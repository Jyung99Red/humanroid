"""PPO, 从零用 PyTorch 写的, 所有张量都留在 GPU 上。

要点:
- 动作是对角高斯分布, 标准差是与状态无关的可学参数。
- 观测用滑动均值方差归一化; 评论家的目标用 PopArt 式的滑动归一化(改统计量时同步缩放输出层, 输出不跳变)。
- GAE; 回合超时(截断)不是失败: 用终止前状态的价值做 bootstrap, 摔倒(终止)则不做。
- 学习率按 KL 自适应(只调策略网络)。
- 观测可以扩展: 检查点记着观测块布局, 续训时环境多了新块, 第一层给新块补零列(优化器动量、归一化统计一起补), 已有行为不变。
"""
import math
import time

import torch
import torch.nn as nn


class RunningMeanStd:
    """每一维各自计数: 后加的输入计数从零开始, 统计量很快跟上(共用一个计数的话, 训练久了新维度几乎不更新)。"""
    def __init__(self, dim, device, eps=1e-4):
        self.mean = torch.zeros(dim, device=device)
        self.var = torch.ones(dim, device=device)
        self.count = torch.full((dim,), eps, device=device)

    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        d, tot = bm - self.mean, self.count + bc
        self.mean = self.mean + d * bc / tot
        self.var = (self.var * self.count + bv * bc + d * d * self.count * bc / tot) / tot
        self.count = tot

    def norm(self, x):
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-10, 10)

    def state(self):
        return dict(mean=self.mean.cpu(), var=self.var.cpu(), count=self.count.cpu())

    def load(self, s, device):
        self.mean, self.var = s["mean"].to(device), s["var"].to(device)
        c = s["count"]                                  # 旧检查点里是一个数
        self.count = (c.to(device) if torch.is_tensor(c) else torch.full_like(self.mean, float(c))).float()


def remap_cols(t, old_layout, new_layout, fill=0.0):
    """t 的最后一维按 old_layout([(块名, 维数), ...])排列, 返回按 new_layout 排列的新张量: 旧块按名字搬过去, 新块填 fill。
    旧块必须都还在、维数不变(删输入或改维数会改变已有行为, 不支持)。"""
    old, o = {}, 0
    for name, d in old_layout:
        old[name] = (o, d)
        o += d
    assert o == t.shape[-1], f"布局 {o} 维, 张量 {t.shape[-1]} 维"
    missing = set(old) - {name for name, _ in new_layout}
    assert not missing, f"新观测里少了旧输入 {sorted(missing)}"
    out = t.new_full(t.shape[:-1] + (sum(d for _, d in new_layout),), fill)
    n = 0
    for name, d in new_layout:
        if name in old:
            s, od = old[name]
            assert od == d, f"观测块 {name} 维数变了: {od} -> {d}"
            out[..., n:n + d] = t[..., s:s + d]
        n += d
    return out


def expand_rms(s, old_layout, new_layout):
    """把归一化统计按新布局扩展: 新维度均值 0、方差 1、计数从零开始。"""
    c = s["count"] if torch.is_tensor(s["count"]) else torch.full_like(s["mean"], float(s["count"]))
    return dict(mean=remap_cols(s["mean"], old_layout, new_layout, 0.0), var=remap_cols(s["var"], old_layout, new_layout, 1.0),
                count=remap_cols(c.float(), old_layout, new_layout, 1e-4))


def mlp(i, o, hidden, out_gain):
    layers, d = [], i
    for h in hidden:
        lin = nn.Linear(d, h)
        nn.init.orthogonal_(lin.weight, math.sqrt(2))
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.ELU()]
        d = h
    out = nn.Linear(d, o)
    nn.init.orthogonal_(out.weight, out_gain)
    nn.init.zeros_(out.bias)
    return nn.Sequential(*layers, out)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, critic_dim, act_dim, hidden=(512, 256, 128), init_std=0.5):
        super().__init__()
        self.actor = mlp(obs_dim, act_dim, hidden, 0.01)
        self.critic = mlp(critic_dim, 1, hidden, 1.0)
        self.log_std = nn.Parameter(torch.full((act_dim,), math.log(init_std)))

    def std(self):
        return self.log_std.clamp(-4.0, 0.7).exp()


class PPO:
    def __init__(self, env, T=24, epochs=5, minibatches=4, lr=1e-3, critic_lr=5e-4, clip=0.2, ent=0.005,
                 gamma=0.99, lam=0.95, kl_target=0.01, init_std=0.5, hidden=(512, 256, 128), max_grad=1.0):
        self.env, self.dev = env, env.dev
        self.T, self.epochs, self.mb, self.clip, self.ent = T, epochs, minibatches, clip, ent
        self.gamma, self.lam, self.kl_target, self.max_grad, self.lr = gamma, lam, kl_target, max_grad, lr
        self.ac = ActorCritic(env.obs_dim, env.critic_dim, env.nu, hidden, init_std).to(self.dev)
        actor_params = list(self.ac.actor.parameters()) + [self.ac.log_std]
        self.actor_params, self.critic_params = actor_params, list(self.ac.critic.parameters())
        self.opt = torch.optim.Adam([dict(params=actor_params, lr=lr), dict(params=self.critic_params, lr=critic_lr)])
        self.obs_rms = RunningMeanStd(env.obs_dim, self.dev)
        self.crit_rms = RunningMeanStd(env.critic_dim, self.dev)
        self.ret_mean, self.ret_std, self.ret_init = 0.0, 1.0, False
        self.obs, self.cobs = env.observe()
        self.ep_ret = torch.zeros(env.n, device=self.dev)
        self.ep_len = torch.zeros(env.n, device=self.dev)
        self.iteration = 0

    # ---- 价值的归一化 ----
    def value(self, cobs_n):
        return self.ac.critic(cobs_n).squeeze(-1) * self.ret_std + self.ret_mean

    def _update_ret_stats(self, ret):
        b_mean, b_std = ret.mean().item(), ret.std().item() + 1e-6
        beta = 1.0 if not self.ret_init else 0.1
        new_mean = (1 - beta) * self.ret_mean + beta * b_mean
        new_std = (1 - beta) * self.ret_std + beta * b_std
        last = self.ac.critic[-1]
        with torch.no_grad():                       # 缩放输出层, 让网络当前的价值输出保持不变
            last.weight.mul_(self.ret_std / new_std)
            last.bias.copy_((self.ret_std * last.bias + self.ret_mean - new_mean) / new_std)
        self.ret_mean, self.ret_std, self.ret_init = new_mean, new_std, True

    # ---- 收集 ----
    @torch.no_grad()
    def collect(self):
        env, T, N = self.env, self.T, self.env.n
        f = dict(device=self.dev, dtype=torch.float32)
        B = dict(obs=torch.zeros(T, N, env.obs_dim, **f), cobs=torch.zeros(T, N, env.critic_dim, **f),
                 act=torch.zeros(T, N, env.nu, **f), mu=torch.zeros(T, N, env.nu, **f), logp=torch.zeros(T, N, **f),
                 val=torch.zeros(T, N, **f), rew=torch.zeros(T, N, **f), done=torch.zeros(T, N, **f))
        raw_obs = torch.zeros(T, N, env.obs_dim, **f)       # 原始(未归一化)观测, 用来更新均值方差
        raw_cobs = torch.zeros(T, N, env.critic_dim, **f)
        std = self.ac.std()
        S = dict(rew=0.0, fin_ret=0.0, fin_len=0.0, fin_n=0.0, term=0.0, lin_err=0.0, vx=0.0)
        for t in range(T):
            on, cn = self.obs_rms.norm(self.obs), self.crit_rms.norm(self.cobs)
            mu = self.ac.actor(on)
            a = mu + std * torch.randn_like(mu)
            logp = (-0.5 * ((a - mu) / std) ** 2 - std.log() - 0.5 * math.log(2 * math.pi)).sum(-1)
            v = self.value(cn)
            B["obs"][t], B["cobs"][t], B["act"][t], B["mu"][t], B["logp"][t], B["val"][t] = on, cn, a, mu, logp, v
            raw_obs[t], raw_cobs[t] = self.obs, self.cobs

            self.obs, self.cobs, r, te, tr, info = env.step(a)
            done = te | tr
            if info["final_critic"] is not None:          # 超时截断: 用终止前状态的价值做 bootstrap
                idx = tr.nonzero(as_tuple=False).squeeze(1)
                vf = self.value(self.crit_rms.norm(info["final_critic"][idx]))
                r = r.clone()
                r[idx] += self.gamma * vf
            B["rew"][t], B["done"][t] = r, done.float()

            self.ep_ret += r
            self.ep_len += 1
            d = done.float()
            S["fin_ret"] += (self.ep_ret * d).sum()
            S["fin_len"] += (self.ep_len * d).sum()
            S["fin_n"] += d.sum()
            S["term"] += te.float().sum()
            self.ep_ret *= 1 - d
            self.ep_len *= 1 - d
            S["rew"] += r.sum()
            S["lin_err"] += env.m_lin_err.sum()
            S["vx"] += env.m_vx.sum()

        # GAE
        vnext = self.value(self.crit_rms.norm(self.cobs))
        adv, last = torch.zeros(T, N, **f), torch.zeros(N, **f)
        for t in reversed(range(T)):
            nd = 1 - B["done"][t]
            nv = vnext if t == T - 1 else B["val"][t + 1]
            delta = B["rew"][t] + self.gamma * nv * nd - B["val"][t]
            last = delta + self.gamma * self.lam * nd * last
            adv[t] = last
        B["adv"], B["ret"] = adv, adv + B["val"]
        S = {k: float(v) for k, v in S.items()}
        S["steps"] = T * N
        S["raw_obs"], S["raw_cobs"] = raw_obs, raw_cobs
        return B, S

    # ---- 更新 ----
    def update(self, B):
        T, N = self.T, self.env.n
        flat = {k: v.reshape(T * N, *v.shape[2:]) for k, v in B.items()}
        n = T * N
        adv = flat["adv"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        if not self.ret_init:                               # 第一轮: 先用这批回报定好归一化(输出层同步缩放, 价值不变)
            self._update_ret_stats(flat["ret"])
        ret_n = (flat["ret"] - self.ret_mean) / self.ret_std
        stats = dict(kl=0.0, vloss=0.0, ploss=0.0, clipfrac=0.0, k=0)
        last_kl, stop = 0.0, False
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.dev)
            ep_kl = []
            for chunk in perm.chunk(self.mb):
                on, cn, act = flat["obs"][chunk], flat["cobs"][chunk], flat["act"][chunk]
                mu_old, logp_old = flat["mu"][chunk], flat["logp"][chunk]
                mu, std = self.ac.actor(on), self.ac.std()
                logp = (-0.5 * ((act - mu) / std) ** 2 - std.log() - 0.5 * math.log(2 * math.pi)).sum(-1)
                ratio = (logp - logp_old).exp()
                a = adv[chunk]
                surr = torch.min(ratio * a, ratio.clamp(1 - self.clip, 1 + self.clip) * a)
                ploss = -surr.mean()
                v = self.ac.critic(cn).squeeze(-1)
                vloss = 0.5 * ((v - ret_n[chunk]) ** 2).mean()
                entropy = (self.ac.log_std.clamp(-4.0, 0.7) + 0.5 * math.log(2 * math.pi * math.e)).sum()
                loss = ploss + vloss - self.ent * entropy
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_params, self.max_grad)
                nn.utils.clip_grad_norm_(self.critic_params, self.max_grad)
                self.opt.step()

                with torch.no_grad():                      # 旧分布 -> 新分布的 KL
                    s_old = self.old_std
                    kl = ((std.log() - s_old.log() + (s_old ** 2 + (mu_old - mu) ** 2) / (2 * std ** 2) - 0.5)
                          .sum(-1).mean().item())
                    ep_kl.append(kl)
                    stats["kl"] += kl
                    stats["vloss"] += vloss.item()
                    stats["ploss"] += ploss.item()
                    stats["clipfrac"] += ((ratio - 1).abs() > self.clip).float().mean().item()
                    stats["k"] += 1
                if kl > 3 * self.kl_target:                # 这一轮策略已经动得够多了, 提前收手
                    stop = True
                    break
            last_kl = sum(ep_kl) / len(ep_kl)
            if stop:
                break
        # 学习率每轮只调一次, 看最后一个 epoch 的平均 KL(KL 是相对收集时的旧策略累积的, 逐小批调会塌缩)
        if last_kl > 2 * self.kl_target:
            self.lr = max(1e-5, self.lr / 1.2)
        elif last_kl < 0.5 * self.kl_target:
            self.lr = min(3e-3, self.lr * 1.2)
        self.opt.param_groups[0]["lr"] = self.lr
        k = stats.pop("k")
        return {key: val / k for key, val in stats.items()}

    def train_iteration(self):
        t0 = time.perf_counter()
        B, S = self.collect()
        t1 = time.perf_counter()
        self.old_std = self.ac.std().detach()               # 收集时的策略标准差, 更新时算 KL 用
        U = self.update(B)
        # 统计量放在更新之后再改, 这一轮收集和更新用的归一化前后一致
        self.obs_rms.update(S.pop("raw_obs").reshape(-1, self.env.obs_dim))
        self.crit_rms.update(S.pop("raw_cobs").reshape(-1, self.env.critic_dim))
        self._update_ret_stats(B["ret"].reshape(-1))
        t2 = time.perf_counter()
        self.iteration += 1
        n = max(S["fin_n"], 1.0)
        return dict(it=self.iteration, steps=S["steps"], t_collect=t1 - t0, t_update=t2 - t1,
                    rew=S["rew"] / S["steps"], ep_ret=S["fin_ret"] / n, ep_len=S["fin_len"] / n,
                    fall=S["term"] / S["steps"], lin_err=S["lin_err"] / S["steps"], vx=S["vx"] / S["steps"],
                    std=float(self.ac.std().detach().mean()), lr=self.lr, **U)

    # ---- 存取 ----
    def state(self):
        return dict(ac=self.ac.state_dict(), opt=self.opt.state_dict(), obs_rms=self.obs_rms.state(),
                    crit_rms=self.crit_rms.state(), ret=(self.ret_mean, self.ret_std, self.ret_init),
                    lr=self.lr, iteration=self.iteration, obs_layout=self.env.obs_layout, critic_layout=self.env.critic_layout)

    def load(self, s):
        """载入检查点。环境的观测布局比检查点多了新块时, 自动扩展(新输入一开始权重为零, 不影响已有行为)。"""
        s = expand_checkpoint(s, self.env.obs_layout, self.env.critic_layout, len(self.actor_params))
        self.ac.load_state_dict(s["ac"])
        self.opt.load_state_dict(s["opt"])
        self.obs_rms.load(s["obs_rms"], self.dev)
        self.crit_rms.load(s["crit_rms"], self.dev)
        self.ret_mean, self.ret_std, self.ret_init = s["ret"]
        self.lr, self.iteration = s["lr"], s["iteration"]
        self.opt.param_groups[0]["lr"] = self.lr


def default_layouts(nu):
    """没记布局的旧检查点(walk1~walk4 和本改动之前的试训)用的是这套默认观测。"""
    obs = [("gravity", 3), ("lin_vel", 3), ("ang_vel", 3), ("joint_pos", nu), ("joint_vel", nu), ("last_action", nu), ("cmd", 3), ("clock", 2)]
    return obs, obs + [("height", 1), ("foot_contact", 2)]


def expand_checkpoint(s, obs_layout, critic_layout, critic_first_param):
    """检查点 s 的观测布局 -> 新布局: 演员和评论家第一层权重、它们的 Adam 动量、两份归一化统计都按块名重排, 新块补零。
    critic_first_param: 优化器参数列表里评论家第一层权重的序号(演员参数 + log_std 之后)。"""
    nu = s["ac"]["log_std"].shape[0]
    old_obs, old_crit = (s["obs_layout"], s["critic_layout"]) if "obs_layout" in s else default_layouts(nu)
    old_obs, old_crit = [tuple(x) for x in old_obs], [tuple(x) for x in old_crit]
    if old_obs == list(obs_layout) and old_crit == list(critic_layout):
        return s
    s = dict(s, ac=dict(s["ac"]), opt=dict(s["opt"], state=dict(s["opt"]["state"])))
    for key, idx, old, new in (("actor.0.weight", 0, old_obs, obs_layout), ("critic.0.weight", critic_first_param, old_crit, critic_layout)):
        s["ac"][key] = remap_cols(s["ac"][key], old, new, 0.0)
        if idx in s["opt"]["state"]:                   # 优化器还没走过一步时没有动量
            st = dict(s["opt"]["state"][idx])
            for m in ("exp_avg", "exp_avg_sq"):
                st[m] = remap_cols(st[m], old, new, 0.0)
            s["opt"]["state"][idx] = st
    s["obs_rms"] = expand_rms(s["obs_rms"], old_obs, obs_layout)
    s["crit_rms"] = expand_rms(s["crit_rms"], old_crit, critic_layout)
    added = [n for n, _ in obs_layout if n not in dict(old_obs)] + [f"{n}(评论家)" for n, _ in critic_layout if n not in dict(old_crit)]
    print(f"观测扩展: 新增 {added}, 新输入的权重从零开始")
    return s
