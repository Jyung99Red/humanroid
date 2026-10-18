"""起身策略 + 走路策略放在同一个环境里, 每个人形按状态切换。回放(export.py --getup)、起身测验(test_getup.py --walk)、以后的网页都用这套规则:
  走 -> 起身: 脚以外的部位碰地(和训练时的摔倒定义一样)。
  起身 -> 走: env.stood_up()(躯干 > 站立的 90%、立直、只有脚着地)连续 hold 秒。
环境按走路策略的设置建(指令、步态时钟都有), 再打开起身模式: 倒地开局、碰地不终止、动作按关节放开到整个活动范围。
两个网络各自按自己的观测块布局取观测; 走路策略训练时动作裁在 ±1.5, 它的动作和它看到的"上一步动作"都裁回 ±1.5。

    env, sw, meta = make_combo("runs/sp2/policy.pt", "runs/getup2/policy.pt", 6, getup_stand_frac=0.5)
    obs, _ = env.observe(); env.step(sw(obs))        # sw.mode: 0 走 1 起身; sw.n_falls: 每个人形摔了几次
"""
import torch

from policy import env_config
from ppo import RunningMeanStd, mlp
from env import HumanoidEnv


def actor_on(s, env, layout, clip=None):
    """策略文件 s 的演员 -> 函数(sl) -> 世界 sl 的动作。观测按 layout 的块名直接从环境取, 不要求和环境的观测布局一致。"""
    dim = sum(d for _, d in layout)
    net = mlp(dim, env.nu, s["hidden"], 0.01).to(env.dev)
    net.load_state_dict(s["actor"])
    rms = RunningMeanStd(dim, env.dev)
    rms.load(s["obs_rms"], env.dev)

    @torch.no_grad()
    def act(sl=slice(None)):
        fns = env._block_fns()
        if clip is not None:
            fns["last_action"] = lambda: env.action.clamp(-clip, clip)
        a = net(rms.norm(torch.cat([fns[k]() for k, _ in layout], dim=1)[sl]))
        return a if clip is None else a.clamp(-clip, clip)

    return act


class Switcher:
    """getup_s 可以是一个起身策略, 也可以是一串(时间线: 世界平均分成几块, 每块用一个快照); tile = True 时每块的开局姿态都和第一块一样"""

    def __init__(self, env, walk_s, getup_s, hold=0.3, tile=False):
        self.env, self.hold, self.tile = env, int(round(hold / env.dt)), tile
        self.walk = actor_on(walk_s, env, [tuple(b) for b in walk_s["obs_layout"]], clip=1.5)
        gl = getup_s if isinstance(getup_s, (list, tuple)) else [getup_s]
        self.K = env.n // len(gl)
        assert self.K * len(gl) == env.n
        self.getups = [actor_on(g, env, [tuple(b) for b in g["obs_layout"]]) for g in gl]
        self.reset()

    def getup(self):
        K = self.K
        return torch.cat([f(slice(i * K, (i + 1) * K)) for i, f in enumerate(self.getups)])

    def reset(self):
        """环境重置后调用: 倒地开局的从起身开始"""
        env = self.env
        if self.tile:
            K = self.K
            for x in (env.qpos, env.qvel, env.getup_pick):
                x[K:] = x[:K].repeat(env.n // K - 1, *([1] * (x.dim() - 1)))
        self.mode = (env.getup_pick >= 0).long()
        self.run = torch.zeros(env.n, dtype=torch.long, device=env.dev)
        self.n_falls = torch.zeros(env.n, dtype=torch.long, device=env.dev)
        self.n_up = torch.zeros(env.n, dtype=torch.long, device=env.dev)

    @torch.no_grad()
    def __call__(self, obs=None):
        env = self.env
        fell = env.ground_contact()[0]
        up = env.stood_up(fell=fell)
        self.run = torch.where(up, self.run + 1, torch.zeros_like(self.run))
        to_getup = (self.mode == 0) & fell
        to_walk = (self.mode == 1) & (self.run >= self.hold)
        self.n_falls += to_getup
        self.n_up += to_walk
        self.mode = torch.where(to_getup, 1, torch.where(to_walk, 0, self.mode))
        env.since_push[to_walk] = 0                       # 刚站起来交给走路策略: 1 秒内不按步态时钟要求(和被推后一样)
        aw, ag = self.walk(), self.getup()
        return torch.where(self.mode[:, None] == 1, ag, aw)


def combo_config(ws, gs, **overrides):
    """走路策略 ws 的环境设置 + 起身模式(按起身策略 gs 的动作范围)"""
    gcfg = gs["env"]
    assert gcfg.get("getup"), "不是起身策略"
    assert gcfg.get("xml") == ws["env"].get("xml") and gcfg.get("arm_pose") == ws["env"].get("arm_pose"), "两个策略的身体或手臂默认姿势不一样"
    return env_config(ws, **dict(dict(getup=True, act_clip=gcfg.get("act_clip", 1.5), ep_len=10 ** 6), **overrides))


def make_combo(walk_path, getup_path, nworld, hold=0.3, **overrides):
    """返回 (环境, 切换器, {walk_iteration, getup_iteration, env})。overrides 覆盖走路策略的环境设置(指令、getup_stand_frac 等)。"""
    ws = torch.load(walk_path, map_location="cuda:0", weights_only=False)
    gs = torch.load(getup_path, map_location="cuda:0", weights_only=False)
    cfg = combo_config(ws, gs, **overrides)
    env = HumanoidEnv(nworld, **cfg)
    sw = Switcher(env, ws, gs, hold)
    return env, sw, dict(walk_iteration=int(ws["iteration"]), getup_iteration=int(gs["iteration"]), env=cfg)
