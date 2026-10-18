"""奖励分项: 固定一个指令走一段, 把每步的走路奖励拆成各项(env._walk_reward 一项一项单独算), 看哪一项在拉后腿。
--obs-cmd 是反事实对比: 策略看到的是 obs-cmd(按它走), 奖励按 --cmd 算。例如策略侧移总是只走到指令的六成:
    python reward_terms.py runs/sp2/policy.pt --cmd 0,0.4,0                    # 照常: 指令 0.4, 实际约 0.24
    python reward_terms.py runs/sp2/policy.pt --cmd 0,0.4,0 --obs-cmd 0,0.8,0  # 让它按 0.8 的步子走(约 0.5), 奖励还按 0.4 算
后者总奖励更高 = 奖励是鼓励走快的, 策略困在慢步子里(探索不到); 更低 = 是奖励在罚快(看是哪几项)。
"""
import argparse
from collections import defaultdict

import torch

from policy import load_policy

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("policy")
    p.add_argument("--cmd", default="0,0.4,0", help="算奖励用的指令 vx,vy,wz")
    p.add_argument("--obs-cmd", default="", help="策略看到的指令(不给 = 同 --cmd)")
    p.add_argument("--gait", default="", help="步频,着地比例; 不给 = 训练范围的中点(指令全 0 时着地比例 1)")
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--secs", type=float, default=6.0)
    p.add_argument("--skip", type=float, default=2.0, help="开头这么多秒不算(起步)")
    a = p.parse_args()
    vec = lambda s: [float(v) for v in s.split(",")]
    env, act, meta = load_policy(a.policy, a.n, push_force=0.0, cmd_period=10 ** 9, ep_len=10 ** 9, cmd_stand=0.0)
    dev = env.dev
    cmd = torch.tensor(vec(a.cmd), device=dev).expand(a.n, 3).clone()
    ocmd = torch.tensor(vec(a.obs_cmd), device=dev).expand(a.n, 3).clone() if a.obs_cmd else cmd
    (f_r, d_r) = env.gait_rng
    g = vec(a.gait) if a.gait else [sum(f_r) / 2, 1.0 if not cmd.abs().sum() else sum(d_r) / 2]
    gait = torch.tensor(g, device=dev).expand(a.n, 2).clone()

    terms, n_acc = defaultdict(float), [0]
    orig = env._walk_reward
    counting = [False]

    def wrapped(rw, q, gv):
        if counting[0]:
            saved = env.foot_prev.clone(), env.point_prev.clone()
            for k, w in rw.items():
                if not w or k.startswith("getup"):
                    continue
                env.foot_prev.copy_(saved[0])
                env.point_prev.copy_(saved[1])
                terms[k] += float(orig({kk: (w if kk == k else 0.0) for kk in rw}, q, gv).mean())
            env.foot_prev.copy_(saved[0])
            env.point_prev.copy_(saved[1])
            n_acc[0] += 1
        return orig(rw, q, gv)

    env._walk_reward = wrapped
    env.reset_idx(torch.arange(a.n, device=dev))
    env.set_cmd(cmd, gait)
    falls, vx, vy = torch.zeros(a.n, dtype=torch.bool, device=dev), [], []
    T, S = int(a.secs / env.dt), int(a.skip / env.dt)
    for t in range(T):
        env.set_cmd(ocmd, gait)
        obs, _ = env.observe()
        action = act(obs)
        env.set_cmd(cmd, gait)
        counting[0] = t >= S
        _, _, _, te, _, _ = env.step(action)
        falls |= te
        if t >= S:
            vx.append(float(env.m_vx.mean()))
            vy.append(float(env.m_vy.mean()))
    total = sum(terms.values()) / n_acc[0]
    print(f"{a.policy}(第 {meta['iteration']} 轮) | 奖励按指令 {vec(a.cmd)}" + (f", 策略看到的是 {vec(a.obs_cmd)}" if a.obs_cmd else "")
          + f" | 实际 vx {sum(vx) / len(vx):+.2f} vy {sum(vy) / len(vy):+.2f} m/s | 摔 {int(falls.sum())}/{a.n}")
    print(f"每步总奖励 {total:.3f}(第 {a.skip:g}~{a.secs:g} 秒平均), 各项:")
    for k, v in sorted(terms.items(), key=lambda kv: kv[1] / n_acc[0]):
        print(f"  {k:14s} {v / n_acc[0]:+.3f}")
