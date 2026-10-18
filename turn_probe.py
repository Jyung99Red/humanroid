"""转身测验: 原地转(前进 0)和边走边转(前进 0.6 m/s), 转向指令 0.3 / 0.6 / 1.0 rad/s(左右各一半), 量实际转速和摔倒。
    python turn_probe.py runs/turn1/policy.pt
"""
import argparse

import torch

from eval_policy import measure
from policy import load_policy

p = argparse.ArgumentParser()
p.add_argument("policy")
p.add_argument("--k", type=int, default=128, help="每种指令几个人形")
a = p.parse_args()

CASES = [(vx, wz) for vx in (0.0, 0.6) for wz in (0.3, 0.6, 1.0)]
env, act, meta = load_policy(a.policy, a.k * len(CASES), cmd_stand=0.0)
f0 = sum(env.gait_rng[0]) / 2
cmd = torch.zeros(env.n, 3, device=env.dev)
gait = torch.zeros(env.n, 2, device=env.dev)
sign = torch.where(torch.arange(env.n, device=env.dev) % 2 == 0, 1.0, -1.0)     # 一半左转一半右转
for i, (vx, wz) in enumerate(CASES):
    s = slice(i * a.k, (i + 1) * a.k)
    cmd[s, 0], cmd[s, 2] = vx, wz * sign[s]
    gait[s] = torch.tensor([f0, 0.6], device=env.dev)
r = measure(env, act, cmd, gait, seconds=8.0, warm=2.0)
print(f"{a.policy}(第 {meta['iteration']} 轮)")
for i, (vx, wz) in enumerate(CASES):
    s = slice(i * a.k, (i + 1) * a.k)
    ok = ~r["fell"][s]
    got = (r["wz"][s] * sign[s])[ok].mean().item() if ok.any() else float("nan")
    print(f"  {'原地转' if vx == 0 else f'走 {vx} m/s 同时转'} {wz:.1f} rad/s: 实际 {got:.2f} rad/s({got / wz * 100:3.0f}%) | 前进 {r['vx'][s][ok].mean():+.2f} m/s"
          f" | 摔 {int((~ok).sum())}/{a.k}")
