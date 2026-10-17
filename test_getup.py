"""起身测验: 从倒地姿态库随机抽开局, 量几秒内站起来(stood_up 连续 0.3 秒)的比例, 按仰躺/趴/侧躺分开。
    python test_getup.py runs/getup1/policy.pt
"""
import argparse

import torch

from policy import load_policy

p = argparse.ArgumentParser()
p.add_argument("policy")
p.add_argument("--n", type=int, default=1024)
p.add_argument("--secs", type=float, default=8.0)
p.add_argument("--hold", type=float, default=0.3, help="站起来的判定要连续保持多少秒")
a = p.parse_args()

env, act, meta = load_policy(a.policy, a.n, getup=True, getup_stand_frac=0.0, ep_len=10 ** 6, seed=7)
assert env.getup, "不是起身策略"
steps, hold = int(a.secs / env.dt), int(a.hold / env.dt)
run = torch.zeros(a.n, dtype=torch.long, device=env.dev)
t_up = torch.full((a.n,), float("inf"), device=env.dev)
lost = torch.zeros(a.n, dtype=torch.bool, device=env.dev)            # 站起来以后又倒了(身体碰地)
cls = env.fallen_cls.to(env.dev)[env.getup_pick]
obs, _ = env.observe()
for t in range(steps):
    obs, _, _, _, _, _ = env.step(act(obs))
    up = env.stood_up()
    run = torch.where(up, run + 1, torch.zeros_like(run))
    newly = (run == hold) & torch.isinf(t_up)
    t_up[newly] = (t + 1) * env.dt - a.hold
    lost |= torch.isfinite(t_up) & env.m_fell
print(f"{a.policy}(第 {meta['iteration']} 轮), {a.n} 个, 站起来 = 躯干 > 90% 站立高度、立直、只有脚着地, 连续 {a.hold} 秒")
for name, m in [("全部", torch.ones_like(cls, dtype=torch.bool))] + [(n, cls == i) for i, n in enumerate(env.fallen_names)]:
    k = int(m.sum())
    if k == 0:
        continue
    row = " ".join(f"{s}秒内 {(t_up[m] <= s).float().mean() * 100:5.1f}%" for s in (2, 3, 5, a.secs))
    ok = torch.isfinite(t_up[m])
    med = t_up[m][ok].median().item() if ok.any() else float("nan")
    print(f"  {name:4s} {k:5d} 个 | {row} | 中位用时 {med:.2f} 秒 | 站起来后又倒 {int((lost & m).sum())}")
from env import quat_to_gravity
h, up = env.qpos[:, 2] / env.root_z0, -quat_to_gravity(env.qpos[:, 3:7])[:, 2]
print(f"  结束时: 还站着 {env.stood_up().float().mean() * 100:.1f}% | 躯干高度 / 站立 平均 {h.mean():.2f}(> 0.5 的 {(h > 0.5).float().mean() * 100:.0f}%)"
      f" | 立直程度平均 {up.mean():.2f} | 身体碰地 {env.m_fell.float().mean() * 100:.0f}%", env.diagnostics())
