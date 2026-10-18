"""起身测验: 从倒地姿态库随机抽开局, 量几秒内站起来(stood_up 连续 0.3 秒)的比例, 按仰躺/趴/侧躺分开。
    python test_getup.py runs/getup2/policy.pt
    python test_getup.py runs/getup2/policy.pt --walk runs/sp2/policy.pt      # 交接测验: 站起来交给走路策略, 一半让站、一半让 0.6 m/s 往前走, 数又摔倒的
"""
import argparse

import torch

from env import quat_to_gravity
from policy import load_policy

p = argparse.ArgumentParser()
p.add_argument("policy")
p.add_argument("--n", type=int, default=1024)
p.add_argument("--secs", type=float, default=8.0, help="起身的时限")
p.add_argument("--hold", type=float, default=0.3, help="站起来的判定要连续保持多少秒")
p.add_argument("--walk", default="", help="走路策略; 给了就做交接测验")
p.add_argument("--after", type=float, default=5.0, help="交接测验: 起身时限之后再跑多少秒")
a = p.parse_args()

if a.walk:
    from combo import make_combo
    env, sw, meta = make_combo(a.walk, a.policy, a.n, hold=a.hold, getup_stand_frac=0.0, seed=7, cmd_stand=0.0)
    meta["iteration"] = meta["getup_iteration"]
    half = torch.arange(a.n, device=env.dev) >= a.n // 2           # 后一半交接后往前走
    f0 = sum(env.gait_rng[0]) / 2
    cmd = torch.zeros(a.n, 3, device=env.dev)
    cmd[half, 0] = 0.6
    gait = torch.stack([torch.full((a.n,), f0, device=env.dev), torch.where(half, 0.6, 1.0)], 1)
    env.set_cmd(cmd, gait)
    env.cmd_t[:] = -10 ** 9                                         # 不让环境自己重抽指令
else:
    env, act, meta = load_policy(a.policy, a.n, getup=True, getup_stand_frac=0.0, ep_len=10 ** 6, seed=7)
assert env.getup, "不是起身策略"
steps, hold = int((a.secs + (a.after if a.walk else 0)) / env.dt), int(a.hold / env.dt)
run = torch.zeros(a.n, dtype=torch.long, device=env.dev)
t_up = torch.full((a.n,), float("inf"), device=env.dev)
lost = torch.zeros(a.n, dtype=torch.bool, device=env.dev)            # 站起来以后又倒了(身体碰地)
cls = env.fallen_cls.to(env.dev)[env.getup_pick]
air, vmax, wmax, hand, rise, prop = (torch.zeros(a.n, device=env.dev) for _ in range(6))   # 起身过程中(站起来之前): 腾空秒数、躯干速度峰值、角速度峰值、手撑地秒数、用时
tau = torch.zeros(a.n, env.nu, device=env.dev)                     # 起身过程中各关节力矩 / 最大力矩的峰值
tau_max = env.tau_max.to(env.dev)
obs, _ = env.observe()
for t in range(steps):
    obs, _, _, _, _, _ = env.step(sw() if a.walk else act(obs))
    rising = torch.isinf(t_up)
    air += (rising & ~env.m_fell & (env.contact.sum(dim=1) == 0)).float() * env.dt
    bg = env.body_ground()
    hand += (rising & bg[:, :2].any(dim=1)).float() * env.dt
    prop += (rising & bg[:, :2].any(dim=1) & ~bg[:, 2]).float() * env.dt          # 手撑着、躯干离地(手在承重)
    rise += rising.float() * env.dt
    tau = torch.where(rising[:, None], torch.maximum(tau, (env.frc[:, env.dadr] / tau_max).abs()), tau)
    vmax = torch.where(rising, torch.maximum(vmax, env.qvel[:, :3].norm(dim=1)), vmax)
    wmax = torch.where(rising, torch.maximum(wmax, env.qvel[:, 3:6].norm(dim=1)), wmax)
    up = env.stood_up()
    run = torch.where(up, run + 1, torch.zeros_like(run))
    newly = (run == hold) & torch.isinf(t_up) & ((t + 1) * env.dt <= a.secs + 1e-6)
    t_up[newly] = (t + 1) * env.dt - a.hold
    lost |= torch.isfinite(t_up) & env.m_fell
print(f"{a.policy}(第 {meta['iteration']} 轮), {a.n} 个, 站起来 = 躯干 > 90% 站立高度、立直、只有脚着地, 连续 {a.hold} 秒"
      + (f"; 站起来后交给 {a.walk}(前一半站、后一半 0.6 m/s 走), 一共跑 {a.secs + a.after:.0f} 秒" if a.walk else ""))
groups = [("全部", torch.ones_like(cls, dtype=torch.bool))] + [(n, cls == i) for i, n in enumerate(env.fallen_names)]
if a.walk:
    groups += [("交接后站", ~half), ("交接后走", half)]
for name, m in groups:
    k = int(m.sum())
    if k == 0:
        continue
    row = " ".join(f"{s}秒内 {(t_up[m] <= s).float().mean() * 100:5.1f}%" for s in sorted({2, 3, 5, a.secs}) if s <= a.secs)
    ok = torch.isfinite(t_up[m])
    med = t_up[m][ok].median().item() if ok.any() else float("nan")
    print(f"  {name:6s} {k:5d} 个 | {row} | 中位用时 {med:.2f} 秒 | 站起来后又倒 {int((lost & m).sum())}/{int(ok.sum())}"
          f" | 手碰地 {hand[m].sum() / rise[m].sum() * 100:.0f}% 的时间, 手承重(躯干离地) {prop[m].sum() / rise[m].sum() * 100:.0f}%(撑过的 {(prop[m] > 0).float().mean() * 100:.0f}%)"
          f" | 起身时腾空 {air[m].mean():.2f} 秒(有腾空的 {(air[m] > 0).float().mean() * 100:.0f}%), 躯干速度峰值 {vmax[m].median():.1f} m/s, 角速度峰值 {wmax[m].median():.1f} rad/s")
import mujoco
jn = [mujoco.mj_id2name(env.mjm, mujoco.mjtObj.mjOBJ_JOINT, int(j)) for j in env.mjm.actuator_trnid[:, 0]]
print("  起身时力矩峰值 / 最大力矩(各人形峰值的中位): " + ", ".join(f"{n} {tau[:, i].median() * 100:.0f}%" for i, n in enumerate(jn)
                                                  if n.endswith("right") or not n.endswith("left")))
h, up = env.qpos[:, 2] / env.root_z0, -quat_to_gravity(env.qpos[:, 3:7])[:, 2]
print(f"  结束时: 还站着 {env.stood_up().float().mean() * 100:.1f}% | 躯干高度 / 站立 平均 {h.mean():.2f}(> 0.5 的 {(h > 0.5).float().mean() * 100:.0f}%)"
      f" | 立直程度平均 {up.mean():.2f} | 身体碰地 {env.m_fell.float().mean() * 100:.0f}%", env.diagnostics())
if a.walk:
    print(f"  交接后的平均前进速度(走的那一半, 结束时): {env.m_vx[half].mean():.2f} m/s; 切换器记到的摔倒 {int((sw.n_falls > 0).sum())}")
