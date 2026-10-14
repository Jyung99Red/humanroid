"""环境冒烟测试: 零动作(PD 保持默认姿态)、随机动作, 看有没有 NaN、会不会摔、重置对不对、整体速度。"""
import sys
import time

import torch
import warp as wp

from env import HumanoidEnv

n = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
xml = sys.argv[2] if len(sys.argv) > 2 else "models/humanoid_pd.xml"
env = HumanoidEnv(n, xml=xml, arm_pose="models/arm_pose.json")
print("模型", xml)
print(f"obs={env.obs_dim} critic={env.critic_dim} act={env.nu} dt={env.dt}s  policy-step/episode={env.ep_len}")
obs, critic = env.observe()
print("初始观测: 形状", tuple(obs.shape), "有NaN", bool(torch.isnan(obs).any()), "| 重力方向(应约 0,0,-1):", obs[0, :3].tolist())


def run(name, act_fn, steps=100):
    ep_done = 0
    term = 0
    ret = torch.zeros(n, device=env.dev)
    zs = []
    t0 = time.perf_counter()
    for _ in range(steps):
        obs, critic, r, te, tr, info = env.step(act_fn())
        ret += r
        term += int(te.sum())
        ep_done += int((te | tr).sum())
        zs.append(env.qpos[:, 2].mean().item())
    wp.synchronize()
    dt = time.perf_counter() - t0
    assert torch.isfinite(env.qpos).all(), "出现 NaN/inf"
    print(f"[{name}] {steps} 步 {dt:.2f}s -> {n*steps/dt/1e3:.0f}k 策略步/秒 ({n*steps*env.dec/dt/1e6:.2f}M 物理步/秒) | "
          f"每步平均奖励 {ret.mean().item()/steps:.3f} | 摔倒/重置 {term}/{ep_done} 次 | 躯干平均高度 {zs[0]:.2f}->{zs[-1]:.2f} | {env.diagnostics()}")


run("零动作(保持默认姿态)", lambda: torch.zeros(n, env.nu, device=env.dev))
run("随机动作 std=0.5", lambda: torch.randn(n, env.nu, device=env.dev) * 0.5)
run("随机动作 std=1.5(猛甩, 查数值稳定)", lambda: torch.randn(n, env.nu, device=env.dev) * 1.5)
