"""诊断: 同一个检查点, 在不同条件下的真实表现(平均前进速度、摔倒率)。
用来区分"回放脚本有 bug"还是"策略只在带噪声时才会走"。"""
import sys

import torch

from env import HumanoidEnv
from ppo import RunningMeanStd, mlp

ck = torch.load(sys.argv[1] if len(sys.argv) > 1 else "runs/walk1/ckpt.pt", map_location="cuda:0", weights_only=False)
ac = ck["ac"]
N, STEPS = 1024, 300
print(f"检查点第 {ck['iteration']} 轮, 训练时 log_std 均值对应 std={ac['log_std'].exp().mean().item():.3f}")


def run(name, deterministic, random_yaw, cmd, use_std=None):
    kw = dict(random_yaw=random_yaw)
    if cmd is not None:
        kw.update(cmd_vx=(cmd, cmd), cmd_vy=(0, 0), cmd_wz=(0, 0))
    env = HumanoidEnv(N, seed=3, **kw)
    env.t[:] = 0
    env.reset_idx(torch.arange(N, device=env.dev))
    actor = mlp(env.obs_dim, env.nu, (512, 256, 128), 0.01).to(env.dev)
    actor.load_state_dict({k[len("actor."):]: v for k, v in ac.items() if k.startswith("actor.")})
    rms = RunningMeanStd(env.obs_dim, env.dev)
    rms.load(ck["obs_rms"], env.dev)
    std = ac["log_std"].exp().to(env.dev) if use_std is None else use_std
    obs, _ = env.observe()
    vx = torch.zeros(N, device=env.dev); cnt = torch.zeros(N, device=env.dev); falls = 0; eps = 0
    with torch.no_grad():
        for _ in range(STEPS):
            mu = actor(rms.norm(obs))
            a = mu if deterministic else mu + std * torch.randn_like(mu)
            obs, _, r, te, tr, _ = env.step(a)
            vx += env.m_vx; cnt += 1
            falls += int(te.sum()); eps += int((te | tr).sum())
    print(f"{name:34s} 平均 vx {(vx / cnt).mean().item():6.2f} | 摔倒 {falls:5d} 次 / {STEPS * N} 步 | 目标 {'随机' if cmd is None else cmd}")


run("训练条件(带噪声, 随机朝向, 随机指令)", False, True, None)
run("带噪声, 固定朝向, 指令 1.0", False, False, 1.0)
run("确定性(无噪声), 随机朝向, 随机指令", True, True, None)
run("确定性(无噪声), 固定朝向, 指令 1.0", True, False, 1.0)
run("确定性, 随机朝向, 指令 0.8", True, True, 0.8)
