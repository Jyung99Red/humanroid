"""检查左右镜像对不对: 镜像约束(train.py --sym)的前提是身体、观测、奖励都左右对称, 镜像映射写错了约束就是在教错的东西。

    python check_mirror.py runs/xxx/policy.pt

两个世界: 世界 0 正常走(用这个策略), 世界 1 从世界 0 的镜像状态出发、每步执行世界 0 动作的镜像。
身体和奖励都对称、映射也写对的话, 世界 1 的观测一直等于世界 0 观测的镜像, 两边奖励一样(物理求解有浮点误差, 会慢慢放大)。
顺带量这个策略本身有多不对称: 策略(镜像观测) 和 镜像(策略(观测)) 差多少。
"""
import sys

import torch

from policy import load_policy

env, act, meta = load_policy(sys.argv[1], 2, random_yaw=False, cmd_vx=(1.0, 1.0), cmd_vy=(0.0, 0.0), cmd_wz=(0.15, 0.15), cmd_stand=0.0, **({"cmd_acc": float(sys.argv[2])} if len(sys.argv) > 2 else {}))
env.ep_len = env.cmd_period = 10 ** 9
idx = torch.arange(2, device=env.dev)
env.t[:] = 0
env.reset_idx(idx)
env._sample_cmd(idx)
with torch.no_grad():
    for _ in range(150):                                   # 先走几步, 让状态里有速度、有接触
        obs, _ = env.observe()
        env.step(act(obs))

    # 世界 1 <- 世界 0 的镜像
    A, P = env.act_mirror, env.obs_mirror
    q, v = env.qpos[0].clone(), env.qvel[0].clone()
    q1, v1 = q.clone(), v.clone()
    q1[1] = -q[1]                                          # 位置 y 变号
    q1[3:7] = q[3:7] * torch.tensor([1.0, -1.0, 1.0, -1.0], device=env.dev)   # 四元数 (w, -x, y, -z)
    q1[env.qadr] = env.mirror(q[env.qadr], A)
    v1[1] = -v[1]                                          # 线速度(世界系) y 变号
    v1[3:6] = v[3:6] * torch.tensor([-1.0, 1.0, -1.0], device=env.dev)       # 角速度(机体系)是赝矢量
    v1[env.dadr] = env.mirror(v[env.dadr], A)
    env.qpos[1], env.qvel[1] = q1, v1
    env.ctrl[1] = env.mirror(env.ctrl[0], A)
    env.warm[:] = 0
    env.phase[1] = (env.phase[0] + 0.5) % 1.0
    env.cmd[1] = env.cmd[0] * torch.tensor([1.0, -1.0, -1.0], device=env.dev)
    env.gait[1] = env.gait[0]
    env.action[1], env.prev_action[1] = env.mirror(env.action[0], A), env.mirror(env.prev_action[0], A)
    env.contact[1] = env.contact[0].flip(0)
    env.cmd_goal[1], env.gait_goal[1] = env.cmd[1], env.gait[1]
    env.t[1], env.cmd_t[1] = env.t[0], env.cmd_t[0]
    flip = torch.tensor([1.0, -1.0], device=env.dev)
    env.foot_prev[1] = env.foot_prev[0].flip(0) * flip
    env.point_prev[1] = env.point_prev[0].flip(0) * torch.tensor([1.0, -1.0, 1.0], device=env.dev)

    worst_obs, worst_r, asym = 0.0, 0.0, []
    for t in range(100):
        obs, _ = env.observe()
        e = (obs[1] - env.mirror(obs[0], P)).abs().max().item()
        a0 = act(obs[:1])
        asym.append((act(env.mirror(obs[:1], P)) - env.mirror(a0, A)).abs().mean().item())
        _, _, r, te, _, _ = env.step(torch.cat([a0, env.mirror(a0, A)]))
        er = abs(r[0] - r[1]).item()
        if t in (0, 1, 2, 5, 10, 20, 50, 99):
            print(f"  第 {t:3d} 步  观测差(最大) {e:.2e}  奖励 {r[0].item():.4f} / {r[1].item():.4f}  差 {er:.2e}  摔 {te.tolist()}")
        if t < 20:
            worst_obs, worst_r = max(worst_obs, e), max(worst_r, er)
print(f"前 20 步: 观测最大差 {worst_obs:.2e}, 奖励最大差 {worst_r:.2e}  ({'镜像正确' if worst_obs < 1e-2 and worst_r < 1e-2 else '有不对称, 检查映射或身体'})")
print(f"策略本身的不对称: |策略(镜像观测) - 镜像(策略(观测))| 平均 {sum(asym) / len(asym):.3f}(动作单位, 0 = 完全对称)")
