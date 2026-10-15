"""按步态时钟量每只脚的俯仰: 直接跑策略(不用导出回放), 左右脚各按自己的时钟分 10 段。
check_feet 给的是整段平均, 这里看"脚跟着地 -> 踩平 -> 蹬地 -> 摆腿"每个相位脚是什么姿态, 左右对不对得上。

    python foot_phase.py runs/xxx/policy.pt [更多策略...] [--freq 0.9 --duty 0.6 --vx 1.0]

俯仰正 = 脚尖朝下。右脚在时钟 [0, 着地比例) 该踩地, 左脚晚半个周期(这里已经换算成左脚自己的时钟)。
"""
import argparse

import numpy as np
import torch

from env import rotate
from policy import load_policy


@torch.no_grad()
def measure(path, n=64, T=600, f=0.9, D=0.6, vx=1.0):
    env, act, meta = load_policy(path, n, seed=3, random_yaw=False, cmd_vx=(vx, vx), cmd_vy=(0.0, 0.0), cmd_wz=(0.0, 0.0),
                                 cmd_freq=(f, f), cmd_duty=(D, D))
    env.ep_len = env.cmd_period = 10 ** 9
    idx = torch.arange(n, device=env.dev)
    env.t[:] = 0
    env.reset_idx(idx)
    env._sample_cmd(idx)
    rec, fell = [], torch.zeros(n, dtype=torch.bool, device=env.dev)
    for t in range(T):
        obs, _ = env.observe()
        _, _, _, te, _, _ = env.step(act(obs))
        fell |= te
        if t < 150:                                           # 前 3 秒不算
            continue
        fq = env.xquat[:, env.foot_ids]
        pitch = torch.rad2deg(torch.asin((-2 * (fq[..., 1] * fq[..., 3] - fq[..., 0] * fq[..., 2])).clamp(-1, 1)))
        ph = torch.stack([env.phase, env.phase - 0.5], 1) % 1.0
        pts = torch.stack([torch.stack([env.xpos[:, b] + rotate(env.xquat[:, b], p) for p in env.foot_pts], 1) for b in env.foot_ids], 1)
        rec.append(torch.cat([pitch, ph, env.contact, pts[..., 2].reshape(n, 4)], 1).cpu().numpy())
    R = np.stack(rec)[:, ~fell.cpu().numpy()]                  # [时间, 没摔的人形, 2 俯仰 + 2 时钟 + 2 接触 + 4 脚跟/脚尖高度]
    print(f"== {path}(第 {meta['iteration']} 轮)  指令 {vx} m/s, 步频 {f * 120:.0f} 步/分, 着地比例 {D} | 摔倒 {int(fell.sum())}/{n}")
    print("  时钟        " + " ".join(f"{b / 10:5.1f}" for b in range(10)) + " | 着地时: 平均俯仰 脚跟离地 脚尖离地 脚尖>2cm")
    for k, side in enumerate(("右脚", "左脚")):
        pitch, ph, con = R[..., k], R[..., 2 + k], R[..., 4 + k] > 0
        heel, toe = R[..., 6 + 2 * k], R[..., 7 + 2 * k]
        bins = [pitch[(ph >= b / 10) & (ph < (b + 1) / 10)].mean() for b in range(10)]
        print(f"  {side}俯仰 °   " + " ".join(f"{b:+5.0f}" for b in bins) +
              f" | {pitch[con].mean():+6.1f}° {heel[con].mean() * 100:5.1f} cm {toe[con].mean() * 100:5.1f} cm {np.mean(toe[con] > 0.02) * 100:4.0f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("policies", nargs="+")
    ap.add_argument("--freq", type=float, default=0.9, help="步频(周期/秒), 步数/分 = 120 x 步频")
    ap.add_argument("--duty", type=float, default=0.6, help="着地比例")
    ap.add_argument("--vx", type=float, default=1.0)
    a = ap.parse_args()
    for p in a.policies:
        measure(p, f=a.freq, D=a.duty, vx=a.vx)
