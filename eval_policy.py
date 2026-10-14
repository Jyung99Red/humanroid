"""测策略跟指令跟得怎么样: 每个人形给一个固定指令, 跑一段时间, 从真实接触量实际的步频、着地比例、速度。

    python eval_policy.py runs/xxx/policy.pt                    # 步态网格: 几种步频 x 几种着地比例
    python eval_policy.py runs/xxx/policy.pt --vx 1.0 --freq 0.8,0.9,1.0 --duty 0.5,0.6

量法(每个人形各自, 只算预热之后、摔倒之前):
  步频 = 每只脚每秒落地几次(离地至少 2 步 = 40 ms 后再接触才算一次, 防止一次着地里接触断续被多数); 着地比例 = 每只脚踩地的时间占比; 速度 = 朝向坐标系里的前进速度。
"""
import argparse

import torch

from policy import load_policy


@torch.no_grad()
def measure(env, act, cmd, gait, seconds=8.0, warm=2.0):
    """cmd [K,3], gait [K,2] 逐个人形固定。返回 dict(每个人形的 freq, duty, vx, wz, fell), 都是 [K] 张量。"""
    K = env.n
    env.cmd_period = env.ep_len = 10 ** 9                  # 测验期间不换指令、不超时
    idx = torch.arange(K, device=env.dev)
    env.reset_idx(idx)
    env.t[:] = 0
    env.cmd[:], env.gait[:] = cmd, gait
    steps, w = int(seconds / env.dt), int(warm / env.dt)
    fell = torch.zeros(K, dtype=torch.bool, device=env.dev)
    touch = torch.zeros(K, 2, device=env.dev)
    on = torch.zeros(K, 2, device=env.dev)
    vx = torch.zeros(K, device=env.dev)
    wz = torch.zeros(K, device=env.dev)
    off = torch.zeros(K, 2, device=env.dev)               # 每只脚连续离地了几步
    n = 0
    for t in range(steps):
        obs, _ = env.observe()
        _, _, _, te, _, _ = env.step(act(obs))
        fell |= te
        env.cmd[:], env.gait[:] = cmd, gait                 # 摔倒重置后会重抽指令, 改回来(摔了的人形不计入结果)
        if t >= w:
            live = (~fell).float()
            touch += ((env.contact > 0) & (off >= 2)).float() * live[:, None]
            on += env.contact * live[:, None]
            vx += env.m_vx * live
            wz += env.qvel[:, 5] * live
            n += 1
        off = torch.where(env.contact > 0, torch.zeros_like(off), off + 1)
    T = n * env.dt
    return dict(freq=touch.mean(1) / T, duty=on.mean(1) / n, vx=vx / n, wz=wz / n, fell=fell)


def grid_report(env, act, vx, freqs, duties, wz=0.0):
    combos = [(f, d) for f in freqs for d in duties]
    reps = env.n // len(combos)
    assert reps >= 1, f"人形数 {env.n} 少于指令组合数 {len(combos)}"
    K = reps * len(combos)
    g = torch.tensor([c for c in combos for _ in range(reps)], device=env.dev, dtype=torch.float32)
    g = torch.cat([g, g[:1].expand(env.n - K, 2)])          # 除不尽的补上第一种组合, 不计入
    cmd = torch.tensor([vx, 0.0, wz], device=env.dev).expand(env.n, 3).clone()
    r = measure(env, act, cmd, g)
    print(f"指令前进 {vx} m/s, 转向 {wz} rad/s; 每种组合 {reps} 个人形, 只算没摔的")
    print("指令 步频(步/分) 着地比例 | 实际 步频(步/分) 着地比例 前进速度 | 摔倒")
    for i, (f, d) in enumerate(combos):
        s = slice(i * reps, (i + 1) * reps)
        ok = ~r["fell"][s]
        m = lambda k: r[k][s][ok].mean().item() if ok.any() else float("nan")
        print(f"     {f * 120:5.0f}      {d:4.2f}    |      {m('freq') * 120:5.0f}      {m('duty'):4.2f}    {m('vx'):5.2f}   | {int((~ok).sum())}/{reps}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("policy")
    ap.add_argument("--worlds", type=int, default=1024)
    ap.add_argument("--vx", type=float, default=1.0)
    ap.add_argument("--freq", default="", help="步频列表(周期/秒), 逗号分隔; 不给就用训练范围的两端和中点")
    ap.add_argument("--duty", default="", help="着地比例列表; 不给就用训练范围的两端和中点")
    a = ap.parse_args()
    env, act, meta = load_policy(a.policy, a.worlds, random_yaw=False)
    pick = lambda s, rng: [float(v) for v in s.split(",")] if s else sorted({rng[0], (rng[0] + rng[1]) / 2, rng[1]})
    print(f"{a.policy} (第 {meta['iteration']} 轮)")
    grid_report(env, act, a.vx, pick(a.freq, env.gait_rng[0]), pick(a.duty, env.gait_rng[1]))
