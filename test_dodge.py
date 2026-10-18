"""躲球测验: 开着发球机(env.BALL_DEFAULT, 速度固定为几档)站着 / 走着, 数被打中的比例和摔倒。策略没练过球(比如 sp4)也能测, 当基线。

    python test_dodge.py runs/sp4/policy.pt
    python test_dodge.py runs/dodge1/policy.pt --speeds 9,12,15
    python test_dodge.py runs/sp4/policy.pt --ball steer=1           # 只靠手写的躲球规划(底层没练过)
"""
import argparse

import torch

from policy import load_policy

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("policy")
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--secs", type=float, default=15.0)
    p.add_argument("--speeds", default="9,12,15")
    p.add_argument("--ball", default="", help="覆盖发球机设置(同 train.py --ball), 例如 dist=7:9,steer=1")
    a = p.parse_args()
    s = torch.load(a.policy, map_location="cuda:0", weights_only=False)
    balls = dict(s["env"].get("balls") or {})                     # 用策略训练时的发球机设置(没练过球的用默认), 只改速度
    balls.update((k, tuple(float(x) for x in v.split(":")) if ":" in v else float(v)) for k, v in (kv.split("=") for kv in a.ball.split(",") if kv))
    env, act, meta = load_policy(a.policy, a.n, balls=balls, push_force=0.0, cmd_period=10 ** 9, ep_len=10 ** 9, cmd_stand=0.0, seed=3)
    dev = env.dev
    (f_r, d_r) = env.gait_rng
    print(f"{a.policy}(第 {meta['iteration']} 轮) | 发球机: 距离 {env.balls['dist']} m, 出手高 {env.balls['muzzle']} m, 打头 {env.balls['head']:.0%}, 散布 {env.balls['spread']} m, "
          f"间隔 {env.balls['every']} s, 躲球规划 {'开' if env.balls['steer'] else '关'} | 每格 {a.n} 个人形 x {a.secs:g} 秒")
    for name, cmd, duty in (("站着", (0.0, 0.0, 0.0), 1.0), ("走 0.8", (0.8, 0.0, 0.0), sum(d_r) / 2)):
        row = []
        for v in (float(x) for x in a.speeds.split(",")):
            env.balls["speed"] = (v, v)
            env.reset_idx(torch.arange(a.n, device=dev))
            c = torch.tensor(cmd, device=dev).expand(a.n, 3).clone()
            g = torch.tensor([sum(f_r) / 2, duty], device=dev).expand(a.n, 2).clone()
            env.set_cmd(c, g)
            env.ball_thrown_sum.zero_()
            env.ball_hit_sum.zero_()
            falls = 0
            obs, _ = env.observe()
            for t in range(int(a.secs / env.dt)):
                env.set_cmd(c, g)
                obs, _, _, te, _, _ = env.step(act(obs))
                falls += int(te.sum())
            n_th, n_hit = float(env.ball_thrown_sum), float(env.ball_hit_sum)
            row.append(f"{v:g} m/s 打中 {n_hit / max(n_th, 1) * 100:4.1f}% ({n_th:.0f} 个) 摔 {falls}")
        print(f"{name:6s} | " + " | ".join(row))
