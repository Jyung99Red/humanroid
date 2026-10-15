"""回归测验: 一条命令测一个策略, 和基线比, 防止改训练后悄悄退步。

    python regress.py runs/xxx/policy.pt --save                               # 测完存成 runs/xxx/regress.json
    python regress.py runs/yyy/policy.pt --baseline runs/xxx/regress.json     # 和基线比, 变差的项标出来

四组测验(指令都取策略自己训练时的范围, 所以范围变了测验也跟着变; 和基线比时只比两边都有的项):
  1. 速度和转向: 前进速度取范围两端和中点, 转向取范围两端, 每种指令 128 个人形跑 8 秒(前 2 秒不算)。
  2. 步态指令: 步频 x 着地比例 各取两端和中点, 共 9 种。量法见 eval_policy.py。
  3. 姿态: 指令取中间值走 10 秒, 量骨盆、两脚间距、抬脚、脚的滚动、头的稳定, 和真人范围对照(量法见 check_feet.py)。
  4. 稳定性: 1024 个人形, 每 2 秒在训练范围里随机换一次指令, 跑 20 秒, 数摔倒。
判定: 误差类(越小越好)比基线大出容差就算退步; 姿态类看离真人范围多远, 比基线远出容差就算退步。
"""
import argparse
import json
import os

import mujoco
import numpy as np
import torch

from check_feet import gait_stats
from eval_policy import measure
from policy import load_policy

# 指标: 键 -> (说明, 类别, 容差, 真人范围)。类别 err = 越小越好; range = 离真人范围越近越好; info = 只显示
METRICS = {
    "vx_err": ("前进速度误差(最大, m/s)", "err", 0.05, None),
    "back_err": ("倒退速度误差(最大, m/s)", "err", 0.05, None),
    "stand_move": ("站定时漂移 m/s", "err", 0.03, None),
    "stand_steps": ("站定时踏步 步/分", "err", 3, None),
    "wz_err": ("转向误差(最大, rad/s)", "err", 0.05, None),
    "track_fall": ("测验 1、2 摔倒(人)", "err", 0, None),
    "freq_err": ("步频误差(最大, 步/分)", "err", 3, None),
    "duty_err": ("着地比例误差(最大)", "err", 0.03, None),
    "pelvis_roll": ("骨盆侧倾 ±°", "range", 1.0, (4, 5)),
    "pelvis_yaw": ("骨盆扭转 ±°", "range", 1.0, (4, 6)),
    "width_cm": ("落地时两脚左右间距 cm", "range", 1.0, (8, 12)),
    "head_pitch": ("头俯仰晃动 ±°", "range", 1.0, (0, 4)),
    "clear_cm": ("抬脚高度 cm", "info", 0, None),
    "land_toe_up": ("落地瞬间脚尖朝上 °(脚跟先着地)", "info", 0, None),
    "push_off": ("支撑末段脚尖朝下 °(蹬地)", "info", 0, None),
    "foot_lr_diff": ("左右脚俯仰差 °", "err", 3.0, None),
    "slip_cms": ("踩地打滑 cm/s", "err", 2.0, None),
    "random_fall": ("随机换指令 20 秒摔倒(人/1024)", "err", 3, None),
}


def mid(r):
    return (r[0] + r[1]) / 2


FWD_POINTS, BACK_POINTS = (0.4, 0.8, 1.2), (-0.3, -0.6)   # 速度测点(只取训练范围里的), 外加范围两端
WALK_V = 0.8                                               # 转向、步态网格测验时的前进速度(不在训练范围里就取中点)


def run_tracking(env, act):
    """测验 1 和 2: 每个人形一个固定指令。返回 (指标, 明细行)。
    测点固定(而不是取范围两端和中点), 这样放宽指令范围后, 前进、转向、步态这几项还能和旧基线逐项比。"""
    (vx_r, _, wz_r), (f_r, d_r) = env.cmd_rng, env.gait_rng
    g_mid = (mid(f_r), mid(d_r))
    inside = lambda v: vx_r[0] <= v <= vx_r[1]
    walk = WALK_V if inside(WALK_V) else mid(vx_r)
    pts = sorted({v for v in FWD_POINTS + BACK_POINTS if inside(v)} | {vx_r[0], vx_r[1]})
    combos = [("vx" if v > 0 else "back", (v, 0.0, 0.0), g_mid) for v in pts if v != 0]
    combos += [("wz", (walk, 0.0, w), g_mid) for w in sorted({wz_r[0], wz_r[1]}) if w != 0]
    combos += [("gait", (walk, 0.0, 0.0), (f, d)) for f in sorted({f_r[0], mid(f_r), f_r[1]}) for d in sorted({d_r[0], mid(d_r), d_r[1]})]
    if env.stand_prob > 0:
        combos.append(("stand", (0.0, 0.0, 0.0), (mid(f_r), 1.0)))
    reps = env.n // len(combos)
    cmd = torch.tensor([c for _, c, _ in combos for _ in range(reps)], device=env.dev, dtype=torch.float32)
    gt = torch.tensor([g for _, _, g in combos for _ in range(reps)], device=env.dev, dtype=torch.float32)
    pad = env.n - len(cmd)
    cmd, gt = torch.cat([cmd, cmd[:1].expand(pad, 3)]), torch.cat([gt, gt[:1].expand(pad, 2)])
    r = measure(env, act, cmd, gt)
    rows, out = [], dict(vx_err=0.0, wz_err=0.0, freq_err=0.0, duty_err=0.0, track_fall=0)
    for i, (kind, c, g) in enumerate(combos):
        s = slice(i * reps, (i + 1) * reps)
        ok = ~r["fell"][s]
        m = lambda k: r[k][s][ok].mean().item() if ok.any() else float("nan")
        out["track_fall"] += int((~ok).sum())
        if kind in ("vx", "back"):
            key = "vx_err" if kind == "vx" else "back_err"
            out[key] = max(out.get(key, 0.0), abs(m("vx") - c[0]))
        elif kind == "wz":
            out["wz_err"] = max(out["wz_err"], abs(m("wz") - c[2]))
        elif kind == "gait":
            out["freq_err"] = max(out["freq_err"], abs(m("freq") - g[0]) * 120)
            out["duty_err"] = max(out["duty_err"], abs(m("duty") - g[1]))
        else:                                              # 站定: 不该走, 也不该原地踏步
            out["stand_move"], out["stand_steps"] = abs(m("vx")), m("freq") * 120
        rows.append(f"  指令 vx {c[0]:4.2f} wz {c[2]:+4.2f} 步频 {g[0] * 120:3.0f} 着地 {g[1]:.2f} | 实际 vx {m('vx'):4.2f} wz {m('wz'):+5.2f}"
                    f" 步频 {m('freq') * 120:3.0f} 着地 {m('duty'):.2f} | 摔 {int((~ok).sum())}/{reps}")
    return out, rows


@torch.no_grad()
def run_pose(env, act, K=32, seconds=10.0):
    """测验 3: 中间指令走 10 秒, 记下前 K 个人形每帧的刚体位姿和接触, 交给 check_feet.gait_stats。"""
    (vx_r, _, _), (f_r, d_r) = env.cmd_rng, env.gait_rng
    vx = 1.0 if vx_r[0] <= 1.0 <= vx_r[1] else mid(vx_r)
    cmd = torch.tensor([vx, 0.0, 0.0], device=env.dev).expand(env.n, 3).clone()
    gt = torch.tensor([mid(f_r), mid(d_r)], device=env.dev).expand(env.n, 2).clone()
    env.cmd_period = env.ep_len = 10 ** 9
    env.reset_idx(torch.arange(env.n, device=env.dev))
    env.t[:] = 0
    env.cmd[:], env.gait[:] = cmd, gt
    T = int(seconds / env.dt)
    xp, xq, con = (torch.zeros(T, K, *shape, device=env.dev) for shape in ((env.xpos.shape[1], 3), (env.xpos.shape[1], 4), (2,)))
    fell = torch.zeros(env.n, dtype=torch.bool, device=env.dev)
    for t in range(T):
        xp[t], xq[t], con[t] = env.xpos[:K], env.xquat[:K], env.contact[:K]
        obs, _ = env.observe()
        _, _, _, te, _, _ = env.step(act(obs))
        fell |= te
        env.cmd[:], env.gait[:] = cmd, gt
    names = [mujoco.mj_id2name(env.mjm, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(env.mjm.nbody)]
    keep = [k for k in range(K) if not fell[k]]
    xp, xq, con = xp.cpu().numpy(), xq.cpu().numpy(), con.cpu().numpy()
    st = gait_stats([(xp[:, k], xq[:, k], con[:, k]) for k in keep], names, env.dt)
    return {k: float(st[k]) for k in ("pelvis_roll", "pelvis_yaw", "width_cm", "head_pitch", "clear_cm", "land_toe_up", "push_off", "slip_cms",
                                         "foot_lr_diff")}


@torch.no_grad()
def run_random(env, act, seconds=20.0):
    """测验 4: 每 2 秒在训练范围里随机换指令, 数摔倒的人形。"""
    env.cmd_period, env.ep_len = int(2.0 / env.dt), 10 ** 9
    idx = torch.arange(env.n, device=env.dev)
    env.reset_idx(idx)
    env.t[:] = 0
    env._sample_cmd(idx)
    fell = torch.zeros(env.n, dtype=torch.bool, device=env.dev)
    for _ in range(int(seconds / env.dt)):
        obs, _ = env.observe()
        _, _, _, te, _, _ = env.step(act(obs))
        fell |= te
    return dict(random_fall=int(fell.sum()))


def verdict(key, val, base):
    _, kind, tol, rng = METRICS[key]
    dist = lambda v: max(rng[0] - v, 0.0, v - rng[1]) if rng else v
    human = "" if not rng else ("在真人范围" if dist(val) == 0 else ("偏小" if val < rng[0] else "偏大"))
    if base is None or kind == "info" or np.isnan(val):
        return human
    worse = dist(val) > dist(base) + tol
    return ("退步 ✗" if worse else "持平/更好") + (f" · {human}" if human else "")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("policy")
    ap.add_argument("--baseline", default="", help="基线 regress.json")
    ap.add_argument("--save", action="store_true", help="结果存到策略所在目录的 regress.json")
    a = ap.parse_args()
    env, act, meta = load_policy(a.policy, 1024, random_yaw=False)
    print(f"{a.policy}(第 {meta['iteration']} 轮)  指令范围 vx {env.cmd_rng[0]} wz {env.cmd_rng[2]} 步频 {env.gait_rng[0]} 着地 {env.gait_rng[1]}")
    res, rows = run_tracking(env, act)
    print("测验 1、2 明细:")
    print("\n".join(rows))
    res.update(run_pose(env, act))
    res.update(run_random(env, act))
    base = json.load(open(a.baseline, encoding="utf-8"))["metrics"] if a.baseline else {}
    print(f"\n{'指标':32s} {'本次':>8s} {'基线':>8s}  判定")
    bad = 0
    for key, (label, kind, tol, rng) in METRICS.items():
        if key not in res:                                 # 这个策略没练过的项(例如没开站定)
            continue
        b = base.get(key)
        v = verdict(key, res[key], b)
        bad += v.startswith("退步")
        print(f"{label:32s} {res[key]:8.2f} {('' if b is None else f'{b:8.2f}'):>8s}  {v}")
    if a.baseline:
        print(f"\n{'有 ' + str(bad) + ' 项退步' if bad else '没有退步'}(基线 {a.baseline})")
    if a.save:
        path = os.path.join(os.path.dirname(a.policy), "regress.json")
        json.dump(dict(policy=a.policy, iteration=meta["iteration"], metrics=res), open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"已存 {path}")
