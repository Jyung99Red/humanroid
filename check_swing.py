"""量一份回放里的摆臂: 上臂前后摆的角度幅度, 以及手臂和同侧/对侧腿的相位关系。
用法: python check_swing.py web/replays/d_final.json [更多回放...]"""
import json
import sys

import numpy as np


def rot_inv(q, v):  # 把世界系向量转到四元数 q(wxyz) 的机体系
    w, u = q[..., :1], -q[..., 1:]
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


for path in sys.argv[1:]:
    d = json.load(open(path, encoding="utf-8"))
    names = [b["name"] for b in d["bodies"]]
    nb, dt = len(names), d["meta"]["dt"]
    amps, c_same, c_opp, vxs, falls = [], [], [], [], 0
    for w in d["worlds"]:
        xp = np.array(w["xpos"]).reshape(-1, nb, 3)
        xq = np.array(w["xquat"]).reshape(-1, nb, 4)
        falls += bool(w["fell"])
        vxs.append(np.mean(w["vx"]))
        if len(xp) < 200:
            continue
        s = slice(100, None)                                   # 去掉起步的 2 秒
        t = names.index("torso")
        rel = lambda b: rot_inv(xq[s, t], xp[s, names.index(b)] - xp[s, t])     # 躯干系里的位置
        pitch, foot = {}, {}
        for side in ("right", "left"):
            v = rel("lower_arm_" + side) - rel("upper_arm_" + side)             # 肩到肘
            pitch[side] = np.degrees(np.arctan2(v[:, 0], -v[:, 2]))            # 上臂往前抬的角度
            foot[side] = rel("foot_" + side)[:, 0]
            amps.append((np.percentile(pitch[side], 95) - np.percentile(pitch[side], 5)) / 2)
        for a, b in (("right", "left"), ("left", "right")):
            c_same.append(np.corrcoef(pitch[a], foot[a])[0, 1])
            c_opp.append(np.corrcoef(pitch[a], foot[b])[0, 1])
    lab = d["meta"]["label"]
    if not amps:
        print(f"{lab}: 回放太短, 没法量")
        continue
    print(f"{lab}: 上臂前后摆幅 ±{np.mean(amps):.1f}° (各臂 {np.round(amps, 1).tolist()}) | "
          f"和同侧脚的相关 {np.mean(c_same):+.2f}, 和对侧脚 {np.mean(c_opp):+.2f} | 平均速度 {np.mean(vxs):.2f} m/s | 摔倒 {falls}/{len(d['worlds'])}")
