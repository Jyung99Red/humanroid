"""起步有多快: 站 1.5 秒后突然给一个指令(阶跃), 量躯干沿指令方向挪开多远(0.3 / 0.5 / 0.8 秒)和 0.5~1.5 秒的平均速度。
在 MuJoCo CPU 版里跑网页用的导出(export_web.py 的 policies.json), 和网页里的手感一致; 不同策略用同一串指令比。
躲球要的是前 0.5 秒: 球 14 m/s 从 7 米外飞来只要 0.5 秒, 躲开一个正对躯干的球要横着让开约 25~30 cm。

    python move_probe.py                                   # 网页现在用的策略
    python move_probe.py --cfg 某个目录/policies.json --acc 3   # 指令加速度上限改成 3 m/s^2(默认用导出时的)
"""
import argparse
import json
import math
import os

import numpy as np

from cpu_check import Controller

CMDS = [("侧移 0.4", (0, 0.4, 0)), ("侧移 0.8", (0, 0.8, 0)), ("倒退 0.6", (-0.6, 0, 0)), ("倒退 1.0", (-1.0, 0, 0)), ("前进 1.2", (1.2, 0, 0))]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="web/play/policies.json")
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--acc", type=float, default=0.0, help="指令加速度上限 m/s^2; 0 = 用导出时的(训练时的)")
    a = p.parse_args()
    cfg = json.load(open(a.cfg))
    ctl = Controller(cfg, os.path.join(os.path.dirname(a.cfg), cfg["xml"]))
    if a.acc:
        ctl.c["cmd_acc"] = a.acc
    T = cfg["torso_body"]
    print(f"{cfg['walk']['path']}(第 {cfg['walk']['iteration']} 轮) | 指令加速度上限 {ctl.c['cmd_acc']} m/s² | 每个指令 {a.trials} 次")
    print("指令       | 沿指令方向挪开: 0.3 s   0.5 s   0.8 s | 0.5~1.5 s 平均速度 | 摔")
    for name, cmd in CMDS:
        rows = []
        for i in range(a.trials):
            ctl.reset_standing()
            for _ in range(75):
                ctl.step()
            q = ctl.d.qpos[3:7]
            yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
            n = math.hypot(cmd[0], cmd[1])
            ux, uy = cmd[0] / n, cmd[1] / n
            e = np.array([math.cos(yaw) * ux - math.sin(yaw) * uy, math.sin(yaw) * ux + math.cos(yaw) * uy])   # 指令方向(世界系)
            p0, f0 = ctl.d.xpos[T][:2].copy(), ctl.n_falls
            ctl.set_cmd(*cmd)
            d = []
            for _ in range(75):
                ctl.step()
                d.append(float((ctl.d.xpos[T][:2] - p0) @ e))
            rows.append([d[14], d[24], d[39], d[74] - d[24], ctl.n_falls - f0])
        r = np.array(rows)
        m = r.mean(0)
        print(f"{name:10s} | {m[0] * 100:7.1f} cm {m[1] * 100:6.1f} {m[2] * 100:6.1f} | {m[3]:5.2f} m/s | {int(r[:, 4].sum())}/{a.trials}")
