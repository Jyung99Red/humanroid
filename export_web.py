"""把走路策略和起身策略导出给网页互动版(web/play/): 网络权重、观测归一化、动作换算(默认姿势、幅度、关节限位、裁剪)、身体模型。
网页和 cpu_check.py 都只用这里导出的东西, 不再读 PyTorch 文件。

    python export_web.py --walk runs/sp2/policy.pt --getup runs/getup7/policy.pt
输出 web/play/policies.json(权重是小端 float32 的 base64)和 web/play/human50_pd.xml。
"""
import argparse
import base64
import json
import os
import shutil

import numpy as np
import torch

from combo import combo_config
from env import HumanoidEnv

p = argparse.ArgumentParser()
p.add_argument("--walk", default="runs/sp2/policy.pt")
p.add_argument("--getup", default="runs/getup7/policy.pt")
p.add_argument("--out", default="web/play")
p.add_argument("--turn-max", type=float, default=1.0, help="网页里转向指令的上限 rad/s(按 Q/E)。sp2 只练到 0.3, 但给 1.0 时实测能转 0.8、随机换指令 20 秒 1024 个不摔")
a = p.parse_args()


def b64(x):
    return base64.b64encode(np.ascontiguousarray(x, dtype="<f4").tobytes()).decode()


def net(s, path):
    """演员网络: 每层 (权重 [输出, 输入] 按行展开, 偏置); 隐藏层后面是 ELU, 最后一层线性"""
    sd = s["actor"]
    keys = sorted({k.rsplit(".", 1)[0] for k in sd}, key=lambda k: int(k))
    layers = [dict(n_in=int(sd[f"{k}.weight"].shape[1]), n_out=int(sd[f"{k}.weight"].shape[0]),
                   w=b64(sd[f"{k}.weight"].cpu().numpy()), b=b64(sd[f"{k}.bias"].cpu().numpy())) for k in keys]
    rms = s["obs_rms"]
    return dict(path=path, iteration=int(s["iteration"]), layout=[[n, int(d)] for n, d in s["obs_layout"]], layers=layers,
                mean=b64(rms["mean"].cpu().numpy()), inv_std=b64(1 / np.sqrt(rms["var"].cpu().numpy() + 1e-8)))


if __name__ == "__main__":
    ws = torch.load(a.walk, map_location="cuda:0", weights_only=False)
    gs = torch.load(a.getup, map_location="cuda:0", weights_only=False)
    cfg = combo_config(ws, gs)
    env = HumanoidEnv(1, **cfg)                              # 只用来取动作换算的常数(和训练完全一样的算法)
    m = env.mjm
    os.makedirs(a.out, exist_ok=True)
    xml_name = os.path.basename(cfg["xml"])
    shutil.copy(cfg["xml"], os.path.join(a.out, xml_name))
    t = lambda x: x.detach().cpu().numpy().astype(float).round(6).tolist()
    gait_mid = [sum(r) / 2 for r in env.gait_rng]             # 走路时步频、着地比例取训练范围的中点
    out = dict(
        xml=xml_name, timestep=float(m.opt.timestep), decimation=env.dec, root_z0=env.root_z0,
        qadr=env.qadr.tolist(), dadr=env.dadr.tolist(), q_def=t(env.q_def), scale=t(env.scale), lo=t(env.lo), hi=t(env.hi),
        clip_getup=t(env.act_clip), clip_walk=1.5,
        foot_bodies=env.foot_ids, torso_body=env.torso_id,
        cmd_range=dict(vx=list(cfg["cmd_vx"]), vy=list(cfg["cmd_vy"]), wz=[-a.turn_max, a.turn_max]), cmd_acc=cfg.get("cmd_acc"),
        cmd_ellipse=bool(cfg.get("cmd_ellipse", False)),          # 前进和侧移合起来限在椭圆里(和训练一样), 网页按键组合时照这个缩
        dodge=bool((cfg.get("balls") or {}).get("steer")),        # 走路策略是带着躲球规划练的: 网页 / cpu_check 也开(balls.py 的 BALL_NEAR 等常数)
        gait_walk=gait_mid, gait_stand=[gait_mid[0], 1.0],
        hold_steps=15, stood=dict(height=0.9, up=0.9),
        walk=net(ws, a.walk), getup=net(gs, a.getup),
    )
    path = os.path.join(a.out, "policies.json")
    json.dump(out, open(path, "w"), separators=(",", ":"))
    print(f"{path}  {os.path.getsize(path) / 1e6:.2f} MB | 走路 {a.walk}(第 {out['walk']['iteration']} 轮)、起身 {a.getup}(第 {out['getup']['iteration']} 轮) | 模型 {xml_name}")
