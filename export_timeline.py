"""把一条训练链(顺着 config.json 的 resume 往回找)的所有快照导出成一条"时间线", 回放页面用轮数滑块逐个看。

    python export_timeline.py runs/knee3 --name base --seq "0:1.0,0,0; 4:0,0,0,0.925,1; 6.5:-0.4,0,0; 10.5:0,0,0,0.925,1" --steps 700
    python export_timeline.py runs/knee3 --name base --from 3000 --every 100          # 只要第 3000 轮以后, 每 100 轮一个

链: runs/knee3 <- knee2 <- stand3 <- stand2 <- stand1 <- sym2 <- sym1, 每段取它自己 snaps/ 里的快照(子段开始以后的父段快照不要)。
同一段的快照放进同一个环境一起跑(每个快照占 --worlds 个世界, 用各自的网络), 指令序列和 export.py 一样。
输出 web/replays/tl_<名字>/: manifest.json(刚体、几何体、每个快照的轮数和每个世界的帧数) + 每个快照一个 it_XXXXX.bin。
.bin 是小端二进制, 按世界依次放: 位置 int16(毫米)、朝向 int16(×32767, wxyz)、前进速度 int16(毫米/秒)、两脚接触 uint8。
再跑一次时, 指令序列/世界数/步数没变的快照直接复用, 只跑新的。
--walk runs/sp2/policy.pt: 链是起身策略时用, 每个快照和这个走路策略切换(combo.py), --lying 比例的人形从倒地开局, 每个快照的开局姿态相同:
    python export_timeline.py runs/getup3 --name getup3 --walk runs/sp2/policy.pt --lying 1 --worlds 4 --steps 500 --seq "0:0,0,0,0.925,1; 5:0.6,0,0" --solo
--solo: 回放列表只留这条时间线, 其他回放的文件移到 web/replays/archive/(不删), 列表项记在 archive/index.json, 放回去就能再看。
"""
import argparse
import json
import os
import shutil
import time

import mujoco
import numpy as np
import torch

from env import HumanoidEnv
from export import fixed_cmd, kinematics, model_info, parse_seq, rollout
from combo import Switcher, combo_config
from policy import env_config, load_actor

POS_SCALE, QUAT_SCALE, VX_SCALE = 1000, 32767, 1000


def chain(run):
    """从根到 run 的各段目录"""
    out = []
    while run and run not in out:
        out.insert(0, run)
        cfg = json.load(open(os.path.join(run, "config.json"), encoding="utf-8"))
        parent = os.path.dirname(cfg.get("resume") or "")
        run = parent if parent and os.path.normpath(parent) != os.path.normpath(run) else None
    return out


def snaps_of(run):
    d = os.path.join(run, "snaps")
    return sorted((int(f[3:8]), os.path.join(d, f)) for f in os.listdir(d) if f.startswith("it_") and f.endswith(".pt")) if os.path.isdir(d) else []


def encode(xp, xq, vx, con):
    big = np.abs(xp).max()
    assert big * POS_SCALE < 32767, f"位置 {big:.1f} m 超出 int16 毫米的范围"
    return b"".join([np.round(xp * POS_SCALE).astype("<i2").tobytes(), np.round(xq * QUAT_SCALE).astype("<i2").tobytes(),
                     np.round(np.clip(vx, -32, 32) * VX_SCALE).astype("<i2").tobytes(), con.astype(np.uint8).tobytes()])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", help="链的最后一段, 例如 runs/knee3")
    p.add_argument("--name", required=True, help="输出目录 web/replays/tl_<name>")
    p.add_argument("--label", default="", help="回放列表里的名字; 不给就用链的首尾")
    p.add_argument("--worlds", type=int, default=3, help="每个快照跑几个人形")
    p.add_argument("--steps", type=int, default=700)
    p.add_argument("--seq", default="0:1.0,0,0; 4:0,0,0,0.925,1; 6.5:-0.4,0,0; 10.5:0,0,0,0.925,1")
    p.add_argument("--push", default="", help="按时间推躯干, 同 export.py --push")
    p.add_argument("--cmd", default="1.0,0,0")
    p.add_argument("--gait", default="")
    p.add_argument("--hold-heading", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--from", dest="it_from", type=int, default=0)
    p.add_argument("--to", dest="it_to", type=int, default=10 ** 9)
    p.add_argument("--every", type=int, default=0, help="只取轮数是它倍数的快照; 0 = 全部")
    p.add_argument("--solo", action="store_true", help="回放列表只留这条时间线, 其他回放移到 web/replays/archive/")
    p.add_argument("--walk", default="", help="链是起身策略时: 和这个走路策略切换")
    p.add_argument("--lying", type=float, default=1.0, help="配合 --walk: 多大比例的人形从倒地开局")
    p.add_argument("--batch", type=int, default=32, help="一个环境里最多同时跑几个快照")
    a = p.parse_args()

    runs = chain(os.path.normpath(a.run).replace("\\", "/"))
    picked = []                                         # [(轮数, 段, 快照路径)]
    for i, r in enumerate(runs):
        nxt = snaps_of(runs[i + 1]) if i + 1 < len(runs) else []
        stop = nxt[0][0] if nxt else 10 ** 9           # 子段开始以后, 父段自己再练出来的快照不属于这条链
        picked += [(it, r, f) for it, f in snaps_of(r) if it < stop]
    picked = [x for x in picked if a.it_from <= x[0] <= a.it_to and (not a.every or x[0] % a.every == 0)]
    assert picked, "没有可导出的快照"

    out_dir = os.path.join("web", "replays", f"tl_{a.name}")
    os.makedirs(out_dir, exist_ok=True)
    params = dict(seq=a.seq, push=a.push, cmd=a.cmd, gait=a.gait, worlds=a.worlds, steps=a.steps, hold_heading=a.hold_heading, seed=a.seed)
    if a.walk:
        params.update(walk=a.walk, lying=a.lying)
        walk_s = torch.load(a.walk, map_location="cuda:0", weights_only=False)
    man_path = os.path.join(out_dir, "manifest.json")
    old = json.load(open(man_path, encoding="utf-8")) if os.path.exists(man_path) else {}
    reuse = {s["it"]: s for s in old.get("snaps", [])
             if old.get("params") == params and os.path.exists(os.path.join(out_dir, s["file"]))}

    seq, pushes = parse_seq(a.seq), parse_seq(a.push)
    fix, (vx0, vy0, wz0) = fixed_cmd(seq, a.cmd, a.gait)
    xml = None
    entries, t0 = {it: reuse[it] for it, _, _ in picked if it in reuse}, time.time()
    todo = [x for x in picked if x[0] not in reuse]
    print(f"链: {' -> '.join(runs)} | 快照 {len(picked)} 个({picked[0][0]}~{picked[-1][0]} 轮), 其中 {len(todo)} 个要跑")
    for r in runs:
        mine = [x for x in todo if x[1] == r]
        for b in range(0, len(mine), a.batch):
            part = mine[b:b + a.batch]
            states = [torch.load(f, map_location="cuda:0", weights_only=False) for _, _, f in part]
            K, S = a.worlds, len(part)
            if a.walk:                                  # 起身策略的链: 每块世界用一个快照起身, 站起来后都交给同一个走路策略
                cfg = combo_config(walk_s, states[0], seed=a.seed, random_yaw=False, getup_stand_frac=1 - a.lying, **fix)
                env = HumanoidEnv(S * K, **cfg)
                act = Switcher(env, walk_s, states, tile=True)
            else:
                cfg = env_config(states[0], seed=a.seed, random_yaw=False, **fix)
                env = HumanoidEnv(S * K, **cfg)
                acts = [load_actor(s, env) for s in states]

                def act(obs):
                    return torch.cat([f(obs[i * K:(i + 1) * K]) for i, f in enumerate(acts)])
            xml = xml or cfg.get("xml", "models/humanoid_pd.xml")
            assert cfg.get("xml", "models/humanoid_pd.xml") == xml, "链上各段的身体模型不一样, 不能放进同一条时间线"

            q_hist, vx_hist, c_hist, end, fell = rollout(env, act, a.steps, seq, wz0, bool(a.gait), a.hold_heading, pushes)
            m, bodies, geoms = model_info(xml)
            d = mujoco.MjData(m)
            for i, (it, _, _) in enumerate(part):
                blob, worlds = [], []
                for k in range(K):
                    w = i * K + k
                    n = int(end[w])
                    xp, xq = kinematics(m, d, q_hist[:n, w])
                    blob.append(encode(xp, xq, vx_hist[:n, w], c_hist[:n, w]))
                    worlds.append(dict(n=n, fell=bool(fell[w])))
                fname = f"it_{it:05d}.bin"
                open(os.path.join(out_dir, fname), "wb").write(b"".join(blob))
                entries[it] = dict(it=it, run=os.path.basename(r), file=fname, worlds=worlds)
            print(f"  {r}: 第 {part[0][0]}~{part[-1][0]} 轮 {S} 个快照, 累计 {time.time() - t0:.0f} 秒", flush=True)
            del env

    m, bodies, geoms = model_info(xml or old.get("xml", "models/human50_pd.xml"))
    snaps = [entries[it] for it, _, _ in picked]
    label = a.label or f"时间线 · {os.path.basename(runs[0])}→{os.path.basename(runs[-1])} · {snaps[0]['it']}~{snaps[-1]['it']}轮"
    json.dump(dict(type="timeline", label=label, params=params, xml=xml or old.get("xml"), dt=0.02, steps=a.steps, nworld=a.worlds,
                   scale=dict(pos=POS_SCALE, quat=QUAT_SCALE, vx=VX_SCALE), runs=[os.path.basename(r) for r in runs],
                   version=int(time.time()), pushes=[[k, *f[:2]] for k, f in pushes], push_dur=0.2, bodies=bodies, geoms=geoms, snaps=snaps),
              open(man_path, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))

    idx_path = os.path.join("web", "replays", "index.json")
    idx = json.load(open(idx_path, encoding="utf-8")) if os.path.exists(idx_path) else []
    file = f"tl_{a.name}/manifest.json"
    idx = [e for e in idx if e["file"] != file]
    idx.append(dict(file=file, label=label, iteration=snaps[-1]["it"], timeline=True))
    if a.solo:                                          # 其他回放归档: 文件(时间线是整个目录)移走, 列表项记到 archive/index.json
        arc_dir = os.path.join("web", "replays", "archive")
        os.makedirs(arc_dir, exist_ok=True)
        arc_path = os.path.join(arc_dir, "index.json")
        arc = json.load(open(arc_path, encoding="utf-8")) if os.path.exists(arc_path) else []
        for e in idx[:-1]:
            src = os.path.join("web", "replays", e["file"].split("/")[0])
            if os.path.exists(src):
                dst = os.path.join(arc_dir, os.path.basename(src))
                if os.path.exists(dst):
                    shutil.rmtree(dst) if os.path.isdir(dst) else os.remove(dst)
                shutil.move(src, dst)
            arc = [x for x in arc if x["file"] != e["file"]] + [e]
        json.dump(arc, open(arc_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"其他 {len(idx) - 1} 条回放移到 {arc_dir}/")
        idx = idx[-1:]
    json.dump(idx, open(idx_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    size = sum(os.path.getsize(os.path.join(out_dir, s["file"])) for s in snaps) / 1e6
    falls = sum(w["fell"] for s in snaps for w in s["worlds"])
    print(f"{man_path}  [{label}]  {len(snaps)} 个快照, 共 {size:.1f} MB | 摔倒的人形 {falls}/{len(snaps) * a.worlds}")
