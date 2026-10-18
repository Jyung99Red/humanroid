"""倒地姿态库: 让人形从站立被猛推一下、随机乱动着倒下, 躺稳后记下 qpos, 起身训练从这里抽开局。
    python make_fallen.py --xml models/human50_pd.xml --out models/fallen_poses.pt
不走策略, 直接控制 PD 目标: 0~0.4 秒推躯干 250~500 N(随机方向) + 随机动作; 到 2.5 秒每 0.5 秒换一次随机动作;
之后 2 秒一半保持最后的动作、一半"放松"(PD 目标 = 当前角度)。最后只留身体碰地、几乎不动的。
按躯干朝向分类: 仰躺(胸口朝上)、趴(胸口朝下)、侧躺、坐/跪(躯干还比较直)。
"""
import argparse
import math

import torch
import warp as wp

from env import HumanoidEnv, quat_to_gravity, rotate

p = argparse.ArgumentParser()
p.add_argument("--xml", default="models/human50_pd.xml")
p.add_argument("--arm-pose", default="models/arm_pose.json")
p.add_argument("--envs", type=int, default=4096)
p.add_argument("--rounds", type=int, default=2)
p.add_argument("--out", default="models/fallen_poses.pt")
p.add_argument("--mid", type=int, default=0, help="往已有的姿态库里追加中间姿态(坐、跪、四肢着地)各这么多个, 不重新生成倒地姿态")
a = p.parse_args()


def classify(q):
    """躯干的前方(胸口)和左方在世界系的 z 分量 + 躯干竖直程度 -> 类别 0 仰躺 1 趴 2 侧躺 3 坐/跪"""
    n = q.shape[0]
    fwd = rotate(q, torch.tensor([1.0, 0, 0], device=q.device).expand(n, 3))[:, 2]
    left = rotate(q, torch.tensor([0, 1.0, 0], device=q.device).expand(n, 3))[:, 2]
    up = -quat_to_gravity(q)[:, 2]
    c = torch.full((n,), 2, dtype=torch.long, device=q.device)
    c[(fwd > 0.6)] = 0
    c[(fwd < -0.6)] = 1
    c[(up > 0.6)] = 3
    c[(left.abs() > 0.7) & (up <= 0.6)] = 2
    return c


NAMES = ("仰躺", "趴", "侧躺", "坐/跪", "四肢着地")


FOURS_PITCH = 1.19         # 弧度(68°): 肩高 - 胯高 = 0.52 m x cos(俯仰) ≈ 手臂长 - 大腿长 = 0.2 m


def mid_poses(env, k):
    """中间姿态(起身路上会经过的, 手常常撑着地): 坐(髋屈 90°、腿伸直)、跪(膝屈 90°、上身直)、四肢着地(躯干前倾 68°、手臂和大腿竖直向下、膝屈 90°)。
    从站姿(含手臂默认姿势)改关节角, 加 ±0.15 rad 的随机, 躯干按姿态俯仰; 根的高度让最低点离地 1 cm, 之后由物理落稳。返回 (qpos [3k, nq], 类别 [3k])"""
    import json
    import mujoco
    import numpy as np
    from env import arm_swing_dirs
    m = env.mjm
    d = mujoco.MjData(m)
    ja = lambda n: int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])
    pose = json.load(open(a.arm_pose))
    base = m.qpos0.copy()
    for n, v in pose.items():
        base[ja(n)] = v
    dirs = arm_swing_dirs(m, pose)
    rng = np.random.default_rng(0)
    out, cls = [], []
    for kind in ("sit", "kneel", "fours"):
        for _ in range(k):
            q = base.copy()
            pitch = 0.0
            for sd in ("right", "left"):
                if kind == "sit":
                    q[ja(f"hip_y_{sd}")] = np.radians(-90)
                elif kind == "kneel":
                    q[ja(f"knee_{sd}")] = np.radians(-90)
                    q[ja(f"hip_y_{sd}")] = np.radians(-10)
                else:                                    # 躯干前倾 FOURS_PITCH(肩比胯高: 手臂约 0.6 m、大腿约 0.4 m, 躯干水平时只有手着地, 会翻倒),
                    pitch = FOURS_PITCH                  # 手臂前举、髋屈同样的角度, 使手臂和大腿都竖直向下; 膝屈 90°, 小腿平放
                    q[ja(f"hip_y_{sd}")], q[ja(f"knee_{sd}")] = -pitch, np.radians(-90)
                    for j in ("shoulder1", "shoulder2"):
                        q[ja(f"{j}_{sd}")] += pitch * dirs[f"{j}_{sd}"]
            if kind == "sit":
                pitch = np.radians(-10)                  # 上身略后仰
            jq = [int(m.jnt_qposadr[j]) for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
            q[jq] += rng.uniform(-0.15, 0.15, len(jq))
            for j in range(m.njnt):                      # 裁到关节范围
                if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                    q[m.jnt_qposadr[j]] = np.clip(q[m.jnt_qposadr[j]], *m.jnt_range[j])
            pitch += rng.uniform(-0.15, 0.15)
            q[3:7] = [np.cos(pitch / 2), 0, np.sin(pitch / 2), 0]
            q[2] = 2.0
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            low = 1e9
            for g in range(1, m.ngeom):                  # 胶囊两端 / 球的最低点
                r, h = m.geom_size[g][0], m.geom_size[g][1]
                ax = d.geom_xmat[g].reshape(3, 3)[:, 2]
                ends = [d.geom_xpos[g]] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE else [d.geom_xpos[g] + ax * h, d.geom_xpos[g] - ax * h]
                low = min(low, min(e[2] for e in ends) - r)
            q[2] -= low - 0.01
            out.append(q)
            cls.append(3 if kind in ("sit", "kneel") else 4)
    return torch.tensor(np.array(out), dtype=torch.float32), torch.tensor(cls)

if __name__ == "__main__":
    env = HumanoidEnv(a.envs, xml=a.xml, arm_pose=a.arm_pose, seed=1)
    n, dev = env.n, env.dev
    keep = []
    if a.mid:                                              # 追加中间姿态: 摆好后 PD 保持这个姿势落稳 1.5 秒
        bank = torch.load(a.out)
        qm, cm = mid_poses(env, a.mid)
        got, gc = [], []
        for b in range(0, len(qm), n):
            q, c = qm[b:b + n].to(dev), cm[b:b + n].to(dev)
            k = len(q)
            env.reset_idx(torch.arange(n, device=dev))
            env.qpos[:k] = q
            env.qvel[:] = 0
            hold = env.qpos[:, env.qadr].clamp(env.lo, env.hi).clone()
            for t in range(75):
                env.ctrl[:] = hold
                env.xfrc[:] = 0
                wp.capture_launch(env.graph)
            wp.synchronize()
            fell, _ = env.ground_contact()
            ok = (fell & torch.isfinite(env.qpos).all(1) & (env.qvel[:, :3].norm(dim=1) < 0.3))[:k]
            got.append(env.qpos[:k][ok].clone().cpu())
            gc.append(c[ok].cpu())
            print(f"中间姿态 {b}~{b + k}: 落稳留下 {int(ok.sum())}/{k}", env.diagnostics())
        qpos, c = torch.cat([bank["qpos"]] + got), torch.cat([bank["cls"]] + gc)
        print("共", qpos.shape[0], "个; " + ", ".join(f"{NAMES[i]} {int((c == i).sum())}" for i in range(len(NAMES))))
        torch.save(dict(qpos=qpos, cls=c, names=NAMES, xml=a.xml), a.out)
        raise SystemExit
    for rd in range(a.rounds):
        torch.manual_seed(100 + rd)
        env.reset_idx(torch.arange(n, device=dev))
        ang = torch.rand(n, device=dev) * 2 * math.pi
        mag = 250 + torch.rand(n, device=dev) * 250
        push = torch.stack([torch.cos(ang), torch.sin(ang)], 1) * mag[:, None]
        act = torch.randn(n, env.nu, device=dev) * 0.5
        limp = torch.rand(n, device=dev) < 0.5
        for t in range(225):                               # 4.5 秒, 每步 0.02 秒
            if t > 0 and t % 25 == 0 and t < 125:
                act = torch.randn(n, env.nu, device=dev) * 0.7
            ctrl = (env.q_def + act * env.scale).clamp(env.lo, env.hi)
            if t >= 125:
                ctrl = torch.where(limp[:, None], env.qpos[:, env.qadr].clamp(env.lo, env.hi), ctrl)
            env.ctrl[:] = ctrl
            env.xfrc[:, env.torso_id, :2] = push * (t < 20)
            wp.capture_launch(env.graph)
        wp.synchronize()
        fell, _ = env.ground_contact()
        ok = fell & torch.isfinite(env.qpos).all(1) & (env.qvel[:, :3].norm(dim=1) < 0.2)
        keep.append(env.qpos[ok].clone())
        print(f"第 {rd + 1} 轮: 留下 {int(ok.sum())}/{n}", env.diagnostics())
    qpos = torch.cat(keep)
    c = classify(qpos[:, 3:7])
    print("共", qpos.shape[0], "个; " + ", ".join(f"{NAMES[i]} {int((c == i).sum())}" for i in range(4)),
          f"; 躯干高度 {qpos[:, 2].min():.2f}~{qpos[:, 2].max():.2f} m(站立 {env.root_z0:.2f})")
    torch.save(dict(qpos=qpos.cpu(), cls=c.cpu(), names=NAMES, xml=a.xml), a.out)
    print("存到", a.out)
