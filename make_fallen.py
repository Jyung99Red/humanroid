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


NAMES = ("仰躺", "趴", "侧躺", "坐/跪")

if __name__ == "__main__":
    env = HumanoidEnv(a.envs, xml=a.xml, arm_pose=a.arm_pose, seed=1)
    n, dev = env.n, env.dev
    keep = []
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
