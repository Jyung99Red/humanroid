"""载入训练好的策略, 并排跑若干个人形, 导出每帧每个刚体的位置和朝向, 给 three.js 回放。

    python export.py runs/walk1/policy.pt --out web/replays/walk1_final.json --label "第 N 轮"
    python export.py runs/walk1/snaps/it_00050.pt --label "第 50 轮" --worlds 6 --steps 400 --cmd 1.0,0,0

策略用确定性动作(均值), 不加探索噪声。每个世界只记录到第一次摔倒/超时为止。
坐标是 MuJoCo 原样(Z 轴朝上, 四元数 wxyz), 转成 three.js 的朝向由网页负责。
"""
import argparse
import json
import os

import mujoco
import numpy as np
import torch

from policy import load_policy

p = argparse.ArgumentParser()
p.add_argument("policy")
p.add_argument("--out", default="")
p.add_argument("--label", default="")
p.add_argument("--worlds", type=int, default=6)
p.add_argument("--steps", type=int, default=400, help="最多记录多少个策略步(每步 0.02 秒)")
p.add_argument("--cmd", default="1.0,0,0", help="速度指令 vx,vy,wz")
p.add_argument("--gait", default="", help="步态指令 步频,着地比例; 不给就用训练时范围的中点")
p.add_argument("--seed", type=int, default=1)
p.add_argument("--hold-heading", type=float, default=1.5, help="保持朝向的增益, 0 = 关闭(策略自己走会慢慢拐弯)")
a = p.parse_args()

vx, vy, wz = (float(v) for v in a.cmd.split(","))
fix = dict(cmd_vx=(vx, vx), cmd_vy=(vy, vy), cmd_wz=(wz, wz))
if a.gait:
    f, duty = (float(v) for v in a.gait.split(","))
    fix.update(cmd_freq=(f, f), cmd_duty=(duty, duty))
env, act, meta = load_policy(a.policy, a.worlds, seed=a.seed, random_yaw=False, **fix)
env.ep_len = max(env.ep_len, a.steps + 5)
if not a.gait:                                                       # 训练时步态指令是个范围的话, 回放取中点
    env.gait_rng = tuple(((lo + hi) / 2,) * 2 for lo, hi in env.gait_rng)
env.t[:] = 0
env.cmd_t[:] = 0
env.reset_idx(torch.arange(a.worlds, device=env.dev))
env._sample_cmd(torch.arange(a.worlds, device=env.dev))

K, T = a.worlds, a.steps
q_hist = torch.zeros(T, K, env.qpos.shape[1], device=env.dev)
vx_hist = torch.zeros(T, K, device=env.dev)
c_hist = torch.zeros(T, K, 2, dtype=torch.uint8, device=env.dev)  # 每帧两只脚的真实接触(和 q_hist 同一时刻)
end = torch.full((K,), T, dtype=torch.long, device=env.dev)       # 每个世界有效帧数
fell = torch.zeros(K, dtype=torch.bool, device=env.dev)
with torch.no_grad():
    for t in range(T):
        if a.hold_heading > 0:      # 偏航角 -> 转向指令(限制在训练时见过的范围内), 然后让观测反映这个指令
            qq = env.qpos[:, 3:7]
            yaw = torch.atan2(2 * (qq[:, 0] * qq[:, 3] + qq[:, 1] * qq[:, 2]), 1 - 2 * (qq[:, 2] ** 2 + qq[:, 3] ** 2))
            env.cmd[:, 2] = (wz - a.hold_heading * yaw).clamp(-0.3, 0.3)
        obs, _ = env.observe()
        q_hist[t] = env.qpos
        c_hist[t] = env.contact.to(torch.uint8)
        _, _, r, te, tr, _ = env.step(act(obs))
        vx_hist[t] = env.m_vx
        newly = (te | tr) & (end == T)
        end[newly] = t + 1                                          # 这一帧(摔倒前的最后状态)还是有效的
        fell |= te & newly
q_hist, vx_hist, c_hist, end, fell = q_hist.cpu().numpy(), vx_hist.cpu().numpy(), c_hist.cpu().numpy(), end.cpu().numpy(), fell.cpu().numpy()

m = mujoco.MjModel.from_xml_path(meta["env"].get("xml", "models/humanoid_pd.xml"))
d = mujoco.MjData(m)
bodies = [dict(name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i), parent=int(m.body_parentid[i])) for i in range(m.nbody)]
TYPES = {int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere", int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule"}
geoms = []
for g in range(m.ngeom):
    if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
        continue
    assert int(m.geom_type[g]) in TYPES, f"geom {g} 类型 {m.geom_type[g]} 网页还不支持"
    geoms.append(dict(body=int(m.geom_bodyid[g]), type=TYPES[int(m.geom_type[g])], size=[round(float(x), 4) for x in m.geom_size[g][:2]],
                      pos=[round(float(x), 4) for x in m.geom_pos[g]], quat=[round(float(x), 4) for x in m.geom_quat[g]]))

worlds = []
for k in range(K):
    n = int(end[k])
    xp = np.zeros((n, m.nbody, 3))
    xq = np.zeros((n, m.nbody, 4))
    for t in range(n):
        d.qpos[:] = q_hist[t, k]
        mujoco.mj_kinematics(m, d)
        xp[t], xq[t] = d.xpos, d.xquat
    worlds.append(dict(n=n, fell=bool(fell[k]), vx=np.round(vx_hist[:n, k], 3).tolist(), contact=c_hist[:n, k].reshape(-1).tolist(),
                       xpos=np.round(xp, 4).reshape(-1).tolist(), xquat=np.round(xq, 4).reshape(-1).tolist()))

label = a.label or f"第 {meta['iteration']} 轮"
out = a.out or f"web/replays/it_{meta['iteration']:05d}.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(dict(meta=dict(label=label, iteration=meta["iteration"], dt=float(env.dt), cmd=[vx, vy, wz], gait=env.gait[0].tolist()),
               bodies=bodies, geoms=geoms, worlds=worlds), open(out, "w"), separators=(",", ":"))

# 更新回放列表(同名文件覆盖并移到最后)
idx_path = os.path.join(os.path.dirname(out), "index.json")
idx = json.load(open(idx_path, encoding="utf-8")) if os.path.exists(idx_path) else []
idx = [e for e in idx if e["file"] != os.path.basename(out)]
idx.append(dict(file=os.path.basename(out), label=label, iteration=meta["iteration"]))   # 最近导出的排最后, 网页默认打开它
json.dump(idx, open(idx_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

secs = [round(int(e) * env.dt, 1) for e in end]
print(f"{out}  [{label}]  {os.path.getsize(out)/1e6:.2f} MB | 各世界撑了(秒): {secs} | 摔倒: {fell.tolist()} | "
      f"平均前进速度: {np.mean([np.mean(w['vx']) for w in worlds]):.2f} m/s (指令 {vx})")
