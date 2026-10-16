"""载入训练好的策略, 并排跑若干个人形, 导出每帧每个刚体的位置和朝向, 给 three.js 回放。

    python export.py runs/walk1/policy.pt --out web/replays/walk1_final.json --label "第 N 轮"
    python export.py runs/walk1/snaps/it_00050.pt --label "第 50 轮" --worlds 6 --steps 400 --cmd 1.0,0,0
    python export.py runs/xxx/policy.pt --steps 600 --seq "0:1.0,0,0; 4:0,0,0,0.9,1; 7:-0.4,0,0"     # 按时间换指令: 走 -> 站定 -> 倒退

策略用确定性动作(均值), 不加探索噪声。每个世界只记录到第一次摔倒/超时为止。
坐标是 MuJoCo 原样(Z 轴朝上, 四元数 wxyz), 转成 three.js 的朝向由网页负责。
推演和模型信息的函数(parse_seq / fixed_cmd / rollout / model_info / kinematics)也给 export_timeline.py 用。
"""
import argparse
import json
import os

import mujoco
import numpy as np
import torch

from policy import load_policy


def parse_seq(s):
    """"秒:vx,vy,wz[,步频,着地比例]; ..." -> [(秒, [数...]), ...], 按时间排好"""
    return sorted((float(k), [float(v) for v in x.split(",")]) for k, x in (e.split(":") for e in s.split(";") if e.strip())) if s else []


def fixed_cmd(seq, cmd="1.0,0,0", gait=""):
    """回放时固定的指令范围(覆盖策略自带的), 返回 (覆盖项, 开头的 vx, vy, wz)"""
    vx, vy, wz = seq[0][1][:3] if seq else (float(v) for v in cmd.split(","))
    fix = dict(cmd_vx=(vx, vx), cmd_vy=(vy, vy), cmd_wz=(wz, wz), cmd_stand=0.0)   # 训练时开了 --stand 的策略, 回放时不要随机抽到站定
    if gait:
        f, duty = (float(v) for v in gait.split(","))
        fix.update(cmd_freq=(f, f), cmd_duty=(duty, duty))
    return fix, (vx, vy, wz)


@torch.no_grad()
def rollout(env, act, steps, seq, wz0, gait_fixed, hold_heading=1.5):
    """所有世界一起从头跑 steps 步。返回 numpy 的 (q_hist [T,N,nq], vx_hist [T,N], c_hist [T,N,2], end [N], fell [N])。
    gait_fixed = False 时步态指令取训练范围的中点。"""
    N = env.n
    env.ep_len = max(env.ep_len, steps + 5)
    if not gait_fixed:                                               # 训练时步态指令是个范围的话, 回放取中点
        env.gait_rng = tuple(((lo + hi) / 2,) * 2 for lo, hi in env.gait_rng)
    env.t[:] = 0
    env.cmd_t[:] = 0
    env.reset_idx(torch.arange(N, device=env.dev))
    env._sample_cmd(torch.arange(N, device=env.dev))
    gait0 = env.gait[0].clone()
    T, wz = steps, wz0
    q_hist = torch.zeros(T, N, env.qpos.shape[1], device=env.dev)
    vx_hist = torch.zeros(T, N, device=env.dev)
    c_hist = torch.zeros(T, N, 2, dtype=torch.uint8, device=env.dev)  # 每帧两只脚的真实接触(和 q_hist 同一时刻)
    end = torch.full((N,), T, dtype=torch.long, device=env.dev)       # 每个世界有效帧数
    fell = torch.zeros(N, dtype=torch.bool, device=env.dev)
    for t in range(T):
        if seq:
            c = [x for k, x in seq if k <= t * env.dt + 1e-9][-1]
            cmd, gait = torch.tensor(c[:3], device=env.dev), torch.tensor(c[3:5], device=env.dev) if len(c) >= 5 else gait0
            if env.cmd_acc is None:
                env.set_cmd(cmd, gait)
            else:                                                # 训练时有指令加速度限制: 只改目标, 让指令按限制过渡
                env.cmd_goal[:], env.gait_goal[:] = cmd, gait
            wz = c[2]
        if hold_heading > 0 and env.cmd[0].abs().sum() > 0:        # 偏航角 -> 转向指令(限制在训练时见过的范围内), 然后让观测反映这个指令
            qq = env.qpos[:, 3:7]
            yaw = torch.atan2(2 * (qq[:, 0] * qq[:, 3] + qq[:, 1] * qq[:, 2]), 1 - 2 * (qq[:, 2] ** 2 + qq[:, 3] ** 2))
            env.cmd[:, 2] = env.cmd_goal[:, 2] = (wz - hold_heading * yaw).clamp(-0.3, 0.3)
        obs, _ = env.observe()
        q_hist[t] = env.qpos
        c_hist[t] = env.contact.to(torch.uint8)
        _, _, r, te, tr, _ = env.step(act(obs))
        vx_hist[t] = env.m_vx
        newly = (te | tr) & (end == T)
        end[newly] = t + 1                                          # 这一帧(摔倒前的最后状态)还是有效的
        fell |= te & newly
    return q_hist.cpu().numpy(), vx_hist.cpu().numpy(), c_hist.cpu().numpy(), end.cpu().numpy(), fell.cpu().numpy()


def model_info(xml):
    """(MjModel, 刚体列表, 几何体列表): 网页按这些搭人形"""
    m = mujoco.MjModel.from_xml_path(xml)
    bodies = [dict(name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i), parent=int(m.body_parentid[i])) for i in range(m.nbody)]
    TYPES = {int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere", int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule"}
    geoms = []
    for g in range(m.ngeom):
        if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        assert int(m.geom_type[g]) in TYPES, f"geom {g} 类型 {m.geom_type[g]} 网页还不支持"
        geoms.append(dict(body=int(m.geom_bodyid[g]), type=TYPES[int(m.geom_type[g])], size=[round(float(x), 4) for x in m.geom_size[g][:2]],
                          pos=[round(float(x), 4) for x in m.geom_pos[g]], quat=[round(float(x), 4) for x in m.geom_quat[g]]))
    return m, bodies, geoms


def kinematics(m, d, q):
    """一个世界的关节轨迹 [n, nq] -> 每帧每个刚体的位置 [n, nbody, 3] 和朝向 [n, nbody, 4](wxyz)"""
    n = len(q)
    xp, xq = np.zeros((n, m.nbody, 3)), np.zeros((n, m.nbody, 4))
    for t in range(n):
        d.qpos[:] = q[t]
        mujoco.mj_kinematics(m, d)
        xp[t], xq[t] = d.xpos, d.xquat
    return xp, xq


if __name__ == "__main__":
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
    p.add_argument("--seq", default="", help="按时间换指令, 分号隔开的 秒:vx,vy,wz[,步频,着地比例]; 给了就不看 --cmd/--gait。站定(速度全 0)时不做保持朝向")
    a = p.parse_args()

    seq = parse_seq(a.seq)
    fix, (vx, vy, wz) = fixed_cmd(seq, a.cmd, a.gait)
    env, act, meta = load_policy(a.policy, a.worlds, seed=a.seed, random_yaw=False, **fix)
    q_hist, vx_hist, c_hist, end, fell = rollout(env, act, a.steps, seq, wz, bool(a.gait), a.hold_heading)

    m, bodies, geoms = model_info(meta["env"].get("xml", "models/humanoid_pd.xml"))
    d = mujoco.MjData(m)
    worlds = []
    for k in range(a.worlds):
        n = int(end[k])
        xp, xq = kinematics(m, d, q_hist[:n, k])
        worlds.append(dict(n=n, fell=bool(fell[k]), vx=np.round(vx_hist[:n, k], 3).tolist(), contact=c_hist[:n, k].reshape(-1).tolist(),
                           xpos=np.round(xp, 4).reshape(-1).tolist(), xquat=np.round(xq, 4).reshape(-1).tolist()))

    label = a.label or f"第 {meta['iteration']} 轮"
    out = a.out or f"web/replays/it_{meta['iteration']:05d}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(dict(meta=dict(label=label, iteration=meta["iteration"], dt=float(env.dt), cmd=[vx, vy, wz], gait=env.gait[0].tolist(), seq=a.seq),
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
