"""找一组关节角, 让手臂自然下垂(上臂朝下略向外、前臂略向前弯), 作为训练时的默认姿势。
官方人形的默认姿势是手臂斜向前下方直直伸着, 像僵尸; 策略也没有理由去动手臂, 所以会一直举着。
输出 models/arm_pose.json: {关节名: 弧度}。"""
import itertools
import json

import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path("models/humanoid_pd.xml")
d = mujoco.MjData(m)
bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
unit = lambda v: v / np.linalg.norm(v)

out = {}
for side, sgn in (("right", -1.0), ("left", 1.0)):         # 右臂在 -y 一侧, 左臂在 +y 一侧
    js = [jid(f"shoulder1_{side}"), jid(f"shoulder2_{side}"), jid(f"elbow_{side}")]
    qa = [m.jnt_qposadr[j] for j in js]
    rng = [m.jnt_range[j] for j in js]
    want_ua = unit(np.array([0.12, sgn * 0.22, -1.0]))       # 上臂: 朝下, 略向外、略向前
    want_fa = unit(np.array([0.45, sgn * 0.10, -1.0]))       # 前臂: 朝前下, 肘微弯
    ua_b, la_b, hand_b = bid(f"upper_arm_{side}"), bid(f"lower_arm_{side}"), bid(f"hand_{side}")
    grids = [np.linspace(r[0], r[1], n) for r, n in zip(rng, (41, 41, 31))]
    best = (1e9, None)
    for q in itertools.product(*grids):
        d.qpos[:] = m.qpos0
        for a, v in zip(qa, q):
            d.qpos[a] = v
        mujoco.mj_kinematics(m, d)
        ua = unit(d.xpos[la_b] - d.xpos[ua_b])
        fa = unit(d.xpos[hand_b] - d.xpos[la_b])
        cost = np.sum((ua - want_ua) ** 2) + np.sum((fa - want_fa) ** 2)
        if cost < best[0]:
            best = (cost, q, ua, fa)
    cost, q, ua, fa = best
    names = [f"shoulder1_{side}", f"shoulder2_{side}", f"elbow_{side}"]
    out.update({n: float(v) for n, v in zip(names, q)})
    print(f"{side}: 误差 {cost:.4f} | 关节角(度) {[round(float(np.degrees(v)), 1) for v in q]} | 上臂方向 {np.round(ua, 2)} (目标 {np.round(want_ua, 2)}) | 前臂方向 {np.round(fa, 2)}")

json.dump(out, open("models/arm_pose.json", "w"), indent=1)
print("已写入 models/arm_pose.json")
