"""体检: 量一个身体模型和真人差多少。改了 make_model.py 的预设后用它复查。

    python check_body.py models/human50_pd.xml
    python check_body.py models/humanoid_pd.xml --no-jump

量的东西:
  - 身高、体重、BMI、质心高度, 各段质量占比 vs 真人(Winter)
  - 各关节最大力矩(Nm 和 Nm/公斤体重) vs 真人量级
  - 手臂能不能摆出拳击要的位置(直拳伸直、拳护下巴)
  - 起跳测验: 站 -> 下蹲 -> 蹬伸 的开环动作(11 个参数), 用 CEM 搜最好的一组, 看质心最高能比站立时高多少。
    判据是"质心高于站立时, 同时双脚离地 15 cm 以上、身体直立", 坐着/跪着/倒地都骗不了。
    这是开环动作的成绩, 训练出来的策略一般只会更好; 真人普通成年人约 40~50 cm。
"""
import argparse
import json

import mujoco
import numpy as np
from mujoco import rollout

HUMAN_FRAC = {"头": 8.1, "躯干": 49.7, "大腿": 20.0, "小腿": 9.3, "脚": 2.9, "上臂": 5.6, "前臂": 3.2, "手": 1.2}
GROUPS = {"头": ["head"], "躯干": ["torso", "waist_lower", "pelvis"], "大腿": ["thigh_right", "thigh_left"], "小腿": ["shin_right", "shin_left"],
          "脚": ["foot_right", "foot_left"], "上臂": ["upper_arm_right", "upper_arm_left"], "前臂": ["lower_arm_right", "lower_arm_left"],
          "手": ["hand_right", "hand_left"]}
HUMAN_NM_KG = {"hip_y": "3-4", "knee": "3-4", "ankle_y": "2-3", "ankle_x": "0.5-1", "hip_x": "1.5-2", "hip_z": "0.8-1.2", "abdomen_y": "2-3",
               "abdomen_x": "1.5-2", "abdomen_z": "0.8-1.2", "shoulder1": "1-1.5", "shoulder2": "1-1.5", "elbow": "1-1.5",
               "neck_y": "0.4-0.6", "neck_z": "0.25-0.4"}
ARM_POSE = json.load(open("models/arm_pose.json"))
SWING = {"shoulder1": -0.377, "shoulder2": 0.910}             # 上臂往前摆 1 弧度时两个肩关节各转多少(见 env.arm_swing_dirs)


def body_report(m):
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    M = m.body_mass.sum()
    top = d.xpos[m.body("head").id][2] + m.geom("head").size[0]
    com = d.subtree_com[m.body("torso").id][2]
    print(f"身高 {top:.2f} m  体重 {M:.1f} kg  BMI {M / top ** 2:.1f}  质心高 {com:.2f} m ({com / top * 100:.0f}% 身高)")
    print("各段质量占比: " + "  ".join(f"{g} {sum(m.body(b).mass[0] for b in bs) / M * 100:.1f}%(真人{HUMAN_FRAC[g]})" for g, bs in GROUPS.items()))
    print("最大力矩 Nm (Nm/kg, 真人量级):")
    row = []
    for a in range(m.nu):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        if name.endswith("_left"):
            continue
        key = name.replace("_right", "")
        g = m.actuator_forcerange[a][1]
        row.append(f"{key} {g:.0f} ({g / M:.1f}, {HUMAN_NM_KG[key]})")
    for i in range(0, len(row), 4):
        print("  " + "   ".join(row[i:i + 4]))
    rng = lambda n: np.degrees(m.joint(n).range).round().astype(int).tolist()
    print(f"活动范围(度): 髋屈伸 {rng('hip_y_right')}  髋外展/内收 {rng('hip_x_right')}  膝 {rng('knee_right')}  "
          f"踝(负=勾脚尖) {rng('ankle_y_right')}  脊柱(负=后仰) {rng('abdomen_y')}"
          + (f"  脖子 左右 {rng('neck_z')} 点头(负=抬头) {rng('neck_y')}" if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "neck_z") >= 0 else "  没有脖子"))


def arm_report(m, n=60000):
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    names = ["shoulder1_right", "shoulder2_right", "elbow_right"]
    adr = [m.jnt_qposadr[m.joint(j).id] for j in names]
    lim = np.array([m.joint(j).range for j in names])
    torso, hand = m.body("torso").id, m.body("hand_right").id
    sh = d.xpos[m.body("upper_arm_right").id] - d.xpos[torso]
    Q = np.random.default_rng(0).uniform(lim[:, 0], lim[:, 1], size=(n, 3))
    H = np.zeros((n, 3))
    for i in range(n):
        d.qpos[:] = m.qpos0
        d.qpos[adr] = Q[i]
        mujoco.mj_kinematics(m, d)
        H[i] = d.xpos[hand] - d.xpos[torso]
    rel = H - sh
    level = (abs(rel[:, 2]) < 0.08) & (abs(rel[:, 1]) < 0.08)
    guard = np.linalg.norm(H - np.array([0.17, -0.11, 0.17]), axis=1).min()
    print(f"手臂: 直拳(肩高正前方)能伸到 {rel[level, 0].max():.2f} m(臂长约 0.62);  拳护下巴最近差 {guard * 100:.1f} cm;  手最高举过肩 {rel[:, 2].max():.2f} m")


# ---------------- 起跳测验 ----------------
DT, T = 0.005, 1.7
N = int(T / DT)
LO = np.array([-2.0, -2.6, -0.8, -0.8, -0.4, -0.2, -0.3, 0.25, 0.04, 0.0, 0.0])
HI = np.array([0.0, -0.3, 0.8, 0.5, 0.4, 0.03, 1.0, 1.0, 0.35, 1.2, 1.6])
# 参数: 下蹲时 髋/膝/踝/脊柱 目标角, 蹬伸时 髋/膝/踝 目标角, 下蹲用时, 蹬伸用时, 手臂后摆, 手臂上摆
# 符号: hip_y 负 = 屈髋, knee 负 = 屈膝, ankle_y 正 = 踮脚, abdomen_y 正 = 前屈


def with_sensors(xml_path):
    spec = open(xml_path, encoding="utf-8").read().replace("</mujoco>", """<sensor>
  <subtreecom name="com" body="torso"/><framepos name="head" objtype="body" objname="head"/>
  <framepos name="fr" objtype="body" objname="foot_right"/><framepos name="fl" objtype="body" objname="foot_left"/>
</sensor></mujoco>""")
    return mujoco.MjModel.from_xml_string(spec)


def controls(m, P):
    idx = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a): a for a in range(m.nu)}
    t = np.arange(N)[None] * DT
    ch, ck, ca, ct, ph, pk, pa, t_down, t_push, a_back, a_up = (P[:, i:i + 1] for i in range(11))
    t0 = 0.3
    x = np.clip((t - t0) / t_down, 0, 1)
    w1 = x * x * (3 - 2 * x)                                       # 0 -> 1: 下蹲(平滑)
    w2 = np.clip((t - t0 - t_down - 0.05) / t_push, 0, 1)          # 0 -> 1: 蹬伸(线性, 要快)
    blend = lambda c, p: c * w1 + (p - c) * w2
    u = np.zeros((len(P), N, m.nu))
    for side, sgn in (("right", 1.0), ("left", -1.0)):
        u[:, :, idx[f"hip_y_{side}"]] = blend(ch, ph)
        u[:, :, idx[f"knee_{side}"]] = blend(ck, pk)
        u[:, :, idx[f"ankle_y_{side}"]] = blend(ca, pa)
        swing = blend(-a_back, a_up)
        for j in ("shoulder1", "shoulder2"):
            u[:, :, idx[f"{j}_{side}"]] = ARM_POSE[f"{j}_{side}"] + sgn * SWING[j] * swing
        u[:, :, idx[f"elbow_{side}"]] = ARM_POSE[f"elbow_{side}"]
    u[:, :, idx["abdomen_y"]] = blend(ct, ct * 0.3)
    return u


def jump_scores(m, P, datas):
    q0 = m.qpos0.copy()
    for k, v in ARM_POSE.items():
        q0[m.jnt_qposadr[m.joint(k).id]] = v
    s0 = np.concatenate([[0.0], q0, np.zeros(m.nv)])
    _, sd = rollout.rollout(m, datas, np.tile(s0, (len(P), 1)), controls(m, P))
    com, head, fr, fl = sd[:, :, 2], sd[:, :, 5], sd[:, :, 8], sd[:, :, 11]
    com0, f0 = com[:, 0].mean(), min(fr[:, 0].mean(), fl[:, 0].mean())
    ok = (np.minimum(fr, fl) > f0 + 0.15) & ((head - com) > 0.35)
    rise = np.where(ok, com - com0, -9.0).max(axis=1)
    return np.where(rise > -9.0, rise, -1.0)


def jump_test(m, seeds=(0, 1, 2), pop=480, iters=40):
    datas = [mujoco.MjData(m) for _ in range(12)]
    best = []
    for seed in seeds:
        rs = np.random.default_rng(seed)
        mu, sd, top = (LO + HI) / 2, (HI - LO) / 3, -9.0
        for _ in range(iters):
            P = np.clip(rs.normal(mu, sd, size=(pop, len(mu))), LO, HI)
            sc = jump_scores(m, P, datas)
            el = P[np.argsort(sc)[-pop // 10:]]
            mu, sd = el.mean(0), el.std(0) + 1e-3
            top = max(top, sc.max())
        best.append(top)
    print(f"起跳测验: 质心最高比站立时高 {max(best) * 100:.0f} cm (各随机种子 {[round(b * 100) for b in best]};"
          f" 真人普通成年人约 40~50 cm, 运动员 60+)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("xml", nargs="?", default="models/human50_pd.xml")
    ap.add_argument("--no-jump", action="store_true")
    a = ap.parse_args()
    m = mujoco.MjModel.from_xml_path(a.xml)
    print(f"== {a.xml} ==")
    body_report(m)
    arm_report(m)
    if not a.no_jump:
        jump_test(with_sensors(a.xml))
