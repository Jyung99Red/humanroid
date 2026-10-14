"""量一份回放里的步态: 步长/步频, 支撑时脚的翻转(roll)/俯仰(pitch)/外八(yaw), 脚踝相对小腿的角度, 两脚间距, 抬脚高度;
骨盆的侧倾/扭转/左右晃, 头的稳定。括号里是真人慢走的参考范围。
用法: python check_feet.py web/replays/d_final.json [更多回放...]
着地用回放里存的真实接触(新导出的回放都有); 旧回放没有, 退回"脚踝高度 < 5.5 cm"的估算(和真实接触只有约 71% 一致)。"""
import json
import sys

import numpy as np

FOOT_POINTS = ((-0.07, 0.0, -0.027), (0.14, 0.0, -0.027))   # 脚跟、脚尖的底面点(脚坐标系), 和 env.FOOT_POINTS 一样


def mat(q):  # wxyz -> 旋转矩阵 [..., 3, 3]
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def rpy(R):  # ZYX 欧拉角(度): roll 绕前后轴, pitch 绕左右轴, yaw 绕竖直轴
    return (np.degrees(np.arctan2(R[..., 2, 1], R[..., 2, 2])), np.degrees(-np.arcsin(np.clip(R[..., 2, 0], -1, 1))),
            np.degrees(np.arctan2(R[..., 1, 0], R[..., 0, 0])))


def stat(a):
    return f"均值 {np.mean(a):+5.1f} 范围 [{np.percentile(a, 2):+5.1f}, {np.percentile(a, 98):+5.1f}]"


def smooth(x, n):
    """长度 n 的滑动平均(两端补边), 用来去掉一个步态周期内的摆动、留下平均朝向/轨迹。"""
    return np.convolve(np.pad(x, (n // 2, n - 1 - n // 2), mode="edge"), np.ones(n) / n, mode="valid")


def amp(a):
    """摆动幅度: 偏离平均值的 95 百分位, 记作 ±。"""
    return np.percentile(np.abs(a), 95)


def gait_samples(worlds, names, dt):
    """worlds: [(xpos [T,nb,3], xquat [T,nb,4], contact [T,2] 或 None), ...]。返回各项的原始样本(合并所有人形), 前 2 秒不算。"""
    acc = {k: [] for k in ("step_len", "step_time", "clear", "width", "roll_st", "pitch_st", "yaw_st", "roll_all", "ank_roll", "ank_pitch",
                           "land_roll", "land_pitch", "slip", "pel_roll", "pel_yaw", "pel_sway", "land_width", "head_pitch", "head_yaw")}
    real = all(c is not None for _, _, c in worlds)
    for xp, xq, con in worlds:
        if len(xp) < 200:
            continue
        s = slice(100, None)
        con = con[s].astype(bool) if real else None
        t = names.index("torso")
        _, _, tyaw = rpy(mat(xq[s, t]))
        c, sn = np.cos(np.radians(tyaw)), np.sin(np.radians(tyaw))
        feet = {}
        for k, (side, sgn) in enumerate((("right", -1), ("left", 1))):
            f, sh = names.index("foot_" + side), names.index("shin_" + side)
            R = mat(xq[s, f])
            roll, pitch, yaw = rpy(R)
            p = xp[s, f]
            st = con[:, k] if real else p[:, 2] < 0.055
            st[1:-1] |= st[:-2] & st[2:]                           # 着地中只断开一帧(20 ms)的接触补上, 不算一次抬脚(和 eval_policy 一致)
            rel = xp[s, f, :2] - xp[s, t, :2]
            lat = -sn * rel[:, 0] + c * rel[:, 1]                  # 脚相对躯干的左右位置(朝向坐标系)
            ar, ap, _ = rpy(np.swapaxes(mat(xq[s, sh]), -1, -2) @ R)   # 脚相对小腿
            dyaw = (yaw - tyaw + 180) % 360 - 180
            acc["roll_st"].append(sgn * roll[st]); acc["pitch_st"].append(pitch[st]); acc["yaw_st"].append(sgn * dyaw[st])
            acc["roll_all"].append(sgn * roll); acc["ank_roll"].append(sgn * ar); acc["ank_pitch"].append(ap)
            lift, land = np.flatnonzero(st[:-1] & ~st[1:]), np.flatnonzero(~st[:-1] & st[1:]) + 1
            acc["clear"].append([p[a:b, 2].max() - 0.027 for a, b in zip(lift, land[np.searchsorted(land, lift[0]):] if len(lift) else []) if b > a][:50])
            acc["land_roll"].append(sgn * roll[land]); acc["land_pitch"].append(pitch[land])
            # 打滑: 脚跟点、脚尖点谁贴地(离地 < 1 cm, 前后两帧都是)谁的水平速度; 绕脚尖转动不算(和 env 的 point_slip 一致)
            for local in FOOT_POINTS:
                q = p + R @ np.array(local)
                v = np.linalg.norm(np.diff(q[:, :2], axis=0), axis=1) / dt
                acc["slip"].append(v[(q[1:, 2] < 0.01) & (q[:-1, 2] < 0.01)])
            feet[side] = (p, st, lat, land)
        (pr, _, latr, landr), (pl, _, latl, landl) = feet["right"], feet["left"]
        acc["width"].append(latl - latr)
        acc["land_width"].append(np.abs((latl - latr)[np.concatenate([landr, landl])]))
        for land, me, other in ((landr, pr, pl), (landl, pl, pr)):
            acc["step_len"].append(np.hypot(*(me[land, :2] - other[land, :2]).T))   # 落地瞬间两脚的前后距离(近似)
            acc["step_time"].append(np.diff(land) * dt)
        # 骨盆和头: 减去 1 秒滑动平均, 剩下的就是一个步态周期里的摆动
        n = int(round(1.0 / dt))
        pv, hd = names.index("pelvis"), names.index("head")
        proll, _, pyaw = rpy(mat(xq[s, pv]))
        pyaw = np.degrees(np.unwrap(np.radians(pyaw)))
        acc["pel_roll"].append(proll - smooth(proll, n)); acc["pel_yaw"].append(pyaw - smooth(pyaw, n))
        h = np.radians(smooth(pyaw, n))
        pp = xp[s, pv, :2]
        dev = pp - np.stack([smooth(pp[:, 0], n), smooth(pp[:, 1], n)], 1)
        acc["pel_sway"].append(-np.sin(h) * dev[:, 0] + np.cos(h) * dev[:, 1])
        _, hpitch, hyaw = rpy(mat(xq[s, hd]))
        acc["head_pitch"].append(hpitch - smooth(hpitch, n)); acc["head_yaw"].append((hyaw - tyaw + 180) % 360 - 180)
    return {k: np.concatenate([np.ravel(x) for x in v]) for k, v in acc.items()}, real


def gait_stats(worlds, names, dt):
    """回归测验用的汇总数字。"""
    cat, _ = gait_samples(worlds, names, dt)
    return dict(cadence=120 / np.mean(cat["step_time"]), step_len=np.mean(cat["step_len"]), clear_cm=np.mean(cat["clear"]) * 100,
                width_cm=np.median(cat["land_width"]) * 100, pelvis_roll=amp(cat["pel_roll"]), pelvis_yaw=amp(cat["pel_yaw"]),
                pelvis_sway_cm=amp(cat["pel_sway"]) * 100, head_pitch=amp(cat["head_pitch"]), stance_toe_down=np.mean(cat["pitch_st"]),
                land_toe_up=-np.mean(cat["land_pitch"]), push_off=np.percentile(cat["pitch_st"], 98), stance_roll=amp(cat["roll_st"]),
                toe_out=amp(cat["yaw_st"]), slip_cms=np.mean(cat["slip"]) * 100)


def report(path):
    d = json.load(open(path, encoding="utf-8"))
    names = [b["name"] for b in d["bodies"]]
    nb, dt = len(names), d["meta"]["dt"]
    worlds = [(np.array(w["xpos"]).reshape(-1, nb, 3), np.array(w["xquat"]).reshape(-1, nb, 4),
               np.array(w["contact"]).reshape(-1, 2) if "contact" in w else None) for w in d["worlds"]]
    cat, real = gait_samples(worlds, names, dt)
    print(f"== {d['meta']['label']} ==  (着地: {'真实接触' if real else '估算, 脚踝 < 5.5 cm'})")
    print(f"  步长(落地时两脚相距) {np.mean(cat['step_len']):.2f} m | 同一只脚两次落地间隔 {np.mean(cat['step_time']):.2f} s (步频 {120 / np.mean(cat['step_time']):.0f} 步/分)"
          f" | 抬脚高度 {np.mean(cat['clear']) * 100:.1f} cm | 两脚左右间距 {stat(cat['width'] * 100)} cm")
    print(f"  支撑时 脚外翻(+)/内翻(-) {stat(cat['roll_st'])}° | 脚尖朝上(-)/朝下(+) {stat(cat['pitch_st'])}° | 外八(+)/内八(-) {stat(cat['yaw_st'])}°")
    print(f"  落地瞬间 翻转 {stat(cat['land_roll'])}° 俯仰 {stat(cat['land_pitch'])}° | 整个周期翻转 {stat(cat['roll_all'])}°")
    print(f"  脚踝相对小腿 翻转 {stat(cat['ank_roll'])}° 俯仰 {stat(cat['ank_pitch'])}° | 着地点打滑(脚跟/脚尖贴地时) {np.mean(cat['slip']) * 100:.1f} cm/s")
    print(f"  骨盆 侧倾 ±{amp(cat['pel_roll']):.1f}°(真人 ±4~5) 扭转 ±{amp(cat['pel_yaw']):.1f}°(±4~6) 左右晃 ±{amp(cat['pel_sway']) * 100:.1f} cm(±2~3)"
          f" | 落地时两脚左右间距 {np.median(cat['land_width']) * 100:.0f} cm(8~12)")
    print(f"  头 俯仰晃动 ±{amp(cat['head_pitch']):.1f}°(±2~4) | 头相对胸左右转 均值 {np.mean(cat['head_yaw']):+.1f}° 摆动 ±{amp(cat['head_yaw'] - np.mean(cat['head_yaw'])):.1f}°")


if __name__ == "__main__":
    for path in sys.argv[1:]:
        report(path)
