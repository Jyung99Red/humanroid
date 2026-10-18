"""发球机的球: 训练环境(env.py)、参考实现(cpu_check.py)、测验共用。球的种类和物理参数在 web/play/balls.json; 网页 web/play/sim.js 里有同样的 addBalls / aim / closestApproach。

- add_balls: 往人形的场景里加一池球(自由刚体, 排在人形后面, 不用时停在高空 gravcomp=1)
- aim: 从 src 以 speed 扔到 dst 的出手速度(和 MuJoCo 一样的离散积分, 含空气阻力; 一次一个, CPU)
- AimTable: 同一套积分在 GPU 上一次算好一张表(速度 x 水平距离 x 高度差 -> 仰角、飞行时间), 训练时每步几十个世界同时出手, 查表插值
- closest_approach: 来球(不计阻力的抛物线)和身体几个点(按躯干现在的水平速度平移)最近时还剩多少空隙、几秒后(球的观测块用)
"""
import math
import re

import numpy as np
import torch

BODY_R = 0.15          # 球的观测块里身体点(头、躯干、胯)当成这么粗的球: 空隙 = 球心到最近身体点的距离 - BODY_R - 球半径
CLOSE_T = 1.5          # 最近点只往后看这么多秒
CLOSE_K = 16           # 在 [0, CLOSE_T] 里均匀取这么多个时刻找最近点
# 手写的躲球规划(env._steer, cpu_check / sim.js 的 Controller 同一套): 预测空隙 < BALL_NEAR 时起躲, 往远离球路的方向全速走
BALL_NEAR = 0.3        # 米, 预测空隙小于它就起躲(env 的 ball_near 奖励也按它)
DODGE_HOLD = 0.3       # 秒, 球结束后躲球指令再保持这么久, 然后按加速度限制回到原指令
DODGE_ACC = 100.0      # m/s^2, 躲球指令往躲的方向变的加速度上限(等于瞬间切换)。试过 6: 躲不开(打中 89%, 不躲 93%); 瞬间切换时 sp4 走着急转会摔, 靠训练学


def add_balls(xml, spec):
    """往场景里加一池球: 每种 per_kind 个自由刚体, 排在人形后面(人形的刚体编号、qpos 地址都不变)。
    不用的球停在高空(gravcomp=1 抵消重力, 碰不到任何东西), 扔的时候才放下来。关键帧的 qpos 长度会对不上, 删掉(没用到)。"""
    xml = re.sub(r"<keyframe>.*?</keyframe>", "", xml, flags=re.S)
    fr = " ".join(str(v) for v in spec["friction"])
    px, py, pz = spec["park"]
    bodies, k = [], 0
    for kd in spec["kinds"]:
        for j in range(spec["per_kind"]):
            bodies.append(f'<body name="ball_{kd["id"]}_{j}" pos="{px + 1.5 * k} {py} {pz}" gravcomp="1"><freejoint/>'
                          f'<geom type="sphere" size="{kd["r"]}" mass="{kd["mass"]}" shellinertia="{str(kd["shell"]).lower()}" '
                          f'priority="1" condim="6" friction="{fr}" solref="{kd["solref"][0]} {kd["solref"][1]}"/></body>')
            k += 1
    return xml.replace("</worldbody>", "".join(bodies) + "</worldbody>")


def drag_coef(spec, kd):
    """空气阻力系数 / 质量: 加速度 = -它 x |v| x v"""
    return 0.5 * spec["air"]["rho"] * spec["air"]["cd"] * math.pi * kd["r"] ** 2 / kd["mass"]


def aim(src, dst, speed, drag, dt, dec, g):
    """从 src 以 speed 扔到 dst(世界系)要的出手速度, 低弹道, 算上空气阻力 drag = 阻力系数 / 质量(加速度 = -drag |v| v)。
    和 MuJoCo 里一样离散积分(阻力每个策略步算一次, 物理步半隐式欧拉)。返回 (速度, 飞行时间, 打得到吗); 打不到时给打得最高的角度"""
    dx, dy = dst[0] - src[0], dst[1] - src[1]
    dist = math.hypot(dx, dy)
    ex, ey = (dx / dist, dy / dist) if dist > 1e-6 else (1.0, 0.0)
    h = dt / dec

    def fly(th):                                       # 飞到水平距离 dist 时的 (高度, 时间); 飞不到返回 None
        vs, vz, s, z, t = speed * math.cos(th), speed * math.sin(th), 0.0, src[2], 0.0
        while t < 3 and z > -1:
            v = math.hypot(vs, vz)
            a_s, a_z = -drag * v * vs, -drag * v * vz - g
            for _ in range(dec):
                s0, z0 = s, z
                vs += h * a_s
                vz += h * a_z
                s += h * vs
                z += h * vz
                t += h
                if s >= dist:
                    f = (dist - s0) / (s - s0)
                    return z0 + f * (z - z0), t - h * (1 - f)
        return None

    def err(th):
        r = fly(th)
        return -1e9 if r is None else r[0] - dst[2]

    ths = [math.radians(a) for a in range(-40, 62, 2)]
    es = [err(th) for th in ths]
    reach = any(e >= 0 for e in es)
    if reach:                                          # 第一个打高了的角度和前一个之间二分
        i = next(i for i, e in enumerate(es) if e >= 0)
        lo, hi = (ths[i - 1], ths[i]) if i > 0 else (ths[0], ths[0])
        for _ in range(20):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if err(mid) >= 0 else (mid, hi)
        th = hi
    else:
        th = ths[max(range(len(ths)), key=lambda i: es[i])]
    r = fly(th)
    c = speed * math.cos(th)
    return np.array([c * ex, c * ey, speed * math.sin(th)]), (r[1] if r else 3.0), reach


class AimTable:
    """aim 的 GPU 查表版。建表: 每个 (速度, 仰角) 飞一条轨迹(同样的离散积分), 记下经过每个水平距离时的高度和时间;
    再对每个 (速度, 距离, 高度差) 找第一个打高了的仰角(低弹道), 和前一个仰角之间线性插值。查表: 三线性插值。"""

    def __init__(self, drag, dt, dec, g, device, speeds=(5.0, 22.0, 0.5), dists=(1.0, 12.0, 0.25), dzs=(-2.0, 2.0, 0.1), ths=(-40.0, 60.0, 0.25)):
        f = dict(device=device, dtype=torch.float32)
        grid = lambda a: torch.arange(a[0], a[1] + a[2] / 2, a[2], **f)
        self.S, self.D, self.Z = grid(speeds), grid(dists), grid(dzs)
        self.lo = torch.tensor([speeds[0], dists[0], dzs[0]], **f)
        self.step = torch.tensor([speeds[2], dists[2], dzs[2]], **f)
        TH = grid(ths) * (math.pi / 180)
        ns, nt, nd = len(self.S), len(TH), len(self.D)
        vs = self.S[:, None] * torch.cos(TH)[None, :]
        vz = self.S[:, None] * torch.sin(TH)[None, :]
        s, z, t = torch.zeros_like(vs), torch.zeros_like(vs), 0.0
        zc = torch.full((ns, nt, nd), float("nan"), **f)
        tc = torch.full_like(zc, float("nan"))
        h = dt / dec
        for _ in range(int(round(3.0 / dt))):
            sp = torch.sqrt(vs * vs + vz * vz)
            a_s, a_z = -drag * sp * vs, -drag * sp * vz - g
            for _ in range(dec):
                s0, z0 = s, z
                vs = vs + h * a_s
                vz = vz + h * a_z
                s = s + h * vs
                z = z + h * vz
                t += h
                cross = (s0[..., None] < self.D) & (s[..., None] >= self.D) & torch.isnan(zc)
                fr = (self.D - s0[..., None]) / (s - s0).clamp(min=1e-9)[..., None]
                zc = torch.where(cross, z0[..., None] + fr * (z - z0)[..., None], zc)
                tc = torch.where(cross, t - h * (1 - fr), tc)
        zf = torch.nan_to_num(zc, nan=-1e9)                                  # 飞不到这个距离 = 无限低
        mask = zf[..., None] >= self.Z                                        # [速度, 仰角, 距离, 高度差]
        self.ok = mask.any(dim=1)                                             # [速度, 距离, 高度差]
        i = mask.float().argmax(dim=1)                                        # 第一个打高了的仰角
        i0 = (i - 1).clamp(min=0)
        take = lambda x, idx: torch.gather(x, 1, idx)                         # x [速度, 仰角, 距离, 高度差]
        nz = len(self.Z)
        Zx = zf[..., None].expand(ns, nt, nd, nz)
        Tx = torch.nan_to_num(tc, nan=3.0)[..., None].expand(ns, nt, nd, nz)
        z_lo, z_hi = take(Zx, i0[:, None]).squeeze(1), take(Zx, i[:, None]).squeeze(1)
        w = ((self.Z - z_lo) / (z_hi - z_lo).clamp(min=1e-9)).clamp(0, 1) * (i > 0)
        self.th = TH[i0] + w * (TH[i] - TH[i0])
        t_lo, t_hi = take(Tx, i0[:, None]).squeeze(1), take(Tx, i[:, None]).squeeze(1)
        self.t = t_lo + w * (t_hi - t_lo)

    def __call__(self, src, dst, speed):
        """src, dst [n, 3], speed [n] -> (出手速度 [n, 3], 飞行时间 [n], 打得到吗 [n])"""
        if not hasattr(self, "corners"):
            dev = self.th.device
            self.corners = torch.tensor([[(c >> 2) & 1, (c >> 1) & 1, c & 1] for c in range(8)], device=dev)    # 三线性插值的 8 个角
            self.hi = torch.tensor(self.th.shape, device=dev) - 1
            self.strides = torch.tensor([self.th.shape[1] * self.th.shape[2], self.th.shape[2], 1], device=dev)
        d = dst[:, :2] - src[:, :2]
        dist = d.norm(dim=1).clamp(min=1e-6)
        x = torch.stack([speed, dist, dst[:, 2] - src[:, 2]], 1)
        u = torch.minimum(((x - self.lo) / self.step).clamp(min=0), self.hi.float())
        i0 = torch.minimum(u.floor().long(), self.hi - 1)
        w = u - i0
        j = ((i0[:, None, :] + self.corners) * self.strides).sum(-1)                                  # [n, 8] 展平后的下标
        wc = torch.where(self.corners.bool(), w[:, None, :], 1 - w[:, None, :]).prod(-1)              # [n, 8]
        th = (wc * self.th.flatten()[j]).sum(1)
        t = (wc * self.t.flatten()[j]).sum(1)
        ok = self.ok.flatten()[j].all(1)
        e = d / dist[:, None]
        vel = torch.cat([(speed * torch.cos(th))[:, None] * e, (speed * torch.sin(th))[:, None]], 1)
        return vel, t, ok


def closest_approach(p, v, pts, vb, g, r_sum):
    """来球 p, v [n, 3](不计阻力的抛物线)和身体点 pts [n, P, 3](按水平速度 vb [n, 3] 平移)在接下来 CLOSE_T 秒里最近的时刻。
    返回 (空隙 [n] = 最近距离 - r_sum, 几秒后 [n])。torch 和 numpy 都能用(numpy 时 n 可以是 1)"""
    xp = torch if isinstance(p, torch.Tensor) else np
    ts = (xp.arange(CLOSE_K) * (CLOSE_T / (CLOSE_K - 1)))
    if xp is torch:
        ts = ts.to(p.device, p.dtype)
    gz = xp.zeros(3) if xp is np else torch.zeros(3, device=p.device, dtype=p.dtype)
    gz[2] = -g
    bp = p[:, None, :] + v[:, None, :] * ts[None, :, None] + 0.5 * gz * (ts * ts)[None, :, None]      # [n, K, 3]
    pp = pts[:, None, :, :] + vb[:, None, None, :] * ts[None, :, None, None]                         # [n, K, P, 3]
    d = ((bp[:, :, None, :] - pp) ** 2).sum(-1) ** 0.5                                              # [n, K, P]
    dk = d.min(-1) if xp is np else d.min(-1).values
    if xp is np:
        k = dk.argmin(-1)
        return dk.min(-1) - r_sum, ts[k]
    m = dk.min(-1)
    return m.values - r_sum, ts[m.indices]
