"""网页互动版的参考实现 + 可行性检查: 只用 export_web.py 导出的 web/play/policies.json, 在 MuJoCo 的 CPU 版(和浏览器里的 WebAssembly 同一套 C 代码)
里跑走路 + 起身切换。训练用的是 GPU 版(MuJoCo Warp), 两者接触求解有细微差别, 策略要在这里也能用, 网页才有戏。
web/play/sim.js 是这里 Controller 的逐行翻译; --dump 存一段轨迹(观测、动作)给 node 端对比。
发球机的球(web/play/balls.json): balls.add_balls 往场景里加一池自由刚体, Controller.throw_ball 扔, balls.aim 算出手速度。

    python cpu_check.py                    # 几个场景各跑若干次: 站、走(前/后/侧)、被推倒再起来、从倒地姿态起身
    python cpu_check.py --balls            # 发球机: 每种球打躯干/头, 命中率和被打倒的比例
    python cpu_check.py --dump web/play/ref_trace.json
"""
import argparse
import base64
import json
import math
import time

import mujoco
import numpy as np
import torch

from balls import BALL_NEAR, BODY_R, CLOSE_T, DODGE_ACC, DODGE_HOLD, add_balls, aim, closest_approach


def f32(s):
    return np.frombuffer(base64.b64decode(s), dtype="<f4").astype(np.float64)


class Net:
    def __init__(self, p):
        self.layout = [(n, d) for n, d in p["layout"]]
        self.mean, self.inv_std = f32(p["mean"]), f32(p["inv_std"])
        self.layers = [(f32(l["w"]).reshape(l["n_out"], l["n_in"]), f32(l["b"])) for l in p["layers"]]

    def __call__(self, obs):
        x = np.clip((obs - self.mean) * self.inv_std, -10, 10)
        for i, (w, b) in enumerate(self.layers):
            x = w @ x + b
            if i < len(self.layers) - 1:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))   # ELU
        return x


def gravity(q):
    w, x, y, z = q
    return np.array([-2 * (x * z - w * y), -2 * (y * z + w * x), -(1 - 2 * (x * x + y * y))])


def rotate_inverse(q, v):
    w, u = q[0], q[1:]
    t = 2 * np.cross(u, v)
    return v - w * t + np.cross(u, t)


class Controller:
    """一个人形: 物理(MuJoCo CPU) + 走路/起身两个策略按状态切换。和 combo.Switcher + env.HumanoidEnv.step 一致。"""

    def __init__(self, cfg, xml_path, balls=None):
        self.c = cfg
        xml = open(xml_path, encoding="utf-8").read()
        self.m = mujoco.MjModel.from_xml_string(add_balls(xml, balls) if balls else xml)
        self.d = mujoco.MjData(self.m)
        self.balls = []                                    # 每个球: 种类、刚体编号、qpos/qvel 地址、阻力系数 / 质量
        if balls:
            n = len(balls["kinds"]) * balls["per_kind"]
            for kind, kd in enumerate(balls["kinds"]):
                for _ in range(balls["per_kind"]):
                    b = self.m.nbody - n + len(self.balls)
                    jnt = self.m.body_jntadr[b]
                    self.balls.append(dict(kind=kind, body=b, qadr=int(self.m.jnt_qposadr[jnt]), dadr=int(self.m.jnt_dofadr[jnt]), r=kd["r"],
                                           drag=0.5 * balls["air"]["rho"] * balls["air"]["cd"] * math.pi * kd["r"] ** 2 / kd["mass"]))
        self.nbh = self.m.nbody - len(self.balls)          # 人形(含 world)的刚体数, 球的刚体编号从这里开始
        self.head_id, self.pelvis_id = (mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("head", "pelvis"))
        self.dodge = bool(cfg.get("dodge"))                # 躲球规划(env._steer): 走路策略是带着它练的(export_web.py 导出)
        self.ball_state = np.zeros(len(self.balls), dtype=int)   # 0 停在高空, 1 飞行中(还没碰到东西), 2 碰过了
        self.ball_seq = np.full(len(self.balls), -1)       # 第几个扔出去的(回收最早的那个)
        self.walk, self.getup = Net(cfg["walk"]), Net(cfg["getup"])
        for k in ("q_def", "scale", "lo", "hi", "clip_getup"):
            setattr(self, k, np.array(cfg[k]))
        self.qadr, self.dadr = np.array(cfg["qadr"]), np.array(cfg["dadr"])
        self.feet = set(cfg["foot_bodies"])
        self.dt = cfg["timestep"] * cfg["decimation"]
        self.reset_standing()

    def reset_standing(self):
        d = self.d
        mujoco.mj_resetData(self.m, d)
        d.qpos[:] = self.m.qpos0
        d.qpos[2] = self.c["root_z0"]
        d.qpos[self.qadr] = self.q_def
        self._reset_state(mode=0)

    def reset_pose(self, qpos):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:len(qpos)] = qpos
        self._reset_state(mode=1)

    def _reset_state(self, mode):
        for i in range(len(self.balls)):
            self.park_ball(i)
        self.n_thrown, self.hits = 0, []                   # hits: 球打到人的记录 (球, 刚体, 位置), 调用方取走
        self.dodge_on, self.dodge_dir, self.ball_after, self._ended = False, np.zeros(2), 10 ** 6, False
        self.d.ctrl[:] = self.q_def
        mujoco.mj_forward(self.m, self.d)
        self.action = np.zeros(self.m.nu)
        self.phase = 0.0
        self.cmd = np.zeros(3)
        self.cmd_goal = np.zeros(3)
        self.gait = np.array(self.c["gait_stand"], dtype=float)
        self.gait_goal = self.gait.copy()
        self.mode, self.run, self.n_falls, self.n_up = mode, 0, 0, 0
        self.push_f, self.push_left = np.zeros(2), 0

    def set_cmd(self, vx, vy, wz):
        """目标指令: 实际指令按加速度限制往这里靠; 速度全 0 就是站定(着地比例 1)"""
        self.cmd_goal[:] = vx, vy, wz
        self.gait_goal[:] = self.c["gait_stand"] if abs(vx) + abs(vy) + abs(wz) < 1e-6 else self.c["gait_walk"]

    def push(self, fx, fy, steps=10):
        self.push_f[:], self.push_left = (fx, fy), steps

    def throw_ball(self, kind, pos, vel, spin=(0, 0, 0)):
        """从 pos 以速度 vel(世界系)扔一个第 kind 种的球; 优先用停着的, 都在用就回收这种球里最早扔出去的。返回球的编号"""
        i = min((i for i, b in enumerate(self.balls) if b["kind"] == kind), key=lambda i: self.ball_seq[i])
        b, d = self.balls[i], self.d
        d.qpos[b["qadr"]:b["qadr"] + 7] = (*pos, 1, 0, 0, 0)
        d.qvel[b["dadr"]:b["dadr"] + 6] = (*vel, *spin)
        d.xfrc_applied[b["body"]] = 0
        self.m.body_gravcomp[b["body"]] = 0
        self.ball_state[i], self.ball_seq[i] = 1, self.n_thrown
        self.n_thrown += 1
        return i

    def park_ball(self, i):
        """把球收回高空"""
        b, d = self.balls[i], self.d
        d.qpos[b["qadr"]:b["qadr"] + 7] = self.m.qpos0[b["qadr"]:b["qadr"] + 7]
        d.qvel[b["dadr"]:b["dadr"] + 6] = 0
        d.xfrc_applied[b["body"]] = 0
        self.m.body_gravcomp[b["body"]] = 1
        self.ball_state[i], self.ball_seq[i] = 0, -1

    def _ball_contacts(self):
        """飞行中的球第一次碰到东西: 碰地算落地, 碰到人记一次命中(躲避球规则: 落过地的球不算)"""
        for j in range(self.d.ncon):
            con = self.d.contact[j]
            b1, b2 = self.m.geom_bodyid[con.geom1], self.m.geom_bodyid[con.geom2]
            for b, o in ((b1, b2), (b2, b1)):
                i = b - self.nbh
                if i >= 0 and self.ball_state[i] == 1 and o < self.nbh:
                    self.ball_state[i] = 2
                    self._ended = True
                    if o > 0:
                        self.hits.append((i, int(o), con.pos.copy()))

    def contacts(self):
        """(脚以外碰地, 两只脚各自着地)。球碰地不算"""
        fell, feet = False, [False, False]
        fid = self.c["foot_bodies"]
        for i in range(self.d.ncon):
            con = self.d.contact[i]
            b1, b2 = self.m.geom_bodyid[con.geom1], self.m.geom_bodyid[con.geom2]
            if (b1 == 0) == (b2 == 0):
                continue
            b = b2 if b1 == 0 else b1
            if b >= self.nbh:
                continue
            if b == fid[0]:
                feet[0] = True
            elif b == fid[1]:
                feet[1] = True
            else:
                fell = True
        return fell, feet

    def _threat(self):
        """飞行中的球里预测空隙最小的那个(训练时每个世界只有一个球): (空隙, 几秒后最近, 位置, 速度); 没有飞行中的球返回 None"""
        d = self.d
        live = [i for i in range(len(self.balls)) if self.ball_state[i] == 1]
        if not live:
            return None
        pts = d.xpos[[self.head_id, self.c["torso_body"], self.pelvis_id]][None]
        vb = np.array([[d.qvel[0], d.qvel[1], 0.0]])
        best = None
        for i in live:
            b = self.balls[i]
            p, v = d.qpos[b["qadr"]:b["qadr"] + 3].copy(), d.qvel[b["dadr"]:b["dadr"] + 3].copy()
            clear, tc = closest_approach(p[None], v[None], pts, vb, -self.m.opt.gravity[2], BODY_R + b["r"])
            if best is None or clear[0] < best[0]:
                best = (clear[0], tc[0], p, v)
        return best

    def ball_obs(self):
        """球的观测块(同 env._ball_obs); 没有飞行中的球就全 0"""
        d, th = self.d, self._threat()
        if th is None:
            return np.zeros(9)
        clear, tc, p, v = th
        q = d.qpos[3:7]
        yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        c, s = math.cos(yaw), math.sin(yaw)
        hf = lambda x: np.array([c * x[0] + s * x[1], -s * x[0] + c * x[1], x[2]])
        rel = p - d.xpos[self.c["torso_body"]]
        return np.concatenate([hf(rel) / 5, hf(v) / 10, [tc / CLOSE_T, min(max(clear, -0.3), 2.0), 1.0]])

    def _steer(self):
        """手写的躲球规划(同 env._steer): 来球预测会打到时速度指令换成"往远离球路的方向全速走", 球结束后 DODGE_HOLD 秒放手"""
        d, c, th = self.d, self.c, self._threat()
        live = th is not None
        if live and th[0] < BALL_NEAR and not self.dodge_on:
            _, tc, p, v = th
            bp = p + v * tc + 0.5 * self.m.opt.gravity * tc * tc
            away = d.xpos[c["torso_body"], :2] + d.qvel[:2] * tc - bp[:2]
            self.dodge_dir = away / max(np.linalg.norm(away), 1e-6)
            self.dodge_on = True
        self.dodge_on = self.dodge_on and (live or self.ball_after * self.dt < DODGE_HOLD)
        if not self.dodge_on:
            return
        q = d.qpos[3:7]
        yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        cy, sy = math.cos(yaw), math.sin(yaw)
        hx, hy = cy * self.dodge_dir[0] + sy * self.dodge_dir[1], -sy * self.dodge_dir[0] + cy * self.dodge_dir[1]
        R = c["cmd_range"]
        ax, ay = (R["vx"][1] if hx >= 0 else -R["vx"][0]), (R["vy"][1] if hy >= 0 else -R["vy"][0])
        k = 1 / max(math.sqrt((hx / ax) ** 2 + (hy / ay) ** 2), 1e-6)
        s = DODGE_ACC * self.dt
        self.cmd += np.clip(np.array([hx * k, hy * k, 0.0]) - self.cmd, -s, s)
        self.gait[1] = c["gait_walk"][1]                   # 站定的也要迈步

    def obs(self, layout, last_action):
        d = self.d
        q = d.qpos[3:7]
        blocks = dict(gravity=lambda: gravity(q), lin_vel=lambda: rotate_inverse(q, d.qvel[:3]) * 2.0, ang_vel=lambda: d.qvel[3:6] * 0.25,
                      joint_pos=lambda: d.qpos[self.qadr], joint_vel=lambda: d.qvel[self.dadr] * 0.05, last_action=lambda: last_action,
                      cmd=lambda: self.cmd, clock=lambda: np.array([math.sin(2 * math.pi * self.phase), math.cos(2 * math.pi * self.phase)]),
                      gait=lambda: self.gait, ball=self.ball_obs)
        return np.concatenate([blocks[n]() for n, _ in layout])

    def step(self):
        """一个策略步(decimation 个物理步)。返回 (观测, 动作) 方便对比。"""
        c, d = self.c, self.d
        fell, _ = self.contacts()
        up = -gravity(d.qpos[3:7])[2]
        stood = d.qpos[2] > c["stood"]["height"] * c["root_z0"] and up > c["stood"]["up"] and not fell
        self.run = self.run + 1 if stood else 0
        if self.mode == 0 and fell:
            self.mode, self.n_falls = 1, self.n_falls + 1
        elif self.mode == 1 and self.run >= c["hold_steps"]:
            self.mode, self.n_up = 0, self.n_up + 1
        if self.mode == 0:
            o = self.obs(c["walk"]["layout"], np.clip(self.action, -c["clip_walk"], c["clip_walk"]))
            a = np.clip(self.walk(o), -c["clip_walk"], c["clip_walk"])
        else:
            o = self.obs(c["getup"]["layout"], self.action)
            a = self.getup(o)
        self.action = np.clip(a, -self.clip_getup, self.clip_getup)
        d.ctrl[:] = np.clip(self.q_def + self.action * self.scale, self.lo, self.hi)
        d.xfrc_applied[c["torso_body"], :2] = self.push_f if self.push_left > 0 else 0
        for i, b in enumerate(self.balls):                 # 球的空气阻力, 每个策略步算一次
            if self.ball_state[i]:
                v = d.qvel[b["dadr"]:b["dadr"] + 3]
                d.xfrc_applied[b["body"], :3] = -b["drag"] * self.m.body_mass[b["body"]] * math.sqrt(v @ v) * v
        self._ended = False
        for _ in range(c["decimation"]):
            mujoco.mj_step(self.m, d)
            if (self.ball_state == 1).any():
                self._ball_contacts()
        self.ball_after = 0 if self._ended else self.ball_after + 1
        self.push_left = max(0, self.push_left - 1)
        self.phase = (self.phase + self.dt * self.gait[0]) % 1.0
        if c["cmd_acc"]:                                   # 同 env._ramp_cmd
            s = c["cmd_acc"] * self.dt
            if not self.dodge_on:                          # 躲球规划接管着时不动(由 _steer 管)
                self.cmd += np.clip(self.cmd_goal - self.cmd, -s, s)
            moving = np.abs(self.cmd).sum() > 1e-6
            self.gait[0] = self.gait_goal[0]
            if not (self.gait_goal[1] >= 1 and moving and self.gait[1] < 1):
                self.gait[1] = self.gait_goal[1]
        else:
            self.cmd[:], self.gait[:] = self.cmd_goal, self.gait_goal
        if self.dodge:
            self._steer()
        return o, a


def throw_at(ctl, kind, body, speed, dist, ang, muzzle=1.0, lead=True):
    """发球机: 在人周围 dist 米、方位角 ang、离地 muzzle 米的地方, 以 speed 把第 kind 种球扔向刚体 body;
    lead = 按躯干现在的水平速度打提前量。和网页上的发球机一样。返回 (球的编号, 打得到吗)"""
    c, d = ctl.c, ctl.d
    p0 = d.xpos[body].copy()
    tb = c["torso_body"]
    src = np.array([d.xpos[tb][0] + dist * math.cos(ang), d.xpos[tb][1] + dist * math.sin(ang), muzzle])
    u = np.array([d.qvel[0], d.qvel[1], 0.0]) if lead else np.zeros(3)
    drag, g = ctl.balls[[b["kind"] for b in ctl.balls].index(kind)]["drag"], -ctl.m.opt.gravity[2]
    tf = 0.0
    for _ in range(3):                                 # 提前量: 飞行时间和落点互相依赖, 迭代几次
        vel, tf, ok = aim(src, p0 + u * tf, speed, drag, ctl.dt, c["decimation"], g)
    return ctl.throw_ball(kind, src, vel), ok


def scenario(ctl, seq, pushes, secs):
    """seq: [(秒, vx, vy, wz)], pushes: [(秒, fx, fy)](世界系)。返回 (摔倒次数, 起身次数, 结束时是否站着, 平均前进速度)"""
    n, vx = int(secs / ctl.dt), []
    for t in range(n):
        now = t * ctl.dt
        for k, *v in seq:
            if abs(now - k) < 1e-9:
                ctl.set_cmd(*v)
        for k, fx, fy in pushes:
            if abs(now - k) < 1e-9:
                ctl.push(fx, fy)
        ctl.step()
        q = ctl.d.qpos[3:7]
        yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        vx.append(math.cos(yaw) * ctl.d.qvel[0] + math.sin(yaw) * ctl.d.qvel[1])
    return ctl.n_falls, ctl.n_up, ctl.mode == 0, float(np.mean(vx[-100:]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="web/play/policies.json")
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--dump", default="", help="存一段轨迹(倒地开局 → 起身 → 走 → 被球打 → 被推)给 node 端逐步对比")
    p.add_argument("--balls", action="store_true", help="发球机测验: 每种球用两种速度打躯干和头, 站着和走着")
    a = p.parse_args()
    cfg = json.load(open(a.cfg))
    spec = json.load(open("web/play/balls.json", encoding="utf-8"))
    ctl = Controller(cfg, f"web/play/{cfg['xml']}", spec)
    bank = torch.load("models/fallen_poses.pt")
    rng = np.random.default_rng(0)
    t0 = time.time()

    if a.dump:
        ctl.reset_pose(bank["qpos"][0].double().numpy())
        ctl.set_cmd(0, 0, 0)
        tr = dict(qpos0=ctl.d.qpos.tolist(), steps=[], throws=[], aims=[])
        head = mujoco.mj_name2id(ctl.m, mujoco.mjtObj.mjOBJ_BODY, "head")
        for t in range(300):
            if t == 150:
                ctl.set_cmd(0.8, 0, 0)
            if t == 250:
                ctl.push(0, -200)
            if t in (200, 230):                            # 篮球打躯干(侧前方), 实心球打头(侧后方); 出手位置和速度存下来, node 端照扔
                kind, body, ang = (1, cfg["torso_body"], 1.0) if t == 200 else (2, head, 2.5)
                i, _ = throw_at(ctl, kind, body, 14.0, 6.0, ang)
                b = ctl.balls[i]
                tr["throws"].append(dict(step=t, kind=kind, pos=ctl.d.qpos[b["qadr"]:b["qadr"] + 3].tolist(), vel=ctl.d.qvel[b["dadr"]:b["dadr"] + 3].tolist()))
            o, act = ctl.step()
            tr["steps"].append(dict(mode=ctl.mode, obs=o.round(6).tolist(), act=act.round(6).tolist(), qpos=ctl.d.qpos.round(6).tolist(),
                                    hits=len(ctl.hits)))
        for kind, dist, dz, speed in [(0, 6, 0.3, 14), (1, 9, -0.5, 10), (2, 5, 0.6, 8), (3, 12, 0.0, 9), (0, 4, 1.0, 25)]:
            drag = ctl.balls[kind * spec["per_kind"]]["drag"]
            src, dst = [0.3, -0.2, 1.0], [0.3 + dist * 0.8, -0.2 + dist * 0.6, 1.0 + dz]
            vel, tf, ok = aim(src, dst, speed, drag, ctl.dt, cfg["decimation"], 9.81)
            tr["aims"].append(dict(src=src, dst=dst, speed=speed, drag=drag, vel=vel.tolist(), t=tf, ok=ok))
        json.dump(tr, open(a.dump, "w"))
        print("存到", a.dump)
        raise SystemExit

    if a.balls:
        head = mujoco.mj_name2id(ctl.m, mujoco.mjtObj.mjOBJ_BODY, "head")
        print(f"发球机离人 6 米、出手高 1 米, 方向随机, 打躯干/头; 每格 {a.trials} 次。命中 = 落地之前直接打到人; 摔倒 = 3 秒内倒地")
        for kind, kd in enumerate(spec["kinds"]):
            for speed in (12, 20):
                row = []
                for walk in (0, 0.8):
                    for name, body in (("躯干", cfg["torso_body"]), ("头", head)):
                        hits = falls = 0
                        for i in range(a.trials):
                            ctl.reset_standing()
                            ctl.set_cmd(walk, 0, 0)
                            for _ in range(int(1.5 / ctl.dt)):
                                ctl.step()
                            throw_at(ctl, kind, body, speed, 6.0, rng.uniform(0, 2 * math.pi))
                            for _ in range(int(3 / ctl.dt)):
                                ctl.step()
                            hits += len(ctl.hits) > 0
                            falls += ctl.n_falls > 0
                        row.append(f"{'走' if walk else '站'}·{name} 中 {hits:2d} 倒 {falls:2d}")
                print(f"{kd['name']:4s} {kd['mass']:5.2f} kg {speed:2d} m/s | " + " | ".join(row))
        print(f"用时 {time.time() - t0:.0f} 秒")
        raise SystemExit

    def run(name, setup, seq, pushes, secs):
        res = []
        for i in range(a.trials):
            setup(i)
            res.append(scenario(ctl, seq, pushes, secs))
        falls = sum(r[0] for r in res)
        ups = sum(r[1] for r in res)
        print(f"{name:28s} | 摔倒 {falls} 次, 起身 {ups} 次, 结束时站着 {sum(r[2] for r in res)}/{a.trials} | 最后 2 秒平均前进 {np.mean([r[3] for r in res]):+.2f} m/s")

    stand = lambda i: ctl.reset_standing()
    run("站 6 秒", stand, [(0, 0, 0, 0)], [], 6)
    run("往前 0.8 m/s 8 秒", stand, [(0, 0.8, 0, 0)], [], 8)
    run("往前 1.2 m/s 8 秒", stand, [(0, 1.2, 0, 0)], [], 8)
    run("倒退 0.4 m/s 8 秒", stand, [(0, -0.4, 0, 0)], [], 8)
    run("左移 0.3 m/s 8 秒", stand, [(0, 0, 0.3, 0)], [], 8)
    run("站着侧推 150 N", stand, [(0, 0, 0, 0)], [(2, 0, 150)], 6)
    run("走着从后推 200 N", stand, [(0, 0.8, 0, 0)], [(3, 200, 0)], 6)
    ang = rng.uniform(0, 2 * math.pi, a.trials)
    run("站着推 500 N(推倒)→起身→走", stand, [(0, 0, 0, 0), (6, 0.6, 0, 0)], [(1, 500, 0)], 10)
    picks = rng.choice(len(bank["qpos"]), a.trials, replace=False)
    run("倒地开局 → 起身 → 站", lambda i: ctl.reset_pose(bank["qpos"][picks[i]].double().numpy()), [(0, 0, 0, 0)], [], 6)
    print(f"用时 {time.time() - t0:.0f} 秒")
