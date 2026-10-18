"""网页互动版的参考实现 + 可行性检查: 只用 export_web.py 导出的 web/play/policies.json, 在 MuJoCo 的 CPU 版(和浏览器里的 WebAssembly 同一套 C 代码)
里跑走路 + 起身切换。训练用的是 GPU 版(MuJoCo Warp), 两者接触求解有细微差别, 策略要在这里也能用, 网页才有戏。
web/play/sim.js 是这里 Controller 的逐行翻译; --dump 存一段轨迹(观测、动作)给 node 端对比。

    python cpu_check.py                    # 几个场景各跑若干次: 站、走(前/后/侧)、被推倒再起来、从倒地姿态起身
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

    def __init__(self, cfg, xml_path):
        self.c = cfg
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.d = mujoco.MjData(self.m)
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
        self.d.qpos[:] = qpos
        self._reset_state(mode=1)

    def _reset_state(self, mode):
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

    def contacts(self):
        """(脚以外碰地, 两只脚各自着地)"""
        fell, feet = False, [False, False]
        fid = self.c["foot_bodies"]
        for i in range(self.d.ncon):
            con = self.d.contact[i]
            b1, b2 = self.m.geom_bodyid[con.geom1], self.m.geom_bodyid[con.geom2]
            if (b1 == 0) == (b2 == 0):
                continue
            b = b2 if b1 == 0 else b1
            if b == fid[0]:
                feet[0] = True
            elif b == fid[1]:
                feet[1] = True
            else:
                fell = True
        return fell, feet

    def obs(self, layout, last_action):
        d = self.d
        q = d.qpos[3:7]
        blocks = dict(gravity=lambda: gravity(q), lin_vel=lambda: rotate_inverse(q, d.qvel[:3]) * 2.0, ang_vel=lambda: d.qvel[3:6] * 0.25,
                      joint_pos=lambda: d.qpos[self.qadr], joint_vel=lambda: d.qvel[self.dadr] * 0.05, last_action=lambda: last_action,
                      cmd=lambda: self.cmd, clock=lambda: np.array([math.sin(2 * math.pi * self.phase), math.cos(2 * math.pi * self.phase)]),
                      gait=lambda: self.gait)
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
        for _ in range(c["decimation"]):
            mujoco.mj_step(self.m, d)
        self.push_left = max(0, self.push_left - 1)
        self.phase = (self.phase + self.dt * self.gait[0]) % 1.0
        if c["cmd_acc"]:                                   # 同 env._ramp_cmd
            s = c["cmd_acc"] * self.dt
            self.cmd += np.clip(self.cmd_goal - self.cmd, -s, s)
            moving = np.abs(self.cmd).sum() > 1e-6
            self.gait[0] = self.gait_goal[0]
            if not (self.gait_goal[1] >= 1 and moving and self.gait[1] < 1):
                self.gait[1] = self.gait_goal[1]
        else:
            self.cmd[:], self.gait[:] = self.cmd_goal, self.gait_goal
        return o, a


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
    p.add_argument("--dump", default="", help="存一段轨迹(倒地开局 → 起身 → 走 → 被推)给 node 端逐步对比")
    a = p.parse_args()
    cfg = json.load(open(a.cfg))
    ctl = Controller(cfg, f"web/play/{cfg['xml']}")
    bank = torch.load("models/fallen_poses.pt")
    rng = np.random.default_rng(0)
    t0 = time.time()

    if a.dump:
        ctl.reset_pose(bank["qpos"][0].double().numpy())
        ctl.set_cmd(0, 0, 0)
        tr = dict(qpos0=ctl.d.qpos.tolist(), steps=[])
        for t in range(300):
            if t == 150:
                ctl.set_cmd(0.8, 0, 0)
            if t == 250:
                ctl.push(0, -200)
            o, act = ctl.step()
            tr["steps"].append(dict(mode=ctl.mode, obs=o.round(6).tolist(), act=act.round(6).tolist(), qpos=ctl.d.qpos.round(6).tolist()))
        json.dump(tr, open(a.dump, "w"))
        print("存到", a.dump)
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
