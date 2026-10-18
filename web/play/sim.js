// 网页互动版的物理 + 策略: cpu_check.py 里 Controller 的逐行翻译(那边是参考实现, 改这里要同步改那边)。
// 浏览器和 node 都能用: 调用方负责把 MuJoCo 模块、模型 XML 文本、policies.json(export_web.py 导出)交进来。
//   const ctl = new Controller(mujoco, xmlText, cfg); ctl.resetStanding(); ctl.setCmd(0.8, 0, 0); ctl.step();
// 一个策略步 = cfg.decimation 个物理步(50 Hz)。mode: 0 走路策略, 1 起身策略。

function f32(b64) {
  const bin = typeof atob === 'function' ? atob(b64) : Buffer.from(b64, 'base64').toString('binary');
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(u8.buffer);
}

class Net {                                            // 演员网络: (观测 - 均值) / 标准差, 裁 ±10; 隐藏层 ELU, 最后一层线性
  constructor(p) {
    this.layout = p.layout;
    this.mean = f32(p.mean); this.invStd = f32(p.inv_std);
    this.layers = p.layers.map(l => ({ nIn: l.n_in, nOut: l.n_out, w: f32(l.w), b: f32(l.b) }));
    this.buf = this.layers.map(l => new Float64Array(l.nOut));
  }
  run(obs) {
    let x = new Float64Array(obs.length);
    for (let i = 0; i < obs.length; i++) x[i] = Math.min(10, Math.max(-10, (obs[i] - this.mean[i]) * this.invStd[i]));
    this.layers.forEach((l, k) => {
      const y = this.buf[k], last = k === this.layers.length - 1;
      for (let o = 0; o < l.nOut; o++) {
        let s = l.b[o];
        const row = o * l.nIn;
        for (let i = 0; i < l.nIn; i++) s += l.w[row + i] * x[i];
        y[o] = last || s > 0 ? s : Math.expm1(s);
      }
      x = y;
    });
    return Array.from(x);
  }
}

export function gravity(q) {                           // 机体系下的重力方向, q = [w, x, y, z]
  const [w, x, y, z] = q;
  return [-2 * (x * z - w * y), -2 * (y * z + w * x), -(1 - 2 * (x * x + y * y))];
}

function rotateInverse(q, v) {                         // 世界系向量 -> 机体系
  const w = q[0], u = [q[1], q[2], q[3]];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const t = cross(u, v).map(e => 2 * e);
  const ut = cross(u, t);
  return [0, 1, 2].map(i => v[i] - w * t[i] + ut[i]);
}

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

export class Controller {
  constructor(mujoco, xmlText, cfg) {
    this.mj = mujoco; this.c = cfg;
    this.m = mujoco.MjModel.from_xml_string(xmlText);
    this.d = new mujoco.MjData(this.m);
    this.walk = new Net(cfg.walk); this.getup = new Net(cfg.getup);
    this.nu = this.m.nu;
    this.dt = cfg.timestep * cfg.decimation;
    this.bodyOf = Array.from(this.m.geom_bodyid);
    this.resetStanding();
  }

  resetStanding() {
    const { m, d, c } = this;
    this.mj.mj_resetData(m, d);
    const qpos = d.qpos;
    qpos.set(m.qpos0);
    qpos[2] = c.root_z0;
    c.qadr.forEach((a, i) => { qpos[a] = c.q_def[i]; });
    this._resetState(0);
  }

  resetPose(q) {                                       // 任意姿态开局(例如倒地), 由起身策略接手
    this.mj.mj_resetData(this.m, this.d);
    this.d.qpos.set(q);
    this._resetState(1);
  }

  _resetState(mode) {
    const { c, d } = this;
    d.ctrl.set(c.q_def);
    this.mj.mj_forward(this.m, d);
    this.action = new Array(this.nu).fill(0);
    this.phase = 0;
    this.cmd = [0, 0, 0]; this.cmdGoal = [0, 0, 0];
    this.gait = [...c.gait_stand]; this.gaitGoal = [...c.gait_stand];
    this.mode = mode; this.run = 0; this.nFalls = 0; this.nUp = 0;
    this.pushF = [0, 0]; this.pushLeft = 0;
  }

  setCmd(vx, vy, wz) {                                 // 目标指令; 实际指令按加速度限制往这里靠。速度全 0 = 站定
    this.cmdGoal = [vx, vy, wz];
    this.gaitGoal = [...(Math.abs(vx) + Math.abs(vy) + Math.abs(wz) < 1e-6 ? this.c.gait_stand : this.c.gait_walk)];
  }

  push(fx, fy, steps = 10) { this.pushF = [fx, fy]; this.pushLeft = steps; }   // 推躯干(世界系, 牛), 默认 0.2 秒

  contacts() {                                         // [脚以外碰地, [右脚着地, 左脚着地]]
    const { d } = this, fid = this.c.foot_bodies;
    let fell = false;
    const feet = [false, false];
    const vec = d.contact, n = d.ncon;
    for (let i = 0; i < n; i++) {
      const con = vec.get(i);
      const b1 = this.bodyOf[con.geom1], b2 = this.bodyOf[con.geom2];
      con.delete();
      if ((b1 === 0) === (b2 === 0)) continue;
      const b = b1 === 0 ? b2 : b1;
      if (b === fid[0]) feet[0] = true;
      else if (b === fid[1]) feet[1] = true;
      else fell = true;
    }
    vec.delete();
    return [fell, feet];
  }

  obs(layout, lastAction) {
    const { d, c } = this;
    const qpos = d.qpos, qvel = d.qvel;
    const q = [qpos[3], qpos[4], qpos[5], qpos[6]];
    const blocks = {
      gravity: () => gravity(q),
      lin_vel: () => rotateInverse(q, [qvel[0], qvel[1], qvel[2]]).map(v => v * 2.0),
      ang_vel: () => [qvel[3] * 0.25, qvel[4] * 0.25, qvel[5] * 0.25],
      joint_pos: () => c.qadr.map(a => qpos[a]),
      joint_vel: () => c.dadr.map(a => qvel[a] * 0.05),
      last_action: () => lastAction,
      cmd: () => this.cmd,
      clock: () => [Math.sin(2 * Math.PI * this.phase), Math.cos(2 * Math.PI * this.phase)],
      gait: () => this.gait,
    };
    return layout.flatMap(([name]) => blocks[name]());
  }

  step() {                                             // 一个策略步; 返回 [观测, 动作](对比用)
    const { c, d, m } = this;
    const [fell] = this.contacts();
    const qpos = d.qpos;
    const up = -gravity([qpos[3], qpos[4], qpos[5], qpos[6]])[2];
    const stood = qpos[2] > c.stood.height * c.root_z0 && up > c.stood.up && !fell;
    this.run = stood ? this.run + 1 : 0;
    if (this.mode === 0 && fell) { this.mode = 1; this.nFalls++; }
    else if (this.mode === 1 && this.run >= c.hold_steps) { this.mode = 0; this.nUp++; }
    let o, a;
    if (this.mode === 0) {
      const cw = c.clip_walk;
      o = this.obs(c.walk.layout, this.action.map(v => clamp(v, -cw, cw)));
      a = this.walk.run(o).map(v => clamp(v, -cw, cw));
    } else {
      o = this.obs(c.getup.layout, this.action);
      a = this.getup.run(o);
    }
    this.action = a.map((v, i) => clamp(v, -c.clip_getup[i], c.clip_getup[i]));
    const ctrl = d.ctrl;
    this.action.forEach((v, i) => { ctrl[i] = clamp(c.q_def[i] + v * c.scale[i], c.lo[i], c.hi[i]); });
    const xf = d.xfrc_applied, tb = c.torso_body * 6;
    xf[tb] = this.pushLeft > 0 ? this.pushF[0] : 0;
    xf[tb + 1] = this.pushLeft > 0 ? this.pushF[1] : 0;
    for (let k = 0; k < c.decimation; k++) this.mj.mj_step(m, d);
    this.pushLeft = Math.max(0, this.pushLeft - 1);
    this.phase = (this.phase + this.dt * this.gait[0]) % 1.0;
    if (c.cmd_acc) {                                   // 同 env._ramp_cmd
      const s = c.cmd_acc * this.dt;
      this.cmd = this.cmd.map((v, i) => v + clamp(this.cmdGoal[i] - v, -s, s));
      const moving = Math.abs(this.cmd[0]) + Math.abs(this.cmd[1]) + Math.abs(this.cmd[2]) > 1e-6;
      this.gait[0] = this.gaitGoal[0];
      if (!(this.gaitGoal[1] >= 1 && moving && this.gait[1] < 1)) this.gait[1] = this.gaitGoal[1];
    } else {
      this.cmd = [...this.cmdGoal]; this.gait = [...this.gaitGoal];
    }
    return [o, a];
  }
}
