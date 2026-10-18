// 网页互动版的物理 + 策略: cpu_check.py 里 Controller 的逐行翻译(那边是参考实现, 改这里要同步改那边)。
// 浏览器和 node 都能用: 调用方负责把 MuJoCo 模块、模型 XML 文本、policies.json(export_web.py 导出)、balls.json(可选)交进来。
//   const ctl = new Controller(mujoco, xmlText, cfg, balls); ctl.resetStanding(); ctl.setCmd(0.8, 0, 0); ctl.step();
// 一个策略步 = cfg.decimation 个物理步(50 Hz)。mode: 0 走路策略, 1 起身策略。
// 发球机: addBalls 往场景里加一池球, ctl.throwBall 扔, aim 算出手速度(都和 cpu_check.py 一致)。

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

// 往场景里加一池球: 每种 per_kind 个自由刚体, 排在人形后面(人形的刚体编号、qpos 地址都不变)。
// 不用的球停在高空(gravcomp=1 抵消重力, 碰不到任何东西), 扔的时候才放下来。关键帧的 qpos 长度会对不上, 删掉(没用到)。
export function addBalls(xml, spec) {
  xml = xml.replace(/<keyframe>[\s\S]*?<\/keyframe>/, '');
  const fr = spec.friction.join(' '), [px, py, pz] = spec.park;
  let bodies = '', k = 0;
  for (const kd of spec.kinds) {
    for (let j = 0; j < spec.per_kind; j++, k++) {
      bodies += `<body name="ball_${kd.id}_${j}" pos="${px + 1.5 * k} ${py} ${pz}" gravcomp="1"><freejoint/>`
        + `<geom type="sphere" size="${kd.r}" mass="${kd.mass}" shellinertia="${kd.shell}" `
        + `priority="1" condim="6" friction="${fr}" solref="${kd.solref[0]} ${kd.solref[1]}"/></body>`;
    }
  }
  return xml.replace('</worldbody>', bodies + '</worldbody>');
}

// 从 src 以 speed 扔到 dst(世界系)要的出手速度, 低弹道, 算上空气阻力 drag = 阻力系数 / 质量(加速度 = -drag |v| v)。
// 和 MuJoCo 里一样离散积分(阻力每个策略步算一次, 物理步半隐式欧拉)。返回 {vel, t, ok}; 打不到(ok = false)时给打得最高的角度
export function aim(src, dst, speed, drag, dt, dec, g) {
  const dx = dst[0] - src[0], dy = dst[1] - src[1], dist = Math.hypot(dx, dy);
  const [ex, ey] = dist > 1e-6 ? [dx / dist, dy / dist] : [1, 0];
  const h = dt / dec;
  const fly = (th) => {                                // 飞到水平距离 dist 时的 [高度, 时间]; 飞不到返回 null
    let vs = speed * Math.cos(th), vz = speed * Math.sin(th), s = 0, z = src[2], t = 0;
    while (t < 3 && z > -1) {
      const v = Math.hypot(vs, vz), as = -drag * v * vs, az = -drag * v * vz - g;
      for (let k = 0; k < dec; k++) {
        const s0 = s, z0 = z;
        vs += h * as; vz += h * az; s += h * vs; z += h * vz; t += h;
        if (s >= dist) { const f = (dist - s0) / (s - s0); return [z0 + f * (z - z0), t - h * (1 - f)]; }
      }
    }
    return null;
  };
  const err = (th) => { const r = fly(th); return r === null ? -1e9 : r[0] - dst[2]; };
  const ths = [];
  for (let a = -40; a < 62; a += 2) ths.push(a * (Math.PI / 180));
  const es = ths.map(err);
  const ok = es.some((e) => e >= 0);
  let th;
  if (ok) {                                            // 第一个打高了的角度和前一个之间二分
    const i = es.findIndex((e) => e >= 0);
    let [lo, hi] = i > 0 ? [ths[i - 1], ths[i]] : [ths[0], ths[0]];
    for (let k = 0; k < 20; k++) { const mid = (lo + hi) / 2; if (err(mid) >= 0) hi = mid; else lo = mid; }
    th = hi;
  } else {
    th = ths[es.indexOf(Math.max(...es))];
  }
  const r = fly(th), c = speed * Math.cos(th);
  return { vel: [c * ex, c * ey, speed * Math.sin(th)], t: r ? r[1] : 3.0, ok };
}

export class Controller {
  constructor(mujoco, xmlText, cfg, balls = null) {
    this.mj = mujoco; this.c = cfg;
    this.m = mujoco.MjModel.from_xml_string(balls ? addBalls(xmlText, balls) : xmlText);
    this.d = new mujoco.MjData(this.m);
    const m = this.m;
    this.balls = [];                                   // 每个球: 种类、刚体编号、qpos/qvel 地址、阻力系数 / 质量
    if (balls) {
      const n = balls.kinds.length * balls.per_kind;
      balls.kinds.forEach((kd, kind) => {
        for (let j = 0; j < balls.per_kind; j++) {
          const b = m.nbody - n + this.balls.length, jnt = m.body_jntadr[b];
          this.balls.push({ kind, body: b, qadr: m.jnt_qposadr[jnt], dadr: m.jnt_dofadr[jnt],
                            drag: 0.5 * balls.air.rho * balls.air.cd * Math.PI * kd.r ** 2 / kd.mass });
        }
      });
    }
    this.nbh = m.nbody - this.balls.length;            // 人形(含 world)的刚体数, 球的刚体编号从这里开始
    this.ballState = new Int8Array(this.balls.length); // 0 停在高空, 1 飞行中(还没碰到东西), 2 碰过了
    this.ballSeq = new Array(this.balls.length).fill(-1);   // 第几个扔出去的(回收最早的那个)
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
    for (let i = 0; i < this.balls.length; i++) this.parkBall(i);
    this.nThrown = 0; this.hits = [];                  // hits: 球打到人的记录 [球, 刚体, 位置], 调用方取走
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

  throwBall(kind, pos, vel, spin = [0, 0, 0]) {        // 从 pos 以速度 vel(世界系)扔一个第 kind 种的球; 优先用停着的, 都在用就回收这种球里最早扔出去的。返回球的编号
    let i = -1;
    this.balls.forEach((b, j) => { if (b.kind === kind && (i < 0 || this.ballSeq[j] < this.ballSeq[i])) i = j; });
    const b = this.balls[i], d = this.d;
    d.qpos.set([...pos, 1, 0, 0, 0], b.qadr);
    d.qvel.set([...vel, ...spin], b.dadr);
    d.xfrc_applied.fill(0, b.body * 6, b.body * 6 + 6);
    this.m.body_gravcomp[b.body] = 0;
    this.ballState[i] = 1; this.ballSeq[i] = this.nThrown++;
    return i;
  }

  parkBall(i) {                                        // 把球收回高空
    const b = this.balls[i], d = this.d, m = this.m;
    d.qpos.set(m.qpos0.subarray(b.qadr, b.qadr + 7), b.qadr);
    d.qvel.fill(0, b.dadr, b.dadr + 6);
    d.xfrc_applied.fill(0, b.body * 6, b.body * 6 + 6);
    m.body_gravcomp[b.body] = 1;
    this.ballState[i] = 0; this.ballSeq[i] = -1;
  }

  _ballContacts() {                                    // 飞行中的球第一次碰到东西: 碰地算落地, 碰到人记一次命中(躲避球规则: 落过地的球不算)
    const vec = this.d.contact, n = this.d.ncon;
    for (let j = 0; j < n; j++) {
      const con = vec.get(j);
      const b1 = this.bodyOf[con.geom1], b2 = this.bodyOf[con.geom2];
      for (const [b, o] of [[b1, b2], [b2, b1]]) {
        const i = b - this.nbh;
        if (i >= 0 && this.ballState[i] === 1 && o < this.nbh) {
          this.ballState[i] = 2;
          if (o > 0) this.hits.push([i, o, Array.from(con.pos)]);
        }
      }
      con.delete();
    }
    vec.delete();
  }

  contacts() {                                         // [脚以外碰地, [右脚着地, 左脚着地]]。球碰地不算
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
      if (b >= this.nbh) continue;
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
    const qvel = d.qvel;
    this.balls.forEach((b, i) => {                     // 球的空气阻力, 每个策略步算一次
      if (!this.ballState[i]) return;
      const v0 = qvel[b.dadr], v1 = qvel[b.dadr + 1], v2 = qvel[b.dadr + 2];
      const k = -b.drag * m.body_mass[b.body] * Math.sqrt(v0 * v0 + v1 * v1 + v2 * v2);
      xf[b.body * 6] = k * v0; xf[b.body * 6 + 1] = k * v1; xf[b.body * 6 + 2] = k * v2;
    });
    for (let k = 0; k < c.decimation; k++) {
      this.mj.mj_step(m, d);
      if (this.ballState.includes(1)) this._ballContacts();
    }
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
