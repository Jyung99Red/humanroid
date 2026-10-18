// node 端对比: 用 sim.js 从同一个姿态跑同样的 300 步(倒地 → 起身 → 站 → 走 → 被球打 → 被推), 和 cpu_check.py --dump 的轨迹逐步比观测、动作、qpos(含球)、命中数,
// 再比几组发球机瞄准(aim)的结果。
//   python cpu_check.py --dump web/play/ref_trace.json && node web/play/check_node.mjs
import fs from 'node:fs';
import loadMujoco from '../vendor/mujoco/mujoco.js';
import { Controller, aim } from './sim.js';

const here = new URL('.', import.meta.url);
const mujoco = await loadMujoco();
const cfg = JSON.parse(fs.readFileSync(new URL('policies.json', here)));
const balls = JSON.parse(fs.readFileSync(new URL('balls.json', here)));
const xml = fs.readFileSync(new URL(cfg.xml, here), 'utf8');
const ref = JSON.parse(fs.readFileSync(new URL('ref_trace.json', here)));
const ctl = new Controller(mujoco, xml, cfg, balls);
ctl.resetPose(ref.qpos0);
ctl.setCmd(0, 0, 0);
let worst = { obs: 0, act: 0, qpos: 0 }, firstBad = -1, modes = 0, hitsBad = 0;
const t0 = performance.now();
ref.steps.forEach((r, t) => {
  if (t === 150) ctl.setCmd(0.8, 0, 0);
  if (t === 250) ctl.push(0, -200);
  for (const th of ref.throws) if (th.step === t) ctl.throwBall(th.kind, th.pos, th.vel);
  const [o, a] = ctl.step();
  const diff = (x, y) => Math.max(...x.map((v, i) => Math.abs(v - y[i])));
  const e = { obs: diff(o, r.obs), act: diff(a, r.act), qpos: diff(Array.from(ctl.d.qpos), r.qpos) };
  for (const k in e) worst[k] = Math.max(worst[k], e[k]);
  if (ctl.mode !== r.mode) modes++;
  if (ctl.hits.length !== r.hits) hitsBad++;
  if (firstBad < 0 && e.qpos > 1e-3) firstBad = t;
  if (t % 50 === 49) console.log(`步 ${t + 1}: 模式 js ${ctl.mode} / py ${r.mode} | 观测差 ${e.obs.toExponential(1)} 动作差 ${e.act.toExponential(1)} qpos 差 ${e.qpos.toExponential(1)} | 躯干高 ${ctl.d.qpos[2].toFixed(3)} | 命中 ${ctl.hits.length}`);
});
const ms = (performance.now() - t0) / ref.steps.length;
console.log(`最大差: 观测 ${worst.obs.toExponential(2)}, 动作 ${worst.act.toExponential(2)}, qpos ${worst.qpos.toExponential(2)}; qpos 差第一次超过 1e-3 在第 ${firstBad} 步; 模式不一致 ${modes} 步; 命中数不一致 ${hitsBad} 步(命中 ${ctl.hits.length} 次: ${ctl.hits.map(([i, b]) => `球 ${i} → 刚体 ${b}`).join(', ')}); 每步 ${ms.toFixed(2)} ms`);
let aimWorst = 0, aimOk = 0;
for (const c of ref.aims) {
  const r = aim(c.src, c.dst, c.speed, c.drag, ctl.dt, cfg.decimation, 9.81);
  aimWorst = Math.max(aimWorst, ...r.vel.map((v, i) => Math.abs(v - c.vel[i])), Math.abs(r.t - c.t));
  aimOk += r.ok === c.ok;
}
console.log(`瞄准: ${ref.aims.length} 组, 打得到与否一致 ${aimOk} 组, 出手速度/飞行时间最大差 ${aimWorst.toExponential(2)}`);
