// node 端对比: 用 sim.js 从同一个姿态跑同样的 300 步(倒地 → 起身 → 站 → 走 → 被推), 和 cpu_check.py --dump 的轨迹逐步比观测和动作。
//   python cpu_check.py --dump web/play/ref_trace.json && node web/play/check_node.mjs
import fs from 'node:fs';
import loadMujoco from '../vendor/mujoco/mujoco.js';
import { Controller } from './sim.js';

const here = new URL('.', import.meta.url);
const mujoco = await loadMujoco();
const cfg = JSON.parse(fs.readFileSync(new URL('policies.json', here)));
const xml = fs.readFileSync(new URL(cfg.xml, here), 'utf8');
const ref = JSON.parse(fs.readFileSync(new URL('ref_trace.json', here)));
const ctl = new Controller(mujoco, xml, cfg);
ctl.resetPose(ref.qpos0);
ctl.setCmd(0, 0, 0);
let worst = { obs: 0, act: 0, qpos: 0 }, firstBad = -1, modes = 0;
const t0 = performance.now();
ref.steps.forEach((r, t) => {
  if (t === 150) ctl.setCmd(0.8, 0, 0);
  if (t === 250) ctl.push(0, -200);
  const [o, a] = ctl.step();
  const diff = (x, y) => Math.max(...x.map((v, i) => Math.abs(v - y[i])));
  const e = { obs: diff(o, r.obs), act: diff(a, r.act), qpos: diff(Array.from(ctl.d.qpos), r.qpos) };
  for (const k in e) worst[k] = Math.max(worst[k], e[k]);
  if (ctl.mode !== r.mode) modes++;
  if (firstBad < 0 && e.qpos > 1e-3) firstBad = t;
  if (t % 50 === 49) console.log(`步 ${t + 1}: 模式 js ${ctl.mode} / py ${r.mode} | 观测差 ${e.obs.toExponential(1)} 动作差 ${e.act.toExponential(1)} qpos 差 ${e.qpos.toExponential(1)} | 躯干高 ${ctl.d.qpos[2].toFixed(3)}`);
});
console.log(`最大差: 观测 ${worst.obs.toExponential(2)}, 动作 ${worst.act.toExponential(2)}, qpos ${worst.qpos.toExponential(2)}; qpos 差第一次超过 1e-3 在第 ${firstBad} 步; 模式不一致 ${modes} 步; 每步 ${((performance.now() - t0) / ref.steps.length).toFixed(2)} ms`);
