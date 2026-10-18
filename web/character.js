// 人形的外观(回放页面 index.html 和互动页面 play/ 共用): 造型参数 LOOK、几何体、穿衣服(dressCharacter)、每帧连上臂(updateLinks)。
// 只影响画面, 和物理、训练无关。data 的格式同 export.py 的 model_info: bodies [{name, parent}], geoms [{body, type, size, pos, quat}]。
import * as THREE from 'three';

// ---------- 造型参数(只影响外观, 和物理、训练完全无关)。想改造型就改这里的数, 刷新页面生效 ----------
export const LOOK = {
  headSize: 0.132, headLift: 0.034, neckRadius: 0.048, eyeScale: 1.25,          // 头的半径, 头抬高多少(= 脖子长度), 脖子半径, 眼睛大小(倍数)
  shoulderLevel: 0.0, shoulderSpan: 0.33, shoulderSize: 0.052,                  // 肩头的高度(越小肩越低, 物理转轴在 0.06), 两肩头的距离, 肩头的半径
  chestWidth: 0.152, chestDepth: 0.094, chestHeight: 0.17, waistWidth: 0.138, hipWidth: 0.158,   // 胸的半宽/半厚/向下延伸多少, 腰和胯的半宽
  armThick: 1.0, legThick: 1.0, sleeve: 0.55, handSize: 0.045, footSize: 1.0,   // 手臂/腿的粗细(倍数), 袖子盖住上臂的比例(1 = 到肘), 手的大小, 鞋的大小(倍数)
  outline: 0.011,                                                               // 描边粗细(0 = 不描边)
  skin: '#f3d9be', hair: '#3a2a22', pants: '#2b3445', shoes: '#f2f2f2',         // 衣服的颜色按跑道区分(SHIRTS)
};
const ARM_REST = 0.28;                              // 上臂建模时的长度, 每帧按肩头到肘的实际距离拉伸(见 updateLinks)
const SPINE_REST = [[0, 0, 0], [-0.01, 0, -0.26], [-0.01, 0, -0.425]];   // 站直时 torso / waist_lower / pelvis 的原点在躯干坐标系里的位置(来自 humanoid.xml), 躯干蒙皮的绑定姿势
const Z_AXIS = new THREE.Vector3(0, 0, 1), D2R = Math.PI / 180;
const lerp = (a, b, t) => a + (b - a) * t;
const smooth = (a, b, x) => { const t = Math.min(Math.max((x - a) / (b - a), 0), 1); return t * t * (3 - 2 * t); };   // x = a 时为 0, x = b 时为 1, 中间平滑

// 几何体按参数缓存: 同一次构建里所有人形共用同一份, 重建时统一释放
let GEO = new Map();
const memo = (key, make) => { let g = GEO.get(key); if (!g) { g = make(); GEO.set(key, g); } return g; };
// 沿 z 轴的一段肢体: radii 是从近端(-z)到远端(+z)等距的几个半径, 中间平滑过渡, 两端是半球, 球心相距 2*half。
// cut < 1 时只做近端的一截并在 cut 处收口(袖子); pad 是比 radii 粗出来的量
function limbGeo(radii, half, cut = 1, pad = 0) {
  return memo(`lg${radii}|${half}|${cut}|${pad}`, () => {
    const curve = new THREE.SplineCurve(radii.map((r, i) => new THREE.Vector2(r + pad, -half + 2 * half * i / (radii.length - 1))));
    const pts = [], N = 8, r0 = radii[0] + pad, r1 = radii[radii.length - 1] + pad;
    for (let i = 0; i < N; i++) { const th = -Math.PI / 2 + (Math.PI / 2) * i / N; pts.push(new THREE.Vector2(r0 * Math.cos(th), -half + r0 * Math.sin(th))); }
    if (cut < 1) {
      for (let i = 0; i <= 12; i++) pts.push(curve.getPoint(cut * i / 12));
      const e = pts[pts.length - 1]; pts.push(new THREE.Vector2(e.x - pad - 0.004, e.y + 0.001));      // 袖口向里收, 贴住手臂
    } else {
      pts.push(...curve.getPoints(20));
      for (let i = 1; i <= N; i++) { const th = (Math.PI / 2) * i / N; pts.push(new THREE.Vector2(r1 * Math.cos(th), half + r1 * Math.sin(th))); }
    }
    return new THREE.LatheGeometry(pts, 24).rotateX(Math.PI / 2);
  });
}
// 把一圈圈的点(每圈点数相同)依次连成一张封闭的皮, 两头用扇形封口。索引的顺序: 第一圈的封口, 各圈之间的带子, 最后一圈的封口
function loftGeo(rings) {
  const n = rings[0].length, K = rings.length, pos = [], idx = [];
  for (const r of rings) for (const p of r) pos.push(p.x, p.y, p.z);
  const mid = (r) => { const c = new THREE.Vector3(); for (const p of r) c.add(p); c.divideScalar(n); pos.push(c.x, c.y, c.z); return pos.length / 3 - 1; };
  const c0 = mid(rings[0]), c1 = mid(rings[K - 1]);
  for (let i = 0; i < n; i++) idx.push(c0, i, (i + 1) % n);
  for (let k = 0; k + 1 < K; k++) for (let i = 0; i < n; i++) { const a = k * n + i, b = k * n + (i + 1) % n; idx.push(a, a + n, b, b, a + n, b + n); }
  for (let i = 0; i < n; i++) idx.push(c1, (K - 1) * n + (i + 1) % n, (K - 1) * n + i);
  const P = (i) => new THREE.Vector3(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
  let vol = 0;                                       // 圈的绕向反了整张皮会里外翻转: 用体积的正负检查, 反了就把每个三角形翻过来
  for (let i = 0; i < idx.length; i += 3) vol += P(idx[i]).dot(P(idx[i + 1]).cross(P(idx[i + 2])));
  if (vol < 0) for (let i = 0; i < idx.length; i += 3) [idx[i + 1], idx[i + 2]] = [idx[i + 2], idx[i + 1]];
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.setIndex(idx); g.computeVertexNormals();
  return g;
}
// 躯干: 从脖子根到裆部的一张皮(绑定姿势, 躯干坐标系), 横截面是略方的椭圆。材质分两组: 0 = 衣服(到下摆), 1 = 裤子。
// 每个顶点按高度把权重分给 torso / waist_lower / pelvis, 过渡区正好跨在两个腰部关节上, 弯腰扭腰时是平滑弯过去的
function torsoGeo(L) {
  return memo('torso', () => {
    const zTop = 0.078, zc = 0.07 - L.chestHeight, zw = -0.26, zh = -0.425, zBot = -0.535, zHem = -0.345, N = 32, E = 0.82;
    const shape = (z) => {                           // 这个高度上的 [半宽, 半厚, 中心的 x]
      if (z >= zc) { const f = Math.sqrt(Math.max(1 - ((z - zc) / (zTop - zc)) ** 2, 0)); return [L.chestWidth * f, L.chestDepth * f, -0.005]; }
      if (z >= zw) { const t = smooth(zc, zw, z); return [lerp(L.chestWidth, L.waistWidth, t), lerp(L.chestDepth, 0.088, t), lerp(-0.005, -0.01, t)]; }
      if (z >= zh) { const t = smooth(zw, zh, z); return [lerp(L.waistWidth, L.hipWidth, t), lerp(0.088, 0.106, t), lerp(-0.01, -0.03, t)]; }
      const f = Math.sqrt(Math.max(1 - ((zh - z) / (zh - zBot)) ** 2, 0)); return [L.hipWidth * f, 0.106 * f, -0.03];
    };
    const zs = [];
    for (let i = 0; i < 7; i++) zs.push(zc + (zTop - zc) * Math.sin((80 - 80 * i / 6) * D2R));      // 胸腔顶部的圆顶
    for (let i = 1; i <= 8; i++) zs.push(lerp(zc, zh, i / 8));
    for (let i = 1; i <= 5; i++) zs.push(zh - (zh - zBot) * Math.sin(16 * i * D2R));                 // 胯底收口
    const rings = [], ring = (z, grow) => rings.push(Array.from({ length: N }, (_, i) => {
      const th = 2 * Math.PI * i / N, c = Math.cos(th), s = Math.sin(th), [ry, rx, cx] = shape(z);
      return new THREE.Vector3(cx + (rx + grow) * Math.sign(c) * Math.abs(c) ** E, (ry + grow) * Math.sign(s) * Math.abs(s) ** E, z);
    }));
    let hemAt = 0;
    for (const z of zs) {
      if (!hemAt && z < zHem + 0.012) { ring(zHem, 0.008); hemAt = rings.length; ring(zHem - 0.002, 0); }   // 衣服下摆比裤腰大一圈, 形成一个小台阶
      if (Math.abs(z - zHem) >= 0.012) ring(z, hemAt ? 0 : 0.008 * smooth(zHem + 0.09, zHem, z));
    }
    const g = loftGeo(rings), nShirt = N * 3 + hemAt * N * 6, pa = g.attributes.position, si = [], sw = [];
    g.addGroup(0, nShirt, 0); g.addGroup(nShirt, g.index.count - nShirt, 1);
    for (let i = 0; i < pa.count; i++) {
      const z = pa.getZ(i), wT = 1 - smooth(-0.15, -0.24, z), wP = smooth(-0.29, -0.36, z);
      si.push(0, 1, 2, 0); sw.push(wT, 1 - wT - wP, wP, 0);
    }
    g.setAttribute('skinIndex', new THREE.Uint16BufferAttribute(si, 4)); g.setAttribute('skinWeight', new THREE.Float32BufferAttribute(sw, 4));
    return g;
  });
}
// 鞋(脚坐标系: x 朝前, 鞋底贴着物理的脚底 z = -0.027)。从鞋尖到鞋跟一圈圈连起来; sole = true 时是鞋底那一片
function shoeGeo(fs, sole) {
  return memo(`shoe${fs}|${sole}`, () => {
    const ST = [[0.158, 0.014, 0.012], [0.148, 0.032, 0.024], [0.122, 0.048, 0.034], [0.082, 0.054, 0.043], [0.04, 0.053, 0.058],
                [0, 0.049, 0.076], [-0.04, 0.046, 0.082], [-0.07, 0.041, 0.074], [-0.09, 0.03, 0.056], [-0.098, 0.014, 0.03]];   // [x, 半宽, 鞋面高]
    const N = 20, x0 = 0.03, z0 = -0.027;            // 放大缩小时 (x0, z0) 不动, 鞋底始终踩在地上
    return loftGeo(ST.map(([x, w, h]) => Array.from({ length: N }, (_, i) => {
      const th = 2 * Math.PI * i / N, c = Math.cos(th), s = Math.sin(th);
      const y = sole ? (w + 0.005) * Math.sign(c) * Math.sqrt(Math.abs(c)) : w * c;
      const z = sole ? 0.011 + 0.011 * Math.sign(s) * Math.sqrt(Math.abs(s)) : 0.016 + (s > 0 ? h : 0.012) * s;
      return new THREE.Vector3(x0 + (x - x0) * fs * (sole ? 1.03 : 1), y * fs, z0 + z * fs);   // 鞋底比鞋面长一点, 两头的封口才不会叠在同一个平面上闪烁
    })));
  });
}
// 头发: 从头顶铺到发际线的一张皮(原点在头的中心, 面朝 +x)。发际线的高度随方位角变化: 前面露出额头, 鬓角垂到耳朵前面, 耳朵上方拱起来, 后面盖到后颈
const HAIRLINE = [[0, 26], [28, 30], [52, 20], [68, 0], [80, -9], [93, 9], [112, -12], [135, -30], [180, -36]];   // [离正前方的方位角, 发际线的仰角](度), 左右对称
function hairGeo(R) {
  return memo('hair', () => {
    const N = 48, K = 9, rings = [];
    const line = (az) => { let i = 0; while (i < HAIRLINE.length - 2 && az > HAIRLINE[i + 1][0]) i++; return lerp(HAIRLINE[i][1], HAIRLINE[i + 1][1], smooth(HAIRLINE[i][0], HAIRLINE[i + 1][0], az)); };
    for (let k = 0; k <= K + 1; k++) {
      const s = k > K ? 1 : Math.max(k / K, 0.04), rho = k > K ? 0.96 : 1.045 + 0.06 * (1 - s * s);   // 头顶厚、发梢薄; 最后一圈向里收, 贴住头皮
      rings.push(Array.from({ length: N }, (_, i) => {
        const az = -180 + 360 * i / N, th = s * (90 - line(Math.abs(az))) * D2R, sn = R * rho * Math.sin(th);
        return new THREE.Vector3(sn * Math.cos(az * D2R), 0.97 * sn * Math.sin(az * D2R), 1.02 * R * rho * Math.cos(th));
      }));
    }
    return loftGeo(rings);
  });
}
const ell = (rx, ry, rz) => memo(`e${rx}|${ry}|${rz}`, () => new THREE.SphereGeometry(1, 36, 24).scale(rx, ry, rz));
const ball = (r) => memo(`b${r}`, () => new THREE.SphereGeometry(r, 28, 18));
const capZ = (r, len) => memo(`cz${r}|${len}`, () => new THREE.CapsuleGeometry(r, len, 6, 16).rotateX(Math.PI / 2));
const cylZ = (r0, r1, h) => memo(`yz${r0}|${r1}|${h}`, () => new THREE.CylinderGeometry(r0, r1, h, 20).rotateX(Math.PI / 2));
const ringZ = (r, tube) => memo(`rz${r}|${tube}`, () => new THREE.TorusGeometry(r, tube, 10, 28));                 // 平放的圆环(衣领)
function outlineMaterial() {                         // 描边: 背面外扩一点点, 做出手办/卡通的轮廓线
  const th = LOOK.outline, m = new THREE.MeshBasicMaterial({ color: 0x0c1018, side: THREE.BackSide });
  m.visible = th > 0;
  m.onBeforeCompile = (sh) => { sh.vertexShader = sh.vertexShader.replace('#include <begin_vertex>', `#include <begin_vertex>\n transformed += normalize(normal) * ${th.toFixed(5)};`); };
  m.customProgramCacheKey = () => 'outline' + th.toFixed(5);   // 厚度不同要重新编译, 否则会复用旧的
  return m;
}
export function makeMats(shirt) {
  return {
    shirt: new THREE.MeshStandardMaterial({ color: shirt, roughness: 0.42 }),
    skin: new THREE.MeshStandardMaterial({ color: new THREE.Color(LOOK.skin), roughness: 0.5 }),
    pants: new THREE.MeshStandardMaterial({ color: new THREE.Color(LOOK.pants), roughness: 0.6 }),
    shoes: new THREE.MeshStandardMaterial({ color: new THREE.Color(LOOK.shoes), roughness: 0.4 }),
    sole: new THREE.MeshStandardMaterial({ color: 0x3b4252, roughness: 0.7 }),
    hair: new THREE.MeshStandardMaterial({ color: new THREE.Color(LOOK.hair), roughness: 0.55 }),
    eye: new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.3 }),
    pupil: new THREE.MeshStandardMaterial({ color: 0x141820, roughness: 0.2 }),
    outline: outlineMaterial(),
  };
}
export function dressCharacter(data, bodyObjs, mats) {
  const L = LOOK, links = [], fore = {};
  const idx = (n) => data.bodies.findIndex((b) => b.name === n);
  const quat = (g) => new THREE.Quaternion(g.quat[1], g.quat[2], g.quat[3], g.quat[0]);
  const part = (bi, geo, kind, pos = [0, 0, 0], q = null, ol = true) => {       // ol = false: 不描边(脸上的小零件描了边会糊成一团)
    const m = new THREE.Mesh(geo, mats[kind]); m.castShadow = ol; m.position.set(...pos); if (q) m.quaternion.copy(q);
    bodyObjs[bi].add(m);
    if (ol) { const o = new THREE.Mesh(geo, mats.outline); o.position.copy(m.position); o.quaternion.copy(m.quaternion); bodyObjs[bi].add(o); }
  };
  const seg = (bi, A, B, radii, kind) => {             // 刚体坐标系里从 A 到 B 的一段肢体(radii: 从 A 到 B 的半径)
    const d = B.clone().sub(A), len = d.length();
    part(bi, limbGeo(radii, len / 2), kind, A.clone().add(B).multiplyScalar(0.5).toArray(), new THREE.Quaternion().setFromUnitVectors(Z_AXIS, d.normalize()));
  };
  const link = (a, pa, b, pb, parts) => {              // 两端分别钉在两个刚体上的一组几何体([几何体, 材质]...), 每帧重新摆放(见 updateLinks)
    const meshes = [];
    for (const [geo, kind] of parts) { const m = new THREE.Mesh(geo, mats[kind]); m.castShadow = true; meshes.push(m, new THREE.Mesh(geo, mats.outline)); }
    bodyObjs[0].add(...meshes);                        // 0 号是 world 刚体, 永远不动, 相当于直接放在这个人形的坐标系里
    links.push({ a, pa, b, pb, meshes });
  };
  // 胶囊的两个端点(刚体坐标系), 离刚体原点(= 关节)近的那一端在前。
  // MuJoCo 的 fromto 胶囊 z 轴是从 to 指向 from 的, 不能假设 +z 朝远端: 以前这里搞反了, 四肢的粗细和肩膀的裁剪全跑到了另一头
  const ends = (g) => {
    const P = new THREE.Vector3(...g.pos), h = new THREE.Vector3(0, 0, g.size[1]).applyQuaternion(quat(g));
    const a = P.clone().sub(h), b = P.clone().add(h);
    return a.lengthSq() <= b.lengthSq() ? [a, b] : [b, a];
  };
  const lt = L.legThick, at = L.armThick, hs = L.handSize, mul = (r, k) => r.map((v) => v * k);
  const LIMB = {   // 从近端到远端的半径(带一点肌肉的起伏), 材质, 近端/远端各自沿轴向外延长多少(物理胶囊到关节之间有空隙, 一直画到关节才连得上)
    thigh: [mul([0.075, 0.075, 0.066, 0.056], lt), 'pants', 0, 0.035], shin: [mul([0.054, 0.058, 0.047, 0.039], lt), 'pants', 0, 0.03],
    lower_arm: [mul([0.036, 0.038, 0.031, 0.026], at), 'skin', 0.017, 0],
  };
  for (const g of data.geoms) {
    const nm = data.bodies[g.body].name, bi = g.body, base = nm.replace(/_(left|right)$/, '');
    if (base === 'torso' || base === 'foot' || base === 'head' || base === 'waist_lower' || base === 'pelvis' || base === 'upper_arm') continue;   // 下面单独做
    if (LIMB[base] && g.type === 'capsule') {
      const [radii, k, extA, extB] = LIMB[base], [a, b] = ends(g), u = b.clone().sub(a).normalize();
      if (base === 'lower_arm') fore[bi] = u.clone();
      seg(bi, a.addScaledVector(u, -extA), b.addScaledVector(u, extB), radii, k);
    }
    else if (base === 'hand') {
      // 手: 连指手套的形状。手指顺着前臂(u); 手掌的法线沿肘关节的转轴(e, 来自 humanoid.xml), 这样手臂下垂时掌心朝向大腿;
      // 拇指在屈肘时前臂转过去的那一侧(w = e × u)
      const s = nm.endsWith('right') ? 1 : -1, u = fore[data.bodies[bi].parent] || new THREE.Vector3(1, s, 1).normalize();
      const e = new THREE.Vector3(0, -1, s).normalize(), w = e.clone().cross(u).normalize(), P = new THREE.Vector3(...g.pos);
      const q = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(w, e, u));
      part(bi, ell(0.78 * hs, 0.42 * hs, 1.25 * hs), 'skin', P.clone().addScaledVector(u, 0.4 * hs).toArray(), q);
      part(bi, ell(0.26 * hs, 0.26 * hs, 0.58 * hs), 'skin', P.clone().addScaledVector(w, 0.72 * hs).addScaledVector(u, 0.12 * hs).toArray(),
           q.clone().multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), 0.6)));
    }
    else part(bi, g.type === 'capsule' ? capZ(g.size[0], g.size[1] * 2) : ball(g.size[0]), 'shirt', g.pos, quat(g));
  }
  // 躯干: 胸、腰、胯是一张蒙皮, 骨头就是 torso / waist_lower / pelvis 三个刚体(不再是三个球叠在一起)
  const t = idx('torso'), hd = idx('head'), wl = idx('waist_lower'), pv = idx('pelvis');
  const skel = new THREE.Skeleton([t, wl, pv].map((i) => bodyObjs[i]), SPINE_REST.map((p) => new THREE.Matrix4().makeTranslation(-p[0], -p[1], -p[2])));
  for (const mat of [[mats.shirt, mats.pants], mats.outline]) {
    const m = new THREE.SkinnedMesh(torsoGeo(L), mat); m.castShadow = Array.isArray(mat); m.frustumCulled = false;   // 人会走远, 而包围球停在原地, 所以不做视锥剔除
    m.bind(skel, new THREE.Matrix4()); bodyObjs[0].add(m);
  }
  // 肩和上臂。物理上肩关节的转轴在躯干坐标系 (0, ±0.17, 0.06), 离下巴只有几厘米, 照着画就是耸肩。
  // 所以视觉上的肩头单独定位(shoulderLevel / shoulderSpan), 固定在躯干上; 上臂不跟 upper_arm 刚体走, 而是每帧从肩头连到
  // 肘关节(lower_arm 的原点): 肩怎么调两头都不会脱开, 代价是上臂的长度会随手臂姿势有一点变化
  const armR = mul([0.045, 0.044, 0.037], at);
  for (const [side, sgn] of [['right', -1], ['left', 1]]) {
    const la = idx('lower_arm_' + side); if (la < 0) continue;
    const S = new THREE.Vector3(0, sgn * L.shoulderSpan / 2, L.shoulderLevel), rs = L.shoulderSize;
    seg(t, new THREE.Vector3(-0.005, sgn * 0.035, 0.03), S, [0.045, rs * 0.9], 'shirt');     // 斜方肌: 从脖子根斜着下到肩头
    part(t, ball(rs), 'shirt', S.toArray());                                                   // 肩头
    link(t, S, la, new THREE.Vector3(), [[limbGeo(armR, ARM_REST / 2), 'skin'], [limbGeo(armR, ARM_REST / 2, L.sleeve, 0.007), 'shirt']]);   // 上臂 + 袖子
    part(la, ball(0.0375 * at), 'skin');                                                       // 肘
  }
  // 脖子和衣领
  const R = L.headSize, es = L.eyeScale, hz = 0.012 + L.headLift;        // hz: 头的中心在头坐标系里的高度
  const neckBottom = 0.05, neckTop = 0.19 + hz - R + 0.03;                // 头底再往里 3 cm, 保证和头连在一起
  if (neckTop > neckBottom) part(t, cylZ(L.neckRadius * 0.96, L.neckRadius * 1.08, neckTop - neckBottom), 'skin', [0.004, 0, (neckBottom + neckTop) / 2]);
  part(t, ringZ(L.neckRadius * 1.08 + 0.004, 0.008), 'shirt', [0.006, 0, 0.071], new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), 0.14));
  // 头(面朝 +x): 比物理的球大一点, 略扁。hp 给出头表面上的点, yaw 朝左为正, pitch 朝上为正(度), rho = 1 正好在表面上
  const hp = (yaw, pitch, rho = 1) => { const cp = Math.cos(pitch * D2R); return new THREE.Vector3(R * rho * cp * Math.cos(yaw * D2R), 0.95 * R * rho * cp * Math.sin(yaw * D2R), hz + 1.02 * R * rho * Math.sin(pitch * D2R)); };
  const decal = (kind, yaw, pitch, a, b, lift, rot = 0, bend = 0) => {    // 贴在头表面上的椭圆片(眼睛、眉毛、嘴): 半轴 a × b 度, 比头皮高出 lift, 转 rot 度, 两端翘起 bend 度
    bodyObjs[hd].add(new THREE.Mesh(memo(`dc${[yaw, pitch, a, b, lift, rot, bend]}`, () => {
      const pos = [], nor = [], ix = [], n = 24, M = 3, cr = Math.cos(rot * D2R), sr = Math.sin(rot * D2R);
      const put = (u, v) => { const p = hp(yaw + (u * cr - v * sr) / Math.cos(pitch * D2R), pitch + u * sr + v * cr, 1 + lift); pos.push(p.x, p.y, p.z); p.z -= hz; p.normalize(); nor.push(p.x, p.y, p.z); };
      put(0, 0);
      for (let m = 1; m <= M; m++) for (let i = 0; i < n; i++) { const c = m / M * Math.cos(2 * Math.PI * i / n); put(a * c, b * m / M * Math.sin(2 * Math.PI * i / n) + bend * c * c); }
      for (let i = 0; i < n; i++) ix.push(0, 1 + i, 1 + (i + 1) % n);
      for (let m = 1; m < M; m++) for (let i = 0; i < n; i++) { const p = 1 + (m - 1) * n + i, q = 1 + (m - 1) * n + (i + 1) % n; ix.push(p, p + n, q, q, p + n, q + n); }
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3)); g.setIndex(ix);
      return g;
    }), mats[kind]));
  };
  part(hd, ell(R, 0.95 * R, 1.02 * R), 'skin', [0, 0, hz]);
  part(hd, hairGeo(R), 'hair', [0, 0, hz]);
  part(hd, ell(R * 0.075, R * 0.06, R * 0.065), 'skin', hp(0, -8, 0.985).toArray(), null, false);                 // 鼻子
  decal('pupil', 0, -30, 6.5, 1.2, 0.004, 0, 2.4);                                                                // 嘴(两角上翘)
  for (const sgn of [-1, 1]) {
    part(hd, ell(R * 0.09, R * 0.055, R * 0.15), 'skin', [-0.04 * R, sgn * 0.955 * R, hz - 0.03 * R]);            // 耳朵
    decal('pupil', sgn * 21, 4, 7.9 * es, 9.9 * es, 0.004);                                                       // 眼眶(比眼白大一圈, 露出来的就是眼线)
    decal('eye', sgn * 21, 4, 7 * es, 9 * es, 0.009);                                                             // 眼白
    decal('pupil', sgn * 20, 3.4, 4.6 * es, 6.4 * es, 0.014);                                                     // 瞳孔
    decal('eye', sgn * 20 + 2.2 * es, 3.4 + 3 * es, 1.5 * es, 1.5 * es, 0.019);                                   // 高光(两只眼睛都在同一侧)
    decal('hair', sgn * 21, 9.5 + 9 * es, 7.5, 1.5, 0.006, sgn * -3);                                             // 眉毛
  }
  // 脚: 鞋面 + 鞋底, 底面就是脚底的接触面
  const fs = L.footSize;
  for (const side of ['right', 'left']) {
    const f = idx('foot_' + side);
    if (f >= 0) {
      part(f, shoeGeo(fs, false), 'shoes'); part(f, shoeGeo(fs, true), 'sole');
      part(f, ball(0.043 * lt), 'pants', [0, 0, 0.055]);         // 脚踝(裤脚): 在踝关节转轴附近, 脚怎么转都和小腿、鞋连着
    }
    for (const j of ['thigh_', 'shin_']) {                       // 关节球: 髋、膝(放在膝关节转轴上), 让腿的连接处圆润
      const bi2 = idx(j + side);
      if (bi2 >= 0) part(bi2, ball((j === 'thigh_' ? 0.076 : 0.06) * lt), 'pants', [0, 0, j === 'thigh_' ? 0 : 0.02]);
    }
  }
  return { links, skel };
}
const tmpA = new THREE.Vector3(), tmpB = new THREE.Vector3(), tmpQ = new THREE.Quaternion();
export function updateLinks(W) {                             // 刚体就位后, 把两头钉在不同刚体上的几何体(上臂和袖子)重新连好
  for (const k of W.links) {
    const a = W.bodyObjs[k.a], b = W.bodyObjs[k.b];
    tmpA.copy(k.pa).applyQuaternion(a.quaternion).add(a.position);
    tmpB.copy(k.pb).applyQuaternion(b.quaternion).add(b.position);
    const len = tmpA.distanceTo(tmpB);
    tmpQ.setFromUnitVectors(Z_AXIS, tmpB.sub(tmpA).divideScalar(len || 1));
    tmpB.multiplyScalar(len / 2).add(tmpA);            // 中点
    for (const m of k.meshes) { m.position.copy(tmpB); m.quaternion.copy(tmpQ); m.scale.z = len / ARM_REST; }
  }
}

export function disposeGeometry() {                  // 重建人形前释放缓存的几何体
  for (const g of GEO.values()) g.dispose();
  GEO = new Map();
}
