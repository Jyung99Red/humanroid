"""把官方人形 humanoid.xml 变成训练用的 PD 版模型, 并按预设调整身体。

    python make_model.py                  # official: 只把 motor 换成 PD, 其余不动 -> models/humanoid_pd.xml (walk1~walk4 用的就是它)
    python make_model.py human50          # 50 kg 真人比例的质量分布和力气 -> models/human50_pd.xml

PD 执行器: 策略输出"每个关节想去哪个角度", 底层 PD 去追。kp = KP*gear, kv = KV*gear, 力矩限制在 ±gear 内。

human50 相对官方模型改了什么(数字来源见 README"身体模型"一节, 用 check_body.py 复查):
  - 总质量 50 kg, 各段质量按 Winter/Dempster 人体测量表分配(官方的躯干太轻、四肢太重)。几何不变, 只改密度, 所以回放页面不用动。
  - 力气 = 真人静态最大力矩(每公斤体重, HUMAN_ISO) x 体重, 各关节之间的比例和真人一致。
  - 转得越快力气越小: 每个关节加阻尼 = 最大力矩 / MUSCLE_SPEED, 关节以 MUSCLE_SPEED 转时力矩被抵消完。
    没有这一项的话执行器任何转速都出满力, 照真人力气给会跳到 85 cm(世界级)。
    阻尼也让关节被迫弯曲时(落地缓冲)阻力更大, 和肌肉"离心收缩更有力"一致。MUSCLE_SPEED 用起跳测验校准到普通成年人水平。
  - 关节活动范围向真人靠拢: 脊柱前屈 45°/后伸 30°(官方 30°/75°)、踝背屈 35°(官方 50°)、髋外展 45°、髋后伸 30°。
  - 积分器改 implicitfast: 力气和阻尼加大后, 原来的显式积分在脚、手这种轻的部位会发散。
  - 加脖子(官方模型头和胸是焊死的): 左右转 neck_z ±70°, 点头 neck_y 抬头 45° / 低头 50°, 转动中心在头中心下方 9 cm(头底)。
    动作维数 21 -> 23, 新关节排在最后。头上的第一人称相机跟着头转。
"""
import argparse
import re

import mujoco

# 各刚体占体重的比例(Winter / Dempster): 躯干总共 49.7%, 按胸 : 腰 : 胯 = 0.42 : 0.29 : 0.29 分到 torso / waist_lower / pelvis;
# 左右对称的段每侧一份。
BODY_FRAC = {"head": 0.081, "torso": 0.497 * 0.42, "waist_lower": 0.497 * 0.29, "pelvis": 0.497 * 0.29}
for side in ("right", "left"):
    for seg, frac in dict(thigh=0.10, shin=0.0465, foot=0.0145, upper_arm=0.028, lower_arm=0.016, hand=0.006).items():
        BODY_FRAC[f"{seg}_{side}"] = frac

# 真人静态最大力矩, Nm/公斤体重(健康成年人的中间量级, 不是某一篇文献的精确值)。髋/膝是屈伸里较强的伸, 踝是踮脚方向
HUMAN_ISO = dict(hip_y=3.5, knee=3.5, ankle_y=2.5, ankle_x=0.75, hip_x=1.75, hip_z=1.0, abdomen_y=2.5, abdomen_x=1.75, abdomen_z=1.0,
                 shoulder1=1.25, shoulder2=1.25, elbow=1.25, neck_y=0.5, neck_z=0.3)

PRESETS = {
    "official": {},
    "human50": dict(
        mass=50.0,
        integrator="implicitfast",
        strength=HUMAN_ISO,
        muscle_speed=15.0,          # 弧度/秒
        ranges={"hip_x": ("-30 10", "-45 25"), "hip_y": ("-150 20", "-150 30"), "abdomen_y": ("-75 30", "-30 45"), "ankle": ("-50 50", "-35 50")},
        neck=True,
    ),
}


def segment_densities(src, mass):
    """用默认密度 1000 编译一次, 按各刚体的目标质量求密度。返回 {刚体名: 密度}。"""
    m = mujoco.MjModel.from_xml_string(src)
    return {b: 1000.0 * f * mass / float(m.body(b).mass[0]) for b, f in BODY_FRAC.items()}


CLASS_OF = {"thigh": "thigh", "shin": "shin", "foot": "foot", "upper_arm": "arm_upper", "lower_arm": "arm_lower", "hand": "hand"}


def set_geom_density(src, dens):
    """把密度写成显式属性: 四肢的 geom 在 class 默认里(左右共用), 躯干和头的 geom 直接写在刚体里。"""
    for seg, cls in CLASS_OF.items():
        src, n = re.subn(r'(<default class="%s">\s*<geom )' % cls, lambda mm: mm.group(1) + f'density="{dens[seg + "_right"]:.1f}" ', src, count=1)
        assert n == 1, cls
    for name, body in (("torso", "torso"), ("waist_upper", "torso"), ("head", "head"), ("waist_lower", "waist_lower"), ("butt", "pelvis")):
        src, n = re.subn(r'(<geom name="%s" )' % name, lambda mm: mm.group(1) + f'density="{dens[body]:.1f}" ', src, count=1)
        assert n == 1, name
    return src


def build(preset, kp_gain=2.0, kv_gain=0.1):
    cfg = PRESETS[preset]
    src = open("models/humanoid.xml", encoding="utf-8").read()
    if cfg.get("mass"):
        dens = segment_densities(src, cfg["mass"])
        src = set_geom_density(src, dens)
    if cfg.get("integrator"):
        src, n = re.subn(r'<option timestep="0.005"', f'<option timestep="0.005" integrator="{cfg["integrator"]}"', src, count=1)
        assert n == 1
    if cfg.get("neck"):                                 # 关节加在头刚体里, 执行器加在最后(gear 只是占位, 下面按 strength 重算)
        neck_joints = ('        <joint name="neck_z" pos="0 0 -.09" axis="0 0 1" range="-70 70"/>\n'
                       '        <joint name="neck_y" pos="0 0 -.09" axis="0 1 0" range="-45 50"/>\n')
        src, n = re.subn(r'(<body name="head" pos="0 0 .19">\n)', lambda mm: mm.group(1) + neck_joints, src, count=1)
        assert n == 1
        neck_motors = ('    <motor name="neck_z" gear="20" joint="neck_z"/>\n'
                       '    <motor name="neck_y" gear="20" joint="neck_y"/>\n')
        src, n = re.subn(r'(  </actuator>)', lambda mm: neck_motors + mm.group(1), src, count=1)
        assert n == 1
    for key, (old, new) in cfg.get("ranges", {}).items():
        if key == "abdomen_y":
            pat = r'(<joint name="abdomen_y"[^>]*range=")%s(")' % re.escape(old)
        else:
            pat = r'(<default class="%s">\s*<joint [^>]*range=")%s(")' % (key, re.escape(old))
        src, n = re.subn(pat, lambda mm: mm.group(1) + new + mm.group(2), src, count=1)
        assert n == 1, f"没找到范围 {key} {old}"

    pat = re.compile(r'<motor name="(?P<name>\w+)"\s+gear="(?P<gear>[\d.]+)"\s+joint="(?P<joint>\w+)"\s*/>')
    gear = {mm.group("joint"): float(mm.group("gear")) for mm in pat.finditer(src)}
    if cfg.get("strength"):
        for j in gear:
            gear[j] = round(cfg["strength"][j.replace("_right", "").replace("_left", "")] * cfg["mass"])
    if cfg.get("muscle_speed"):
        for j, g in gear.items():                       # 写在关节元素上, 覆盖 class 里的默认阻尼
            src, n = re.subn(r'(<joint name="%s")' % j, lambda mm: mm.group(1) + f' damping="{g / cfg["muscle_speed"]:.2f}"', src, count=1)
            assert n == 1, j

    def repl(mm):
        name, joint = mm.group("name"), mm.group("joint")
        g = gear[joint]
        return (f'<position name="{name}" joint="{joint}" kp="{kp_gain * g:g}" kv="{kv_gain * g:g}" '
                f'forcerange="-{g:g} {g:g}" ctrllimited="false"/>')

    dst, n = pat.subn(repl, src)
    assert n == len(gear), f"有 {len(gear)} 个 motor, 换了 {n} 个"
    return dst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("preset", nargs="?", default="official", choices=list(PRESETS))
    ap.add_argument("--kp", type=float, default=2.0)
    ap.add_argument("--kv", type=float, default=0.1)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or ("models/humanoid_pd.xml" if a.preset == "official" else f"models/{a.preset}_pd.xml")
    open(out, "w", encoding="utf-8").write(build(a.preset, a.kp, a.kv))
    m = mujoco.MjModel.from_xml_path(out)
    print(f"{out}: 预设 {a.preset}, 总质量 {m.body_mass.sum():.1f} kg, kp={a.kp}*gear, kv={a.kv}*gear")
