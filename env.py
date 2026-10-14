"""批量人形环境: N 个人形同时在 GPU 上跑, 观测/奖励/重置全是 torch 张量, 不回 CPU。

动作: 21 维, 乘以每个关节的幅度加到默认姿态上, 作为 PD 的目标角度。
观测由有名字的块拼成(OBS_BLOCKS 里定义, 顺序和选哪些由 obs / priv 参数决定), 默认:
  演员 77 维 = 重力方向(机体系) 3 | 线速度(机体系) 3 | 角速度(机体系) 3 | 关节角 21 | 关节速度 21 | 上一步动作 21 | 速度指令 3 | 步态时钟 2
  评论家另外看到 3 维 = 躯干高度 1 | 两只脚是否着地(真实接触) 2
策略文件记下自己的块布局 [(名字, 维数), ...]; 续训时多了新块, ppo.PPO.load 会给网络第一层补零列, 已有行为不变。
指令 = 速度 (vx, vy, wz) + 步态 (步频 f, 着地比例 D), 站/走/跑都用这一套:
  步频 f: 每秒几个步态周期(一个周期左右脚各迈一步, 步数/分 = 120 f); 步态时钟按 f 走。
  着地比例 D: 每只脚在一个周期里该踩地的时间占比, 右脚在时钟 [0, D) 踩地, 左脚晚半个周期。
  D = 0.5 左右脚严格轮换(以前唯一的模式); D > 0.5 有双脚支撑, 是走; D < 0.5 有腾空, 是跑; D = 1 两脚一直踩地, 是站。
摔倒 = 脚以外的任何部位碰到地面(扫接触点, 不看高度和倾角, 所以下蹲、弯腰、落地缓冲都不算摔)。
坐标约定: MuJoCo 四元数 wxyz, Z 轴朝上; 自由关节线速度在世界系, 角速度在机体系。
"""
import json
import math

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

wp.config.quiet = True

# 奖励权重(正数是奖励, 惩罚项在代码里取负)
REWARD = dict(
    lin_vel=1.0,       # 跟踪前进/侧移速度指令
    ang_vel=0.5,       # 跟踪转向速度指令
    alive=0.2,         # 活着就给一点
    clock=0.3,         # 脚的着地节奏和步态时钟对齐
    upright=1.0,       # 惩罚躯干倾斜
    action_rate=0.01,  # 惩罚动作变化太快(抽搐)
    energy=5e-4,       # 惩罚耗能
    arm_dev=0.0,       # 惩罚手臂偏离默认姿势(只在给了 arm_pose 时有意义, 例如 0.1)
    arm_swing=0.0,     # 手臂跟着步态时钟前后摆、和同侧的腿反相(只在给了 arm_pose 时有意义, 例如 0.5; 用它时把 arm_dev 设成 0, 两者是冲突的)
    foot_roll=0.0,     # 惩罚脚左右翻转(内翻/外翻, 看起来像要崴脚), 例如 2.0。前后俯仰不罚, 保留脚跟先着地、脚尖蹬地
    foot_yaw=0.0,      # 惩罚脚尖不朝前(内八/外八), 例如 1.5
    foot_slip=0.0,     # 惩罚脚踩在地上时还在滑(看脚中心的速度; 蹬地时脚绕脚尖转也会被罚, 新配方用 point_slip), 例如 1.0
    point_slip=0.0,    # 脚跟点、脚尖点谁贴地(离地 < 1 cm)谁就不许水平移动; 绕脚尖转动不罚。例如 3.0
    head_level=0.0,    # 头保持水平、朝前(惩罚头相对地面的倾斜和脖子左右转; 只在有脖子的身体上有意义), 例如 5.0
    waist_twist=0.0,   # 惩罚腰左右扭(abdomen_z)和侧弯(abdomen_x), 即骨盆相对胸口的扭转和侧倾。不罚前后弯。例如 2.0
    pelvis_level=0.0,  # 惩罚骨盆相对地面左右侧倾(走路像模特扭胯); 真人约 ±4~5°。例如 10.0
    step_width=0.0,    # 两脚左右间距(朝向坐标系)小于 STEP_WIDTH_MIN 时罚, 贴着中线或交叉时罚满 1。例如 1.0
    heel_down=0.0,     # 每次着地的前 70%(按步态时钟)脚跟要踩实: 罚脚尖朝下(踮脚); 最后 30% 留给蹬地, 落地时脚尖朝上不罚。例如 2.0
    toe_off=0.0,       # 每次着地的最后 30% 脚还踩着地时脚跟抬起、脚尖压地(蹬地)给奖励, 压到 20° 封顶(每只脚最多 1)。例如 0.5
    swing_flat=0.0,    # 摆动中段(进度 25%~75%)脚要放平: 罚脚相对地面前后倾斜(朝上朝下都罚)。例如 2.0
    foot_clear=0.0,    # 步态时钟说该摆动的脚, 离地高度要跟上 FOOT_CLEAR_H * sin(pi * 摆动进度), 低了就罚。例如 1.0
)
STEP_WIDTH_MIN = 0.08     # 米, 真人走路两脚间距约 8~12 cm
FOOT_CLEAR_H = 0.05       # 米, 摆动中途脚底抬起的高度(脚刚体原点平放时离地 0.027)。真人摆到中途脚底只离地几厘米; 定成 10 cm 会逼出踮脚走


def quat_to_gravity(q):
    """机体系下的重力方向: 世界 (0,0,-1) 用四元数的逆旋转。站直时约等于 (0,0,-1)。"""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack([
        -2 * (x * z - w * y),
        -2 * (y * z + w * x),
        -(1 - 2 * (x * x + y * y)),
    ], dim=1)


def rotate(q, v):
    """把机体系向量 v 旋转到世界系(q 是机体的四元数)。"""
    w, xyz = q[:, :1], q[:, 1:]
    t = 2 * torch.cross(xyz, v, dim=1)
    return v + w * t + torch.cross(xyz, t, dim=1)


FOOT_POINTS = ((-0.07, 0.0, -0.027), (0.14, 0.0, -0.027))   # 脚跟、脚尖的底面点(脚坐标系; 脚的胶囊从 -0.07 到 0.14, 半径 0.027)


def rotate_inverse(q, v):
    """把世界系向量 v 旋转到四元数 q 的机体系。"""
    w, xyz = q[:, :1], q[:, 1:]
    t = 2 * torch.cross(xyz, v, dim=1)
    return v - w * t + torch.cross(xyz, t, dim=1)


def arm_swing_dirs(mjm, pose):
    """上臂绕左右轴往前摆 1 弧度时, 两个肩关节各要转多少弧度。返回 {关节名: 弧度}。
    肩关节的两根轴都是斜的, 没法直接看出来, 所以在手臂的默认姿势 pose 附近用运动学数值求: 让肘的位移等于"绕 -y 轴转"的位移。"""
    d = mujoco.MjData(mjm)
    jadr = lambda n: mjm.jnt_qposadr[mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, n)]
    bid = lambda n: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, n)
    base = mjm.qpos0.copy()                              # 躯干站直、朝向 +x, 所以世界系就是躯干系
    for n, v in pose.items():
        base[jadr(n)] = v
    out = {}
    for side in ("right", "left"):
        adr = [jadr(f"shoulder1_{side}"), jadr(f"shoulder2_{side}")]
        ua, la = bid(f"upper_arm_{side}"), bid(f"lower_arm_{side}")

        def elbow(dq):                                   # 肩到肘的向量
            d.qpos[:] = base
            d.qpos[adr] += dq
            mujoco.mj_kinematics(mjm, d)
            return d.xpos[la] - d.xpos[ua]

        r0, eps = elbow([0, 0]), 1e-5
        jac = np.stack([(elbow([eps, 0]) - r0) / eps, (elbow([0, eps]) - r0) / eps], axis=1)
        sol = np.linalg.lstsq(jac, np.cross([0, -1, 0], r0), rcond=None)[0]
        out[f"shoulder1_{side}"], out[f"shoulder2_{side}"] = float(sol[0]), float(sol[1])
    return out

# 观测块: 名字 -> 说明。实际计算在 HumanoidEnv._block_fns 里; 新加输入就在那里加一项, 不要改已有块的含义和缩放
OBS_BLOCKS = dict(
    gravity="重力方向(机体系)", lin_vel="线速度(机体系, x2)", ang_vel="角速度(机体系, x0.25)", joint_pos="关节角",
    joint_vel="关节速度(x0.05)", last_action="上一步动作", cmd="速度指令 vx,vy,wz", clock="步态时钟 sin/cos", gait="步态指令 步频,着地比例",
    height="躯干高度", foot_contact="两只脚是否着地",
)
DEFAULT_OBS = ("gravity", "lin_vel", "ang_vel", "joint_pos", "joint_vel", "last_action", "cmd", "clock")
DEFAULT_PRIV = ("height", "foot_contact")          # 只给评论家(训练时的特权信息)


class HumanoidEnv:
    def __init__(self, nworld, xml="models/humanoid_pd.xml", decimation=4, ep_len=500, action_frac=0.3,
                 act_clip=1.5, gait_period=0.8, cmd_vx=(0.4, 1.2), cmd_vy=(0.0, 0.0),
                 cmd_wz=(-0.3, 0.3), cmd_period=250, nconmax=64, njmax=160, seed=0, reward=None, random_yaw=True, arm_pose=None,
                 arm_swing_amp=0.35, obs=DEFAULT_OBS, priv=DEFAULT_PRIV, cmd_freq=None, cmd_duty=(0.5, 0.5)):
        self.n, self.dev = nworld, torch.device("cuda:0")
        assert not (set(obs) | set(priv)) - set(OBS_BLOCKS), f"未知观测块 {(set(obs) | set(priv)) - set(OBS_BLOCKS)}, 可用: {list(OBS_BLOCKS)}"
        self.obs_names = list(obs)
        self.priv_names = [k for k in priv if k not in self.obs_names]     # 演员已经看到的块, 评论家不重复
        self.dec, self.ep_len, self.act_clip = decimation, ep_len, act_clip
        self.cmd_period, self.random_yaw = cmd_period, random_yaw
        self.cmd_rng = (cmd_vx, cmd_vy, cmd_wz)
        # 步态指令的范围; 没给步频就用固定的 1 / gait_period(旧配置只有 gait_period)
        self.gait_rng = (cmd_freq or (1.0 / gait_period, 1.0 / gait_period), cmd_duty)
        self.rw = dict(REWARD, **(reward or {}))
        torch.manual_seed(seed)
        wp.init()

        self.mjm = mjm = mujoco.MjModel.from_xml_path(xml)
        self.dt = mjm.opt.timestep * decimation          # 一个策略步的时间
        self.nu = mjm.nu
        # 执行器顺序 != qpos 顺序, 建立映射
        jid = mjm.actuator_trnid[:, 0]
        qadr = torch.tensor(mjm.jnt_qposadr[jid], dtype=torch.long)
        dadr = torch.tensor(mjm.jnt_dofadr[jid], dtype=torch.long)
        rng = torch.tensor(mjm.jnt_range[jid], dtype=torch.float32)
        self.qadr, self.dadr = qadr.to(self.dev), dadr.to(self.dev)
        self.lo, self.hi = rng[:, 0].to(self.dev), rng[:, 1].to(self.dev)
        self.q_def = torch.tensor(mjm.qpos0[mjm.jnt_qposadr[jid]], dtype=torch.float32, device=self.dev)
        # 手臂默认姿势: 官方模型的默认是手臂斜向前伸(像僵尸), 给了 arm_pose(关节名->弧度)就改成自然下垂
        names = [mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, int(j)) for j in jid]
        self.arm_idx = torch.tensor([i for i, n in enumerate(names) if n.startswith(("shoulder", "elbow"))], dtype=torch.long, device=self.dev)
        self.arm_pose = arm_pose
        if arm_pose:
            pose = json.load(open(arm_pose)) if isinstance(arm_pose, str) else arm_pose
            for i, n in enumerate(names):
                if n in pose:
                    self.q_def[i] = pose[n]
        self.scale = action_frac * (self.hi - self.lo)    # 每个关节动作 ±1 对应的角度幅度
        if arm_pose:
            self.scale[self.arm_idx] *= 0.5               # 手臂幅度减半, 不然探索阶段会乱甩
        # 摆臂的目标: 手臂各关节相对默认姿势的偏移 = 这个向量 x 摆动量。摆动量为正时右臂往前、左臂往后; 肘不参与(保持默认角度)。
        # arm_swing_amp 是指令 1 m/s 时上臂前后各摆多少弧度
        self.swing_amp = arm_swing_amp
        self.swing_dir = torch.zeros(len(self.arm_idx), device=self.dev)
        if arm_pose:
            dirs = arm_swing_dirs(mjm, pose)
            self.swing_dir = torch.tensor([dirs.get(names[i], 0.0) * (-1.0 if names[i].endswith("left") else 1.0) for i in self.arm_idx.tolist()],
                                          dtype=torch.float32, device=self.dev)
        self.foot_ids = [mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, b) for b in ("foot_right", "foot_left")]
        self.head_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "head")
        nz = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "neck_z")
        self.neck_z_adr = int(mjm.jnt_qposadr[nz]) if nz >= 0 else None
        self.pelvis_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.waist_adr = [int(mjm.jnt_qposadr[mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, j)]) for j in ("abdomen_x", "abdomen_z")]
        self.root_z0 = float(mjm.qpos0[2]) + 0.02

        wp.set_device("cuda:0")
        self.m = mjw.put_model(mjm)
        self.m.opt.warn_overflow = 0
        self.d = mjw.make_data(mjm, nworld=nworld, nconmax=nconmax, njmax=njmax)
        self.qpos, self.qvel = wp.to_torch(self.d.qpos), wp.to_torch(self.d.qvel)
        self.ctrl, self.warm = wp.to_torch(self.d.ctrl), wp.to_torch(self.d.qacc_warmstart)
        self.xpos, self.frc = wp.to_torch(self.d.xpos), wp.to_torch(self.d.qfrc_actuator)
        self.xquat = wp.to_torch(self.d.xquat)
        self.foot_prev = torch.zeros(nworld, 2, 2, device=self.dev)     # 上一步两只脚的水平位置, 用来算打滑的速度
        self.point_prev = torch.zeros(nworld, 2, 2, 3, device=self.dev)  # 上一步两只脚的脚跟、脚尖点位置 [世界, 脚, 点, xyz]
        self.foot_pts = [torch.tensor(p, device=self.dev).expand(nworld, 3) for p in FOOT_POINTS]
        self.overflow, self.nefc = wp.to_torch(self.d.overflow), wp.to_torch(self.d.nefc)
        # 接触点: 所有世界共用一个池子, 前 nacon 个有效。每个 geom 标一个角色: 0 右脚, 1 左脚, 2 身体其他部位, 3 地面
        con = self.d.contact
        self.c_geom, self.c_world, self.c_type = wp.to_torch(con.geom), wp.to_torch(con.worldid), wp.to_torch(con.type)
        self.nacon = wp.to_torch(self.d.nacon)
        self.c_idx = torch.arange(self.c_geom.shape[0], device=self.dev)
        role = torch.full((mjm.ngeom,), 2, dtype=torch.long)
        for g in range(mjm.ngeom):
            body = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, int(mjm.geom_bodyid[g]))
            if mjm.geom_bodyid[g] == 0:
                role[g] = 3
            elif body in ("foot_right", "foot_left"):
                role[g] = 0 if body == "foot_right" else 1
        self.geom_role = role.to(self.dev)

        mjw.step(self.m, self.d)                           # 热身, 编译 kernel
        with wp.ScopedCapture() as cap:                    # 一个策略步 = decimation 次物理步, 抓成一个 graph
            for _ in range(decimation):
                mjw.step(self.m, self.d)
        self.graph = cap.graph

        f = dict(device=self.dev, dtype=torch.float32)
        self.action = torch.zeros(nworld, self.nu, **f)
        self.prev_action = torch.zeros_like(self.action)
        self.cmd = torch.zeros(nworld, 3, **f)
        self.gait = torch.zeros(nworld, 2, **f)              # 步频 f(周期/秒), 着地比例 D
        self.phase = torch.rand(nworld, **f)
        self.contact = torch.ones(nworld, 2, **f)
        # 开局把各环境的回合进度打散, 避免同时超时、同时重置
        self.t = torch.randint(0, ep_len, (nworld,), device=self.dev)
        self.cmd_t = torch.randint(0, cmd_period, (nworld,), device=self.dev)
        fns = self._block_fns()
        self.obs_layout = [(k, fns[k]().shape[1]) for k in self.obs_names]
        self.critic_layout = self.obs_layout + [(k, fns[k]().shape[1]) for k in self.priv_names]
        self.obs_dim, self.critic_dim = sum(d for _, d in self.obs_layout), sum(d for _, d in self.critic_layout)
        self.reset_idx(torch.arange(nworld, device=self.dev), fresh=True)
        self._sample_cmd(torch.arange(nworld, device=self.dev))

    # ---------------- 重置 ----------------
    def reset_idx(self, idx, fresh=False):
        k = idx.numel()
        if k == 0:
            return
        f = dict(device=self.dev, dtype=torch.float32)
        yaw = (torch.rand(k, **f) * 2 - 1) * math.pi if self.random_yaw else torch.zeros(k, **f)
        q = torch.zeros(k, 7, **f)
        q[:, 2] = self.root_z0
        q[:, 3], q[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        qpos = torch.zeros(k, self.qpos.shape[1], **f)
        qpos[:, :7] = q
        qpos[:, self.qadr] = self.q_def + (torch.rand(k, self.nu, **f) * 2 - 1) * 0.05
        qvel = torch.zeros(k, self.qvel.shape[1], **f)
        qvel[:, :6] = (torch.rand(k, 6, **f) * 2 - 1) * 0.1
        self.qpos[idx], self.qvel[idx] = qpos, qvel
        self.ctrl[idx] = self.q_def
        self.warm[idx] = 0
        self.overflow[idx] = 0
        self.action[idx] = 0
        self.prev_action[idx] = 0
        self.contact[idx] = 1
        self.phase[idx] = torch.rand(k, **f)
        if not fresh:
            self.t[idx] = 0

    def _sample_cmd(self, idx):
        k = idx.numel()
        for j, (lo, hi) in enumerate(self.cmd_rng):
            self.cmd[idx, j] = lo + (hi - lo) * torch.rand(k, device=self.dev)
        for j, (lo, hi) in enumerate(self.gait_rng):
            self.gait[idx, j] = lo + (hi - lo) * torch.rand(k, device=self.dev)
        self.cmd_t[idx] = 0

    # ---------------- 观测 ----------------
    def _block_fns(self):
        """每个观测块怎么算(用到才算)。每块返回 [n, 维数]。"""
        q = self.qpos[:, 3:7]
        return dict(
            gravity=lambda: quat_to_gravity(q),
            lin_vel=lambda: rotate_inverse(q, self.qvel[:, :3]) * 2.0,
            ang_vel=lambda: self.qvel[:, 3:6] * 0.25,
            joint_pos=lambda: self.qpos[:, self.qadr],
            joint_vel=lambda: self.qvel[:, self.dadr] * 0.05,
            last_action=lambda: self.action,
            cmd=lambda: self.cmd,
            clock=lambda: torch.stack([torch.sin(2 * math.pi * self.phase), torch.cos(2 * math.pi * self.phase)], dim=1),
            gait=lambda: self.gait,
            height=lambda: self.qpos[:, 2:3],
            foot_contact=lambda: self.contact,
        )

    def _obs(self):
        fns = self._block_fns()
        obs = torch.cat([fns[k]() for k in self.obs_names], dim=1)
        return obs, torch.cat([obs] + [fns[k]() for k in self.priv_names], dim=1)

    def ground_contact(self):
        """返回 (身体其他部位碰地 [n], 两只脚各自着地 [n, 2])。全在 GPU 上算, 不同步回 CPU。"""
        g = self.c_geom.long()
        valid = (self.c_idx < self.nacon[0]) & ((self.c_type & 1) != 0)              # 只看参与求解的接触
        r0, r1 = self.geom_role[g[:, 0].clamp(min=0)], self.geom_role[g[:, 1].clamp(min=0)]
        other = torch.where(r0 == 3, r1, r0)                                          # 地面以外的那个 geom 的角色
        hit = valid & ((r0 == 3) | (r1 == 3)) & (other != 3)
        key = self.c_world.long().clamp(0, self.n - 1) * 3 + other.clamp(max=2)
        cnt = torch.bincount(key, weights=hit.float(), minlength=self.n * 3).view(self.n, 3)
        return cnt[:, 2] > 0, cnt[:, :2] > 0

    def observe(self):
        return self._obs()

    # ---------------- 一步 ----------------
    @torch.no_grad()
    def step(self, action):
        self.prev_action.copy_(self.action)
        self.action = action.clamp(-self.act_clip, self.act_clip)
        self.ctrl[:] = (self.q_def + self.action * self.scale).clamp(self.lo, self.hi)
        wp.capture_launch(self.graph)

        self.t += 1
        self.cmd_t += 1
        self.phase = (self.phase + self.dt * self.gait[:, 0]) % 1.0

        q = self.qpos[:, 3:7]
        g = quat_to_gravity(q)
        # 摔倒(脚以外碰地)和两只脚是否着地, 都来自真实接触点(以前用"脚踝高度 < 5.5 cm"估算, 和真实接触只有 71% 一致)
        fell, feet = self.ground_contact()
        self.contact = feet.float()

        # 在"朝向对齐"的水平坐标系里比较速度, 这样躯干前倾也不会影响速度跟踪
        w, x, y, zq = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        yaw = torch.atan2(2 * (w * zq + x * y), 1 - 2 * (y * y + zq * zq))
        c, s = torch.cos(yaw), torch.sin(yaw)
        vx_w, vy_w = self.qvel[:, 0], self.qvel[:, 1]
        vx_h, vy_h = c * vx_w + s * vy_w, -s * vx_w + c * vy_w
        wz = self.qvel[:, 5]

        rw = self.rw
        lin_err = (vx_h - self.cmd[:, 0]) ** 2 + (vy_h - self.cmd[:, 1]) ** 2
        self.m_lin_err, self.m_vx = lin_err, vx_h          # 给训练日志看的监控量
        r = rw["lin_vel"] * torch.exp(-lin_err / 0.25)
        r = r + rw["ang_vel"] * torch.exp(-((wz - self.cmd[:, 2]) ** 2) / 0.25)
        r = r + rw["alive"]
        # 步态时钟: 右脚在 [0, D) 该踩地, 左脚晚半个周期; D = 0.5 就是左右严格轮换, D = 1 两脚都该踩地(站)
        want = (torch.stack([self.phase, self.phase - 0.5], dim=1) % 1.0 < self.gait[:, 1:2]).float()
        r = r + rw["clock"] * (self.contact == want).float().mean(dim=1)
        r = r - rw["upright"] * (g[:, :2] ** 2).sum(dim=1)
        r = r - rw["action_rate"] * ((self.action - self.prev_action) ** 2).sum(dim=1)
        power = (self.frc[:, self.dadr] * self.qvel[:, self.dadr]).abs().sum(dim=1)
        r = r - rw["energy"] * power
        if rw["arm_dev"]:
            r = r - rw["arm_dev"] * ((self.qpos[:, self.qadr[self.arm_idx]] - self.q_def[self.arm_idx]) ** 2).sum(dim=1)
        if rw["arm_swing"]:
            # 前半周期右脚着地往后蹬, 右臂从最后摆到最前(左臂相反); 走得越快摆得越大, 指令为 0 时目标就是默认姿势
            swing = -torch.cos(2 * math.pi * self.phase) * self.swing_amp * self.cmd[:, 0].clamp(0, 1.5)
            arm_err = self.qpos[:, self.qadr[self.arm_idx]] - self.q_def[self.arm_idx] - swing[:, None] * self.swing_dir
            self.m_arm_err = (arm_err ** 2).sum(dim=1)
            r = r + rw["arm_swing"] * torch.exp(-self.m_arm_err / 0.15)
        # 脚的姿态: 从脚的四元数取出它的前后轴和左右轴(世界系)
        fw, fx, fy, fz = self.xquat[:, self.foot_ids].unbind(dim=2)
        if rw["foot_roll"]:                                # 左右轴翘起来的程度 = cos(俯仰) * sin(翻转), 脚放平时为 0
            r = r - rw["foot_roll"] * ((2 * (fy * fz + fw * fx)) ** 2).sum(dim=1)
        if rw["foot_yaw"]:                                 # 前后轴在"朝向对齐"坐标系里的横向分量 = sin(脚尖偏离前进方向的角度)
            ax, ay = 1 - 2 * (fy * fy + fz * fz), 2 * (fx * fy + fw * fz)
            r = r - rw["foot_yaw"] * ((-s[:, None] * ax + c[:, None] * ay) ** 2).sum(dim=1)
        if rw["waist_twist"]:
            r = r - rw["waist_twist"] * (self.qpos[:, self.waist_adr] ** 2).sum(dim=1)
        if rw["pelvis_level"]:                             # 重力在骨盆坐标系里的左右分量 = sin(侧倾角)
            r = r - rw["pelvis_level"] * quat_to_gravity(self.xquat[:, self.pelvis_id])[:, 1] ** 2
        if rw["step_width"]:                               # 左脚相对右脚的左右距离, 在朝向坐标系里(左为正)
            d = self.xpos[:, self.foot_ids[1], :2] - self.xpos[:, self.foot_ids[0], :2]
            lat = -s * d[:, 0] + c * d[:, 1]
            r = r - rw["step_width"] * ((STEP_WIDTH_MIN - lat).clamp(min=0) / STEP_WIDTH_MIN).clamp(max=1) ** 2
        if rw["heel_down"] or rw["toe_off"] or rw["swing_flat"]:
            # 按步态时钟分段: 着地前 70% 脚跟踩实, 着地最后 30% 蹬地, 摆动中段脚放平
            ph = torch.stack([self.phase, self.phase - 0.5], dim=1) % 1.0
            D = self.gait[:, 1:2]
            fw_, fx_, fy_, fz_ = self.xquat[:, self.foot_ids].unbind(dim=2)
            sin_down = -2 * (fx_ * fz_ - fw_ * fy_)        # 脚的前后轴朝下的分量 = sin(脚尖朝下的角度), 朝上为负
            swing_prog = (ph - D) / (1 - D).clamp(min=1e-3)
            r = r - rw["heel_down"] * ((ph < 0.7 * D) * sin_down.clamp(min=0) ** 2).sum(dim=1)
            late = (ph >= 0.7 * D) & (ph < D) & (self.contact > 0)    # 要真踩着地: 否则脚已经往前摆了还保持脚尖朝下拿奖励, 脚尖就拖地
            r = r + rw["toe_off"] * (late * (sin_down.clamp(min=0) / math.sin(math.radians(20))).clamp(max=1)).sum(dim=1)
            mid_swing = (ph >= D) & (swing_prog >= 0.25) & (swing_prog < 0.75)
            r = r - rw["swing_flat"] * (mid_swing * sin_down ** 2).sum(dim=1)
        if rw["foot_clear"]:                               # 摆动进度 0 -> 1 时目标高度 0 -> H -> 0; 只罚低于目标
            ph = torch.stack([self.phase, self.phase - 0.5], dim=1) % 1.0
            D = self.gait[:, 1:2]
            prog = ((ph - D) / (1 - D).clamp(min=1e-3)).clamp(0, 1) * (ph >= D)
            target = FOOT_CLEAR_H * torch.sin(math.pi * prog)
            short = (target - (self.xpos[:, self.foot_ids, 2] - 0.027)).clamp(min=0) / FOOT_CLEAR_H
            r = r - rw["foot_clear"] * (short ** 2).sum(dim=1)
        if rw["head_level"] and self.neck_z_adr is not None:   # 头的倾斜(重力在头坐标系里的水平分量) + 脖子左右转的角度
            gh = quat_to_gravity(self.xquat[:, self.head_id])
            r = r - rw["head_level"] * ((gh[:, :2] ** 2).sum(dim=1) + self.qpos[:, self.neck_z_adr] ** 2)
        if rw["foot_slip"]:                                # 着地的脚的水平速度(两步之间的位移 / 时间); 重置后的第一步位置是跳变的, 不算
            fxy = self.xpos[:, self.foot_ids, :2]
            v2 = (((fxy - self.foot_prev) / self.dt) ** 2).sum(dim=2).clamp(max=1.0)
            r = r - rw["foot_slip"] * (self.contact * v2).sum(dim=1) * (self.t > 1)
            self.foot_prev.copy_(fxy)
        if rw["point_slip"]:                               # 两只脚各两个点(脚跟、脚尖)在世界系的位置
            pts = torch.stack([torch.stack([self.xpos[:, f] + rotate(self.xquat[:, f], p) for p in self.foot_pts], dim=1)
                               for f in self.foot_ids], dim=1)
            on = (pts[..., 2] < 0.01) & (self.point_prev[..., 2] < 0.01)
            v2 = (((pts[..., :2] - self.point_prev[..., :2]) / self.dt) ** 2).sum(dim=-1).clamp(max=1.0)
            r = r - rw["point_slip"] * (on * v2).sum(dim=(1, 2)) * (self.t > 1)
            self.point_prev.copy_(pts)

        terminated = fell | ~torch.isfinite(self.qpos).all(dim=1)
        truncated = (self.t >= self.ep_len) & ~terminated
        done = terminated | truncated

        # 超时截断不是失败: 需要用终止前的状态估价值, 所以先把"重置前"的观测留下来
        final_critic = self._obs()[1] if truncated.any() else None
        info = dict(final_critic=final_critic)

        idx = done.nonzero(as_tuple=False).squeeze(1)
        self.reset_idx(idx)
        due = (self.cmd_t >= self.cmd_period) | done
        self._sample_cmd(due.nonzero(as_tuple=False).squeeze(1))
        obs, critic = self._obs()
        return obs, critic, r, terminated, truncated, info

    def diagnostics(self):
        """dropped_worlds: 接触/约束池溢出(会悄悄丢约束, 必须为 0); iter_limit_worlds: 求解器迭代到上限(只是没完全收敛, 少量无妨)。"""
        iters = 512 | 1024                                 # mujoco_warp OverflowType.ITERATIONS | LS_ITERATIONS
        return dict(dropped_worlds=int(((self.overflow & ~iters) != 0).sum()), iter_limit_worlds=int(((self.overflow & iters) != 0).sum()),
                    max_nefc=int(self.nefc.max()))
