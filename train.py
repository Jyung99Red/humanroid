"""训练入口。例:
    python train.py --iters 300 --out runs/walk1
    python train.py --resume runs/walk1/ckpt.pt --iters 500 --out runs/walk1       # 接着练
    python train.py --iters 20 --envs 1024 --out runs/test                          # 快速试跑
"""
import argparse
import csv
import json
import math
import os
import time

import torch

from env import DEFAULT_OBS, DEFAULT_PRIV, OBS_BLOCKS, HumanoidEnv, REWARD
from ppo import PPO

p = argparse.ArgumentParser()
p.add_argument("--iters", type=int, default=300)
p.add_argument("--envs", type=int, default=4096)
p.add_argument("--T", type=int, default=24, help="每轮每个环境走多少步")
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--mb", type=int, default=4, help="每个 epoch 分几个小批")
p.add_argument("--lr", type=float, default=5e-4)
p.add_argument("--critic-lr", type=float, default=5e-4)
p.add_argument("--ent", type=float, default=0.005)
p.add_argument("--init-std", type=float, default=0.5)
p.add_argument("--ep-len", type=int, default=500, help="回合长度(策略步, 每步 0.02 秒)")
p.add_argument("--action-frac", type=float, default=0.3)
p.add_argument("--xml", type=str, default="models/humanoid_pd.xml", help="身体模型(make_model.py 生成); 会存进策略文件, 回放时用同一个")
p.add_argument("--arm-pose", type=str, default="", help="手臂默认姿势 json, 例如 models/arm_pose.json; 不给就是官方默认(手臂前伸)")
p.add_argument("--rw", type=str, default="", help="覆盖奖励权重, 例如 clock=0.5,alive=0.1")
p.add_argument("--swing-amp", type=float, default=0.35, help="摆臂幅度: 指令 1 m/s 时上臂前后各摆多少弧度(配合 --rw arm_swing=...)")
p.add_argument("--gait-period", type=float, default=0.8, help="步态周期(秒, 左右脚各迈一步): 越大步频越慢, 同样速度下步子越大; 给了 --freq 就不用它")
p.add_argument("--vx", default="0.4,1.2", help="指令范围 lo,hi: 前进速度 m/s; 负数开头要写成 --vx=-0.6,1.2")
p.add_argument("--vy", default="0,0", help="指令范围: 侧移速度 m/s")
p.add_argument("--wz", default="-0.3,0.3", help="指令范围: 转向速度 rad/s")
p.add_argument("--freq", default="", help="指令范围: 步频(步态周期/秒, 步数/分 = 120 x 步频); 不给就固定为 1/gait-period")
p.add_argument("--duty", default="0.5,0.5", help="指令范围: 着地比例(>0.5 走, <0.5 跑, 1 站)")
p.add_argument("--stand", type=float, default=0.0, help="抽指令时站定(速度全 0、着地比例 1)的概率, 例如 0.15")
p.add_argument("--cmd-period", default="", help="多少秒换一次指令 lo,hi(每次随机抽), 例如 1,5; 不给就固定 5 秒")
p.add_argument("--cmd-acc", type=float, default=0.0, help="指令加速度限制 m/s^2(转向 rad/s^2 同值), 例如 1.5; 0 = 新指令瞬间生效")
p.add_argument("--out", type=str, default="runs/walk1")
p.add_argument("--resume", type=str, default="")
p.add_argument("--obs-add", type=str, default="", help="给演员加观测块(逗号分隔), 例如续训时加新输入; 可用块见 env.OBS_BLOCKS")
p.add_argument("--priv-add", type=str, default="", help="只给评论家加的观测块(逗号分隔)")
p.add_argument("--save-every", type=int, default=10)
p.add_argument("--snap-every", type=int, default=50, help="每多少轮存一个快照到 snaps/(每个约 330 KB, 导出成时间线约 0.5 MB); 回放时间线按快照逐个看")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--push", type=float, default=0.0, help="随机推躯干的最大力(牛), 例如 150; 0 = 不推。每次推 0.2 秒, 方向随机, 大小在 0~它之间")
p.add_argument("--push-period", default="2,5", help="多少秒推一次 lo,hi(每次随机抽)")
p.add_argument("--kl", type=float, default=0.01, help="KL 目标: 学习率按每轮的 KL 自动调(大于 2 倍降、小于一半升)。std 塌到 0.15 左右时学习率会被压到 1e-5 附近, 调大它能学快一些, 例如 0.02")
p.add_argument("--std", type=float, default=0.0, help="续训时把探索噪声的标准差重置成这个值(例如 0.3); 0 = 沿用检查点的。"
               "std 塌到 0.15 左右时, 学习率按 KL 自动调会被压到下限, 策略几乎学不动新东西")
p.add_argument("--bound", type=float, default=0.0, help="动作边界损失的权重(例如 10): 动作均值超出目标角不被关节限位裁掉的范围时罚; 0 = 不加")
p.add_argument("--getup", action="store_true", help="训练起身策略: 从倒地姿态库(make_fallen.py)开局, 碰地不终止, 用 getup_* 奖励; 观测不带指令和步态时钟")
p.add_argument("--sym", type=float, default=0.0, help="左右镜像约束的权重(例如 4); 0 = 不加。身体和奖励必须左右对称, 用 check_mirror.py 检查")
a = p.parse_args()

os.makedirs(os.path.join(a.out, "snaps"), exist_ok=True)
rw = {k: float(v) for k, v in (kv.split("=") for kv in a.rw.split(",") if kv)}
unknown = set(rw) - set(REWARD)
assert not unknown, f"未知奖励项: {unknown}, 可用: {list(REWARD)}"
# 观测块: 续训时沿用检查点的布局(旧检查点没记就是默认那套), 再加上 --obs-add / --priv-add
ck = torch.load(a.resume, map_location="cuda:0", weights_only=False) if a.resume else None
if ck is not None and "obs_layout" in ck:
    obs = [n for n, _ in ck["obs_layout"]]
    priv = [n for n, _ in ck["critic_layout"]][len(obs):]
else:
    obs, priv = list(DEFAULT_OBS) + ["gait"], list(DEFAULT_PRIV)      # 从头训练: 默认观测 + 步态指令
    if a.getup:                                                       # 起身策略不跟指令、不按步态走
        obs = [n for n in DEFAULT_OBS if n not in ("cmd", "clock")]
split = lambda s: [x for x in s.split(",") if x]
unknown = set(split(a.obs_add) + split(a.priv_add)) - set(OBS_BLOCKS)
assert not unknown, f"未知观测块: {unknown}, 可用: {list(OBS_BLOCKS)}"
obs += [n for n in split(a.obs_add) if n not in obs]            # 评论家的特权块也可以加给演员, 环境会自动不重复
priv += [n for n in split(a.priv_add) if n not in obs + priv]
rng = lambda s: tuple(float(v) for v in s.split(","))
env_cfg = dict(xml=a.xml, obs=obs, priv=priv, cmd_vx=rng(a.vx), cmd_vy=rng(a.vy), cmd_wz=rng(a.wz), cmd_duty=rng(a.duty),
               cmd_freq=rng(a.freq) if a.freq else None, ep_len=a.ep_len, action_frac=a.action_frac, reward=dict(REWARD, **rw), arm_pose=a.arm_pose or None, arm_swing_amp=a.swing_amp,
               gait_period=a.gait_period, cmd_stand=a.stand, cmd_acc=a.cmd_acc or None,
               push_force=a.push, push_period=tuple(int(round(float(v) / 0.02)) for v in a.push_period.split(",")),
               getup=a.getup, cmd_period=tuple(int(round(float(v) / 0.02)) for v in a.cmd_period.split(",")) if a.cmd_period else 250)

env = HumanoidEnv(a.envs, seed=a.seed, **env_cfg)
ppo = PPO(env, T=a.T, epochs=a.epochs, minibatches=a.mb, lr=a.lr, critic_lr=a.critic_lr, ent=a.ent, init_std=a.init_std, sym=a.sym, bound=a.bound, kl_target=a.kl)
if ck is not None:
    ppo.load(ck)
    print(f"从 {a.resume} 接着练, 已有 {ppo.iteration} 轮")
    if a.std:
        with torch.no_grad():
            print(f"探索噪声 std {ppo.ac.std().mean().item():.3f} -> {a.std}")
            ppo.ac.log_std.fill_(math.log(a.std))

json.dump(dict(vars(a), env=env_cfg), open(os.path.join(a.out, "config.json"), "w"), ensure_ascii=False, indent=1)
print(f"{a.envs} 个环境 x {a.T} 步 = 每轮 {a.envs * a.T} 个样本; 观测 {env.obs_dim}(评论家 {env.critic_dim}), 动作 {env.nu}; 输出目录 {a.out}")
print("观测块:", " | ".join(f"{n} {d}" for n, d in env.obs_layout), " 评论家另加:", " | ".join(f"{n} {d}" for n, d in env.critic_layout[len(env.obs_layout):]))

log_path = os.path.join(a.out, "train_log.csv")
new_log = not os.path.exists(log_path)
logf = open(log_path, "a", newline="")
writer = None
t_start = time.perf_counter()


def save(path):
    torch.save(dict(ppo.state(), env=env_cfg), path)


def save_policy(path):
    """只存回放需要的东西: 策略网络 + 观测归一化 + 环境设置。"""
    torch.save(dict(actor=ppo.ac.actor.state_dict(), obs_rms=ppo.obs_rms.state(), env=env_cfg, obs_layout=env.obs_layout,
                    iteration=ppo.iteration, hidden=(512, 256, 128)), path)


for _ in range(a.iters):
    m = ppo.train_iteration()
    m["time"] = time.perf_counter() - t_start
    if writer is None:
        writer = csv.DictWriter(logf, fieldnames=list(m))
        if new_log:
            writer.writeheader()
    writer.writerow(m)
    logf.flush()
    sps = m["steps"] / (m["t_collect"] + m["t_update"])
    print(f"it {m['it']:4d} | {sps/1e3:5.0f}k步/s (采{m['t_collect']:.2f}s 训{m['t_update']:.2f}s) | 每步奖励 {m['rew']:6.3f} | "
          f"回合长 {m['ep_len']:6.1f} | 回合回报 {m['ep_ret']:7.1f} | 摔倒 {m['fall']*100:5.2f}%/步 | "
          f"vx {m['vx']:5.2f} 误差 {m['lin_err']:5.3f} | KL {m['kl']:.4f} lr {m['lr']:.1e} std {m['std']:.3f} | "
          f"vloss {m['vloss']:.3f} clip {m['clipfrac']:.2f}" + (f" | 不对称 {m['sym']:.4f}" if "sym" in m else "")
          + (f" | 越界 {m['bound']:.3f}" if "bound" in m else ""), flush=True)
    if m["it"] % a.save_every == 0:
        save(os.path.join(a.out, "ckpt.pt"))
        save_policy(os.path.join(a.out, "policy.pt"))
    if m["it"] % a.snap_every == 0:
        save_policy(os.path.join(a.out, "snaps", f"it_{m['it']:05d}.pt"))

save(os.path.join(a.out, "ckpt.pt"))
save_policy(os.path.join(a.out, "policy.pt"))
print(f"训练结束, 共 {ppo.iteration} 轮, 用时 {(time.perf_counter()-t_start)/60:.1f} 分钟")
