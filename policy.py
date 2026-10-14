"""载入策略文件(policy.pt 或 snaps/*.pt), 按它自己存的环境设置建环境。回放、测验都用这里, 保证和训练时一致。

    env, act, meta = load_policy("runs/walk4/policy.pt", 6, cmd_vx=(1.0, 1.0), random_yaw=False)
    obs, _ = env.observe(); env.step(act(obs))
"""
import torch

from env import DEFAULT_OBS, HumanoidEnv
from ppo import RunningMeanStd, mlp

# 环境设置里这些键会原样传给 HumanoidEnv; 旧策略文件没有的键用 HumanoidEnv 的默认值
ENV_KEYS = ("xml", "obs", "ep_len", "action_frac", "reward", "arm_pose", "arm_swing_amp", "gait_period", "cmd_vx", "cmd_vy", "cmd_wz",
            "cmd_freq", "cmd_duty")


def load_policy(path, nworld, **overrides):
    """返回 (环境, 确定性动作函数 obs -> action, {iteration, env: 实际用的环境设置})。
    overrides 覆盖策略自带的环境设置(例如固定的指令、ep_len)。"""
    s = torch.load(path, map_location="cuda:0", weights_only=False)
    cfg = {k: v for k, v in s["env"].items() if k in ENV_KEYS}
    cfg.setdefault("obs", list(DEFAULT_OBS))
    cfg.update(overrides)
    env = HumanoidEnv(nworld, **cfg)
    assert [tuple(b) for b in s.get("obs_layout", env.obs_layout)] == env.obs_layout, "策略的观测布局和环境不一致"
    actor = mlp(env.obs_dim, env.nu, s["hidden"], 0.01).to(env.dev)
    actor.load_state_dict(s["actor"])
    rms = RunningMeanStd(env.obs_dim, env.dev)
    rms.load(s["obs_rms"], env.dev)

    @torch.no_grad()
    def act(obs):
        return actor(rms.norm(obs))

    return env, act, dict(iteration=int(s["iteration"]), env=cfg)
