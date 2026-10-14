"""GPU 吞吐测试: N 个人形并行, 随机动作, 看每秒能算多少物理步。"""
import sys, time
import mujoco
import mujoco_warp as mjw
import warp as wp
import torch

wp.config.quiet = True
wp.init()

nworld = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
mjm = mujoco.MjModel.from_xml_path("models/humanoid.xml")
print(f"nq={mjm.nq} nv={mjm.nv} nu={mjm.nu} dt={mjm.opt.timestep}")

with wp.ScopedDevice("cuda:0"):
    m = mjw.put_model(mjm)
    m.opt.warn_overflow = 0               # 随机动作下 linesearch 常到上限, 只是刷屏, 先关掉
    d = mjw.make_data(mjm, nworld=nworld)
    ctrl = wp.to_torch(d.ctrl)            # [nworld, nu], 和 torch 共享显存, 不用拷贝
    qpos = wp.to_torch(d.qpos)

    # 抓成 CUDA graph, 之后每步只是一次 graph 启动
    mjw.step(m, d)                        # 先热身(编译 kernel)
    with wp.ScopedCapture() as cap:
        mjw.step(m, d)
    graph = cap.graph

    torch.manual_seed(0)
    n = 200
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        ctrl.uniform_(-1, 1)
        wp.capture_launch(graph)
    wp.synchronize()
    dt = time.perf_counter() - t0
    sps = nworld * n / dt
    print(f"nworld={nworld}: {n} 步用时 {dt:.2f}s -> {sps/1e6:.2f}M 环境步/秒 (每步 {dt/n*1e3:.1f} ms)")
    z = qpos[:, 2]
    print(f"随机动作 {n} 步后躯干高度: 均值 {z.mean().item():.2f} m, 最小 {z.min().item():.2f}, 有 NaN: {bool(torch.isnan(qpos).any())}")
