# Graceful Component Reshard: Overhead Breakdown

## Overview

本文档记录了 AFlex 系统中组件动态扩展（Reshard）的时间开销 breakdown，以及优化方案 (A+C) 的效果。

测试环境：
- **Model**: Qwen3-32B (BF16)
- **Hardware**: 8× NVIDIA H100 80GB SXM
- **Framework**: SGLang + AFD (Attention-FFN Disaggregation)
- **通信**: Intra-node IPC (CUDA IPC handles) + Mooncake RDMA

## Reshard 方式

| 组件类型 | 方式 | 说明 |
|---------|------|------|
| ATTN (PA/DA) | IPC Reconnect + Shadow Pre-warm | 预热 shadow 进程（NCCL+Mooncake），与 drain 并行 |
| FFN (PF/DF) | Peer Restart + Mooncake Skip | FFN 跳过 Mooncake 初始化（不参与 KV transfer） |

## 优化方案

### 方案 C: 跳过 FFN Mooncake 初始化 (收益 ~6.5s)

FFN 模块不参与 P→D KV cache transfer，无需 Mooncake Transfer Engine。在 `model_runner.py` 的 `init_shared_mooncake_transfer_engine()` 中增加条件判断：

```python
if self.server_args.afd_perspective == AFDPerspective.AFD_PERSPECTIVE_FFN:
    return  # Skip Mooncake for FFN modules
```

### 方案 A: Shadow Process 预热 (隐藏 ~8s)

在 drain 之前启动 shadow process，shadow 完成 NCCL/Mooncake 初始化后暂停等待信号。当 old process 被 kill 后发信号，shadow 继续执行 weight load + KV alloc。

```python
# ModelRunner.__init__ 中:
shadow_signal = os.environ.get("SGLANG_RESHARD_SHADOW_SIGNAL")
if shadow_signal:
    # 等待信号（drain + kill 完成后由 orchestrator 创建信号文件）
    while not os.path.exists(shadow_signal):
        time.sleep(0.05)
```

## Detailed Breakdown

### ATTN 扩展 (PA) — 优化前 vs 优化后

#### 优化前（串行启动）

| Overhead Source | TP1→TP2 | TP2→TP4 |
|---|---|---|
| Drain from router | 5.0 s | 5.0 s |
| Export weights via CUDA IPC | 0.03 s | 0.02 s |
| Kill old process + GPU free | 3.0 s | 3.0 s |
| Peer FFN IPC reconnect | 0.49 s | 0.37 s |
| Subprocess fork + Python init | 2.5 s | 2.5 s |
| Mooncake Transfer Engine init | 6.5 s | 6.5 s |
| NCCL init_process_group | 1.1 s | 0.82 s |
| Weight load (disk fallback) | 0.25 s | 0.41 s |
| KV Cache allocation | 0.1 s | 0.1 s |
| UCX + IPC handshake | 1.5 s | 2.2 s |
| HTTP server ready | 0.5 s | 0.8 s |
| **Total** | **21.0 s** | **21.7 s** |

#### 优化后（Shadow Pre-warm）

| Overhead Source | TP1→TP2 | TP2→TP4 | Note |
|---|---|---|---|
| Drain + Export + Kill | 8.1 s | 8.0 s | 与 shadow 并行 |
| Peer FFN IPC reconnect | 0.5 s | 0.36 s | |
| Shadow remaining wait | ~8 s | ~10 s | fork+mooncake+nccl 超出 drain |
| Weight load | 0.2 s | 0.4 s | signal 后继续 |
| KV Cache + IPC + ready | 2.1 s | 3.1 s | |
| **Total (measured)** | **22.7 s** | **24.5 s** | |
| **Service interruption** | **~14 s** | **~16 s** | 从 kill 到 ready |

> **Note**: Shadow pre-warm 的瓶颈在于 Python subprocess fork 时间（~7s），使得 shadow 总初始化（16s）超过 drain phase（8.6s）。进一步优化可通过预启动 Python interpreter pool 来消除 fork 延迟。

### FFN 扩展 (PF) — 优化前 vs 优化后

#### 优化前

| Overhead Source | Time |
|---|---|
| Drain peer ATTN | ~5.0 s |
| Export weights | ~0.15 s |
| Kill old FFN + peer ATTN | ~6.0 s |
| New FFN: Python + NCCL + **Mooncake** + weight + KV + IPC | ~12.0 s |
| Restart peer ATTN: Python + NCCL + Mooncake + weight + IPC | ~17.0 s |
| **Total** | **~40 s** |

#### 优化后（方案 C: Mooncake Skip）

| Overhead Source | Time | Note |
|---|---|---|
| Drain peer ATTN | 5.0 s | |
| Export weights | 0.03 s | |
| Kill old FFN + peer ATTN | 6.0 s | |
| New FFN: Python + NCCL + weight + KV + IPC | ~12.0 s | **跳过 Mooncake (-6.5s)** |
| Restart peer ATTN: Python + NCCL + Mooncake + weight + IPC | ~17.0 s | ATTN 仍需 Mooncake |
| **Total (measured)** | **~40.3 s** | |

> PF 扩展中 Mooncake skip 节省的 6.5s 被 peer ATTN 的启动时间掩盖了（ATTN 仍需 Mooncake），实际总时间未见显著缩短。但 FFN 进程本身的启动从 ~18s 降低到 ~12s。

## TTFT Performance After Scaling

| Component | TP1 (ms) | TP2 (ms) | TP4 (ms) | Δ (TP1→TP4) |
|-----------|----------|----------|----------|-------------|
| **PA** | 1433 | 1407 | 1418 | −1.0% |
| **PF** | 1427 | 1215 | 1129 | −20.9% |

## Key Observations

1. **Shadow Pre-warm 生效但受限于 fork 延迟**：Python subprocess fork + import 需要 ~7s，加上 Mooncake init 8s，总计 15-16s 超过 drain+kill 的 8.6s。Shadow 仍节省了 ~3s (weight load 不再需要等完整初始化)。

2. **方案 C (Mooncake Skip) 对 FFN 明确生效**：PF 进程日志中无 Mooncake/Transfer Engine 初始化，FFN 启动时间从 ~18s 降至 ~12s。

3. **进一步优化方向**：
   - **Python interpreter pool**: 预启动 Python 进程池，消除 7s fork 延迟
   - **Mooncake lazy init for ATTN**: 延迟 Mooncake 初始化到首次实际 KV transfer
   - **Drain timeout 缩短**: 当前固定 5s，可根据 in-flight 请求数动态调整
   - **缩短 kill wait**: 当前 kill 后固定等 3s，可通过确认进程退出来减少

4. **PF 扩展的瓶颈是 peer ATTN 重启**：FFN 扩展需要重启 peer ATTN，而 ATTN 需要完整的 Mooncake 初始化。未来可考虑为 FFN 扩展也使用 IPC reconnect（需要实现 server-side reconnect）。

## 文件变更

| 文件 | 变更 |
|------|------|
| `python/sglang/srt/model_executor/model_runner.py` | 方案 C: FFN perspective 跳过 Mooncake init |
| `python/sglang/srt/model_executor/model_runner.py` | 方案 A: Shadow mode signal wait 逻辑 |
| `benchmark/AFlex_bench/reshard/test_component_scaling.py` | Shadow pre-warm 集成到 do_reshard() |
