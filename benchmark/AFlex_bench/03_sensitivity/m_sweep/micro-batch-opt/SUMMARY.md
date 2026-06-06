# Micro-Batch Optimization: Interleaved Async Pipeline

## 概述

在 AF（Attention-FFN）分离架构中，通过 micro-batch 切分 + interleaved async pipeline，实现 DA（Decode Attention）和 DF（Decode FFN）的计算 overlap，在大 batch 场景下吞吐提升 58%。

## 核心结果

### 实际 PD+AF 系统（batch=256, output=200）

| 配置 | MaxBatch | 吞吐 | 成功率 | vs M=1 |
|------|----------|------|--------|--------|
| M=1 | 256 | 1537.0 tok/s | 256/256 | — |
| **M=2 async** | **256** | **2432.7 tok/s** | **256/256** | **+58.3%** |

### Synthetic Decode-Only（batch=256, output=2048）

| 配置 | TPOT | 吞吐 | vs M=1 |
|------|------|------|--------|
| M=1 | 138.9ms | 1842.4 tok/s | — |
| M=2 async | 104.6ms | 2446.6 tok/s | TPOT -25%, 吞吐 +33% |

## 原理

### M=1 的时间线（每层）

```
DA GPU: [===Attn(256)===][idle............][===Attn(256)===]
DF GPU: [idle...........][===FFN(256)====][idle............]
IPC:    ←──send──→←recv→              ←──send──→←recv→
```

每层总时间 = Attn(256) + IPC + FFN(256) + IPC

### M=2 Interleaved 的时间线（每层）

```
DA GPU: [=Attn(128)=][=Attn(128)=][idle][=Attn(128)=][=Attn(128)=]
DF GPU: [idle.......][==FFN(128)==][==FFN(128)==][idle][==FFN(128)==]
IPC:    ←s→          ←s→←r→       ←r→←s→       ←s→←r→
                     ↑ overlap!
```

DA 做 Attn(mb1) 的时间与 DF 做 FFN(mb0) overlap，节省了一个 Attn 的时间。

### 为什么大 batch 下有效

| Batch | FFN 状态 | 2×FFN(N/2) vs 1×FFN(N) | M=2 效果 |
|-------|---------|------------------------|---------|
| 100 | 过渡区 | 2×FFN(50) > FFN(100) | -8%（GEMM 效率损失 > overlap 收益） |
| 256 | compute-bound | 2×FFN(128) ≈ FFN(256) | +58%（overlap 收益，无效率损失） |

batch=256 时 FFN 完全 compute-bound，切半后 FFN(128) 仍然高效（SM 利用率高），overlap 带来纯收益。

## 实现

### 新增文件

| 文件 | 功能 |
|------|------|
| `python/sglang/srt/layers/afd_async_pipeline.py` | AsyncPipelineExecutor — interleaved 流水线调度 |
| `python/sglang/srt/layers/afd_mixin.py` | `forward_afd_A_compute` / `forward_afd_F_compute` 纯计算接口 |
| `sgl-kernel/csrc/afd_ipc/afd_pipeline_driver.h/.cpp` | C++ PipelineDriver 批量 send/recv |

### 修改文件

| 文件 | 修改 |
|------|------|
| `python/sglang/srt/layers/afd.py` | 集成 `--afd-async-pipeline` 入口 |
| `python/sglang/srt/server_args.py` | 新增 CLI 参数 `--afd-async-pipeline` |
| `sgl-kernel/csrc/afd_ipc/afd_ipc.cpp/.h` | 新增 `send_gpu_only` / `recv_gpu_only`（GPU signal/wait） |
| `sgl-kernel/csrc/afd_ipc/afd_ipc_pybind.cpp` | 暴露 `send_tensor_gpu` / `recv_tensor_gpu` / `PipelineDriver` |
| `python/sglang/srt/layers/afd_ipc_cpp/communicator.py` | 新增 `send_tensor_gpu_only` / `recv_tensor_gpu_only` |

### 启用方式

```bash
# 服务端参数
--afd-micro-batch 2 --afd-async-pipeline --afd-disagg-interleave-poll

# 或环境变量
AFD_ASYNC_PIPELINE=1
```

## Batch=100 瓶颈排查

### 根因

之前测试中 batch 始终卡在 100，经过排查发现根因是 **aiohttp 客户端默认连接池限制为 100 个并发连接**（`TCPConnector(limit=100)`）。

### 排除的假设

| 假设 | 排除原因 |
|------|---------|
| KV cache 容量不够 | `max_total_num_tokens=189463`，token usage 最高 33% |
| Mooncake transfer 太慢 | 增大线程池到 128 无效；fake transfer 也卡在 100 |
| Scheduler 每步只能处理有限请求 | `max_running_requests=2487`，`polling_interval=1` |
| Prefill 侧批次限制 | Prefill 一次可处理 1820 个请求 |
| Tokenizer_manager 串行处理 | 是次要因素，但不是 100 的硬限制 |

### 修复

```python
# 客户端修复
connector = aiohttp.TCPConnector(limit=0)  # 去掉连接池限制
async with aiohttp.ClientSession(connector=connector) as session:
    ...
```

### 生产环境配置

```bash
# 增大 Mooncake transfer 并发度（加速 KV 传输）
export SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=128
export SGLANG_DISAGGREGATION_QUEUE_SIZE=32

# 减少 KV cache 预留（允许更多请求同时 decode）
--num-reserved-decode-tokens 64
```

## DF OOM 问题

### 现象

batch=300 + output=2048 时 DF 侧 OOM：
```
RuntimeError: Out of memory. Try to lower your batch size.
Try to allocate 426 tokens. Available tokens: 154
```

### 原因

DF 加载了完整的 FFN 权重（~54GB for Qwen3-32B），GPU 内存剩余给 KV cache 的空间很少：
- DA: `max_total_num_tokens=198983`（Attn 权重小，剩余内存多）
- DF: `max_total_num_tokens=64751`（FFN 权重大，剩余内存少）

### 解决方案

1. **控制 batch × output 不超过 DF 的 token 容量**：`batch × (input + output) < 64751`
2. **降低 output 长度**：batch=256 时 output ≤ 200 可安全运行
3. **长期优化**：DF 侧不需要存储实际 KV cache 数据（只在 DA 侧），可以修改 DF 的 token pool 只分配 metadata

## M=3 测试结论

M=3（batch 切成 3 份）在所有场景下都比 M=2 差：

| 配置 | TPOT | vs M=1 | 原因 |
|------|------|--------|------|
| M=2 interleaved | 73.1ms | +7.7% | 最优 overlap/通信比 |
| M=3 interleaved | 130.6ms | +92% | 通信次数增加 50%，CPU poll 开销累积 |

M=3 的额外通信开销（每层 6 次 IPC vs M=2 的 4 次）远超 overlap 收益。

## 测试脚本

| 脚本 | 用途 |
|------|------|
| `run_async_pipeline.py` | M=2 async pipeline 基础测试 |
| `run_m2_vs_m3_async.py` | M=2 vs M=3 对比 |
| `run_high_batch.py` | 高并发测试（发现 batch=100 瓶颈） |
| `run_preload_batch.py` | Synthetic decode-only（fake transfer，验证大 batch 效果） |
| `run_real_pd_256.py` | 实际 PD+AF 系统 batch=256 测试 |

## 关键发现总结

1. **M=2 async pipeline 在 batch≥200 时有显著收益**（+58% 吞吐）
2. **batch<100 时 M=2 反而更差**（GEMM 效率损失 > overlap 收益）
3. **batch=100 的瓶颈是客户端连接池限制**，不是服务端问题
4. **DF 的 KV cache 容量是生产环境的约束**，需要控制 batch×output < 64751
5. **GPU signal/wait 路径已实现但未启用**（需要 `gpu_signal` sync mode），预计可进一步降低 CPU poll 开销
