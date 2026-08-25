# Qwen3-30B-A3B：A800 上的 Ampere EP 适配测试

## 1. 测试目的

验证新增的 `ampere_ep` 后端在 NVIDIA A800（SM80）上的以下特性：

1. 是否真正执行 FlashInfer `MoeAlltoAll`，而不是退化到标准 EP 的 AllGather/AllReduce。
2. 相比 `moe_a2a_backend=none`，是否降低通信量和通信时间。
3. 在不同 EP 规模、并发和 workload 下的吞吐、延迟与扩展性。
4. 相比纯 TP4，EP4 是否带来性能或容量收益。

## 2. 测试环境

- 容器：`moe-energy`
- GPU：8 × NVIDIA A800-SXM4-80GB
- Compute Capability：8.0（SM80）
- 模型：`/models/Qwen3-30B-A3B/`
- 模型类型：`Qwen3MoeForCausalLM`
- 权重类型：BF16
- 层数：48
- Hidden size：2048
- MoE intermediate size：768
- Routed experts：128
- Top-k：8
- PyTorch：`2.13.0+cu129`
- CUDA：12.9

环境中存在 FlashInfer 包版本不一致：

- `flashinfer-python`：`0.6.15.post1`
- `flashinfer-cubin`：`0.6.6`

所有相关测试均使用：

```bash
FLASHINFER_DISABLE_VERSION_CHECK=1
```

绕过版本一致性检查。长期使用应统一这两个包的版本。

## 3. Ampere EP 适配说明

SGLang 原有 `moe_a2a_backend=flashinfer` 强制绑定 FlashInfer MoE runner，例如 `flashinfer_cutlass`、`flashinfer_cutedsl` 和 `flashinfer_trtllm_routed`。这些 runner 没有适用于 A800/SM80 BF16 的完整路径。

新增的 `ampere_ep` 路径：

- 通信复用 FlashInfer `MoeAlltoAll` dispatch/combine kernel。
- MoE 计算使用 SM80 可运行的 Triton runner。
- 将通信后的 global expert ID 转换为当前 rank 的 local expert ID。
- 推荐组合：

```bash
--moe-a2a-backend ampere_ep \
--moe-runner-backend triton \
--enable-dp-attention \
--tp-size N \
--dp-size N \
--ep-size N
```

当前约束：

- `ep_size = tp_size`
- `dp_size = tp_size`
- 必须开启 `--enable-dp-attention`
- 会自动关闭 shared-experts fusion
- 单卡 EP=1 无法作为合法 `ampere_ep` 基线，因为单卡时 DP attention 会被关闭，随后触发参数约束

## 4. EP 扩展性测试

### 4.1 Workload

- 随机 token ID 输入（完全离线）
- 请求数：128
- 输入长度：1024
- 输出长度：256
- 最大并发：64
- Warmup：8 requests
- `request-rate=inf`

### 4.2 结果

| EP/GPU 数 | 总吞吐 (tok/s) | 输出吞吐 (tok/s) | 平均 E2E (ms) | TTFT (ms) | TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 11,443 | 2,289 | 7,127 | 1,205 | 23.23 |
| 4 | 13,920 | 2,784 | 5,855 | 871 | 19.54 |
| 8 | 18,492 | 3,698 | 4,395 | 701 | 14.48 |

以 EP=2 为基线：

- EP=4 总吞吐提升 21.6%，E2E 降低 17.9%。
- EP=8 总吞吐提升 61.6%，E2E 降低 38.3%。
- EP=8 相比 EP=4 吞吐继续提升 32.8%。
- EP=8 的 TTFT 相比 EP=2 降低 41.8%。
- EP=8 的 TPOT 相比 EP=2 降低 37.7%。

扩展不是线性的：EP=2 → EP=8 使用 4 倍 GPU，但吞吐约为 1.62 倍，相对线性扩展效率约 40.4%。此外，`ampere_ep` 强制 EP、TP、DP 同时变化，因此该结果代表完整并行配置的扩展性，而不是只隔离 EP 变量。

原始结果：

```text
/tmp/moe_ep_bench/ep2.json
/tmp/moe_ep_bench/ep4.json
/tmp/moe_ep_bench/ep8.json
```

## 5. All-to-All 实际执行验证

### 5.1 测试配置

- 4 × A800
- `TP=DP=EP=4`
- 输入/输出：256/16
- 并发：16
- 捕获 8 个 forward step
- 关闭 CUDA Graph 和 Radix Cache，便于直接观察 kernel
- 对照组仅切换：
  - `moe_a2a_backend=none`
  - `moe_a2a_backend=ampere_ep`

### 5.2 Profiler 直接证据

`none` trace 中没有 MoE A2A kernel，主要出现：

- `ncclDevKernel_AllGather_RING_LL`
- `ncclDevKernel_AllReduce_Sum_bf16_RING_LL`
- `ncclDevKernel_Reduce_Sum_bf16_RING_LL`

`ampere_ep` 的每张 GPU trace 中明确出现：

- 384 次 `moeA2ADispatchKernel<8>`
- 384 次 `moeA2ACombineKernel<bf16, 8>`

调用数与模型结构完全一致：

```text
48 个 MoE 层 × 8 个 forward step = 384
```

因此可以确认 `ampere_ep` 每层都真正执行了 FlashInfer A2A dispatch/combine，而不是仅初始化对象或回退到 NCCL AllGather/AllReduce。

注意：kernel 名称中的 `<8>` 是 router top-k=8，不是 EP size。

原始 trace：

```text
/tmp/moe_comm_profile_none/
/tmp/moe_comm_profile_ampere/
```

## 6. NVLink 通信总量对比

### 6.1 Workload

- 4 × A800，`TP=DP=EP=4`
- 输入/输出：1024/64
- 请求数/并发：32
- 通过 `nvidia-smi nvlink -gt d` 读取测试前后的硬件计数器差值

### 6.2 结果

| 后端 | 4 卡累计 TX | TX+RX | 平均每卡 TX |
| --- | ---: | ---: | ---: |
| `none` | 60.38 GiB | 120.77 GiB | 15.10 GiB |
| `ampere_ep` | 36.30 GiB | 72.61 GiB | 9.08 GiB |

同一负载下：

- NVLink 发送量减少 24.08 GiB。
- 总链路通信量降低约 39.9%。
- `TX+RX` 会在链路两端重复观察同一份数据，判断逻辑通信量时应优先看 TX。

NVLink 计数器包含测试窗口内所有 GPU 间通信，而不只包含 MoE payload。但两组配置除 MoE A2A backend 外完全相同；结合 profiler 中 `none` 的 NCCL MoE 通信消失、`ampere_ep` 的 dispatch/combine kernel 出现，可以判断这部分下降主要来自 MoE 通信路径替换。

对应端到端结果：

| 后端 | Duration (s) | Output tok/s | TTFT (ms) | TPOT (ms) |
| --- | ---: | ---: | ---: | ---: |
| `none` | 6.265 | 326.88 | 637.33 | 88.60 |
| `ampere_ep` | 6.076 | 337.08 | 622.86 | 85.87 |

`ampere_ep` 输出吞吐提升约 3.1%，总时长降低约 3.0%。

原始结果：

```text
/tmp/nvlink_none_delta.json
/tmp/nvlink_ampere_delta.json
/tmp/moe_comm_none_link_bench.json
/tmp/moe_comm_ampere_link_bench.json
```

## 7. 通信延迟瓶颈测试

### 7.1 Workload 设计

为了让通信延迟而不是字节带宽成为瓶颈，使用极薄 decode batch：

- 4 × A800，`TP=DP=EP=4`
- 输入：1 token
- 输出：128 tokens
- 并发：4，即每个 DP rank 每步只有约 1 token
- 关闭 CUDA Graph 和 Radix Cache
- 使用模型真实 expert 路由

该模型每 token、每 MoE 层的计算量约为：

```text
3 × 2 × hidden_size(2048) × moe_intermediate_size(768) × topk(8)
= 75.5 MFLOPs
```

`ampere_ep` BF16 dispatch+combine 逻辑 payload 约为：

```text
2 × hidden_size + 8 × topk + 2 × hidden_size
= 8,256 bytes/token/layer
```

EP=4 且均匀路由时，约 75% payload 发往远端。每 token 跨 48 层约 290 KiB，按每卡约 200 GB/s NVLink 能力计算，纯字节传输理论下界只有约 1.49 μs/token。因此该场景主要暴露大量小消息的 kernel 启动和跨 rank 同步延迟，而不是链路带宽饱和。

### 7.2 端到端结果

| 后端 | Output tok/s | TPOT (ms) | E2E (s) | TTFT (ms) |
| --- | ---: | ---: | ---: | ---: |
| `none` | 49.64 | 78.90 | 10.275 | 253.86 |
| `ampere_ep` | 49.26 | 79.61 | 10.354 | 243.58 |

在每 rank 只有一个 token 的极小 batch 下，两者吞吐基本持平，`ampere_ep` 低约 0.8%。

### 7.3 四步 Profiler 结果（TP0）

`none`：

- 总 GPU kernel 时间：293.51 ms
- NCCL Reduce/ReduceScatter：154.60 ms，占 52.67%
- NCCL AllGather：62.21 ms，占 21.20%
- NCCL AllReduce：49.61 ms，占 16.90%
- 集合通信合计：266.42 ms，占 90.77%
- MoE compute：7.96 ms，占 2.71%

`ampere_ep`：

- 总 GPU kernel 时间：120.48 ms
- A2A dispatch：46.61 ms，占 38.68%
- A2A combine：45.92 ms，占 38.11%
- A2A 合计：92.52 ms，占 76.79%
- 残余 NCCL AllReduce/AllGather：约 1.20 ms，占约 1.0%
- MoE compute：7.50 ms，占 6.22%

结论：

- 成功构造了通信主导的 GPU workload。
- `ampere_ep` 将 GPU 通信 kernel 时间从约 266.4 ms 降至约 93.7 ms，下降约 64.8%。
- 总 GPU kernel 时间下降约 59.0%。
- 但端到端吞吐没有同步提升，说明极薄 batch 下关键路径还受到 CPU 调度、DP dispatch、每 token 同步及大量小 kernel 启动开销影响。

完整短 trace：

```text
/tmp/moe_latency_none_short/
/tmp/moe_latency_ampere_short/
```

曾尝试捕获 64 steps，但 trace 每 rank 约 120 MiB，写盘期间服务失去心跳；进程被终止后 gzip 尾部损坏，因此 `/tmp/moe_latency_none/` 不应作为正式证据。

## 8. 输入 32、输出 256 的并发阶梯

### 8.1 测试配置

- 4 × A800
- 输入：32 tokens
- 输出：256 tokens
- 每个点 `num_prompts = max_concurrency`
- 请求同时发送
- 随机种子：42
- 比较：
  - 纯 TP4：`TP=4, DP=1, EP=1`
  - 标准 EP4：`TP=DP=EP=4, moe_a2a_backend=none`
  - Ampere EP4：`TP=DP=EP=4, moe_a2a_backend=ampere_ep`

### 8.2 输出吞吐

| 并发 | 纯 TP4 (tok/s) | EP4 + none (tok/s) | EP4 + ampere_ep (tok/s) |
| ---: | ---: | ---: | ---: |
| 64 | 4,231 | 3,221 | 3,162 |
| 128 | 6,980 | 5,657 | 5,724 |
| 256 | 11,506 | 9,563 | 10,045 |
| 512 | 16,529 | 14,581 | 15,205 |
| 1024 | 9,870 | 15,702 | 15,982 |
| 2048 | 12,554 | 12,974 | 12,614 |
| 4096 | OOM | 11,621 | 11,217 |
| 6144 | — | OOM | 10,681 |
| 7168 | — | — | 10,759 |
| 8192 | — | — | 10,464 |
| 10240 | — | — | OOM |

### 8.3 Ampere EP4 相对标准 EP4

| 并发 | Output throughput 变化 | TPOT 变化 |
| ---: | ---: | ---: |
| 64 | -1.8% | +0.4% |
| 128 | +1.2% | -0.8% |
| 256 | +5.0% | -6.6% |
| 512 | +4.3% | -5.8% |
| 1024 | +1.8% | -2.4% |
| 2048 | -2.8% | +2.7% |
| 4096 | -3.5% | +4.9% |

`ampere_ep` 的最佳收益区间是并发 256–512，输出吞吐提升约 4%–5%。进入 2048 以上的深度饱和区后，排队、调度、logits 处理和显存压力超过 MoE 通信影响，`ampere_ep` 反而低约 3%。

### 8.4 EP4 相对纯 TP4

`ampere_ep` 相对纯 TP4：

| 并发 | 吞吐变化 |
| ---: | ---: |
| 64 | -25.3% |
| 128 | -18.0% |
| 256 | -12.7% |
| 512 | -8.0% |
| 1024 | +61.9% |
| 2048 | +0.5% |

现象：

- 低到中并发（≤512）时，纯 TP4 更快。
- 纯 TP4 的峰值在并发 512，约 16.53k output tok/s。
- Ampere EP4 的峰值在并发 1024，约 15.98k output tok/s。
- 比较各自最佳点，纯 TP4 峰值约高 3.4%。
- 并发 1024 时，纯 TP4 吞吐明显下滑，而 Ampere EP4 仍维持接近峰值，提升约 61.9%。
- EP4 的主要优势是高并发容量和高负载稳定吞吐，而不是低并发峰值。

## 9. 容量与 OOM 边界

最大完整成功阶梯：

| 配置 | 最大成功并发 | 下一失败点 |
| --- | ---: | ---: |
| 纯 TP4 | 2048 | 4096 |
| EP4 + none | 4096 | 6144 |
| EP4 + ampere_ep | 8192 | 10240 |

这些测试都没有真正填满 KV cache，而是先撞上 logits TP AllGather 的瞬时显存峰值：

- EP4 + none 在 6144 并发：
  - 每 rank 约 1450 个 running requests
  - KV token usage 约 12%
  - logits all-gather 需要额外申请约 1.74 GiB
  - CUDA OOM
- EP4 + ampere_ep 在 10240 并发：
  - 每 rank 约 2174 个 running requests
  - KV token usage 约 17%
  - logits all-gather 需要额外申请约 2.48 GiB
  - CUDA OOM
- 纯 TP4 在 4096 并发：
  - logits all-gather 需要额外申请约 1.16 GiB
  - CUDA OOM

因此“输入 32、输出 256、高并发”并不能填满当前约 551,659 tokens/rank 的 KV cache。若目标是填满 KV cache，需要显著增加单请求序列长度，或降低/分块 logits 的瞬时显存开销。

## 10. 总结

1. `ampere_ep + triton` 已在 A800/SM80、Qwen3-30B-A3B BF16 上完成端到端验证。
2. Profiler 明确捕获到每层执行 FlashInfer `moeA2ADispatchKernel` 和 `moeA2ACombineKernel`，证明是真正的 All-to-All。
3. 相比标准 EP，NVLink 总发送量降低约 39.9%。
4. 在通信延迟主导的极薄 decode workload 中，GPU 通信 kernel 时间下降约 64.8%，但端到端吞吐基本持平，说明还存在通信之外的调度和同步瓶颈。
5. 在输入 32、输出 256 的实际并发阶梯中，`ampere_ep` 在并发 256–512 相比标准 EP 快约 4%–5%。
6. 纯 TP4 在并发 ≤512 时仍然更快；EP4 的优势主要从并发 1024 开始体现。
7. 最大完整成功并发从纯 TP4 的 2048、标准 EP4 的 4096，提高到 Ampere EP4 的 8192。
8. 当前实用建议：
   - 低并发/追求单实例峰值：纯 TP4，并发约 512。
   - 高并发（约 1024）或更重视容量：EP4 + `ampere_ep`。
   - 超大并发虽然能运行，但延迟非常高，不具有实际服务价值。

并发阶梯原始 JSON（容器内历史路径 `/tmp/moe_batch_sweep/`，导出至 `benchmark/moe-energy/benchmark/data/ep_test/`）：

```text
benchmark/moe-energy/benchmark/data/ep_test/
```
