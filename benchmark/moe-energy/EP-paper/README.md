# 单节点 8 GPU MoE EP 论文部署方案

本文整理适合在单节点 8 GPU 上复现或对齐的 MoE Expert Parallelism（EP）论文部署方案，重点记录模型、并行拓扑、硬件、负载、通信及与本项目的匹配程度。

资料核验日期：2026-08-19。

## 1. 结论与推荐顺序

当前项目使用 `Qwen3-30B-A3B`，运行在单节点 8 张 A800-SXM4-80GB 上，并已覆盖 EP2/EP4/EP8 的 kernel-only profiling。因此推荐顺序如下：

1. **METRO**：主对齐工作。模型同为 Qwen3-30B-A3B，真实系统为单节点 8×A100、Attention DP8、MoE full EP8。
2. **AMoE / AEP**：同步 EP8 与 Attention–Expert 解耦、异步执行的系统对比。
3. **Semantic Parallelism / Sem-MoE**：单节点 8 GPU SGLang EP，通过模型—数据协同调度减少 A2A。
4. **MoETuner**：单节点 8×H100，但采用 4EP×2TP，适合作为 Hybrid TP–EP 基线。
5. **SGLang 官方 DPA+EP8**：不是论文创新方案，但代表当前常见、可直接落地的单节点 EP8 配置。

> 当前 `profiling/data/kernel-EP` 只测 `run_moe_core()`，且 `moe-a2a-backend=none`，不包含真实跨 rank dispatch/combine、Attention、continuous batching 或请求调度。它可与论文做 kernel 机理对照，但不能直接作为端到端吞吐或 TPOT 复现结果。

## 2. 方案总览

| 工作 | 模型 | 硬件 | Attention | Expert | 框架 | 主要研究问题 | 与当前项目匹配度 |
|---|---|---|---|---|---|---|---|
| METRO | Qwen3-30B-A3B | 1×8 A100 40GB，600 GB/s NVLink | DP8、TP1 | Full EP8 | vLLM | Decode memory-bound 下应平衡 activated experts 而非 tokens | **最高** |
| AMoE / AEP | Mixtral 8×7B | 1×8 A100 80GB / DGX A100 | 基线 DP8；AMoE 用 4 GPU | 基线 EP8；AMoE 用另 4 GPU | SGLang baseline + 原型 | 异步 EP、去 barrier、跨层动态 rebatch | 高，但模型不同 |
| Semantic Parallelism | DeepSeek-V2-Lite 等 | 1×8 GPU，每卡 96GB，互联 >400 GB/s | Attention-DP 或 Attention-TP | 8 GPU EP | SGLang 插件 | 按 token–expert affinity 减少远程 A2A | 高，但需要额外调度实现 |
| MoETuner | Mixtral 8×7B | 1×8 H100 | TP2 | EP4 | Megatron-LM 修改版 | ILP expert placement，降低负载和通信尾延迟 | 中，拓扑不是 full EP8 |
| SGLang 官方基线 | DeepSeek/Qwen MoE | 单节点 8 GPU NVLink 域 | DPA8、TP1 | EP8 | SGLang | 通用高吞吐 Serving | 工程基线 |

## 3. METRO：最推荐的 EP8 对齐工作

论文：**Efficient MoE Serving in the Memory-Bound Regime: Balance Activated Experts, Not Tokens**  
链接：https://arxiv.org/html/2512.09277

### 3.1 真实系统配置

| 参数 | 配置 |
|---|---|
| 模型 | Qwen3-30B-A3B |
| 节点 | 1 |
| GPU | 8×NVIDIA A100 40GB |
| GPU 互联 | 600 GB/s NVLink，8 GPU 位于同一互联域 |
| Serving 框架 | vLLM |
| Attention 并行 | DP8（每个 rank TP1） |
| Expert FFN 并行 | Full EP8 |
| Prefill/Decode | 同一实例混部 |
| Context length | 8K |
| Decode 最大 batch | 每 GPU 32 tokens；全局最多约 256 decode tokens |
| Prefill 最大 batch | 每 GPU 32 prompts |
| Decode CUDA Graph | 每 GPU batch `1, 2, 4, 8, 16, 32`，非 2 次幂向上 padding |
| 数据集 | InstructCoder、NuminaMath-1.5 |
| 对照 | EPLB placement/routing 与 METRO routing |
| Replication ratio | 1.0×、1.125×、1.25×、1.375×、1.5× |

Qwen3-30B-A3B 有 128 个 routed experts。对应冗余 expert slots：

| Replication ratio | 总物理 expert slots | 冗余 experts |
|---:|---:|---:|
| 1.0× | 128 | 0 |
| 1.125× | 144 | 16 |
| 1.25× | 160 | 32 |
| 1.375× | 176 | 48 |
| 1.5× | 192 | 64 |

### 3.2 关键思想

传统 EPLB 在每个 GPU 间平衡 token 数。METRO 指出，在 decode 的 memory-bound 区间，运行时间更取决于每张 GPU 实际加载了多少个 expert 权重，即 `activated experts per GPU`，而不是 token 数。METRO 因此选择 replica，使各 GPU 的 activated expert 数尽量小且均衡。

论文真实系统结果报告：相对 EPLB，METRO 在不同 workload 和 replication ratio 下可降低 decode latency，并改善 prefill/decode 混部时的总 token throughput。所有比例均是相对论文自身 baseline，不能直接与本项目绝对数值横比。

### 3.3 本项目对齐建议

端到端拓扑建议：

```text
Model: Qwen3-30B-A3B
Hardware: 1 node × 8 A800-SXM4-80GB
Attention: DP8 × TP1
Expert FFN: EP8
Context: 8192
Decode batch per GPU: 1, 2, 4, 8, 16, 32
Prefill prompts per GPU: 1, 8, 16, 32
Routing: natural + balanced + middle_rank0 + skewed_rank0
Redundant experts: 0, 16, 32, 48, 64
```

建议测量：

- TTFT、TPOT p50/p95/p99；
- input/output token throughput；
- dispatch、combine、FFN 分项 latency；
- 每 GPU token 数和 activated expert 数；
- `max/mean` rank load 和 expert-load CV；
- 总能耗、J/input-token、J/output-token、tokens/J。

当前 kernel-only 数据可以直接补齐 METRO 的 decode batch 档位 `2` 和 `16`，但必须先确认脚本中的 `batch_size` 是否等价于论文的“每 GPU decode tokens”，不能只按数值相同认定语义相同。

## 4. AMoE：同步 EP8 与异步解耦 EP

论文：**Toward Cost-Efficient Serving of Mixture-of-Experts with Asynchrony**  
链接：https://arxiv.org/html/2505.08944

### 4.1 部署配置

| 参数 | 标准 SGLang baseline | AMoE |
|---|---|---|
| 模型 | Mixtral 8×7B | Mixtral 8×7B |
| 硬件 | 单节点 8×A100 80GB / NVSwitch | 相同 8 GPU |
| Attention | DP8，位于全部 GPU | 4 GPU Attention DP |
| Expert | EP8，位于全部 GPU | 另外 4 GPU 承载 Expert |
| 执行方式 | 固定 batch、同步 A2A barrier | layer-wise µ-queue、异步 rebatch |
| 数据集 | Databricks Dolly-15K | 同 baseline |
| 路由 | Top-1 / Top-2 | Top-1 / Top-2 |

论文还用标准 SGLang EP8 在 DGX A100 40GB 上，以约 100 req/s 输入研究 Mixtral 的 expert load skew 和 GPU stall。

### 4.2 与当前项目的关系

AMoE 的价值不是提供一个普通 EP8 参数，而是给出两种 8 GPU 资源映射：

```text
同步基线：8 GPU 同时运行 Attention DP8 + Expert EP8
解耦方案：4 GPU Attention + 4 GPU Expert
```

适合增加以下对照：

1. 标准同步 full EP8；
2. 两 microbatch overlap；
3. Attention/Expert 4+4 解耦；
4. balanced 与 skewed 下的 GPU idle、能耗和尾延迟。

AMoE 需要自定义异步 runtime，不能只通过现有 SGLang flag 完整复现。

## 5. Semantic Parallelism / Sem-MoE

论文：**Semantic Parallelism: Redefining Efficient MoE Inference via Model-Data Co-Scheduling**，ICLR 2026  
OpenReview：https://openreview.net/pdf?id=MSHPrMpIHZ  
arXiv：https://arxiv.org/html/2503.04398v5

### 5.1 部署配置

| 参数 | 配置 |
|---|---|
| 节点 | 1 |
| GPU | 8 GPU，每卡 96GB HBM |
| GPU 互联 | 同构高速互联，论文描述带宽 >400 GB/s |
| CPU/内存 | 2×44-core CPU，2TB DDR5 |
| Serving 框架 | 基于 SGLang 的插件和自定义 Triton kernels |
| Attention 方案 A | Attention-DP：按 request affinity 将请求调度到 DP rank |
| Attention 方案 B | Attention-TP：在 reduce-scatter 等路径融合 token reshuffle |
| Expert | 8 GPU EP、优化后的 A2A |

### 5.2 关键思想

Sem-MoE 联合优化 expert placement 和输入调度：

1. 离线统计 token/request 与 expert 的激活亲和性；
2. 将经常共同激活的 experts 放置到相近设备；
3. Attention-DP 下，将请求发往更可能本地命中 expert 的 rank；
4. Attention-TP 下，在已有通信中融合 token 重排；
5. 减少需要经过全局 A2A 的远程 expert activation。

### 5.3 复现边界

它只在单节点 8 GPU 上验证，规模与本项目吻合；但完整复现需要 affinity profiling、expert placement 求解、请求调度和自定义通信 kernel。可以先复现普通 SGLang EP8 baseline，再把自然路由 trace 的“本地 expert 命中率”作为扩展指标。

## 6. MoETuner：单节点 4EP×2TP

论文：**MoETuner: Optimized Mixture of Expert Serving with Balanced Expert Placement and Token Routing**  
链接：https://arxiv.org/html/2502.06643

### 6.1 部署配置

| 参数 | 配置 |
|---|---|
| 模型 | Mixtral 8×7B，8 experts/layer，Top-2 |
| 单节点硬件 | 8×NVIDIA H100 |
| 并行拓扑 | 4EP × 2TP = 8 GPU |
| 多节点扩展 | 2 节点 16×H200，4EP × 4TP |
| 框架 | 修改 Megatron-LM 的 A2A 和 expert placement 模块 |
| Placement | ILP，同时建模参数容量、token load 和跨层 expert affinity |
| ILP solver | Gurobi 12.0，求解 tolerance 0.025 |

### 6.2 拓扑含义

MoETuner 不是 full EP8，而是：

```text
4 个 expert-parallel 分区
每个 expert 在 2 GPU 上做 tensor parallel
总 GPU = EP4 × TP2 = 8
```

它适合以下情况：

- 单个 expert 权重或 GEMM 太大，不适合完整放在单 GPU；
- 极小 batch 下 full EP8 的 expert GEMM 太碎；
- 希望将 TP 通信留在 NVLink 域，同时降低 EP degree 和负载偏斜。

对本项目可增加 `EP8×TP1` 与 `EP4×TP2` 的对照，观察通信、kernel 粒度和 energy-delay product。

## 7. SGLang 官方常见 EP8 部署

文档：https://docs.sglang.ai/advanced_features/expert_parallelism.html  
DPA 指南：https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/dp_dpa_smg_guide.md

推荐拓扑：

```text
World size: 8
Attention: DP8 × TP1
MoE: EP8
A2A: DeepEP
MoE GEMM: auto / DeepGEMM / Triton（按模型、精度和 GPU 选择）
```

示例：

```bash
python -m sglang.launch_server \
  --model-path /models/Qwen3-30B-A3B \
  --tp 8 \
  --dp-size 8 \
  --ep 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
  --moe-runner-backend triton \
  --context-length 8192
```

参数名称以当前 checkout 的 `python -m sglang.launch_server --help` 为准。对于 Qwen3-30B-A3B，runner/backend 的兼容性也应按当前分支实测。

如果只写：

```text
--tp 8 --ep 8
```

Attention 可能仍是 TP8，而不是论文中 METRO 使用的 DP8×TP1。要对齐 METRO，必须显式确认 DPA 已启用并检查运行日志中的实际 parallel groups。

建议分别运行：

```text
EP8-kernel-none:
  当前 kernel-only，A2A=none

EP8-full-none:
  全栈 serving，通用 collective 路径

EP8-full-deepep:
  全栈 serving，真实 DeepEP dispatch/combine
```

## 8. 建议实验矩阵

### 8.1 第一阶段：对齐 METRO 的最小矩阵

固定：

```text
model = Qwen3-30B-A3B
node = 1
GPU = 8×A800-SXM4-80GB
Attention = DP8
MoE = EP8
context = 8192
```

变化维度：

```text
phase/co-location = prefill-only, decode-only, co-located
decode batch per GPU = 1, 2, 4, 8, 16, 32
prefill prompts per GPU = 1, 8, 16, 32
routing = natural, balanced, middle_rank0, skewed_rank0
redundant experts = 0, 16, 32, 48, 64
GPU frequency = 210, 690, 930, 1410 MHz
```

为避免组合爆炸，先执行：

1. 自然路由、默认频率、冗余 expert 全档位；
2. 固定冗余 expert=0，扫强制 routing；
3. 从前两步选 2–3 个代表配置扫 GPU frequency。

### 8.2 第二阶段：并行拓扑对照

```text
P1: TP8，EP1
P2: TP2，EP4（MoETuner 风格）
P3: TP1，EP8（METRO/SGLang 风格）
```

在同一个 global token budget 和 SLO 下比较，不能用不同 per-rank batch 的结果直接横比。

### 8.3 最低报告指标

```text
TTFT p50/p95/p99
TPOT p50/p95/p99
input/output token throughput
MoE dispatch / GEMM / combine latency
每 rank tokens
每 rank activated experts
expert-load CV
max-rank / mean-rank load
GPU energy
J/input-token
J/output-token
tokens/J
EDP 或 ED²P
```

## 9. 与现有 kernel-EP 数据的关系

现有文档：`../profiling/data/kernel-EP/doc.md`。

当前配置：

```text
model = Qwen3-30B-A3B
GPU = 8×A800-SXM4-80GB
EP size = 2, 4, 8
runner = Triton
moe-a2a-backend = none
measurement = 单层 run_moe_core()
routing = balanced, middle_rank0, skewed_rank0
```

可以对齐的部分：

- Qwen3-30B-A3B expert kernel；
- EP8 rank 数；
- batch、routing skew 与 GPU frequency 的机理；
- latency/energy 分解。

尚未对齐的部分：

- Attention DP8；
- 真实 token dispatch/combine；
- DeepEP/NCCL 通信；
- 自然请求 trace；
- EPLB/replica placement；
- context=8K；
- prefill/decode 混部；
- TTFT/TPOT 与 token throughput。

因此，当前结果应表述为：

> A kernel-level characterization using the same Qwen3-30B-A3B model and EP8 degree as METRO, without end-to-end attention execution or cross-rank token dispatch/combine.

补齐全栈后才可表述为复现其 single-node full-EP deployment topology。

## 10. 引用列表

1. METRO, *Efficient MoE Serving in the Memory-Bound Regime: Balance Activated Experts, Not Tokens*. https://arxiv.org/html/2512.09277
2. AMoE, *Toward Cost-Efficient Serving of Mixture-of-Experts with Asynchrony*. https://arxiv.org/html/2505.08944
3. Semantic Parallelism, ICLR 2026. https://openreview.net/pdf?id=MSHPrMpIHZ
4. MoETuner, *Optimized Mixture of Expert Serving with Balanced Expert Placement and Token Routing*. https://arxiv.org/html/2502.06643
5. SGLang Expert Parallelism documentation. https://docs.sglang.ai/advanced_features/expert_parallelism.html
6. SGLang DP/DPA guide. https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/dp_dpa_smg_guide.md
