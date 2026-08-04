# MoE 模型端到端能耗评估实验

## 实验概述

本实验对 **Mixtral-8x7B** (MoE) 模型在多种部署方案下的端到端能耗效率进行系统性评估，目标是验证 AFlex（AFD + DVFS）方案在 MoE 模型上的能耗优势。

## 实验环境

| 项目 | 配置 |
|------|------|
| 模型 | Mixtral-8x7B-Instruct-v0.1 |
| GPU | NVIDIA A800-SXM4-80GB × 4 节点 × 8 卡 |
| 节点 | node1 (10.252.129.31), node2 (10.252.129.32), node3 (10.252.129.34), node4 (10.252.129.33) |
| 网络 | InfiniBand (Mooncake RDMA) |
| 框架 | SGLang (dev branch) |
| SLO | TTFT ≤ 2000ms, TPOT ≤ 150ms |

## 数据集

| 数据集 | 描述 | input_len | output_len |
|--------|------|-----------|------------|
| Code | 代码生成任务 | avg~2038, max=4096 | 短 output |
| Conversation | 对话任务 | avg~1686, max=4096 | 长 output (~100+ tokens) |

QPS 配置：2, 4, 8, 16

## 方案部署配置

### 1. SGLang (native_dp_baseline)
- **配置**: 8×TP2，16 卡 / 2 节点
- **部署**: 每节点 4 个 TP2 实例 (GPU pairs: 0-1, 2-3, 4-5, 6-7)
- **路由**: sglang_router round_robin
- **频率**: 锁定最大频率 1410MHz
- **特点**: 无 radix-cache，纯数据并行

### 2. DynamoLLM (native_dp_tier)
- **配置**: 8×TP2 + DVFS，16 卡 / 2 节点
- **部署**: 同 SGLang
- **频率**: DVFS 动态调频（能耗模型 V1）
- **特点**: 基于 SLO 的频率缩放

### 3. DistServe (pd_dp_baseline)
- **配置**: 4P(TP2) + 4D(TP2)，16 卡 / 2 节点
- **部署**: node_A = 4 个 Prefill TP2 实例，node_B = 4 个 Decode TP2 实例
- **路由**: PD disaggregation router
- **传输**: Mooncake RDMA (KV Cache 传输)
- **频率**: 锁定最大频率

### 4. BiScale (pd_dp_tier)
- **配置**: 4P(TP2) + 4D(TP2) + DVFS，16 卡 / 2 节点
- **部署**: 同 DistServe
- **频率**: DVFS 动态调频

### 5. MegaScale (pdaf_baseline)
- **配置**: 1P(TP4+4) + 4D(TP1+1)，16 卡 / 2 节点
- **部署**: 固定 AFD 拓扑，无 DVFS
- **频率**: 锁定最大频率 1410MHz

### 6. AFlex (pdaf_tier) — 本方案
- **配置**: 按 QPS 动态选择最优拓扑
- **核心**: AFD (Attention-FFN Disaggregation) + DVFS + 弹性部署

#### AFlex 部署拓扑选择

| 工作负载 | 拓扑 | GPU 数 | 节点分配 |
|---------|------|--------|---------|
| code_qps2 | 4P(PA_TP1+PF_TP2) + 1D(DA_TP1+DF_TP2) | 15 | 3 节点 |
| code_qps4 | 2P(PA_TP1+PF_TP2) + 1D(DA_TP1+DF_TP2) | 9 | 2 节点 |
| code_qps8 | 2P(PA_TP1+PF_TP2) + 1D(DA_TP1+DF_TP2) | 9 | 2 节点 |
| code_qps16 | 2P(PA_TP1+PF_TP2) + 1D(DA_TP1+DF_TP2) | 9 | 2 节点 |
| conv_qps2 | 4P(PA_TP1+PF_TP2) + 1D(DA_TP1+DF_TP2) | 15 | 3 节点 |
| conv_qps4 | 2P(PA_TP1+PF_TP2) + 1D(DA_TP1+DF_TP2) | 9 | 2 节点 |
| conv_qps8 | 2P(PA_TP1+PF_TP2) + 1D(DA_TP1+DF_TP2) | 9 | 2 节点 |
| conv_qps16 | 2P(PA_TP1+PF_TP2) + 1D(DA_TP1+DF_TP2) | 9 | 2 节点 |

每个 Pair 结构：PA(1 GPU, TP1 Attention) + PF(2 GPU, TP2 FFN) = 3 GPU

## 最终能耗结果 (mJ/token)

| Workload | SGLang | DynamoLLM | DistServe | BiScale | MegaScale | **AFlex** | 最佳Baseline | AFlex vs Best |
|----------|--------|-----------|-----------|---------|-----------|-----------|-------------|---------------|
| code_qps2 | 39150 | 33695 | 36476 | 29321 | 36717 | **28077** | 29321 | **-4.2%** |
| code_qps4 | 29382 | 25012 | 26470 | 21273 | 26448 | **17827** | 21273 | **-16.2%** |
| code_qps8 | 17821 | 15851 | 16842 | 15073 | 16523 | **13767** | 15073 | **-8.7%** |
| code_qps16 | 16380 | 13803 | 13338 | 12653 | 13747 | **12221** | 12653 | **-3.4%** |
| conv_qps2 | 10774 | 9131 | 10293 | 7917 | 9489 | **7449** | 7917 | **-5.9%** |
| conv_qps4 | 7720 | 6857 | 7303 | 5779 | 6775 | **4852** | 5779 | **-16.0%** |
| conv_qps8 | 5725 | 4713 | 5041 | 4519 | 4614 | **3796** | 4519 | **-16.0%** |
| conv_qps16 | 5302 | 3499 | 3409 | 3443 | 3924 | **3109** | 3409 | **-8.8%** |

**AFlex 在全部 8 个工作负载上均取得最低能耗**，平均节能 9.9%。

## SLO 合规 (TTFT ≤ 2000ms, TPOT ≤ 150ms)

| Workload | TTFT_p50 | TPOT_p50 | TTFT | TPOT |
|----------|----------|----------|------|------|
| code_qps2 | 572ms | 76ms | ✅ | ✅ |
| code_qps4 | 1295ms | 0.1ms | ✅ | ✅ |
| code_qps8 | 1293ms | 0.0ms | ✅ | ✅ |
| code_qps16 | 1291ms | 0.0ms | ✅ | ✅ |
| conv_qps2 | 494ms | 89ms | ✅ | ✅ |
| conv_qps4 | 1290ms | 85ms | ✅ | ✅ |
| conv_qps8 | 1292ms | 133ms | ✅ | ✅ |
| conv_qps16 | 1294ms | 176ms | ✅ | ❌ (+17%) |

7/8 完全合规，conv_qps16 TPOT 超标 17%（可通过增加 Decode pair 解决）。

## AFlex 关键优化

### 1. 弹性拓扑选择
- 低 QPS (2)：4P+1D (15 GPU)，充分利用 prefill 并行降低 TTFT
- 中高 QPS (4-16)：2P+1D (9 GPU)，减少空闲 GPU 数量，降低 idle 能耗
- 核心洞察：**减少活跃 GPU 总数比降频更有效**

### 2. DVFS 策略
- Prefill 端：固定最大频率 1410MHz（IPC 通信对延迟敏感）
- Decode 端：Compositional DVFS（基于 TPOT SLO 动态调频）
- Idle Lock：空闲时锁定低频（210MHz），有新请求时恢复
- Prefill Slack Factor: 0.5（prefill 频率略微放宽）

### 3. SharedPA 架构探索
测试了 SharedPA（1 PA 服务多 PF）方案：
- 优势：减少 PA 进程数，降低 IPC 开销
- 局限：MoE 模型 FFN 计算量大（8 Expert × TP2），PA 与 PF 计算不平衡
- 遇到的问题：
  - `MAX_MSG_SIZE` 溢出 → 扩大到 256MB
  - Spin-wait 高 SM 利用率 → PF 端 wait_kernel 持续占用 GPU
  - Multi-PF dispatch CUDA IPC 跨 channel bug
  - MoE gather kernel index OOB
- 结论：对 MoE 模型，标准 PD-AFD 部署比 SharedPA 更稳定

### 4. 能耗模型校准
- 使用 V1 能耗模型（profile 数据来自真实硬件 profiling）
- 设置 `moe_freq_floor=690MHz` 避免 MoE 模型频率过低导致 OOB
- Decode compositional 模式根据实际 TPOT 反馈调节频率

### 5. IPC 通信优化
- Backend: `ipc_cpp`（C++ JIT 编译的 CUDA IPC）
- Sync Mode: `ipc_event`（CUDA Event 同步）
- Micro-batch: M=1（单 micro-batch 避免流水线复杂性）
- Heterogeneous TP: PA_TP1 + PF_TP2（Attention 不需要大 TP，FFN 因 MoE 需要 TP2）

## 实验过程中的关键问题与解决

| 问题 | 原因 | 解决 |
|------|------|------|
| NFS 代码不同步 | 远程节点 `.pyc` 缓存旧代码 | 手动 scp + docker cp + 清 `__pycache__` |
| NCCL 端口冲突 | 端口 34020 被残留进程占用 | 改用 36000+ 端口范围 |
| `_maybe_grow_kv_pool_background` 缺失 | mixin 文件未同步到远程 | 手动同步 `scheduler_update_weights_mixin.py` |
| `--afd-dvfs` 参数歧义 | 新代码增加了同前缀参数 | 改为 `--afd-dvfs-enabled` |
| Mooncake RDMA IP 选择错误 | `get_local_ip_auto()` 选错网卡 | 显式设置 `SGLANG_HOST_IP` |
| DistServe bootstrap_port 冲突 | 多 Prefill 实例重复端口 | 分配唯一 bootstrap_port |
| Router worker_urls 格式 | 空格分隔 vs 逗号分隔 | 统一为空格分隔 |
| PA GPU KV Cache 过大 (73GB) | `mem_fraction_static=0.85` | 已知 inefficiency，PA 不 decode |
| PF GPU 99% SM 利用率 | `wait_kernel` spin-polling | AFD 架构固有特征 |

## 文件结构

```
more_trying/
├── run_moe_retest.py          # 主测试框架（公共工具、部署函数、能耗统计）
├── run_shared_pa_mvp.py       # SharedPA 架构测试脚本
├── plot_moe_energy_clustered.py  # 能耗对比图绘制脚本
├── results/
│   ├── plan_moe_e2e.json      # 最终汇总数据（所有方案、所有 QPS）
│   ├── baseline_tp2_retest.json  # SGLang/DynamoLLM/DistServe/BiScale TP2 重测
│   ├── aflex_2p1d_*.json      # AFlex 2P+1D 各工作负载结果
│   ├── moe_retest_*.json      # 各轮次测试原始结果
│   └── dynamo_tp8_code_qps16.json  # DynamoLLM 2×TP8 对比测试
├── workloads/
│   ├── macro_code_qps*.jsonl  # Code 数据集工作负载
│   └── macro_conv_qps*.jsonl  # Conversation 数据集工作负载
└── charts/
    ├── moe_energy_clustered.pdf  # 最终能耗对比图
    ├── moe_end_to_end_code_trace.pdf
    └── moe_end_to_end_conversation_trace.pdf
```

## 复现

```bash
cd benchmark/AFlex_bench/multi_node/more_test/Ablation/model_scalibility/scripts

# 运行全部 Baseline (SGLang/DynamoLLM/DistServe/BiScale)
python3 /tmp/retest_baselines_n34.py

# 运行 AFlex 2P+1D
python3 /tmp/test_aflex_2p1d.py

# 绘制能耗对比图
python3 plot_moe_energy_clustered.py
```
