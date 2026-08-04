# 六方案部署配置总览

> 模型: Qwen3-32B | 硬件: 2×node (A100-80G ×8/node) | 总预算: 16 GPU
> 节点: node3 (10.252.129.34) + node4 (10.252.129.33)

---

## 方案一览

| # | 方案名 | 架构 | DVFS | GPU分配 | 论文对应 |
|---|--------|------|------|---------|----------|
| 1 | SGLang | Native DP (TP=1) | 无 (锁1410MHz) | 16×TP1 | 基线 |
| 2 | DynamoLLM | Native DP (TP=1) | 有 (Unified) | 16×TP1 | DynamoLLM |
| 3 | DistServe | PD 分离 | 无 (锁1410MHz) | 2P(TP4) + 4D(TP2) | DistServe |
| 4 | BiScale | PD 分离 | 有 (BiScale policy) | 2P(TP4) + 4D(TP2) | BiScale |
| 5 | MegaScale | AFD (TP_A+TP_F) | 无 (锁1410MHz) | Solver 决定, 用满16卡 | MegaScale |
| 6 | AFlex | AFD (TP_A+TP_F) | 有 (Per-component) | Solver 决定, 按需分配 | AFlex (ours) |

---

## 1. SGLang（基线）

**架构**: 16 个 TP=1 独立实例，round-robin 路由

```
node3: GPU[0..7] → 8×TP1 实例 (port 53200, 53210, ..., 53270)
node4: GPU[0..7] → 8×TP1 实例 (port 53280, 53290, ..., 53350)
Router: round_robin, port 42000
```

**频率**: 所有 GPU 锁定 1410MHz
**特点**: 无 PD 分离, 无 DVFS, 纯数据并行

---

## 2. DynamoLLM

**架构**: 与 SGLang 完全相同 (16×TP1, round-robin)

```
node3: GPU[0..7] → 8×TP1 实例
node4: GPU[0..7] → 8×TP1 实例
Router: round_robin, port 42000
```

**频率**: DVFS 动态调频 (Unified policy)
- `--dvfs-enabled --dvfs-energy-model-dir <models_v1>`
- `--dvfs-ttft-slo-ms 2000 --dvfs-tpot-slo-us 100000`

**特点**: 同 SGLang 架构 + 统一 DVFS 调频

---

## 3. DistServe

**架构**: PD 分离, 2×Prefill(TP4) + 4×Decode(TP2) = 16 GPU

```
node3 (Prefill):
  P0: GPU[0,1,2,3] TP=4, port 53100
  P1: GPU[4,5,6,7] TP=4, port 53110

node4 (Decode):
  D0: GPU[0,1] TP=2, port 53150
  D1: GPU[2,3] TP=2, port 53160
  D2: GPU[4,5] TP=2, port 53170
  D3: GPU[6,7] TP=2, port 53180

Router: PD-disaggregation, port 42000
  prefill → P0, P1
  decode  → D0, D1, D2, D3
Transfer: mooncake IPC
```

**频率**: 所有 GPU 锁定 1410MHz
**特点**: 纯 PD 分离 (无 AFD), 无 DVFS

---

## 4. BiScale

**架构**: 与 DistServe 完全相同 (2P(TP4) + 4D(TP2) = 16GPU)

```
(部署拓扑同 DistServe)
```

**频率**: DVFS 动态调频 (BiScale policy)
- `--dvfs-enabled --dvfs-policy biscale`
- `--dvfs-energy-model-dir <models_v1>`
- `--dvfs-ttft-slo-ms 2000 --dvfs-tpot-slo-us 100000`

**特点**: PD 分离 + BiScale 分频策略

---

## 5. MegaScale (新, Solver V2.4)

**架构**: AFD 分离 (TP_A + TP_F), Tier1 Solver 求解最大吞吐配置

**频率**: 所有 GPU 锁定 1410MHz, 不使用 DVFS
**约束**: `used == G` (必须用满全部 16 GPU)

### Conv 数据集配置

| QPS | 部署 | GPU | 说明 |
|-----|------|-----|------|
| 2 | 1P(TP4+4) + 1D(TP4+4) | 16 | 1 组 Prefill, 1 组 Decode, 各占 8 卡 |
| 4 | 2P(TP2+2) + 1D(TP4+4) | 16 | 2 组 Prefill 各 4 卡, Decode 8 卡 |
| 8 | 4P(TP1+1) + 1D(TP4+4) | 16 | 4 组 Prefill 各 2 卡, Decode 8 卡 |
| 16 | 6P(TP1+1) + 1D(TP2+2) | 16 | 6 组 Prefill 各 2 卡, Decode 4 卡 |

### Code 数据集配置

| QPS | 部署 | GPU | 说明 |
|-----|------|-----|------|
| 2 | 1P(TP4+4) + 1D(TP4+4) | 16 | 同 conv |
| 4 | 2P(TP2+2) + 1D(TP4+4) | 16 | 同 conv |
| 8 | 4P(TP1+1) + 1D(TP4+4) | 16 | 同 conv |
| 16 | 6P(TP1+1) + 1D(TP2+2) | 16 | 同 conv |

**部署方式**: AFD IPC (ipc_cpp backend), mooncake 跨节点传输, sub-router round-robin

---

## 6. AFlex (ours, Solver V2)

**架构**: AFD 分离 (TP_A + TP_F), Tier1 Solver 求解最小能耗配置

**频率**: Per-component DVFS (PA/PF/DA/DF 各自独立频率)
- `--afd-dvfs-enabled --afd-dvfs-decode-compositional --afd-dvfs-idle-lock`

### Conv 数据集配置 (il=1630, ol=42)

| QPS | 部署 | GPU | PA freq | PF freq | DA freq | DF freq |
|-----|------|-----|---------|---------|---------|---------|
| 2 | 1P(TP2+1) + 1D(TP1+1) | 5 | 1170 | 1170 | 930 | 930 |
| 4 | 2P(TP2+1) + 1D(TP1+1) | 8 | 1170 | 1170 | 930 | 930 |
| 8 | 3P(TP2+2) + 1D(TP2+1) | 15 | 930 | 1170 | 930 | 930 |
| 16 | 3P(TP1+2) + 1D(TP4+1) | 14 | 1410 | 1410 | 450 | 930 |

### Code 数据集配置 (il=2040, ol=10)

| QPS | 部署 | GPU | PA freq | PF freq | DA freq | DF freq |
|-----|------|-----|---------|---------|---------|---------|
| 2 | 1P(TP2+2) + 1D(TP1+1) | 6 | 1170 | 930 | 930 | 930 |
| 4 | 2P(TP2+2) + 1D(TP1+1) | 10 | 1170 | 930 | 930 | 930 |
| 8 | 3P(TP2+2) + 1D(TP1+1) | 14 | 1170 | 1410 | 930 | 930 |
| 16 | 7P(TP1+1) + 1D(TP1+1) | 16 | 1410 | 1410 | 930 | 930 |

**部署方式**: 同 MegaScale (AFD IPC + sub-router), 但频率按 Solver 输出逐 GPU 设置

---

## AFD 部署结构详解 (MegaScale / AFlex 共用)

```
┌──────────────────────────────────────────────────────────┐
│                    Client Requests                        │
│                         │                                │
│              ┌──────────▼──────────┐                     │
│              │  Sub-Router (RR)    │ × k_P 个            │
│              │  每个 P pair 1 个    │                     │
│              └────┬─────────┬─────┘                     │
│                   │         │                            │
│         ┌────────▼──┐  ┌──▼────────┐                   │
│         │ Prefill   │  │  Decode    │                   │
│         │ (PA + PF) │  │  (DA + DF) │                   │
│         └───────────┘  └───────────┘                    │
└──────────────────────────────────────────────────────────┘

单个 AF Pair 内部:
  ┌─────────────┐     IPC (cuda_ipc / ipc_event)     ┌─────────────┐
  │   FFN (PF)  │ ◄──────────────────────────────────► │  Attn (PA)  │
  │  TP = tp_F  │          hidden state 交换          │  TP = tp_A  │
  │  GPU: [0..] │                                     │  GPU: [n..] │
  └─────────────┘                                     └─────────────┘
  CVD = ffn_gpus + attn_gpus (contiguous allocation)
```

**Sub-Router 分配策略**: 每个 P pair 分配一个 sub-router, sub-router 到 D instance 使用 round-robin:
```
sub_router[i] → decode_eps[i % k_d]
```

---

## 公共参数

| 参数 | 值 |
|------|-----|
| Model | `/models/Qwen3-32B/` |
| mem-fraction-static | 0.85 |
| disable-cuda-graph | ✓ |
| disable-radix-cache | ✓ |
| skip-server-warmup | ✓ |
| watchdog-timeout | 600s |
| max-running-requests | 512 |
| TTFT SLO | 5000ms (AFlex/MegaScale) / 2000ms (其他4方案) |
| TPOT SLO | 300ms (AFlex/MegaScale) / 100ms (其他4方案) |

---

## 数据集 (Azure LLM Inference Trace)

| 数据集 | 输入长度 (il) | 输出长度 (ol) | 特征 |
|--------|---------------|---------------|------|
| conv (conversation) | ~1630 tokens | ~42 tokens | 长上下文, 短输出 (对话) |
| code | ~2040 tokens | ~10 tokens | 超长上下文, 极短输出 (代码补全) |

每个 QPS 对应一个 workload 文件: `macro_{dataset}_qps{N}.jsonl` (200 requests each)

---

## 测试脚本索引

### 核心基础设施

| 文件 | 作用 |
|------|------|
| `node_scalibility_macro/run_macro_benchmark.py` | **公共工具库**: cleanup_all, dexec_local/remote, wait_health, lock/unlock_freq, run_workload, test_generate, energy 读取等 |
| `more_trying/other_tier1/bench_tier1_v2.py` | **AFD 部署引擎**: plan_allocation, deploy (异构 TP), _run_workload_rr (client-side round-robin) |
| `more_trying/run_fixed_6scheme_7dataset.py` | 6 方案 × 7 数据集的原始一体化测试 (已弃用, 但 _workload_file 等工具函数仍被引用) |

### 按方案分类的测试脚本

#### AFlex + MegaScale (AFD 架构)

| 脚本 | 数据集 | QPS | 说明 |
|------|--------|-----|------|
| `more_test/bench_conv_all_qps.py` | conv | 2,4,6,8,12,16 | AFlex(DVFS) + MegaScale(旧拓扑同AFlex, 锁频) |
| `more_test/bench_code_all_qps.py` | code | 2,4,6,8,12,16 | 同上, code 数据集 |
| `more_test/run_megascale_16g_sweep.py` | conv+code | 2,4,8,16 | **新版 MegaScale**: Solver 全卡全频 (solve_max_throughput, used==16) |
| `more_test/bench_conv_v24.py` | conv | 8,12,16 | V2.4 solver 配置验证 (修正参数后) |
| `other_tier1/bench_tier1_all_qps.py` | code | 2,4,6,8,12,16 | AFlex Tier1 最优配置全 QPS sweep |
| `other_tier1/bench_megascale_all_qps.py` | code | 2,4,6,8,12,16 | MegaScale (旧: 同 AFlex 拓扑但锁频) |
| `other_tier1/bench_tier1_v2.py` | code | 16 | 单 QPS 快速迭代测试 (开发调试用) |
| `other_tier1/bench_tier1_configs.py` | - | - | 手动定义的探索性配置 |
| `other_tier1/bench_tier1_kp1.py` | - | - | k_P=1 限制下的配置验证 |

#### SGLang + DynamoLLM (Native DP)

| 脚本 | 数据集 | QPS | 说明 |
|------|--------|-----|------|
| `more_test/run_sglang_dynamo_conv_sweep.py` | conv | 2,4,8,16 | SGLang + DynamoLLM, 16×TP1, 每 QPS 重启 |
| `more_test/run_code_4schemes_sweep.py` | code | 2,4,8,16 | 4 方案合一 (SGLang/DynamoLLM/DistServe/BiScale) |

#### DistServe + BiScale (PD 分离)

| 脚本 | 数据集 | QPS | 说明 |
|------|--------|-----|------|
| `more_test/run_ds_bs_conv_sweep.py` | conv | 2,4,8,16 | DistServe + BiScale, 2P(TP4)+4D(TP2), 每 QPS 重启 |
| `more_test/run_distserve_custom.py` | conv | 16 | DistServe 自定义配置 (3D vs 4D 对比) |
| `more_test/run_code_4schemes_sweep.py` | code | 2,4,8,16 | (同上, 含 DistServe/BiScale) |
| `other_tier1/run_biscale_distserve_perqps.py` | - | - | BiScale/DistServe 逐 QPS 测试 (旧版) |

---

### 测试执行方式

```bash
# 1. AFlex + MegaScale (新版, 全卡全频)
cd more_test/
python3 run_megascale_16g_sweep.py --dataset both --qps-list 2,4,8,16

# 2. AFlex + MegaScale (旧版, 同拓扑)
python3 bench_conv_all_qps.py --mode both --qps-list 2,4,8,16
python3 bench_code_all_qps.py --mode both --qps-list 2,4,8,16

# 3. SGLang + DynamoLLM (conv)
python3 run_sglang_dynamo_conv_sweep.py

# 4. DistServe + BiScale (conv)
python3 run_ds_bs_conv_sweep.py

# 5. 四方案合一 (code)
python3 run_code_4schemes_sweep.py
```

### 结果存储

```
more_test/results/
├── megascale_16g_both_YYYYMMDD_HHMMSS.json   ← MegaScale 新版
├── conv_allqps_both_YYYYMMDD_HHMMSS.json     ← AFlex+MegaScale conv
├── code_allqps_both_YYYYMMDD_HHMMSS.json     ← AFlex+MegaScale code
├── ds_bs_conv_sweep_YYYYMMDD_HHMMSS.json     ← DistServe+BiScale conv
├── sglang_dynamo_conv_YYYYMMDD_HHMMSS.json   ← SGLang+DynamoLLM conv
└── code_4schemes_YYYYMMDD_HHMMSS.json        ← 4方案 code

more_trying/Evaluation/End-to-end/
├── data/plan_dense_e2e.json                   ← 合并后的标准数据源
└── charts/*.pdf                               ← Dashboard 图表
```

### 脚本依赖关系

```
run_macro_benchmark.py (公共工具)
    ↑
bench_tier1_v2.py (AFD 部署引擎: deploy, run_benchmark, plan_allocation)
    ↑
bench_conv_all_qps.py / bench_code_all_qps.py / run_megascale_16g_sweep.py
    (引用 deploy + run_benchmark, 定义各 QPS 的 TestConfig)

run_macro_benchmark.py
    ↑
run_code_4schemes_sweep.py / run_ds_bs_conv_sweep.py / run_sglang_dynamo_conv_sweep.py
    (直接使用 RMB 工具函数, 自行实现 deploy 逻辑)
```
