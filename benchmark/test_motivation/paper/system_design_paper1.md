# Paper 1: AF 分离下的算子级动态调频能效优化系统设计

> **目标**: 在 PD 分离的基础上，进一步做 Attention/FFN (AF) 算子级分离，通过动态调整 A/F 实例配比和各自频率，在满足 SLO 的前提下实现系统能耗最低。

---

## 一、三篇相关工作分析

### 1.1 DynamoLLM (HPCA 2025)

- **架构**: 非分离 (collocated)
- **系统设计**: 三层分层控制器——集群管理器 (~30min, Scale in/out 实例数量)、池管理器 (~5min, 调整 TP 并行度)、实例管理器 (~5s, 调整 GPU 频率)。将请求按输入/输出长度分成 9 类，分配到不同配置池 (MultiPool)。
- **调频设计**: 全池统一频率，以 5 秒为周期调整。基于离线 Profiling 构建配置表，使用 MILP 求解最小能耗配置。频率调整粒度较粗 (per-epoch)。
- **核心特点**: 请求异构感知 + 多维配置空间联合优化 + 低开销重配置技术。
- **能耗节省**: ~52% (vs 统一配置)

### 1.2 BiScale (arXiv 2026)

- **架构**: PD 分离 (disaggregated)
- **系统设计**: 两层控制——Tier 1 粗粒度 Provisioning (每 5min, 确定 P/D 实例数量、TP 度、基线频率) + Tier 2 细粒度 DVFS (每次迭代级)。核心在于**阶段感知**: 针对 Prefill 和 Decode 的不同计算特征采用不同控制策略。
- **调频设计**:
  - Prefill: MPC (模型预测控制)，考虑未来 K=8 个批次的队列演化，运行时间约 4ms
  - Decode: 轻量 per-batch 频率选择，利用 TBT slack 选择满足约束的最低频率
- **核心特点**: 首个面向 PD 分离的能效系统，揭示阶段不对称性。
- **能耗节省**: ~39% prefill, ~48% decode (vs DistServe)

### 1.3 throttLL'eM (arXiv 2025)

- **架构**: 非分离 (collocated)
- **系统设计**: 核心是**预测 + 控制**闭环。组件包括: 生成长度预测器、Scoreboard + KV/Batch 投影、准入控制 (虚拟调度 + SLO 检验)、GBDT 性能预测模型、二分频率搜索。Autoscaling 以 ~10s 周期调整 TP 引擎规模。
- **调频设计**: 迭代级 (毫秒级) 调频。在频率可行域上**二分搜索满足 SLO 的最低频率**，全实例 GPU 统一频率。
- **核心特点**: 最细粒度的预测式控制，准入控制换取降频空间，KV 作为性能代理特征。
- **能耗节省**: ~24.7-43.8% (vs Triton)

### 1.4 三篇工作的共同点

| 维度 | 共性 |
|------|------|
| **目标** | 在满足 SLO (延迟约束) 的前提下最小化能耗 |
| **核心旋钮** | GPU DVFS (频率调整) 是核心节能手段 |
| **建模基础** | 均依赖离线 Profiling 构建性能/功耗模型 (GBDT/查找表/回归等) |
| **频率策略** | 均遵循"选择满足 SLO 的最低频率"这一基本原则 |
| **多旋钮协同** | 均将 DVFS 与资源配置 (实例数/TP度) 联合优化 |
| **SLO 安全机制** | 均有降级/回退机制——违反时升频、请求迁移或排队等 |
| **负载感知** | 均感知负载动态变化，按需调整配置 |

### 1.5 三篇工作的差异

| 维度 | DynamoLLM | BiScale | throttLL'eM |
|------|-----------|---------|-------------|
| **架构假设** | 非分离 | PD 分离 | 非分离 |
| **控制层次** | 三层 (集群/池/实例) | 两层 (placement/DVFS) | 两层 (Autoscaling/DVFS) |
| **频率调整粒度** | ~5s, per-epoch | 每次迭代, per-batch | 每次迭代, per-batch |
| **频率调整范围** | 全池统一 | 逐实例、阶段差异化 | 全实例统一 |
| **阶段感知** | 无 | 有 (P: MPC, D: per-batch) | 无 (P 阶段未优化) |
| **搜索算法** | MILP | ILP + MPC + 贪心 | 二分搜索 |
| **请求异构处理** | 9 类请求池 + 长度预测 | 依赖 trace 统计 | 生成长度预测 + Scoreboard |
| **准入控制** | 无显式 | 无显式 | 有 (虚拟调度 + SLO 检验) |
| **评估规模** | 大 (40 节点 H100) | 中 (16 GPU H100) | 小 (8 GPU A100) |

---
## 二·五、核心洞察与量化分析：Motivation 量化分析：AF 分离 vs BiScale

> **分析脚本**: `benchmark/test_motivation/analyze_slo_motivation.py`
> **数据来源**: `decode_data_v1.txt` (7530 行, 修正版) + `prefill_data_v1.txt` (906 行, 修正版)
> **模型**: Qwen3-32B, GPU: A800-80GB SXM, TP: {1,2,4,8}, Freq: {210,450,690,930,1170,1410} MHz

### 2.3 三种策略定义

| 策略 | 频率选法 | 对应工作 |
|---|---|---|
| **BiScale** | 选满足 SLO 的**最低频率**（lowest-freq-first） | BiScale Decode 实际策略 |
| **L e x**（修复后 BiScale） | 选满足 SLO 的**最低能耗统一频率**（min-energy unified） | BiScale Prefill 策略 / 假想修复版 |
| **AF 分离**（本文） | A GPU 和 F GPU 各自独立选频，最小化 `E_A(f_A)+E_F(f_F)` | 本文方案 |

> **注意**：Lex 是假想的理想统一调频上界，现实中没有系统实现。用于 ablation 分解收益来源，不作为主要对比 baseline。

---

### 2.4 BiScale 自身的问题：Decode U 型能耗陷阱

**问题**：BiScale Decode 策略是 lowest-freq-first，但能耗曲线是 U 型——低频时间长导致能耗反而升高。SLO 越松，BiScale 越容易选到 U 型曲线左侧的高能耗低频区。

**Prefill 阶段没有此问题**：BiScale Prefill 使用 MPC min-energy 策略，等价于 Lex。

**BiScale vs Lex 的差距（Decode，tp=1，mixed workload 加权）**：

| SLO 倍数 | BiScale 比 Lex 多耗能（均值） | 掉坑比例 |
|---|---|---|
| ×1.0 | 0.0% | 0% |
| ×1.2 | 4.0% | 36% |
| ×1.5 | **14.2%** | 95% |
| ×2.0 | **25.2%** | 100% |
| ×3.0 | **55.6%** | 100% |

**结论**：这是 BiScale 策略设计的缺陷，与 AF 分离无关，可以独立修复（改为 min-energy 选频）。

---

### 2.5 AF 分离 vs 原始 BiScale 的系统级收益

**Decode 阶段**（mixed workload：il 分布 {128:30%,512:30%,1024:20%,2048:10%,4096:7%}，bs 加权，decode_data_v1）：

| TP | SLO×1.0 | SLO×1.1 | SLO×1.5 | SLO×2.0 |
|---|---|---|---|---|
| tp=1 | 7.6% | 0.7% | 8.3% | 14.2% |
| tp=2 | **8.9%** | 1.7% | 12.1% | 15.0% |
| tp=4 | 7.9% | 1.8% | 13.5% | 26.3% |
| tp=8 | 5.3% | 4.1% | 19.4% | 23.6% |

**收益来源分解（tp=4 为例）**：

| SLO | 来源1：BiScale U型陷阱 | 来源2：AF差异化本身 | 合计 |
|---|---|---|---|
| ×1.0 | 0.0% | **7.9%** | 7.9% |
| ×1.1 | 0.4% | 1.4% | 1.8% |
| ×1.5 | **12.4%** | 1.1% | 13.5% |
| ×2.0 | **25.3%** | 1.0% | 26.3% |

**Prefill 阶段**（BiScale Prefill ≡ Lex，收益全部来自 AF 差异化，prefill_data_v1 修正数据）：

| TP | SLO×1.0 | SLO×1.05 | SLO×1.1 | SLO×1.5+ |
|---|---|---|---|---|
| tp=1 | 0.0% | 4.0% | 3.9% | 0.0% |
| tp=2 | 0.4% | 4.1% | 8.2% | 0.0% |
| tp=4 | 0.7% | 4.5% | 12.2% | 0.1% |
| tp=8 | 1.7% | 5.1% | 8.8% | 0.3%   |

> 与旧数据（prefill_data.txt）的主要差异：旧数据 tp=8 SLO×1.0 为 14.7%，新数据修正为 1.7%。原因：旧数据 tp=8 时 F 延迟不随频率变化（疑似测量 bug），新数据修正后 F 在所有 TP 下均为 compute-bound（ratio≈0.18~0.22）。
> tp=4 SLO×1.1 的 12.2% 仍为窗口效应：1170MHz 刚好超出 SLO，Lex 被迫选 1410MHz，AF 可让 A/F 各自降频，节省显著。SLO×1.2 后消失。

---

### 2.6 AF 分离 vs Lex（修复后 BiScale）的纯粹收益

去掉 BiScale 策略缺陷的贡献，只看 AF 差异化本身的价值：

**Decode**（mixed workload 加权，decode_data_v1）：

| TP | SLO×1.0 | SLO×1.1 | SLO×1.5 | SLO×2.0 |
|---|---|---|---|---|
| tp=1 | 7.6% | 0.6% | 0.4% | 0.4% |
| tp=2 | **8.9%** | 1.7% | 1.2% | 1.1% |
| tp=4 | 7.9% | 1.4% | 1.1% | 1.0% |
| tp=8 | 5.2% | 1.5% | 1.2% | 1.3% |

**规律**：
- 严格 SLO（×1.0）下收益最大（5~9%）：Lex 被迫选高频，A/F 各自可独立降频
- SLO 放松后收益快速衰减到 ~1%（旧数据为 3~4%），说明新数据中 A/F 最优频率差异在宽松 SLO 下更小
- 收益不随 SLO 放松而增大（与 BiScale 对比不同）

**Prefill 收益集中在 SLO×1.05~1.1 窗口**：

F 延迟频率敏感性 `F_lat(1410MHz)/F_lat(210MHz)`（bs=4）：

| TP | il=128 | il=512 | il=1024 | il=2048 | il=4096 |
|---|---|---|---|---|---|
| tp=1 | 0.167 | 0.166 | 0.166 | 0.167 | 0.168 |
| tp=4 | 0.187 | 0.173 | 0.177 | 0.178 | 0.178 |
| tp=8 | 0.222 | 0.184 | 0.184 | 0.186 | 0.191 |

**修正（prefill_data_v1）**：新数据显示 F 在所有 TP 下均为 compute-bound（ratio≈0.17~0.22），tp=8 时 F 并非 memory-bound。旧数据 tp=8 F ratio≈1.0 为测量异常。因此 Prefill 阶段 AF 差异化的收益主要来自 SLO×1.05~1.1 窗口内 A/F 各自选择不同中间频率（如 A 降到 930MHz、F 保持 1410MHz），而非 tp=8 F 免费降频。SLO×1.0 下收益很小（0~1.7%），SLO×1.5+ 后几乎为零。

---

### 2.7 配比优化（N_A:N_F）的论证现状

**关键认识**：纯配比优化（不调频）在相同 GPU 总数下吞吐量与 BiScale 相同，能耗收益为零。

**数学证明**：
```
BiScale N GPU: 吞吐 = N / (t_A + t_F)
AF分离 N_A:N_F = 1:k，N_A+N_F = N:
  最优 k = t_F/t_A，吞吐 = N_A/t_A = N/(t_A*(1+k)) = N/(t_A+t_F)  ← 相同
```

**配比优化的真实价值**：

| 价值 | 类型 | 当前数据能否量化 |
|---|---|---|
| 结合调频，F GPU 独立降频 | **能耗收益** | ✅ 已量化（即 2.5/2.9 节的数字） |
| 独立扩缩 A/F 实例应对负载变化 | 运维弹性 | ❌ 需要 goodput 数据（T2-2） |
| A 用低端 GPU，F 用高端 GPU | 成本优化 | ❌ 超出能耗论文范围 |

**联合优化问题**（需要 T2-2 吞吐量数据才能量化）：
```
给定总 GPU 数 N 和请求率 λ，选 (N_A, N_F, f_A, f_F) 最小化总能耗
约束：N_A + N_F = N
      N_A / t_A(f_A) >= λ   （A 吞吐量满足请求率）
      N_F / t_F(f_F) >= λ   （F 吞吐量满足请求率）
      t_A(f_A) + t_F(f_F) <= SLO
```

---

### 2.8 遗漏角度与后续工作

| 角度 | 说明 | 所需数据 | 优先级 |
|---|---|---|---|
| **Pareto 前沿对比图** | 在延迟-能耗 Pareto 图上加 BiScale 曲线 | 现有数据可做 | ⭐⭐⭐ 立即可做 |
| **绝对能耗数字** | mJ/token 或 Wh/1000 req，比百分比更有说服力 | 现有数据可做 | ⭐⭐⭐ 立即可做 |
| **通信开销敏感性** | AF 分离后 hidden state 传输压缩 SLO 预算 | 需跑 `bench_af_comm.py` | ⭐⭐⭐ 立即可做 |
| **配比优化联合收益** | 调频+配比联合优化 vs BiScale | 需 T2-2 goodput 数据 | ⭐⭐ 下一阶段 |
| **动态负载模拟** | trace-driven，最接近真实系统 | 需 Azure/ShareGPT trace | ⭐⭐ 下一阶段 |
| **第二个模型** | 泛化性验证（如 Llama-3-70B） | 需重新 Profiling | ⭐ 可选 |

### 2.9 AFlex 收益全景分类总结（修订版）

> **甜区定义（最终版）**：AFlex vs Lex 节省 > 5%，不加任何 A_ratio 限制。直接用收益数字判断，覆盖所有 AF 差异化调频有收益的配置（包括 F 降频、A/F 各自降到不同中间频率等场景）。
>
> **数据来源**：decode_data_v1.txt（7530行）+ prefill_data_v1.txt（906行, 修正版），分析脚本 `analyze_decode_v1.py` + `analyze_prefill_v1.py`

> **命名约定**: AFlex = AF 分离独立调频（本文方案）；Lex = min-energy 统一调频（假想最优 baseline）

#### A. Decode 阶段：AFlex vs BiScale / Lex（mixed workload 加权，decode_data_v1）

| | SLO×1.0 | SLO×1.1 | SLO×1.5 | SLO×2.0 | SLO×5.0 |
|---|---|---|---|---|---|
| tp=1 vs BiScale | 7.6% | 0.7% | 8.3% | 14.2% | **34.1%** |
| tp=1 vs Lex | 7.6% | 0.6% | 0.4% | 0.4% | 0.4% |
| tp=2 vs BiScale | **8.9%** | 1.7% | 12.1% | 15.0% | **35.5%** |
| tp=2 vs Lex | **8.9%** | 1.7% | 1.2% | 1.1% | 1.1% |
| tp=4 vs BiScale | 7.9% | 1.8% | 13.5% | 26.3% | **35.2%** |
| tp=4 vs Lex | 7.9% | 1.4% | 1.1% | 1.0% | 1.0% |
| tp=8 vs BiScale | 5.3% | 4.1% | 19.4% | 23.6% | 27.8% |
| tp=8 vs Lex | 5.2% | 1.5% | 1.2% | 1.3% | 1.3% |

**收益来源分解（tp=4）**：

| SLO | 来源1：BiScale U型陷阱 | 来源2：AFlex差异化本身 | 合计 |
|---|---|---|---|
| ×1.0 | 0.0% | **7.9%** | 7.9% |
| ×1.5 | **12.4%** | 1.1% | 13.5% |
| ×2.0 | **25.3%** | 1.0% | 26.3% |
| ×5.0 | **34.2%** | 1.0% | 35.2% |

#### B. Decode：按 (tp, bs) 细分，AFlex vs Lex，SLO×1.0（decode_data_v1）

| TP | bs=1 | bs=2 | bs=4 | bs=8 | bs=16 | bs=32 | bs=64 | bs=128 | bs=256 |
|---|---|---|---|---|---|---|---|---|---|
| tp=1 | ★15.2% | ▲5.6% | ▲6.3% | 3.8% | 4.9% | 5.0% | ▲6.8% | 0.0% | N/A |
| tp=2 | ★10.0% | ★10.7% | ★10.5% | ▲9.4% | ▲9.3% | ▲8.5% | ▲6.3% | 4.8% | ▲5.1% |
| tp=4 | ★10.8% | ▲9.9% | 5.0% | ▲6.2% | ▲6.3% | ▲7.7% | ▲7.3% | ▲7.2% | 4.6% |
| tp=8 | 2.5% | 3.9% | 3.7% | 4.6% | 4.8% | 4.9% | ▲6.1% | ▲8.9% | ▲6.8% |
| tp=8 | ▲7.6% | ▲8.2% | ▲9.0% | ▲5.7% | ▲8.9% | ▲5.8% | ▲8.5% | ▲5.5% |

★≥10%  ▲5~10%  空格<5%

**规律**：tp=2 大 bs（64~256）是最强甜区（★）；tp=4 大 bs（64）是甜区；tp=8 全 bs 均在 ▲ 以上；tp=1 仅少数 bs 进入 ▲。

#### C. Decode：按 (tp, il) 细分，AFlex vs Lex，SLO×1.0，bs=16

| TP | il=128 | il=512 | il=1024 | il=2048 | il=4096 | il=8192 |
|---|---|---|---|---|---|---|
| tp=1 | 4.2% | ▲8.7% | 0.0% | N/A | N/A | N/A |
| tp=2 | ▲8.2% | ▲6.6% | ★11.5% | ▲5.5% | ▲7.5% | N/A |
| tp=4 | ★13.2% | ★13.5% | ★14.2% | ★10.9% | 3.3% | 1.6% |
| tp=8 | ▲6.6% | ▲8.5% | ▲6.5% | 3.7% | 2.5% | N/A |

**规律**：tp=4 在短到中等 il（128~2048）是最强甜区；tp=2 在 il=1024 进入★；tp=8 在短 il 稳定在 ▲；il 对收益的影响弱于 tp。

#### D. 甜区分析（SLO×1.0，Decode，decode_data_v1）

**甜区定义（修正版）**：AFlex vs Lex 节省 > 5%，不加 A_ratio 限制

| TP | 甜区配置数 | 总配置数 | 甜区占比 | 甜区内均值 | 甜区内最大 |
|---|---|---|---|---|---|
| tp=1 | 137 | 212 | **65%** | 11.1% | 24.7% |
| tp=2 | 221 | 329 | **67%** | 12.8% | 39.4% |
| tp=4 | 133 | 364 | 37% | **18.4%** | **46.8%** |
| tp=8 | 91 | 350 | 26% | 15.8% | 42.5% |

**甜区内最优频率选择分布**（前5，全 TP 合并，SLO×1.0）：

| f_A | f_F | 含义 |
|---|---|---|
| 930MHz | 1410MHz | A降频到930，F保持1410 |
| 690MHz | 1410MHz | A降频到690，F保持1410 |
| 450MHz | 930MHz | A/F各自降频到中间频率 |
| 210MHz | 930MHz | A大幅降频，F降到930 |
| 1410MHz | 1170MHz | F降频，A保持1410 |

**规律**：新数据中甜区频率选择更多样化，不再集中于 f_F=1410MHz。tp=4/8 出现较多 f_F=930MHz 的配置（A/F 各自降到不同中间频率），说明新数据中 F 的能耗-频率曲线在中间频率有更优的能效点。

#### E. Prefill 阶段：AFlex vs BiScale(=Lex)（prefill_data_v1 修正数据）

| TP | SLO×1.0 | SLO×1.05 | SLO×1.1 | SLO×1.5+ |
|---|---|---|---|---|
| tp=1 | 0.0% | 4.0% | 3.9% | 0.0% |
| tp=2 | 0.4% | 4.1% | 8.2% | 0.0% |
| tp=4 | 0.7% | 4.5% | 12.2%* | 0.1% |
| tp=8 | 1.7% | 5.1% | 8.8% | 0.3% |

*tp=4 SLO×1.1 为窗口效应，见 2.5 节说明

**F 延迟频率敏感性**（F_lat(1410)/F_lat(210)，bs=4）：

| TP | il=128 | il=512 | il=1024 | il=2048 | il=4096 |
|---|---|---|---|---|---|
| tp=1 | 0.167 | 0.166 | 0.166 | 0.167 | 0.168 |
| tp=4 | 0.187 | 0.173 | 0.177 | 0.178 | 0.178 |
| tp=8 | 0.222 | 0.184 | 0.184 | 0.186 | 0.191 |

**修正**：新数据显示 F 在所有 TP 下均为 compute-bound（ratio≈0.17~0.22）。旧数据 tp=8 ratio≈1.0 为测量异常。Prefill 收益主要来自 SLO×1.05~1.1 窗口内 A/F 差异化选频。

#### F. 端到端（E_total = E_prefill + E_decode × ol，prefill bs=1，decode bs=16）

| TP | SLO×1.0 vs BiScale | SLO×1.0 vs Lex | SLO×2.0 vs BiScale | SLO×2.0 vs Lex |
|---|---|---|---|---|
| tp=1 | 4.9% | 4.9% | 23.8% | 3.0% |
| tp=2 | 7.1% | 7.1% | 17.8% | 3.9% |
| tp=4 | **13.4%** | **13.4%** | **31.8%** | 5.5% |
| tp=8 | 6.2% | 5.4% | 25.4% | 4.0% |

**端到端收益分解（SLO×1.0，vs Lex）**：

| TP | Prefill占总能耗 | Decode占总能耗 | Prefill贡献节省 | Decode贡献节省 | 合计 |
|---|---|---|---|---|---|
| tp=1 | 1% | 99% | 0.00% | 4.90% | 4.90% |
| tp=2 | 1% | 99% | 0.00% | 7.10% | 7.10% |
| tp=4 | 1% | 99% | 0.01% | 13.40% | 13.41% |
| tp=8 | 1% | 99% | 0.20% | 5.20% | 5.40% |

**关键发现**：Prefill 占总能耗 <2%，端到端收益几乎完全由 Decode 决定。Prefill 阶段在 SLO×1.0 下收益很小（0~1.7%），端到端贡献可忽略。

---

#### 综合结论（decode_data_v1 + prefill_data_v1 更新）

| 维度 | 结论 |
|---|---|
| **vs BiScale** | 严格SLO 5~9%，宽松SLO 14~35%（含U型陷阱修复） |
| **vs Lex（纯AF差异化）** | 严格SLO 5~9%，宽松SLO后饱和在 **~1%** |
| **端到端** | Decode主导（99%），端到端收益≈Decode收益 |
| **甜区条件** | AFlex vs Lex 节省 > 5%（不加 A_ratio 限制） |
| **甜区占比** | tp=1: 65%，tp=2: 67%，tp=4: 37%，tp=8: 26% |
| **甜区收益** | 均值 11~18%，最大 46.8% |
| **tp=1/2** | 甜区占比最高（65~67%），收益来自 A/F 差异化选频 |
| **tp=4/8** | 甜区占比较低（26~37%），但甜区内均值更高（15~18%） |
| **频率选择规律** | 甜区内频率选择多样化，tp=4/8 出现较多 f_F=930MHz 的中间频率配置 |
| **SLO敏感性** | vs Lex在SLO×1.1后快速饱和（~1%）；vs BiScale随SLO放松持续增大 |

#### AFlex 收益的两层递进结构（论文 Motivation 逻辑）

> **可视化脚本**：`plot_fig1_pareto.py`（Pareto 前沿对比）、`plot_fig2_slo_curve.py`（收益 vs SLO 全景）、`plot_fig2b_Lex_only.py`（vs Lex 放大图 + error bar）、`plot_fig3_heatmap.py`（甜区热力图）
> **输出目录**：`benchmark/test_motivation/figures/`

**核心发现：AFlex 的收益来自两个独立来源，在不同 SLO 条件下主导地位不同。**

**第一层（核心贡献）：AF 分离差异化调频——严格 SLO 下的独特价值**

- AFlex vs Lex（统一调频理论上界）在 SLO×1.0 时收益最大：tp=4 达 9.2%，tp=2 达 8.3%，tp=8 达 7.8%
- 收益在 SLO×1.0 → ×1.05 这一小段下降最陡（如 tp=4 从 9.2% 降到 4.4%），×1.1 后进入饱和带（3~5%）
- P25~P75 分布：SLO×1.0 时 band 很宽（tp=8 的 P75 达 12.2%，max 达 38.8%），说明甜区配置收益远高于均值；SLO 放松后 band 收窄，所有配置趋同
- 物理原因：严格 SLO 下 Lex 被迫选高频统一调频，而 A（memory-bound）降频几乎不掉速度，AFlex 的"A 降频、F 保持高频"差异化空间最大；SLO 宽松后 Lex 自己也能选低频，差异化空间被压缩
- 这是统一调频的理论上界都做不到的，是 AF 分离的独有价值

**第二层（附带收益）：频率选择策略修复 BiScale U 型陷阱**

- AFlex vs BiScale 的收益随 SLO 放松持续增大（SLO×5.0 时达 31~42%），但增量全部来自 BiScale 的 lowest-freq-first 策略缺陷
- 这部分收益 Lex（min-energy 统一调频）也能拿到，不是 AF 分离的功劳
- 但说明 AFlex 在所有 SLO 下都不比 BiScale 差，宽松 SLO 下的大数字是锦上添花

**甜区在不同 SLO 下的变化**

| 维度 | SLO×1.0（严格） | SLO≥×1.5（宽松） |
|---|---|---|
| 甜区占比 | tp=2: 75%, tp=4: 56%, tp=8: 60% | 大幅缩小（大部分配置降到 3~5%，低于 5% 门槛） |
| 甜区内均值 | 8~14% | 降到 3~5% |
| 频率模式 | A 大幅降频，F 保持最高频（63% 配置 f_F=1410MHz） | A/F 都降频，差距缩小 |
| 物理原因 | Lex 被迫高频，AFlex 差异化空间大 | Lex 自己也能低频，差异化空间被压缩 |

**AFlex vs Lex 严格 SLO 放大数据（fig2b）**

| TP | SLO×1.0 | SLO×1.02 | SLO×1.05 | SLO×1.1 | SLO×1.5 |
|---|---|---|---|---|---|
| tp=1 | 5.0% (P75=7.4%) | 3.5% | 3.6% | 3.2% | 2.3% |
| tp=2 | 8.3% (P75=11.6%) | 7.0% | 5.2% | 3.9% | 2.8% |
| tp=4 | **9.2%** (P75=12.4%) | 7.1% | 4.4% | 3.3% | 3.2% |
| tp=8 | 7.8% (P75=12.2%) | 5.6% | 4.8% | 4.7% | 4.2% |

**论文定位**：AFlex 的核心价值在严格 SLO（×1.0~×1.1）+ 甜区（tp≥2），这恰好是生产环境最常见的场景。严格 SLO 下甜区内均值 8~14%、最高 39% 的能耗节省，是统一调频的理论上界都无法达到的。

---

### 2.10 能耗收益多维度变化规律

> **数据来源**：decode_data_v1.txt（7530行），分析脚本 `benchmark/test_motivation/analyze_decode_v1.py`
> **对比基准**：AFlex vs Lex（min-energy 统一调频），SLO×1.0 除非特别说明

#### 维度1：随 SLO 放松的变化——"快速衰减后饱和"（decode_data_v1）

| TP | SLO×1.0 | SLO×1.1 | SLO×1.2 | SLO×1.5 | SLO×2.0 | SLO×3.0 | SLO×5.0 |
|---|---|---|---|---|---|---|---|
| tp=1 | 7.6% | 0.6% | 0.4% | 0.4% | 0.4% | 0.4% | 0.4% |
| tp=2 | **8.9%** | 1.7% | 1.3% | 1.2% | 1.1% | 1.1% | 1.1% |
| tp=4 | 7.9% | 1.4% | 1.2% | 1.1% | 1.0% | 1.0% | 1.0% |
| tp=8 | 5.2% | 1.5% | 1.4% | 1.2% | 1.3% | 1.3% | 1.3% |

**规律**：SLO 从 ×1.0 放松到 ×1.1 时收益急剧下降（从 5~9% 降到 ~1.5%），之后饱和在 ~1%。衰减比旧数据更陡峭。

**物理原因**：SLO 严格时 Lex 被迫选高频（如 1410MHz），AF 可把 A 降到 930MHz 而 F 保持 1410MHz，差异化空间大。SLO 一旦放松，Lex 自己也能降到中间频率，AF 的额外自由度带来的边际收益就小了。

---

#### 维度2：随 TP 变化——"tp=2 甜区最广，tp=4 均值最高"

| TP | 均值 | 中位数 | 最大值 | >5% 占比 | 配置总数 |
|---|---|---|---|---|---|
| tp=1 | 4.6% | 4.8% | 12.4% | 49% | 59 |
| tp=2 | 7.9% | 7.8% | 37.5% | **75%** | 114 |
| tp=4 | **8.4%** | 8.6% | 33.6% | 56% | 133 |
| tp=8 | 7.7% | 7.4% | 38.8% | 60% | 119 |

**A_ratio（A_lat@1410 / A_lat@210）随 TP 的变化**（越接近 1.0 = A 越 memory-bound，ol=64，decode_data_v1）：

| TP | bs=1 | bs=4 | bs=16 | bs=64 | bs=256 |
|---|---|---|---|---|---|
| tp=1 | 0.530 | 0.529 | 0.521 | 0.379 | N/A |
| tp=2 | 0.819 | 0.854 | 0.768 | 0.680 | 0.269 |
| tp=4 | 1.005 | 1.006 | 1.014 | 0.947 | 0.430 |
| tp=8 | 1.004 | 1.020 | 1.003 | 1.011 | 0.680 |

**物理原因**：TP 越大，A 的权重被切分到更多 GPU，每块 GPU 的计算量越小，A 越 memory-bound，降频几乎不增加 A 延迟。tp=8 时 A 完全 memory-bound（ratio≈1.0），但 F 也开始变 memory-bound（权重同样被切分），F 降频空间也大，AF 差异化的"相对优势"反而不如 tp=2 突出，所以 tp=8 均值最低（5.2%）。

---

#### 维度3：随 batch_size 变化（decode_data_v1）

| TP | bs=1 | bs=2 | bs=4 | bs=8 | bs=16 | bs=32 | bs=64 | bs=128 | bs=256 |
|---|---|---|---|---|---|---|---|---|---|
| tp=1 | **15.2%** | 5.6% | 6.3% | 3.8% | 4.9% | 5.0% | 6.8% | 0.0% | N/A |
| tp=2 | 10.0% | 10.7% | 10.5% | 9.4% | 9.3% | 8.5% | 6.3% | 4.8% | 5.1% |
| tp=4 | 10.8% | 9.9% | 5.0% | 6.2% | 6.3% | 7.7% | 7.3% | 7.2% | 4.6% |
| tp=8 | 2.5% | 3.9% | 3.7% | 4.6% | 4.8% | 4.9% | 6.1% | **8.9%** | 6.8% |

**规律**：
- tp=1：bs=1 时收益最大（15.2%），大 bs 时波动较大
- tp=2：小 bs（1~4）收益最高（~10%），随 bs 增大缓慢下降
- tp=4：bs=1 时 10.8%，中间 bs 稳定在 6~8%
- tp=8：随 bs 增大收益递增，bs=128 达到 8.9%

---

#### 维度4：随 input_len 变化——"中等 il 收益最高，长 il 急剧下降"

（bs=16，SLO×1.0）

| TP | il=128 | il=512 | il=1024 | il=2048 | il=4096 | il=8192 |
|---|---|---|---|---|---|---|
| tp=1 | 4.2% | 8.7% | 0.0% | N/A | N/A | N/A |
| tp=2 | 8.2% | 6.6% | **11.5%** | 5.5% | 7.5% | N/A |
| tp=4 | 13.2% | 13.5% | **14.2%** | 10.9% | 3.3% | 1.6% |
| tp=8 | 6.6% | **8.5%** | 6.5% | 3.7% | 2.5% | N/A |

**规律**：短到中等 il（128~1024）收益最高，长 il（4096+）收益急剧下降。

**物理原因**：il 越长，KV cache 越大，A 的访存量越大，A 越 memory-bound——这本来应该让收益更大。但长 il 时 F 的 GEMM 规模不变而 A 的 GEMM 规模随 il 增大，F 反而成为瓶颈，Lex 被迫给 F 选高频，AF 的差异化空间被 F 的约束压缩。

---

#### 维度5：去掉 A_ratio 限制前后的甜区变化

| TP | 旧甜区占比（A_ratio>0.7 且 saving>5%） | 新甜区占比（saving>5%） | 变化 |
|---|---|---|---|
| tp=1 | 0% | **49%** | +49% |
| tp=2 | 21% | **75%** | +54% |
| tp=4 | 41% | 56% | +15% |
| tp=8 | 59% | 60% | +1% |

**结论**：旧定义对 tp=1/2 的低估最严重。tp=1 的 A 虽然是 compute-bound（A_ratio≈0.45），但 F 是强 compute-bound，F 降频空间很大，旧定义完全忽略了这部分收益。tp=8 的 A 本来就完全 memory-bound，两个定义几乎等价（+1%）。

**核心结论**：AF 差异化调频的收益来源有两类——
1. **A 降频**（A memory-bound 时）：tp 越大越显著，tp=8 最强
2. **F 降频**（F memory-bound 时）：tp=1 时 F 是 compute-bound，降频收益大；tp=8 时 F 也变 memory-bound，降频收益反而小

两类来源在不同 tp 下此消彼长，共同构成了 AF 差异化的总收益。

---

#### 维度6：Mixed Workload 权重敏感性分析（decode_data_v1）

> **问题**：论文默认的 il 权重分布是否会显著影响结论？

**SLO×1.0 下不同 workload 分布的加权结果**：

| Workload | il 权重分布 | tp=1 | tp=2 | tp=4 | tp=8 | 均值 |
|---|---|---|---|---|---|---|
| 论文默认 | 128:30%/512:30%/1024:20%/2048:10%/4096:7% | 7.6% | 8.9% | 7.9% | 5.2% | **7.4%** |
| 短请求为主 | 128:60%/512:25%/1024:10%/2048:5% | 8.0% | 8.7% | 8.5% | 5.5% | **7.7%** |
| 中等请求为主 | 128:10%/512:20%/1024:40%/2048:20%/4096:10% | 6.6% | 8.8% | 6.8% | 4.6% | **6.7%** |
| 长请求为主 | 128:5%/512:5%/1024:10%/2048:20%/4096:60% | 6.8% | 9.1% | 7.1% | 5.6% | **7.1%** |
| 均匀分布 | 各 20% | 7.1% | 8.9% | 7.5% | 5.1% | **7.2%** |

**结论**：

| 维度 | 结论 |
|---|---|
| **SLO×1.0 的波动范围** | 6.7%~7.7%（极端 workload 间差距约 1.0%） |
| **最有利 workload** | 短请求为主（saving 最高 7.7%） |
| **最不利 workload** | 中等请求为主（saving 最低 6.7%） |
| **结论稳健性** | **高**——不同 workload 下 saving 均在 6.7~7.7%（SLO×1.0），结论不随权重变化而翻转 |

**物理解释**：各 il 段的 saving 本身差距不大（4~11%），因为 AF 差异化的核心驱动是 TP 决定的 A memory-bound 程度，而非 il。il 只是通过影响 KV cache 大小间接影响 A 的访存压力，但在 bs=16 的典型 decode 场景下，bs 对 A memory-bound 的影响远大于 il。

---

## 三、系统设计

### 3.1 问题定义

**前提假设**: PD 分离已完成 (Prefill 池和 Decode 池独立部署)，在此基础上进一步做 AF 分离。

**四个算子池**:

| 池 | 计算特征 | 频率敏感性 |
|---|---------|-----------|
| **PA** (Prefill-Attn) | 偏 memory-bound | 中等 |
| **PF** (Prefill-FFN) | compute-bound | **高** |
| **DA** (Decode-Attn) | 强 memory-bound | **低** |
| **DF** (Decode-FFN) | 偏 memory-bound | 中等 |

**优化变量**:
1. A/F 实例配比: n_A 和 n_F 的 GPU 数量分配 (Prefill 内和 Decode 内各自独立)
2. 算子级频率: f_A 和 f_F 独立设置 (每个 iteration 可调)

**目标**: 满足 SLO (TTFT_P99, TPOT_P99) 的前提下最小化总能耗

**系统参数**: 频率切换开销 P50 ~4.5ms, avg ~6ms, P99 ~13ms (A800-80GB SXM 实测, SetGpuLockedClocks)

### 3.2 整体架构: 两层控制

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     AF-Disaggregated Serving Cluster                    │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Tier 1: Joint Resource Provisioner (每 T₁ = 几分钟)               │  │
│  │                                                                   │  │
│  │  输入: 近期负载统计 (到达率 λ, 输入/输出长度分布, 活跃 decode 数)   │  │
│  │  联合决策 (一个 ILP 同时输出):                                     │  │
│  │    • P/D 资源划分: n_P = n_PA + n_PF,  n_D = n_DA + n_DF          │  │
│  │    • A/F 实例配比: n_PA vs n_PF,  n_DA vs n_DF                    │  │
│  │    • TP 并行度:    tp_PA, tp_PF, tp_DA, tp_DF                     │  │
│  │    • 基线频率:     f̄_PA, f̄_PF, f̄_DA, f̄_DF                        │  │
│  │  约束: n_PA + n_PF + n_DA + n_DF ≤ G_total                        │  │
│  │  方法: ILP (Profile 表驱动)                                        │  │
│  │                                                                   │  │
│  │  关键: P/D 分配与 A/F 配比存在耦合，联合优化可发现分层方案无法         │  │
│  │        达到的全局最优配置 (如: 减少 PA 给 DF 以换取更低频率)          │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                          │                                              │
│          ┌───────────────┴───────────────┐                              │
│          ▼                               ▼                              │
│  ┌──────────────────────┐  ┌──────────────────────────┐                 │
│  │ Prefill AF Pipeline   │  │ Decode AF Pipeline        │                │
│  │                       │  │                           │                │
│  │  Tier 2 Controller    │  │  Tier 2 Controller        │                │
│  │  (每次 iteration)     │  │  (每次 iteration)         │                │
│  │                       │  │                           │                │
│  │  ┌─────┐   ┌─────┐   │  │  ┌─────┐    ┌─────┐      │                │
│  │  │ PA  │──→│ PF  │   │  │  │ DA  │──→ │ DF  │      │                │
│  │  │f_PA │   │f_PF │   │  │  │f_DA │    │f_DF │      │                │
│  │  └─────┘   └─────┘   │  │  └─────┘    └─────┘      │                │
│  │                       │  │                           │                │
│  │  SLO: TTFT            │  │  SLO: TPOT                │                │
│  └──────────────────────┘  └──────────────────────────┘                 │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Monitoring & Feedback (~10-30s)                                   │  │
│  │  • SLO 违反率 → 触发 Tier 1 重规划                                  │  │
│  │  • P/D 吞吐失衡检测 → 触发 P/D 资源重分配                           │  │
│  │  • A/F 负载不均衡检测 → 触发 A/F 配比调整                            │  │
│  │  • 负载模式突变检测 (请求长度分布变化) → 触发全局重规划               │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Tier 1: P/D + A/F 联合资源规划

#### 3.3.1 设计选择: 联合优化 vs 分层优化

一种自然的做法是**分层优化**: 先决定 P/D 资源划分 (复用 BiScale/DistServe)，再在各自内部优化 A/F 配比。但我们选择**联合优化**，原因如下:

**P/D 分配与 A/F 配比存在耦合**，分层优化会丢失全局最优:

```
例: 总共 16 GPU

分层优化:
  Level 0: 先按吞吐匹配决定 n_P=8, n_D=8
  Level 1: 在 n_P=8 内优化 → PA=3, PF=5, DA=4, DF=4

联合优化可能发现:
  PA=2, PF=4, DA=4, DF=6 (即 n_P=6, n_D=10)
  → Decode-FFN 在低频下延迟增加大，多给它 GPU 可用更低频率
  → 虽然 n_P=6 不是分层方案的最优 P/D 比，但整体能耗更低
  → 这种跨层优化只有联合方案能发现
```

**联合优化的可行性**: 搜索空间虽大于分层，但实际可控:
- TP 度候选有限: {1, 2, 4, 8}
- 频率候选有限: {210, 450, 690, 930, 1170, 1410}
- 副本数受 GPU 总量约束: k ≤ G / tp
- 大量组合因违反 SLO 或超出资源可提前剪枝
- ILP 求解器 (Gurobi/CPLEX/PuLP) 在此规模下通常秒级可解

#### 3.3.2 四池联合 ILP 优化公式

**目标函数**: 最小化四个算子池的总能耗

```
minimize:
  Σ_{c ∈ C} k_c × E_c(tp_c, f̄_c, workload_c)

  其中 C = {PA, PF, DA, DF}
  E_c = 池 c 中单个实例在给定配置和负载下的能耗 (来自 Profile 表)
```

**约束集合**:

```
// ===== 资源约束 =====
(1)  n_PA + n_PF + n_DA + n_DF ≤ G                  // GPU 总量上限
     其中 n_c = k_c × tp_c

// ===== TP 整数约束 =====
(2)  tp_c ∈ {1, 2, 4, 8}                            // TP 度候选集
(3)  k_c ∈ Z⁺                                        // 正整数副本数

// ===== 频率选择 =====
(4)  f̄_c ∈ {210, 450, 690, 930, 1170, 1410}             // 离散频率候选

// ===== 延迟 SLO 约束 (P/D 和 A/F 耦合的关键) =====
(5)  ∀ 请求类型 r ∈ R_prefill:
     t_PA(r, tp_PA, f̄_PA) + t_PF(r, tp_PF, f̄_PF) + t_AF_comm ≤ TTFT_SLO / L
     // L = 模型层数, 单层延迟约束

(6)  ∀ 请求类型 r ∈ R_decode:
     t_DA(r, tp_DA, f̄_DA) + t_DF(r, tp_DF, f̄_DF) + t_AF_comm ≤ TPOT_SLO

// ===== 吞吐容量约束 (P/D 耦合) =====
(7)  k_PA × Throughput_PA(tp_PA, f̄_PA) ≥ (1+α) × λ
(8)  k_PF × Throughput_PF(tp_PF, f̄_PF) ≥ (1+α) × λ
     // PA 和 PF 都需要匹配 prefill 到达率 λ

(9)  k_DA × Throughput_DA(tp_DA, f̄_DA) ≥ (1+α) × N_active
(10) k_DF × Throughput_DF(tp_DF, f̄_DF) ≥ (1+α) × N_active
     // DA 和 DF 都需要匹配活跃 decode 请求数 N_active

// ===== A/F 流水线平衡约束 =====
(11) |k_PA × Thpt_PA - k_PF × Thpt_PF| ≤ ε × max(k_PA × Thpt_PA, k_PF × Thpt_PF)
(12) |k_DA × Thpt_DA - k_DF × Thpt_DF| ≤ ε × max(k_DA × Thpt_DA, k_DF × Thpt_DF)
     // A/F 吞吐差距不超过 ε (如 10%), 避免严重的流水线气泡
```

**变量汇总**:

| 变量 | 含义 | 类型 | 范围 |
|------|------|------|------|
| k_PA, k_PF, k_DA, k_DF | 各池实例副本数 | 正整数 | [1, G/tp_min] |
| tp_PA, tp_PF, tp_DA, tp_DF | 各池 TP 并行度 | 离散 | {1, 2, 4, 8} |
| f̄_PA, f̄_PF, f̄_DA, f̄_DF | 各池基线频率 | 离散 | {210, 450, 690, 930, 1170, 1410} |

**输入参数**:

| 参数 | 含义 | 来源 |
|------|------|------|
| G | GPU 总量 | 集群配置 |
| λ | Prefill 请求到达率 (req/s) | 负载监控窗口统计 |
| N_active | 活跃 Decode 请求数 | 负载监控窗口统计 |
| R_prefill, R_decode | 请求类型集合 (按长度分桶) | 负载监控窗口统计 |
| TTFT_SLO, TPOT_SLO | 延迟 SLO 目标 | 用户配置 |
| α | 容量裕度 | 超参 (如 0.1-0.2) |
| ε | A/F 平衡松弛度 | 超参 (如 0.1) |
| t_AF_comm | A→F 中间激活传输延迟 | Profiling 测量 |

#### 3.3.3 联合优化为什么 P/D 和 A/F 耦合

从约束可见，P/D 和 A/F 之间存在多层耦合:

1. **资源竞争**: n_PA + n_PF + n_DA + n_DF ≤ G，给 PF 多一个 GPU，DA 就少一个
2. **延迟耦合**: PA 和 PF 的频率/TP 联合决定能否满足 TTFT SLO (约束 5)
3. **吞吐耦合**: PA 和 PF 的副本数各自需匹配 λ (约束 7-8)，但用不同 TP/频率时所需副本数不同
4. **能耗-频率权衡的跨池传递**: 降低 f̄_DA 省能耗但可能需要更多 DA 副本 → 挤占 PF 的 GPU → PF 需要更高频率 → 能耗在 DA 和 PF 之间转移

**联合优化捕捉这些耦合，找到全局最优的资源-频率分配。**

#### 3.3.4 ILP 求解加速

为控制求解时间，采用以下策略:

```
1. Profile 表预计算
   - 离线穷举所有 (tp, freq, workload_bin) 组合
   - 每个组合记录 (latency, power, throughput)
   - ILP 中直接查表，无需在线推理

2. 对称性与支配性剪枝
   - 若配置 A Pareto 支配配置 B (延迟更低且能耗更低)，剪掉 B
   - 每个池的候选配置从 |TP| × |F| = 16 种降至 Pareto 前沿上的少数几种

3. Warm Start
   - 用上一个规划窗口的解作为初始可行解
   - ILP 求解器可在此基础上快速改进

4. 负载分桶
   - 将连续的请求长度分布离散化为若干代表性桶
   - 约束 (5)(6) 只针对桶的代表值，而非每个请求
```

#### 3.3.5 A/F 负载不对称性驱动配比

F/A 延迟比随配置变化，是联合优化发现差异化配比的关键数据基础:

| 配置 | P_A (us) | P_F (us) | F/A 比 | 联合优化的含义 |
|------|---------|---------|--------|-------------|
| tp=1, in=128, bs=1, 1410MHz | 340 | 576 | 1.69 | F 是瓶颈 → 多给 PF GPU |
| tp=1, in=8192, bs=1, 1410MHz | 11,899 | 25,440 | 2.14 | F 是瓶颈 → 多给 PF GPU |
| tp=1, in=16384, bs=1, 1410MHz | 33,528 | 51,011 | 1.52 | F 是瓶颈但差距缩小 |
| tp=4, in=128, bs=1, 1410MHz | 338 | 254 | **0.75** | A 是瓶颈 → 多给 PA GPU |
| tp=4, in=1024, bs=4, 1410MHz | 1,011 | 4,132 | **4.09** | F 强瓶颈 → 倾斜给 PF |
| tp=8, in=128, bs=4, 1410MHz | 343 | 417 | **1.22** | 接近均衡 |

**当负载以长请求为主时，ILP 倾向给 PF 分配更多 GPU (降低 f̄_PF 的同时保持吞吐); 短请求为主时反之。** 新数据修正：tp=8 时 F/A 比不再极端（旧数据 tp=8 bs=4 il=8192 F/A=13.6x 为测量异常），实际 F/A 比在 1~4x 范围内，ILP 配比仍有优化空间但不如旧数据预期的那么极端。

#### 3.3.6 重规划触发与开销控制

**触发条件** (Monitoring 模块, 每 10-30s 采样):

```
metric_1: SLO_violation_rate > threshold        // SLO 违反率过高
metric_2: |A_util - F_util| > δ                  // A/F 利用率失衡
metric_3: |P_util - D_util| > δ                  // P/D 利用率失衡
metric_4: workload_distribution_shift detected   // 请求长度分布突变

任一指标在连续多个窗口超过阈值 → 触发 Tier 1 ILP 重规划
```

**开销控制**:
- 重规划涉及 GPU 在四个池之间的角色切换 (加载不同权重)
- 预缓存 PA/PF/DA/DF 四种角色的权重在本地磁盘
- Shadow instancing: 后台预创建新配置的实例，ready 后原子切换
- 控制重规划间隔不低于 T₁_min (如 2-5 分钟)

#### 3.3.7 Profile 表构建

离线对每个池独立 profiling:

```
PA_Profile[tp, input_len, batch_size, freq] → (latency_PA, power_PA, throughput_PA)
PF_Profile[tp, input_len, batch_size, freq] → (latency_PF, power_PF, throughput_PF)
DA_Profile[tp, input_len, output_len, batch_size, freq] → (latency_DA, power_DA, throughput_DA)
DF_Profile[tp, input_len, output_len, batch_size, freq] → (latency_DF, power_DF, throughput_DF)
```

#### 3.3.8 Ablation 设计: 联合 vs 分层

在 Evaluation 中通过以下对比验证联合优化的优越性:

| Baseline | 描述 | 预期结果 |
|----------|------|---------|
| **Hierarchical** | 先按吞吐匹配决定 n_P/n_D，再各自内部优化 A/F | 能耗高于联合方案 |
| **Fixed-Ratio** | 固定 A/F=1:1，仅优化频率 | 流水线气泡导致资源浪费 |
| **PD-Only** | 仅 P/D 分离，无 AF 分离 (如 BiScale) | 缺少算子级频率自由度 |
| **Joint (本文)** | 联合优化 P/D+A/F 配比+频率 | **最优** |

这组 ablation 可以量化两个维度的增益:
- **AF 分离的增量收益**: Joint vs PD-Only
- **联合优化的增量收益**: Joint vs Hierarchical

### 3.4 Tier 2: Per-Iteration 算子级 DVFS

每次 iteration，独立选择 f_A 和 f_F，目标是找到满足 SLO 的最低能耗频率组合。

#### 3.4.1 Prefill 侧 DVFS

Prefill 突发性强，延迟敏感 (TTFT)，采用基于 SLO slack 的联合频率搜索:

```
算法: Prefill AF-DVFS (每次 iteration)

输入:
  当前 batch B_PA (已调度到 PA 的请求集合)
  每个请求 r 的 TTFT deadline: d_r
  PA/PF Profile 模型 M_PA, M_PF

步骤:
  1. 对当前 batch, 计算 TTFT slack:
     slack = min_{r ∈ B} (d_r - elapsed_r) - t_AF_comm

  2. 搜索 (f_PA, f_PF) 组合:
     candidates = {}
     for f_PA in [210, 450, 690, 930, 1170, 1410]:
       for f_PF in [210, 450, 690, 930, 1170, 1410]:
         t_PA = M_PA.predict(batch, f_PA)
         t_PF = M_PF.predict(batch, f_PF)
         if (t_PA + t_PF + t_AF_comm) × remaining_layers ≤ slack:
           e = power_PA(f_PA) × t_PA + power_PF(f_PF) × t_PF
           candidates.add((f_PA, f_PF, e))

  3. 选择能耗最低的可行组合:
     (f_PA*, f_PF*) = argmin_{(f_PA, f_PF) ∈ candidates} e

  4. 设置频率并执行

复杂度: O(|F|²) = O(16) — 常数时间，可忽略
```

**设计选择说明 (为什么不用 MPC)**:
- BiScale 的 MPC 是因为 Prefill 只有一个频率旋钮 (f_P)，需要在时间维度上优化未来 K 个 batch 的频率序列
- AF 分离提供了空间维度的额外自由度 (f_A vs f_F)，使得单步的 (f_A, f_F) 联合搜索就已经有足够大的节能空间
- 如需进一步优化，可扩展为 MPC

#### 3.4.2 Decode 侧 DVFS

Decode 负载平滑，延迟不那么敏感，采用 per-batch slack-aware 策略:

```
算法: Decode AF-DVFS (每次 iteration)

输入:
  当前 batch B_DA (活跃 decode 请求)
  TPOT budget: SLO_TPOT
  DA/DF Profile 模型 M_DA, M_DF

步骤:
  1. 获取当前 batch 特征: batch_size, avg_kv_blocks

  2. 按能耗升序搜索可行 (f_DA, f_DF):
     for (f_DA, f_DF) in sorted_by_energy(all_combos):
       t_DA = M_DA.predict(batch, f_DA)
       t_DF = M_DF.predict(batch, f_DF)
       if t_DA + t_DF + t_AF_comm ≤ SLO_TPOT:
         return (f_DA, f_DF)

     // 若无可行组合, 使用最高频率
     return (1410, 1410)

  注: 由于 DA 对频率非常不敏感 (强 memory-bound),
      很可能最优解总是 f_DA = 最低频, f_DF 按 slack 选择
```

#### 3.4.3 频率切换开销处理 (实测数据)

> **实验平台**: A800-80GB SXM × 8, NVIDIA Driver, SetGpuLockedClocks API
>
> **实验结论** (benchmark/test_motivation/bench_dvfs_overhead.py):
>
> | 指标 | SetApplicationsClocks | SetGpuLockedClocks |
> |------|----------------------|-------------------|
> | API latency P50 | ~2.7 ms | ~4.5 ms |
> | API latency avg | ~3.4 ms | ~6.0 ms |
> | API latency P99 | ~10 ms | ~13 ms |
> | 降频生效? | **不生效** (仅升频有效) | **立即生效** (升降均有效) |
> | Settle 100%? | 仅降到最低频 | **全部 100%** |
> | Settle avg | N/A (超时) | ~0.9 ms |
>
> **必须使用 SetGpuLockedClocks**: SetApplicationsClocks 是 "建议" 频率, 降频不生效, 不适合精确调频场景。
>
> **多 GPU 切频**: 受 NVIDIA 内核驱动全局锁限制, N 卡切频被串行化, 总开销 ~N × 6ms。
> 多线程/多进程均无法绕过 (锁在 nvidia.ko 内核模块中)。
> 8 卡串行: ~46ms, 8 卡多线程: ~46ms, 8 卡多进程: ~56ms (额外 IPC 开销)。
>
> **对系统设计的影响**:
> - 单 GPU 切频 ~6ms, 在 AF 分离架构中每个池独立调频, 单次涉及 1-2 卡, 开销 6-12ms
> - Tier 2 调频粒度应为 per-batch 而非 per-iteration (Decode iteration ~1-2ms, 切频开销远大于 iteration)
> - 惰性切频策略为必选项, 不能频繁切频

**AF 分离架构下**: A 和 F 在不同 GPU 上，各自维持各自的频率，**不存在切频开销**。切频开销仅在以下场景相关:

- **Tier 1 重规划**: 调整某个池的基线频率 → ~6ms 完全可接受 (T₁ 是分钟级)
- **Tier 2 相邻 batch 间切换频率**: 每个池内部切频 ~6ms (per-batch, 非 per-iteration)

**对 Decode 的影响**:
- Decode iteration 典型耗时 ~1ms (bs=1) 到 ~2ms (bs=256)
- ~6ms 切频 + 1ms iteration = 7ms → per-iteration 切频开销过大, 不可行
- 应采用 per-batch 调频: batch 间隔 ~50-200ms, 6ms 开销占 3-12%, 可接受

**Decode 惰性切频策略**:
```
if |f_new - f_current| ≥ threshold AND estimated_savings > switching_cost:
  switch_freq(f_new)      // 承担 ~6ms 开销, 需确保后续运行足够多 iteration 回本
else:
  keep f_current           // 维持当前频率, 避免开销
```

**对 Prefill 的影响**:
- Prefill iteration 通常 ≥10ms (长序列可达数十到数百 ms)
- ~6ms 切频开销比例较小 (<60%)，per-batch 调频可行

### 3.5 能耗模型

```
单次 iteration 能耗:
  E_iter = P_A(f_A, batch, ctx) × t_A(f_A, batch, ctx)   // Attention 能耗
         + P_F(f_F, batch, ctx) × t_F(f_F, batch, ctx)   // FFN 能耗
         + E_comm                                         // AF 通信能耗
         + E_idle                                         // 流水线空闲能耗

AF 分离的 Pareto 优势:
  E_AF(f_A, f_F) ≤ E_unified(f)  当 f 使得延迟相同时

  原理: 当 A 是 memory-bound 时, 降 f_A 几乎不增加 t_A,
        但大幅降低 P_A (功耗 ∝ f^α)。
        同时 f_F 可以保持高频以保证 t_F 不成为瓶颈。
```



## 四、与已有工作的定位对比

| 维度 | DynamoLLM | throttLL'eM | BiScale | **本文 (Paper 1)** |
|------|-----------|------------|---------|-------------------|
| 分离粒度 | 无 | 无 | PD 两阶段 | **PD + AF 四算子** |
| 频率自由度 | 1 | 1 | 2 (f_P, f_D) | **4 (f_PA, f_PF, f_DA, f_DF)** |
| 核心 insight | 请求异构 | 预测式控制 | 阶段不对称 | **算子级频率异构** |
| 资源配比 | 固定 TP 池 | 动态 TP | P/D 实例数 | **A/F 实例配比** |
| 切频开销 | 50-80ms | ~200ms | 未讨论 | **~6ms (P50 4.5ms)** |

---

## 五、核心贡献点

1. **首次揭示 Attention 和 FFN 的频率敏感性异构**——并用 Profiling 数据量化 (Pareto 图)
2. **AF 分离 + 差异化调频**——比 PD 级 DVFS 多出两个频率自由度，Pareto 前沿更优
3. **P/D + A/F 联合资源规划**——首次将 P/D 分配和 A/F 配比视为一个联合优化问题，揭示并利用跨层耦合效应，发现分层分解无法达到的全局最优配置
4. **动态 A/F 配比**——首次将 A/F 负载不对称性作为资源分配的依据
5. **低切频开销 (P50 ~4.5ms, avg ~6ms) 使 per-batch 级调频成为可能**——比 throttLL'eM (200ms) 低一个数量级; 实测验证 SetGpuLockedClocks 升降频均立即生效, 多 GPU 受内核驱动全局锁限制需串行化但在 AF 独立进程架构下影响有限

---

## 六、论文结构建议

```
1. Introduction
   - LLM 推理能耗问题
   - AF 分离作为新的能效优化维度

2. Background & Motivation
   - PD 分离与 AF 分离的背景
   - 算子级频率异构性观察 (Pareto 图作为 motivation)
     → 统一调频 vs AF 差异化调频的 Pareto 前沿对比
   - A/F 负载不对称性随工作负载变化 (F/A ratio 数据)

3. System Design
   - 两层控制架构
   - Tier 1: P/D + A/F 联合 ILP 资源规划 (四池联合优化)
   - Tier 2: Per-batch (f_A, f_F) 联合搜索 (实测表明 per-iteration 不可行)
   - 切频开销处理 (~6ms overhead analysis, SetGpuLockedClocks 实测)
   - A/F 流水线平衡约束
   - 联合优化 vs 分层优化的设计选择论证

4. Implementation
   - 基于 SGLang 的 AF 分离实现
   - Profile 表构建
   - 频率控制接口 (NVML)

5. Evaluation
   - vs 统一调频 (同一频率给 A 和 F)
   - vs BiScale (PD 分离但无 AF 分离)
   - vs DynamoLLM (无分离)
   - Ablation: Tier 1 alone, Tier 2 alone, combined
   - Ablation: 联合优化 vs 分层优化 (量化跨层耦合的收益)
   - 不同负载模式下的 P/D+A/F 配比自适应
   - 切频开销分析

6. Related Work
7. Conclusion
```

---

## 七、前置工作清单

### Phase 1: Profiling 数据采集 (详细执行方案)

#### 现有基础设施

| 组件 | 路径 | 状态 |
|------|------|------|
| DVFSController (Python) | `python/sglang/srt/layers/dvfs.py` | ✅ 可用，支持 `lock_sm_clock()` + `get_power()` |
| libdvfs_ctrl.so (C++ NVML) | `benchmark/test_motivation/dvfs/` | ✅ 已编译，需 `cd dvfs && make` |
| DVFS 开销 benchmark | `benchmark/test_motivation/bench_dvfs_overhead.py` | ✅ 已完成 |
| AFD 实现 (Attention/FFN 分离) | `python/sglang/srt/layers/afd_mixin.py` | ✅ 可独立调用 self_attn / mlp |
| A/F Profiling 脚本 | — | ❌ 不存在，需新建 |
| NVML 能耗计数器接口 | `dvfs_ctrl.cpp` | ❌ 需新增 `dvfs_get_energy_mj()` |

#### 能耗采集方案: NVML 硬件能耗计数器

NVML 提供 `nvmlDeviceGetTotalEnergyConsumption()` API，返回 GPU 自驱动加载以来的
**累积能耗** (单位 mJ)。这是硬件级计数器，连续累积，精度远高于功耗采样。

```
直接测能耗 (推荐):
  e0 = nvmlDeviceGetTotalEnergyConsumption()   // mJ
  执行 Attention × N 次
  e1 = nvmlDeviceGetTotalEnergyConsumption()   // mJ
  energy_A = (e1 - e0) / N                    // mJ/次 (精确)

vs 间接算能耗 (旧方案, 已弃用):
  power_samples = [后台线程采样功耗 W]          // 50-100ms 间隔, 可能漏掉瞬态
  energy_A ≈ mean(power_samples) × latency     // 近似值
```

**优势**: 无需后台采样线程，无采样对齐问题，几行代码即可完成。

**需要新增的接口** (一次性开发):

```cpp
// dvfs_ctrl.cpp 新增
int dvfs_get_energy_mj(int gpu_idx) {
    nvmlDevice_t dev = get_device(gpu_idx);
    if (!dev) return -1;
    unsigned long long energy = 0;
    nvmlReturn_t ret = nvmlDeviceGetTotalEnergyConsumption(dev, &energy);
    if (ret != NVML_SUCCESS) return -1;
    return (long long)energy;  // millijoules
}
```

```python
# dvfs.py 新增
def get_energy_mj(self) -> int:
    """Cumulative energy consumption in millijoules since driver load."""
    return self._lib.dvfs_get_energy_mj(self.device_index)
```

#### A/F 独立 Profiling 方案

**核心思路**: 在同一块 GPU 上，**时间上隔离** Attention 和 FFN，独立测量各自的延迟和能耗。
不需要 AF 分离基础设施，不需要多 GPU，不需要 AFD 通信。

```
同一块 GPU，时间上隔离:

  ┌── Attention × N 次 ──┐    ┌── FFN × N 次 ──┐
  │  e0            e1    │    │  e2         e3  │
  └──────────────────────┘    └─────────────────┘
  
  energy_A = (e1 - e0) / N    energy_F = (e3 - e2) / N
  latency_A = wall_time / N    latency_F = wall_time / N
  power_A = energy_A / latency_A  (反推平均功率, 可选)
```

SGLang `LlamaDecoderLayer` 的两个子模块可直接独立调用:
```python
layer.self_attn(positions, hidden_states, forward_batch)  # Attention
layer.mlp(hidden_states)                                   # FFN
```

---

#### [P0-1] Prefill A/F 延迟 + 功耗 Profiling

**现有数据对比**:

| | prefill_data.txt (旧) | prefill_data_v0.txt (旧) | prefill_data_v1.txt (新) |
|--|----------------------|------------------------|------------------------|
| batch_size | {1, 16, 256(tp=8)} | {1} | **{1, 2, 4, 8, 16, 32}** |
| input_len | {128, 512, 4096} | {128..40000, 10 档} | **{128..40000, 10 档}** |
| tp | {1, 2, 4, 8} | {1, 2, 4, 8} | {1, 2, 4, 8} |
| gpu_clock | {210, 540, 870, 1200} | {210, 540, 870, 1200} | **{210, 450, 690, 930, 1170, 1410}** |
| 功耗 | ❌ | ❌ | **✅ P_A_power, P_F_power** |

**采集矩阵**:

```
tp         ∈ {1, 2, 4, 8}                                              → 4 值
input_len  ∈ {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32000}     → 9 值
batch_size ∈ {1, 2, 4, 8, 16, 32}                                      → 6 值
gpu_clock  ∈ {210, 450, 690, 930, 1170, 1410}                          → 6 值

总配置数 = 4 × 9 × 6 × 6 = 1296 个组合
需排除不可行组合 (OOM 等), 预计实际 ~900-1000 个

注意:
  - tp=1 时 input_len=32000 + bs=32 可能 OOM → 自动跳过
  - tp=8 时可额外测 bs=64, 128, 256
  - 频率值需 snap_to_supported() 对齐到 GPU 实际支持的频率点
```

**每个配置的执行流程**:

```
对每个 (tp, input_len, batch_size, gpu_clock):
  1. 设置 TP 并行度 (需启动对应的模型实例)
  2. lock_sm_clock(gpu_clock)
  3. 构造输入: input_ids = random tokens, shape = (batch_size, input_len)
  4. 预热 10 次 forward (单层)
  5. 启动功耗采样线程
  6. 执行 50 次 forward (单层), 用 CUDA events 分别记录 t_A, t_F
  7. 停止功耗采样
  8. 记录:
     - P_A_lat = median(t_A_list) (us)
     - P_F_lat = median(t_F_list) (us)
     - P_A_power = mean(power_samples_during_A) (W)  -- 若无法对齐则用总 power
     - P_F_power = mean(power_samples_during_F) (W)
     - P_total_power = mean(all_power_samples) (W)
  9. unlock_sm_clock()
```

**A/F 功耗分离: 采用方案 B (独立执行 + 独立功耗采集)**

在 AF 分离架构下，A 和 F 在不同 GPU 上独立运行、独立设频，能耗模型为:
```
E = P_A(f_A) × t_A(f_A) + P_F(f_F) × t_F(f_F)
```
P_A(f) 和 P_F(f) 是两个独立的函数——Attention (memory-bound) 和 FFN (compute-bound)
在同一频率下的功耗不同，功耗-频率关系也不同。因此**必须分别测量**。

**SGLang 已有 AFD 支持，可直接独立调用 Attention / FFN**:

```python
# LlamaDecoderLayer 的两个子模块可独立调用:
#   self.self_attn(positions, hidden_states, forward_batch)  → Attention
#   self.mlp(hidden_states)                                  → FFN
# 
# 或使用 AFD mixin 的接口:
#   layer.forward_afd_A(positions, hidden_states, forward_batch, residual)
#   layer.forward_afd_F(hidden_states, forward_batch, residual)
#
# 关键文件:
#   python/sglang/srt/layers/afd_mixin.py  → _run_attn(), _run_mlp()
#   python/sglang/srt/models/llama.py      → LlamaDecoderLayer
```

**方案 B 执行流程 (每个配置点)**:

```
Phase A: 纯 Attention 功耗采集
  1. lock_sm_clock(target_freq)
  2. 构造输入 hidden_states, positions, forward_batch
  3. 预热: 执行 self.self_attn(...) × N_warmup 次
  4. 启动功耗采样线程
  5. 连续执行 self.self_attn(...) × N_repeat 次, CUDA events 计时
  6. 停止采样 → power_A(freq), t_A(freq)

Phase F: 纯 FFN 功耗采集
  7. 构造 FFN 输入 (Attention 的输出 shape)
  8. 预热: 执行 self.mlp(...) × N_warmup 次
  9. 启动功耗采样线程
  10. 连续执行 self.mlp(...) × N_repeat 次, CUDA events 计时
  11. 停止采样 → power_F(freq), t_F(freq)
  12. unlock_sm_clock()

N_repeat ≥ 50, 确保采样线程 (~20ms 间隔) 采到 ≥10 个功耗样本
```

**输出格式**: `prefill_data_v1.txt` (Tab 分隔)

```
tp  input_len  gpu_clock  batch_size  P_A_lat  P_F_lat  P_A_power  P_F_power  idle_power_W
```

**时间估算**:

```
每个配置: ~10s (预热 + 50次执行 + 功耗采样)
每个 TP 需要重新启动模型: ~2-5 min (模型加载)
总配置: ~700 个 ÷ 4 TP = ~175 个/TP
每个 TP: 175 × 10s ≈ 30 min + 5 min 加载 ≈ 35 min
总时间: 4 × 35 min ≈ 2.5 小时
加上 OOM 重试和间歇: ~3-4 小时
```

---

#### [P0-2] Decode A/F 延迟 + 功耗 Profiling

**采集矩阵**:

```
tp         ∈ {1, 2, 4, 8}                                → 4 值
input_len  ∈ {128, 512, 1024, 2048, 4096, 8192}           → 6 值
output_len ∈ {64, 256, 512, 4096}                         → 4 值
batch_size ∈ {1, 4, 8, 16, 32, 64, 128, 256}              → 8 值
gpu_clock  ∈ {210, 450, 690, 930, 1170, 1410}              → 6 值

总配置数 = 4 × 6 × 4 × 8 × 6 = 4608 个组合
去除 OOM 等: 预计实际 ~1200-1400 个
```

**Decode 特殊处理 (方案 B: 独立 A/F 功耗)**:

```
对每个 (tp, input_len, output_len, batch_size, gpu_clock):

  准备阶段:
    1. 先执行 Prefill (input_len tokens) 填充 KV cache
    2. 执行 output_len 步 Decode 生成 KV cache 到目标长度
       (此时 KV cache 包含 input_len + output_len 个 token)

  数据含义: 测量的是生成第 output_len 个 output token 时的单步 Decode 延迟和功耗。
            此时 KV cache 中有 (input_len + output_len - 1) 个 token 的 KV 条目
            (input_len 个来自 Prefill, output_len - 1 个来自前序 Decode 步)。
            output_len 越大 → KV cache 越大 → Attention 访存量越大 → 延迟越高。

  Phase A: 纯 Decode-Attention 功耗
    3. lock_sm_clock(gpu_clock)
    4. 连续执行 self.self_attn(...) × N_repeat 次 (batch_size 个请求同时 decode)
    5. 后台采样功耗 → D_A_power(freq)
    6. CUDA events → D_A_lat(freq)

  Phase F: 纯 Decode-FFN 功耗
    7. 连续执行 self.mlp(...) × N_repeat 次
    8. 后台采样功耗 → D_F_power(freq)
    9. CUDA events → D_F_lat(freq)
    10. unlock_sm_clock()
```

**输出格式**: `decode_data_v1.txt` (Tab 分隔)

```
tp  input_len  output_len  gpu_clock  batch_size  D_A_lat  D_F_lat  D_A_power  D_F_power  idle_power_W
```

**时间估算**: ~2-3 小时

---

#### [P0-3] AF 通信开销测量 (t_AF_comm)

**目的**: 测量 Attention GPU → FFN GPU 之间中间激活传输延迟。

**测量方法**:

```python
# 伪代码
import torch
import torch.distributed as dist

hidden_size = 8192  # Llama 70B

for bs in [1, 4, 16, 64, 256]:
    for seq_len in [1, 128, 1024, 8192]:
        tensor = torch.randn(bs, seq_len, hidden_size, dtype=torch.bfloat16, device='cuda:0')
        
        # 预热
        for _ in range(10):
            dist.send(tensor, dst=1)  # 或 NCCL P2P
        
        # 计时
        times = []
        for _ in range(100):
            torch.cuda.synchronize()
            t0 = time.perf_counter_ns()
            dist.send(tensor, dst=1)
            torch.cuda.synchronize()
            t1 = time.perf_counter_ns()
            times.append((t1 - t0) / 1000)  # us
        
        data_bytes = bs * seq_len * hidden_size * 2  # bf16 = 2 bytes
        latency_us = median(times)
        bw_gbps = data_bytes / (latency_us * 1e-6) / 1e9
```

**测量维度**:

```
传输方式:
  - NVLink 直连 (同节点, GPU 0 → GPU 1)
  - NCCL send/recv
  - torch.distributed P2P

batch_size ∈ {1, 4, 16, 64, 256}
seq_len    ∈ {1, 128, 1024, 8192}
  注: Decode 时 seq_len=1 (仅传 1 token 的 hidden)
      Prefill 时 seq_len = input_len
```

**输出格式**: `af_comm_overhead.txt`

```
transport  batch_size  seq_len  data_bytes  latency_us  bandwidth_gbps
```

**时间估算**: ~30 分钟

---

#### [P0-4] 频率切换开销 (已完成)

> ✅ 已完成。脚本: `bench_dvfs_overhead.py`
> 结论: SetGpuLockedClocks P50 ~4.5ms, avg ~6ms, P99 ~13ms。
> 多 GPU 受内核驱动全局锁限制串行化。

---

#### [P0-5] 空闲功耗基线

**目的**: 测量各频率下 GPU 空闲功耗，用于计算净功耗。

```
对每个 gpu_clock ∈ {210, 450, 690, 930, 1170, 1410}:
  1. lock_sm_clock(gpu_clock)
  2. 无任何 CUDA kernel 运行
  3. 采样 30s 功耗
  4. 记录 idle_power_W

输出: idle_power.txt
  gpu_clock  idle_power_W  idle_power_std_W
```

**时间估算**: ~5 分钟

---

#### 基于现有 AFD 代码的 Profiling 实现方案

SGLang 已有完整的 AFD 实现，可以直接利用:

```
关键代码文件:
  python/sglang/srt/layers/afd_mixin.py    → forward_afd_A(), forward_afd_F()
  python/sglang/srt/layers/afd.py          → model_forward_afd(), AFDProxyAttention/MLP
  python/sglang/srt/models/llama.py        → LlamaDecoderLayer (已集成 AFD mixin)
  python/sglang/srt/layers/dvfs.py         → DVFSController (锁频 + 功耗读取)
```

**两种 Profiling 方式**:

```
方式 1 (推荐, 更简单): 单 GPU 直接调用子模块
  → 加载完整 LlamaDecoderLayer 到单 GPU
  → 独立调用 layer.self_attn(...) 采集 Attention 延迟+功耗
  → 独立调用 layer.mlp(...) 采集 FFN 延迟+功耗
  → 优点: 无需多 GPU, 无需启动 AFD 通信, 简单快速
  → 缺点: 不含 LayerNorm/residual 开销 (很小, 可忽略)

方式 2 (完整, 更精确): 启动两个 AFD 进程
  → 用 --afd-perspective attn 启动 Attention 节点
  → 用 --afd-perspective ffn 启动 FFN 节点
  → 各自锁频, 各自采集功耗
  → 优点: 包含完整 AFD 通信 + norm 开销
  → 缺点: 需要多 GPU + 配置 AFD 通信后端, 复杂度高
```

**方式 1 的核心代码框架**:

```python
import torch
from sglang.srt.layers.dvfs import DVFSController

# 加载单层 LlamaDecoderLayer (而非整个模型, 节省显存)
layer = load_single_decoder_layer(model_path, layer_id=0, tp=tp, device='cuda:0')
ctrl = DVFSController(device_index=0)

FREQS = [210, 450, 690, 930, 1170, 1410]

for freq in FREQS:
    freq = ctrl.snap_to_supported(freq)
    ctrl.lock_sm_clock(freq)
    
    # --- Attention Profiling ---
    hidden = torch.randn(batch_size, input_len, hidden_size, device='cuda:0', dtype=torch.bfloat16)
    # warmup
    for _ in range(10):
        _ = layer.self_attn(positions=pos, hidden_states=hidden, forward_batch=fb)
    # measure
    power_sampler.start()
    attn_times = []
    for _ in range(N_REPEAT):
        start_event.record()
        _ = layer.self_attn(positions=pos, hidden_states=hidden, forward_batch=fb)
        end_event.record()
        torch.cuda.synchronize()
        attn_times.append(start_event.elapsed_time(end_event) * 1000)  # us
    power_A = power_sampler.stop_and_mean()
    t_A = median(attn_times)
    
    # --- FFN Profiling ---
    ffn_input = torch.randn(batch_size, input_len, hidden_size, device='cuda:0', dtype=torch.bfloat16)
    for _ in range(10):
        _ = layer.mlp(ffn_input)
    power_sampler.start()
    ffn_times = []
    for _ in range(N_REPEAT):
        start_event.record()
        _ = layer.mlp(ffn_input)
        end_event.record()
        torch.cuda.synchronize()
        ffn_times.append(start_event.elapsed_time(end_event) * 1000)
    power_F = power_sampler.stop_and_mean()
    t_F = median(ffn_times)
    
    ctrl.unlock_sm_clock()
    # 写入结果: tp, input_len, freq, batch_size, t_A, t_F, power_A, power_F
```

**关键挑战: 如何加载单层**

```
问题: SGLang 模型加载是整体的 (所有层 + embedding + lm_head)
      Llama 70B 整体加载需要大量显存

解决方案 (按推荐度排序):

1. 只加载 1 层的权重 (最省显存)
   → 用 safetensors.torch.load_file() 仅加载 layer.0 的权重
   → 手动构造 LlamaDecoderLayer + LlamaAttention + LlamaMLP
   → 需要模型 config (hidden_size, num_heads, num_kv_heads 等)

2. 加载完整模型但只 profile 第 0 层 (最简单)
   → 正常启动 SGLang server / model_runner
   → 在 LlamaDecoderLayer.forward 中插入 profiling hooks
   → 利用已有的 forward_batch 构造机制

3. 使用 HuggingFace transformers 加载单层 (中等复杂度)
   → from transformers import LlamaForCausalLM
   → model.model.layers[0].self_attn / .mlp
   → 但 kernel 实现不同 (无 FlashAttention 等优化)
   → 仅适合初步验证, 不代表 SGLang 真实性能

推荐先用方案 2 (加载完整模型), 开发效率最高。
对于 tp=1/2/4, 70B 模型可以放下。
```

**Decode Profiling 的特殊要求**:

```
Decode 阶段 Attention 需要 KV cache:
  → 必须先 Prefill 填充 KV cache 到目标长度
  → 然后在目标 KV cache 大小下反复执行 Decode Attention
  → Decode-FFN 与 KV cache 大小无关 (仅 batch_size 影响)

实现:
  1. forward_batch 设置为 extend/prefill 模式, 执行 Prefill 填充 KV cache
  2. forward_batch 切换为 decode 模式 (seq_len=1)
  3. 在 decode 模式下反复执行 layer.self_attn() 采集延迟+功耗
  4. 在 decode 模式下反复执行 layer.mlp() 采集延迟+功耗
```

---

#### 需新建的脚本清单

| 脚本 | 功能 | 依赖 |
|------|------|------|
| `bench_prefill_af.py` | Prefill A/F 延迟+功耗 profiling (方案 B) | DVFSController, 模型加载, CUDA events |
| `bench_decode_af.py` | Decode A/F 延迟+功耗 profiling (方案 B) | 同上 + KV cache 管理 |
| `bench_af_comm.py` | AF 通信开销测量 | torch.distributed, NCCL |
| `bench_idle_power.py` | 空闲功耗基线 | DVFSController |
| `utils_profiling.py` | 公共工具 (功耗采样线程, CUDA 计时, 结果输出) | DVFSController |

**公共工具 `utils_profiling.py` 包含**:

```python
class PowerSampler:
    """后台线程持续采样 GPU 功耗 (via DVFSController.get_power())"""
    def __init__(self, ctrl: DVFSController, interval_ms=20): ...
    def start(self): ...
    def stop_and_mean(self) -> float: ...  # 返回平均功耗 W
    def stop_and_samples(self) -> list[float]: ...  # 返回所有样本

class CUDATimer:
    """CUDA events 计时器, 自动管理 start/end events"""
    def __init__(self): ...
    def start(self): ...
    def stop(self) -> float: ...  # 返回 us

def load_single_decoder_layer(model_path, layer_id, tp, device):
    """加载单个 LlamaDecoderLayer 的权重到指定设备"""
    ...

def run_profiling_sweep(layer, configs, output_file, phase='prefill'):
    """通用 profiling 扫描循环"""
    ...
```

---

#### 执行顺序与依赖

```
Step 1: 编写 utils_profiling.py (公共工具)
  └── 功耗采样线程, CUDA 计时, 结果输出格式

Step 2: 编写 bench_idle_power.py → 跑一次 [P0-5]    (~5 min)
  └── 验证功耗采集链路正确

Step 3: 编写 bench_prefill_af.py → 跑一次 [P0-1]    (~3-4 hours)
  └── 需要: 模型加载 (Llama 70B), A/F 分离计时方案确认

Step 4: 编写 bench_decode_af.py → 跑一次 [P0-2]     (~2-3 hours)
  └── 可与 Step 3 合并为一个脚本 (prefill+decode 一起跑)

Step 5: 编写 bench_af_comm.py → 跑一次 [P0-3]       (~30 min)
  └── 需要多 GPU 环境

Step 6: 数据验证
  └── 与旧数据 (prefill_data_v0.txt, decode_data.txt) 的延迟列对比
  └── 验证趋势一致性
  └── 更新 Pareto 图 (用真实功耗替代 f^α 代理)

总计: ~1 天 (脚本编写) + ~1 天 (数据采集) + ~半天 (验证)
```

### Phase 2: 模型构建与验证

#### [P1-1] 能耗模型验证与拟合

```
任务:
  1. 用 [P0-1/P0-2] 的真实功耗数据拟合:
     - power(f) = a × f^α + b，分别为 A 和 F 拟合 α_A, α_F
     - 评估查找表 vs 线性回归 vs f^α 公式的 MAPE
  2. 确定 Tier 2 使用哪种功耗预测方法:
     - 查找表 + 线性插值 (BiScale 方案, Prefill 功耗 MAPE 4.1%)
     - GBDT (throttLL'eM 方案, IPS MAPE 2.8-5.8%)
     - 简单回归 (BiScale Decode 功耗 MAPE 1.0%)
  3. 验证 Pareto 图结论在真实功耗下是否仍然成立
     (即 AF grid 是否仍然 Pareto 优于 unified DVFS)
```

#### [P1-2] 吞吐量 Profiling (ILP Throughput 参数)

```
任务:
  对每个 (池类型, tp, freq) 配置:
    - 用递增的 RPS 压力发送请求
    - 找到满足 SLO 的最大吞吐 → Throughput_c(tp_c, f̄_c)
    - 记录 goodput vs RPS 曲线
  
  或使用 BiScale 的方法:
    - 构建频率感知的迭代级模拟器
    - 二分搜索确定每个配置的最大可行 goodput
```

### Phase 3: 系统实现

#### [P2-1] SGLang AF 分离推理引擎

```
任务:
  - Attention 和 FFN 在不同 GPU 组上执行
  - A→F 中间激活传输 (hidden states via NCCL/P2P)
  - KV cache 管理 (Attention GPU 上)
  - AF 流水线调度器
```

#### [P2-2] NVML 频率控制接口集成

```
任务:
  - 封装 SetGpuLockedClocks 为 Python 可调用接口
  - 每个 GPU 独立设频, 不影响同节点其他 GPU
  - 验证推理中动态切频不导致 CUDA 错误或精度问题
  - 实现惰性切频策略 (Decode 侧)
```

#### [P2-3] Tier 1 ILP 求解器

```
任务:
  - 基于 PuLP/Gurobi 实现四池联合 ILP
  - Profile 表加载与查询
  - Pareto 剪枝 + Warm Start
  - 负载分桶
```

#### [P2-4] Tier 2 DVFS 控制器

```
任务:
  - Prefill: (f_A, f_F) 联合搜索 (SLO slack-aware)
  - Decode: per-batch slack-aware + 惰性切频
  - 功耗预测模型在线推理
  - SLO 违反回退机制 (升频到最高)
```

### Phase 4: 评估

#### [P3-1] Trace 准备

```
任务:
  - 获取 Azure LLM trace 或 ShareGPT/LMSYS 公开 trace
  - 提取: 到达率 λ(t)、输入长度分布、输出长度分布
```

#### [P3-2] Baseline 实现

```
需要准备:
  - Max-Freq: 所有池最高频率 (DistServe 风格)
  - Unified DVFS: 同频给 A 和 F
  - PD-Only: PD 分离但无 AF 分离 (BiScale 方案)
  - Hierarchical: 分层优化 (先 P/D 再 A/F)
  - Fixed-Ratio: A/F=1:1 固定, 仅优化频率
  - Joint (本文): 联合优化 P/D+A/F 配比+频率
```

### 全部工作项 (按执行顺序, 含具体产出和时间)

```
═══════════════════════════════════════════════════════════════════
 Phase 0: 基础设施准备 (~1 天)                       ✅ 已完成
═══════════════════════════════════════════════════════════════════

 [T0-1] dvfs_ctrl.cpp 新增能耗计数器接口              ✅ 已完成
   产出: dvfs_get_energy_mj() 函数 (封装 nvmlDeviceGetTotalEnergyConsumption)
   文件: benchmark/test_motivation/dvfs/dvfs_ctrl.cpp

 [T0-2] dvfs.py 新增 get_energy_mj() Python 接口      ✅ 已完成
   产出: DVFSController.get_energy_mj() 方法
   文件: python/sglang/srt/layers/dvfs.py

 [T0-3] 编写 utils_profiling.py 公共工具               ✅ 已完成
   产出: EnergyMeter, load_single_decoder_layer(), run_profiling_sweep()
   文件: benchmark/test_motivation/utils_profiling.py

 [T0-4] 验证单层独立调用可行性                         ✅ 已完成

═══════════════════════════════════════════════════════════════════
 Phase 1: Profiling 数据采集 (~1 周)                  ✅ 大部分完成
═══════════════════════════════════════════════════════════════════

 [T1-1] 编写 bench_prefill_af.py                      ✅ 已完成
   文件: benchmark/test_motivation/bench_prefill_af.py

 [T1-2] 运行 Prefill Profiling                        ✅ 已完成（v1 修正版）
   产出: prefill_data_v1.txt (906 行, 修正 F 测量, 含 A_energy_mj/F_energy_mj)
     tp  input_len  gpu_clock  batch_size  A  F  TTFT_ms  (A+F)*64_ms  A_energy_mj  F_energy_mj
   频率: {210, 450, 690, 930, 1170, 1410}
   bs: {1,2,4,8,16,32,64,128}  il: {128..40000}
   注: 旧 prefill_data.txt tp=8 F 延迟不随频率变化（测量 bug），v1 已修正

 [T1-3] 编写 bench_decode_af.py                       ✅ 已完成
   文件: benchmark/test_motivation/bench_decode_af.py

 [T1-4] 运行 Decode Profiling                         ✅ 已完成（v1 修正版）
   产出: decode_data_v1.txt (7530 行, 含 D_A_energy/D_F_energy)
     tp  input_len  output_len  gpu_clock  batch_size  A  F  TPOT_ms  (A+F)*64_ms  A_energy_mj  F_energy_mj
   il: {128,256,512,1024,2048,4096}  ol: {64,128,256,512,1024,2048,4096}  bs: {1..256}
   频率: {210, 450, 690, 930, 1170, 1410}

 [T1-5] 编写+运行 bench_af_comm.py                    ❌ 未完成 (需多 GPU 环境)
   产出: af_comm_overhead.txt

 [T1-6] 编写+运行 bench_idle_power.py                 ❌ 未完成
   产出: idle_power.txt (各频率下的空闲功耗基线)

 [T1-7] 数据验证 + 更新 Pareto 图                     ✅ 已完成
   产出: plot_decode_pareto.py, plot_prefill_pareto.py, figures/
   已用真实能耗数据生成 Pareto 图, 确认 AF grid Pareto 优于 unified DVFS

═══════════════════════════════════════════════════════════════════
 Phase 2: 模型构建与分析 (~1 周)
═══════════════════════════════════════════════════════════════════

 [T2-1] 能耗模型拟合                                  ⏱ 3h
   输入: [T1-2] [T1-4] 数据
   任务:
     - 拟合 energy_A(f, input_len, bs) 和 energy_F(f, input_len, bs)
     - 对比查找表 vs 回归 vs GBDT 的 MAPE
     - 分别为 Prefill/Decode × Attention/FFN 拟合
   产出: energy_model.py (模型代码) + 精度报告

 [T2-2] 吞吐量 Profiling                              ⏱ 1 天
   输入: [T1-2] [T1-4] 数据 + 模型实例
   任务: 测量每个 (tp, freq) 配置的最大可行 goodput
   方法: 递增 RPS 压测 或 模拟器二分搜索
   产出: throughput_table.txt (ILP 输入)

 [T2-3] Motivation 论证撰写                           ⏱ 1 天
   输入: [T1-7] Pareto 图 + decode_data_v1.txt + prefill_data_v1.txt
   已完成分析 (见 Section 二·五 ~ 二·九):
     ✅ BiScale U型陷阱量化 (Decode SLO×1.5 时 BiScale 比 Lex 多耗能 14~25%)
     ✅ AF vs 原始BiScale 系统级收益 (Decode 5~9%, Prefill SLO×1.05~1.1 窗口 4~12%)
     ✅ AF vs Lex 纯粹收益 (Decode 5~9% @SLO×1.0, 饱和到 2~4%)
     ✅ 配比优化收益分析 (纯配比不带来能耗收益，需与调频联合)
     ✅ 收益来源分解 (来源1: BiScale策略缺陷; 来源2: AF差异化本身)
     ✅ 两层递进结构分析 (严格SLO下AF差异化 + 宽松SLO下U型修复)
     ✅ 甜区多维度分析 (tp×bs×il×SLO 全景)
   已完成图表:
     ✅ Pareto 前沿对比图 → fig1_pareto_comparison.png (plot_fig1_pareto.py)
     ✅ 收益 vs SLO 全景曲线 → fig2_saving_vs_slo.png (plot_fig2_slo_curve.py)
     ✅ AFlex vs Lex 放大图 + P25~P75 band → fig2b_aflex_vs_Lex_zoom.png (plot_fig2b_Lex_only.py)
     ✅ 甜区热力图 (tp×bs) → fig3_sweetspot_heatmap.png (plot_fig3_heatmap.py)
   待完成任务:
     - [ ] 绝对能耗数字（mJ/token）→ 现有数据可做
     - [ ] 通信开销敏感性分析 → 需跑 bench_af_comm.py (需多 GPU)
     - [ ] 撰写 paper Section 2 (Background & Motivation) 正式文本
   分析脚本: analyze_slo_motivation.py, analyze_sweet_spot.py, analyze_decode_energy.py
   产出: motivation 文本 + 论文图表

═══════════════════════════════════════════════════════════════════
 Phase 3: 系统实现 (~2-3 周)
═══════════════════════════════════════════════════════════════════

 [T3-1] Tier 1: ILP 联合求解器                        ⏱ 3-4 天
   输入: [T2-1] [T2-2] 能耗模型 + 吞吐表
   任务:
     - 基于 PuLP 实现四池联合 ILP (PA/PF/DA/DF)
     - Profile 表加载与查询
     - Pareto 剪枝 + Warm Start + 负载分桶
     - 输入: 负载参数 (λ, N_active, 长度分布, SLO)
     - 输出: (k_PA, k_PF, k_DA, k_DF, tp_*, f̄_*)
   产出: tier1_solver.py
   验证: 用 [T1-2][T1-4] 的 Profile 数据, 验证 ILP 有可行解

 [T3-2] Tier 2: DVFS 控制器                           ⏱ 2-3 天
   输入: [T2-1] 能耗模型
   任务:
     - Prefill: (f_A, f_F) 联合搜索 (O(36) 全枚举, SLO slack-aware)
     - Decode: per-batch slack-aware + 惰性切频
     - SLO 违反回退机制 (升频到最高)
   产出: tier2_dvfs.py

 [T3-3] AFD + DVFS 集成                               ⏱ 3-4 天
   输入: [T3-2], SGLang AFD 代码 (已有)
   任务:
     - 在 AFD pipeline (model_forward_afd) 中集成 DVFS 控制
     - forward_afd_A 前后调用 lock_sm_clock(f_A)
     - forward_afd_F 前后调用 lock_sm_clock(f_F)
     - 惰性切频逻辑 (Decode 侧避免频繁切换)
   产出: 修改 afd_mixin.py / afd.py + 新增 dvfs_scheduler.py
   注意: SGLang AFD 引擎已存在 (afd.py + afd_mixin.py),
         不需要从零实现 AF 分离, 只需集成 DVFS 控制

 [T3-4] Tier 1 + Tier 2 + Monitoring 联调             ⏱ 2-3 天
   任务:
     - Tier 1 输出配置 → 启动 AFD 实例 → Tier 2 运行时调频
     - Monitoring 模块: SLO 违反率、A/F 利用率、负载变化检测
     - 触发 Tier 1 重规划的完整闭环
   产出: 端到端可运行的系统原型

═══════════════════════════════════════════════════════════════════
 Phase 4: 评估 (~1-2 周)
═══════════════════════════════════════════════════════════════════

 [T4-1] Trace 准备                                    ⏱ 1 天
   任务:
     - 获取 Azure LLM trace 或 ShareGPT/LMSYS 公开 trace
     - 提取: 到达率 λ(t)、输入/输出长度分布
     - 构造不同负载模式 (高/中/低, 长请求/短请求为主)

 [T4-2] Baseline 实现                                 ⏱ 2 天
   需要实现的对比系统:
     ① Max-Freq: 所有 GPU 最高频率 (DistServe 风格)
     ② Unified-DVFS: 同频给 A 和 F (统一调频)
     ③ PD-Only: PD 分离但无 AF 分离 (BiScale 方案)
     ④ Hierarchical: 分层优化 (先 P/D 再 A/F)
     ⑤ Fixed-Ratio: A/F=1:1 固定, 仅优化频率
     ⑥ Joint (本文): 联合优化 P/D+A/F 配比+频率

 [T4-3] 端到端实验                                    ⏱ 3-4 天
   实验矩阵:
     - 6 个系统 × 3+ 种负载模式 × 2 种 SLO 设置
   采集指标:
     - 总能耗 (mJ), 能效 (tokens/J)
     - TTFT P99, TPOT P99, SLO 满足率
     - GPU 频率分布, A/F 利用率
   产出: 评估结果表格 + 论文图表

 [T4-4] Ablation 实验                                 ⏱ 2 天
   实验:
     - Tier 1 alone vs Tier 2 alone vs Combined
     - 联合优化 vs 分层优化 (量化跨层耦合收益)
     - AF 分离的增量收益 (Joint vs PD-Only)
     - 不同频率档数的影响 (4档 vs 6档)
   产出: ablation 结果 + 分析

 [T4-5] 论文撰写                                     ⏱ 1-2 周
   基于上述所有产出撰写完整论文

═══════════════════════════════════════════════════════════════════
 总计: ~6-8 周
═══════════════════════════════════════════════════════════════════
```

### 依赖关系图

```
T0-1 → T0-2 → T0-3 ─┬→ T1-1 → T1-2 ──┐
                      │                 │
            T0-4 ─────┼→ T1-3 → T1-4 ──┼→ T1-7 → T2-1 → T3-1 ─┐
                      │                 │         T2-2 ─────────┤
                      └→ T1-5           │         T2-3          │
                         T1-6           │                       │
                                        └→ T2-1 → T3-2 → T3-3 → T3-4
                                                                   │
                                           T4-1 ──────────────────┤
                                           T4-2 ──────────────────┤
                                                                   ↓
                                                          T4-3 → T4-4 → T4-5
```

### 关键路径（更新后）

```
Phase 0/1 已完成 → T2-1 → T3-1 → T3-3 → T3-4 → T4-3 → T4-5
                   T2-2 ─────────┤
(已完成)           (建模)   (系统实现)              (评估)    (论文)
                    1周        2-3周                1-2周     1-2周
```

### 当前可立即开始的工作

```
可并行启动:
  [T2-1] 能耗模型拟合 (数据已就绪: decode_data_v1.txt + prefill_data_v1.txt)
  [T2-3] Motivation 论文文本撰写 (数据分析已完成, 图表已生成)
  [T4-1] Trace 准备 (下载 + 分析)
  [T1-5] AF 通信开销测量 (需多 GPU 环境)
```

---

## 八、保留工作

### 保留 1: 动态部署模式切换 (Paper 2)

> 在不同负载下动态选择部署模式 (全 AF 分离 / 仅 PD 分离 / 仅 AF 分离 / 全共置)，将部署模式本身作为能效优化的决策变量。不同负载场景适合不同模式:
>
> | 负载场景 | 最优部署模式 |
> |----------|------------|
> | 高负载、A/F 频率差异大 | 全分离 (PA+PF+DA+DF) |
> | 中负载、AF 通信开销大 | PD 分离 (P 池 + D 池) |
> | P/D 负载均衡 | AF 分离 (A 池 + F 池) |
> | 低负载 | 全共置 (统一池) |

### 保留 2: 分层优化方案 (Tier 1 替代设计)

> 将 Tier 1 的联合优化替换为分层优化，作为对比方案或简化实现:
>
> ```
> Level 0: P/D 资源划分 (~10-30 min)
>   • 依据: 长期负载趋势，匹配 Prefill/Decode 吞吐到工作负载
>   • 决策: n_P, n_D
>   • 方法: 复用 BiScale/DistServe 的 provisioning 方案
>
> Level 1: A/F 配比 + 基线频率 (~几分钟)
>   • 输入: n_P (来自 Level 0)
>   • 决策: n_PA, n_PF, f̄_PA, f̄_PF (使得 n_PA + n_PF = n_P)
>   • 同理: n_DA, n_DF, f̄_DA, f̄_DF (使得 n_DA + n_DF = n_D)
>   • 目标: 在给定 P/D 资源下最小化能耗
>
> Level 2: Per-iteration DVFS (同 Tier 2, 不变)
> ```
>
> **优点**: 分解降低求解复杂度，P/D 层可直接复用已有方案，层次清晰易理解。
>
> **缺点**: P/D 分配与 A/F 配比存在耦合 (资源竞争、能耗跨池转移)，分层分解可能丢失全局最优解。
>
> **用途**: 在 Evaluation 中作为 ablation baseline (Hierarchical)，量化联合优化相对于分层优化的增量收益。

### 保留 3: AF灵活共享SM分配
