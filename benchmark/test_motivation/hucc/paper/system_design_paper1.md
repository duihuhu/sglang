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
> **Decode 配置**: il: {128,256,512,1024,2048,4096}, ol: {64,128,256,512,1024,2048,4096}, bs: {1..256}
> **Prefill 配置**: il: {128,256,512,1024,2048,4096,8192,16384,32000,40000}, bs: {1..128}

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
| **第二个模型** | 泛化性验证（如 Llama-3-70B 或 DeepSeek-V3） | 需重新 Profiling | ⭐ 可选 |

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

> **已合并到 [system_design_unified.md](system_design_unified.md)**，本章不再维护。
>
> 包含：问题定义、两层控制架构、Tier 1 ILP 联合资源规划、Tier 2 算子级 DVFS、能耗模型、动态扩缩容（遗留）、端到端闭环。

---

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

1. **首次揭示 Attention 和 FFN 的频率敏感性异构**——并用 Profiling 数据量化 (Pareto 图)，包括 Decode-FFN 随 batch_size 从 memory-bound 转为 compute-bound 的动态特性
2. **AF 分离 + 差异化调频**——比 PD 级 DVFS 多出两个频率自由度，Pareto 前沿更优
3. **P/D + A/F 联合资源规划**——首次将 P/D 分配和 A/F 配比视为一个联合优化问题，揭示并利用跨层耦合效应（资源竞争、延迟耦合、吞吐耦合、显存耦合、pipeline bubble 耦合），发现分层分解无法达到的全局最优配置
4. **动态 A/F 配比**——首次将 A/F 负载不对称性作为资源分配的依据
5. **低切频开销 (P50 ~4.5ms, avg ~6ms) 使差异化调频成为可能**——Prefill per-request 调频 + Decode per-window 惰性调频; 比 throttLL'eM (200ms) 低一个数量级; 实测验证 SetGpuLockedClocks 升降频均立即生效

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
   - Tier 1: P/D + A/F 联合 ILP 资源规划 (四池联合优化, 含显存约束 + pipeline bubble 能耗)
   - Tier 2: Prefill per-request + Decode per-window 差异化调频
   - Microbatch pipeline (M ∈ {1,2,3}) 与 A/F 延迟均衡
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
| batch_size | {1, 16, 256(tp=8)} | {1} | **{1, 2, 4, 8, 16, 32, 64, 128}** |
| input_len | {128, 512, 4096} | {128..40000, 10 档} | **{128,256,512,1024,2048,4096,8192,16384,32000,40000}** |
| tp | {1, 2, 4, 8} | {1, 2, 4, 8} | {1, 2, 4, 8} |
| gpu_clock | {210, 540, 870, 1200} | {210, 540, 870, 1200} | **{210, 450, 690, 930, 1170, 1410}** |
| 功耗 | ❌ | ❌ | **✅ A_energy_mj, F_energy_mj** |
| 总配置数 | — | — | **906** (tp=1: 222, tp=2/4/8: 各 228) |

**采集矩阵**（已按实际采集更新）:

```
tp         ∈ {1, 2, 4, 8}                                                        → 4 值
input_len  ∈ {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32000, 40000}        → 10 值
             (tp=1 无 40000, 实际 9 值; tp=2/4/8 含 40000, 实际 10 值)
batch_size ∈ {1, 2, 4, 8, 16, 32, 64, 128}                                       → 8 值
gpu_clock  ∈ {210, 450, 690, 930, 1170, 1410}                                    → 6 值

实际采集配置数 = 906 个 (去除 OOM 等不可行组合)
  tp=1: 222 个, tp=2: 228 个, tp=4: 228 个, tp=8: 228 个
```

**每个配置的执行流程**:

```
对每个 (tp, input_len, batch_size, gpu_clock):
  1. 设置 TP 并行度 (需启动对应的模型实例)
  2. lock_sm_clock(gpu_clock)
  3. 构造输入: input_ids = random tokens, shape = (batch_size, input_len)
  4. 预热 10 次 forward (单层)
  5. e0 = get_energy_mj()  // NVML 能耗计数器
  6. 执行 50 次 forward (单层), 用 CUDA events 分别记录 t_A, t_F
  7. e1 = get_energy_mj()
  8. 记录:
     - A = median(t_A_list) (us)
     - F = median(t_F_list) (us)
     - A_energy_mj, F_energy_mj (NVML 硬件能耗计数器差值)
  9. unlock_sm_clock()
```

**A/F 能耗分离: 采用方案 B (独立执行 + NVML 能耗计数器)**

在 AF 分离架构下，A 和 F 在不同 GPU 上独立运行、独立设频，能耗模型为:
```
E = E_A(f_A) + E_F(f_F)
  = A_energy_mj(f_A) + F_energy_mj(f_F)
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

**方案 B 执行流程 (每个配置点, 已采用 NVML 能耗计数器)**:

```
Phase A: 纯 Attention 能耗采集
  1. lock_sm_clock(target_freq)
  2. 构造输入 hidden_states, positions, forward_batch
  3. 预热: 执行 self.self_attn(...) × N_warmup 次
  4. e0 = nvmlDeviceGetTotalEnergyConsumption()  // mJ
  5. 连续执行 self.self_attn(...) × N_repeat 次, CUDA events 计时
  6. e1 = nvmlDeviceGetTotalEnergyConsumption()  // mJ
     → A_energy_mj = (e1 - e0) / N_repeat, t_A = median(cuda_times)

Phase F: 纯 FFN 能耗采集
  7. 构造 FFN 输入 (Attention 的输出 shape)
  8. 预热: 执行 self.mlp(...) × N_warmup 次
  9. e2 = nvmlDeviceGetTotalEnergyConsumption()
  10. 连续执行 self.mlp(...) × N_repeat 次, CUDA events 计时
  11. e3 = nvmlDeviceGetTotalEnergyConsumption()
      → F_energy_mj = (e3 - e2) / N_repeat, t_F = median(cuda_times)
  12. unlock_sm_clock()

N_repeat ≥ 50, 能耗计数器精度远高于功耗采样 (无采样对齐问题)
```

**输出格式**: `prefill_data_v1.txt` (Tab 分隔)

```
tp  input_len  gpu_clock  batch_size  A  F  TTFT_ms  (A+F)*64_ms  A_energy_mj  F_energy_mj
```

**时间估算**:

```
实际采集: 906 个配置, 每个 TP 约 222-228 个
每个配置: ~10s (预热 + 50次执行 + 能耗采集)
每个 TP 需要重新启动模型: ~2-5 min (模型加载)
总时间: ~3-4 小时 (已完成)
```

---

#### [P0-2] Decode A/F 延迟 + 功耗 Profiling

**采集矩阵**（已按实际采集更新）:

```
tp         ∈ {1, 2, 4, 8}                                → 4 值
input_len  ∈ {128, 256, 512, 1024, 2048, 4096}            → 6 值
output_len ∈ {64, 128, 256, 512, 1024, 2048, 4096}        → 7 值
batch_size ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256}           → 9 值
             (tp=1 无 bs=256, 实际 8 值; tp=2/4/8 含 bs=256, 实际 9 值)
gpu_clock  ∈ {210, 450, 690, 930, 1170, 1410}              → 6 值

实际采集配置数 = 7530 个 (去除 OOM 等不可行组合)
  tp=1: 1272 个, tp=2: 1974 个, tp=4: 2184 个, tp=8: 2100 个
```

**Decode 特殊处理 (方案 B: 独立 A/F 功耗)**:

```
对每个 (tp, input_len, output_len, batch_size, gpu_clock):

  准备阶段:
    1. 先执行 Prefill (input_len tokens) 填充 KV cache
    2. 执行 output_len 步 Decode 生成 KV cache 到目标长度
       (此时 KV cache 包含 input_len + output_len 个 token)

  数据含义: 测量的是生成第 output_len 个 output token 时的单步 Decode 延迟和能耗。
            此时 KV cache 中有 (input_len + output_len - 1) 个 token 的 KV 条目
            (input_len 个来自 Prefill, output_len - 1 个来自前序 Decode 步)。
            output_len 越大 → KV cache 越大 → Attention 访存量越大 → 延迟越高。

  Phase A: 纯 Decode-Attention 能耗
    3. lock_sm_clock(gpu_clock)
    4. e0 = get_energy_mj()
    5. 连续执行 self.self_attn(...) × N_repeat 次 (batch_size 个请求同时 decode)
    6. e1 = get_energy_mj()
       → A_energy_mj = (e1 - e0) / N_repeat, A = median(cuda_times)

  Phase F: 纯 Decode-FFN 能耗
    7. e2 = get_energy_mj()
    8. 连续执行 self.mlp(...) × N_repeat 次
    9. e3 = get_energy_mj()
       → F_energy_mj = (e3 - e2) / N_repeat, F = median(cuda_times)
    10. unlock_sm_clock()
```

**输出格式**: `decode_data_v1.txt` (Tab 分隔)

```
tp  input_len  output_len  gpu_clock  batch_size  A  F  TPOT_ms  (A+F)*64_ms  A_energy_mj  F_energy_mj
```

**时间估算**: 实际采集 7530 个配置, ~3-4 小时 (已完成)

---

#### [P0-3] AF 通信开销测量 (t_AF_comm)

**目的**: 测量 AFD 跨节点通信延迟的三个组成部分。

**背景**: AFD 跨节点传输的 tensor 是 all-reduce 后的**完整** hidden_states `(bs, seq_len, H)`，
所有 TP rank 持有相同数据。因此跨节点只需 rank 0 单连接发送，其余 rank 通过 NVLink broadcast 获取。
TP 度不影响跨节点数据量（恒为 2NH/层），只影响节点内分发。

**脚本**: `benchmark/test_motivation/bench_af_comm.py` ✅ 已就绪

**三种测量模式**:

```
Mode 1 (p2p):      单 rank P2P 传输 (模拟 ZMQ/UCX 跨节点)
                    GPU 0 → GPU 1, tensor = (bs, seq_len, H) 全量
                    torchrun --nproc_per_node=2 bench_af_comm.py --mode p2p

Mode 2 (broadcast): NVLink broadcast (模拟节点内分发)
                    GPU 0 → GPU 0..N-1, tensor = (bs, seq_len, H) 全量
                    torchrun --nproc_per_node=4 bench_af_comm.py --mode broadcast

Mode 3 (e2e):      端到端 = P2P + broadcast (模拟完整 AFD 通信)
                    Attn[0] → FFN[N/2] (P2P) → FFN[N/2..N-1] (broadcast)
                    torchrun --nproc_per_node=8 bench_af_comm.py --mode e2e
```

**测量维度**:

```
hidden_size = 5120 (Qwen3-32B, 完整 H, 不除以 TP)
batch_size  ∈ {1, 4, 16, 64, 128, 256}
seq_len     ∈ {1, 128, 512, 1024, 2048, 4096, 8192}
  注: Decode 时 seq_len=1 (仅传 1 token 的 hidden)
      Prefill 时 seq_len = input_len
```

**输出格式**: `af_comm_{mode}.txt`

```
mode  n_gpus  batch_size  seq_len  data_bytes  latency_us  bandwidth_gbps
```

**时间估算**: 每种 mode ~15 分钟, 共 ~45 分钟

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

# 加载单层 DecoderLayer (而非整个模型, 节省显存)
layer = load_single_decoder_layer(model_path, layer_id=0, tp=tp, device='cuda:0')
ctrl = DVFSController(device_index=0)
energy_meter = EnergyMeter(ctrl)

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
    energy_meter.start()
    attn_times = []
    for _ in range(N_REPEAT):
        start_event.record()
        _ = layer.self_attn(positions=pos, hidden_states=hidden, forward_batch=fb)
        end_event.record()
        torch.cuda.synchronize()
        attn_times.append(start_event.elapsed_time(end_event) * 1000)  # us
    A_energy_mj = energy_meter.stop() / N_REPEAT
    t_A = median(attn_times)
    
    # --- FFN Profiling ---
    ffn_input = torch.randn(batch_size, input_len, hidden_size, device='cuda:0', dtype=torch.bfloat16)
    for _ in range(10):
        _ = layer.mlp(ffn_input)
    energy_meter.start()
    ffn_times = []
    for _ in range(N_REPEAT):
        start_event.record()
        _ = layer.mlp(ffn_input)
        end_event.record()
        torch.cuda.synchronize()
        ffn_times.append(start_event.elapsed_time(end_event) * 1000)
    F_energy_mj = energy_meter.stop() / N_REPEAT
    t_F = median(ffn_times)
    
    ctrl.unlock_sm_clock()
    # 写入结果: tp, input_len, freq, batch_size, t_A, t_F, A_energy_mj, F_energy_mj
```

**关键挑战: 如何加载单层**

```
问题: SGLang 模型加载是整体的 (所有层 + embedding + lm_head)
      Qwen3-32B 整体加载需要大量显存

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
对于 tp=1/2/4, Qwen3-32B 模型可以放下。
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
| `utils_profiling.py` | 公共工具 (NVML 能耗计数器, CUDA 计时, 结果输出) | DVFSController |

**公共工具 `utils_profiling.py` 包含**:

```python
class EnergyMeter:
    """NVML 硬件能耗计数器 (via DVFSController.get_energy_mj())"""
    def __init__(self, ctrl: DVFSController): ...
    def start(self): ...  # 记录 e0
    def stop(self) -> float: ...  # 返回 energy_mj = e1 - e0

class CUDATimer:
    """CUDA events 计时器, 自动管理 start/end events"""
    def __init__(self): ...
    def start(self): ...
    def stop(self) -> float: ...  # 返回 us

def load_single_decoder_layer(model_path, layer_id, tp, device):
    """加载单个 DecoderLayer 的权重到指定设备"""
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
  └── 需要: 模型加载 (Qwen3-32B), A/F 分离计时方案确认

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

#### [P1-1] 能耗模型验证与拟合 ✅ 已完成

```
结果 (5-fold CV MAPE%):

  能耗模型:
    子模型          LUT(KNN)   LinearReg    GBDT      最佳
    Prefill_A       55.7%      17.6%        23.5%     LinearReg ✓
    Prefill_F       57.4%      13.8%        21.9%     LinearReg ✓
    Decode_A         7.9%      11.3%         3.7%     GBDT ✓
    Decode_F         3.3%      10.6%         2.0%     GBDT ✓

  延迟模型:
    Prefill_A_lat   50.7%      17.4%        74.3%     LinearReg ✓
    Prefill_F_lat   57.0%       9.2%        63.7%     LinearReg ✓
    Decode_A_lat     5.2%       9.4%         3.0%     GBDT ✓
    Decode_F_lat     3.0%       7.8%         1.8%     GBDT ✓

  结论:
    - Decode: GBDT 最优 (能耗 2-4%, 延迟 2-3%), 数据量大 (7530 行)
    - Prefill: LinearReg 最优 (能耗 14-18%, 延迟 9-17%), 数据量小 (906 行)
    - LUT train MAPE=0% (精确匹配), CV 高是因为测试点不在网格上
    - 实际系统建议: LUT 精确匹配 + GBDT/LinearReg 插值兜底

  脚本: benchmark/test_motivation/energy_model.py
  产出: energy_models/ (pickle 模型 + cv_mape_report.tsv)
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
   产出: EnergyMeter, CUDATimer, load_single_decoder_layer(), run_profiling_sweep()
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
   bs: {1,2,4,8,16,32,64,128}  il: {128,256,512,1024,2048,4096,8192,16384,32000,40000}
   tp=1: 222 配置 (无 il=40000), tp=2/4/8: 各 228 配置
   注: 旧 prefill_data.txt tp=8 F 延迟不随频率变化（测量 bug），v1 已修正

 [T1-3] 编写 bench_decode_af.py                       ✅ 已完成
   文件: benchmark/test_motivation/bench_decode_af.py

 [T1-4] 运行 Decode Profiling                         ✅ 已完成（v1 修正版）
   产出: decode_data_v1.txt (7530 行, 含 D_A_energy/D_F_energy)
     tp  input_len  output_len  gpu_clock  batch_size  A  F  TPOT_ms  (A+F)*64_ms  A_energy_mj  F_energy_mj
   il: {128,256,512,1024,2048,4096}  ol: {64,128,256,512,1024,2048,4096}  bs: {1..256}
   tp=1: 1272 配置 (bs 最大 128), tp=2: 1974, tp=4: 2184, tp=8: 2100 (bs 最大 256)
   频率: {210, 450, 690, 930, 1170, 1410}

 [T1-5] 编写+运行 bench_af_comm.py                    ⏳ 脚本已就绪 (需 2/4/8 GPU 环境运行)
   脚本: benchmark/test_motivation/bench_af_comm.py
   三种模式:
     p2p:       rank0→rank1 单 P2P (模拟 ZMQ/UCX 跨节点), 需 2 GPU
     broadcast: rank0→all NVLink broadcast (模拟节点内分发), 需 4+ GPU
     e2e:       P2P + broadcast 端到端 (模拟完整 AFD 通信), 需 4+ GPU (偶数)
   tensor shape: (bs, seq_len, H=5120) — 完整 hidden_states, 不除以 TP
   运行:
     torchrun --nproc_per_node=2 bench_af_comm.py --mode p2p
     torchrun --nproc_per_node=4 bench_af_comm.py --mode broadcast
     torchrun --nproc_per_node=8 bench_af_comm.py --mode e2e
   产出: af_comm_p2p.txt, af_comm_broadcast.txt, af_comm_e2e.txt

 [T1-6] 编写+运行 bench_idle_power.py                 ⏳ 脚本已就绪 (需 GPU 环境运行)
   脚本: benchmark/test_motivation/bench_idle_power.py
   更新: 支持多 GPU (--gpu 0 1 2 3), 多轮采样 (--rounds 3), 标准差统计
   运行: sudo python bench_idle_power.py [--gpu 0] [--rounds 3]
   产出: idle_power.txt (gpu_idx, gpu_clock, idle_power_W, power_std_W, energy_mj, duration_s)

 [T1-7] 数据验证 + 更新 Pareto 图                     ✅ 已完成
   产出: plot_decode_pareto.py, plot_prefill_pareto.py, figures/
   已用真实能耗数据生成 Pareto 图, 确认 AF grid Pareto 优于 unified DVFS

═══════════════════════════════════════════════════════════════════
 Phase 2: 模型构建与分析 (~1 周)
═══════════════════════════════════════════════════════════════════

 [T2-1] 能耗模型拟合                                  ✅ 已完成
   脚本: benchmark/test_motivation/energy_model.py
   结果: Decode 用 GBDT (能耗 2-4%, 延迟 2-3%), Prefill 用 LinearReg (能耗 14-18%, 延迟 9-17%)
   产出: energy_models/ (pickle 模型 + cv_mape_report.tsv)

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

 [T4-1] Trace 准备                                    ✅ 已完成
   脚本: benchmark/test_motivation/prepare_trace.py
   输入: AzurePublicDataset/data/ (DynamoLLM HPCA'25 Azure traces)
   产出: trace_processed/ 目录:
     - code_summary.json, conv_summary.json (统计摘要)
     - code_bucketed.csv, conv_bucketed.csv (il/ol 对齐 profiling 网格)
     - {code,conv}_{low,mid,high}.csv (0.5x/1x/2x 负载级别)
   Trace 特征:
     Code (Prefill-heavy): 8819 req, 2.6 RPS, il_p50=1469, ol_p50=13
     Conv (Decode-heavy):  19366 req, 5.5 RPS, il_p50=1020, ol_p50=129
   注: Code 有 14% 请求 il>4096 (超出 Decode profiling 范围, 已 snap 到 4096)

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
  [T2-1] 能耗模型拟合 (数据已就绪: decode_data_v1.txt 7530行 + prefill_data_v1.txt 906行, 含 A_energy_mj/F_energy_mj)
  [T2-3] Motivation 论文文本撰写 (数据分析已完成, 图表已生成)
  [T4-1] Trace 准备 (下载 + 分析)
  [T1-5] AF 通信开销测量 (需多 GPU 环境, 脚本 bench_af_comm.py 已就绪)
  [T1-6] 空闲功耗基线 (需 GPU 环境)
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

---

## 九、代码实现规划：基于 SGLang AFD 现有代码

> **已合并到 [system_design_unified.md](system_design_unified.md) 第 8-12 节**，本章不再维护。
>
> 包含：现有代码基础盘点、实现优先级与依赖、新增文件清单、工时明细、审阅意见处理记录。
