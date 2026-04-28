# Paper 1: AF 分离下的算子级动态调频能效优化系统设计

> **目标**: 在 PD 分离的基础上，进一步做 Attention/FFN (AF) 算子级分离，通过动态调整 A/F 实例配比和各自频率，在满足 SLO 的前提下实现系统能耗最低。

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

### 3.1 问题定义

**前提假设**: PD 分离已完成 (Prefill 池和 Decode 池独立部署)，在此基础上进一步做 AF 分离。

**四个算子池**:

| 池 | 计算特征 | 频率敏感性 |
|---|---------|-----------|
| **PA** (Prefill-Attn) | 偏 memory-bound | 中等 |
| **PF** (Prefill-FFN) | compute-bound | **高** |
| **DA** (Decode-Attn) | 强 memory-bound | **低** |
| **DF** (Decode-FFN) | memory→compute 随 bs 变化 | **中→高** (bs 依赖) |

> **DF 频率敏感性说明**: Decode-FFN 的计算特征随 batch_size 动态变化。小 bs (1~8) 时 FFN 的 arithmetic intensity 低，偏 memory-bound，频率敏感性中等；大 bs (64~256) 时 FFN 的矩阵乘法充分利用 Tensor Core，转为 compute-bound，频率敏感性显著升高。Tier 2 调频需感知当前 bs 选择 f_DF。

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
│  │  约束: k_P×(tp_PA+tp_PF) + k_D×(tp_DA+tp_DF) ≤ G_total              │  │
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
  Level 1: 在 n_P=8 内优化 → k_P=2 对 (tp_PA=1, tp_PF=3 → 每对 4 GPU, 共 8)
           在 n_D=8 内优化 → k_D=2 对 (tp_DA=2, tp_DF=2 → 每对 4 GPU, 共 8)

联合优化可能发现:
  k_P=1 对 (tp_PA=2, tp_PF=4 → 6 GPU), k_D=2 对 (tp_DA=1, tp_DF=4 → 每对 5, 共 10)
  → Decode-FFN 在低频下延迟增加大，多给它 GPU (tp_DF=4) 可用更低频率
  → 虽然 Prefill 只有 1 对 (6 GPU)，但高 TP 下单实例吞吐足够
  → 这种跨层优化只有联合方案能发现
```

**联合优化的可行性**: 搜索空间虽大于分层，但实际可控:
- TP 度候选有限: {1, 2, 4, 8}
- 频率候选有限: {210, 450, 690, 930, 1170, 1410}
- 配对约束使独立变量从 4 个 k 降为 2 个 (k_P, k_D)
- 副本数受 GPU 总量约束: k_P × (tp_PA + tp_PF) + k_D × (tp_DA + tp_DF) ≤ G
- 大量组合因违反 SLO 或超出资源可提前剪枝
- ILP 求解器 (Gurobi/CPLEX/PuLP) 在此规模下通常秒级可解

#### 3.3.2 四池联合 ILP 优化公式

**目标函数**: 最小化四个算子池的总能耗（含 pipeline bubble 能耗）

```
minimize:
  Σ_{c_pair ∈ {P, D}} k_{c_pair} × [
    E_A(tp_{c_A}, f̄_{c_A}, workload) + E_F(tp_{c_F}, f̄_{c_F}, workload)
    + E_bubble(tp_{c_A}, tp_{c_F}, f̄_{c_A}, f̄_{c_F}, workload, M)
  ]

  其中:
  - c_pair ∈ {P, D}, 对应 Prefill 对 (PA,PF) 和 Decode 对 (DA,DF)
  - E_A, E_F = A/F 单实例在给定配置下的计算能耗 (来自 Profile 表, mJ)
  - E_bubble = pipeline 气泡能耗, 快的一方等待慢的一方时的空闲功耗:
      E_bubble = P_idle(f_fast) × |t_A - t_F| × (M-1)/M
    其中 P_idle(f) 为频率 f 下的 GPU 空闲功耗 (来自 idle_power.txt)
    M = microbatch 数 (M ∈ {1, 2, 3})
    当 M=1 时 (M-1)/M=0, 无 pipeline 故无 bubble (但延迟为 t_A+t_F)
    当 M>1 时, bubble 时间 = |t_A - t_F| × (M-1)/M

注: E_bubble 在 A/F 延迟接近时很小, 但当 F/A 比达到 3~4x 时
    (如 tp=4, il=1024, bs=4: F/A=4.09x), bubble 能耗不可忽略。
    ILP 通过 E_bubble 项自动倾向选择 A/F 延迟更均衡的配置。
```

**约束集合**:

```
// ===== 资源约束 =====
(1)  k_PA × (tp_PA + tp_PF) + k_DA × (tp_DA + tp_DF) ≤ G
     // 每对 AF 实例占用 tp_A + tp_F 个 GPU (因配对约束 k_PA=k_PF, k_DA=k_DF)

// ===== TP 整数约束 =====
(2)  tp_c ∈ {1, 2, 4, 8}                            // TP 度候选集
     // 受 num_kv_heads 整除约束: Qwen3-32B num_kv_heads=8, 故 tp ∈ {1,2,4,8}
     // FFN 受 intermediate_size 整除约束: 18944 = 2^9 × 37, {1,2,4,8} 均可整除
(3)  k_c ∈ Z⁺                                        // 正整数副本数

// ===== Microbatch 数 =====
(2.5) M ∈ {1, 2, 3}                                  // microbatch pipeline 深度
     // M=1: 无 pipeline, A/F 串行, 延迟 = t_A + t_F + t_comm, 无 bubble
     // M=2: 2 级 pipeline, 延迟 ≈ max(t_A, t_F) + (t_A+t_F)/(2M) + t_comm/M
     // M=3: 3 级 pipeline, 延迟进一步降低, 但每个 microbatch 的 bs 变为 bs/3
     //       → GPU 利用率下降 (尤其 FFN compute intensity 降低)
     //       → 通信次数从 1 次变为 M 次
     // 实践中 M=2 是常用选择: pipeline 收益显著且 bs 不至于过小
     // M 作为系统配置参数, 不参与 ILP 优化 (固定为 afd_micro_batch 参数)

// ===== 频率选择 =====
(4)  f̄_c ∈ {210, 450, 690, 930, 1170, 1410}             // 离散频率候选

// ===== 延迟 SLO 约束 (P/D 和 A/F 耦合的关键) =====
//
// AFD 使用 microbatch pipeline: 一个 batch 被切成 M 个 microbatch,
// A 和 F 交替执行形成流水线, 通信与计算重叠。
// 单层延迟 = pipeline 启动 + 稳态 + 排空:
//   t_layer = max(t_A, t_F) + t_A/M + t_F/M + t_AF_comm × 2/M
//   简化 (M 较大时): t_layer ≈ max(t_A, t_F) + t_AF_comm/M
//   保守上界 (M=1, 无 pipeline): t_layer = t_A + t_F + t_AF_comm
//
// 下面使用保守上界, 确保 SLO 在最坏情况下也满足。
// 实际系统中 M>1 时有额外 slack, Tier 2 可利用这部分 slack 进一步降频。

(5)  ∀ 请求类型 r ∈ R_prefill:
     t_PA(r, tp_PA, f̄_PA) + t_PF(r, tp_PF, f̄_PF) + t_AF_comm ≤ TTFT_SLO / L
     // L = 模型层数 (Qwen3-32B: L=64), 单层延迟约束 (保守上界, M=1)
     // 若 M>1: 实际约束为 max(t_PA, t_PF) + t_AF_comm/M ≤ TTFT_SLO / L
     //         Tier 2 运行时使用精确公式

(6)  ∀ 请求类型 r ∈ R_decode:
     t_DA(r, tp_DA, f̄_DA) + t_DF(r, tp_DF, f̄_DF) + t_AF_comm ≤ TPOT_SLO
     // 同上, 保守上界; Tier 2 运行时使用 max(t_DA, t_DF) + t_AF_comm/M

// ===== 吞吐容量约束 (P/D 耦合, pipeline 感知) =====
// 在 AF pipeline 中, A 和 F 是 1:1 配对的, 一对 AF 的吞吐量受限于较慢的一方。
// 定义 pipeline 感知吞吐量:
//   Thpt_pair_P(tp_PA, tp_PF, f̄_PA, f̄_PF, M) = 1 / t_layer_P
//   其中 t_layer_P = max(t_PA, t_PF) + (t_PA + t_PF) / (2M) + t_AF_comm / M  (M>1)
//                  = t_PA + t_PF + t_AF_comm                                   (M=1)
// 类似地定义 Thpt_pair_D。
//
// 注: 这里的吞吐量是 pipeline 对的联合吞吐量, 而非 A/F 各自独立的吞吐量。
//     Profile 表提供 t_PA, t_PF 各自的延迟, pipeline 吞吐量由公式计算。

(7)  k_P × Thpt_pair_P(tp_PA, tp_PF, f̄_PA, f̄_PF, M) ≥ (1+α) × λ
     // Prefill pipeline 对的总吞吐需匹配到达率 λ

(8)  k_D × Thpt_pair_D(tp_DA, tp_DF, f̄_DA, f̄_DF, M) ≥ (1+α) × N_active
     // Decode pipeline 对的总吞吐需匹配活跃 decode 请求数 N_active

// ===== A/F 流水线平衡约束 =====
// pipeline 效率取决于 A/F 延迟的均衡程度。当 |t_A - t_F| 大时:
//   - pipeline bubble 增大 → 能耗浪费 (已在目标函数 E_bubble 中建模)
//   - pipeline 吞吐量下降 → 需要更多副本 (已在约束 7/8 中通过 Thpt_pair 建模)
// 以下约束作为辅助剪枝, 排除 A/F 延迟严重失衡的配置:
(9)  |t_A(r_typ, tp_PA, f̄_PA) - t_F(r_typ, tp_PF, f̄_PF)| ≤ β × max(t_A, t_F)
(10) |t_A(r_typ, tp_DA, f̄_DA) - t_F(r_typ, tp_DF, f̄_DF)| ≤ β × max(t_A, t_F)
     // β ∈ [0.5, 0.8]: 允许一定程度的不均衡 (完全均衡不现实)
     // r_typ: 代表性请求类型 (负载分桶的中位数桶)
     // 此约束主要用于剪枝, 减少 ILP 搜索空间; 精确的 bubble 代价由目标函数处理

// ===== A/F 配对部署约束 (AFD 架构要求) =====
// AFD 中每个 A 实例必须有一个 F 实例对接 (1:1 配对)。
// 异构 TP 时 (tp_A ≠ tp_F), 一对 AF 占用 tp_A + tp_F 个 GPU。
// 因此 A 和 F 的实例副本数必须相等:
(13) k_PA = k_PF                                       // Prefill 侧 A/F 配对
(14) k_DA = k_DF                                       // Decode 侧 A/F 配对

// 资源约束 (1) 相应改写为:
//   k_PA × (tp_PA + tp_PF) + k_DA × (tp_DA + tp_DF) ≤ G
// 即每对 AF 实例占用 tp_A + tp_F 个 GPU

// ===== 显存约束 =====
// A 实例和 F 实例的显存需求不同:
//   - A 实例: Attention 权重 + KV cache (随 batch_size × seq_len 增长)
//   - F 实例: FFN 权重 (intermediate_size 大, 但无 KV cache)
// 不同 TP 下每卡显存占用不同 (权重按 TP 切分)。
// 显存约束确保 ILP 不会选出运行时 OOM 的配置。

(15) Mem_A(tp_{c_A}, bs_max, seq_max) ≤ GPU_MEM / tp_{c_A}    // 每卡 A 显存
     // Mem_A = W_attn/tp + KV_cache(bs_max, seq_max, tp)
     // W_attn: Attention 权重 (Q/K/V/O proj), 按 TP 切分
     // KV_cache: 2 × L × num_kv_heads/tp × head_dim × bs_max × seq_max × dtype_size
     // bs_max, seq_max: 该池预期的最大 batch_size 和最大序列长度 (来自负载分桶)

(16) Mem_F(tp_{c_F}) ≤ GPU_MEM / tp_{c_F}                     // 每卡 F 显存
     // Mem_F = W_ffn/tp + activation_buffer
     // W_ffn: FFN 权重 (gate/up/down proj), 按 TP 切分
     // activation_buffer: hidden_states 中间激活 (bs × seq_len × H)
     // F 实例无 KV cache, 显存压力主要来自权重

// 注: Profile 表中已排除 OOM 配置, 但运行时 batch_size 可能超过 Profile 时的值。
//     显存约束使用保守估计 (bs_max 取负载分桶的 P99), 确保运行时不 OOM。
//     Qwen3-32B 显存估算:
//       Attention 权重/层: 4 × H × (H/num_heads × num_kv_heads) × 2B ≈ 52.4 MB (bf16)
//       FFN 权重/层: 3 × H × intermediate_size × 2B ≈ 582 MB (bf16)
//       KV cache/层/token: 2 × num_kv_heads × head_dim × 2B ≈ 2.56 KB (bf16)
//       → bs=256, seq=4096: KV cache ≈ 256 × 4096 × 2.56KB × 64层 ≈ 164 GB (需 TP 切分)
```

> **注**: 约束 (11)(12) 使得 ILP 搜索空间大幅缩小——原来 4 个独立的 k 变量
> 变为 2 个 (k_P = k_PA = k_PF, k_D = k_DA = k_DF)。
> 约束 (9)(10) 作为辅助剪枝，排除 A/F 延迟严重失衡的配置。
> 约束 (15)(16) 确保选出的配置在运行时不会 OOM。

**变量汇总** (配对约束后):

| 变量 | 含义 | 类型 | 范围 |
|------|------|------|------|
| k_P (= k_PA = k_PF) | Prefill AF 对数 | 正整数 | [1, G/(tp_PA+tp_PF)] |
| k_D (= k_DA = k_DF) | Decode AF 对数 | 正整数 | [1, G/(tp_DA+tp_DF)] |
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
| M | microbatch 数 | AFD 配置 (M ∈ {1, 2, 3}) |
| α | 容量裕度 | 超参 (如 0.1-0.2) |
| β | A/F 延迟平衡松弛度 | 超参 (如 0.5-0.8) |
| t_AF_comm | A→F 中间激活传输延迟 | Profiling 测量 |
| GPU_MEM | 单卡显存容量 | 硬件规格 (A800: 80GB) |
| P_idle(f) | 频率 f 下 GPU 空闲功耗 | idle_power.txt |
| bs_max, seq_max | 各池预期最大 batch_size/seq_len | 负载分桶 P99 |

#### 3.3.3 联合优化为什么 P/D 和 A/F 耦合

从约束可见，P/D 和 A/F 之间存在多层耦合:

1. **资源竞争**: k_P × (tp_PA + tp_PF) + k_D × (tp_DA + tp_DF) ≤ G，给 Prefill 多一对 AF，Decode 就少 GPU
2. **延迟耦合**: PA 和 PF 的频率/TP 联合决定能否满足 TTFT SLO (约束 5)
3. **吞吐耦合**: pipeline 对的吞吐量取决于 A/F 中较慢的一方 (约束 7-8)，TP/频率选择同时影响 A 和 F 的延迟
4. **能耗-频率权衡的跨池传递**: 降低 f̄_DA 省能耗但可能需要更多 DA 副本 → 挤占 PF 的 GPU → PF 需要更高频率 → 能耗在 DA 和 PF 之间转移
5. **显存耦合**: A 实例的 KV cache 显存随 bs/seq 增长，限制了 DA 的最大 batch_size → 影响 Decode 吞吐 → 可能需要更多 DA 副本 → 挤占其他池的 GPU
6. **Pipeline bubble 耦合**: A/F 延迟差距越大，bubble 能耗越高 (目标函数 E_bubble)，ILP 自动倾向选择 A/F 延迟更均衡的 TP/频率组合

**联合优化捕捉这些耦合，找到全局最优的资源-频率分配。**

#### 3.3.4 ILP 求解加速

为控制求解时间，采用以下策略:

```
1. Profile 表预计算 (含 TP 硬件约束剪枝)
   - 离线穷举所有 (tp, freq, workload_bin) 组合
   - TP 候选受 num_kv_heads 整除约束: 只 Profile tp ∈ {1,2,4,8}
   - 不合法的 (tp, freq) 组合不生成 Profile 条目 → 剪枝在建模阶段完成
   - 每个组合记录 (latency_A, latency_F, energy_A_mj, energy_F_mj)
   - ILP 中直接查表，无需在线推理

2. 对称性与支配性剪枝
   - 若配置 A Pareto 支配配置 B (延迟更低且能耗更低)，剪掉 B
   - 每个池的候选配置从 |TP| × |F| = 4×6=24 种降至 Pareto 前沿上的少数几种

3. Warm Start
   - 用上一个规划窗口的解作为初始可行解
   - ILP 求解器可在此基础上快速改进

4. 负载分桶 (代表值选择策略)
   - 将连续的请求长度分布离散化为若干代表性桶
   - 约束 (5)(6) 只针对桶的代表值，而非每个请求
   - 代表值选择: 使用桶内 P90 (而非均值或最大值)
     → 均值会低估长尾请求的延迟, 导致 SLO 违反
     → 最大值过于保守, 浪费资源
     → P90 在保守性和资源效率之间取得平衡
   - Prefill 分桶: 按 input_len 分桶 (如 [0,512), [512,2048), [2048,8192), [8192,+∞))
   - Decode 分桶: 按 (input_len, output_len) 联合分桶
   - 显存约束 (15)(16) 中的 bs_max, seq_max 取桶内 P99
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

**过渡期处理 (重规划期间的请求连续性)**:

```
场景 1: 仅频率变化 (tp 不变, 仅调整 f̄)
  → 最轻量: 直接切频 (~6ms), 无需迁移, 请求不中断
  → Tier 2 在下一个决策窗口自动使用新基线频率

场景 2: TP 变化但池角色不变 (如 tp_DA: 2→4)
  → 需要重新分配 GPU 并重新加载权重 (按新 TP 切分)
  → Decode 侧 KV cache 迁移:
    - KV cache 按 num_kv_heads 在 TP rank 间分布
    - TP 变化 → head 分布变化 → 需要 all-to-all 重分布
    - 迁移策略: drain-then-switch
      (1) 停止接收新 Decode 请求到旧实例
      (2) 等待旧实例上的活跃请求完成 (或达到超时)
      (3) 启动新 TP 配置的实例 (shadow instancing 已预创建)
      (4) 新请求路由到新实例
    - 不做在线 KV cache 迁移 (复杂度高, 收益低):
      旧实例上的请求自然完成, 新请求在新实例上从头开始
    - 过渡期时长: 取决于旧实例上最长请求的剩余生成长度
      典型值: 数秒到数十秒 (Decode 请求的剩余 output tokens)

场景 3: P/D 资源重分配 (如 n_P: 8→6, n_D: 8→10)
  → 涉及 GPU 角色切换: 部分 GPU 从 Prefill 角色转为 Decode 角色
  → 需要加载不同的权重子集 (A 权重 vs F 权重)
  → 策略: 先缩 Prefill (drain), 再扩 Decode (shadow instancing)
  → 过渡期 Prefill 容量暂时下降, Tier 2 自动升频补偿

紧急回退: 若重规划期间 SLO 违反率飙升
  → Tier 2 立即升频到最高 (f_max = 1410MHz)
  → 若升频仍不够 (资源不足), 启用准入控制: 排队新请求, 优先完成已有请求
```

#### 3.3.7 Profile 表构建

离线对每个池独立 profiling:

```
PA_Profile[tp, input_len, batch_size, freq] → (latency_PA, energy_PA_mj)
PF_Profile[tp, input_len, batch_size, freq] → (latency_PF, energy_PF_mj)
DA_Profile[tp, input_len, output_len, batch_size, freq] → (latency_DA, energy_DA_mj)
DF_Profile[tp, input_len, output_len, batch_size, freq] → (latency_DF, energy_DF_mj)
```

> 注: 实际数据中直接记录了 A_energy_mj / F_energy_mj (NVML 硬件能耗计数器),
> 功率可由 power = energy / latency 反推。throughput 需额外 Profiling (T2-2)。

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

### 3.4 Tier 2: 算子级 DVFS

A GPU 和 F GPU 独立选频，目标是找到满足 SLO 的最低能耗频率组合。
调频粒度因阶段而异: Prefill 为 per-request，Decode 为 per-window（频率决策窗口）。

> **与 AFD microbatch pipeline 的关系**:
> AFD 将一个 batch 切成 M 个 microbatch 形成流水线 (M ∈ {1, 2, 3})，A 和 F 交替执行。
> 单层实际延迟 ≈ max(t_A, t_F) + t_AF_comm/M（pipeline 稳态, M>1）。
> 当 M=1 时无 pipeline，延迟 = t_A + t_F + t_comm。
> Tier 2 频率选择使用精确的 pipeline 延迟公式（而非 Tier 1 的保守上界）。
>
> **M 的选择**: M=2 是默认推荐值。M=1 适用于小 batch (bs≤4, pipeline 收益小);
> M=3 适用于大 batch (bs≥64, 每个 microbatch 仍有足够的 compute intensity)。
> M 过大会导致每个 microbatch 的 bs 过小，GPU 利用率下降，且通信次数增加。

#### 3.4.1 Prefill 侧 DVFS

Prefill 突发性强，延迟敏感 (TTFT)，采用基于 SLO slack 的联合频率搜索。

**切频粒度**: per-request（而非 per-layer）。原因:
- 短序列 (il=128, bs=1) 单层延迟仅 ~0.3-1ms，切频 ~6ms 远大于单层延迟
- 长序列 (il=8192+) 单层延迟 ~10-50ms，per-layer 切频理论可行但收益有限
- 统一采用 per-request: 在请求开始前设一次频率，整个 Prefill 过程保持不变
- 切频开销 ~6ms 相对于 TTFT (通常 100ms~数秒) 占比很小 (<6%)

```
算法: Prefill AF-DVFS (per-request)

输入:
  当前 batch B_PA (已调度到 PA 的请求集合)
  每个请求 r 的 TTFT deadline: d_r
  PA/PF Profile 模型 M_PA, M_PF
  M = microbatch 数 (M ∈ {1, 2, 3})
  t_switch = 频率切换开销 (~6ms)

步骤:
  1. 对当前 batch, 计算 TTFT slack (扣除切频开销):
     slack = min_{r ∈ B} (d_r - elapsed_r) - t_switch
     // 注: remaining_layers × t_layer 假设所有层延迟相同
     // 实际上各层结构相同 (Transformer), 延迟差异 <1%, 近似合理
     // 若有 prefix caching, 已缓存层跳过 Attention, 需调整 remaining_layers

  2. 搜索 (f_PA, f_PF) 组合:
     candidates = {}
     for f_PA in [210, 450, 690, 930, 1170, 1410]:
       for f_PF in [210, 450, 690, 930, 1170, 1410]:
         t_PA = M_PA.predict_latency(batch, f_PA)
         t_PF = M_PF.predict_latency(batch, f_PF)
         // pipeline 延迟: A/F 并行执行, 通信被 microbatch 重叠
         t_layer = max(t_PA, t_PF) + t_AF_comm / M
         if t_layer × remaining_layers ≤ slack:
           e_PA = M_PA.predict_energy(batch, f_PA)
           e_PF = M_PF.predict_energy(batch, f_PF)
           e_bubble = P_idle × |t_PA - t_PF| × (M-1)/M × remaining_layers
           candidates.add((f_PA, f_PF, e_PA + e_PF + e_bubble))

  3. 选择能耗最低的可行组合:
     (f_PA*, f_PF*) = argmin_{(f_PA, f_PF) ∈ candidates} e

  4. 设置频率并执行 (整个 Prefill 请求保持该频率)

复杂度: O(|F|²) = O(36) — 常数时间，可忽略
```

**设计选择说明 (为什么不用 MPC)**:
- BiScale 的 MPC 是因为 Prefill 只有一个频率旋钮 (f_P)，需要在时间维度上优化未来 K 个 batch 的频率序列
- AF 分离提供了空间维度的额外自由度 (f_A vs f_F)，使得单步的 (f_A, f_F) 联合搜索就已经有足够大的节能空间
- 如需进一步优化，可扩展为 MPC

#### 3.4.2 Decode 侧 DVFS

Decode 使用 continuous batching，batch 组成每个 iteration 都可能变化（请求加入/退出）。
需要明确定义频率决策的时机和粒度。

**Batch 边界定义**: 在 continuous batching 中不存在天然的 "batch 边界"。
我们定义**频率决策窗口**为连续 W 个 iteration（W 由切频开销决定）:

```
算法: Decode AF-DVFS (频率决策窗口)

频率决策窗口 W 的确定:
  - 切频开销 ~6ms, Decode iteration ~1-2ms
  - 要求切频开销占比 < 10%: W × t_iter ≥ 10 × t_switch
  - 典型值: W = 60ms / 1ms = 60 iterations (即每 ~60ms 做一次频率决策)
  - 自适应: W = max(W_min, ceil(10 × t_switch / t_iter_avg))

触发频率重评估的条件 (满足任一即触发):
  cond_1: 距上次决策已过 W 个 iteration
  cond_2: batch_size 变化超过 30% (请求大量加入/退出)
  cond_3: SLO 违反检测 (当前 TPOT > SLO_TPOT × 0.9)

输入:
  当前 batch B_DA (活跃 decode 请求)
  TPOT budget: SLO_TPOT
  DA/DF Profile 模型 M_DA, M_DF
  M = microbatch 数 (M ∈ {1, 2, 3})
  当前频率: (f_DA_cur, f_DF_cur)

步骤:
  1. 获取当前 batch 特征: batch_size, avg_kv_blocks

  2. 按能耗升序搜索可行 (f_DA, f_DF):
     // 注: DF 的频率敏感性随 bs 变化 (小 bs 偏 memory-bound, 大 bs 偏 compute-bound)
     // Profile 模型 M_DF 已包含 bs 维度, 自动捕获这种变化
     for (f_DA, f_DF) in sorted_by_energy(all_combos):
       t_DA = M_DA.predict_latency(batch, f_DA)
       t_DF = M_DF.predict_latency(batch, f_DF)
       t_layer = max(t_DA, t_DF) + t_AF_comm / M
       if t_layer ≤ SLO_TPOT:
         (f_DA_new, f_DF_new) = (f_DA, f_DF)
         break

     // 若无可行组合, 使用最高频率
     if no feasible: (f_DA_new, f_DF_new) = (1410, 1410)

  3. 惰性切频 (避免频繁切换):
     // 只有当新频率的节能收益足以覆盖切频开销时才切换
     E_saved = (E_cur - E_new) × W_remaining  // 剩余窗口内的预期节能
     E_switch = P_idle × t_switch × N_gpu      // 切频期间的能耗开销
     if E_saved > γ × E_switch:                // γ ≥ 2, 要求至少 2 倍回本
       switch_freq(f_DA_new, f_DF_new)
     else:
       keep (f_DA_cur, f_DF_cur)

  注: DA 对频率非常不敏感 (强 memory-bound),
      实践中 f_DA 几乎总是最低频, 切频主要发生在 f_DF 上。
      大 bs 时 DF 转为 compute-bound, f_DF 的选择空间更大。
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

**AF 分离架构下**: A 和 F 在不同 GPU 上，各自维持各自的频率，**不存在 A/F 之间的切频开销**。切频开销仅在以下场景相关:

- **Tier 1 重规划**: 调整某个池的基线频率 → ~6ms 完全可接受 (T₁ 是分钟级)
- **Tier 2 Prefill**: per-request 切频, ~6ms 相对于 TTFT (100ms~数秒) 占比 <6%
- **Tier 2 Decode**: per-window 切频 (窗口 ~60ms), 6ms 开销占 ~10%, 通过惰性策略进一步降低

**对 Decode 的影响**:
- Decode iteration 典型耗时 ~1ms (bs=1) 到 ~2ms (bs=256)
- ~6ms 切频 + 1ms iteration = 7ms → per-iteration 切频开销过大, 不可行
- 采用 per-window 调频 (窗口 W ≈ 60 iterations ≈ 60ms), 6ms 开销占 ~10%
- 惰性切频策略 (3.4.2 节) 确保只在节能收益 > 2× 切频开销时才切换
- 多 GPU 串行切频: AF 分离下每个池的 TP 通常 1-4 卡, 切频 6-24ms, 仍在窗口内可接受

**对 Prefill 的影响**:
- 采用 per-request 切频: 在请求开始前设一次频率, 整个 Prefill 保持不变
- ~6ms 切频开销相对于 TTFT 占比很小
- 短序列 (il=128, bs=1): TTFT ≈ 60ms, 切频占 10% — 可接受
- 长序列 (il=8192+): TTFT ≈ 1-10s, 切频占 <1% — 可忽略

### 3.5 能耗模型

```
单次 iteration 能耗 (直接使用 NVML 硬件能耗计数器):

  M=1 模型 (单层, A/F 串行, 无 pipeline):
    E_iter = E_A(f_A, batch, ctx) + E_F(f_F, batch, ctx) + E_comm
    // 无 bubble: A 和 F 串行执行, 各自 GPU 在对方执行时空闲
    // 空闲 GPU 的能耗已包含在 E_A/E_F 的 Profile 测量中 (NVML 计数器)
    // 注: 此处 E_idle 不单独建模, 因为 Profile 时 GPU 也处于类似空闲状态

  M>1 模型 (M 个 microbatch, A/F pipeline 并行):
    E_iter = E_A(f_A, batch, ctx) + E_F(f_F, batch, ctx) + E_comm
           + E_pipeline_bubble
    // E_A, E_F 总量不变 (同样的计算量), 但 pipeline 使得:
    //   - 延迟降低: t_layer ≈ max(t_A, t_F) + t_comm/M
    //   - A GPU 在等 F 完成时处于空闲 (或反之), 产生 pipeline bubble 能耗
    //   - E_pipeline_bubble = P_idle(f_fast) × |t_A - t_F| × (M-1)/M
    //   - 当 t_A ≈ t_F 时 bubble 最小 (这也是 A/F 平衡约束的物理意义)
    //   - P_idle(f) 从 idle_power.txt 查询 (频率相关的空闲功耗)

  注: E_A, E_F 直接从 Profile 表查询 (mJ), 无需 power × latency 间接计算
      E_pipeline_bubble 可用 idle_power (T1-6) × bubble 时间估算
      DF 的能耗特性随 bs 变化: 小 bs 时 E_DF 对频率不敏感 (memory-bound),
      大 bs 时 E_DF 随频率显著变化 (compute-bound) — Profile 表已包含 bs 维度

AF 分离的 Pareto 优势:
  E_AF(f_A, f_F) ≤ E_unified(f)  当 f 使得延迟相同时

  原理: 当 A 是 memory-bound 时, 降 f_A 几乎不增加 t_A,
        但大幅降低 E_A (能耗随频率降低)。
        同时 f_F 可以保持高频以保证 t_F 不成为瓶颈。
        Decode 大 bs 时 DF 转为 compute-bound, f_DF 的降频空间缩小,
        但 DA 仍为强 memory-bound, f_DA 可大幅降频 → 差异化收益仍然存在。
```



## 九、代码实现规划：基于 SGLang AFD 现有代码

### 9.1 现有代码基础盘点

| 组件 | 状态 | 关键文件 |
|------|------|---------|
| AFD 核心 (A/F 分离执行) | ✅ 已有 | `afd.py`, `afd_mixin.py`, `afd_overlap.py` |
| AFD 调度器 | ✅ 已有 | `scheduler.py` (`event_loop_afd`), `scheduler_afd_mixin.py` |
| AFD 通信 (ZMQ/StepMesh/UCX) | ✅ 已有 | `afd.py` (3 种后端), `rdma_comm.py` |
| AFD 异构 TP (tp_A ≠ tp_F) | ✅ 已有 | `server_args.py` (`afd_attn_tp`, `afd_ffn_tp`) |
| AFD 权重过滤 | ✅ 已有 | `afd_mixin.py` (`AFDWeightFilter`) |
| DVFS 控制器 | ✅ 已有 | `dvfs.py` (`DVFSController/DVFSManager`), `dvfs_ctrl.cpp` |
| Profile 数据 | ✅ 已有 | `prefill_data_v1.txt` (906行), `decode_data_v1.txt` (7530行) |
| 能耗模型脚本 | ✅ 已有 | `energy_model.py` (待运行拟合) |
| Trace 数据 | ✅ 已有 | `prepare_trace.py`, Azure Code/Conv traces |
| **Tier 2 DVFS 集成** | ❌ 缺失 | DVFS 未接入推理流水线 |
| **Tier 1 ILP 求解器** | ❌ 缺失 | 联合资源规划未实现 |
| **动态扩缩容/重规划** | ❌ 缺失 | 无运行时 TP/配比切换 |
| **Monitoring 闭环** | ❌ 缺失 | 无 SLO 监控 → 触发重规划 |
| **权重缓存 (TP 快速切换)** | ❌ 缺失 | 无多 TP 权重预缓存 |

### 9.2 三大功能模块实现规划

---

#### 模块 A: AF 动态调频 (Tier 2 DVFS)

**目标**: 在 AFD 推理流水线中，A GPU 和 F GPU 各自独立调频，满足 SLO 下最小化能耗。

**依赖**: DVFSController (已有) + Profile 表 (已有) + 能耗模型 (T2-1 待拟合)

```
A-1. 能耗/延迟预测模型 (在线推理用)                    ⏱ 2 天
  输入: energy_model.py 拟合产出 (pickle 模型)
  任务: 封装为轻量在线查询接口
    - AFProfilePredictor 类: predict_latency(phase, op, tp, freq, bs, il, ol)
                             predict_energy(phase, op, tp, freq, bs, il, ol)
    - 支持查找表 + 线性插值 (首选) 或 GBDT
    - 加载 Profile 表到内存, O(1) 查询
  产出: python/sglang/srt/layers/af_profile_predictor.py

A-2. Tier 2 DVFS 控制器                                ⏱ 3 天
  任务: 实现 Prefill per-request + Decode per-window 调频策略
    - AFDVFSController 类:
      - select_freq_prefill(batch, slack, M) → (f_A, f_F)
      - select_freq_decode(batch, slo_tpot, M) → (f_A, f_F)
      - should_switch(f_new, f_cur, window_remaining) → bool  (惰性切频)
    - 内部调用 AFProfilePredictor 做 O(36) 全枚举
    - SLO 违反回退: 升频到 f_max
  产出: python/sglang/srt/layers/af_dvfs_controller.py

A-3. 集成到 AFD 推理流水线                              ⏱ 2 天
  任务: 在 model_forward_afd() 的 stage 执行前后插入调频
    - 修改 afd.py: model_forward_afd() 入口处调用 AFDVFSController
    - Prefill: 请求开始前 lock_sm_clock(f_A/f_F), 请求结束后 unlock
    - Decode: 每 W 个 iteration 检查一次, 惰性切频
    - 修改 scheduler.py: event_loop_afd 中传递 SLO deadline 信息
  修改文件: afd.py, scheduler.py (少量), 新增 af_dvfs_controller.py
```

---

#### 模块 B: Tier 1 ILP 联合资源规划

**目标**: 给定集群 GPU 总量和负载统计，联合决策 P/D 划分、A/F 配比、TP 度、基线频率。

**依赖**: Profile 表 + 吞吐量 Profiling (T2-2) + 能耗模型

```
B-1. Profile 表加载与查询模块                           ⏱ 1 天
  任务: 将 prefill_data_v1.txt / decode_data_v1.txt 加载为结构化查询接口
    - ProfileTable 类: query(phase, op, tp, freq, bs, il, ol) → (latency, energy)
    - Pareto 剪枝: 预计算每个 (phase, workload_bin) 的 Pareto 前沿配置
    - 显存估算: mem_attn(tp, bs_max, seq_max), mem_ffn(tp)
  产出: python/sglang/srt/energy/profile_table.py

B-2. ILP 求解器                                         ⏱ 3 天
  任务: 基于 PuLP 实现四池联合 ILP
    - Tier1Solver 类:
      - 输入: G, λ, N_active, R_prefill, R_decode, SLO, M, idle_power
      - 输出: (k_P, k_D, tp_PA, tp_PF, tp_DA, tp_DF, f̄_PA, f̄_PF, f̄_DA, f̄_DF)
    - 目标函数: Σ E_compute + E_bubble (含 pipeline bubble)
    - 约束: 资源(1) + TP(2) + 频率(4) + SLO(5,6) + 吞吐(7,8) + 平衡(9,10)
            + 配对(11,12) + 显存(15,16)
    - 枚举配置法: 预生成所有合法 (tp, freq) 组合, 用二元变量选择
    - Warm Start + Pareto 剪枝 + 负载分桶 (P90 代表值)
  产出: python/sglang/srt/energy/tier1_solver.py

B-3. 负载监控模块                                       ⏱ 1 天
  任务: 采集 Tier 1 所需的负载统计
    - WorkloadMonitor 类:
      - 滑动窗口统计: λ(t), N_active(t), il/ol 分布
      - 负载分桶: 按 il 分桶 (Prefill), 按 (il, ol) 联合分桶 (Decode)
      - 突变检测: KL 散度 / 均值偏移
    - 集成到 Scheduler 的 Monitoring 模块
  产出: python/sglang/srt/energy/workload_monitor.py
```

---

#### 模块 C: AF 弹性扩缩容 + 动态 TP 切换

**目标**: 运行时根据 Tier 1 ILP 输出，动态调整 A/F 实例数、TP 度，无需重启服务。

**核心挑战**: SGLang 当前 TP 在启动时固定，无运行时切换支持。

**关键洞察**: AF 分离后，A 实例只加载 Attention 权重 (~52MB/层 bf16)，F 实例只加载 FFN 权重 (~582MB/层 bf16)。单侧权重远小于完整模型，可以在显存/内存中预缓存多个 TP 配置的权重。

```
C-1. 多 TP 权重预缓存                                   ⏱ 3 天
  任务: 预加载多个 TP 配置的权重到显存或 CPU 内存
    - WeightCache 类:
      - 启动时为每个候选 TP (如 {1,2,4,8}) 预切分权重
      - AF 分离后单侧权重量小:
        Attn 权重 (Qwen3-32B): 52MB/层 × 64层 ≈ 3.3 GB (全量)
        FFN 权重 (Qwen3-32B): 582MB/层 × 64层 ≈ 36.4 GB (全量)
      - 缓存策略:
        (a) 显存缓存: A 实例显存充裕 (无 FFN 权重), 可缓存多个 TP 的 Attn 权重
            tp=1: 3.3GB, tp=2: 1.65GB/卡, tp=4: 0.83GB/卡 → 全部缓存 < 6GB
        (b) CPU 内存缓存: F 实例的 FFN 权重较大, 优先缓存在 CPU pinned memory
            tp=1: 36.4GB, tp=2: 18.2GB/卡 → CPU 内存通常足够
        (c) 混合: 当前 TP 的权重在 GPU, 其他 TP 的在 CPU pinned memory
      - 切换时: GPU↔CPU 拷贝 (PCIe Gen4: ~25GB/s, 3.3GB Attn ≈ 130ms)
    - 修改 AFDWeightFilter: 支持按 TP 切分后缓存多份
  产出: python/sglang/srt/energy/weight_cache.py
  修改: afd_mixin.py (权重加载时同时缓存多 TP 版本)

C-2. NCCL 通信组动态重建                                ⏱ 2 天
  任务: TP 切换时重建 NCCL 通信组
    - TP 变化意味着参与 all-reduce 的 rank 集合变了
    - 方案: 预创建所有候选 TP 的 NCCL 通信组 (启动时)
      tp=1: 无需 all-reduce
      tp=2: 预建 (0,1), (2,3), ... 的通信组
      tp=4: 预建 (0,1,2,3), (4,5,6,7), ... 的通信组
      tp=8: 预建 (0..7) 的通信组
    - 切换时: 选择对应的预建通信组, 无需 destroy/create
    - 修改 LayerCommunicator: 支持运行时切换 process_group
  产出: python/sglang/srt/energy/tp_group_manager.py
  修改: layers/communicator.py (支持动态 process_group)

C-3. 扩缩容编排器 (Orchestrator)                        ⏱ 4 天
  任务: 根据 Tier 1 ILP 输出, 编排 A/F 实例的创建/销毁/TP 切换
    - AFOrchestrator 类:
      - apply_plan(new_config) → 执行重规划
      - 三种场景的处理:
        场景 1 (仅频率变化): 直接调用 DVFSController, ~6ms
        场景 2 (TP 变化):
          (a) drain 旧实例上的活跃请求 (停止接收新请求, 等待完成)
          (b) 从 WeightCache 加载新 TP 的权重到 GPU
          (c) 切换 NCCL 通信组
          (d) 重新分配 KV cache (Attn 侧, head 数变化)
          (e) 恢复接收请求
        场景 3 (P/D 资源重分配):
          (a) drain Prefill 侧
          (b) 释放 GPU → 重新分配角色 (A/F 权重切换)
          (c) 启动新角色实例
      - 过渡期 SLO 保护: Tier 2 自动升频 + 准入控制
    - 集成到 Scheduler: 接收 Tier 1 输出, 触发 Orchestrator
  产出: python/sglang/srt/energy/af_orchestrator.py
  修改: scheduler.py (新增重规划触发逻辑)

C-4. KV Cache 容量感知 + 重分配                          ⏱ 3 天
  任务: TP 变化时处理 Attn 侧的 KV cache, 并在 ILP 和编排中考虑 KV cache 容量

  **KV cache 容量与 TP 的关系** (Qwen3-32B, bf16):
    每 token KV cache = 2 × (num_kv_heads/tp) × head_dim × num_layers × 2B
    tp=1: 2 × 8 × 128 × 64 × 2 = 2.62 MB/token
    tp=2: 2 × 4 × 128 × 64 × 2 = 1.31 MB/token
    tp=4: 2 × 2 × 128 × 64 × 2 = 0.66 MB/token
    tp=8: 2 × 1 × 128 × 64 × 2 = 0.33 MB/token

    A 实例显存 (80GB) - Attn 权重 (~3.3GB) - 多 TP 权重缓存 (~6GB) ≈ 70GB 可用
    tp=1: 70GB / 2.62MB ≈ 26K tokens → bs=256, avg_seq=100 → 仅支持 ~26K/100 ≈ 260 并发
    tp=2: 70GB / 1.31MB ≈ 53K tokens/卡 → 并发能力翻倍
    tp=4: 70GB / 0.66MB ≈ 106K tokens/卡

  **对 ILP 的影响** (补充约束):
    ILP 显存约束 (15) 中的 bs_max 不是自由参数, 而是受 KV cache 容量限制:
      bs_max(tp) = KV_capacity(tp) / avg_seq_len
    当 ILP 选择低 TP (如 tp_DA=1) 时, bs_max 较小 → 单实例吞吐受限
    → 需要更多副本 k_D → 消耗更多 GPU → 可能不如选 tp_DA=2 划算
    ILP 自动通过吞吐约束 (7)(8) 和显存约束 (15) 捕获这种 trade-off

  **TP 切换时的 KV cache 处理**:
    - 策略: drain-then-switch (不做在线迁移)
      → 旧请求在旧 TP 配置下自然完成
      → 新请求在新 TP 配置下从头分配 KV cache
    - 容量缩减风险: tp 从大变小时 (如 tp=4→2), 每卡 KV cache 容量下降
      → drain 条件: 等待活跃请求的 KV cache 总量 < 新容量的 80%
      → 若 drain 超时 (如 >30s): 强制驱逐最长请求 (preempt + recompute)
    - 容量扩增: tp 从小变大时 (如 tp=2→4), 每卡容量增加, 无风险
    - 需要修改 KV cache 分配器: 支持按新 TP 的 head 分布重新初始化
      MHATokenToKVPool: head_num = num_kv_heads / tp_new
    - 过渡期: 旧 KV cache pool 和新 KV cache pool 短暂共存
      → 旧 pool 只读 (服务 drain 中的请求), 新 pool 读写 (服务新请求)
      → 旧请求全部完成后释放旧 pool
  修改: mem_cache/memory_pool.py, mem_cache/allocator.py, tier1_solver.py (显存约束)
```

---

### 9.3 端到端集成与闭环

```
D-1. Monitoring → Tier 1 → Orchestrator 闭环             ⏱ 2 天
  任务: 将 B-3 (监控) → B-2 (ILP) → C-3 (编排) 串联
    - WorkloadMonitor 每 10-30s 采样
    - 触发条件满足 → 调用 Tier1Solver.solve()
    - ILP 输出新配置 → AFOrchestrator.apply_plan()
    - 过渡期 Tier 2 自动升频保护
  修改: scheduler.py (主循环中加入监控检查)

D-2. 配置热加载                                          ⏱ 1 天
  任务: 支持运行时修改 SLO、M、α 等超参
    - 通过 HTTP API 或配置文件热加载
    - 触发 Tier 1 重规划
```

### 9.4 实现优先级与依赖

```
                    ┌─────────────────────────────────────────┐
                    │  已有基础: AFD + DVFS + Profile 数据     │
                    └──────────────┬──────────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
     ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐
     │ A-1 预测模型  │    │ B-1 Profile表 │    │ C-1 权重预缓存    │
     │   (2天)      │    │   (1天)       │    │   (3天)          │
     └──────┬──────┘    └──────┬───────┘    └────────┬─────────┘
            │                  │                     │
            ▼                  ▼                     ▼
     ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐
     │ A-2 DVFS控制 │    │ B-2 ILP求解器 │    │ C-2 NCCL组管理    │
     │   (3天)      │    │   (3天)       │    │   (2天)          │
     └──────┬──────┘    └──────┬───────┘    └────────┬─────────┘
            │                  │                     │
            ▼                  ▼                     ▼
     ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐
     │ A-3 流水线集成│    │ B-3 负载监控  │    │ C-3 扩缩容编排    │
     │   (2天)      │    │   (1天)       │    │   (4天)          │
     └──────┬──────┘    └──────┬───────┘    │ C-4 KV cache (2天)│
            │                  │            └────────┬─────────┘
            └────────────┬─────┘                     │
                         ▼                           │
                  ┌──────────────┐                   │
                  │ D-1 闭环集成  │◄──────────────────┘
                  │   (2天)      │
                  └──────────────┘

推荐实现顺序:
  Phase 1 (核心, 可独立验证):  A-1 → A-2 → A-3          (7天, 动态调频)
  Phase 2 (规划, 可独立验证):  B-1 → B-2 → B-3          (5天, ILP 求解)
  Phase 3 (弹性, 依赖 Phase 2): C-1 → C-2 → C-3 → C-4  (12天, 扩缩容+KV cache)
  Phase 4 (集成):              D-1 → D-2                 (3天, 闭环)

总计: ~27 天 (Phase 1/2 可并行, 实际 ~3-4 周)
```

### 9.5 新增文件清单

| 文件 | 功能 | 依赖 |
|------|------|------|
| `srt/energy/af_profile_predictor.py` | 在线延迟/能耗预测 | Profile 数据 |
| `srt/energy/af_dvfs_controller.py` | Tier 2 调频策略 | af_profile_predictor |
| `srt/energy/profile_table.py` | Profile 表加载与查询 | Profile 数据 |
| `srt/energy/tier1_solver.py` | Tier 1 ILP 求解器 | profile_table, PuLP |
| `srt/energy/workload_monitor.py` | 负载监控与分桶 | scheduler |
| `srt/energy/weight_cache.py` | 多 TP 权重预缓存 | afd_mixin |
| `srt/energy/tp_group_manager.py` | NCCL 通信组管理 | torch.distributed |
| `srt/energy/af_orchestrator.py` | 扩缩容编排器 | 以上所有 |

### 9.6 关键风险与缓解

| 风险 | 影响 | 缓解方案 |
|------|------|---------|
| NCCL 通信组预创建占用资源 | 启动时间增加 | 仅预建当前 TP ± 1 级的通信组, 按需扩展 |
| TP 切换期间请求延迟抖动 | SLO 违反 | drain-then-switch + Tier 2 升频保护 |
| 权重 GPU↔CPU 拷贝延迟 | 切换时间 ~130ms (Attn) | 异步拷贝 + shadow instancing |
| ILP 求解时间过长 | 重规划延迟 | Pareto 剪枝 + Warm Start, 目标 <1s |
| Profile 表不覆盖运行时配置 | 预测不准 | 线性插值 + 运行时校准 (实测 vs 预测偏差修正) |
| Decode 大 bs 时 DF compute-bound | 调频策略失效 | Profile 模型含 bs 维度, 自动适应 |
| TP 缩减时 KV cache 容量不足 | drain 超时, 过渡期过长 | drain 超时后强制 preempt + recompute; ILP 优先选容量安全的 TP |
| 低 TP 下 KV cache 限制并发数 | 吞吐不足 | ILP 显存约束自动捕获, 必要时增加副本数或选更高 TP |
| 多 TP 权重缓存占用 A 实例显存 | KV cache 可用空间减少 | Attn 权重小 (全 TP 缓存 <6GB), 影响有限; 可选择性只缓存相邻 TP |
