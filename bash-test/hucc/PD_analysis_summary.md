# PD 分离能效优化分析报告

## 数据概况

- **Prefill 数据**: 906 条记录
- **Decode 数据**: 7530 条记录
- **TP 配置**: 1, 2, 4, 8
- **频率范围**: 210-1410 MHz (6档)
- **Input lengths**: 128-40000 tokens
- **Batch sizes**: 1-256

---

## 核心发现

### 1. 频率敏感性分析

#### Prefill 阶段
- **Attention (A)**: 高度 compute-bound
  - A_ratio (210MHz/1410MHz) ≈ 5.9-6.3
  - 降频会显著增加延迟（约6倍）
  - TP 对 A 的 memory-bound 特性影响较小

- **FFN (F)**: 同样 compute-bound
  - F_ratio ≈ 5.3-6.0
  - TP 越大，F 越 memory-bound（tp=8 时 ratio=5.26）

#### Decode 阶段（关键差异）
- **Attention (A)**: 随 TP 增大变 memory-bound
  - tp=1: ratio=1.94-3.93 (compute-bound)
  - tp=4: ratio=0.99-3.13 (小 bs 时 memory-bound)
  - **tp=8: ratio=0.98-2.56 (强 memory-bound)**
  
- **FFN (F)**: 始终较 compute-bound
  - tp=1: ratio=3.06-4.88
  - tp=8: ratio=1.76-4.15
  - F 的降频空间小于 A

**关键洞察**: Decode 阶段 A 的 memory-bound 特性（尤其 tp≥4）为 PD 分离调频提供了优化空间。

---

### 2. 端到端 PD 分离收益

#### SLO × 1.0（严格 SLO）
- **所有 TP 收益为 0%**
- 原因：严格 SLO 下，P 和 D 都必须选最高频（1410MHz），无差异化空间

#### SLO × 1.1（轻微放松，最佳甜区）

| TP | 平均节能 | 中位数 | 最大节能 | 甜点占比 (>5%) |
|---|---|---|---|---|
| tp=1 | **8.40%** | 2.69% | 21.91% | 39.4% |
| tp=2 | **10.74%** | 13.56% | 22.05% | 63.6% |
| tp=4 | **16.32%** | 17.73% | **23.31%** | **100%** |
| tp=8 | **15.97%** | 17.42% | 23.10% | 97.0% |

**核心发现**:
- **tp=4 和 tp=8 是最强甜区**：几乎所有配置都能节能 >5%
- **平均节能 10-16%**，最高可达 23%
- SLO×1.1 是 PD 分离的"黄金窗口"

#### SLO × 1.2（中等放松）

| TP | 平均节能 | 甜点占比 |
|---|---|---|
| tp=1 | 1.76% | 0% |
| tp=2 | 1.38% | 0% |
| tp=4 | 3.97% | 24.2% |
| tp=8 | 4.76% | 54.5% |

收益快速衰减，但 tp=8 仍保持一定优势。

#### SLO ≥ 1.5（宽松 SLO）
- **收益接近 0%** (0.0-0.4%)
- 原因：P 和 D 都能选到低频，统一调频已足够优化

---

### 3. 能耗构成分析

**Prefill vs Decode 能耗占比** (output_len=64):

| TP | Prefill 占比 | Decode 占比 |
|---|---|---|
| tp=1 | 30-32% | 68-70% |
| tp=2 | 30% | 70% |
| tp=4 | 27-29% | 71-73% |
| tp=8 | 22-26% | 74-78% |

**关键发现**:
- Decode 占总能耗 **68-78%**
- TP 越大，Decode 占比越高
- 与 AF 分离论文不同（Decode 占 99%），PD 分离中 **Prefill 能耗占比显著（22-32%）**

---

## 与 AF 分离论文的对比

| 维度 | AF 分离 | PD 分离 |
|---|---|---|
| **最佳 SLO 窗口** | ×1.0（严格） | ×1.1（轻微放松） |
| **严格 SLO 收益** | 5-9% | 0% |
| **最佳 SLO 收益** | 5-9% | 10-16% |
| **宽松 SLO 收益** | 2-4% | 0% |
| **Prefill 能耗占比** | 1% | 22-32% |
| **甜区 TP** | tp=2,4 | tp=4,8 |
| **收益来源** | A memory-bound | P/D 阶段差异 |

---

## 物理机制解释

### 为什么 SLO×1.0 收益为 0？

在严格 SLO 下：
- Prefill 的 A 和 F 都是 compute-bound（ratio≈6），必须用最高频
- Decode 虽然 A 是 memory-bound，但 F 仍是 compute-bound
- **统一调频被 Prefill 的高频需求锁死**，P 和 D 都必须 1410MHz
- PD 分离无法提供额外自由度

### 为什么 SLO×1.1 收益最大？

SLO 放松 10% 后：
- **Prefill 可以降频**：从 1410MHz 降到 1170MHz 或 930MHz
- **Decode 可以进一步降频**：利用 A 的 memory-bound 特性降到更低频率
- P 和 D 的最优频率出现分化，PD 分离的差异化空间打开
- 例如：P 选 1170MHz，D 选 690MHz，节能 16-23%

### 为什么 SLO≥1.5 收益消失？

SLO 过度放松后：
- 统一调频自己也能选到很低的频率（如 450MHz）
- P 和 D 的最优频率都很低，差异化空间被压缩
- PD 分离的边际收益趋近于 0

---

## 甜区配置特征

### SLO×1.1 下的最优频率选择模式

通过分析 `results_slo_1.1.csv`，典型模式：

1. **P 高频 + D 低频** (最常见)
   - P: 1170-1410 MHz
   - D: 450-690 MHz
   - 适用于 tp=4,8 的大部分配置

2. **P 中频 + D 低频**
   - P: 930 MHz
   - D: 210-450 MHz
   - 适用于 input_len 较短的场景

3. **P/D 都中频但不同**
   - P: 1170 MHz
   - D: 930 MHz
   - 适用于 tp=1,2 的部分配置

---

## 结论与建议

### 核心结论

1. **PD 分离在 SLO×1.1 下最有价值**
   - 平均节能 10-16%，最高 23%
   - tp=4 和 tp=8 是最强甜区（100% 和 97% 配置受益）

2. **与 AF 分离互补**
   - AF 分离：严格 SLO 下优化 Decode 内部
   - PD 分离：轻微放松 SLO 下优化 P/D 阶段差异
   - 两者可叠加：先 PD 分离，再在 D 内部做 AF 分离

3. **Prefill 能耗不可忽略**
   - 占总能耗 22-32%（vs AF 论文的 1%）
   - 说明 PD 分离的优化空间更大

### 系统设计建议

1. **SLO 策略**
   - 生产环境设置 SLO = base_latency × 1.1
   - 避免过严（×1.0，无收益）或过松（×1.5+，无收益）

2. **TP 选择**
   - 优先使用 tp=4 或 tp=8
   - tp=8 在 Decode 阶段 A 完全 memory-bound，降频空间最大

3. **频率控制器设计**
   - P 控制器：基于 input_len 和 batch_size 选频
   - D 控制器：基于 KV cache 大小（input_len × batch_size）选频
   - 两者独立决策，每次迭代级调整

4. **与 AF 分离结合**
   - 第一层：PD 分离（粗粒度，阶段级）
   - 第二层：D 内部 AF 分离（细粒度，算子级）
   - 理论叠加收益：16% (PD) + 9% (AF) ≈ 25%

---

## 数据文件

- 原始数据：`/workspace/benchmark/sglang-main/bash-test/hucc/data/`
- 分析结果：`/workspace/benchmark/sglang-main/bash-test/hucc/results_slo_*.csv`
- 分析脚本：`/workspace/benchmark/sglang-main/bash-test/hucc/analyze_pd_slo.py`
