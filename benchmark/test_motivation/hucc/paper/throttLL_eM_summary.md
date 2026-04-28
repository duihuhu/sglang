# throttLL'eM：面向 SLO 的预测式 GPU 节流与 LLM 推理能效

> **正式标题**: *throttLL'eM: Predictive GPU Throttling for Energy Efficient LLM Inference Serving*（文中 Section 4 亦表述为 SLO-aware & energy-efficient LLM inference serving）  
> **arXiv**: [2408.05235v2](https://arxiv.org/abs/2408.05235) [cs.DC]，文中标注 2025-12-03  
> **作者**: Andreas Kosmas Kakolyris (ETH Zürich), Dimosthenis Masouros, Petros Vavaroutsos, Sotirios Xydis, Dimitrios Soudris (NTUA)  
> **代码**: <https://github.com/WilliamBlaskowicz/throttLL-eM>

---

## 一、研究动机

- **推理占主导**: LLM 在数据中心的算力消耗中，推理占比极高；单次响应能耗可达约 Wh 量级，规模化后能耗与环境、成本压力显著。
- **能效 vs 体验**: 降低功耗常与低延迟、高吞吐冲突。交互式服务用 **SLO**（如 p99 端到端延迟、token 间时间 TBT）刻画用户体验，需在满足 SLO 的前提下节能。
- **传统手段不适配 LLM serving**: 静态功耗封顶、电源超配、面向 CNN 的 batching、race-to-idle 等，在 **自回归**、**动态 batch**、**持续有请求** 的 LLM 场景下易推高尾延迟或难以进入有效 idle。

**核心思路**: 利用 **GPU 动态调频（DVFS / throttling）** 在 **迭代级**（毫秒级）调节功耗；相比整请求级控制更细，有望在保证 SLO 的同时压低能耗。难点在于生成过程与负载的 **不可预测性**，需要 **预测 + 控制** 闭环。

---

## 二、LLM 推理节能的三类障碍（论文归纳）

1. **自回归**: 生成长度随机，难以预先为每个请求固定分配算力。
2. **动态 batch 组成**（如 inflight batching）: batch 大小变化会带来显著性能波动（文中示例：TBT/E2E 等可恶化约 45% 量级），使 SLO 下的能效优化复杂化。
3. **KV cache 可变占用**: 序列增长导致 KV 显存占用变化；KV 增大与性能下降相关（文中提到可达约 18.2% 量级的退化分析场景）。

---

## 三、测量驱动的主要结论（Section 3，Llama2-13B 等）

### 3.1 Batch 大小 × GPU 频率

- 吞吐随 batch 与频率升高而升高；E2E 与 TBT 随频率升高而改善，但随 batch 增大而变差。
- **功耗** 主要由 **频率** 决定，同频下不同 batch 的功耗相对平稳。
- **能效（TPJ, tokens/J）** 存在 **甜点频率**：并非频率越低越省能；例如文中给出在最大 batch 下，约 **1050 MHz** 相对峰值频率可带来显著能效提升，同时对吞吐/E2E/TBT 的影响在所述实验设定下相对温和；过低频率则吞吐损失抵消省电收益。

### 3.2 KV 占用

- KV block 增多 → 吞吐（IPS）下降、TBT 近似线性上升；KV 与功耗正相关，高频下随 KV 增长的功耗斜率更陡。
- **Pearson 相关**: KV 与 TBT 约 **0.92**，KV 与吞吐约 **-0.92**，故 KV 可作为性能建模的重要代理特征。

### 3.3 并行划分（单节点）

- 相比 DDP/PP，**Tensor Parallelism (TP)** 在吞吐与能效上更优；但 **更小引擎规模** 在接近其可承受的最大 batch 时可能 **TPJ 更高**，提示需要 **按负载动态调整引擎规模（实例并行度）**。

### 3.4 生产 trace 特征（Azure）

- Prompt / 生成长度均为 **长尾**；到达率 **非均匀、持续有负载**，削弱 race-to-idle 类策略。

---

## 四、系统：throttLL'eM 架构

**目标 SLO**: 重点保证 **E2E** 与 **TBT**（TTFT 未作为主要优化对象，见 Discussion）。

**主要组件**:

| 组件 | 作用 |
|------|------|
| **生成长度预测** | 估计每个请求的生成 token 数 \(\hat{\|r_i\|}\)，驱动后续 KV/batch 投影；可与文献中 bucket/回归等方法对接。 |
| **Scoreboard + KV & Batch 投影** | 基于调度迭代 \(s_i\)、输入长度 \(\|q_i\|\)、\(\hat{\|r_i\|}\) 解析地构造未来每步的 **batch 向量 B** 与 **KV 占用向量 KV**（按 KV block 容量 \(N\) 分块计数）。该模块延迟约 **<2 ms**。 |
| **查询调度 / 准入控制** | 新请求先 **虚拟** 写入 Scoreboard，检查：① KV 是否超容量（避免换入主机内存等严重退化）；② 用性能模型在 **最大频率** 下预测每步 IPS→TBT，是否满足 TBT SLO；③ 用累计时间估计是否满足各请求 E2E deadline。不满足则 **排队**。 |
| **性能预测模型 M** | **梯度提升树（GBDT）**，输入：**引擎规模、batch、KV 占用、GPU 频率**；输出：**IPS**。CPU 推理约 **3 ms**；与调度器合计约 **35 ms**（重载下）。 |
| **GPU 频率节流** | 准入通过后，在频率可行域上 **二分搜索** **满足 SLO 的最低频率**；全实例 GPU 统一频率。若存在已标记 “lost” 的请求则回退最高频尽力满足。 |
| **Autoscaling** | 以约 **10 s** 为周期，按 RPS 与预刻画容量调整 **TP 引擎规模**；用 **shadow instancing** 掩盖新引擎冷启动（>20s）；**grace period** 内允许升规格、避免过早降规格以减少双引擎空转浪费。 |

**预测误差处理**: 对 \(\hat{\|r_i\|}\) 做保守放大；若实际长度超过调整值则回退到 **max_tokens** 等上限并更新 Scoreboard。

---

## 五、实验设置与结果摘要（Section 5）

### 5.1 设置

- **硬件**: AWS `p4d.24xlarge`，8× A100 40GB。  
- **软件**: **NVIDIA Triton + TensorRT-LLM**，inflight batching、paged KV、FlashAttention。  
- **模型**: Llama3-8B、Llama2-13B（TP1/2/4）、Llama3-70B（TP8）等。  
- **SLO**: TBT 目标 **平均 <200 ms**（对应约 250 wpm 阅读速度）；E2E SLO 取各引擎在 **最高频、饱和负载** 下的 **p99 响应时间**。  
- **负载**: Azure LLM 推理 trace，按各引擎最大 RPS 做缩放；因隐私用合成 query 匹配 trace 中的长度分布。

### 5.2 模型与投影精度

- **IPS 预测**: 90/10 划分下 **R² ≥ 0.97**（多数 >0.98），**MAE < 1 IPS**，MAPE 约 **2.8%–5.8%**（随配置变化）；10/90 训练仍较稳健。  
- **Batch / KV 投影**: 平均误差约 **0.19%** / **2.26%**；累计时间漂移约 **0.43 ms/迭代**（相对 TBT 15–30 ms 较小）。

### 5.3 相对 Triton 的端到端（无 autoscaling）

- 在 oracle 生成长度下，平均 **节能约 24.7%**，最高约 **30.7%**；**能效（TPJ）平均约 +36.3%**，最高配置可达约 **+44.3%**（相对 Triton）。  
- 30% p95 长度预测误差时，能效增益仍约 **30.0%**。  
- 多数配置满足 E2E p99 SLO；**Llama2-13B TP1** 因容量紧、baseline 贴近 SLO，节流空间受限。  
- **TTFT 可能增加**（排队 + prefill 阶段仍受较低频率影响）。

### 5.4 启用 autoscaling（Llama2-13B TP1/2/4）

- 相对 **Triton 固定 TP4**：**能耗最高降约 43.8%**（oracle），30% 预测误差时约 **41.7%**；**能效约 1.71×–1.78×**（TPJ 约 0.69 → 1.19–1.23）。  
- 对比分解：**仅 autoscaling** 约节能 **20.8%**，**仅节流** 约 **30.6%**，二者结合最优。  
- E2E 相对 baseline 会变差但仍可压在 TP4 的 p99 SLO 以下；TBT 仍满足。

### 5.5 运行时行为

- 轻载选较低频；接近引擎最大负载时提高频率以避 SLO 违反。  
- 违反主要出现在 **超过最大引擎容量** 或 **RPS 突增尚未完成升规格** 的瞬态。

---

## 六、局限与讨论（与后续工作衔接）

- **TTFT / prefill**: 未专门优化；因 **切频开销**（文中约 **200 ms**）相对 **prefill 时长**（约 **175 ms** 量级）不划算；提及 **Splitwise** 类 prefill/decode 分离可独立优化。  
- **扩展性**: 聚焦单节点 **TP**；跨节点 **PP** 需额外编排（方法上仍可逐节点建模）。与 **DDP 多副本** 正交。  
- **开销**: 生成长度预测器延迟文献量级约 **7.6 ms**；性能模型 **离线 profiling** 对大模型可达约 **一天**（一次性）。  
- 与同目录 **BiScale** 等工作的关系：BiScale 强调 **disaggregated** 下 prefill/decode 不对称与组合配置；throttLL'eM 面向 **单体 Triton+TRT-LLM 栈** 的 **迭代级 DVFS + TP autoscaling**，二者问题设定与栈假设不同，可互为参照。

---

## 七、可迁移的要点（用于 motivation / 实验设计）

1. **KV、batch、频率** 是迭代级性能与功耗的强相关量，适合作为 **轻量 ML 或查表模型** 的特征。  
2. **“最低满足 SLO 的频率”** 可通过 **在投影轨迹上重复 SLO 检验 + 二分** 实现，无需连续优化。  
3. **准入控制 + 排队** 是换取 **可证明的 SLO 安全裕度** 与 **降频空间** 的杠杆；代价是 **TTFT/排队时间**。  
4. **Autoscaling 与 DVFS 互补**：单独节流或单独缩实例均有收益，组合在论文 trace 上最优。

---

*文档根据仓库内 PDF 文本提取整理，公式与数值以原论文为准。*
