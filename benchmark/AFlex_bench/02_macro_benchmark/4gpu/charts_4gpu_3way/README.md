# 4-GPU 三方案对比：PDAF / PD DP2 / Native DP4（Tier vs 满频）

GPU 4-7，Qwen3-32B，3 个变长 workload（steady/varying/heavy）。
SLO：TTFT < 2000ms（纯处理时间口径，去排队）/ TPOT < 150ms。
所有配置 SLO 违背率均为 **0.0%**。

数据来源：
- Tier（DVFS, `--freq auto`）：`scripts/bench/results/4gpu_3way/json/`
- 满频对照（`--freq max`）：`scripts/bench/results/4gpu_3way_bl/json/`
绘图脚本：`scripts/bench/plot_4gpu_3way.py`

<!-- TABLES -->
## Tier 节能百分比

节能 % = (满频能耗 − Tier 能耗) / 满频能耗

| 架构 | steady | varying | heavy |
|---|---:|---:|---:|
| PDAF+Tier | 30.2% | 27.7% | 29.3% |
| PD DP2+Tier | 1.3% | 1.3% | −1.7% |
| Native DP4+Tier | 39.2% | 34.4% | 36.2% |

## 总能耗（满频 → Tier，J）

| 架构 | steady | varying | heavy |
|---|---:|---:|---:|
| PDAF | 99255 → 69260 | 75476 → 54555 | 86340 → 61069 |
| PD DP2 | 229949 → 227018 | 90031 → 88906 | 102377 → 104096 |
| Native DP4 | 169632 → 103066 | 127009 → 83288 | 140703 → 89797 |

## 吞吐 (tok/s) / TPOT (ms)，Tier

| 架构 | steady | varying | heavy |
|---|---:|---:|---:|
| PDAF+Tier | 711.6 / 83 | 181.4 / 64 | 260.7 / 69 |
| PD DP2+Tier | 341.6 / 48 | 183.0 / 48 | 239.1 / 48 |
| Native DP4+Tier | 737.2 / 74 | 182.1 / 63 | 264.2 / 71 |

## TTFT 处理时间 (ms，去排队)，满频 → Tier

| 架构 | steady | varying | heavy |
|---|---:|---:|---:|
| PDAF | 115 → 535 | 90 → 259 | 117 → 694 |
| PD DP2 | 50 → 49 | 49 → 49 | 48 → 50 |
| Native DP4 | 157 → 237 | 131 → 199 | 159 → 238 |



<!-- FIGS -->
## 图表

### 三方案对比（Tier vs 满频，2×2 指标）
![comparison](4gpu_3way_comparison.png)

### Tier 节能百分比
![savings](4gpu_3way_savings.png)

### PDAF / PD 的 P/D 能耗拆分（Tier）
![breakdown](4gpu_3way_pd_breakdown.png)



<!-- CONCLUSION -->
## 关键结论

1. **绝对能耗最低始终是 PDAF**：即便 Native 的 Tier 节能比例最高（34–39%），其 Tier 后能耗（83–103 kJ）仍高于 PDAF 满频（75–99 kJ），更高于 PDAF+Tier（54–69 kJ）。
2. **Native 的 Tier 节能比例最大（~36%）**：满频下 4 张单卡 TP=1 利用率低、SLO slack 大，降频空间充足（与 DynamoLLM 观察一致）。
3. **PD DP2 几乎无 Tier 收益（~1%，heavy 略负）**：能耗集中在 Decode（D 占 ~74%），而 decode 是 memory-bound、对频率不敏感（TPOT 恒 48ms）；Prefill 占比小、可降频空间有限（对应 BiScale "decode 调频收益远小于 prefill"）。
4. **延迟代价**：Tier 用 TTFT 处理时间换能耗。PDAF proc TTFT 115→535ms，Native 157→237ms，PD 几乎不变（~49ms，本就贴地）。
5. **公平性提醒**：本对比用去排队的 TTFT 处理时间口径，三方案口径一致。PD 的端到端 TTFT（含排队）在 steady/heavy 极高（74s/14s），说明该口径会掩盖 PD 单卡 1P1D 的排队塌陷，解读时需结合吞吐与端到端 TTFT。

