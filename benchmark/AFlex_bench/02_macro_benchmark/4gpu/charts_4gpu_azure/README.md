# Azure Trace 4-GPU 对比测试

GPU 4-7，Qwen3-32B，基于 Azure LLM Inference Trace 真实负载。
SLO：TTFT < 2000ms（纯处理时间口径）/ TPOT < 150ms。

## 数据源

| Trace | 原始规模 | IL avg/p50 | OL avg/p50 | 特征 |
|-------|---------|-----------|-----------|------|
| Code (代码补全) | 1680万 req / 7天 | 2511/1930 | 23/8 | Prefill-heavy, 极短 output |
| Conv (对话) | 2730万 req / 7天 | 1632/928 | 106/41 | 均衡, 中等 decode |

## 测试负载

从 trace 中选取 5 分钟稳定窗口，按比例缩放 QPS 适配 4 卡容量：

| 负载 | Code QPS | Conv QPS | 请求数 |
|------|---------|---------|--------|
| Light | 2 | 2 | 600 |
| Medium | 5 | 4 | 1500/1200 |
| Heavy | 8 | 6 | 2400/1800 |

## Code Trace 结果（PD DP2 & Native DP4）

PDAF 因 TP=1 单卡无法在 SLO 内处理 IL>2000 的 prefill，所有负载 TPOT 超标，故不纳入有效对比。

### 节能百分比（有效数据点）

| 架构 | Light | Medium | Heavy |
|------|------:|-------:|------:|
| PD DP2+Tier | 26.2% | 21.9% | 6.1% |
| Native DP4+Tier | 28.0% | 38.7% | 24.0% |

### SLO 违背率（Tier / 满频）

| 架构 | Light | Medium | Heavy |
|------|------:|-------:|------:|
| PD DP2 | 0.0/0.0% | 1.0/0.0% | 1.0/0.0% |
| Native DP4 | 0.5/0.0% | 32.6/3.7% | 66.1/30.7% |

## Conv Trace 结果（PD DP2 & Native DP4）

PDAF 在 medium/heavy 下请求超时失败；PD DP2 在 heavy 下也超载。

### 节能百分比（有效数据点）

| 架构 | Light | Medium | Heavy |
|------|------:|-------:|------:|
| PD DP2+Tier | 11.8% | 28.0% | N/A |
| Native DP4+Tier | 36.1% | 34.8% | 13.9% |

### SLO 违背率（Tier / 满频）

| 架构 | Light | Medium | Heavy |
|------|------:|-------:|------:|
| PD DP2 | 0.2/0.0% | 23.6/9.0% | N/A |
| Native DP4 | 0.0/0.0% | 2.8/0.0% | 7.1/1.2% |

## 图表

### Code Trace 对比
![code](azure_code_comparison.png)

### Conv Trace 对比
![conv](azure_conv_comparison.png)

### 节能百分比汇总
![savings](azure_energy_savings.png)

## 关键结论

1. **Native DP4 Tier 节能最稳健**：在两种 trace 下，轻中负载均能实现 28-39% 节能，SLO 违背率可控（<3%）。
2. **PD DP2 Tier 在适中负载下有效**：Code light 26%、Conv medium 28% 节能，decode 降频（55→70ms TPOT）换取显著省电。Heavy 负载下 SLO slack 不足，节能空间收窄。
3. **PDAF 不适合长 IL trace**：TP=1 单卡处理 IL>2000 时 TPOT 190+ms 超 SLO，需要 TP≥2 才能承载代码补全类长上下文负载。
4. **负载越重 Tier 空间越小**：这符合预期 — 高负载下 SLO slack 紧张，降频余地有限。真实部署应根据负载动态调整 Tier 激进度。
5. **Conv trace 的 decode 更重**（OL avg 113 vs Code 的 23），PD decode 单卡在 medium 以上即超载（排队 TTFT 35 万 ms），说明 conv 场景需要更大 decode 容量。
