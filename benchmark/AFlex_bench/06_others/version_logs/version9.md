### 2025-06-12 ~ 2025-06-13

#### MoE 模型（Qwen3-30B-A3B）全量 Benchmark

完成 MoE 模型在 8 GPU 下三方案六配置的全量 Azure workload 测试：

| 方案 | code_medium SLO% | conv_light SLO% | conv_medium SLO% | conv_heavy SLO% | 节能(vs无Tier) |
|------|:-:|:-:|:-:|:-:|:-:|
| Native DP8 | 0.0% | 0.1% | 0.0% | 0.0% | — |
| Native DP8 + Tier | 0.1% | 0.0% | 0.1% | 0.0% | 7-9% |
| PD DP4 | 0.6% | 0.0% | 0.1% | 0.3% | — |
| PD DP4 + Tier | 0.8% | 0.0% | 0.1% | 0.5% | 10-11% |
| PDAF TP2 | 1.4% | 0.1% | 0.2% | 37.1% | — |
| PDAF TP2 + Tier | 20.0% | 11.3% | 16.2% | 84.1% | 20-24% |

结论：Native+Tier 和 PD+Tier 表现正常（SLO 违反 <1%）。**PDAF+Tier 在 MoE 模型上 SLO 违反严重（11-84%）**，即使 SLO 已放宽到 250ms。

---

#### 根因分析：MoE 模型 DVFS 失效原因

通过对比 Qwen3-32B（Dense）和 Qwen3-30B-A3B（MoE）两个模型在 PDAF+Tier 下的表现差异，识别出以下根因：

##### 1. MoE 模型是 Memory-Bound，频率对延迟几乎无影响

| 频率 (f_A=f_F) | Dense 32B iter_lat | MoE 30B iter_lat |
|:-:|:-:|:-:|
| 1410 MHz (max) | 154 ms | 148 ms |
| 930 MHz | 158 ms | 150 ms |
| 690 MHz | 178 ms | 151 ms |
| 450 MHz (min) | 189 ms | 153 ms |
| **降频增幅** | **+35ms (23%)** | **+5ms (4%)** |

- Dense 模型是 **compute-bound**：降频导致延迟显著增加 → DVFS 控制器被 SLO 自然约束，只选到 ~930MHz
- MoE 模型是 **memory-bound**：128 个 expert 权重加载是瓶颈，频率对 HBM 带宽无影响 → 控制器看到"任何频率都满足 SLO"，选最低频（210MHz）

##### 2. DVFS 控制器决策失误

由于模型预测"降到任何频率延迟都只有 ~150ms"（远低于 SLO=250ms），控制器选择了最低频率以最大化节能。但实际上：
- 降频虽然单步只慢 5ms，但累积效应导致吞吐下降
- 吞吐下降 → 请求队列积压 → 排队时间大增 → 端到端 TPOT 飙升
- 这种"排队雪球效应"在能耗模型中完全没有建模

##### 3. MoE 延迟预测本质困难：Expert 路由不可预测

能耗模型仅使用 `(tp, M, f_A, f_F, input_len, batch_size)` 6 个特征预测延迟，但 MoE 的实际延迟还依赖于 **expert 路由分布**：

- **Profiling 时**：使用合成 prompt（`"Hello " * N`），所有 token 走相同的 expert 路由 → 延迟稳定
- **实际 Serving 时**：不同请求内容完全不同（代码/对话/技术文档），一个 batch 中 token 的 expert 分布是随机且不均匀的
- 当热门 expert 被大量 token 集中调用时，负载不均衡导致延迟突增
- 这种语义相关的延迟波动无法被 `(input_len, batch_size)` 特征捕获

**证据**：即使在最高频（无 Tier）下，MoE 的 TPOT p99/p50 = 2.0-2.7x（高尾部波动），而 Dense 只有 1.07-1.18x（极其稳定）。

##### 4. Batch Size 断崖效应

MoE profiling 数据显示 `bs=64→96` 存在延迟断崖（85ms → 177ms，翻倍），而 Dense 模型是线性增长。这可能与 expert 并行度、all-to-all 通信瓶颈或 profiling 方法中 prefill 排队的 artifact 有关。在线 serving 中，batch size 动态穿越该临界点时会触发延迟跳变。

##### 5. 对比总结

| 维度 | Qwen3-32B (Dense) | Qwen3-30B-A3B (MoE) |
|------|:-:|:-:|
| 计算瓶颈 | Compute-bound | Memory-bound |
| 降频延迟增加 | +35ms (23%) | +5ms (4%) |
| DVFS 实际选频 | ~930MHz（受 SLO 约束） | 210MHz（无约束） |
| TPOT p99/p50 (无 Tier) | 1.07-1.18x | 2.0-2.7x |
| 延迟可预测性 | 高（固定计算路径） | 低（依赖 expert 路由） |
| PDAF+Tier SLO 违反 | ≤0.3% | 11-84% |
| Tier 节能效果 | 有效且安全 | 节能高但 SLO 不达标 |

---

#### 结论

1. **PD DP4 + Tier 是 MoE 模型的最优部署方案**：节能 10-11%，SLO 违反 <1%
2. **PDAF + Tier 不适合 Memory-bound 的 MoE 模型**：频率调节空间极小，且延迟预测因 expert 路由不确定性而失准
3. **改进方向**：若要让 PDAF+Tier 适用于 MoE，需要引入 expert 负载感知的延迟预测（如加入 expert 分布特征），或在控制器中设置保守的频率下限
