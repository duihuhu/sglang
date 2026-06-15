### 2025-06-15

#### MoE 模型异构 PDAF 部署方案测试

在 version9 中发现对称 PDAF (4P+4D) + Tier 对 MoE 模型存在严重的排队问题。本轮测试引入**异构 PDAF (3P+5D)** 方案，通过给 Decode 侧分配更多 GPU 来缓解排队瓶颈。

---

#### 1. TPOT 计算方法修正

之前 MoE 的 TPOT 采用 `(E2E - TTFT) / (tokens - 1)` 计算，包含了排队延迟。本轮改为与 Dense 一致的**流式计算**：

```
TPOT = (last_token_time - first_token_time) / (token_count - 1)
TTFT_proc = server-side pure prefill time (不含排队)
```

这样 TPOT 只反映纯 GPU 处理时间，排队问题（与能耗优化正交）被分离出去。

---

#### 2. 异构方案设计

由于 MoE 30B 的 FFN 权重（56GB FP16）要求至少 TP=2，无法像 Dense（Llama-8B）那样用 TP=1 缩减 Prefill GPU。最终方案：

**PDAF Asym (3P+5D)**: `PA(TP=1, 1GPU) + PF(TP=2, 2GPU) + DA(TP=1, 1GPU) + DF(TP=4, 4GPU)`

| 组件 | TP | GPU数 | 说明 |
|------|----|-------|------|
| PA (Prefill Attn) | 1 | 1 | Attn 权重仅 9GB，TP=1 足够 |
| PF (Prefill FFN) | 2 | 2 | FFN 权重 56GB，TP=2 必须 (28GB/卡) |
| DA (Decode Attn) | 1 | 1 | Attn 权重仅 9GB |
| DF (Decode FFN) | 4 | 4 | TP=4 增大 KV cache 容量，缓解排队 |

对比对称方案 (4P+4D)：PA(TP=2)+PF(TP=2)+DA(TP=2)+DF(TP=2)，异构方案将 Decode 侧从 4GPU 扩展到 5GPU。

---

#### 3. 全量 Benchmark 结果

8 配置 × 4 workloads = 32 组实验，GPU 独占环境。

##### 能耗对比

| 方案 | Code Medium | Conv Light | Conv Medium | Conv Heavy |
|------|:-:|:-:|:-:|:-:|
| Native DP8 | 230 kJ | 271 kJ | 284 kJ | 311 kJ |
| Native DP8 + Tier | 210 kJ (**-8.8%**) | 246 kJ (**-9.2%**) | 260 kJ (**-8.4%**) | 287 kJ (**-7.8%**) |
| PD DP4 | 223 kJ | 253 kJ | 259 kJ | 278 kJ |
| PD DP4 + Tier | 202 kJ (**-9.2%**) | 227 kJ (**-10.2%**) | 232 kJ (**-10.5%**) | 251 kJ (**-9.6%**) |
| PDAF Sym (4P+4D) | 253 kJ | 263 kJ | 264 kJ | 311 kJ |
| PDAF Sym + Tier | 194 kJ (**-23.1%**) | 202 kJ (**-23.3%**) | 209 kJ (**-21.0%**) | 293 kJ (**-5.9%**) |
| PDAF Asym (3P+5D) | 251 kJ | 262 kJ | 265 kJ | 286 kJ |
| PDAF Asym + Tier | 193 kJ (**-23.1%**) | 202 kJ (**-23.2%**) | 200 kJ (**-24.6%**) | 238 kJ (**-16.8%**) |

##### 吞吐量 (tok/s)

| 方案 | Code Medium | Conv Light | Conv Medium | Conv Heavy |
|------|:-:|:-:|:-:|:-:|
| Native DP8 / +Tier | 112 / 112 | 332 / 331 | 598 / 599 | 942 / 943 |
| PD DP4 / +Tier | 112 / 112 | 331 / 333 | 598 / 598 | 942 / 941 |
| PDAF Sym / +Tier | 107 / 106 | 316 / 314 | 572 / **534** | 815 / **619** |
| PDAF Asym / +Tier | 109 / 108 | 323 / 323 | 583 / 578 | 900 / **789** |

##### TPOT (ms, 纯处理, 流式测量)

| 方案 | Code Medium | Conv Light | Conv Medium | Conv Heavy |
|------|:-:|:-:|:-:|:-:|
| Native DP8 / +Tier | 55 / 56 | 56 / 56 | 58 / 58 | 60 / 60 |
| PD DP4 / +Tier | 71 / 69 | 62 / 62 | 62 / 62 | 62 / 62 |
| PDAF Sym / +Tier | 87 / 95 | 87 / 104 | 92 / **137** | 106 / **141** |
| PDAF Asym / +Tier | 75 / 80 | 76 / 77 | 76 / 88 | 89 / 107 |

##### SLO 违反率 (TPOT SLO=250ms)

所有方案均 ≤ 0.2%（修正计算方法后，纯处理 TPOT 远在 SLO 内）。

---

#### 4. 关键发现

##### (1) 异构 PDAF 在重负载下显著优于对称方案

| 指标 | PDAF Sym +Tier (Conv Heavy) | PDAF Asym +Tier (Conv Heavy) | 提升 |
|------|:-:|:-:|:-:|
| 能耗节省 | 5.9% | **16.8%** | +11pp |
| 吞吐量 | 619 tok/s | **789 tok/s** | +27% |
| TPOT | 141 ms | **107 ms** | -24% |

- 对称方案在 Conv Heavy 下因排队严重，Tier 几乎无法降频（只省 5.9%）
- 异构方案 Decode 侧 5GPU (DF TP=4) 增大了 KV cache 容量 → 排队缓解 → Tier 可以安全降频 → 省 16.8%

##### (2) 轻负载下两者等效

Code Medium / Conv Light 下，异构和对称的能耗节省几乎相同（~23%），因为轻负载不触发排队。

##### (3) PDAF Asym 基线吞吐也优于 PDAF Sym

即使不开 Tier，异构方案的吞吐也更高（Conv Heavy: 900 vs 815 tok/s），因为 Decode 侧更多 GPU 提供了更大 batch capacity。

##### (4) PD DP4 + Tier 仍然是最稳定的方案

- 吞吐几乎无损（±0.2%）
- 能耗稳定节省 9-10.5%
- SLO ≤ 0.1%
- 适合对延迟敏感的生产场景

---

#### 5. 方案推荐

| 场景 | 推荐方案 | 节能 | 说明 |
|------|----------|:----:|------|
| 延迟最敏感 | PD DP4 + Tier | 9-10% | 吞吐无损，TPOT 最低 |
| 最大能耗节省（轻-中负载） | PDAF Asym + Tier | 23-25% | TPOT 仍在 SLO 内 |
| 最大能耗节省（重负载） | PDAF Asym + Tier | 17% | 优于 Sym 的 6%，但吞吐损失 12% |
| 不推荐 | PDAF Sym + Tier (重负载) | 6% | 排队严重，吞吐损失 24% |

---

#### 6. 图表

对比图见：`benchmark/AFlex_bench/06_others/more_model/charts/moe_simplified_comparison.png`

包含三个子图：Total Energy、TTFT (Pure Processing)、TPOT (Streaming)，展示四方案 × Tier/NoTier × 4 workloads 的完整对比。
