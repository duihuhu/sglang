# Preliminary Hybrid Motivation Results

> 三次重复；同一 Kimi-Linear-48B-A3B TP=2 server process；结果仅用于 motivation go/no-go。

- Artifacts: 24
- Server version: 0.0.0.dev16772+g0077f84d3.d20260820
- 所有请求成功，所有正式 condition 的 retraction 数为 0。

## M-A: Lifetime Order Sensitivity

| condition | short p95 JCT (s) | short p95 TTFT (s) | overall p95 JCT (s) | <=10s completions |
|---|---:|---:|---:|---:|
| short-first | 4.02 +/- 1.01 | 3.11 +/- 1.00 | 12.06 +/- 1.07 | 96.0 |
| long-first | 4.53 +/- 0.01 | 3.90 +/- 0.01 | 11.26 +/- 0.01 | 96.0 |
| interleave | 5.09 +/- 0.88 | 4.46 +/- 0.88 | 11.77 +/- 0.88 | 96.0 |

- `short-first` 相对 `long-first` 的 short p95 JCT 改善：11.1% (paired bootstrap 95% CI -14.9% to 24.3%)。
- 10s SLO goodput 改善：0.0%。
- Preliminary gate: **FAIL**。

解释：FAIL 不表示调度顺序无影响，而是简单的 short-lifetime-first 没有表现出预期优势。当前结果不支持把 Topic A 直接包装成 SJF/lifetime heuristic；需要先做 state-cap sensitivity 和 resource-time-vs-SJF cross case。

## M-B: Equal Tokens, Unequal State Pressure

| condition | group | p95 JCT (s) | p95 TTFT (s) | peak decode state slots |
|---|---|---:|---:|---:|
| state-isolated | state-heavy | 3.41 | 2.50 | 480 |
| kv-isolated | kv-heavy | 12.56 | 6.58 | 30 |
| state-first | kv-heavy | 15.45 | 9.47 | 30 |
| state-first | state-heavy | 4.58 | 2.60 | 480 |
| kv-first | kv-heavy | 16.59 | 7.56 | 30 |
| kv-first | state-heavy | 11.09 | 10.29 | 455 |
| interleave | kv-heavy | 15.23 | 8.16 | 30 |
| interleave | state-heavy | 9.78 | 8.98 | 455 |

- 两组 completed token service 最大误差：0.00%。
- state-heavy / KV-heavy peak decode state-slot ratio：16.0x。
- 最大 mean p95 JCT slowdown：3.26x。
- state-heavy 的 slowdown 在 `kv-first` 相对 `state-first` 放大：2.42x。
- KV-heavy 在 `state-first` 下的 mean p95 TTFT slowdown：1.46x。
- Preliminary gate: **PASS**。

解释：PASS 只证明 equal-token groups 的物理资源压力与干扰不相等；还没有证明 VTC/DRF baseline 失败。下一阶段必须加入可信 tenant ID、精确 allocator timestamps 和 tuned VTC 对照。

## Decision

- Topic A preliminary gate: **FAIL**。不进入在线 lifetime predictor 实现，先做更便宜的 capacity sensitivity/cross-case。
- Topic B physical-motivation gate: **PASS**。可以进入精确 tenant resource accounting 和 VTC kill test。
- 当前不应把两个 topic 都当作已成立；证据优先支持 Topic B。
