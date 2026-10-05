# Dynamic request-cost motivation results

Artifacts analyzed: 108

## Gate decisions

| gate | decision | measured evidence |
|---|---|---|
| Request cost is non-monotone in raw length | PASS | `1K+512` mean JCT 2.608s vs `16K+32` 1.218s; 90% reuse cuts concurrent `16K+32` p95 JCT from 8.789s to 2.077s. |
| Batch externality depends on incremental cost | PASS | long-unique background p95 JCT increase 70.4%, paired 95% CI [69.1%, 71.5%]; background SLO goodput 0/0/0/0/0 in all reps. long-reused increase 13.2%, CI [12.8%, 13.6%]. |
| Cache state lifetime changes recompute cost | PASS | KV pressure evicts the 16K prefix in 5/5 reps and raises probe TTFT by 25.2x, paired 95% CI [24.9x, 25.5x]; state pressure preserves it in 5/5 reps. |
| A simple request-order rule yields >=15% gain | FAIL | Best tested lifetime order improves overall p95 JCT by only 5.9%; state-KV order effects are smaller, and reuse right-first is worse. |
| Phase-specific scalar retuning has material regret | FAIL | Isolated scalar sweeps change phase latency gaps by less than 4%; no fixed weight incurs the preregistered 15% regret, so switch/confirm is not triggered. |

## Request cost surface

| P | O | N | reuse | reps | cached | mean TTFT (s) | p95 JCT (s) | state-slot-s | KV-token-s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 32 | 1 | 0% | 3 | 0 | 0.052 | 0.213 | 0.626 | 216 |
| 1024 | 32 | 16 | 0% | 3 | 0 | 0.373 | 0.854 | 34.150 | 11766 |
| 1024 | 512 | 1 | 0% | 3 | 0 | 0.049 | 2.608 | 7.812 | 3324 |
| 1024 | 512 | 16 | 0% | 3 | 0 | 0.369 | 7.062 | 332.141 | 140693 |
| 4096 | 32 | 1 | 0% | 3 | 0 | 0.140 | 0.298 | 0.878 | 1201 |
| 16384 | 32 | 1 | 0% | 3 | 0 | 1.053 | 1.218 | 3.632 | 17715 |
| 16384 | 32 | 8 | 0% | 3 | 0 | 4.783 | 8.789 | 122.761 | 653682 |
| 16384 | 32 | 8 | 90% | 3 | 14720 | 1.334 | 2.077 | 38.606 | 210766 |
| 16384 | 512 | 1 | 0% | 3 | 0 | 1.054 | 3.793 | 11.355 | 60591 |
| 16384 | 512 | 8 | 90% | 3 | 14720 | 1.335 | 7.988 | 180.436 | 998253 |

## Rank inversions

| pair | order | reps | makespan (s) | overall p95 JCT (s) | SLO goodput | left p95 JCT | right p95 JCT | retractions |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| lifetime | interleave | 3 | 18.279 | 18.264 | 96.0 | 18.268 | 8.574 | 0.0 |
| lifetime | left-first | 3 | 18.665 | 18.654 | 96.0 | 18.656 | 8.886 | 0.0 |
| lifetime | right-first | 3 | 17.605 | 17.547 | 96.0 | 17.550 | 7.678 | 0.0 |
| reuse | interleave | 3 | 17.215 | 17.197 | 96.0 | 17.197 | 17.196 | 0.0 |
| reuse | left-first | 3 | 16.405 | 16.388 | 96.0 | 16.390 | 16.348 | 0.0 |
| reuse | right-first | 3 | 17.607 | 17.588 | 96.0 | 17.537 | 17.592 | 0.0 |
| state-kv | interleave | 3 | 15.215 | 14.878 | 102.0 | 9.769 | 15.209 | 0.0 |
| state-kv | left-first | 3 | 15.517 | 14.535 | 102.0 | 4.599 | 15.419 | 0.0 |
| state-kv | right-first | 3 | 15.207 | 14.946 | 102.0 | 9.799 | 15.205 | 0.0 |

## Batch externality

| candidate | reps | background p95 JCT (s) | background p95 ITL (s) | background max ITL (s) | candidate TTFT (s) | cached |
|---|---:|---:|---:|---:|---:|---:|
| long-reused | 5 | 15.030 | 0.0240 | 3.3052 | 2.371 | 14720 |
| long-unique | 5 | 22.615 | 0.0241 | 10.9335 | 6.300 | 0 |
| none | 5 | 13.274 | 0.0229 | 1.6392 | - | - |
| short-unique | 5 | 13.519 | 0.0231 | 1.8552 | 1.192 | 0 |

## Cache lifetime and eviction

| pressure | reps | cached before | cached after | TTFT before (s) | TTFT after (s) | pressure p95 JCT (s) |
|---|---:|---:|---:|---:|---:|---:|
| kv | 5 | 16384 | 0 | 0.043 | 1.090 | 17.547 |
| none | 5 | 16384 | 16384 | 0.044 | 0.043 | - |
| state | 5 | 16384 | 16384 | 0.043 | 0.045 | 14.730 |

## Dynamic phase switch

| pattern | policy | weight/schedule | reps | mean phase gap | max phase gap | final resource gap | makespan (s) | p95 JCT (s) | SLO goodput |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| kv | fcfs | 0 | 1 | 0.284 | 0.284 | 0.387 | 11.682 | 10.968 | 20.0 |
| kv | scalar | 0 | 1 | 0.286 | 0.286 | 0.450 | 11.467 | 10.753 | 20.0 |
| kv | scalar | 1024 | 1 | 0.278 | 0.278 | 0.444 | 11.802 | 11.088 | 20.0 |
| kv | scalar | 16384 | 1 | 0.279 | 0.279 | 0.528 | 11.640 | 10.926 | 20.0 |
| kv | scalar | 262144 | 1 | 0.280 | 0.280 | 0.536 | 11.819 | 11.105 | 20.0 |
| kv | scalar | 4096 | 1 | 0.281 | 0.281 | 0.511 | 11.705 | 10.990 | 20.0 |
| kv | scalar | 65536 | 1 | 0.284 | 0.284 | 0.549 | 11.753 | 11.039 | 20.0 |
| kv | vector | 0 | 1 | 0.287 | 0.287 | 0.348 | 11.585 | 10.871 | 20.0 |
| state | fcfs | 0 | 1 | 0.372 | 0.372 | 0.505 | 11.043 | 8.116 | 68.0 |
| state | scalar | 0 | 1 | 0.383 | 0.383 | 0.534 | 10.958 | 8.001 | 68.0 |
| state | scalar | 1024 | 1 | 0.376 | 0.376 | 0.637 | 11.085 | 8.139 | 68.0 |
| state | scalar | 16384 | 1 | 0.369 | 0.369 | 0.691 | 11.090 | 8.169 | 68.0 |
| state | scalar | 262144 | 1 | 0.373 | 0.373 | 0.696 | 11.194 | 8.246 | 68.0 |
| state | scalar | 4096 | 1 | 0.379 | 0.379 | 0.679 | 11.104 | 8.145 | 68.0 |
| state | scalar | 65536 | 1 | 0.373 | 0.373 | 0.696 | 11.219 | 8.268 | 68.0 |
| state | vector | 0 | 1 | 0.368 | 0.368 | 0.384 | 11.214 | 8.284 | 68.0 |

