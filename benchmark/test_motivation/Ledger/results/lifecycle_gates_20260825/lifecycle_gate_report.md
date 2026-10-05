# Lifecycle-cost motivation gate results

Artifacts analyzed: 120

| gate | decision |
|---|---|
| M1 | PASS |
| M2 | FAIL |
| M3 | FAIL |
| M4 | PASS |
| M5 | FAIL |

## M1: token/length ranking

Decision: **PASS**. Comparable pairs: 324; >5% inversions: 44 (13.6%); largest inversion: 10.06x.

| expensive lower-token class | cheaper higher-token class | p95 ratio | stable across reps |
|---|---|---:|---|
| P=1024, O=512, reuse=0.0 | P=4096, O=32, reuse=0.9 | 10.06x | yes |
| P=1024, O=512, reuse=0.5 | P=4096, O=32, reuse=0.9 | 9.70x | yes |
| P=1024, O=512, reuse=0.9 | P=4096, O=32, reuse=0.9 | 9.43x | yes |
| P=1024, O=512, reuse=0.0 | P=4096, O=32, reuse=0.5 | 5.29x | yes |
| P=1024, O=512, reuse=0.5 | P=4096, O=32, reuse=0.5 | 5.11x | yes |
| P=1024, O=512, reuse=0.9 | P=4096, O=32, reuse=0.5 | 4.96x | yes |
| P=1024, O=512, reuse=0.0 | P=4096, O=32, reuse=0.0 | 4.26x | yes |
| P=1024, O=512, reuse=0.5 | P=4096, O=32, reuse=0.0 | 4.11x | yes |
| P=1024, O=512, reuse=0.9 | P=4096, O=32, reuse=0.0 | 4.00x | yes |
| P=1024, O=512, reuse=0.0 | P=4096, O=128, reuse=0.9 | 3.55x | yes |

## M2: state-dependent ranking

Decision: **FAIL**. State/KV best-order reversal: yes; best fixed order worst-regime regret: 2.0%.

| regime | decode first p95 | prefill first p95 | interleave p95 | best |
|---|---:|---:|---:|---|
| compute-bound | 25.913 | 24.539 | 25.424 | prefill-first |
| kv-bound | 25.268 | 24.759 | 25.447 | prefill-first |
| state-bound | 28.292 | 28.850 | 29.245 | decode-first |

## M3: admission/batch/eviction coupling

Decision: **FAIL**. FCFS reuse-first p95 gain: 3.3%; cached-token delta: 1824; residual gain over pressure-first LPM: 4.6%.

| server policy | emulated order | overall p95 | background p95 | reuse cached tokens | makespan |
|---|---|---:|---:|---:|---:|
| fcfs | interleave | 31.483 | 31.484 | 12448 | 31.506 |
| fcfs | pressure-first | 32.000 | 32.002 | 10432 | 32.023 |
| fcfs | reuse-first | 30.947 | 30.948 | 12256 | 30.969 |
| lpm | interleave | 30.867 | 30.867 | 12208 | 30.889 |
| lpm | pressure-first | 32.456 | 32.457 | 10432 | 32.478 |
| lpm | reuse-first | 30.798 | 30.798 | 12256 | 30.820 |

## M4: equal-token tenant cost transfer

Decision: **PASS**. Requested tokens: split=33792, packed=33792; maximum per-token resource ratio: 6.60x; VTC normalized-slowdown gap: 101.5%.

| tenant shape | isolated p95 | mixed VTC p95 | normalized slowdown | state-slot-s/token | KV-token-s/token |
|---|---:|---:|---:|---:|---:|
| split | 1.725 | 3.914 | 126.8% | 0.002107 | 0.726 |
| packed | 5.554 | 6.962 | 25.3% | 0.000891 | 4.794 |

## M5: empirical oracle headroom

Decision: **FAIL**. Orders completed: 24/24; oracle=decode,medium,reuse,prefill; strongest baseline=future-memory; p95 headroom=2.5%; throughput bound=pass.

| policy/order | class order | overall p95 | SLO goodput | makespan |
|---|---|---:|---:|---:|
| fcfs | decode,prefill,reuse,medium | 21.341 | 28.0 | 21.345 |
| sjf | prefill,reuse,medium,decode | 22.585 | 28.0 | 22.639 |
| lpm | reuse,decode,prefill,medium | 21.896 | 28.0 | 22.099 |
| future-memory | decode,medium,prefill,reuse | 20.754 | 28.0 | 20.757 |
| composite | reuse,decode,medium,prefill | 21.268 | 28.0 | 21.470 |
| empirical oracle | decode,medium,reuse,prefill | 20.226 | 28.0 | 20.229 |

