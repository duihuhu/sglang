# M2 activation pacing: exploratory results

Runs: 48; requests: 3840; complete=True

All gates are reuse-aware. Positive reduction means a smaller measured value, not automatically a service improvement.

| Regime / batching / order | New limit | Paired runs | All mean JCT reduction | Short mean reduction | Long mean reduction | Makespan reduction | Pacing events/run |
|---|---:|---:|---:|---:|---:|---:|---:|
| state8 / mixed-1024 / long-first | 1 | 2 | -5.39% | -6.13% | -3.65% | -2.47% | 38.0 |
| state8 / mixed-1024 / long-first | 4 | 2 | -0.96% | -0.97% | -0.93% | -1.09% | 0.0 |
| state8 / mixed-1024 / short-first | 1 | 2 | -1.05% | -1.20% | -0.71% | -0.38% | 33.0 |
| state8 / mixed-1024 / short-first | 4 | 2 | +2.09% | +2.13% | +2.01% | +1.83% | 0.0 |
| state8 / mixed-4096 / long-first | 1 | 2 | -3.72% | -4.04% | -2.99% | -1.86% | 37.0 |
| state8 / mixed-4096 / long-first | 4 | 2 | -1.06% | -1.08% | -1.01% | -0.74% | 5.0 |
| state8 / mixed-4096 / short-first | 1 | 2 | -3.59% | -3.94% | -2.86% | -1.66% | 33.0 |
| state8 / mixed-4096 / short-first | 4 | 2 | -1.33% | -1.48% | -1.03% | -0.83% | 5.0 |
| state8 / separate-4096 / long-first | 1 | 2 | -11.03% | -11.67% | -9.52% | -8.09% | 57.0 |
| state8 / separate-4096 / long-first | 4 | 2 | -1.59% | -1.67% | -1.41% | -1.19% | 9.0 |
| state8 / separate-4096 / short-first | 1 | 2 | -11.43% | -12.29% | -9.60% | -8.04% | 55.0 |
| state8 / separate-4096 / short-first | 4 | 2 | -2.18% | -2.34% | -1.86% | -1.55% | 9.0 |
| state8 / unchunked / long-first | 1 | 2 | -12.08% | -12.65% | -10.71% | -9.19% | 69.0 |
| state8 / unchunked / long-first | 4 | 2 | -1.08% | -1.20% | -0.79% | -0.60% | 9.0 |
| state8 / unchunked / short-first | 1 | 2 | -16.23% | -16.57% | -15.50% | -13.65% | 69.0 |
| state8 / unchunked / short-first | 4 | 2 | -1.41% | -1.49% | -1.24% | -1.03% | 9.0 |

## Limits

Screening averages are not independent confirmation; report every cell, including regressions and inactive interventions.
Pacing events precede the remaining candidate budgets: they do not prove feasible admissions were rejected.
Nearest-rank p95, per-group queues/lifetimes, TPOT, state/KV integrals and counter gaps are in analysis.json.
VTC counter gaps include lifts and unconditioned backlog; they are not violations of its fairness guarantee.
Finite-wave completion is not steady-state goodput. No SLO thresholds were inferred from intervention results.
Native prefill-size and free-slot-delay controls, natural pressure, held-out traces and full literature baselines remain separate stages.
