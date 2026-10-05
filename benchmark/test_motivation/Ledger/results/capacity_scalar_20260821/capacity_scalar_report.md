# Topic B Capacity x Scalar-Weight Cross Experiment

> The screening phase uses one repetition. It identifies bottleneck regimes and candidate weights; it is not statistical evidence until finalists are repeated.

## Capacity calibration

| capacity | actual state cap | actual KV cap | state admission util. | KV util. | normalized pressure state/KV | regime | FCFS retractions |
|---|---:|---:|---:|---:|---:|---|---:|
| s240-k196608 | 240 | 196,608 | 1.000 | 0.989 | 0.429 | dual-bound | 0.0 |
| s240-k393216 | 240 | 393,216 | 1.000 | 0.573 | 0.916 | state-bound | 0.0 |
| s240-k786432 | 240 | 786,432 | 1.000 | 0.286 | 1.832 | state-bound | 0.0 |
| s480-k196608 | 480 | 196,608 | 1.000 | 0.988 | 0.277 | dual-bound | 0.0 |
| s480-k393216 | 480 | 393,216 | 1.000 | 0.688 | 0.650 | state-bound | 0.0 |
| s480-k786432 | 480 | 786,432 | 1.000 | 0.344 | 1.301 | state-bound | 0.0 |
| s720-k196608 | 720 | 196,608 | 0.701 | 0.988 | 0.182 | KV-bound | 0.0 |
| s720-k393216 | 720 | 393,216 | 1.000 | 0.699 | 0.502 | state-bound | 0.0 |
| s720-k786432 | 720 | 786,432 | 1.000 | 0.350 | 1.003 | state-bound | 0.0 |

## Per-capacity oracle comparison

A scalar candidate is eligible only when its mean makespan is no more than 5% above FCFS. The gap below is scale-free: `|d_state-d_KV| / mean(d_state,d_KV)`.

| capacity | regime | best eligible scalar | scalar rel. gap | vector rel. gap | vector vs oracle | scalar makespan vs FCFS | vector makespan vs FCFS |
|---|---|---:|---:|---:|---:|---:|---:|
| s240-k196608 | dual-bound | 4096 | 0.835 | 1.003 | -20.1% | -0.2% | -1.0% |
| s240-k393216 | state-bound | 0 | 0.430 | 0.460 | -7.1% | -0.9% | -2.2% |
| s240-k786432 | state-bound | 262144 | 0.024 | 0.052 | -120.1% | -3.9% | -6.8% |
| s480-k196608 | dual-bound | 0 | 1.195 | 1.177 | +1.5% | +0.2% | -1.6% |
| s480-k393216 | state-bound | 0 | 0.510 | 0.529 | -3.7% | -0.6% | -3.7% |
| s480-k786432 | state-bound | 1024 | 0.079 | 0.061 | +23.2% | -0.6% | -4.1% |
| s720-k196608 | KV-bound | 4096 | 1.201 | 1.192 | +0.8% | +3.9% | -0.5% |
| s720-k393216 | state-bound | 0 | 0.597 | 0.616 | -3.2% | +1.0% | -1.8% |
| s720-k786432 | state-bound | 1024 | 0.028 | 0.121 | -330.2% | +1.7% | -1.6% |

## Fixed scalar robustness

Best fixed scalar weight across all completed cells: `0`; mean relative gap 0.623, worst-cell gap 1.306, mean regret versus per-cell oracle 0.078.

## Aggregate checks

- Mean per-cell oracle relative gap: 0.544; mean vector gap: 0.579; vector change: -6.4%.
- Vector wins by at least 10% in 1 cell across 1 regime: state-bound.
- Total retractions across all analyzed policy runs: 3.

## Decision

**Kill condition:** a throughput-bounded per-capacity scalar oracle matches or beats vector. Do not claim that vector accounting is necessary for this workload.
