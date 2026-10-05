# Topic B Sustained Policy Kill Test

> 12 waves per run, 16 state-heavy plus 1 KV-heavy request per wave, 3 repetitions, identical Kimi-Linear TP=2 capacity and arrival trace.

| policy | state p95 JCT (s) | KV p95 JCT (s) | <=20s goodput | makespan (s) | vs FCFS makespan | max token lag | dominant-debt gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| fcfs | 14.18 +/- 0.02 | 26.67 +/- 0.02 | 194.0 | 27.65 +/- 0.02 | +0.0% | 22,945 | 3.60 +/- 0.00 |
| vtc-token | 15.12 +/- 0.01 | 26.52 +/- 0.01 | 194.0 | 27.31 +/- 0.01 | -1.2% | 15,345 | 3.44 +/- 0.00 |
| vtc-r8076 | 16.66 +/- 0.03 | 26.92 +/- 0.03 | 194.0 | 27.34 +/- 0.03 | -1.1% | 159,378 | 1.41 +/- 0.00 |
| vtc-r13460 | 15.76 +/- 0.02 | 26.01 +/- 0.03 | 195.0 | 26.44 +/- 0.03 | -4.4% | 155,620 | 3.56 +/- 0.01 |
| vector | 15.75 +/- 0.01 | 26.00 +/- 0.01 | 195.0 | 26.43 +/- 0.01 | -4.4% | 180,220 | 0.82 +/- 0.00 |

## Findings

- Best scalar baseline: `vtc-r8076`, dominant-debt gap 1.41.
- Vector debt gap: 0.82, 41.6% below the best tested scalar VTC.
- Vector makespan change versus FCFS: -4.4%; <=20s goodput change: +1.0 requests.
- Token VTC keeps token-service lag low, while vector debt intentionally permits token lag to equalize normalized physical resource use. These are different fairness objectives.
- Scalar tuning is sensitive: a physical-slot conversion (8,076) is materially better than token-only VTC, while the 5-slot admission conversion (13,460) regresses close to FCFS on debt gap.

## Decision

This is a **partial pass**, not a paper-level pass. At the current capacity point vector debt beats all tested scalar VTC settings without throughput loss, but GPU-time accounting, broader scalar tuning, state/KV bottleneck sweeps, request-splitting resistance, and a fairness-efficiency frontier remain required.
