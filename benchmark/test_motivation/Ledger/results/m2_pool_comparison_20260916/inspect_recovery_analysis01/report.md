# M2 Pool Comparison

Stage: inspect; complete: True; counts: {'runs': 2, 'requests': 1214, 'warmups': 2, 'warmup_requests': 1214}

Inspect times are diagnostic only.

| Pool | Mode | Cap | N | Mean JCT | Queue | p95 JCT | Makespan | Peak active |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| static-rebalanced | mixed-4096 | 607 | 1 | 22.753 | 6.827 | 24.703 | 25.508 | 607.0 |
| unified-raised-cap | mixed-4096 | 607 | 1 | 23.418 | 6.923 | 25.499 | 26.314 | 607.0 |
