# M2 Pool Comparison

Stage: inspect; complete: False; counts: {'runs': 2, 'requests': 1214, 'warmups': 2, 'warmup_requests': 1214}

Inspect times are diagnostic only.

| Pool | Mode | Cap | N | Mean JCT | Queue | p95 JCT | Makespan | Peak active |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| static-natural | mixed-4096 | 485 | 1 | 22.198 | 7.604 | 27.426 | 28.282 | 485.0 |
| unified-old-cap | mixed-4096 | 485 | 1 | 23.034 | 7.816 | 28.392 | 29.251 | 485.0 |
