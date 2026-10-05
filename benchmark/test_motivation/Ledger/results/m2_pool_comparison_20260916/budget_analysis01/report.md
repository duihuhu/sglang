# M2 Pool Comparison

Stage: inspect; complete: True; counts: {'runs': 4, 'requests': 2428, 'warmups': 4, 'warmup_requests': 2428}

Inspect times are diagnostic only.

| Pool | Mode | Cap | N | Mean JCT | Queue | p95 JCT | Makespan | Peak active |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| static-natural | mixed-4096 | 485 | 1 | 22.168 | 7.591 | 27.373 | 28.221 | 485.0 |
| static-rebalanced | mixed-4096 | 607 | 1 | 22.508 | 6.759 | 24.458 | 25.264 | 607.0 |
| unified-old-cap | mixed-4096 | 485 | 1 | 23.116 | 7.804 | 28.586 | 29.454 | 485.0 |
| unified-raised-cap | mixed-4096 | 607 | 1 | 23.338 | 6.900 | 25.424 | 26.237 | 607.0 |
