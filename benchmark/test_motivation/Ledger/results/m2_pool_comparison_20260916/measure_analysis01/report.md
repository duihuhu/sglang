# M2 Pool Comparison

Stage: measure; complete: True; counts: {'runs': 24, 'requests': 14568, 'warmups': 24, 'warmup_requests': 14568}

Inspect times are diagnostic only.

| Pool | Mode | Cap | N | Mean JCT | Queue | p95 JCT | Makespan | Peak active |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| static-natural | mixed-4096 | 485 | 2 | 22.151 | 7.593 | 27.350 | 28.200 | 485.0 |
| static-natural | separate-4096 | 485 | 2 | 16.474 | 5.684 | 23.145 | 23.674 | 485.0 |
| static-rebalanced | mixed-4096 | 485 | 2 | 22.189 | 7.613 | 27.336 | 28.189 | 485.0 |
| static-rebalanced | mixed-4096 | 607 | 2 | 22.455 | 6.765 | 24.344 | 25.151 | 607.0 |
| static-rebalanced | separate-4096 | 485 | 2 | 16.601 | 5.711 | 23.413 | 23.941 | 485.0 |
| static-rebalanced | separate-4096 | 607 | 2 | 17.666 | 4.100 | 17.889 | 18.029 | 607.0 |
| unified-old-cap | mixed-4096 | 485 | 2 | 23.018 | 7.807 | 28.325 | 29.184 | 485.0 |
| unified-old-cap | separate-4096 | 485 | 2 | 16.960 | 5.837 | 23.817 | 24.346 | 485.0 |
| unified-raised-cap | mixed-4096 | 485 | 2 | 22.947 | 7.786 | 28.297 | 29.160 | 485.0 |
| unified-raised-cap | mixed-4096 | 607 | 2 | 23.252 | 6.907 | 25.314 | 26.127 | 607.0 |
| unified-raised-cap | separate-4096 | 485 | 2 | 16.956 | 5.844 | 23.719 | 24.247 | 485.0 |
| unified-raised-cap | separate-4096 | 607 | 2 | 18.157 | 4.200 | 18.382 | 18.510 | 607.0 |

## Paired Changes

Positive gain is better. Two repeats do not give a precise population CI.

| Mode | Comparison | Mean JCT gain % | p95 gain % | Makespan gain % | Both repeats constrained benefit |
|---|---|---:|---:|---:|---|
| separate-4096 | static-repartition | -7.23 | 22.71 | 23.85 | False |
| separate-4096 | unified-and-cap | -10.21 | 20.58 | 21.81 | False |
| separate-4096 | unified-switch-only | -2.95 | -2.90 | -2.84 | False |
| separate-4096 | unified-vs-rebalanced-607 | -2.78 | -2.76 | -2.67 | False |
| separate-4096 | unified-vs-rebalanced-485 | -2.14 | -1.31 | -1.28 | False |
| separate-4096 | static-cap-relaxation | -6.42 | 23.59 | 24.70 | False |
| separate-4096 | unified-cap-relaxation | -7.09 | 22.50 | 23.66 | False |
| mixed-4096 | static-repartition | -1.37 | 10.99 | 10.81 | False |
| mixed-4096 | unified-and-cap | -4.97 | 7.44 | 7.35 | False |
| mixed-4096 | unified-switch-only | -3.91 | -3.57 | -3.49 | False |
| mixed-4096 | unified-vs-rebalanced-607 | -3.55 | -3.99 | -3.88 | False |
| mixed-4096 | unified-vs-rebalanced-485 | -3.42 | -3.52 | -3.44 | False |
| mixed-4096 | static-cap-relaxation | -1.20 | 10.94 | 10.78 | False |
| mixed-4096 | unified-cap-relaxation | -1.33 | 10.54 | 10.40 | False |
