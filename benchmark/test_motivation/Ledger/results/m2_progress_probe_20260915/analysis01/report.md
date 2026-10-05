# M2 natural-capacity progress probe

Complete: True; runs: 20

All slot gates corrected; activation limit off. Unselected time is a CPU-launch proxy, not wasted GPU time.

| Mode | Shape | Rep | Mean JCT s | Peak active | State slot-s | Decode-ready unselected slot-s | With pending slot-s |
|---|---|---:|---:|---:|---:|---:|---:|
| mixed-4096 | long | 1 | 12.926 | 16 | 158.526 | 0.000 | 0.000 |
| mixed-4096 | long | 2 | 13.014 | 16 | 159.716 | 0.000 | 0.000 |
| mixed-4096 | medium | 1 | 10.753 | 64 | 544.304 | 0.000 | 0.000 |
| mixed-4096 | medium | 2 | 10.735 | 64 | 543.801 | 0.000 | 0.000 |
| mixed-4096 | mixed | 1 | 12.177 | 136 | 1241.934 | 0.000 | 0.000 |
| mixed-4096 | mixed | 2 | 12.357 | 136 | 1267.043 | 0.000 | 0.000 |
| mixed-4096 | short | 1 | 11.383 | 256 | 2336.951 | 0.000 | 0.000 |
| mixed-4096 | short | 2 | 11.282 | 256 | 2317.560 | 0.000 | 0.000 |
| mixed-4096 | state-pressure | 1 | 22.360 | 485 | 8941.391 | 0.000 | 0.000 |
| mixed-4096 | state-pressure | 2 | 22.051 | 485 | 8775.643 | 0.000 | 0.000 |
| separate-4096 | long | 1 | 12.557 | 16 | 158.345 | 39.018 | 36.284 |
| separate-4096 | long | 2 | 12.610 | 16 | 158.644 | 39.013 | 36.287 |
| separate-4096 | medium | 1 | 10.387 | 64 | 550.224 | 110.435 | 106.294 |
| separate-4096 | medium | 2 | 10.506 | 64 | 558.750 | 110.109 | 105.851 |
| separate-4096 | mixed | 1 | 11.486 | 136 | 1227.764 | 277.469 | 269.080 |
| separate-4096 | mixed | 2 | 11.558 | 136 | 1239.131 | 278.056 | 269.675 |
| separate-4096 | short | 1 | 10.679 | 256 | 2280.855 | 433.302 | 416.453 |
| separate-4096 | short | 2 | 10.872 | 256 | 2336.168 | 432.213 | 414.972 |
| separate-4096 | state-pressure | 1 | 16.442 | 485 | 6497.739 | 1746.805 | 1739.337 |
| separate-4096 | state-pressure | 2 | 16.493 | 485 | 6546.029 | 1738.344 | 1730.737 |

Different shapes have different output work; their absolute latencies are not causal state-cost comparisons.
Two restarts per configuration are exploratory. No confirmatory CI, winner selection, or SLO/fairness claim.
