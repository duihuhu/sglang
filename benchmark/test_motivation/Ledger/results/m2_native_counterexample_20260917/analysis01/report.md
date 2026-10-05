# M2 Native Calibration

Complete: True; {'runs': 12, 'requests': 6912, 'warmup_requests': 6912}

| Phase | Policy | Rep | Mean JCT | P95 JCT | Makespan | New TTFT | Slot delays | Request-limit blocks |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| far | cap512 | 61 | 17.932 | 33.928 | 42.596 | 19.256 | 0 | 1 |
| near | cap512 | 61 | 17.390 | 28.828 | 42.650 | 14.127 | 0 | 1 |
| far | native607 | 61 | 38.443 | 39.927 | 40.036 | 11.889 | 0 | 0 |
| near | native607 | 61 | 38.157 | 39.671 | 41.376 | 11.904 | 0 | 0 |
| far | slots96 | 61 | 17.952 | 33.909 | 42.578 | 19.281 | 113 | 0 |
| near | slots96 | 61 | 17.364 | 28.753 | 42.575 | 14.113 | 34 | 0 |
| far | slots96 | 62 | 18.048 | 34.106 | 42.775 | 19.378 | 113 | 0 |
| near | slots96 | 62 | 17.485 | 28.942 | 42.765 | 14.233 | 35 | 0 |
| far | cap512 | 62 | 18.076 | 34.041 | 42.708 | 19.406 | 0 | 1 |
| near | cap512 | 62 | 17.471 | 28.901 | 42.724 | 14.222 | 0 | 1 |
| far | native607 | 62 | 38.577 | 40.069 | 40.178 | 11.920 | 0 | 0 |
| near | native607 | 62 | 38.263 | 39.790 | 41.383 | 11.888 | 0 | 0 |

Positive percent means improvement; two exploratory repeats, not confidence intervals.

| Phase | Policy | Mean gain % | P95 gain % | Makespan gain % | New mean TTFT gain % | Both constrained/triggered |
|---|---|---:|---:|---:|---:|---|
| far | cap512 | 53.25 | 15.04 | -6.35 | -62.38 | False |
| far | slots96 | 53.26 | 14.98 | -6.41 | -62.37 | False |
| near | cap512 | 54.38 | 27.35 | -3.16 | -19.15 | True |
| near | slots96 | 54.40 | 27.39 | -3.12 | -19.14 | True |
