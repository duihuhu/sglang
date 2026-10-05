# M2 fixed-workload state-cap mediation

Both stages complete: True; measured runs: 26

Every measured run uses 607 P512/O128 requests. All configured cells are retained.

| Pool | Mode | Cap | Rep | Mean JCT s | Queue s | Makespan s | State slot-s | Peak active | Peak KV % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| natural | mixed-1024 | 128 | 11 | 33.999 | 23.326 | 56.436 | 6472.336 | 128 | 5.32 |
| natural | mixed-1024 | 128 | 12 | 38.347 | 27.418 | 60.393 | 6627.890 | 128 | 5.32 |
| natural | mixed-1024 | 256 | 11 | 46.467 | 25.612 | 69.243 | 12639.858 | 229 | 9.07 |
| natural | mixed-1024 | 256 | 12 | 32.285 | 17.317 | 53.963 | 9076.392 | 229 | 9.07 |
| natural | mixed-1024 | 485 | 11 | 34.816 | 20.832 | 54.390 | 8478.485 | 229 | 9.07 |
| natural | mixed-1024 | 485 | 12 | 43.289 | 25.439 | 61.196 | 10818.478 | 229 | 9.07 |
| natural | mixed-4096 | 128 | 11 | 40.530 | 26.120 | 73.719 | 8734.069 | 128 | 5.54 |
| natural | mixed-4096 | 128 | 12 | 25.398 | 17.068 | 43.413 | 5051.603 | 128 | 5.54 |
| natural | mixed-4096 | 256 | 11 | 20.566 | 10.159 | 31.595 | 6307.229 | 256 | 10.94 |
| natural | mixed-4096 | 256 | 12 | 20.815 | 10.280 | 31.987 | 6387.443 | 256 | 10.94 |
| natural | mixed-4096 | 485 | 11 | 29.464 | 9.259 | 40.673 | 12232.492 | 485 | 20.22 |
| natural | mixed-4096 | 485 | 12 | 22.005 | 7.553 | 27.937 | 8760.734 | 485 | 20.22 |
| natural | separate-4096 | 128 | 11 | 34.862 | 24.347 | 58.311 | 6365.997 | 128 | 5.62 |
| natural | separate-4096 | 128 | 12 | 24.870 | 17.063 | 43.079 | 4730.286 | 128 | 5.62 |
| natural | separate-4096 | 256 | 11 | 18.135 | 9.324 | 29.885 | 5332.651 | 256 | 11.23 |
| natural | separate-4096 | 256 | 12 | 18.198 | 9.356 | 29.959 | 5351.898 | 256 | 11.23 |
| natural | separate-4096 | 485 | 11 | 18.727 | 5.818 | 33.804 | 7804.361 | 485 | 21.28 |
| natural | separate-4096 | 485 | 12 | 16.415 | 5.688 | 23.500 | 6481.737 | 485 | 21.28 |
| wide | mixed-4096 | 485 | 11 | 22.042 | 7.567 | 28.132 | 8775.210 | 485 | 36.66 |
| wide | mixed-4096 | 485 | 12 | 22.055 | 7.557 | 28.214 | 8788.858 | 485 | 36.66 |
| wide | mixed-4096 | 607 | 11 | 22.535 | 6.758 | 25.226 | 9564.863 | 607 | 45.24 |
| wide | mixed-4096 | 607 | 12 | 22.474 | 6.757 | 25.130 | 9527.829 | 607 | 45.24 |
| wide | separate-4096 | 485 | 11 | 16.516 | 5.695 | 23.713 | 6540.900 | 485 | 38.58 |
| wide | separate-4096 | 485 | 12 | 16.428 | 5.672 | 23.620 | 6499.619 | 485 | 38.58 |
| wide | separate-4096 | 607 | 11 | 17.621 | 4.077 | 17.973 | 8178.231 | 607 | 48.29 |
| wide | separate-4096 | 607 | 12 | 17.628 | 4.106 | 17.978 | 8165.743 | 607 | 48.29 |

## Prespecified JCT Contrast

D = Mixed-4K minus Separate-4K. Delta = D(wide,485) minus D(wide,607). Positive Delta supports amplification by the tighter cap; negative means the opposite.

| Rep | D natural485, s | D wide485, s | D wide607, s | Delta, s |
|---|---:|---:|---:|---:|
| 11 | 10.737 | 5.526 | 4.915 | 0.612 |
| 12 | 5.590 | 5.628 | 4.847 | 0.781 |

Only two independent starts per configuration. No confirmatory CI or post-hoc winner claim.
A/B pool comparisons are cross-session; B/C cap interventions share the same physical pool. No GPU-kernel attribution or hybrid exclusivity is implied.
