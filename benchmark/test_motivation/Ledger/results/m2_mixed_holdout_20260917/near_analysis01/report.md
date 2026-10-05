# M2 Mixed Near-Only Screen

Complete: True; {'runs': 16, 'requests': 8192, 'warmups': 16, 'warmup_requests': 8192}

Training-only selection: {'policy': 'native607', 'training_scores': {'native607': 1.0}, 'selection_uses_held_out': False, 'scope': 'near only'}

| Profile | Policy | Rep | Mean JCT | P95 | Makespan | New TTFT | Alive | Remaining | Triggered | Scene errors |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| hold384 | cap512 | 81 | 36.215 | 50.290 | 68.891 | 23.001 | 320 | 42.0 | False | [] |
| train512 | cap512 | 81 | 29.454 | 40.768 | 61.612 | 22.945 | 488 | 32.0 | False | [] |
| hold384 | native607 | 81 | 37.657 | 52.238 | 69.947 | 23.679 | 336 | 44.0 | True | [] |
| train512 | native607 | 81 | 29.431 | 40.735 | 61.559 | 22.916 | 488 | 32.0 | True | [] |
| hold384 | slots96 | 81 | 36.169 | 50.231 | 68.772 | 22.948 | 320 | 42.0 | False | [] |
| train512 | slots96 | 81 | 29.872 | 41.115 | 62.021 | 23.131 | 496 | 32.5 | False | [] |
| hold384 | decode32 | 81 | 29.793 | 39.893 | 65.376 | 20.888 | 320 | 42.0 | True | [] |
| train512 | decode32 | 81 | 22.368 | 36.861 | 57.750 | 19.959 | 480 | 33.0 | True | [] |
| hold384 | decode32 | 82 | 29.817 | 39.974 | 65.498 | 20.904 | 320 | 42.0 | True | [] |
| train512 | decode32 | 82 | 22.566 | 37.061 | 57.979 | 20.088 | 488 | 33.0 | True | [] |
| hold384 | slots96 | 82 | 35.893 | 49.871 | 68.631 | 22.884 | 316 | 41.5 | False | [] |
| train512 | slots96 | 82 | 29.521 | 40.836 | 61.655 | 23.011 | 488 | 33.0 | False | [] |
| hold384 | cap512 | 82 | 35.928 | 49.897 | 68.595 | 22.919 | 316 | 42.5 | False | [] |
| train512 | cap512 | 82 | 29.444 | 40.745 | 61.622 | 22.926 | 488 | 32.0 | False | [] |
| hold384 | native607 | 82 | 35.860 | 49.812 | 68.540 | 22.836 | 316 | 41.5 | True | [] |
| train512 | native607 | 82 | 29.458 | 40.765 | 61.633 | 22.942 | 488 | 32.0 | True | [] |

Positive percentage means improvement; two repeats are not confidence intervals.

| Profile | Reference | Treatment | Mean gain % | P95 gain % | Makespan gain % | New TTFT gain % | Both valid/triggered/constrained |
|---|---|---|---:|---:|---:|---:|---|
| train512 | native607 | cap512 | -0.02 | -0.02 | -0.03 | -0.03 | False |
| train512 | native607 | slots96 | -0.86 | -0.55 | -0.39 | -0.62 | False |
| train512 | native607 | decode32 | 23.70 | 9.30 | 6.06 | 12.67 | False |
| train512 | cap512 | decode32 | 23.71 | 9.31 | 6.09 | 12.70 | False |
| train512 | slots96 | decode32 | 24.34 | 9.80 | 6.42 | 13.21 | False |
| hold384 | native607 | cap512 | 1.82 | 1.78 | 0.71 | 1.25 | False |
| hold384 | native607 | slots96 | 1.93 | 1.86 | 0.77 | 1.44 | False |
| hold384 | native607 | decode32 | 18.87 | 21.69 | 5.49 | 10.13 | False |
| hold384 | cap512 | decode32 | 17.37 | 20.28 | 4.81 | 8.99 | False |
| hold384 | slots96 | decode32 | 17.28 | 20.21 | 4.75 | 8.82 | False |
