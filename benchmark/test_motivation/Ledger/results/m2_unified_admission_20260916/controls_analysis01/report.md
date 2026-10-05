# M2 Unified Admission

Complete: True; {'runs': 32, 'requests': 11776, 'warmup_requests': 11776}

| Profile | Phase | Policy | Rep | Mean JCT | P95 | Makespan | Control delays | Scene errors |
|---|---|---|---:|---:|---:|---:|---:|---|
| kv | far | native | 51 | 96.009 | 144.737 | 145.467 | 0 | [] |
| kv | near | native | 51 | 94.116 | 138.997 | 148.213 | 0 | [] |
| state | far | native | 51 | 38.184 | 39.659 | 39.765 | 0 | [] |
| state | near | native | 51 | 37.884 | 39.392 | 41.166 | 0 | [] |
| kv | far | decode32 | 51 | 96.908 | 144.998 | 147.399 | 32 | [] |
| kv | near | decode32 | 51 | 94.977 | 139.473 | 148.672 | 32 | [] |
| state | far | decode32 | 51 | 38.443 | 39.689 | 39.798 | 32 | [] |
| state | near | decode32 | 51 | 17.189 | 28.487 | 42.309 | 32 | [] |
| kv | far | decode32 | 52 | 96.743 | 144.724 | 147.436 | 32 | [] |
| kv | near | decode32 | 52 | 84.024 | 121.481 | 147.789 | 30 | [] |
| state | far | decode32 | 52 | 38.678 | 39.933 | 40.042 | 32 | [] |
| state | near | decode32 | 52 | 17.294 | 28.690 | 42.511 | 32 | [] |
| kv | far | native | 52 | 95.914 | 144.545 | 145.535 | 0 | [] |
| kv | near | native | 52 | 93.842 | 138.581 | 147.853 | 0 | [] |
| state | far | native | 52 | 38.249 | 39.728 | 39.836 | 0 | [] |
| state | near | native | 52 | 37.881 | 39.391 | 41.145 | 0 | [] |
| kv | far | slots64 | 51 | 96.297 | 145.015 | 145.942 | 0 | [] |
| kv | near | slots64 | 51 | 94.021 | 138.849 | 148.084 | 0 | [] |
| state | far | slots64 | 51 | 37.826 | 39.737 | 40.709 | 65 | [] |
| state | near | slots64 | 51 | 27.533 | 28.880 | 42.730 | 31 | [] |
| kv | far | delayer32 | 51 | 96.257 | 145.089 | 145.604 | 0 | [] |
| kv | near | delayer32 | 51 | 94.188 | 139.082 | 148.328 | 0 | [] |
| state | far | delayer32 | 51 | 38.201 | 39.677 | 39.783 | 0 | [] |
| state | near | delayer32 | 51 | 37.881 | 39.383 | 41.200 | 0 | [] |
| kv | far | delayer32 | 52 | 95.868 | 144.510 | 145.449 | 0 | [] |
| kv | near | delayer32 | 52 | 93.803 | 138.590 | 147.738 | 0 | [] |
| state | far | delayer32 | 52 | 38.259 | 39.738 | 39.844 | 0 | [] |
| state | near | delayer32 | 52 | 37.935 | 39.445 | 41.195 | 0 | [] |
| kv | far | slots64 | 52 | 95.823 | 144.445 | 145.438 | 0 | [] |
| kv | near | slots64 | 52 | 93.637 | 138.351 | 147.529 | 0 | [] |
| state | far | slots64 | 52 | 37.830 | 39.735 | 40.750 | 65 | [] |
| state | near | slots64 | 52 | 27.618 | 28.890 | 42.740 | 32 | [] |

Positive percent means improvement. At most two paired repeats; no precise CI. An incomplete pair matrix is pending, not a failed motivation.

| Profile | Phase | Policy | Paired reps | Mean gain % | P95 gain % | Makespan gain % | Status |
|---|---|---|---:|---:|---:|---:|---|
| state | far | decode32 | 2 | -0.90 | -0.30 | -0.30 | NO CONSISTENT CONSTRAINED GAIN |
| state | far | slots64 | 2 | 1.02 | -0.11 | -2.33 | SCREEN POSITIVE |
| state | far | delayer32 | 2 | -0.04 | -0.04 | -0.03 | NOT TRIGGERED |
| state | near | decode32 | 2 | 54.49 | 27.43 | -3.05 | SCREEN POSITIVE |
| state | near | slots64 | 2 | 27.21 | 26.67 | -3.84 | SCREEN POSITIVE |
| state | near | delayer32 | 2 | -0.07 | -0.06 | -0.10 | NOT TRIGGERED |
| kv | far | decode32 | 2 | -0.90 | -0.15 | -1.32 | NO CONSISTENT CONSTRAINED GAIN |
| kv | far | slots64 | 2 | -0.10 | -0.06 | -0.13 | NOT TRIGGERED |
| kv | far | delayer32 | 2 | -0.11 | -0.11 | -0.02 | NOT TRIGGERED |
| kv | near | decode32 | 2 | 4.77 | 6.00 | -0.13 | NO CONSISTENT CONSTRAINED GAIN |
| kv | near | slots64 | 2 | 0.16 | 0.14 | 0.15 | NOT TRIGGERED |
| kv | near | delayer32 | 2 | -0.02 | -0.03 | 0.00 | NOT TRIGGERED |
