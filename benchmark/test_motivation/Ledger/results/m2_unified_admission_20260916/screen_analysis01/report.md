# M2 Unified Admission

Complete: True; {'runs': 16, 'requests': 5888, 'warmup_requests': 5888}

| Profile | Phase | Policy | Rep | Mean JCT | P95 | Makespan | Delays | Scene errors |
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

Positive percent means improvement. At most two paired repeats; no precise CI. An incomplete pair matrix is pending, not a failed motivation.

| Profile | Phase | Policy | Paired reps | Mean gain % | P95 gain % | Makespan gain % | Status |
|---|---|---|---:|---:|---:|---:|---|
| state | far | decode32 | 2 | -0.90 | -0.30 | -0.30 | NO CONSISTENT CONSTRAINED GAIN |
| state | near | decode32 | 2 | 54.49 | 27.43 | -3.05 | SCREEN POSITIVE |
| kv | far | decode32 | 2 | -0.90 | -0.15 | -1.32 | NO CONSISTENT CONSTRAINED GAIN |
| kv | near | decode32 | 2 | 4.77 | 6.00 | -0.13 | NO CONSISTENT CONSTRAINED GAIN |
