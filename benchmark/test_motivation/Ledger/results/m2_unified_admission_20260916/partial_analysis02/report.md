# M2 Unified Admission

Complete: False; {'runs': 12, 'requests': 4416, 'warmup_requests': 4416}

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

Positive percent means improvement. At most two paired repeats; no precise CI. An incomplete pair matrix is pending, not a failed motivation.

| Profile | Phase | Policy | Paired reps | Mean gain % | P95 gain % | Makespan gain % | Status |
|---|---|---|---:|---:|---:|---:|---|
| state | far | decode32 | 1 | -0.68 | -0.07 | -0.08 | PENDING: incomplete pairs |
| state | near | decode32 | 1 | 54.63 | 27.68 | -2.78 | PENDING: incomplete pairs |
| kv | far | decode32 | 1 | -0.94 | -0.18 | -1.33 | PENDING: incomplete pairs |
| kv | near | decode32 | 1 | -0.91 | -0.34 | -0.31 | PENDING: incomplete pairs |
