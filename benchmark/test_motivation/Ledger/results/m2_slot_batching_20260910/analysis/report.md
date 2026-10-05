# M2 slot accounting and batching

Formal runs: 100; requests: 2000; complete=True

| Configuration | Gate | Order | N | Short p95 | Short mean | Short max | Long p95 | Makespan | Premature slot denials | Mixed batches |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| state8-mixed1024 | legacy | long-first | 5 | 12.447 | 7.680 | 12.487 | 12.899 | 12.984 | 3.0 | 37.0 |
| state8-mixed1024 | legacy | short-first | 5 | 9.296 | 6.620 | 12.062 | 12.889 | 13.006 | 10.0 | 45.0 |
| state8-mixed1024 | reuse-aware | long-first | 5 | 12.357 | 7.679 | 12.397 | 12.816 | 12.893 | 0.0 | 35.0 |
| state8-mixed1024 | reuse-aware | short-first | 5 | 9.291 | 6.669 | 12.075 | 12.844 | 12.962 | 0.0 | 43.0 |
| state8-mixed4096 | legacy | long-first | 5 | 12.339 | 7.322 | 12.340 | 12.495 | 12.515 | 1.0 | 12.0 |
| state8-mixed4096 | legacy | short-first | 5 | 9.085 | 6.487 | 12.363 | 12.531 | 12.600 | 1.0 | 14.0 |
| state8-mixed4096 | reuse-aware | long-first | 5 | 12.255 | 7.245 | 12.256 | 12.409 | 12.429 | 0.0 | 9.0 |
| state8-mixed4096 | reuse-aware | short-first | 5 | 8.899 | 6.417 | 12.314 | 12.483 | 12.553 | 0.0 | 11.0 |
| state8-prefill4096 | legacy | long-first | 5 | 12.145 | 7.044 | 12.145 | 12.207 | 12.210 | 0.0 | 0.0 |
| state8-prefill4096 | legacy | short-first | 5 | 12.311 | 6.874 | 12.312 | 12.274 | 12.326 | 1.0 | 0.0 |
| state8-prefill4096 | reuse-aware | long-first | 5 | 12.137 | 7.028 | 12.138 | 12.200 | 12.203 | 0.0 | 0.0 |
| state8-prefill4096 | reuse-aware | short-first | 5 | 8.738 | 6.299 | 12.257 | 12.221 | 12.272 | 0.0 | 0.0 |
| state8-unchunked | legacy | long-first | 5 | 11.528 | 6.694 | 11.529 | 11.590 | 11.593 | 0.0 | 0.0 |
| state8-unchunked | legacy | short-first | 5 | 8.435 | 6.087 | 11.608 | 11.572 | 11.624 | 0.0 | 0.0 |
| state8-unchunked | reuse-aware | long-first | 5 | 11.531 | 6.699 | 11.532 | 11.593 | 11.597 | 0.0 | 0.0 |
| state8-unchunked | reuse-aware | short-first | 5 | 8.516 | 6.152 | 11.711 | 11.674 | 11.726 | 0.0 | 0.0 |
| wide-cap8-prefill4096 | legacy | long-first | 5 | 12.192 | 7.041 | 12.192 | 12.254 | 12.257 | 0.0 | 0.0 |
| wide-cap8-prefill4096 | legacy | short-first | 5 | 8.803 | 6.343 | 12.350 | 12.313 | 12.365 | 0.0 | 0.0 |
| wide-cap8-prefill4096 | reuse-aware | long-first | 5 | 12.142 | 7.043 | 12.143 | 12.204 | 12.207 | 0.0 | 0.0 |
| wide-cap8-prefill4096 | reuse-aware | short-first | 5 | 8.804 | 6.343 | 12.349 | 12.312 | 12.365 | 0.0 | 0.0 |

## Within-session correction effects

Positive is improvement; percentages, paired run-level bootstrap intervals.

| Configuration | Order | Short p95 | Long p95 | Makespan |
|---|---|---:|---:|---:|
| state8-prefill4096 | short-first | +29.01 [+28.04, +30.03] | +0.42 [-0.96, +1.85] | +0.42 [-0.95, +1.84] |
| state8-prefill4096 | long-first | +0.06 [-0.35, +0.66] | +0.06 [-0.34, +0.67] | +0.06 [-0.35, +0.66] |
| state8-mixed4096 | short-first | +2.05 [+1.45, +2.68] | +0.37 [-0.22, +0.89] | +0.37 [-0.22, +0.89] |
| state8-mixed4096 | long-first | +0.68 [+0.46, +0.93] | +0.68 [+0.46, +0.91] | +0.68 [+0.46, +0.91] |
| state8-mixed1024 | short-first | +0.05 [-1.17, +1.25] | +0.34 [-0.85, +1.63] | +0.33 [-0.85, +1.62] |
| state8-mixed1024 | long-first | +0.71 [-0.65, +1.91] | +0.63 [-0.72, +1.82] | +0.69 [-0.66, +1.88] |
| state8-unchunked | short-first | -0.97 [-2.13, +0.19] | -0.89 [-1.85, +0.07] | -0.89 [-1.84, +0.07] |
| state8-unchunked | long-first | -0.03 [-0.59, +0.58] | -0.03 [-0.58, +0.58] | -0.03 [-0.58, +0.58] |
| wide-cap8-prefill4096 | short-first | -0.01 [-0.66, +0.62] | +0.01 [-0.47, +0.48] | +0.00 [-0.47, +0.48] |
| wide-cap8-prefill4096 | long-first | +0.41 [+0.04, +0.75] | +0.41 [+0.04, +0.75] | +0.40 [+0.04, +0.75] |

Reference opportunity: True

All batching comparisons use the same token-VTC. No external SOTA system was faithfully reproduced.
Cross-batching comparisons are session-blocked; gate A/B is randomized within-session.
With only 16 short requests, linearly interpolated p95 is sensitive to the final two ranks; mean, max and first-repetition ECDF accompany it.
Absolute latency differences between batching families include compute/batch effects; no KDA-only causality claim.
