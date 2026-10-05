# M2 batch frontier measurements

Artifacts: 108; requests: 12896. Paired payload identity verified.

These are native scheduling controls, not full Sarathi/FairBatching reproductions.
One-repeat screening is descriptive, not confirmatory evidence.

| Stage / shape | Configuration | Runs | Split p95 (s) | Packed p95 (s) | Makespan (s) | Split slowdown | Packed slowdown |
|---|---|---:|---:|---:|---:|---:|---:|
| confirm/f16 | separate-4096/n48/vtc | 5 | 7.825 | 20.116 | 21.229 | 8.479 | 1.381 |
| confirm/f16 | separate-4096/n96/vtc | 5 | 10.635 | 19.994 | 20.745 | 11.599 | 1.373 |
| confirm/f32 | separate-4096/n48/vtc | 5 | 7.071 | 20.197 | 21.336 | 8.889 | 1.385 |
| confirm/f32 | separate-4096/n96/vtc | 5 | 7.097 | 20.140 | 20.935 | 8.931 | 1.381 |
| screen/f16 | mixed-1024/n24/instant | 1 | 5.933 | 28.075 | 31.429 | 2.871 | 1.482 |
| screen/f16 | mixed-1024/n24/vtc | 1 | 16.967 | 29.760 | 32.185 | 8.211 | 1.571 |
| screen/f16 | mixed-1024/n48/instant | 1 | 8.600 | 28.669 | 31.580 | 4.017 | 1.508 |
| screen/f16 | mixed-1024/n48/vtc | 1 | 16.844 | 29.542 | 31.968 | 7.867 | 1.554 |
| screen/f16 | mixed-1024/n9/instant | 1 | 9.164 | 32.026 | 34.968 | 2.620 | 1.690 |
| screen/f16 | mixed-1024/n9/vtc | 1 | 22.216 | 27.293 | 36.091 | 6.351 | 1.440 |
| screen/f16 | mixed-1024/n96/instant | 1 | 8.593 | 28.661 | 31.572 | 4.061 | 1.511 |
| screen/f16 | mixed-1024/n96/vtc | 1 | 16.846 | 29.552 | 31.969 | 7.961 | 1.558 |
| screen/f16 | mixed-4096/n24/instant | 1 | 4.139 | 21.391 | 24.244 | 4.301 | 1.364 |
| screen/f16 | mixed-4096/n24/vtc | 1 | 8.639 | 21.646 | 23.731 | 8.976 | 1.380 |
| screen/f16 | mixed-4096/n48/instant | 1 | 8.488 | 21.856 | 23.291 | 2.898 | 1.328 |
| screen/f16 | mixed-4096/n48/vtc | 1 | 9.492 | 21.829 | 23.094 | 3.241 | 1.326 |
| screen/f16 | mixed-4096/n9/instant | 1 | 4.177 | 25.276 | 27.660 | 3.182 | 1.609 |
| screen/f16 | mixed-4096/n9/vtc | 1 | 14.984 | 19.775 | 28.523 | 11.414 | 1.259 |
| screen/f16 | mixed-4096/n96/instant | 1 | 10.870 | 22.179 | 23.274 | 11.309 | 1.413 |
| screen/f16 | mixed-4096/n96/vtc | 1 | 10.910 | 22.047 | 23.082 | 11.352 | 1.405 |
| screen/f16 | separate-4096/n24/instant | 1 | 3.948 | 20.151 | 22.657 | 4.216 | 1.380 |
| screen/f16 | separate-4096/n24/vtc | 1 | 7.247 | 20.280 | 22.230 | 7.741 | 1.389 |
| screen/f16 | separate-4096/n48/instant | 1 | 7.758 | 19.986 | 21.102 | 8.300 | 1.370 |
| screen/f16 | separate-4096/n48/vtc | 1 | 7.895 | 20.202 | 21.323 | 8.447 | 1.385 |
| screen/f16 | separate-4096/n9/instant | 1 | 2.988 | 22.311 | 25.846 | 2.471 | 1.528 |
| screen/f16 | separate-4096/n9/vtc | 1 | 13.444 | 18.344 | 26.994 | 11.117 | 1.256 |
| screen/f16 | separate-4096/n96/instant | 1 | 10.348 | 20.050 | 20.801 | 11.075 | 1.376 |
| screen/f16 | separate-4096/n96/vtc | 1 | 10.709 | 20.101 | 20.853 | 11.461 | 1.380 |

## Interpretation

Frontiers minimize both tenants' absolute p95 and makespan. They are descriptive;
a dominated sample mean is not proof of a significant or SOTA-wide improvement.
Completed token equality is not a backlogged-window fairness guarantee.
Matched isolation uses the same payload, arrival offsets and configuration.
Decode intervals and launch gaps are not synchronized GPU kernel timings.
Missing instant-share queue measurements remain null.

## Descriptive frontiers

- confirm/f16: separate-4096/n48/vtc, separate-4096/n96/vtc
- confirm/f32: separate-4096/n48/vtc, separate-4096/n96/vtc
- screen/f16: separate-4096/n24/instant, separate-4096/n24/vtc, separate-4096/n48/instant, separate-4096/n9/instant, separate-4096/n9/vtc, separate-4096/n96/instant
