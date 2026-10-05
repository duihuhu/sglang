# M2 batch frontier measurements

Artifacts: 16; requests: 1632. Paired payload identity verified.

These are native scheduling controls, not full Sarathi/FairBatching reproductions.
One-repeat screening is descriptive, not confirmatory evidence.

| Stage / shape | Configuration | Runs | Split p95 (s) | Packed p95 (s) | Makespan (s) | Split slowdown | Packed slowdown |
|---|---|---:|---:|---:|---:|---:|---:|
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

- screen/f16: separate-4096/n24/instant, separate-4096/n24/vtc, separate-4096/n48/instant, separate-4096/n9/instant, separate-4096/n9/vtc, separate-4096/n96/instant
