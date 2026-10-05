# Topic B Exact Active-Resource Results

> 3 repetitions on Kimi-Linear-48B-A3B, TP=2, FCFS; scheduler allocator timestamps rather than streaming-response proxies.

- Physical recurrent-state capacity: 485 slots.
- Full-KV capacity: 1,458,422 token slots.
- Scope: active private request state/KV; inactive radix-cache ownership and GPU time are not included.
- All requests completed; no retractions were observed.

| condition | group | state-slot-s | KV-token-s | normalized state debt | normalized KV debt | dominant debt | peak state slots |
|---|---|---:|---:|---:|---:|---:|---:|
| state-isolated | state-heavy | 636.011 +/- 1.577 | 218446.636 +/- 539.170 | 1.311 +/- 0.003 | 0.150 +/- 0.000 | 1.311 +/- 0.003 | 288 |
| kv-isolated | kv-heavy | 169.134 +/- 0.099 | 919898.419 +/- 572.476 | 0.349 +/- 0.000 | 0.631 +/- 0.000 | 0.631 +/- 0.000 | 18 |
| state-first | kv-heavy | 171.490 +/- 0.103 | 928920.680 +/- 537.753 | 0.354 +/- 0.000 | 0.637 +/- 0.000 | 0.637 +/- 0.000 | 18 |
| state-first | state-heavy | 975.080 +/- 0.935 | 334255.868 +/- 319.441 | 2.010 +/- 0.002 | 0.229 +/- 0.000 | 2.010 +/- 0.002 | 288 |
| kv-first | kv-heavy | 225.466 +/- 0.260 | 1227589.834 +/- 1392.412 | 0.465 +/- 0.001 | 0.842 +/- 0.001 | 0.842 +/- 0.001 | 18 |
| kv-first | state-heavy | 602.769 +/- 0.826 | 207100.141 +/- 283.663 | 1.243 +/- 0.002 | 0.142 +/- 0.000 | 1.243 +/- 0.002 | 273 |
| interleave | kv-heavy | 206.179 +/- 0.228 | 1121024.724 +/- 718.766 | 0.425 +/- 0.000 | 0.769 +/- 0.000 | 0.769 +/- 0.000 | 18 |
| interleave | state-heavy | 1388.083 +/- 5.155 | 475156.065 +/- 1761.284 | 2.862 +/- 0.011 | 0.326 +/- 0.001 | 2.862 +/- 0.011 | 273 |

## Findings

- Completed-token mismatch: 0.00%.
- Mean physical peak state ratio (state-heavy / KV-heavy): 15.4x.
- Every co-run shows resource specialization: state-heavy is state-dominant and KV-heavy is KV-dominant (9/9 runs).
- Equal token service maps to a mean 2.79x difference in cumulative normalized dominant debt.
- Maximum mean p95 JCT slowdown versus isolated execution: 2.87x.

## Decision

Exact active-resource accounting confirms the physical premise of Topic B, but it does not yet prove VTC inadequate. The next gate remains a sustained-arrival comparison of tuned scalar VTC against vector debt, including GPU time and a throughput-loss bound.
