# Lifecycle-cost fairness held-out generalization

Overall decision: **FAIL**

## Integrity

Formal artifacts: 626; requests: 64152; failures: 0; exact traces: 626/626; resolved mixed-policy metadata: 486/486; GPU 6,7 metadata: 626/626.

## Screening selection

| cell | best request-tax | strongest static | lifecycle oracle | token gap | fixed lifecycle gap |
|---|---|---|---|---:|---:|
| arrival1 | token-vtc | instant-share | lifecycle-s0-k1 | 343.7% | 223.7% |
| arrival3 | request-tax-r32768-i1-o1 | request-tax-r32768-i1-o1 | lifecycle-s0-k1 | 57.5% | 59.5% |
| base | token-vtc | instant-share | lifecycle-s0-k1 | 368.1% | 201.7% |
| kv393 | token-vtc | instant-share | lifecycle-s0-k1 | 361.9% | 248.7% |
| output1k | request-tax-r65536-i1-o1 | request-tax-r65536-i1-o1 | lifecycle-s0-k1 | 114.3% | 104.9% |
| output2k | request-tax-r1024-i1-o1 | request-tax-r1024-i1-o1 | lifecycle-s1-k1 | 93.3% | 91.2% |
| relaxed | token-vtc | fcfs | lifecycle-s0.25-k1 | 257.7% | 250.1% |
| split32 | token-vtc | instant-share | lifecycle-s0-k1 | 354.2% | 123.9% |
| split4 | token-vtc | fcfs | lifecycle-s0-k1 | 285.5% | 281.0% |
| state96 | token-vtc | vector-debt | lifecycle-s0.5-k1 | 271.1% | 269.0% |

## Confirmation by cell

| cell | strongest confirmed static | lifecycle oracle | token gap | strongest static gap | fixed lifecycle gap | oracle gap | fixed improvement | price regret | makespan change | direction |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| arrival1 | instant-share | lifecycle-s0-k1 | 675.7% | 84.9% | 132.8% | 85.7% | -56.5% | 55.0% | -2.9% | no |
| arrival3 | instant-share | lifecycle-s0.25-k1 | 28.8% | 10.1% | 16.2% | 16.2% | -60.3% | 0.0% | 1.7% | no |
| base | instant-share | lifecycle-s0.25-k1 | 916.4% | 253.1% | 143.1% | 143.1% | 43.5% | 0.0% | -2.4% | yes |
| kv393 | instant-share | lifecycle-s0-k1 | 918.8% | 279.7% | 238.6% | 136.6% | 14.7% | 74.7% | -0.5% | yes |
| output1k | instant-share | lifecycle-s0-k1 | 390.9% | 72.2% | 75.4% | 72.9% | -4.5% | 3.4% | 4.2% | no |
| output2k | instant-share | lifecycle-s0.25-k1 | 124.5% | 62.2% | 45.7% | 45.7% | 26.5% | 0.0% | 2.2% | yes |
| relaxed | instant-share | lifecycle-s0.25-k1 | 606.3% | 517.5% | 217.7% | 217.7% | 57.9% | 0.0% | 3.5% | yes |
| split32 | instant-share | lifecycle-s0-k1 | 910.6% | 162.1% | 95.5% | 77.0% | 41.1% | 24.0% | -3.7% | yes |
| split4 | instant-share | lifecycle-s0-k1 | 751.5% | 389.6% | 354.8% | 330.2% | 8.9% | 7.5% | 0.9% | yes |
| state96 | instant-share | lifecycle-s0.25-k1 | 609.2% | 337.4% | 225.2% | 225.2% | 33.2% | 0.0% | 3.8% | yes |

## Generalization gate

Decision: **FAIL**. Mean gap improvement over the strongest per-cell static baseline: 10.4%; paired direction: 7/10 cells; mean makespan change: 0.7%; median fixed-price regret: 1.7%.

## Independent instantaneous-share kill confirm

| cell | token gap | instant-share gap | fixed lifecycle gap | lifecycle improvement | makespan change | direction |
|---|---:|---:|---:|---:|---:|---|
| arrival1 | 676.8% | 85.3% | 170.6% | -99.9% | -2.7% | no |
| arrival3 | 31.1% | 9.5% | 12.4% | -30.6% | 1.7% | no |
| base | 924.8% | 272.4% | 146.4% | 46.2% | -2.8% | yes |
| kv393 | 920.4% | 280.7% | 253.8% | 9.6% | -1.0% | yes |
| output1k | 397.1% | 71.9% | 77.0% | -7.0% | 3.9% | no |
| output2k | 123.5% | 60.8% | 55.9% | 8.0% | 2.3% | yes |
| relaxed | 609.5% | 518.4% | 216.6% | 58.2% | 3.1% | yes |
| split32 | 913.8% | 122.8% | 100.2% | 18.4% | -4.2% | yes |
| split4 | 760.9% | 394.3% | 358.5% | 9.1% | 0.7% | yes |
| state96 | 605.5% | 336.8% | 241.2% | 28.4% | 2.8% | yes |

Decision: **FAIL**. Mean gap improvement over instantaneous-share: 4.0%, cell bootstrap 95% CI [-24.7%, 27.3%]; paired direction: 7/10 cells; mean makespan change vs token VTC: 0.4%.

