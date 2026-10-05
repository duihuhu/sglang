# Lifecycle-cost fairness held-out generalization

Overall decision: **PENDING**

## Integrity

Formal artifacts: 200; requests: 13224; failures: 0; exact traces: 200/200; resolved mixed-policy metadata: 180/180; GPU 6,7 metadata: 200/200.

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

## Confirmation

Not run.

