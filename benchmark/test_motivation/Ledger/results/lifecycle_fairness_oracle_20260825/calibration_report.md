# Lifecycle-cost fairness oracle results

Artifacts analyzed: 18

Overall decision: **PENDING**

## Calibration

| policy | split slowdown | packed slowdown | slowdown gap | makespan (s) | SLO goodput | selected |
|---|---:|---:|---:|---:|---:|---|
| realized-cost-s0-k1 | 196.7% | 48.2% | 148.5% | 24.113 | 136 |  |
| realized-cost-s0.25-k1 | 193.3% | 53.6% | 139.7% | 24.207 | 136 | yes |
| realized-cost-s0.5-k1 | 305.7% | 57.8% | 247.9% | 24.453 | 136 |  |
| realized-cost-s1-k1 | 304.8% | 51.8% | 253.0% | 24.914 | 136 |  |
| realized-cost-s16-k1 | 742.7% | 29.7% | 713.0% | 24.550 | 136 |  |
| realized-cost-s2-k1 | 424.3% | 36.9% | 387.3% | 24.447 | 136 |  |
| realized-cost-s4-k1 | 806.0% | 33.2% | 772.8% | 23.969 | 136 |  |
| realized-cost-s8-k1 | 845.7% | 31.8% | 813.8% | 23.794 | 136 |  |
| scalar-vtc-r0 | 931.4% | 23.8% | 907.6% | 24.716 | 136 | yes |
| scalar-vtc-r1024 | 1076.7% | 17.2% | 1059.5% | 23.573 | 136 |  |
| scalar-vtc-r12288 | 1226.8% | 17.6% | 1209.2% | 23.288 | 136 |  |
| scalar-vtc-r16384 | 1182.4% | 18.5% | 1163.9% | 23.000 | 136 |  |
| scalar-vtc-r32768 | 1167.4% | 17.6% | 1149.8% | 22.834 | 136 |  |
| scalar-vtc-r4096 | 1178.9% | 17.1% | 1161.8% | 23.051 | 136 |  |
| scalar-vtc-r65536 | 1179.3% | 18.2% | 1161.1% | 22.970 | 136 |  |
| scalar-vtc-r8192 | 1146.3% | 17.7% | 1128.6% | 22.815 | 136 |  |
| token-vtc | 931.4% | 23.8% | 907.6% | 24.716 | 136 | yes |

## Confirmation

Not run.

## Frozen selection

```json
{
  "scalar_label": "scalar-vtc-r0",
  "scalar_request_weight": 0.0,
  "lifecycle_label": "realized-cost-s0.25-k1",
  "lifecycle_state_weight": 0.25,
  "lifecycle_kv_weight": 1.0
}
```

