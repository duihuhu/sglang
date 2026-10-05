# M2 hybrid factors measurements

Formal runs: 132; requests: 2119; paired payloads match.
Complete planned matrix: True; one-state-per-active: True.

## E1 activation

| Shape | Runs | Peak state slots | Peak KV tokens | KV-progress/no-new-state events |
|---|---:|---:|---:|---:|
| n1 | 3 | 1.0 | 16512.0 | 3.0 |
| n16 | 3 | 16.0 | 18432.0 | 1.0 |
| n32 | 3 | 32.0 | 20480.0 | 2.0 |
| n4 | 3 | 4.0 | 16896.0 | 0.0 |

## E2 fixed candidates

| Regime | Submission | Policy | Runs | Short p95 | Long p95 | Makespan | Request-limit / no-token blocks |
|---|---|---|---:|---:|---:|---:|---:|
| cap-control | long-first | instant | 5 | 8.606 | 12.200 | 12.202 | 2.0 / 0.0 |
| cap-control | long-first | vtc | 5 | 12.047 | 12.109 | 12.112 | 2.0 / 0.0 |
| cap-control | short-first | instant | 5 | 8.703 | 12.184 | 12.236 | 2.0 / 0.0 |
| cap-control | short-first | vtc | 5 | 8.669 | 12.131 | 12.182 | 2.0 / 0.0 |
| kv-low | long-first | instant | 5 | 7.919 | 14.999 | 15.572 | 0.0 / 5.0 |
| kv-low | long-first | vtc | 5 | 11.781 | 15.057 | 15.628 | 0.0 / 4.0 |
| kv-low | short-first | instant | 5 | 8.047 | 15.016 | 15.637 | 0.0 / 5.0 |
| kv-low | short-first | vtc | 5 | 8.078 | 15.013 | 15.631 | 0.0 / 4.0 |
| state-low | long-first | instant | 5 | 8.582 | 12.159 | 12.161 | 2.0 / 0.0 |
| state-low | long-first | vtc | 5 | 12.006 | 12.069 | 12.072 | 2.0 / 0.0 |
| state-low | short-first | instant | 5 | 12.145 | 12.108 | 12.160 | 2.0 / 0.0 |
| state-low | short-first | vtc | 5 | 12.181 | 12.145 | 12.197 | 2.0 / 0.0 |
| wide | long-first | instant | 5 | 5.094 | 5.145 | 5.146 | 0.0 / 0.0 |
| wide | long-first | vtc | 5 | 5.049 | 5.099 | 5.100 | 0.0 / 0.0 |
| wide | short-first | instant | 5 | 5.198 | 5.149 | 5.201 | 0.0 / 0.0 |
| wide | short-first | vtc | 5 | 5.239 | 5.190 | 5.242 | 0.0 / 0.0 |

## E3 probe

| Regime | Background output | Runs | Probe queue | TTFT | JCT | Live background at arrival | Valid fraction |
|---|---|---:|---:|---:|---:|---:|---:|
| cap-control | o256 | 5 | 10.993 | 11.111 | 11.913 | 8.0 | 1.0 |
| cap-control | o32 | 5 | 0.004 | 0.124 | 0.926 | 0.0 | 1.0 |
| kv-low | o256 | 5 | 0.034 | 0.153 | 0.965 | 8.0 | 1.0 |
| kv-low | o32 | 5 | 0.004 | 0.125 | 0.934 | 0.0 | 1.0 |
| state-low | o256 | 5 | 10.966 | 11.083 | 11.883 | 8.0 | 1.0 |
| state-low | o32 | 5 | 0.004 | 0.124 | 0.924 | 0.0 | 1.0 |
| wide | o256 | 5 | 0.034 | 0.153 | 0.959 | 8.0 | 1.0 |
| wide | o32 | 5 | 0.004 | 0.125 | 0.923 | 0.0 | 1.0 |

## Holding interaction relative to wide

Long-minus-short background probe queue effect, minus the same effect on wide pools; seconds.

- state-low: 10.932, CI [10.921, 10.948], valid=True.
- cap-control: 10.960, CI [10.840, 11.107], valid=True.
- kv-low: 0.000, CI [-0.017, 0.018], valid=True.

## Boundaries

Activation is first scheduler-observed active ownership, not exact GPU start time.
Admission reasons are observed return paths; physical pool counts can exclude pending selected reservations.
A positive effect reproduced by cap-control is not unique KDA execution causality.
E1 output work and E3 background work differ intentionally; neither is an equal-total-work fairness comparison.
Policy effect intervals are exploratory multiple contrasts, not simultaneous significance claims.
No complete SOTA implementation or calibrated-model comparison was performed.
