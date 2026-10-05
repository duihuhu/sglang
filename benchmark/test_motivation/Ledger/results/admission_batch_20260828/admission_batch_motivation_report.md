# Admission-batch and runtime-cost motivation report

## Integrity

Artifacts 57; requests 6120; failures 0; cached tokens 0; retractions 0; exact resource traces 57/57; batch traces 57/57; dropped batch events 0; resolved policy metadata 57/57; equal-token mixed 33/33.

## M2: admission-batch coupling

| cell | state / max-running | split/packed p95 JCT | peak active | split active/request | split decode interval | split queue p95 | mean decode batch |
|---|---:|---:|---:|---:|---:|---:|---:|
| low-state-auto | 48 / 9 | 11.51/15.89 | 9 | 0.65 | 0.0159 | 10.79 | 6.68 |
| high-state-capped | 480 / 9 | 11.45/15.93 | 9 | 0.64 | 0.0157 | 10.76 | 6.68 |
| high-state-auto | 480 / 96 | 9.85/18.94 | 96 | 4.49 | 0.1348 | 2.58 | 14.89 |

Decision: **PASS**. At fixed max-running=9, raising state 48->480 changes mean tenant p95 JCT by 0.4%. At fixed state=480, raising max-running 9->96 changes peak active by 10.67x, reduces split queue p95 by 76.0%, inflates split active residency by 6.99x and decode interval by 8.61x, while packed p95 changes by 18.9%.

## M3a: static cost versus runtime resource signal

| cell | token gap | profiled gap | instant gap | instant improvement over strongest scalar |
|---|---:|---:|---:|---:|
| low-state-auto | 920.4% | 869.0% | 336.2% | 61.3% |
| high-state-capped | 916.1% | 871.7% | 149.2% | 82.9% |
| high-state-auto | 976.4% | 934.7% | 947.0% | -1.3% |

Decision: **PASS**. Profiled scalar mean improvement over token VTC is 4.9%; instant-share mean improvement in the two max-running=9 cells is 72.1%, but only -1.3% when max-running=96.

## M3b: instantaneous-share history reset probe

| policy | max delivered-service gap | gap area | max pulser lead | slowdown gap | makespan |
|---|---:|---:|---:|---:|---:|
| token-vtc | 34.9% | 25.8% | -6.3% | 1242.3% | 31.40 |
| instant-share | 47.6% | 27.1% | 47.6% | 533.6% | 32.28 |

Decision: **PASS**. Instant-share/token-VTC maximum service-gap ratio is 1.36x, normalized gap-area ratio is 1.05x, and makespan change is 2.8%.

## Overall

Motivation decision: **PASS**. M2 and M3 are reported separately; a failed sub-gate is retained as a falsified explanation, not rewritten post hoc.

