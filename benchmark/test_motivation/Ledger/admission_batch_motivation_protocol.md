# Admission-batch and runtime-cost motivation protocol

Date: 2026-08-28

## Scope

This experiment supplements the equal-token VTC observation with two narrowly
defined motivation questions:

1. M2: does recurrent-state capacity affect mixed-workload latency primarily
   through the admission concurrency that it permits?
2. M3: can an accurate static request cost or a memory-only instantaneous-share
   policy represent both capacity-limited and batch-limited regimes?

All formal runs use `/models/Kimi-Linear-48B-A3B`, physical A800 GPUs 6 and 7,
TP=2, unique cache salts, and no retraction or prefix reuse.

## M2: capacity-concurrency intervention

The equal-token workload gives each tenant 16,384 input and 512 output tokens
per wave. The split tenant uses sixteen `(1024,32)` requests while the packed
tenant uses one `(16384,512)` request. Eight waves arrive 1.5 seconds apart.

| cell | state cap | KV cap | max running | intervention |
|---|---:|---:|---:|---|
| low-state-auto | 48 | 196608 | derived 9 | original state-limited system |
| high-state-capped | 480 | 196608 | explicit 9 | remove state scarcity while holding concurrency |
| high-state-auto | 480 | 196608 | derived 96 | expose the admission/batch coupling |

Each cell has three fresh repetitions of matching split-only and packed-only
isolation plus token VTC, the frozen isolation-profiled VTC, and
instantaneous state/KV share. Repetition two reverses equal-time tie order.

M2 passes only if all integrity checks pass and:

1. raising state 48 to 480 at fixed max-running=9 changes mean absolute tenant
   p95 JCT by at most 10%; and
2. raising max-running 9 to 96 at fixed state=480 increases peak active
   requests by at least 4x, reduces split queue p95 by at least 50%, inflates
   split active residency or mean decode interval by at least 2x, and increases
   packed mixed p95 JCT by at least 15%.

The second condition deliberately requires a real neighbor externality, not
only a normalized-slowdown denominator change.

## M3a: static and instantaneous cost boundary

The frozen profiled VTC uses the six non-negative polynomial weights fitted on
the disjoint 2026-08-26 isolation data. No weight is selected from this matrix.

M3a passes when:

1. profiled VTC improves the mean gap over token VTC by less than 15%;
2. instantaneous share improves over the strongest token/profiled scalar by
   at least 15% on average in the two max-running=9 cells; and
3. its improvement is at most 10% in the max-running=96 cell.

This gate supports a runtime-state signal while exposing that memory-only
instantaneous share loses leverage once execution interference dominates. It
does not claim that every scalar function is impossible.

## M3b: pulsing tenant history-reset probe

At state=480, KV=196608, and explicit max-running=4, an incumbent submits eight
`(16384,512)` requests at time zero. A pulsing tenant submits eight waves of
sixteen `(1024,32)` requests starting at 0.75 seconds. The two tenants have
exactly equal total input and output service. Three repetitions compare token
VTC and instantaneous share with matching isolation.

Delivered token service at time `t` is `input tokens whose first token has
arrived + 2 * delivered output tokens`. The formal window begins after the last
pulse arrival and ends when the first tenant drains, so both tenants have all
work offered. M3b passes when instantaneous share increases either maximum
normalized service gap or normalized gap area by at least 20% over token VTC,
while changing makespan by at most +5%.

This is a motivation probe for missing historical debt, not a complete
long-term fairness evaluation.

## Exact observation and integrity

Every artifact must include:

- exact active request, recurrent-state-slot, and KV-token resource time;
- scheduler batch events with forward mode, batch size, tenant composition,
  extend tokens, queue depth, and CPU-observed launch gaps;
- resolved policy and VTC weights;
- effective state/KV capacity and max-running metadata;
- zero failures, cache hits, retractions, and dropped batch events.

The batch trace does not synchronize the GPU. Launch gaps and streaming token
events establish execution stretching without perturbing the serving path with
per-step CUDA synchronization.
