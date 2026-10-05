# Lifecycle-Cost Motivation Gate Protocol

This protocol freezes the five motivation gates before the A800 runs. Results
must be interpreted against these thresholds; a failed gate is not retuned into
a pass after observing the data.

## Hardware and model

- Host: `182.43.57.115`
- Container: `sglang`
- GPUs: physical A800 devices 6 and 7 only, tensor parallel size 2
- Model: `/models/Kimi-Linear-48B-A3B`
- Repository: host `/mnt/nvme0/mllm/sglang`, container `/workspace/sglang`
- Every server enables exact active-request state/KV resource-time tracking.

## M1: token and length ranking

Run the Cartesian product `P={1K,4K,16K}`, `O={32,128,512}` and
`reuse={0,0.5,0.9}` with eight concurrent requests and two repetitions. A raw
`P+O` ordering inversion requires the lower-token class to have at least 5%
higher p95 JCT. The gate passes only when at least 10% of comparable class pairs
invert and at least one 15% inversion has the same direction in every
repetition.

## M2: state-dependent ranking

Use the same 16 `P=1K,O=512` decode-heavy and 16 `P=16K,O=32`
prefill-heavy requests in every run. Compare decode-first, prefill-first and
interleaving under state-bound `(state=48, KV=393216)`, KV-bound
`(state=480, KV=196608)` and compute-bound `(state=480, KV=393216)` servers.
The gate requires both a best-order reversal between state-bound and KV-bound
regimes and at least 15% worst-regime regret for every fixed order.

## M3: cross-layer coupling

Warm a 90%-reused 16K prefix, start 32 decode-heavy background requests, then
queue 12 long unique pressure requests and eight requests that reuse the warm
prefix. Compare pressure-first, reuse-first and interleaving on both FCFS and
SGLang LPM servers. The gate requires reuse-first to improve FCFS overall p95 by
15%, preserve at least 8K additional cached tokens, and retain at least 5%
benefit over pressure-first LPM. This last condition prevents an existing
locality policy from being relabeled as a joint contribution.

## M4: equal-token tenant cost transfer

Tenant `split` sends 32 `P=1K,O=32` requests; tenant `packed` sends two
`P=16K,O=512` requests. Both request exactly 33,792 input-plus-output tokens.
Run both tenants in isolation, then together under FCFS and token VTC, with
three repetitions. The gate requires a 2x difference in at least one measured
per-token resource-time dimension and a 15 percentage-point difference in
VTC-normalized slowdown relative to isolation.

## M5: empirical oracle headroom

Use four request classes: decode-heavy, unique-prefill, reused-prefill and
medium. Under the balanced constrained server `(state=96, KV=196608)`, enumerate
all 24 class-block orders. Compare the best observed order with preregistered
FCFS, SJF, LPM, future-memory and reuse-plus-future-memory composite orders. The
gate requires at least 15% p95 JCT improvement over the strongest baseline while
keeping makespan within 5%.

The enumerated best order is only an empirical block-order oracle. Passing this
gate establishes mechanism headroom; it does not validate an online predictor
or claim a production scheduler improvement.
