# M2: request-slot correction and batching comparison

Frozen before GPU measurements. Objective: demonstrate a concrete problem and
measurable opportunity under stated conditions, not universal inapplicability.

## Intervention and controls

The original gate charges all selected requests against free request slots.
The corrected gate separately bounds total batch participation by the runtime
cap and counts only newly allocated request slots against the free pool.
Continuations reuse their existing slots. KV/state safeguards and token-VTC
ordering/weights (input=1, output=2) remain unchanged.

Use an idle-only, tenant-experiment-guarded runtime switch to compare the exact
legacy gate and corrected gate in the same server session. Default production
behavior is corrected. No legacy branch is enabled outside explicit opt-in.
Record the resolved gate on both ranks and reused-slot counts at denials.

## Batching families

All are native SGLang implementations, not complete external paper reproductions:

1. Unchunked prefill-first (`chunk=-1`): no continuing-chunk negative control.
2. Chunked prefill-first (`chunk=4096`): the observed problem configuration.
3. Mixed prefill/decode (`chunk=4096` and `1024`): existing running decodes join
   prefill batches; decode occupancy is reserved in the token budget.

The two mixed chunk sizes are configurations of one family, not two algorithms.
Keep token-VTC fixed. FCFS/instant-share are not the batching axis in this study.
Sarathi-Serve, FairBatching and LPRS/APC have not been faithfully ported here;
do not label native mixed mode as a complete SOTA reproduction.

## Work and execution

- Kimi-Linear-48B-A3B, TP2, physical GPUs 6/7 only, port 30128.
- Disable radix cache and both CUDA-graph backends; max-prefill=16384;
  mem-fraction-static=.85. Same controls for both gates and batching families.
- Four configurations use state8/KV131072/effective8/runtime8. A fifth uses
  state128/KV131072/effective64/runtime8 with chunked prefill-first4096,
  separating a tight request pool from a cap-only condition.
- Each replay: 16 P512/O64 short + 4 P8192/O64 long requests, the E2 workload.
  Identical input IDs, output lengths and nominal arrivals for every paired
  configuration/gate; short-first and long-first alter only cohort starts
  (0 and .05s, intra-cohort .001s). Fresh repetitions20--24, never reuse old
  timing samples as baselines.
- Warm both gates and both orders for every configuration, exclude warmups.
  Randomize the four gate/order cases within each repetition and session.
  5 configurations x 2 gates x 2 orders x 5 repetitions = 100 formal runs,
  2000 formal requests. Server-session confounding remains for comparisons
  between batching configurations, but not within-session gate A/B pairs.
- Pilot: separate4096/state8, both gates and both orders, excluded. Resolve
  correctness failures before formal work; record any protocol amendment.

## Measurements and interpretation

Record exact output success, no prefix hits/retractions, no trace drops,
one active state per request, all resources returned, actual active/activation
trajectories, queue/TTFT/JCT, completed token-service totals, decode intervals,
mixed-batch presence and denial reasons. No inter-group equal-work claim:
the two tenants have different aggregate work, identical across paired runs.

Primary causal contrast: corrected versus legacy within each configuration/order.
Show short and long p95 JCT, makespan and queue with all run-level values.
Use run-level paired bootstrap CIs, not requests as independent repetitions.
Predeclared reference opportunity: separate4096/state8/short-first improves
short p95 by at least5%, CI lower bound above0; long p95 and makespan degradation
CI upper bounds each at most5%. A failure is informative, not grounds to
discard runs or tune after outcomes. Other contrasts are exploratory.

No-chunk and wide-pool controls check mechanism specificity; do not require
statistical equivalence or zero numerical timing noise. Check whether mixed
batching eliminates, reduces or preserves the gate-level opportunity. Compare
batching families under the same corrected gate to avoid attributing a known
implementation artifact to the entire family. Same VTC code is not a claim of
equal realized service gaps or improved fairness; report absolute endpoints.

This is a finite synthetic replay with manual pool limits. It demonstrates
local problems/opportunity, not production-wide prevalence, steady-state
capacity, KDA-exclusive causality, or a complete new scheduling algorithm.

## Pilot hardware amendment, before formal measurements

The first pilot (`smoke01`, 80 requests) completed request/resource checks but
both GPUs remained at 210MHz under full load. Its timing is invalid for gain
estimation; retain all four artifacts as functional diagnostics only. The user
approved restoring default GPU clocks on GPUs6/7 only. No power-limit changes.
Before restarting the pilot/formal block, add read-only GPU telemetry at about
0.5s intervals to every run, identical for all configurations and gates.
Abort the block if either GPU has three consecutive samples with utilization
>=80% and SM clock<=300MHz, or if telemetry fails. Keep invalid artifacts;
do not silently exclude failed runs from a claimed completed matrix. This is
a guard against the observed clock fault, not proof of zero hardware drift.
The workload, endpoints, repetitions and opportunity criterion are unchanged.
