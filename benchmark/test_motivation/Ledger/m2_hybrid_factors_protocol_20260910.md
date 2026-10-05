# M2 hybrid admission factors: protocol

Frozen before the new GPU runs. Old cap48/cap96 gates remain unchanged and
are auxiliary evidence, not acceptance criteria for this study.

## Question

Do request activation, separate state/KV constraints, and persistent state
occupancy introduce decision-relevant factors not adequately handled by the
tested admission/batch formation combinations? Distinguish structural facts,
causal effects, and evidence of an existing policy's avoidable loss.
Do not require an unfavorable result for existing methods.

## Shared controls

- Kimi-Linear-48B-A3B, TP2, physical GPUs 6/7 only, port 30127.
- Disable radix cache: one recurrent-state slot per active request, no prefix
  reuse, cached tokens or retractions. This removes checkpoint-buffer changes
  as a confound and does not model prefix caching.
- Explicitly disable both prefill and decode CUDA graphs across all regimes;
  chunk=4096, max-prefill-tokens=16384, mem-fraction-static=.85.
- Regimes: wide (state128, KV131072, requested max-running64, runtime cap64),
  state-low (state8, KV131072, requested64, effective/runtime8),
  cap-control (wide pools/effective64, runtime8),
  kv-low (state128, KV16384, effective/runtime64).
- The cap-control is mandatory: an effect reproduced by this control is not
  evidence that KDA computation or physical pool exhaustion uniquely causes it.
- Same token IDs and output lengths across regimes/policies within each paired
  repetition. Swap submission order without changing prompt content.
- Warm each workload/policy before formal samples. Three repetitions for
  structural activation measurements; five for paired policy/pressure probes.
- Resource events report scheduler-observed activation/release and aggregate
  allocated state/KV. They are not GPU kernel timings or exact allocator calls.
- Admission-block telemetry records actual request-limit or PrefillAdder return
  reason and remaining budgets. NO_TOKEN alone is not labeled state exhaustion.
- Keep all formal runs. Require successful exact output, zero cached tokens,
  retractions and trace drops. Report any failed validity conditions explicitly.

## E1: activation cost versus token work

On wide pools and FCFS, process aggregate 16384 input tokens as 1, 4, 16 or 32
requests. Each requests 128 output tokens so states remain observable; total
output work differs and end-to-end performance across these shapes is NOT a
fairness/performance comparison. Inspect event-level state/KV increments and
chunked prefill traces. Structural support requires one state per active
request and observations where KV advances while state holders stay fixed.
This establishes a resource factor, not a scheduler defect.

## E2: fixed candidates, distinct resource constraints

Each run contains 16 short requests (P512/O64) and four long requests
(P8192/O64). Test token-VTC and frozen-score instant-share, both with short-first
and long-first submission. Cohort starts are 0 and .05 seconds; within a cohort
requests are staggered by .001 seconds. This is a submission-order intervention,
not a faithful reproduction of another batch scheduler. Record actual resource
activation order and do not assume all candidates were queued simultaneously.

Across the four regimes record both groups' p95 JCT, makespan, first-observed
queue delay, peak state/KV, service totals and typed admission blocks. Five
paired repetitions use new inputs; configurations retain identical contents.
Identify the actual bottleneck, not just the regime's requested name.

Report VTC versus instant effects with run-level paired bootstrap CIs for each
predeclared order/regime; no post-hoc best-order oracle or minimum-regret gate.
These are exploratory multiple contrasts, not simultaneous significance claims.
A ranking reversal alone does not demonstrate meaningful loss or novelty.
Unchanged service objectives and stronger calibrated baselines are still needed
before a general "existing methods fail on hybrid" claim.

## E3: persistent occupancy and downstream admission

Use eight P512 background requests with O32 or O256, then at t=3.0s send one
P512/O16 probe. Use FCFS in all four regimes, five paired repetitions. Background
work intentionally differs; do not call this an equal-total-work comparison.
Report probe queue/TTFT/JCT, whether all background holders remain at probe
arrival, and whether probe activation follows their resource release.

The key contrast is whether long-versus-short background holding has extra
probe queue impact in state-low relative to wide. Cap-control and kv-low are
negative/alternative controls. If both lengths are still active, or both have
finished by probe arrival, mark the intended lifecycle contrast inconclusive.
Do not implement a lifetime predictor or equate all waiting with state tax.

## Scope and safety

Correct SGLang capacity checks stay enabled. Only opt-in read-only telemetry is
added; queue ordering/admission behavior is unchanged. Local token-VTC and
instant-share are the tested implementations, not complete SOTA reproductions.
Pure KV or operator controls and calibrated advanced policies remain necessary
for hybrid-specific causal and novelty claims. Run only on idle GPUs, hold the
existing GPU67 lock, terminate only the owned server process group, preserve
all prior results, and synchronize analysis rather than all raw traces.

## Pilot amendment before formal runs

The wide-pool smoke (excluded from inference) produced the 32nd background token
at approximately 2.016s with eager execution. The originally planned t=1.0s
probe therefore cannot separate short-completed from long-active holders.
Freeze probe arrival at t=3.0s for all formal E3 regimes; retain the validity
check and mark failures inconclusive. No formal outcomes were used to choose
this time. Admission telemetry also records candidates selected in the current
prefill pass, because those reservations may not yet appear in free-pool counts.
