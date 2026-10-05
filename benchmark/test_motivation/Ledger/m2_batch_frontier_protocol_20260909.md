# M2 batch admission validation protocol

Frozen before new GPU measurements on 2026-09-09. This protocol supplements,
and does not rewrite, the 2026-08-28 admission-batch gates.

## Questions and boundaries

1. With identical state/KV capacity and graph coverage, does admission
   concurrency change queueing, execution and neighbor latency?
2. Do native mixed chunked prefill and a smaller chunk budget remove the
   observed loss? This is a mechanism control, not a complete Sarathi-Serve
   reproduction, and not a FairBatching reproduction.
3. At the same concurrency limit, does changing the candidate ordering change
   actual batch composition and the efficiency/isolation frontier?
4. Does a calibrated strong dynamic batching policy leave an avoidable loss?
   This last question remains OPEN until the relevant implementations and
   held-out comparisons actually exist. A positive configuration sweep cannot
   be promoted to a SOTA-wide result.

## Environment and controls

- Host 182.43.57.115, container sglang, /workspace/sglang.
- Only physical GPU 6 and 7, Kimi-Linear-48B-A3B, TP=2.
- State slots=480, KV tokens=196608, startup max-running=96.
- Change the existing pp_max_micro_batch_size runtime admission control only
  while idle; verify readback and observed peak active count. Runtime limits
  9, 24, 48 and 96 do not rebuild memory pools or CUDA graphs.
- Decode graph coverage stays at max batch size 96; prefill graphs disabled
  to prevent chunk-size-dependent graph capture from becoming a hidden factor.
- Native batching modes: separate-4096, mixed-4096 and mixed-1024. Each mode
  has an explicit chunk size and max-prefill token budget 16384.
- Preserve cache implementation and unique per-request salts. Require zero
  prefix hits/retractions/failures; no radix on/off claim in this stage.
- Reuse exactly the same input token IDs and requested output lengths across
  configurations and policies in each repetition. Equal-time ordering reverses
  between repetitions. This also controls MoE input-content differences.
- Record the complete server readback, GPU UUIDs, launch command, source file
  hashes, arrival deviations and batch/resource traces.

## Workloads and policies

Use the M1 partition workload: 8 waves, 1.5 seconds apart. Each tenant offers
16384 input and 512 output tokens per wave. Packed has one request; split has
16 requests. Split factor 32 and changed arrival rates are held-out workloads,
not silently pooled into the first configuration scan.

Each cap/repetition includes matched split-only and packed-only FCFS isolation.
Mixed policies are local token-VTC and the existing frozen-score instant-share
control. The latter is not full DRF and is not claimed as a new design.
All first-stage configurations are reported, not only a winning cap.

Warmup artifacts are explicitly excluded. Start with smoke, then a one-repeat
screen, then independent confirmation repetitions for any claimed improvement.
Confirmatory comparisons must freeze candidate and comparator before running.
At least five paired confirmation repetitions are needed for a benefit claim;
the one-repeat screen is descriptive only.

## Outcomes and interpretation

Primary measurements: both tenants' absolute p95 JCT, makespan and successful
completed work. Report queue p95 where actually available, active seconds per
request, decode interval, matched-isolation slowdown, peak state/KV and batch
composition. Frozen-score instant-share has no scheduler queue-wait trace;
missing values stay null, never zero.

For each repetition, compute a three-dimensional configuration frontier using
split p95, packed p95 and makespan (all minimized). An observed dominance is
descriptive and does not prove SOTA inferiority or a fairness guarantee.

For a confirmatory avoidable-loss claim, preselect the primary latency endpoint.
Require its paired relative improvement to have a 95% bootstrap CI above zero,
and mean improvement at least 10%; the other tenant's p95 and makespan must
each have an upper 95% relative-regression CI at most 5%. Report all endpoints
and token-service observations, not just slowdown-gap reduction. These
noninferiority margins are experimental criteria, not VTC fairness guarantees.
For many tested candidates, screening selects one pair and fresh confirmation
tests it; do not attach unadjusted significance claims to the entire screen.

No observed benefit, or a mixed-batching baseline that removes the loss, is a
valid result. In that case narrow the motivation instead of inventing a new
scheduler. Broad multi-tenant fairness claims additionally require legal
backlogged-window service accounting, not merely equal completed token totals.

## Related mechanism controls

- Sarathi-Serve: chunked prefill and stall-free schedules. The native mixed
  implementation here tests this family of mechanisms, not its entire system:
  https://www.usenix.org/conference/osdi24/presentation/agrawal
- FairBatching: adaptive time budgets, request slack and urgent/non-urgent
  decode selection. These are NOT implemented by enabling mixed chunks:
  https://arxiv.org/html/2510.14392v1
- SOLA, JITServe and Equinox need separate faithful adaptation and calibration
  before labeling a comparison with those names. No OOM-prone KV-only memory
  checker will be used as an artificially weak hybrid baseline.

## Safety

The new runner refuses an occupied port or GPUs. It creates a unique result
directory and an exclusive experiment lock. It only terminates the process
group it started, never uses name-wide pkill, and does not overwrite old raw
results or modify SGLang production scheduling code.

## Confirmation amendment, frozen after screening

The selection is recorded in m2_confirmation_selection_20260909.json: keep
separate-4096 and token-VTC fixed, compare cap 96 with cap 48, use fresh
repetitions 10 through 14. Independently test factors 16 and 32 with the same
thresholds; never pool these workloads or replace a failed candidate post hoc.
Factor 32 doubles split request count while preserving offered token totals.

The mixed-4096 screening split-only/cap48 run showed a slow first wave
(mean JCT 2.921s versus about 0.94s later), despite negligible arrival lateness.
One mixed wave is therefore insufficient warmup for all execution shapes.
Before confirmation, run eight-wave warmups of both isolations and mixed VTC
at both selected caps. Exclude these predeclared warmups, retain all formal
repetitions, and do not selectively drop slow formal samples. This amendment
does not change screening results or the outcome thresholds.

An additional relevant baseline is LPRS/APC in Fairness-Aware and
Latency-Controllable Scheduling for Chunked-Prefill LLM Serving (2026-06):
https://arxiv.org/html/2606.09061v1 and its official implementation
https://github.com/Charmstok/Chunked_Prefill_Serving . It combines learned
batch-latency prediction and active-prefill control. It is not implemented
in this experiment; Kimi-Linear adaptation and local profiling are necessary
before using its name for a comparative claim.
