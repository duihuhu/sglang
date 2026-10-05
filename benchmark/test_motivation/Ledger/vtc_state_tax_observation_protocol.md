# VTC state-tax observation protocol

Date: 2026-08-26

## Scope and question

This experiment deliberately disables cross-request prefix reuse and asks:

> Can VTC remain fair in its declared scalar service while request partitioning
> in a Hybrid-Attention model creates unequal recurrent-state pressure and
> unequal tenant slowdown? Can a profiled or fixed-request-tax VTC repair the
> mismatch, or is an explicit state/KV resource signal required?

The experiment does not test DLPM. Every request has a unique `cache_salt`, the
cache is flushed before every artifact, and a formal artifact is invalid if any
request reports nonzero `cached_tokens`.

Here, "short requests are more expensive" has a precise and limited meaning:
each active request pays a fixed recurrent-state allocation, independent of its
token length. Therefore, at equal aggregate input/output tokens, splitting work
into many short requests raises instantaneous state-slot demand and changes the
state/KV resource composition. It does **not** mean that one short request has a
larger total footprint or more compute than one long request. The experiment
separately reports peak demand and integrated resource-time so that these two
claims cannot be conflated.

All runs use `/models/Kimi-Linear-48B-A3B`, TP=2, and physical NVIDIA A800 GPUs
6 and 7 in the `sglang` container on `182.43.57.115`. Smoke runs, if any, are
excluded from selection and every decision.

## Cost model under test

Standard VTC charges scalar service

\[
W_u = w_p n^p_u + w_q n^q_u,
\]

with `w_p=1,w_q=2`. At fixed total input and output tokens this cost is
invariant to request partitioning. Hybrid active memory is not:

\[
M_u(t) = S N_u^{active}(t) + K L_u^{resident}(t).
\]

The profiled VTC baseline uses the preregistered nonnegative polynomial

\[
h(p,q)=r+a p+b p^2+c q+d p q+e q^2.
\]

Its coefficients are fitted once from isolated A800 latency profiles and frozen
before any policy screening. The request-tax family is the simpler special case

\[
h(p,q)=r+p+2q.
\]

## Profile calibration

Run the Cartesian product `P={1024,4096,16384}` and `O={32,128,512}` as
single-request, no-reuse FCFS artifacts with three repetitions: 27 formal
artifacts. Fit the six nonnegative coefficients with NNLS using scaled features
`[1,P/1024,(P/1024)^2,O/32,(P/1024)(O/32),(O/32)^2]`, then convert them to raw
token units for the scheduler. The fit is acceptable when median relative
prediction error is at most 20% and maximum relative error is at most 35%.
Coefficients and errors are saved before screening.

## Partition and bottleneck cells

Every mixed wave gives tenant `split` and tenant `packed` exactly 16,384 input
tokens and 512 output tokens. `packed` submits one `P=16384,O=512` request;
`split` submits `f` requests of `P=16384/f,O=512/f`.

| cell | split factor | interval | state cap | KV cap | purpose |
|---|---:|---:|---:|---:|---|
| f1-base | 1 | 1.5s | 48 | 196608 | identical-shape negative control |
| f4-base | 4 | 1.5s | 48 | 196608 | moderate partitioning |
| f16-base | 16 | 1.5s | 48 | 196608 | calibration state pressure |
| f32-base | 32 | 1.5s | 48 | 196608 | high partitioning |
| f16-state-relaxed | 16 | 1.5s | 480 | 196608 | remove the state wall |
| f16-kv-relaxed | 16 | 1.5s | 48 | 393216 | strengthen relative state pressure |
| f16-full | 16 | 1.5s | 480 | 393216 | relaxed-capacity control |
| f16-fast | 16 | 1.0s | 48 | 196608 | higher request density |
| f16-slow | 16 | 3.0s | 48 | 196608 | lower request density |

Screening uses four waves and one repetition. Confirmation uses eight waves
and three new repetitions. Repetitions 1/3 place split first on counter ties;
repetition 2 places packed first. Every mixed repetition has matching FCFS
split-only and packed-only isolation artifacts.

## Policies

Screening runs, per cell:

- FCFS;
- request-count VTC (`r=1,w_p=w_q=0`);
- standard token VTC (`r=0,w_p=1,w_q=2`);
- request-tax VTC with `r={1024,4096,8192,16384,32768,65536}` and
  `w_p=1,w_q=2`;
- frozen profiled VTC;
- instantaneous normalized dominant state/KV share.

The global request tax is selected only from `f4-base`, `f16-base`, and
`f16-kv-relaxed`, minimizing mean isolation-normalized p95 slowdown gap subject
to makespan at most 5% above standard VTC. A per-cell request-tax oracle is also
frozen from screening, but is an upper bound rather than a deployable policy.

Confirmation reruns matching isolation, FCFS, request-count VTC, standard VTC,
profiled VTC, the frozen global request tax, the frozen per-cell request-tax
oracle, and instantaneous state/KV share. Selection and confirmation requests
use disjoint run IDs and salts.

## Metrics

VTC-native metrics:

- offered and completed input/output service per tenant;
- final and maximum observed virtual-counter gap;
- work conservation, total throughput, and makespan.

Hybrid physical metrics:

- exact active-request-seconds, state-slot-seconds, and KV-token-seconds;
- peak active requests, peak state slots, and peak KV tokens;
- per-token state/KV resource-time and peak state-to-KV resource composition;
- exact VTC queue-wait distributions and peak waiting requests per tenant;
- matching-isolation normalized p95 JCT/TTFT and their tenant gap;
- SLO goodput, failures, retractions, and cached tokens.

## Preregistered observation gates

All gates require zero failed requests, zero cached tokens, exact resource
traces, exact resolved policy/weight metadata, and physical-GPU metadata `6,7`.

### O1: VTC abstraction mismatch

On confirmed standard VTC in both `f16-base` and `f32-base`:

1. completed input and output service differs by at most 1% between tenants;
2. maximum observed VTC-counter gap is no greater than the conservative bound
   of two maximum full-request charges (one admission lead plus one outstanding
   decode-charge lead);
3. at least one exact per-token state/KV resource-time dimension differs by at
   least 2x, and the tenant peak state-to-KV composition differs by at least
   4x; and
4. the absolute p95 normalized-slowdown gap is at least 50 percentage points.

This gate says VTC succeeds in its own currency while that currency is
misaligned with physical resource service and user slowdown.

### O2: hybrid state specificity

For standard VTC, the `f16-state-relaxed` slowdown gap must be at least 20%
smaller than `f16-base`, while `f16-kv-relaxed` must not reduce it by more than
the state-relaxed cell. The `f1-base` identical-shape control must have a gap
below 25 percentage points. This ties the partition effect to recurrent-state
pressure instead of tenant identity or arrival order.

### O3: scalar repair insufficiency

On fresh confirmation data, compare the strongest of standard VTC, frozen
profiled VTC, global request-tax VTC, and the frozen per-cell request-tax oracle
against instantaneous state/KV share. The gate passes when instantaneous share:

1. reduces mean slowdown gap by at least 15%;
2. wins in at least 7/9 cells; and
3. changes mean makespan by at most +5% relative to standard VTC.

If O1/O2 pass but O3 fails, the observation remains a critique of standard VTC
but does not establish that scalar customization is insufficient. If O3 passes,
the motivation supports an explicit hybrid resource signal, not any particular
historical-debt or lifecycle scheduler.

## Interpretation boundary

The experiment does not claim that VTC violates its published token-service
fairness guarantee. It tests whether that declared service remains a sufficient
currency for Hybrid-Attention serving. It also does not claim an
instantaneous-share scheduler is a complete fair-serving solution; that policy
lacks VTC's long-term service bound and is used only as the simplest
physical-resource-aware diagnostic baseline.
