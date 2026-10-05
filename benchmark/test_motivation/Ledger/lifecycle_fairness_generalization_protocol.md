# Lifecycle-cost fairness generalization protocol

Date: 2026-08-25

## Question

Does a frozen realized lifecycle-cost ledger improve tenant fairness across
equal-token workloads with different request splitting, prompt/output shapes,
arrival rates, and state/KV capacities, compared with strong token, request,
static-tax, instantaneous-share, and cumulative-vector baselines?

All formal runs use `/models/Kimi-Linear-48B-A3B` on physical A800 GPUs 6 and
7 with TP=2. Smoke artifacts are excluded from every decision.

## Workloads

Each wave gives both tenants exactly the same input and output tokens. The
packed tenant sends one request `(P, O)`; the split tenant sends `f` requests
`(P/f, O/f)`.

| cell | split | packed P/O | wave interval | state cap | KV cap |
|---|---:|---:|---:|---:|---:|
| base | 16 | 16384/512 | 1.5s | 48 | 196608 |
| split4 | 4 | 16384/512 | 1.5s | 48 | 196608 |
| split32 | 32 | 16384/512 | 1.5s | 48 | 196608 |
| output1k | 16 | 8192/1024 | 1.5s | 48 | 196608 |
| output2k | 16 | 4096/2048 | 1.5s | 48 | 196608 |
| arrival1 | 16 | 16384/512 | 1.0s | 48 | 196608 |
| arrival3 | 16 | 16384/512 | 3.0s | 48 | 196608 |
| state96 | 16 | 16384/512 | 1.5s | 96 | 196608 |
| kv393 | 16 | 16384/512 | 1.5s | 48 | 393216 |
| relaxed | 16 | 16384/512 | 1.5s | 96 | 393216 |

Screening uses four waves and one repetition. Confirmation uses eight waves
and three fresh repetitions. Every repetition has matching split-only and
packed-only isolation runs for normalized slowdown.

## Policies

- `fcfs`: global FCFS.
- `request-vtc`: request-count service, `request=1,input=0,output=0`.
- `token-vtc`: input/output token service, `request=0,input=1,output=1`.
- `request-tax`: token VTC plus a fixed request charge swept over
  `0,1024,4096,8192,16384,32768,65536`.
- `instant-share`: lowest current normalized dominant state/KV share first.
- `vector-debt`: lowest accumulated normalized dominant resource-time first.
- `lifecycle`: lowest accumulated
  `s * state-slot-s/state-cap + k * KV-token-s/KV-cap` first. Screening fixes
  `k=1` and sweeps `s=0,0.25,0.5,1,2,4,8`.

For each cell, screening freezes the best throughput-bounded request-tax
baseline, strongest non-lifecycle baseline, and lifecycle oracle. Confirmation
always reruns token VTC, request VTC, instantaneous share, vector debt, the
selected request-tax, frozen lifecycle `s=0.25,k=1`, and the selected lifecycle
oracle. Selection and confirmation data are disjoint.

## Metrics and gates

Primary fairness metric is the absolute gap between split and packed p95
normalized slowdown, where each tenant is normalized by its same-repetition
isolation p95. Supporting metrics are tenant p95 JCT/TTFT, SLO goodput,
makespan, exact state-slot-seconds, exact KV-token-seconds, failures, and
retractions.

A policy is throughput-bounded when its makespan is at most 5% above token
VTC. The frozen lifecycle gate passes only if:

1. all formal artifacts have exact scheduler traces, resolved policy metadata,
   physical-GPU metadata `6,7`, and zero failed requests;
2. mean slowdown-gap improvement over the per-cell strongest static baseline
   is at least 15%, with the paired direction in at least 8/10 cells;
3. mean makespan change relative to token VTC is at most +5%; and
4. median direct slowdown-gap regret relative to the per-cell lifecycle oracle
   is below 15%.

If gate 2 passes but gate 4 fails, realized lifecycle accounting is supported
but a frozen resource price is not; online scarcity pricing becomes the next
mechanism gate. If a tuned request-tax or vector baseline matches the lifecycle
policy, the cost-denominated fairness mechanism claim does not pass.

## Independent strongest-baseline kill confirm

The first confirmation exposed `instant-share` as the strongest confirmed
non-lifecycle baseline in all ten cells, even where one-repetition screening
had selected another label. Because selecting it on the same confirmation data
would be post-hoc, a fresh kill-confirm is preregistered before inspecting any
new run. It uses new run IDs and request salts and reruns, per cell and for
three repetitions, matching isolation, token VTC, `instant-share`, and frozen
`lifecycle-s0.25-k1` only. No weights or labels are selected from kill data.

The lifecycle mechanism survives only if it improves the mean slowdown gap
over `instant-share` by at least 15%, wins in at least 8/10 cells, and keeps
mean makespan within +5% of token VTC. This independent gate supersedes both
the optimistic screening-frozen comparison and the conservative post-hoc
comparison in the first confirmation.

## Scope boundary

The earlier A800 lifecycle request-ranking gates remain authoritative for
admission/eviction efficiency. This protocol does not rerun failed ordering
mechanisms under new names. DCP8 is not claimed as measured because the user
restricted the experiment allocation to physical GPUs 6 and 7; topology
dependence remains an external SGLang observation until a real multi-GPU DCP
experiment is authorized.
