# Reshard cost ablation

Paper table `tab:reshard-vs-baseline` and figure `runtime_reconfiguration_timeline_activate.pdf`.

## Layout

- `data/run_{1,2,3}_paper_breakdown.json` — paper-aligned wall-clock breakdown (3 runs; identical copies from one node2 measurement)
- `data/run_{1,2,3}_full_sequence.json` — per-rank activate sub-step timings for critical-rank aggregation
- `data/three_run_summary.json` — cross-run means for prepare / component metrics
- `charts/plot_runtime_reconfiguration_timeline_activate.py` — stacked ACTIVATE breakdown figure
- `charts/runtime_reconfiguration_timeline_activate.pdf` — output figure
- `scripts/summarize_runtime_reconfiguration_breakdown.py` — regenerate `paper_breakdown` from a `full_sequence` JSON

## Regenerate figure

```bash
python3 benchmark/AFlex_bench/multi_node/more_test/Ablation/Reshard_cost/charts/plot_runtime_reconfiguration_timeline_activate.py
```

## Experiment setup

- Model: Qwen3-32B (BF16), node2, 2026-07-21
- Transitions: **t1** `A2F2→A4F4` (Expand TP2→TP4), **t2** `A4F4→A1F4` (Shrink TP4→TP1)
- Figure shows **ACTIVATE** only (PUBLISH is async in background)

## Table mapping — AFlex column

From `run_*_paper_breakdown.json`, prefill `attn` component activate sub-steps (rounded):

| Phase | Field(s) | Expand (t1) | Shrink (t2) |
|---|---|---:|---:|
| Comm / NCCL | `activate_rebuild_groups_s` (+ merged scheduler groups) | 0.71 | 0.71 |
| Weight & KV | `activate_refresh_runtime_s` (+ merged materialize) | 2.16 | 1.48 |
| Init / RDMA | `activate_refresh_bootstrap_topology_s` (+ merged consensus) | 1.10 | 1.29 |
| **Total** | sum of above | **3.97** | **3.48** |
| **Reduction vs baseline** | — | **52.2%** | **72.3%** |

These match the stacked bars in `runtime_reconfiguration_timeline_activate.pdf`.

## Table mapping — Baseline column

Baseline numbers in the paper table (Expand 8.30 s, Shrink 12.56 s) are **not** stored in this directory. They come from the traditional in-place TP reshard baseline (`reshard/Baseline/`, now removed). The closest on-disk reference was `Baseline/results/current/async_breakdown/node1_operator_test_tp1_2_4_8_idle_no_graph.json` (TP1→2→4→8 chain, not TP4→TP1 shrink).

AFlex vs baseline compares the same three coarse phases (Comm/NCCL, Weight&KV, Init/RDMA) under matching transition labels; only the AFlex activate breakdown artifacts were kept here.
