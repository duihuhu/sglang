# 02 — UCX / IPC Communication Benchmarks

Benchmarks the inter-GPU communication layer — UCX RDMA vs CUDA IPC. Measures transfer latency, cycle-level breakdown, and wire time for small decode tensors.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/bench_ucx_gap_analysis.py` | Measure UCX transfer gap — time between send and receive |
| `scripts/bench_ucx_gap_analysis_v2.py` | Refined gap analysis with per-micro-batch tracking |
| `scripts/bench_ucx_transfer.py` | Standalone UCX transfer bandwidth/latency benchmark |
| `scripts/run_ipc_breakdown.py` | Run benchmark with CUDA IPC backend, collect timing |
| `scripts/run_wire_breakdown.py` | Run benchmark with wire-level breakdown logging |
| `scripts/plot_full_cycle_breakdown.py` | Plot full transfer cycle breakdown (prepare/transfer/complete) |
| `scripts/plot_full_cycle_precise.py` | Precise cycle timing with CUDA events |
| `scripts/plot_ipc_cycle_precise.py` | IPC-specific cycle timing visualization |
| `scripts/plot_wire_breakdown.py` | Wire-level transfer breakdown chart |
| `scripts/config_ipc_breakdown.json` | Configuration for IPC breakdown experiments |

## Output

- **`charts/`** — Communication timing charts:
  - `gantt_cycle_sync_merged.png` — Sync point timing across cycles
  - `gantt_cycle_ucx_vs_ipc.png` — UCX vs IPC cycle comparison
  - `gantt_full_cycle_breakdown.png` — Full cycle breakdown bars
  - `gantt_full_cycle_precise.png` — Precise cycle Gantt
  - `gantt_ipc_cycle_precise.png` — IPC cycle timing
  - `gantt_ucx_vs_ipc_breakdown.png` — Side-by-side breakdown
  - `gantt_wire_breakdown_inner.png` — Inner wire transfer breakdown
  - `ucx_vs_ipc_comparison.png` — Summary UCX vs IPC comparison

- **`logs/`** — Raw server logs from experiments:
  - `ipc_breakdown_logs/` — DA, DF, PA, PF, router logs (IPC backend)
  - `wire_breakdown_logs/` — DA, DF, PA, PF, router logs (wire breakdown mode)

## Usage

```bash
# Run IPC breakdown benchmark
python benchmark/af_bench/02_communication/scripts/run_ipc_breakdown.py

# Plot results
python benchmark/af_bench/02_communication/scripts/plot_full_cycle_breakdown.py
```
