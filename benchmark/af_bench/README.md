# AF Bench — Pipeline Analysis & Benchmark Suite for Attention-FFN Disaggregation

This directory contains all benchmark scripts, analysis tools, and visualizations for the **Attention-FFN Disaggregation (AFD)** pipeline research on SGLang.

Originally from `benchmark/test_motivation/AzurePublicDataset/pipeline_analysis/`, reorganized here for clarity.

## Quick Start

Each subdirectory is self-contained. Run benchmark scripts from the SGLang repo root:

```bash
# Example: run a micro-batch sweep experiment
python benchmark/af_bench/04_msweep/run_m3.py

# Example: run pipeline breakdown analysis
python benchmark/af_bench/01_dstage_breakdown/scripts/analyze_pipeline_breakdown.py
```

## Directory Structure

| Directory | Description |
|-----------|-------------|
| [01_dstage_breakdown/](01_dstage_breakdown/) | Pipeline stage breakdown analysis — per-layer A/F timing, schedule analysis, Gantt charts |
| [02_communication/](02_communication/) | UCX RDMA vs CUDA IPC communication benchmarks — gap analysis, wire breakdown, cycle precision |
| [03_pipeline_viz/](03_pipeline_viz/) | Pipeline visualization scripts — Gantt charts, pipeline schematic diagrams |
| [04_msweep/](04_msweep/) | Micro-batch number sweep (M=1~5) — throughput/latency vs micro-batch count |
| [05_qps_sweep/](05_qps_sweep/) | QPS/request concurrency sweeps — throughput/latency under varying load |
| [06_af_experiments/](06_af_experiments/) | AF-only experiments — pure AF (no PD), baselines, CUDA graph tests, final comparison |
| [07_stress_timing/](07_stress_timing/) | Stress tests, sequence length sweeps, async verification, bubble breakdown |
| [08_throughput/](08_throughput/) | General throughput benchmarks, Gantt bench, quick timing scripts |
| [logs/](logs/) | Raw experiment logs (bubble, IPC, wire breakdown, throughput) |
| [utils/](utils/) | Shared utility scripts (log parsing, chart generation) |

## Common Test Configuration

- **Model**: Qwen3-32B (64 layers, hidden=5120, kv_heads=8)
- **Architecture**: PD+AF Disaggregated (4 GPUs: DF/DA/PF/PA)
- **TP per module**: 1 (4 GPUs total)
- **Interconnect**: UCX RDMA (mlx5_4) / CUDA IPC
- **GPU**: NVIDIA A800-SXM4-80GB
- **Server launch pattern**: See individual scripts for `sglang serve` commands
- **Environment variable**: `AFD_DETAILED_TIMING=1` enables per-step timing output

## Key Finding

The M=3 pipeline suffers **~33% GPU utilization** because:
1. Each layer requires 2 sequential UCX transfers (DA→DF, DF→DA)
2. No overlap between DA and DF compute due to data dependencies
3. UCX transfer latency dominates (~200ms cumulative wait per forward pass)

See [01_dstage_breakdown/README.md](01_dstage_breakdown/README.md) for detailed analysis.
