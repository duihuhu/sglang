# 06 — AF-Only Experiments & Baselines

Experiments that focus on the AF (Attention-FFN) disaggregation aspect, often without PD (prefill-decode) disaggregation. Includes baselines for comparison.

## Scripts

| Script | Purpose |
|--------|---------|
| `run_af_only.py` | Pure AF disaggregation benchmark (no PD) |
| `run_pure_af.py` | AF-only with full control over TP and overlap settings |
| `run_baselines.py` | Run baseline configurations (native TP4, native TP2, etc.) |
| `run_final_comparison.py` | Final comparison across all configurations |
| `run_cuda_graph_test.py` | Test CUDA graph impact on AF performance |
| `run_ovdec.py` | Overlap decode — test decode overlap strategies |

## Data

Raw logs and JSON results in `logs/throughput_logs/`. Key files:

- `pure_af_m{1,3}_{a,f}.log` — AF-only logs
- `pure_af_m{1,3}_tp2_{a,f}.log` — AF-only with TP=2
- `pure_af_results.json`, `pure_af_tp2_results.json`
- `baselines_4gpu.json` — 4-GPU baseline comparison
- `final_comparison.json` — Final aggregated comparison
- `cuda_graph_comparison.json` — CUDA graph on/off comparison

## Usage

```bash
# Run AF-only benchmark
python benchmark/af_bench/06_af_experiments/run_pure_af.py

# Run baselines
python benchmark/af_bench/06_af_experiments/run_baselines.py

# Final comparison
python benchmark/af_bench/06_af_experiments/run_final_comparison.py
```
