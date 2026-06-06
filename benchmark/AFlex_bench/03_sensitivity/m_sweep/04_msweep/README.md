# 04 — Micro-batch Number Sweep (M=1~5)

Sweeps the pipeline micro-batch count (M) to study its impact on throughput, latency, and pipeline bubble ratio.

## Scripts

| Script | Purpose |
|--------|---------|
| `run_m1.py` | Run benchmark with M=1 (no pipelining) |
| `run_m3.py` | Run benchmark with M=3 (standard) |
| `run_m5_only.py` | Run benchmark with M=5 |
| `run_msweep.py` | Sweep M=1~5 in sequence, collect all metrics |
| `run_msweep_m35.py` | Focused sweep: M=3 vs M=5 comparison |
| `run_m1_vs_m3_sweep.py` | Direct M=1 vs M=3 throughput/latency comparison |
| `run_m3_async.py` | M=3 with async recv pipeline optimization |
| `run_m3_vs_m1_highconcurrency.py` | M=3 vs M=1 under high concurrency |
| `run_m3_large_batch.py` | M=3 with large decode batch size |

## Data

Raw logs and result JSONs are in `logs/throughput_logs/` (shared across experiments). Key files:

- `msweep_m{1-5}_*.log` — Per-server logs for each M value
- `msweep_m{1-5}_results.json` — Parsed results per M
- `msweep_all_results.json` — Aggregated results across all M values
- `m1_vs_m3_sweep.json` — Direct comparison results

## Usage

```bash
# Full M sweep
python benchmark/af_bench/04_msweep/run_msweep.py

# Compare M=1 vs M=3
python benchmark/af_bench/04_msweep/run_m1_vs_m3_sweep.py
```
