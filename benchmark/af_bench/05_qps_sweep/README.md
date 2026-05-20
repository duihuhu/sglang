# 05 — QPS / Concurrency Sweeps

Measures throughput and latency under varying request rates and concurrency levels. Evaluates how PD+AF disaggregation behaves under load.

## Scripts

| Script | Purpose |
|--------|---------|
| `run_qps_sweep.py` | QPS sweep across standard range |
| `run_qps_sweep_m1.py` | QPS sweep with M=1 configuration |
| `run_qps_sweep_m3.py` | QPS sweep with M=3 configuration |
| `run_qps_sweep_native.py` | QPS sweep on native (non-disaggregated) baseline |
| `run_qps_sweep_parallel.py` | Parallel QPS sweep (multiple concurrent clients) |
| `run_high_qps_sweep.py` | Extended high-QPS range sweep |

## Data

Raw logs and JSON results are in `logs/throughput_logs/`. Key files:

- `qps_sweep_*.log` / `qps_sweep_m{1,3}_*.log` — Server logs per config
- `qps_sweep_results.json`, `qps_sweep_m1_results.json`, `qps_sweep_m3_results.json`
- `hqsweep_m{1,2,3}_*.log` — High QPS sweep logs
- `qps_sweep_native*.log` — Native baseline logs

## Usage

```bash
# Standard QPS sweep
python benchmark/af_bench/05_qps_sweep/run_qps_sweep.py

# Native (no disaggregation) baseline
python benchmark/af_bench/05_qps_sweep/run_qps_sweep_native.py
```
