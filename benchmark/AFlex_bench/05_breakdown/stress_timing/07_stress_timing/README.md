# 07 — Stress Tests & Timing Analysis

Stress tests, sequence length sweeps, asynchronous verification, and pipeline bubble breakdown experiments.

## Scripts

| Script | Purpose |
|--------|---------|
| `run_stress.py` | Long-duration stress test with sustained request load |
| `run_stress2.py` | Extended stress test with QPS spike patterns |
| `run_seq_len_test.py` | Sweep input/output sequence length impact on throughput |
| `run_async_verify.py` | Verify correctness of async recv pipeline optimization |
| `run_bubble_breakdown.py` | Measure and analyze pipeline bubble ratio under various conditions |

## Data

Raw logs in `logs/throughput_logs/` and `logs/bubble_logs/`:

- `stress_comparison.json` — Stress test results
- `seq_len_comparison.json` — Sequence length sweep results
- `bubble_logs/` — Bubble analysis raw logs

## Usage

```bash
# Run stress test
python benchmark/af_bench/07_stress_timing/run_stress.py

# Sequence length sweep
python benchmark/af_bench/07_stress_timing/run_seq_len_test.py

# Bubble breakdown
python benchmark/af_bench/07_stress_timing/run_bubble_breakdown.py
```
