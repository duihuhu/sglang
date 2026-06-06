# 08 — Throughput & Timing

General throughput benchmarks, Gantt chart benchmarks, quick timing scripts, and timeline validation.

## Scripts

| Script | Purpose |
|--------|---------|
| `throughput_bench.py` | General throughput benchmark with configurable params |
| `run_gantt_bench.py` | Run benchmark collecting data for Gantt chart generation |
| `run_quick_timing.sh` | Shell script for quick timing sanity checks |
| `test_m3_timeline.py` | Validate M=3 pipeline timeline with detailed AFD_TIMELINE logs |

## Usage

```bash
# Throughput benchmark
python benchmark/af_bench/08_throughput/throughput_bench.py

# Quick timing check
bash benchmark/af_bench/08_throughput/run_quick_timing.sh

# Validate M=3 timeline
python benchmark/af_bench/08_throughput/test_m3_timeline.py
```
