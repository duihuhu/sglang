# ILP solver cost ablation

Paper table `tab:tier1-solver-breakdown` (average ILP latency breakdown, scheduling window = 5 min).

## Layout

- `data/solver_avg_breakdown_summary.csv` — 6 rows (Code + Conversation × G=8/16/32)
- `data/solver_avg_breakdown_raw.json` — 180 measured solver runs
- `charts/plot_solver_avg_breakdown.py` — stacked breakdown figure
- `charts/solver_avg_breakdown.pdf` — output figure
- `scripts/benchmark_solver_avg_breakdown.py` — benchmark entry
- `scripts/benchmark_tier1_overhead.py` — instrumentation helper (imported by benchmark)

## Regenerate

```bash
# Run benchmark (requires energy model + Qwen3 profiles)
python3 benchmark/AFlex_bench/multi_node/more_test/Ablation/ILP_solver_cost/scripts/benchmark_solver_avg_breakdown.py

# Plot
python3 benchmark/AFlex_bench/multi_node/more_test/Ablation/ILP_solver_cost/charts/plot_solver_avg_breakdown.py
```

## Table mapping

From `solver_avg_breakdown_summary.csv`:

- **Search** ≈ `max(decode_enumeration_mean_ms, prefill_enumeration_mean_ms)`
- **Other** = `pareto_mean_ms + global_search_mean_ms + other_mean_ms`
- **Total** = Search + Other
- **Overhead (%)** = Total / (5×60×1000) × 100
