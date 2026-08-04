# Dynamic micro-batch ablation

Paper figure: `charts/dynamic_m_design_aligned.pdf`

## Layout

- `charts/plot_design_aligned_motivation.py` — generate the design-aligned A/F timeline figure
- `data/design_aligned_cases.json` — frozen per-layer A/F latencies used by the plot
- `scripts/run_breakdown.py` — low/high load M=1/M=2 breakdown benchmark (Qwen3-32B, 4-GPU PD+A/F)
- `scripts/parse_breakdown.py` — parse breakdown logs into summary JSON/CSV
- `scripts/search_unequal_partition.py` — profile-guided M=2 partition search
- `scripts/analyze_node3_bs_curve.py` — batch-curve validation from measured decode sweeps

## Regenerate figure

```bash
python3 benchmark/AFlex_bench/multi_node/more_test/Ablation/Dynamic_micro_batch/charts/plot_design_aligned_motivation.py
```

## Data provenance

| Case | Source |
|------|--------|
| low_m1 / low_m2 | `run_breakdown.py` m1-low / m2-low (bs=2 vs 1+1) |
| balanced_m1 / balanced_m2 | DA 210 MHz + DF 300 MHz, total bs=384 |
| unequal | Measured batch curves + partition search (128+256) |

Batch-curve raw data: `data/decode_bs_curve_node3_shortctx_run*.txt`, `decode_bs_curve_node3_physical_f300.txt`.
