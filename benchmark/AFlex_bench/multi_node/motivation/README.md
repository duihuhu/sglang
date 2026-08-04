# Motivation figures

Paper motivation figures (fig1–fig5). `data/` is a symlink to `energy_model/Qwen3-32B/data` (no duplicate copies).

## Layout

```
motivation/
├── data/                           # → ../../energy_model/Qwen3-32B/data
├── fig1_heter_homo/                  plot script + fig1_combined_2panel.pdf
├── fig2_latency_freq/
├── fig3_energy_heatmap/
├── fig4_AF_ratio/
└── fig5_reshard/
    ├── plot_fig5_combined_2panel.py + fig5_combined_2panel.pdf
    └── data/                         fig5-only JSON/CSV inputs
```

## Regenerate all figures

```bash
ROOT=benchmark/AFlex_bench/multi_node/motivation
python3 $ROOT/fig1_heter_homo/plot_fig1_heter_vs_homo.py
python3 $ROOT/fig2_latency_freq/plot_fig2_3panel.py
python3 $ROOT/fig3_energy_heatmap/plot_fig3_decode_lat_vs_energy.py
python3 $ROOT/fig4_AF_ratio/plot_fig4_combined_4panel.py
python3 $ROOT/fig5_reshard/plot_fig5_combined_2panel.py
```

## Figure index

| Figure | PDF | Script | Data source |
|---|---|---|---|
| Fig.1 Heter vs Homo | `fig1_heter_homo/fig1_combined_2panel.pdf` | `plot_fig1_heter_vs_homo.py` | `data/v1_layer_profile/{prefill,decode}_data_v1.txt` |
| Fig.2 Latency & Energy vs Freq | `fig2_latency_freq/fig2_combined_2panel.pdf` | `plot_fig2_3panel.py` | same layer profile (TP=4, bs=256) |
| Fig.3 Decode sensitivity heatmap | `fig3_energy_heatmap/fig3_combined_4panel.pdf` | `plot_fig3_decode_lat_vs_energy.py` | `decode_data_v1.txt` |
| Fig.4 F/A ratio & bubble | `fig4_AF_ratio/fig4_combined_4panel.pdf` | `plot_fig4_combined_4panel.py` | both profile files |
| Fig.5 Startup & AF/DVFS overhead | `fig5_reshard/fig5_combined_2panel.pdf` | `plot_fig5_combined_2panel.py` | `fig5_reshard/data/*.json` + `dvfs_switch_summary_latest.csv` |

### Fig.5 panel (a)

`fig5_reshard/data/startup_overhead_by_tp.json` — Qwen3-32B cold-start breakdown by TP (no CUDA graph).

### Fig.5 panel (b)

- `af_all_tp_results.json` — native vs AF-disagg TTFT/TPOT at TP1/2/4/8
- `dvfs_switch_summary_latest.csv` — DVFS `SetGpuLockedClocks` wall-time p50 by GPU count
