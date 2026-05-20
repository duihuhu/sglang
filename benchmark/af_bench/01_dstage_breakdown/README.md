# 01 — Pipeline D-Stage Breakdown Analysis

Analyzes the per-layer, per-micro-batch timing of the Attention and FFN stages in the PD+AF pipeline. Identifies pipeline bubbles caused by schedule mismatch between DA and DF.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/analyze_pipeline_breakdown.py` | Parse raw logs → compute per-stage A/F timing, bubble analysis |
| `scripts/analyze_pipeline_breakdown_v2.py` | Extended breakdown with micro-batch-level granularity |
| `scripts/analyze_schedule.py` | Analyze schedule divergence between DA and DF |
| `scripts/plot_dstage_pipeline.py` | Basic dstage pipeline visualization |
| `scripts/plot_dstage_pipeline_v2.py` | Improved per-layer stacked charts |
| `scripts/plot_dstage_pipeline_v3.py` | Gantt-style pipeline timeline |
| `scripts/plot_gantt_with_breakdown.py` | Gantt chart with compute/wait breakdown overlay |

## Output

- **`charts/`** — All generated visualizations (`.png` + `.svg`):
  - `dstage_breakdown_m{1-5}_l4.png` — Per-micro-batch breakdown for M=1~5
  - `dstage_breakdown_summary.png` — Summary comparison across M values
  - `pipeline_dstage*.png` — Pipeline stage visualizations
  - `dstage_pipeline_gantt_v3_*.png` — Gantt-style timelines
  - `per_layer_breakdown.*` — Per-layer A+F duration bars
  - `pipeline_bubble_explanation.*` — Bubble explanation diagram
  - `pipeline_full_gantt.*` — Full 64-layer × 3-MB Gantt chart
  - `schedule_divergence.*` — DA vs DF schedule mismatch

- **`results/`** — Parsed JSON data:
  - `parsed_timeline_m3.json` — 384-step per-stage timing
  - `breakdown_stats_m3.json` — Aggregate CUDA event timing
  - `analysis_summary.json` — Summary metrics

## Key Finding

DA and DF execute **different** pipeline schedules, causing ~67% of forward pass time spent waiting for UCX transfers, not computing. GPU utilization is only ~33%.

## Detailed Analysis Report

See [`breakdown_analysis_report.md`](breakdown_analysis_report.md) for the complete analysis writeup including data tables, schedule divergence explanation, and bottleneck analysis.

## Usage

```bash
# Run full analysis + chart generation
python benchmark/af_bench/utils/gen_all_breakdown_charts.py

# Individual analysis
python benchmark/af_bench/01_dstage_breakdown/scripts/analyze_pipeline_breakdown.py
```
