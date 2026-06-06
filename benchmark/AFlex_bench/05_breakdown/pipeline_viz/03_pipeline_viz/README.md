# 03 — Pipeline Visualization

Drawing and visualization scripts for pipeline Gantt charts, schematic diagrams, and multi-pipeline composition views.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/draw_gantt_actual.py` | Draw actual Gantt chart from real timing data (M=1, M=3) |
| `scripts/draw_multi_pipeline.py` | Multi-pipeline composition visualization |
| `scripts/draw_schematic.py` | High-level pipeline schematic diagram |
| `scripts/visualize_pipeline.py` | General pipeline visualization with configurable views |
| `scripts/plot_m3_new_schedule.py` | Plot M=3 new schedule proposal timing |

## Output

- **`charts/`** — Pipeline visualization charts:
  - `gantt_m1_actual.png` — Actual Gantt for M=1 pipeline
  - `gantt_m3_actual.png` — Actual Gantt for M=3 pipeline
  - `gantt_m1_vs_m3_combined.png` — Side-by-side M=1 vs M=3 comparison
  - `gantt_m1_ucx_breakdown.png` — M=1 with UCX transfer breakdown
  - `gantt_m3_new_schedule_precise.png` — M=3 proposed schedule

## Usage

```bash
# Generate Gantt charts from existing timing data
python benchmark/af_bench/03_pipeline_viz/scripts/draw_gantt_actual.py

# Draw pipeline schematic
python benchmark/af_bench/03_pipeline_viz/scripts/draw_schematic.py
```
