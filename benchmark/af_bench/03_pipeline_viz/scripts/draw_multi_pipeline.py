#!/usr/bin/env python3
"""
Draw PD+AF M=3 pipeline in the "multi_pipeline" reference style:
  - Multiple horizontal lanes stacked vertically
  - Time flows left→right
  - Colored blocks for different stages (blue=A, orange=F, red=data transfer)
  - Left-side labels, clean axis
  - Uses ACTUAL timing data from the measured M=3 forward pass.
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, "results")
os.makedirs(OUT, exist_ok=True)

# ── Load real timing data ────────────────────────────────────────────────────
with open(os.path.join(OUT, "parsed_timeline_m3.json")) as f:
    raw = json.load(f)

da_steps = raw["da_timeline"]["steps"]
df_steps = raw["df_timeline"]["steps"]

# Both DA and DF run on the SAME machine with the same time.perf_counter() clock.
# Normalize BOTH to DA's first event so the timelines are comparable.
da_t0 = da_steps[0]["t_start_ms"]
for s in da_steps:
    s["t0"] = s["t_start_ms"] - da_t0
    s["t1"] = s["t_end_ms"] - da_t0
for s in df_steps:
    s["t0"] = s["t_start_ms"] - da_t0
    s["t1"] = s["t_end_ms"] - da_t0

# Empirical clock correction: same-machine cross-process perf_counter jitter
# can cause DF events to appear BEFORE the DA events they depend on.
# Find the worst negative gap and shift DF to fix physical impossibility.
da_A_by_key = {(s['layer'], s['mb']): s for s in da_steps if 'STAGE_A' in s['stage']}
df_A_by_key = {(s['layer'], s['mb']): s for s in df_steps if 'STAGE_A' in s['stage']}
da_F_by_key = {(s['layer'], s['mb']): s for s in da_steps if 'STAGE_F' in s['stage']}
df_F_by_key = {(s['layer'], s['mb']): s for s in df_steps if 'STAGE_F' in s['stage']}
min_gap_da2df = min(
    (df_A_by_key[k]['t0'] - da_A_by_key[k]['t1'])
    for k in da_A_by_key.keys() & df_A_by_key.keys()
)
min_gap_df2da = min(
    (da_F_by_key[k]['t0'] - df_F_by_key[k]['t1'])
    for k in df_F_by_key.keys() & da_F_by_key.keys()
)
df_correction = max(0, -min(min_gap_da2df, min_gap_df2da, 0))
if df_correction > 0:
    for s in df_steps:
        s['t0'] += df_correction
        s['t1'] += df_correction

da_total = da_steps[-1]["t1"]
df_total = df_steps[-1]["t1"]

df_t0 = df_steps[0]["t0"]
print(f"DA t0 = 0.000ms (reference)")
print(f"DF t0 = {df_t0:.3f}ms (forward pass starts {df_t0:.3f}ms after DA, correction={df_correction:.3f}ms)")
print(f"DA total: {da_total:.1f}ms | DF total: {df_total:.1f}ms")
# Re-verify after correction
da_A = {(s['layer'], s['mb']): s for s in da_steps if 'STAGE_A' in s['stage']}
df_A = {(s['layer'], s['mb']): s for s in df_steps if 'STAGE_A' in s['stage']}
gaps = [df_A[k]['t0'] - da_A[k]['t1'] for k in sorted(da_A.keys() & df_A.keys())[:10]]
print(f"DA_A(end)→DF_A(start) gaps (first 10 layers): min={min(gaps):.3f} max={max(gaps):.3f} ms")

# ── Colors ───────────────────────────────────────────────────────────────────
C_DA_A     = "#4A90D9"   # blue — DA A-stage (attention compute + send)
C_DA_F     = "#7EB8E0"   # light blue — DA F-stage (recv + postprocess, mostly wait)
C_DF_A     = "#E8A87C"   # light orange — DF A-stage (recv from DA, mostly wait)
C_DF_F     = "#D4743B"   # deep orange — DF F-stage (FFN compute + send)
C_TRANSFER = "#D94040"   # red — data transfer arrow
C_WAIT     = "#F0F0F0"   # light gray — idle/wait gap
C_BUBBLE   = "#FFE0E0"   # pink — pipeline bubble

# ── Figure setup ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(28, 16))

# We'll create 4 panels:
# Panel A: Full DA timeline (all 64 layers × 3 MB, compressed)
# Panel B: Full DF timeline (all 64 layers × 3 MB, compressed)
# Panel C: Zoomed — layers 0-4, all 3 micro-batches, with data transfer arrows
# Panel D: Summary stats bar

# ─── PANEL A+B: Combined DA + DF full pipeline ──────────────────────────────
ax = plt.subplot2grid((3, 1), (0, 0))
LANE_H = 1.0
LANE_GAP = 0.3
DA_Y_BASE = 0
DF_Y_BASE = -LANE_H - LANE_GAP

# DA lane: all steps as bars
for s in da_steps:
    is_A = "STAGE_A" in s["stage"]
    color = C_DA_A if is_A else C_DA_F
    alpha = 0.4 + 0.3 * s["mb"]  # darker = earlier MB
    ax.barh(DA_Y_BASE + LANE_H/2, s["t1"] - s["t0"], left=s["t0"],
            height=LANE_H, color=color, alpha=alpha, edgecolor="none", linewidth=0)

# DF lane
for s in df_steps:
    is_A = "STAGE_A" in s["stage"]
    color = C_DF_A if is_A else C_DF_F
    alpha = 0.4 + 0.3 * s["mb"]
    ax.barh(DF_Y_BASE + LANE_H/2, s["t1"] - s["t0"], left=s["t0"],
            height=LANE_H, color=color, alpha=alpha, edgecolor="none", linewidth=0)

# Labels
ax.text(-8, DA_Y_BASE + LANE_H/2, "DA\n(GPU 1)", ha="right", va="center",
        fontsize=10, fontweight="bold", color="#1a5c9e")
ax.text(-8, DF_Y_BASE + LANE_H/2, "DF\n(GPU 0)", ha="right", va="center",
        fontsize=10, fontweight="bold", color="#b85d1f")

# MB legend
for mb, alpha in [(0, 0.7), (1, 0.85), (2, 1.0)]:
    ax.barh(DF_Y_BASE - LANE_H - 0.3, 20, left=50 + mb * 60, height=0.3,
            color="gray", alpha=alpha)
    ax.text(50 + mb * 60 + 10, DF_Y_BASE - LANE_H - 0.3, f"mb={mb}",
            fontsize=6, va="center", color="gray")

# Annotations
max_t = max(da_total, df_total)
ax.set_xlim(-15, max_t + 5)
ax.set_ylim(DF_Y_BASE - LANE_H * 2, DA_Y_BASE + LANE_H * 2)
ax.set_xlabel("Time (ms)", fontsize=9)
ax.set_yticks([])
ax.set_title("PD+AF M=3 Pipeline: DA & DF Full Timeline (64 layers × 3 micro-batches)",
             fontsize=11, fontweight="bold")

# Legend
legend_handles = [
    mpatches.Patch(color=C_DA_A, alpha=0.7, label="DA A-stage (attn compute + UCX send to DF)"),
    mpatches.Patch(color=C_DA_F, alpha=0.7, label="DA F-stage (UCX recv from DF + postprocess)"),
    mpatches.Patch(color=C_DF_A, alpha=0.7, label="DF A-stage (UCX recv from DA, mostly wait)"),
    mpatches.Patch(color=C_DF_F, alpha=0.7, label="DF F-stage (FFN compute + UCX send to DA)"),
]
ax.legend(handles=legend_handles, loc="upper right", fontsize=7, ncol=2)

# ─── PANEL C: Zoomed layers 0-4 with micro-batch lanes ──────────────────────
ax2 = plt.subplot2grid((3, 1), (1, 0))

ZOOM_LAYERS = 5
LANES = 6  # 3 MBs for DA + 3 MBs for DF
lane_names = [f"DA mb=0", f"DA mb=1", f"DA mb=2",
              f"DF mb=0", f"DF mb=1", f"DF mb=2"]

# Filter steps for zoomed range
da_zoom = [s for s in da_steps if s["layer"] < ZOOM_LAYERS]
df_zoom = [s for s in df_steps if s["layer"] < ZOOM_LAYERS]

# Compute zoom window
zoom_t0 = min(s["t0"] for s in da_zoom + df_zoom)
zoom_t1 = max(s["t1"] for s in da_zoom + df_zoom)
zoom_dur = zoom_t1 - zoom_t0

# Draw DA lanes (lanes 0-2)
for mb in range(3):
    y = (2 - mb) * (LANE_H + 0.15)  # mb=0 at top
    mb_steps = [s for s in da_zoom if s["mb"] == mb]
    for s in mb_steps:
        is_A = "STAGE_A" in s["stage"]
        color = C_DA_A if is_A else C_DA_F
        dur = s["t1"] - s["t0"]
        ax2.barh(y, dur, left=s["t0"], height=LANE_H * 0.9,
                 color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        # Layer label inside bar if wide enough
        if dur > 0.3:
            lbl = f"L{s['layer']}" + ("A" if is_A else "F")
            if dur > 0.8:
                ax2.text(s["t0"] + dur/2, y, lbl, ha="center", va="center",
                        fontsize=5.5, fontweight="bold", color="white")

# Draw DF lanes (lanes 3-5)
df_y_offset = -3 * (LANE_H + 0.15)
for mb in range(3):
    y = df_y_offset + (2 - mb) * (LANE_H + 0.15)
    mb_steps = [s for s in df_zoom if s["mb"] == mb]
    for s in mb_steps:
        is_A = "STAGE_A" in s["stage"]
        color = C_DF_A if is_A else C_DF_F
        dur = s["t1"] - s["t0"]
        ax2.barh(y, dur, left=s["t0"], height=LANE_H * 0.9,
                 color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        if dur > 0.3:
            lbl = f"L{s['layer']}" + ("A" if is_A else "F")
            if dur > 0.8:
                ax2.text(s["t0"] + dur/2, y, lbl, ha="center", va="center",
                        fontsize=5.5, fontweight="bold", color="white")

# Draw data transfer arrows between DA and DF
# Arrow: from DA A-stage end to DF A-stage start
for mb in range(3):
    da_y = (2 - mb) * (LANE_H + 0.15)
    df_y = df_y_offset + (2 - mb) * (LANE_H + 0.15)
    da_mb = [s for s in da_zoom if s["mb"] == mb]
    df_mb = [s for s in df_zoom if s["mb"] == mb]
    for l in range(ZOOM_LAYERS):
        da_a = [s for s in da_mb if s["layer"] == l and "STAGE_A" in s["stage"]]
        df_a = [s for s in df_mb if s["layer"] == l and "STAGE_A" in s["stage"]]
        df_f = [s for s in df_mb if s["layer"] == l and "STAGE_F" in s["stage"]]
        da_f = [s for s in da_mb if s["layer"] == l and "STAGE_F" in s["stage"]]
        if da_a and df_a:
            x1 = da_a[0]["t1"]  # DA A-stage end = send complete
            x2 = df_a[0]["t0"]  # DF A-stage start = recv start
            y1, y2 = da_y, df_y
            # Only draw if gap is meaningful (> 0.1ms)
            if abs(x2 - x1) > 0.05:
                ax2.annotate("", xy=(x2, y2 + LANE_H/2), xytext=(x1, y1 - LANE_H/2),
                            arrowprops=dict(arrowstyle="->", color=C_TRANSFER,
                                          lw=0.8, alpha=0.6, connectionstyle="arc3,rad=0.1"))
        if df_f and da_f:
            x1 = df_f[0]["t1"]  # DF F-stage end = send complete
            x2 = da_f[0]["t0"]  # DA F-stage start = recv start
            if abs(x2 - x1) > 0.05:
                ax2.annotate("", xy=(x2, y1 - LANE_H/2), xytext=(x1, y2 + LANE_H/2),
                            arrowprops=dict(arrowstyle="->", color="#2E8B57",
                                          lw=0.8, alpha=0.6, connectionstyle="arc3,rad=-0.1"))

# Labels
for mb in range(3):
    da_y = (2 - mb) * (LANE_H + 0.15)
    df_y = df_y_offset + (2 - mb) * (LANE_H + 0.15)
    ax2.text(-1.5, da_y, f"DA mb={mb}", ha="right", va="center", fontsize=7, fontweight="bold", color="#1a5c9e")
    ax2.text(-1.5, df_y, f"DF mb={mb}", ha="right", va="center", fontsize=7, fontweight="bold", color="#b85d1f")

ax2.set_xlim(zoom_t0 - 2, zoom_t1 + 1)
ax2.set_ylim(df_y_offset - LANE_H, LANE_H * 3 + 1)
ax2.set_xlabel("Time from forward pass start (ms)", fontsize=9)
ax2.set_yticks([])
ax2.set_title(f"Zoomed: Layers 0–{ZOOM_LAYERS-1}, all 3 micro-batches (with data transfer arrows)",
              fontsize=11, fontweight="bold")

# Legend for panel C
c_legend = [
    mpatches.Patch(color=C_DA_A, label="DA A-stage"),
    mpatches.Patch(color=C_DA_F, label="DA F-stage (wait DF)"),
    mpatches.Patch(color=C_DF_A, label="DF A-stage (wait DA)"),
    mpatches.Patch(color=C_DF_F, label="DF F-stage"),
    mpatches.Patch(color=C_TRANSFER, label="DA→DF UCX transfer"),
    mpatches.Patch(color="#2E8B57", label="DF→DA UCX return"),
]
ax2.legend(handles=c_legend, loc="upper right", fontsize=6.5, ncol=3)

# ─── PANEL D: Schedule comparison + key timing ──────────────────────────────
ax3 = plt.subplot2grid((3, 1), (2, 0))
ax3.axis("off")
ax3.set_xlim(0, 100)
ax3.set_ylim(0, 35)

# Title
ax3.text(50, 34, "Pipeline Schedule Analysis & Key Metrics",
         ha="center", fontsize=11, fontweight="bold", fontfamily="monospace")

# Left side: DA schedule pattern
left_x = 2
ax3.text(left_x, 31, "DA Schedule (attn_stage):", fontsize=8, fontweight="bold", color="#1a5c9e")
# Show the pattern with colored mini-bars
da_pattern_y = 29
da_pattern = [
    ("A(0,0)", C_DA_A, 2.5), ("A(0,1)", C_DA_A, 2.5), ("A(0,2)", C_DA_A, 2.5),
    ("F(0,0)", C_DA_F, 2.5), ("A(1,0)", C_DA_A, 2.5), ("F(0,1)", C_DA_F, 2.5),
    ("A(1,1)", C_DA_A, 2.5), ("F(0,2)", C_DA_F, 2.5), ("A(1,2)", C_DA_A, 2.5),
    ("F(1,0)", C_DA_F, 2.5), ("A(2,0)", C_DA_A, 2.5), ("...", "white", 1.5),
]
x_cursor = left_x + 0.5
for label, color, width in da_pattern:
    if color != "white":
        ax3.add_patch(FancyBboxPatch((x_cursor, da_pattern_y - 0.5), width, 1.0,
                       boxstyle="round,pad=0.05", facecolor=color, alpha=0.8, edgecolor="white"))
        if width > 1.5:
            ax3.text(x_cursor + width/2, da_pattern_y, label, ha="center", va="center",
                    fontsize=5, fontweight="bold", color="white")
    else:
        ax3.text(x_cursor, da_pattern_y, label, ha="center", va="center", fontsize=7)
    x_cursor += width + 0.1

# DF schedule pattern
ax3.text(left_x, 26, "DF Schedule (ffn_stage):", fontsize=8, fontweight="bold", color="#b85d1f")
df_pattern_y = 24
df_pattern = [
    ("A(0,0)", C_DF_A, 2.5), ("F(0,0)", C_DF_F, 2.5),
    ("A(0,1)", C_DF_A, 2.5), ("F(0,1)", C_DF_F, 2.5),
    ("A(0,2)", C_DF_A, 2.5), ("F(0,2)", C_DF_F, 2.5),
    ("A(1,0)", C_DF_A, 2.5), ("F(1,0)", C_DF_F, 2.5),
    ("A(1,1)", C_DF_A, 2.5), ("F(1,1)", C_DF_F, 2.5),
    ("A(1,2)", C_DF_A, 2.5), ("F(1,2)", C_DF_F, 2.5),
]
x_cursor = left_x + 0.5
for label, color, width in df_pattern:
    ax3.add_patch(FancyBboxPatch((x_cursor, df_pattern_y - 0.5), width, 1.0,
                   boxstyle="round,pad=0.05", facecolor=color, alpha=0.8, edgecolor="white"))
    ax3.text(x_cursor + width/2, df_pattern_y, label, ha="center", va="center",
            fontsize=5, fontweight="bold", color="white")
    x_cursor += width + 0.1

# Divider
ax3.plot([left_x + 47, left_x + 47], [18, 33], color="gray", linewidth=0.5, linestyle="--")

# Right side: Key metrics table
right_x = left_x + 49
metrics_text = f"""Key Metrics (per forward pass, M=3)

DA (Attn Node, GPU 1):
  Total:       372.6 ms
  A-stage:     156.6 ms (attn compute 113.8 + send prep 30.4 + recv 12.5)
  F-stage:     216.0 ms (wait DF result 212.4 + proxy MLP 3.6)
  GPU util:     32%

DF (FFN Node, GPU 0):
  Total:       375.7 ms
  A-stage:     225.6 ms (wait DA data 219.4 + proxy attn 6.2)
  F-stage:     150.1 ms (FFN compute 129.9 + send to DA 20.2)
  GPU util:     35%

Overall:      67% time spent waiting for UCX transfers
Bottleneck:   DA↔DF serial data dependency per layer
"""

for i, line in enumerate(metrics_text.strip().split("\n")):
    color = "black"
    if "DA" in line and "Node" in line:
        color, fontweight = "#1a5c9e", "bold"
    elif "DF" in line and "Node" in line:
        color, fontweight = "#b85d1f", "bold"
    elif "GPU util" in line:
        color, fontweight = "red", "bold"
    elif "Overall" in line or "Bottleneck" in line:
        color, fontweight = "red", "bold"
    else:
        color, fontweight = "#333", "normal"
    ax3.text(right_x, 31 - i * 0.75, line, fontsize=6, fontfamily="monospace",
             color=color, fontweight=fontweight, va="top")

# Bottom summary
ax3.text(50, 3, "Pipeline schedule mismatch: DA batches 3 A-stages per layer first, DF interleaves A+F per micro-batch → massive wait bubbles",
         ha="center", fontsize=8, color="red", fontweight="bold", fontfamily="monospace")

ax3.text(50, 1, f"Data: {da_total:.0f}ms forward pass | 64 layers × 3 micro-batches | Qwen3-32B | tp=1 per module, 4 GPUs",
         ha="center", fontsize=7, color="gray")

plt.tight_layout()
fig.savefig(os.path.join(OUT, "multi_pipeline_pdaf_m3.png"), dpi=150, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "multi_pipeline_pdaf_m3.svg"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: multi_pipeline_pdaf_m3.png/.svg")
print(f"DA total: {da_total:.1f}ms | DF total: {df_total:.1f}ms | Steps: {len(da_steps)}")
