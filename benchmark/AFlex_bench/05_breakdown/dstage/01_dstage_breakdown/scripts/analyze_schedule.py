#!/usr/bin/env python3
"""Illustrate DA vs DF schedule divergence and the resulting pipeline bubbles."""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_HERE, "results")
os.makedirs(OUT_DIR, exist_ok=True)

with open(os.path.join(OUT_DIR, "parsed_timeline_m3.json")) as f:
    data = json.load(f)

da = data["da_timeline"]["steps"]
df = data["df_timeline"]["steps"]

# Normalize
da_t0 = da[0]["t_start_ms"]
df_t0 = df[0]["t_start_ms"]
for s in da: s["t0"], s["t1"] = s["t_start_ms"] - da_t0, s["t_end_ms"] - da_t0
for s in df: s["t0"], s["t1"] = s["t_start_ms"] - df_t0, s["t_end_ms"] - df_t0

# ── Figure 1: Schedule pattern — first 30 steps side by side ─────────────────
fig, ax = plt.subplots(1, 1, figsize=(20, 10))
ax2 = ax.twiny()

N_STEPS = 36
HEIGHT = 0.8
A_COLOR, F_COLOR = "#2196F3", "#FF5722"
MB_COLORS_ALPHA = {0: 0.5, 1: 0.7, 2: 0.9}

# DA steps going down (top half)
for s in da[:N_STEPS]:
    y = -s["step"] * 0.25
    is_A = "STAGE_A" in s["stage"]
    color = A_COLOR if is_A else F_COLOR

    # X position: normalize by step order + micro-batch offset
    x_start = s["t0"]
    dur = s["t1"] - s["t0"]

    ax.barh(y, dur, left=x_start, height=0.2,
            color=color, alpha=0.7, edgecolor="white", linewidth=0.3)

    label = f"{'A' if is_A else 'F'}({s['layer']},{s['mb']})"
    ax.text(x_start + dur + 0.3, y, label, fontsize=5, va="center")

# DF steps going down (bottom half)
df_y_offset = -N_STEPS * 0.25 - 1
for s in df[:N_STEPS]:
    y = df_y_offset - s["step"] * 0.25
    is_A = "STAGE_A" in s["stage"]
    color = A_COLOR if is_A else F_COLOR

    x_start = s["t0"]
    dur = s["t1"] - s["t0"]

    ax.barh(y, dur, left=x_start, height=0.2,
            color=color, alpha=0.7, edgecolor="white", linewidth=0.3)

    label = f"{'A' if is_A else 'F'}({s['layer']},{s['mb']})"
    ax.text(x_start + dur + 0.3, y, label, fontsize=5, va="center")

ax.set_xlabel("Time (ms)")
ax.set_ylim(df_y_offset - N_STEPS * 0.25 - 1, 0.5)
ax.set_yticks([])

# Annotations
ax.text(0.5, 1, "DA (Attn Node) — attn_stage schedule", transform=ax.transAxes,
        ha="center", fontsize=11, fontweight="bold", color=A_COLOR)
ax.text(0.5, 0.54, "DF (FFN Node) — ffn_stage schedule", transform=ax.transAxes,
        ha="center", fontsize=11, fontweight="bold", color=F_COLOR)
ax.axhline(y=df_y_offset + 0.2, color="gray", linewidth=0.5, linestyle="--")

legend_elements = [
    mpatches.Patch(color=A_COLOR, label="A stage (attn compute + send)"),
    mpatches.Patch(color=F_COLOR, label="F stage (recv + postprocess)"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
ax.set_title("DA vs DF Schedule Divergence (first 36 steps)", fontsize=12, fontweight="bold")

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "schedule_divergence.png"), dpi=150, bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "schedule_divergence.svg"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: schedule_divergence.png/.svg")

# ── Figure 2: Show the data flow connections — who sends what to whom ─────────
fig, axes = plt.subplots(2, 1, figsize=(22, 12), gridspec_kw={"height_ratios": [1, 1]})

# DA per-layer breakdown
da_layers_A = {}
da_layers_F = {}
for s in da:
    l = s["layer"]
    dur = s["dur_ms"]
    if "STAGE_A" in s["stage"]:
        da_layers_A[l] = da_layers_A.get(l, 0) + dur
    else:
        da_layers_F[l] = da_layers_F.get(l, 0) + dur

df_layers_A = {}
df_layers_F = {}
for s in df:
    l = s["layer"]
    dur = s["dur_ms"]
    if "STAGE_A" in s["stage"]:
        df_layers_A[l] = df_layers_A.get(l, 0) + dur
    else:
        df_layers_F[l] = df_layers_F.get(l, 0) + dur

layers_sorted = sorted(set(list(da_layers_A.keys()) + list(df_layers_A.keys())))
x = np.arange(len(layers_sorted))
w = 0.35

ax0 = axes[0]
a_bars = ax0.bar(x - w/2, [da_layers_A.get(l, 0) for l in layers_sorted], w,
                 color=A_COLOR, alpha=0.7, label="A stage total (3 MBs)", edgecolor="white")
f_bars = ax0.bar(x - w/2, [da_layers_F.get(l, 0) for l in layers_sorted], w,
                 bottom=[da_layers_A.get(l, 0) for l in layers_sorted],
                 color=F_COLOR, alpha=0.7, label="F stage total (3 MBs)", edgecolor="white")
ax0.set_ylabel("Time (ms)")
ax0.set_title("DA (Attn Node): Per-layer A+F total duration across 3 micro-batches", fontsize=11)
ax0.set_xticks(x[::4])
ax0.set_xticklabels([f"L{l}" for l in layers_sorted[::4]], rotation=45)
ax0.legend(fontsize=8)
ax0.set_ylim(0, 7)
# Annotate averages
avg_a = sum(da_layers_A.values()) / len(da_layers_A)
avg_f = sum(da_layers_F.values()) / len(da_layers_F)
ax0.axhline(y=avg_a, color=A_COLOR, linewidth=0.8, linestyle="--", alpha=0.5)
ax0.axhline(y=avg_a + avg_f, color=F_COLOR, linewidth=0.8, linestyle="--", alpha=0.5)
ax0.text(len(layers_sorted) - 2, avg_a / 2, f"A avg={avg_a:.2f}ms", fontsize=7, color=A_COLOR)
ax0.text(len(layers_sorted) - 2, avg_a + avg_f / 2, f"F avg={avg_f:.2f}ms", fontsize=7, color=F_COLOR)

ax1 = axes[1]
ax1.bar(x - w/2, [df_layers_A.get(l, 0) for l in layers_sorted], w,
        color=A_COLOR, alpha=0.7, label="A stage total (3 MBs)", edgecolor="white")
ax1.bar(x - w/2, [df_layers_F.get(l, 0) for l in layers_sorted], w,
        bottom=[df_layers_A.get(l, 0) for l in layers_sorted],
        color=F_COLOR, alpha=0.7, label="F stage total (3 MBs)", edgecolor="white")
ax1.set_ylabel("Time (ms)")
ax1.set_xlabel("Layer")
ax1.set_title("DF (FFN Node): Per-layer A+F total duration across 3 micro-batches", fontsize=11)
ax1.set_xticks(x[::4])
ax1.set_xticklabels([f"L{l}" for l in layers_sorted[::4]], rotation=45)
ax1.legend(fontsize=8)
ax1.set_ylim(0, 7)
avg_a = sum(df_layers_A.values()) / len(df_layers_A)
avg_f = sum(df_layers_F.values()) / len(df_layers_F)
ax1.axhline(y=avg_a, color=A_COLOR, linewidth=0.8, linestyle="--", alpha=0.5)
ax1.axhline(y=avg_a + avg_f, color=F_COLOR, linewidth=0.8, linestyle="--", alpha=0.5)
ax1.text(len(layers_sorted) - 2, avg_a / 2, f"A avg={avg_a:.2f}ms", fontsize=7, color=A_COLOR)
ax1.text(len(layers_sorted) - 2, avg_a + avg_f / 2, f"F avg={avg_f:.2f}ms", fontsize=7, color=F_COLOR)

fig.suptitle("Per-Layer Stage Duration (summed across 3 micro-batches) — Qwen3-32B M=3", fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "per_layer_breakdown.png"), dpi=150, bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "per_layer_breakdown.svg"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: per_layer_breakdown.png/.svg")

# ── Figure 3: The "why" diagram — show the direct pipeline dependency ────────
fig, ax = plt.subplots(figsize=(22, 8))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

title = "PD+AF M=3 Pipeline: Why Data Transfer IS Batched but Schedule Mismatch Creates Bubbles"
ax.text(50, 97, title, ha="center", fontsize=13, fontweight="bold", fontfamily="monospace")
ax.text(50, 93, "DA processes A(l,0) A(l,1) A(l,2) first → sends 3 batches to DF  |  DF processes A(l,0) F(l,0) A(l,1) F(l,1) A(l,2) F(l,2)",
        ha="center", fontsize=9, fontfamily="monospace", color="gray")

# DA row
DA_Y = 75
DF_Y = 45

# Draw DA timeline for layer 0
da_boxes = [
    ("A(0,0)", 0, 0.8, "#2196F3"),
    ("A(0,1)", 0.9, 0.8, "#2196F3"),
    ("A(0,2)", 1.8, 0.8, "#2196F3"),
    ("F(0,0)", 2.7, 1.3, "#FF5722"),
    ("A(1,0)", 4.1, 0.8, "#2196F3"),
    ("F(0,1)", 5.0, 1.3, "#FF5722"),
    ("A(1,1)", 6.4, 0.8, "#2196F3"),
    ("F(0,2)", 7.3, 1.3, "#FF5722"),
    ("A(1,2)", 8.7, 0.8, "#2196F3"),
    ("F(1,0)", 9.6, 1.3, "#FF5722"),
]
for label, x_start, width, color in da_boxes:
    x = 5 + x_start * 6
    w = width * 6
    ax.add_patch(plt.Rectangle((x, DA_Y - 5), w, 10, facecolor=color, alpha=0.7, edgecolor="white"))
    ax.text(x + w/2, DA_Y, label, ha="center", va="center", fontsize=7, fontweight="bold", color="white")

ax.text(3, DA_Y + 7, "DA:", fontsize=10, fontweight="bold", color="#2196F3")
ax.text(3, DA_Y - 7, "Sends 3 A-stages\nfor same layer\nbefore any F", fontsize=7, color="gray")

# Draw DF timeline for layer 0
df_boxes = [
    ("A(0,0)", 0, 1.0, "#2196F3"),
    ("F(0,0)", 1.1, 0.8, "#FF5722"),
    ("A(0,1)", 2.0, 1.0, "#2196F3"),
    ("F(0,1)", 3.1, 0.8, "#FF5722"),
    ("A(0,2)", 4.0, 1.0, "#2196F3"),
    ("F(0,2)", 5.1, 0.8, "#FF5722"),
    ("A(1,0)", 6.0, 1.0, "#2196F3"),
    ("F(1,0)", 7.1, 0.8, "#FF5722"),
    ("A(1,1)", 8.0, 1.0, "#2196F3"),
    ("F(1,1)", 9.1, 0.8, "#FF5722"),
]
for label, x_start, width, color in df_boxes:
    x = 5 + x_start * 6
    w = width * 6
    ax.add_patch(plt.Rectangle((x, DF_Y - 5), w, 10, facecolor=color, alpha=0.7, edgecolor="white"))
    ax.text(x + w/2, DF_Y, label, ha="center", va="center", fontsize=7, fontweight="bold", color="white")

ax.text(3, DF_Y + 7, "DF:", fontsize=10, fontweight="bold", color="#FF5722")
ax.text(3, DF_Y - 7, "Expects A→F→A→F\nfor each layer\nand micro-batch", fontsize=7, color="gray")

# Arrows showing data flow
arrow_y_start = DA_Y - 8
arrow_y_end = DF_Y + 8
for i, x_start in enumerate([0, 0.9, 1.8]):
    x_mid = 5 + (x_start + 0.4) * 6
    ax.annotate("", xy=(x_mid, arrow_y_end), xytext=(x_mid, arrow_y_start),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5,
                              connectionstyle="arc3,rad=0.1"))
    ax.text(x_mid + 1, (DA_Y + DF_Y) / 2, f"UCX send\nmb={i}", fontsize=5, color="red", va="center")

# Return arrows
for i, x_start in enumerate([2.7, 5.0, 7.3]):
    x_mid = 5 + (x_start + 0.65) * 6
    ax.annotate("", xy=(x_mid, arrow_y_start), xytext=(x_mid, arrow_y_end),
                arrowprops=dict(arrowstyle="->", color="green", lw=1.5,
                              connectionstyle="arc3,rad=-0.1"))
    ax.text(x_mid + 1, (DA_Y + DF_Y) / 2, f"UCX return\nmb={i}", fontsize=5, color="green", va="center")

# Summary box
summary_box = plt.Rectangle((5, 5), 90, 30, facecolor="#f5f5f5", edgecolor="#ccc", linewidth=1)
ax.add_patch(summary_box)
summary_text = """
Key Insight: M=3 data transfers ARE batched — DA sends 3 micro-batches of layer output together to DF via ring buffer (RING_SIZE=3).
But DF's schedule (ffn_stage) interleaves A and F per micro-batch: A(0,0) F(0,0) A(0,1) F(0,1) A(0,2) F(0,2).
DA's schedule (attn_stage) batches A-stages first: A(0,0) A(0,1) A(0,2) F(0,0) F(0,1) F(0,2).

Result: When DA begins F(0,0) (waiting for DF's result), DF is still processing F(0,1) or later — creating a ~212ms bubble on DA.
When DF begins A(0,1) (waiting for DA's result), DA has already sent it but DF can't process it yet — creating a ~219ms bubble on DF.

The M=3 pipeline creates 3× more data in flight but doesn't overlap DA and DF compute — the schedule mismatch prevents true pipelining.
"""
ax.text(8, 33, summary_text, fontsize=7.5, fontfamily="monospace", va="top", linespacing=1.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "pipeline_bubble_explanation.png"), dpi=150, bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "pipeline_bubble_explanation.svg"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: pipeline_bubble_explanation.png/.svg")
