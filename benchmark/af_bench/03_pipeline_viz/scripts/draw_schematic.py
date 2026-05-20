#!/usr/bin/env python3
"""Create high-level M=3 pipeline schematic showing the abstract flow pattern."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_HERE, "results")

fig, ax = plt.subplots(1, 1, figsize=(24, 14))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# Title
ax.text(50, 98, "PD+AF M=3 Pipeline Schematic — Qwen3-32B Decode (tp=4×1, 4 GPUs)",
        ha="center", fontsize=14, fontweight="bold", fontfamily="monospace")
ax.text(50, 94, "Data Flow: DA (Attn) ↔ DF (FFN) via UCX RDMA | 64 Layers × 3 Micro-batches",
        ha="center", fontsize=10, fontfamily="monospace", color="gray")

# ── Layout parameters ────────────────────────────────────────────────────────
DA_X, DF_X = 20, 60  # X positions for DA and DF columns
TOP_Y = 88
BAR_HEIGHT = 0.6
LAYER_GAP = 1.2
MB_GAP = 0.3

# ── Legend box ────────────────────────────────────────────────────────────────
legend_box = FancyBboxPatch((2, 78), 35, 12, boxstyle="round,pad=0.3",
                            facecolor="#f5f5f5", edgecolor="#ccc", linewidth=1, zorder=0)
ax.add_patch(legend_box)
ax.text(4, 88, "Legend", fontsize=9, fontweight="bold")

colors_legend = [
    ("#2196F3", "A Stage (attn compute + send to DF)"),
    ("#FF5722", "F Stage (recv from DF + postprocess)"),
    ("#4CAF50", "Real GPU Compute"),
    ("#FF9800", "Wait (UCX transfer + bubble)"),
    ("#9E9E9E", "Proxy/No-op"),
]
for i, (color, label) in enumerate(colors_legend):
    ax.add_patch(plt.Rectangle((5, 86 - i * 2.5), 3, 1.5, facecolor=color, edgecolor="white"))
    ax.text(9.5, 86.2 - i * 2.5, label, fontsize=8, va="center")

# ── DA Side (Attn Node) ──────────────────────────────────────────────────────
ax.text(DA_X - 5, TOP_Y + 2, "DA (GPU 1)\nAttn Node", ha="center", fontsize=10, fontweight="bold", color="#2196F3")

# ── DF Side (FFN Node) ───────────────────────────────────────────────────────
ax.text(DF_X + 5, TOP_Y + 2, "DF (GPU 0)\nFFN Node", ha="center", fontsize=10, fontweight="bold", color="#FF5722")

# ── UCX Transfer arrows ──────────────────────────────────────────────────────
arrow_y = TOP_Y - 10
ax.annotate("", xy=(DF_X - 2, arrow_y), xytext=(DA_X + 2, arrow_y),
            arrowprops=dict(arrowstyle="<->", color="red", lw=2, connectionstyle="arc3,rad=0"))
ax.text((DA_X + DF_X) / 2, arrow_y + 1, "UCX RDMA\nDA↔DF", ha="center", fontsize=7, color="red", fontweight="bold")

# ── Draw pipeline stages for a representative subset (layers 0-9, all 3 micro-batches) ──

# Timings from actual data (ms), normalized per-layer, per-stage
# DA: A_stage ≈ 2.4ms/layer/stage (156.6ms / 64 layers / 3 MB ≈ 0.8ms per stage)
#     F_stage ≈ 3.4ms/layer/stage (216.0ms / 64 layers / 3 MB ≈ 1.1ms per stage)
# DF: A_stage ≈ 3.5ms/layer/stage (225.6ms / 64 layers / 3 MB ≈ 1.2ms per stage)
#     F_stage ≈ 2.3ms/layer/stage (150.1ms / 64 layers / 3 MB ≈ 0.8ms per stage)
#
# But these are very small. For the schematic, I'll use normalized times.
# Let me show the pattern with representative widths.

# ── Micro-batch 0 ──
mb0_y = TOP_Y
# Layer 0
for l in range(5):
    y = mb0_y - l * (BAR_HEIGHT * 2 * 3 + MB_GAP * 2)

    # DA: A(0,0), F(0,0)
    da_a_x = 0 + (l * 4.5 % 30) if l > 0 else 0  # for visual clarity, shift slightly
    da_a_x = l * 0.8
    da_a_w = 0.8
    da_f_x = da_a_x + da_a_w + 0.01
    da_f_w = 1.2

    ax.add_patch(FancyBboxPatch((DA_X - 4, y), da_a_w * 3, BAR_HEIGHT,
                                 boxstyle="round,pad=0.05", facecolor="#2196F3", alpha=0.7, edgecolor="white"))
    ax.add_patch(FancyBboxPatch((DA_X - 4 + da_a_w * 3, y), da_f_w * 3, BAR_HEIGHT,
                                 boxstyle="round,pad=0.05", facecolor="#FF5722", alpha=0.7, edgecolor="white"))
    ax.text(DA_X - 4 - 0.3, y + BAR_HEIGHT / 2, f"L{l}", fontsize=5, ha="right", va="center")

    # DF: A(0,0), F(0,0)
    ax.add_patch(FancyBboxPatch((DF_X - 4, y), da_a_w * 3, BAR_HEIGHT,
                                 boxstyle="round,pad=0.05", facecolor="#2196F3", alpha=0.7, edgecolor="white"))
    ax.add_patch(FancyBboxPatch((DF_X - 4 + da_a_w * 3, y), da_f_w * 3, BAR_HEIGHT,
                                 boxstyle="round,pad=0.05", facecolor="#FF5722", alpha=0.7, edgecolor="white"))

# ── Micro-batch 1 ──
mb1_y = mb0_y - 7
ax.text(DA_X - 5, mb1_y + 0.5, "mb=1", fontsize=7, color="gray")
for l in range(5):
    y = mb1_y - l * (BAR_HEIGHT * 2 * 3 + MB_GAP * 2)
    da_a_x = l * 0.8
    da_a_w = 0.8
    ax.add_patch(FancyBboxPatch((DA_X - 4, y), da_a_w * 3, BAR_HEIGHT,
                                 boxstyle="round,pad=0.05", facecolor="#2196F3", alpha=0.5, edgecolor="white"))
    ax.add_patch(FancyBboxPatch((DA_X - 4 + da_a_w * 3, y), 1.2 * 3, BAR_HEIGHT,
                                 boxstyle="round,pad=0.05", facecolor="#FF5722", alpha=0.5, edgecolor="white"))
    ax.text(DA_X - 4 - 0.3, y + BAR_HEIGHT / 2, f"L{l}", fontsize=5, ha="right", va="center")
    ax.add_patch(FancyBboxPatch((DF_X - 4, y), da_a_w * 3, BAR_HEIGHT,
                                 boxstyle="round,pad=0.05", facecolor="#2196F3", alpha=0.5, edgecolor="white"))
    ax.add_patch(FancyBboxPatch((DF_X - 4 + da_a_w * 3, y), 1.2 * 3, BAR_HEIGHT,
                                 boxstyle="round,pad=0.05", facecolor="#FF5722", alpha=0.5, edgecolor="white"))

# ── Micro-batch 2 ──
mb2_y = mb1_y - 7
ax.text(DA_X - 5, mb2_y + 0.5, "mb=2", fontsize=7, color="gray")
for l in range(5):
    y = mb2_y - l * (BAR_HEIGHT * 2 * 3 + MB_GAP * 2)
    da_a_x = l * 0.8
    da_a_w = 0.8
    ax.add_patch(FancyBboxPatch((DA_X - 4, y), da_a_w * 3, BAR_HEIGHT,
                                 boxstyle="round,pad=0.05", facecolor="#2196F3", alpha=0.3, edgecolor="white"))
    ax.add_patch(FancyBboxPatch((DA_X - 4 + da_a_w * 3, y), 1.2 * 3, BAR_HEIGHT,
                                 boxstyle="round,pad=0.05", facecolor="#FF5722", alpha=0.3, edgecolor="white"))
    ax.text(DA_X - 4 - 0.3, y + BAR_HEIGHT / 2, f"L{l}", fontsize=5, ha="right", va="center")
    ax.add_patch(FancyBboxPatch((DF_X - 4, y), da_a_w * 3, BAR_HEIGHT,
                                 boxstyle="round,pad=0.05", facecolor="#2196F3", alpha=0.3, edgecolor="white"))
    ax.add_patch(FancyBboxPatch((DF_X - 4 + da_a_w * 3, y), 1.2 * 3, BAR_HEIGHT,
                                 boxstyle="round,pad=0.05", facecolor="#FF5722", alpha=0.3, edgecolor="white"))

# ── Pipeline schedule notation ───────────────────────────────────────────────
schedule_box = FancyBboxPatch((2, 2), 96, 30, boxstyle="round,pad=0.3",
                              facecolor="#fafafa", edgecolor="#ddd", linewidth=1, zorder=0)
ax.add_patch(schedule_box)

schedule_text = """
Pipeline Schedule (from AFDStageScheduleGenerator):

  attn_stage (DA):  A(0,0) F(0,0) A(1,0) F(1,0) ... A(63,0) F(63,0)  ── mb=0 ──
                    A(0,1) F(0,1) A(1,1) F(1,1) ... A(63,1) F(63,1)  ── mb=1 ──
                    A(0,2) F(0,2) A(1,2) F(1,2) ... A(63,2) F(63,2)  ── mb=2 ──

  ffn_stage  (DF):  A(0,0) F(0,0) A(1,0) F(1,0) ... A(63,0) F(63,0)  ── mb=0 ──
                    A(0,1) F(0,1) A(1,1) F(1,1) ... A(63,1) F(63,1)  ── mb=1 ──
                    A(0,2) F(0,2) A(1,2) F(1,2) ... A(63,2) F(63,2)  ── mb=2 ──

  Each A stage on DA:  prep_attn (recv + norm) → attn (compute) → prep_mlp (send to DF)
  Each F stage on DA:  proxy_mlp → postprocess (recv from DF + norm)   ← WAITS for DF
  Each A stage on DF:  prep_attn (recv from DA) → proxy_attn → prep_mlp  ← WAITS for DA
  Each F stage on DF:  mlp (compute) → postprocess (send to DA)

  Total steps: 64 layers × 3 micro-batches × 2 stages = 384 steps per forward pass
  Total time:  ~375ms per forward pass
  GPU utilization:  DA=32%  DF=35%  Overall=33%
  Bottleneck:  UCX RDMA transfers (DA↔DF) account for ~65-68% of forward pass time
"""
ax.text(4, 30, schedule_text, fontsize=7, fontfamily="monospace", va="top", linespacing=1.1)

# ── Key timing annotations ───────────────────────────────────────────────────
timing_box = FancyBboxPatch((72, 48), 26, 18, boxstyle="round,pad=0.3",
                            facecolor="#fff3e0", edgecolor="#FF9800", linewidth=1.5)
ax.add_patch(timing_box)
timing_text = """KEY METRICS (per forward pass)

DA Attn Node:
  A stage:  156.6ms (42%)
    - attn compute: 113.8ms
    - send prep:     30.4ms
    - recv+norm:     12.5ms
  F stage:  216.0ms (58%)
    - wait DF result: 212.4ms
    - proxy MLP:        3.6ms

DF FFN Node:
  A stage:  225.6ms (60%)
    - wait DA data: 219.4ms
    - proxy attn:     6.2ms
  F stage:  150.1ms (40%)
    - FFN compute: 129.9ms
    - send to DA:    20.2ms

Total: DA=372.6ms  DF=375.7ms
Compute efficiency: 33%
"""
ax.text(73, 64, timing_text, fontsize=6.5, fontfamily="monospace", va="top", linespacing=1.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "pipeline_schematic.png"), dpi=150, bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "pipeline_schematic.svg"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: pipeline_schematic.png/.svg")
