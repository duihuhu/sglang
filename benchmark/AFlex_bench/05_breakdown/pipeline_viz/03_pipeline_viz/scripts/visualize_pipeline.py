#!/usr/bin/env python3
"""Create detailed M=3 PD+AF pipeline visualization from parsed timeline data.

Generates a multi-panel figure:
  1. Full pipeline Gantt chart (DA + DF, all 64 layers, M=3 micro-batches)
  2. Zoomed view of first 10 layers showing the A/F interleaving pattern
  3. Aggregate compute vs wait breakdown per stage
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_HERE, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load parsed data ─────────────────────────────────────────────────────────

with open(os.path.join(OUT_DIR, "parsed_timeline_m3.json")) as f:
    data = json.load(f)

with open(os.path.join(OUT_DIR, "breakdown_stats_m3.json")) as f:
    stats = json.load(f)

da_steps = data["da_timeline"]["steps"]
df_steps = data["df_timeline"]["steps"]

# ── Normalize timelines to start at t=0 ──────────────────────────────────────

da_t0 = da_steps[0]["t_start_ms"]
df_t0 = df_steps[0]["t_start_ms"]

for s in da_steps:
    s["t0"] = s["t_start_ms"] - da_t0
    s["t1"] = s["t_end_ms"] - da_t0
for s in df_steps:
    s["t0"] = s["t_start_ms"] - df_t0
    s["t1"] = s["t_end_ms"] - df_t0

da_total = da_steps[-1]["t1"]
df_total = df_steps[-1]["t1"]
max_total = max(da_total, df_total)

# ── Extract per-micro-batch stats ────────────────────────────────────────────

def compute_mb_stats(steps, perspective):
    """Compute per-micro-batch A and F stage durations."""
    mbs = {}
    for s in steps:
        mb = s["mb"]
        if mb not in mbs:
            mbs[mb] = {"A_total": 0, "F_total": 0, "A_count": 0, "F_count": 0,
                       "layers_A": [], "layers_F": []}
        dur = s["dur_ms"]
        if s["stage"] == "AFD_FORWARD_STAGE_A":
            mbs[mb]["A_total"] += dur
            mbs[mb]["A_count"] += 1
            mbs[mb]["layers_A"].append(dur)
        else:
            mbs[mb]["F_total"] += dur
            mbs[mb]["F_count"] += 1
            mbs[mb]["layers_F"].append(dur)
    return mbs

da_mb = compute_mb_stats(da_steps, "attn")
df_mb = compute_mb_stats(df_steps, "ffn")

# ── Figure 1: Full Pipeline Gantt Chart ──────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(22, 14), gridspec_kw={"height_ratios": [1, 1]})
fig.suptitle(
    "PD+AF M=3 Decode Pipeline — Qwen3-32B (64 layers, tp=4×1)\n"
    f"DA Attn Node vs DF FFN Node | Total forward pass: DA={da_total:.1f}ms, DF={df_total:.1f}ms",
    fontsize=13, fontweight="bold",
)

# Color scheme
A_COLOR = "#2196F3"       # blue — A stage (attention compute)
F_COLOR = "#FF5722"       # deep orange — F stage (FFN compute)
MB_COLORS = ["#1a1a2e", "#16213e", "#0f3460"]  # micro-batch shades

for ax_idx, (steps, title, total_ms) in enumerate([
    (da_steps, "DA (Decode Attn Node, GPU 1)", da_total),
    (df_steps, "DF (Decode FFN Node, GPU 0)", df_total),
]):
    ax = axes[ax_idx]
    height = 0.8

    # Plot each step as a horizontal bar
    for s in steps:
        layer = s["layer"]
        mb = s["mb"]
        is_A = s["stage"] == "AFD_FORWARD_STAGE_A"

        y_pos = layer * 3 + mb  # stack 3 micro-batches per layer
        color = A_COLOR if is_A else F_COLOR
        alpha = 0.5 + 0.25 * mb  # lighter for earlier micro-batches

        ax.barh(y_pos, s["t1"] - s["t0"], left=s["t0"], height=height,
                color=color, alpha=alpha, edgecolor="none", linewidth=0)

    ax.set_ylabel("Layer × Micro-batch (y = layer*3 + mb)")
    ax.set_xlabel("Time (ms)")
    ax.set_title(f"{title} — total={total_ms:.1f}ms")
    ax.set_xlim(0, max_total * 1.02)
    ax.set_ylim(-1, 64 * 3 + 1)
    ax.invert_yaxis()

    # Legend
    legend_elements = [
        mpatches.Patch(color=A_COLOR, alpha=0.75, label="A stage (attn compute + prep_mlp send)"),
        mpatches.Patch(color=F_COLOR, alpha=0.75, label="F stage (recv from peer + postprocess)"),
        mpatches.Patch(color="gray", alpha=0.3, label="mb=0 (darker)"),
        mpatches.Patch(color="gray", alpha=0.6, label="mb=1"),
        mpatches.Patch(color="gray", alpha=0.9, label="mb=2 (lighter)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7, ncol=5)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "pipeline_full_gantt.png"), dpi=150, bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "pipeline_full_gantt.svg"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: pipeline_full_gantt.png/.svg")

# ── Figure 2: Zoomed view — first 10 layers × 3 micro-batches ────────────────

fig, axes = plt.subplots(1, 2, figsize=(22, 10))
fig.suptitle(
    "PD+AF M=3 Pipeline — Zoomed: Layers 0–9 × 3 micro-batches\n"
    "Left: DA (Attn Node) | Right: DF (FFN Node)",
    fontsize=12, fontweight="bold",
)

ZOOM_LAYERS = 10

for ax_idx, (steps, title, total_ms) in enumerate([
    (da_steps, "DA (Decode Attn Node)", da_total),
    (df_steps, "DF (Decode FFN Node)", df_total),
]):
    ax = axes[ax_idx]
    zoom_steps = [s for s in steps if s["layer"] < ZOOM_LAYERS]
    zoom_t0 = min(s["t0"] for s in zoom_steps)
    zoom_t1 = max(s["t1"] for s in zoom_steps)

    for s in zoom_steps:
        layer = s["layer"]
        mb = s["mb"]
        is_A = s["stage"] == "AFD_FORWARD_STAGE_A"
        color = A_COLOR if is_A else F_COLOR

        y_pos = layer * 3 + mb
        ax.barh(y_pos, s["t1"] - s["t0"], left=s["t0"], height=0.8,
                color=color, alpha=0.7, edgecolor="white", linewidth=0.3)

        # Annotate with duration if > 1ms
        dur = s["dur_ms"]
        if dur > 1.0:
            ax.text(s["t0"] + dur / 2, y_pos,
                    f"{dur:.0f}ms", ha="center", va="center",
                    fontsize=5, fontweight="bold", color="white")

    # Layer labels
    ax.set_yticks([l * 3 + 1 for l in range(ZOOM_LAYERS)])
    ax.set_yticklabels([f"L{l}" for l in range(ZOOM_LAYERS)])
    ax.set_xlabel("Time (ms)")
    ax.set_title(f"{title}")
    ax.invert_yaxis()

    # Add micro-batch annotation on the right
    for mb in range(3):
        ax.annotate(f"mb={mb}", xy=(1.01, 0.95 - mb * 0.05), xycoords="axes fraction",
                    fontsize=8, fontstyle="italic")

    # Legend
    legend_elements = [
        mpatches.Patch(color=A_COLOR, label="A (attn+send)"),
        mpatches.Patch(color=F_COLOR, label="F (recv+postprocess)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "pipeline_zoomed_l0_l9.png"), dpi=150, bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "pipeline_zoomed_l0_l9.svg"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: pipeline_zoomed_l0_l9.png/.svg")

# ── Figure 3: Aggregate breakdown — compute vs wait per stage ─────────────────

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("PD+AF M=3 — Aggregate Stage Breakdown (64 layers × 3 micro-batches)", fontsize=12, fontweight="bold")

# From parsed AFD_BREAKDOWN
da_breakdown_raw = data.get("da_breakdown", "")
df_breakdown_raw = data.get("df_breakdown", "")

# Parse detailed sub-stage timings
def parse_breakdown(raw):
    import re
    result = {}
    # Order matters: match longer keys before shorter ones to avoid substring matches
    for key in ["total", "A_stage", "F_stage", "prep_attn", "prep_mlp", "postprocess", "attn", "mlp"]:
        # Use word boundary or space to avoid matching sub-keys
        m = re.search(rf"\b{key}=([\d.]+)ms", raw)
        if m:
            result[key] = float(m.group(1))
    return result

da_bd = parse_breakdown(da_breakdown_raw)
df_bd = parse_breakdown(df_breakdown_raw)

def plot_breakdown(ax, bd, title, node_color):
    """Plot stacked bar showing compute vs wait breakdown."""
    if not bd:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return

    categories = ["A_stage", "F_stage"]
    x = np.arange(len(categories))
    width = 0.6

    # For DA (Attn node):
    #   A_stage: prep_attn (recv+norm) ≈ 12.5ms, attn (real compute) ≈ 113.8ms, prep_mlp (send) ≈ 30.4ms
    #   F_stage: mlp (proxy, cheap) ≈ 3.6ms, postprocess (wait for DF+norm) ≈ 212.4ms
    #
    # For DF (FFN node):
    #   A_stage: prep_attn (wait for DA+norm) ≈ 2.5 + 3.7 + 219.4ms → mostly prep_mlp wait
    #   F_stage: mlp (real compute) ≈ 129.9ms, postprocess (send) ≈ 20.2ms

    if "attn" in title.lower():
        # DA: A_stage breakdown
        a_compute = bd.get("attn", 0)       # actual attention
        a_send = bd.get("prep_mlp", 0)      # preparation + send to DF
        a_recv = bd.get("prep_attn", 0)     # receive + norm
        # F_stage breakdown
        f_wait = bd.get("postprocess", 0)   # waiting for DF result
        f_proxy = bd.get("mlp", 0)          # proxy MLP (cheap)

        bars_a = [a_recv, a_compute, a_send]
        bars_f = [f_wait, f_proxy]
        colors_a = ["#FFEB3B", "#4CAF50", "#2196F3"]
        colors_f = ["#FF9800", "#9E9E9E"]
        labels_a = [f"recv+norm ({a_recv:.0f}ms)", f"attn compute ({a_compute:.0f}ms)", f"prep_mlp+send ({a_send:.0f}ms)"]
        labels_f = [f"wait DF result ({f_wait:.0f}ms)", f"proxy MLP ({f_proxy:.0f}ms)"]
    else:
        # DF: A_stage breakdown
        a_wait = bd.get("prep_mlp", 0)      # waiting for DA data
        a_other = (bd.get("prep_attn", 0) + bd.get("attn", 0))  # cheap proxy
        # F_stage breakdown
        f_compute = bd.get("mlp", 0)         # real FFN compute
        f_send = bd.get("postprocess", 0)    # postprocess + send to DA

        bars_a = [a_wait, a_other]
        bars_f = [f_compute, f_send]
        colors_a = ["#FF9800", "#9E9E9E"]
        colors_f = ["#4CAF50", "#2196F3"]
        labels_a = [f"wait DA data ({a_wait:.0f}ms)", f"proxy A ({a_other:.0f}ms)"]
        labels_f = [f"FFN compute ({f_compute:.0f}ms)", f"postprocess+send ({f_send:.0f}ms)"]

    # Plot A stage
    bottom = 0
    for i, (val, color, label) in enumerate(zip(bars_a, colors_a, labels_a)):
        ax.bar(0, val, width, bottom=bottom, color=color, label=label, edgecolor="white")
        if val > 3:
            ax.text(0, bottom + val / 2, label, ha="center", va="center", fontsize=9, fontweight="bold")
        bottom += val

    # Plot F stage
    bottom = 0
    for i, (val, color, label) in enumerate(zip(bars_f, colors_f, labels_f)):
        ax.bar(1, val, width, bottom=bottom, color=color, label=label, edgecolor="white")
        if val > 3:
            ax.text(1, bottom + val / 2, label, ha="center", va="center", fontsize=9, fontweight="bold")
        bottom += val

    # Total annotations
    ax.text(0, bd.get("A_stage", 0) + 3, f"A={bd.get('A_stage', 0):.0f}ms", ha="center", fontsize=10, fontweight="bold")
    ax.text(1, bd.get("F_stage", 0) + 3, f"F={bd.get('F_stage', 0):.0f}ms", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["A Stage\n(per-layer)", "F Stage\n(per-layer)"])
    ax.set_ylabel("Time (ms)")
    ax.set_title(title)
    ax.set_ylim(0, bd.get("total", 400) * 1.15)
    ax.legend(loc="upper right", fontsize=7, ncol=1)

plot_breakdown(axes[0], da_bd, f"DA Attn Node\nTotal: {da_bd.get('total', 0):.0f}ms (nA={192}, nF={192})", "#2196F3")
plot_breakdown(axes[1], df_bd, f"DF FFN Node\nTotal: {df_bd.get('total', 0):.0f}ms (nA={192}, nF={192})", "#FF5722")

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "pipeline_breakdown_bars.png"), dpi=150, bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "pipeline_breakdown_bars.svg"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: pipeline_breakdown_bars.png/.svg")

# ── Figure 4: Compute vs Wait summary ────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 7))

# Compute split on DA and DF
da_total_ms = da_bd.get("total", 0)
df_total_ms = df_bd.get("total", 0)

# DA: real compute = attn (113.8) + proxy mlp (3.6) = 117.4ms, wait = the rest
da_compute = da_bd.get("attn", 0) + da_bd.get("mlp", 0)
da_wait = da_total_ms - da_compute

# DF: real compute = mlp (129.9), wait = the rest
df_compute = df_bd.get("mlp", 0)
df_wait = df_total_ms - df_compute

categories = ["DA (Attn Node)", "DF (FFN Node)"]
compute_vals = [da_compute, df_compute]
wait_vals = [da_wait, df_wait]

x = np.arange(len(categories))
width = 0.5

bars_compute = ax.bar(x, compute_vals, width, color="#4CAF50", label="Real GPU Compute", edgecolor="white")
bars_wait = ax.bar(x, wait_vals, width, bottom=compute_vals, color="#FF9800", label="Wait (UCX transfer + pipeline bubble)", edgecolor="white")

# Annotate
for i, (comp, wait) in enumerate(zip(compute_vals, wait_vals)):
    total = comp + wait
    ax.text(i, comp / 2, f"{comp:.0f}ms\n({comp/total*100:.0f}%)", ha="center", va="center", fontsize=12, fontweight="bold", color="white")
    ax.text(i, comp + wait / 2, f"{wait:.0f}ms\n({wait/total*100:.0f}%)", ha="center", va="center", fontsize=12, fontweight="bold", color="white")
    ax.text(i, total + 5, f"Total: {total:.0f}ms", ha="center", fontsize=11, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylabel("Time per forward pass (ms)")
ax.set_title("PD+AF M=3: Compute vs Wait per Forward Pass\n(64 layers × 3 micro-batches, Qwen3-32B)", fontsize=12, fontweight="bold")
ax.legend(loc="upper right", fontsize=10)
ax.set_ylim(0, max(da_total_ms, df_total_ms) * 1.2)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "pipeline_compute_vs_wait.png"), dpi=150, bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "pipeline_compute_vs_wait.svg"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: pipeline_compute_vs_wait.png/.svg")

# ── Figure 5: Per-layer latency heatmap ──────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(22, 10))
fig.suptitle("PD+AF M=3 — Per-Layer A & F Stage Duration Heatmap", fontsize=12, fontweight="bold")

for ax_idx, (steps, title) in enumerate([
    (da_steps, "DA (Attn Node)"),
    (df_steps, "DF (FFN Node)"),
]):
    ax = axes[ax_idx]
    num_layers = 64
    num_mb = 3

    # Build matrix: rows=layers, cols=micro-batches, separate for A and F
    matrix = np.zeros((num_layers, num_mb * 2))
    for s in steps:
        layer = s["layer"]
        mb = s["mb"]
        is_A = s["stage"] == "AFD_FORWARD_STAGE_A"
        col = mb * 2 + (0 if is_A else 1)
        matrix[layer, col] = s["dur_ms"]

    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(num_mb * 2))
    ax.set_xticklabels([f"mb0-A", f"mb0-F", f"mb1-A", f"mb1-F", f"mb2-A", f"mb2-F"], rotation=45)
    ax.set_ylabel("Layer")
    ax.set_title(f"{title} — Stage Duration (ms)")
    plt.colorbar(im, ax=ax, label="Duration (ms)")

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "pipeline_per_layer_heatmap.png"), dpi=150, bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "pipeline_per_layer_heatmap.svg"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: pipeline_per_layer_heatmap.png/.svg")

# ── Save data summary ────────────────────────────────────────────────────────

summary = {
    "config": {
        "model": "Qwen3-32B",
        "layers": 64,
        "micro_batches": 3,
        "tp": "tp=1 per module, 4 GPUs total (DA/DF/PA/PF)",
        "decode_batch_size": "~6 requests → M=3",
        "output_tokens_per_request": 8,
    },
    "da_aggregate_ms": da_bd,
    "df_aggregate_ms": df_bd,
    "compute_vs_wait": {
        "da": {"compute_ms": round(da_compute, 1), "wait_ms": round(da_wait, 1),
               "compute_pct": round(da_compute / da_total_ms * 100, 1)},
        "df": {"compute_ms": round(df_compute, 1), "wait_ms": round(df_wait, 1),
               "compute_pct": round(df_compute / df_total_ms * 100, 1)},
    },
    "pipeline_efficiency": {
        "da_utilization_pct": round(da_compute / da_total_ms * 100, 1),
        "df_utilization_pct": round(df_compute / df_total_ms * 100, 1),
        "overall_utilization_pct": round((da_compute + df_compute) / (da_total_ms + df_total_ms) * 100, 1),
        "note": "Wait time dominated by UCX RDMA transfers between DA↔DF and pipeline serialization"
    }
}

with open(os.path.join(OUT_DIR, "analysis_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved: analysis_summary.json")

print(f"\n{'='*60}")
print(f"Pipeline Analysis Complete")
print(f"{'='*60}")
print(f"DA total: {da_total_ms:.1f}ms | Compute: {da_compute:.0f}ms ({da_compute/da_total_ms*100:.0f}%) | Wait: {da_wait:.0f}ms ({da_wait/da_total_ms*100:.0f}%)")
print(f"DF total: {df_total_ms:.1f}ms | Compute: {df_compute:.0f}ms ({df_compute/df_total_ms*100:.0f}%) | Wait: {df_wait:.0f}ms ({df_wait/df_total_ms*100:.0f}%)")
print(f"Overall GPU utilization: {(da_compute+df_compute)/(da_total_ms+df_total_ms)*100:.0f}%")
print(f"\nResults saved to: {OUT_DIR}")
