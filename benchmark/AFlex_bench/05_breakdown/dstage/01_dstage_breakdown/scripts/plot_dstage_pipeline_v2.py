#!/usr/bin/env python3
"""Plot D-stage pipeline Gantt chart using actual AFD_PER_STEP CUDA event timings.

Plots a 3-row Gantt:
  Row 1: DA (attn node) GPU timeline — sub-stage granularity
  Row 2: DF (ffn node) GPU timeline
  Row 3: Communication arrows between DA and DF

Usage: python3 plot_dstage_pipeline_v2.py [--layers N] [--save]
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────
MAX_LAYERS_TO_SHOW = 4
FIGURE_WIDTH = 20
FIGURE_HEIGHT = 7.5

COLORS = {
    "prep_attn":   "#5dade2",  # blue  - input norm
    "attn":        "#2e86c1",  # dark blue - attention compute
    "prep_mlp":    "#f4d03f",  # yellow - output + send
    "mlp":         "#e74c3c",  # red - MLP compute (FFN node only)
    "postprocess": "#8e44ad",  # purple - recv + output norm
    "recv_wait":   "#f5b041",  # orange - waiting for data
    "proxy":       "#d5dbdb",  # gray - no-op proxy
}

SUB_STAGE_LABELS = {
    "prep_attn": "prep_attn",
    "attn": "attn",
    "prep_mlp": "prep_mlp + send",
    "mlp": "mlp",
    "postprocess": "recv + postprocess",
}

COMM_COLORS = {"DA_to_DF": "#e67e22", "DF_to_DA": "#2ecc71"}


def parse_steps(line: str) -> list:
    idx = line.rindex("steps=")
    return json.loads(line[idx + 6:])


def build_timeline(steps: list):
    """Convert per-step CUDA events into sequential GPU timeline.

    Returns (timeline, total_duration) and event dicts for comm alignment:
    - send_A: t_end of prep_mlp in A-stage (= DA sends to DF)
    - recv_F: t_end of postprocess in F-stage (= DF sends to DA / DA recv done)
    - recv_start: t_start of prep_attn in A-stage (= DF begins receiving)
    - recv_end:   t_end   of prep_attn in A-stage (= DF recv completed)
    """
    tl = []
    t = 0.0
    send_A = {}
    recv_F = {}
    recv_start = {}
    recv_end = {}

    for s in steps:
        stage = s["stage"]
        lid = s["layer_id"]
        mb = s["mb"]
        bs = s.get("batch_size", -1)

        if stage == "A":
            subs = ["prep_attn", "attn", "prep_mlp"]
        else:
            subs = ["mlp", "postprocess"]

        for ss in subs:
            dur = s.get(f"{ss}_ms", 0)
            dur = max(dur, 0.001)
            entry = {
                "t_start": t, "t_end": t + dur,
                "sub_stage": ss, "stage": stage,
                "layer_id": lid, "mb": mb, "batch_size": bs,
            }
            tl.append(entry)

            if stage == "A" and ss == "prep_attn":
                recv_start[(lid, mb)] = t
                recv_end[(lid, mb)] = t + dur
            elif stage == "A" and ss == "prep_mlp":
                send_A[(lid, mb)] = t + dur
            elif stage == "F" and ss == "postprocess":
                recv_F[(lid, mb)] = t + dur

            t += dur

    return tl, t, send_A, recv_F, recv_start, recv_end


def find_matching_iteration(da_path, df_path):
    """Find the max-batch iteration with matching batch signature."""
    with open(da_path) as f:
        da_lines = [l for l in f if "[AFD_PER_STEP]" in l]
    with open(df_path) as f:
        df_lines = [l for l in f if "[AFD_PER_STEP]" in l]

    def best_match(lines):
        best = None
        best_sum = -1
        for line in lines:
            steps = parse_steps(line)
            bs_sum = sum(s["batch_size"] for s in steps if s["stage"] == "A")
            if bs_sum > best_sum:
                best_sum = bs_sum
                best = line
        return parse_steps(best), best_sum

    da_steps, da_sum = best_match(da_lines)
    df_steps, df_sum = best_match(df_lines)
    return da_steps, df_steps, da_sum, df_sum


def format_label(layer_id, mb, stage):
    return f"L{layer_id}.{mb}{stage}"


def plot_gantt(da_steps, df_steps, max_layers=MAX_LAYERS_TO_SHOW, save_path=None,
               variant="m3", bs_sig=None):
    da_tl, da_total, da_send, da_recv, da_rstart, da_rend = build_timeline(da_steps)
    df_tl, df_total, df_send, df_recv, df_rstart, df_rend = build_timeline(df_steps)

    print(f"DA: {len(da_tl)} bars, total = {da_total:.2f}ms")
    print(f"DF: {len(df_tl)} bars, total = {df_total:.2f}ms")

    # ── Estimate DF offset for wall-time alignment ──────────────────────
    # Align so that DA send always precedes DF recv (forward arrows).
    # offset = max(da_send - df_rstart) for the displayed layers ensures
    # no DA→DF arrow points backward (means DF receives before DA sends).
    #
    # Physical interpretation of the offset:
    #   DF's GPU timeline starts ~offset ms after DA's due to PD KV cache
    #   transfer overhead (MoonCake). DF events at DF_t appear at wall
    #   time = DF_t + offset relative to DA_start.
    delays = []
    for key in sorted(set(da_send.keys()) & set(df_rstart.keys())):
        lid, mb = key
        if lid < max_layers:
            delays.append(da_send[key] - df_rstart[key])
    df_offset = max(delays) if delays else 0
    df_offset = max(df_offset, 0)

    print(f"Computed DF offset: {df_offset:.3f}ms "
          f"(max of {len(delays)} da_send-df_rstart pairs, layers 0-{max_layers - 1})")
    print(f"  (PD KV cache transfer delay via MoonCake ≈ {df_offset:.0f}ms)")

    # Apply offset to DF
    def shift_tl(tl, offset):
        out = []
        for r in tl:
            r = dict(r)
            r["t_start"] += offset
            r["t_end"] += offset
            out.append(r)
        return out

    df_tl = shift_tl(df_tl, df_offset)

    def shift_dict(d, offset):
        return {k: v + offset for k, v in d.items()}

    df_send = shift_dict(df_send, df_offset)
    df_recv = shift_dict(df_recv, df_offset)
    df_rstart = shift_dict(df_rstart, df_offset)
    df_rend = shift_dict(df_rend, df_offset)

    # Filter to first N layers
    da_tl_f = [r for r in da_tl if r["layer_id"] < max_layers]
    df_tl_f = [r for r in df_tl if r["layer_id"] < max_layers]

    da_send_f = {k: v for k, v in da_send.items() if k[0] < max_layers}
    da_recv_f = {k: v for k, v in da_recv.items() if k[0] < max_layers}
    df_send_f = {k: v for k, v in df_send.items() if k[0] < max_layers}
    df_recv_f = {k: v for k, v in df_recv.items() if k[0] < max_layers}
    df_rstart_f = {k: v for k, v in df_rstart.items() if k[0] < max_layers}
    df_rend_f = {k: v for k, v in df_rend.items() if k[0] < max_layers}

    t_max = max(
        max((r["t_end"] for r in da_tl_f), default=0),
        max((r["t_end"] for r in df_tl_f), default=0),
    )
    t_max = max(t_max, 0.1) * 1.05  # 5% padding on right

    # ── Figure ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    fig.patch.set_facecolor("white")

    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.0, 1.2],
                          hspace=0.08, left=0.08, right=0.97,
                          top=0.93, bottom=0.08)
    ax_da = fig.add_subplot(gs[0])
    ax_comm = fig.add_subplot(gs[1], sharex=ax_da)
    ax_df = fig.add_subplot(gs[2], sharex=ax_da)

    y_center = 0

    def draw_node_bars(ax, tl, y_mid):
        BAR_H = 18
        for rec in tl:
            dur = rec["t_end"] - rec["t_start"]
            if dur < 0.002:
                continue
            ss = rec["sub_stage"]
            color = COLORS.get(ss, "#ccc")
            label = f"L{rec['layer_id']}.{rec['mb']}{rec['stage']}"
            ax.barh(y_mid, dur, left=rec["t_start"], height=BAR_H,
                    color=color, edgecolor="white", linewidth=0.3)
            text_w = max(12, len(label) * 4 + 4)
            if dur * 25 > text_w:
                ax.text(rec["t_start"] + dur / 2, y_mid, label,
                        ha="center", va="center", fontsize=5.5,
                        fontweight="bold", color="white")

    # ── DA row ─────────────────────────────────────────────────────────────
    draw_node_bars(ax_da, da_tl_f, y_center)
    seen = set()
    for rec in da_tl_f:
        if rec["stage"] == "A" and rec["mb"] == 0 and rec["layer_id"] > 0 \
           and rec["layer_id"] not in seen:
            ax_da.axvline(x=rec["t_start"], color="gray", linewidth=0.6,
                          linestyle="--", alpha=0.4)
            seen.add(rec["layer_id"])
    ax_da.set_ylabel("DA (Attn GPU)", fontsize=10, fontweight="bold")
    ax_da.set_yticks([])
    ax_da.set_ylim(-28, 28)
    ax_da.grid(axis="x", alpha=0.3, linewidth=0.3)
    da_handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[ss],
                   label=SUB_STAGE_LABELS.get(ss, ss))
                  for ss in ["prep_attn", "attn", "prep_mlp", "postprocess"]]
    ax_da.legend(handles=da_handles, loc="upper right", fontsize=6,
                 ncol=6, framealpha=0.85, edgecolor="gray")
    ax_da.text(0.98, 0.06, f"DA total: {da_total:.1f}ms",
               transform=ax_da.transAxes, fontsize=8, ha="right", va="bottom",
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                         edgecolor="gray", alpha=0.8))

    # ── DF row ─────────────────────────────────────────────────────────────
    draw_node_bars(ax_df, df_tl_f, y_center)
    seen.clear()
    for rec in df_tl_f:
        if rec["stage"] == "A" and rec["mb"] == 0 and rec["layer_id"] > 0 \
           and rec["layer_id"] not in seen:
            ax_df.axvline(x=rec["t_start"], color="gray", linewidth=0.6,
                          linestyle="--", alpha=0.4)
            seen.add(rec["layer_id"])
    ax_df.set_ylabel("DF (FFN GPU)", fontsize=10, fontweight="bold")
    ax_df.set_yticks([])
    ax_df.set_ylim(-28, 28)
    ax_df.set_xlabel("Time (ms) — CUDA event elapsed time",
                     fontsize=9)
    ax_df.grid(axis="x", alpha=0.3, linewidth=0.3)
    df_handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[ss],
                   label=SUB_STAGE_LABELS.get(ss, ss))
                  for ss in ["prep_mlp", "mlp", "postprocess"]]
    ax_df.legend(handles=df_handles, loc="upper right", fontsize=6,
                 ncol=5, framealpha=0.85, edgecolor="gray")
    ax_df.text(0.98, 0.06, f"DF total: {df_total:.1f}ms",
               transform=ax_df.transAxes, fontsize=8, ha="right", va="bottom",
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                         edgecolor="gray", alpha=0.8))

    # ── Communication row ──────────────────────────────────────────────────
    ax_comm.set_ylabel("Comm", fontsize=10, fontweight="bold")
    ax_comm.set_yticks([])
    ax_comm.set_ylim(-30, 30)
    ax_comm.grid(axis="x", alpha=0.3, linewidth=0.3)

    # Communication arrows
    # Upper half (y=+12): DA→DF — DA sends (prep_mlp end) → DF receives (prep_attn start)
    # Lower half (y=-12): DF→DA — DF sends (postprocess end) → DA receives (postprocess start)
    all_keys = sorted(set(da_send_f.keys()) | set(df_rstart_f.keys())
                      | set(df_recv_f.keys()) | set(da_recv_f.keys()))
    for key in all_keys:
        lid, mb = key
        if key in da_send_f and key in df_rstart_f:
            c = COMM_COLORS["DA_to_DF"]
            from_t = da_send_f[key]
            to_t = df_rstart_f[key]
            if abs(from_t - to_t) > 0.01:
                t0, t1 = (from_t, to_t) if from_t <= to_t else (to_t, from_t)
                ax_comm.annotate(
                    "", xy=(t1, 12), xytext=(t0, 12),
                    arrowprops=dict(arrowstyle="->",
                                    color=c, linewidth=3.0,
                                    connectionstyle="arc3,rad=0.2"),
                )
        if key in df_recv_f and key in da_recv_f:
            c = COMM_COLORS["DF_to_DA"]
            from_t = df_recv_f[key]
            to_t = da_recv_f[key]
            if abs(from_t - to_t) > 0.01:
                t0, t1 = (from_t, to_t) if from_t <= to_t else (to_t, from_t)
                ax_comm.annotate(
                    "", xy=(t1, -12), xytext=(t0, -12),
                    arrowprops=dict(arrowstyle="->",
                                    color=c, linewidth=3.0,
                                    connectionstyle="arc3,rad=-0.2"),
                )

    ax_comm.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=COMM_COLORS["DA_to_DF"],
                          label="DA→DF (attn sends hidden_states)"),
            plt.Rectangle((0, 0), 1, 1, color=COMM_COLORS["DF_to_DA"],
                          label="DF→DA (ffn returns MLP result)"),
        ],
        loc="upper right", fontsize=6, ncol=2,
        framealpha=0.85, edgecolor="gray",
    )
    ax_comm.text(
        0.5, 0.88,
        f"DF GPU timeline shifted by +{df_offset:.2f}ms "
        f"(PD KV xfer via MoonCake)",
        transform=ax_comm.transAxes, fontsize=6.5, ha="center", va="center",
        style="italic", color="gray",
    )

    # Set common x-axis limits
    for ax in [ax_da, ax_comm, ax_df]:
        ax.set_xlim(0, t_max)

    # ── Title ──────────────────────────────────────────────────────────────
    m_label = variant.upper().replace("M", "M=")
    bs_str = str(bs_sig) if bs_sig else ""
    fig.suptitle(
        f"PD+AF {m_label} D-Stage Pipeline (CUDA Event Timings, Max-Batch Iteration)\n"
        f"DA(attn) total={da_total:.1f}ms  DF(ffn) total={df_total:.1f}ms  |  "
        f"Layers 0-{max_layers - 1}  |  QPS=2  {bs_str}  "
        f"|  DF offset +{df_offset:.1f}ms",
        fontsize=11, fontweight="bold",
    )

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=MAX_LAYERS_TO_SHOW)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--variant", type=str, default="m3", choices=["m1", "m3"])
    args = parser.parse_args()

    log_dir = os.path.join(os.path.dirname(__file__), "throughput_logs")
    da_path = os.path.join(log_dir, f"qps_sweep_{args.variant}_da.log")
    df_path = os.path.join(log_dir, f"qps_sweep_{args.variant}_df.log")

    if not os.path.exists(da_path):
        print(f"ERROR: {da_path} not found")
        sys.exit(1)
    if not os.path.exists(df_path):
        print(f"ERROR: {df_path} not found")
        sys.exit(1)

    da_steps, df_steps, da_sum, df_sum = find_matching_iteration(da_path, df_path)
    sig_da = [s["batch_size"] for s in da_steps if s["stage"] == "A"][:3]
    sig_df = [s["batch_size"] for s in df_steps if s["stage"] == "A"][:3]
    print(f"DA: total_bs={da_sum}, sig={sig_da}, nsteps={len(da_steps)}")
    print(f"DF: total_bs={df_sum}, sig={sig_df}, nsteps={len(df_steps)}")

    save_path = args.save or os.path.join(
        os.path.dirname(__file__),
        f"dstage_pipeline_gantt_v2_l{args.layers}.png",
    )
    plot_gantt(da_steps, df_steps, max_layers=args.layers, save_path=save_path,
               variant=args.variant, bs_sig=sig_da)


if __name__ == "__main__":
    main()
