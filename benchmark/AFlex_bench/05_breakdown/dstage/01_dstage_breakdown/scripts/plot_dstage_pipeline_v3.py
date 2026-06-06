#!/usr/bin/env python3
"""Plot D-stage pipeline Gantt chart using wall-clock aligned timestamps.

DA and DF events are aligned to the same host wall-clock, so the
3-row Gantt shows true cross-GPU pipeline overlap without any offset hack.

Usage: python3 plot_dstage_pipeline_v3.py [--layers N] [--save PATH]
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────────────────────
MAX_LAYERS_TO_SHOW = 4
FIGURE_WIDTH = 24
FIGURE_HEIGHT = 8.0

COLORS = {
    "prep_attn":   "#5dade2",  # blue  - input norm
    "attn":        "#2e86c1",  # dark blue - attention compute
    "prep_mlp":    "#f4d03f",  # yellow - output + send
    "mlp":         "#e74c3c",  # red - MLP compute (FFN node only)
    "postprocess": "#8e44ad",  # purple - recv + output norm
    "recv_wait":   "#f5b041",  # orange - waiting for data
    "proxy":       "#d5dbdb",  # gray - no-op proxy
}

SUB_STAGE_LABELS_DA = {
    "prep_attn": "input norm",
    "attn": "attention",
    "prep_mlp": "output norm + send→DF",
    "mlp": "(proxy)",
    "postprocess": "recv←DF + output norm",
}

SUB_STAGE_LABELS_DF = {
    "prep_attn": "(no-op)",
    "attn": "(proxy)",
    "prep_mlp": "recv←DA",
    "mlp": "FFN compute",
    "postprocess": "send→DA",
}

COMM_COLORS = {"DA_to_DF": "#e67e22", "DF_to_DA": "#2ecc71"}


def parse_steps(line: str) -> list:
    idx = line.rindex("steps=")
    return json.loads(line[idx + 6:])


def has_wall_clock(steps: list) -> bool:
    """Check if steps have wall-clock timestamp fields (new afd.py)."""
    for s in steps:
        for key in s:
            if key.endswith("_wall_start_ms"):
                return True
    return False


def build_timeline_wallclock(steps: list):
    """Build timeline using absolute wall-clock timestamps.

    Returns (timeline, total_duration, send_A, recv_F, recv_start, recv_end).
    Wall-clock keys:
      A-stage: prep_attn_wall_start/end, attn_wall_start/end, prep_mlp_wall_start/end
      F-stage: mlp_wall_start/end, postprocess_wall_start/end
    """
    tl = []
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
            t_start = s.get(f"{ss}_wall_start_ms", 0)
            t_end = s.get(f"{ss}_wall_end_ms", 0)
            dur = max(t_end - t_start, 0.001)

            entry = {
                "t_start": t_start, "t_end": t_end,
                "sub_stage": ss, "stage": stage,
                "layer_id": lid, "mb": mb, "batch_size": bs,
            }
            tl.append(entry)

            if stage == "A" and ss == "prep_attn":
                recv_start[(lid, mb)] = t_start
                recv_end[(lid, mb)] = t_end
            elif stage == "A" and ss == "prep_mlp":
                send_A[(lid, mb)] = t_end
            elif stage == "F" and ss == "postprocess":
                recv_F[(lid, mb)] = t_end

    return tl, send_A, recv_F, recv_start, recv_end


def build_timeline_fallback(steps: list):
    """Fallback: sequential elapsed-time timeline (old afd.py without wall-clock).

    Returns same structure as build_timeline_wallclock.
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

    return tl, send_A, recv_F, recv_start, recv_end


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


def plot_gantt(da_steps, df_steps, max_layers=MAX_LAYERS_TO_SHOW, save_path=None,
               variant="m3", bs_sig=None, use_wallclock=True):
    if use_wallclock:
        da_tl, da_send, da_recv, da_rstart, da_rend = build_timeline_wallclock(da_steps)
        df_tl, df_send, df_recv, df_rstart, df_rend = build_timeline_wallclock(df_steps)
        # Normalize: subtract global minimum wall-clock time
        t_da_min = min((r["t_start"] for r in da_tl), default=0)
        t_df_min = min((r["t_start"] for r in df_tl), default=0)
        t_min = min(t_da_min, t_df_min)

        def renorm(tl):
            for r in tl:
                r["t_start"] -= t_min
                r["t_end"] -= t_min

        def renorm_dict(d):
            return {k: v - t_min for k, v in d.items()}

        renorm(da_tl)
        renorm(df_tl)
        da_send = renorm_dict(da_send)
        da_recv = renorm_dict(da_recv)
        da_rstart = renorm_dict(da_rstart)
        da_rend = renorm_dict(da_rend)
        df_send = renorm_dict(df_send)
        df_recv = renorm_dict(df_recv)
        df_rstart = renorm_dict(df_rstart)
        df_rend = renorm_dict(df_rend)

        print(f"Wall-clock normalization: t_min = {t_min:.2f}ms (host epoch)")
        alignment_label = "Wall-Clock Aligned (same host)"
    else:
        da_tl, _, da_send, da_recv, da_rstart, da_rend = build_timeline_fallback(da_steps)
        df_tl, _, df_send, df_recv, df_rstart, df_rend = build_timeline_fallback(df_steps)

        # Old df_offset hack
        delays = []
        for key in sorted(set(da_send.keys()) & set(df_rstart.keys())):
            lid, mb = key
            if lid < max_layers:
                delays.append(da_send[key] - df_rstart[key])
        df_offset = max(delays) if delays else 0
        df_offset = max(df_offset, 0)

        def shift_tl(tl, offset):
            return [dict(r, t_start=r["t_start"]+offset, t_end=r["t_end"]+offset) for r in tl]
        def shift_dict(d, offset):
            return {k: v+offset for k,v in d.items()}

        df_tl = shift_tl(df_tl, df_offset)
        df_send = shift_dict(df_send, df_offset)
        df_recv = shift_dict(df_recv, df_offset)
        df_rstart = shift_dict(df_rstart, df_offset)
        df_rend = shift_dict(df_rend, df_offset)
        alignment_label = f"DF shifted +{df_offset:.1f}ms (estimated)"

    # Compute totals
    if use_wallclock:
        da_total = sum(r["t_end"] - r["t_start"] for r in da_tl if r["sub_stage"] == "attn") + \
                   sum(r["t_end"] - r["t_start"] for r in da_tl if r["sub_stage"] != "attn")
        df_total = sum(r["t_end"] - r["t_start"] for r in df_tl)
    else:
        da_total = da_tl[-1]["t_end"] if da_tl else 0
        df_total = df_tl[-1]["t_end"] if df_tl else 0

    print(f"DA: {len(da_tl)} bars, total_gpu = {da_total:.2f}ms")
    print(f"DF: {len(df_tl)} bars, total_gpu = {df_total:.2f}ms")

    # Filter to first N layers
    da_tl_f = [r for r in da_tl if r["layer_id"] < max_layers]
    df_tl_f = [r for r in df_tl if r["layer_id"] < max_layers]

    da_send_f = {k: v for k, v in da_send.items() if k[0] < max_layers}
    da_recv_f = {k: v for k, v in da_recv.items() if k[0] < max_layers}
    df_send_f = {k: v for k, v in df_send.items() if k[0] < max_layers}
    df_recv_f = {k: v for k, v in df_recv.items() if k[0] < max_layers}
    df_rstart_f = {k: v for k, v in df_rstart.items() if k[0] < max_layers}
    df_rend_f = {k: v for k, v in df_rend.items() if k[0] < max_layers}

    # Wall-clock range (not just GPU time)
    t_max = max(
        max((r["t_end"] for r in da_tl_f), default=0),
        max((r["t_end"] for r in df_tl_f), default=0),
    )
    t_max = max(t_max, 0.1) * 1.05

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
            label = f"L{rec['layer_id']}.{rec['mb']}"
            ax.barh(y_mid, dur, left=rec["t_start"], height=BAR_H,
                    color=color, edgecolor="white", linewidth=0.3)
            # Always show label: inside bar if wide enough, else slightly above
            text_w = max(10, len(label) * 5 + 2)
            if dur * 30 > text_w:
                ax.text(rec["t_start"] + dur / 2, y_mid, label,
                        ha="center", va="center", fontsize=5,
                        fontweight="bold", color="white")
            else:
                ax.text(rec["t_start"] + dur / 2, y_mid + BAR_H / 2 + 1,
                        label, ha="center", va="bottom", fontsize=4.5,
                        fontweight="bold", color=color)

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
                   label=SUB_STAGE_LABELS_DA.get(ss, ss))
                  for ss in ["prep_attn", "attn", "prep_mlp", "postprocess"]]
    ax_da.legend(handles=da_handles, loc="upper right", fontsize=6,
                 ncol=4, framealpha=0.85, edgecolor="gray")

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
    ax_df.set_xlabel("Wall-Clock Time (ms, relative)", fontsize=9)
    ax_df.grid(axis="x", alpha=0.3, linewidth=0.3)
    df_handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[ss],
                   label=SUB_STAGE_LABELS_DF.get(ss, ss))
                  for ss in ["prep_attn", "attn", "prep_mlp", "mlp", "postprocess"]]
    ax_df.legend(handles=df_handles, loc="upper right", fontsize=6,
                 ncol=3, framealpha=0.85, edgecolor="gray")

    # ── Communication row ──────────────────────────────────────────────────
    ax_comm.set_ylabel("Comm", fontsize=10, fontweight="bold")
    ax_comm.set_yticks([])
    ax_comm.set_ylim(-30, 30)
    ax_comm.grid(axis="x", alpha=0.3, linewidth=0.3)

    # Communication arrows (wall-clock aligned!)
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
        alignment_label,
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
        f"PD+AF {m_label} D-Stage Pipeline ({alignment_label})\n"
        f"DA(attn) GPU={da_total:.1f}ms  DF(ffn) GPU={df_total:.1f}ms  |  "
        f"Layers 0-{max_layers - 1}  |  QPS=2  {bs_str}",
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
    parser.add_argument("--fallback", action="store_true",
                        help="Use elapsed-time fallback instead of wall-clock")
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

    use_wc = has_wall_clock(da_steps) and not args.fallback
    print(f"Using {'wall-clock' if use_wc else 'elapsed-time fallback'} timestamps")

    save_path = args.save or os.path.join(
        os.path.dirname(__file__),
        f"dstage_pipeline_gantt_v3_{args.variant}_l{args.layers}.png",
    )
    plot_gantt(da_steps, df_steps, max_layers=args.layers, save_path=save_path,
               variant=args.variant, bs_sig=sig_da, use_wallclock=use_wc)


if __name__ == "__main__":
    main()
