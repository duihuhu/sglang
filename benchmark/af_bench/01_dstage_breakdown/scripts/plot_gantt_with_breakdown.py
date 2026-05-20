#!/usr/bin/env python3
"""Combined Gantt + Breakdown chart for DA→DF pipeline analysis.

Top: 3-row Gantt (DA / Comm / DF) — wall-clock aligned
Bottom: ZMQ latency timeline + UCX gap breakdown + UCX daemon thread detail

Usage: python plot_gantt_with_breakdown.py [--layers N] [--save PATH]
"""

import argparse, json, os, re, sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────
MAX_LAYERS = 4
FIG_WIDTH = 26
FIG_HEIGHT = 12.5

COLORS = {
    "prep_attn":   "#5dade2",
    "attn":        "#2e86c1",
    "prep_mlp":    "#f4d03f",
    "mlp":         "#e74c3c",
    "postprocess": "#8e44ad",
    "recv_wait":   "#f5b041",
    "proxy":       "#d5dbdb",
}
SUB_STAGE_LABELS_DA = {
    "prep_attn": "input norm", "attn": "attention",
    "prep_mlp": "output norm + send→DF", "mlp": "(proxy)",
    "postprocess": "recv←DF + output norm",
}
SUB_STAGE_LABELS_DF = {
    "prep_attn": "(no-op)", "attn": "(proxy)",
    "prep_mlp": "recv←DA", "mlp": "FFN compute",
    "postprocess": "send→DA",
}
COMM_COLORS = {"DA_to_DF": "#e67e22", "DF_to_DA": "#2ecc71"}


# ── Data loaders ─────────────────────────────────────────────────────────────

def parse_per_step(line: str) -> list:
    idx = line.rindex("steps=")
    return json.loads(line[idx + 6:])

def has_wall_clock(steps: list) -> bool:
    for s in steps:
        for key in s:
            if key.endswith("_wall_start_ms"):
                return True
    return False

def build_timeline_wallclock(steps, max_layers):
    tl = []
    send_A, recv_F, recv_start, recv_end = {}, {}, {}, {}
    for s in steps:
        stage, lid, mb = s["stage"], s["layer_id"], s["mb"]
        subs = ["prep_attn", "attn", "prep_mlp"] if stage == "A" else ["mlp", "postprocess"]
        for ss in subs:
            t_start = s.get(f"{ss}_wall_start_ms", 0)
            t_end = s.get(f"{ss}_wall_end_ms", 0)
            dur = max(t_end - t_start, 0.001)
            tl.append({"t_start": t_start, "t_end": t_end, "sub_stage": ss,
                       "stage": stage, "layer_id": lid, "mb": mb,
                       "batch_size": s.get("batch_size", -1)})
            if stage == "A" and ss == "prep_attn":
                recv_start[(lid, mb)] = t_start; recv_end[(lid, mb)] = t_end
            elif stage == "A" and ss == "prep_mlp":
                send_A[(lid, mb)] = t_end
            elif stage == "F" and ss == "postprocess":
                recv_F[(lid, mb)] = t_end
    return tl, send_A, recv_F, recv_start, recv_end

def extract_host_events(log_path):
    """Extract [AFD_HOST_EVENTS] and [AFD_SCHED_TS] lines."""
    events, sched_ts = [], []
    pat_host = re.compile(r'\[AFD_HOST_EVENTS\]\s+role=(\S+).*?events=(\[.*\])')
    pat_sched = re.compile(r'\[AFD_SCHED_TS\]\s+sched_timestamps=(\{.*?\})')
    with open(log_path) as f:
        for line in f:
            m = pat_host.search(line)
            if m:
                try: events.append({"role": m.group(1), "events": json.loads(m.group(2))})
                except: pass
            m = pat_sched.search(line)
            if m:
                try: sched_ts.append(json.loads(m.group(1)))
                except: pass
    return events, sched_ts

def match_send_recv_pairs(da_events, df_events):
    """Match DA send with DF recv by (layer, mb) in sequential order."""
    da_sends, df_recvs = [], []
    for ev_set in da_events:
        pending = {}
        for e in ev_set["events"]:
            if e["role"] == "DA" and e["event"] == "send_start":
                pending[(e["layer"], e["mb"])] = e["ts_ms"]
            elif e["role"] == "DA" and e["event"] == "send_end":
                k = (e["layer"], e["mb"])
                if k in pending:
                    da_sends.append({"layer": e["layer"], "mb": e["mb"],
                                     "send_start": pending[k], "send_end": e["ts_ms"]})
                    del pending[k]
    for ev_set in df_events:
        pending = {}
        for e in ev_set["events"]:
            if e["role"] == "DF" and e["event"] == "recv_start":
                pending[(e["layer"], e["mb"])] = e["ts_ms"]
            elif e["role"] == "DF" and e["event"] == "recv_end":
                k = (e["layer"], e["mb"])
                if k in pending:
                    df_recvs.append({"layer": e["layer"], "mb": e["mb"],
                                     "recv_start": pending[k], "recv_end": e["ts_ms"],
                                     "recv_dur_us": e.get("recv_dur_us", 0)})
                    del pending[k]
    # Match by sequential order per key
    sends_by_key = defaultdict(list)
    for s in da_sends: sends_by_key[(s["layer"], s["mb"])].append(s)
    recvs_by_key = defaultdict(list)
    for r in df_recvs: recvs_by_key[(r["layer"], r["mb"])].append(r)
    pairs = []
    for key in sorted(sends_by_key):
        ss, rr = sends_by_key[key], recvs_by_key.get(key, [])
        for i in range(min(len(ss), len(rr))):
            s, r = ss[i], rr[i]
            pairs.append({"layer": key[0], "mb": key[1],
                          "send_start": s["send_start"], "send_end": s["send_end"],
                          "recv_end": r["recv_end"], "recv_start": r["recv_start"],
                          "gap_ms": r["recv_end"] - s["send_start"],
                          "send_dur_us": (s["send_end"] - s["send_start"]) * 1e3,
                          "recv_dur_us": r["recv_dur_us"]})
    return pairs

def extract_ucx_sends(da_events):
    ucx = []
    for ev_set in da_events:
        launch = cuda_sync = send_done = None
        for e in ev_set["events"]:
            if e["role"] == "SENDER":
                if e["event"] == "ucx_send_launched": launch = e["ts_ms"]
                elif e["event"] == "ucx_cuda_sync_done": cuda_sync = e["ts_ms"]
                elif e["event"] == "ucx_send_done":
                    if launch and cuda_sync:
                        ucx.append({"launch": launch, "cuda_sync": cuda_sync,
                                    "ucx_done": e["ts_ms"],
                                    "cuda_us": (cuda_sync - launch) * 1e3,
                                    "send_us": (e["ts_ms"] - cuda_sync) * 1e3})
                    launch = cuda_sync = send_done = None
    return ucx


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_combined(da_steps, df_steps, da_events, df_events, da_sched, df_sched,
                  max_layers, save_path, variant="m3"):
    da_tl, da_send, da_recv, da_rstart, da_rend = build_timeline_wallclock(da_steps, max_layers)
    df_tl, df_send, df_recv, df_rstart, df_rend = build_timeline_wallclock(df_steps, max_layers)

    # Normalize wall-clock
    t_min = min(min(r["t_start"] for r in da_tl), min(r["t_start"] for r in df_tl))
    def renorm(tl):
        for r in tl: r["t_start"] -= t_min; r["t_end"] -= t_min
    def rd(d): return {k: v - t_min for k, v in d.items()}
    renorm(da_tl); renorm(df_tl)
    da_send, da_recv, da_rstart, da_rend = rd(da_send), rd(da_recv), rd(da_rstart), rd(da_rend)
    df_send, df_recv, df_rstart, df_rend = rd(df_send), rd(df_recv), rd(df_rstart), rd(df_rend)

    # Match send→recv pairs and normalize times
    pairs = match_send_recv_pairs(da_events, df_events)
    for p in pairs:
        p["send_start_n"] = p["send_start"] - t_min
        p["send_end_n"] = p["send_end"] - t_min
        p["recv_end_n"] = p["recv_end"] - t_min
        p["recv_start_n"] = p["recv_start"] - t_min

    ucx_sends = extract_ucx_sends(da_events)

    # Build per-iteration ZMQ data (normalized)
    zmq_data = []
    for i in range(min(len(da_sched), len(df_sched))):
        zsent = da_sched[i].get("zmq_sent")
        zrecv = df_sched[i].get("zmq_recv")
        da_fs = da_sched[i].get("forward_start")
        df_fs = df_sched[i].get("forward_start")
        if zsent and zrecv:
            zmq_data.append({
                "iter": i,
                "zmq_lat": zrecv - zsent,
                "da_sent": zsent - t_min,
                "df_recv": zrecv - t_min,
                "da_fwd_start": (da_fs - t_min) if da_fs else 0,
                "df_fwd_start": (df_fs - t_min) if df_fs else 0,
            })

    # Filter: first few layers
    da_tl_f = [r for r in da_tl if r["layer_id"] < max_layers]
    df_tl_f = [r for r in df_tl if r["layer_id"] < max_layers]
    t_max = max(max((r["t_end"] for r in da_tl_f), default=0),
                max((r["t_end"] for r in df_tl_f), default=0)) * 1.05

    # ── Figure grid ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    fig.patch.set_facecolor("white")

    gs = fig.add_gridspec(5, 2, height_ratios=[1.2, 0.7, 1.2, 1.0, 1.0],
                          hspace=0.12, wspace=0.18,
                          left=0.06, right=0.97, top=0.95, bottom=0.05)

    ax_da = fig.add_subplot(gs[0, :])
    ax_comm = fig.add_subplot(gs[1, :], sharex=ax_da)
    ax_df = fig.add_subplot(gs[2, :], sharex=ax_da)

    # ── Gantt rows ────────────────────────────────────────────────────────
    y_center = 0; BAR_H = 18
    def draw_bars(ax, tl):
        for rec in tl:
            dur = rec["t_end"] - rec["t_start"]
            if dur < 0.002: continue
            ss = rec["sub_stage"]
            color = COLORS.get(ss, "#ccc")
            label = f"L{rec['layer_id']}.{rec['mb']}"
            ax.barh(y_center, dur, left=rec["t_start"], height=BAR_H,
                    color=color, edgecolor="white", linewidth=0.3)
            text_w = max(10, len(label) * 5 + 2)
            if dur * 30 > text_w:
                ax.text(rec["t_start"] + dur / 2, y_center, label,
                        ha="center", va="center", fontsize=5,
                        fontweight="bold", color="white")
            else:
                ax.text(rec["t_start"] + dur / 2, y_center + BAR_H / 2 + 1,
                        label, ha="center", va="bottom", fontsize=4.5,
                        fontweight="bold", color=color)

    draw_bars(ax_da, da_tl_f)
    ax_da.set_ylabel("DA\n(Attn GPU)", fontsize=9, fontweight="bold")
    ax_da.set_yticks([]); ax_da.set_ylim(-28, 28)
    ax_da.grid(axis="x", alpha=0.3, linewidth=0.3)
    da_handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[ss],
                   label=SUB_STAGE_LABELS_DA.get(ss, ss))
                  for ss in ["prep_attn", "attn", "prep_mlp", "postprocess"]]
    ax_da.legend(handles=da_handles, loc="upper right", fontsize=6, ncol=4,
                 framealpha=0.85, edgecolor="gray")

    draw_bars(ax_df, df_tl_f)
    ax_df.set_ylabel("DF\n(FFN GPU)", fontsize=9, fontweight="bold")
    ax_df.set_yticks([]); ax_df.set_ylim(-28, 28)
    ax_df.set_xlabel("Wall-Clock Time (ms, relative to iteration start)", fontsize=8)
    ax_df.grid(axis="x", alpha=0.3, linewidth=0.3)
    df_handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[ss],
                   label=SUB_STAGE_LABELS_DF.get(ss, ss))
                  for ss in ["prep_attn", "attn", "prep_mlp", "mlp", "postprocess"]]
    ax_df.legend(handles=df_handles, loc="upper right", fontsize=6, ncol=3,
                 framealpha=0.85, edgecolor="gray")

    # Comm arrows
    ax_comm.set_ylabel("Comm", fontsize=9, fontweight="bold")
    ax_comm.set_yticks([]); ax_comm.set_ylim(-30, 30)
    ax_comm.grid(axis="x", alpha=0.3, linewidth=0.3)
    da_send_f = {k: v for k, v in da_send.items() if k[0] < max_layers}
    df_rstart_f = {k: v for k, v in df_rstart.items() if k[0] < max_layers}
    df_recv_f = {k: v for k, v in df_recv.items() if k[0] < max_layers}
    da_recv_f = {k: v for k, v in da_recv.items() if k[0] < max_layers}
    all_keys = sorted(set(da_send_f.keys()) | set(df_rstart_f.keys())
                      | set(df_recv_f.keys()) | set(da_recv_f.keys()))
    for key in all_keys:
        lid, mb = key
        if key in da_send_f and key in df_rstart_f:
            c = COMM_COLORS["DA_to_DF"]
            ft, tt = da_send_f[key], df_rstart_f[key]
            if abs(ft - tt) > 0.01:
                t0, t1 = (ft, tt) if ft <= tt else (tt, ft)
                ax_comm.annotate("", xy=(t1, 12), xytext=(t0, 12),
                    arrowprops=dict(arrowstyle="->", color=c, linewidth=3.0,
                                    connectionstyle="arc3,rad=0.2"))
        if key in df_recv_f and key in da_recv_f:
            c = COMM_COLORS["DF_to_DA"]
            ft, tt = df_recv_f[key], da_recv_f[key]
            if abs(ft - tt) > 0.01:
                t0, t1 = (ft, tt) if ft <= tt else (tt, ft)
                ax_comm.annotate("", xy=(t1, -12), xytext=(t0, -12),
                    arrowprops=dict(arrowstyle="->", color=c, linewidth=3.0,
                                    connectionstyle="arc3,rad=-0.2"))
    ax_comm.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color=COMM_COLORS["DA_to_DF"],
                      label="DA→DF (attn→ffn hidden_states)"),
        plt.Rectangle((0, 0), 1, 1, color=COMM_COLORS["DF_to_DA"],
                      label="DF→DA (ffn→attn result)"),
    ], loc="upper right", fontsize=6, ncol=2, framealpha=0.85, edgecolor="gray")

    for ax in [ax_da, ax_comm, ax_df]:
        ax.set_xlim(0, t_max)

    # ── Bottom-left: ZMQ latency over iterations ──────────────────────────
    ax_zmq = fig.add_subplot(gs[3, 0])
    if zmq_data:
        iters = [z["iter"] for z in zmq_data]
        zmq_lats = [z["zmq_lat"] * 1000 for z in zmq_data]  # convert to μs
        colors_zmq = ["#e74c3c" if lat > 1000 else "#2ecc71" for lat in zmq_lats]
        ax_zmq.bar(iters, zmq_lats, color=colors_zmq, width=0.8, edgecolor="white", linewidth=0.3)
        ax_zmq.axhline(y=np.mean(zmq_lats), color="red", linestyle="--", linewidth=1,
                       label=f"Mean: {np.mean(zmq_lats):.0f} μs")
        ax_zmq.set_xlabel("Iteration", fontsize=8)
        ax_zmq.set_ylabel("ZMQ Latency (μs)", fontsize=8)
        ax_zmq.set_title("Scheduler ZMQ: DA send → DF recv", fontsize=9, fontweight="bold")
        ax_zmq.legend(fontsize=6)
        ax_zmq.grid(axis="y", alpha=0.3, linewidth=0.3)
        ax_zmq.tick_params(labelsize=7)

    # ── Bottom-right: DA→DF UCX gap distribution ──────────────────────────
    ax_gap = fig.add_subplot(gs[3, 1])
    if pairs:
        # Filter outliers for histogram (>10ms are cold-start)
        gaps = [p["gap_ms"] for p in pairs if p["gap_ms"] < 10]
        n_bins = 80
        ax_gap.hist(gaps, bins=n_bins, color="#3498db", edgecolor="white", alpha=0.85, linewidth=0.3)
        p50 = np.percentile(gaps, 50)
        p90 = np.percentile(gaps, 90)
        p99 = np.percentile(gaps, 99)
        for pct, val, c in [("P50", p50, "#e74c3c"), ("P90", p90, "#f39c12"), ("P99", p99, "#8e44ad")]:
            ax_gap.axvline(x=val, color=c, linestyle="--", linewidth=1.2,
                          label=f"{pct}: {val:.3f} ms")
        ax_gap.set_xlabel("DA→DF Gap (ms)", fontsize=8)
        ax_gap.set_ylabel("Count", fontsize=8)
        ax_gap.set_title(f"DA→DF Transfer Gap Distribution\n(n={len(gaps)}, excl. cold-start >10ms)",
                         fontsize=9, fontweight="bold")
        ax_gap.legend(fontsize=6, loc="upper right")
        ax_gap.grid(axis="y", alpha=0.3, linewidth=0.3)
        ax_gap.tick_params(labelsize=7)

    # ── Row 5 left: UCX daemon thread breakdown ──────────────────────────
    ax_ucx = fig.add_subplot(gs[4, 0])
    if ucx_sends:
        # Average stacked bar
        avg_cuda = np.mean([u["cuda_us"] for u in ucx_sends])
        avg_ucx = np.mean([u["send_us"] for u in ucx_sends])
        bars = ax_ucx.barh(0, avg_cuda, color="#e74c3c", edgecolor="white",
                           linewidth=0.5, label=f"CUDA fence: {avg_cuda:.0f} μs")
        ax_ucx.barh(0, avg_ucx, left=avg_cuda, color="#f39c12", edgecolor="white",
                    linewidth=0.5, label=f"UCX send_tensor: {avg_ucx:.0f} μs")
        ax_ucx.set_yticks([])
        ax_ucx.set_xlabel("Time (μs)", fontsize=8)
        ax_ucx.set_title(f"UCX Send Daemon Thread (avg, n={len(ucx_sends)})",
                         fontsize=9, fontweight="bold")
        ax_ucx.legend(fontsize=7, loc="upper right")
        ax_ucx.grid(axis="x", alpha=0.3, linewidth=0.3)

    # ── Row 5 right: Steady-state gap per micro-batch ────────────────────
    ax_mb = fig.add_subplot(gs[4, 1])
    if pairs:
        steady_gaps = [p for p in pairs if p["gap_ms"] < 10]
        by_mb = defaultdict(list)
        for p in steady_gaps:
            by_mb[p["mb"]].append(p["gap_ms"])
        mb_keys = sorted(by_mb.keys())
        mb_avgs = [np.mean(by_mb[k]) for k in mb_keys]
        mb_stds = [np.std(by_mb[k]) for k in mb_keys]
        colors_mb = ["#2e86c1", "#27ae60", "#e74c3c"][:len(mb_keys)]
        bars = ax_mb.bar(range(len(mb_keys)), mb_avgs, yerr=mb_stds,
                        color=colors_mb, edgecolor="white", linewidth=0.5, capsize=4)
        ax_mb.set_xticks(range(len(mb_keys)))
        ax_mb.set_xticklabels([f"MB={k}" for k in mb_keys], fontsize=8)
        ax_mb.set_ylabel("Avg Gap (ms)", fontsize=8)
        ax_mb.set_title("Per-Microbatch DA→DF Gap (steady state)",
                        fontsize=9, fontweight="bold")
        ax_mb.grid(axis="y", alpha=0.3, linewidth=0.3)
        ax_mb.tick_params(labelsize=7)
        for bar, avg in zip(bars, mb_avgs):
            ax_mb.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                      f"{avg:.3f} ms", ha="center", fontsize=7, fontweight="bold")

    # ── Suptitle ──────────────────────────────────────────────────────────
    m_label = variant.upper().replace("M", "M=")
    fig.suptitle(
        f"PD+AF {m_label} D-Stage Pipeline with Breakdown  |  Layers 0-{max_layers - 1}  |  QPS=2  |  Wall-Clock Aligned",
        fontsize=11, fontweight="bold",
    )

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=MAX_LAYERS)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--variant", type=str, default="m3", choices=["m1", "m3"])
    parser.add_argument("--log-prefix", type=str, default="quick_m3",
                        help="Log file prefix (e.g., 'quick_m3' or 'qps_sweep_m3')")
    args = parser.parse_args()

    log_dir = os.path.join(os.path.dirname(__file__), "throughput_logs")
    da_log = os.path.join(log_dir, f"{args.log_prefix}_da.log")
    df_log = os.path.join(log_dir, f"{args.log_prefix}_df.log")

    if not os.path.exists(da_log):
        print(f"ERROR: {da_log} not found")
        sys.exit(1)
    if not os.path.exists(df_log):
        print(f"ERROR: {df_log} not found")
        sys.exit(1)

    # Load AFD_PER_STEP for Gantt
    with open(da_log) as f:
        da_perstep_lines = [l for l in f if "[AFD_PER_STEP]" in l]
    with open(df_log) as f:
        df_perstep_lines = [l for l in f if "[AFD_PER_STEP]" in l]

    def best_match(lines):
        best_line, best_sum = None, -1
        for line in lines:
            steps = parse_per_step(line)
            bs_sum = sum(s["batch_size"] for s in steps if s["stage"] == "A")
            if bs_sum > best_sum:
                best_sum = bs_sum
                best_line = line
        return parse_per_step(best_line), best_sum

    da_steps, da_sum = best_match(da_perstep_lines)
    df_steps, df_sum = best_match(df_perstep_lines)

    # Load AFD_HOST_EVENTS + AFD_SCHED_TS for breakdown
    da_events, da_sched = extract_host_events(da_log)
    df_events, df_sched = extract_host_events(df_log)

    print(f"DA: AFD_PER_STEP={len(da_perstep_lines)} lines, chosen bs_sum={da_sum}")
    print(f"DF: AFD_PER_STEP={len(df_perstep_lines)} lines, chosen bs_sum={df_sum}")
    print(f"DA: HOST_EVENTS={len(da_events)}, SCHED_TS={len(da_sched)}")
    print(f"DF: HOST_EVENTS={len(df_events)}, SCHED_TS={len(df_sched)}")

    use_wc = has_wall_clock(da_steps)
    if not use_wc:
        print("ERROR: Wall-clock timestamps required. Use latest afd.py.")
        sys.exit(1)

    save_path = args.save or os.path.join(
        os.path.dirname(__file__),
        f"dstage_gantt_breakdown_{args.variant}_l{args.layers}.png",
    )
    plot_combined(da_steps, df_steps, da_events, df_events, da_sched, df_sched,
                  max_layers=args.layers, save_path=save_path, variant=args.variant)


if __name__ == "__main__":
    main()
