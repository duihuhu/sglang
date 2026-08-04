#!/usr/bin/env python3
"""Plot GPU frequency timelines for 3 tier schemes (DynamoLLM, BiScale, AFlex).

Uses the LATEST frequency data referenced by plan_dense_e2e.json:
- DynamoLLM: results/freq_timelines/native_tp1_tier/
- BiScale:   results/freq_timelines/pd_hetero_tier_biscale/
- AFlex:     results/freq_timelines/aflex_q{N}/ (per QPS)

Output: freq_timeline/{code,conv}_qps{2,4,8,16}_freq_timeline.png
        freq_timeline/{code,conv}_combined_freq_timeline.png
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.ticker import FixedLocator
import numpy as np

# Typography aligned with macro/charts/plot_e2e_dashboard.py.
MICRO_REFERENCE = {
    "fig_width": 9.0,
    "font_size": 14,
    "panel_title": 15,
    "legend": 13,
}
MACRO_FIG_WIDTH = 13.0
PAPER_FONT_SCALE = MACRO_FIG_WIDTH / MICRO_REFERENCE["fig_width"]
FONT_SIZE = MICRO_REFERENCE["font_size"] * PAPER_FONT_SCALE
PANEL_TITLE_FONTSIZE = MICRO_REFERENCE["panel_title"] * PAPER_FONT_SCALE
LEGEND_FONT_SIZE = MICRO_REFERENCE["legend"] * PAPER_FONT_SCALE

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "figure.dpi": 150,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

HERE = Path(__file__).resolve().parent
FREQ_DIR = HERE.parent / "data" / "freq_timelines"
E2E_JSON = HERE.parent / "data" / "plan_dense_e2e.json"
OUT_DIR = HERE

NODE1 = "10.252.129.34"  # Prefill node
NODE2 = "10.252.129.33"  # Decode node

QPS_LIST = [2, 4, 8, 16]
BIN_S = 0.25
SEGMENT_DURATION_S = 25.0
Y_LABEL = "GPU Frequency (MHz)"
Y_TICKS = [300, 600, 900, 1200, 1500]
PANEL_SUBTITLE_Y = -0.22
PANEL_A_SUBTITLE_Y = -0.16
PANEL_BCDE_SUBTITLE_Y = -0.18
PANEL_A_XLABEL_PAD = 8
LEGEND_LINE_WIDTH = 3.5
LEGEND_PAD = 0.004
LEGEND_COLUMNSPACING = 1.5
LEGEND_HANDLETEXTPAD = 0.6
LEGEND_HANDLELENGTH = 1.8
FIG_TOP = 0.96
SAVE_PAD_INCHES = 0.015
QPS_LABEL_FONTSIZE = FONT_SIZE * 0.85
X_TICKS = [0, 25, 50, 75, 100]
COMBINED_FIG_SIZE = (MACRO_FIG_WIDTH, 10.0)
OUTER_HSPACE = 0.22
BOTTOM_HSPACE = 0.42
SUPYLABEL_X = 0.008

# Load e2e config for AFlex directory mapping
_e2e_data = None

def _get_e2e_data():
    global _e2e_data
    if _e2e_data is None:
        _e2e_data = json.loads(E2E_JSON.read_text())
    return _e2e_data


def _aflex_dir_for(dataset: str, qps: int) -> str:
    """Get the correct AFlex freq_timeline directory name for a given dataset/qps."""
    e2e = _get_e2e_data()
    key = f"{dataset}_qps{qps}"
    aflex = e2e["results"]["aflex_tier1"]
    if key in aflex and "config" in aflex[key]:
        name = aflex[key]["config"]["name"]
        # Check if directory actually exists, fallback to aflex_q{qps}
        if (FREQ_DIR / name / f"{dataset}_qps{qps}.json").exists():
            return name
    # Fallback: standard naming
    fallback = f"aflex_q{qps}"
    if (FREQ_DIR / fallback / f"{dataset}_qps{qps}.json").exists():
        return fallback
    # Return config name anyway (will trigger FileNotFoundError later)
    if key in aflex and "config" in aflex[key]:
        return aflex[key]["config"]["name"]
    return fallback


def _aflex_config_for(dataset: str, qps: int) -> dict:
    """Get the AFlex config (tp_pa, tp_pf, etc.) for a given dataset/qps."""
    e2e = _get_e2e_data()
    key = f"{dataset}_qps{qps}"
    aflex = e2e["results"]["aflex_tier1"]
    if key in aflex and "config" in aflex[key]:
        return aflex[key]["config"]
    return {"tp_pa": 2, "tp_pf": 2, "tp_da": 1, "tp_df": 1}


def load_timeline(scheme: str, dataset: str, qps: int) -> tuple[list[dict], float]:
    """Load frequency timeline data for a given scheme/dataset/qps."""
    if scheme == "aflex":
        dir_name = _aflex_dir_for(dataset, qps)
        path = FREQ_DIR / dir_name / f"{dataset}_qps{qps}.json"
    else:
        path = FREQ_DIR / scheme / f"{dataset}_qps{qps}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    samples = payload["nvidia_smi_timeline"]
    if scheme == "aflex" and (role_mapping := payload.get("role_mapping")):
        role_by_gpu = {}
        for role, role_data in role_mapping.items():
            placements = (
                role_data.get("placements", [])
                if isinstance(role_data, dict)
                else role_data
            )
            for placement in placements:
                node = placement["node"]
                for gpu in placement["gpus"]:
                    role_by_gpu[(node, gpu)] = role
        for sample in samples:
            role = role_by_gpu.get((sample["node"], sample["gpu"]))
            if role is not None:
                sample["_aflex_role"] = role
    duration = float(payload.get("duration_s", 0))
    return samples, duration


def bin_series(
    samples: list[dict], groups: dict[str, callable], bin_s: float = BIN_S,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Bin samples into time series per group."""
    if not samples:
        return {}
    t0 = min(s["t"] for s in samples)
    acc: dict[str, dict[int, list[int]]] = {g: defaultdict(list) for g in groups}
    for s in samples:
        b = int((s["t"] - t0) / bin_s)
        for gname, pred in groups.items():
            if pred(s):
                acc[gname][b].append(s["freq_mhz"])

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    max_bin = max(b for g in acc.values() for b in g) if any(acc[g] for g in acc) else 0
    xs = np.arange(max_bin + 1) * bin_s
    for gname, bins in acc.items():
        ys = []
        for b in range(max_bin + 1):
            vals = bins.get(b, [])
            ys.append(float(np.mean(vals)) if vals else np.nan)
        ys_arr = np.array(ys, dtype=float)
        mask = ~np.isnan(ys_arr)
        if mask.any():
            last = ys_arr[mask][0]
            for i in range(len(ys_arr)):
                if np.isnan(ys_arr[i]):
                    ys_arr[i] = last
                else:
                    last = ys_arr[i]
        out[gname] = (xs, ys_arr)
    return out


SCHEMES = [
    ("native_tp1_tier", "DynamoLLM", "#aec7e8"),
    ("pd_hetero_tier_biscale", "BiScale", "#ffbb78"),
    ("aflex", "AFlex", "#d62728"),
]


def _aflex_role_groups(dataset: str, qps: int) -> dict:
    """Build PA/PF/DA/DF filter predicates based on AFlex config.

    Layout rule:
      Each prefill worker uses tp_pa + tp_pf GPUs: [PA*tp_pa, PF*tp_pf]
      Each decode worker uses tp_da + tp_df GPUs:  [DA*tp_da, DF*tp_df]
      Workers fill GPUs sequentially: Node1 first, then Node2.
      k_p prefill workers + k_d decode workers = total GPUs used.
    """
    cfg = _aflex_config_for(dataset, qps)
    tp_pa = cfg.get("tp_pa", 2)
    tp_pf = cfg.get("tp_pf", 2)
    tp_da = cfg.get("tp_da", 1)
    tp_df = cfg.get("tp_df", 1)
    k_p = cfg.get("k_p", 1)
    k_d = cfg.get("k_d", 1)

    gpus_per_p = tp_pa + tp_pf
    gpus_per_d = tp_da + tp_df
    total_gpus = gpus_per_p * k_p + gpus_per_d * k_d

    # Build flat GPU list: (node, gpu_id) in assignment order
    all_gpu_slots = []
    nodes = [NODE1, NODE2]
    for node in nodes:
        for g in range(8):
            all_gpu_slots.append((node, g))

    # Assign roles sequentially
    pa_set = set()
    pf_set = set()
    da_set = set()
    df_set = set()

    idx = 0
    for _ in range(k_p):
        for i in range(tp_pa):
            if idx < len(all_gpu_slots):
                pa_set.add(all_gpu_slots[idx])
                idx += 1
        for i in range(tp_pf):
            if idx < len(all_gpu_slots):
                pf_set.add(all_gpu_slots[idx])
                idx += 1
    for _ in range(k_d):
        for i in range(tp_da):
            if idx < len(all_gpu_slots):
                da_set.add(all_gpu_slots[idx])
                idx += 1
        for i in range(tp_df):
            if idx < len(all_gpu_slots):
                df_set.add(all_gpu_slots[idx])
                idx += 1

    def role_predicate(role: str, fallback_gpus: set[tuple[str, int]]):
        def matches(sample: dict) -> bool:
            if "_aflex_role" in sample:
                return sample["_aflex_role"] == role
            return (sample["node"], sample["gpu"]) in fallback_gpus

        return matches

    return {
        "PA": role_predicate("PA", pa_set),
        "PF": role_predicate("PF", pf_set),
        "DA": role_predicate("DA", da_set),
        "DF": role_predicate("DF", df_set),
    }


def _detect_active_aflex_nodes(samples: list[dict]) -> tuple[str, str]:
    """Detect which node is P and which is D from actual freq data."""
    from collections import Counter
    node_activity = defaultdict(list)
    for s in samples:
        node_activity[s["node"]].append(s["freq_mhz"])
    # Node with higher average freq is more likely Prefill (PA at high freq)
    avgs = {n: np.mean(f) for n, f in node_activity.items()}
    nodes_sorted = sorted(avgs.keys(), key=lambda n: avgs[n], reverse=True)
    if len(nodes_sorted) >= 2:
        return nodes_sorted[0], nodes_sorted[1]
    return NODE1, NODE2


def plot_freq_timeline(dataset: str, qps: int, out_path: Path) -> None:
    dataset_titles = {"code": "Code", "conv": "Conversation"}
    ds_title = dataset_titles.get(dataset, dataset)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"GPU Frequency Timeline — {ds_title} Trace (QPS={qps})"
        " | 16-GPU 2-Node | Qwen3-32B",
        fontsize=13, fontweight="bold", y=0.98,
    )

    # --- Panel 1: All GPUs avg for all 3 schemes ---
    ax = axes[0]
    for scheme, label, color in SCHEMES:
        try:
            samples, duration = load_timeline(scheme, dataset, qps)
        except FileNotFoundError:
            continue
        groups = {"All GPUs": lambda s: True}
        series = bin_series(samples, groups)
        if "All GPUs" in series:
            xs, ys = series["All GPUs"]
            ax.step(xs, ys, where="post", label=f"{label} ({duration:.0f}s)",
                    color=color, linestyle="-", linewidth=2.0)
    ax.set_ylabel("SM Clock (MHz)")
    ax.set_title("All GPUs Average — 3 Schemes Comparison",
                 fontweight="bold", loc="left")
    ax.set_ylim(0, 1500)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")

    # --- Panel 2: BiScale P/D vs AFlex P/D ---
    ax = axes[1]
    pd_styles = {
        "BiScale Prefill": ("#ff7f0e", "-", 1.8),
        "BiScale Decode": ("#ffbb78", "-", 1.8),
        "AFlex Prefill (PA+PF)": ("#e377c2", "--", 1.8),
        "AFlex Decode (DA+DF)": ("#c5b0d5", "--", 1.8),
    }
    # BiScale P/D
    try:
        samples_bs, _ = load_timeline("pd_hetero_tier_biscale", dataset, qps)
        groups_bs = {
            "BiScale Prefill": lambda s: s["node"] == NODE1,
            "BiScale Decode": lambda s: s["node"] == NODE2,
        }
        series_bs = bin_series(samples_bs, groups_bs)
        for gname, (xs, ys) in series_bs.items():
            c, ls, lw = pd_styles[gname]
            ax.step(xs, ys, where="post", label=gname, color=c,
                    linestyle=ls, linewidth=lw)
    except FileNotFoundError:
        pass
    # AFlex P/D
    try:
        samples_af, _ = load_timeline("aflex", dataset, qps)
        role_groups = _aflex_role_groups(dataset, qps)
        groups_af_pd = {
            "AFlex Prefill (PA+PF)": lambda s: (
                role_groups["PA"](s) or role_groups["PF"](s)
            ),
            "AFlex Decode (DA+DF)": lambda s: (
                role_groups["DA"](s) or role_groups["DF"](s)
            ),
        }
        series_af = bin_series(samples_af, groups_af_pd)
        for gname, (xs, ys) in series_af.items():
            c, ls, lw = pd_styles[gname]
            ax.step(xs, ys, where="post", label=gname, color=c,
                    linestyle=ls, linewidth=lw)
    except FileNotFoundError:
        pass
    ax.set_ylabel("SM Clock (MHz)")
    ax.set_title("Prefill vs Decode — BiScale & AFlex",
                 fontweight="bold", loc="left")
    ax.set_ylim(0, 1500)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right", ncol=2)

    # --- Panel 3: AFlex PA/PF/DA/DF ---
    ax = axes[2]
    try:
        samples_af, _ = load_timeline("aflex", dataset, qps)
        role_groups = _aflex_role_groups(dataset, qps)
        af4_colors = {"PA": "#d62728", "PF": "#ff7f0e", "DA": "#17becf", "DF": "#2ca02c"}
        series_af4 = bin_series(samples_af, role_groups)
        for gname in ["PA", "PF", "DA", "DF"]:
            if gname in series_af4:
                xs, ys = series_af4[gname]
                ax.step(xs, ys, where="post", label=gname,
                        color=af4_colors[gname], linestyle="-", linewidth=1.6)
    except FileNotFoundError:
        ax.text(0.5, 0.5, "No AFlex data", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_ylabel("SM Clock (MHz)")
    ax.set_title("AFlex — PA / PF / DA / DF Breakdown",
                 fontweight="bold", loc="left")
    ax.set_ylim(0, 1500)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right", ncol=4)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _clip_series(samples: list[dict], duration_s: float) -> list[dict]:
    if not samples:
        return samples
    t0 = min(s["t"] for s in samples)
    return [s for s in samples if s["t"] - t0 <= duration_s]


def _offset_samples(samples: list[dict], offset: float) -> list[dict]:
    if not samples:
        return samples
    t0 = min(s["t"] for s in samples)
    return [{**s, "t": s["t"] - t0 + offset} for s in samples]


def _subtitle_below_xlabel(ax, text: str, y: float = PANEL_SUBTITLE_Y) -> None:
    ax.text(
        0.5,
        y,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=PANEL_TITLE_FONTSIZE,
    )


def _style_combined_axis(ax) -> None:
    ax.set_ylim(0, 1500)
    ax.yaxis.set_major_locator(FixedLocator(Y_TICKS))


def plot_freq_timeline_combined(dataset: str, out_path: Path) -> None:
    """Plot combined timeline from pre-computed freq_timeline_data.json."""
    data_file = HERE / "freq_timeline_data.json"
    if not data_file.exists():
        print(f"WARN: {data_file} not found, skipping combined plot.")
        return
    data = json.loads(data_file.read_text())
    meta = data["meta"]

    total_bins = meta["total_bins"]
    bin_size = meta["bin_size_s"]
    segment_dur = meta["segment_duration_s"]
    qps_segments = meta["qps_segments"]

    xs = [i * bin_size + bin_size / 2 for i in range(total_bins)]

    fig = plt.figure(figsize=COMBINED_FIG_SIZE)
    gs_outer = fig.add_gridspec(
        2, 1,
        height_ratios=[1.0, 2.0],
        hspace=OUTER_HSPACE,
        top=FIG_TOP,
        bottom=0.10,
        left=0.10,
        right=0.98,
    )
    ax_top = fig.add_subplot(gs_outer[0])
    gs_bottom = gs_outer[1].subgridspec(2, 2, hspace=BOTTOM_HSPACE, wspace=0.18)
    axes = [
        ax_top,
        fig.add_subplot(gs_bottom[0, 0]),
        fig.add_subplot(gs_bottom[0, 1]),
        fig.add_subplot(gs_bottom[1, 0]),
        fig.add_subplot(gs_bottom[1, 1]),
    ]
    for ax in axes[1:]:
        ax.sharex(axes[0])

    # Segment boundaries for QPS labels
    segment_boundaries = []
    offset = 0.0
    for qps in qps_segments:
        segment_boundaries.append((offset, qps))
        offset += segment_dur

    def _draw_boundaries(ax):
        qps_label_y = 105  # center of unused 0–210 MHz band
        for boff, bqps in segment_boundaries:
            if boff > 0:
                ax.axvline(boff, color="#888888", linestyle=":", linewidth=1.0, alpha=0.7)
            mid = boff + segment_dur / 2
            ax.text(
                mid,
                qps_label_y,
                f"RPS={bqps}",
                ha="center",
                va="center",
                fontsize=QPS_LABEL_FONTSIZE,
                color="#555555",
            )

    def _mean_mhz(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    # --- Panel 1: All GPUs avg for 3 schemes ---
    ax = axes[0]
    scheme_colors = {"DynamoLLM": "#aec7e8", "BiScale": "#ffbb78", "AFlex": "#d62728"}
    panel1 = data["panel1_all_gpus_avg"]
    scheme_handles = []
    for scheme in ["DynamoLLM", "BiScale", "AFlex"]:
        ys = panel1.get(scheme, [])
        if ys:
            if scheme == "AFlex":
                avg = _mean_mhz(ys)
                ax.axhline(
                    avg,
                    color=scheme_colors[scheme],
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.75,
                )
            (line,) = ax.plot(
                xs[: len(ys)],
                ys,
                label=scheme,
                color=scheme_colors[scheme],
                linestyle="-",
                linewidth=2.0,
            )
            scheme_handles.append(line)
    ax.set_ylabel("")
    _style_combined_axis(ax)
    ax.grid(True, alpha=0.3)
    _draw_boundaries(ax)

    # --- Panels 2-5: one AFlex role per panel ---
    af4_colors = {
        "PA": "#2ca02c",
        "PF": "#9467bd",
        "DA": "#17becf",
        "DF": "#e377c2",
    }
    panel_labels = {"PA": "(b)", "PF": "(c)", "DA": "(d)", "DF": "(e)"}
    panel_keys = {"PA": "panel2_aflex_PA", "PF": "panel3_aflex_PF",
                  "DA": "panel4_aflex_DA", "DF": "panel5_aflex_DF"}
    role_handles = []
    for ax, role in zip(axes[1:], ["PA", "PF", "DA", "DF"]):
        ys = data.get(panel_keys[role], [])
        if ys:
            avg = _mean_mhz(ys)
            ax.axhline(
                avg,
                color=af4_colors[role],
                linestyle="--",
                linewidth=1.2,
                alpha=0.75,
            )
            (line,) = ax.plot(
                xs[: len(ys)],
                ys,
                label=role,
                color=af4_colors[role],
                linestyle="-",
                linewidth=1.6,
            )
            role_handles.append(line)
        else:
            ax.text(0.5, 0.5, f"No AFlex {role} data", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_ylabel("")
        _style_combined_axis(ax)
        ax.grid(True, alpha=0.3)
        _draw_boundaries(ax)
        _subtitle_below_xlabel(ax, f"{panel_labels[role]} AFlex — {role}", y=PANEL_BCDE_SUBTITLE_Y)

    for ax in axes:
        ax.set_xlim(0, offset)
        ax.set_xticks(X_TICKS)
        ax.set_xticklabels([f"{t}s" for t in X_TICKS])
    axes[0].set_xlabel("")
    for ax in axes[1:]:
        ax.set_xlabel("")
    _subtitle_below_xlabel(axes[0], "(a) DVFS-Enabled Systems", y=PANEL_A_SUBTITLE_Y)

    # Place y-label at the vertical center of each row (top & bottom)
    pos_top = axes[0].get_position()
    y_top_center = pos_top.y0 + pos_top.height / 2
    pos_bottom_rows = [ax.get_position() for ax in axes[1:]]
    y_bottom_center = sum(p.y0 + p.height / 2 for p in pos_bottom_rows) / len(pos_bottom_rows)
    for y_center in (y_top_center, y_bottom_center):
        fig.text(
            SUPYLABEL_X, y_center, Y_LABEL,
            fontsize=FONT_SIZE,
            va="center", ha="center", rotation="vertical",
        )

    all_handles = scheme_handles + role_handles
    all_labels = [h.get_label() for h in all_handles]
    fig.canvas.draw()
    pos_top = axes[0].get_position()
    legend_y = pos_top.y1 + LEGEND_PAD
    legend = fig.legend(
        all_handles, all_labels,
        fontsize=LEGEND_FONT_SIZE,
        loc="lower center",
        ncol=len(all_handles),
        frameon=False,
        columnspacing=LEGEND_COLUMNSPACING,
        handletextpad=LEGEND_HANDLETEXTPAD,
        handlelength=LEGEND_HANDLELENGTH,
        bbox_to_anchor=(0.5, legend_y),
        bbox_transform=fig.transFigure,
    )
    for line in legend.get_lines():
        line.set_linewidth(LEGEND_LINE_WIDTH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["code", "conv"], default=None)
    parser.add_argument("--combined-only", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = [args.dataset] if args.dataset else ["code", "conv"]
    for dataset in datasets:
        if not args.combined_only:
            for qps in QPS_LIST:
                out = OUT_DIR / f"{dataset}_qps{qps}_freq_timeline.pdf"
                try:
                    plot_freq_timeline(dataset, qps, out)
                except Exception as e:
                    print(f"WARN: {dataset}/qps{qps} failed: {e}")
        out = OUT_DIR / f"{dataset}_combined_freq_timeline.pdf"
        try:
            plot_freq_timeline_combined(dataset, out)
        except Exception as e:
            import traceback
            print(f"WARN: {dataset}/combined failed: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
