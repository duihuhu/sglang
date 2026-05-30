#!/usr/bin/env python3
"""Plot fixed-QPS 3-mode comparison per il/ol group.

Auto-discovers all il<I>_ol<O>_qps<N>_<mode>_results.json files under
results/fixed_qps and produces one 6-panel figure per il/ol group
(energy / power / throughput / TTFT / TPOT / SLO vs QPS).
"""

import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE / "results" / "fixed_qps"
JSON_DIR = RES / "json"
FIG_DIR = RES / "figures"

MODES = ["tier1_freq", "max_freq", "auto_freq"]
LABELS = {
    "tier1_freq": "Tier1+Tier2 (freq-only)",
    "max_freq": "No-Tier max freq (1410MHz)",
    "auto_freq": "No-Tier auto freq",
}
COLORS = {"tier1_freq": "#2ca02c", "max_freq": "#d62728", "auto_freq": "#1f77b4"}
MARKERS = {"tier1_freq": "o", "max_freq": "s", "auto_freq": "^"}

PANELS = [
    ("total_energy_j", "Total Energy (J)", "Total GPU Energy"),
    ("prefill_energy_j", "Prefill Energy (J)", "Prefill-stage Energy (GPU6,7)"),
    ("decode_energy_j", "Decode Energy (J)", "Decode-stage Energy (GPU4,5)"),
    ("throughput_tok_s", "Throughput (tok/s)", "Output Throughput"),
    ("ttft_avg_ms", "TTFT avg (ms)", "Time-To-First-Token"),
    ("tpot_avg_ms", "TPOT avg (ms)", "Time-Per-Output-Token"),
    ("ttft_violation_rate", "TTFT violation (%)", "TTFT SLO Violation (>5000ms)"),
    ("tpot_violation_rate", "TPOT violation (%)", "TPOT SLO Violation (>300ms)"),
]


def discover():
    # data[group][mode][qps] = result dict
    data = defaultdict(lambda: defaultdict(dict))
    for f in glob.glob(str(JSON_DIR / "il*_results.json")):
        m = re.match(r"(il\d+_ol\d+)_qps([0-9p]+)_(tier1_freq|max_freq|auto_freq)_results\.json",
                     Path(f).name)
        if not m:
            continue
        group, qps, mode = m.group(1), float(m.group(2).replace("p", ".")), m.group(3)
        d = json.load(open(f))
        # Derive per-cause violation rates (TTFT vs TPOT). Denominator matches
        # the total-requests basis used for slo_violation_rate.
        tot = d.get("total_requests", 0) or 0
        if tot > 0:
            d["ttft_violation_rate"] = round(100.0 * d.get("ttft_violations", 0) / tot, 2)
            d["tpot_violation_rate"] = round(100.0 * d.get("tpot_violations", 0) / tot, 2)
        else:
            d["ttft_violation_rate"] = 0.0
            d["tpot_violation_rate"] = 0.0
        data[group][mode][qps] = d
    return data


def plot_group(group, gdata):
    qps_all = sorted({q for mode in gdata.values() for q in mode})
    fig, axes = plt.subplots(2, 4, figsize=(21, 9))
    fig.suptitle(
        f"Fixed-length {group} — 3 frequency-control policies\n"
        "Qwen3-32B, PD+AF, A800x4 (prefill=GPU6,7 / decode=GPU4,5)",
        fontsize=14, fontweight="bold",
    )
    for ax, (key, ylabel, title) in zip(axes.flat, PANELS):
        for mode in MODES:
            if mode not in gdata:
                continue
            xs = [q for q in qps_all if q in gdata[mode] and key in gdata[mode][q]]
            ys = [gdata[mode][q][key] for q in xs]
            if not xs:
                continue
            ax.plot(xs, ys, marker=MARKERS[mode], color=COLORS[mode],
                    label=LABELS[mode], linewidth=2, markersize=7)
        ax.set_xlabel("QPS (req/s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(qps_all)

    # Annotate energy-saving % vs max_freq on each energy panel (total/P/D).
    if "max_freq" in gdata and "tier1_freq" in gdata:
        energy_panels = {
            "total_energy_j": axes.flat[0],
            "prefill_energy_j": axes.flat[1],
            "decode_energy_j": axes.flat[2],
        }
        for ekey, ax_e in energy_panels.items():
            for q in qps_all:
                mx = gdata["max_freq"].get(q, {})
                t1 = gdata["tier1_freq"].get(q, {})
                if ekey in mx and ekey in t1 and mx[ekey] > 0:
                    save = (mx[ekey] - t1[ekey]) / mx[ekey] * 100
                    ax_e.annotate(f"-{save:.0f}%", (q, t1[ekey]),
                                  textcoords="offset points", xytext=(0, -14),
                                  ha="center", fontsize=8,
                                  color=COLORS["tier1_freq"])
    axes.flat[0].legend(loc="upper left", fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIG_DIR / f"compare_{group}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = discover()
    for group in sorted(data):
        plot_group(group, data[group])


if __name__ == "__main__":
    main()
