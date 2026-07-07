#!/usr/bin/env python3
"""Plot DVFS decision frequency timeline from biscale_het_*.jsonl logs."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CHARTS = HERE / "charts"
LOG_DIR = HERE.parent.parent / "logs"
BENCH_LOG = LOG_DIR / "biscale_pd_hetero_code.log"

INST_MAP = {
    "biscale_het_p0_gpu0": "P0",
    "biscale_het_p1_gpu2": "P1",
    "biscale_het_p2_gpu4": "P2",
    "biscale_het_p3_gpu6": "P3",
    "biscale_het_d0_gpu0": "D0",
    "biscale_het_d1_gpu4": "D1",
}


def load_decisions(log_dir: Path) -> dict[str, list[tuple[float, int, str]]]:
    """inst -> [(t_rel, sel_f, phase)]"""
    raw: dict[str, list[tuple[float, int, str]]] = defaultdict(list)
    t0 = None
    for path in sorted(log_dir.glob("biscale_het_*.jsonl")):
        inst = INST_MAP.get(path.stem)
        if not inst:
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            t = row.get("t")
            f = row.get("sel_f")
            if t is None or f is None:
                continue
            if t0 is None:
                t0 = t
            raw[inst].append((t - t0, int(f), row.get("phase", "")))
    if t0 is None:
        return {}
    return dict(raw)


def parse_workload_spans(bench_log: Path, t0_unix: float) -> list[tuple[str, float, float]]:
    """Return (workload, start_rel, end_rel) from benchmark log."""
    if not bench_log.exists():
        return []
    wl_re = re.compile(r"code_qps\d+")
    spans: list[tuple[str, float, float]] = []
    current = None
    start = None
    for line in bench_log.read_text().splitlines():
        if "code_qps" in line and "run_window" in line:
            m = wl_re.search(line)
            if m:
                if current and start is not None:
                    spans.append((current, start, None))
                current = m.group(0)
                ts_m = re.match(r"(\d{2}:\d{2}:\d{2})", line)
                if ts_m:
                    h, mi, s = map(int, ts_m.group(1).split(":"))
                    start = h * 3600 + mi * 60 + s
        elif current and "Thpt=" in line:
            ts_m = re.match(r"(\d{2}:\d{2}:\d{2})", line)
            if ts_m and start is not None:
                h, mi, s = map(int, ts_m.group(1).split(":"))
                end = h * 3600 + mi * 60 + s
                spans.append((current, start, end))
                current = None
                start = None
    if spans and spans[0][1] is not None:
        base = spans[0][1]
        return [(w, s - base, (e - base) if e else 0) for w, s, e in spans]
    return []


def plot_timeline(by_inst: dict, spans: list, out: Path, title: str):
    if not by_inst:
        print("no decisions")
        return
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax, phase, label in [(axes[0], "P", "Prefill (4×TP2)"),
                             (axes[1], "D", "Decode (2×TP4)")]:
        insts = sorted(k for k in by_inst if k.startswith(phase))
        for inst in insts:
            pts = by_inst[inst]
            xs, ys = zip(*[(t, f) for t, f, _ in pts])
            ax.step(xs, ys, where="post", linewidth=1.4, label=inst)
        for wl, s, e in spans:
            if e > s:
                ax.axvspan(s, e, alpha=0.08, color="gray")
        ax.set_ylabel("Selected freq (MHz)")
        ax.set_title(label)
        ax.set_ylim(0, 1500)
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=4, fontsize=8, loc="upper right")
    axes[1].set_xlabel("Time since first DVFS decision (s)")
    fig.suptitle(title, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def plot_by_workload(by_inst: dict, spans: list, out: Path):
    if not spans:
        plot_timeline(by_inst, spans, out, "BiScale hetero — DVFS decisions")
        return
    n = len(spans)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.8 * n), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, (wl, s, e) in zip(axes, spans):
        for inst in sorted(by_inst):
            pts = [(t, f) for t, f, _ in by_inst[inst] if s <= t <= e]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.step([x - s for x in xs], ys, where="post", linewidth=1.2, label=inst)
        ax.set_title(wl)
        ax.set_ylabel("MHz")
        ax.set_ylim(0, 1500)
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=6, fontsize=6, loc="upper right")
    axes[-1].set_xlabel("Time within workload (s)")
    fig.suptitle("BiScale P4×TP2+D2×TP4 — freq by QPS (DVFS decisions)", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def print_summary(by_inst: dict):
    print("Per-instance freq summary (DVFS sel_f):")
    for inst in sorted(by_inst):
        freqs = [f for _, f, _ in by_inst[inst]]
        uniq = sorted(set(freqs))
        print(f"  {inst}: n={len(freqs)} min={min(freqs)} max={max(freqs)} "
              f"mean={sum(freqs)/len(freqs):.0f} unique={len(uniq)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path,
                        default=LOG_DIR / "dvfs_decisions")
    parser.add_argument("--bench-log", type=Path, default=BENCH_LOG)
    parser.add_argument("--out-dir", type=Path, default=CHARTS)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    by_inst = load_decisions(args.log_dir)
    print_summary(by_inst)
    spans = parse_workload_spans(args.bench_log, 0)
    plot_timeline(
        by_inst, spans,
        args.out_dir / "biscale_het_code_freq_timeline.png",
        "BiScale P4×TP2+D2×TP4 — DVFS frequency timeline (code)",
    )
    plot_by_workload(
        by_inst, spans,
        args.out_dir / "biscale_het_code_freq_by_qps.png",
    )


if __name__ == "__main__":
    main()
