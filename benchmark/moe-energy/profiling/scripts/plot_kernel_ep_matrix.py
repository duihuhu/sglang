#!/usr/bin/env python3
"""Plot the final kernel-EP block-NVML matrix along its main dimensions."""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

PROFILE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROFILE_ROOT / "data" / "kernel-EP"
OUT_DIR = DATA_DIR / "pic"
ROUTINGS = ("balanced", "middle", "skewed")
COLORS = {"balanced": "#4C78A8", "middle": "#F2A541", "skewed": "#E45756"}
MARKERS = {"balanced": "o", "middle": "^", "skewed": "s"}
PHASES = ("PF", "DF")
PHASE_LABEL = {"PF": "Prefill", "DF": "Decode"}
LENGTH_COL = {"PF": "input_len", "DF": "context_len"}
BASE = {"ws": 8, "length": 512, "freq": 930, "batch": 32}
AXIS_LABEL = {
    "batch": "Batch size",
    "length": "Input/context length",
    "freq": "GPU clock (MHz)",
    "ws": "EP world size",
}
METRIC_LABEL = {
    "latency": "Max-rank local-wall latency (µs, barrier excluded)",
    "energy": "Cluster energy per call (mJ, block-level NVML)",
}

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 220,
    }
)


def load(phase: str, routing: str) -> list[dict]:
    rows = []
    with (DATA_DIR / f"{phase}-{routing}.txt").open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rows.append(
                {
                    "ws": int(row["size"]),
                    "length": int(row[LENGTH_COL[phase]]),
                    "freq": int(row["gpu_clock"]),
                    "batch": int(row["batch_size"]),
                    "latency": float(row["latency_us"]),
                    "energy": float(row["energy_mj"]),
                }
            )
    return rows


DATA = {(phase, routing): load(phase, routing) for phase in PHASES for routing in ROUTINGS}


def select(phase: str, routing: str, varying: str, ws: int | None = None) -> list[dict]:
    selected = []
    for row in DATA[(phase, routing)]:
        if ws is not None and row["ws"] != ws:
            continue
        if any(
            row[dim] != value
            for dim, value in BASE.items()
            if dim != varying and not (dim == "ws" and ws is not None)
        ):
            continue
        selected.append(row)
    return sorted(selected, key=lambda row: row[varying])


def save(fig, filename: str, caption: str) -> None:
    fig.text(0.5, 0.008, caption, ha="center", va="bottom", fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    path = OUT_DIR / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def plot_dimension(varying: str, metric: str) -> None:
    if varying == "ws":
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        for ax, phase in zip(axes, PHASES):
            for routing in ROUTINGS:
                rows = select(phase, routing, varying)
                ax.plot(
                    [row[varying] for row in rows],
                    [row[metric] for row in rows],
                    marker=MARKERS[routing], color=COLORS[routing], linewidth=2,
                    markersize=5, label=routing,
                )
            ax.set_title(PHASE_LABEL[phase])
            ax.set_xlabel(AXIS_LABEL[varying])
            ax.set_ylabel(METRIC_LABEL[metric])
            ax.set_xticks([2, 4, 8])
            ax.grid(True, alpha=0.25)
            ax.legend()
        fixed = "length=512, batch=32, clock=930 MHz"
        fig.suptitle(
            f"Kernel-EP routing comparison vs {AXIS_LABEL[varying]} — {METRIC_LABEL[metric]}\n"
            f"Fixed: {fixed}", fontsize=12,
        )
    else:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for row_index, phase in enumerate(PHASES):
            for col_index, ws in enumerate((2, 4, 8)):
                ax = axes[row_index, col_index]
                all_rows = []
                for routing in ROUTINGS:
                    rows = select(phase, routing, varying, ws=ws)
                    all_rows.extend(rows)
                    if rows:
                        ax.plot(
                            [row[varying] for row in rows],
                            [row[metric] for row in rows],
                            marker=MARKERS[routing], color=COLORS[routing], linewidth=1.8,
                            markersize=4.5, label=routing,
                        )
                ax.set_title(f"{PHASE_LABEL[phase]} · ws={ws}")
                ax.set_xlabel(AXIS_LABEL[varying])
                ax.set_ylabel(METRIC_LABEL[metric])
                if varying in ("batch", "length"):
                    ax.set_xscale("log", base=2)
                xs = sorted({row[varying] for row in all_rows})
                if xs:
                    ax.set_xticks(xs)
                    ax.xaxis.set_major_formatter(ScalarFormatter())
                ax.grid(True, alpha=0.25)
                ax.legend()
        fixed = [f"{dim}={value}" for dim, value in BASE.items() if dim not in (varying, "ws")]
        fixed = ["clock=930 MHz" if item == "freq=930" else item for item in fixed]
        fig.suptitle(
            f"Kernel-EP routing comparison vs {AXIS_LABEL[varying]} — {METRIC_LABEL[metric]}\n"
            f"Fixed per panel: {', '.join(fixed)}", fontsize=12,
        )
    save(
        fig, f"{metric}_vary_{varying}.png",
        "Source: final 1020-row block-NVML kernel-EP refresh · Qwen3-30B-A3B · local-wall excludes barrier",
    )


def plot_ratio_by_batch() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)
    for row_index, phase in enumerate(PHASES):
        for col_index, ws in enumerate((2, 4, 8)):
            ax = axes[row_index, col_index]
            balanced = {
                (row["length"], row["batch"], row["freq"]): row
                for row in DATA[(phase, "balanced")] if row["ws"] == ws
            }
            for routing in ("middle", "skewed"):
                other = {
                    (row["length"], row["batch"], row["freq"]): row
                    for row in DATA[(phase, routing)] if row["ws"] == ws
                }
                common = balanced.keys() & other.keys()
                batches, latency_ratios, energy_ratios = [], [], []
                for batch in sorted({key[1] for key in common}):
                    keys = [key for key in common if key[1] == batch]
                    batches.append(batch)
                    latency_ratios.append(
                        statistics.median(other[key]["latency"] / balanced[key]["latency"] for key in keys)
                    )
                    energy_ratios.append(
                        statistics.median(other[key]["energy"] / balanced[key]["energy"] for key in keys)
                    )
                ax.plot(
                    batches, latency_ratios, marker=MARKERS[routing], color=COLORS[routing],
                    linewidth=2, label=f"{routing}/bal latency",
                )
                ax.plot(
                    batches, energy_ratios, marker=MARKERS[routing], color=COLORS[routing],
                    linewidth=1.5, linestyle="--", label=f"{routing}/bal energy",
                )
            ax.axhline(1.0, color="#666666", linewidth=1, linestyle=":")
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks([1, 8, 32, 1024])
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.set_xlabel("Batch size")
            ax.set_ylabel("Median ratio to balanced (log scale)")
            ax.set_title(f"{PHASE_LABEL[phase]} · ws={ws}")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(fontsize=7)
    fig.suptitle(
        "Kernel-EP routing ratios by batch\n"
        "Median over lengths and GPU clocks; >1 means slower / more energy than balanced",
        fontsize=12,
    )
    save(
        fig, "ratio_by_batch.png",
        "Source: final block-NVML matrix · solid=barrier-excluded local-wall ratio · dashed=cluster-energy ratio",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for varying in ("batch", "length", "freq", "ws"):
        for metric in ("latency", "energy"):
            plot_dimension(varying, metric)
    plot_ratio_by_batch()


if __name__ == "__main__":
    main()
