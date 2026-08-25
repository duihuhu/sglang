#!/usr/bin/env python3
"""Plot balanced vs skewed_rank0 EP routing comparison charts."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import statistics as stats

PROFILE_ROOT = Path(__file__).resolve().parent.parent
EP_DIR = PROFILE_ROOT / "data" / "EP"
FIG_DIR = PROFILE_ROOT / "fig"

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": 150,
    }
)


def load(path: Path, len_col: str) -> dict:
    rows = {}
    with path.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            key = (int(r["size"]), int(r[len_col]), int(r["gpu_clock"]), int(r["batch_size"]))
            rows[key] = {"lat": float(r["latency_us"]), "en": float(r["energy_mj"])}
    return rows


def dim_stats(bal: dict, skew: dict, index: int) -> list[dict]:
    common = sorted(set(bal) & set(skew))
    groups: dict[int, list] = defaultdict(list)
    for k in common:
        groups[k[index]].append(k)
    rows = []
    for dim in sorted(groups):
        sub = groups[dim]
        lat_bal = sum(1 for k in sub if skew[k]["lat"] > bal[k]["lat"])
        en_bal = sum(1 for k in sub if skew[k]["en"] > bal[k]["en"])
        lr = [skew[k]["lat"] / bal[k]["lat"] for k in sub]
        er = [skew[k]["en"] / bal[k]["en"] for k in sub]
        rows.append(
            {
                "dim": dim,
                "n": len(sub),
                "lat_bal_pct": 100 * lat_bal / len(sub),
                "en_bal_pct": 100 * en_bal / len(sub),
                "lat_ratio": stats.median(lr),
                "en_ratio": stats.median(er),
            }
        )
    return rows


def _plot_dim_panel(ax, rows, xlabel: str, title: str, log_x: bool = False) -> None:
    xs = [r["dim"] for r in rows]
    lat_pct = [r["lat_bal_pct"] for r in rows]
    lat_ratio = [r["lat_ratio"] for r in rows]

    color_bar = "#4C78A8"
    color_line = "#E45756"
    ax2 = ax.twinx()
    bars = ax.bar(xs, lat_pct, color=color_bar, alpha=0.75, label="balanced latency win rate (%)")
    line = ax2.plot(xs, lat_ratio, color=color_line, marker="o", linewidth=2, label="skew/bal latency ratio")
    ax2.axhline(1.0, color="#666", linestyle="--", linewidth=1, alpha=0.7)

    ax.set_ylabel("balanced win rate (%)", color=color_bar)
    ax2.set_ylabel("skew/bal latency ratio", color=color_line)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_ylim(0, 105)
    ax2.set_ylim(0, max(4.0, max(lat_ratio) * 1.15))
    if log_x:
        ax.set_xscale("log", base=2)
        ax2.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs])
    ax.tick_params(axis="y", labelcolor=color_bar)
    ax2.tick_params(axis="y", labelcolor=color_line)

    handles = [bars, line[0]]
    labels = ["balanced win rate (%)", "skew/bal ratio (>1 balanced better)"]
    ax.legend(handles, labels, loc="upper left", framealpha=0.9)


def plot_by_dimension(
    pf_bal: dict,
    pf_skew: dict,
    df_bal: dict,
    df_skew: dict,
    index: int,
    xlabel: str,
    outfile: str,
    title_prefix: str,
    log_x: bool = False,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    pf_rows = dim_stats(pf_bal, pf_skew, index)
    df_rows = dim_stats(df_bal, df_skew, index)
    _plot_dim_panel(axes[0], pf_rows, xlabel, f"Prefill — {title_prefix}", log_x=log_x)
    _plot_dim_panel(axes[1], df_rows, xlabel, f"Decode — {title_prefix}", log_x=log_x)
    fig.tight_layout()
    fig.savefig(FIG_DIR / outfile, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(bal: dict, skew: dict, phase: str, outfile: str) -> None:
    common = sorted(set(bal) & set(skew))
    batches = sorted({k[3] for k in common})
    lengths = sorted({k[1] for k in common})
    grid = []
    for length in lengths:
        row = []
        for batch in batches:
            sub = [k for k in common if k[1] == length and k[3] == batch]
            if not sub:
                row.append(float("nan"))
                continue
            row.append(stats.median([skew[k]["lat"] / bal[k]["lat"] for k in sub]))
        grid.append(row)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn_r", vmin=0.6, vmax=3.5)
    ax.set_xticks(range(len(batches)))
    ax.set_xticklabels(batches, rotation=45, ha="right")
    ax.set_yticks(range(len(lengths)))
    ax.set_yticklabels(lengths)
    ax.set_xlabel("batch_size")
    ax.set_ylabel("length")
    ax.set_title(f"{phase}: skew/bal latency ratio (green=skew, red=balanced)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("skew/bal latency ratio")
    fig.tight_layout()
    fig.savefig(FIG_DIR / outfile, bbox_inches="tight")
    plt.close(fig)


def plot_overview_summary(
    pf_bal: dict,
    pf_skew: dict,
    df_bal: dict,
    df_skew: dict,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    configs = [
        (0, "batch_size", "by batch", True),
        (0, "EP size (ws)", "by EP size", False),
        (1, "length", "by length", False),
    ]
    for col, (index, xlabel, title, log_x) in enumerate(configs):
        pf_rows = dim_stats(pf_bal, pf_skew, index)
        df_rows = dim_stats(df_bal, df_skew, index)
        _plot_dim_panel(axes[0, col], pf_rows, xlabel, f"Prefill {title}", log_x=log_x)
        _plot_dim_panel(axes[1, col], df_rows, xlabel, f"Decode {title}", log_x=log_x)
    fig.suptitle("Legacy EP (none): balanced vs skewed_rank0 routing", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ep_overview_bal_vs_skew.png", bbox_inches="tight")
    plt.close(fig)


def _paired_improvements(bal: dict, skew: dict) -> list[dict]:
    """Improvement when switching skewed -> balanced (positive = balanced better)."""
    rows = []
    for k in sorted(set(bal) & set(skew)):
        b, s = bal[k], skew[k]
        lat_red = (s["lat"] - b["lat"]) / s["lat"] * 100
        en_red = (s["en"] - b["en"]) / s["en"] * 100
        lat_ratio = s["lat"] / b["lat"]
        en_ratio = s["en"] / b["en"]
        asym = lat_red - en_red  # >0: latency gains exceed energy gains
        rows.append(
            {
                "key": k,
                "ws": k[0],
                "length": k[1],
                "freq": k[2],
                "batch": k[3],
                "lat_red": lat_red,
                "en_red": en_red,
                "lat_ratio": lat_ratio,
                "en_ratio": en_ratio,
                "asym": asym,
                "bal_lat": b["lat"],
                "bal_en": b["en"],
                "skew_lat": s["lat"],
                "skew_en": s["en"],
                "throughput_bal": k[3] / b["lat"],
                "throughput_skew": k[3] / s["lat"],
                "eff_bal": k[3] / b["en"],
                "eff_skew": k[3] / s["en"],
            }
        )
    return rows


def plot_tradeoff_reduction_scatter(
    pf_rows: list[dict],
    df_rows: list[dict],
) -> None:
    """Scatter: energy vs latency reduction % (skew->balanced). Above diagonal = disproportionate latency win."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, rows, title in zip(axes, [pf_rows, df_rows], ["Prefill", "Decode"]):
        en_red = [r["en_red"] for r in rows]
        lat_red = [r["lat_red"] for r in rows]
        batches = [r["batch"] for r in rows]
        sc = ax.scatter(
            en_red,
            lat_red,
            c=batches,
            cmap="viridis",
            s=14,
            alpha=0.55,
            edgecolors="none",
        )
        lim_lo = min(min(en_red), min(lat_red), -15)
        lim_hi = max(max(en_red), max(lat_red), 15) * 1.05
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", linewidth=1, alpha=0.6, label="proportional (y=x)")
        ax.axhline(0, color="#999", linewidth=0.8)
        ax.axvline(0, color="#999", linewidth=0.8)
        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
        ax.set_xlabel("energy reduction % (skew -> balanced)")
        ax.set_ylabel("latency reduction % (skew -> balanced)")
        ax.set_title(title)
        above = sum(1 for r in rows if r["lat_red"] > r["en_red"])
        med_lat = stats.median(lat_red)
        med_en = stats.median(en_red)
        ax.text(
            0.03,
            0.97,
            f"n={len(rows)}\nabove diagonal: {above}/{len(rows)} ({100*above/len(rows):.0f}%)\n"
            f"median lat_red={med_lat:.1f}%\nmedian en_red={med_en:.1f}%",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )
        fig.colorbar(sc, ax=ax, label="batch_size")
    fig.suptitle(
        "Latency–energy tradeoff: skewed -> balanced (above diagonal = latency improves more than energy)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ep_tradeoff_reduction_scatter.png", bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff_ratio_roofline(
    pf_rows: list[dict],
    df_rows: list[dict],
) -> None:
    """Log-ratio roofline: x=en_ratio, y=lat_ratio. Diagonal = proportional slowdown."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, rows, title in zip(axes, [pf_rows, df_rows], ["Prefill", "Decode"]):
        x = [r["en_ratio"] for r in rows]
        y = [r["lat_ratio"] for r in rows]
        batches = [r["batch"] for r in rows]
        sc = ax.scatter(x, y, c=batches, cmap="viridis", s=14, alpha=0.55, edgecolors="none")
        lo = min(min(x), min(y), 0.5)
        hi = max(max(x), max(y), 2.0) * 1.08
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.6, label="proportional (lat=en)")
        ax.axhline(1, color="#999", linewidth=0.8)
        ax.axvline(1, color="#999", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("energy ratio skew/balanced (>1 balanced better)")
        ax.set_ylabel("latency ratio skew/balanced (>1 balanced better)")
        ax.set_title(title)
        above = sum(1 for r in rows if r["lat_ratio"] > r["en_ratio"])
        ax.text(
            0.03,
            0.97,
            f"above diagonal: {above}/{len(rows)} ({100*above/len(rows):.0f}%)\n"
            f"median lat_ratio={stats.median(y):.2f}\nmedian en_ratio={stats.median(x):.2f}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )
        fig.colorbar(sc, ax=ax, label="batch_size")
    fig.suptitle(
        "Ratio roofline: skew/balanced (>1 = balanced better; above diagonal = latency gap > energy gap)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ep_tradeoff_ratio_roofline.png", bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff_efficiency_frontier(
    pf_rows: list[dict],
    df_rows: list[dict],
    sample_max: int = 400,
) -> None:
    """Throughput vs energy-efficiency frontier; arrows skew -> balanced."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, rows, title in zip(axes, [pf_rows, df_rows], ["Prefill", "Decode"]):
        if len(rows) > sample_max:
            step = len(rows) // sample_max
            plot_rows = rows[: :step]
        else:
            plot_rows = rows
        for r in plot_rows:
            ax.annotate(
                "",
                xy=(r["eff_bal"], r["throughput_bal"]),
                xytext=(r["eff_skew"], r["throughput_skew"]),
                arrowprops=dict(arrowstyle="->", color="#888", alpha=0.25, lw=0.6),
            )
        ax.scatter(
            [r["eff_skew"] for r in rows],
            [r["throughput_skew"] for r in rows],
            c="#E45756",
            s=8,
            alpha=0.35,
            label="skewed",
        )
        ax.scatter(
            [r["eff_bal"] for r in rows],
            [r["throughput_bal"] for r in rows],
            c="#4C78A8",
            s=8,
            alpha=0.35,
            label="balanced",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("energy efficiency (batch / energy_mj)")
        ax.set_ylabel("throughput (batch / latency_us)")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle(
        "Efficiency frontier: arrows skewed -> balanced (non-uniform latency/energy movement)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ep_tradeoff_efficiency_frontier.png", bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff_by_batch_bars(
    pf_rows: list[dict],
    df_rows: list[dict],
) -> None:
    """Grouped bars: median latency vs energy reduction % by batch."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, rows, title in zip(axes, [pf_rows, df_rows], ["Prefill", "Decode"]):
        batches = sorted({r["batch"] for r in rows})
        lat_meds = []
        en_meds = []
        for b in batches:
            sub = [r for r in rows if r["batch"] == b]
            lat_meds.append(stats.median([r["lat_red"] for r in sub]))
            en_meds.append(stats.median([r["en_red"] for r in sub]))
        x = range(len(batches))
        w = 0.35
        ax.bar([i - w / 2 for i in x], lat_meds, w, label="latency reduction %", color="#4C78A8")
        ax.bar([i + w / 2 for i in x], en_meds, w, label="energy reduction %", color="#E45756")
        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in batches], rotation=45, ha="right")
        ax.axhline(0, color="#999", linewidth=0.8)
        ax.set_xlabel("batch_size")
        ax.set_ylabel("reduction % (skew -> balanced)")
        ax.set_title(title)
        ax.legend(fontsize=9)
    fig.suptitle("Median latency vs energy reduction by batch (gap = non-proportional tradeoff)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ep_tradeoff_by_batch.png", bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff_asymmetry_hist(
    pf_rows: list[dict],
    df_rows: list[dict],
) -> None:
    """Histogram of latency_reduction - energy_reduction asymmetry."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, rows, title in zip(axes, [pf_rows, df_rows], ["Prefill", "Decode"]):
        asym = [r["asym"] for r in rows]
        ax.hist(asym, bins=40, color="#72B7B2", edgecolor="white", alpha=0.85)
        ax.axvline(0, color="k", linewidth=1)
        ax.axvline(stats.median(asym), color="#E45756", linewidth=2, label=f"median={stats.median(asym):.1f}%")
        ax.set_xlabel("asymmetry: latency_red% - energy_red%")
        ax.set_ylabel("count")
        ax.set_title(title)
        ax.legend(fontsize=9)
    fig.suptitle("Asymmetry distribution (>0 = latency improves more than energy)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ep_tradeoff_asymmetry_hist.png", bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff_all(pf_bal: dict, pf_skew: dict, df_bal: dict, df_skew: dict) -> None:
    pf_rows = _paired_improvements(pf_bal, pf_skew)
    df_rows = _paired_improvements(df_bal, df_skew)
    plot_tradeoff_reduction_scatter(pf_rows, df_rows)
    plot_tradeoff_ratio_roofline(pf_rows, df_rows)
    plot_tradeoff_efficiency_frontier(pf_rows, df_rows)
    plot_tradeoff_by_batch_bars(pf_rows, df_rows)
    plot_tradeoff_asymmetry_hist(pf_rows, df_rows)


BASELINE = {"ws": 4, "length": 512, "freq": 930, "batch": 32}
KEY_INDEX = {"ws": 0, "length": 1, "freq": 2, "batch": 3}
AXIS_LABEL = {
    "ws": "EP size (world size)",
    "length": "input/context length",
    "freq": "GPU frequency (MHz)",
    "batch": "batch size",
}


def _fixed_slice(bal: dict, skew: dict, varying: str) -> list[tuple[int, dict, dict]]:
    """Return paired points while varying one axis and fixing the other three."""
    varying_index = KEY_INDEX[varying]
    rows = []
    for key in sorted(set(bal) & set(skew), key=lambda item: item[varying_index]):
        if any(
            key[KEY_INDEX[axis]] != value
            for axis, value in BASELINE.items()
            if axis != varying
        ):
            continue
        rows.append((key[varying_index], bal[key], skew[key]))
    if not rows:
        fixed = ", ".join(
            f"{axis}={value}" for axis, value in BASELINE.items() if axis != varying
        )
        raise RuntimeError(f"no paired rows for varying={varying}, fixed {fixed}")
    return rows


def _plot_metric(
    ax,
    rows: list[tuple[int, dict, dict]],
    metric: str,
    varying: str,
    phase: str,
) -> None:
    xs = [row[0] for row in rows]
    balanced = [row[1][metric] for row in rows]
    skewed = [row[2][metric] for row in rows]

    ax.plot(xs, balanced, marker="o", linewidth=2, color="#4C78A8", label="balanced")
    ax.plot(xs, skewed, marker="s", linewidth=2, color="#E45756", label="skewed_rank0")
    ax.set_yscale("log")
    if varying in {"batch", "length"}:
        ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(value) for value in xs])
    ax.set_xlabel(AXIS_LABEL[varying])
    unit = "us" if metric == "lat" else "mJ"
    name = "Latency" if metric == "lat" else "Energy"
    ax.set_ylabel(f"{name} ({unit}, log scale)")
    ax.set_title(f"{phase} {name}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()


def plot_fixed_dimension(
    pf_bal: dict,
    pf_skew: dict,
    df_bal: dict,
    df_skew: dict,
    varying: str,
) -> None:
    """Plot balanced/skewed latency and energy with three axes held constant."""
    pf_rows = _fixed_slice(pf_bal, pf_skew, varying)
    df_rows = _fixed_slice(df_bal, df_skew, varying)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    _plot_metric(axes[0, 0], pf_rows, "lat", varying, "Prefill")
    _plot_metric(axes[0, 1], pf_rows, "en", varying, "Prefill")
    _plot_metric(axes[1, 0], df_rows, "lat", varying, "Decode")
    _plot_metric(axes[1, 1], df_rows, "en", varying, "Decode")

    fixed = ", ".join(
        f"{axis}={value}" for axis, value in BASELINE.items() if axis != varying
    )
    fig.suptitle(
        f"Balanced vs skewed while varying {AXIS_LABEL[varying]}\nFixed: {fixed}",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"ep_fixed_vary_{varying}.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pf_bal = load(EP_DIR / "PF-balanced.txt", "input_len")
    pf_skew = load(EP_DIR / "PF-skewed.txt", "input_len")
    df_bal = load(EP_DIR / "DF-balanced.txt", "context_len")
    df_skew = load(EP_DIR / "DF-skewed.txt", "context_len")

    for varying in ("batch", "ws", "length", "freq"):
        plot_fixed_dimension(pf_bal, pf_skew, df_bal, df_skew, varying)

    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
