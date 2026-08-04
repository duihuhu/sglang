#!/usr/bin/env python3
"""Plot Vanilla, +Scheduler, and AFlex Energy/Token."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SHARED_ROOT = ROOT.parent / "shared_py"
sys.path.insert(0, str(SHARED_ROOT))
import bench_common as BC

DEFAULT_DATA = ROOT / "data/tier_perf_energy_all.json"
CHARTS_DIR = HERE
PAPER_PDF = HERE / "tier_perf_energy.pdf"
FIG_SIZE = (BC.FIG_SIZE[0], 4.1)
QPS_LIST = (2, 4, 8, 16)
PANEL_GAP_RATIO = 0.13
LEGEND_FONTSIZE = BC.LEGEND_FONT_SIZE
SUBTITLE_LABELPAD = 2
ENERGY_YLABEL = "Energy per Token (J)"
LEGEND_PAD = -0.03
LABEL_FONTSIZE = BC.FONT_SIZE * 0.55 + 2
LABEL_Y_PAD = 0.006
LABEL_ROTATION = 90
ARROW_LW = 1.6
ARROW_MUTATION_SCALE = 14
ARROW_RAD = -0.22
SERIES = (
    ("megascale", "Vanilla AFD", "#7f7f7f"),
    ("aflex_no_dvfs", "AFlex w/o DVFS", "#228B22"),
    ("aflex_dvfs", "AFlex", "#d62728"),
)


def _qps_tick_labels() -> list[str]:
    return [str(q) for q in QPS_LIST]


def _subtitle_below_xlabel(ax, text: str) -> None:
    ax.set_xlabel(text, fontsize=BC.PANEL_TITLE_FONTSIZE, labelpad=SUBTITLE_LABELPAD)


def _create_ab_axes(fig: plt.Figure) -> list[plt.Axes]:
    gs = fig.add_gridspec(1, 3, width_ratios=[1, PANEL_GAP_RATIO, 1], wspace=0)
    axes = [fig.add_subplot(gs[0, i]) for i in (0, 2)]
    spacer = fig.add_subplot(gs[0, 1])
    spacer.axis("off")
    spacer.plot([0.5], [0.5], alpha=0.0, transform=spacer.transAxes)
    return axes


PANEL_A_YMAX = 0.8
PANEL_B_YMAX = 0.6
Y_TICK_STEP = 0.2


def _format_y_tick_1dp(val: float, _pos: int) -> str:
    if abs(val) < 1e-9:
        return "0"
    return f"{val:.1f}"


def _apply_y_axis_fixed(ax: plt.Axes, ymax: float) -> None:
    ticks = np.arange(0.0, ymax + Y_TICK_STEP * 0.5, Y_TICK_STEP)
    ax.set_ylim(0, ymax)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(_format_y_tick_1dp))


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"required input is missing: {path}")
    return json.loads(path.read_text())


def _point_energy_j(scheme: str, point: dict, workload_key: str, plot_metric: dict) -> float:
    metric = plot_metric.get(scheme, "energy_per_token_j_all_tokens")
    if metric == "energy_per_token_mj_over_1000":
        value = point.get("energy_per_token_mj")
        if value is None:
            raise ValueError(f"missing energy_per_token_mj for {workload_key}")
        return value / 1000.0
    return BC.energy_per_token_j_all_tokens(point, workload_key)


def _panel_values(sources: dict, dataset: str, plot_metric: dict) -> list[list[float]]:
    rows = []
    for scheme, _, _ in SERIES:
        rows.append(
            [
                _point_energy_j(
                    scheme,
                    sources[scheme][f"{dataset}_qps{q}"],
                    f"{dataset}_qps{q}",
                    plot_metric,
                )
                for q in QPS_LIST
            ]
        )
    return rows


def validate_points(results: dict, path: Path, prefix: str = "") -> None:
    missing = []
    invalid = []
    for dataset in ("code", "conv"):
        for qps in QPS_LIST:
            key = f"{dataset}_qps{qps}"
            point = results.get(key)
            qualified = f"{prefix}{key}"
            if not isinstance(point, dict):
                missing.append(qualified)
            elif point.get("status") != "PASS" or point.get("energy_per_token_mj") is None:
                invalid.append(f"{qualified} (status={point.get('status')!r})")
    if missing or invalid:
        details = (["missing: " + ", ".join(missing)] if missing else []) + (
            ["not plottable: " + ", ".join(invalid)] if invalid else []
        )
        raise ValueError(f"benchmark data are incomplete in {path}; " + "; ".join(details))


def _add_improvement_labels(
    ax: plt.Axes,
    x: np.ndarray,
    vanilla: np.ndarray,
    vanilla_g: np.ndarray,
    aflex: np.ndarray,
    green_offset: float,
    red_offset: float,
) -> None:
    label_kw = dict(
        ha="center",
        va="bottom",
        rotation=LABEL_ROTATION,
        fontsize=LABEL_FONTSIZE,
        clip_on=False,
    )
    for pos, v0, v1, v2 in zip(x, vanilla, vanilla_g, aflex):
        if v0 > 0:
            green_pct = (v0 - v1) / v0 * 100.0
            ax.text(
                pos + green_offset,
                v1 + LABEL_Y_PAD,
                f"-{green_pct:.1f}%",
                color="#006400",
                **label_kw,
            )
        if v1 > 0:
            red_pct = (v1 - v2) / v1 * 100.0
            ax.text(
                pos + red_offset,
                v2 + LABEL_Y_PAD,
                f"-{red_pct:.1f}%",
                color="#d62728",
                **label_kw,
            )


def _bar_center_x(x_idx: int, series_idx: int, bar_width: float) -> float:
    return float(x_idx) + (series_idx - (len(SERIES) - 1) / 2) * bar_width


def _add_reference_arrows(
    ax: plt.Axes,
    x_idx: int,
    vanilla: float,
    vanilla_g: float,
    aflex: float,
    bar_width: float,
) -> None:
    vanilla_x = _bar_center_x(x_idx, 0, bar_width)
    green_x = _bar_center_x(x_idx, 1, bar_width)
    red_x = _bar_center_x(x_idx, 2, bar_width)
    green_label_y = vanilla_g + LABEL_Y_PAD
    red_label_y = aflex + LABEL_Y_PAD

    arrow_kw = dict(
        arrowstyle="->",
        lw=ARROW_LW,
        shrinkA=2,
        shrinkB=2,
        mutation_scale=ARROW_MUTATION_SCALE,
        connectionstyle=f"arc3,rad={ARROW_RAD}",
    )
    ax.annotate(
        "",
        xy=(vanilla_x, vanilla),
        xytext=(green_x, green_label_y),
        arrowprops=dict(**arrow_kw, color="#006400"),
        zorder=4,
    )
    ax.annotate(
        "",
        xy=(green_x, vanilla_g),
        xytext=(red_x, red_label_y),
        arrowprops=dict(**arrow_kw, color="#d62728"),
        zorder=4,
    )


def _set_panel_a_ylim(ax: plt.Axes, panel_ys: list[float]) -> None:
    _apply_y_axis_fixed(ax, PANEL_A_YMAX)


def _set_panel_b_ylim(ax: plt.Axes, panel_ys: list[float]) -> None:
    _apply_y_axis_fixed(ax, PANEL_B_YMAX)


def plot_figure(
    sources: dict,
    output_pdf: Path,
    plot_metric: dict,
) -> None:
    BC.apply_plot_style()
    fig = plt.figure(figsize=FIG_SIZE)
    axes = _create_ab_axes(fig)
    x = np.arange(len(QPS_LIST), dtype=float)
    bar_width = 0.84 / len(SERIES)

    for ax_idx, (dataset, title) in enumerate(zip(("conv", "code"), ("(a) Conversation", "(b) Coding"))):
        ax = axes[ax_idx]
        values = _panel_values(sources, dataset, plot_metric)
        panel_ys: list[float] = []
        for idx, (_, label, color) in enumerate(SERIES):
            ys = values[idx]
            panel_ys.extend(ys)
            offset = (idx - (len(SERIES) - 1) / 2) * bar_width
            ax.bar(
                x + offset,
                ys,
                width=bar_width * 0.92,
                color=color,
                edgecolor="none",
                linewidth=0,
                label=label,
                zorder=3,
            )

        vanilla = np.asarray(values[0], dtype=float)
        vanilla_g = np.asarray(values[1], dtype=float)
        aflex = np.asarray(values[2], dtype=float)
        green_offset = (1 - (len(SERIES) - 1) / 2) * bar_width
        red_offset = (2 - (len(SERIES) - 1) / 2) * bar_width

        ax.set_xticks(x, _qps_tick_labels())
        ax.set_ylabel(ENERGY_YLABEL, labelpad=2)
        _subtitle_below_xlabel(ax, title)
        ax.grid(True, alpha=0.3)
        ax.spines["right"].set_visible(True)
        if ax_idx == 0:
            _set_panel_a_ylim(ax, panel_ys)
        else:
            ax.set_ylabel("")
            _set_panel_b_ylim(ax, panel_ys)

        _add_improvement_labels(ax, x, vanilla, vanilla_g, aflex, green_offset, red_offset)
        if ax_idx == 0:
            _add_reference_arrows(ax, 0, vanilla[0], vanilla_g[0], aflex[0], bar_width)

    fig.subplots_adjust(left=0.11, right=0.95, top=0.90, bottom=0.20, wspace=0)
    legend_labels = [label for _, label, _ in SERIES]
    legend_handles = [
        Patch(facecolor=color, edgecolor="none") for _, _, color in SERIES
    ]
    fig.canvas.draw()
    pos_a = axes[0].get_position()
    pos_b = axes[1].get_position()
    legend_x = (pos_a.x0 + pos_b.x1) / 2
    legend_y = max(pos_a.y1, pos_b.y1) + LEGEND_PAD
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        fontsize=LEGEND_FONTSIZE,
        loc="lower center",
        ncol=len(SERIES),
        frameon=False,
        columnspacing=2,
        handletextpad=0.4,
        bbox_to_anchor=(legend_x, legend_y),
        bbox_transform=fig.transFigure,
    )

    BC.save_ablation_clustered_figure(fig, output_pdf, save_png=False)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=PAPER_PDF)
    args = parser.parse_args()
    try:
        payload = load(args.data)
        plot_metric = payload.get("meta", {}).get("plot_energy_metric", {})
        for scheme, _, _ in SERIES:
            validate_points(payload["results"][scheme], args.data, f"{scheme}.")
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Cannot plot Tier1 sensitivity: {exc}", file=sys.stderr)
        return 2

    plot_figure(payload["results"], args.output, plot_metric)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
