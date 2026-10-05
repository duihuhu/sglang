"""Descriptive preflight figure, with no significance or causal labels."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def plot(summary, output):
    data = json.loads(summary.read_text())
    rows = {r["experiment"]["neighbor"]: r for r in data["rows"]
            if r["experiment"]["observer"] == "on" and r["experiment"]["target"] == "split"}
    order, labels = ("alone", "packed", "split"), ("Alone", "+ long neighbor", "+ short neighbor")
    colors = ("#64748b", "#b45309", "#047857")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))
    fig.subplots_adjust(top=.87, bottom=.14, left=.075, right=.98, hspace=.60, wspace=.27)
    fig.suptitle("M2 preflight: fixed short-request target, different neighbors", fontsize=15, y=.975)
    fig.text(.5, .93, "One run per cell; token budgets fixed. Descriptive observations, not a fairness or hybrid-causality test.",
             ha="center", fontsize=10, color="#475569")

    ax = axes[0, 0]
    values = [rows[n]["groups"]["A"]["p95_jct_s"] for n in order]
    ax.bar(range(3), values, color=colors, width=.55)
    for i, v in enumerate(values):
        ax.text(i, v + .10, f"{v:.2f}s", ha="center")
    ax.set(xticks=range(3), xticklabels=labels, ylabel="Target p95 JCT (s)", ylim=(0, 6.1))
    ax.set_title("(a) Same target, different service outcomes", loc="left", pad=14)

    ax = axes[0, 1]
    pressure = [100 * max(sum(v[1] * 21708800 + v[2] * 8064 for v in e["tenants"].values())
                         for e in rows[n]["resource_curves"]) / 19481072000 for n in order]
    ax.bar(range(3), pressure, color=colors, width=.55)
    for i, v in enumerate(pressure):
        ax.text(i, v + .18, f"{v:.1f}%", ha="center")
    ax.set(xticks=range(3), xticklabels=labels, ylabel="Peak active state + KV / raw pool (%)", ylim=(0, 12.2))
    ax.set_title("(b) No state/KV capacity denial observed", loc="left", pad=14)
    ax.text(.5, .91, "Raw pool: 19.48 GB/rank; active-cache byte proxy", transform=ax.transAxes,
            ha="center", fontsize=9)

    case = rows["packed"]
    for ax, role, title in zip(axes[1], ("A", "B"),
                               ("(c) Short target A: accounting vs work", "(d) Long neighbor B: accounting vs work")):
        curve = [(e["at_s"], e["tenants"][role]) for e in case["counter_execution_curves"] if role in e["tenants"]]
        x = [t for t, _ in curve]
        ax.step(x, [v["charge_without_lift"] for _, v in curve], where="post", color="#b45309",
                linestyle="--", linewidth=1.7, label="Token charges (lift excluded)")
        ax.step(x, [v["executed_weighted_tokens"] for _, v in curve], where="post", color="#0369a1",
                linewidth=1.7, label="Completed input + 2 x output")
        ax.set(xlim=(0, 6), ylim=(0, 38000), xlabel="Time from scheduled start (s); first 6s shown",
               ylabel="Weighted tokens")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}K"))
        ax.set_title(title, loc="left", pad=12)
        ax.legend(loc="lower right", frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=.16)
        ax.set_axisbelow(True)
    fig.text(.075, .046, "A: 2 waves x 16 requests, P1024/O32. Neighbor: P16384/O512 or 16 x P1024/O32 per wave.\n"
             "Mixed4096, token-VTC, unified pool, TP2. CPU result observations are not kernel timings. No confidence intervals.",
             fontsize=9, color="#475569")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.summary, args.output)
