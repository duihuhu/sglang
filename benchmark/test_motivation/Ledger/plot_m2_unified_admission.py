"""Plot one complete pair without imputing the missing second baseline."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.analysis.read_text())
    selected = [r for r in result["rows"] if r["experiment"]["profile"] == "state"
                and r["experiment"]["phase"] == "near" and r["experiment"]["rep"] == 51]
    rows = {r["experiment"]["policy"]: r for r in selected}
    if set(rows) != {"native", "decode32"} or len(selected) != 2:
        raise ValueError("Need exactly one complete state-near rep51 pair")
    if any(r["scene_errors"] for r in selected):
        raise ValueError("Cannot illustrate an invalid scene as a valid pair")
    if rows["native"]["fingerprint"] != rows["decode32"]["fingerprint"]:
        raise ValueError("Unpaired payload")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.4, 4.5))
    colors = {"native": "#267C9B", "decode32": "#D16E45"}
    labels = {"native": "Native", "decode32": "One-shot decode32"}
    x = np.arange(3)
    for i, policy in enumerate(("native", "decode32")):
        metrics = rows[policy]["metrics"]
        values = [metrics["holder"]["mean_jct_s"], metrics["candidate"]["mean_jct_s"], metrics["mean_jct_s"]]
        bars = left.bar(x+(i-.5)*.34, values, .34, color=colors[policy], label=labels[policy])
        left.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=4, fontsize=9)
    left.set_xticks(x, ["Existing (512)", "New (64)", "All (576)"])
    left.set_ylabel("Mean JCT (s)")
    left.set_ylim(0, 48)
    left.set_title("A. Existing requests gain; new requests pay", fontsize=11, pad=14)
    left.legend(frameon=False, loc="upper center", ncol=2, fontsize=9)
    left.set_axisbelow(True)
    left.grid(axis="y", alpha=.18)

    for y, policy in enumerate(("decode32", "native")):
        mechanism = rows[policy]["metrics"]["mechanism"]
        release = mechanism["holder_last_release_after_arrival_s"]
        activation = mechanism["first_candidate_activation_after_arrival_s"]
        right.barh(y, release, .20, color=colors[policy])
        right.scatter([activation], [y], marker="v", color="#252525", s=52, zorder=3)
        right.text(release+.65, y, f"{release:.2f}s", va="center", fontsize=9)
        right.text(.25, y+.23, f"First new activation: {activation:.3f}s", fontsize=9)
    right.set_yticks([0, 1], [labels["decode32"], labels["native"]])
    right.set_xlim(0, 31)
    right.set_ylim(-.45, 1.55)
    right.set_xlabel("Time after candidate burst starts (s)")
    right.set_title("B. Time until all 512 existing holders release", fontsize=11, pad=14)
    right.set_axisbelow(True)
    right.grid(axis="x", alpha=.18)
    fig.suptitle("Unified pool, Separate4K: state-dominant / near completion", fontsize=13, y=.98)
    fig.text(.5, .035, "Rep51 only (one pair), not a CI. No capacity rejection in this pair; state-capacity mediation is not established.",
             ha="center", fontsize=9, color="#444444")
    fig.subplots_adjust(left=.065, right=.98, bottom=.19, top=.81, wspace=.40)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(args.output.with_suffix(suffix), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
