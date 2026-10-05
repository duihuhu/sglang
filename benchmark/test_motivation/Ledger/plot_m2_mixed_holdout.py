"""Show both repetitions and new-request cost for the frozen Mixed screen."""

import argparse
import json
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.analysis.read_text())
    phases = data.get("phases", ["far", "near"])
    if phases not in (["near"], ["far", "near"]) or not data["complete"] or data["counts"]["runs"] != 16 * len(phases):
        parser.error("Need the complete declared phase matrix")
    policies = ("native607", "cap512", "slots96", "decode32")
    labels = ("Native607", "Cap512", "Slots96", "Decode32")
    colors = ("#267C9B", "#37886D", "#86699D", "#B07142")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for profile in ("train512", "hold384"):
        fig, axes = plt.subplots(2, len(phases), figsize=(11 if len(phases) == 2 else 6.5, 7.5), squeeze=False)
        invalid = False
        for col, phase in enumerate(phases):
            for row, metric in enumerate(("jct", "ttft")):
                ax, maximum = axes[row, col], 0
                for x, (policy, color) in enumerate(zip(policies, colors)):
                    own = sorted([r for r in data["rows"]
                                  if r["experiment"]["policy"] == policy
                                  and r["experiment"]["profile"] == profile
                                  and r["experiment"]["phase"] == phase],
                                 key=lambda r: r["experiment"]["rep"])
                    if [r["experiment"]["rep"] for r in own] != [81, 82]:
                        raise ValueError("Incomplete paired cell")
                    values = [r["metrics"]["mean_jct_s"] if metric == "jct"
                              else r["metrics"]["candidate"]["mean_ttft_s"] for r in own]
                    average = statistics.fmean(values)
                    bad = any(r["scene_errors"] for r in own)
                    invalid |= bad
                    ax.bar(x, average, width=.58, color=color,
                           hatch="//" if bad else None, edgecolor="#333333" if bad else color)
                    for offset, marker, value in zip((-.07, .07), ("o", "s"), values):
                        ax.scatter(x + offset, value, marker=marker, s=24, facecolors="white",
                                   edgecolors="#252525", zorder=3)
                    ax.annotate(f"{average:.2f}", (x, max(values)), xytext=(0, 7),
                                textcoords="offset points", ha="center", fontsize=10)
                    maximum = max(maximum, *values)
                ax.set_xticks(range(4), labels, fontsize=9)
                ax.set_ylim(0, maximum * 1.25)
                ax.set_title(phase.capitalize())
                ax.set_ylabel("Overall mean JCT (s)" if metric == "jct" else "New-request mean TTFT (s)")
                ax.set_axisbelow(True)
                ax.grid(axis="y", alpha=.18)
                ax.spines[["top", "right"]].set_visible(False)
        title = "512 x P512/O128 background" if profile == "train512" else "Held-out: 384 x P1024/O128 background"
        fig.suptitle(f"Mixed4K / same unified pool\n{title}", fontsize=13)
        note = "Bars: two-session means. Circle: rep81; square: rep82.\nPoints are observations, not confidence intervals."
        if invalid:
            note += "\nHatched bars contain an unmatched arrival scene; not a valid near/far causal contrast."
        mismatched = any(not s["scene_matched"] for c in data["contrasts"]
                         if c["profile"] == profile for s in c["samples"])
        if mismatched:
            note += "\nDescriptive comparison: not all pairs match arrival states."
        own_rows = [r for r in data["rows"] if r["experiment"]["profile"] == profile]
        cap_idle = all(not r["metrics"]["blocks"].get("request_limit", 0)
                       for r in own_rows if r["experiment"]["policy"] == "cap512")
        slots_idle = all(not r["control"]["native_slot_delays"]
                         for r in own_rows if r["experiment"]["policy"] == "slots96")
        if cap_idle and slots_idle:
            note += "\nCap512 and Slots96 did not trigger in this profile."
        fig.text(.5, .025, note, ha="center", va="bottom", fontsize=8.5)
        fig.subplots_adjust(left=.12, right=.98, top=.87, bottom=.20, hspace=.4, wspace=.25)
        for suffix in (".png", ".pdf"):
            fig.savefig(args.output_dir / f"m2_mixed_{profile}{suffix}", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
