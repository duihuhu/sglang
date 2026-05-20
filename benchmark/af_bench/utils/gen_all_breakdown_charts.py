#!/usr/bin/env python3
"""Generate combined Gantt+breakdown charts for M=1,2,3,4,5 and a comparison montage."""
import subprocess, sys, os, json

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "throughput_logs")

def gen_one(m, prefix="msweep"):
    da_log = os.path.join(LOG_DIR, f"{prefix}_m{m}_da.log")
    df_log = os.path.join(LOG_DIR, f"{prefix}_m{m}_df.log")
    if not os.path.exists(da_log):
        print(f"  M={m}: DA log not found ({da_log}), skipping")
        return None
    if not os.path.exists(df_log):
        print(f"  M={m}: DF log not found ({df_log}), skipping")
        return None

    out = os.path.join(HERE, f"dstage_breakdown_m{m}_l4.png")
    cmd = [
        sys.executable, os.path.join(HERE, "plot_gantt_with_breakdown.py"),
        "--layers", "4",
        "--variant", "m3",  # just for title, overridden by log-prefix
        "--log-prefix", f"{prefix}_m{m}",
        "--save", out,
    ]
    print(f"  Generating M={m} chart → {out}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR M={m}: {result.stderr[:500]}")
        return None
    print(f"  M={m}: {result.stdout.strip().split(chr(10))[-1]}")
    return out

def main():
    charts = {}
    for m in [1, 2, 3, 4, 5]:
        path = gen_one(m)
        if path:
            charts[m] = path

    print("\nGenerated charts:")
    for m, p in charts.items():
        size_kb = os.path.getsize(p) / 1024
        print(f"  M={m}: {p} ({size_kb:.0f} KB)")

    # Generate summary comparison figure
    if charts:
        summary_path = os.path.join(HERE, "dstage_breakdown_summary.png")
        print(f"\nGenerating summary montage → {summary_path}")
        # Use matplotlib to create a 3x2 grid of the saved images
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        fig, axes = plt.subplots(2, 3, figsize=(36, 20))
        fig.patch.set_facecolor("white")

        for idx, m in enumerate([1, 2, 3, 4, 5]):
            ax = axes[idx // 3][idx % 3]
            if m in charts:
                img = mpimg.imread(charts[m])
                ax.imshow(img)
                ax.set_title(f"M = {m}", fontsize=14, fontweight="bold")
            ax.axis("off")

        # Remove extra subplot
        axes[1][2].axis("off")
        axes[1][2].text(0.5, 0.5, f"PD+AF Pipeline Breakdown\nMicro-batch sweep\nM=1..5\n\n50 requests, QPS=2\nQwen3-32B",
                       ha="center", va="center", fontsize=12, fontweight="bold",
                       transform=axes[1][2].transAxes)

        fig.suptitle("PD+AF D-Stage Pipeline Breakdown — Micro-batch Sweep (M=1..5)",
                    fontsize=16, fontweight="bold")
        plt.tight_layout()
        fig.savefig(summary_path, dpi=150, bbox_inches="tight")
        print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
