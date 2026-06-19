#!/usr/bin/env python3
"""Analyze V2 profiling data: max_tokens_per_expert vs TPOT scatter + linear fit."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

DATA_PATH = Path("/workspace/sglang-tier/benchmark/AFlex_bench/06_others/more_model/data/expert_load_vs_latency_batch_v2.tsv")
OUT_DIR = Path("/workspace/sglang-tier/benchmark/AFlex_bench/06_others/more_model/analysis_expert_load")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH, sep="\t")
    df["tpot_ms_avg"] = df["tpot_ms_avg"].astype(float)
    df["max_tokens_per_expert"] = df["max_tokens_per_expert"].astype(int)
    df["els"] = df["els"].astype(float)
    df["batch_size"] = df["batch_size"].astype(int)
    return df


def plot_scatter_with_fit(df):
    """Main plot: max_tokens_per_expert vs TPOT, colored by batch_size."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("MoE Expert Load vs TPOT (Qwen3-30B-A3B, TP=2, 128 output tokens)",
                 fontsize=13, fontweight='bold')

    # Left: Full scatter with color = batch_size
    ax = axes[0]
    batch_sizes = sorted(df["batch_size"].unique())
    cmap = plt.cm.viridis(np.linspace(0, 0.95, len(batch_sizes)))

    for i, bs in enumerate(batch_sizes):
        subset = df[df["batch_size"] == bs]
        ax.scatter(subset["max_tokens_per_expert"], subset["tpot_ms_avg"],
                   c=[cmap[i]], s=25, alpha=0.7, label=f'bs={bs}', edgecolors='none')

    # Global linear fit
    x = df["max_tokens_per_expert"].values.astype(float)
    y = df["tpot_ms_avg"].values.astype(float)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r--', linewidth=2,
            label=f'Linear fit: R²={r_value**2:.4f}\ny={slope:.4e}·x + {intercept:.1f}')

    ax.set_xlabel('max_tokens_per_expert', fontsize=10)
    ax.set_ylabel('TPOT (ms)', fontsize=10)
    ax.set_title('All Data Points (color = batch_size)')
    ax.legend(fontsize=7, ncol=2, loc='upper left')
    ax.grid(alpha=0.3)

    # Right: Same but with outlier removal (IQR method per bs)
    ax = axes[1]
    clean_dfs = []
    for bs in batch_sizes:
        subset = df[df["batch_size"] == bs]
        q1 = subset["tpot_ms_avg"].quantile(0.1)
        q3 = subset["tpot_ms_avg"].quantile(0.9)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        clean = subset[(subset["tpot_ms_avg"] >= lower) & (subset["tpot_ms_avg"] <= upper)]
        clean_dfs.append(clean)

    df_clean = pd.concat(clean_dfs)
    for i, bs in enumerate(batch_sizes):
        subset = df_clean[df_clean["batch_size"] == bs]
        ax.scatter(subset["max_tokens_per_expert"], subset["tpot_ms_avg"],
                   c=[cmap[i]], s=25, alpha=0.7, label=f'bs={bs}', edgecolors='none')

    x_c = df_clean["max_tokens_per_expert"].values.astype(float)
    y_c = df_clean["tpot_ms_avg"].values.astype(float)
    slope_c, intercept_c, r_c, p_c, std_err_c = stats.linregress(x_c, y_c)
    x_line_c = np.linspace(x_c.min(), x_c.max(), 100)
    y_line_c = slope_c * x_line_c + intercept_c
    ax.plot(x_line_c, y_line_c, 'r--', linewidth=2,
            label=f'Linear fit (cleaned): R²={r_c**2:.4f}\ny={slope_c:.4e}·x + {intercept_c:.1f}')

    ax.set_xlabel('max_tokens_per_expert', fontsize=10)
    ax.set_ylabel('TPOT (ms)', fontsize=10)
    ax.set_title(f'Outliers removed ({len(df) - len(df_clean)} pts removed)')
    ax.legend(fontsize=7, ncol=2, loc='upper left')
    ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = OUT_DIR / "v2_max_expert_vs_tpot_scatter.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close(fig)

    return slope, intercept, r_value, slope_c, intercept_c, r_c


def plot_violin(df):
    """Violin plot: TPOT distribution per batch_size."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    fig.suptitle("TPOT Distribution per Batch Size (TP=2, 128 output tokens)",
                 fontsize=13, fontweight='bold')

    batch_sizes = sorted(df["batch_size"].unique())
    data = [df[df["batch_size"] == bs]["tpot_ms_avg"].values for bs in batch_sizes]

    parts = ax.violinplot(data, positions=range(len(batch_sizes)), showmeans=True,
                          showmedians=True)
    ax.set_xticks(range(len(batch_sizes)))
    ax.set_xticklabels([str(bs) for bs in batch_sizes])
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('TPOT (ms)')
    ax.set_title('TPOT Distribution (violin)')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = OUT_DIR / "v2_tpot_violin_per_bs.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close(fig)


def plot_per_bs_linear(df):
    """Per-batch-size linear fits to see if relationship holds within each bs."""
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    fig.suptitle("max_tokens_per_expert vs TPOT (per batch_size, with linear fit)",
                 fontsize=13, fontweight='bold')

    batch_sizes = sorted(df["batch_size"].unique())
    axes_flat = axes.flatten()

    for i, bs in enumerate(batch_sizes):
        ax = axes_flat[i]
        subset = df[df["batch_size"] == bs]
        x = subset["max_tokens_per_expert"].values.astype(float)
        y = subset["tpot_ms_avg"].values.astype(float)

        ax.scatter(x, y, s=20, alpha=0.7, color='steelblue', edgecolors='none')

        if len(x) > 3:
            slope, intercept, r, p, se = stats.linregress(x, y)
            x_fit = np.linspace(x.min(), x.max(), 50)
            ax.plot(x_fit, slope * x_fit + intercept, 'r-', linewidth=1.5)
            ax.set_title(f'BS={bs} (R²={r**2:.3f}, n={len(x)})', fontsize=9)
        else:
            ax.set_title(f'BS={bs} (n={len(x)})', fontsize=9)

        ax.set_xlabel('max_tokens_per_expert', fontsize=8)
        ax.set_ylabel('TPOT (ms)', fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for j in range(len(batch_sizes), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = OUT_DIR / "v2_per_bs_linear_fit.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close(fig)


def print_summary(df, slope, intercept, r_val, slope_c, intercept_c, r_c):
    """Print statistical summary."""
    print("\n" + "=" * 70)
    print("  V2 PROFILING SUMMARY")
    print("=" * 70)
    print(f"  Total data points: {len(df)}")
    print(f"  Batch sizes: {sorted(df['batch_size'].unique())}")
    print(f"  max_tokens_per_expert range: [{df['max_tokens_per_expert'].min()}, {df['max_tokens_per_expert'].max()}]")
    print(f"  TPOT range: [{df['tpot_ms_avg'].min():.1f}, {df['tpot_ms_avg'].max():.1f}] ms")
    print(f"\n  Global linear fit (all data):")
    print(f"    TPOT = {slope:.6e} * max_tokens_per_expert + {intercept:.2f}")
    print(f"    R² = {r_val**2:.4f}")
    print(f"\n  Cleaned linear fit (outliers removed):")
    print(f"    TPOT = {slope_c:.6e} * max_tokens_per_expert + {intercept_c:.2f}")
    print(f"    R² = {r_c**2:.4f}")

    print(f"\n  Per-BS linear R²:")
    for bs in sorted(df["batch_size"].unique()):
        subset = df[df["batch_size"] == bs]
        x = subset["max_tokens_per_expert"].values.astype(float)
        y = subset["tpot_ms_avg"].values.astype(float)
        if len(x) > 3:
            _, _, r, _, _ = stats.linregress(x, y)
            print(f"    bs={bs:>3d}: R²={r**2:.4f} (n={len(x)}, "
                  f"maxE range=[{x.min():.0f}, {x.max():.0f}])")


def main():
    print("Loading V2 profiling data...")
    df = load_data()
    print(f"  {len(df)} rows loaded")

    slope, intercept, r, slope_c, intercept_c, r_c = plot_scatter_with_fit(df)
    plot_violin(df)
    plot_per_bs_linear(df)
    print_summary(df, slope, intercept, r, slope_c, intercept_c, r_c)


if __name__ == "__main__":
    main()
