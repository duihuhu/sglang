#!/usr/bin/env python3
"""Generate MoE prefill characteristic plots matching dense_pic/ style.

Produces 3 figures:
  1. latency_freq.png   — Prefill latency vs GPU frequency (per input_len)
  2. latency_bs_tp.png  — Prefill latency vs batch size
  3. U-shaped.png       — Energy vs frequency (U-shaped curve)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

MOE_PATH = Path("/workspace/sglang-tier/benchmark/AFlex_bench/06_others/more_model/energy_model/data/prefill_moe_tp1.txt")
OUT_DIR = Path(__file__).resolve().parent / "moe_pic"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_moe():
    df = pd.read_csv(MOE_PATH, sep="\t")
    df = df[df["tp"] == 1].copy()
    # Total per-layer latency and energy (A + F)
    df["total_lat_us"] = df["P_A_lat"] + df["P_F_lat"]
    df["total_lat_ms"] = df["total_lat_us"] / 1000.0
    df["total_energy_mj"] = df["P_A_energy"] + df["P_F_energy"]
    # 48 layers total latency
    df["prefill_lat_ms"] = df["total_lat_us"] * 48 / 1000.0
    df["prefill_energy_mj"] = df["total_energy_mj"] * 48
    return df


def plot_latency_freq(df):
    """Fig 1: Prefill latency vs GPU frequency, one line per input_len (fixed bs=1)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("MoE (Qwen3-30B-A3B) Prefill Latency vs Frequency (TP=1)",
                 fontsize=13, fontweight='bold')

    colors = plt.cm.viridis(np.linspace(0, 0.9, 8))

    # Left: bs=1, various input_len
    bs = 1
    ax = axes[0]
    subset = df[df["batch_size"] == bs].sort_values("gpu_clock")
    for i, il in enumerate(sorted(subset["input_len"].unique())):
        data = subset[subset["input_len"] == il]
        ax.plot(data["gpu_clock"], data["prefill_lat_ms"],
                '-o', ms=5, color=colors[i], linewidth=1.5,
                label=f'IL={il}')
    ax.set_xlabel('GPU Frequency (MHz)', fontsize=10)
    ax.set_ylabel('Prefill Latency (ms)', fontsize=10)
    ax.set_title(f'BS=1, varying Input Length', fontsize=11)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_xticks([210, 450, 690, 930, 1170, 1410])

    # Right: bs=8, various input_len
    bs = 8
    ax = axes[1]
    subset = df[df["batch_size"] == bs].sort_values("gpu_clock")
    for i, il in enumerate(sorted(subset["input_len"].unique())):
        data = subset[subset["input_len"] == il]
        ax.plot(data["gpu_clock"], data["prefill_lat_ms"],
                '-o', ms=5, color=colors[i], linewidth=1.5,
                label=f'IL={il}')
    ax.set_xlabel('GPU Frequency (MHz)', fontsize=10)
    ax.set_ylabel('Prefill Latency (ms)', fontsize=10)
    ax.set_title(f'BS=8, varying Input Length', fontsize=11)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_xticks([210, 450, 690, 930, 1170, 1410])

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = OUT_DIR / "latency_freq.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close(fig)


def plot_latency_bs(df):
    """Fig 2: Heatmap of Lat(1410)/Lat(210) — frequency sensitivity (A/F x Dense/MoE)."""
    import seaborn as sns

    DENSE_PATH = Path("/workspace/sglang-tier/benchmark/AFlex_bench/"
                      "03_sensitivity/slo_sweep/retrain/data/v1_layer_profile/prefill_data_v1.txt")

    df_dense = pd.read_csv(DENSE_PATH, sep="\t", skiprows=1)
    df_dense = df_dense[df_dense["tp"] == 1].copy()
    df_dense["A_lat_ms"] = df_dense["A"] * 64 / 1000.0
    df_dense["F_lat_ms"] = df_dense["F"] * 64 / 1000.0

    df["A_lat_ms"] = df["P_A_lat"] * 48 / 1000.0
    df["F_lat_ms"] = df["P_F_lat"] * 48 / 1000.0

    def build_ratio(data, lat_col, bs_col="batch_size", il_col="input_len",
                    freq_col="gpu_clock"):
        batch_sizes = sorted(data[bs_col].unique())
        input_lens = sorted(data[il_col].unique())
        matrix = np.full((len(input_lens), len(batch_sizes)), np.nan)
        for i, il in enumerate(input_lens):
            for j, bs in enumerate(batch_sizes):
                lat_hi = data[(data[il_col] == il) & (data[bs_col] == bs)
                              & (data[freq_col] == 1410)][lat_col]
                lat_lo = data[(data[il_col] == il) & (data[bs_col] == bs)
                              & (data[freq_col] == 210)][lat_col]
                if len(lat_hi) > 0 and len(lat_lo) > 0:
                    matrix[i, j] = lat_hi.values[0] / lat_lo.values[0]
        return matrix, input_lens, batch_sizes

    dense_A, dense_ils, dense_bss = build_ratio(df_dense, "A_lat_ms")
    dense_F, _, _ = build_ratio(df_dense, "F_lat_ms")
    moe_A, moe_ils, moe_bss = build_ratio(df, "A_lat_ms")
    moe_F, _, _ = build_ratio(df, "F_lat_ms")

    all_vals = np.concatenate([
        dense_A[~np.isnan(dense_A)], dense_F[~np.isnan(dense_F)],
        moe_A[~np.isnan(moe_A)], moe_F[~np.isnan(moe_F)]])
    vmin, vmax = float(all_vals.min()), float(all_vals.max())

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(
        "Frequency Sensitivity: Lat(1410) / Lat(210)  (TP=1)\n"
        "Closer to 1 → memory-bound (freq insensitive) | Closer to 0 → compute-bound (freq sensitive)",
        fontsize=14, fontweight='bold')

    configs = [
        (axes[0, 0], dense_A, dense_ils, dense_bss, "Dense Attention (Llama3.1-8B)"),
        (axes[0, 1], moe_A, moe_ils, moe_bss, "MoE Attention (Qwen3-30B-A3B)"),
        (axes[1, 0], dense_F, dense_ils, dense_bss, "Dense FFN (Llama3.1-8B)"),
        (axes[1, 1], moe_F, moe_ils, moe_bss, "MoE FFN (Qwen3-30B-A3B)"),
    ]

    for ax, matrix, ils, bss, title in configs:
        sns.heatmap(matrix, annot=True, fmt=".3f", cmap="YlGnBu",
                    xticklabels=[str(b) for b in bss],
                    yticklabels=[str(il) for il in ils],
                    ax=ax, vmin=vmin, vmax=vmax,
                    linewidths=0.5, linecolor='white',
                    cbar_kws={'label': 'Lat(1410) / Lat(210)'})
        ax.set_xlabel('Batch Size', fontsize=10)
        ax.set_ylabel('Input Length', fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.invert_yaxis()

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = OUT_DIR / "latency_bs_tp.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close(fig)


def plot_u_shaped(df):
    """Fig 3: Energy vs frequency (U-shaped curve)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("MoE (Qwen3-30B-A3B) Energy vs Frequency — U-Shaped Curve (TP=1)",
                 fontsize=13, fontweight='bold')

    colors = plt.cm.viridis(np.linspace(0, 0.9, 8))

    # Left: bs=1, various input_len
    bs = 1
    ax = axes[0]
    subset = df[df["batch_size"] == bs].sort_values("gpu_clock")
    for i, il in enumerate(sorted(subset["input_len"].unique())):
        data = subset[subset["input_len"] == il]
        ax.plot(data["gpu_clock"], data["prefill_energy_mj"],
                '-o', ms=5, color=colors[i], linewidth=1.5,
                label=f'IL={il}')
        # Mark minimum energy point
        min_idx = data["prefill_energy_mj"].idxmin()
        ax.plot(data.loc[min_idx, "gpu_clock"], data.loc[min_idx, "prefill_energy_mj"],
                '*', ms=12, color=colors[i], zorder=5)
    ax.set_xlabel('GPU Frequency (MHz)', fontsize=10)
    ax.set_ylabel('Prefill Energy (mJ, 48 layers)', fontsize=10)
    ax.set_title(f'BS=1 (★ = optimal freq)', fontsize=11)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_xticks([210, 450, 690, 930, 1170, 1410])

    # Right: bs=8, various input_len
    bs = 8
    ax = axes[1]
    subset = df[df["batch_size"] == bs].sort_values("gpu_clock")
    for i, il in enumerate(sorted(subset["input_len"].unique())):
        data = subset[subset["input_len"] == il]
        ax.plot(data["gpu_clock"], data["prefill_energy_mj"],
                '-o', ms=5, color=colors[i], linewidth=1.5,
                label=f'IL={il}')
        min_idx = data["prefill_energy_mj"].idxmin()
        ax.plot(data.loc[min_idx, "gpu_clock"], data.loc[min_idx, "prefill_energy_mj"],
                '*', ms=12, color=colors[i], zorder=5)
    ax.set_xlabel('GPU Frequency (MHz)', fontsize=10)
    ax.set_ylabel('Prefill Energy (mJ, 48 layers)', fontsize=10)
    ax.set_title(f'BS=8 (★ = optimal freq)', fontsize=11)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_xticks([210, 450, 690, 930, 1170, 1410])

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = OUT_DIR / "U-shaped.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close(fig)


def print_optimal_freqs(df):
    """Print optimal frequency for each (bs, il) combination."""
    print("\n=== MoE Optimal Frequency (minimum energy) ===")
    print(f"{'BS':>4} {'IL':>6} {'Opt Freq':>9} {'Energy@Opt':>11} {'Energy@1410':>12} {'Saving%':>8}")
    print("-" * 55)
    for bs in sorted(df["batch_size"].unique()):
        for il in sorted(df["input_len"].unique()):
            subset = df[(df["batch_size"] == bs) & (df["input_len"] == il)]
            if len(subset) < 2:
                continue
            min_row = subset.loc[subset["prefill_energy_mj"].idxmin()]
            max_freq_row = subset[subset["gpu_clock"] == 1410]
            if len(max_freq_row) == 0:
                continue
            e_max = max_freq_row["prefill_energy_mj"].values[0]
            e_opt = min_row["prefill_energy_mj"]
            saving = (1 - e_opt / e_max) * 100
            print(f"{bs:>4} {il:>6} {int(min_row['gpu_clock']):>9} "
                  f"{e_opt:>11.1f} {e_max:>12.1f} {saving:>7.1f}%")


def main():
    print("Loading MoE data...")
    df = load_moe()
    print(f"  Rows: {len(df)}")

    plot_latency_freq(df)
    plot_latency_bs(df)
    plot_u_shaped(df)
    print_optimal_freqs(df)
    print(f"\nAll figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
