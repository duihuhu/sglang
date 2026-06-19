#!/usr/bin/env python3
"""Generate decode version of latency_bs_tp.png — frequency sensitivity heatmap (2x2: A/F x Dense/MoE).

Uses heterogeneous frequency data to isolate A vs F sensitivity:
  - A sensitivity: fix f_F=1410, measure Lat(f_A=1410)/Lat(f_A=210)
  - F sensitivity: fix f_A=1410, measure Lat(f_F=1410)/Lat(f_F=210)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "moe_pic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DENSE_DECODE_PATH = Path("/workspace/sglang-tier/benchmark/AFlex_bench/"
                         "03_sensitivity/slo_sweep/retrain/data/v2_pipeline_profile/decode_pipeline_v1.txt")
MOE_DECODE_PATH = Path("/workspace/sglang-tier/benchmark/AFlex_bench/"
                       "06_others/more_model/energy_model/data/decode_pipeline_moe_tp1.txt")


def load_decode_dense():
    df = pd.read_csv(DENSE_DECODE_PATH, sep="\t")
    df = df[(df["tp"] == 1) & (df["M"] == 1)].copy()
    df["total_lat_ms"] = df["iter_lat_us"] / 1000.0
    return df


def load_decode_moe():
    df = pd.read_csv(MOE_DECODE_PATH, sep="\t")
    df = df[(df["tp"] == 1) & (df["M"] == 1)].copy()
    df["total_lat_ms"] = df["iter_lat_us"] / 1000.0
    return df


def build_ratio_hetero(data, bs_col="batch_size", il_col="input_len",
                       fix_col="f_F", vary_col="f_A",
                       freq_hi=1410, freq_lo=450):
    """Build Lat(vary=hi)/Lat(vary=lo) matrix, fixing the other freq at freq_hi."""
    subset = data[data[fix_col] == freq_hi].copy()
    batch_sizes = sorted(subset[bs_col].unique())
    input_lens = sorted(subset[il_col].unique())
    matrix = np.full((len(input_lens), len(batch_sizes)), np.nan)
    for i, il in enumerate(input_lens):
        for j, bs in enumerate(batch_sizes):
            lat_hi = subset[(subset[il_col] == il) & (subset[bs_col] == bs)
                            & (subset[vary_col] == freq_hi)]["total_lat_ms"]
            lat_lo = subset[(subset[il_col] == il) & (subset[bs_col] == bs)
                            & (subset[vary_col] == freq_lo)]["total_lat_ms"]
            if len(lat_hi) > 0 and len(lat_lo) > 0:
                matrix[i, j] = lat_hi.values[0] / lat_lo.values[0]
    return matrix, input_lens, batch_sizes


def main():
    print("Loading decode data...")
    df_dense = load_decode_dense()
    df_moe = load_decode_moe()
    print(f"  Dense decode rows (tp=1, M=1): {len(df_dense)}")
    print(f"  MoE decode rows (tp=1, M=1): {len(df_moe)}")

    # Build 4 matrices: A-sensitivity and F-sensitivity for Dense and MoE
    # A sensitivity: fix f_F=1410, vary f_A → measures how much A phase depends on freq
    # F sensitivity: fix f_A=1410, vary f_F → measures how much F phase depends on freq
    dense_A, dense_A_ils, dense_A_bss = build_ratio_hetero(
        df_dense, fix_col="f_F", vary_col="f_A")
    dense_F, dense_F_ils, dense_F_bss = build_ratio_hetero(
        df_dense, fix_col="f_A", vary_col="f_F")
    moe_A, moe_A_ils, moe_A_bss = build_ratio_hetero(
        df_moe, fix_col="f_F", vary_col="f_A")
    moe_F, moe_F_ils, moe_F_bss = build_ratio_hetero(
        df_moe, fix_col="f_A", vary_col="f_F")

    all_vals = np.concatenate([
        dense_A[~np.isnan(dense_A)], dense_F[~np.isnan(dense_F)],
        moe_A[~np.isnan(moe_A)], moe_F[~np.isnan(moe_F)]])
    vmin, vmax = float(all_vals.min()), float(all_vals.max())

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(
        "Decode: Frequency Sensitivity — Lat(1410) / Lat(450)  (TP=1)\n"
        "Closer to 1 → memory-bound (freq insensitive) | "
        "Closer to 0 → compute-bound (freq sensitive)",
        fontsize=14, fontweight='bold')

    configs = [
        (axes[0, 0], dense_A, dense_A_ils, dense_A_bss,
         "Dense Attention (Llama3.1-8B, TP=1)\nfix f_F=1410, Lat(1410)/Lat(450)"),
        (axes[0, 1], moe_A, moe_A_ils, moe_A_bss,
         "MoE Attention (Qwen3-30B-A3B, TP=1)\nfix f_F=1410, Lat(1410)/Lat(450)"),
        (axes[1, 0], dense_F, dense_F_ils, dense_F_bss,
         "Dense FFN (Llama3.1-8B, TP=1)\nfix f_A=1410, Lat(1410)/Lat(450)"),
        (axes[1, 1], moe_F, moe_F_ils, moe_F_bss,
         "MoE FFN (Qwen3-30B-A3B, TP=1)\nfix f_A=1410, Lat(1410)/Lat(450)"),
    ]

    for ax, matrix, ils, bss, title in configs:
        sns.heatmap(matrix, annot=True, fmt=".3f", cmap="YlGnBu",
                    xticklabels=[str(b) for b in bss],
                    yticklabels=[str(il) for il in ils],
                    ax=ax, vmin=vmin, vmax=vmax,
                    linewidths=0.5, linecolor='white',
                    cbar_kws={'label': 'Lat(1410) / Lat(450)'})
        ax.set_xlabel('Batch Size', fontsize=10)
        ax.set_ylabel('Input Length', fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.invert_yaxis()

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    path = OUT_DIR / "decode_latency_bs_tp.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
