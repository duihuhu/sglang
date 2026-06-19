#!/usr/bin/env python3
"""Compare Dense (Llama3.1-8B) vs MoE (Qwen3-30B-A3B) prefill characteristics at TP=1.

Generates comparison charts:
  1. Latency vs input_len at different batch sizes (A and F operators)
  2. Energy vs input_len at different batch sizes
  3. F/A ratio comparison (compute balance)
  4. Energy efficiency (mJ per token) vs frequency
  5. Frequency sensitivity heatmap

All outputs saved to the same analysis/ directory.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE
OUT_DIR.mkdir(parents=True, exist_ok=True)

DENSE_PATH = Path("/workspace/sglang-tier/benchmark/AFlex_bench/03_sensitivity/slo_sweep/retrain/data/v1_layer_profile/prefill_data_v1.txt")
MOE_PATH = Path("/workspace/sglang-tier/benchmark/AFlex_bench/06_others/more_model/energy_model/data/prefill_moe_tp1.txt")


def load_dense():
    df = pd.read_csv(DENSE_PATH, sep="\t", skiprows=1)
    df = df[df["tp"] == 1].copy()
    df.rename(columns={
        "A": "A_lat_us", "F": "F_lat_us",
        "A_energy_mj": "A_energy_mj", "F_energy_mj": "F_energy_mj"
    }, inplace=True)
    df["total_lat_us"] = df["A_lat_us"] + df["F_lat_us"]
    df["total_energy_mj"] = df["A_energy_mj"] + df["F_energy_mj"]
    df["model"] = "Dense (Llama3.1-8B)"
    return df


def load_moe():
    df = pd.read_csv(MOE_PATH, sep="\t")
    df = df[df["tp"] == 1].copy()
    df.rename(columns={
        "P_A_lat": "A_lat_us", "P_F_lat": "F_lat_us",
        "P_A_energy": "A_energy_mj", "P_F_energy": "F_energy_mj"
    }, inplace=True)
    df["total_lat_us"] = df["A_lat_us"] + df["F_lat_us"]
    df["total_energy_mj"] = df["A_energy_mj"] + df["F_energy_mj"]
    df["model"] = "MoE (Qwen3-30B-A3B)"
    return df


def plot_latency_vs_input_len(dense, moe):
    """Fig 1: Latency (A, F, Total) vs input_len for common batch sizes."""
    common_ils = sorted(set(dense["input_len"]) & set(moe["input_len"]))
    common_bs = sorted(set(dense["batch_size"]) & set(moe["batch_size"]))
    freq = 1410  # max freq for fair comparison

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Prefill Latency: Dense vs MoE (TP=1, freq=1410MHz)", fontsize=13, fontweight='bold')

    bs_to_plot = [bs for bs in [1, 4, 16, 32, 64, 128] if bs in common_bs][:6]

    for idx, bs in enumerate(bs_to_plot):
        ax = axes[idx // 3, idx % 3]
        d = dense[(dense["batch_size"] == bs) & (dense["gpu_clock"] == freq)]
        m = moe[(moe["batch_size"] == bs) & (moe["gpu_clock"] == freq)]

        d = d[d["input_len"].isin(common_ils)].sort_values("input_len")
        m = m[m["input_len"].isin(common_ils)].sort_values("input_len")

        if len(d) > 0:
            ax.plot(d["input_len"], d["A_lat_us"] / 1000, 'b-o', ms=4, label='Dense A')
            ax.plot(d["input_len"], d["F_lat_us"] / 1000, 'b--s', ms=4, label='Dense F')
        if len(m) > 0:
            ax.plot(m["input_len"], m["A_lat_us"] / 1000, 'r-o', ms=4, label='MoE A')
            ax.plot(m["input_len"], m["F_lat_us"] / 1000, 'r--s', ms=4, label='MoE F')

        ax.set_xlabel('Input Length (tokens)')
        ax.set_ylabel('Latency (ms)')
        ax.set_title(f'BS={bs}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.set_xscale('log', base=2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = OUT_DIR / "fig1_latency_vs_input_len.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close(fig)


def plot_energy_vs_input_len(dense, moe):
    """Fig 2: Energy (A, F) vs input_len for common batch sizes."""
    common_ils = sorted(set(dense["input_len"]) & set(moe["input_len"]))
    common_bs = sorted(set(dense["batch_size"]) & set(moe["batch_size"]))
    freq = 1410

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Prefill Energy: Dense vs MoE (TP=1, freq=1410MHz)", fontsize=13, fontweight='bold')

    bs_to_plot = [bs for bs in [1, 4, 16, 32, 64, 128] if bs in common_bs][:6]

    for idx, bs in enumerate(bs_to_plot):
        ax = axes[idx // 3, idx % 3]
        d = dense[(dense["batch_size"] == bs) & (dense["gpu_clock"] == freq)]
        m = moe[(moe["batch_size"] == bs) & (moe["gpu_clock"] == freq)]

        d = d[d["input_len"].isin(common_ils)].sort_values("input_len")
        m = m[m["input_len"].isin(common_ils)].sort_values("input_len")

        if len(d) > 0:
            ax.plot(d["input_len"], d["A_energy_mj"], 'b-o', ms=4, label='Dense A')
            ax.plot(d["input_len"], d["F_energy_mj"], 'b--s', ms=4, label='Dense F')
        if len(m) > 0:
            ax.plot(m["input_len"], m["A_energy_mj"], 'r-o', ms=4, label='MoE A')
            ax.plot(m["input_len"], m["F_energy_mj"], 'r--s', ms=4, label='MoE F')

        ax.set_xlabel('Input Length (tokens)')
        ax.set_ylabel('Energy (mJ)')
        ax.set_title(f'BS={bs}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = OUT_DIR / "fig2_energy_vs_input_len.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close(fig)


def plot_fa_ratio(dense, moe):
    """Fig 3: F/A latency and energy ratio comparison."""
    freq = 1410
    common_ils = sorted(set(dense["input_len"]) & set(moe["input_len"]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("F/A Ratio: Dense vs MoE (TP=1, freq=1410MHz)", fontsize=13, fontweight='bold')

    for bs in [1, 4, 8, 16, 32]:
        d = dense[(dense["batch_size"] == bs) & (dense["gpu_clock"] == freq)]
        d = d[d["input_len"].isin(common_ils)].sort_values("input_len")
        if len(d) > 0:
            ratio_lat = d["F_lat_us"] / d["A_lat_us"]
            axes[0].plot(d["input_len"], ratio_lat, '-o', ms=3, alpha=0.7, label=f'Dense bs={bs}')

    for bs in [1, 4, 8, 16, 32]:
        m = moe[(moe["batch_size"] == bs) & (moe["gpu_clock"] == freq)]
        m = m[m["input_len"].isin(common_ils)].sort_values("input_len")
        if len(m) > 0:
            ratio_lat = m["F_lat_us"] / m["A_lat_us"]
            axes[0].plot(m["input_len"], ratio_lat, '--s', ms=3, alpha=0.7, label=f'MoE bs={bs}')

    axes[0].set_xlabel('Input Length (tokens)')
    axes[0].set_ylabel('F/A Latency Ratio')
    axes[0].set_title('Latency Ratio (F_lat / A_lat)')
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(alpha=0.3)
    axes[0].set_xscale('log', base=2)
    axes[0].axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)

    for bs in [1, 4, 8, 16, 32]:
        d = dense[(dense["batch_size"] == bs) & (dense["gpu_clock"] == freq)]
        d = d[d["input_len"].isin(common_ils)].sort_values("input_len")
        if len(d) > 0:
            ratio_e = d["F_energy_mj"] / d["A_energy_mj"]
            axes[1].plot(d["input_len"], ratio_e, '-o', ms=3, alpha=0.7, label=f'Dense bs={bs}')

    for bs in [1, 4, 8, 16, 32]:
        m = moe[(moe["batch_size"] == bs) & (moe["gpu_clock"] == freq)]
        m = m[m["input_len"].isin(common_ils)].sort_values("input_len")
        if len(m) > 0:
            ratio_e = m["F_energy_mj"] / m["A_energy_mj"]
            axes[1].plot(m["input_len"], ratio_e, '--s', ms=3, alpha=0.7, label=f'MoE bs={bs}')

    axes[1].set_xlabel('Input Length (tokens)')
    axes[1].set_ylabel('F/A Energy Ratio')
    axes[1].set_title('Energy Ratio (F_energy / A_energy)')
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(alpha=0.3)
    axes[1].set_xscale('log', base=2)
    axes[1].axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = OUT_DIR / "fig3_fa_ratio.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close(fig)


def plot_freq_sensitivity(dense, moe):
    """Fig 4: Frequency sensitivity — latency and energy at fixed bs=8, il=1024."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Frequency Sensitivity: Dense vs MoE (TP=1, BS=8, IL=1024)",
                 fontsize=13, fontweight='bold')

    il = 1024
    bs = 8

    d = dense[(dense["batch_size"] == bs) & (dense["input_len"] == il)].sort_values("gpu_clock")
    m = moe[(moe["batch_size"] == bs) & (moe["input_len"] == il)].sort_values("gpu_clock")

    # Latency vs freq
    if len(d) > 0:
        axes[0, 0].plot(d["gpu_clock"], d["A_lat_us"] / 1000, 'b-o', ms=5, label='Dense A')
        axes[0, 0].plot(d["gpu_clock"], d["F_lat_us"] / 1000, 'b--s', ms=5, label='Dense F')
    if len(m) > 0:
        axes[0, 0].plot(m["gpu_clock"], m["A_lat_us"] / 1000, 'r-o', ms=5, label='MoE A')
        axes[0, 0].plot(m["gpu_clock"], m["F_lat_us"] / 1000, 'r--s', ms=5, label='MoE F')
    axes[0, 0].set_xlabel('GPU Frequency (MHz)')
    axes[0, 0].set_ylabel('Latency (ms)')
    axes[0, 0].set_title('Operator Latency vs Frequency')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.3)

    # Energy vs freq
    if len(d) > 0:
        axes[0, 1].plot(d["gpu_clock"], d["A_energy_mj"], 'b-o', ms=5, label='Dense A')
        axes[0, 1].plot(d["gpu_clock"], d["F_energy_mj"], 'b--s', ms=5, label='Dense F')
    if len(m) > 0:
        axes[0, 1].plot(m["gpu_clock"], m["A_energy_mj"], 'r-o', ms=5, label='MoE A')
        axes[0, 1].plot(m["gpu_clock"], m["F_energy_mj"], 'r--s', ms=5, label='MoE F')
    axes[0, 1].set_xlabel('GPU Frequency (MHz)')
    axes[0, 1].set_ylabel('Energy (mJ)')
    axes[0, 1].set_title('Operator Energy vs Frequency')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.3)

    # Total (A+F) latency vs freq
    if len(d) > 0:
        axes[1, 0].plot(d["gpu_clock"], d["total_lat_us"] / 1000, 'b-o', ms=5, label='Dense Total')
    if len(m) > 0:
        axes[1, 0].plot(m["gpu_clock"], m["total_lat_us"] / 1000, 'r-o', ms=5, label='MoE Total')
    axes[1, 0].set_xlabel('GPU Frequency (MHz)')
    axes[1, 0].set_ylabel('Latency (ms)')
    axes[1, 0].set_title('Total Latency (A+F) vs Frequency')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3)

    # Total energy vs freq
    if len(d) > 0:
        axes[1, 1].plot(d["gpu_clock"], d["total_energy_mj"], 'b-o', ms=5, label='Dense Total')
    if len(m) > 0:
        axes[1, 1].plot(m["gpu_clock"], m["total_energy_mj"], 'r-o', ms=5, label='MoE Total')
    axes[1, 1].set_xlabel('GPU Frequency (MHz)')
    axes[1, 1].set_ylabel('Energy (mJ)')
    axes[1, 1].set_title('Total Energy (A+F) vs Frequency')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = OUT_DIR / "fig4_freq_sensitivity.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close(fig)


def plot_energy_per_token(dense, moe):
    """Fig 5: Energy efficiency (mJ per token) across frequencies."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Energy Efficiency: Dense vs MoE (TP=1, per token·layer)",
                 fontsize=13, fontweight='bold')

    common_ils = sorted(set(dense["input_len"]) & set(moe["input_len"]))

    # At max freq, energy/token vs input_len for different bs
    freq = 1410
    for bs in [1, 4, 8, 16, 32]:
        d = dense[(dense["batch_size"] == bs) & (dense["gpu_clock"] == freq)]
        d = d[d["input_len"].isin(common_ils)].sort_values("input_len")
        if len(d) > 0:
            ept = d["total_energy_mj"] / (d["input_len"] * d["batch_size"])
            axes[0].plot(d["input_len"], ept, '-o', ms=3, alpha=0.7, label=f'Dense bs={bs}')

    for bs in [1, 4, 8, 16, 32]:
        m = moe[(moe["batch_size"] == bs) & (moe["gpu_clock"] == freq)]
        m = m[m["input_len"].isin(common_ils)].sort_values("input_len")
        if len(m) > 0:
            ept = m["total_energy_mj"] / (m["input_len"] * m["batch_size"])
            axes[0].plot(m["input_len"], ept, '--s', ms=3, alpha=0.7, label=f'MoE bs={bs}')

    axes[0].set_xlabel('Input Length (tokens)')
    axes[0].set_ylabel('Energy per token (mJ/tok)')
    axes[0].set_title(f'Energy/Token vs Input Length (freq={freq}MHz)')
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(alpha=0.3)
    axes[0].set_xscale('log', base=2)

    # Energy/token vs frequency at fixed il=1024
    il = 1024
    for bs in [1, 4, 8, 16, 32]:
        d = dense[(dense["batch_size"] == bs) & (dense["input_len"] == il)]
        d = d.sort_values("gpu_clock")
        if len(d) > 0:
            ept = d["total_energy_mj"] / (il * bs)
            axes[1].plot(d["gpu_clock"], ept, '-o', ms=3, alpha=0.7, label=f'Dense bs={bs}')

    for bs in [1, 4, 8, 16, 32]:
        m = moe[(moe["batch_size"] == bs) & (moe["input_len"] == il)]
        m = m.sort_values("gpu_clock")
        if len(m) > 0:
            ept = m["total_energy_mj"] / (il * bs)
            axes[1].plot(m["gpu_clock"], ept, '--s', ms=3, alpha=0.7, label=f'MoE bs={bs}')

    axes[1].set_xlabel('GPU Frequency (MHz)')
    axes[1].set_ylabel('Energy per token (mJ/tok)')
    axes[1].set_title(f'Energy/Token vs Frequency (IL={il})')
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = OUT_DIR / "fig5_energy_per_token.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close(fig)


def plot_dvfs_saving_potential(dense, moe):
    """Fig 6: DVFS saving potential — energy saving from freq=1410 to optimal."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("DVFS Saving Potential (Energy at optimal freq vs max freq, TP=1)",
                 fontsize=13, fontweight='bold')

    common_ils = sorted(set(dense["input_len"]) & set(moe["input_len"]))

    for idx, (df, name, color) in enumerate([(dense, "Dense", "blue"), (moe, "MoE", "red")]):
        ax = axes[idx]
        for bs in [1, 4, 8, 16, 32]:
            savings = []
            ils_used = []
            for il in common_ils:
                subset = df[(df["batch_size"] == bs) & (df["input_len"] == il)]
                if len(subset) < 2:
                    continue
                e_max = subset[subset["gpu_clock"] == 1410]["total_energy_mj"]
                e_min = subset["total_energy_mj"].min()
                if len(e_max) > 0 and e_max.values[0] > 0:
                    saving_pct = (1 - e_min / e_max.values[0]) * 100
                    savings.append(saving_pct)
                    ils_used.append(il)
            if savings:
                ax.plot(ils_used, savings, '-o', ms=4, label=f'bs={bs}')

        ax.set_xlabel('Input Length (tokens)')
        ax.set_ylabel('Energy Saving (%)')
        ax.set_title(f'{name}: Max Energy Saving (opt vs 1410MHz)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xscale('log', base=2)
        ax.set_ylim(bottom=0)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = OUT_DIR / "fig6_dvfs_saving_potential.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close(fig)


def compute_statistics(dense, moe):
    """Compute and print statistical summary for the report."""
    freq = 1410
    common_ils = sorted(set(dense["input_len"]) & set(moe["input_len"]))
    common_bs = sorted(set(dense["batch_size"]) & set(moe["batch_size"]))

    print("\n" + "=" * 70)
    print("  Dense vs MoE Prefill Statistics (TP=1, freq=1410MHz)")
    print("=" * 70)

    rows = []
    for il in common_ils:
        for bs in common_bs:
            d = dense[(dense["input_len"] == il) & (dense["batch_size"] == bs) & (dense["gpu_clock"] == freq)]
            m = moe[(moe["input_len"] == il) & (moe["batch_size"] == bs) & (moe["gpu_clock"] == freq)]
            if len(d) == 0 or len(m) == 0:
                continue
            rows.append({
                "il": il, "bs": bs,
                "dense_A_lat": d["A_lat_us"].values[0],
                "dense_F_lat": d["F_lat_us"].values[0],
                "moe_A_lat": m["A_lat_us"].values[0],
                "moe_F_lat": m["F_lat_us"].values[0],
                "dense_A_e": d["A_energy_mj"].values[0],
                "dense_F_e": d["F_energy_mj"].values[0],
                "moe_A_e": m["A_energy_mj"].values[0],
                "moe_F_e": m["F_energy_mj"].values[0],
            })

    stats_df = pd.DataFrame(rows)
    if len(stats_df) == 0:
        print("No overlapping data points!")
        return {}

    stats_df["A_lat_ratio"] = stats_df["moe_A_lat"] / stats_df["dense_A_lat"]
    stats_df["F_lat_ratio"] = stats_df["moe_F_lat"] / stats_df["dense_F_lat"]
    stats_df["A_e_ratio"] = stats_df["moe_A_e"] / stats_df["dense_A_e"]
    stats_df["F_e_ratio"] = stats_df["moe_F_e"] / stats_df["dense_F_e"]
    stats_df["dense_FA_lat_ratio"] = stats_df["dense_F_lat"] / stats_df["dense_A_lat"]
    stats_df["moe_FA_lat_ratio"] = stats_df["moe_F_lat"] / stats_df["moe_A_lat"]

    print(f"\nOverlapping data points: {len(stats_df)}")
    print(f"\nMoE/Dense ratios at freq=1410MHz:")
    print(f"  A latency: avg={stats_df['A_lat_ratio'].mean():.2f}x, "
          f"median={stats_df['A_lat_ratio'].median():.2f}x, "
          f"range=[{stats_df['A_lat_ratio'].min():.2f}, {stats_df['A_lat_ratio'].max():.2f}]")
    print(f"  F latency: avg={stats_df['F_lat_ratio'].mean():.2f}x, "
          f"median={stats_df['F_lat_ratio'].median():.2f}x, "
          f"range=[{stats_df['F_lat_ratio'].min():.2f}, {stats_df['F_lat_ratio'].max():.2f}]")
    print(f"  A energy:  avg={stats_df['A_e_ratio'].mean():.2f}x, "
          f"median={stats_df['A_e_ratio'].median():.2f}x, "
          f"range=[{stats_df['A_e_ratio'].min():.2f}, {stats_df['A_e_ratio'].max():.2f}]")
    print(f"  F energy:  avg={stats_df['F_e_ratio'].mean():.2f}x, "
          f"median={stats_df['F_e_ratio'].median():.2f}x, "
          f"range=[{stats_df['F_e_ratio'].min():.2f}, {stats_df['F_e_ratio'].max():.2f}]")
    print(f"\nF/A latency ratio within each model:")
    print(f"  Dense: avg={stats_df['dense_FA_lat_ratio'].mean():.2f}x, "
          f"range=[{stats_df['dense_FA_lat_ratio'].min():.2f}, {stats_df['dense_FA_lat_ratio'].max():.2f}]")
    print(f"  MoE:   avg={stats_df['moe_FA_lat_ratio'].mean():.2f}x, "
          f"range=[{stats_df['moe_FA_lat_ratio'].min():.2f}, {stats_df['moe_FA_lat_ratio'].max():.2f}]")

    # DVFS saving potential
    print(f"\nDVFS Energy Saving Potential (optimal freq vs 1410MHz):")
    for name, df in [("Dense", dense), ("MoE", moe)]:
        savings_all = []
        for il in common_ils:
            for bs in common_bs:
                subset = df[(df["batch_size"] == bs) & (df["input_len"] == il)]
                if len(subset) < 2:
                    continue
                e_max = subset[subset["gpu_clock"] == 1410]["total_energy_mj"]
                e_min = subset["total_energy_mj"].min()
                if len(e_max) > 0 and e_max.values[0] > 0:
                    savings_all.append((1 - e_min / e_max.values[0]) * 100)
        if savings_all:
            print(f"  {name}: avg={np.mean(savings_all):.1f}%, "
                  f"median={np.median(savings_all):.1f}%, "
                  f"max={np.max(savings_all):.1f}%")

    return {
        "A_lat_ratio_avg": stats_df["A_lat_ratio"].mean(),
        "F_lat_ratio_avg": stats_df["F_lat_ratio"].mean(),
        "A_e_ratio_avg": stats_df["A_e_ratio"].mean(),
        "F_e_ratio_avg": stats_df["F_e_ratio"].mean(),
        "dense_FA_lat_ratio_avg": stats_df["dense_FA_lat_ratio"].mean(),
        "moe_FA_lat_ratio_avg": stats_df["moe_FA_lat_ratio"].mean(),
    }


def main():
    print("Loading data...")
    dense = load_dense()
    moe = load_moe()
    print(f"  Dense TP=1: {len(dense)} rows")
    print(f"  MoE TP=1:   {len(moe)} rows")

    print("\nGenerating figures...")
    plot_latency_vs_input_len(dense, moe)
    plot_energy_vs_input_len(dense, moe)
    plot_fa_ratio(dense, moe)
    plot_freq_sensitivity(dense, moe)
    plot_energy_per_token(dense, moe)
    plot_dvfs_saving_potential(dense, moe)

    stats = compute_statistics(dense, moe)
    print("\nDone! All figures saved to:", OUT_DIR)
    return stats


if __name__ == "__main__":
    main()
