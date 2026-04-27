#!/usr/bin/env python3
"""
PD 分离结果可视化
生成类似 AF 分离论文的图表
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

RESULTS_DIR = "/workspace/benchmark/sglang-main/bash-test/hucc"
OUTPUT_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_all_results():
    """加载所有 SLO 的结果"""
    slo_multipliers = [1.0, 1.1, 1.2, 1.5, 2.0]
    all_data = []

    for slo in slo_multipliers:
        file_path = os.path.join(RESULTS_DIR, f"results_slo_{slo}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['slo_mult'] = slo
            all_data.append(df)

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return None

def plot_slo_curve(df):
    """
    图1: 收益 vs SLO 曲线
    类似论文 Fig 2
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    slo_mults = sorted(df['slo_mult'].unique())

    for tp in sorted(df['tp'].unique()):
        tp_data = df[df['tp'] == tp]

        means = []
        stds = []
        p25s = []
        p75s = []

        for slo in slo_mults:
            slo_data = tp_data[tp_data['slo_mult'] == slo]['saving_pct']
            means.append(slo_data.mean())
            stds.append(slo_data.std())
            p25s.append(slo_data.quantile(0.25))
            p75s.append(slo_data.quantile(0.75))

        # 绘制均值线
        ax.plot(slo_mults, means, marker='o', linewidth=2, label=f'TP={tp}', markersize=8)

        # 绘制 P25-P75 区间
        ax.fill_between(slo_mults, p25s, p75s, alpha=0.2)

    ax.set_xlabel('SLO Multiplier', fontsize=12)
    ax.set_ylabel('Energy Saving (%)', fontsize=12)
    ax.set_title('PD Separation: Energy Savings vs SLO', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xlim(0.95, max(slo_mults) + 0.1)

    # 添加甜区标注
    ax.axvspan(1.05, 1.15, alpha=0.1, color='green', label='Sweet Spot')
    ax.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='5% Threshold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_slo_curve.png'), dpi=300, bbox_inches='tight')
    print(f"已保存: {OUTPUT_DIR}/fig1_slo_curve.png")
    plt.close()

def plot_tp_comparison(df):
    """
    图2: 不同 TP 的收益对比（柱状图）
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 子图1: SLO×1.1 的详细统计
    slo_11_data = df[df['slo_mult'] == 1.1]

    tp_stats = []
    for tp in sorted(slo_11_data['tp'].unique()):
        tp_data = slo_11_data[slo_11_data['tp'] == tp]['saving_pct']
        tp_stats.append({
            'tp': tp,
            'mean': tp_data.mean(),
            'median': tp_data.median(),
            'max': tp_data.max(),
            'sweet_ratio': (tp_data > 5).sum() / len(tp_data) * 100
        })

    stats_df = pd.DataFrame(tp_stats)

    x = np.arange(len(stats_df))
    width = 0.25

    axes[0].bar(x - width, stats_df['mean'], width, label='Mean', alpha=0.8)
    axes[0].bar(x, stats_df['median'], width, label='Median', alpha=0.8)
    axes[0].bar(x + width, stats_df['max'], width, label='Max', alpha=0.8)

    axes[0].set_xlabel('Tensor Parallelism', fontsize=12)
    axes[0].set_ylabel('Energy Saving (%)', fontsize=12)
    axes[0].set_title('SLO×1.1: Savings by TP', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'TP={tp}' for tp in stats_df['tp']])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # 子图2: 甜区占比
    axes[1].bar(x, stats_df['sweet_ratio'], color='green', alpha=0.7)
    axes[1].set_xlabel('Tensor Parallelism', fontsize=12)
    axes[1].set_ylabel('Sweet Spot Ratio (%)', fontsize=12)
    axes[1].set_title('SLO×1.1: Configs with >5% Savings', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'TP={tp}' for tp in stats_df['tp']])
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for i, v in enumerate(stats_df['sweet_ratio']):
        axes[1].text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_tp_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"已保存: {OUTPUT_DIR}/fig2_tp_comparison.png")
    plt.close()

def plot_energy_composition(df):
    """
    图3: Prefill vs Decode 能耗占比
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    slo_11_data = df[df['slo_mult'] == 1.1]

    tp_list = sorted(slo_11_data['tp'].unique())
    p_ratios = []
    d_ratios = []

    for tp in tp_list:
        tp_data = slo_11_data[slo_11_data['tp'] == tp]
        p_ratios.append(tp_data['p_energy_ratio'].mean())
        d_ratios.append(tp_data['d_energy_ratio'].mean())

    x = np.arange(len(tp_list))
    width = 0.6

    p1 = ax.bar(x, p_ratios, width, label='Prefill', color='skyblue', alpha=0.8)
    p2 = ax.bar(x, d_ratios, width, bottom=p_ratios, label='Decode', color='coral', alpha=0.8)

    ax.set_xlabel('Tensor Parallelism', fontsize=12)
    ax.set_ylabel('Energy Composition (%)', fontsize=12)
    ax.set_title('Prefill vs Decode Energy Composition (SLO×1.1)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'TP={tp}' for tp in tp_list])
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for i, (p, d) in enumerate(zip(p_ratios, d_ratios)):
        ax.text(i, p/2, f'{p:.1f}%', ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(i, p + d/2, f'{d:.1f}%', ha='center', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_energy_composition.png'), dpi=300, bbox_inches='tight')
    print(f"已保存: {OUTPUT_DIR}/fig3_energy_composition.png")
    plt.close()

def plot_frequency_distribution(df):
    """
    图4: 最优频率选择分布（SLO×1.1）
    """
    slo_11_data = df[df['slo_mult'] == 1.1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, tp in enumerate(sorted(slo_11_data['tp'].unique())):
        tp_data = slo_11_data[slo_11_data['tp'] == tp]

        # 统计 (p_freq, d_freq) 组合
        freq_pairs = tp_data.groupby(['p_freq', 'd_freq']).size().reset_index(name='count')
        freq_pairs = freq_pairs.sort_values('count', ascending=False).head(10)

        # 创建标签
        labels = [f'P:{p}MHz\nD:{d}MHz' for p, d in zip(freq_pairs['p_freq'], freq_pairs['d_freq'])]

        axes[idx].barh(range(len(freq_pairs)), freq_pairs['count'], color='steelblue', alpha=0.7)
        axes[idx].set_yticks(range(len(freq_pairs)))
        axes[idx].set_yticklabels(labels, fontsize=8)
        axes[idx].set_xlabel('Count', fontsize=10)
        axes[idx].set_title(f'TP={tp}: Top 10 Freq Pairs', fontsize=11, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='x')
        axes[idx].invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_freq_distribution.png'), dpi=300, bbox_inches='tight')
    print(f"已保存: {OUTPUT_DIR}/fig4_freq_distribution.png")
    plt.close()

def plot_heatmap_savings(df):
    """
    图5: 热力图 - 不同 (TP, SLO) 下的平均节能
    """
    # 计算每个 (tp, slo_mult) 的平均节能
    heatmap_data = df.groupby(['tp', 'slo_mult'])['saving_pct'].mean().reset_index()
    pivot = heatmap_data.pivot(index='tp', columns='slo_mult', values='saving_pct')

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=20)

    # 设置刻度
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([f'{x:.1f}' for x in pivot.columns])
    ax.set_yticklabels([f'TP={int(x)}' for x in pivot.index])

    # 添加数值标签
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            text = ax.text(j, i, f'{pivot.values[i, j]:.1f}%',
                          ha="center", va="center", color="black", fontsize=11, fontweight='bold')

    ax.set_xlabel('SLO Multiplier', fontsize=12)
    ax.set_ylabel('Tensor Parallelism', fontsize=12)
    ax.set_title('PD Separation: Average Energy Savings Heatmap', fontsize=14, fontweight='bold')

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Energy Saving (%)', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_heatmap.png'), dpi=300, bbox_inches='tight')
    print(f"已保存: {OUTPUT_DIR}/fig5_heatmap.png")
    plt.close()

def plot_comparison_with_af(df):
    """
    图6: PD 分离 vs AF 分离对比
    """
    # AF 分离的典型数据（来自论文）
    af_data = {
        'tp': [1, 2, 4, 8],
        'slo_1.0': [5.0, 8.3, 9.2, 7.8],
        'slo_1.1': [3.2, 3.9, 3.3, 4.7],
        'slo_1.5': [2.3, 2.8, 3.1, 4.2]
    }

    # PD 分离数据
    pd_data = {
        'tp': [],
        'slo_1.0': [],
        'slo_1.1': [],
        'slo_1.5': []
    }

    for tp in sorted(df['tp'].unique()):
        pd_data['tp'].append(tp)
        for slo in [1.0, 1.1, 1.5]:
            slo_data = df[(df['tp'] == tp) & (df['slo_mult'] == slo)]['saving_pct']
            pd_data[f'slo_{slo}'].append(slo_data.mean())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    slo_labels = ['SLO×1.0', 'SLO×1.1', 'SLO×1.5']
    slo_keys = ['slo_1.0', 'slo_1.1', 'slo_1.5']

    for idx, (label, key) in enumerate(zip(slo_labels, slo_keys)):
        x = np.arange(len(af_data['tp']))
        width = 0.35

        axes[idx].bar(x - width/2, af_data[key], width, label='AF Separation', alpha=0.8, color='steelblue')
        axes[idx].bar(x + width/2, pd_data[key], width, label='PD Separation', alpha=0.8, color='coral')

        axes[idx].set_xlabel('Tensor Parallelism', fontsize=11)
        axes[idx].set_ylabel('Energy Saving (%)', fontsize=11)
        axes[idx].set_title(label, fontsize=12, fontweight='bold')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels([f'TP={tp}' for tp in af_data['tp']])
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3, axis='y')
        axes[idx].set_ylim(0, max(max(af_data[key]), max(pd_data[key])) * 1.2)

    plt.suptitle('PD Separation vs AF Separation: Energy Savings Comparison',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig6_pd_vs_af.png'), dpi=300, bbox_inches='tight')
    print(f"已保存: {OUTPUT_DIR}/fig6_pd_vs_af.png")
    plt.close()

def generate_summary_table(df):
    """生成汇总表格"""
    summary = []

    for slo in sorted(df['slo_mult'].unique()):
        slo_data = df[df['slo_mult'] == slo]

        for tp in sorted(slo_data['tp'].unique()):
            tp_data = slo_data[slo_data['tp'] == tp]['saving_pct']

            summary.append({
                'SLO': f'×{slo}',
                'TP': tp,
                'Mean (%)': f'{tp_data.mean():.2f}',
                'Median (%)': f'{tp_data.median():.2f}',
                'Max (%)': f'{tp_data.max():.2f}',
                'Sweet Spot (%)': f'{(tp_data > 5).sum() / len(tp_data) * 100:.1f}',
                'Configs': len(tp_data)
            })

    summary_df = pd.DataFrame(summary)
    output_file = os.path.join(RESULTS_DIR, 'summary_table.csv')
    summary_df.to_csv(output_file, index=False)
    print(f"\n汇总表格已保存: {output_file}")

    return summary_df

def main():
    print("="*80)
    print("PD 分离结果可视化")
    print("="*80)

    # 加载数据
    df = load_all_results()

    if df is None or len(df) == 0:
        print("错误: 未找到结果数据")
        return

    print(f"\n加载了 {len(df)} 条记录")
    print(f"SLO 范围: {sorted(df['slo_mult'].unique())}")
    print(f"TP 范围: {sorted(df['tp'].unique())}")

    # 生成图表
    print("\n生成图表...")

    plot_slo_curve(df)
    plot_tp_comparison(df)
    plot_energy_composition(df)
    plot_frequency_distribution(df)
    plot_heatmap_savings(df)
    plot_comparison_with_af(df)

    # 生成汇总表
    print("\n生成汇总表...")
    summary_df = generate_summary_table(df)
    print("\n汇总表预览:")
    print(summary_df.to_string(index=False))

    print("\n" + "="*80)
    print(f"所有图表已保存到: {OUTPUT_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()
