#!/usr/bin/env python3
"""
PD 分离数据的 SLO 和甜点分析
采用与 AF 分离论文相同的分析方法
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import os

# 数据路径
DATA_DIR = "/workspace/benchmark/sglang-main/bash-test/hucc/data"
P_DATA = os.path.join(DATA_DIR, "P_data.csv")
D_DATA = os.path.join(DATA_DIR, "D_data.csv")

def load_data():
    """加载 Prefill 和 Decode 数据"""
    # Prefill 数据
    p_df = pd.read_csv(P_DATA, skiprows=1)
    p_df.columns = ['tp', 'input_len', 'gpu_clock', 'batch_size', 'A', 'F', 'TTFT_ms', 'AF_64_ms', 'A_energy_mj', 'F_energy_mj']

    # Decode 数据
    d_df = pd.read_csv(D_DATA, skiprows=1)
    d_df.columns = ['tp', 'input_len', 'output_len', 'gpu_clock', 'batch_size', 'A', 'F', 'TPOT_ms', 'AF_64_ms', 'A_energy_mj', 'F_energy_mj']

    # 计算总延迟和总能耗
    p_df['total_latency'] = p_df['A'] + p_df['F']
    p_df['total_energy'] = p_df['A_energy_mj'] + p_df['F_energy_mj']

    d_df['total_latency'] = d_df['A'] + d_df['F']
    d_df['total_energy'] = d_df['A_energy_mj'] + d_df['F_energy_mj']

    return p_df, d_df

def analyze_frequency_sensitivity(df, stage_name):
    """分析频率敏感性（类似 A_ratio 分析）"""
    print(f"\n{'='*80}")
    print(f"{stage_name} 阶段频率敏感性分析")
    print(f"{'='*80}")

    freqs = sorted(df['gpu_clock'].unique())
    min_freq = min(freqs)
    max_freq = max(freqs)

    print(f"\n频率范围: {min_freq} - {max_freq} MHz")

    # 计算 A 和 F 的频率敏感性比率
    results = []

    for tp in sorted(df['tp'].unique()):
        for bs in sorted(df['batch_size'].unique()):
            tp_bs_data = df[(df['tp'] == tp) & (df['batch_size'] == bs)]

            if len(tp_bs_data) < 2:
                continue

            # 获取最低频和最高频的数据
            min_freq_data = tp_bs_data[tp_bs_data['gpu_clock'] == min_freq]
            max_freq_data = tp_bs_data[tp_bs_data['gpu_clock'] == max_freq]

            if len(min_freq_data) == 0 or len(max_freq_data) == 0:
                continue

            # 计算平均值
            A_ratio = min_freq_data['A'].mean() / max_freq_data['A'].mean()
            F_ratio = min_freq_data['F'].mean() / max_freq_data['F'].mean()

            results.append({
                'tp': tp,
                'batch_size': bs,
                'A_ratio': A_ratio,
                'F_ratio': F_ratio
            })

    if results:
        ratio_df = pd.DataFrame(results)

        print(f"\nA 延迟频率敏感性 (A_lat@{min_freq}MHz / A_lat@{max_freq}MHz):")
        print("越接近 1.0 = 越 memory-bound，降频影响小")
        pivot_a = ratio_df.pivot_table(values='A_ratio', index='tp', columns='batch_size')
        print(pivot_a.to_string(float_format=lambda x: f"{x:.3f}"))

        print(f"\nF 延迟频率敏感性 (F_lat@{min_freq}MHz / F_lat@{max_freq}MHz):")
        pivot_f = ratio_df.pivot_table(values='F_ratio', index='tp', columns='batch_size')
        print(pivot_f.to_string(float_format=lambda x: f"{x:.3f}"))

    return results

def find_optimal_freq_unified(group, slo_multiplier):
    """
    统一调频策略：选择满足 SLO 的最低能耗频率
    对应论文中的 OptB (min-energy unified)
    """
    base_latency = group['total_latency'].min()
    slo = base_latency * slo_multiplier

    # 筛选满足 SLO 的配置
    feasible = group[group['total_latency'] <= slo]

    if len(feasible) == 0:
        return None

    # 选择能耗最低的
    optimal = feasible.loc[feasible['total_energy'].idxmin()]
    return optimal

def find_optimal_freq_pd_separate(p_group, d_group, slo_multiplier):
    """
    PD 分离调频策略：P 和 D 各自独立选频，最小化总能耗
    类似论文中的 AF 分离策略
    """
    # Prefill 的 base latency 和 SLO
    p_base_latency = p_group['total_latency'].min()
    p_slo = p_base_latency * slo_multiplier

    # Decode 的 base latency 和 SLO
    d_base_latency = d_group['total_latency'].min()
    d_slo = d_base_latency * slo_multiplier

    # Prefill 可行频率
    p_feasible = p_group[p_group['total_latency'] <= p_slo]
    if len(p_feasible) == 0:
        return None, None

    # Decode 可行频率
    d_feasible = d_group[d_group['total_latency'] <= d_slo]
    if len(d_feasible) == 0:
        return None, None

    # 选择各自能耗最低的频率
    p_optimal = p_feasible.loc[p_feasible['total_energy'].idxmin()]
    d_optimal = d_feasible.loc[d_feasible['total_energy'].idxmin()]

    return p_optimal, d_optimal

def analyze_slo_savings(p_df, d_df, stage='decode', slo_multipliers=[1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0]):
    """
    分析不同 SLO 下的能耗节省
    对比 PD 分离 vs 统一调频
    """
    print(f"\n{'='*80}")
    print(f"{stage.upper()} 阶段 SLO 分析: PD 分离 vs 统一调频")
    print(f"{'='*80}")

    if stage == 'decode':
        df = d_df
        group_cols = ['tp', 'input_len', 'output_len', 'batch_size']
    else:  # prefill
        df = p_df
        group_cols = ['tp', 'input_len', 'batch_size']

    results = []

    for name, group in df.groupby(group_cols):
        if len(group) < 2:
            continue

        config = dict(zip(group_cols, name))

        for slo_mult in slo_multipliers:
            # 统一调频策略
            unified_opt = find_optimal_freq_unified(group, slo_mult)

            if unified_opt is None:
                continue

            # PD 分离策略（这里简化为单独优化，实际应该考虑 P+D 联合）
            # 对于单阶段分析，PD 分离等价于 P 和 D 各自独立优化
            pd_opt = unified_opt  # 占位，后续实现端到端分析

            result = {
                **config,
                'slo_mult': slo_mult,
                'unified_freq': unified_opt['gpu_clock'],
                'unified_latency': unified_opt['total_latency'],
                'unified_energy': unified_opt['total_energy'],
                'pd_freq': pd_opt['gpu_clock'],
                'pd_latency': pd_opt['total_latency'],
                'pd_energy': pd_opt['total_energy'],
                'saving_pct': (unified_opt['total_energy'] - pd_opt['total_energy']) / unified_opt['total_energy'] * 100
            }
            results.append(result)

    if not results:
        print("没有足够的数据进行分析")
        return None

    result_df = pd.DataFrame(results)

    # 按 TP 和 SLO 汇总
    print("\n按 TP 和 SLO 汇总的平均节能百分比:")
    summary = result_df.groupby(['tp', 'slo_mult'])['saving_pct'].mean().reset_index()
    pivot = summary.pivot(index='tp', columns='slo_mult', values='saving_pct')
    print(pivot.to_string(float_format=lambda x: f"{x:.1f}%"))

    return result_df

def analyze_sweet_spot(p_df, d_df, slo_multiplier=1.0, threshold=5.0):
    """
    甜点分析：找出 PD 分离收益 > threshold% 的配置
    """
    print(f"\n{'='*80}")
    print(f"甜点分析 (SLO×{slo_multiplier}, 收益阈值 > {threshold}%)")
    print(f"{'='*80}")

    # Decode 阶段甜点分析
    print("\n[Decode 阶段]")
    d_results = []

    for name, group in d_df.groupby(['tp', 'input_len', 'output_len', 'batch_size']):
        if len(group) < 2:
            continue

        tp, il, ol, bs = name

        unified_opt = find_optimal_freq_unified(group, slo_multiplier)
        if unified_opt is None:
            continue

        # 简化：PD 分离在单阶段等价于统一优化
        # 真正的收益来自 P 和 D 的联合优化
        saving = 0.0  # 占位

        if saving > threshold:
            d_results.append({
                'tp': tp,
                'input_len': il,
                'output_len': ol,
                'batch_size': bs,
                'saving_pct': saving,
                'unified_freq': unified_opt['gpu_clock'],
                'unified_energy': unified_opt['total_energy']
            })

    if d_results:
        sweet_df = pd.DataFrame(d_results)
        print(f"\n找到 {len(sweet_df)} 个甜点配置")

        # 按 TP 统计
        tp_stats = sweet_df.groupby('tp').agg({
            'saving_pct': ['count', 'mean', 'max']
        }).round(1)
        print("\n按 TP 统计:")
        print(tp_stats)
    else:
        print("\n未找到满足条件的甜点配置")

    return d_results

def analyze_end_to_end(p_df, d_df, slo_multiplier=1.0, output_len=64):
    """
    端到端分析：E_total = E_prefill + E_decode × output_len
    这里才能真正体现 PD 分离的价值
    """
    print(f"\n{'='*80}")
    print(f"端到端分析 (SLO×{slo_multiplier}, output_len={output_len})")
    print(f"{'='*80}")

    results = []

    # 遍历所有 (tp, input_len, batch_size) 组合
    for tp in sorted(p_df['tp'].unique()):
        for il in sorted(p_df['input_len'].unique()):
            for bs in sorted(p_df['batch_size'].unique()):
                # Prefill 数据
                p_group = p_df[(p_df['tp'] == tp) &
                               (p_df['input_len'] == il) &
                               (p_df['batch_size'] == bs)]

                # Decode 数据
                d_group = d_df[(d_df['tp'] == tp) &
                               (d_df['input_len'] == il) &
                               (d_df['output_len'] == output_len) &
                               (d_df['batch_size'] == bs)]

                if len(p_group) == 0 or len(d_group) == 0:
                    continue

                # 统一调频：P 和 D 必须用相同频率
                unified_results = []
                for freq in sorted(p_group['gpu_clock'].unique()):
                    p_data = p_group[p_group['gpu_clock'] == freq]
                    d_data = d_group[d_group['gpu_clock'] == freq]

                    if len(p_data) == 0 or len(d_data) == 0:
                        continue

                    p_lat = p_data['total_latency'].iloc[0]
                    d_lat = d_data['total_latency'].iloc[0]
                    p_energy = p_data['total_energy'].iloc[0]
                    d_energy = d_data['total_energy'].iloc[0]

                    # 检查是否满足 SLO
                    p_base = p_group['total_latency'].min()
                    d_base = d_group['total_latency'].min()

                    if p_lat <= p_base * slo_multiplier and d_lat <= d_base * slo_multiplier:
                        total_energy = p_energy + d_energy * output_len
                        unified_results.append({
                            'freq': freq,
                            'total_energy': total_energy,
                            'p_latency': p_lat,
                            'd_latency': d_lat
                        })

                if not unified_results:
                    continue

                # 选择能耗最低的统一频率
                unified_opt = min(unified_results, key=lambda x: x['total_energy'])

                # PD 分离：P 和 D 各自独立选频
                p_opt, d_opt = find_optimal_freq_pd_separate(p_group, d_group, slo_multiplier)

                if p_opt is None or d_opt is None:
                    continue

                pd_total_energy = p_opt['total_energy'] + d_opt['total_energy'] * output_len

                saving_pct = (unified_opt['total_energy'] - pd_total_energy) / unified_opt['total_energy'] * 100

                results.append({
                    'tp': tp,
                    'input_len': il,
                    'batch_size': bs,
                    'unified_freq': unified_opt['freq'],
                    'unified_energy': unified_opt['total_energy'],
                    'p_freq': p_opt['gpu_clock'],
                    'd_freq': d_opt['gpu_clock'],
                    'pd_energy': pd_total_energy,
                    'saving_pct': saving_pct,
                    'p_energy_ratio': p_opt['total_energy'] / unified_opt['total_energy'] * 100,
                    'd_energy_ratio': d_opt['total_energy'] * output_len / unified_opt['total_energy'] * 100
                })

    if not results:
        print("没有足够的数据进行端到端分析")
        return None

    result_df = pd.DataFrame(results)

    # 汇总统计
    print("\n按 TP 汇总的端到端节能:")
    tp_summary = result_df.groupby('tp').agg({
        'saving_pct': ['mean', 'median', 'max', 'min'],
        'p_energy_ratio': 'mean',
        'd_energy_ratio': 'mean'
    }).round(2)
    print(tp_summary)

    print("\n各 TP 的详细统计:")
    for tp in sorted(result_df['tp'].unique()):
        tp_data = result_df[result_df['tp'] == tp]
        print(f"\nTP={tp}:")
        print(f"  配置数: {len(tp_data)}")
        print(f"  平均节能: {tp_data['saving_pct'].mean():.2f}%")
        print(f"  中位数: {tp_data['saving_pct'].median():.2f}%")
        print(f"  最大节能: {tp_data['saving_pct'].max():.2f}%")
        print(f"  Prefill 占总能耗: {tp_data['p_energy_ratio'].mean():.1f}%")
        print(f"  Decode 占总能耗: {tp_data['d_energy_ratio'].mean():.1f}%")

        # 甜点配置
        sweet = tp_data[tp_data['saving_pct'] > 5.0]
        if len(sweet) > 0:
            print(f"  甜点配置数 (>5%): {len(sweet)} ({len(sweet)/len(tp_data)*100:.1f}%)")
            print(f"  甜点平均节能: {sweet['saving_pct'].mean():.2f}%")

    return result_df

def main():
    print("="*80)
    print("PD 分离数据 SLO 和甜点分析")
    print("="*80)

    # 加载数据
    print("\n加载数据...")
    p_df, d_df = load_data()

    print(f"Prefill 数据: {len(p_df)} 行")
    print(f"  TP: {sorted(p_df['tp'].unique())}")
    print(f"  Freq: {sorted(p_df['gpu_clock'].unique())} MHz")
    print(f"  Input lengths: {sorted(p_df['input_len'].unique())}")
    print(f"  Batch sizes: {sorted(p_df['batch_size'].unique())}")

    print(f"\nDecode 数据: {len(d_df)} 行")
    print(f"  TP: {sorted(d_df['tp'].unique())}")
    print(f"  Freq: {sorted(d_df['gpu_clock'].unique())} MHz")
    print(f"  Input lengths: {sorted(d_df['input_len'].unique())}")
    print(f"  Output lengths: {sorted(d_df['output_len'].unique())}")
    print(f"  Batch sizes: {sorted(d_df['batch_size'].unique())}")

    # 1. 频率敏感性分析
    analyze_frequency_sensitivity(p_df, "Prefill")
    analyze_frequency_sensitivity(d_df, "Decode")

    # 2. 端到端分析（最重要）
    print("\n" + "="*80)
    print("核心分析：端到端 PD 分离收益")
    print("="*80)

    for slo_mult in [1.0, 1.1, 1.2, 1.5, 2.0]:
        print(f"\n{'='*80}")
        print(f"SLO × {slo_mult}")
        print(f"{'='*80}")
        e2e_df = analyze_end_to_end(p_df, d_df, slo_multiplier=slo_mult, output_len=64)

        if e2e_df is not None and len(e2e_df) > 0:
            # 保存结果
            output_file = f"/workspace/benchmark/sglang-main/bash-test/hucc/results_slo_{slo_mult}.csv"
            e2e_df.to_csv(output_file, index=False)
            print(f"\n结果已保存到: {output_file}")

    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)

if __name__ == "__main__":
    main()
