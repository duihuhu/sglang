#!/usr/bin/env python3
"""Plot Mixtral-8x7B 8-GPU benchmark results: Baseline vs Tier, 3 architectures."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data from benchmark results
scenarios = ['chatbot', 'qa', 'rag', 'summary']
qps_list = [1, 3, 5]

# Baseline results
baseline = {
    'native_dp': {
        'chatbot': {'thpt': [872.3, 1733.0, 1914.0], 'ttft': [78.0, 76.7, 76.9], 'tpot': [36.2, 42.6, 43.0], 'mjpt': [2664.6, 1464.7, 1359.9]},
        'qa': {'thpt': [246.2, 683.0, 1041.9], 'ttft': [68.0, 67.9, 74.8], 'tpot': [32.8, 33.6, 40.7], 'mjpt': [6956.1, 3185.3, 2258.4]},
        'rag': {'thpt': [63.3, 186.2, 304.2], 'ttft': [70.6, 68.1, 67.8], 'tpot': [31.9, 33.3, 34.1], 'mjpt': [17435.3, 8309.0, 5917.4]},
        'summary': {'thpt': [63.3, 186.3, 304.0], 'ttft': [71.0, 68.3, 68.0], 'tpot': [32.1, 33.4, 34.3], 'mjpt': [17867.3, 8502.0, 6147.1]},
    },
    'pd_dp': {
        'chatbot': {'thpt': [859.0, 1623.1, 1793.4], 'ttft': [38.9, 36.7, 38.4], 'tpot': [42.0, 46.1, 46.5], 'mjpt': [1901.4, 1034.6, 944.2]},
        'qa': {'thpt': [245.4, 673.8, 1014.0], 'ttft': [38.2, 38.2, 40.0], 'tpot': [34.8, 45.1, 47.2], 'mjpt': [5571.9, 2468.2, 1646.7]},
        'rag': {'thpt': [63.3, 185.9, 303.9], 'ttft': [38.3, 38.5, 39.1], 'tpot': [33.4, 33.9, 35.6], 'mjpt': [15161.6, 6753.7, 4695.7]},
        'summary': {'thpt': [63.3, 186.2, 303.6], 'ttft': [38.2, 37.8, 40.8], 'tpot': [33.5, 34.0, 36.2], 'mjpt': [16034.9, 7042.6, 4939.9]},
    },
    'pdaf': {
        'chatbot': {'thpt': [742.7, 925.1, 981.1], 'ttft': [71.5, 70.6, 71.2], 'tpot': [85.6, 92.8, 92.7], 'mjpt': [1304.6, 1077.6, 1022.4]},
        'qa': {'thpt': [240.3, 613.1, 828.7], 'ttft': [85.4, 87.1, 90.4], 'tpot': [62.1, 76.5, 84.9], 'mjpt': [4194.8, 1752.8, 1347.8]},
        'rag': {'thpt': [62.9, 182.6, 293.2], 'ttft': [152.7, 178.8, 246.5], 'tpot': [53.1, 63.0, 65.9], 'mjpt': [16119.4, 6718.1, 4632.9]},
        'summary': {'thpt': [62.9, 182.4, 204.4], 'ttft': [265.4, 573.5, 1147.6], 'tpot': [52.6, 62.5, 62.9], 'mjpt': [15823.9, 7448.5, 7082.0]},
    },
}

# Tier results
tier = {
    'native_dp': {
        'chatbot': {'thpt': [878.2, 1737.9, 1934.8], 'ttft': [78.1, 76.6, 76.1], 'tpot': [35.9, 42.6, 42.9], 'mjpt': [2619.7, 1461.6, 1340.6]},
        'qa': {'thpt': [246.0, 682.0, 1035.3], 'ttft': [68.1, 68.4, 74.3], 'tpot': [32.8, 33.8, 40.5], 'mjpt': [6819.7, 3179.7, 2358.0]},
        'rag': {'thpt': [63.3, 186.2, 304.2], 'ttft': [74.9, 70.1, 68.3], 'tpot': [33.3, 33.4, 34.2], 'mjpt': [15374.5, 7993.9, 5787.2]},
        'summary': {'thpt': [63.3, 185.8, 304.7], 'ttft': [76.8, 69.1, 68.7], 'tpot': [33.4, 33.7, 34.5], 'mjpt': [15746.8, 8250.7, 5995.4]},
    },
    'pd_dp': {
        'chatbot': {'thpt': [859.3, 1639.5, 1800.9], 'ttft': [38.8, 36.9, 38.3], 'tpot': [41.6, 45.9, 46.1], 'mjpt': [1842.6, 983.1, 895.6]},
        'qa': {'thpt': [246.1, 672.8, 1007.3], 'ttft': [36.4, 36.0, 39.4], 'tpot': [33.9, 44.9, 47.0], 'mjpt': [5440.0, 2432.7, 1623.0]},
        'rag': {'thpt': [63.3, 186.1, 303.9], 'ttft': [37.5, 38.5, 38.3], 'tpot': [32.6, 32.9, 33.8], 'mjpt': [14354.4, 6465.6, 4480.4]},
        'summary': {'thpt': [63.3, 185.9, 303.1], 'ttft': [38.0, 38.1, 41.4], 'tpot': [32.5, 33.5, 35.3], 'mjpt': [14518.7, 6769.1, 4695.6]},
    },
    'pdaf': {
        'chatbot': {'thpt': [588.8, 697.6, 736.1], 'ttft': [169.3, 189.6, 219.6], 'tpot': [125.4, 127.6, 126.0], 'mjpt': [1104.8, 924.6, 872.8]},
        'qa': {'thpt': [233.7, 558.3, 645.0], 'ttft': [293.9, 519.9, 947.3], 'tpot': [89.4, 111.6, 114.7], 'mjpt': [2878.6, 1251.9, 1070.4]},
        'rag': {'thpt': [62.5, 71.5, 70.6], 'ttft': [1652.7, 6649.0, 6749.9], 'tpot': [63.5, 66.7, 66.8], 'mjpt': [11082.1, 9828.2, 9943.2]},
        'summary': {'thpt': [31.7, 0.3, None], 'ttft': [6840.7, 337799.7, None], 'tpot': [64.9, 0.0, None], 'mjpt': [21670.2, 2007302.2, None]},
    },
}

colors = {'native_dp': '#2196F3', 'pd_dp': '#4CAF50', 'pdaf': '#FF9800'}
labels = {'native_dp': 'Native DP4', 'pd_dp': 'PD 2P+2D', 'pdaf': 'PDAF'}

# ============================================================
# Figure 1: Baseline comparison (QPS=3, all scenarios)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Mixtral-8x7B 8-GPU Baseline (QPS=3)', fontsize=14, fontweight='bold')

metrics = [('thpt', 'Throughput (tok/s)'), ('ttft', 'TTFT (ms)'), ('tpot', 'TPOT (ms)'), ('mjpt', 'Energy (mJ/tok)')]
for ax_idx, (metric, ylabel) in enumerate(metrics):
    ax = axes[ax_idx // 2][ax_idx % 2]
    x = np.arange(len(scenarios))
    width = 0.25
    for i, arch in enumerate(['native_dp', 'pd_dp', 'pdaf']):
        vals = [baseline[arch][s][metric][1] for s in scenarios]  # QPS=3
        ax.bar(x + i * width, vals, width, label=labels[arch], color=colors[arch], alpha=0.85)
    ax.set_xlabel('Scenario')
    ax.set_ylabel(ylabel)
    ax.set_xticks(x + width)
    ax.set_xticklabels(scenarios)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/sglang/benchmark/AFlex_bench/06_others/Mixtral_test/micro_benchmark/8gpu/baseline_qps3.png', dpi=150)
plt.close()

# ============================================================
# Figure 2: Baseline vs Tier energy saving (QPS=3)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Mixtral-8x7B 8-GPU: Baseline vs Tier Energy (QPS=3)', fontsize=14, fontweight='bold')

# Energy comparison bar
ax = axes[0]
x = np.arange(len(scenarios))
width = 0.13
for i, arch in enumerate(['native_dp', 'pd_dp', 'pdaf']):
    base_vals = [baseline[arch][s]['mjpt'][1] for s in scenarios]
    tier_vals = [tier[arch][s]['mjpt'][1] if tier[arch][s]['mjpt'][1] and tier[arch][s]['mjpt'][1] < 50000 else None for s in scenarios]
    ax.bar(x + i * width * 2, base_vals, width, label=f'{labels[arch]} Base', color=colors[arch], alpha=0.6)
    tier_plot = [v if v else 0 for v in tier_vals]
    ax.bar(x + i * width * 2 + width, tier_plot, width, label=f'{labels[arch]} Tier', color=colors[arch], alpha=1.0)
ax.set_xlabel('Scenario')
ax.set_ylabel('mJ/tok')
ax.set_xticks(x + 0.2)
ax.set_xticklabels(scenarios)
ax.legend(fontsize=8, ncol=2)
ax.grid(axis='y', alpha=0.3)
ax.set_title('Energy per Token')

# Energy saving percentage (exclude PDAF rag/summary where it breaks)
ax = axes[1]
for arch in ['native_dp', 'pd_dp', 'pdaf']:
    savings = []
    valid_scenarios = []
    for s in scenarios:
        b = baseline[arch][s]['mjpt'][1]
        t = tier[arch][s]['mjpt'][1]
        if t and t < 50000 and b:
            savings.append((b - t) / b * 100)
            valid_scenarios.append(s)
    if savings:
        ax.plot(valid_scenarios, savings, 'o-', label=labels[arch], color=colors[arch], linewidth=2, markersize=8)
ax.set_xlabel('Scenario')
ax.set_ylabel('Energy Saving (%)')
ax.set_title('Tier Energy Saving vs Baseline')
ax.legend()
ax.grid(alpha=0.3)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('/workspace/sglang/benchmark/AFlex_bench/06_others/Mixtral_test/micro_benchmark/8gpu/baseline_vs_tier.png', dpi=150)
plt.close()

# ============================================================
# Figure 3: Throughput vs QPS for all architectures
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Mixtral-8x7B 8-GPU: Throughput vs QPS (Baseline vs Tier)', fontsize=14, fontweight='bold')

for s_idx, scenario in enumerate(scenarios):
    ax = axes[s_idx // 2][s_idx % 2]
    for arch in ['native_dp', 'pd_dp', 'pdaf']:
        base_thpt = baseline[arch][scenario]['thpt']
        tier_thpt = [v for v in tier[arch][scenario]['thpt'] if v is not None]
        tier_qps = qps_list[:len(tier_thpt)]
        ax.plot(qps_list, base_thpt, 'o-', label=f'{labels[arch]} Base', color=colors[arch], alpha=0.6, linewidth=2)
        ax.plot(tier_qps, tier_thpt, 's--', label=f'{labels[arch]} Tier', color=colors[arch], alpha=1.0, linewidth=2)
    ax.set_xlabel('QPS')
    ax.set_ylabel('Throughput (tok/s)')
    ax.set_title(scenario)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/sglang/benchmark/AFlex_bench/06_others/Mixtral_test/micro_benchmark/8gpu/thpt_vs_qps.png', dpi=150)
plt.close()

# ============================================================
# Figure 4: TTFT comparison (Baseline vs Tier, QPS=3)
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
fig.suptitle('Mixtral-8x7B 8-GPU: TTFT Comparison (QPS=3)', fontsize=14, fontweight='bold')

x = np.arange(len(scenarios))
width = 0.13
for i, arch in enumerate(['native_dp', 'pd_dp', 'pdaf']):
    base_ttft = [baseline[arch][s]['ttft'][1] for s in scenarios]
    tier_ttft = [tier[arch][s]['ttft'][1] if tier[arch][s]['ttft'][1] and tier[arch][s]['ttft'][1] < 1000 else None for s in scenarios]
    ax.bar(x + i * width * 2, base_ttft, width, label=f'{labels[arch]} Base', color=colors[arch], alpha=0.6)
    tier_plot = [v if v else 0 for v in tier_ttft]
    ax.bar(x + i * width * 2 + width, tier_plot, width, label=f'{labels[arch]} Tier', color=colors[arch], alpha=1.0)

ax.set_xlabel('Scenario')
ax.set_ylabel('TTFT (ms)')
ax.set_xticks(x + 0.2)
ax.set_xticklabels(scenarios)
ax.legend(fontsize=8, ncol=2)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/sglang/benchmark/AFlex_bench/06_others/Mixtral_test/micro_benchmark/8gpu/ttft_comparison.png', dpi=150)
plt.close()

print("All plots saved to micro_benchmark/8gpu/")
