#!/usr/bin/env python3
"""Generate workload CDF plot for MoE benchmark datasets (same style as 8gpu_azure_workload_cdf.png)."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

WORKLOAD_DIR = Path("/workspace/sglang-tier/benchmark/AFlex_bench/02_macro_benchmark/workloads")
OUT_DIR = Path(__file__).resolve().parent / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WORKLOADS = {
    'Code Medium': 'workload_azure_code_medium_real.jsonl',
    'Conv Light': 'workload_azure_conv_light_real.jsonl',
    'Conv Medium': 'workload_azure_conv_medium_real.jsonl',
    'Conv Heavy': 'workload_azure_conv_heavy_real.jsonl',
}

colors = ['#4472C4', '#ED7D31', '#70AD47', '#FFC000']
linestyles = ['-', '--', '-.', ':']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Azure Workload Characteristics (CDF)", fontsize=13, fontweight='bold')

for wi, (wl_name, wl_file) in enumerate(WORKLOADS.items()):
    fpath = WORKLOAD_DIR / wl_file
    input_lens = []
    output_lens = []
    arrival_times = []
    with open(fpath) as f:
        for line in f:
            d = json.loads(line)
            input_lens.append(d['input_len'])
            output_lens.append(d['output_len'])
            arrival_times.append(d['arrival_time_s'])

    input_lens = np.array(input_lens)
    output_lens = np.array(output_lens)
    arrival_times = np.array(arrival_times)

    # Input length CDF
    ax = axes[0]
    sorted_il = np.sort(input_lens)
    cdf = np.arange(1, len(sorted_il) + 1) / len(sorted_il)
    ax.plot(sorted_il, cdf, color=colors[wi], linestyle=linestyles[wi],
            linewidth=1.8, label=f'{wl_name} (n={len(input_lens)})')

    # Output length CDF
    ax = axes[1]
    sorted_ol = np.sort(output_lens)
    cdf = np.arange(1, len(sorted_ol) + 1) / len(sorted_ol)
    ax.plot(sorted_ol, cdf, color=colors[wi], linestyle=linestyles[wi],
            linewidth=1.8, label=wl_name)

    # Inter-arrival time CDF
    ax = axes[2]
    if len(arrival_times) > 1:
        iat = np.diff(np.sort(arrival_times))
        sorted_iat = np.sort(iat)
        cdf = np.arange(1, len(sorted_iat) + 1) / len(sorted_iat)
        ax.plot(sorted_iat, cdf, color=colors[wi], linestyle=linestyles[wi],
                linewidth=1.8, label=wl_name)

# Format axes
axes[0].set_xlabel('Input Length (tokens)', fontsize=10)
axes[0].set_ylabel('CDF', fontsize=10)
axes[0].set_title('Input Length Distribution', fontsize=11, fontweight='bold')
axes[0].legend(fontsize=8, loc='lower right')
axes[0].grid(alpha=0.3)
axes[0].set_xlim(left=0)

axes[1].set_xlabel('Output Length (tokens)', fontsize=10)
axes[1].set_ylabel('CDF', fontsize=10)
axes[1].set_title('Output Length Distribution', fontsize=11, fontweight='bold')
axes[1].legend(fontsize=8, loc='lower right')
axes[1].grid(alpha=0.3)
axes[1].set_xlim(left=0)

axes[2].set_xlabel('Inter-arrival Time (s)', fontsize=10)
axes[2].set_ylabel('CDF', fontsize=10)
axes[2].set_title('Request Inter-arrival Time', fontsize=11, fontweight='bold')
axes[2].legend(fontsize=8, loc='lower right')
axes[2].grid(alpha=0.3)
axes[2].set_xlim(left=0)

plt.tight_layout(rect=[0, 0, 1, 0.94])
save_path = OUT_DIR / "8gpu_moe_workload_cdf.png"
fig.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Saved: {save_path}")

# Print summary
print("\n=== Workload Summary ===")
for wl_name, wl_file in WORKLOADS.items():
    fpath = WORKLOAD_DIR / wl_file
    ils, ols = [], []
    with open(fpath) as f:
        for line in f:
            d = json.loads(line)
            ils.append(d['input_len'])
            ols.append(d['output_len'])
    ils, ols = np.array(ils), np.array(ols)
    print(f"  {wl_name:15s}: n={len(ils):4d}, IL avg={ils.mean():.0f} p50={np.median(ils):.0f} p99={np.percentile(ils,99):.0f}, OL avg={ols.mean():.0f} p50={np.median(ols):.0f} p99={np.percentile(ols,99):.0f}")
