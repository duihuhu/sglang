#!/usr/bin/env python3
"""Generate workload CDF plot: Input Length, Output Length, and QPS distributions."""
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
    arrival_times = np.sort(np.array(arrival_times))

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

    # QPS CDF (requests per second, using 1-second sliding window)
    ax = axes[2]
    if len(arrival_times) > 1:
        window_size = 1.0  # 1 second window
        total_duration = arrival_times[-1] - arrival_times[0]
        # Count requests in each 1-second bin
        bin_start = arrival_times[0]
        bin_end = arrival_times[-1]
        n_bins = int(np.ceil(total_duration / window_size))
        if n_bins > 0:
            qps_values = []
            for i in range(n_bins):
                t_start = bin_start + i * window_size
                t_end = t_start + window_size
                count = np.sum((arrival_times >= t_start) & (arrival_times < t_end))
                qps_values.append(count)
            qps_values = np.array(qps_values)
            sorted_qps = np.sort(qps_values)
            cdf = np.arange(1, len(sorted_qps) + 1) / len(sorted_qps)
            ax.plot(sorted_qps, cdf, color=colors[wi], linestyle=linestyles[wi],
                    linewidth=1.8, label=f'{wl_name} (avg={qps_values.mean():.1f})')

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

axes[2].set_xlabel('QPS (requests/sec)', fontsize=10)
axes[2].set_ylabel('CDF', fontsize=10)
axes[2].set_title('QPS Distribution (1s window)', fontsize=11, fontweight='bold')
axes[2].legend(fontsize=8, loc='lower right')
axes[2].grid(alpha=0.3)
axes[2].set_xlim(left=0)

plt.tight_layout(rect=[0, 0, 1, 0.94])
save_path = OUT_DIR / "workload_input_output_qps_cdf.png"
fig.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Saved: {save_path}")

# Print summary
print("\n=== Workload Summary ===")
for wl_name, wl_file in WORKLOADS.items():
    fpath = WORKLOAD_DIR / wl_file
    ils, ols, ats = [], [], []
    with open(fpath) as f:
        for line in f:
            d = json.loads(line)
            ils.append(d['input_len'])
            ols.append(d['output_len'])
            ats.append(d['arrival_time_s'])
    ils, ols, ats = np.array(ils), np.array(ols), np.sort(np.array(ats))
    duration = ats[-1] - ats[0]
    avg_qps = len(ats) / duration if duration > 0 else 0
    print(f"  {wl_name:15s}: n={len(ils):4d}, "
          f"IL avg={ils.mean():.0f} p50={np.median(ils):.0f} p99={np.percentile(ils,99):.0f}, "
          f"OL avg={ols.mean():.0f} p50={np.median(ols):.0f} p99={np.percentile(ols,99):.0f}, "
          f"duration={duration:.0f}s avg_qps={avg_qps:.2f}")
