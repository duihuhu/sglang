#!/usr/bin/env python3
"""NVLink vs RDMA throughput comparison chart."""

import matplotlib.pyplot as plt
import numpy as np

# RDMA bandwidth data (from ib_write_bw results)
rdma_sizes_bytes = [2, 4, 8, 16, 32, 64, 128, 256, 512,
                    1024, 2048, 4096, 8192, 16384, 32768,
                    65536, 131072, 262144, 524288,
                    1048576, 2097152, 4194304, 8388608]
rdma_bw_gbps = [0.086, 0.18, 0.35, 0.70, 1.39, 2.81, 5.61, 11.22, 22.28,
                44.61, 84.78, 135.89, 192.57, 196.36, 196.63,
                196.73, 196.76, 196.78, 196.79,
                196.80, 196.81, 196.81, 196.80]
rdma_bw_GBs = [x / 8 for x in rdma_bw_gbps]

# 4-port RDMA aggregate (.36 <-> .35, ib_write_bw -a, 4× mlx5_0/1/4/5)
rdma_4port_sizes_bytes = [2, 4, 8, 16, 32, 64, 128, 256, 512,
                          1024, 2048, 4096, 8192, 16384, 32768,
                          65536, 131072, 262144, 524288,
                          1048576, 2097152, 4194304, 8388608]
rdma_4port_bw_mb_s = [46.86, 96.78, 193.06, 385.70, 780.08, 1563.19, 3140.33,
                      6254.62, 12261.09, 24198.11, 44714.89, 71396.04,
                      92294.60, 93668.49, 93765.91, 93801.44, 93824.15,
                      93829.11, 93825.91, 93838.71, 93453.17, 93831.57, 90040.30]
rdma_4port_bw_GBs = [x / 1024 for x in rdma_4port_bw_mb_s]

# NVLink bandwidth data (from communicaton_cost_nvlink.txt)
# Using seq_len series with batch_size=1: data size = seq_len * 4096 * 2 (hidden=4096, fp16)
# seq=1,bs=1..256 gives sizes from 8KB to 2MB
# We use the full dataset aligned by actual throughput
nvlink_sizes_bytes = []
nvlink_bw_GBs = []

# seq_len=1, varying batch_size (each element = 4096*2=8192 bytes per seq per batch)
# Actually the raw data gives throughput directly, let's reconstruct by data size
# data_size = seq_len * batch_size * hidden_dim * dtype_size
# From the data pattern, using seq=1 row: throughput goes from 0.24 to 71.23 GB/s
# The sizes are: seq_len * batch_size * 8192 bytes (assuming hidden=4096, fp16)
# But let's just use the throughput values directly and compute equivalent message sizes

# Better approach: use latency to compute size
# throughput = size / latency => size = throughput * latency
# From the data: seq=1, bs=1: 0.24 GB/s, 41.96us => size = 0.24e9 * 41.96e-6 = 10071 bytes ~ 10KB
# This suggests hidden_dim * 2 * seq * bs for fp16

# Let's just map NVLink data by computing transfer size = throughput_GB_s * latency_us * 1e-6 * 1e9
# Or simpler: use the known formula. Looking at the data:
# seq=1, bs=1, latency=41.96us, throughput=0.24 GB/s => size = 0.24*1e9 * 41.96*1e-6 = ~10KB
# seq=1, bs=2 => 0.56 GB/s, 36.59us => size = 0.56e9*36.59e-6 = ~20KB
# So base unit is ~10KB (= 5120 elements * 2 bytes, suggesting hidden=5120 for A800 model)

# Direct computation: size = throughput * latency
nvlink_raw = [
    # (seq_len, batch_size, latency_us, throughput_GB_s)
    (1, 1, 41.96, 0.24),
    (1, 2, 36.59, 0.56),
    (1, 4, 36.55, 1.12),
    (1, 8, 37.44, 2.19),
    (1, 16, 36.77, 4.46),
    (1, 32, 36.74, 8.92),
    (1, 64, 36.91, 17.76),
    (1, 128, 37.08, 35.34),
    (1, 256, 36.80, 71.23),
    (128, 1, 37.05, 35.38),
    (128, 2, 37.11, 70.63),
    (128, 4, 48.64, 107.79),
    (128, 8, 81.49, 128.67),
    (128, 16, 146.05, 143.60),
    (128, 32, 282.52, 148.46),
    (128, 64, 549.31, 152.71),
    (128, 128, 965.07, 173.84),
    (128, 256, 1909.54, 175.72),
    (256, 256, 3801.44, 176.54),
    (512, 256, 7590.20, 176.83),
    (1024, 256, 15163.62, 177.03),
    (2048, 256, 30335.97, 176.98),
    (4096, 256, 60633.29, 177.09),
    (8192, 256, 121238.55, 177.13),
]

for seq, bs, lat_us, tp_GBs in nvlink_raw:
    size = tp_GBs * 1e9 * lat_us * 1e-6  # bytes
    nvlink_sizes_bytes.append(size)
    nvlink_bw_GBs.append(tp_GBs)

# Sort by size
nvlink_sorted = sorted(zip(nvlink_sizes_bytes, nvlink_bw_GBs))
nvlink_sizes_bytes = [x[0] for x in nvlink_sorted]
nvlink_bw_GBs = [x[1] for x in nvlink_sorted]

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Left: Throughput (GB/s) vs Message Size ---
ax1.semilogx(rdma_sizes_bytes, rdma_bw_GBs, 'o-', color='#2196F3',
             linewidth=2, markersize=4, label='RDMA')
ax1.semilogx(rdma_4port_sizes_bytes, rdma_4port_bw_GBs, 'D-', color='#1565C0',
             linewidth=2, markersize=5, label='RDMA 4×')
ax1.semilogx(nvlink_sizes_bytes, nvlink_bw_GBs, 's-', color='#FF5722',
             linewidth=2, markersize=4, label='NVLink')

ax1.set_xlabel('Message Size (Bytes)', fontsize=12)
ax1.set_ylabel('Throughput (GB/s)', fontsize=12)
ax1.set_title('Throughput vs Message Size', fontsize=14)
ax1.legend(fontsize=11, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(1, 1e8)
ax1.set_ylim(0, 200)

rdma_4port_peak = max(rdma_4port_bw_GBs)
ax1.axhline(y=rdma_4port_peak, color='#1565C0', linestyle='--', alpha=0.5, linewidth=1)

# Add size labels on x-axis
size_labels = [(1, '1B'), (1024, '1KB'), (1048576, '1MB'), (8388608, '8MB')]
ax1.set_xticks([1, 16, 256, 1024, 16384, 262144, 1048576, 8388608, 1e8])
ax1.set_xticklabels(['1B', '16B', '256B', '1KB', '16KB', '256KB', '1MB', '8MB', '100MB'])

# Add horizontal reference lines
ax1.axhline(y=24.6, color='#2196F3', linestyle='--', alpha=0.5, linewidth=1)
ax1.axhline(y=177, color='#FF5722', linestyle='--', alpha=0.5, linewidth=1)
ax1.text(2, 26, '24.6 GB/s (1× RDMA peak)', fontsize=9, color='#2196F3', alpha=0.7)
ax1.text(2, rdma_4port_peak + 2, f'{rdma_4port_peak:.1f} GB/s (4× RDMA)', fontsize=9,
         color='#1565C0', alpha=0.7)
ax1.text(2, 179, '177 GB/s (NVLink peak)', fontsize=9, color='#FF5722', alpha=0.7)

# --- Right: Latency comparison ---
# RDMA latency (.36 <-> .35, ib_write_lat -a, mlx5_0, MTU 4096)
rdma_lat_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512,
                  1024, 2048, 4096, 8192, 16384, 32768,
                  65536, 131072, 262144, 524288,
                  1048576, 2097152, 4194304, 8388608]
rdma_lat_us = [1.82, 1.82, 1.83, 1.82, 1.86, 1.86, 1.93, 2.70, 2.74,
               2.90, 2.95, 3.55, 3.78, 4.13, 4.82,
               6.15, 8.81, 14.12, 24.76,
               46.73, 89.44, 174.69, 345.19]

# NVLink latency (from raw data)
nvlink_lat_sorted = sorted(zip(nvlink_sizes_bytes, [x[2] for x in nvlink_raw]))
nvlink_lat_sizes = [x[0] for x in nvlink_lat_sorted]
nvlink_lat_us = [x[1] for x in nvlink_lat_sorted]

ax2.loglog(rdma_lat_sizes, rdma_lat_us, 'o-', color='#2196F3',
           linewidth=2, markersize=4, label='RDMA')
ax2.loglog(nvlink_lat_sizes, nvlink_lat_us, 's-', color='#FF5722',
           linewidth=2, markersize=4, label='NVLink')

ax2.set_xlabel('Message Size (Bytes)', fontsize=12)
ax2.set_ylabel('Latency (μs)', fontsize=12)
ax2.set_title('Latency vs Message Size', fontsize=14)
ax2.legend(fontsize=11, loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/sglang/benchmark/AFlex_bench/comm/nvlink_vs_rdma.png', dpi=150, bbox_inches='tight')
plt.savefig('/workspace/sglang/benchmark/AFlex_bench/comm/nvlink_vs_rdma.pdf', bbox_inches='tight')
print("Charts saved to:")
print("  /workspace/sglang/benchmark/AFlex_bench/comm/nvlink_vs_rdma.png")
print("  /workspace/sglang/benchmark/AFlex_bench/comm/nvlink_vs_rdma.pdf")
