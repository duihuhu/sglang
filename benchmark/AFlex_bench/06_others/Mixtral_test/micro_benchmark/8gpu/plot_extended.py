#!/usr/bin/env python3
"""Plot extended QPS sweep results for Mixtral-8x7B 8-GPU."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

qps_range = list(range(1, 10))
scenarios = ['chatbot', 'qa', 'rag', 'summary']

# Baseline data
B = {
 'native_dp': {
  'chatbot': {'thpt': [882.5,1486.3,1746.0,1850.2,1928.5,1983.5,2000.5,2046.3,2072.2], 'tpot': [35.8,41.5,42.5,42.8,43.1,43.0,43.3,43.2,43.3], 'mjpt': [2670.3,1679.0,1449.2,1405.2,1355.9,1318.8,1307.3,1266.9,1268.6], 'ttft': [78.3,75.6,76.1,76.1,76.4,76.3,76.3,76.0,76.5]},
  'qa': {'thpt': [246.2,473.7,683.9,871.7,1037.9,1183.4,1320.6,1440.3,1496.6], 'tpot': [32.7,33.4,33.7,38.0,40.4,43.3,45.8,46.7,47.0], 'mjpt': [6922.1,4267.7,3098.4,2749.9,2367.5,2112.7,1813.5,1727.0,1708.9], 'ttft': [67.4,67.9,67.9,72.3,73.8,75.7,77.1,77.3,77.0]},
  'rag': {'thpt': [63.4,125.5,186.3,245.8,304.4,361.5,418.3,471.7,526.2], 'tpot': [31.4,32.0,32.6,33.5,33.8,34.3,34.4,34.9,35.0], 'mjpt': [17138.7,10737.7,8387.2,7004.4,6004.3,5200.3,4597.9,4162.2,3840.1], 'ttft': [68.6,67.0,66.8,67.6,67.3,67.5,67.2,67.4,66.9]},
  'summary': {'thpt': [63.4,125.5,186.3,246.0,303.9,361.6,417.7,472.2,527.4], 'tpot': [31.4,32.1,32.5,33.7,33.6,34.5,34.8,35.3,35.4], 'mjpt': [17504.5,10861.0,8515.8,7123.1,6200.3,5363.6,4813.4,4421.0,4003.6], 'ttft': [69.2,67.3,66.7,67.9,66.8,68.0,67.9,68.0,67.6]},
 },
 'pd_dp': {
  'chatbot': {'thpt': [864.6,1436.6,1625.0,1732.9,1784.1,1847.5,1867.0,1912.4,1943.7], 'tpot': [42.0,45.5,46.2,46.4,46.5,46.4,46.5,46.5,46.4], 'mjpt': [1903.2,1158.3,1036.2,979.2,940.2,926.7,902.2,888.8,878.4], 'ttft': [38.5,36.9,37.5,38.8,37.8,37.8,39.4,40.6,40.8]},
  'qa': {'thpt': [245.8,471.4,673.9,849.4,1013.5,1147.9,1288.5,1399.3,1443.6], 'tpot': [34.1,40.5,45.1,46.4,47.3,48.2,48.4,49.2,49.2], 'mjpt': [5714.3,3445.9,2475.2,1983.3,1660.3,1463.5,1300.7,1215.8,1176.1], 'ttft': [37.0,36.2,37.8,39.7,38.5,38.3,39.5,39.3,41.7]},
  'rag': {'thpt': [63.3,125.3,186.1,246.0,303.0,360.3,417.1,469.9,523.3], 'tpot': [33.0,33.9,34.0,34.6,35.0,35.5,36.5,38.2,38.6], 'mjpt': [15670.8,9163.0,6909.8,5505.3,4667.2,4074.0,3667.6,3337.9,3063.6], 'ttft': [37.3,38.1,38.4,37.4,37.7,40.5,40.0,39.9,42.0]},
  'summary': {'thpt': [63.3,125.3,185.8,245.0,302.7,360.7,416.3,470.9,520.2], 'tpot': [33.3,33.9,35.1,35.0,36.7,37.2,40.3,42.5,44.0], 'mjpt': [15977.2,9463.9,7085.1,5721.0,4865.8,4309.5,3866.0,3489.6,3211.5], 'ttft': [38.7,36.9,36.9,37.9,40.4,40.4,40.8,40.2,43.0]},
 },
 'pdaf': {
  'chatbot': {'thpt': [743.6,873.2,920.1,952.9,966.6,974.9,987.2,996.5,1001.6], 'tpot': [85.2,91.8,93.3,93.3,94.2,94.7,94.6,94.5,94.7], 'mjpt': [1285.8,1129.0,1081.6,1044.3,1019.4,1015.3,1000.7,992.2,985.3], 'ttft': [71.6,70.3,70.8,71.8,70.7,70.4,70.9,70.8,71.7]},
  'qa': {'thpt': [240.4,445.4,614.5,753.0,821.0,863.2,883.9,902.4,928.1], 'tpot': [62.0,68.7,75.1,82.3,86.2,86.3,87.3,88.3,87.6], 'mjpt': [4160.9,2361.5,1745.8,1460.7,1347.8,1290.6,1258.3,1231.8,1196.1], 'ttft': [84.1,86.1,86.4,88.2,90.0,91.4,95.2,96.0,96.4]},
  'rag': {'thpt': [63.0,123.4,182.6,239.0,292.7,344.2,391.5,384.2,386.1], 'tpot': [53.2,57.7,62.5,65.1,66.9,68.9,71.0,68.4,68.3], 'mjpt': [16013.3,9119.9,6705.0,5222.7,4387.3,3878.1,3554.9,3661.8,3668.2], 'ttft': [150.2,159.7,176.4,212.5,241.4,301.2,466.2,1096.2,1109.7]},
  'summary': {'thpt': [62.9,123.8,182.3,205.2,204.9,41.8,0.7,None,None], 'tpot': [52.7,56.7,62.4,64.3,64.1,63.8,0,None,None], 'mjpt': [16020.8,9604.8,7312.3,7008.6,6954.3,20300.6,993035.6,None,None], 'ttft': [271.7,365.5,564.7,1139.0,1143.7,1159.7,134641.4,None,None]},
 },
}

# Tier data
T = {
 'native_dp': {
  'chatbot': {'thpt': [871.3,1480.8,1729.5,1843.7,1921.4,1968.9,2015.7,2037.7,2072.7], 'tpot': [36.3,41.6,42.7,42.9,43.1,43.1,43.2,43.4,43.3], 'mjpt': [2651.9,1679.2,1449.0,1397.0,1353.5,1320.2,1280.7,1277.1,1265.1], 'ttft': [78.6,76.2,76.6,76.7,76.5,76.4,76.5,78.3,78.3]},
  'qa': {'thpt': [245.6,471.3,683.0,868.1,1033.6,1182.7,1317.4,1437.2,1491.2], 'tpot': [33.2,33.7,33.8,38.3,40.5,44.1,45.9,46.7,46.9], 'mjpt': [6814.3,4271.6,3168.3,2741.7,2343.9,2001.7,1887.0,1761.9,1659.4], 'ttft': [69.5,68.5,68.3,73.2,74.8,76.7,77.5,77.7,77.6]},
  'rag': {'thpt': [63.3,125.4,186.2,245.9,304.4,361.9,417.4,472.0,525.6], 'tpot': [33.3,33.3,33.4,33.6,34.2,34.5,35.2,35.4,35.7], 'mjpt': [15659.6,10226.0,8107.6,6833.0,5891.8,5121.9,4536.1,4114.8,3759.7], 'ttft': [75.2,70.0,69.0,67.9,68.3,68.0,68.7,68.5,68.4]},
  'summary': {'thpt': [63.3,125.2,186.2,245.5,303.9,361.6,415.2,472.3,526.3], 'tpot': [33.1,33.2,33.5,33.9,34.2,34.5,34.9,35.6,35.9], 'mjpt': [15621.7,10270.5,8258.4,7037.3,6120.4,5366.0,4862.7,4321.6,3951.6], 'ttft': [76.3,70.0,69.0,68.5,68.1,67.8,67.9,68.2,70.2]},
 },
 'pd_dp': {
  'chatbot': {'thpt': [862.1,1447.0,1632.9,1730.6,1792.7,1859.5,1895.3,1913.9,1913.4], 'tpot': [41.8,45.3,46.1,46.4,46.5,46.4,46.3,46.4,46.6], 'mjpt': [1837.8,1116.3,987.8,951.8,920.2,891.6,865.9,857.2,855.2], 'ttft': [40.7,38.9,38.8,37.7,38.3,38.6,40.5,40.5,41.1]},
  'qa': {'thpt': [245.7,472.0,668.4,850.4,1013.2,1152.4,1293.7,1402.2,1436.0], 'tpot': [34.2,40.5,45.1,46.4,47.1,47.9,48.5,49.0,49.1], 'mjpt': [5405.6,3293.7,2424.0,1915.8,1604.9,1417.6,1272.2,1176.4,1143.2], 'ttft': [38.3,37.7,38.8,38.5,39.0,41.0,41.9,41.4,42.2]},
  'rag': {'thpt': [63.3,125.4,186.1,245.8,304.1,360.8,416.2,470.9,521.4], 'tpot': [33.3,33.4,33.6,34.1,34.3,35.0,36.1,37.5,38.9], 'mjpt': [14246.0,8662.4,6567.8,5356.4,4534.0,3932.7,3552.0,3247.6,2993.9], 'ttft': [38.4,38.4,39.3,39.3,39.7,40.0,41.3,42.4,42.0]},
  'summary': {'thpt': [63.3,125.4,185.6,245.8,302.9,359.5,416.7,466.8,521.3], 'tpot': [33.2,33.2,33.8,34.1,35.9,38.6,39.5,41.8,43.9], 'mjpt': [14556.2,9013.6,6813.4,5508.4,4680.3,4118.6,3686.0,3369.2,3062.2], 'ttft': [39.6,38.6,38.9,39.1,41.0,40.4,42.7,40.5,41.1]},
 },
 'pdaf': {
  'chatbot': {'thpt': [474.4,600.7,700.4,720.3,727.8,736.8,740.9,746.1,752.6], 'tpot': [128.1,128.6,127.0,126.9,127.5,127.3,127.6,127.5,127.0], 'mjpt': [1372.7,1078.7,923.4,894.7,882.7,870.6,864.0,856.5,848.2], 'ttft': [168.5,175.1,183.3,196.9,211.8,225.6,243.7,245.0,273.8]},
  'qa': {'thpt': [233.7,418.4,558.4,613.1,644.7,650.5,622.1,606.5,606.1], 'tpot': [89.3,99.2,111.7,116.0,115.2,111.7,109.4,109.3,107.4], 'mjpt': [2885.7,1655.6,1255.3,1134.0,1068.8,1050.5,1088.0,1116.9,1126.1], 'ttft': [293.8,378.3,515.2,684.2,966.2,1879.2,3444.6,4200.6,5054.7]},
  'rag': {'thpt': [62.5,70.5,70.6,70.5,70.8,41.2,0.4,None,None], 'tpot': [63.8,66.3,67.9,67.8,67.8,67.5,0,None,None], 'mjpt': [11057.5,9922.7,9956.0,9977.8,9951.0,15871.4,1372687.2,None,None], 'ttft': [1670.8,6734.5,6739.8,6757.8,6725.1,6854.4,234016.3,None,None]},
  'summary': {'thpt': [None]*9, 'tpot': [None]*9, 'mjpt': [None]*9, 'ttft': [None]*9},
 },
}

colors = {'native_dp': '#2196F3', 'pd_dp': '#4CAF50', 'pdaf': '#FF9800'}
labels = {'native_dp': 'Native DP4', 'pd_dp': 'PD 2P+2D', 'pdaf': 'PDAF'}

def safe_plot(ax, x, y, **kwargs):
    """Plot only non-None values."""
    xf = [xi for xi, yi in zip(x, y) if yi is not None]
    yf = [yi for yi in y if yi is not None]
    if xf:
        ax.plot(xf, yf, **kwargs)

# ============================================================
# Figure 1: Throughput vs QPS (4 scenarios, Base vs Tier)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle('Mixtral-8x7B 8-GPU: Throughput vs QPS', fontsize=14, fontweight='bold')

for s_idx, sc in enumerate(scenarios):
    ax = axes[s_idx // 2][s_idx % 2]
    for arch in ['native_dp', 'pd_dp', 'pdaf']:
        safe_plot(ax, qps_range, B[arch][sc]['thpt'], marker='o', linestyle='-',
                  label=f'{labels[arch]} Base', color=colors[arch], alpha=0.6, linewidth=2)
        safe_plot(ax, qps_range, T[arch][sc]['thpt'], marker='s', linestyle='--',
                  label=f'{labels[arch]} Tier', color=colors[arch], alpha=1.0, linewidth=2)
    ax.set_xlabel('QPS')
    ax.set_ylabel('Throughput (tok/s)')
    ax.set_title(sc)
    ax.legend(fontsize=7, loc='best')
    ax.grid(alpha=0.3)
    ax.set_xticks(qps_range)

plt.tight_layout()
plt.savefig('/workspace/sglang/benchmark/AFlex_bench/06_others/Mixtral_test/micro_benchmark/8gpu/ext_throughput.png', dpi=150)
plt.close()

# ============================================================
# Figure 2: Energy (mJ/tok) vs QPS
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle('Mixtral-8x7B 8-GPU: Energy per Token vs QPS', fontsize=14, fontweight='bold')

for s_idx, sc in enumerate(scenarios):
    ax = axes[s_idx // 2][s_idx % 2]
    for arch in ['native_dp', 'pd_dp', 'pdaf']:
        # Filter out extreme values for readability
        b_mjpt = [v if v and v < 25000 else None for v in B[arch][sc]['mjpt']]
        t_mjpt = [v if v and v < 25000 else None for v in T[arch][sc]['mjpt']]
        safe_plot(ax, qps_range, b_mjpt, marker='o', linestyle='-',
                  label=f'{labels[arch]} Base', color=colors[arch], alpha=0.6, linewidth=2)
        safe_plot(ax, qps_range, t_mjpt, marker='s', linestyle='--',
                  label=f'{labels[arch]} Tier', color=colors[arch], alpha=1.0, linewidth=2)
    ax.set_xlabel('QPS')
    ax.set_ylabel('mJ/tok')
    ax.set_title(sc)
    ax.legend(fontsize=7, loc='best')
    ax.grid(alpha=0.3)
    ax.set_xticks(qps_range)

plt.tight_layout()
plt.savefig('/workspace/sglang/benchmark/AFlex_bench/06_others/Mixtral_test/micro_benchmark/8gpu/ext_energy.png', dpi=150)
plt.close()

# ============================================================
# Figure 3: TPOT vs QPS
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle('Mixtral-8x7B 8-GPU: TPOT vs QPS', fontsize=14, fontweight='bold')

for s_idx, sc in enumerate(scenarios):
    ax = axes[s_idx // 2][s_idx % 2]
    for arch in ['native_dp', 'pd_dp', 'pdaf']:
        b_tpot = [v if v and v > 0 else None for v in B[arch][sc]['tpot']]
        t_tpot = [v if v and v > 0 else None for v in T[arch][sc]['tpot']]
        safe_plot(ax, qps_range, b_tpot, marker='o', linestyle='-',
                  label=f'{labels[arch]} Base', color=colors[arch], alpha=0.6, linewidth=2)
        safe_plot(ax, qps_range, t_tpot, marker='s', linestyle='--',
                  label=f'{labels[arch]} Tier', color=colors[arch], alpha=1.0, linewidth=2)
    ax.set_xlabel('QPS')
    ax.set_ylabel('TPOT (ms)')
    ax.set_title(sc)
    ax.legend(fontsize=7, loc='best')
    ax.grid(alpha=0.3)
    ax.set_xticks(qps_range)

plt.tight_layout()
plt.savefig('/workspace/sglang/benchmark/AFlex_bench/06_others/Mixtral_test/micro_benchmark/8gpu/ext_tpot.png', dpi=150)
plt.close()

# ============================================================
# Figure 4: Energy Saving % (Tier vs Base) across QPS
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle('Mixtral-8x7B 8-GPU: Tier Energy Saving % vs QPS', fontsize=14, fontweight='bold')

for s_idx, sc in enumerate(scenarios):
    ax = axes[s_idx // 2][s_idx % 2]
    for arch in ['native_dp', 'pd_dp', 'pdaf']:
        savings = []
        valid_qps = []
        for q_idx, qps in enumerate(qps_range):
            bv = B[arch][sc]['mjpt'][q_idx]
            tv = T[arch][sc]['mjpt'][q_idx]
            if bv and tv and bv < 25000 and tv < 25000:
                savings.append((bv - tv) / bv * 100)
                valid_qps.append(qps)
        if valid_qps:
            ax.plot(valid_qps, savings, 'o-', label=labels[arch], color=colors[arch], linewidth=2, markersize=6)
    ax.set_xlabel('QPS')
    ax.set_ylabel('Energy Saving (%)')
    ax.set_title(sc)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(qps_range)

plt.tight_layout()
plt.savefig('/workspace/sglang/benchmark/AFlex_bench/06_others/Mixtral_test/micro_benchmark/8gpu/ext_energy_saving.png', dpi=150)
plt.close()

print("All extended plots saved!")
