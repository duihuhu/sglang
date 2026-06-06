# Version 5: 4-GPU Variable-Length Workload Results

## 配置

- **GPU**: 4卡 (GPU 4-7), NVIDIA H800
- **模型**: Qwen3-32B
- **方案**:
  - PD TP2: Prefill TP2 (GPU 4,5) + Decode TP2 (GPU 6,7)
  - Native TP2 DP2: 2x 全实例 TP2 (GPU 4,5 / GPU 6,7)
  - PDAF DynM: PF/PA (GPU 4,5) + DF/DA (GPU 6,7), 动态微批
  - PDAF Tier+DynM: 同上 + DVFS 调频
- **SLO**: TTFT ≤ 5000ms, TPOT ≤ 300ms

## 变长 Workload 说明

| Workload | 请求数 | 特点 |
|----------|--------|------|
| steady | 600 | 稳定 QPS=5, 混合长度 |
| varying | 280 | 突发模式, 低→高→低 |
| heavy | 360 | 长序列, 高计算需求 |
| overload | 690 | 高 QPS=10, 压力测试 |
| tier1_demo | 780 | 混合模式, 展示 Tier1 效果 |

## 4-GPU 变长 Workload 方案对比

| 方案 | Workload | Thpt(tok/s) | TTFT(ms) | TPOT(ms) | Energy(J) | J/GPU | SLO违背(%) |
|------|----------|------------|---------|---------|----------|-------|-----------|
| PD TP2 | steady | 755.3 | 138 | 44.4 | 104636 | 26159 | 0.0 |
| PD TP2 | varying | 183.1 | 122 | 43.4 | 79671 | 19918 | 0.0 |
| PD TP2 | heavy | 263.1 | 138 | 45.3 | 89477 | 22369 | 0.0 |
| PD TP2 | overload | 1005.1 | 138 | 47.7 | 101676 | 25419 | 0.0 |
| PD TP2 | tier1_demo | 883.2 | 133 | 46.7 | 118317 | 29579 | 0.0 |
| Native TP2 DP2 | steady | 748.0 | 142 | 68.2 | 95382 | 23846 | 0.0 |
| Native TP2 DP2 | varying | 183.2 | 124 | 52.9 | 75877 | 18969 | 0.0 |
| Native TP2 DP2 | heavy | 262.7 | 139 | 62.4 | 84488 | 21122 | 0.0 |
| Native TP2 DP2 | overload | 944.6 | 11292 | 71.1 | 95653 | 23913 | 68.0 |
| Native TP2 DP2 | tier1_demo | 880.4 | 10611 | 71.6 | 106570 | 26643 | 69.6 |
| PD DP2 | steady | 750.0 | 194 | 49.0 | 122488 | 30622 | 0.0 |
| PD DP2 | varying | 183.0 | 142 | 47.4 | 97614 | 24403 | 0.0 |
| PD DP2 | heavy | 259.1 | 190 | 48.2 | 108794 | 27198 | 0.0 |
| PD DP2 | overload | 835.6 | 19262 | 48.9 | 132690 | 33173 | 76.5 |
| PD DP2 | tier1_demo | 806.5 | 19436 | 48.9 | 143795 | 35949 | 79.0 |
| PDAF DynM | steady | 724.0 | 213 | 71.5 | 113498 | 28375 | 0.0 |
| PDAF DynM | varying | 181.8 | 297 | 64.4 | 81854 | 20464 | 0.0 |
| PDAF DynM | heavy | 265.9 | 209 | 67.5 | 94027 | 23507 | 0.0 |
| PDAF DynM | overload | 909.8 | 11303 | 73.3 | 109899 | 27475 | 69.1 |
| PDAF DynM | tier1_demo | 866.1 | 10873 | 73.1 | 119843 | 29961 | 69.1 |
| PDAF Tier+DynM | steady | 685.2 | 3004 | 103.5 | 76839 | 19210 | 12.3 |
| PDAF Tier+DynM | varying | 179.6 | 835 | 92.6 | 54462 | 13615 | 0.0 |
| PDAF Tier+DynM | heavy | 260.1 | 2850 | 95.2 | 63891 | 15973 | 15.0 |
| PDAF Tier+DynM | overload | 722.6 | 25089 | 96.1 | 88310 | 22078 | 76.5 |
| PDAF Tier+DynM | tier1_demo | 692.4 | 27160 | 96.3 | 94002 | 23500 | 79.0 |

## 关键发现

1. **PDAF Tier+DynM 总能耗节省 27.3%**（相比 PDAF DynM）
2. **PDAF Tier+DynM 比 PD TP2 节省 23.5% 能耗**
3. **PD TP2 在所有 workload 上 SLO 违背 = 0%**，延迟表现最佳
4. **Native TP2 DP2 和 PDAF DynM** 在 overload/tier1_demo 高负载下 SLO 违背约 69%
5. **PDAF Tier+DynM 在轻负载(varying)下 0% SLO 违背**，能耗最低

## 图表

![Throughput](../charts_4gpu_var/throughput_comparison.png)

![TTFT](../charts_4gpu_var/ttft_comparison.png)

![TPOT](../charts_4gpu_var/tpot_comparison.png)

![Energy](../charts_4gpu_var/energy_comparison.png)

![SLO](../charts_4gpu_var/slo_comparison.png)

![Energy Saving](../charts_4gpu_var/energy_saving.png)

![Energy Breakdown](../charts_4gpu_var/energy_breakdown.png)

![Radar](../charts_4gpu_var/radar_comparison.png)

