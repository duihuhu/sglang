# PCIe / RDMA / NVLink 对比图

## 矩阵

- 网络：PCIe（host-staged）、RDMA、NVLink
- 架构：Native、PD、AF
- 工作负载：QA、Chatbot、Balanced、RAG、Summary、LongContext
- 每个点 3 次重复；柱高为均值，误差棒为样本标准差

## 图表

- `ttft_p90_ms_network_architecture_comparison.{png,pdf}`：客户端观测 TTFT P90，单位 ms，纵轴为对数尺度。
- `tpot_p90_ms_network_architecture_comparison.{png,pdf}`：TPOT P90，单位 ms/token，纵轴为对数尺度。
- `total_throughput_tokens_s_network_architecture_comparison.{png,pdf}`：总吞吐，即 `(input tokens + output tokens) / duration`，单位 tokens/s。
- `energy_per_total_token_j_network_architecture_comparison.{png,pdf}`：总 token 能耗，即 `total GPU energy / (input tokens + output tokens)`，单位 J/token，纵轴为对数尺度。
- `all_metrics_relative_overview.{png,pdf}`：四项指标的 workload 内归一化总览；1.0 表示该 workload 的最佳配置。
- `network_architecture_workload_metrics.csv`：绘图使用的 216 行聚合值（3 网络 × 3 架构 × 6 workload × 4 指标）。
- `plot_manifest.json`：指标定义、矩阵范围和输出文件清单。

## 数据来源

- PCIe/NVLink：`rq2_network_energy/results/raw/measurement_local_2gpu_pcie_nvlink_20260827/formal/*/summary.json`
- RDMA：通过 `data/rdma_2gpu_qwen3_30b_a3b.json` 中 54 个有效 artifact 引用读取对应原始 `summary.json`

所有 162 个运行（每种网络 54 个）均为 64/64 请求成功；每个矩阵点均有 3 次有效重复。

## 组合图

- `balanced_four_metrics_network_architecture.{png,pdf}`：固定 Balanced（512 input + 256 output），在一张 2×2 图中比较 PCIe、RDMA、NVLink 下 Native、PD、AF 的 TTFT、TPOT、总 token 吞吐和总 token 能耗。
- `nvlink_six_workloads_four_metrics_architecture.{png,pdf}`：固定 NVLink，在一张 2×2 图中比较 Native、PD、AF 在六个工作负载下的四项指标。
