# PD+AF 多并发性能测试结果

## 测试环境

- 模型：Qwen3-32B，A800-SXM4-80GB
- GPU：4-7 号卡
- 四种架构：PD TP=2（4卡）、PD+AF M=2 async（4卡）、PD+AF M=1（4卡）、PD TP=1（2卡）
- 并发范围：1 ~ 2048

## 核心结论

1. **PD TP=2 全面最优**：在所有并发下延迟最低、吞吐最高，峰值吞吐 3704 tok/s（conc=768），之后饱和在 ~3500 tok/s。

2. **AF M=2 async pipeline 在高并发下逼近 TP=2**：conc=2048 时吞吐达 3281 tok/s，仅比 TP=2 低 6%。M=2 的 interleaved pipeline 在 batch≥200 时 overlap 收益显著。

3. **AF M=1 吞吐饱和在 ~1950 tok/s**：约为 M=2 的 60%，验证了 async pipeline 在大 batch 下的必要性。

4. **PD TP=1 吞吐硬顶 ~710 tok/s**：单卡 decode 容量有限，高并发下 TPOT 线性增长至秒级。

5. **小并发（≤32）下 AF 方案无优势**：M=2 因 GEMM 效率损失反而最差；M=1 与 PD TP=1 接近。AF 的收益拐点在 conc≈64。

## 数据与图表

- `results/all_concurrency_results.csv`：完整数据（57 条记录）
- `results/plot_tpot.png`、`plot_ttft.png`、`plot_throughput.png`：折线图
