# Mixtral-8x7B MoE 模型测试报告

## 环境信息

- **模型**: Mixtral-8x7B (MoE, 8 experts × top-2, 32 layers)
- **GPU**: 4 × 80GB (TP=2 per instance)
- **部署拓扑**:
  - Native DP2: 2 instances × TP=2
  - PD: P(TP=2, GPU 0,1) + D(TP=2, GPU 2,3)
  - PDAF: 需要 8 GPU（4 卡环境不适用）

## 1. 部署验证

| 架构 | TP | DP | 状态 |
|------|----|----|------|
| Native DP2 | 2 | 2 | ✓ 正常启动 |
| PD | 2 | 1 | ✓ Mooncake RDMA 传输正常 |
| PDAF | 2/comp | - | ✗ 需要 8 GPU |

## 2. 能耗模型

### 数据采集 (quick sweep)

| 数据集 | 行数 | 维度 |
|--------|------|------|
| V1 Prefill | 27 | tp=2, freq=[210,690,1410], il=[128,1024,4096], bs=[1,4,16] |
| V1 Decode | 37 | tp=2, freq=[210,690,1410], il=[128,1024], ol=[64,256], bs=[1,4,16,64] |
| V2 Pipeline | 150 | tp=2, M=1, f_A/f_F=[210,690,1410], il=[128,512,2048], bs=[1..128] |

### 模型训练 CV MAPE

| 子任务 | Best Model | CV MAPE |
|--------|-----------|---------|
| Prefill_A (energy) | LinearReg | 37.8% |
| Prefill_F (energy) | LinearReg | 37.8% |
| Decode_A (energy) | LinearReg | 17.1% |
| Decode_F (energy) | LinearReg | 17.1% |
| Prefill_A_lat | LinearReg | 25.5% |
| Prefill_F_lat | LinearReg | 25.5% |
| Decode_A_lat | LinearReg | 15.4% |
| Decode_F_lat | LinearReg | 15.4% |
| Decode_iter_lat (V2) | GBDT | 21.0% |
| Decode_iter_energy (V2) | GBDT | 13.0% |

模型路径: `energy_model/models_v1/` 和 `energy_model/models_v2/`

## 3. Benchmark 结果 (chatbot, QPS=1, 200 reqs)

### Baseline vs Tier DVFS

| 架构 | 模式 | Thpt (tok/s) | TTFT (ms) | TPOT (ms) | Energy (J) | mJ/tok | 节能比 |
|------|------|------|------|------|--------|--------|--------|
| Native DP2(TP=2) | baseline | 864.1 | 80.2 | 42.5 | 300567 | 1467.6 | — |
| Native DP2(TP=2) | **Tier** | 863.5 | 79.7 | 42.5 | 309530 | 1511.4 | **-3.0%** |
| PD(P-TP2+D-TP2) | baseline | 849.8 | 37.0 | 46.2 | 207927 | 1015.3 | — |
| PD(P-TP2+D-TP2) | **Tier** | 850.7 | 37.4 | 46.1 | 199300 | 973.1 | **4.1%** |

### 关键观察

1. **MoE 模型吞吐远高于 Dense**：Mixtral 864 tok/s vs Qwen3-32B 14.9 tok/s（MoE 仅激活 2/8 experts）
2. **PD 分离的能耗优势明显**：PD baseline 1015 mJ/tok vs Native 1468 mJ/tok（-31%）
3. **Tier DVFS 效果有限**：
   - Native Tier 反而增加了 3% 能耗（MoE 的计算本身很轻，频率已经很低时降频反而延长执行时间增加静态功耗）
   - PD Tier 节能 4.1%（decode 侧有一定降频空间）
4. **TTFT/TPOT 无明显退化**：SLO=0%，无违约

### 与 Dense 模型对比

| 指标 | Qwen3-32B (Dense) | Mixtral-8x7B (MoE) |
|------|-------------------|---------------------|
| 吞吐 | 14.9 tok/s | 864.1 tok/s |
| TTFT | 90.5 ms | 80.2 ms |
| TPOT | 43.5 ms | 42.5 ms |
| mJ/tok (Native) | 34747 | 1468 |
| Tier 节能 (PD) | 10.3% | 4.1% |

## 4. 文件结构

```
Mixtral_test/
├── test.md                    # 本报告
├── scripts/
│   ├── run_micro_bench.py     # Mixtral benchmark 脚本
│   └── profile_energy.py      # 能耗 profiling 采集脚本
├── energy_model/
│   ├── data/                  # profiling 原始数据
│   │   ├── prefill_data_v1.txt
│   │   ├── decode_data_v1.txt
│   │   └── decode_pipeline_v1.txt
│   ├── models_v1/             # V1 layer 级模型
│   └── models_v2/             # V2 pipeline 级模型 (用于 Tier DVFS)
├── logs/                      # 服务器运行日志
└── results/                   # benchmark JSON 结果
```

## 5. 后续建议

1. **补充 profiling 数据**：当前用 quick 模式采集点数较少，V1 模型 MAPE 偏高。建议补充完整的 freq/bs/il 组合
2. **PDAF 需要 8 卡**：MoE 模型单卡放不下，PDAF 的 AF 分离需要每个 component 至少 TP=2 = 8 卡
3. **Tier DVFS 对 MoE 效果有限**：因为 MoE 本身计算量小（只激活 2/8 experts），GPU 利用率低，降频的节能空间被静态功耗抵消
4. **可考虑 Expert-aware DVFS**：根据 Expert Load Imbalance Factor (LIF) 动态调频可能更有效
