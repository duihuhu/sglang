# Native / PD Tier 调频模型指南

## 模型文件位置

```
/workspace/sglang-tier/benchmark/AFlex_bench/06_others/more_model/energy_model/models/
```

### 文件清单

| 文件 | 用途 |
|------|------|
| `Prefill_A_GBDT.pkl` / `Prefill_A_LUT.pkl` / `Prefill_A_LinearReg.pkl` | Prefill Attention 能耗模型 |
| `Prefill_F_GBDT.pkl` / `Prefill_F_LUT.pkl` / `Prefill_F_LinearReg.pkl` | Prefill FFN 能耗模型 |
| `Prefill_A_lat_GBDT.pkl` / `Prefill_A_lat_LUT.pkl` / `Prefill_A_lat_LinearReg.pkl` | Prefill Attention 延迟模型 |
| `Prefill_F_lat_GBDT.pkl` / `Prefill_F_lat_LUT.pkl` / `Prefill_F_lat_LinearReg.pkl` | Prefill FFN 延迟模型 |
| `Decode_iter_lat_GBDT.pkl` / `Decode_iter_lat_LUT.pkl` | Decode iteration 延迟（V2 coupled pipeline） |
| `Decode_iter_energy_A_GBDT.pkl` / `Decode_iter_energy_A_LUT.pkl` | Decode Attention 能耗（V2 coupled） |
| `Decode_iter_energy_F_GBDT.pkl` / `Decode_iter_energy_F_LUT.pkl` | Decode FFN 能耗（V2 coupled） |

注意：没有 V1 的 `Decode_A_*.pkl` / `Decode_F_*.pkl`，只有 V2 coupled 模型。

## 启动参数

```bash
# Native / PD 实例启用 Tier DVFS
--dvfs-enabled \
--dvfs-energy-model-dir /workspace/sglang-tier/benchmark/AFlex_bench/06_others/more_model/energy_model/models \
--dvfs-ttft-slo-ms 2000 \
--dvfs-tpot-slo-us 250000
```

## 代码调用链

```
启动参数: --dvfs-enabled --dvfs-energy-model-dir <path>
         │
         ▼
scheduler.py: _init_unified_dvfs(server_args)
         │
         ▼
AFProfilePredictor(dvfs_energy_model_dir)      ← 加载所有 pkl 模型
  文件: python/sglang/srt/energy/af_profile_predictor.py
         │
         ▼
UnifiedDVFSController(predictor, num_layers, tp)
  文件: python/sglang/srt/energy/unified_dvfs_controller.py
         │
         ▼
每个 batch 前: scheduler._unified_dvfs_before_batch(batch)
  ├── Prefill: select_freq_prefill(bs, il, slack_us)
  │     └── 遍历频率，用 Prefill_A + Prefill_F 求和预测延迟/能耗
  │         选最低能耗且满足 TTFT SLO 的频率
  └── Decode: select_freq_decode(bs, il, ol, slo_tpot_us)
        └── 遍历频率，用 V2 coupled 模型预测 decode iteration 延迟/能耗
            选最低能耗且满足 TPOT SLO 的频率
```

## 核心逻辑 (UnifiedDVFSController)

### Prefill 调频
- 搜索所有候选频率 [210, 450, 690, 930, 1170, 1410] MHz
- 对每个频率，用 `Prefill_A_lat + Prefill_F_lat` 预测全模型 prefill 延迟
- 用 `Prefill_A + Prefill_F` 预测全模型 prefill 能耗
- 选择: 延迟 ≤ TTFT SLO 前提下，能耗最低的频率

### Decode 调频
- 使用 80% TPOT SLO 作为预算（conservative margin 应对 chunked prefill 阻塞）
- 按能耗从低到高搜索候选频率
- 用 `Decode_iter_lat` 预测单次 decode iteration 延迟
- 选择: 延迟 ≤ effective SLO 前提下，能耗最低的频率

### 保护机制
- **Lazy switching**: 只有节能 > 切换成本（1800 mJ）才实际切换频率
- **KV guard**: KV-cache 利用率 > 85% 时强制最大频率防 OOM
- **Fallback**: 没有频率满足 SLO → 使用最大频率 1410 MHz
- **Decode-aware prefill floor**: 有 decode 请求在队列时，prefill 频率不低于 decode 所需最低频率

## Native vs PD 的区别

两者共用同一个 `UnifiedDVFSController`，区别在于：

| 维度 | Native (DP8) | PD (DP4) |
|------|-------------|----------|
| GPU 拓扑 | 每个 GPU 独立跑完整 prefill+decode | Prefill GPU + Decode GPU 分离 |
| 调频粒度 | 同一 GPU 上 prefill/decode 交替，频率随 batch 类型切换 | P-GPU 只做 prefill 调频，D-GPU 只做 decode 调频 |
| chunked prefill 影响 | prefill chunk 会阻塞 decode → 需要 decode-aware floor | 无此问题，prefill 不阻塞 decode |
| 适用参数 | `--dvfs-enabled` | `--dvfs-enabled` |

## 关键源码文件

| 文件 | 功能 |
|------|------|
| `python/sglang/srt/energy/unified_dvfs_controller.py` | Native/PD 的单频率控制器核心 |
| `python/sglang/srt/energy/af_profile_predictor.py` | 能耗/延迟预测器（加载 pkl 模型） |
| `python/sglang/srt/energy/af_dvfs_controller.py` | PDAF 的双频率控制器（f_A / f_F 独立） |
| `python/sglang/srt/managers/scheduler.py` | 调度器集成入口（_init_unified_dvfs, _unified_dvfs_before_batch） |
| `python/sglang/srt/layers/dvfs.py` | NVML 硬件层（DVFSController, lock_sm_clock） |
| `python/sglang/srt/server_args.py` | 命令行参数定义 |
