# MoE Model (Qwen3-30B-A3B) Benchmark Status

## 模型信息
- **模型**: Qwen3-30B-A3B (MoE, 128 experts, 8 active/token, 48 layers)
- **大小**: ~57GB (BF16)
- **GPU**: 8x A800-80GB
- **最大频率**: 1410 MHz

## 部署方案（统一 8 张卡，公平对比）

| 方案 | 配置 | GPU分配 |
|------|------|---------|
| Native DP8 | 8x TP=1 独立实例 + Router (round_robin) | GPU 0-7 各一个实例 |
| PD DP4 | 4x (Prefill-TP1 + Decode-TP1) + Router | 4对: (0,1)(2,3)(4,5)(6,7) |
| PDAF DP2 | 2x (PA+PF+DA+DF, 各TP=1) + Router | 2组: (0,1,2,3)(4,5,6,7) |

每种方案有 baseline 和 +Tier DVFS 两个版本，共 6 种配置。

## 频率控制策略
- **无 Tier（baseline）**: 测试前锁频 1410 MHz，测完后解锁
- **有 Tier（DVFS）**: 不锁频，由 DVFS 动态调节，测完后解锁确保清理

## Workloads
- `workload_azure_code_light_real.jsonl` — 681 请求
- `workload_azure_code_medium_real.jsonl` — 1600 请求
- `workload_azure_code_heavy_real.jsonl` — 2678 请求

## 当前进度

| # | 配置 | light | medium | heavy | 状态 |
|---|------|-------|--------|-------|------|
| 1 | native_dp8 | ✅ | ✅ | ✅ | 完成 |
| 2 | native_dp8_tier | 🔄 运行中 | ⏳ | ⏳ | 进行中 |
| 3 | pd_dp4 | ⏳ | ⏳ | ⏳ | 等待 |
| 4 | pd_dp4_tier | ⏳ | ⏳ | ⏳ | 等待 |
| 5 | pdaf_dp2 | ⏳ | ⏳ | ⏳ | 等待 |
| 6 | pdaf_dp2_tier | ⏳ | ⏳ | ⏳ | 等待 |

## 已完成结果

### Native DP8 (baseline, 锁频 1410 MHz)

| Workload | TTFT p50 | TTFT p99 | E2E p50 | E2E p99 | TPOT p50 | TPOT p99 |
|----------|----------|----------|---------|---------|----------|----------|
| light (681) | 113.7ms | 262.9ms | 0.54s | 15.79s | 54.0ms | 72.3ms |
| medium (1600) | 113.3ms | 119.2ms | 0.54s | 16.17s | 56.8ms | 99.3ms |
| heavy (2678) | 113.5ms | 119.3ms | 0.63s | 17.36s | 59.9ms | 112.1ms |

## 脚本信息
- **脚本**: `benchmark/AFlex_bench/06_others/more_model/scripts/run_moe_bench.py`
- **日志**: `/tmp/moe_8gpu_workload_v3.log`
- **结果目录**: `benchmark/AFlex_bench/06_others/more_model/results/`
- **服务器日志**: `benchmark/AFlex_bench/06_others/more_model/logs/<deploy_name>/`

## 运行命令

```bash
cd /workspace/sglang-tier/benchmark/AFlex_bench/06_others/more_model/scripts

WORKLOADS="/workspace/sglang-tier/benchmark/AFlex_bench/02_macro_benchmark/workloads/workload_azure_code_light_real.jsonl,/workspace/sglang-tier/benchmark/AFlex_bench/02_macro_benchmark/workloads/workload_azure_code_medium_real.jsonl,/workspace/sglang-tier/benchmark/AFlex_bench/02_macro_benchmark/workloads/workload_azure_code_heavy_real.jsonl"

PYTHONUNBUFFERED=1 /workspace/env/sglang-tier/bin/python run_moe_bench.py \
  --deploy native_dp8,native_dp8_tier,pd_dp4,pd_dp4_tier,pdaf_dp2,pdaf_dp2_tier \
  --workload "$WORKLOADS" \
  --max-run-s 600 > /tmp/moe_8gpu_workload_v3.log 2>&1
```

## 关键修复记录
1. **AFDDecoderLayerMixin bug** — `qwen3_moe.py` 中 import 作用域问题，已修复
2. **mem-fraction-static=0.75** — MoE 模型 57GB，80GB 卡上需要精确计算 KV cache 分配
3. **Native DP Router** — 之前 workload 只发到第一个实例，已加 `sglang_router` round-robin 负载均衡
4. **PDAF DP2 端口冲突** — UCX/SCHED 端口需要 +1000 偏移避免两组实例冲突
5. **频率控制** — baseline 锁频 1410MHz 确保公平对比，Tier 让 DVFS 动态调频
