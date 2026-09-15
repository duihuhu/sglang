# AFlex RQ5/RQ6 多机基准套件

本目录是自包含、可恢复的 1–4 节点（每节点 8 GPU）评测工具。默认命令均只做生成、校验或 dry-run；只有 `run_benchmark.py --execute` 会启动服务。实验点统一标记 `experimental` 且需要预检，默认保持 `requires_preflight`；只有同时传入 `--allow-experimental` 才会显式解锁。

## 快速开始

```bash
python3 scripts/generate_workloads.py
python3 -m pytest -q
python3 scripts/run_benchmark.py --matrix configs/rq5_32gpu.json --inventory --dry-run
python3 scripts/run_benchmark.py --matrix configs/rq6_scaling.json --priority 1 --dry-run
python3 scripts/run_benchmark.py --matrix configs/smoke.json --smoke --dry-run
python3 scripts/run_benchmark.py --matrix configs/smoke_32gpu.json --smoke --dry-run
python3 scripts/preflight.py   # 只读；失败会写 results/preflight.json
```

单节点 Smoke 覆盖 Dense、MoE 两模型的 native/PD/AF/PDAF 共 8 点；每点固定为单节点 8 GPU、`fixed_short`、QPS 1，并且仅由 `--smoke --dry-run` 清点，不启动真实服务。资源划分为 native TP8×1、PD P/D 各 1×TP4、AF F4/A4、PDAF PF2/PA2/DF2/DA2。`smoke_32gpu.json` 的四个 32 GPU 点通过 `recipe` 适配遗留拓扑：`legacy_native_tp`、`legacy_pd_dual`、`legacy_af_profile_replicas`、`legacy_pdaf_3p1d`。其中 AF 点当前整体标记为 `experimental=true`、`requires_preflight=true`：TP2 shared 通信成功后发生 CUDA index 越界（`results/smoke_32gpu_operator_test_af_shared/51314528102c901b/`），TP2 per-rank 卡在首 tensor（`results/smoke_32gpu_operator_test_af_per_rank/73dfe6f7475522ff/`），TP1 的 16 副本有部分进程退出（`results/smoke_32gpu_operator_test_af_tp1/9b492780bc056ce8/`）。默认 smoke 清点显示 `requires_preflight` 且不会执行；单独实验须显式加 `--allow-experimental`。Native/PD/PDAF 保持 `ready`。通用 builder 仍服务其他矩阵，但不再用于这四个 smoke 拓扑。配方来源和精确形状见 `docs/runbook.md`。

部署生命周期使用 `ProcessSpec.startup_stage` 描述依赖：stage 按升序执行，同一 stage 并行 launch，再并行等待存活与 health；任一失败仍进入统一日志收集与清理。manifest 会记录每个进程的 stage，具体配方分层见 `docs/runbook.md`。

RQ5 覆盖两模型、native/PD/AF/PDAF、6 个定长与 conv/code、QPS 1/4/8。RQ6 覆盖 1/2/3/4 节点典型点。原始请求记录逐 token 时间戳、ITL、客户端/服务端 TTFT、TPOT、E2E；summary 给出 p50/p90/p95/p99。能耗以 NVML 累计毫焦读数差值计算并输出 GPU、节点、集群三级焦耳。详见 `docs/`。

node3/node4 的 16 GPU 矩阵见 `configs/smoke_16gpu_node34.json`。minimal AF 已通过（`results/node34_af_minimal_single/d5849d869704f545/`）；AF pool 跳过逐 endpoint warmup，使用 90 秒请求超时，仍需 `--allow-experimental`。PDAF 的推理路径已有逐 token 产出证据，使用 120 秒请求超时并保留 router warmup。Native/PD 不变。

QPS policy：正式 workload 与 matrix point 的硬上限为 16，超限配置会触发 `ConfigError`；历史生成 trace 可保留但不进入正式矩阵。node3/node4 AF 在 scheduler-local perspective 修复后已通过 1/2/4/8 副本阶梯，8 副本 artifact 为 `results/node34_af_eight_replicas/a07927517bec675e/`，现为非实验 `ready`。PDAF 的 64 个长输出请求现象属于并发堆积而非部署失败，后续使用 QPS ≤16 及 admission/inflight 控制。

## node3/node4 统一 QPS 16 轻量测试

`configs/bench_16gpu_node34_qps16.json` 沿用 16 GPU smoke 的四种部署配方和资源划分，四点统一运行 `fixed_qps16_light`（64 请求、Poisson open-loop QPS 16、input 128/output 64）并设置 `max_inflight=16`。Native、PD、AF 的请求超时为 60 秒；AF 已完成多机 preflight 且跳过 warmup；PDAF 保留每架构一次默认 warmup（不计入 body 能耗），请求超时为 90 秒。只做清点：

```bash
python3 scripts/run_benchmark.py --matrix configs/bench_16gpu_node34_qps16.json --smoke --dry-run
```

runner 保留所有原始计划到达时间，但只有取得 admission semaphore 后才发送；服务积压时不会丢弃请求，`arrival_lag_ms` 会显式记录排队导致的实际发送偏差。

## 历史 Tier1 Code QPS16 复现

`configs/reproduce_tier1_code_qps16_node34.json` 使用 `legacy_tier1_layout` 精确重建 node3/node4 上的历史 7P+1D PDAF 布局，并直接引用原始 800 请求 trace，不复制大文件。`QPS=16` 只描述 open-loop 到达速率，**不等于** `max_inflight=16`；历史复现点不设置 admission semaphore，因此不会用并发上限改变原始到达过程。单请求超时设为 180 秒，覆盖历史约 69 秒运行窗口。

只做编译与清点（不会启动服务、写 IB JSON 或锁频）：

```bash
python3 scripts/run_benchmark.py --matrix configs/reproduce_tier1_code_qps16_node34.json --smoke --dry-run
```

## node3/node4 持久 AF QPS 1–16 sweep

`run_af_qps_sweep.py` 使用 `af_node34_a1f1_pool` 一次部署 8 个 A1/F1 replica，跳过已完成的 preflight/warmup，在同一生命周期内依次测试 QPS 1–16，最后统一清理。默认仅编译计划，不会部署：

```bash
python3 scripts/run_af_qps_sweep.py --plan
```

确认计划后才显式执行；每档 frozen workload、raw request、system/energy、summary/status 分别写入 `results/sweep/run/`：

```bash
python3 scripts/run_af_qps_sweep.py --execute
python3 scripts/run_af_qps_sweep.py --start-qps 8 --end-qps 16 --execute
python3 scripts/run_af_qps_sweep.py --continue-on-partial --execute
```

每档 32 个 input=128/output=64 请求，使用按 QPS 独立固定 seed 的 Poisson 到达；QPS 仅表示 offered arrival rate，所有档统一 `sweep_max_inflight=16`，单请求 timeout 45 秒。每档 wall limit 为 `min(90, max(60, last_arrival + 45 + 5))` 秒，admission 排队仅记录 `arrival_lag_ms`，wall limit 到达时只取消仍未完成的请求。默认任一请求失败即将该档记为边界并停止；`--continue-on-partial` 只允许部分请求失败后继续，0 成功、endpoint 探测失败、进程退出或 fatal 日志仍会立即停止。总览写入 `sweep_summary.json`，逐档记录 `offered_qps`、`max_inflight`、`arrival_span` 和 `wall_limit`。
