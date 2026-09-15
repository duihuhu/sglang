# 运行手册

1. 生成工作负载：`python3 scripts/generate_workloads.py`。若真实 conv/code trace 缺失，会生成明确标记为 `synthetic_proxy` 的确定性代理，不伪装为真实 trace。正式工作负载和每个 point 的 QPS 上限均为 16；配置加载会拒绝任何大于 16 的值。已有高 QPS trace 可留作历史 artifact，但不纳入正式矩阵。
2. 运行单元检查：`python3 -m pytest -q` 与 `python3 -m py_compile scripts/*.py src/aflex_benchmark/**/*.py`。
3. 只读预检：`python3 scripts/preflight.py`。它只执行 SSH、容器/路径/GPU查询，不启动服务、不写远端、不改时钟；无论成功失败都写本地 JSON。
4. 清点：`python3 scripts/run_benchmark.py --matrix configs/rq5_32gpu.json --inventory --dry-run`。用 `--priority 1` 过滤核心点。
5. 冒烟清点：`python3 scripts/run_benchmark.py --matrix configs/smoke.json --smoke --dry-run`。该矩阵固定包含 Dense/MoE × native/PD/AF/PDAF 的 8 点，每点为单节点 8 GPU、`fixed_short`、QPS 1；资源分别为 native TP8×1、PD P/D 各 1×TP4、AF F4/A4、PDAF PF2/PA2/DF2/DA2。此命令不得添加 `--execute`。
6. 32 GPU 遗留配方冒烟清点：`python3 scripts/run_benchmark.py --matrix configs/smoke_32gpu.json --smoke --dry-run`。四点均为 Qwen3-32B、`fixed_short`、QPS 1。默认队列中 Native/PD/PDAF 为 `ready`，AF 为 `requires_preflight`，统计中的 `blocked` 包含该 AF run；清点命令不得添加 `--execute`：
   - `legacy_native_tp`：来源 `multi_node/more_test/Ablation/node_scalibility/scripts/run_4node_scalability_benchmark.py::_deploy_native_tp`；4 节点各 1×TP8，顶层 round-robin。
   - `legacy_pd_dual`：来源同文件 `_deploy_pd_instance`/`deploy_pd_dual`；两个 P/D 节点对，每对 2×P(TP4)+4×D(TP2)，各自 PD subrouter，再由 top router 汇聚。
   - `legacy_af_profile_replicas`：该 AF 点整体为 `experimental=true`、`requires_preflight=true`，默认不可执行。当前三种 32 GPU 实测均未通过：TP2 `shared_tp0` 通信成功后 CUDA index 越界，artifact 为 `results/smoke_32gpu_operator_test_af_shared/51314528102c901b/`；TP2 `per_rank` 卡在首 tensor，artifact 为 `results/smoke_32gpu_operator_test_af_per_rank/73dfe6f7475522ff/`；TP1 每节点 4 个 pair、共 16 副本时部分进程退出，artifact 为 `results/smoke_32gpu_operator_test_af_tp1/9b492780bc056ce8/`。配方仍可供用户单点实验：在人工预检后，同时指定 AF 的 `--point-id`、`--execute` 和 `--allow-experimental` 显式解锁。AF-only 命令不含 PD flags。
   - `legacy_pdaf_3p1d`：来源 `run_4node_scalability_benchmark.py::deploy_aflex_3p1d_on_node`/`deploy_aflex_4node`；每节点 3 个 prefill F1/A1 pair + 1 个 decode F1/A1 pair，3 层 subrouter/node router/top router；baseline 明确不启用 DVFS。原 3p1d 的同 pair 两侧使用相同 sched port，隐式推导可工作；本 adapter 仍为每个 pair 显式设置相同且全局唯一的 channel base，避免依赖该推导。
   `ProcessSpec` 清点 HTTP、bootstrap、NCCL、UCX、sched 端口，并用独立的 `startup_stage` 字段记录启动依赖；该字段也写入 deployment manifest。生命周期按 stage 升序建立屏障：同 stage 的进程并行 launch，全部 launch 完成后再并行检查存活、等待 health（沿用配置的 health timeout）及执行激活 warmup；任一任务失败会取消尚未开始的同批任务并进入原有日志收集/清理流程。遗留配方阶段为 Native server 0/router 1；PD P 0、D 1、subrouter 2、top 3；AF 所有 F 0、所有 A 1；PDAF PF 0、PA 与 DF 1、DA 2、subrouter 3、node router 4、top 5。这样 AF/PDAF 同类副本并发启动，同时保持 F/A handshake 的紧邻依赖。shared sched 只在 pair 的一侧登记以避免重复占用误报；metadata 记录 `channel_strategy`/`channel_id`（shared 模式为 `None`），并保留 channel base。宿主机 deep cleanup 由 `cluster_operator_test.json::host_deep_cleanup_cmd` 经 SSH 直接执行，并在执行前后检查 GPU compute app。
7. node3/node4 的 16 GPU 清点使用 `configs/smoke_16gpu_node34.json`。AF 的 scheduler-local perspective 修复后，A1/F1 的 1、2、4、8 副本阶梯均已成功；8 副本（16 GPU）完整 artifact 为 `results/node34_af_eight_replicas/a07927517bec675e/`，8/8 请求成功。因此该 AF point 已标记 `multinode_preflight_complete=true`、`experimental=false`，默认状态为 `ready`，仍保留 `skip_warmup=true`。PDAF 的 router warmup 与部署路径已通过；此前 64 个长输出请求在客户端堆积并超时，是无限制并发造成的长输出 backlog，不是部署失败。artifact `results/node34_16gpu_pdaf/5107d6a84244456b/requests.jsonl` 已记录逐 token 产出。后续 PDAF 正式运行必须保持 QPS ≤16，并增加 admission/inflight 控制，避免一次性放入全部长请求；当前 point 保留 `request_timeout_s=120`。Native/PD 配置不变。
8. node3/node4 的统一轻量 QPS 16 清点使用 `python3 scripts/run_benchmark.py --matrix configs/bench_16gpu_node34_qps16.json --smoke --dry-run`。四点沿用 smoke 配方及 16 GPU 资源，统一 `fixed_qps16_light`、QPS 16、`max_inflight=16`。64 个任务仍按固定 seed 的 Poisson/open-loop arrival 调度；到达后若 16 个 admission slot 已满则等待，发送偏差写入 `arrival_lag_ms`。Native/PD/AF 请求 timeout 60 秒，AF 已 preflight 且跳过 warmup；PDAF 保留一次默认 warmup（发生在能耗 body 前），timeout 90 秒。
9. 经人工确认后才可加 `--execute`。实验点还必须显式加 `--allow-experimental`；该开关不影响非实验点，也不会解除普通 `blocked` 状态。建议始终配合 `--point-id` 单独运行实验点。结果按稳定 run_id 存放，已有 complete summary 自动跳过。
10. 汇总：`python3 scripts/analyze_results.py --results results --matrix configs/smoke_32gpu.json --smoke --output results/aggregate.json`。带 `--matrix` 时报告附带队列 `ready`/`blocked`/`requires_preflight` 统计；若只分析已落盘结果则可省略。

故障恢复：查看 run 的 summary/error 和 logs；修复后重跑同命令。清理和 GPU 解锁位于 finally，但高成本执行前仍应人工确认无其他作业。


## 历史 Tier1 Code QPS16 复现

`configs/reproduce_tier1_code_qps16_node34.json` 使用 `legacy_tier1_layout` 精确重建 node3/node4 上的历史 7P+1D PDAF 布局，并直接引用原始 800 请求 trace，不复制大文件。`QPS=16` 只描述 open-loop 到达速率，**不等于** `max_inflight=16`；历史复现点不设置 admission semaphore，因此不会用并发上限改变原始到达过程。单请求超时设为 180 秒，覆盖历史约 69 秒运行窗口。

只做编译与清点（不会启动服务、写 IB JSON 或锁频）：

```bash
python3 scripts/run_benchmark.py --matrix configs/reproduce_tier1_code_qps16_node34.json --smoke --dry-run
```

## AF QPS 1..16 sweep

`configs/sweep_af_qps1_16_node34.json` 中的 QPS 只控制 Poisson/open-loop 到达速率，不控制并发。所有 QPS 档统一使用 `sweep_max_inflight=16`；到达后等待 admission slot 只增加 `arrival_lag_ms`，本身不判为请求失败。每档请求 timeout 为 45 秒，point wall limit 按 `min(point_wall_cap_s, max(60, last_arrival + request_timeout_s + 5))` 动态计算，当前 cap 为 90 秒，以覆盖完整到达窗口及末个请求的完成时间。wall limit 到达时只把仍未完成的 future 标为 point wall timeout。

只编译计划、不执行部署：

```bash
python3 scripts/run_af_qps_sweep.py --plan
```
