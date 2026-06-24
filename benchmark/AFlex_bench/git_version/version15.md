# Version 15: 16 卡双节点多架构基准 + Tier1 优雅重载 + Mixtral MoE 测试 + 能耗模型/依赖修复

## 概述

本次版本是一个综合提交，包含多个并行推进的工作线。主体是新增 **16 卡双节点多架构基准**
（`multi_node/`），同时合入了 **Tier1 优雅重载（graceful reload）** 核心机制、**Mixtral-8x7B
MoE 模型测试**、能耗模型加载兼容性修复、通信带宽测试、以及构建依赖调整。

> 这是一次"攒了多批改动一起提交"的版本。下面按工作线分别说明；§9 给出完整文件清单。

---

# 一、16 卡双节点多架构基准（multi_node/，本版本主体）

将单节点微基准（`retesting/scripts/run_micro_bench.py`）扩展到 **16 卡 / 双 8 卡节点**，
新增 `benchmark/AFlex_bench/multi_node/`。在两台 8×A800 节点上对比
**4 种架构 × 2 种功耗模式**，由 node1 宿主机统一编排（`ssh + docker exec`）双节点容器。

- node1 `10.252.129.36`（Prefill 侧）、node2 `10.252.129.35`（Decode 侧），各 8×A800-80GB。
- 跨节点 KV：mooncake RDMA over RoCE `mlx5_bond_0`；节点内 Attn↔FFN：CUDA IPC（`ipc_cpp`）。
- 模型 `Qwen3-32B`（Dense）。

## 1.1 四种架构

| arch | 说明 | GPU 布局 |
|---|---|---|
| `native_dp` | Native DP16，16×TP=1 实例 + round_robin router | node1 GPU0-7 + node2 GPU0-7 |
| `pd_dp_xnode` | PD DP8，8 对 P-D，**跨节点**配对 | P=node1 GPU0-7，D=node2 GPU0-7 |
| `pd_dp_intra` | PD DP8，8 对 P-D，**节点内**配对 | 每节点 4P(GPU0-3)+4D(GPU4-7) |
| `pdaf` | PDAF 4PA+4PF+4DA+4DF（AF 算子拆分，tp=4） | P=node1 8 卡，D=node2 8 卡 |

两种功耗模式：`baseline`（GPU 锁频 1410 MHz）/ `tier`（按能耗模型 per-instance DVFS，
Native/PD 用 `--dvfs-*`，PDAF 用 `--afd-dvfs-*` + `--afd-dvfs-idle-lock`）。

## 1.2 关键架构理解

- **AFD 算子拆分（Attn↔FFN）只能在节点内**：`ipc_cpp` 走 CUDA IPC，无法跨节点。→ PDAF 的
  Prefill 整占 node1、Decode 整占 node2，唯一跨节点的是 PD KV 传输与 router HTTP。
- **`AFD_IPC_PEER_OFFSET=±tp`**：FFN peer=local+tp、ATTN peer=local−tp，tp=4 每 rank 正确配对。
- **网络**：纯 IB 口（`mlx5_0/1/4/5`）的 IPoIB 未起、`ibdev2netdev` 显示 Down 但 RDMA ACTIVE；
  跨节点 mooncake 走有可路由 IP 的 RoCE `mlx5_bond_0`（已 smoke 验证）。

## 1.3 解决的关键问题

| 问题 | 根因 | 解决 |
|---|---|---|
| 容器内无法编排 | 容器无 `docker` 命令 | 编排器移到宿主机 |
| 容器间 ssh 不通 | 容器缺 ssh 私钥 | 把宿主机 ssh key 拷入容器 |
| `EADDRINUSE`（torch dist） | 同节点多服务分布式端口冲突 | 每服务独立 `--nccl-port` |
| router 残留占端口 | router 进程名是 `sglang::router`，`pkill -f` 漏杀 | `cleanup_node.sh` 按进程名 + 端口段 kill |
| 工负载超时 | 低 QPS 长尾请求晚到 | 运行窗口 `last_arrival+150s` |
| 跨节点 mooncake 走错网卡 | 纯 IB 口 IPoIB 未起 | 用 RoCE `mlx5_bond_0` |

## 1.4 验证状态与结果

四架构 × 两模式均已逐个 smoke 验证（chatbot QPS1）。**PDAF 完整结果**
（`results/pdaf_16gpu_tp4_*.json`，4 场景 × QPS1-4，全部 SLO 0%）：chatbot 670→844 tok/s，
TTFT 随 input_len 递增，mJ/token 随 QPS 上升而下降，Decode 侧能耗高于 Prefill 侧。
完整「4 架构 × 2 模式 × 4 场景 × QPS{1,2,4}」批跑脚本就绪，本版本未跑全量。

详见 `multi_node/README.md`。

---

# 二、Tier1 优雅重载（graceful reload / drain-then-switch）

给 AFD Tier1 重规划（TP/k 变化触发的 full reload）增加"先排空再切换"机制，把 reshard 的服务
中断从"全杀全启 ≈50s、期间全部 503"压缩到"已在 decode 的请求不受影响、中断 ≈5-9s"。

## 2.1 调度器侧（`managers/scheduler.py`）

- `_trigger_full_reload(old, new)`：新增 `old` 入参（graceful 需要旧拓扑做 diff 与排空）。
- 默认走 `graceful_orchestrator`（drain-then-switch）；`--tier1-no-graceful-reload` 时回退
  旧的 `reload_orchestrator`（kill-all）。
- reload config 增加 `old_solution` 与 `drain_timeout_s`。

## 2.2 新增编排器（`energy/graceful_orchestrator.py`）

独立进程，流程：识别 TP 变化的模块 → 通知 router 排空（停新流量）→ 并行启动新模块 →
等旧模块 idle（可超时）→ 杀旧模块 → 通知 router 激活新模块 → 写 ready 信号。

## 2.3 信号协议（`energy/reload_signal.py`）

- 状态机扩展：`idle / draining / starting / switching / ready / error`（原只有 idle/reloading/ready）。
- 新增 `write_signal()` 统一写入；`is_reloading()` 覆盖所有非终态。

## 2.4 Router 排空 API（`sgl-model-gateway/.../mini_lb.py`）

- `MiniLoadBalancer` 增加 `draining_prefill_urls / draining_decode_urls / is_draining_all` 状态。
- `select_pair()` 过滤掉正在排空的 server；全部排空时 `/generate` 返回 503。
- 新增 admin 端点 `/admin/drain_module`、`/admin/activate_module`（按 URL 标记/恢复）。

## 2.5 server_args（`server_args.py`）

- `--tier1-graceful-reload`（默认 True）/ `--tier1-no-graceful-reload`（关闭，回退 kill-all）。
- `--tier1-drain-timeout`（默认 30s，超时后 abort 剩余请求并强杀）。

## 2.6 HTTP 接口（`entrypoints/http_server.py`）

- 新增 `GET /is_idle`：返回 `{idle, inflight}`，供 graceful orchestrator 探测模块是否排空完毕。

## 2.7 设计文档与验证（`benchmark/AFlex_bench/reshard/`）

- `graceful_reload_impl.md` / `rebuild.md`：增量权重继承 + 细粒度阻塞的设计。
- `bench_reshard.py` / `test_reshard.py` / `plot_reshard.py` + results/charts/logs：重载中断耗时测试。
- 单测 `test/srt/test_graceful_reload.py`：信号状态转移、router 排空 API、idle 探测与超时 abort、
  模块 diff 检测。

---

# 三、Mixtral-8x7B MoE 模型测试（06_others/Mixtral_test/）

在已有 Qwen3-30B-A3B MoE 测试基础上，换用新 MoE 模型 **Mixtral-8x7B** 跑一套部署 + 能耗 + 调频测试。

- `test.md`：从原始任务清单更新为完整测试报告（Native DP2 / PD 部署验证、能耗模型采集、
  Tier 调频、定长 + Azure 真实数据集）。Mixtral-8x7B 用 TP=2/实例；PDAF 需 8 GPU（4 卡环境不适用）。
- 新增脚本：`run_micro_bench.py`、`run_hetero_pdaf_bench.py`、`profile_energy.py`、`profile_tp4.py`、
  `profile_v2_*.py`、若干 `run_*.sh` 启停脚本。
- 新增 `macro_benchmark/`、`micro_benchmark/` 结果目录。
- **删除旧目录** `06_others/more_model/`（27 个文件，Qwen3-30B-A3B 的旧 MoE 测试脚本/结果），
  被 Mixtral_test 取代。

---

# 四、能耗模型加载兼容性修复（energy/af_profile_predictor.py）

- `_ModelUnpickler` 重构：明确区分 V1（`energy_model` / `energy_model_v1`）与 V2/V3
  （`energy_model_v2/v3`）模型类的反序列化，V1 类按需从 `benchmark/test_motivation/energy_model.py`
  动态加载并缓存；`__main__` 提供回退映射。修复不同版本能耗模型 pkl 混用时的反序列化失败。
- V2 fallback 预测特征修正：Decode_iter 延迟/能量预测的特征向量从 `[tp_a, M, f_a, f_f, il, bs]`
  改为 `[M, f_a, f_f, il, bs]`（V2 单 tp 模型不含 tp 维），消除特征维度不匹配。

---

# 五、通信带宽测试（06_others 下新增 comm/）

NVLink vs RDMA 带宽对比：`bench_gpudirect_rdma.py`、`run_rdma_bw_test.sh`、`run_rdma_send_test.sh`、
`plot_nvlink_vs_rdma.py` + 结果图 `nvlink_vs_rdma.png/pdf` 与 `read.md`。

---

# 六、构建依赖与基准脚本调整

## 6.1 依赖（pyproject.toml）

- `python/pyproject.toml`：`ucx-py-cu12==0.45.0` 从可选 extra `afd-ucx` 提升为主依赖
  （`libucx-cu12` 由其传递引入，不再显式 pin 1.19.0 以免 resolver 冲突）；新增 `socksio`
  （`all_proxy`/clash 环境下 httpx 走 SOCKS 代理，避免 pip/maturin 构建失败）。
- `sgl-model-gateway/bindings/python/pyproject.toml`：build 依赖加 `socksio`；readme 指向
  本地 `README.md`，去掉 maturin 的 README exclude。

## 6.2 retesting/scripts/run_micro_bench.py

- PD DP 的 prefill/decode 启动支持 `--tier` 时附加 `--dvfs-*`（与 PDAF 的 `--afd-dvfs-*` 对应）。
- 请求体加 `ignore_eos: True`（定长输出，保证 TPOT/吞吐口径一致）。
- 其他：memlock unlimited、端口清理等小修。
- 另有未跟踪辅助文件 `run_micro_bench_patched.py`、`retesting/micro_benchmark/{4gpu,8gpu}/` 下
  补充的 plot 脚本与 8gpu 结果。

---

# 七、建议 Commit Message

```
Multi-node 16-GPU benchmark + Tier1 graceful reload + Mixtral MoE tests.

- multi_node/: orchestrate two 8xA800 nodes from node1 host (ssh+docker exec);
  compare native_dp / pd_dp_xnode / pd_dp_intra / pdaf in baseline & tier (DVFS)
  modes; AF IPC stays intra-node, cross-node PD KV via mooncake RoCE.
- Tier1 graceful reload: drain-then-switch orchestrator + router drain API
  (/admin/drain_module, /admin/activate_module) + /is_idle endpoint +
  --tier1-graceful-reload / --tier1-drain-timeout; cuts reshard downtime from
  ~50s (kill-all) to ~5-9s with no impact on in-decode requests.
- Mixtral-8x7B MoE test suite (replaces old more_model dir).
- Fix energy-model unpickler V1/V2 compatibility and V2 decode feature vector.
- Add NVLink-vs-RDMA comm benchmark; deps: promote ucx-py-cu12 to main, add socksio.
```

---

# 八、备注

- `multi_node/logs/` 已被 `.gitignore` 忽略（GB 级服务 stdout），不进 commit。
- 这些改动来自多条工作线、跨越多日，统一在本次一起提交。

---

# 九、本次改动文件清单

## 核心源码（M）

```
M python/sglang/srt/managers/scheduler.py          # graceful reload 触发
M python/sglang/srt/server_args.py                 # --tier1-graceful-reload / --tier1-drain-timeout
M python/sglang/srt/entrypoints/http_server.py     # GET /is_idle
M python/sglang/srt/energy/reload_signal.py        # 信号状态机扩展 + write_signal
M python/sglang/srt/energy/af_profile_predictor.py # 能耗模型 unpickler V1/V2 兼容 + 特征修正
M sgl-model-gateway/bindings/python/src/sglang_router/mini_lb.py  # router 排空 API
M python/pyproject.toml                             # ucx-py 转主依赖 + socksio
M sgl-model-gateway/bindings/python/pyproject.toml  # socksio + readme 路径
M benchmark/AFlex_bench/retesting/scripts/run_micro_bench.py      # PD tier DVFS + ignore_eos
```

## 新增（A / 未跟踪）

```
A python/sglang/srt/energy/graceful_orchestrator.py
A test/srt/test_graceful_reload.py
A benchmark/AFlex_bench/multi_node/                 # 16 卡多架构基准（README/scripts/results/charts）
A benchmark/AFlex_bench/git_version/version15.md
A benchmark/AFlex_bench/reshard/                    # graceful reload 设计文档 + reshard 测试
A benchmark/AFlex_bench/comm/                       # NVLink vs RDMA 带宽测试
A benchmark/AFlex_bench/06_others/Mixtral_test/{macro_benchmark,micro_benchmark,scripts/...}  # Mixtral MoE 测试
A benchmark/AFlex_bench/03_sensitivity/slo_sweep/READ.md
A benchmark/AFlex_bench/retesting/micro_benchmark/8gpu/
A benchmark/AFlex_bench/retesting/micro_benchmark/4gpu/plot_baseline_vs_tier.py
A benchmark/AFlex_bench/retesting/micro_benchmark/4gpu/plot_old_vs_new.py
A benchmark/AFlex_bench/retesting/scripts/run_micro_bench_patched.py
```

## 删除（D）

```
D benchmark/AFlex_bench/06_others/more_model/...    # 27 个文件，旧 Qwen3-30B-A3B MoE 测试，被 Mixtral_test 取代
D benchmark/AFlex_bench/06_others/Mixtral_test/scripts/test_pdaf_deploy.sh
```
