# v10 版本 -- 多卡 DVFS 调频修复 + 8 卡全量压测 + DP/TP 部署对比

## 概述

本版本以一次 **TP>1 部署下的多卡 DVFS 调频缺陷修复** 为核心：此前每个 TP=2 的 AF 进程只有一张卡被调频，另一张始终停在 auto-boost，导致 tier 方案能耗节省被低估约一半。修复后每个进程锁住其 TP 组内全部物理卡，并把锁频/决策日志收敛到 tp_rank0 以消除并发写损坏。配套完成：①修复 `ipc_cpp` 后端在 TP>1 时的张量通信兼容性；②8 卡 A800 上 5 种部署方案的全量压测与两个 tier 方案的修复后重跑；③新增 PD DP=2 部署方式并与 TP=2 做同卡数对照。详细评测结论见 `benchmark/energy_bench/versions/version2.md`（部署拓扑对比）与 `version3.md`（8 卡压测）。

代码改动统计（相对 v9）：

| 文件 | 变更 | 说明 |
|------|------|------|
| `srt/managers/scheduler.py` | ~+30 行 | 多卡 DVFS 控制器列表 + tp_rank0 gate |
| `srt/layers/afd.py` | +13 行 | `ipc_cpp` + TP>1 的 Broadcast 包装 + stream-ordered 别名 |
| `benchmark/.../run_fixed_qps_bench.py` | +54 行 | `run_workload` 支持按部署传入 GPU 映射（能耗按卡归属） |
| `benchmark/.../run_deploy_bench.py` | 新增 | 4 卡部署拓扑对比（含 pd_dp2 DP 方式） |
| `benchmark/.../run_8gpu_deploy_bench.py` | 新增 | 8 卡 5 方案全量压测（含多卡 NVML index 传递） |
| `benchmark/.../plot_{deploy,8gpu,8gpu_tier}.py` | 新增 | 对比图生成 |

<!-- SECTIONS -->

## 一、多卡 DVFS 调频修复（核心，`scheduler.py`）

### 1.1 问题

在 8 卡 tier 方案（PA/PF/DA/DF 各 TP=2）首轮跑完后发现：**同一个 TP=2 进程占两张物理卡，但只有一张被调频**，另一张始终停在 auto-boost。实测铁证：杀进程后 PF 进程的 GPU0 锁在 930MHz、GPU1 停在 210MHz。

两个根因叠加：
- **单卡控制器**：`self._dvfs_hw = DVFSController(device_index=单个index)` 只持一张卡的 NVML 句柄，`_apply_freq()` 只锁那一张。
- **脚本未传 NVML index**：8 卡脚本只设了 `AFD_IPC_PEER_DEVICE`，没设 `AFD_NVML_DEVICE_INDEX`，运行时 fallback 到 `torch.cuda.current_device()`，在 CUDA_VISIBLE_DEVICES 重映射下可能锁错卡；决策日志名全部为 `gpu-1`。

### 1.2 修复

- 引入 `self._dvfs_hw_list`，每个进程为其 TP 组内**全部**物理卡各建一个 `DVFSController`；`_apply_freq()` 对列表内所有卡统一 `lock_sm_clock()`。
- 物理卡列表来自新环境变量 `AFD_NVML_DEVICE_INDICES`（CSV），由 launcher/bench 脚本按 `CVD[base:base+tp]` 计算后传入；缺失时回退到单 index。
- **回归修复**：NVML 锁频是系统级的，锁频 + 决策日志只在 `tp_rank==0` 执行。否则 TP0/TP1 两 rank 并发写同一决策日志文件，产生交错损坏的 JSON（`{"{"...`）。

### 1.3 验证

修复后日志显示每进程持有正确卡对（PF→[0,1] PA→[2,3] DF→[4,5] DA→[6,7]），同实例两卡频率完全同步（如 DF 的 4,5 一起跳 1170、DA 的 6,7 一起回 930），决策日志 `ok=57 bad=0` 无损坏，文件名带正确 GPU index。**修复前 tier 数据只调了一半卡，能耗节省被低估；两个 tier 方案已用修复后代码完整重跑。**

## 二、`ipc_cpp` 后端 TP>1 兼容修复（`afd.py`）

`get_tensor_communicator()` 在 `ipc_cpp` + `local_tp > 1` 时，原本各 rank 独立 IPC，导致 hidden_states 损坏、rotary_embedding shape 报错。修复：对该场景用 `BroadcastTensorCommunicator` 包装底层 `CppIpcTensorCommunicator`（仅 rank0 持有真实 inner_comm，其余 rank 走 broadcast），并为 `BroadcastTensorCommunicator` 补 `send_stream_ordered`/`recv_stream_ordered` 别名以兼容调用方。这是 8 卡 AF（每进程 TP=2）能跑通的前提。

## 三、Benchmark 改动

### 3.1 `run_fixed_qps_bench.py`：按部署传入 GPU 映射
`run_workload` 新增 `gpu_indices/prefill_gpus/decode_gpus` 参数（缺省回退到模块级默认），使不同部署拓扑的 P/D 能耗能按各自 GPU 映射正确归属（NVML 按卡采集）。向后兼容。

### 3.2 `run_8gpu_deploy_bench.py`：8 卡 5 方案压测
新增脚本，参数化 8 卡部署：pd_p4d4 / pdaf_8g_m1 / pdaf_8g_m2 / pdaf_8g_m1_tier / pdaf_8g_m2_tier。`_owned_gpus(cvd, base, tp)` 计算每进程物理卡并通过 `AFD_NVML_DEVICE_INDICES` 传入（多卡 DVFS 修复的脚本侧配套）。

### 3.3 `run_deploy_bench.py`：4 卡部署拓扑对比 + DP 方式
新增 `pd_dp2`（PD DP=2）：两个独立 1P1D 实例（各占 2 卡、TP=1、独立 bootstrap port），用完整 router（`round_robin`）做请求级负载均衡。配套 `start_pd_dp` + `start_router_multi`（多 prefill/decode 端点）。顺带修复 `cleanup_procs` 误 reset 他人 GPU（改为按当前方案 GPU reset）。

## 四、测试进度与结论

### 4.1 8 卡全量压测（version3）
5 方案全部测完，两个 tier 方案修复后重跑。核心结论：
- DVFS 在健康 QPS（SLO 0%）下稳定回收 **21-26%** 能耗，M=1/M=2 收益相当，prefill-heavy 负载最稳定。
- 与同 8 卡纯 PD TP4 横向对比：PD+AF DVFS 在 prefill-heavy 明显胜出（il2048 q3 省约 30%，il4096 q2 省约 29%）。
- 逼近 collapse 时 decode 预测器低估约 67%，导致 DVFS 压频过度、TTFT 崩溃（il2048 q6 tier TTFT 16s）——校准 decode 预测器是 DVFS 推向高 QPS 的前置条件。

### 4.2 DP=2 vs TP=2（version2/version3 补充，4 卡）
同 4 卡下吞吐基本持平，但 TP=2 延迟/能效稳定略优（J/tok 低 15-20%）；decode 重负载是分水岭：`il256_ol512 q4` TP=2 扛住（0% 违背），DP=2 崩溃（TTFT 16462ms、75% 违背），因 DP=2 每实例 decode 只有单卡 KV 容量。结论：单机 NVLink + dense 模型下优先 TP 切分。

### 4.3 产物
- 数据：`benchmark/energy_bench/scripts/bench/results/{8gpu,deploy}/json/`
- 图表：`8gpu_tier_tradeoff.png`、`8gpu_tier_generalization.png`、`deploy_tier_tradeoff.png`（6 路对比）
- 文档：`benchmark/energy_bench/versions/version2.md`（部署对比）、`version3.md`（8 卡压测 + DP/TP）
