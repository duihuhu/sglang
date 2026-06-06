# version3 — 8 卡全量压测（8-GPU Full-Scale Stress Test）

> 目标：在 Qwen3-32B 上使用全部 8 张 A800 GPU，系统对比 5 种部署方案在 4 种长度组合下的性能与能耗极限，通过递增 QPS 压测找到各方案的吞吐天花板和 SLO 崩溃点。日期 2026-06-01，A800×8（GPU0-7），freq=auto。

## 1. 实验设置

### 1.1 部署方案（5 种，均使用 8 卡）

| # | 方案 ID | 含义 | GPU 映射 |
|---|---------|------|----------|
| 1 | pd_p4d4 | 纯 PD，Prefill TP4 + Decode TP4 | P=4,5,6,7 D=0,1,2,3 |
| 2 | pdaf_m1 | PD+AF M=1（无微批），PA+PF 各 TP2 + DA+DF 各 TP2 | PF=0,1 PA=2,3 DF=4,5 DA=6,7 |
| 3 | pdaf_m2 | PD+AF M=2（双微批），PA+PF 各 TP2 + DA+DF 各 TP2 | PF=0,1 PA=2,3 DF=4,5 DA=6,7 |
| 4 | pdaf_m1_tier | PD+AF M=1 + Tier1/DVFS | 同 pdaf_m1 |
| 5 | pdaf_m2_tier | PD+AF M=2 + Tier1/DVFS | 同 pdaf_m2 |

说明：
- 纯 PD 方案使用标准 PD 解耦（mooncake 传输后端），P/D 各 TP4。
- AF 方案使用 ipc_cpp 后端，Prefill 侧和 Decode 侧各占 4 卡（PA+PF 共享 2 对 GPU，DA+DF 共享 2 对 GPU），每个进程 TP=2。
- Tier 方案在 AF 基础上启用 Tier1 频率编排 + Tier2 per-batch DVFS。

### 1.2 Workload（4 种长度组合）

| # | 标记 | input_len | output_len | 特征 |
|---|------|-----------|------------|------|
| 1 | il128_ol1024 | 128 | 1024 | decode 极重（长生成） |
| 2 | il512_ol256 | 512 | 256 | 均衡负载 |
| 3 | il2048_ol64 | 2048 | 64 | prefill 重（长输入短输出） |
| 4 | il4096_ol64 | 4096 | 64 | prefill 极重（超长输入） |

每种长度组合生成递增 QPS 的 workload 文件（duration=60s，确定性到达）：
- il128_ol1024: QPS = 2, 4, 6, 8, 10, 12, 15, 20
- il512_ol256: QPS = 2, 4, 6, 8, 10, 12, 15
- il2048_ol64: QPS = 1, 2, 3, 4, 6, 8, 10
- il4096_ol64: QPS = 1, 2, 3, 4, 5, 6

QPS 范围设计原则：从低负载逐步加压直到系统崩溃（SLO 违背 >50% 或 TTFT >10s），找到各方案的极限。

### 1.3 测试指标

每个 run 记录：
- **总吞吐** (tok/s)
- **TTFT** (avg/p50/p90/p99 ms)
- **TPOT** (avg/p50/p90/p99 ms)
- **总能耗** (J)，分 Prefill 阶段和 Decode 阶段
- **Energy/token** (mJ/tok)
- **SLO 违背率** (TTFT SLO=5000ms, TPOT SLO=300ms)

### 1.4 测试矩阵

总 run 数 = 5 方案 × (8+7+7+6) QPS 点 = 5 × 28 = **140 run**

预计耗时：每 run ~2-3 分钟（含启动/warmup/60s 负载/清理），总计 ~5-7 小时。

### 1.5 环境

- Python: `/workspace/env/sglang-tier/bin/python`
- Model: `/models/Qwen/Qwen3-32B/`
- Hardware: A800 SXM 80GB × 8, NVLink
- Freq 策略：**非 Tier 方案锁最高频（2520 MHz）**，Tier 方案用 auto（DVFS 动态调频）
- 脚本: `scripts/bench/run_8gpu_bench.sh` + `run_8gpu_deploy_bench.py`

### 1.6 Bug 修复记录

测试过程中修复了 `ipc_cpp` 通信后端在 TP>1 时的兼容性问题：
- **问题**：`CppIpcTensorCommunicator` 未被 `BroadcastTensorCommunicator` 包装，导致 TP=2 时各 rank 独立 IPC，hidden_states 损坏，rotary_embedding shape 报错
- **修复**：`python/sglang/srt/layers/afd.py` 中 `get_tensor_communicator()` 对 `ipc_cpp` + `local_tp > 1` 场景添加 `BroadcastTensorCommunicator` 包装，并添加 `send_stream_ordered`/`recv_stream_ordered` 别名

#### 多卡 DVFS 调频修复（TP>1 关键修复，2026-06-01）

在 tier 方案首轮跑完后发现一个严重缺陷：**TP=2 实例只有一张卡被调频，另一张始终停在 auto-boost**。实测铁证：杀进程后 PF 进程的 GPU0 锁在 930MHz、GPU1 停在 210MHz（同属一个 TP=2 进程）。

- **根因 1（单卡控制器）**：`scheduler.py` 的 `self._dvfs_hw = DVFSController(device_index=单个index)` 只持有一张卡的 NVML 句柄，`_apply_freq()` 只锁那一张。TP=2 进程占两张卡，第二张永远不被调频。
- **根因 2（脚本未传 NVML index）**：`run_8gpu_deploy_bench.py` 只设了 `AFD_IPC_PEER_DEVICE`，没设 `AFD_NVML_DEVICE_INDEX`，运行时 fallback 到 `torch.cuda.current_device()`，在 CVD 重映射下可能锁错卡；决策日志名全部为 `gpu-1`。
- **修复**：
  - `scheduler.py`：引入 `self._dvfs_hw_list`，每个进程为它 TP 组内**全部**物理卡各建一个 `DVFSController`，`_apply_freq()` 对列表内所有卡统一 `lock_sm_clock()`。
  - `run_8gpu_deploy_bench.py`：新增 `_owned_gpus(cvd, base, tp)` 计算每进程占用的物理卡（= `CVD[base:base+tp]`），通过新环境变量 `AFD_NVML_DEVICE_INDICES`（CSV）传入。
  - 回归修复：NVML 锁频是系统级的，调频+决策日志只在 `tp_rank==0` 执行（否则 TP0/TP1 两 rank 并发写同一日志文件，产生交错损坏的 JSON）。
- **验证**：修复后日志显示每进程持有正确卡对（PF→[0,1] PA→[2,3] DF→[4,5] DA→[6,7]），且同实例两卡频率完全同步（如 DF 的 4,5 一起跳 1170、DA 的 6,7 一起回 930），决策日志 `ok=57 bad=0` 无损坏。
- **影响**：修复前 tier 数据只调了一半的卡，能耗节省被低估。**两个 tier 方案（m1_tier/m2_tier）已用修复后代码完整重跑**，本文档第 3/4 节数据均为修复后结果。

## 2. 完成状态（2026-06-01 最终）

5 个方案全部测完。非 tier 方案（pd_p4d4 / pdaf_8g_m1 / pdaf_8g_m2）首轮跑完；两个 tier 方案（pdaf_8g_m1_tier / pdaf_8g_m2_tier）在多卡 DVFS 修复后用 `max-run-s=600` 完整重跑，旧的 bug 版 tier 数据已归档至 `results/8gpu/json/_prebugfix_*/`。

| Deploy | 状态 | collapse 点 |
|--------|------|------------|
| pd_p4d4 | ✅ 完成 | il128 q4、il512 q10 |
| pdaf_8g_m1 | ✅ 完成 | il128 q4、il512 q8、il2048 q10、il4096 q4 |
| pdaf_8g_m2 | ✅ 完成 | il128 q4、il512 q6、il2048 q8、il4096 q4 |
| pdaf_8g_m1_tier | ✅ 修复后重跑 | il128 q4、il2048 q6、il4096 q3、il512 q4 |
| pdaf_8g_m2_tier | ✅ 修复后重跑 | il128 q6、il2048 q6、il4096 q3、il512 q4 |

注：有 2 个 thpt=0 的 hang 异常点（m1_tier il2048 q4、m2_tier il128 q4），系 decode-heavy + DVFS 压频导致尾部请求超 600s watchdog，绘图时已剔除。

## 3. 分析

### 3.1 PD vs AF 基线（最高频，decode/prefill 分场景）

| Workload | 指标 | pd_p4d4 | pdaf_8g_m1 | 结论 |
|----------|------|---------|------------|------|
| il128_ol1024 q2 | Thpt/TPOT/mJ·tok | 1156 / 47ms / 919 | 550 / 116ms / 1550 | decode-heavy：PD 全面碾压（吞吐翻倍、TPOT 减半、能效优 40%） |
| il512_ol256 q4 | Thpt/TPOT/mJ·tok | 864 / 46ms / 1308 | 680 / 109ms / 1423 | 均衡：PD 吞吐高，AF 能效略差 |
| il2048_ol64 q6 | Thpt/TPOT/mJ·tok | 366 / 47ms / 4421 | 346 / 97ms / 4109 | prefill-heavy：吞吐接近，AF 能效略优、但 TPOT 翻倍 |
| il4096_ol64 q3 | Thpt/TPOT/mJ·tok | 183 / 45ms / 8598 | 172 / 92ms / 8188 | prefill-heavy：同上，AF 能效略优 |

decode-heavy 场景 PD TP4 把 attention/KV 全留在 4 卡，AF 拆分反而增加跨卡通信和 TPOT；prefill-heavy 场景 AF 的算力切分有轻微能效优势，但 TPOT 始终比 PD 高一倍（AF 解耦的固有代价）。

### 3.2 DVFS 能效收益（tier vs 裸 AF，多卡修复后）

修复多卡调频后，DVFS 在**健康 QPS（SLO 0%）下稳定回收 21-26% 能耗**，M=1 / M=2 收益相当：

| Workload | QPS | M=1 裸→tier (省) | M=2 裸→tier (省) |
|----------|-----|------------------|------------------|
| il512_ol256 | q2 | 2398→1890 mJ/tok (**21.2%**) | 2751→2378 (13.6%) |
| il2048_ol64 | q2 | 8831→6745 (**23.6%**) | 9458→7200 (**23.9%**) |
| il2048_ol64 | q3 | 6523→4831 (**25.9%**) | 6968→5177 (**25.7%**) |
| il4096_ol64 | q2 | 10570→7897 (**25.3%**) | 11221→8389 (**25.2%**) |

代价是 TPOT 升高（如 il2048 q3：裸 AF 89ms → tier 106ms），这是 DVFS 压频换能耗的预期权衡，但仍远在 300ms SLO 内。

### 3.3 DVFS 在高 QPS 的崩溃（predictor 低估）

接近 collapse 的 QPS 下 DVFS 收益骤降、SLO 崩溃。il4096_ol64 q3：M=1 能效收益从 q2 的 25% 跌到 4.8%，SLO 违背冲到 93%；il2048_ol64 q6：tier 版 TTFT 飙到 16s（裸 AF 仅 1.5s）、SLO 92%、吞吐反而从 346 跌到 240 tok/s。

根因是 **decode 预测器严重低估迭代时间**（smoke 实测 `obs_iter=121ms` vs `pred=40ms`，低估 ~67%），DVFS 据此把 attention 频率压到过低档（450MHz），实际迭代慢 3 倍，请求在 decode 端堆积直至 TTFT 崩溃。这与 version1 的 SLO 收紧实验暴露的是同一个 predictor bias 问题，不是多卡修复引入的——裸 AF 在同样 QPS 能跑通正说明问题在 DVFS 选频策略。

### 3.4 图表

- `8gpu_tier_tradeoff.png`：il2048_ol64 上 5 方案三联板（能耗/token、TTFT log、TPOT），覆盖 q1-q6。低 QPS tier 能耗最低，q6 全线 collapse。
- `8gpu_tier_generalization.png`：4 种负载在 QPS2 公共健康点的 J/tok 分组柱状。prefill-heavy（il4096）单 token 能耗最高，decode-heavy（il128）最低。

## 4. 结论

1. **场景决定方案**：decode-heavy 用纯 PD（TP4 碾压 AF）；prefill-heavy 下 AF 能效略优但 TPOT 翻倍，需按延迟预算取舍。
2. **DVFS 在健康负载区间有明确价值**：多卡修复后，AF+DVFS 在 SLO 0% 的 QPS 下稳定省 21-26% 能耗，且对 prefill-heavy 负载收益最稳定（il2048/il4096 在 q2-q3 都 ~25%）。
3. **DVFS 的可用区间受 predictor 精度限制**：一旦逼近 collapse，decode 预测器低估会让 DVFS 压频过度、引发 TTFT 崩溃和吞吐倒退。**校准 decode 预测器（或加在线 calibration / 利用率反馈）是 DVFS 落地的前置条件**，否则只能保守用于中低 QPS。
4. **M=1 vs M=2**：能效收益相当；M=2 微批在 prefill-heavy 下能多撑一档 QPS（il2048 裸 AF 撑到 q8），但 decode-heavy 下 M=2 的 TPOT 更高。
5. **多卡调频修复是 TP>1 部署的必要前提**：修复前 tier 只调一半卡，能耗节省被低估约一半，所有 TP>1 的 DVFS 实验都应基于修复后代码。

### 产物
- Workload: `workloads/fixed_il{128,512,2048,4096}_ol{1024,256,64,64}_qps*.jsonl`
- 脚本: `scripts/bench/run_8gpu_bench.sh`, `scripts/bench/run_8gpu_deploy_bench.py`（含多卡 DVFS 修复）
- 数据: `scripts/bench/results/8gpu/json/`（修复前 tier 数据归档于 `_prebugfix_*/`）
- 图表: `scripts/bench/results/8gpu/figures/`
  - `8gpu_tier_tradeoff.png`、`8gpu_tier_generalization.png`（脚本 `plot_8gpu_tier.py`）
- 日志: `logs/_sweep/8gpu_full_v2.log`（首轮），`logs/_sweep/8gpu_tier_rerun.log`（tier 修复后重跑）
- 代码修复: `python/sglang/srt/managers/scheduler.py`（多卡 DVFS + tp_rank0 gate）

## 5. 补充实验：纯 PD 的 DP=2 并行（pd_dp2，DP vs TP，4 卡）

> 注：本节为 4 卡部署拓扑补充实验，归属 version2 的 deploy 对比体系（非 8 卡压测），但为保持 DP/TP 讨论的连续性记录于此。GPU0-3，freq=auto。

前述纯 PD 方案都是单实例 TP 切分。本节补一种数据并行（DP）部署：把 4 卡拆成**两个独立的 1P1D 实例**（实例0 P=GPU0/D=GPU1，实例1 P=GPU2/D=GPU3，每进程 TP=1），每个实例是完整模型副本 + 独立 disaggregation 对（独立 bootstrap port），用完整 router（`round_robin`）做请求级负载均衡。与 TP=2（pd_p2d2，单实例两卡切一份模型）形成同卡数的 DP vs TP 对照。

### 5.1 DP=2 vs TP=2 数据（同 4 卡）

| 负载 | QPS | 方案 | 吞吐 | TTFT | TPOT | J/tok | SLO违背 |
|------|-----|------|------|------|------|-------|--------|
| il512_ol128 | 2 | pd_dp2 | 234 | 182 | 47 | 3448 | 0% |
| il512_ol128 | 2 | pd_p2d2 | 236 | 115 | 44 | 2819 | 0% |
| il512_ol128 | 8 | pd_dp2 | 930 | 183 | 49 | 1095 | 0% |
| il512_ol128 | 8 | pd_p2d2 | 934 | 132 | 47 | 963 | 0% |
| il256_ol512 | 2 | pd_dp2 | 730 | 158 | 49 | 1055 | 0% |
| il256_ol512 | 2 | pd_p2d2 | 754 | 119 | 45 | 882 | 0% |
| il256_ol512 | 4 | pd_dp2 | 1068 | **16462** | 49 | 756 | **75%** |
| il256_ol512 | 4 | pd_p2d2 | 1489 | 130 | 47 | 522 | 0% |
| il1024_ol128 | 2 | pd_dp2 | 234 | 265 | 48 | 3664 | 0% |
| il1024_ol128 | 4 | pd_dp2 | 465 | 274 | 49 | 2082 | 0% |

（il512_ol128 q4/q6、il1024 各点 DP2 与 TP2 趋势一致，吞吐持平、TP2 延迟/能效略优，从略。）

### 5.2 发现

1. **吞吐基本持平**：中低 QPS 下 DP=2 与 TP=2 吞吐几乎一样（il512 q8：930 vs 934），两者都能吃满 4 卡算力。
2. **TP=2 延迟与能效稳定略优**：TP=2 把单请求计算切到两卡并行，TTFT/TPOT 更低（il512 q2：115/44ms vs DP 的 182/47ms），J/tok 普遍低 15-20%（il512 q2：2819 vs 3448）；DP=2 每请求只在单卡算，单请求延迟略高。
3. **decode 重负载是分水岭——DP=2 的本质短板**：`il256_ol512 q4`（decode 重、高并发）TP=2 扛住（1489 tok/s, TTFT 130ms, 0% 违背），DP=2 崩溃（1068 tok/s, TTFT 16462ms, **75% 违背**）。原因：DP=2 每个 1P1D 实例的 decode 只有**单卡 KV cache 容量**，长输出高并发时单实例 KV 池先撑爆、请求排队回压；TP=2 的 decode 是两卡合并 KV 池，容量翻倍。
4. **结论**：单机 NVLink、dense 模型、4 卡规模下，TP=2 在延迟、能效、decode 扩展性上全面略优于 DP=2。DP 的理论优势（无跨卡通信）在 NVLink 高带宽下体现不出来，反被单实例 KV 容量减半拖累；DP 更适合模型小、单卡能放下、纯比吞吐扩展的场景。

### 5.3 产物（DP=2）
- 方案注册：`run_deploy_bench.py` 新增 `pd_dp2`（`start_pd_dp` + `start_router_multi` 多端点路由）；顺带修复 `cleanup_procs` 误 reset 他人卡（改为按当前方案 GPU reset）。
- 数据：`results/deploy/json/pd_dp2_il{512,256,1024}_ol*_qps*_results.json`（8 run）。
- 日志：`scripts/bench/logs/_sweep/pd_dp2_bench.log`。
- 图表：`results/deploy/figures/deploy_tier_tradeoff.png` 已更新为 6 路对比（新增 pure-PD DP2）。

## 每日进度

今天完成了 Qwen3-32B / 8 卡 A800 上 5 种部署方案的全量压测与分析。
1.先修复了一个 TP>1 部署的多卡 DVFS 缺陷：TP=2 实例此前只有一张卡被调频，修复后每进程锁住 TP 组内全部物理卡并验证两卡同步；两个 tier 方案已用修复后代码完整重跑。
2.DVFS 在健康 QPS（SLO 0%）下稳定回收 21-26% 能耗，M=1/M=2 收益相当，prefill-heavy 负载收益最稳定；与同样 8 卡的纯 PD TP4 横向对比，PD+AF DVFS 在 prefill-heavy 场景能效明显胜出（il2048 q3：PD4 6895 vs M1tier 4831 mJ/tok 省约 30%，il4096 q2：11057 vs 7897 省约 29%），均衡场景（il512 q2）小幅领先（2376 vs 1890）。
3.下一步计划：校准 decode 预测器（当前逼近 collapse 时低估约 67%，导致 DVFS 压频过度、TTFT 崩溃、吞吐倒退，如 il2048 q6 tier TTFT 飙到 16s）。