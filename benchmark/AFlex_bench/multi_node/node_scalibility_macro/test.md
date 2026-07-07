# AFlex conv 优化尝试记录

测试环境：node3 (`10.252.129.34`) + node4 (`10.252.129.33`)，Qwen3-32B，A800-SXM4-80GB (SM80 Ampere)。
数据集 `conv`，QPS `[8,12,16]`，SLO `TTFT=5000ms / TPOT=300ms`，PDAF `M=1`。
基准来自 `charts/conv_6scheme_slo5000_300_dashboard.png`：AFlex (`pdaf_tier`) 在吞吐/TPOT 上打不过 DistServe/BiScale。

目标：缩小 AFlex 与 PD 方案（DistServe/BiScale）的差距，优先 TPOT/吞吐，再看 energy/token。

---

## 背景结论：AFlex 为什么打不过 PD

三个结构性原因：

1. **强制 `M=1`**，AF operator disaggregation 的逐层 IPC/sync 无法被 pipeline overlap 掩盖。
2. **单 decode pipeline（DA/DF TP4）**，而 DistServe/BiScale 是 `2×TP4` decode worker，decode 并发度差一半。
3. `conv` 宽松 SLO + decode 访存受限，DVFS 能省的能耗空间本就很小。

---

## 尝试 1：脚本级参数 sweep（M2 / dynamicM / V1DVFS / 2Decode）

脚本：`run_conv_aflex_opt_sweep.py`，分析：`analyze_conv_aflex_opt_sweep.py`
结果：`results/conv_aflex_opt_sweep_20260704_204328.json`，图：`charts/conv_aflex_opt_sweep_dashboard.png`

| 变体 | 结论 |
|---|---|
| `M=2` | 明显退化：TPOT/吞吐都比 M=1 差（单 decode 上 M2 反而增加开销） |
| `dynamicM` | 只在高 QPS 勉强回到 M=1 水平，无决定性改善 |
| `V1 compositional DVFS` | **单 decode 下最佳**：qps8/12/16 的 TPOT 最低、能耗最低、SLO 低 |
| `2Decode` | 初次测试 TTFT 爆炸（见尝试 2） |

**落地**：把主流程 `run_conv_6scheme_slo5000_300.py` 的 AFlex tier 从 V2 coupled 切到 V1 compositional（`--afd-dvfs-decode-compositional` + `models_v1`），M=1 不变。

---

## 尝试 2：2Decode router bug 定位与修复

现象：2Decode（1 prefill TP4 + 2 decode TP2）初测 TTFT 5-7s、SLO violation 40%+。

排查过程（probe：`probe_2decode_router_node34.py`）：
- 先怀疑 router 未分流 → 改成"两个 PD 子 router + 顶层 round_robin"，仍超时。
- 直接分别打子 router：`d0` 2.8s 正常，`d1` 45s 超时 → 缩小到第二个 decode 实例。
- 读 DF1 日志定位根因：**D1 的 bootstrap port 配错**（`BS_PORT+1` 指向不存在的 prefill bootstrap 端点）。

**修复**（`run_conv_pdaf_2decode.py`）：`d1_cf = _afd_common(..., BS_PORT)`，两个 decode 连同一个 prefill bootstrap。

修复后 2Decode V1 完整结果（`results/conv_aflex_opt_sweep_2decode_fixed_20260704_222318.json`）：

| qps | thpt | TPOT | mJ/tok | SLO% |
|---|---|---|---|---|
| 8  | 379.0 | 89.9 | 3826 | 0.0 |
| 12 | 578.7 | 91.8 | 2592 | 0.0 |
| 16 | 596.1 | 96.8 | 2511 | 0.2 |

TTFT 恢复到 270-335ms，SLO 基本 0。2Decode 修复后 TPOT/吞吐明显接近 PD。

---

## 尝试 3：引擎层 AF 通信优化（fused / gpu_only）

### 3a. 2Decode 上的 fused / comm-stream A/B
脚本：`run_conv_2decode_afcomm_ab.py`，结果：`results/conv_2decode_afcomm_ab_20260704_235133.json`

| 模式 | qps16 TPOT | qps16 thpt |
|---|---|---|
| baseline | 96.6 | 596.4 |
| fused pipeline | 94.5 | 608.8 |
| fused + comm_stream | 96.2 | 590.8 |

结论：fused 只有 <2% 微弱正收益，comm_stream overlap 无效甚至略负。

### 3b. TP4 单 decode 上做 gpu_only（改 C++）
- 发现 TP4 下 decode rank0 被 `BroadcastTensorCommunicator` 包裹，缺 `send_tensor_gpu_only`/`get_fused_pipeline`，脚本级开关**静默回退**，根本没生效。
- 给 wrapper 补 gpu_only 委托 + stream-ordered 广播，并把优化 gate 到 decode 侧（prefill 保持默认，否则 health check 超时）。
- 撞到 C++ bug：`send_tensor()` 首调会重置 recv 元数据缓存，导致 FFN 侧 `recv→send→recv_gpu` 崩溃。
- **修复 C++**（`sgl-kernel/csrc/afd_ipc/afd_ipc_pybind.cpp`）：移除 `send_tensor` 里错误的 `meta_cached_recv_=false`（send/recv 缓存本独立）。重编 kernel 后默认路径验证正确（temp0 输出确定性一致）。
- 但 gpu_only（gpu_signal）**输出损坏且非确定性**：根因是 `signal_kernel` 与异步 D2D copy 无跨流完成排序（`__threadfence_system` 管不到独立 copy 引擎），是真实数据竞争。默认 IPC_EVENT 用 `cudaStreamWaitEvent` 有硬排序，gpu_signal 没有。

**决策**：放弃 gpu_signal 路径（要修得重实现 IPC_EVENT 级排序，且收益 <2%）。回退所有 gpu_only Python 改动，`afd.py` 恢复基线。**保留 C++ recv-meta 缓存 bug 修复**（独立的正确性改进）。

关键洞察：默认 `recv_tensor()` 已在 CPU busy-poll 时释放 GIL 且用 stream event 排序、不做 `cudaStreamSynchronize` —— **CPU busy-poll 不阻塞 GPU，AF 通信在 M=1/conv 下压根不是瓶颈**。

---

## 尝试 4：DVFS 方向

### 4a. headroom-aggressive DVFS
新增 server arg `--afd-dvfs-headroom-aggressive`（把代码里硬编码关闭的逻辑打开，SLO-safe，默认关闭）。
脚本：`run_conv_tp4_headroom_ab.py`，结果：`results/conv_tp4_headroom_ab_20260705_143526.json`

| mode | qps8/12/16 mJ/tok | SLO% |
|---|---|---|
| v1_base | 3697 / 2626 / 2513 | 0/0/0 |
| v1_hr06 | 3762 / 2637 / 2502 | 0/0.3/0.1 |

结论：**无效**。energy/token 基本不降（甚至略升），还引入 SLO violation。原因：AFlex 已开 `--afd-dvfs-idle-lock`，迭代间 idle 功耗已被捕获，headroom 想省的是同一块，重叠。

### 4b. decode kernel / batch 调度
- 硬件 A800 (SM80)，decode 默认用 flashinfer（Ampere 最快，fa3 是 Hopper 专属），无可换 backend。
- `num_continuous_decode_steps` 不作用于 AFD decode 事件循环。
- AFD decode loop (`event_loop_afd_disagg_decode`) 是全同步（无 CPU/GPU overlap），但实测 `AFD_FWD_OVERHEAD` 显示暴露的 CPU 开销 ~0.1ms/iter（**<0.2%**）。加 overlap 有跨进程死锁风险且收益 <0.2%，不值得。

### 4c. DVFS 调频粒度 + 预测器偏差（重要发现）
开 `AFD_DVFS_DECISION_LOG` 逐实例记录决策。
- **调频粒度确认独立**：DA0→GPU[0,2]、DA1→GPU[4,6]，各有独立 controller/日志/记录数，不是整体调频；只是两个 decode 负载相同所以选频一致。
- **发现预测器严重高估**：预测 decode 迭代 226ms vs 实测 83ms，`pred_err_pct ~175%`，导致频率卡在 1170 不降。
- 开 `--afd-dvfs-online-calibration`（结果：`results/conv_2decode_v1_calib_20260705_153338.json`）：校准生效，`pred_err_pct` 收敛到 ~0%，但 **energy/token 反而 +10~12%**：

| qps | mJ/tok (无校准→校准) |
|---|---|
| 8  | 3816 → 4285 (+12%) |
| 12 | 2567 → 2817 (+10%) |
| 16 | 2510 → 2762 (+10%) |

原因：修正预测后 controller 更"自信"地偶尔升频到 1410 压 TPOT，但 **decode 访存受限，升频不怎么加速却明显更耗电**。印证 conv/宽松 SLO/decode-bound 下 DVFS 无有效能耗杠杆。

**决策**：保留 `--afd-dvfs-online-calibration`（默认关闭，紧 SLO/计算受限场景可能有用），conv 下不开。

---

## 尝试 5：3-Decode 配比（用户 idea，有真实收益）

思路：P 从 8 卡砍到 4 卡（TP2），省出的 4 卡再做第 3 个 decode 实例。
脚本：`run_conv_pdaf_3decode.py`（拓扑 node1: P(TP2 GPU0-3)+D0(TP2 GPU4-7)；node2: D1(TP2 GPU0-3)+D2(TP2 GPU4-7)），probe：`probe_pdaf_3decode.py`，sweep：`run_conv_3decode_sweep.py`
结果：`results/conv_3decode_v1_20260705_163351.json`

三种 decode 配比对比（都 V1 compositional DVFS，M=1，conv）：

| qps | 指标 | 1Decode(TP4) | 2Decode(2×TP2) | **3Decode(3×TP2)** |
|---|---|---|---|---|
| 8  | thpt / TPOT / mJ/tok | 367.6 / 93.3 / 3697 | 379.0 / 89.9 / 3826 | 378.8 / 88.4 / 4154 |
| 12 | thpt / TPOT / mJ/tok | 534.0 / 105.1 / 2626 | 578.7 / 91.8 / 2592 | **592.4** / 87.9 / 2769 |
| 16 | thpt / TPOT / mJ/tok | 561.8 / 109.9 / 2513 | 596.1 / 96.8 / 2511 | **637.8** / 89.1 / 2620 |

结论：
- **吞吐**：3Decode qps16 达 637.8 tok/s（比 2Decode +13.5%，比 1Decode 更高），qps12 +11%。
- **TPOT**：3Decode 稳定在 ~88-89ms，几乎不随 QPS 上升（1Decode 93→110ms），decode 并发吃掉了排队。
- **TTFT**：升到 339-445ms（prefill TP2 变慢），但离 5000ms SLO 有 10 倍余量，安全。
- **能耗**：3Decode energy/token 比 2Decode 高约 4-9%（prefill TP2 效率下降 + 多一个实例固定功耗；decode 访存受限，多实例不等比省能）。

**权衡**：2Decode 是吞吐/能耗的平衡点；3Decode 是纯吞吐/TPOT 最优。这是本系列唯一真实、可解释的性能杠杆（对准 decode 并发的结构性瓶颈）。

---

## 总体结论

1. **engine 层微优化（AF 通信 / DVFS 选频 / decode kernel-scheduling）在 conv/宽松 SLO/decode-bound 下均无有效收益** —— 通信已被 IPC_EVENT 做对 overlap、能耗已被 idle-lock+V1 选频吃掉、decode CPU 开销 <0.2%、DVFS 因访存受限无杠杆。
2. **真正的杠杆是 decode 并发度（拓扑）**：修复 2Decode bug + 3Decode 配比直接打在结构性瓶颈上，吞吐/TPOT 明显改善。
3. 附带修复了一个真实的 C++ bug（recv-meta 缓存被 send 错误重置）。

## 代码改动状态

保留：
- `sgl-kernel/csrc/afd_ipc/afd_ipc_pybind.cpp`：recv-meta 缓存 bug 修复（正确性改进）。
- `server_args.py` + `scheduler.py`：新增 `--afd-dvfs-headroom-aggressive`（SLO-safe，默认关闭）。
- `run_conv_pdaf_2decode.py`：bootstrap port 修复 + 两级 round-robin router + env 透传。
- 新增脚本：`run_conv_pdaf_3decode.py`、`run_conv_3decode_sweep.py`、`probe_pdaf_3decode.py`、`run_conv_2decode_afcomm_ab.py`、`run_conv_tp4_headroom_ab.py`、`run_conv_aflex_opt_sweep.py`、`analyze_conv_aflex_opt_sweep.py`、`probe_2decode_router_node34.py`、`probe_pdaf_tp4_afcomm.py`。

已回退：
- `afd.py`：gpu_only Python 改动全部回退，恢复基线（md5 `73e32dbb`）。

## 后续可探索
- 把 1D/2D/3D 三种配比画成对比图入主 dashboard。
- 3Decode 换更省的 prefill 配置，补回能耗劣势。
- 动态配比：低 QPS 用 2Decode（省能耗），高 QPS 切 3Decode（提吞吐）。
