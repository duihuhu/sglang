# 节点可扩展性能耗测试 — 完整测试方案 (Node Scalability Benchmark Plan)

> 目标:在单节点 / 跨节点、不同 GPU 总数下,对比 **6 种方案** (3 架构 × 2 功耗模式) 的
> TTFT、TPOT、总能耗、energy-per-token、吞吐。模型固定 **Qwen3-32B (Dense)**。
>
> 本文档自包含,换设备后照此即可继续测试。所有路径、端口、亲和性、已知坑都写清楚了。

---

## ★ Native DVFS Bug Fix & 重跑计划 (2026-06-27)

### 修复内容:`notify_freq_override` — prefill/decode 频率状态同步

**问题**: 在 Native 连续批处理模式下,prefill 通过 `_apply_freq_single(f)` 改变了硬件 GPU 频率
(例如降到 690MHz),但 controller 内部的 `_decode_state.cur_f` 没有被更新(仍记着旧值如 930MHz)。
当 decode 阶段调用 `select_freq_decode` 选出最优频率 930 时,`_should_switch(930, 930)` 发现
`f_new == f_cur` → 返回 False → 不切换 → 硬件一直保持 prefill 设的频率(如 690),decode 无法
正确调整到最优频率。

**结果**: Native+Tier 之前仅能节约 ~2% 能耗(实际是 decode 被困在错误频率)。

**修复** (两个文件):
1. `python/sglang/srt/managers/scheduler.py` — prefill 切频后调用 `ctrl.notify_freq_override(decision.f)`:
   ```python
   if decision.switched:
       self._apply_freq_single(decision.f)
       ctrl.notify_freq_override(decision.f)  # NEW: sync decode state
   ```
2. `python/sglang/srt/energy/unified_dvfs_controller.py` — 新增方法:
   ```python
   def notify_freq_override(self, f: int):
       """Notify that hardware freq was changed externally (e.g. by prefill)."""
       self._decode_state.cur_f = f
       self._cur_f = f
   ```

**验证结果 (8卡, qa, QPS=1)**:
| 方案 | 修复前 (mJ/tok) | 修复后 (mJ/tok) | 节能 |
|------|----------------|----------------|------|
| Native DP8 baseline | 10211 | 10211 | — |
| Native DP8+Tier | 9988 (-2.2%) | **6497 (-36.4%)** | ✓ |
| Native TP2×DP4 baseline | 7592 | 7592 | — |
| Native TP2×DP4+Tier | — | **5389 (-29.0%)** | ✓ |

**影响范围**: 只影响 `DisaggregationMode.NULL`(Native 模式)。PD/PDAF 不受影响(prefill/decode 分
在不同实例,不存在跨阶段频率冲突),已有 PD/PDAF 数据无需重跑。

### 需要重跑的测试

所有之前跑过的 `native_dp_baseline` 和 `native_dp_tier` 数据需要 **用修复后的代码重跑**。
同时新增 **Native TP2/TP4/TP8** 的测试维度。

---

### 重跑测试矩阵

#### A. Micro-benchmark (4 数据集: chatbot/qa/rag/summary)

脚本: `benchmark/AFlex_bench/multi_node/node_scalibility/run_node_scalability.py`

| # | GPU总数 | TP | DP (每节点) | 方案名 | QPS | 状态 |
|---|---------|----|----|--------|-----|------|
| A1 | 8卡 | TP1 | DP4 | native_dp baseline | 1,2,3,4,5,6 | 待重跑 |
| A2 | 8卡 | TP1 | DP4 | native_dp tier | 1,2,3,4,5,6 | 待重跑 |
| A3 | 8卡 | TP2 | DP2 | native_tp2 baseline | 1,2,3,4,5,6 | 待跑 |
| A4 | 8卡 | TP2 | DP2 | native_tp2 tier | 1,2,3,4,5,6 | 待跑 |
| A5 | 8卡 | TP4 | DP1 | native_tp4 baseline | 1,2,3,4,5,6 | 待跑 |
| A6 | 8卡 | TP4 | DP1 | native_tp4 tier | 1,2,3,4,5,6 | 待跑 |
| A7 | 16卡 | TP1 | DP8 | native_dp baseline | 1,2,4,6,8 | 待重跑 |
| A8 | 16卡 | TP1 | DP8 | native_dp tier | 1,2,4,6,8 | 待重跑 |
| A9 | 16卡 | TP2 | DP4 | native_tp2 baseline | 1,2,4,6,8 | 待跑 |
| A10 | 16卡 | TP2 | DP4 | native_tp2 tier | 1,2,4,6,8 | 待跑 |
| A11 | 16卡 | TP4 | DP2 | native_tp4 baseline | 1,2,4,6,8 | 待跑 |
| A12 | 16卡 | TP4 | DP2 | native_tp4 tier | 1,2,4,6,8 | 待跑 |
| A13 | 16卡 | TP8 | DP1 | native_tp8 baseline | 1,2,4,6,8 | 待跑 |
| A14 | 16卡 | TP8 | DP1 | native_tp8 tier | 1,2,4,6,8 | 待跑 |

> 注: 8卡不测 TP8(无法分到 2 个节点,每节点只有 4 卡)。
> 16卡 TP8 = 每节点 1 个 TP8 实例(利用 NVLink 全互联),DP1×2节点。

数据集 × QPS 总计:
- 8卡: 6 方案 × 4 数据集 × 6 QPS = **144 个测试点**
- 16卡: 8 方案 × 4 数据集 × 5 QPS = **160 个测试点**
- 合计: **304 个测试点**

#### B. Macro-benchmark (2 数据集: conv/code, Azure traces)

脚本: `benchmark/AFlex_bench/multi_node/node_scalibility_macro/run_macro_benchmark.py`

| # | GPU总数 | TP | DP (每节点) | 方案名 | QPS | 状态 |
|---|---------|----|----|--------|-----|------|
| B1 | 8卡 | TP1 | DP4 | native_dp baseline | 1–8 | 待重跑 |
| B2 | 8卡 | TP1 | DP4 | native_dp tier | 1–8 | 待重跑 |
| B3 | 8卡 | TP2 | DP2 | native_tp2 baseline | 1–8 | 待跑 |
| B4 | 8卡 | TP2 | DP2 | native_tp2 tier | 1–8 | 待跑 |
| B5 | 8卡 | TP4 | DP1 | native_tp4 baseline | 1–8 | 待跑 |
| B6 | 8卡 | TP4 | DP1 | native_tp4 tier | 1–8 | 待跑 |
| B7 | 16卡 | TP1 | DP8 | native_dp baseline | 1–16 | 待重跑 |
| B8 | 16卡 | TP1 | DP8 | native_dp tier | 1–16 | 待重跑 |
| B9 | 16卡 | TP2 | DP4 | native_tp2 baseline | 1–16 | 待跑 |
| B10 | 16卡 | TP2 | DP4 | native_tp2 tier | 1–16 | 待跑 |
| B11 | 16卡 | TP4 | DP2 | native_tp4 baseline | 1–16 | 待跑 |
| B12 | 16卡 | TP4 | DP2 | native_tp4 tier | 1–16 | 待跑 |
| B13 | 16卡 | TP8 | DP1 | native_tp8 baseline | 1–16 | 待跑 |
| B14 | 16卡 | TP8 | DP1 | native_tp8 tier | 1–16 | 待跑 |

数据集 × QPS 总计:
- 8卡: 6 方案 × 2 数据集 × 8 QPS = **96 个测试点**
- 16卡: 8 方案 × 2 数据集 × 16 QPS = **256 个测试点**
- 合计: **352 个测试点**

### TP 部署详情

| GPU总数 | TP | 每节点实例数 | 每实例GPU | 部署说明 |
|---------|----|----|-----|------|
| 8卡 | TP1 | 4 | [4],[5],[6],[7] | 4 个 DP 实例/节点, router round_robin |
| 8卡 | TP2 | 2 | [4,5],[6,7] | 2 个 TP2 实例/节点, NVLink 内通信 |
| 8卡 | TP4 | 1 | [4,5,6,7] | 1 个 TP4 实例/节点, NVLink 内通信 |
| 16卡 | TP1 | 8 | [0]~[7] | 8 个 DP 实例/节点 |
| 16卡 | TP2 | 4 | [0,1],[2,3],[4,5],[6,7] | 4 个 TP2 实例/节点 |
| 16卡 | TP4 | 2 | [0,1,2,3],[4,5,6,7] | 2 个 TP4 实例/节点 |
| 16卡 | TP8 | 1 | [0,1,2,3,4,5,6,7] | 1 个 TP8 实例/节点 |

### 脚本改动 (执行前需完成)

1. **`run_node_scalability.py`**: 已支持 `--native-tp` 参数和 `native_tp` deploy 选项。
   需新增 TP4/TP8 的 GPU 分组逻辑(当前 `_launch_tp` 按 TP 度切分 `card_gpus`)。
2. **`run_macro_benchmark.py`**: 需复制 `native_tp` 部署逻辑(与 micro 脚本对齐),
   添加 `--native-tp` 参数和 `start_native_tp()` 函数。
3. 代码已修复(`scheduler.py` + `unified_dvfs_controller.py`),确保两节点容器已同步。

### 执行顺序建议

1. 先跑 **8卡 micro** (A1–A6),验证各 TP 下 tier 节能效果
2. 再跑 **8卡 macro** (B1–B6)
3. 然后 **16卡 micro** (A7–A14)
4. 最后 **16卡 macro** (B7–B14)

每组内按 baseline → tier 交替跑(脚本 `--mode all` 自动处理)。

---

## 0. 关键结论 / 注意事项 (先读这一段)

1. **6 种方案** = 3 架构 (`native_dp` / `pd_dp` / `pdaf`) × 2 功耗模式 (`baseline` / `tier`)。
2. **所有方案启动后先把 GPU 锁到 1410MHz** 作为统一基线;`baseline` 全程锁 1410,
   `tier` 在此基础上由 DVFS 往下调频(而不是从硬件默认的 1140MHz 应用时钟起调)。
3. **能耗模型分 V1/V2**:
   - `native_dp` / `pd_dp` → **V1 模型**(A/F 分层,`models_v1`),走 `UnifiedDVFSController`。
   - `pdaf` → **V2 模型**(耦合 decode 迭代,`models_v2`),走 AFD DVFS controller。
   - 用错模型会导致 DVFS 决策恒为 1410MHz(latency 预测 `ValueError` → fallback F_MAX)。
4. **GPU↔RDMA 网卡亲和性必须保持**(每节点拓扑相同):

   | GPU | RDMA NIC | 拓扑 |
   |-----|----------|------|
   | GPU0,1 | mlx5_0 (NIC0) | PXB |
   | GPU2,3 | mlx5_1 (NIC1) | PXB |
   | GPU4,5 | mlx5_4 (NIC2) | PXB |
   | GPU6,7 | mlx5_5 (NIC3) | PXB |
   | 全部 GPU 间 | NVLink | NV8 |

5. **DVFS 硬件控制依赖** `libdvfs_ctrl.so`(NVML `SetGpuLockedClocks`),两节点容器内都必须有,
   否则调频决策不会落到硬件。位置见 §5。
6. 容器内需要 `scikit-learn` + `lightgbm`(能耗模型加载),两节点都要装。
7. **已修复的关键 bug(勿回退)**:见 §6。

---

## 1. 环境与拓扑

| 项 | 值 |
|----|----|
| node1 (本地, prefill 侧) | `10.252.129.36` (`MN_NODE1_IP`) |
| node2 (远端, decode 侧) | `10.252.129.35` (`MN_NODE2_IP`) |
| 容器名 | `operator_test` (`MN_CONTAINER`) |
| 模型 | `/models/Qwen3-32B/` |
| Python | `/usr/bin/python3` |
| 每节点 GPU | 8 卡 (GPU0–7),NVLink 全互联 (NV8) |
| 每节点 RDMA | mlx5_0 / mlx5_1 / mlx5_4 / mlx5_5 (各 PXB 亲和 2 张 GPU) |

> 主控脚本在 **node1 宿主机** 运行;它通过 `docker exec`(本地)和 `ssh + docker exec`(node2)
> 拉起两节点容器内的 sglang 进程。node2 免密 SSH 必须可用。

### 卡数 → 每节点 GPU 选取(保持亲和)

| GPU 总数 | 每节点用卡 | 使用网卡 | 说明 |
|----------|-----------|----------|------|
| 4 卡 (2节点) | `[6,7]` | mlx5_5 | 每节点 2 卡 |
| 8 卡 (2节点) | `[4,5,6,7]` | mlx5_4 + mlx5_5 | 每节点 4 卡 |
| **16 卡 (2节点)** | `[0,1,2,3,4,5,6,7]` | mlx5_0/1/4/5 (4 张全用) | 每节点 8 卡,全 GPU(**新增**) |
| 8 卡 (单节点) | 单机 `[0..7]` | — | 参考数据,见 retesting/micro_benchmark |
| 4 卡 (单节点) | 单机 `[0..3]` | — | 参考数据 |

---

## 2. 六种方案的部署方法

主脚本:`run_node_scalability.py`(跨节点),单节点参考:`retesting/scripts/run_micro_bench.py`。

### 2.1 `native_dp` — 纯数据并行
- N 个 `TP=1` 实例,均分到两节点(node1 一半 GPU + node2 一半 GPU)。
- 前面挂 `sglang_router`,`--policy round_robin`。
- baseline: 全程锁 1410;tier: 每实例带 V1 DVFS flags。

### 2.2 `pd_dp` — Prefill/Decode 分离 (均衡分布)
- **关键设计**:不是「一台机全 prefill、另一台全 decode」,而是 **N/2 个自包含 PD 对**,
  每对的 prefill+decode 在**同一节点**(节点内 KV 传输),两两分布到两节点。
  这样两节点都同时承担 prefill 和 decode,避免 decode-heavy 负载时一台机空转。
- GPU 配对取亲和卡:`(prefill_gpu, decode_gpu)` 用同一 NIC 域。
  - 8 卡:4 个 PD 对,每节点 2 个 → node1: PD0(P=4,D=5),PD1(P=6,D=7);node2 同。
  - 4 卡:2 个 PD 对,每节点 1 个 → P=6,D=7。
- KV 传输后端:`mooncake`,`--disaggregation-ib-device` 用配对 GPU 的亲和 NIC。
- router:`--pd-disaggregation`,列出所有 prefill / decode worker。

### 2.3 `pdaf` — Attention/FFN 算子级分离 (AF Disagg)
- PA+PF 在 **node1**(prefill 侧),DA+DF 在 **node2**(decode 侧)。
- **交错亲和布局**(关键):attn 放偶数序 GPU、ffn 放奇数序 GPU,每个 attn rank 独占一张 NIC。
  - 4 卡:`tp=1`,PA=GPU6 (mlx5_5),PF=GPU7,`step=1`。
  - 8 卡:`tp=2`,PA=GPU4,6 (mlx5_4 + mlx5_5,**两张 NIC**),PF=GPU5,7,`step=2`。
- 通信后端:`--afd-comm-backend ipc_cpp`(节点内 IPC),KV 走 mooncake + IB JSON 映射。
- micro-batch:`--afd-micro-batch 2 --afd-dynamic-micro-batch`(M 动态,峰值 2)。
- IB 设备用 JSON 文件 `/tmp/ib_scal_map.json`(GPU→NIC 映射),通过 `docker cp` 下发到容器。
- 关键环境变量(`_afd_env`):`AFD_NVML_DEVICE_INDICES`、`AFD_IPC_PEER_OFFSET`(attn=+1/ffn=−1)、
  `SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0`(防 decode 侧内存泄漏误报导致 watchdog timeout)。
- router:`--pd-disaggregation --mini-lb`,prefill→PA,decode→DA。

---

## 3. 运行命令

### 3.1 跨节点(在 node1 宿主机运行)
```bash
cd /mnt/workspace/lt/sglang/benchmark/AFlex_bench/multi_node/node_scalibility

# 全 6 方案 / 全 4 数据集 / 指定 QPS
python3 run_node_scalability.py --ngpu 8 --deploy all --mode all --scenario all --qps 1,2,3,4,5,6

# 只跑某架构(如重测 pdaf)
python3 run_node_scalability.py --ngpu 8 --deploy pdaf --mode all --scenario qa --qps 1

# 只跑某模式
python3 run_node_scalability.py --ngpu 4 --deploy all --mode tier --qps 1,2,3
```
参数:`--ngpu {4,8,16*}` `--deploy {native_dp,pd_dp,pdaf,all}` `--mode {baseline,tier,all}`
`--scenario {chatbot,qa,rag,summary,all}` `--qps 逗号分隔` `--max-run-s 400`
(*16 卡需先按 §7 扩展脚本)

结果增量保存到 `results/scal_{ngpu}card_{timestamp}.json`(每个方案跑完即落盘,中途挂掉不丢)。

### 3.2 单节点参考(在对应节点容器内)
```bash
# 8 卡单机
cd /mnt/workspace/lt/sglang/benchmark/AFlex_bench/retesting
python3 scripts/run_micro_bench.py   # 见该脚本参数
```

---

## 4. 数据集 (README 271-286 对应的 4 个场景)

| 场景 | input_len | output_len | 特征 |
|------|-----------|------------|------|
| chatbot | 128 | 1024 | decode 重 |
| qa | 512 | 256 | 均衡 |
| rag | 2048 | 64 | prefill 重 |
| summary | 4096 | 64 | prefill 极重 |

workload 文件:`retesting/workloads/micro_{scenario}_qps{N}.jsonl`(已生成 qps1–9)。

**QPS 选取**:4 卡测 1,2,3;8 卡测 1–6;16 卡可测更高(见 §7)。
> 注意:`summary`(in=4096) 在高 QPS(≥7)易把系统压崩,出无效点,按需限制上限。

### SLO
- TTFT SLO = 5000 ms,TPOT SLO = 300 ms。

---

## 5. DVFS / 能耗模型 / 依赖

### 能耗模型目录
```
V1 (native_dp/pd_dp): .../AFlex_bench/03_sensitivity/slo_sweep/retrain/models_v1
V2 (pdaf):            .../AFlex_bench/03_sensitivity/slo_sweep/retrain/models_v2
```

### libdvfs_ctrl.so(硬件调频依赖)
- 源码/Makefile:`benchmark/test_motivation/hucc/dvfs/`
- 编译:`cd benchmark/test_motivation/hucc/dvfs && make`
- **两节点容器内都要有**,否则 tier 调频决策不会落到硬件(频率不变)。
- 校验:tier 模式下 `nvidia-smi --query-gpu=clocks.sm` 应出现 < 1410 的频率(如 930/1170)。

### Python 依赖(两节点容器)
```bash
pip install scikit-learn lightgbm
```

---

## 6. 已修复的关键 Bug(勿回退)

1. **`_launch` 环境变量丢失**(影响 pdaf tier 节能):
   `setsid prlimit` 必须插在 `export ...;` **之后**、`python -m` **之前**;
   若放最前面会让 prlimit 去 exec `export`,导致 `AFD_NVML_DEVICE_INDICES` 等丢失,
   DVFS 绑错 GPU(绑到 GPU0 而非活跃卡),tier 几乎不省电。
2. **IB device 校验**:`server_args.py` 的 `_validate_ib_devices` 已放开,允许 JSON 字符串/文件路径。
3. **JSON 下发**:用 `docker cp`(本地)/ `scp`+`docker cp`(远端)传 IB 映射文件,勿用命令行内联(会 JSON 解析失败)。
4. **decode 内存泄漏误报**:AFD 服务加 `SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0`,否则 watchdog timeout。
5. **端口冲突 / 卡死**:每次部署前后都 `cleanup_all`;若手动跑过验证测试,务必清干净
   (`pkill -9 -f sglang.launch_server` 两节点 + 检查端口 42010/42011/42020/42021/42000)。
   DA bind 42020 失败会让主脚本在健康检查里死等到超时。
6. **node2 磁盘**:containerd 数据已迁到 `/mnt/data/containerd`(原 `/var/lib/containerd` 撑爆根分区)。

---

## 7. 【新增】16 卡方案(2 节点全 GPU)设计

> 每节点用满 8 卡(GPU0–7),两节点共 16 卡,**4 张 RDMA 网卡全用上**。

### 7.1 各架构布局
- **native_dp**:16 个 `TP=1` 实例(每节点 8 个),round_robin。
- **pd_dp**:8 个 PD 对,每节点 4 个,节点内配对取亲和卡:
  `(P=0,D=1)|(P=2,D=3)|(P=4,D=5)|(P=6,D=7)`,NIC 分别 mlx5_0/1/4/5。
- **pdaf**:`tp=4`,交错亲和布局:
  - PA = GPU `0,2,4,6`(mlx5_0/1/4/5,**4 张 NIC 各一**),PF = GPU `1,3,5,7`,`step=2`。
  - node1 跑 PA+PF,node2 跑 DA+DF,IB JSON 映射 8 卡全量。

### 7.2 需要的脚本改动(`run_node_scalability.py`)
1. `argparse` 的 `--ngpu` choices 加 `16`。
2. `card_gpus()` 增加分支:
   ```python
   def card_gpus(ngpu):
       if ngpu == 4:
           return [6, 7]
       if ngpu == 8:
           return [4, 5, 6, 7]
       return [0, 1, 2, 3, 4, 5, 6, 7]   # 16 卡:每节点 8 卡
   ```
3. `GPU_NIC` 补全 8 卡映射:
   ```python
   GPU_NIC = {0:"mlx5_0",1:"mlx5_0",2:"mlx5_1",3:"mlx5_1",
              4:"mlx5_4",5:"mlx5_4",6:"mlx5_5",7:"mlx5_5"}
   ```
4. `start_pdaf()` 增加 16 卡分支:`tp=4, step=2, attn_gpus=[0,2,4,6], ffn_gpus=[1,3,5,7]`,
   `attn_base=0, ffn_base=1`,`ib_map = {str(g):GPU_NIC[g] for g in gpus}`。
5. `start_pd_dp()` / `start_native_dp()` 已按 `card_gpus` 通用化,16 卡自动展开(确认端口不冲突)。
6. QPS:16 卡建议测 `1,2,3,4,6,8,10`(吞吐更高,压力点上移)。

### 7.3 16 卡运行
```bash
python3 run_node_scalability.py --ngpu 16 --deploy all --mode all --scenario all --qps 1,2,4,6,8,10
```

---

## 8. 完整测试矩阵(要做的所有测试)

| 部署 | GPU 总数 | 方案 | 数据集 | QPS | 状态 |
|------|---------|------|--------|-----|------|
| 单节点 | 4 卡 | 6 方案 | 4 数据集 | 1,2,3 | 参考数据已有(`retesting/micro_benchmark/4gpu`),缺的补 |
| 单节点 | 8 卡 | 6 方案 | 4 数据集 | 1–6 | 参考数据部分有,**rag/summary 待补** |
| 跨节点 | 4 卡 | 6 方案 | 4 数据集 | 1,2,3 | **待跑** |
| 跨节点 | 8 卡 | 6 方案 | 4 数据集 | 1–6 | 进行中(qa/qps1 已验证 6 方案,其余待跑) |
| 跨节点 | **16 卡** | 6 方案 | 4 数据集 | 1,2,4,6,8,10 | **待跑**(先按 §7 改脚本) |

对比指标:**TTFT、TPOT、总能耗(J)、energy-per-token(mJ/tok)、吞吐(tok/s)**。

最终交付:
- 6 方案 × {单节点,2节点} × {4,8,16 卡} × 4 数据集 × 多 QPS 的结果表。
- 跨架构 / 单机 vs 跨机 对比图(`analyze_scalability.py`、`plot_6scheme_comparison.py`)。

---

## 9. 分析与绘图

```bash
cd .../node_scalibility
# 单条 6 方案对比图(指定结果 json)
python3 plot_6scheme_comparison.py results/scal_8card_merged_qa_qps1.json
# 单机 vs 跨机全量对比(自动扫 results/ + retesting/micro_benchmark/)
python3 analyze_scalability.py
```
- 图输出:`charts/`
- 表/分析:`scalability_comparison.md`
- 多次 run 的同方案结果可手动 merge 同一 json 后再绘(参考已生成的 `scal_8card_merged_qa_qps1.json`)。

---

## 10. 当前进度快照(换设备时参考)

**已完成:**
- (DONE) 跨节点脚本 `run_node_scalability.py` 6 方案部署逻辑(含亲和、PD 均衡、PDAF 交错)。
- (DONE) DVFS V1/V2 模型目录修正 + `libdvfs_ctrl.so` 编译 + sklearn/lightgbm 安装(两节点)。
- (DONE) `_launch` 环境变量 bug 修复 → pdaf tier 调频生效(8卡 qa/qps1:节能约 22%,吞吐持平)。
- (DONE) node2 containerd 迁移到 `/mnt/data`,根分区恢复。
- (DONE) 8 卡 qa/qps1 全 6 方案验证 + `charts/6scheme_8card_qa_qps1.png`。

**待办:**
- (TODO) 跨节点 8 卡:补齐 chatbot/rag/summary × QPS 1–6 全量。
- (TODO) 跨节点 4 卡:6 方案 × 4 数据集 × QPS 1–3。
- (TODO) 单机 8 卡:补 rag/summary 缺失数据。
- (TODO) **16 卡:按 §7 改脚本后全量测试**。
- (TODO) `analyze_scalability.py` 出最终单机 vs 跨机对比表 + 图。

---

## 11. 2026-06-28 16卡 Micro-Benchmark 重跑进展

### 背景
修复了 Native DVFS 频率同步 bug (`notify_freq_override`)后，需要重跑所有方案。
新增 5 种数据集组合: `qa_lpld`(128,64), `chatbot_lphd`(128,1024), `balanced_mpmd`(512,256), `rag_hpld`(4096,64), `summary_hphd`(4096,1024)。
QPS: 2,4,6,8,12,16。Native 只测 TP2。

### 方案命名映射
| 脚本内部名 | 论文名 |
|---|---|
| native_tp2_baseline | SGLang |
| native_tp2_tier | DynamoLLM |
| pd_dp_baseline | DistServe |
| pd_dp_tier | BiScale |
| pdaf_baseline | MegaScale |
| pdaf_tier | AFlex |

### 测试结果状态

**第一轮 (16:14~19:48, log: /tmp/micro_16card_v2.log)**
| Scheme | 状态 | Scenarios |
|---|---|---|
| native_tp2_baseline (SGLang) | DONE | 30/30 (summary_hphd_qps12/16 FAIL) |
| native_tp2_tier (DynamoLLM) | DONE | 30/30 (chatbot_lphd_qps4 FAIL) |
| pd_dp_baseline (DistServe) | DEPLOY_FAILED | 端口冲突(容器被paused残留进程) |
| pd_dp_tier (BiScale) | DONE | 30/30 (summary_hphd_qps12/16 FAIL) |
| pdaf_baseline (MegaScale) | DEPLOY_FAILED | 残留进程导致 |
| pdaf_tier (AFlex) | DONE | 30/30 |

结果文件: `results/scal_16card_20260628_194842.json`

**第二轮补跑 (19:57~进行中, log: /tmp/micro_16card_patch.log)**
只跑: `pd_dp_baseline` + `pdaf_baseline` (baseline mode)
| Scheme | 状态 | 进展 |
|---|---|---|
| pd_dp_baseline (DistServe) | DONE | 30/30 完成 (19:57~20:40) |
| pdaf_baseline (MegaScale) | IN_PROGRESS | qa+chatbot完成, balanced进行中 (~21:15) |

预计 pdaf_baseline 约 22:10 全部跑完。

### 代码改动
1. `run_node_scalability.py`: 增加 `_check_ports()` 和 `_force_kill_sglang()` 函数,
   `cleanup_all()` 增强为清理后自动检测端口占用并 force kill 残留进程。
   `run_deploy()` 部署前增加端口预检查+重试。
2. 根因: node1 容器 `operator_test` 被 paused 导致之前的 sglang 进程残留,
   占用端口 53100-53161,使新的 PD/PDAF 部署无法绑定端口。

### 跑完后续步骤
1. 合并第一轮和第二轮结果到统一 JSON
2. 检查 FAIL 的 scenarios (summary_hphd 高 QPS OOM) 是否需要重跑
3. 生成新图表 (per-dataset 4子图格式 + 能耗对比标注)
4. 如有需要,补跑 8 卡 Native TP2/TP4 + Macro benchmark
