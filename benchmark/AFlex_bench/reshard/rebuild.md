# Tier 1 实现审查报告

基于文档 `benchmark/test_motivation/hucc/paper/system_detail_v1.md` 做对照审查。

**总体评价：85% 完成度** — 核心算法和架构全部打通，主要差距在"优雅过渡"和"物理拓扑感知"。

---

## 涉及文件

| 文件 | 行数 | 角色 |
|------|------|------|
| `python/sglang/srt/energy/tier1_solver.py` | 567 | ILP 求解器 |
| `python/sglang/srt/energy/workload_monitor.py` | 269 | 重规划触发器 |
| `python/sglang/srt/energy/workload_collector.py` | 545 | 运行时指标采集 |
| `python/sglang/srt/energy/profile_table.py` | 277 | Profile 数据 + 预测查询 |
| `python/sglang/srt/energy/af_profile_predictor.py` | — | ML 预测器 |
| `python/sglang/srt/energy/af_dvfs_controller.py` | 668 | Tier2 DVFS 控制器 |
| `python/sglang/srt/energy/af_launcher.py` | 558 | 批量启动器（含预飞求解） |
| `python/sglang/srt/energy/reload_orchestrator.py` | 302 | 全量 Reload 编排器 |
| `python/sglang/srt/energy/reload_signal.py` | 73 | 信号协议 |
| `python/sglang/srt/managers/scheduler.py` | — | 调度器集成（~200 行 tier1 相关） |

---

## 一、已完成模块

### 1.1 ILP 求解器（`tier1_solver.py`）

**目标函数（文档 1.1 节）** ✅
- `minimize Σ k_c × [E_A + E_F + E_bubble(M)]`
- `E_bubble = P_idle × |t_A - t_F| × (M-1)/M`, `P_IDLE_W = 80.0`, `M=2` 默认

**约束（文档 1.2 节）** ✅ 全部 13 条约束已实现

| 约束 | 实现位置 | 说明 |
|------|----------|------|
| (1) GPU 总量 | `_search_kp_kd()` L480-482 | `k_P×(tp_PA+tp_PF) + k_D×(tp_DA+tp_DF) ≤ G` |
| (2-4) 变量域 | `_VALID_TP`, `_VALID_FREQS`, `range(1, max_k+1)` | tp∈{1,2,4,8}, freq∈{210,450,690,930,1170,1410}, k∈Z⁺ |
| (5) Prefill SLO | `_enumerate_pairs()` L354 | `t_A + t_F + t_comm ≤ TTFT_SLO/L`, M=1 保守上界 |
| (6) Decode SLO | 同上 | `t_A + t_F + t_comm ≤ TPOT_SLO/L` |
| (7) Prefill 吞吐 | `_select_pair()` L522 | `k_P × Thpt_pair ≥ (1+α) × λ`, α=0.15 |
| (8) Decode 吞吐 | 同上 | `k_D × Thpt_pair ≥ (1+α) × N_active` |
| (9) A/F 平衡 | `_enumerate_pairs()` L358 | `|t_A - t_F| ≤ 0.8 × max(t_A, t_F)`，降低搜索空间~60% |
| (10-13) 显存 | `_check_attn_memory/ffn_memory` | 4 池分别约束：PA/PF/DA/DF |

**搜索剪枝（文档 1.3 节）** ✅
1. 资源约束 → 排除超限组合
2. 显存约束 → 排除 OOM 组合
3. SLO 约束 → 排除低频不可行组合
4. 平衡约束 → 排除极端失衡组合
5. Pareto 过滤 → 保留非支配前沿

### 1.2 4 路监控触发器（`workload_monitor.py`）

**4 个指标（文档 1.4 节）** ✅

| 指标 | 阈值 | 实现 |
|------|------|------|
| SLO 违反率 | > 1% | `_SLO_VIOLATION_THRESHOLD = 0.01` |
| A/F 利用率差 | > 0.2 | `_AF_UTIL_IMBALANCE_THRESHOLD = 0.2` |
| P/D 利用率差 | > 0.2 | `_PD_UTIL_IMBALANCE_THRESHOLD = 0.2` |
| KL 散度 | > 2.0 | `_KL_DIVERGENCE_THRESHOLD = 2.0` |

**滞回机制** ✅：连续 2 个窗口超阈值才触发（`_CONSECUTIVE_WINDOWS = 2`）

**Tier 2 节能抑制** ✅：`is_tier2_energy_saving` 标志抑制 A/F 不平衡误报（`workload_monitor.py` L187-192）

### 1.3 运行时指标采集（`workload_collector.py`）

**（文档 1.4 节配套）** ✅

- 逐请求 TTFT/TPOT 追踪（`record_batch_end()`）
- NVML 后台线程 GPU 利用率轮询（`_start_nvml_polling()`, 1Hz）
- 跨进程 Decode 耗时采集（DA→PA 通过共享 stats 文件）
- IL 分桶 + P99 延迟计算
- MonitoringWindow 构建

### 1.4 调度器集成（`scheduler.py`）

**（文档 3.1 节时序全景）** ✅

- `_tier1_monitor_check()` 主循环每 ~30s 一次（L1680）
- `_tier1_record_batch_start/end()` 全流程埋点（L1618, L1636）
- `_replan_tier1()` 重新求解 + 切换配置（L1750）
- `_apply_tier1_transition()` 增量/全量切换双路径（L1811）

### 1.5 启动器 + 预飞求解（`af_launcher.py`）

**（文档 0 节 — 路径 A）** ✅

- `--start-with-workload` 运行预飞 ILP 求解
- 序列化 `Tier1Solution` 为 JSON，通过 `--tier1-initial-solution` 传给 PA
- 自动读取 ILP 输出的 tp/freq 分配 GPU，预锁频率
- 两种启动模式：预求解（延迟创建求解器）vs 进程内求解

### 1.6 Reload 编排器（`reload_orchestrator.py`）

**（文档 1.5 节 — TP 变化过渡）** ✅ 基本功能

- 写信号 → 杀进程 → 释放端口 → 重置 GPU 时钟 → 重启 → 健康检查
- 信号协议文件 `tier1_reload_signal.json`（状态：`idle → reloading → ready/error`）

### 1.7 Tier 1 + Tier 2 协同（文档 3 节）

- `AFDVFSController.update_baseline()` 接收 Tier 1 新基线频率
- `is_energy_saving()` 抑制 Tier 1 误触发
- `--tier1-disable-reload` 限制 Tier 1 只做频率调整，不触发全量重启

---

## 二、已验证的端到端通路

```
af_launcher.py                    scheduler.py (PA)
     │                                  │
     │  --start-with-workload           │  --enable-tier1-pa
     │  ┌─────────────────┐             │  ┌───────────────────────┐
     │  │ Pre-flight ILP  │  solve()    │  │ WorkloadMonitor       │
     │  │ Tier1Solver     │─────────────│  │  - 4 metrics          │
     │  │ → Tier1Solution │  JSON       │  │  - 2-window hysteresis│
     │  └─────────────────┘             │  │  - should_replan()    │
     │          │                       │  └───────┬───────────────┘
     │          │ --tier1-initial-      │          │ True
     │          │   solution            │  ┌───────┴───────────────┐
     │          ▼                       │  │ _replan_tier1()       │
     │  ┌─────────────────┐             │  │  → lazy solver init  │
     │  │ Load Solution   │             │  │  → Tier1Solver.solve │
     │  │ Apply tp/freq   │             │  │  → new Tier1Solution │
     │  │ per module      │             │  └───────┬───────────────┘
     │  └─────────────────┘             │          │
     │                                 ┌──────────┴──────────┐
     │                                 │ _apply_transition()  │
     │                                 │  ├─ freq-only: NVML │
     │                                 │  └─ full: spawn     │
     │                                 │    reload_orchestrator
     │                                 └─────────────────────┘
```

---

## 三、与文档的偏差

### ⚠️ 轻微偏差（功能正确性不受影响）

**1. E_bubble 空闲功率固定 80W**

- 文档设计：`P_idle(f_wait)`，取决于完成更快那方的频率
- 当前实现：`_P_IDLE_W = 80.0` 固定值
- 影响：E_bubble 是目标函数的次要项（通常 < 5%），固定值对最优解方向影响有限。后续可通过 idle power vs freq 表替换常量。

**2. t_comm 静态化**

- 文档设计：`t_comm(r) = latency_overhead + tensor_size(r) / bandwidth`，按请求桶动态计算
- 当前实现：decode 固定 40us，prefill 回退 283us，支持文件查找特定 (bs, seq_len)
- 影响：对大 batch 长序列通信开销估计可能失真，但 ILP 的 M=1 保守公式提供了足够 slack 吸收误差。

**3. SLO 约束单一代表值**

- 文档设计：多负载分桶，每个桶有各自的 P90 值，产生多条约束
- 当前实现：`WorkloadProfile` 单组 il_rep/bs_avg，每条 phase 一条 SLO 约束
- 影响：对负载混合场景的建模精度有限，但 Tier 2 的运行时 DVFS 会实时补偿。

### ❌ 实质性缺失

**1. Graceful drain-then-switch 过渡（最高优先级）**

| 方面 | 文档设计 | 当前实现 |
|------|---------|---------|
| 数据面连续性 | 停止接收 → 等待活跃完成 → 激活 shadow | 直接 kill 所有进程 |
| 活跃请求处理 | 等待完成 | 全部丢弃 |
| Shadow 实例 | 预先创建 | 无 |

- 影响：reload 期间 inflight 请求全部丢失。对于延迟敏感的在线推理场景，这是最严重的功能缺口。
- 修复方向：在 `_trigger_full_reload()` 中先让 router 停止转发新请求 → 等待 drain timeout → 再执行重启。reload 完成后 router 恢复转发。

**2. 多节点拓扑约束**

- 文档设计：考虑"每节点 GPU 数 ≤ 8"、"k_c × tp_c ≤ G_node × N_nodes_c"
- 当前实现：GPU 池平坦化，无节点拓扑感知
- 影响：多节点部署时 ILP 可能给出跨节点分配不合理的解（如一个 AF 对跨 3 个节点），导致实际不可部署。
- 修复方向：在 `Tier1Solver.__init__` 中加入 `gpus_per_node` 参数，在 `_enumerate_pairs()` 的 GPU 计数中增加节点容量约束。

**3. P/D 重分配走统一 reload 路径**

- 文档设计：场景 3 有专门的"先缩再扩"过渡策略
- 当前实现：走通用 `reload_orchestrator` 全杀全启
- 影响：同 issue 1，但频率较低（P/D 重分配场景少）

---

## 四、推荐修复优先级

| 优先级 | 项目 | 工作量 | 影响面 |
|--------|------|--------|--------|
| P0 | Drain-then-switch（保留活跃请求） | 中 | reload 数据面连续性 |
| P1 | 多节点拓扑约束 | 小 | 多节点部署正确性 |
| P2 | E_bubble 频率感知空闲功率 | 小 | 能效精度 |
| P3 | 多负载分桶 SLO 约束 | 中 | 混合负载 ILP 精度 |
| P4 | Shadow 实例预创建 | 大 | 过渡平滑度 |
| P5 | P/D 专门过渡策略 | 中 | 最小化中断 |

---

*审查日期: 2026-06-20*
*基于: `benchmark/test_motivation/hucc/paper/system_detail_v1.md`*
