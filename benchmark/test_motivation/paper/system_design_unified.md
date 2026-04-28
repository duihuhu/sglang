# AFlex 系统设计与实现规划（统一版）

> 整合自 system_design_paper1.md 第三章（系统设计）与第九章（代码实现规划）。
> 每个设计组件直接对应实现方案，形成 **设计→算法→实现→文件→状态** 的完整链条。

---

## 1. 问题定义

**前提**: PD 分离已完成，在此基础上进一步做 AF 分离。

**四个算子池**:

| 池 | 计算特征 | 频率敏感性 |
|---|---------|-----------|
| **PA** (Prefill-Attn) | 偏 memory-bound | 中等 |
| **PF** (Prefill-FFN) | compute-bound | **高** |
| **DA** (Decode-Attn) | 强 memory-bound | **低** |
| **DF** (Decode-FFN) | memory→compute 随 bs 变化 | **中→高** |

**优化变量**: (1) A/F 实例配比 (2) 算子级频率 f_A, f_F

**目标**: 满足 SLO (TTFT_P99, TPOT_P99) 下最小化总能耗

**系统参数**: 频率切换 P50 ~4.5ms, avg ~6ms; 频率候选 {210, 450, 690, 930, 1170, 1410} MHz

---

## 2. 整体架构: 两层控制

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     AF-Disaggregated Serving Cluster                    │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Tier 1: Joint Resource Provisioner (每 T₁ = 几分钟)               │  │
│  │  输入: 近期负载统计 (λ, 长度分布, 活跃 decode 数)                   │  │
│  │  联合决策: k_P, k_D, tp_PA/PF/DA/DF, f̄_PA/PF/DA/DF               │  │
│  │  约束: k_P×(tp_PA+tp_PF) + k_D×(tp_DA+tp_DF) ≤ G_total           │  │
│  │  方法: ILP (Profile 表驱动)                                        │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              ↓ 配置下发                                 │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Tier 2: Per-Iteration AF-DVFS (每次 iteration / 每个决策窗口)      │  │
│  │  在 Tier 1 基线频率附近微调 f_A, f_F                                │  │
│  │  Prefill: per-request 调频 (SLO slack 驱动)                        │  │
│  │  Decode:  per-window 调频 (频率决策窗口 ~60 iterations)             │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │ PA 实例群  │  │ PF 实例群  │  │ DA 实例群  │  │ DF 实例群  │               │
│  │ (f_PA)    │  │ (f_PF)    │  │ (f_DA)    │  │ (f_DF)    │               │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Tier 1: 联合资源规划 (ILP)

### 3.1 设计: 联合优化 vs 分层优化

一种自然的做法是**分层优化**: 先决定 P/D 资源划分 (复用 BiScale/DistServe)，再在各自内部优化 A/F 配比。但我们选择**联合优化**，原因如下:

**P/D 分配与 A/F 配比存在 6 条耦合**:

1. **资源竞争**: k_P×(tp_PA+tp_PF) + k_D×(tp_DA+tp_DF) ≤ G，给 Prefill 多一对 AF，Decode 就少 GPU
2. **延迟耦合**: PA 和 PF 的频率/TP 联合决定能否满足 TTFT SLO
3. **吞吐耦合**: PA 和 PF 的副本数各自需匹配 λ，但用不同 TP/频率时所需副本数不同
4. **能耗-频率权衡的跨池传递**: 降低 f̄_DA 省能耗但可能需要更多 DA 副本 → 挤占 PF 的 GPU → PF 需要更高频率
5. **显存耦合**: A 实例有 KV cache (显存随 bs×seq 增长)，F 实例无 KV cache 但权重大；不同 TP 下显存分布不同
6. **Pipeline bubble 耦合**: A/F 延迟越不均衡，bubble 越大；配比和频率共同决定均衡度

```
例: 总共 16 GPU

分层优化:
  Level 0: 先按吞吐匹配决定 n_P=8, n_D=8
  Level 1: 在 n_P=8 内优化 → k_P=2 对 (tp_PA=1, tp_PF=3 → 每对 4 GPU, 共 8)
           在 n_D=8 内优化 → k_D=2 对 (tp_DA=2, tp_DF=2 → 每对 4 GPU, 共 8)

联合优化可能发现:
  k_P=1 对 (tp_PA=2, tp_PF=4 → 6 GPU), k_D=2 对 (tp_DA=1, tp_DF=4 → 每对 5, 共 10)
  → Decode-FFN 在低频下延迟增加大，多给它 GPU (tp_DF=4) 可用更低频率
  → 虽然 Prefill 只有 1 对 (6 GPU)，但高 TP 下单实例吞吐足够
  → 这种跨层优化只有联合方案能发现
```

**联合优化的可行性**: 搜索空间虽大于分层，但实际可控:
- TP 度候选有限: {1, 2, 4, 8}
- 频率候选有限: {210, 450, 690, 930, 1170, 1410}
- 配对约束使独立变量从 4 个 k 降为 2 个 (k_P, k_D)
- 大量组合因违反 SLO 或超出资源可提前剪枝
- ILP 求解器 (Gurobi/CPLEX/PuLP) 在此规模下通常秒级可解

### 3.2 ILP 公式

```
minimize:
  Σ_{c ∈ {P,D}} k_c × [E_A(tp_cA, f̄_cA, wl) + E_F(tp_cF, f̄_cF, wl) + E_bubble(M)]

subject to:
  (1)  k_P × (tp_PA + tp_PF) + k_D × (tp_DA + tp_DF) ≤ G
  (2)  tp_c ∈ {1, 2, 4, 8}
  (3)  k_P, k_D ∈ Z⁺
  (4)  f̄_c ∈ {210, 450, 690, 930, 1170, 1410}

  // 延迟 SLO (保守上界, M=1; Tier 2 运行时用精确 pipeline 公式)
  (5)  ∀r ∈ R_prefill: t_PA(r) + t_PF(r) + t_comm ≤ TTFT_SLO / L
  (6)  ∀r ∈ R_decode:  t_DA(r) + t_DF(r) + t_comm ≤ TPOT_SLO

  // 吞吐容量
  (7)  k_P × min(Thpt_PA, Thpt_PF) ≥ (1+α) × λ
  (8)  k_D × min(Thpt_DA, Thpt_DF) ≥ (1+α) × N_active

  // A/F 延迟平衡 (剪枝用)
  (9)  |t_A - t_F| ≤ β × max(t_A, t_F)    // β ∈ [0.5, 0.8]

  // 显存
  (10) Mem_A(tp_cA, bs_max, seq_max) ≤ GPU_MEM
       // Mem_A = W_attn/tp + KV_cache(bs_max, seq_max, tp)
       // KV_cache = 2 × L × num_kv_heads/tp × head_dim × bs_max × seq_max × dtype_size
  (11) Mem_F(tp_cF) ≤ GPU_MEM
       // Mem_F = W_ffn/tp + activation_buffer (无 KV cache)
```

### 3.3 A/F 负载不对称性驱动配比

F/A 延迟比随配置变化，是联合优化发现差异化配比的关键数据基础:

| 配置 | P_A (us) | P_F (us) | F/A 比 | 联合优化的含义 |
|------|---------|---------|--------|-------------|
| tp=1, in=128, bs=1, 1410MHz | 340 | 576 | 1.69 | F 是瓶颈 → 多给 PF GPU |
| tp=1, in=8192, bs=1, 1410MHz | 11,899 | 25,440 | 2.14 | F 是瓶颈 → 多给 PF GPU |
| tp=4, in=128, bs=1, 1410MHz | 338 | 254 | **0.75** | A 是瓶颈 → 多给 PA GPU |
| tp=4, in=1024, bs=4, 1410MHz | 1,011 | 4,132 | **4.09** | F 强瓶颈 → 倾斜给 PF |
| tp=8, in=128, bs=4, 1410MHz | 343 | 417 | **1.22** | 接近均衡 |

当负载以长请求为主时，ILP 倾向给 PF 分配更多 GPU；短请求为主时反之。F/A 比在 0.75~4x 范围内，ILP 配比有显著优化空间。

**变量汇总**:

| 变量 | 含义 | 类型 |
|------|------|------|
| k_P (= k_PA = k_PF) | Prefill AF 对数 | 正整数 |
| k_D (= k_DA = k_DF) | Decode AF 对数 | 正整数 |
| tp_PA, tp_PF, tp_DA, tp_DF | 各池 TP 度 | {1,2,4,8} |
| f̄_PA, f̄_PF, f̄_DA, f̄_DF | 各池基线频率 | {210..1410} |

**输入参数**:

| 参数 | 来源 |
|------|------|
| G (GPU 总量) | 集群配置 |
| λ (Prefill RPS) | 负载监控 |
| N_active (活跃 Decode 数) | 负载监控 |
| R_prefill, R_decode (请求类型桶) | 负载监控 |
| TTFT_SLO, TPOT_SLO | 用户配置 |
| M (microbatch 数) | AFD 配置 |

### 3.4 ILP 求解加速

1. **Profile 表预计算**: 离线穷举 (tp, freq, workload_bin) → (latency, energy_mj)
2. **Pareto 剪枝**: 24 种 (tp, freq) 配置降至 Pareto 前沿上的少数几种
3. **Warm Start**: 上一窗口的解作为初始可行解
4. **负载分桶**: 约束只针对桶的 P90 代表值（P90 在保守性和资源效率之间取得平衡）
   - Prefill 分桶: 按 input_len 分桶 (如 [0,512), [512,2048), [2048,8192), [8192,+∞))
   - Decode 分桶: 按 (input_len, output_len) 联合分桶
   - 显存约束中的 bs_max, seq_max 取桶内 P99

### 3.5 Ablation 设计 (评估用)

| 系统 | 描述 | 预期能耗 |
|------|------|---------|
| **Max-Freq** | 所有 GPU 最高频率 (DistServe 风格) | 最高 (基线) |
| **Unified-DVFS** | 同频给 A 和 F (统一调频) | 中等 |
| **PD-Only** | PD 分离但无 AF 分离 (BiScale 方案) | 中等 |
| **Hierarchical** | 分层优化 (先 P/D 再 A/F) | 较低 |
| **Fixed-Ratio** | A/F=1:1 固定, 仅优化频率 | 较低 |
| **Joint (本文)** | 联合优化 P/D+A/F 配比+频率 | **最优** |

量化两个维度的增益: AF 分离增量 (Joint vs PD-Only) + 联合优化增量 (Joint vs Hierarchical)

### 3.6 重规划触发与过渡

**触发条件** (Monitoring 模块, 每 10-30s 采样):
```
metric_1: SLO_violation_rate > threshold        // SLO 违反率过高
metric_2: |A_util - F_util| > δ                  // A/F 利用率失衡
metric_3: |P_util - D_util| > δ                  // P/D 利用率失衡
metric_4: workload_distribution_shift detected   // 请求长度分布突变 (KL 散度 / 均值偏移)

任一指标在连续多个窗口超过阈值 → 触发 Tier 1 ILP 重规划
```

**开销控制**:
- 预缓存 PA/PF/DA/DF 四种角色的权重在本地磁盘/CPU 内存
- Shadow instancing: 后台预创建新配置的实例，ready 后原子切换
- 控制重规划间隔不低于 T₁_min (如 2-5 分钟)

**过渡策略**:
```
场景 1: 仅频率变化 (tp 不变, 仅调整 f̄)
  → 最轻量: 直接切频 (~6ms), 无需迁移, 请求不中断

场景 2: TP 变化但池角色不变 (如 tp_DA: 2→4)
  → drain-then-switch:
    (1) 停止接收新 Decode 请求到旧实例
    (2) 等待旧实例上的活跃请求完成 (或达到超时)
    (3) 启动新 TP 配置的实例 (shadow instancing 已预创建)
    (4) 新请求路由到新实例
  → 不做在线 KV cache 迁移 (复杂度高, 收益低)
  → 过渡期: 数秒到数十秒 (取决于剩余 output tokens)

场景 3: P/D 资源重分配 (如 n_P: 8→6, n_D: 8→10)
  → 先缩 Prefill (drain), 再扩 Decode (shadow instancing)
  → 过渡期 Prefill 容量暂时下降, Tier 2 自动升频补偿

紧急回退: 若重规划期间 SLO 违反率飙升
  → Tier 2 立即升频到 f_max (1410MHz)
  → 若仍不够, 启用准入控制: 排队新请求, 优先完成已有请求
```

### 3.5 实现规划

| 子任务 | 产出文件 | 耗时 | 状态 |
|--------|---------|------|------|
| B-1 Profile 表构建 | `sglang/srt/energy/profile_table.py` | 1 天 | ❌ |
| B-2 ILP 求解器 | `sglang/srt/energy/tier1_solver.py` | 4 天 | ❌ |
| B-3 负载监控 | `sglang/srt/energy/workload_monitor.py` | 2 天 | ❌ |

**B-1 Profile 表构建**:
```python
class ProfileTable:
    """加载 prefill_data_v1.txt / decode_data_v1.txt, 构建查找表"""
    def lookup(self, phase, op, tp, freq, il, ol, bs) -> (latency_us, energy_mj)
    def get_pareto_configs(self, phase) -> list[Config]  # Pareto 剪枝
```

**B-2 ILP 求解器**:
```python
class Tier1Solver:
    """PuLP/Gurobi ILP 求解"""
    def solve(self, G, lambda_rps, N_active, workload_bins,
              ttft_slo, tpot_slo, M) -> ClusterConfig
    # ClusterConfig = (k_P, k_D, tp_PA, tp_PF, tp_DA, tp_DF, f̄_PA, f̄_PF, f̄_DA, f̄_DF)
```

**B-3 负载监控**:
```python
class WorkloadMonitor:
    """滑动窗口统计 + 重规划触发"""
    def update(self, request) -> None
    def should_replan(self) -> bool
    def get_stats(self) -> WorkloadStats  # λ, N_active, 长度分布
```

---

## 4. Tier 2: 算子级 DVFS

### 4.1 设计

A GPU 和 F GPU 独立选频，满足 SLO 下最小化能耗。

**与 AFD microbatch pipeline 的关系**:
- M 个 microbatch 形成流水线，A/F 交替执行
- 单层延迟: `max(t_A, t_F) + t_comm/M` (M>1) 或 `t_A + t_F + t_comm` (M=1)
- Tier 2 使用精确 pipeline 公式（Tier 1 用保守上界）
- M 的选择: M=2 默认推荐; M=1 适用于小 batch (bs≤4); M=3 适用于大 batch (bs≥64)

**调频粒度**:
- Prefill: per-request（切频 ~6ms vs 单请求 ~10-500ms，开销可接受）
- Decode: per-window（窗口 ~60 iterations ≈ 60ms，切频 ~6ms 占 10%）

> **频率切换实测数据** (A800-80GB SXM, SetGpuLockedClocks):
>
> | 指标 | 值 |
> |------|-----|
> | API latency P50 | ~4.5 ms |
> | API latency avg | ~6.0 ms |
> | API latency P99 | ~13 ms |
> | Settle avg | ~0.9 ms |
> | 多 GPU 串行 (N 卡) | ~N × 6ms (NVIDIA 内核驱动全局锁) |
>
> AF 分离架构中每个池独立调频，单次涉及 1-2 卡，开销 6-12ms。

**Prefill per-request 论证**:
- 短序列 (il=128, bs=1) 单层延迟仅 ~0.3-1ms，per-layer 切频 ~6ms 远大于单层延迟 → 不可行
- 长序列 (il=8192+) 单层延迟 ~10-50ms，per-layer 理论可行但收益有限
- 统一采用 per-request: 请求开始前选频一次，整个请求期间保持不变
- 切频开销 ~6ms vs 请求总延迟 ~10-500ms，开销占比 1-60%，可接受

**为什么不用 MPC**:
- BiScale 的 MPC 是因为 Prefill 只有一个频率旋钮 (f_P)，需要在时间维度上优化未来 K 个 batch 的频率序列
- AF 分离提供了空间维度的额外自由度 (f_A vs f_F)，使得单步的 (f_A, f_F) 联合搜索就已经有足够大的节能空间
- 单步搜索 O(36) 常数时间，无需预测未来负载，实现简单且鲁棒

### 4.2 Prefill DVFS 算法

```
输入: batch B, 每个请求的 TTFT deadline d_r, M

1. slack = min_{r∈B}(d_r - elapsed_r)
2. for (f_A, f_F) in all 36 combos:
     t_layer = max(M_PA.lat(f_A), M_PF.lat(f_F)) + t_comm/M
     if t_layer × remaining_layers ≤ slack:
       e = M_PA.energy(f_A) + M_PF.energy(f_F)
       candidates.add((f_A, f_F, e))
3. return argmin energy in candidates
```

### 4.3 Decode DVFS 算法

```
频率决策窗口 W = max(W_min, ceil(10 × t_switch / t_iter_avg))  // 典型 ~60 iters

触发条件 (任一):
  - 距上次决策已过 W 个 iteration
  - batch_size 变化 > 30%
  - TPOT > SLO × 0.9

输入: batch B, SLO_TPOT, M, 当前频率 (f_cur_A, f_cur_F)

1. for (f_A, f_F) in sorted_by_energy(all_combos):
     t_layer = max(M_DA.lat(f_A), M_DF.lat(f_F)) + t_comm/M
     if t_layer ≤ SLO_TPOT:
       if should_switch(f_new, f_cur):  // 惰性切频
         return (f_A, f_F)
2. return (1410, 1410)  // fallback
```

**惰性切频**: 避免频繁切频的开销浪费
```
should_switch(f_new, f_cur, W_remaining) =
  (f_new ≠ f_cur)
  AND (|E(f_cur) - E(f_new)| × W_remaining > E_switch)
  AND (|f_new - f_cur| ≥ f_step_min)

其中:
  E_switch = P_avg × t_switch ≈ 300W × 6ms = 1.8 J = 1800 mJ
  W_remaining = 当前窗口剩余 iteration 数
  f_step_min = 最小频率跳变 (如 240 MHz, 即至少跳一档)
```

### 4.4 实现规划

| 子任务 | 产出文件 | 耗时 | 状态 |
|--------|---------|------|------|
| A-1 预测模型接口 | `sglang/srt/energy/af_profile_predictor.py` | 2 天 | ❌ |
| A-2 DVFS 控制器 | `sglang/srt/energy/af_dvfs_controller.py` | 3 天 | ❌ |
| A-3 调度器集成 | 修改 `scheduler.py`, `afd.py` | 2 天 | ❌ |

**A-1 预测模型接口**:
```python
class AFProfilePredictor:
    """LUT 精确匹配 + GBDT/LinearReg 插值兜底"""
    def predict_latency(self, phase, op, tp, freq, bs, il, ol=None) -> float
    def predict_energy(self, phase, op, tp, freq, bs, il, ol=None) -> float
    # Decode: GBDT (MAPE 2-4%), Prefill: LinearReg (MAPE 14-18%)
```

**A-2 DVFS 控制器**:
```python
class AFDVFSController:
    def select_freq_prefill(self, batch, slack, M) -> (f_A, f_F)
    def select_freq_decode(self, batch, slo_tpot, M) -> (f_A, f_F)
    def should_switch(self, f_new, f_cur, window_remaining) -> bool
```

**A-3 调度器集成** (插入点):
```python
# scheduler.py — event_loop_afd 中:
# Prefill 路径: 调度 batch 后, forward 前
f_A, f_F = dvfs_ctrl.select_freq_prefill(batch, slack, M)
dvfs_mgr.lock_clocks(attn_gpu=f_A, ffn_gpu=f_F)

# Decode 路径: 每个决策窗口
if dvfs_ctrl.should_reevaluate(window_counter, batch_changed):
    f_A, f_F = dvfs_ctrl.select_freq_decode(batch, slo_tpot, M)
    if dvfs_ctrl.should_switch(f_new=(f_A, f_F), f_cur=current_freq):
        dvfs_mgr.lock_clocks(attn_gpu=f_A, ffn_gpu=f_F)
```

---

## 5. 能耗模型

```
M=1 (无 pipeline):
  E_iter = E_A(f_A, batch) + E_F(f_F, batch) + E_comm

M>1 (pipeline):
  E_iter = E_A + E_F + E_comm + E_bubble
  E_bubble = P_idle(f_fast) × |t_A - t_F| × (M-1)/M
  // P_idle(f) 来自 idle_power.txt (T1-6)
  // 当 t_A ≈ t_F 时 bubble 最小 → A/F 吞吐平衡约束的物理意义
```

> **DF 频率敏感性随 bs 变化**:
> - bs=1~4: DF 偏 memory-bound (权重加载主导), 降频影响小
> - bs=16~64: DF 转为 compute-bound (矩阵乘主导), 降频显著增加延迟
> - bs=128+: DF 强 compute-bound, 频率敏感性最高
> - 因此 Tier 2 在大 bs 时倾向给 DF 保持高频, 小 bs 时可大幅降频

**AF 分离的 Pareto 优势论证**:
```
E_AF(f_A, f_F) ≤ E_unified(f)  当 f 使得延迟相同时

证明 (Decode 场景):
  DA 是强 memory-bound → t_DA(f) ≈ const (降频几乎不增延迟)
                        → E_DA(f) ∝ f (能耗随频率线性降低)
  DF 是 compute-bound  → t_DF(f) ∝ 1/f (降频线性增延迟)
                        → E_DF(f) ≈ const (能耗不随频率变化)

  统一调频 f: E_unified = E_DA(f) + E_DF(f), 受 t_DA(f)+t_DF(f) ≤ SLO 约束
  AF 调频:   E_AF = E_DA(f_A) + E_DF(f_F), f_A 可以远低于 f_F
             → E_DA(f_A) << E_DA(f) 且 t_DA(f_A) ≈ t_DA(f)
             → 总延迟不变, 总能耗降低

  实测: Decode tp=1 bs=16 il=1024 场景, AF 差异化调频比统一调频节省 5-9% 能耗
```

**模型拟合结果** (5-fold CV MAPE):

| 子模型 | LUT | LinearReg | GBDT | 最佳 |
|--------|-----|-----------|------|------|
| Prefill_A energy | 55.7% | **17.6%** | 23.5% | LinearReg |
| Prefill_F energy | 57.4% | **13.8%** | 21.9% | LinearReg |
| Decode_A energy | 7.9% | 11.3% | **3.7%** | GBDT |
| Decode_F energy | 3.3% | 10.6% | **2.0%** | GBDT |

**系统中的使用**:
- Tier 1 ILP: LUT 精确查表 (0% 误差, 输入都是网格点)
- Tier 2 运行时: GBDT (Decode) / LinearReg (Prefill) 插值预测
- 兜底: LUT 优先精确匹配, 查不到再用模型插值

---

## 6. 动态扩缩容 (模块 C)

### 6.1 设计

Tier 1 重规划可能改变 TP 度或 P/D 资源分配，需要运行时切换。核心挑战：
- 权重按 TP 切分，TP 变化需要重新加载权重 (~2-5s)
- NCCL 通信组与 TP 绑定，需要预建多组
- KV cache 按 num_kv_heads 分布在 TP rank 间，TP 变化需要重分布

### 6.2 实现规划

| 子任务 | 产出文件 | 耗时 | 状态 |
|--------|---------|------|------|
| C-1 权重预缓存 | `sglang/srt/energy/weight_cache.py` | 3 天 | ❌ |
| C-2 NCCL 组管理 | `sglang/srt/energy/tp_group_manager.py` | 2 天 | ❌ |
| C-3 扩缩容编排器 | `sglang/srt/energy/orchestrator.py` | 4 天 | ❌ |

**C-1 权重预缓存**:
```python
class WeightCache:
    """预加载 4 种 TP 切分的权重到 CPU/NVMe"""
    # 启动时: 对 tp ∈ {1,2,4,8}, 预切分权重并缓存
    # 切换时: GPU 从 CPU 加载对应 TP 的权重 (~2-5s, PCIe 带宽瓶颈)
    # 显存估算 (Qwen3-32B, bf16):
    #   tp=1: ~60GB/卡, tp=2: ~30GB/卡, tp=4: ~15GB/卡, tp=8: ~7.5GB/卡
    #   CPU 缓存 4 种 TP: ~60GB × 4 = 240GB CPU 内存
    def load_weights(self, tp: int, gpu_ids: list[int]) -> None
    def preload_all_tp(self) -> None
```

**C-2 NCCL 组管理**:
```python
class TPGroupManager:
    """预建所有 TP 度的 NCCL 通信组"""
    # 启动时预建:
    #   tp=1: 无需 all-reduce
    #   tp=2: 预建 (0,1), (2,3), ... 的通信组
    #   tp=4: 预建 (0,1,2,3), (4,5,6,7), ... 的通信组
    #   tp=8: 预建 (0..7) 的通信组
    # 切换时: 选择对应的预建通信组, 无需 destroy/create
    def get_group(self, tp: int, replica_id: int) -> ProcessGroup
    def switch_tp(self, new_tp: int) -> None
```

**C-3 扩缩容编排器**:
```python
class AFOrchestrator:
    """根据 Tier 1 输出编排实例创建/销毁/TP 切换"""
    def apply_plan(self, new_config: ClusterConfig) -> None:
        # 场景 1 (仅频率): dvfs_mgr.lock_clocks(), ~6ms
        # 场景 2 (TP 变化):
        #   (a) drain 旧实例 (停止接收新请求, 等待完成)
        #   (b) WeightCache.load_weights(new_tp, gpus)
        #   (c) TPGroupManager.switch_tp(new_tp)
        #   (d) 恢复接收请求
        # 场景 3 (P/D 重分配):
        #   (a) drain Prefill 实例
        #   (b) 加载 Decode 角色权重
        #   (c) 启动新 Decode 实例
```

---

## 7. 端到端闭环

```
WorkloadMonitor (每 10-30s)
    ↓ 触发条件满足
Tier1Solver.solve() → ClusterConfig
    ↓
AFOrchestrator.apply_plan()
    ↓ 过渡期
Tier 2 自动升频保护
    ↓ 稳态
Tier 2 正常调频
```

| 子任务 | 产出 | 耗时 | 状态 |
|--------|------|------|------|
| D-1 闭环串联 | 修改 `scheduler.py` | 2 天 | ❌ |
| D-2 配置热加载 | HTTP API | 1 天 | ❌ |

---

## 8. 现有代码基础

| 组件 | 状态 | 关键文件 |
|------|------|---------|
| AFD 核心 | ✅ | `afd.py`, `afd_mixin.py`, `afd_overlap.py` |
| AFD 调度器 | ✅ | `scheduler.py` (`event_loop_afd`) |
| AFD 通信 (ZMQ/UCX) | ✅ | `afd.py`, `rdma_comm.py` |
| AFD 异构 TP | ✅ | `server_args.py` (`afd_attn_tp`, `afd_ffn_tp`) |
| DVFS 控制器 | ✅ | `dvfs.py`, `dvfs_ctrl.cpp` |
| Profile 数据 | ✅ | `prefill_data_v1.txt` (906行), `decode_data_v1.txt` (7530行) |
| 能耗模型 | ✅ | `energy_model.py` + `energy_models/` |
| Trace 数据 | ✅ | `prepare_trace.py`, Azure Code/Conv traces |
| **Tier 2 DVFS 集成** | ❌ | A-1 ~ A-3 |
| **Tier 1 ILP** | ❌ | B-1 ~ B-3 |
| **动态扩缩容** | ❌ | C-1 ~ C-3 |
| **端到端闭环** | ❌ | D-1 ~ D-2 |

---

## 9. 实现优先级与依赖

```
                    ┌─────────────────────────────────────────┐
                    │  已有基础: AFD + DVFS + Profile 数据     │
                    └──────────────┬──────────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
     ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐
     │ A-1 预测模型  │    │ B-1 Profile表 │    │ C-1 权重预缓存    │
     │   (2天)      │    │   (1天)       │    │   (3天)          │
     └──────┬──────┘    └──────┬───────┘    └────────┬─────────┘
            │                  │                     │
            ▼                  ▼                     ▼
     ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐
     │ A-2 DVFS控制 │    │ B-2 ILP求解器 │    │ C-2 NCCL组管理    │
     │   (3天)      │    │   (4天)       │    │   (2天)          │
     └──────┬──────┘    └──────┬───────┘    └────────┬─────────┘
            │                  │                     │
            ▼                  ▼                     ▼
     ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐
     │ A-3 调度集成  │    │ B-3 负载监控  │    │ C-3 编排器        │
     │   (2天)      │    │   (2天)       │    │   (4天)          │
     └──────┬──────┘    └──────┬───────┘    └────────┬─────────┘
            │                  │                     │
            └────────────────┬─┴─────────────────────┘
                             ▼
                    ┌──────────────────┐
                    │ D-1 端到端闭环     │
                    │   (2天)           │
                    └──────────────────┘
```

**推荐实现顺序**:
1. **Phase 1 (1 周)**: A-1 → A-2 → A-3 (Tier 2 DVFS, 最快出结果)
2. **Phase 2 (1 周)**: B-1 → B-2 → B-3 (Tier 1 ILP)
3. **Phase 3 (2 周)**: C-1 → C-2 → C-3 (动态扩缩容)
4. **Phase 4 (3 天)**: D-1 → D-2 (端到端闭环)

**总计**: ~4-5 周

---

## 10. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| NVIDIA 内核驱动全局锁导致多 GPU 切频串行 | 8 卡切频 ~46ms, 可能超过 Decode 决策窗口 | AF 分离后每次只切 1-2 卡 (~6-12ms); 惰性切频减少切频次数 |
| Prefill 能耗模型 MAPE 14-18% | Tier 2 选频可能不是最优 | LUT 精确匹配兜底; Prefill 请求少, 影响有限 |
| ILP 求解时间超预期 | 重规划延迟增加 | Warm Start + Pareto 剪枝; 设置求解时间上限 (如 10s) |
| TP 切换期间服务中断 | 过渡期 SLO 违反 | drain-then-switch + Tier 2 升频保护 + 准入控制 |
| 负载突变导致频繁重规划 | 系统不稳定 | 重规划间隔下限 T₁_min; 触发条件需连续多窗口满足 |
| Profile 数据不覆盖运行时参数 | 模型外推误差大 | GBDT/LinearReg 插值; 运行时监控 MAPE, 超阈值回退 f_max |

## 11. 新增文件清单

```
python/sglang/srt/energy/
├── af_profile_predictor.py    # A-1: 能耗/延迟预测接口
├── af_dvfs_controller.py      # A-2: Tier 2 DVFS 控制器
├── profile_table.py           # B-1: Profile 表构建与查询
├── tier1_solver.py            # B-2: ILP 求解器
├── workload_monitor.py        # B-3: 负载监控
├── weight_cache.py            # C-1: 权重预缓存
├── tp_group_manager.py        # C-2: NCCL 组管理
└── orchestrator.py            # C-3: 扩缩容编排器

修改文件:
├── scheduler.py               # A-3: Tier 2 调频插入点 + D-1: 闭环
├── afd.py                     # A-3: model_forward_afd 中加入调频
├── server_args.py             # 新增 energy 相关参数
└── layers/communicator.py     # C-2: 支持动态 process_group
```

## 12. 工时明细

| Phase | 模块 | 子任务 | 工时 | 累计 |
|-------|------|--------|------|------|
| 1 | A (Tier 2 DVFS) | A-1 预测模型 | 2 天 | 2 天 |
| 1 | A | A-2 DVFS 控制器 | 3 天 | 5 天 |
| 1 | A | A-3 调度器集成 | 2 天 | **7 天** |
| 2 | B (Tier 1 ILP) | B-1 Profile 表 | 1 天 | 8 天 |
| 2 | B | B-2 ILP 求解器 | 4 天 | 12 天 |
| 2 | B | B-3 负载监控 | 2 天 | **14 天** |
| 3 | C (扩缩容) | C-1 权重预缓存 | 3 天 | 17 天 |
| 3 | C | C-2 NCCL 组管理 | 2 天 | 19 天 |
| 3 | C | C-3 编排器 | 4 天 | **23 天** |
| 4 | D (闭环) | D-1 端到端串联 | 2 天 | 25 天 |
| 4 | D | D-2 配置热加载 | 1 天 | **26 天** |
