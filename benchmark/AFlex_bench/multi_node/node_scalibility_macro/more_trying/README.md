# PDAF 部署形态 Sweep（MegaScale / AFlex）

两节点 **16 卡**（node34 + node33，各 8 卡）、**code** 数据集、**QPS 2/4/6/8/12/16**。

只测 **MegaScale**（`pdaf_*_baseline`）与 **AFlex**（`pdaf_*_tier`）。PDAF 路径默认 **`--disable-radix-cache`**。

SLO：TTFT=5s，TPOT=300ms。

---

## 硬约束（PDAF 架构）

| 约束 | 说明 |
|------|------|
| A↔F 同节点 | PA↔PF、DA↔DF 必须节点内 **CUDA IPC**，不能跨节点 |
| P/D 可跨节点 | Prefill（PA+PF）与 Decode（DA+DF）之间 KV 走 **mooncake/RDMA** |
| 同组同 TP | 同一 PA/PF（或 DA/DF）对内 `--tp` 通常必须一致；**AF 异构**（如 PA_TP2+PF_TP4）需引擎支持，单独标注 |

---

## 三类部署（均占满 16 卡）

### 第 3 类 — 4 实例 × 跨节点 PD × TP1

**router：4P + 4D**。每个实例 **4 卡/节点**，四组件均为 **TP1**。

```
node34:  inst0..3 的 PA/PF（各 TP1，gpu 0-1 / 2-3 / 4-5 / 6-7）
node33:  inst0..3 的 DA/DF（各 TP1，同上）
```

| 项 | 值 |
|----|-----|
| 实例数 | 4 |
| 跨节点 | 是（每实例 P@node1，D@node2） |
| 占卡 | 4×4 = 16 |
| 代码 | `start_pdaf_multi(1)` → `pdaf_tp1_{baseline,tier}` |
| 状态 | **已测**（`pdaf_code_final.json`） |

---

### 第 2 类 — 2 实例 × 单节点完整 PDAF（不跨节点）

**router：2 worker**。每个节点 **1 个完整 PDAF 实例**，**8 卡全在本节点**（P+D 均在同一节点，无跨节点 KV）。

```
node34:  实例 #0 — PA/PF/DA/DF 全在 node34（8 卡）
node33:  实例 #1 — PA/PF/DA/DF 全在 node33（8 卡）
```

> 与第 1/3 类不同：**实例内部不跨节点**，避免 xnode 通讯开销。

每个节点 8 卡、1 个实例内的两种放法：

| 变体 | 四组件布局 | 占 8 卡 |
|------|-----------|--------|
| **2a** | PA/PF/DA/DF **全是 TP2** | 4×TP2 = 8 |
| **2b** | PA/PF/DA/DF **各 2 个 TP1** | 8×TP1 = 8 |

| 项 | 值 |
|----|-----|
| 实例数 | 2（每节点 1 个） |
| 跨节点 | **否**（实例内）；router 跨 2 节点做负载均衡 |
| 占卡 | 8+8 = 16 |
| 代码 | 需新 `start_pdaf_intra_full`（单节点 8 卡完整 PDAF） |
| 状态 | **未实现** |

建议 scheme key：`pdaf_intra_tp2_n1_{baseline,tier}`、`pdaf_intra_tp1x2_n1_{baseline,tier}`

---

### 第 1 类 — 1 实例 × 跨节点 PD（搜索空间最大）

**router：1P + 1D**（或 1 组 PA/PF 对 1 组 DA/DF）。整集群 **1 个逻辑 PDAF 部署**：

```
node34 (8卡):  Prefill  — PA + PF 的 AF 布局（可多种组合）
node33 (8卡):  Decode   — DA + DF 的 AF 布局（可多种组合）
```

P 侧与 D 侧 **独立配置**；两侧可同构或异构。

#### 单侧 8 卡上常见的 AF 同构布局

| 布局 ID | P 或 D 单侧 | 含义 | 占 8 卡 |
|---------|------------|------|--------|
| `tp4x1` | TP4 × 1 组 | 1×(PA_TP4 + PF_TP4) | 8 |
| `tp2x2` | TP2 × 2 组 | 2×(PA_TP2 + PF_TP2) | 8 |
| `tp1x4` | TP1 × 4 组 | 4×(PA_TP1 + PF_TP1) | 8 |

Decode 侧同理（DA/DF 替换 PA/PF）。

#### 第 1 类搜索维度（组合爆炸处）

1. **P 侧布局** × **D 侧布局**（9 种同构配对，如 P_tp2x2 + D_tp4x1）
2. **P/D 非对称**（两侧选不同 tp 网格）
3. **AF 异构**（同组内 PA_TP ≠ PF_TP，若引擎支持）
4. **MegaScale vs AFlex**（×2）

优先 sweep 建议（同构 P=D）：

```
(tp4x1, tp4x1)  (tp2x2, tp2x2)  (tp1x4, tp1x4)
(tp2x2, tp4x1)  (tp1x4, tp2x2)  … 非对称扩展
```

| 项 | 值 |
|----|-----|
| 实例数 | 1（逻辑） |
| 跨节点 | **是**（P 整节点 / D 整节点） |
| 占卡 | 8+8 = 16 |
| 代码 | 需新 `start_pdaf_xnode_single`（可配 P/D 侧 tp 网格） |
| 状态 | **未实现**（`start_pdaf(16)` 仅覆盖部分 TP4 单布局） |

建议 scheme key 示例：`pdaf_xnode_p{tp2x2}_d{tp4x1}_{baseline,tier}`

---

## 三类对比

| 类 | 实例数 | 实例内跨节点 | 16 卡用法 | 变量 |
|----|--------|-------------|----------|------|
| **1** | 1 | 是（P/D 分离） | 8+8 | **P/D 两侧 AF 组合**（最大） |
| **2** | 2 | 否 | 8+8 | TP2 同构 vs 8×TP1 |
| **3** | 4 | 是（每实例 P/D 分离） | 4×4 | 固定 TP1×4 |

---

## 实现与测试顺序

1. **第 3 类** — 已有，作 anchor（`run_pdaf_multi` / `pdaf_code_final.json`）
2. **第 2 类** — 单节点完整 PDAF launch + sweep（2a / 2b）
3. **第 1 类** — 单实例跨节点 + 可配 P/D 侧布局；先同构 3×3 网格，再非对称与 AF 异构

脚本规划：

| 脚本 | 用途 |
|------|------|
| `run_pdaf_deploy_sweep_code.py` | 统一 sweep 入口（按类/变体选择） |
| `plot_pdaf_deploy_sweep_code.py` | 横向 TP/布局对比图 |

结果命名：`pdaf_deploy_{class}_{variant}_{baseline,tier}_*.json`

---

## 与 6 方案主图的关系

`code_dashboard_4panel.png` 里的 MegaScale/AFlex 对应 **第 3 类**（`pdaf_tp1`）。  
本 sweep 用于在同一 code 数据集上找出 **更优 PDAF 部署**，再决定是否替换主图配置。

---

## 公平性

- PDAF：**始终** `--disable-radix-cache`
- 其他方案对比见 `noradix_3scheme_code_final_*.json`、`distserve_pd_hetero_code_final_*.json` 等
- 勿使用已删除的 `fixed_6scheme_*`（开缓存）旧结果
