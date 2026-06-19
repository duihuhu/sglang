# Tier1 Solver Bug 修正记录

## 修正总览

将"四池独立枚举 → 事后拼接"改为"AF Pair 联合优化"，修复 6 项错误。

| 错误 | 根因 | 修正方案 |
|------|------|----------|
| 1. Profile 查询绑定同 tp/freq | `query_metrics(tp, freq)` 同时返回 A+F，候选绑定在一行上 | 新增 `_enumerate_op_candidates` 分别独立枚举 A 和 F |
| 2. 能耗双重计算 (~2×) | `e_total = e_a + e_f` 在 PA 和 PF 各算一次，相加后 A/F 各计两遍 | `_AFPairCandidate.e_pair_mj = E_A(tp_A,f_A) + E_F(tp_F,f_F)` 每算子只计一次 |
| 3. SLO 用假 pair 剪枝 | 同行 `t_A(tp,f) + t_F(tp,f)` 做剪枝，与最终异构 pair 不一致 | 联合剪枝 `t_A(tp_A,f_A) + t_F(tp_F,f_F) + comm` |
| 4. 吞吐按池独立 | PA/PF 各自声明满足 λ，未验证 pipeline 瓶颈 | `throughput_per_pair` 基于整个 pair 的 `t_pair_us` 计算 |
| 5. Bubble 基础错误 | 公式正确但输入选自错误能耗目标 | `_pair_bubble_energy` 直接取正确选出的 pair |
| 6. Hetero TP 事后拼接 | 搜索空间 `(tp, freq)` 绑定 A 和 F | 笛卡尔积 `cands_A × cands_F` 原生枚举异构 pair |

## 数据结构变更

### 删除: `_PoolCandidate`

```python
# 旧: 一个候选绑定 A+F 在同一个 (tp, freq) 上
@dataclass
class _PoolCandidate:
    tp: int
    freq: int
    t_a_us: float
    t_f_us: float
    e_a_mj: float
    e_f_mj: float
    t_layer_us: float = 0.0
    e_total_mj: float = 0.0        # ← BUG: PA 池也包含 F 能耗
    throughput_per_pair: float = 0.0

    def __post_init__(self):
        self.e_total_mj = self.e_a_mj + self.e_f_mj  # ← 双重计算根源
```

### 新增: `_OpCandidate` + `_AFPairCandidate`

```python
@dataclass
class _OpCandidate:
    """单算子候选，只带自己的 latency/energy"""
    tp: int
    freq: int
    t_us: float       # 仅 A 或仅 F 的延迟
    e_mj: float       # 仅 A 或仅 F 的能耗


@dataclass
class _AFPairCandidate:
    """联合优化单元: (tp_A, freq_A, tp_F, freq_F)"""
    tp_a: int
    freq_a: int
    tp_f: int
    freq_f: int
    t_a_us: float           # t_A(tp_A, f_A)
    t_f_us: float           # t_F(tp_F, f_F)
    e_a_mj: float           # E_A(tp_A, f_A)
    e_f_mj: float           # E_F(tp_F, f_F)
    t_pair_us: float = 0.0  # t_A + t_F + t_comm
    e_pair_mj: float = 0.0  # E_A + E_F (无双重计算)
    throughput_per_pair: float = 0.0

    def __post_init__(self):
        self.e_pair_mj = self.e_a_mj + self.e_f_mj
```

## 算法流程变更

### Phase 1: 候选枚举

```python
# 旧: 每个池枚举 4×6=24 个 (tp,freq)，A/F 绑定
cand_pa = _enumerate_pool("prefill", "A", ...)  # 24 candidates
cand_pf = _enumerate_pool("prefill", "F", ...)  # 24 candidates (本质相同)

# 新: 先独立枚举 A 和 F，再做笛卡尔积
cands_a = _enumerate_op_candidates("prefill", "A", ...)  # 24 candidates
cands_f = _enumerate_op_candidates("prefill", "F", ...)  # 24 candidates
pairs_p = []
for ca in cands_a:
    for cf in cands_f:
        t_pair = ca.t_us + cf.t_us + t_comm  # 联合 SLO (修复错误3)
        if t_pair > slo_per_layer:
            continue
        if abs(ca.t_us - cf.t_us) > beta * max(ca.t_us, cf.t_us):
            continue
        pairs_p.append(_AFPairCandidate(...))
# 枚举空间: 24×24 = 576 (含 hetero TP, 修复错误6)
```

### Phase 2: Pareto 过滤

```python
# 旧: 对每个池按 (t_layer, e_total) 过滤
pareto = _pareto_filter(cand_pa)  # e_total 含双重计数

# 新: 对 pair 按 (t_pair_us, e_pair_mj) 过滤
pareto = _pareto_filter_pairs(pairs_p)  # e_pair 正确
```

### Phase 3: (k_P, k_D) 搜索

```python
# 旧: 每个池独立选最优 → 事后拼接
sel_pa = _select_for_pool(cand_pa, k, λ, α)  # 错误4: 独立吞吐
sel_pf = _select_for_pool(cand_pf, k, λ, α)
e_total = sel_pa.e_total + sel_pf.e_total     # 错误2: 双重计算

# 新: 从 pair 候选中选最优
sel_p = _select_pair(pairs_p, k, λ, α)       # pair 级吞吐 (修复错误4)
e_total = sel_p.e_pair_mj                     # 正确: E_A + E_F 只算一次 (修复错误2)
```

### Bubble 计算

```python
# 旧: 参数来自两个独立选出的池候选
t_wait = abs(cand_a.t_a_us - cand_f.t_f_us)

# 新: 直接取 pair 内部值
t_wait = abs(pair.t_a_us - pair.t_f_us)       # 同一 pair, 选择基础正确 (修复错误5)
```

## 搜索空间对比

| 维度 | 四池独立（旧） | AF Pair 联合（新） |
|------|---|---|
| 最终解空间（单 phase） | 576 | 576 |
| 最终解空间（P+D） | 331,776 | 331,776 |
| 枚举候选数（单 phase） | ~48 (PA+PF) | 576 (笛卡尔积) |
| 枚举候选数（P+D） | ~96 | 1,152 |
| 每个 k 探索的联合组合 | 1 组（贪心独立选） | 剪枝后全部 pair 里选最优 |
| 相对倍数 | 1× | ~12× (576/48) |

倍数来自：旧代码每个 (tp, freq) 只查一次 profile 行；联合方式对每个 (tp_A, tp_F, f_A, f_F) 分别查 t_A(tp_A,f_A) 和 t_F(tp_F,f_F)。

## 数值验证 (bug.md 中的例子)

假设 profile 数据 (bs=4, il=1024):

| (tp, freq) | t_A (us) | t_F (us) | E_A (mJ) | E_F (mJ) |
|---|---|---|---|---|
| (4, 690) | 1000 | 4000 | 10 | 40 |
| (8, 690) | 600 | 2000 | 8 | 35 |

选异构 pair: tp_A=4, f_A=690 + tp_F=8, f_F=690

| 计算方式 | 公式 | 结果 |
|---|---|---|
| 旧代码 (双重) | (10+40) + (8+35) = 93 mJ | **93 mJ** (错误) |
| 新代码 (正确) | E_A(4,690) + E_F(8,690) = 10 + 35 | **45 mJ** |
| 误差 | 93/45 | **2.07×** |
