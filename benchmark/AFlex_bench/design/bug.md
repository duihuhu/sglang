错误 1：Profile 查询假设「同 tp、同 freq 绑定」
每次枚举调用的是：


tier1_solver.py
Lines 324-346
                m = self.pt.query_metrics(phase, tp, freq, bs, il, ol)
                t_layer = m.t_a_us + m.t_f_us + t_comm_us
                ...
                candidates.append(_PoolCandidate(
                    tp=tp, freq=freq,
                    t_a_us=m.t_a_us, t_f_us=m.t_f_us,
                    e_a_mj=m.e_a_mj, e_f_mj=m.e_f_mj,
                ))
query_metrics 一次返回 同一行 的 A 和 F：


profile_table.py
Lines 209-229
    def query_metrics(self, phase, tp, freq, bs, il, ol):
        # 索引 key = (tp, freq, bs, il) — 没有分开的 tp_A / tp_F
        la = self.predictor.predict_latency(phase, "A", tp, freq, ...)
        lf = self.predictor.predict_latency(phase, "F", tp, freq, ...)
        return LayerMetrics(t_a_us=la.value, t_f_us=lf.value, ...)
也就是说，每个候选隐含假设：Attention 和 FFN 跑在同一个 tp、同一个 freq 上。

op 参数（"A" 或 "F"）只用于内存检查，不影响取哪个算子的 latency/energy：


tier1_solver.py
Lines 317-320
            if op == "A" and not self._check_attn_memory(...):
                continue
            if op == "F" and not self._check_ffn_memory(tp):
                continue
            # 之后 op 再也没用过
所以 cand_pa 和 cand_pf 本质上是 同一批 (tp, freq) 组合，只是内存剪枝条件不同，并不是「PA 池候选」和「PF 池候选」。

错误 2：能耗被双重计算（最严重）
每个 _PoolCandidate 的能耗是：


tier1_solver.py
Lines 108-109
    def __post_init__(self):
        self.e_total_mj = self.e_a_mj + self.e_f_mj   # 一行 profile 里 A+F 加起来
选 PA 时最小化 e_total（= e_a + e_f），选 PF 时也最小化 e_total，最后加总：


tier1_solver.py
Lines 484-487
                e_total = (
                    k_p * (sel_pa.e_total_mj + sel_pf.e_total_mj + e_bubble_p)
                    + k_d * (sel_da.e_total_mj + sel_df.e_total_mj + e_bubble_d)
                )
数值例子
假设 profile 数据（bs=4, il=1024）：

(tp, freq)	t_A (us)	t_F (us)	E_A (mJ)	E_F (mJ)
(4, 690)
1000
4000
10
40
(8, 690)
600
2000
8
35
若最终选 hetero pair：tp_A=4, f_A=690 + tp_F=8, f_F=690

论文正确能耗（每层）：

E = E_A(4,690) + E_F(8,690) = 10 + 35 = 45 mJ
当前代码：

sel_pa 选 (4,690): e_total_pa = 10 + 40 = 50   ← 多加了 F 在 tp=4 下的能耗
sel_pf 选 (8,690): e_total_pf = 8 + 35 = 43    ← 多加了 A 在 tp=8 下的能耗
报告总能耗 = 50 + 43 = 93 mJ                    ← 几乎是正确值的 2 倍
PA 池实际只跑 Attention，PF 池只跑 FFN，但 Solver 把 两个算子在每一行的能耗都算进各自池里，再相加，等于 A 和 F 各被算了两次。

错误 3：SLO 剪枝用的是「假 pair」，和最终组合不一致
枚举 PA 候选时，SLO 检查是：


tier1_solver.py
Lines 328-332
                t_layer = m.t_a_us + m.t_f_us + t_comm_us
                if t_layer > slo_per_layer_us:
                    continue   # 剪掉
这里 t_A 和 t_F 来自 同一个 (tp, freq)。

但最终 pair 可能是 sel_pa(tp=4) + sel_pf(tp=8)，真实 SLO 应是：

t_A(4, f_pa) + t_F(8, f_pf) + t_comm
例子
配置	t_A	t_F	t_layer (M=1)
同行 (tp=4, f=690)
1000
4000
5000 + comm
异构 pair (tp_A=4, tp_F=8)
1000
2000
3000 + comm
枚举 PA 时，(tp=4) 行用 1000+4000 做 SLO 剪枝 → 可能 被剪掉
但若最终 PF 选 tp=8，真实 pair latency 只有 3000 → 其实可行
反过来也可能发生：两个池各自通过剪枝，拼起来却 违反 SLO。

错误 4：吞吐约束按池独立，不是 pipeline pair 吞吐

tier1_solver.py
Lines 457-468
                sel_pa = self._select_for_pool(cand_pa, k_p, lambda_prefill, alpha)
                sel_pf = self._select_for_pool(cand_pf, k_p, lambda_prefill, alpha)
_select_for_pool 要求 每个池独立 满足：


tier1_solver.py
Lines 543-546
            if k * c.throughput_per_pair >= (1 + alpha) * demand:
                if best is None or c.e_total_mj < best.e_total_mj:
                    best = c
而 throughput_per_pair 来自 该候选自己的 t_layer_us（仍是同行 A+F）：


tier1_solver.py
Lines 339-339
                thpt = self._pair_throughput(bs, t_layer, self.num_layers, phase)
论文里 pair 吞吐应由 pipeline 瓶颈 决定（慢的那边），不是 PA、PF 各自 declare 自己够快。

例子
PA 选了一个极快配置：单独看 k × thpt_pa ≥ λ ✓
PF 选了一个极慢配置：单独看也刚好 k × thpt_pf ≥ λ ✓
但拼成 pipeline 后，瓶颈在 PF → 实际吞吐远低于 λ
错误 5：Bubble energy 部分对、但整体仍建立在错误选择上

tier1_solver.py
Lines 519-519
        t_wait_us = abs(cand_a.t_a_us - cand_f.t_f_us)
这里确实用了 sel_pa.t_a 和 sel_pf.t_f，hetero TP 下 这一项本身方向是对的。

但问题是：

sel_pa / sel_pf 是按 错误能耗目标（e_a+e_f 双重计数）选出来的，不是按 bubble 或 pair SLO 联合优化的
A/F balance 剪枝在枚举时仍用 同行 的 |t_a - t_f|，对最终 hetero pair 无效
bubble 加在「已双重计数的能耗」上，目标函数整体仍偏
错误 6：hetero TP 只能「事后拼接」，无法作为候选原生存在
论文决策空间是：

(tp_A, tp_F, f_A, f_F)  四个自由度独立组合
例如: tp_A=8, tp_F=2, f_A=690, f_F=450
当前枚举空间是：

for tp in [1,2,4,8]:
    for freq in [210,...,1410]:
        候选 = (tp, freq) 同时约束 A 和 F
hetero TP 只能靠 Phase 3 碰巧 拼出 sel_pa.tp ≠ sel_pf.tp，但：

枚举时从未以 (tp_A=8, tp_F=2) 为单元做 SLO/吞吐/能耗评估
profile 虽支持 predict_latency(phase, "A", tp=8, ...) 和 predict_latency(phase, "F", tp=2, ...) 分开查，Solver 没用上
总结对照
维度	论文（AF Pair 联合）	当前代码（四池独立）	后果
优化单元
(tp_A, tp_F, f_A, f_F) 一个 pair
四个池各选 (tp, freq) 再拼接
hetero TP 无法正确搜索
能耗
E_A(tp_A,f_A) + E_F(tp_F,f_F)
(E_A+E_F)_pa + (E_A+E_F)_pf
约 2 倍高估
SLO 剪枝
t_A(tp_A,f_A) + t_F(tp_F,f_F) + comm
同行 t_A(tp,f) + t_F(tp,f) + comm
误剪/漏剪可行解
吞吐
pair 级 μ(tp_A,tp_F,...)
PA、PF 各自满足 λ
虚高估计容量
A/F balance 剪枝
pair 级 `
t_A - t_F
`
Bubble
`
t_A(tp_A,f_A) - t_F(tp_F,f_F)