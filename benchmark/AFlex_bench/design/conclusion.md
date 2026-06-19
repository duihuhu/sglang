六项错误全部成立
错误1- Profile查询绑定同tp、同freg:query metrics只接受一个tp参数，同时传给A和F的 predictor。op参数仅用于内存检查，之后不再使用。cand pa和 candpf本质是同一批候选。
错误2-能耗双重计算(最严重):_PoolCandidate.e_total_mjef(同一行)，然后 e_total=selpa.e_toteat+ sel pf.o total。PA跑Attention 却被算了 FFN 能耗，PF 反之。你的数值例子成立:报告 93mJ vs 正确值45mJ，几乎翻倍
错误3-SLo剪枝用假pair:枚举PA候选时，t1layer=tA(tpf)+tF(tpf)来自同一行。最终PA(tp=4)+PF(tp=8)的真实 SLo可能可行，但枚举时被同一行的错误t1ayer 剪掉了。
错误4-吞吐约束按池独立:每个池用自己的 throughput per pair(从同行tAttF算出串行吞吐)独立通过检查，没有跨池联合验证 pipeline瓶颈吞吐。
错误5-Bubble 计算本身正确但基础错误:abs(cand a.ta us-cand f.tf us)确实用了异构 pair 的真实值，但selpa/selpf的选择目标(双重计数的能量)已经错了。
错误6-Hetero TP只是事后拼接:搜索空间是for tp in [1,2,4,8] x for freg in [... 1,一个tp同时约束A和F。(tp A-8,tp F=2)从未作为单元被枚举、评估 SLO/吞吐/能耗，只能靠 Phase 3 独立选择碰巧拼出。
根源:代码是"四个独立池"的模型，每个候选携带完整A+F指标;论文是"一个AFpair"联合优化的模型。需要把枚举、剪枝、能耗计算全部改为 pair 级别