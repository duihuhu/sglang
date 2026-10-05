# M2 F1：从token计量到租户服务的最小补测规格

2026-09-17。根据[60组旧数据审计](m2_tenant_service_audit_20260917.md)确定必须补采的证据。**这是观测与对照规格，尚不是冻结的GPU性能协议；本轮未执行新GPU实验。** 正式样本数、到达负载、服务窗口和代价阈值在观测正确性预检后冻结，不从正式结果反向选取。

## 1. 单一问题与理论前提

已知前提来自M1：单位token的固定state摊销随请求长度变化。F1不重新证明这一公式，而检查：**不同成本的token服务经过现有公平计量、排序和执行之后，租户实际获得的服务机会如何变化？**

固定形态的成本差异不等于公平价格；相同token不保证计算量相同；运行resource-time也不是外生计费成本。F1先识别跨租户形态效应，F2才区分state/KV、计量粒度和通用执行干扰，F3再验证同公平约束下可避免性。

## 2. 最小工作集与配对要求

以原M1每wave的16384 input + 512 output为起点。A有两个单独固定的目标形态：packed为1个P16384/O512；split为16个P1024/O32。对每个A形态比较A独跑、A+B-packed、A+B-split，形成六种基本条件；B两种形态的每wave输入、输出总量和权重相同。

必须建立以下独立fingerprint，而不只是整批请求的一个hash：

- A的input IDs、输出长度、到达偏移不依赖B的partition、策略、服务会话或租户标签。独跑使用混跑中A的原样子集。
- B从每wave同一份规范token序列组织为packed/split；这控制输入token集合，不声称拆分后attention计算或语义相同。输出长度按同一预算拆分。
- 逻辑请求角色与API租户标签分离，交换标签不重新生成payload。
- A先/B先作为独立的预设条件，配对的B-packed/B-split使用相同A到达。不能把相同rep同时换payload与先后顺序后，当作纯到达顺序消融。

六种基本条件只是功能预检起点，不等于六个样本足以确认motivation。正式阶段需要独立重复、顺序平衡和角色镜像；不能把8个wave或大量请求当作独立系统重复。

## 3. 基线与资源预算

保持当前模型TP2、无prefix复用，使用corrected slot gate、经审计的请求表/有效cap及正确池配置。本地token-VTC固定input1/output2/request0，租户等权；先不更换公平成本，也不引入按租户惩罚或新调度规则。

初始观测采用固定原生batching配置，完整记录Mixed/Separate、chunk、prefill budget、graph、radix、通信backend和GPU健康状态。新配置不得与旧radix-on/额外state缓冲数据直接配对；如需radix-off，作为新的共同基线冻结。所有请求总量、实际活跃资源与真实拒绝都要记录，不预设必须达到state wall。

采用统一池时须记录raw字节、请求表、实际有效请求上限及完整HBM/workspace预算。若没有空间拒绝，仍可观察服务转换，但不能宣称其由state容量介导。若后来需要压力干预，另立F2协议，不在F1中静默缩池、缩cap或修改state padding制造“hybrid因果”。

## 4. 必须补充的观测

所有服务端事件共享monotonic原点和递增序号，携带run ID、rank及决策/批次标识；显式导出客户端起点与服务端原点的对应，只有同主机同monotonic时钟可直接相减，否则需独立校准误差。不能通过延迟曲线拟合补造时钟对齐。

| 事件 | 最小字段 | 用途 |
|---|---|---|
| queue enter / lift | rid、tenant、时间、入队前后waiting数量、counter前后、lift理由 | 区分历史计量、空闲返回校正与新增需求 |
| sort decision | 决策ID、各租户counter、waiting/running数量、候选顺序、shadow charge | 说明排序依据，不以最终counter倒推 |
| candidate attempt | 决策ID、rid、首次激活/复用、检查过的预算、通过/拒绝原因、是否提前终止扫描 | 区分不可行和策略选择；未被考察不等于被内存拒绝 |
| admission / charge | 新增rid、实际新增slot、input记账增量、剩余队列 | 区分新增holder与继续chunk，区分收费与执行 |
| batch launch / result | 批次ID、每rid的prefill/decode及实际提交工作量、结果更新 | 连接实际处理与输出，不能仅用tenant_counts分摊extend tokens |
| output charge | rid、旧/新输出计数、收费增量、counter前后 | 核验decode记账与实际进度 |
| resource change / finish | rid/tenant、state/KV主体、可用空间、释放理由 | 连接一方持有与另一方后续获准 |

CPU launch/result事件不是GPU kernel时间。优先复用已有resource/progress trace，但VTC目前只有汇总，需新增默认关闭、只读的事件观测。不得为观测更改排序或容量判定，也不增加CUDA同步；容量上界、trace drop及观测开销须单独验证。

## 5. 开始性能统计之前的预检门槛

1. 三类fingerprint测试：目标A在所有邻居条件完全一致、标签交换不改payload、B总token预算一致；实际到达误差符合另行冻结的阈值。
2. 计量守恒：每请求input只记一次，output增量与真实结果对应；lift单列。继续chunk不重复新增state或input charge，运行结束资源归零。
3. 时间与轨迹完整性：事件序号和时钟有序、零丢失、客户端流与rid完整对应；缺字段不以汇总代替。
4. 观测开/关功能对照：相同生成输出和配置，检验额外观测没有改变请求语义；独立评估记录开销，不将带profiler结果混进普通性能统计。
5. 积压条件检查：分别识别双方waiting-backlogged和只有running未完成的区间。先在预检确定可重复的窗口规则；没有合适窗口就不做VTC过程公平结论。

功能预检不按收益选最佳配置。失败只修测量或调整独立预检协议；不将其混进后续确认样本。

## 6. 预期交付，不预设正结果

每个正式条件至少交付每租户counter/实际处理/交付曲线、准入与资源时间线、绝对TTFT/TPOT/JCT、匹配独跑slowdown及整体效率。公平服务使用预先声明的同一口径，减去窗口起点，披露请求粒度与不可抢占区间；等输出、等请求数和等资源占用不是预设公平目标。

若邻居partition没有稳定影响，如实保留负结果；若变化只来自记账粒度或通用计算干扰，就缩小hybrid归因；若原生控制已在共同公平约束下修复，就不主张新算法必需。

本规格完成后的执行顺序是：**补只读观测和测试 -> 六条件功能预检 -> 冻结正式F1矩阵与公平窗口 -> 跨租户验证 -> 必要的F2/F3干预 -> 后续效率优化。** 当前不进入lifetime predictor，也不恢复单租户JCT扫描作为M2主验证。
