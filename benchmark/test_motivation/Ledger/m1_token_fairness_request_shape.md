# M1：相同 token 计量，不同 hybrid 资源代价与服务隔离

> 整理日期：2026-09-08。本文独立记录 M1 的三张图、实验口径、观察、问题与 insight。
> 范围：Kimi-Linear-48B-A3B，KDA + MLA，A800 GPU 6/7，TP=2，无 prefix reuse。
> 数据来自 2026-08-26 已有实验，本次未新增 GPU 实验。图中实测点为 3 runs 均值，当前紧凑汇总未提供 CI。
> [整体故事线](hybrid_state_aware_admission_motivation.md)仍保留 M1/M2/M3；本文是 M1 的独立整理版本，不修改历史实验 gates。

## 1. M1 要揭示的第一个问题

**当前基于累计 token 服务量进行公平决策的方法，没有充分刻画请求长短所对应的 hybrid 资源代价。在所测混合模型中，相同 token 预算的长短请求租户承受显著不同的混跑延迟膨胀，暴露出 token 公平计量与服务隔离之间的缺口。**

这里检查的是本地集成的 token-VTC 风格策略。输入、输出长度已经进入计量，因此不是说 VTC 完全不考虑长度；问题是累计 token 计量没有显式区分请求形态带来的固定 state 成本和资源构成。

完整论证顺序是：

1. 两租户具有相同的输入、输出 token 预算，按当前权重计量相同。
2. Hybrid 请求具有固定 recurrent state，短请求的单位 resident token 固定 state 摊销更高。
3. 实测中，两类租户的 state/KV 占用构成确实不同。
4. 相同请求形态时 slowdown 接近，拆成更多短请求后服务隔离明显不对称。
5. 因此，累计 token 数不能充分表征这里的资源压力和服务隔离表现；请求形态是需要独立观察的变量。

## 2. 实验设置与指标

每个 wave，两租户均提供 16,384 input tokens 和 512 output tokens。Packed 租户始终提供一个 `(16K input, 512 output)` 请求；split 租户提供 f 个 `(16K/f input, 512/f output)` 请求，f 取 1、4、16、32。

当前 token service 计量为：

\[
W_u = \sum_i(P_i+2O_i)=17,408\quad\text{per wave}.
\]

这表示完整服务这些请求后的累计 charge 相同，不表示任意时刻 counter 相同，也不表示已经证明 backlogged-window token fairness。请求粒度与记账时机仍可能改变实际 admission 顺序。

主要控制与定义：

- 每个租户 8 waves，base 扫描固定 wave interval=1.5s、state cap=48、KV cap=196,608。
- 每个请求使用唯一 cache salt，排除 prefix reuse；不等于关闭 radix 缓存。
- 每个配置和租户形态有匹配的 FCFS 单租户独跑基线。
- 图中 slowdown ratio 为 `mixed p95 JCT / isolated p95 JCT`，先逐 run 计算，再取均值。
- Slowdown gap 为逐 run 两租户 slowdown ratio 的绝对差再取均值，以百分点展示；不是两列均值之差。
- Equal tokens 控制的是工作量计数，不保证计算量、业务语义或资源需求等价。当前按独立 API 请求评价，不用 request p95 替代整个应用或 wave 的完成时间。

## 3. 图 1：计量相同，固定 state 成本和服务结果不同

![M1 开场：相同 token 预算、不同固定 state tax 与 slowdown](figures/m1_token_state_fairness.png)

### 实验与读图

图 1 采用 f16 对照：长请求侧为 `1 x (16K,512)`，短请求侧为 `16 x (1K,32)`。

- **面板 A，计量视角：** 两侧的 `input + 2*output` 均为 17,408。
- **面板 B，资源视角：** 固定 active-state 大小为 S 时，单位初始 resident token 的固定 state 项为 `S/L`。1K 请求是 16K 请求的 16 倍。
- **面板 C，服务视角：** f16-base 中，长请求侧 p95 slowdown ratio 为 1.21，短请求侧为 10.34。

**Observation 1：token 计量相同的请求集合，可以具有不同的单位 token 固定 state 负担，并呈现显著不同的混跑延迟膨胀。**

### 为什么短请求存在固定 state tax

对固定模型与 topology，活跃请求内存可近似写为：

\[
M_{active}(t)\approx S_{request}N_{active}(t)+K_{token}T_{resident}(t).
\]

固定 state 按请求承担，KV 随 resident tokens 增长。请求越短，固定项摊销到每个 resident token 上越高。并且 state 与 KV 是分别受限的池，不能只依据总字节之和判断压力。

**“短请求更贵”的准确含义是单位 token 固定 state 摊销更高。** 16 倍是面板 B 的解析值，不是本机实测的总内存倍率，不假设全部短请求同时准入，也不代表累计 state-time 更高：短请求可能释放得更快。

*图注：面板 A 是 offered token charge；B 是固定 state 项推导；C 是 3 runs 的匹配独跑 p95 比值均值。并列展示尚未证明 fixed state tax 单独导致 slowdown 差异。*

## 4. 图 2：实测确认 state/KV 资源构成不同

![M1 实测 state 与 KV 资源占用](figures/m1_measured_resource_pressure.png)

### 实验与读图

固定 f=16 和 KV cap=196,608，分别展示 base 与 state-relaxed 配置。同一配置内，两租户 offered tokens 相同。

| 配置 | 长请求 peak state slots | 短请求 peak state slots | 长请求 peak KV tokens | 短请求 peak KV tokens |
|---|---:|---:|---:|---:|
| Base，state cap=48 | 18 | 24 | 100,166 | 8,448 |
| State-relaxed，state cap=480 | 24 | 270 | 135,086 | 94,845 |

**Observation 2：相同 token 预算没有产生相同的资源使用者。短请求集合偏向请求级 state 占用，长请求集合偏向上下文 KV 占用。**

这为图 1 的资源差异提供实测支撑。一个累计 token 数掩盖了不同资源方向，但不能据此说短请求在所有资源上都更贵。

*图注：数值为各 run 峰值的均值；state/KV 峰值不一定同时发生。两面板分别按各自 pool cap 归一化，比例不能相加。State slots 按当前缓存/缓冲分配路径统计，不等于同一时刻的 active request 数。State cap 改变也会改变派生并发；此图展示资源构成，不将跨配置变化解释成仅容量改变的因果效应。*

## 5. 图 3：请求形态扫描显示服务隔离不对称

![M1 请求拆分形态与 slowdown、排队时间](figures/m1_partition_sensitivity.png)

### 实验与读图

Packed 侧固定不变，split factor 从 1 增加到 32；每侧 token 总量、base 容量和到达间隔保持不变。

| Split 形态 | 长请求侧 slowdown ratio | 拆分侧 slowdown ratio | 平均绝对 slowdown gap |
|---|---:|---:|---:|
| 1 x (16K,512)，相同形态对照 | 1.77 | 1.77 | 1.2 个百分点 |
| 4 x (4K,128) | 1.32 | 8.90 | 757.7 个百分点 |
| 16 x (1K,32) | 1.21 | 10.34 | 913.2 个百分点 |
| 32 x (512,16) | 1.23 | 10.47 | 924.6 个百分点 |

相同形态的 f1 是负对照：两租户 slowdown 接近。改变 partition 后，短请求侧相对独跑的延迟膨胀明显增加，长请求侧约为 1.2--1.3 倍。

右图补充实际排队时间：f16 短/长请求 queue p95 为 10.69/8.18s，f32 为 11.05/8.28s。请求变短之后，短请求仍承受约 11 秒的 p95 排队，没有随自身独跑成本成比例下降。

**Observation 3：在当前 token-VTC 和 serving 配置下，相同 token 总量的工作负载一旦被拆为更多短请求，就出现明显的请求形态相关服务隔离差异。**

*图注：P、O 和请求数一起变化，识别的是 partition/shape 效应，不是仅输入长度的单因素效应。Slowdown 的分母是各自匹配的独跑 p95；10.34/1.21 不是两租户绝对 JCT 之比。表中 gap 的聚合方式不同于直接相减两列均值。每点 3 runs，当前紧凑来源未提供 CI。*

## 6. 从三张图收束为问题与 insight

**完整逻辑：相同 token 计量，不同固定 state 摊销；实测资源构成不同，并且当前方法下的服务隔离显著依赖请求形态。**

**问题：当前 token-VTC 下，相同 token 总量的长短请求租户承受明显不对称的混跑延迟膨胀。仅按累计 token 计量，不能充分刻画请求长短对应的 hybrid 资源约束与服务代价。**

**Insight：request partition 是服务隔离相关变量，而不只是 token 总量的表示方式。** 固定 state 使请求数具有独立资源含义；因此，需要区分累计 token 服务、当前 state/KV 占用和混跑 slowdown，才能解释不同租户承担的服务代价。

这一步不预设新算法，不推导出应当惩罚短请求，也不以先证明某种向量成本优于所有 scalar cost 为成立条件。

## 7. 已验证与尚未验证

### 已有证据支持

- 固定 state 使单位 resident token 的资源代价依赖请求长度。
- 相同 token 预算的长短请求集合，在当前系统中呈现不同的实测 state/KV 占用构成。
- 本地 token-VTC 的服务隔离表现对请求形态敏感；相同形态对照与拆分扫描给出一致的问题方向。

### 不由这三张图推出

- 不能将资源差异与 slowdown 并列，直接视为固定 state 单独导致全部服务差异的证明。
- 不能将本地 VTC 风格实现的观察推广为所有公平策略失效，或声称 VTC 违反其 token-service 公平保证。
- 不能声称这些差异全部为 hybrid 独有；通用 batching、非线性 prefill 成本和缓存策略仍可能参与。
- 尚未证明某个新策略可以在维持整体效率与 token 服务约束的同时消除问题。

### 正式论文仍需补强

1. 补 run 级不确定性、更多负载点；同时报告绝对 JCT、worst-tenant slowdown、token-service gap 和整体效率，不只看 slowdown gap。
2. 固定 max-running，独立改变 state/KV cap；补固定总 token、不同请求数的直接 footprint 和可行并发曲线。
3. 加同模型 radix on/off 与合适的 KV-only 对照，区分缓存策略、通用请求形态效应与 hybrid 固定 state 的贡献。
4. 检查更多模型/topology 的适用范围。外部 serving figures 不替代本机测量。

## 8. 与 M2 的衔接

2026-09-17重新定位：M1回答“token计量遗漏了什么，以及当前系统出现什么服务差异”。M2接着问：**固定一个租户的请求和权重，邻居在相同token需求下的请求形态变化，会如何经过现有公平排序、准入与组批，影响该租户的服务机会与隔离？是否存在相同公平约束下可避免的损失？**

这里单位token资源成本不同，不等于业务价值不同，也不直接推导出应惩罚短请求租户。需分开记录公平counter、真实准入、执行进度、state/KV持有与交付服务；相同最终token或相同排序算法不等于过程公平相同。

因此M2不仅问“准入多少个”，还问“哪个租户被激活、哪个租户实际推进、谁持有的资源影响了谁的后续机会”。新增holder、当前瓶颈与释放时刻是解释这条跨租户链路的机制，不是独立替代公平问题的终点。早期双租户cap干预保留为局部证据，但需补过程公平；单租户mean JCT收益只作执行机制辅助。

当前入口为[M2：租户服务机会与隔离](m2_tenant_service_opportunity_isolation.md)。先验证邻居形态效应、定位传递路径，再检查相同公平口径下的可避免性；之后才探索如何更快完成请求。历史[Hybrid准入因素](m2_hybrid_admission_factors.md)、[双租户cap/batching](m2_admission_batch_formation.md)和[2026-08-28报告](results/admission_batch_20260828/admission_batch_motivation_report.md)保留原数据与判定，不视为新M2已经完成。

## 9. 数据与复现入口

- [原始汇总 analysis.json](results/vtc_state_tax_20260826/analysis.json)；[实验报告](results/vtc_state_tax_20260826/vtc_state_tax_observation_report.md)；[实验协议](vtc_state_tax_observation_protocol.md)。整套报告包含 387 artifacts、34,107 requests；并非全部用于这三张图。
- [三图数值与来源记录](figures/m1_evidence_series.json)，包含源文件 SHA-256 与指标解释。
- [统一绘图程序](plot_m1_evidence_series.py)；[数据校验测试](test_m1_evidence_series.py)。
- 向量图：[图 1](figures/m1_token_state_fairness.svg)、[图 2](figures/m1_measured_resource_pressure.svg)、[图 3](figures/m1_partition_sensitivity.svg)。图片复用同目录已有资产，不复制实验数据。

从仓库根目录复现：

```bash
python3 benchmark/test_motivation/Ledger/plot_m1_evidence_series.py \
  benchmark/test_motivation/Ledger/results/vtc_state_tax_20260826/analysis.json \
  benchmark/test_motivation/Ledger/figures
python3 -m unittest discover -s benchmark/test_motivation/Ledger \
  -p test_m1_evidence_series.py -v
```
