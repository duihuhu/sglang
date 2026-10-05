# M2：state归因前的原生控制排除实验

2026-09-17。Kimi-Linear-48B-A3B，TP2，GPU4/5。状态：4个独立服务会话、12组正式6912请求及等量完整预热全部完成；45项CPU回归通过。本轮一次完整执行，没有失败重试或样本排除。

**结论：State-near的约54% mean JCT改善可以由纯运行cap512或原生空槽阈值96实现，不需要新的state记账或lifetime预测。两者使新请求TTFT增加约19%；State-far的mean改善虽也超过53%，但makespan增加约6.4%、新请求TTFT增加约62%，未通过原5% makespan约束。当前验证的是已有控制可处理的阶段相关服务权衡，不是state容量额外放大的证据。**

接续[32组统一池准入筛选](m2_unified_admission_complete_20260917.md)，按[本轮冻结协议](m2_native_counterexample_protocol_20260917.md)执行。没有改动旧测量代码、模型算子、allocator或服务端源码；新脚本只使用已有启动/运行参数与旧观测wrapper。

## 1. 先纠正实验的可识别性

当前无prefix/spec/retraction时，一个active请求对应一个state holder，逐事件完整性检查核对`active_requests == state_slots`。因此：

\[
N_{state}(t)=N_{active}(t),\qquad
N_{state}(t)\le L\iff N_{active}(t)\le L.
\]

**只添加一个逻辑state-slot限制，与添加相同的普通请求上限等价。** 如果仍只比较这两种限制，不能通过给干预换名字来证明新的hybrid机制。该等价只针对本配置，不推广到prefix/checkpoint、多份state或不同state规格的情况。

源码另有两项耦合：`KVCacheConfigurator.resolve_max_num_reqs`按启动state数裁剪实际请求上限；`init_unified_mamba_pools`用启动state与KV的字节和构造共享raw池。直接改变`max-mamba-cache-size`还可能同时改变池大小与请求表，不能称为独立state容量消融。

统一池在`PrefillAdder`中已经用`mamba_slot_full_token_cost`预留新增state的共享字节，并检查`rem_mamba_slots`。因此本topic不能把现有问题写成“系统没有检查state”。近似容量关系是：

\[
S N_{active}(t)+K T_{resident}(t)\le B,
\quad N_{active}(t)\le C.
\]

固定S不使请求数维度消失；但单独观察N上限的后果，也不自动区分固定state压力与普通并发控制。

## 2. 本轮检验什么

保持同一共享raw预算19481072000 bytes/rank、启动state/request607、KV参数781725、Separate4K及相同token-VTC，重新比较Native607、Slots96和Cap512。

初始512个背景对应607-512=95空槽。Slots96可以在新候选首次激活前等待；Cap512则仅依靠普通运行请求上限限制新增请求。若它们得到接近历史诊断动作的改善，说明此前State-near的大幅收益可以由已有机制产生，尚不能作为新资源抽象的必要性证据。

本轮只是State-far/near校准，rep61/62为新的配对；不混入旧rep51/52，不把新种子称为held-out工作负载。Native607与Cap512在同一个服务会话内交错，另一轮反转；Slots96使用独立会话。每个正式条件都有完整同形预热。

## 3. 性能与机制结果

预注册主目标为mean JCT下降，整体p95 JCT与makespan各退化不超过5%；同时披露候选JCT/TTFT、state-time、触发和准入阻塞原因。两次独立会话不支持精确总体CI。

### 3.1 整体收益与新请求代价

下表单位为秒，均为两次run的均值，p95使用nearest-rank。

| 场景 | 策略 | Mean JCT | P95 JCT | Makespan | 新请求mean TTFT |
|---|---|---:|---:|---:|---:|
| State-far | Native607 | 38.510 | 39.998 | 40.107 | 11.904 |
| State-far | Cap512 | 18.004 | 33.984 | 42.652 | 19.331 |
| State-far | Slots96 | 18.000 | 34.008 | 42.676 | 19.330 |
| State-near | Native607 | 38.210 | 39.731 | 41.380 | 11.896 |
| State-near | Cap512 | 17.430 | 28.864 | 42.687 | 14.174 |
| State-near | Slots96 | 17.424 | 28.847 | 42.670 | 14.173 |

![相同统一池下原生控制的收益与候选TTFT代价](figures/m2_native_counterexample.png)

柱为两次会话均值，点为rep61/62，不是置信区间。

| 相对Native607，逐rep百分比均值 | Mean JCT改善 | P95改善 | Makespan增加 | 新请求mean TTFT增加 | 两次均通过原约束 |
|---|---:|---:|---:|---:|---|
| Far / Cap512 | 53.25% | 15.04% | 6.35% | 62.38% | 否 |
| Far / Slots96 | 53.26% | 14.98% | 6.41% | 62.37% | 否 |
| Near / Cap512 | 54.38% | 27.35% | 3.16% | 19.15% | 是 |
| Near / Slots96 | 54.40% | 27.39% | 3.12% | 19.14% | 是 |

Near中Cap512的mean改善逐次为54.43%/54.34%，Slots96为54.49%/54.30%。两者达到此前诊断Decode32的收益量级，但旧诊断属于rep51/52，本轮为rep61/62，不把跨轮近似数值写成严格性能等效或统计无差异。

Near中新请求mean JCT由27.555s增加至28.862/28.845s，约增加4.74%/4.68%；新请求p95 TTFT由22.203s增加至24.419/24.417s。Far中新请求mean JCT由28.109s增加至33.982/34.005s，约增加20.89%/20.98%。这些代价不能被整体平均数掩盖：512个已有请求占576请求的88.9%，整体mean对它们的改善更敏感。本轮为单租户实验，不是公平收益。

**需要更新旧观察：不是“只有near能改善mean JCT”，而是near在本轮同时满足整体p95/makespan约束，far则付出更大的新请求等待和整批完成代价。** 旧Decode32的far负结果仍成立；那是最多32次延迟扫描的特定动作，不是所有等待策略的上界。

### 3.2 收益由什么执行变化产生

| 执行证据 | Native607 | Cap512 | Slots96 |
|---|---|---|---|
| 两场景、两轮peak active | 576 | 512 | 512 |
| 已有请求全部释放前候选prefill批数 | 每run 128 | 每run 0 | 每run 0 |
| 上述候选prefill token数 | 每run 524288 | 0 | 0 |
| Near最长无已有decode调度间隔 | 23.11--23.13s | 0.25s左右 | 0.25s左右 |
| Near到达后已有请求全部释放 | 25.89--25.96s | 2.37--2.45s | 2.35--2.46s |
| Far到达后已有请求全部释放 | 31.30--31.44s | 7.50--7.65s | 7.52--7.60s |
| Request-limit记录 | 0 | 每run 1 | 0 |
| 原生空槽延迟次数 | 0 | 0 | Near 34/35，Far 113/113 |

Cap512触发的是普通runtime请求上限；Slots96触发的是已有空槽阈值。两种方式下，所有512个背景均在第一个候选激活前释放。所有正式run无prefill-NO_TOKEN，不能把Cap512人为产生的request-limit记录改名为state字节不足。拒绝事件次数也不是等待时长，scheduler置batch-is-full后不需要每轮重复记录。

Near到达时三策略每rep的median remaining完全相同：rep61为34，rep62为35。Far为113，只有Cap512/rep62为114；均通过冻结场景检查。上述时间是CPU观察的调度/释放事件，不是GPU kernel计时。

Near的state-time从19418.06 slot-seconds降至Cap512的7311.43、Slots96的7310.78，约减少62.35%；Far约减少62.7%。池raw预算没有变化，这是逻辑占用时间下降，不是CUDA驻留显存下降相同比例。

## 4. 对hybrid因果验证的含义

本轮不是独立state干预。它提供了“只有新的state感知方法才能取得前述收益”的实际反例：已有通用cap和空槽阈值已经取得相同量级的受约束改善。参数96/512使用了已知512背景规模，因此只能说明校准能力，尚未证明可泛化的自动选择，更不是所有原生方法的最优性能。

这也不证明固定state无关。模型确实存在固定state占用，执行推迟确实延长其持有；本轮没有改变这项成本，因此不能估计它对性能的边际因果贡献。已有控制覆盖收益与state是否因果放大是两个不同判断，不能混为一个PASS/FAIL。

当前M2建议保留的问题是：**同一资源可行域下，立即加入长prefill会延迟已有请求完成及state释放，通用并发/空槽控制能避免大量损失，但等待的代价取决于执行阶段和服务目标。** 该观察成立；“现有控制无法处理”及“本正例必须用新hybrid机制”不再作为结论。

真正的后续证据应识别共享预算内固定state占用的边际影响，而不仅是人为降低N上限：匹配计算与请求、核对独立cap及KV需求，记录state继续持有、空间不足、后续请求等待、释放后准入的链路。只减小整个共享池识别的是总内存压力；改变state dtype还需要数值正确性和宽容量kernel负对照。人为padding或扣账只能算人工压力实验，不能冒充自然部署缺陷。

## 5. 完整性、边界与下一步

12/12正式、6912/6912请求和12组完整预热通过检查，预热不进入统计；回放零新建连接，正式最大提交迟到1.32ms，结束active归零。源码/协议/输入计划哈希未变，没有修改时钟、功率或其他任务。运行器与自己的模型服务已退出，结束快照GPU4/5各1MiB、利用率0。

4个服务会话各有每rank一次custom-all-reduce回退警告，共8条；首会话有2条可恢复CUDA分配重试，其余0。不把完整性通过写成无分配压力。正式高负载SM1365--1410MHz，最高66摄氏度、NVML峰值81176MiB；固定raw预算并不等于不同batch的运行峰值显存严格相等。45项CPU回归通过，另有17条工具链/无GPU测试后端警告；PNG/PDF已生成并目视核对。

本轮没有测试KV场景、Mixed、其他背景规模/长度、生产graphs、稳态SLO或多租户。两次重复不足以报告总体CI；独立会话和共享宿主机仍有时间漂移风险。不能把当前阈值当成通用最优值，也不能据far的约束失败直接证明自适应或predictor必需。

下一步应冻结已校准的原生基线，再检查Mixed和未参与调参的背景规模/到达形态是否还有明确的受约束缺口；只有出现该缺口，才继续寻找与固定state相关的容量介导路径。没有缺口时应保留负结果，不以改变压力定义制造hybrid特有性。

本地汇总：[完整分析与逐run执行审计](results/m2_native_counterexample_20260917/analysis01/analysis.json)、[配对表](results/m2_native_counterexample_20260917/analysis01/report.md)、[冻结manifest](results/m2_native_counterexample_20260917/measure01/manifest.json)、[完成标记](results/m2_native_counterexample_20260917/measure01/completed.json)。大型原始request/batch/resource/GPU轨迹和服务日志在远端`/workspace/sglang/benchmark/test_motivation/Ledger/results/m2_native_counterexample_20260917/measure01`；本地不是完整原始数据镜像，汇总保留每个artifact的SHA256。
