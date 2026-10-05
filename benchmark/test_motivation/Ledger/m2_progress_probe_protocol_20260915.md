# M2：自然容量下的激活、执行进度与后续准入

2026-09-15，数据采集前冻结。本轮是定位实验，不预设新算法，也不把旧 slot 修复收益计入新机会。

## 1. 可证伪问题

在无 prefix、无 retraction、单模型固定 state 的条件下，独立 state 池的数量约束可以退化为并发上限。仅发现 state 满或排队不足以形成新的 hybrid 调度问题。

本轮检查：正确 slot 门限下，是否自然出现请求已占用 state、但在其他请求组批期间持续未被选择，同时还有请求等待准入？这种时序观察是否在 mixed batching 下已被消除？它只是候选机制；要称可避免损失，还需要同一请求集、相同目标和预算下的干预见证。

不以 client TTFT 前的全部时间当成无效驻留，因为其中包含真实 prefill。新增 opt-in 进度日志记录每个已提交 batch 的请求 ID、prefill/decode、实际批级 Python 列表中的 work/prefix tokens。不能从 Req 可变计数反推 overlap 下的过去工作量，不读取 CUDA tensor，不引入 GPU 同步。CPU launch 间隔及 result 记录时间不是 kernel 时间。

## 2. 本轮阶段

1. `inspect`：自然池启动，读取实际 state/KV/request/cap，回放少量功能请求，验证日志与资源回收。不人工指定 state、KV、max-running，也不把 runtime cap 改为 8。
2. `observe`：Separate-4K 与 Mixed-4K；三个输入形态：256x512、64x2048、16x8192，均总 input=131072，output=128/请求；另 mixed 为 128x512 + 8x8192，总 input=131072。再增加一个自然压力点：请求数为 `min(1024, ceil(1.25 * inspect_state_slots))`，每请求 P512/O128。数量由 inspect 容量文件固定，各模式、重复一致，不按性能选择。所有请求在一个有限 burst 中到达，组内间隔 1ms。每配置两个独立服务启动，rep 相同的输入、输出、到达配对，预热排除。若自然容量大于最大负载且未出现资源等待，明确报告未形成 state 压力，不临时把池缩为 8。
3. `intervene`：只有获得可定位的等待/执行事件，才单独冻结后续矩阵。先用校准的原生 batching/active-prefill 控制；不自动重新实施固定 per-pass 限额，不将原生 mixed 称为完整 Sarathi/FairBatching/APC。

observe 共 2 modes x 5 shapes x 2 starts = 20 runs，测量请求总数为 `1888 + 4 * pressure_count`。每个会话用各形态的 1/8 数量预热（每组向上取整），全部排除。形态之间 output 总量不同，因此它们只用于资源/瓶颈结构比较；性能比较只能在同一形态、同一 rep 内进行。两次启动是探索性，不作确认性 CI、SLO goodput 或公平保证改善。

## 3. 固定条件与安全

Kimi-Linear-48B-A3B、TP2，仅物理 GPU 6/7；mem-fraction-static=0.85；radix off、CUDA graphs off、spec off；max-prefill-tokens=16384、chunk=4096。共同 token-VTC input=1/output=2/request=0，其余成本项为 0；corrected slot gate=1，额外 activation limit=0。可选延迟和 prefill-size 控制默认关闭，不改变原服务策略。

自然容量指这些明确部署选项下自动解析的容量，不代表所有默认生产配置。统一池本轮不启用；后续若比较应核验模型支持、总字节预算、调度预算适配及运行开销，不能仅增加资源。

启动前检查 GPU 与端口，不停止其他任务、不调整时钟/功率。只有本轮启动的进程组可以清理。每 run 校验请求完整性、输出长度、prefix/retraction=0、健康读数、日志无丢失、容量 readback 一致、结束 active=0、一个 active 对应一个 state。实际提交迟到超过 50ms 标为无效；超时或校验失败原样保留，不静默纳入。

## 4. 指标与解释边界

- 每请求完整 prefill work 之和应等于输入长度；记录 decode 参与次数与输出数的差异，识别 overlap 下已提交但未交付的工作，不把它称为公平服务。
- 激活、首次提交工作、首次输出、最终释放的时间；首轮工作占输入比例；prefill 分块数；两次 batch 参与之间的间隔。
- 活跃 holder 在相邻 CPU launch 区间没有被该批选择的 slot-time 代理量，以及期间待准入请求数。它不是 GPU 空转/浪费量；overlap、网络和正常时间复用都可能影响指标。
- state/KV/request 约束实际触发、峰值与驻留积分、全部及分组 JCT/TTFT/TPOT、完成时间。没有选中比例高不自动代表存在可避免损失。
- 暂不宣称精确双方 backlogged-window 公平性。候选干预仍需补全相应队列事件和预先指定的服务代价约束。

## 5. 判读

若仅 Separate 出现未参与区间，而 Mixed 在相同请求集与容量下消除并改善服务，应先归为已有 mixed batching 可处理的通用干扰；不能直接成为 hybrid 新贡献。若两者均有损失但普通 matched-cap 控制复现，应说明 state 的作用是容量中介。只有证明自然 state/内存耦合改变了可行选择或放大损失，并在修正、调优后的已有方法上仍有剩余机会，才升级 M2。
