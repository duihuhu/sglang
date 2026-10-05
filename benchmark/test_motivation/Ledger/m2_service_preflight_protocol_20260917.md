# M2 服务机会观测：功能预检协议

2026-09-17，在首个GPU预检之前保存。依据 [F1观测规格](m2_tenant_service_f1_measurement_spec_20260917.md)。本轮只验证观测正确性与工作集可用性，不作motivation成立、hybrid因果或公平改善的PASS判断。

## 工作集与基线

- 六条件：A为packed/split，分别独跑、与B-packed、与B-split混跑。每角色每wave 16384输入+512输出；packed为1请求，split为16请求。两wave间隔1.5s，A先到，B延后20ms，同角色同wave同时到达。角色/顺序镜像留给正式协议。
- A payload、输出预算、到达不依赖邻居、观测开关或会话。B规范token序列相同，改变partition；不声称计算或语义相同。
- rep200只用于每条件完整同形态warmup；rep201用于功能预检。不是独立系统重复。不得因观察到收益选择性报告某个条件。
- Kimi-Linear-48B-A3B，TP2，Mixed4096，max-prefill16384，VTC input1/output2/request0，corrected slot gate，runtime cap607，请求表607，KV参数781725，统一池19481072000 B/rank。radix/graph/spec关闭，原生控制无额外delay或activation pacing。
- 仅选空闲GPU，绑定物理卡身份、文件锁和源文件hash，保存启动命令/实际readback/HBM预算/GPU健康状态。只停止本轮启动的进程组。

## 开关与验证

独立启动两个会话：observer-off，再observer-on。两侧保留旧resource/progress观测，仅新事件观测不同。新启动器默认off，生产调度文件不变。每条件先warmup再预检，共24次负载回放（12次warmup、12次预检）。

每会话另作P512/P8192各两次、O16贪心非流式输出正确性检查，保存实际输出token IDs及logprobs。仅这一独立正确性检查启用logprobs，六条件工作集不启用，避免改变Mixed batching条件。比较off/on相同输出及会话内重复一致性；不把它描述为全部混合工作集逐token一致性验证。

新观测记录queue/lift、VTC候选shadow order、实际candidate返回、scan-stop、admission收费、CPU batch launch/result、output收费，并接入现有资源事件；不更改返回值、排序或准入，不读取device tensor值。

primary scheduler采集，单次策略reset重建记录器；最多20000事件，每成员表最多2048请求，零drop/截断方可继续。导出boot ID、monotonic实现、绝对原点，客户端同容器运行。资源事件保留原始观察时刻；其释放原因明确标为未在allocator call处观测，不伪装成精确释放调用。

自动核查每请求input只收费一次、output增量守恒、chunk复用、prefill工作量守恒、batch身份一一对应、激活/释放各一次、counter变化能由收费和lift完全解释、完整时钟和fingerprint、50ms客户端提交迟到门槛、HTTP预连接、旧资源/GPU健康门槛。

`OTHER`/`NO_TOKEN`是方法返回值而非精确内部分支原因，候选已selected后停止扫描须单列。不能把未检查候选视为内存拒绝。新观测会引入CPU记录开销；off/on仅一次会话，首轮只能检查明显异常，不能得到无偏、稳定的开销估计。

## 本轮可以与不可以回答

可以回答：是否能连接计量、准入、实际执行、输出和资源；固定目标配对是否正确；双方实际waiting-backlogged区间是否存在、持续多久。

不可以回答：VTC违反公平保证、hybrid容量导致稳定隔离缺陷、新算法必要、统一池仍有可避免损失。尚需确定公平口径、冻结独立重复/顺序镜像/负载与窗口、量化观测开销，再做正式F1及F2介导对照。本轮不缩cap、不填padding、不进入lifetime predictor。
