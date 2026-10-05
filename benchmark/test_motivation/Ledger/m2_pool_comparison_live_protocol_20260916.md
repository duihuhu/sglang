# M2 池管理对照：GPU 实测增补协议

用户已授权使用任意空闲卡，替代原协议仅GPU6/7的限制。本轮选择GPU0/1，两张A800-SXM4-80GB，同NUMA、NV8互联；每次启动前复核两张卡连续两次低显存、零利用率，并持有逐卡实验锁。不停止已有作业、不改频率或功率、不把旧6/7延迟直接配对。

## 阶段与预先约定

1. Inspect：按原静态、统一池旧cap、重划静态、统一池新cap顺序，先用Mixed4K。新cap固定607，旧cap以本轮自然启动读回为准。启动参数/backend/dtype/context与原协议一致。逐条件做512与8192输入、16输出的greedy生成，各自重复两次并跨池比较token IDs；不一致就停止，保留logprobs调查，不放宽正确性门槛。随后各做607请求完整预热和一次容量回放，仅用于诊断、不纳入正式延迟统计。
2. 内存审计：独立launcher仅在idle `/get_server_info` 时，对两个TP rank分别记录Torch allocated/reserved/peak、CUDA used/free、Python可见GPU tensor storage去重账本、请求表形状与池属性。所有未归因分配保留为residual。每次审计后reset peaks；容量回放结束后读取峰值。该hook不运行于任何batch/allocator/kernel路径，正式性能进程完全不加载该hook。
3. Budget gate：以A的实际完整驻留预算和峰值安全余量为参考，按字节计算B/C所需减少的KV，不看诊断延迟挑配置。重新启动检查配置、输出与预算；若预算/正确性不通过，不进入正式实验。
4. Measure：在预算确认后写入独立frozen-plan JSON，记录具体参数、容差、顺序、重复数和代码哈希，再运行Separate4K、Mixed4K配对对照。每个条件完整607请求预热排除；固定P512/O128、1ms间隔，v3 HTTP预建/GC隔离/0.5s GPU监测，延迟误差门槛50ms不放宽。

原脚本仅监测6/7，本轮新增独立可选卡monitor，不修改旧冻结测量脚本。payload fingerprint来自同一原生spec generator，忽略run_id，包含group、input_ids、output_len和arrival；正式样本不与inspect/warmup混合。

主目标mean JCT，同时报告queue、p95、TTFT/TPOT、makespan、state-time、peak active及拒绝原因。只有mean改善且p95/makespan均不劣于5%的对照才称本工作集的有约束服务收益；反方向的目标权衡单独报告。统一池容量分母不能当成两个可同时填满的硬池。

这批同形短请求不验证多租户公平或模型普适性。若统一池无法实际启动/正确推理，报告配置兼容性阻碍，不把失败当作调度motivation证明。若原静态池之外的已有机制已修复容量与主要服务损失，收窄M2，不强行提出新算法。
