### 2025-06-07

今日进展：
1. 修正 PD 分离架构的 Tier 调频：发现 PD DP2 的 Tier DVFS 不生效（频率锁定 1410MHz）。修复后 PD DP2 Tier 节能从 0.6% 提升至 22-26%，3-way 对比图已更新。
2. 整理实验文档。
3. Azure 真实 trace 测试（进行中）：分析 Azure Code（IL 均值~1200）和 Conv（IL~100）两个数据集，提取 light/medium/heavy workload。PDAF light 节能 21%。但 code trace 长 prefill 导致 TTFT 超 SLO 是架构容量问题；Conv medium/heavy 因 KV-cache OOM crash，需进一步排查。

### 2025-06-08

今日进展：
1. 8 卡 PDAF Tier 节能效果不佳（仅~3%），诊断发现根因：decode coupled model（`decode_pipeline_v1.txt`）仅有 TP=1 数据，而 8 卡 PDAF decode 阶段运行 TP=2。模型不含 tp 特征，无法区分 TP 配置差异（TP=2 实际比 TP=1 慢 1.3-1.6x），导致 DVFS 频率决策不准。
2. 补充采集 TP=2 profile 数据：编写 `bench_decode_pipeline_tp2.py`，8 卡布局（每组件 TP=2），覆盖 freq×5、IL=[128-4096]、BS=[1-256]、M=[1,2]，共 1864 个配置，耗时约 9 小时完成。
3. 合并 TP=1+TP=2 数据（3623 行）重训 GBDT 模型，feature 加入 tp 维度，MAPE<1%。更新 predictor/controller 接口传入 tp 参数。
4. 新模型重跑 8 卡 PDAF Tier（进行中）。

### 2025-06-09

今日进展：
1. 完成 8 卡三方案（Native DP8、PD DP4、PDAF）Baseline + Tier 全量测试。从 Azure Code/Conv 原始 trace 中筛选出 5 组满足所有方案 Tier SLO 违背 <1% 的数据集（code_D、conv_A/C/D/E）。
2. 分析原始 Azure trace 特征：生成 CDF 图和时间轴图。结论：Code trace 自然存在低 QPS 窗口可直接截取；Conv trace QPS 整体过高，需等比例降采样至可服务范围。

### 2025-06-10

今日进展：
1. 差异化数据集测试：从 Azure trace 中按请求特征（IL/OL 范围）构造 7 组差异化 workload（prefill-heavy、balanced、lightweight、mid-decode、high-IL+mid-OL、low-IL+mid-OL、short+highQPS），覆盖不同负载模式。6 组满足三方案 Tier SLO<1%。PDAF+Tier 相比 Native+Tier 节能 26%-70%，相比 PD+Tier 节能 14%-28%。
2. QPS 扫描测试：固定短请求特征（IL≤256, OL≤50），QPS 从 10 逐步加到 30，三方案均满足 SLO<1%。PDAF+Tier 在 QPS=10 时节能最显著（-69% vs Native Tier），随 QPS 增大能耗差距缩小但始终优于 Native/PD。
