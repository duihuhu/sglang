# RQ1 数据字典

负载使用固定 token 长度：QA 128/64、Chatbot 128/1024、Balanced 512/256、RAG 4096/64、Summary 4096/1024、LongContext 16384/256。每档 64 请求，Poisson 开环到达，QPS 为 2、4、8、16。

每次运行记录 deployment manifest、逐请求延迟/token、系统状态和计划占用 GPU 的 NVML 能量差。核心指标包括 achieved QPS、TTFT/TPOT/E2E p90、成功率、J/request、J/input token、J/output token、input/output tokens/J 和平均功率。只有三个独立 repeat 完整且满足同一 SLA 的组进入直接能效比较。

模型规模同时记录 family、总参数与激活参数；Small/Middle/Large 是部署档位，不表示 Dense 与 MoE 严格等参数。
