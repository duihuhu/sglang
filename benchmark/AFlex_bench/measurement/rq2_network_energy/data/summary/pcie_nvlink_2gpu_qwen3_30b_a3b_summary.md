# Qwen3-30B-A3B 单机双卡 PCIe/NVLink 测试总结

## 测试范围

- 架构：Native、PD、AF。
- 通信：NVLink 与强制 host-staged PCIe 对照。
- 工作负载：QA、Chatbot、Balanced、RAG、Summary、LongContext。
- 每组 64 个请求、Poisson QPS=2、3 次重复。
- 完整性：108/108 个运行完成；每个运行均为 64/64 请求成功。

## 核心结论

1. NVLink 下，PD 在 Chatbot、Balanced 上取得最高 QPS，并在 QA、Chatbot、Balanced、RAG 上能耗最低；AF 在 QA、RAG、Summary 上取得最高 QPS，在 Summary、LongContext 上能耗最低；Native 在 LongContext 上吞吐最高。
2. Host-staged PCIe 下，PD 在 QA、Chatbot、Balanced、RAG 上同时取得最高 QPS和最低能耗；Summary 与 LongContext 则由 Native 同时取得最高 QPS和最低能耗。
3. AF 对通信路径最敏感。PCIe 相比 NVLink，AF 的 RAG、Summary、LongContext QPS 分别显著下降；LongContext 的平均 QPS 从约 0.546 降至 0.052，平均总能量从约 45.9 kJ 增至 176.2 kJ。
4. PD 对链路变化相对稳健，但在 PCIe 下 Summary 的 QPS 下降约 26.2%；Native 的 QA、Chatbot、Balanced、RAG 吞吐变化较小，而 LongContext 在 PCIe 下下降约 39.3%。
5. PCIe 是在 NVLink 机器上通过禁用直连并强制 SHM/TCP/ZMQ host staging 构造的通信对照，不代表独立 PCIe-only 服务器。

## 按架构的跨链路变化


### NATIVE

- QA: PCIe 相比 NVLink，QPS -0.1%，总能量 -10.1%。
- Chatbot: PCIe 相比 NVLink，QPS +0.1%，总能量 -11.5%。
- Balanced: PCIe 相比 NVLink，QPS +0.0%，总能量 -7.2%。
- RAG: PCIe 相比 NVLink，QPS -0.6%，总能量 +2.3%。
- Summary: PCIe 相比 NVLink，QPS -3.8%，总能量 -11.1%。
- LongContext: PCIe 相比 NVLink，QPS -39.3%，总能量 +16.9%。

### PD

- QA: PCIe 相比 NVLink，QPS +0.1%，总能量 -0.9%。
- Chatbot: PCIe 相比 NVLink，QPS -4.1%，总能量 +1.4%。
- Balanced: PCIe 相比 NVLink，QPS -0.5%，总能量 +4.1%。
- RAG: PCIe 相比 NVLink，QPS -0.0%，总能量 +2.0%。
- Summary: PCIe 相比 NVLink，QPS -26.2%，总能量 +25.2%。
- LongContext: PCIe 相比 NVLink，QPS -5.4%，总能量 +6.9%。

### AF

- QA: PCIe 相比 NVLink，QPS -3.5%，总能量 -3.1%。
- Chatbot: PCIe 相比 NVLink，QPS -29.3%，总能量 +24.6%。
- Balanced: PCIe 相比 NVLink，QPS -20.0%，总能量 -2.1%。
- RAG: PCIe 相比 NVLink，QPS -88.1%，总能量 +291.9%。
- Summary: PCIe 相比 NVLink，QPS -67.9%，总能量 +102.4%。
- LongContext: PCIe 相比 NVLink，QPS -90.6%，总能量 +284.0%。

## 数据质量与边界

- 所有 108 个正式运行均通过请求完整性检查。
- NVLink 的 54 个运行均通过物理链路验证。
- PCIe Native 的 18 个运行均通过物理验证；PCIe PD/AF 中部分运行使用语义通信账本验证 host staging，其余通过物理验证。
- 106 个运行使用 GPU0/1；PCIe AF 第一轮 QA 与 Chatbot 使用 GPU6/7。两对 GPU 型号一致且均为 NV8，但 NUMA 域不同。
- AF-PCIe LongContext 的极低吞吐和高能耗在三次完整重复中均出现，应作为结果保留，而不是判作失败。

## 输出文件

- `nvlink_2gpu_qwen3_30b_a3b.json`：NVLink 54 个运行的聚合结果与原始引用。
- `pcie_2gpu_qwen3_30b_a3b.json`：host-staged PCIe 54 个运行的聚合结果与原始引用。
