# RQ5/RQ6 设计

## 研究矩阵
RQ5 固定 32 GPU，模型为 dense Qwen3-32B 与 MoE Qwen3-30B-A3B，架构为 native、PD、AF-only、PDAF。每个正式点展开 6 定长（128/128、2048/128、4096/128、128/1024、1024/1024、4096/1024）、conv、code 与 QPS 1/4/8。正式矩阵的硬上限为 QPS 16；任何 point 配置超过 16 都会在加载时触发 `ConfigError`。已有的高 QPS trace 文件可保留用于历史复现，但不得进入正式矩阵。EP 是附加实验点，`experimental=true`，不混入主结论。

RQ6 对 dense 模型在 1–4 节点上选择每架构一个资源守恒的典型点，观察强扩展；EP 仅作实验性附录。

## 架构不变量
- native：节点内 TP 组，可多副本。
- PD：P/D 服务及 PD router。
- AF-only：进程角色类型严格只有 A 和 F，命令不得出现 `--disaggregation-mode` 或 PD router；跨节点使用 AFD UCX。
- PDAF：PA/PF/DA/DF 四池并配 PD router。
- 每点 GPU 分区无重叠且总数等于节点 GPU 总数。

## 测量
请求按确定性泊松到达计划发送；每个流式 chunk 保存单调时钟时间戳并推导 ITL、TTFT、TPOT、E2E。报告 p50/p90/p95/p99。运行前后读取每 GPU NVML 累计能耗，差值聚合为 per-GPU、per-node、cluster，并派生 J/token 与 J/request。
