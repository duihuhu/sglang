# 测试内容

# 默认参数

* 模型：Qwen3-30B-A3B  
* 数据集：

| Input\\Output | 64 | 256 | 1024 |
| :---: | :---: | :---: | :---: |
| 128 | QA(LPLD) |  | Chatbot(LPHD) |
| 512 |  | Balanced(MPMD) |  |
| 4096 | RAG(HPLD) |  | Summary(HPHD) |
| 16K |  | LongContext |  |

* 默认部署方式：

8GPU

| Native | TP8 |
| :---- | :---- |
| PD | P TP4 D TP4 |
| AF | A TP4 F TP4 |

# 不同数据集的架构能效探究

* 不同尺寸的分离能效探究

| Type\\Size | Small | Middle | Large |
| :---: | :---: | :---: | :---: |
| Dense | Llama3.1-8B | Qwen3-32B | Llama-3.3-70B-Instruct |
| MoE | Mixtral-8x7B | Qwen3-30B-A3B | Mixtral-8x22B |

RQ1：哪种数据集对哪种部署架构能耗效益较高？MoE or Dense，结论：后续采用MoE。

全矩阵：6模型 3架构 6数据集 QPS[2 4 8 16]，每点3次独立重复。

测试目录：[RQ1模型与数据集能效](../rq1_model_dataset_energy/README.md)

# 不同网络的分离能效探究

模型：Qwen3-30B-A3B  
GPU数：4
数据集：六种定长数据集  
QPS：4
部署方法：

|  | PCIe(1node) | RDMA(2node) | RDMA(4node) | NVLink(1node) |
| :---- | :---- | :---- | :---- | :---- |
| Native | TP2 | 2nodeTP2 | 4nodeTP4 | TP2 |
| PD | P TP1 D TP1 | P 1nodeTP1 D 1nodeTP1 | P 2nodeTP2 D 2nodeTP2 | P TP1 D TP1 |
| AF | A TP1 F TP1 | A 1nodeTP1 F 1nodeTP1 | A 2nodeTP2 F 2nodeTP2 | A TP1 F TP1 |

RQ2：哪种通信效率更有利于哪些分离架构？

全矩阵：3网络 3架构 6数据集

测试目录：[RQ2网络能效](../rq2_network_energy/README.md)

# 不同分离架构的并行部署范式探究

* DP部署对比

| Native | DP8 |  |  |
| :---- | :---- | :---- | :---- |
| PD | 1\* P DP4  D DP4 | 2\* P DP2  D DP2 | 4\* P DP1  D DP1 |
| AF | 1\* A DP4  F DP4 | 2\* A DP2  F DP2 | 4\* A DP1  F DP1  |

* EP部署对比

| Native | A DP8  F EP8 |  |  |
| :---- | :---- | :---- | :---- |
| PD | 1\* P(A DP4 F EP4) D(A DP4 F EP4) | 2\* P(A DP2 F EP2) D(A DP2 F EP2) | EP1同DP1 |
| AF | 1\* A DP4  F EP4 | 2\* A DP2 F EP2 | EP1同DP1 |

* TP部署对比

| Native | TP8 |  |  |
| :---- | :---- | :---- | :---- |
| PD | 1\* P TP4 D TP4 | 2\* P TP2 D TP2 | TP1同DP1 |
| AF | 1\* A TP4 F TP4 | 2\* A TP2 F TP2 | TP1同DP1 |

* 常见混合部署对比

| Native | 2\*TP4 | 4\*TP2 |  |
| :---- | :---- | :---- | :---- |
| PD | P 2\*TP2 D 2\*TP2 |  |  |
| AF | A 2\*TP2 F 2\*TP2 | A 2\*DP2 F 2\*EP2 |  |

RQ3：不同的并行部署范式如何影响不同的分离架构？

全矩阵：22部署 3架构 1数据集（Balanced(MPMD)）

# 不同计算显存配比的分离能效探究

A800/L40S  
RQ4：探究哪种计算显存配比对哪种分离架构能耗效益前景最高?

# 显存频率调节探索

RQ5：显存频率调节。5090/L20  


# 两节点 QPS2（总 2 GPU）安全恢复

该 runner 默认仅做本地 dry-run/inventory，不会 SSH。只有同时给出 `--resume --execute` 才允许远程执行；canary 与 formal 共用 `scheduler.lock`。当前资源释放暂停原因固定为 `user_requested_resource_release`，已有 valid logical key 不会重跑。

资源恢复后的顺序：

```bash
cd /mnt/nvme1/lt/Measurement_sglang/benchmark/AFlex_bench/measurement/rq2_network_energy
python3 scripts/run_measurement_rdma2n_qps2.py --offline-preflight
python3 scripts/run_measurement_rdma2n_qps2.py --preflight-only --execute
python3 scripts/run_measurement_rdma2n_qps2.py --check-gate
# 若 gate 因配置、runtime 或资源映射 hash 变化失效：
python3 scripts/validate_rdma2n_qps2_canaries.py --execute
python3 scripts/run_measurement_rdma2n_qps2.py --resume --execute --retry-blocked
python3 scripts/report_measurement_rdma2n_qps2.py
python3 scripts/merge_measurement_results.py \
  --rdma4n results/raw/measurement_rdma_qps4_20260825 \
  --rdma2n results/raw/measurement_rdma2n_qps2_20260825
```

在线 preflight 只读检查 node3/node4 的容器、GPU 0/2/4/6 空闲、拓扑、runtime 路径；不清理、不杀进程。报告不足 54/54 时只写 `measurement_rdma2n_interim.json`，不会覆盖最终 data JSON。PD 的 D 组件曾偶发退出，恢复后仍需通过实机 canary/health barrier 确认。
