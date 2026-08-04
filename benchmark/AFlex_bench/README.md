# AFlex Bench 实验导航与复现手册

> 本文以当前仓库中的脚本、数据和子目录说明为准，目标是帮助读者区分“正式论文数据”“可直接重绘的数据”“需占用集群重跑的实验”和“历史/探索性遗留”。命令默认从 SGLang 仓库根目录执行；凡依赖四节点、模型、RDMA 或特权容器的命令，均应先完成运行前检查，不能视为可移植的一键复现。

## 1. 快速导航

AFlex Bench 顶层有四个职责不同的目录：

| 目录 | 用途 | 正式入口 | 主要输入 | 主要输出 | 依赖关系 |
|---|---|---|---|---|---|
| [`comm/`](comm/) | NVLink、Mooncake RDMA、原生 RDMA 与 GPU/NIC affinity 通信基准 | [`comm/run_comm_bench.sh`](comm/run_comm_bench.sh)、[`comm/read.md`](comm/read.md) | 两节点 GPU、IB/RoCE NIC、Mooncake | `comm/results/*.json`，绘图脚本生成的图 | 独立于 E2E，但为 AFD/PD 通信假设提供依据 |
| [`common/`](common/) | 跨实验共享的数据口径工具 | [`common/energy_per_token.py`](common/energy_per_token.py) | macro workload JSONL 与结果条目 | 按 input+output token 归一化后的 energy/token | 被正式 macro/绘图逻辑引用；不要用旧 output-only 口径替代 |
| [`energy_model/`](energy_model/) | Qwen3-32B、Mixtral-8x7B 的 A/F latency/energy profile、训练与评估 | Qwen3 的 [`retrain_models_v1.sh`](energy_model/Qwen3-32B/retrain_models_v1.sh)、[`eval_predictor_fast.py`](energy_model/Qwen3-32B/eval_predictor_fast.py)；Mixtral 的 profile 脚本 | V1 layer profile、V2 pipeline profile | 序列化 predictor（若本地生成）、CV/MAPE 报告 | AFlex solver、DVFS、ILP ablation 和 motivation 依赖 profile/模型 |
| [`multi_node/`](multi_node/) | 四节点环境、PD/PDAF smoke、正式 macro/micro、ablation、motivation | smoke 位于 [`multi_node/scripts/`](multi_node/scripts/)；正式 E2E 位于 `more_test/{macro,micro}` | workload、模型、energy model、4×8 GPU 集群 | canonical packed JSON、partial/raw JSON、图表 | 同时依赖 `common/`、`energy_model/`，跨节点传输依赖 Mooncake/RDMA |

入口可靠性建议分为四级：

- **A级：正式/可信入口**：当前 canonical 数据明确由其读取或生成，参数和依赖相对集中，如 macro `scripts/run_benchmark.py`、micro baseline runner、各图的 plot 脚本。
- **B级：可重绘入口**：已有冻结数据，脚本可生成图，但不一定能重新采集原始测量，如 Reshard_cost。
- **C级：环境绑定 runner**：逻辑有效，但硬编码节点、容器、模型、端口或设备，运行前必须核对，如 MoE、node scalability。
- **D级：历史/失效入口**：引用已删除文件、`/tmp` 脚本或旧路径；只能作为 provenance 线索，不能直接照抄执行。

## 2. 推荐工作流

1. **先审计环境，不启动服务**：确认节点、容器挂载、GPU/NIC、模型和 predictor 文件。
2. **先做通信与 smoke**：单 pair 用 `smoke_pdaf_pair.sh`；要覆盖六种节点组合，用 `run_smoke_pairs_live.sh`。
3. **只重绘已有数据**：先运行 macro/micro、motivation、ablation 的绘图入口，确认 Python 依赖和数据 schema。
4. **再跑正式 baseline**：从单个 scheme、dataset、QPS 开始；确认 partial JSON、日志和能量计数范围后再扩大 sweep。
5. **最后跑 AFlex/并行任务**：AFlex 会读 layout/predictor，并进行跨节点部署与清理；四节点 `parallel` 会同时占用两个 16-GPU pair。
6. **打包而非手工拼 JSON**：使用正式 `pack`/迁移/构建脚本；保留 source、meta、energy denominator 和 patch 记录。
7. **提交前只核验 artifact**：不要默认 PDF/PNG 已提交；检查生成物、数据来源、单位和 git diff。

## 3. 多节点环境与拓扑

### 3.1 四节点资产

当前脚本采用以下映射，每节点 8 GPU，物理集群总计 32 GPU：

| 名称 | IP | 典型角色 |
|---|---|---|
| node1 | `10.252.129.36` | 操作机；Group A 第一个节点 |
| node2 | `10.252.129.35` | Group A 第二个节点 |
| node3 | `10.252.129.34` | Group B 第一个节点 |
| node4 | `10.252.129.33` | Group B 第二个节点 |

容器名固定为 `operator_test`。环境文档记录宿主机 `/mnt/workspace/lt` 挂载到容器 `/workspace`，因此仓库常见路径为：

- 宿主机：`/mnt/workspace/lt/sglang`
- 容器内：`/workspace/sglang`
- 模型挂载：宿主机模型盘到容器 `/models`
- Qwen3：`/models/Qwen3-32B/`
- Mixtral：`/models/Mixtral-8x7B/`

环境说明见 [`multi_node/env.md`](multi_node/env.md)。现有多个 profile 与 Fig.5 数据明确标注 A800，`env.md` 也写“4 nodes × 8 A800”；但 [`multi_node/test.md`](multi_node/test.md) 写“A100-80G ×8/node”，形成仓库内真实存在的 **A800/A100 文档冲突**。因此现有证据整体指向 A800，但复现时仍必须保存每节点 `nvidia-smi -L`、GPU 型号、驱动与时钟表，并以运行时采集确认实际设备。

### 3.2 32 GPU 集群与单次 16 GPU pair

“4 节点 × 8 GPU”描述的是可用集群容量，不代表每个 E2E 点使用 32 GPU。正式 macro/micro 常规单次实验占用两个节点，即一个 **16 GPU pair**。`parallel` 模式把 32 GPU 分为两组并发：

- Group A：`.36 + .35`
- Group B：`.34 + .33`

node scalability 的四节点点才明确使用最多 32 GPU。smoke 的 `smoke_pdaf_pair.sh` 同样是任意两个节点组成的 16-card pair；`run_smoke_pairs_live.sh` 顺序遍历四节点的 6 个无序 pair，并非同时运行 6 个实验。

### 3.3 Mooncake、RDMA 与 affinity

- PD 的 Prefill→Decode KV 传输使用 Mooncake，跨节点经 RDMA/RoCE；默认设备常为 `mlx5_bond_0`。
- 同节点 A/F（FFN↔Attention）通信在 smoke 中使用 `ipc_cpp`、CUDA IPC/Event；它与跨节点 Mooncake 是两条不同数据路径。
- 多数正式脚本记录 GPU/NIC affinity：GPU `0,1→mlx5_0`，`2,3→mlx5_1`，`4,5→mlx5_4`，`6,7→mlx5_5`。这反映当前机器拓扑，不应无验证迁移到其他集群。
- `comm/run_comm_bench.sh` 的 4-NIC 集合为 `mlx5_0,mlx5_1,mlx5_4,mlx5_5`，单 NIC 默认 `mlx5_bond_0`。
- 运行前确认 Mooncake 在所有容器内可导入、NIC 可见、GID/路由一致、GPU Direct/RDMA 权限正常。

### 3.4 SSH、memlock 与容器要求

- 编排脚本依赖从操作机到四节点的免交互 SSH；smoke 使用 `BatchMode=yes`，部分脚本关闭 strict host-key check。
- 容器需 host network、host IPC、全部 GPU；环境记录为 privileged。不要在共享环境中未经授权复刻这些高权限设置。
- RDMA 大消息和服务启动依赖足够 memlock。smoke 用 `prlimit --memlock=unlimited:unlimited`；通信文档指出容器默认 64 KB 会限制大消息，需要两端核对 `ulimit -l`。
- 确认宿主机与容器路径内容一致。历史记录显示远端代码、`.pyc`、挂载不同步会造成参数或 mixin 不一致。

## 4. PD/PDAF smoke：只验证连通性

顶层 [`multi_node/scripts/`](multi_node/scripts/) 中的 smoke 用于验证服务启动、AFD 节点内通信、PD 跨节点传输、router 和一次生成请求。**它们不是性能或能耗实验**：关闭 CUDA graph、跳过正式 workload，只有单个短请求，不能据此报告吞吐、SLO 或 energy/token。

推荐入口：

```bash
# 从仓库根执行；先挑一对空闲节点
PREFILL_IP=10.252.129.36 DECODE_IP=10.252.129.35 \
  bash benchmark/AFlex_bench/multi_node/scripts/smoke_pdaf_pair.sh mlx5_bond_0 4

# 覆盖四节点全部 6 个 pair；会频繁清理四节点
bash benchmark/AFlex_bench/multi_node/scripts/run_smoke_pairs_live.sh mlx5_bond_0 4
```

`smoke_pdaf_pair.sh` 的关键事实：

- 默认 TP=4，8 卡映射为 GPU `0..3` 承载 FFN（PF/DF），GPU `4..7` 承载 Attention（PA/DA）。
- 启动顺序是 **PF → PA → DF → DA → Router**。FFN 先启动是为了让对应 Attention peer 接入；不要随意调换。
- 主要 HTTP 端口：Router `42000`、PA `42010`、PF `42011`、DA `42020`、DF `42021`；Mooncake bootstrap `49999`。另有 AFD base/scheduler 端口 `28200/28300`、`68400/68500`。
- PF/DF 用 `/get_model_info`，PA/DA/Router 用 `/health` 轮询；日志中若出现端口占用、Traceback、fatal error 或 OOM 会提前失败。
- 日志容器路径为 `/workspace/sglang/benchmark/AFlex_bench/multi_node/logs`，宿主机对应 `/mnt/workspace/lt/sglang/benchmark/AFlex_bench/multi_node/logs`。
- 成功标准仅为 Router `/generate` 返回含 `text` 的响应。

清理具有侵入性：pair 脚本在成功或失败退出时清理两个节点；`run_smoke_pairs_live.sh` 还调用 `cleanup_all_nodes.sh`，会对四节点 `pkill -9` SGLang scheduler/router，并杀占用一组端口的进程。共享集群上必须先确认没有他人任务。旧的 `smoke_pdaf_node23.sh`、`node34.sh`、`xnode.sh` 是固定拓扑入口，优先使用可参数化 pair 脚本。

## 5. 正式 E2E：六方案定义与论文口径

历史六方案语义如下：

| 方案 | 仓库 scheme（常见） | 架构与频率策略 |
|---|---|---|
| SGLang | `native_tp1_baseline` / MoE `native_dp_baseline` | Native DP；固定最高频率 |
| DynamoLLM | `native_tp1_tier` / `native_dp_tier` | 与 Native DP 同拓扑；统一 DVFS |
| DistServe | `pd_hetero_baseline` / `pd_dp_baseline` | Prefill/Decode 分离；固定最高频率 |
| BiScale | `pd_hetero_tier_biscale` / `pd_dp_tier` | PD 分离；P/D 级 DVFS |
| MegaScale | `pdaf_baseline` | A/F 分离，固定频率；历史 solver 常要求用满预算 |
| AFlex | `aflex_tier1` / `pdaf_tier` | A/F 分离 + component DVFS + layout/弹性选择 |

注意：当前正式 dense macro/micro runner 和论文图通常只保留 **SGLang、DynamoLLM、DistServe、BiScale、AFlex 五系列**，不含 MegaScale。macro `bench_common.py` 的正式 scheme 表就是五项，`plot_e2e_dashboard.py` 明确排除 MegaScale。六方案脚本及数据仍有历史价值，但不能把 MegaScale 自动加入当前 canonical 图。

### 5.1 Macro：Code / Conversation

正式配置：Code 与 Conversation（键名 `code`、`conv`），QPS `2/4/8/16`。workload 位于 [`multi_node/more_test/macro/data/workloads/`](multi_node/more_test/macro/data/workloads/)，正式 canonical packed 数据是 [`macro_e2e_all.json`](multi_node/more_test/macro/data/macro_e2e_all.json)。AFlex 布局来源是 [`plan_dense_e2e.json`](multi_node/more_test/macro/data/plan_dense_e2e.json)，应逐点查看其中 `config`、`tier1_layout` 和 GPU 数，避免引用旧 `test.md` 表格覆盖当前计划。

正式入口：[`multi_node/more_test/macro/scripts/run_benchmark.py`](multi_node/more_test/macro/scripts/run_benchmark.py)。

```bash
cd benchmark/AFlex_bench/multi_node/more_test/macro/scripts

# 当前节点 pair 上运行；建议先用 --schemes/--datasets/--qps 缩小范围
python3 run_benchmark.py run --schemes native_tp1_baseline --datasets code --qps 2

# 两组节点并行：A 跑 SGLang/DynamoLLM/DistServe；B 跑 BiScale/AFlex
python3 run_benchmark.py parallel

# 将指定 partial/final 结果打包为 canonical schema（先确认 --input）
python3 run_benchmark.py pack --input ../data/<明确的结果文件>.json

# 从 canonical 或显式输入重绘
python3 run_benchmark.py plot
```

不要把尖括号占位命令直接运行。`parallel` 默认 Group A `.36+.35`、Group B `.34+.33`，两组会同时部署并在结束后合并。运行过程持续写 partial；`pack` 用于 schema 归一和重算元数据，不等于验证所有点均 PASS。

### 5.2 Micro：四种长度形态

| 数据集 | input_len | output_len | 含义 | 正式 QPS |
|---|---:|---:|---|---|
| `qa_lpld` | 128 | 64 | Low Prompt / Low Decode | 2/4/8/16 |
| `chatbot_lphd` | 128 | 1024 | Low Prompt / High Decode | 2/4/8/16 |
| `rag_hpld` | 4096 | 64 | High Prompt / Low Decode | 2/4/8/16 |
| `summary_hphd` | 4096 | 1024 | High Prompt / High Decode | 2/4/8/16 |

正式 baseline runner 是 [`multi_node/more_test/micro/scripts/run_benchmark.py`](multi_node/more_test/micro/scripts/run_benchmark.py)，它直接采集四个 baseline；AFlex 数据不是由该 runner 现场部署，而是通过 [`migrate_aflex.py`](multi_node/more_test/micro/scripts/migrate_aflex.py) 从 legacy `micro_4ds_e2e.json` 迁移，并在迁移时重算 percentile 与 input+output energy/token。正式 canonical 数据是四个 per-dataset JSON：

- [`micro_e2e_qa_lpld.json`](multi_node/more_test/micro/data/micro_e2e_qa_lpld.json)
- [`micro_e2e_chatbot_lphd.json`](multi_node/more_test/micro/data/micro_e2e_chatbot_lphd.json)
- [`micro_e2e_rag_hpld.json`](multi_node/more_test/micro/data/micro_e2e_rag_hpld.json)
- [`micro_e2e_summary_hphd.json`](multi_node/more_test/micro/data/micro_e2e_summary_hphd.json)

`micro_4ds_e2e.json` 是 legacy 聚合源，不是当前首选引用。canonical 文件还可能记录 `other_test` 的能量修正或 TP2 patch；这些 patch 是 provenance 的一部分，不应静默删除。

```bash
cd benchmark/AFlex_bench/multi_node/more_test/micro/scripts
python3 migrate_aflex.py
python3 run_benchmark.py run --datasets qa_lpld --qps 2 --schemes native_tp1_baseline
python3 run_benchmark.py parallel
python3 run_benchmark.py plot
```

并行 micro 的分工是 Group A 跑 `qa_lpld,chatbot_lphd`，Group B 跑 `rag_hpld,summary_hphd`，且每组跑四个 baseline；合并时会保留已迁移 AFlex。

## 6. 数据与 SLO 口径

### 6.1 Energy/token

正式 energy/token 分母为 **input tokens + output tokens**。共享实现见 [`common/energy_per_token.py`](common/energy_per_token.py)：优先用 `total_energy_j * 1000 / total_all_tokens`；若只有旧 `energy_per_token_mj` 和 output token 数，则按比例转换。

历史数据存在两类修补：

- **output-only → all-token**：旧结果只按 completion token 归一，packed/migrate 时改为 input+output。
- **active-GPU energy scope**：部分 AFlex 重跑只统计实际分配 GPU，随后由 `micro/other_test` patch 回 canonical 文件。比较前必须确认各方案的能量边界一致；不能把“少统计空闲 GPU”误当作系统节能。

### 6.2 SLO 不全目录统一

必须区分：

1. 正式 macro/micro runner 用 TTFT 5000 ms、TPOT 300 ms 记录 SLO violation；`PASS` 仅表示请求成功且没有 timeout/missing，不要求请求满足这些 SLO。
2. 绘图中的横向参考线，可能只是论文展示阈值，不一定参与 PASS 判定。
3. MoE README 声称 TTFT 2000 ms、TPOT 150 ms；旧 dense 文档还出现 baseline 2000/100 与 AFlex 5000/300 的混合口径。

因此报告每张图时应注明具体脚本/JSON 的阈值，不要声称整个 `AFlex_bench` 统一采用一个 SLO。

### 6.3 数据层级

- **canonical packed**：macro `macro_e2e_all.json`、micro 四个 per-dataset JSON、各 ablation 明确的汇总 JSON。
- **partial/raw**：运行中断点、逐请求记录、solver raw、日志；可用于审计，但不应直接混画。
- **other_test patch**：针对能量范围或异常拓扑的定向重跑/修补，需保留 patch metadata。
- **历史备份**：`.bak`、带时间戳 JSON、旧聚合文件，仅作 provenance。
- **预期生成图**：plot 脚本可能输出 PDF/PNG，但当前 checkout 未必提交该文件。文档只说明“运行后生成”，不声称 PDF 已在仓库。

## 7. Ablation 导航

### 7.1 Energy_breakdown

- **研究问题**：AFlex 节能来自拓扑/调度还是 DVFS？三系列为 Vanilla AFD（JSON 键 `megascale`）、AFlex w/o DVFS、完整 AFlex。
- **可信入口**：[`Energy_breakdown/scripts/run_all.py`](multi_node/more_test/Ablation/Energy_breakdown/scripts/run_all.py)，或分别运行 `run_vanilla_1p1d_tp4_no_dvfs.py`、`run_aflex_e2e_no_dvfs.py`，再 `merge_tier_perf_data.py` 和 [`plot_tier_perf.py`](multi_node/more_test/Ablation/Energy_breakdown/charts/plot_tier_perf.py)。
- **输入/输出**：Code/Conv × QPS 2/4/8/16；canonical [`tier_perf_energy_all.json`](multi_node/more_test/Ablation/Energy_breakdown/data/tier_perf_energy_all.json)；图由 plot 脚本生成。
- **状态**：已有汇总数据，可重绘；重跑需要两节点集群并会强制清理。
- **陷阱**：顶层 [`Ablation/run_breakdown.py`](multi_node/more_test/Ablation/run_breakdown.py) 声明的 `run_megascale_fixed_1p1d_tp4.py`、`run_aflex_tier1_no_dvfs.py`、`prepare_existing_data.py` 当前目录中缺失。因此其多个 `run/prepare` target 是失效入口；以子目录 README 和现存 runner 为准。

### 7.2 Dynamic_micro_batch

- **研究问题**：低/高负载下 M=1 与多 micro-batch 的流水化收益，以及不等分 batch 的价值。
- **可信入口**：采集 [`scripts/run_breakdown.py`](multi_node/more_test/Ablation/Dynamic_micro_batch/scripts/run_breakdown.py)，解析 `parse_breakdown.py`，冻结图 [`plot_design_aligned_motivation.py`](multi_node/more_test/Ablation/Dynamic_micro_batch/charts/plot_design_aligned_motivation.py)。
- **输入/输出**：Qwen3-32B、单节点 4 GPU（物理 GPU 4–7）、breakdown 日志/JSON/CSV；冻结图读 [`design_aligned_cases.json`](multi_node/more_test/Ablation/Dynamic_micro_batch/data/design_aligned_cases.json)，batch 曲线来自 `decode_bs_curve_*`。
- **状态**：冻结数据可重绘；现场采集为环境绑定 C 级入口。
- **陷阱**：README 写“low/high M=1/M=2”，但 runner 默认 cases 是 `m1-low,m1-high,m3-low,m3-high`，即高负载默认 M=3；若要复现 M=2，必须显式 `--cases m1-low m2-low m1-high m2-high` 并在报告中写明。脚本默认 Mooncake、`mlx5_4`，但四角色同节点使用 IPC；不要把该实验误称跨节点吞吐测试。

### 7.3 ILP_solver_cost

- **研究问题**：单次 solver 调用中，各求解阶段的延迟构成。
- **可信入口**：[`benchmark_solver_avg_breakdown.py`](multi_node/more_test/Ablation/ILP_solver_cost/scripts/benchmark_solver_avg_breakdown.py)，绘图 [`plot_solver_avg_breakdown.py`](multi_node/more_test/Ablation/ILP_solver_cost/charts/plot_solver_avg_breakdown.py)。
- **输入/输出**：Qwen3 profile + 可加载 energy model；180 次 raw 运行写 [`solver_avg_breakdown_raw.json`](multi_node/more_test/Ablation/ILP_solver_cost/data/solver_avg_breakdown_raw.json)，汇总 6 行 CSV。
- **状态**：已有 raw/summary，可重绘；重跑前确认 predictor 序列化文件存在。
- **陷阱**：图堆叠 Decode enumeration、Prefill enumeration、Pareto、Global search 和 Other；Global search 字段包含其内部逻辑，Other 是总耗时扣除这些显式阶段后的余量。该图展示平均 solver latency，不计算相对 5 分钟调度窗口的占比；不要将各项改写成另一套 Search/Overhead 定义。

### 7.4 Reshard_cost

- **研究问题**：A/F runtime reconfiguration 的 Expand/Shrink activate 开销及其组件分解。
- **可信入口**：仅推荐用 [`plot_runtime_reconfiguration_timeline_activate.py`](multi_node/more_test/Ablation/Reshard_cost/charts/plot_runtime_reconfiguration_timeline_activate.py) 重绘；`summarize_runtime_reconfiguration_breakdown.py` 可从 full sequence 重算 paper breakdown。
- **输入/输出**：`run_{1,2,3}_{full_sequence,paper_breakdown}.json`、`three_run_summary.json`；输出 timeline 图。
- **状态**：**只能可信重绘现有 AFlex 数据，不能从本目录完整重跑或复现 baseline 对比**。
- **陷阱**：三份 paper breakdown 是同一次 node2 测量的相同副本，并非三次独立实验；传统 baseline 的原始数据已缺失，论文 baseline 数字不存于此目录。图只展示 ACTIVATE，PUBLISH 为异步背景阶段。

### 7.5 model_scalibility

> 保留历史拼写 `model_scalibility`，不要重命名目录。

- **研究问题**：AFlex 在 Mixtral-8x7B MoE 上相对五类方案的能耗表现。
- **可信入口**：当前最接近主 runner 的是 [`scripts/run_moe_retest.py`](multi_node/more_test/Ablation/model_scalibility/scripts/run_moe_retest.py)；绘图使用 `charts/plot_moe_energy.py` 或 `plot_moe_energy_clustered.py`。当前 `plot_moe_energy.py` 无论是否传入 `--paper` 都排除 MegaScale。
- **输入/输出**：Mixtral 模型、macro Code/Conversation workload、Mixtral V1 predictor；汇总入口数据为 [`plan_moe_e2e.json`](multi_node/more_test/Ablation/model_scalibility/data/plan_moe_e2e.json)。
- **状态**：有汇总数据与 runner，但重跑高度环境绑定，结果需重新审计。
- **陷阱**：README 的复现命令引用 `/tmp/retest_baselines_n34.py` 和 `/tmp/test_aflex_2p1d.py`，当前仓库不可用；README 节点列表使用 `.31/.32/.34/.33`，与正式四节点 `.36/.35/.34/.33` 冲突；其 MoE SLO 2000/150 与 dense runner 5000/300 不同；“4节点×8卡”与单点实际 GPU 分配也不能混为一谈。以运行脚本和结果 meta 为准。

### 7.6 node_scalibility

> 保留历史拼写 `node_scalibility`，不要重命名目录。

- **研究问题**：系统从 1→2→4 节点扩展时的 energy/token 趋势。
- **可信入口**：采集包括 `run_1node_8gpu_parallel.py`、两节点正式 macro、[`run_4node_scalability_benchmark.py`](multi_node/more_test/Ablation/node_scalibility/scripts/run_4node_scalability_benchmark.py) 与若干定向补跑；汇总使用 [`build_node_scalability_all.py`](multi_node/more_test/Ablation/node_scalibility/scripts/build_node_scalability_all.py)，绘图使用 `charts/plot_node_scalability_energy.py`。
- **输入/输出**：Code/Conv；最终 canonical [`node_scalability_all.json`](multi_node/more_test/Ablation/node_scalibility/data/node_scalability_all.json)。
- **状态**：汇总 JSON 可绘图；完整重采集不是单一 runner 的一次调用。
- **陷阱**：节点数增加时 QPS 同时从 8→16→32，图不是固定负载的纯 strong-scaling；最终 JSON 从 1/2/4 节点的多个文件拼装。构建脚本明确 source mapping，需先确认源文件齐全。四节点 runner 跳过 MegaScale，最终图是五系列。

### 7.7 Model_accuracy

- **研究问题**：模型/预测器准确率如何呈现。
- **入口与产物**：目录仅含 [`paper.tex`](multi_node/more_test/Ablation/Model_accuracy/paper.tex)。
- **状态**：这是论文 artifact，不是可执行 benchmark；预测器实际评估可参考 Qwen3 `eval_predictor_fast.py` 和 macro `measure_aflex_predictor_accuracy.py`。
- **陷阱**：TeX 表值不能当作本目录可重跑产物，必须追溯其数据生成路径和模型版本。

### 7.8 Table

- **用途**：论文表格汇总 artifact。
- **入口与产物**：目录仅含 [`all.tex`](multi_node/more_test/Ablation/Table/all.tex)。
- **状态/陷阱**：不可执行，不提供原始采集 provenance；引用前应与 canonical JSON 和论文版本逐项核对。

## 8. Motivation Fig.1–Fig.5

[`multi_node/motivation/data`](multi_node/motivation/data) 是指向 `energy_model/Qwen3-32B/data` 的符号链接，profile 不是重复副本。统一重绘命令如下；只承诺脚本会尝试生成对应 PDF/PNG，不声称这些图当前已提交：

```bash
ROOT=benchmark/AFlex_bench/multi_node/motivation
python3 "$ROOT/fig1_heter_homo/plot_fig1_heter_vs_homo.py"
python3 "$ROOT/fig2_latency_freq/plot_fig2_3panel.py"
python3 "$ROOT/fig3_energy_heatmap/plot_fig3_decode_lat_vs_energy.py"
python3 "$ROOT/fig4_AF_ratio/plot_fig4_combined_4panel.py"
python3 "$ROOT/fig5_reshard/plot_fig5_combined_2panel.py"
```

### Fig.1：Homo / Hetero A/F split

- **数据源**：Qwen3 V1 `prefill_data_v1.txt`、`decode_data_v1.txt`。
- **指标/筛选**：在 SLO budget 内搜索 baseline unified frequency、同构 split 独立频率、异构 TP+频率；energy saving 相对 baseline。当前脚本使用 8-GPU PA/PF/DA/DF 公平预算，并对每个 SLO 取 saving top-10 后求均值。
- **限制**：这是 profile 上的理想组合搜索，不是在线 scheduler 的端到端测量；top-10 是选择性聚合，报告时必须披露。

### Fig.2：频率敏感性

- **数据源**：同一 V1 profile；固定 TP4，但 Prefill/Decode 和 A/F 曲线取不同代表点。
- **指标/筛选**：PA latency 用 input=1、PF latency 用 input=4、Prefill energy 用 input=8（bs=256）；DA/DF 用 input=1024、output=256、bs=64。
- **限制**：每条曲线分别除以自身最大值，且不同曲线不是同一个 workload 点；只能说明形状/敏感性，不能横向比较绝对 latency 或 energy。

### Fig.3：Decode latency/energy sensitivity heatmap

- **数据源**：V1 decode profile。
- **指标/筛选**：TP 1/2/4/8、batch 1/4/16/64/256；每组计算 `1-min/max`，再按 TP、batch 平均。
- **限制**：grouping 是 `tp,batch_size,input_len`，**未包含 output_len**，因此不同 output_len 可能被合入同一组；解释时应视为脚本当前聚合口径，而非严格控制 output length 的实验。

### Fig.4：F/A 比与 stall energy

- **数据源**：V1 prefill/decode profile。
- **指标/筛选**：F/A latency ratio 按 TP 和 batch 展示；stall/bubble energy 用 `|A-F| × TP × idle_power` 估算占总 A+F energy 的比例。
- **限制**：stall energy 不是直接测量；idle power 在脚本中按 210–1410 MHz 硬编码，缺失频点回退 80 W。不能将其表述为 NVML 实测逐点 bubble energy。

### Fig.5：启动与 AF/DVFS overhead

- **数据源**：本图专属 [`fig5_reshard/data/`](multi_node/motivation/fig5_reshard/data/)：`startup_overhead_by_tp.json`、`af_all_tp_results.json`、`dvfs_switch_summary_latest.csv`。
- **指标/筛选**：panel (a) 为无 CUDA graph 的 TP1/2/4/8 cold-start build/materialize/init；panel (b) 对比 native/AF-disagg TTFT/TPOT 与 DVFS SetGpuLockedClocks wall-time p50。
- **限制**：部分 JSON 有简短 source 字段，但目录未保留完整采集脚本、命令、日志、commit 与机器快照；因此缺少部分原始采集 provenance。可重绘，不应声称能从当前目录完整重采。

## 9. Energy model

### 9.1 Qwen3-32B

- **V1 layer profile**：[`data/v1_layer_profile/`](energy_model/Qwen3-32B/data/v1_layer_profile/) 含 Prefill/Decode 的 TP、长度、频率、batch 与 A/F latency/energy 网格。`retrain_models_v1.sh` 调用 `benchmark/test_motivation/energy_model.py` 训练 LUT、LinearReg、GBDT；README 强调 bs=256 必须在完整训练网格中。
- **V2 pipeline profile**：[`data/v2_pipeline_profile/`](energy_model/Qwen3-32B/data/v2_pipeline_profile/) 的 Prefill 仍用逐层数据，Decode 含 TP1、TP2 和异构 2A4F pipeline 数据；README 还描述 V3 merged 训练，但对应训练脚本可能位于主代码/历史分支而非本目录。
- **评估**：[`eval_predictor_fast.py`](energy_model/Qwen3-32B/eval_predictor_fast.py) 默认从 `models_v1` 加载模型并计算准确率。

当前 `models_v1/`、`models_v2/` 在此 checkout 中可能只有 CV/MAPE 报告，没有 predictor 所需的 pickle/joblib/LUT 序列化文件。运行 solver、DVFS 或 evaluator 前先列出目录并让 predictor 完成一次加载测试；“报告存在”不代表运行时模型存在。

### 9.2 Mixtral-8x7B

- `profile_tp2_afd_layer.py`、`profile_tp4_layer.py`、`profile_tp4_full.py` 等生成 V1 layer profile。
- `profile_tp4_pipeline.py` 与 fast 版本用于 pipeline profile，数据位于 `data/v2_pipeline_profile/`；V1/V2 model 目录当前同样可能只有报告。
- `profile_tp4_pipeline.py` 的实现把 `nvidia-smi --query-gpu=power.draw` 读数作为瞬时功率采样后积分，变量/注释中仍混用 `energy_counter_mj`、W、J、mJ；最终 `energy_mj = energy_j`，并按固定 40%/60% 拆 DA/DF，同时 sweep 中 `f_f` 没有实际用于锁频。故使用这条 pipeline profiling 数据训练或报告前，应客观复核频率是否真正分别施加、能量单位转换及 A/F 分摊方法；这不等于断言所有 Mixtral profile 都无效。

## 10. Comm 通信实验

正式导航见 [`comm/read.md`](comm/read.md)，编排入口为 [`comm/run_comm_bench.sh`](comm/run_comm_bench.sh)。但该脚本在容器命令中硬编码 `cd /mnt/workspace/lt/sglang`，与环境记录的容器路径 `/workspace/sglang` 冲突；在确认旧路径确实存在或另行修正脚本前，下面命令**不能视为可直接执行**。本文档任务不修改该脚本：

```bash
cd benchmark/AFlex_bench/comm
bash run_comm_bench.sh --quick
# 完整 sweep 在确认节点空闲、RDMA 和 memlock 后再去掉 --quick
python3 plot_comm_comparison.py
```

脚本从 node1 宿主机协调 node2，依次测：

1. 同节点 GPU0→GPU4 NVLink P2P；
2. Mooncake RDMA Write 单 NIC；
3. 4 NIC aggregate；
4. 64-block batch transfer；
5. AFD activation tensor 尺寸。

输入是消息尺寸、迭代数、NIC/GPU 映射；输出是 `comm/results/*.json`。`read.md` 还记录 `ib_write_bw/lat`、UCX GPUDirect 以及 affinity 辅助脚本。Mooncake benchmark使用 SGLang PD 相同类型的 `transfer_sync_write`/batch 原语，比 perftest 更贴近软件路径；但现有结果绑定特定 A800/NIC 拓扑，迁移后必须重测。绘图中的 embedded/reference 数据与新采集 JSON要明确区分。

## 11. 运行前检查清单

### 集群与权限

- [ ] 四个 IP 可 SSH，`BatchMode=yes` 不提示密码或 host-key 交互。
- [ ] `operator_test` 在目标节点运行，挂载 `/workspace/sglang` 与当前代码一致。
- [ ] 目标节点确实空闲；已通知共享集群用户清理范围。
- [ ] 容器可访问 8 GPU，记录实际 GPU 型号以解决 A800/A100 冲突。
- [ ] `nvidia-smi -lgc/-rgc` 权限可用，实验后能够恢复频率。
- [ ] memlock 足够，RDMA 设备和端口处于 ACTIVE；Mooncake 可导入。

### 模型与软件

- [ ] `/models/Qwen3-32B/` 或 `/models/Mixtral-8x7B/` 在每个目标容器可读。
- [ ] 各节点 SGLang commit、Python package、router 和 CUDA/Mooncake 版本一致。
- [ ] predictor 目录含实际序列化模型，并通过最小加载测试，不只有报告 TSV。
- [ ] workload 文件存在，request 数、input/output token 总数符合预期。

### 网络、端口与数据

- [ ] GPU/NIC affinity 与本机 `nvidia-smi topo -m`、`ibdev2netdev` 一致。
- [ ] smoke/benchmark/router/bootstrap/NCCL/Prometheus 端口无占用。
- [ ] 明确本次 energy scope：全部预算 GPU还是 active GPU。
- [ ] 明确 PASS SLO、绘图 SLO 线和 MoE SLO，写入结果 meta。
- [ ] 输出使用新的 prefix/时间戳；不要覆盖 canonical，验证后再 pack/patch。

## 12. 术语表

- **P / D**：Prefill / Decode 阶段。
- **A / F**：Attention / FFN 算子部分。
- **PA/PF/DA/DF**：Prefill-Attention、Prefill-FFN、Decode-Attention、Decode-FFN。
- **PD disaggregation**：Prefill 与 Decode 服务分离，KV cache 跨服务传输。
- **AFD / PDAF**：Attention 与 FFN 进一步分离；PDAF 同时包含 PD 与 A/F 分离。
- **TP**：Tensor Parallel degree。
- **M**：AFD micro-batch 数。
- **DVFS**：动态电压频率调节；本仓库主要控制 GPU graphics clock。
- **TTFT**：Time To First Token。
- **TPOT**：Time Per Output Token。
- **SLO**：服务时延目标；具体阈值以实验入口为准。
- **canonical packed data**：已按正式 schema、token 与 meta 口径整理的引用数据。
- **partial/raw**：中间或原始记录，可能未覆盖全实验矩阵。
- **active-GPU energy**：只对实际分配 GPU 求和；必须与比较基线的边界对齐。
- **Mooncake**：SGLang PD 跨节点 KV/数据传输后端，通常经 RDMA。
- **CUDA IPC**：同节点进程间共享 GPU memory/事件的通信机制。

## 13. Git 提交前建议检查

本文档任务本身不要求提交。若未来准备提交实验 artifact，建议从仓库根执行只读检查：

```bash
git status --short
git diff -- benchmark/AFlex_bench/
```

并人工确认：

- 没有模型权重、超大日志、临时 PID/端口文件、远端凭据或私有 IP 之外的敏感配置意外入库；
- canonical JSON 与 partial/raw/backup 分开，source 和 patch metadata 未丢失；
- energy/token 使用 input+output，单位 mJ/token 与 J 没有混淆；
- 图中的系列、SLO 线和数据文件一致，论文五系列未意外加入 MegaScale；
- PDF/PNG 确实由当前脚本生成且需要提交时才加入，不因 README 提到路径就假定文件存在；
- 未重命名历史目录 `model_scalibility`、`node_scalibility`，避免破坏脚本路径。

最后，任何“全量复现”都应记录 git commit、容器镜像、GPU 型号、驱动/CUDA、模型 revision、四节点拓扑、NIC/GID、predictor hash、SLO、能量边界和实际执行命令。当前仓库包含正式入口、冻结数据与历史探索脚本的混合体；可靠复现依赖先分类，再运行，而不是寻找一个不存在的全目录一键命令。
