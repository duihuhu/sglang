# AFlex 端到端宏基准

此目录独立保留原宏基准的真实部署实现，不导入或运行原被忽略的实验目录。来源为原 `run_benchmark.py` 对应的 `macro_bench/aflex.py`、`deployments.py`、`fabric_affinity.py` 和 `bench_common.py` 的 indexed SSE sender；保留仓库许可证。只发布源码、配置和两条小型合成请求，不包含原始 trace、profile、模型预测器、pkl 或结果。

## 支持方案

- `aflex_tier1`：decode-first、FFN-first GPU 分配，异构 Attention/FFN TP，IPC C++，Mooncake PD，多个 Prefill 子路由；客户端依据每对的 `max(tp_pa,tp_pf)` 加权轮询；计划频率初始化及运行时 compositional DVFS。
- `native_tp1_baseline` / `native_tp1_tier`：两节点各八个 TP1 worker，round-robin 路由，固定频率或 DynamoLLM unified DVFS。
- `pd_hetero_baseline` / `pd_hetero_tier_biscale`：2×TP4 Prefill 与 4×TP2 Decode，Mooncake，固定频率或 BiScale DVFS。

这些 baseline 是仓库 SGLang runtime 内实现的比较策略，不是外部 Dynamo/DistServe 官方部署。当前不包含 MegaScale、绘图、频率时间线、RDMA/NVLink 计数器和四节点并行。

## 配置和输入

`examples/config.json` 的 IP 是文档保留地址，不能直接用于真实实验。所有容器内路径、SSH 用户、容器名、模型、能耗预测器、GPU/NIC 对应、Mooncake rank IP 基址、端口、频率、决策与评估 SLO、每个测试点的请求文件及计划文件均需自行配置。

本地 `workload` / `plan` 路径相对于配置文件；容器路径必须是绝对路径且不含 shell 特殊字符。当前为了保持原 baseline 对照语义，要求每节点物理 GPU 池为 `0..7`；AFlex 只使用实际分配的子集。GPU/NIC 映射可逐节点不同。`mooncake_ip_base` 最后一段加物理 GPU ID 必须得到真实可用的 RDMA rank IP，网络环境需管理员预先配置。

必须配置 `paths.dvfs_library`（示例为 `/assets/libdvfs_ctrl.so`），并将同一路径列入 `preflight.native_libraries`。该路径须在两个节点的容器内均可用，且为不含 shell 特殊字符的绝对路径。harness 在进入配置的 repo 后为所有远端命令导出 `DVFS_CTRL_LIB`，服务启动及其子进程继承此变量；不依赖本地终端或容器预设的环境变量。缺少路径或未列入清单时，本地配置校验直接失败。

请求输入为 JSONL，每行至少包含 `input_len`、`output_len` 正整数和非负、非递减 `arrival_time_s`。可选 `use_input_ids:true` 使用 token ID 1000 重复生成精确长度输入；默认沿用原脚本 `"x" * input_len`，字符数不保证等于 tokenizer 的 token 数。

AFlex 计划文件为一个 JSON 对象，字段见 `examples/plan.json`：`name,k_p,k_d,tp_pa,tp_pf,tp_da,tp_df,f_pa,f_pf,f_da,f_df,tier`。频率单位 MHz；可选 `prefill_specs` 是长度严格等于 `k_p` 的对象数组，每项包含 `tp_pa,tp_pf,f_pa,f_pf`，可覆盖每个 Prefill 对的 TP/频率；配置、渲染、部署和失败清理范围均使用同一个解析器。这里不运行离线优化器，用户须提供可行计划。小型合成计划不代表任何性能最优结论。

## CPU 安全入口

从仓库根目录运行：

```bash
python benchmark/test_aflex/run_benchmark.py --help
python benchmark/test_aflex/run_benchmark.py preflight --config benchmark/test_aflex/examples/config.json
python benchmark/test_aflex/run_benchmark.py dry-run --config benchmark/test_aflex/examples/config.json
python -m pytest test/srt/test_aflex_benchmark.py -q
```

`preflight` 校验本地配置、请求、计划并通过真实部署函数渲染；`dry-run` 输出完整命令 JSON。两者禁用实际 transport、HTTP、NVML 和等待，不会 SSH、docker 或操作 GPU，不写结果文件。不需要安装 torch、CUDA 或 pynvml。离线预检不会检查远端模型/预测器、端口占用、RDMA 连通性、容器依赖或 GPU 型号；`OFFLINE_OK` 不意味着远端可运行。`run --execute` 则在任何锁频或服务部署之前，对两个节点进行只读远端预检：Python 版本、runtime/router 模块、配置的 native 动态库及其装载、模型目录及 config.json、预测器文件清单（存在、可读、非空）。远端预检失败会汇总节点错误并终止，不解锁或修改 GPU。

## 准备 Azure trace

```bash
python benchmark/test_aflex/prepare_trace.py --input /external/AzureLLMInferenceTrace_code.csv --output /external/macro_code_qps2.jsonl --qps 2 --n-requests 200 --seed 42
```

遵循原宏工作负载准备脚本的转换语义：读取前 500000 行 `ContextTokens,GeneratedTokens`，过滤非正长度，分别截断至 4096/1024，使用 seed 抽样后 shuffle，再用 `seed+qps` 生成指数间隔、累计并保留四位小数。`TIMESTAMP` 不参与 Poisson 到达生成；这不是对 Azure 原时间戳的直接重放，也不是 motivation 中的网格吸附转换。输出默认采用原宏基准的文本输入语义；输出文件存在时拒绝覆盖。

## 真实运行（需要显式授权开关）

```bash
python benchmark/test_aflex/run_benchmark.py run --config /external/cluster.json --execute --output /external/results/run01.json
```

请先部署与当前仓库匹配的 runtime、sglang_router、IPC JIT 构建环境、Mooncake、NVML Python binding 和能耗模型资产。`preflight.native_libraries` 必须列出容器内所需共享库绝对路径，`preflight.prediction_files` 必须列出相对于 energy_model 的所有必需预测文件（示例采用 V1 默认预测器组合）；二者均不可为空，清单由实验使用者维护，不反序列化 pkl。`ports.bootstrap+i` 是第 i 个 PA 的显式 bootstrap 端口，启动前检查空闲，并将同一值传给服务与子路由。配置的 repo 应已安装到容器 Python 环境。此工具不拷贝或改写远端 runtime 文件。SSH 使用系统 known_hosts 验证，不自动绕过主机校验。运行需要锁频权限，并且独占配置 GPU。每个测试点重新部署，健康检查、warmup 后按到达时间并发 SSE 请求；请求总窗口遵循原 `min(max(400,last_arrival+150),900)` 秒规则，随后记录结果。

完整成功必须收到 `finish_reason.type=length` 且报告完成 token 数匹配目标；chunk 数不充当 token 数。统计 TTFT、TPOT 分位数、吞吐、逐请求结果、超时/失败、请求级 SLO 并集违约率。TTFT 优先使用服务端纯处理时间，无此字段时使用客户端 TTFT。能耗为实际分配 GPU 的 NVML 累计毫焦计数差；缺失、非正或回退计数报错，绝不静默补零。mJ/token 分母为成功请求的输入加输出长度；文本输入长度字段可能仅为字符预算，论文报告应明确输入模式。部署和 warmup 不计入该请求窗口能耗。

## 安全与限制

只终止本次进程：每个服务使用独立 session 和随机 `AFLEX_BENCH_OWNER`，保留 PID；清理前核对 `/proc/PID/environ` 的 token 与 session ID，再向该进程组发送 TERM 并等待存活成员退出。leader 已消失时扫描同 session 的 token-owned 后代；遇到无法验证的成员则拒绝发送信号。默认等待 15 秒，超时保留 ownership 记录并报错；可用 `cleanup.force_kill:true` 在 TERM 超时后重新验证 ownership 再 KILL，随后仍等待退出。清理可重复调用，已退出 session 视为成功，只有确认退出才删除记录。无 broad pkill、按端口杀进程或调用旧 cleanup 脚本。服务端口占用时失败，不杀已有监听者。锁频只针对配置/分配 GPU，结束时重置频率；不会恢复进入实验前的任意自定义频率设置，因此必须独占这些 GPU。

强制结束 orchestrator、SSH 断连、进程忽略 TERM（且未启用安全强杀）或进程逃离受管 session 时可能需要管理员根据日志和 ownership token 手工处理；不提供跨运行全局清理。日志写入配置 remote_output，结果写入指定外部输出文件，不自动打包进源码目录。当前不自动 resume、选择最优计划、下载资产或生成论文图。真实 GPU、SSH、docker 链路尚未在此交付环境执行验证，CPU 单测只能证明配置、命令渲染和统计逻辑。
