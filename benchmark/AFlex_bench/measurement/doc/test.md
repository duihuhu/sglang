# RQ1 测试交接与恢复说明

最后核验时间：2026-09-11 23:09 CST（UTC+8）

## 2026-09-13 01:41 更新（最新口径）

- 活跃 QPS 集合改为 `[1, 2, 4, 8]`，不再测试低于 1 的 QPS。
- 每点请求数依次为：QPS1=32、QPS2=32、QPS4=64、QPS8=64。
- 每点仍执行 3 次独立 repeat；正式矩阵规模改为 1296 点。
- 正式请求总量为 62,208；理论到达窗口总下限约 6.31 小时，考虑逐点部署、排空和大模型开销，预计连续运行约 3–4 天。
- 648 个 QPS0.25/0.5 formal 行已标为 `superseded`；12 个旧请求数下完成的 QPS1/2/4/8 点已重置为 pending，原 artifact 均保留。
- Canary 改为 Balanced/QPS1/32 请求；旧 QPS0.25 canary 已标为 `superseded`。
- 新活跃矩阵清点结果：1296 formal pending、18 canary pending。
- v1 关键文件 SHA-256 在迁移前后保持不变。
- v1/v2 回归测试共 22 项通过，静态诊断无错误。
- 新的 18 个 canary 已启动；18/18 通过后自动进入 1296 点正式矩阵。

## 2026-09-13 更新（优先于下文旧状态）

- RQ1-v2 已实现，相关脚本、配置、测试和隔离结果目录均已创建。
- 正式矩阵仍为 1944 点，18/18 canary 已通过。
- QPS 0.25 已从 256 请求/点调整为 64 请求/点；QPS 0.5、1、2、4、8 仍为 256 请求/点。
- QPS 0.25 的归一化到达窗口由 1020 秒缩短为 252 秒。
- 4 个使用旧 256 请求口径完成的正式 QPS 0.25 点已重置为 pending；旧结果保存在各行的 `superseded_results` 和原 artifact 中，没有删除。
- 迁移记录 ID：`qps0.25_request_count_64_v1`。
- 调整后正式请求总数为 435,456；理论到达窗口下限约 4.65 天，现实预计约 12–14 天。
- v1 的 `results/default/progress.json` 和 `data/workloads/index.json` 在迁移前后 SHA-256 不变。
- v1/v2 回归测试共 22 项通过，静态诊断无错误。
- 2026-09-13 00:45 CST 已从断点恢复正式矩阵；启动时状态为 15 valid、1929 pending。
- 正式运行监督器会在外部占卡或环境故障时安全暂停，不会继承此前的一次性清场授权。

## 1. 测试目标

RQ1 探究不同模型、数据集与部署架构的性能和能效表现。

完整测试维度：

- 6 个模型
- 3 种部署架构
- 6 种定长数据集
- 多档 QPS
- 每个测试点 3 次独立重复

### 1.1 模型

Dense：

1. `dense_llama3_1_8b`
   - Llama-3.1-8B
   - Small
   - 路径：`/models/llama3.1-8`

2. `dense_qwen3_32b`
   - Qwen3-32B
   - Medium
   - 路径：`/models/Qwen/Qwen3-32B`

3. `dense_llama3_3_70b`
   - Llama-3.3-70B-Instruct
   - Large
   - 路径：`/models/meta-llama/Llama-3.3-70B-Instruct`

MoE：

4. `moe_mixtral_8x7b`
   - Mixtral-8x7B
   - Small
   - 路径：`/models/Mixtral/Mixtral-8x7B-v0.1`

5. `moe_qwen3_30b_a3b`
   - Qwen3-30B-A3B
   - Medium
   - 路径：`/models/Qwen3-30B-A3B`

6. `moe_mixtral_8x22b`
   - Mixtral-8x22B
   - Large
   - 路径：`/models/Mixtral/Mixtral-8x22B-v0.1`

模型配置文件：

`benchmark/AFlex_bench/measurement/rq1_model_dataset_energy/configs/models.json`

### 1.2 架构

每种架构使用 8 张 GPU：

1. Native
   - `Native TP8`

2. PD
   - Prefill TP4
   - Decode TP4
   - 同节点 CUDA IPC KV Cache 传输

3. AF
   - Attention TP4
   - FFN TP4

架构配置文件：

`benchmark/AFlex_bench/measurement/rq1_model_dataset_energy/configs/matrix.json`

### 1.3 数据集

六种固定输入/输出长度：

| 数据集 | 标识 | 输入 token | 输出 token |
|---|---|---:|---:|
| QA | `qa_lpld` | 128 | 64 |
| Chatbot | `chatbot_lphd` | 128 | 1024 |
| Balanced | `balanced_mpmd` | 512 | 256 |
| RAG | `rag_hpld` | 4096 | 64 |
| Summary | `summary_hphd` | 4096 | 1024 |
| LongContext | `longcontext` | 16384 | 256 |

负载配置文件：

`benchmark/AFlex_bench/measurement/rq1_model_dataset_energy/configs/workloads.json`

---

## 2. RQ1-v1 旧矩阵

### 2.1 旧测试参数

RQ1-v1 使用：

- QPS：`[2, 4, 8, 16]`
- 每点 3 次 repeat
- 每个 trace 64 个请求
- 6 模型 × 3 架构 × 6 数据集 × 4 QPS × 3 repeats
- 总计：1296 次运行

旧测试目录：

`benchmark/AFlex_bench/measurement/rq1_model_dataset_energy`

旧结果目录：

`benchmark/AFlex_bench/measurement/rq1_model_dataset_energy/results/default`

### 2.2 旧矩阵当前状态

最后确认的正式矩阵状态：

- 总数：1296
- `valid`：367
- `skipped_saturated`：854
- `pending`：41
- `failed`：34
- 已形成 valid 或 skip 结论：1221 / 1296
- 表面完成率：94.2%

注意：由于后面确认了饱和判定存在设计缺陷，这个“完成率”不代表 1221 个点均经过实际测试。

模型级进度：

- Llama-3.1-8B：216 / 216 已处理
- Qwen3-32B：216 / 216 已处理
- Llama-3.3-70B：216 / 216 已处理
- Mixtral-8x7B：216 / 216 已处理
- Qwen3-30B-A3B：216 / 216 已处理
- Mixtral-8x22B：未完成

Mixtral-8x22B 状态：

- Native：
  - valid：20
  - skipped：52
  - 72 / 72 已处理

- PD：
  - valid：17
  - skipped：51
  - failed：4

- AF：
  - valid：1
  - failed：30
  - pending：41

剩余 34 个 failed 均为容器或 Router 环境中断，不是模型性能失败。

主要错误为：

- `container ... is not running`
- `ModuleNotFoundError: No module named 'sglang_router'`
- 容器被终止后产生的部署阶段连锁失败

这些失败不能作为性能或兼容性结论。

### 2.3 旧结果关键路径

完整运行状态：

`benchmark/AFlex_bench/measurement/rq1_model_dataset_energy/results/default/progress.json`

聚合 CSV：

`benchmark/AFlex_bench/measurement/rq1_model_dataset_energy/results/default/rq1_report.csv`

聚合 JSON：

`benchmark/AFlex_bench/measurement/rq1_model_dataset_energy/results/default/rq1_report.json`

原始 artifact：

`benchmark/AFlex_bench/measurement/rq1_model_dataset_energy/results/default/artifacts`

每个 artifact 通常包含：

- `summary.json`
- `deployment_manifest.json`
- 请求级结果
- 服务日志
- GPU 能耗与系统信息

---

## 3. RQ1-v1 已确认的测试设计问题

不要继续使用旧口径判断模型的最大可支持 QPS。

### 3.1 不同 workload 使用不同的 Poisson 到达序列

旧生成器：

`benchmark/AFlex_bench/measurement/rq1_model_dataset_energy/scripts/generate_workloads.py`

其随机种子包含：

- input length
- output length
- QPS

因此相同目标 QPS 下，不同 workload 使用不同的随机到达序列。

旧逻辑类似：

```python
rng = random.Random(
    seed
    + input_len * 1009
    + output_len * 9176
    + qps
)
```

这导致 QA 和 RAG 虽然都标为 QPS=2，但有限 trace 中的实际到达窗口不同。

### 3.2 QA 与 RAG 的具体异常

QPS=2、64 请求时：

QA：

- 输入/输出：128/64
- arrival span：35.430 秒
- 零服务时间下的理论统计吞吐：
  - `64 / 35.430 = 1.806 QPS`
- 0.9 阈值要求：
  - `2 × 0.9 = 1.8 QPS`
- 留给最后请求排空的时间：
  - `64 / 1.8 - 35.430 ≈ 0.125 秒`

RAG：

- 输入/输出：4096/64
- arrival span：31.097 秒
- 零服务时间下的理论统计吞吐：
  - `64 / 31.097 = 2.058 QPS`
- 留给最后请求排空的时间：
  - `64 / 1.8 - 31.097 ≈ 4.459 秒`

因此：

- QA 只要最后一个请求耗时超过约 125ms，就会低于 0.9 阈值。
- RAG 即使请求更重，仍有约 4.46 秒的排空余量。
- 这会制造“RAG 能支持的 QPS 比 QA 更高”的假象。

该现象不是模型真实性能结论。

### 3.3 Achieved QPS 混入排空时间

共享统计代码：

`benchmark/AFlex_bench/measurement/benchmark/src/aflex_benchmark/stats.py`

旧 achieved QPS 计算为：

```text
成功请求数 /
（最后一个请求完成时间 - 第一个请求发送时间）
```

对应代码使用：

- 最早 `sent_offset_s`
- 最晚 `completed_offset_s`

这个指标实际是包含 drain 的 completion throughput，同时受到以下因素影响：

- Poisson trace 的随机首尾长度
- 请求执行时间
- 队列等待
- 最后请求的长尾延迟
- 整个队列的排空时间

它不应直接用于判断负载生成器是否成功提供目标 offered QPS。

### 3.4 单次 repeat 会触发更高 QPS 全部跳过

旧饱和逻辑：

`benchmark/AFlex_bench/measurement/rq1_model_dataset_energy/scripts/rq1lib.py`

旧规则为：

- 如果同一模型、架构、workload、repeat 的较低 QPS：
  - `achieved_qps / target_qps < 0.9`
- 则同一 repeat 的所有更高 QPS 都标记为：
  - `skipped_saturated`

例如：

- repeat 1 的 QPS=2 因短 trace 首尾偏差只得到 1.74 QPS；
- 1.74 / 2 = 0.87；
- QPS 4、8、16 被直接跳过；
- 这些高 QPS 实际上没有运行。

因此旧结果中的 `skipped_saturated` 不能称为“实测饱和点”。

### 3.5 旧数据可以如何使用

仍可局部参考：

- 已实际完成的 valid 点
- 单点 TTFT、TPOT、E2E
- 单点能耗
- 单点平均功率
- 相同 trace 条件下的架构比较

不应直接使用：

- 最大支持 QPS
- workload 间容量排序
- skipped 点作为实际测量值
- “RAG 比 QA 吞吐更高”的结论
- 由单个低 QPS repeat 推断出的饱和边界

---

## 4. RQ1-v2 最终测试口径

RQ1-v2 必须独立于旧结果实现。

### 4.1 矩阵

RQ1-v2 使用：

- 6 个模型
- 3 种架构
- 6 个 workload
- QPS：`[0.25, 0.5, 1, 2, 4, 8]`
- 每点 3 次独立 repeat
- 每个 trace 256 个请求
- 总计：
  - `6 × 3 × 6 × 6 × 3 = 1944` 次正式运行

### 4.2 共享 arrival trace

对于相同的：

- QPS
- repeat

六个 workload 必须共享完全相同的 `arrival_time_s` 序列。

也就是说：

```text
arrival_trace(workload, qps, repeat)
=
shared_arrival_trace(qps, repeat)
```

workload 的 input/output length 不得进入 arrival seed。

### 4.3 归一化有限 Poisson trace

先生成 Poisson inter-arrival，然后进行线性归一化。

要求：

```text
first arrival = 0
last arrival = (N - 1) / target_qps
```

其中：

```text
N = 256
```

中间点保持原 Poisson 序列的相对间隔形状。

这样有限 trace 的 realized offered QPS 必须严格为：

```text
(N - 1) / (last_arrival - first_arrival)
= target_qps
```

### 4.4 Repeat 独立

repeat 1、2、3 必须使用不同 seed。

例如：

```text
seed = hash(base_seed, qps, repeat)
```

不能让 workload 进入 seed。

要求：

- 同 QPS、同 repeat、不同 workload：arrival 完全一致
- 同 QPS、不同 repeat：arrival 不同
- 所有 trace 首尾仍严格归一化

### 4.5 禁止 early saturation skip

RQ1-v2 中：

- 禁止调用旧 `saturated()` 逻辑
- 禁止因为低 QPS 的一个 repeat 失败而跳过高 QPS
- 1944 个正式点全部实际运行
- 即使某个 QPS 不满足 SLA，也继续测更高 QPS
- 除非出现明确资源冲突、环境错误或用户要求停止

### 4.6 指标定义

RQ1-v2 至少记录以下指标。

#### Target/offered QPS

配置中指定的目标 QPS：

```text
offered_qps = target_qps
```

#### Realized offered QPS

根据计划 arrival trace 计算：

```text
realized_offered_qps
=
(N - 1) /
(last_scheduled_arrival - first_scheduled_arrival)
```

归一化后应接近 target QPS，误差只来自浮点精度。

#### Send QPS

根据实际请求发送时间计算：

```text
send_qps
=
(N - 1) /
(last_sent_time - first_sent_time)
```

用于判断负载生成器是否跟上了目标 arrival。

#### Completion QPS

保留旧 achieved QPS 的含义，但明确命名：

```text
completion_qps
=
successful_requests /
(last_completed_time - first_sent_time)
```

为了兼容旧报告，可以继续保留：

```text
achieved_qps = completion_qps
```

但不得再将它用于判断 offered load 是否达成。

#### Arrival lag

每个请求：

```text
arrival_lag
=
actual_sent_time - scheduled_arrival_time
```

至少报告：

- arrival lag P50
- arrival lag P90
- arrival lag P99
- arrival lag max

#### Drain time

定义：

```text
drain_time_s
=
last_completed_time - last_scheduled_arrival_time
```

也可额外记录：

```text
last_completed_time - last_actual_send_time
```

但字段含义必须明确，不能混用。

#### 其他指标

继续记录：

- success rate
- TTFT P50/P90/P95/P99
- TPOT P50/P90/P95/P99
- E2E P50/P90/P95/P99
- input/output token throughput
- total energy
- average cluster power
- J/request
- J/input token
- J/output token
- input/output tokens/J

### 4.7 Load generator 健康判定

负载生成器是否健康应检查：

```text
send_qps / target_qps >= 0.98
```

同时检查：

- arrival lag P90/P99
- 请求线程池是否达到 max inflight
- 请求是否在计划 arrival 后明显延迟发送

如果 load generator 自身无法跟上：

- 该点应标记为 `loadgen_invalid`
- 不应标记为模型饱和
- 不应参与架构性能比较

### 4.8 容量和 SLA 判定

模型容量不再使用：

```text
completion_qps / target_qps >= 0.9
```

容量与 SLA 主要依据：

- success rate
- TTFT
- TPOT
- E2E
- load generator 健康状态
- 队列积压与 drain time

原 SLA 可暂时保留：

- success rate ≥ 0.99
- TTFT P90 ≤ 5000ms
- TPOT P90 ≤ 250ms
- E2E P90 ≤ 60000ms

但是否“达到 offered QPS”应由 send QPS 和 arrival lag 判断，而不是 completion QPS。

---

## 5. RQ1-v2 文件与结果隔离

绝对不能覆盖以下旧路径：

```text
benchmark/AFlex_bench/measurement/rq1_model_dataset_energy/data/workloads
benchmark/AFlex_bench/measurement/rq1_model_dataset_energy/results/default
```

建议创建：

```text
benchmark/AFlex_bench/measurement/rq1_model_dataset_energy/configs/matrix_v2.json
benchmark/AFlex_bench/measurement/rq1_model_dataset_energy/configs/workloads_v2.json

benchmark/AFlex_bench/measurement/rq1_model_dataset_energy/data/workloads_v2

benchmark/AFlex_bench/measurement/rq1_model_dataset_energy/results/v2

benchmark/AFlex_bench/measurement/rq1_model_dataset_energy/scripts/generate_workloads_v2.py
benchmark/AFlex_bench/measurement/rq1_model_dataset_energy/scripts/run_rq1_v2.py
benchmark/AFlex_bench/measurement/rq1_model_dataset_energy/scripts/report_rq1_v2.py

benchmark/AFlex_bench/measurement/rq1_model_dataset_energy/tests/test_rq1_v2.py
```

建议使用新的 phase 或 namespace：

```text
canary_v2
formal_v2
```

或者保证 stable ID 中包含：

```text
rq1-v2
```

防止和旧 run ID 冲突。

---

## 6. RQ1-v2 当前实现状态

截至最后交接时间：

- `generate_workloads_v2.py` 尚未创建
- `run_rq1_v2.py` 尚未创建
- `report_rq1_v2.py` 尚未创建
- `test_rq1_v2.py` 尚未创建
- v2 workload 尚未生成
- v2 dry-run inventory 尚未生成
- v2 canary 尚未运行
- v2 正式矩阵尚未运行

原因是原 Agent 会话没有挂载普通文本编辑工具，不能合规创建源码文件。

不要因为 GPU 已空闲就启动旧 v1 矩阵代替 v2。

---

## 7. RQ1-v2 必须添加的测试

至少覆盖：

1. 完整矩阵规模：

```text
6 × 3 × 6 × 6 × 3 = 1944
```

2. run ID 唯一：

```text
len(set(run_ids)) == 1944
```

3. float QPS stable ID 稳定：

```text
0.25, 0.5, 1, 2, 4, 8
```

不会因字符串格式变化产生不同 ID。

4. 同 QPS、同 repeat 下，六个 workload 的 arrival 完全相同。

5. 不同 repeat 的 arrival 序列不同。

6. 每个 trace：

```text
len(requests) == 256
first_arrival == 0
last_arrival == (255 / qps)
```

7. realized offered QPS 与 target QPS 一致。

8. send QPS 计算公式正确。

9. completion QPS 计算公式正确。

10. drain time 计算公式正确。

11. arrival lag P90/P99 正确。

12. 正式矩阵不调用 early saturation skip。

13. 低 QPS 不满足 SLA 时，高 QPS 仍保持可执行状态。

14. v2 不读取或修改：

```text
results/default
data/workloads
```

15. 环境失败与模型性能失败分类明确。

---

## 8. CUDA IPC interior-pointer 修复

Mixtral-8x22B 的 PD 曾因 CUDA IPC interior pointer 失败。

根因：

- Mixtral-8x22B 权重大
- 每层 KV buffer 较小
- 多层 KV tensor 共享同一个底层 CUDA allocation
- 原实现错误地要求每个 KV tensor pointer 都是 allocation base

已修复为：

- 导出 CUDA allocation base 的 IPC handle
- 同时传输每个 tensor 相对 base 的 offset
- 传输 allocation size
- 接收端使用 `opened_base + offset`
- 同一 handle 只 open 一次
- 同一 base 只 close 一次
- 校验 offset + length 不越界
- 兼容旧扩展只返回 bytes handle 的情况

修复文件：

```text
sgl-kernel/csrc/afd_ipc/afd_ipc_pybind.cpp
python/sglang/srt/disaggregation/cuda_ipc/conn.py
test/registered/unit/disaggregation/test_cuda_ipc_protocol.py
```

验证结果：

- CUDA IPC 协议单测：7 passed
- Python compile：通过
- C++ 扩展在容器中实际重新编译成功
- Mixtral-8x22B PD 真实 8-GPU canary：valid
- Native、PD、AF 三架构 canary 均曾通过

不要撤销这些修改。

构建时曾使用：

```bash
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions_cuda_ipc_offset_v2
```

运行 PD 时也应保持该缓存目录，或确认容器已经加载根据新源码编译的扩展。

---

## 9. measurement 容器环境

### 9.1 容器历史

原 `measurement` 容器曾正常运行全部架构，但之后：

- 先以 exit 137 停止
- 后来被删除
- 无法直接 `docker start`
- 使用原镜像和已知挂载重建同名容器

重建所用镜像：

```text
operator_test:migration-20260825-clean
```

已知挂载：

```text
/mnt/nvme1/lt -> /workspace
/mnt/nvme1/models -> /models
```

使用：

- `--gpus all`
- `--network host`
- `--ipc host`
- unlimited memlock
- 工作目录 `/workspace/Measurement_sglang`

### 9.2 Router 环境问题

重建后，`sglang-router 0.3.2` 的 editable 安装仍指向旧路径：

```text
/workspace/sglang/sgl-model-gateway/bindings/python
```

但实际代码位于：

```text
/workspace/Measurement_sglang/sgl-model-gateway/bindings/python
```

因此一度出现：

```text
ModuleNotFoundError: No module named 'sglang_router'
```

表现为：

- PD Prefill 服务正常
- PD Decode 服务正常
- Router 无法启动
- runner 等待 Router 健康检查
- 负载生成器不启动

### 9.3 Router 恢复

网络下载 `sglang-router==0.3.2` wheel 一度挂起。

最终从 `moe-energy` 容器复制了兼容模块：

- Python 3.10
- glibc 2.35
- `sglang-router 0.3.2`

不要使用 `gxt_ghostserve` 中的二进制，因为其 glibc 2.39，与当前容器 glibc 2.35 不兼容。

恢复后已验证：

```python
import sglang_router
import sglang_router.sglang_router_rs
```

均成功。

临时 Router 启动验证：

- 测试端口：43920
- `/health` 返回 HTTP 200
- body：`OK`
- 临时 Router 已清理

新设备启动正式测试前仍应重新检查：

```bash
docker start measurement

docker exec measurement python3 -c "
import importlib.metadata as md
import sglang_router
import sglang_router.sglang_router_rs as rs
print(md.version('sglang-router'))
print(sglang_router.__file__)
print(rs.__file__)
"
```

并验证：

```text
router version = 0.3.2
```

---

## 10. 2026-09-11 23:00 GPU 清场状态

用户为当晚 23:00 预约了 8 张 GPU，并明确授权当次清场：

- 可以终止 8 张 GPU 上的全部 compute 进程
- 包括其他用户和其他容器中的任务

该授权仅适用于当晚这次预约清场，不得在新会话中视为永久授权。

清场时识别到：

1. PegaFlow/vLLM runtime
   - GPU 0、1、2
   - PegaFlow server
   - 三个 VLLM EngineCore
   - 对应 runtime 容器

2. PegaFlow client
   - AgentTrajectory/ShareGPT 客户端
   - 不直接持有 GPU，但会继续驱动服务

3. `moe-energy`
   - GPU 3–6 上的 SGLang scheduler

已执行：

- 停止 PegaFlow runtime 容器
- 停止 PegaFlow client 容器
- 停止 `moe-energy`
- 将相关容器 restart policy 设为 `no`
- PegaFlow runtime 一度卡在 Docker teardown
- 最后通过容器 cgroup 清理成功退出

最后确认：

```text
GPU 0: 1 MiB, 0%
GPU 1: 1 MiB, 0%
GPU 2: 1 MiB, 0%
GPU 3: 1 MiB, 0%
GPU 4: 1 MiB, 0%
GPU 5: 1 MiB, 0%
GPU 6: 1 MiB, 0%
GPU 7: 1 MiB, 0%
```

没有 GPU compute 进程。

但在新设备上开始操作前，必须实时重新检查，不能仅依赖本文档中的状态：

```bash
date '+%Y-%m-%d %H:%M:%S %Z (%z)'

nvidia-smi \
  --query-gpu=index,memory.used,utilization.gpu,power.draw \
  --format=csv,noheader

nvidia-smi \
  --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
  --format=csv,noheader
```

如果发现新的外部 GPU 任务：

- 不要擅自终止
- 先暂停自己的测试
- 仅在用户再次明确授权新的清场范围后处理

---

## 11. 新设备建议执行顺序

工作目录：

```bash
cd /mnt/nvme1/lt/Measurement_sglang/benchmark/AFlex_bench/measurement/rq1_model_dataset_energy
```

### 阶段 1：实现 RQ1-v2

创建并完善：

```text
configs/matrix_v2.json
configs/workloads_v2.json
scripts/generate_workloads_v2.py
scripts/run_rq1_v2.py
scripts/report_rq1_v2.py
tests/test_rq1_v2.py
```

必要时修改共享统计：

```text
benchmark/AFlex_bench/measurement/benchmark/src/aflex_benchmark/stats.py
benchmark/AFlex_bench/measurement/benchmark/src/aflex_benchmark/runner.py
```

但必须：

- 保持旧字段兼容
- 不改变旧结果文件
- 对新指标增加新字段
- 为旧逻辑添加回归测试

### 阶段 2：运行 CPU 单测

建议：

```bash
python3 -m py_compile \
  scripts/generate_workloads_v2.py \
  scripts/run_rq1_v2.py \
  scripts/report_rq1_v2.py \
  tests/test_rq1_v2.py
```

```bash
python3 -m pytest tests/test_rq1_v2.py -xvs
```

然后运行旧测试，确认没有破坏 v1：

```bash
python3 -m pytest tests/test_rq1.py -xvs
```

如果修改共享 runner/stats，还应运行相关共享测试。

### 阶段 3：生成 v2 workload

预期命令：

```bash
python3 scripts/generate_workloads_v2.py
```

检查：

- 6 workloads
- 6 QPS
- 3 repeats
- 每文件 256 请求
- 共 108 个 workload 文件：
  - `6 × 6 × 3 = 108`
- index 中包含 SHA256
- 同 qps/repeat 的六个 workload arrival 完全相同
- first=0
- last=255/qps

### 阶段 4：dry-run inventory

预期命令：

```bash
python3 scripts/run_rq1_v2.py \
  --node node1
```

必须确认：

```text
formal selected = 1944
```

并确认结果只写入：

```text
results/v2
```

不能访问或修改：

```text
results/default
```

### 阶段 5：容器和环境检查

```bash
docker start measurement
```

检查挂载：

```bash
docker inspect measurement \
  --format '{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}'
```

检查模型：

```bash
docker exec measurement bash -lc '
for p in \
  /models/llama3.1-8 \
  /models/Qwen/Qwen3-32B \
  /models/meta-llama/Llama-3.3-70B-Instruct \
  /models/Mixtral/Mixtral-8x7B-v0.1 \
  /models/Qwen3-30B-A3B \
  /models/Mixtral/Mixtral-8x22B-v0.1
do
  test -f "$p/config.json" || {
    echo "missing config: $p"
    exit 1
  }
done
'
```

检查 Router：

```bash
docker exec measurement python3 -c "
import importlib.metadata as md
import sglang_router
import sglang_router.sglang_router_rs as rs
print(md.version('sglang-router'))
print(sglang_router.__file__)
print(rs.__file__)
"
```

检查新 CUDA IPC 源码可见：

```bash
docker exec measurement test -f \
  /workspace/Measurement_sglang/python/sglang/srt/disaggregation/cuda_ipc/conn.py
```

检查 GPU：

```bash
nvidia-smi \
  --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
  --format=csv,noheader
```

只有确认 8 卡无 compute 进程后才能启动 canary。

### 阶段 6：运行 RQ1-v2 canary

建议：

- 6 模型 × 3 架构
- 共 18 个 canary
- workload：Balanced
- QPS：0.25
- canary 使用独立 run ID 和结果目录

预期命令：

```bash
python3 scripts/run_rq1_v2.py \
  --node node1 \
  --phase canary_v2 \
  --execute
```

然后检查 gate：

```bash
python3 scripts/run_rq1_v2.py \
  --node node1 \
  --check-gate
```

必须满足：

```text
18 / 18 canary valid
```

如果某个 canary 失败：

- 不要启动 formal
- 检查 artifact 和服务日志
- 区分环境失败、模型错误、GPU 资源冲突和 loadgen 错误
- 修复后只重试失败 canary

### 阶段 7：运行正式矩阵

只有 canary 18/18 后：

```bash
python3 scripts/run_rq1_v2.py \
  --node node1 \
  --phase formal_v2 \
  --execute \
  --retry-failed
```

建议继续使用监督器：

- 遇到外部 GPU 占用时安全暂停
- 保持当前点 pending
- 保存 pause reason
- GPU 释放后自动恢复
- 不终止外部进程，除非用户重新明确授权
- 完成条件必须严格为：
  - 1944 点全部有实际 valid/invalid/SLA 结果
  - 不能把 blocked 当完成
  - 不能生成 skipped_saturated

### 阶段 8：生成报告

预期命令：

```bash
python3 scripts/report_rq1_v2.py \
  --results results/v2
```

报告需要区分：

- 正常完成
- SLA pass/fail
- loadgen invalid
- 环境失败
- pending
- blocked

不得把环境失败计入性能均值。

---

## 12. 关键源码路径

RQ1 目录：

```text
benchmark/AFlex_bench/measurement/rq1_model_dataset_energy
```

旧负载生成器：

```text
benchmark/AFlex_bench/measurement/rq1_model_dataset_energy/scripts/generate_workloads.py
```

旧 runner：

```text
benchmark/AFlex_bench/measurement/rq1_model_dataset_energy/scripts/run_rq1.py
```

旧 RQ1 helper：

```text
benchmark/AFlex_bench/measurement/rq1_model_dataset_energy/scripts/rq1lib.py
```

旧运行手册：

```text
benchmark/AFlex_bench/measurement/rq1_model_dataset_energy/docs/runbook.md
```

旧数据字典：

```text
benchmark/AFlex_bench/measurement/rq1_model_dataset_energy/docs/data_dictionary.md
```

共享请求生成与采集：

```text
benchmark/AFlex_bench/measurement/benchmark/src/aflex_benchmark/collect/requests.py
```

共享统计：

```text
benchmark/AFlex_bench/measurement/benchmark/src/aflex_benchmark/stats.py
```

共享 runner：

```text
benchmark/AFlex_bench/measurement/benchmark/src/aflex_benchmark/runner.py
```

共享部署生命周期：

```text
benchmark/AFlex_bench/measurement/benchmark/src/aflex_benchmark/deploy/base.py
```

CUDA IPC 修复：

```text
sgl-kernel/csrc/afd_ipc/afd_ipc_pybind.cpp
python/sglang/srt/disaggregation/cuda_ipc/conn.py
test/registered/unit/disaggregation/test_cuda_ipc_protocol.py
```

---

## 13. 明确禁止事项

1. 不要继续运行 RQ1-v1 剩余 75 点。
2. 不要把 `skipped_saturated` 当成实际执行结果。
3. 不要用旧矩阵推导模型最大 QPS。
4. 不要覆盖 `results/default`。
5. 不要覆盖 `data/workloads`。
6. 不要修改或删除旧 artifact。
7. 不要撤销 CUDA IPC interior-pointer 修复。
8. 不要在 RQ1-v2 单测和 dry-run 通过前占用 GPU。
9. 不要在 canary 18/18 前启动正式矩阵。
10. 不要把容器、Router 或 GPU 冲突导致的失败解释为模型失败。
11. 不要默认继承 2026-09-11 晚上的“杀全部 GPU 进程”授权。
12. 新会话发现外部占卡时，先询问用户或安全等待。

---

## 14. 新 Agent 接续提示词

可以将下面提示直接发送给新 Agent：

> 阅读 `benchmark/AFlex_bench/measurement/doc/test.md`，接续 RQ1-v2 工作。先确认旧结果与 CUDA IPC 修复不被覆盖，然后实现共享归一化 arrival trace、QPS `[0.25,0.5,1,2,4,8]`、256 请求、3 repeats、无 early skip 的独立 v2 runner。运行全部 CPU 单测、生成 workload 并 dry-run 确认正式矩阵为 1944 点。之后实时检查容器、Router、模型和 8 张 GPU；只有 18/18 canary 通过后才能启动正式矩阵。不要默认终止其他用户任务，2026-09-11 当晚的全部 GPU 清场授权不可延续。