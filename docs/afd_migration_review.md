# SGLang AFD（Attention-FFN Disaggregation）迁移与优化报告

## 一、概述

### 什么是 AFD

AFD 将 Transformer 每一层拆分为 Attention（A）和 FFN（F）两部分，分别部署在不同 GPU 节点上。每层有 2 次跨节点传输（A->F 和 F->A），通过 microbatch 流水线隐藏通信延迟。

### 核心价值

- MoE 模型中 Attn 节点不加载 Expert 权重，KV Cache 容量提升约 4 倍
- 更高并发 → 更高吞吐 + 更低 P99 延迟

### 迁移来源与目标

- **源**: `sglang_af/sglang_v1`（AFD PoC 实现）
- **目标**: `sglang`（新版，原本无 AFD 代码）

---

## 二、文件清单

### 新建文件（7 个）

| 文件路径 | 作用 |
|----------|------|
| `python/sglang/srt/layers/afd_type.py` | `AFDPerspective` 枚举和 micro_batch 参数解析 |
| `python/sglang/srt/layers/afd.py` | 核心逻辑：通信器（ZMQ/StepMesh/分片并行）、流水线调度、`model_forward_afd`、`AFDCommunicator`、`AsyncTensorCommunicator`、Proxy 模块 |
| `python/sglang/srt/layers/afd_mixin.py` | 通用 Mixin（`AFDDecoderLayerMixin`）和权重过滤器（`AFDWeightFilter`） |
| `python/sglang/srt/batch_overlap/afd_overlap.py` | `AfdForwardBatchPreparer`——microbatch 切分与子 batch 组装 |
| `python/sglang/srt/managers/scheduler_afd_mixin.py` | 可复用的 AFD 调度辅助方法 Mixin |
| `test/srt/test_afd_fixture.py` | 测试 fixture（双进程启动） |
| `test/srt/test_afd_basic.py` | 端到端测试用例 |

### 修改文件（13 个）

| 文件路径 | 修改内容 |
|----------|----------|
| `server_args.py` | 新增 `afd_perspective`、`afd_micro_batch`、`afd_attn_ratio`、`afd_attn_tp`、`afd_ffn_tp` + `_handle_afd()` 验证 |
| `io_struct.py` | 新增 `AFDReqInput`（继承 `BaseReq`，含 `req_ids`/`seq_lens`/`extend_lens`） |
| `schedule_batch.py` | `ScheduleBatch` + `ModelWorkerBatch` 新增 `afd_split_seq_index: Optional[List[int]]` |
| `forward_batch_info.py` | AFD 字段 + `init_new` 直接传入 `afd_split_seq_index` |
| `model_runner.py` | `init_attention_backend` 新增 `AfdAttnBackend` 分支 |
| `tbo_backend.py` | 新增 `AfdAttnBackend` 类 |
| `qwen3_moe.py` | DecoderLayer 加入 `_afd_init`/`forward_afd_A`/`forward_afd_F`；`load_weights` 加入 `AFDWeightFilter` |
| `qwen2_moe.py` | 同上 + Model.forward 加入 AFD 分支 + `load_weights` 加入 `AFDWeightFilter` |
| `qwen3.py` | DecoderLayer 加入 AFD；Qwen3Model 重写 forward 加入 AFD 分支 |
| `deepseek_v2.py` | DecoderLayer 加入 AFD；Model.forward 加入 AFD 分支 |
| `scheduler.py` | AFD ZMQ 通道 + `event_loop_afd` + `dispatch_event_loop` PREFILL/DECODE AFD 分支 |
| `disaggregation/prefill.py` | 新增 `event_loop_afd_disagg_prefill` |
| `disaggregation/decode.py` | 新增 `event_loop_afd_disagg_decode` |

---

## 三、已应用的优化

### 通信层

| 编号 | 内容 | 位置 |
|------|------|------|
| C1 | ZMQ 零拷贝传输（metadata + raw bytes） | `afd.py` ZMQSimpleTensorCommunicator |
| C2 | 独立 CUDA stream + `AsyncTensorCommunicator` + 跨 stage `postprocess_layer_start_recv` | `afd.py` |
| C3 | `deque` 替代 `list`、去掉 `zero_()`、动态 buffer pool | `afd.py` StepMeshTensorCommunicator |
| C5 | 端口通过环境变量可配置 | `afd.py` + `scheduler.py` |

### 调度层

| 编号 | 内容 | 位置 |
|------|------|------|
| S1 | `AFDReqInput` 传递 `req_ids` 精确对齐 batch | `io_struct.py` + `scheduler_afd_mixin.py` |
| S2 | `zmq.Poller` 替代 busy-wait | `scheduler.py` event_loop_afd |
| S3 | `AFDReqInput` 含 `seq_lens`/`extend_lens` | `io_struct.py` |

### 计算层

| 编号 | 内容 | 位置 |
|------|------|------|
| G3 | Token-count-aware 贪心切分 | `afd_overlap.py` |
| G5 | 预分配 output tensor，消除 `torch.cat` | `afd.py` model_forward_afd |
| G6 | `NamedTuple`（`StageIO`）替代 `dict` | `afd.py` |
| G4 | `--afd-attn-ratio` 非对称 microbatch 参数 | `server_args.py` |

### 工程化

| 编号 | 内容 | 位置 |
|------|------|------|
| E1 | 通用 Mixin + `AFDWeightFilter` | `afd_mixin.py` |
| E2 | AFD 代码独立文件 | `batch_overlap/afd_overlap.py` |
| E3 | 通用 m 路切分（循环实现） | `afd_overlap.py` |
| E5 | ZMQ 超时 + 异常处理 | `afd.py` |

### 异构 TP

| 编号 | 内容 | 位置 |
|------|------|------|
| Phase 6 | `--afd-attn-tp`/`--afd-ffn-tp` + `ShardedParallelCommunicator`（分片并行 + metadata 传递 + padding 截断） | `server_args.py` + `afd.py` |

### PD+AFD 独立控制

| 编号 | 内容 | 位置 |
|------|------|------|
| Task 6 | P/D 各自独立控制 `--afd-perspective`；`SchedulerAFDMixin` 可复用调度方法；PREFILL/DECODE 分支均支持 AFD | `scheduler_afd_mixin.py` + `prefill.py` + `decode.py` + `scheduler.py` |

---

## 四、支持的模型

| 模型 | 类型 | AFD 状态 | 说明 |
|------|------|----------|------|
| Qwen3-MoE | MoE | 已集成 | Mixin + 权重过滤 |
| Qwen2-MoE | MoE | 已集成 | Mixin + 权重过滤 + Model.forward AFD 分支 |
| DeepSeek-V2/V3 | MoE | 已集成 | Mixin + Model.forward AFD 分支 |
| Qwen3（Dense） | Dense | 已集成 | Mixin + Qwen3Model 重写 forward |
| 其他使用 LayerCommunicator 的模型 | 各种 | 可快速适配 | DecoderLayer 加 `_afd_init()` + `forward_afd_A/F` |
| Qwen2（Dense） | Dense | 不支持 | DecoderLayer 不使用 LayerCommunicator，无法直接适配 |

---

## 五、运行方式

### ZMQ 通信（快速测试）

```bash
# Attn 节点
export CUDA_VISIBLE_DEVICES=0,1,2,3
export AFD_SCHED_HOST=<ffn_ip>

python -m sglang.launch_server \
    --model-path <模型> \
    --disable-overlap-schedule --disable-cuda-graph \
    --afd-perspective attn --afd-micro-batch 3

# FFN 节点
export CUDA_VISIBLE_DEVICES=4,5,6,7
export AFD_SCHED_HOST=<ffn_ip>

python -m sglang.launch_server \
    --model-path <模型> \
    --disable-overlap-schedule --disable-cuda-graph \
    --port <端口> --skip-server-warmup --watchdog-timeout 3600 \
    --afd-perspective ffn --afd-micro-batch 3
```

### StepMesh 通信（高性能 RDMA）

```bash
# 额外设置：
export MLC_INTERFACE=<RDMA 网卡名>
export DMLC_PS_ROOT_URI=<root_ip>
export DMLC_NUM_SERVER=1 && export DMLC_NUM_WORKER=1 && export DMLC_GROUP_SIZE=1
```

### 异构 TP

```bash
# Attn (TP=2): --afd-ffn-tp 8
# FFN  (TP=8): --afd-attn-tp 2
```

### PD + AFD

```bash
# Prefill 服务器 + AFD
python -m sglang.launch_server --disaggregation-mode prefill --afd-perspective attn ...

# Decode 服务器 + AFD
python -m sglang.launch_server --disaggregation-mode decode --afd-perspective attn ...
```

---

## 六、架构限制与待办

| 编号 | 描述 | 状态 |
|------|------|------|
| F1 | StepMesh N:M 双向分片通信（详见下方） | **已实现** |
| F2 | 需要 `--disable-cuda-graph --disable-overlap-schedule` | 待实现 |
| F3 | TBO 与 AFD 互斥，不能同时启用 | 设计决策（可后续扩展） |
| F4 | 自动 profiling + `afd_attn_ratio` 自适应调参 | 待实现 |
| F6 | DeepSeek-V2 `load_weights` 添加 AFDWeightFilter（generator 包装过滤） | **已修复** |
| F7 | `afd_prepare_overlap` 添加 `is_target_verify()` 处理（scheduler.py + scheduler_afd_mixin.py） | **已修复** |
| F8 | `afd_overlap.py` 移除冗余 `m is None` 检查 | **已修复** |
| F9 | F→A 跨节点流量 4NH 是 StepMesh push_pull API 固有下限，降到 NH 需分组 StepMesh 实例 | 远期目标 |
| F5 | Qwen2（不使用 LayerCommunicator 的旧 Dense 模型）不支持 AFD | 需重构 Qwen2DecoderLayer |

---

## 七、F1 StepMesh N:M 双向分片通信

### 设计原理

通过调研 StepMesh 源码（`af_tensor_app.h` + `public.hpp`）发现：

1. **`push_pull` 支持多个 pull_tensor**：`pull_tensors` 数组大小 = `server_count × batch_size`，每个 Server 有独立的 pull 缓冲区，不会覆盖
2. **`get_batch` 按 Worker rank 排序返回**：内部用 `q_[worker_rank]` 队列，返回顺序即 rank 顺序
3. **`respond` 按 key 匹配**：Server respond 的数据写入 Worker 对应 key 的 pull 缓冲区
4. **`RegisterRecvTensor` 支持按 Worker 分片注册**：同一 tensor 的不同 chunk 注册给不同 Worker

### 实现方式

**A→F 方向（分片广播）**：
- `attn_send`：每个 Attn rank 只 push 1/TP_A 的 token shard（广播到所有 FFN）
- 同时注册 **TP_F 个 pull_tensor**（每个 FFN Server 一个独立缓冲区用于接收 F→A shard）
- `ffn_recv`：`get_batch()` 返回 TP_A 个 shard（按 Worker rank 排序），`torch.cat` 拼接

**F→A 方向（分片 respond）**：
- `ffn_send`：每个 FFN rank respond 自己的 1/TP_F shard 给每个 Worker（按 key 匹配写入 Worker 对应的 pull 缓冲区）
- `attn_recv`：`wait` 后所有 TP_F 个 pull_tensor 已填充，`torch.cat` + 截断 padding

### 跨节点流量（4A:8F 示例）

| 方向 | 流量 |
|------|------|
| A→F | 4 × (NH/4) × 8 = 8NH（每个 Attn 发 1/4，广播 8 份） |
| F→A | 8 × (NH/8) × 4 = 4NH（每个 FFN 发 1/8，respond 给 4 个 Worker） |
| **总计** | **12NH**（vs 同构 1:1 的 2NH，vs 全广播的 64NH） |

### 向后兼容

同构 TP（`attn_tp == ffn_tp`）时 `_heterogeneous=False`，所有方法走原始 1:1 代码路径。

### 本轮 Review 已修复的问题

| 编号 | 问题 | 修复 |
|------|------|------|
| H1 | `attn_send` 中 `self.key += 2` 只留 2 个 key 空间，但 pull 需要 TP_F 个 key，第二次调用时 key 碰撞 | 改为 `self.key += 1 + self.ffn_tp` |
| H2 | docstring 描述了旧的"选择性 respond + all_gather"方案 | 更新为多 pull_tensor 方案 |
| H3 | `ffn_recv` 只为第一个 Worker 的 key 注册了 recv buffer | 改为为每个 Worker 的 key 都注册 |

### 遗留事项

| 编号 | 描述 |
|------|------|
| G3 | token 数不整除 TP 时末尾 shard 较短，StepMesh RDMA 写入到较大的 pull_buf 时尾部有未初始化数据。`attn_recv` 中 `full[:original_num_tokens]` 已截断，但 RDMA 层是否要求 respond 数据大小 == pull_buf 大小需实际验证 |
| G6 | `attn_send` 异构路径没有 buffer 复用（每次 `torch.empty_like`），同构路径用 `StepMeshTensorCache` + free pool。异构路径可优化但不影响正确性 |

### F→A 流量优化分析

当前 F→A 流量为 4NH（4A:8F 示例），原因是每个 FFN Server 必须 respond 给每个 Worker（StepMesh `push_pull` 要求所有 pull 请求都被 respond，否则 Worker hang）。

**进一步优化方案**：每个 Worker 只注册 1 个 pull_tensor（只从 1 个 Server 取 1/TP_F shard），Attn 侧 NVLink all_gather 拼回完整 tensor。F→A 跨节点降到 NH，代价是增加一次节点内 all_gather（NVLink ~600GB/s，延迟 5-10us）。

但此方案在 StepMesh 层有限制：`ZPull_` 内部 `pull_tensors.size() / server_count` 计算 `pull_batch_size`，如果 pull_tensor 数 < server_count 则 `pull_batch_size=0`，不发 pull。因此 **Worker 必须传 server_count 的整数倍个 pull_tensor**。

可行的折中方案：传 TP_F 个 pull_tensor 但只让需要的 Server respond 真实数据、其余 respond 空占位（避免 hang），然后 Attn 只取需要的 shard + all_gather。这等价于当前实现 + 加 all_gather 但不减少跨节点流量（仍是 4NH），无实际收益。

**结论**：4NH 是 StepMesh push_pull API 的固有下限（每个 Server 必须 respond 每个 Worker 的 pull）。真正降到 NH 需要分组 StepMesh 实例（极高复杂度），建议作为远期目标。实际部署中 RDMA 200Gbps+ 下 4NH 的绝对延迟仍在 SLA 范围内。

---

## 八、最终 Review（当前轮）

### 发现的问题

| 编号 | 严重度 | 文件 | 问题 |
|------|--------|------|------|
| F6 | **严重** | `deepseek_v2.py` | `load_weights` 缺少 `AFDWeightFilter`，Attn/FFN 节点都加载全部权重，浪费显存 |
| F7 | **中等** | `scheduler.py` + `scheduler_afd_mixin.py` | `afd_prepare_overlap` 未处理 `TARGET_VERIFY` 模式，speculative decoding 下 AFD microbatch 切分不工作 |
| F8 | **低** | `afd_overlap.py` | `m is None` 检查冗余（`_get_afd_micro_batch()` 总返回 int） |

### 已验证为正确的部分

| 检查项 | 状态 |
|--------|------|
| `afd_split_seq_index` 字段流转（ScheduleBatch → ModelWorkerBatch → ForwardBatch） | OK |
| StepMesh `attn_send` key 分配（`self.key += 1 + self.ffn_tp`） | OK |
| StepMesh `ffn_send` respond n_workers 次 | OK |
| StepMesh 异构路径（分片广播 + 多 pull_tensor）| OK |
| `model_forward_afd` 流水线 overlap（`postprocess_layer_start_recv`） | OK |
| Qwen3Model.forward AFD 分支 | OK |
| Qwen2.py 不含 AFD（Qwen2DecoderLayer 无 LayerCommunicator，正确排除） | OK |
| AFDReqInput 继承 BaseReq | OK |
| server_args `afd_micro_batch` 拼写 | OK |
| `event_loop_afd_disagg_prefill/decode` 调用 Mixin 方法顺序 | OK |
| `_compute_token_indices_m_way` 自主计算 | OK |

### 后续任务优先级

| 优先级 | 任务 | 描述 | 状态 |
|--------|------|------|------|
| ~~P0~~ | ~~F6~~ | ~~DeepSeek-V2 添加 AFDWeightFilter~~ | **已修复** |
| ~~P0~~ | ~~F7~~ | ~~`afd_prepare_overlap` 添加 `is_target_verify()` 处理~~ | **已修复** |
| ~~P1~~ | ~~F8~~ | ~~移除冗余 `m is None` 检查~~ | **已修复** |
| P2 | F2 | CUDA Graph 支持 | 待实现 |
| P2 | F3 | TBO + AFD 共存 | 待实现 |
| P3 | F4 | 自动 profiling | 待实现 |
| P3 | F9 | 分组 StepMesh 实现 F→A 流量 NH | 待实现 |
| ~~P3~~ | ~~G3~~ | ~~RDMA 末尾 shard 短于 pull_buf~~ — `ffn_send` 中对末尾 shard padding 到 `chunk_size`，`attn_recv` 中 `full[:original_num_tokens]` 截断 | **已修复** |
| P3 | G6 | 异构路径 buffer 复用 | 待优化 |
