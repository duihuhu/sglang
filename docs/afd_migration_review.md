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
| F1 | StepMesh 尚未适配 N:M 拓扑（仅 ZMQ 路径支持分片并行） | 待实现 |
| F2 | 需要 `--disable-cuda-graph --disable-overlap-schedule` | 待实现 |
| F3 | TBO 与 AFD 互斥，不能同时启用 | 设计决策（可后续扩展） |
| F4 | 自动 profiling + `afd_attn_ratio` 自适应调参 | 待实现 |
| F5 | Qwen2（不使用 LayerCommunicator 的旧 Dense 模型）不支持 AFD | 需重构 Qwen2DecoderLayer |
