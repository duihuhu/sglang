# v5 版本 -- Interleaved Schedule + Per-Slot Recv Stream + Per-MB Channel 两阶段初始化

## 概述

本版本完成了三项核心工作:
1. **Interleaved Schedule 调度策略**：替代 AsyncMbDriver 复杂的事件驱动调度，用静态交错 schedule 实现 mb 级流水线
2. **Per-Slot Recv Stream**：每个 recv slot 使用独立 CUDA stream，消除多 recv 的 GPU memcpy 串行化
3. **Per-MB Channel 两阶段初始化**：解决 FFN/Attn 双端 per-mb channel 建连死锁问题

---

## 一、Interleaved Schedule 调度策略（核心变更）

### 1.1 动机

v4 的 `AsyncMbDriver` 使用 per-mb channel + 事件驱动状态机实现异步调度，但引入了大量复杂性：
- 每个 mb 需要独立的通信 channel（端口/SHM 路径隔离）
- 事件驱动循环需要 bg thread + callback + queue 协调
- 调试困难，死锁风险高

**核心洞察**：只要 schedule 本身是交错的（mb0 完成 F 后立即进入下一层 A），就能在单一共享 channel 上实现相同的 compute-communication overlap，无需 per-mb channel。

### 1.2 Interleaved Schedule 设计

对于 M=3, 3 layers 的 Attn 端 schedule：
```
A(0,0) A(0,1) A(0,2)                    -- Layer 0: 所有 mb 的 A-stage
F(0,0) A(1,0) F(0,1) A(1,1) F(0,2) A(1,2)  -- 交错: recv F(prev) → compute A(cur)
F(1,0) A(2,0) F(1,1) A(2,1) F(1,2) A(2,2)  -- 同上
F(2,0) F(2,1) F(2,2)                    -- 最后一层: 只有 F-stage
```

**关键性质**：
- 每个 mb 在收到 F 结果后立即进入下一层 A，不等待其他 mb
- FIFO channel 保证正确性：DA 和 DF 都按 0,1,2 顺序处理 mb
- 无需 per-mb channel，单一共享 channel 即可

### 1.3 实现

新增 `AFDStageScheduleGenerator.attn_stage_interleaved()` 静态方法：
- M=1 退化为普通 batch schedule
- num_layers=1 退化为 A-batch + F-batch
- 一般情况：Layer 0 全 A → 中间层交错 F(prev)+A(cur) → 最后层全 F

### 1.4 调度器集成

- `--afd-async-schedule` 标志现在选择 interleaved schedule（而非 AsyncMbDriver）
- `scheduler.py` 中 `event_loop_afd` 初始化简化：只需初始化普通共享 communicator，无需 per-mb channel
- `model_forward_afd()` 中 AsyncMbDriver 分支标记为 dead code（保留参考）

### 1.5 涉及的代码文件
| 文件 | 改动 |
|------|------|
| `layers/afd.py` | 新增 `attn_stage_interleaved()` 方法 (+55 行) |
| `layers/afd.py` | `model_forward_afd()` 中 `_async_sched_enabled` 选择 interleaved schedule |
| `managers/scheduler.py` | `event_loop_afd` 简化为只初始化共享 communicator |

---

## 二、Preissue Schedule 预计算优化

### 2.1 问题

原有 preissue 逻辑在每次 pipeline 迭代中执行 O(n) 的 `sum()` 和条件判断，对于 64 层 × M=3 = 384 步的 pipeline 有不必要的开销。

### 2.2 解决方案

在 pipeline 循环开始前，一次性预计算 `_preissue_after: list[bool]`：
- 遍历 schedule，模拟 pending recv 计数
- 对 Attn 端：每个 A-stage 后如果 ring 有空位且后续有 F-stage，标记 preissue
- 对 FFN 端：每个 F-stage 后如果下一步是 A-stage 且 ring 有空位，标记 preissue

循环中只需 `if _preissue_after[i]` 即可，O(1) 判断。

### 2.3 涉及的代码文件
| 文件 | 改动 |
|------|------|
| `layers/afd.py` | 新增 `_preissue_after` 预计算逻辑 (+38 行)；删除循环内 O(n) 判断 (-31 行) |

---

## 三、Per-Slot Recv Stream（并行 GPU memcpy）

### 3.1 问题

`AsyncTensorCommunicator` 的所有 recv 都共享单一 `comm_stream`，导致多个 bg recv 完成后的 GPU memcpy 被串行化。

### 3.2 解决方案

为每个 ring slot 分配独立的 CUDA stream (`_recv_streams[idx]`)：
- bg thread 完成 UCX recv 后，在 slot 对应的 stream 上执行 GPU 操作
- 多个 slot 的 memcpy 可以真正并行执行
- `recv_wait()` 时 synchronize 对应 slot 的 event 即可

### 3.3 涉及的代码文件
| 文件 | 改动 |
|------|------|
| `layers/afd.py` | `AsyncTensorCommunicator.__init__` 新增 `_recv_streams` 列表 |
| `layers/afd.py` | `recv_start()` 中 bg thread 使用 `_recv_streams[idx]` 替代共享 `comm_stream` |

---

## 四、Per-MB Channel 两阶段初始化（解决死锁）

### 4.1 问题

v4 中 per-mb channel 串行创建：FFN 端 `listen()` → Attn 端 `connect()`。当 M>1 时，如果 FFN 端还没 listen 完所有 mb 的端口，Attn 端就开始 connect，会导致超时/死锁。

### 4.2 解决方案：Two-Phase Initialization

**FFN 端**：
1. Phase 1: 创建所有 inner comm（`defer_connect=True`），调用 `start_listen()` 绑定端口（非阻塞）
2. Phase 2: 等待所有 Attn 端连接完成（`wait_connected()`）

**Attn 端**：
1. Phase 1: 创建所有 inner comm（`defer_connect=True`）
2. Phase 2: 串行 `connect()` 每个 channel（UCX 内部有重试机制）

### 4.3 UCX 层支持

`_UcxP2PCommunicator` 新增：
- `start_listen()`: FFN 端绑定 listener 但不等待连接
- `wait_connected(timeout)`: 阻塞等待 peer 连接

`UcxTensorCommunicator` 新增：
- `defer_connect` 构造参数：跳过 `connect()` 和 warmup
- `start_listen()` / `wait_connected()` 代理方法

### 4.4 涉及的代码文件
| 文件 | 改动 |
|------|------|
| `layers/rdma_comm.py` | `_UcxP2PCommunicator` 新增 `start_listen()` / `wait_connected()` |
| `layers/rdma_comm.py` | `UcxTensorCommunicator` 新增 `defer_connect` 参数 + 代理方法 |
| `layers/afd_per_mb_channel.py` | `get_per_mb_channel_set()` 重写为两阶段初始化 |

---

## 五、UCX Recv Buffer 管理优化

### 5.1 问题

`_UcxP2PCommunicator._async_recv()` 中 `_last_recv_buf` 机制在 recv 开始时 put 上一次的 buffer 回 pool，但这在 M>1 时可能导致 buffer 被过早回收（上层还在使用）。

### 5.2 修复

移除 `_last_recv_buf` 字段，buffer 生命周期完全由上层 (`AsyncTensorCommunicator`) 管理。recv 只负责从 pool 取 buffer、接收数据、返回 tensor。

### 5.3 涉及的代码文件
| 文件 | 改动 |
|------|------|
| `layers/rdma_comm.py` | 删除 `_last_recv_buf` 字段及相关 put 逻辑 |

---

## 六、Benchmark 配置更新

### 6.1 bench_afd_full_compare.py

新增 IPC backend 的 M=1 配置：
- `M1_IPC_noopt`: IPC 通信 + 无优化
- `M1_IPC_opt`: IPC 通信 + 优化

注释标记 IPC+M=3 在 PD 模式下有死锁问题，暂时跳过。

### 6.2 涉及的代码文件
| 文件 | 改动 |
|------|------|
| `bench_afd_full_compare.py` | 新增 IPC M=1 配置 (+3 行) |

---

## 七、Git 变更汇总

```
Modified (5 files, +257 -68):
  bench_afd_full_compare.py                 +3    IPC M=1 配置
  layers/afd.py                             +164  interleaved schedule + preissue 预计算 + per-slot recv stream
  layers/afd_per_mb_channel.py              +83   两阶段初始化
  layers/rdma_comm.py                       +56   defer_connect + start_listen/wait_connected + 移除 _last_recv_buf
  managers/scheduler.py                     +19   简化 event_loop_afd 初始化
```

---

## 八、当前状态总结

### 已完成
- [x] Interleaved schedule 替代 AsyncMbDriver（简化架构，保持 overlap 效果）
- [x] Preissue schedule 预计算（消除循环内 O(n) 开销）
- [x] Per-slot recv stream（并行 GPU memcpy）
- [x] Per-MB channel 两阶段初始化（解决死锁）
- [x] UCX recv buffer 管理优化（移除 _last_recv_buf）
- [x] IPC M=1 benchmark 配置

### 待完成 (下一步目标)
- [ ] **Interleaved schedule 端到端验证**：在 IPC 下测试 M=3 interleaved 的吞吐提升
- [ ] **IPC send 异步化**：实现 `send_tensor_nonblocking` 消除主线程 ~105μs 阻塞
- [ ] **IPC recv memory ordering**：2-phase recv 中 flag 与数据的写顺序保证
- [ ] **warp 优化**：M=3_async_warp 架构中 SDPA attention fusion
- [ ] **全面 Benchmark**：4-Arch 在 IPC 下的吞吐和延迟对比
