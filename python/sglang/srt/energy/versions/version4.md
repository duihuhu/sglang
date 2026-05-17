# v4 版本 -- UCX Protocol 修复 + IPC 通信器 + M=3 异步调度架构

## 概述

本版本完成了三项核心工作:
1. **修复 UCX M=3 interleaving bug**，恢复 100% 成功率
2. **开发 CUDA IPC 通信器** (ipc_comm.py)，替代 UCX over TCP
3. **构建完全异步 M=3 pipeline 基础设施**：per-mb channel + AsyncMbDriver + 2-phase recv

---

## 一、Bug 修复：UCX Protocol Interleaving（已完成）

### 1.1 问题
M=3 多请求下 UCX send coroutine 交错执行 (meta_0→meta_1→data_0→data_1)，接收端解析错误 → OOM/hang。

### 1.2 修复
`_async_send()` / `_async_recv()` 加 `asyncio.Lock`，保证 meta+data 原子性。

### 1.3 涉及的代码文件
| 文件 | 改动 |
|------|------|
| `layers/rdma_comm.py` | `_async_send` 加 `asyncio.Lock`; `_async_recv` 加 `_recv_lock_async` |

---

## 二、M=1 vs M=3 性能对比实测（已完成）

### 2.1 测试配置

| 参数 | 值 |
|------|------|
| 模型 | Qwen3-32B (64 layers) |
| GPU | DA=GPU0+1, DF=GPU4+5 (TP=2) |
| 通信 | UCX over TCP (loopback 127.0.0.1) |
| max_running | 64 |
| input_len=1024, output_len=128, QPS=8, 100 requests |

### 2.2 实测结果

| 指标 | M=1 | M=2 | M=3 | M=3/M=1 |
|------|----:|----:|----:|--------:|
| **Output Throughput** | **225.1** | **169.9** | **118.6** | **0.53x** |
| Mean TPOT (ms) | 181.0 | 244.2 | 380.4 | 2.1x |
| Wall Duration (s) | 56.9 | 74.7 | 107.9 | 1.9x |

### 2.3 结论
**M>1 在 UCX over TCP 下无法获得吞吐收益。** 瓶颈是 comm round-trip (~450-670μs/次) 远超 per-mb attn kernel (~200μs@batch=21)。

---

## 三、4-Architecture Benchmark 框架（已完成）

新增 `bench_afd_full_compare.py` 对比 4 种架构:

| 架构 | 通信层 | 调度层 | 含义 |
|------|--------|--------|------|
| `M=1` | 单 shared comm (UCX/IPC/ZMQ) | 静态 Schedule (M=1) | 基准 |
| `M=3` | 单 shared comm | 静态 Schedule (M=3) | 经典 pipeline |
| `M=3_async` | M 个独立 per-mb channel | AsyncMbDriver (事件驱动) | 异步调度 |
| `M=3_async_warp` | M 个独立 per-mb channel | AsyncMbDriver + warp 优化 | 异步 + attention SDPA kernel 融合 |

输出对比表 (TTFT / TPOT / throughput / success rate / wall duration) + per-step 时间线。

---

## 四、IPC 通信器：ipc_comm.py（已完成）

### 4.1 架构
替代 UCX-Py，使用直接 CUDA IPC + POSIX 共享内存实现单节点 A↔F 通信：

- **数据传输**: `cudaMemcpyPeer` over NVLink (跨设备拷贝)
- **同步**: POSIX `/dev/shm` 共享内存中的 per-slot volatile uint64 标志位 (纳秒级轮询)
- **Ring buffer**: RING_SIZE=4 个 slot，支持 M≤3 的流水线

### 4.2 SHM 布局
```
offset 0-31:   flag_a2f[0..3]  (4 × uint64)
offset 32-63:  flag_f2a[0..3]  (4 × uint64)
offset 64-95:  size_a2f[0..3]  (4 × uint64)
offset 96-127: size_f2a[0..3]  (4 × uint64)
```

### 4.3 关键设计决策
- **2-phase recv** (recv_poll + recv_complete)：bg 线程只做 flag 轮询 (无 GPU 操作)，主线程做 GPU copy + event sync (~80μs)。避免了 `event.synchronize()` 在 bg 线程中阻塞 GIL 的问题 (10-40ms)。
- **Send 走 comm_stream** (Path 2)：主线程阻塞 ~105μs，在 400-550μs GPU compute 下可接受。
- **每条消息带 size**：接收端直接从 SHM 读取数据大小，一次 cudaMemcpyPeer 拷贝 header+data。
- **IPC handshake via Unix socket**：FFN 端 listen，Attn 端 connect，交换 `_share_cuda_()` 导出的 IPC handle。

### 4.4 涉及的代码文件
| 文件 | 改动 |
|------|------|
| `layers/ipc_comm.py` | **新增**: CUDA IPC 通信器 `IpcTensorCommunicator` |
| `layers/afd.py` | `get_tensor_communicator()` 添加 `ipc` backend 分支 |

---

## 五、Async Send Pipeline 优化（已完成）

### 5.1 Persistent Sender Daemon
替代每次 send 创建新 daemon thread 的旧方案。单 persistent 线程持有 CUDA context，从 `queue.Queue` 取 `(tensor, compute_event)` 执行 send。

### 5.2 Pre-allocated CUDA Event Pool
预分配 16 个 `torch.cuda.Event(enable_timing=False)`，避免每次 send 创建 Event 对象 (~15μs)。

### 5.3 UCX Stream-Ordered Send (send_tensor_nonblocking_stream_ordered)
UCX send 的新路径：在调用线程做 `comm_stream.synchronize()` (~5μs) 确认 GPU kernel 完成，然后提交纯异步 UCX send 到 bridge event loop。消除了旧 daemon-thread 方案的 `event.synchronize()` CPU 开销 (38-288μs)。

### 5.4 Drain/Fence 机制
```python
drain_sends():
  - queue.join()  # 等待 persistent daemon 清空队列
  - 等待所有 UCX Future 完成
  - inner.fence()  # 最终 GPU sync
```

### 5.5 涉及的代码文件
| 文件 | 改动 |
|------|------|
| `layers/afd.py` | `send_async()` 重写: event pool + persistent daemon; `recv_wait()` 支持 2-phase IPC; `drain_sends()` 改版 |
| `layers/rdma_comm.py` | `send_nonblocking` / `send_nonblocking_stream_ordered` / `send_tensor_nonblocking_stream_ordered` 新增 profiling 和流排序支持 |

---

## 六、Per-MB Channel + AsyncMbDriver（进行中）

### 6.1 afd_per_mb_channel.py
将单一共享 `AsyncTensorCommunicator` 拆分为 M 个独立 per-mb channel：

- 每个 channel 持有独立的 `FifoTensorCommunicator` 后端（不同 port/SHM path/ZMQ port）
- `PerMbChannel` 封装 `send_async` / `recv_start` / `recv_wait` 语义
- `recv_start()` 使用 watcher daemon 监听 bg recv 线程完成，通过 callback 通知 driver
- `MultiMbChannelSet` 管理 M 个 channel 的工厂和生命周期
- 支持所有 backend (UCX/IPC/ZMQ)，自动通过 `mb_id` 区分端口/SHM 路径

### 6.2 afd_async_sched.py (AsyncMbDriver)
数据驱动的 M=3 异步调度器，替代静态 `AFDStageScheduleGenerator`:

```
每个 micro-batch 是独立的状态机:
  _State.READY_TO_COMPUTE  →  执行 A/F step  →  发送结果 (fire-and-forget)
  _State.WAIT_RECV         →  等待 recv 完成  →  READY_TO_COMPUTE
  _State.DONE              →  完成所有层
```

主循环：
1. 任选一个 READY_TO_COMPUTE 的 mb (按 layer 优先, mb 次之)
2. 选择该 mb 的 per-mb channel，设置 `_per_mb_async_override`
3. 调用 `layer.forward_afd_A()` / `layer.forward_afd_F()`
4. 如果下一步需要 recv，`recv_start()` 提交背景 recv，状态转为 WAIT_RECV
5. 如果无 READY 的 mb，阻塞在 `_recv_done.get()` (queue)

### 6.3 涉及的代码文件
| 文件 | 改动 |
|------|------|
| `layers/afd_per_mb_channel.py` | **新增**: PerMbChannel + MultiMbChannelSet |
| `layers/afd_async_sched.py` | **新增**: AsyncMbDriver 事件驱动调度器 |
| `layers/afd.py` | `get_async_communicator()` 支持 `_per_mb_async_override`; `model_forward_afd()` 支持 `afd_async_schedule` flag; schedule 改为 batch-A-then-batch-F |
| `disaggregation/decode.py` | `afd_async_schedule` 时初始化 `MultiMbChannelSet` |
| `disaggregation/prefill.py` | 同上 |
| `server_args.py` | 添加 `--afd-async-schedule` 参数 |

---

## 七、Tier 1 监控 + TPOT Breakdown Timing（已完成）

### 7.1 监控指标
- `[AFD_PER_STEP]` 日志：每次 forward 的 per-step GPU 时间分解（各 sub-stage elapsed time + wall clock）
- `[AFD_TIMELINE]` 日志：per-stage wall-clock 时间线
- `_afd_host_events`：主机端 profiling (send/recv 各阶段微秒级分解)
- `_afd_timing_records`：GPU CUDA event 汇总 (整体 TPOT 分解)

### 7.2 涉及的代码文件
| 文件 | 改动 |
|------|------|
| `layers/afd.py` | `_log_afd_breakdown()` 扩展：per-step JSON + wall-clock alignment |
| `layers/afd_mixin.py` | `_afd_host_events` / `_afd_sched_ts` 全局 profiling context |
| `layers/afd.py` | `recv_start()` / `recv_wait()` / `recv_sync()` 添加详细 profiling |

---

## 八、Git 待提交变更汇总

```
Modified (12 files, +905 -130):
  disaggregation/decode.py       +43   per-mb channel 初始化
  disaggregation/prefill.py      +43   同上
  energy/af_launch_config.json   +6   新增 async 架构配置
  energy/af_launcher.py          +63   4-arch benchmark launch 逻辑
  layers/afd.py                  +596  核心变更：async send/recv pipeline、per-mb override、IPC backend、batch schedule、per-step profiling
  layers/afd_mixin.py            +25   profiling context
  layers/communicator.py         +2    import
  layers/rdma_comm.py            +183  UCX lock 修复、stream-ordered send、mb_id 端口隔离、profiling
  layers/rotary_embedding/base.py +2   小修复
  managers/scheduler.py          +56   支持 afd_async_schedule
  models/utils.py                +2    小修复
  server_args.py                 +14   --afd-async-schedule 参数

New files (untracked):
  layers/ipc_comm.py             CUDA IPC 通信器 (cudaMemcpyPeer + SHM flags)
  layers/afd_async_sched.py      AsyncMbDriver: 数据驱动事件循环
  layers/afd_per_mb_channel.py   PerMbChannel + MultiMbChannelSet 工厂
  test/srt/test_afd_async_sched.py  AsyncMbDriver 测试
  bench_afd_async_e2e.py         4-arch 端到端 benchmark
  bench_afd_async_smoke.py       Smoke test
  bench_afd_full_compare.py      4-arch 对比框架
  bench_pdaf_compare_v2.py       PD+AF 对比 v2
  test/srt/test_ipc_m3.py        IPC M=3 测试
  test/srt/test_ipc_quick.py     IPC 快速测试
```

---

## 九、当前状态总结

### 已完成
- [x] UCX M=3 interleaving lock 修复
- [x] CUDA IPC 通信器 (`ipc_comm.py`)：单 channel 模式可 work
- [x] Async send pipeline (persistent daemon + event pool + stream-ordered send)
- [x] Per-mb channel 工厂 (`afd_per_mb_channel.py`)
- [x] AsyncMbDriver 事件驱动调度器 (`afd_async_sched.py`)
- [x] 4-Arch benchmark 框架
- [x] Tier 1 监控 + TPOT per-step breakdown
- [x] IPC 2-phase recv (poll + complete) 解决 bg-thread GIL 阻塞

### 待完成 (下一步目标)
- [ ] **IPC M=3 完全异步测试与调试**：确保 per-mb channel + AsyncMbDriver 端到端可运行
- [ ] **Send 完全去阻塞**：当前 IPC send 仍然是同步的 (走 comm_stream, 主线程阻塞 ~105μs)。需要实现 IPC 端的真正 nonblocking send
- [ ] **recv_poll 返回后 GPU 数据就绪保证**：2-phase recv 中，bg 线程 poll flag 后，主线程做 cudaMemcpyPeer。需要在 flag 写入和数据写入之间加 CUDA event/memory fence 保证写顺序
- [ ] **全面 Benchmark**: M=1 / M=3 / M=3_async / M=3_async_warp 在 IPC 下的吞吐和延迟
- [ ] **warp 优化**：M=3_async_warp 架构中 attention kernel 融合 (SDPA) 的实现和测试

---

## 十、下一步规划：实现完全异步的 M=3 流水线

### 目标
在 IPC 通信层上实现 **零阻塞 M=3 pipeline**：
- 每个 micro-batch 的 A/F compute 和 send/recv 完全 overlap
- 主线程只做 "pick ready mb + fire compute" 和 "GPU copy for completed recv"
- 预期效果：M=3 throughput 超越 M=1 (在 batch≤10 场景)

### 待攻克的技术问题

#### 1. IPC Send 异步化
当前 IPC send 是同步的 (`send_tensor` 在 `comm_stream` 中阻塞主线程 ~105μs)。
需要实现 IPC `send_tensor_nonblocking`：
- GPU copy (`send_buf.copy_(x)`) 提交到 comm_stream
- `event.record()` + flag write 在 GPU copy 完成后由 persistent daemon 完成
- 主线程只做 `event.record()` → `queue.put_nowait((slot, x))` (~3μs)

**接口设计**：
```python
# ipc_comm.py
def send_tensor_nonblocking(self, x, compute_event=None):
    """
    提交 GPU copy 到 comm_stream，放入 daemon 队列。
    主线程代价 ~3-5μs。
    """
    slot = self._send_slot
    self._send_slot = (slot + 1) % self.RING_SIZE
    self._send_queue_for_daemon.put_nowait((slot, x, compute_event))
```

Daemon 线程：
```python
def _send_daemon_loop(self):
    while True:
        slot, x, ev = self._send_queue.get()
        if ev: ev.synchronize()
        # GPU copy on bg_stream
        with torch.cuda.stream(self._bg_stream):
            self._send_buf[slot][:n].copy_(x.view(uint8))
            self._send_event[slot].record()
        self._send_event[slot].synchronize()
        # write flag + size to SHM
        self._write_u64(size_off, n)
        self._write_u64(flag_off, 1)
```

#### 2. IPC recv_poll 排序保证
当前 2-phase recv 有潜在的 race condition：
- Sender: `copy_(send_buf)` → `event.record()` → `event.synchronize()` → `write flag`
- 但 `cudaMemcpyPeer` (recv 侧) 需要在 sender 的 write 序后被保证已写入 peer memory。

**修复**：Sender 端在 `event.synchronize()` 之后加 `__threadfence()` 或使用 `cudaMemcpyPeer` 的同步语义保证 peer memory visible。

#### 3. AsyncMbDriver 集成 IPC channel
在 PerMbChannel 中，每个 channel 持有独立的 `AsyncTensorCommunicator(IpcTensorCommunicator(perspective, mb_id=N))`。
- `recv_start()` → 调用 `AsyncTensorCommunicator.recv_start()` → bg thread 调用 `IpcTensorCommunicator.recv_poll()` (只有 flag polling)
- `recv_wait()` → 调用 `AsyncTensorCommunicator.recv_wait()` → 主线程调用 `IpcTensorCommunicator.recv_complete()` (GPU copy + event sync)

#### 4. 验证 M=3 async 收益
关键条件：
```
max(comm_roundtrip/M, compute_per_mb) < compute_per_layer (M=1)
```
IPC 下 comm 降到 ~80μs (cudaMemcpyPeer)，compute_per_mb ~200μs (attn, batch=7)，compute_per_layer(M=1) ~900μs。
→ 理论上可实现 send/recv 完全 overlap compute。

### 实现路线图

| 阶段 | 任务 | 预估工作量 |
|------|------|-----------|
| Phase 1 | IPC `send_tensor_nonblocking` + persistent send daemon | 2-3h |
| Phase 2 | IPC 2-phase recv memory ordering 修复 | 1-2h |
| Phase 3 | AsyncMbDriver + PerMbChannel 端到端集成测试 | 2-3h |
| Phase 4 | 4-Arch benchmark 全面对比 (IPC 下) | 1-2h |
| Phase 5 | warp 优化 (SDPA attention fusion) | 2-3h |
