# AFlex Graceful Reshard — IPC Reconnect 实现

## 概述

本模块实现了 AFlex (AF Disaggregation) 架构下的**单组件在线扩缩容**能力。核心创新是 **IPC Reconnect** 机制，允许在不重启整个 Prefill pair 的情况下，单独对 PA（Prefill Attention）进行 TP 扩展。

传统方案需要同时重启 PA + PF（因为 IPC 通信是一次性 handshake），导致 6 GPU 的方案（新开一整套 PA+PF）。IPC Reconnect 让 PF 能在 PA 重启后重新建立连接，实现真正的 5 GPU reshard（仅扩展 PA）。

## 架构

```
Before (4 GPU):
  PF(TP1, GPU0) ←IPC→ PA(TP1, GPU1)   |   DF(TP1, GPU2) ←IPC→ DA(TP1, GPU3)

After reshard PA TP1→TP2 (5 GPU):
  PF(TP1, GPU0) ←IPC→ PA(TP2, GPU1+GPU4)   |   DF(TP1, GPU2) ←IPC→ DA(TP1, GPU3)
                  ↑
          IPC Reconnect: PF 不重启，仅重新 handshake
```

## Reshard 流程

```
时间 →

[正常服务] → [Export IPC] → [Drain] → [Kill PA] → [PF Reconnect] → [Start New PA] → [Handshake] → [Activate] → [恢复服务]
              0.02s          2s         3s           0.3s             ← 等待 PA 启动 22s →              0s
                                                                                                         
              ←────── 中断期间 PF 空闲但不处理请求 ──────→                                              
```

### 各步骤说明

| 步骤 | 耗时 | 描述 |
|------|------|------|
| Export IPC handles | ~0.02s | 从旧 PA 导出 weight 的 CUDA IPC handles |
| Drain + Kill | ~5s | Router 停止分发请求到旧 PA，等待飞行请求完成，kill 旧 PA |
| PF IPC Reconnect | ~0.3s | PF 的 scheduler 销毁旧 IPC comm，创建新的并开始 listen |
| New PA startup | ~22s | 新 PA 启动（可通过 IPC fast path 加载 weight 跳过磁盘 I/O） |
| Activate | ~0s | Router 切换流量到新 PA |

## 代码修改

### 核心文件

| 文件 | 修改 |
|------|------|
| `python/sglang/srt/layers/afd_ipc_cpp/communicator.py` | `CppIpcTensorCommunicator` 新增 `reconnect()` 和 `cleanup()` |
| `python/sglang/srt/layers/afd.py` | `BroadcastTensorCommunicator.reconnect()` + 全局 `trigger_ipc_reconnect()` |
| `python/sglang/srt/managers/scheduler_update_weights_mixin.py` | `_ipc_reconnect()` 处理 scheduler 端 reconnect |
| `python/sglang/srt/managers/io_struct.py` | `ReshardReqInput.action` 新增 `"ipc_reconnect"` |
| `python/sglang/srt/entrypoints/http_server.py` | `/admin/ipc_reconnect` HTTP endpoint |
| `sgl-kernel/csrc/afd_ipc/afd_ipc_pybind.cpp` | `handshake()` 添加 GIL release |

### 关键设计决策

1. **通过 Scheduler 通道执行 reconnect**：HTTP server 和 Scheduler 运行在不同进程中，IPC communicator 存在于 Scheduler 进程。通过 `ReshardReqInput` → ZMQ → Scheduler 的管理通道发送 reconnect 命令。

2. **GIL Release**：C++ `handshake()` 中的 `accept()` 是 blocking I/O，必须释放 GIL 否则会阻塞整个 Python 进程（包括 scheduler 的事件循环）。

3. **后台线程 handshake**：`reconnect()` 创建新 comm 后在后台线程中执行 handshake（listen + accept），主线程立即返回。新 PA 连接上来后 handshake 完成，下一次 forward pass 的 `_wait_ready()` 会检测到 ready 状态。

## 测试结果

### PA TP1 → TP2 → TP4 扩展 (Qwen3-32B, 2048 tokens prefill)

| PA TP | GPU 数量 | Avg TTFT | 成功率 |
|-------|----------|----------|--------|
| TP1   | 4        | 1467ms   | 10/10  |
| TP2   | 5        | 1417ms   | 10/10  |
| TP4   | 7        | 1410ms   | 10/10  |

### Reshard 性能

| 步骤 | TP1→TP2 | TP2→TP4 |
|------|---------|---------|
| IPC Reconnect | 0.44s | 0.33s |
| New PA Startup | 22.0s | 24.0s |
| Total | 27.5s | 29.4s |

TTFT 改善不大是因为 2048 token prefill 场景中 FFN（PF 仍为 TP1）是瓶颈，PA attention 计算本身占比较小。对于更长序列（如 8K+ tokens），attention 占比增大，PA TP 扩展的收益会更显著。

## 运行测试

```bash
# 在 Docker 容器内执行
cd /workspace/sglang

# 5 GPU reshard 测试（PA TP1→TP2，带 QPS=1 连续流量）
python3 benchmark/AFlex_bench/reshard/test_correct_reshard_5gpu.py

# PA 连续扩展测试（TP1→TP2→TP4）
python3 benchmark/AFlex_bench/reshard/test_reshard_pa_scaling.py
```

## 后续工作

- TP4→TP8 扩展（需要 8 GPU 全部给 PA，或跨节点）
- 减少 New PA 启动时间（IPC fast path weight 继承优化）
- PF 扩展（PF TP1→TP2）的对称实现
- Decode 侧（DA/DF）的独立扩展
