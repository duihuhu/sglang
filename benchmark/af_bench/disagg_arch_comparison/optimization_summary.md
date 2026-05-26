# AF 分离通信优化总结

**模型**: Qwen3-32B, 64 layers, A800-SXM4-80GB  
**Baseline**: PD TP=1 (2 GPU), TPOT ≈ 44ms

## 优化历程

| 阶段 | TPOT | vs Baseline | 每层通信开销 | 优化内容 |
|------|------|-------------|------------|---------|
| 初始版本 | 117.8 ms | +160% | 1.14 ms | 原始 Python IPC |
| Fast-Path | 93.3 ms | +106% | 0.75 ms | M=1 fast-path 循环，跳过 schedule/deque/event 创建 |
| Pre-launch 缓存 | 80.8 ms | +79% | 0.56 ms | 缓存 metadata encode/decode，event.query busy-wait |
| C++ IPC | 68.9 ms | +56% | 0.39 ms | C++/pybind11 重写通信协议，消除 Python 热路径 |
| **消除 sync+clone** | **46.6 ms** | **+6%** | **0.04 ms** | cudaStreamWaitEvent 纯 GPU 同步 + buffer view 零拷贝 |

## PD+AF vs 纯 PD 多并发对比

**条件**: GPU 4-7, input=512, output=128, 40 requests per level

| Concurrency | PD+AF TPOT | PD TP=1 TPOT | TPOT Overhead | PD+AF TTFT | PD TP=1 TTFT | TTFT 改善 |
|-------------|-----------|-------------|---------------|-----------|-------------|----------|
| 1 | 46.8 ms | 44.3 ms | +5.6% | 174.9 ms | 163.4 ms | +7.0% |
| 2 | 48.3 ms | 45.7 ms | +5.7% | **187.3 ms** | 284.2 ms | **-34.1%** |
| 4 | 49.2 ms | 46.2 ms | +6.5% | **250.0 ms** | 400.7 ms | **-37.6%** |
| 8 | 50.9 ms | 47.4 ms | +7.4% | **417.5 ms** | 657.7 ms | **-36.5%** |

## 最优配置（ipc_event 模式）启动参数

达到 TPOT 46.6ms（+6% overhead）的完整启动配置：

### 环境变量

| 变量 | 值 | 说明 |
|------|------|------|
| `CUDA_VISIBLE_DEVICES` | `<gpu_a>,<gpu_b>` | 同一对 AF 节点共享两张 GPU（如 `4,5` 或 `6,7`） |
| `AFD_IPC_SYNC_MODE` | `ipc_event` | **关键**：使用 cudaEventRecord + cudaStreamWaitEvent 纯 GPU 同步，消除 cudaStreamSynchronize 阻塞 |
| `AFD_IPC_PEER_DEVICE` | `0` 或 `1` | 对端 GPU 在 CUDA_VISIBLE_DEVICES 中的索引（FFN 填对端 ATTN 的 id，反之亦然） |
| `AFD_SCHED_PORT` | `<port>` | AFD 调度器端口（同一 PD 对内的 A/F 节点需一致） |
| `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` | `600` | Mooncake bootstrap 超时（秒） |
| `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` | `600` | 等待对端超时（秒） |
| `SGLANG_DISABLE_REQUEST_LOGGING` | `true` | 关闭请求日志减少开销 |

### 启动命令参数

```bash
python -m sglang.launch_server \
    --model-path <model> \
    --tp 1 \
    --host 127.0.0.1 --port <port> \
    --afd-perspective <ffn|attn> \
    --afd-comm-backend ipc_cpp \
    --afd-micro-batch 1 \
    --mem-fraction-static 0.85 \
    --disaggregation-mode <prefill|decode> \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-bootstrap-port 18999 \
    --disaggregation-ib-device mlx5_4 \
    --base-gpu-id <0|1> \
    --skip-server-warmup \
    --disable-cuda-graph \
    --disable-piecewise-cuda-graph \
    --max-running-requests 128
```

### 关键参数说明

| 参数 | 说明 |
|------|------|
| `--afd-comm-backend ipc_cpp` | 使用 C++ pybind11 IPC 通信后端（vs `ipc` Python 版本） |
| `--afd-micro-batch 1` | M=1 触发 fast-path 循环，跳过 pipeline 调度开销 |
| `--disable-cuda-graph` | AF 模式下 CUDA Graph 与 IPC 通信不兼容，必须关闭 |
| `--base-gpu-id` | 当前进程使用的 GPU 在 CUDA_VISIBLE_DEVICES 中的索引 |
| `AFD_IPC_SYNC_MODE=ipc_event` | **核心优化**：send 端 `cudaEventRecord` 后写 SHM flag（不 sync），recv 端 `cudaStreamWaitEvent` 在 compute stream 上等待，GPU 硬件保证顺序，完全消除 CPU-GPU 同步 |

### 4-GPU 典型部署示例（PF+PA+DF+DA）

```
# Prefill-FFN (GPU4)
CUDA_VISIBLE_DEVICES=4,5 AFD_IPC_SYNC_MODE=ipc_event AFD_IPC_PEER_DEVICE=1 \
  --afd-perspective ffn --disaggregation-mode prefill --base-gpu-id 0

# Prefill-ATTN (GPU5)
CUDA_VISIBLE_DEVICES=4,5 AFD_IPC_SYNC_MODE=ipc_event AFD_IPC_PEER_DEVICE=0 \
  --afd-perspective attn --disaggregation-mode prefill --base-gpu-id 1

# Decode-FFN (GPU6)
CUDA_VISIBLE_DEVICES=6,7 AFD_IPC_SYNC_MODE=ipc_event AFD_IPC_PEER_DEVICE=1 \
  --afd-perspective ffn --disaggregation-mode decode --base-gpu-id 0

# Decode-ATTN (GPU7)
CUDA_VISIBLE_DEVICES=6,7 AFD_IPC_SYNC_MODE=ipc_event AFD_IPC_PEER_DEVICE=0 \
  --afd-perspective attn --disaggregation-mode decode --base-gpu-id 1
```

### 验证结果（2026-05-25 实测）

| Input | Output | TPOT | vs Baseline (44ms) |
|-------|--------|------|-------------------|
| 128 | 128 | 46.3 ms | +5.2% |
| 512 | 128 | 46.3 ms | +5.2% |
| 1024 | 128 | 46.4 ms | +5.5% |
| 2048 | 128 | 46.9 ms | +6.6% |
