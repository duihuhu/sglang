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
