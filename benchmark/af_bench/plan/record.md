目前对比了 PD+AF 方案和纯 PD 方案的性能（Qwen3-32B, 64 layers, A800-SXM4-80GB，单请求，禁用 CUDA Graph + Radix Cache）。

优化前数据：AF M=1（IPC）的 TTFT（513 tokens）为 277ms，TPOT 为 118ms，对比 PD TP=1 的 TTFT 217ms / TPOT 67ms，分别慢 28% 和 76%；对比 PD TP=2 的 TTFT 153ms / TPOT 41ms，分别慢 81% 和 2.9x。分析发现主要开销来自两方面：一是 Python 循环调度开销每层 ~0.28ms（CUDA event 创建、dict 更新、deque 操作等），64 层 x 2 stages 累计约 36ms；二是 IPC 通信的 event.synchronize() 协议开销每次 ~100us。

优化措施：M=1 时走 fast-path 直连循环（跳过 pipeline schedule/deque/context/preissue）；timing CUDA event 默认关闭；修复 IPC device_id 映射和 peer access 启用。

优化后：Python overhead 从 36ms 降为 0，TTFT（513 tokens）从 277ms 降到 253ms（-9%），TPOT 从 118ms 降到 94ms（-21%），TPOT 与 PD TP=1 的差距从 1.76x 缩至 1.40x。但瓶颈仍在：逐层分析显示 AF 每层比 PD TP=1 多 531us，其中 83%（441us）来自 recv wait（等远程 FFN + IPC 返回），24%（126us）来自 IPC send 的 event.synchronize() 开销。NVLink 带宽实际不是瓶颈（2.5MB DMA 仅需 8us），根因在 CPU-GPU 同步延迟。

下一步计划：用 torch.distributed.ProcessGroupNCCL（NCCL P2P）替代裸 IPC，目标将通信开销从 170us/send 降到 30-40us/send，力争 TPOT 从 94ms 进一步降到 ~80ms。

---

## 阶段总结：C++ IPC 通信协议优化 (2026-05-24)

在 Python IPC 优化（TPOT 118→94→81ms）的基础上，用 C++ 重写了整个通信协议，最终将 TPOT 从 81ms 降到 47ms，与 PD TP=1 baseline（44ms）仅差 6%。

优化分两步完成：

第一步，C++ 热路径替代 Python（81→69ms，-15%）：用 pybind11 封装 SHM ring buffer + CUDA IPC handle 交换 + cudaMemcpyPeerAsync 为 `.so` 库，热路径零 Python 对象分配，消除了 GIL 竞争和函数调用框架开销。每层通信开销从 0.56ms 降到 0.39ms。

第二步，消除 cudaStreamSynchronize + clone（69→47ms，-32%）：(1) metadata 改为通过 SHM 直接传递（CPU 读写，不走 GPU buffer），无需 D2H copy 和同步；(2) recv 端用 cudaStreamWaitEvent 替代 cudaStreamSynchronize，GPU 通过 IPC Event 自行保证数据可见性，CPU 完全不阻塞；(3) recv_tensor 直接返回 recv_pool 的 buffer view，不做 clone。每层通信开销从 0.39ms 降到 0.04ms。

最终性能（单请求 decode，Qwen3-32B，GPU 4-7）：TPOT = 46.6ms，vs PD TP=1 的 44.0ms，overhead 仅 +6%。多并发场景下 TPOT overhead 稳定在 +6~7%，同时 TTFT 在 conc≥2 时比纯 PD 快 34~37%（因为 prefill/decode 使用不同 GPU 组互不阻塞）。

代码位置：`sgl-kernel/csrc/afd_ipc/`（C++ 核心）+ `python/sglang/srt/layers/afd_ipc_cpp/`（Python 集成），通过 `--afd-comm-backend ipc_cpp` 启用。

---

## 进展同步（2026-05-24）

PD+AF M=1 通信优化完成。经过四轮迭代，TPOT 从初始 118ms 降至 47ms，与纯 PD TP=1 baseline（44ms）仅差 6%。优化路径：Python fast-path 消除调度开销（118→93ms）；Pre-launch 缓存跳过 metadata 编解码（93→81ms）；C++ pybind11 重写通信协议消除 Python/GIL 开销（81→69ms）；最后用 CUDA IPC Event 纯 GPU 同步替代 cudaStreamSynchronize、metadata 走 SHM 免 D2H copy、返回 buffer view 免 clone（69→47ms）。每层通信开销从 1.14ms 降至 0.04ms，减少 96%。多并发测试显示 TPOT overhead 稳定 +6~7%，同时 TTFT 在 conc≥2 时比纯 PD 快 34~37%，因为 AF 分离让 prefill/decode 使用不同 GPU 组互不阻塞。
