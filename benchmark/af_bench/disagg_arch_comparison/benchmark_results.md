# PD+AF vs PD TP=1 vs PD TP=2 性能对比报告

**测试日期**: 2026-05-25  
**模型**: Qwen3-32B (64 layers, hidden=5120, FFN=27648, bf16)  
**硬件**: A800-SXM4-80GB × 4 (GPU 4-7), NVLink 400GB/s  
**PD+AF 配置**: ipc_event 同步模式, M=1, C++ IPC 后端

---

## 一、架构说明

| 方案 | GPU 分配 | Prefill | Decode | 总 GPU 数 |
|------|---------|---------|--------|----------|
| PD TP=1 | P=GPU4, D=GPU5 | 1 GPU (full model) | 1 GPU (full model) | 2 |
| PD+AF M=1 | PF=GPU6, PA=GPU7, DF=GPU4, DA=GPU5 | 2 GPU (FFN+ATTN 分离) | 2 GPU (FFN+ATTN 分离) | 4 |
| PD TP=2 | P=GPU6,7, D=GPU4,5 | 2 GPU (tensor parallel) | 2 GPU (tensor parallel) | 4 |

---

## 二、单请求性能 (concurrency=1)

### Input 扫描 (output=128, PD+AF ipc_event 模式)

| Input | PD+AF TTFT | PD+AF TPOT | PD+AF 吞吐 |
|-------|-----------|-----------|-----------|
| 128 | 67.7ms | 46.4ms | 21.5 tok/s |
| 256 | 98.2ms | 46.5ms | — |
| 512 | 144.0ms | 46.6ms | 21.1 tok/s |
| 1024 | 217.8ms | 46.6ms | — |
| 2048 | 382.0ms | 46.8ms | — |

**结论**: TPOT 与 input 长度无关（46.4~46.8ms），仅 TTFT 随 input 线性增长。

---

## 三、多并发性能对比

### 3.1 Input=512, Output=128 (公平对比, 无 max_running_requests)

| Conc | PD+AF TPOT | PD TP=2 TPOT | PD TP=1 TPOT | PD+AF 吞吐 | PD TP=2 吞吐 | PD TP=1 吞吐 |
|------|-----------|-------------|-------------|-----------|-------------|-------------|
| 1 | 46.6ms | 39.8ms | 44.1ms | 21.1 | 24.8 | 22.3 |
| 2 | 48.2ms | 42.3ms | 45.7ms | 40.6 | 45.9 | 42.1 |
| 4 | 49.2ms | 42.2ms | 46.2ms | 78.9 | 90.2 | 81.7 |
| 8 | 50.8ms | 41.9ms | 47.4ms | 145.6 | 175.6 | 153.2 |
| 16 | 52.4ms | 42.3ms | 48.3ms | 227.3 | 282.8 | 239.4 |
| 32 | 55.5ms | 42.9ms | 49.1ms | 309.3 | 391.0 | 333.6 |
| 64 | 63.2ms | 44.0ms | 49.3ms | 637.7 | 896.4 | 354.6 |
| 128 | 69.3ms | 45.5ms | 49.3ms | 778.6 | 1063.4 | 406.4 |
| 256 | 73.1ms | 45.8ms | 49.4ms | 939.6 | 1355.3 | 438.9 |
| 512 | 74.9ms | 47.5ms | 49.4ms | 1023.6 | 1520.1 | 457.7 |
| 1024 | 76.3ms | 47.5ms | 49.4ms | 1115.9 | 1636.4 | 458.7 |

### 3.2 Input=16, Output=128 (消除 prefill 瓶颈)

| Conc | PD+AF TPOT | PD TP=2 TPOT | PD+AF 吞吐 | PD TP=2 吞吐 |
|------|-----------|-------------|-----------|-------------|
| 1 | 46.4ms | 39.8ms | 21.5 | 25.0 |
| 8 | 50.0ms | 43.2ms | 156.7 | 181.8 |
| 32 | 53.6ms | 43.2ms | 363.2 | 441.9 |
| 64 | 60.6ms | 44.9ms | 951.0 | 1332.8 |
| 128 | 66.3ms | 44.9ms | 967.6 | 1390.6 |
| 256 | 68.5ms | 45.1ms | 1210.8 | 1788.0 |
| 512 | 69.7ms | 45.6ms | 1245.7 | 1815.5 |
| 1024 | 70.0ms | 45.8ms | 1324.6 | 1980.7 |

### 3.3 TTFT 对比 (Input=512, Output=128)

| Conc | PD+AF TTFT | PD TP=2 TTFT | PD TP=1 TTFT | PD+AF vs PD TP=1 |
|------|-----------|-------------|-------------|-----------------|
| 1 | 144ms | 92ms | 134ms | +7.5% |
| 2 | 187ms | 190ms | 284ms | **-34.2%** |
| 4 | 223ms | 268ms | 401ms | **-44.3%** |
| 8 | 446ms | 394ms | 658ms | **-32.2%** |
| 16 | 906ms | 630ms | 1074ms | **-15.6%** |
| 64 | 3491ms | 2944ms | 6908ms | **-49.5%** |
| 256 | 13465ms | 9538ms | 33165ms | **-59.4%** |

---

## 四、真实 GPU Batch Size (Decode 阶段)

| 方案 | 最大 running-req | token usage 峰值 | 瓶颈 |
|------|-----------------|-----------------|------|
| PD TP=1 | **31** | 47% | 单卡 KV cache 容量（模型权重占 64GB，仅剩 ~12GB） |
| PD+AF | **100** | DA: 33%, **DF: 97%** | DF 端 KV pool 容量（FFN 权重占 ~43GB） |
| PD TP=2 | **100** | 12% | Prefill→Decode 流水线平衡（非显存瓶颈） |

---

## 五、TPOT Breakdown (高并发 batch≈100)

| 组件 | PD TP=2 | PD+AF | 差异原因 |
|------|---------|-------|---------|
| FFN/层 | 0.28ms (2 GPU 并行) | 0.56ms (1 GPU) | TP=2 切半，AF 整个 FFN 在 DF 单卡 |
| Attention/层 | 0.30ms | 0.30ms | 相同 |
| AllReduce/层 | 0.02ms | 0ms | TP=2 需要，AF 不需要 |
| IPC 通信/层 | 0ms | 0.08ms | AF 每层 2 次 NVLink P2P |
| **每层合计** | **0.60ms** | **0.94ms** | **+57%** |
| **64 层总计** | **38.4ms** | **60.2ms** | — |
| **+ 调度开销** | **+9ms** | **+16ms** | — |
| **= TPOT** | **47.5ms** | **76.3ms** | **+61%** |

---

## 六、关键结论

### 1. PD+AF 的优势
- **vs PD TP=1 (同 2 GPU 对比)**: TPOT 仅 +6% overhead (46.6ms vs 44ms)，TTFT 在高并发下快 34~60%
- **显存效率**: AF 分离让 DA 端只存 Attention 权重，KV cache 容量大幅提升（token usage 仅 33%）
- **吞吐**: 比 PD TP=1 高 2.4x（1116 vs 458 tok/s），因为真实 batch 从 31 提升到 100

### 2. PD+AF 的劣势 (vs PD TP=2, 同 4 GPU)
- **TPOT 高并发劣化**: batch=100 时 76ms vs 47.5ms (+61%)，因为 FFN 在单卡上从 memory-bound 变 compute-bound
- **吞吐低**: 1325 vs 1981 tok/s (-33%)
- **DF 显存瓶颈**: FFN 权重占 43GB，KV pool 只剩 ~25GB，限制 batch=100

### 3. AF 架构的固有瓶颈
- **DF 端显存**: FFN 权重占大部分显存，KV pool 容量有限 → batch 上限 100
- **FFN 单卡计算**: 高 batch 时 FFN 变成 compute-bound，TPOT 线性增长
- **串行通信**: 每层 2 次 IPC 同步（DA→DF + DF→DA），虽然单次仅 0.04ms，但 64 层累积 5ms

### 4. 适用场景
- **PD+AF 适合**: 低并发、延迟敏感场景（conc≤8 时 TPOT 仅 +6%），或 GPU 数量有限（2 GPU 即可获得 PD 分离收益）
- **PD TP=2 适合**: 高吞吐场景（4 GPU 可用时，TP=2 全面优于 AF）
- **PD+AF 的独特价值**: 异构 GPU 部署（ATTN 用小显存卡，FFN 用大显存卡）、能耗优化（DVFS 独立调频）

---

## 七、IPC 通信优化历程

| 阶段 | TPOT (conc=1) | vs Baseline | 每层通信开销 |
|------|--------------|-------------|------------|
| 初始 Python IPC | 117.8ms | +160% | 1.14ms |
| Fast-Path | 93.3ms | +106% | 0.75ms |
| Pre-launch 缓存 | 80.8ms | +79% | 0.56ms |
| C++ IPC (cpu_flag) | 68.9ms | +56% | 0.39ms |
| **C++ IPC (ipc_event)** | **46.6ms** | **+6%** | **0.04ms** |

---

## 八、关键启动参数

达到最优性能的 PD+AF 配置：
```
AFD_IPC_SYNC_MODE=ipc_event      # 纯 GPU 同步，消除 cudaStreamSynchronize
AFD_IPC_PEER_DEVICE=<0|1>        # 对端 GPU 索引
--afd-comm-backend ipc_cpp       # C++ IPC 后端
--afd-micro-batch 1              # M=1 fast-path
--disable-cuda-graph             # AF 模式必须关闭
```

Buffer 修复（支持高并发）：
```
MAX_MSG_SIZE = 128MB             # sgl-kernel/csrc/afd_ipc/afd_ipc.h (原 32MB)
```
