# Graceful Reload v2: 增量权重继承 + 细粒度阻塞

> 目标：将 TP 变化时的服务中断从 ~50s 压缩到 ~5-9s，且已在 decode 的请求完全不受影响。

---

## 一、问题背景

当前 reshard (TP 变化) 的流程是"全杀全启"：

```
kill 所有模块 → 启动新模块 → 加载权重(~10s) → 分配 KV cache(~3s) → ready
```

中断窗口 ≈ 50s，期间所有请求失败 (503)。

---

## 二、核心方案

### 2.1 整体时间线 (以 PA: TP1→TP2 为例)

```
T0: 旧 PA (TP1, GPU 0) 正常服务
T1: 后台在 GPU 4 加载新 PA rank 1 权重 (~10s, 完全不影响前台服务)
T2: rank 1 就绪，等待信号
T3: 触发 drain → router 暂停分配新 prefill 请求给旧 PA (排队而非拒绝)
T4: inflight prefill 请求完成 → 旧 PA idle (~1-5s)
T5: 新 PA rank 0 从旧权重 IPC/slice 获取前半权重 (~1s)
T6: 建立 NCCL 通信组 + 分配 KV cache (~3s)
T7: 新 PA ready → router 切流量 → 排队的 prefill 请求开始执行
T8: kill 旧 PA 进程 → 释放 GPU 0 多余显存
```

**用户可见中断 = T3→T7 ≈ 5-9s** (仅新 prefill 请求排队, decode 不受影响)

### 2.2 反向收缩 (PA: TP2→TP1)

```
T0: 旧 PA (TP2, rank0=GPU0, rank1=GPU4) 正常服务
T1: 触发 drain → 暂停新 prefill
T2: inflight 完成 → 旧 PA idle
T3: 新 PA (TP1) 从 rank0+rank1 拼接完整权重 → GPU 0 (~2s, GPU间NVLink传输)
T4: 分配 KV cache (~3s)
T5: 新 PA ready → router 切流量
T6: kill 旧 PA → 释放 GPU 4
```

---

## 三、关键技术点

### 3.1 权重继承 (避免磁盘 IO)

TP 切分规则:
- Column parallel (gate_proj, up_proj, q/k/v_proj): 按列切, rank i 取第 i 份
- Row parallel (down_proj, o_proj): 按行切, rank i 取第 i 份
- Embedding/LM head: 按 vocab 维度切

**扩展 (TP1→TP2)**:
- rank 0 = 旧完整权重的前半部分 (slice, ~0 拷贝时间)
- rank 1 = 旧完整权重的后半部分 (从 GPU 0 → GPU 4, NVLink ~0.5s)

**收缩 (TP2→TP1)**:
- 新完整权重 = concat(rank0 权重, rank1 权重)
- rank 1 → rank 0 传输后拼接 (~0.5s)

### 3.2 CUDA IPC 权重传递

```python
# 旧进程 (PA TP1): 导出权重句柄
import torch.multiprocessing as mp

handles = {}
for name, param in model.named_parameters():
    handles[name] = torch.cuda.ipc_collect()  # 获取 IPC handle

# 写句柄到共享文件
torch.save(handles, "/tmp/pa_weight_handles.pt")

# 新进程 (PA TP2 rank 0): 导入并 slice
loaded_handles = torch.load("/tmp/pa_weight_handles.pt")
for name, handle in loaded_handles.items():
    full_weight = handle.open()  # 打开 IPC 共享显存
    my_slice = full_weight[:, :half_dim]  # view, 零拷贝
    new_param.data.copy_(my_slice)  # copy 到自己的空间
```

### 3.3 细粒度阻塞 (模块级隔离)

PDAF 四模块独立:

| 变化模块 | 受影响的请求阶段 | 不受影响的阶段 |
|---------|---------------|-------------|
| PA reshard | 新 prefill 排队 | decode 正常继续 |
| PF reshard | 新 prefill 排队 | decode 正常继续 |
| DA reshard | decode step 暂停 | prefill 正常 |
| DF reshard | decode step 暂停 | prefill 正常 |

Router 行为:
- **不返回 503** (与当前实现的区别)
- 只将去往变化模块的新请求**排队**
- 其他模块的流量照常转发
- reshard 完成后，排队请求立即分发

### 3.4 显存预算分析 (Qwen3-32B, A800-80GB)

**PA 扩展 (TP1→TP2), GPU 0 上的共存期 (T5 时刻)**:

| 组件 | 显存占用 |
|------|---------|
| 旧 PA TP1 权重 | 21 GB |
| 旧 PA KV cache (idle, 已空) | 0 GB |
| 新 PA rank 0 权重 (slice) | 10.5 GB |
| CUDA context + activation buffer | ~3 GB |
| **总计** | **~34.5 GB** |
| GPU 容量 | 80 GB |
| **剩余 (给新 KV cache)** | **~45 GB** |

显存充裕, 无风险。

---

## 四、与当前实现的对比

| 指标 | 当前 (全杀全启) | v1 (shadow并行) | v2 (本方案) |
|------|---------------|----------------|------------|
| 中断时间 | ~50s | ~50s (GPU 重叠时) | **5-9s** |
| decode 影响 | 全部中断 | 全部中断 | **不受影响** |
| 新 prefill | 503 拒绝 | 503 拒绝 | **排队等待** |
| 磁盘 IO | 全量重加载 | 全量重加载 | **仅 rank 1 加载 (并行)** |
| GPU 0 额外显存 | 0 | +23.5 GB | **+10.5 GB (仅权重)** |

---

## 五、实现模块划分

### 5.1 权重导出服务 (`weight_exporter.py`)

旧进程在收到 drain 信号后:
1. 停止推理循环
2. 将模型权重的 CUDA IPC handle 序列化到共享路径
3. 等待新进程确认接收后再退出

### 5.2 快速加载器 (`fast_tp_loader.py`)

新进程启动时:
1. 检测是否存在 IPC handle 文件
2. 如有, 从 IPC 获取权重并按新 TP 配置 slice
3. 如无, fallback 到标准 safetensors 加载

### 5.3 Router 排队逻辑 (`mini_lb.py` 修改)

新增 `/admin/pause_module` 端点:
- 参数: `module_type` (pa/pf/da/df)
- 行为: 对应模块的新请求进入等待队列
- 恢复: `/admin/resume_module` 后队列请求立即分发

### 5.4 Orchestrator (`graceful_orchestrator_v2.py`)

编排整体流程:
1. 提前启动新 rank 1 (后台, 不影响服务)
2. rank 1 ready 后触发 drain
3. 等待 idle → 权重导出 → 新 rank 0 加载 → NCCL init
4. 新模块 ready → 切流量 → kill 旧进程

---

## 六、适用场景与限制

### 适用:
- 单模块 TP 变化 (PA/PF/DA/DF 任一)
- 同机多卡, NVLink 互联
- GPU 显存 >= 40 GB (需容纳短暂权重共存)

### 限制:
- 多模块同时变 TP: 需要串行执行 (先变 PA, 再变 DA 等)
- 跨机 TP 变化: IPC 不可用, 需走 RDMA 权重传输 (退化为 ~3-5s 额外延迟)
- KV cache 不迁移: reshard 后旧 KV 丢弃, 正在 decode 的请求在旧模块完成后新请求才用新配置

---

## 七、预期性能

基于 Qwen3-32B, 8×A800-SXM4-80GB:

| 操作 | 耗时 |
|------|------|
| rank 1 后台加载 (磁盘→GPU4) | ~10s (隐藏) |
| drain + 等待 idle | ~1-5s |
| IPC 权重 slice (rank 0) | ~1s |
| NCCL 通信组建立 | ~1s |
| KV cache 分配 | ~2s |
| **总可见中断** | **~5-9s** |

相较于当前 ~50s 中断, **缩短 80-90%**, 且 decode 流量零中断。
