# NVLink vs RDMA 通信性能对比分析

## 测试环境

| 项目 | 配置 |
|------|------|
| GPU | 8× NVIDIA A800-SXM4-80GB |
| NVLink | NV8 全互联，每条 25 GB/s，8 条链路 |
| RDMA | Mellanox ConnectX-6 HDR 200Gb/s ×4 (mlx5_0/1/4/5) + bond0 (mlx5_bond_0, RoCE) |
| 连接类型 | IB, RC, MTU 4096B |
| 节点 | 10.252.129.34 ↔ 10.252.129.35；10.252.129.36 ↔ 10.252.129.35 |

## 文件结构

```
comm/
├── read.md                         # 本文档
├── bench_mooncake_rdma_write.py    # ★ Mooncake Transfer Engine RDMA Write 基准
│                                   #   (使用 sglang 跨机 PD 传输的同一原语)
├── bench_gpudirect_rdma.py         # UCX GPUDirect send/recv 基准
├── plot_comm_comparison.py         # ★ 综合对比绘图（NVLink vs RDMA ×1/×4/batch）
├── plot_nvlink_vs_rdma.py          # 早期 ib_write_bw 对比图
├── run_comm_bench.sh               # ★ 全套测试编排脚本（宿主机运行）
├── run_rdma_bw_test.sh             # ib_write_bw perftest 脚本
├── run_rdma_send_test.sh           # ib_send_bw/lat perftest 脚本
├── results/                        # JSON 测试结果
└── charts/                         # 生成的对比图表
```

## 测试结果摘要

### RDMA (ib_write_bw / ib_write_lat)

| 消息大小 | 带宽 (GB/s) | 延迟 (μs) |
|---------|------------|----------|
| 2B | 0.011 | 1.84 |
| 4B | 0.023 | 1.84 |
| 8B | 0.044 | 1.84 |
| 16B | 0.088 | 1.84 |
| 32B | 0.174 | 1.88 |
| 64B | 0.351 | 1.88 |
| 128B | 0.701 | 1.94 |
| 256B | 1.403 | 2.71 |
| 512B | 2.785 | 2.75 |
| 1KB | 5.576 | 2.83 |
| 2KB | 10.598 | 2.96 |
| 4KB | 16.986 | 3.53 |
| 8KB | 24.071 | 3.91 |
| 16KB | 24.545 | 4.17 |
| 32KB | 24.579 | 4.93 |
| 64KB | 24.591 | 6.29 |
| 128KB | 24.596 | 8.98 |
| 256KB | 24.598 | 14.35 |
| 512KB | 24.599 | 25.03 |
| 1MB | 24.600 | 46.92 |
| 2MB | 24.601 | 89.63 |
| 4MB | 24.601 | 174.91 |
| 8MB | 24.600 | 345.45 |

### RDMA 4 卡聚合 (10.252.129.36 ↔ 10.252.129.35)

工具 `ib_write_bw -a`，4 路并行（mlx5_0/1/4/5，端口 18515–18518），`ulimit -l unlimited`。

**峰值带宽（各尺寸合计）：**

| 消息大小 | 带宽 (GB/s) | 消息大小 | 带宽 (GB/s) |
|---------|------------|---------|------------|
| 2B | 0.05 | 16KB | 91.5 |
| 4B | 0.09 | 32KB | 91.6 |
| 8B | 0.19 | 64KB | 91.6 |
| 16B | 0.38 | 128KB | 91.6 |
| 32B | 0.76 | 256KB | 91.6 |
| 64B | 1.53 | 512KB | 91.6 |
| 128B | 3.07 | 1MB | 91.6 |
| 256B | 6.11 | 2MB | 91.3 |
| 512B | 11.97 | 4MB | 91.6 |
| 1KB | 23.6 | 8MB | 87.9 |
| 2KB | 43.7 | | |
| 4KB | 69.7 | | |
| 8KB | 90.1 | | |

**单卡 @64KB（饱和点）：**

| 设备 | 带宽 (MB/s) | 带宽 (GB/s) | 本地 LID | 远端 LID |
|------|------------|-------------|----------|----------|
| mlx5_0 | 23448.65 | 22.90 | 36 | 242 |
| mlx5_1 | 23448.10 | 22.90 | 35 | 244 |
| mlx5_4 | 23391.47 | 22.84 | 231 | 246 |
| mlx5_5 | 23446.91 | 22.90 | 232 | 245 |
| **合计** | **93735.13** | **91.6** | — | — |

### RDMA 延迟 (10.252.129.36 ↔ 10.252.129.35, ib_write_lat -a, mlx5_0)

| 消息大小 | 延迟 (μs) | 消息大小 | 延迟 (μs) |
|---------|----------|---------|----------|
| 2B | 1.82 | 16KB | 4.13 |
| 4B | 1.82 | 32KB | 4.82 |
| 8B | 1.83 | 64KB | 6.15 |
| 16B | 1.82 | 128KB | 8.81 |
| 32B | 1.86 | 256KB | 14.12 |
| 64B | 1.86 | 512KB | 24.76 |
| 128B | 1.93 | 1MB | 46.73 |
| 256B | 2.70 | 2MB | 89.44 |
| 512B | 2.74 | 4MB | 174.69 |
| 1KB | 2.90 | 8MB | 345.19 |
| 2KB | 2.95 | | |
| 4KB | 3.55 | | |
| 8KB | 3.78 | | |

注：容器内默认 `ulimit -l = 64KB`，64KB 消息可正常测试；更大消息需 `ulimit -l unlimited`（两端均需设置）。

### NVLink (P2P GPU 间传输)

| 消息大小 | 带宽 (GB/s) | 延迟 (μs) |
|---------|------------|----------|
| 2B | 0.24 | 41.96 |
| 4B | 0.56 | 36.59 |
| 8B | 1.12 | 36.55 |
| 16B | 2.19 | 37.44 |
| 32B | 4.46 | 36.77 |
| 64B | 8.92 | 36.74 |
| 128B | 17.76 | 36.91 |
| 256B | 35.34 | 37.08 |
| 512B | 71.23 | 36.80 |
| 1KB | 35.38 | 37.05 |
| 2KB | 70.63 | 37.11 |
| 4KB | 107.79 | 48.64 |
| 8KB | 128.67 | 81.49 |
| 16KB | 143.60 | 146.05 |
| 32KB | 148.46 | 282.52 |
| 64KB | 152.71 | 549.31 |
| 128KB | 173.84 | 965.07 |
| 256KB | 175.72 | 1909.54 |
| 512KB | 176.54 | 3801.44 |
| 1MB | 177.03 | 7592.25 |
| 2MB | 176.98 | 15164.85 |
| 4MB | 177.09 | 30340.55 |
| 8MB | 177.13 | 60622.28 |

注：NVLink 数据来源于 `communicaton_cost_nvlink.txt`，使用 seq_len=128,bs=1 起始对应 ~1KB（假设 hidden=4096, fp16），此处按实测吞吐量对齐消息大小。

## 关键发现

1. **RDMA 小消息延迟极低**：1.84μs（2B），比 NVLink 的 37μs 快 ~20×
2. **NVLink 大消息带宽远超 RDMA 单卡**：177 GB/s vs 24.6 GB/s，约 7.2×
3. **4 卡 RDMA 聚合可达 ~91.6 GB/s**（.36↔.35 实测，16KB 起饱和），约为 NVLink 峰值的 52%，单卡 ~22.9 GB/s
4. **RDMA 延迟极度稳定**（.36↔.35）：2B 仅 1.82μs，stdev 0.03–0.06μs，P99.9 < 5μs（小消息）
5. **带宽饱和点**：RDMA ~16KB 即饱和；NVLink 需要 ~256KB 才接近峰值
6. **跨节点大 tensor 传输是核心瓶颈**：1MB 数据 RDMA 需 47μs，NVLink 吞吐更高

## 对 AFD 的启示

- 控制信号/元数据跨节点传输代价极低（<2μs）
- 大 activation tensor 跨节点传输是性能关键路径
- 4 端口 IB 聚合实测 ~91.6 GB/s（.36↔.35），仍比 NVLink 慢 ~1.9×
- 设计 microbatch pipeline 时需充分利用通信-计算重叠掩盖跨节点延迟

---

## Mooncake RDMA Write 基准（匹配 sglang PD 传输原语）

### 背景

SGLang 的跨节点 KV cache 传输（PD disaggregation）使用 **Mooncake Transfer Engine** 的 `transfer_sync_write` / `batch_transfer_sync_write` API。底层是 **RDMA Write**（单边操作，写入远端已注册的 GPU 显存），不是 UCX send/recv 也不是 ib_write_bw perftest。

数据路径：
```
Prefill GPU memory (registered) 
    → mooncake transfer_sync_write(session_id, src_ptr, dst_ptr, length)
    → RDMA Write via IB NIC → 远端 NIC → Decode GPU memory (registered)
```

与 perftest 的区别：
- mooncake 会做 memory region 注册、session 管理、batch 传输优化
- 实际开销包含 mooncake 软件栈开销 + RDMA verbs 开销
- batch mode 一次提交多个 transfer（模拟多层 KV 并发传输）

### 测试脚本使用

```bash
# ── 全套自动化（在 node1 宿主机执行）──
cd /mnt/workspace/lt/sglang/benchmark/AFlex_bench/comm
bash run_comm_bench.sh          # 完整版（~15 分钟）
bash run_comm_bench.sh --quick  # 快速版（~5 分钟，减少迭代）

# ── 手动分步执行 ──

# 1) NVLink baseline（node1 容器内）
docker exec operator_test python3 /mnt/workspace/lt/sglang/benchmark/AFlex_bench/comm/bench_mooncake_rdma_write.py \
    nvlink --src-gpu 0 --dst-gpu 4 --outfile results/nvlink.json

# 2) Mooncake RDMA Write 单卡
#    node2 (receiver):
python3 bench_mooncake_rdma_write.py receiver --gpu 0 --ib-device mlx5_bond_0
#    node1 (sender):
python3 bench_mooncake_rdma_write.py sender --gpu 0 --ib-device mlx5_bond_0 \
    --remote-host 10.252.129.35 --outfile results/rdma_single.json

# 3) Mooncake RDMA Write 4卡聚合
#    node2:
python3 bench_mooncake_rdma_write.py receiver --multi-nic 4
#    node1:
python3 bench_mooncake_rdma_write.py sender --multi-nic 4 \
    --remote-host 10.252.129.35 --outfile results/rdma_4nic.json

# 4) Batch mode（模拟 64 层 KV 并发传输）
#    node2:
python3 bench_mooncake_rdma_write.py receiver --gpu 0 --ib-device mlx5_bond_0 --batch-count 64
#    node1:
python3 bench_mooncake_rdma_write.py sender --gpu 0 --ib-device mlx5_bond_0 \
    --remote-host 10.252.129.35 --use-batch --batch-count 64 \
    --outfile results/rdma_batch64.json

# 5) AFD activation tensor 尺寸
#    node2:
python3 bench_mooncake_rdma_write.py receiver --gpu 0 --ib-device mlx5_bond_0 --sizes afd
#    node1:
python3 bench_mooncake_rdma_write.py sender --gpu 0 --ib-device mlx5_bond_0 \
    --remote-host 10.252.129.35 --sizes afd --outfile results/rdma_afd.json
```

### 测试场景

| # | 场景 | 说明 | 对标 |
|---|------|------|------|
| 1 | NVLink P2P | 同节点 GPU0→GPU4 cudaMemcpyPeerAsync | 节点内 AFD 基准上界 |
| 2 | RDMA Write 1×NIC | mlx5_bond_0 (RoCE)，单 GPU→单 GPU 跨节点 | PD 单实例传输 |
| 3 | RDMA Write 4×NIC | mlx5_0/1/4/5 并行，4 GPU→4 GPU | PD TP=4 多卡并发传输 |
| 4 | Batch(64) | 单 NIC，batch_transfer_sync_write 64 块 | PD 所有层 KV 一次性批量 |
| 5 | AFD sizes | activation tensor 尺寸 (10KB~10MB) | AFD 算子间跨节点传输 |

### 消息尺寸与 KV 传输的对应

Qwen3-32B (hidden=5120, num_kv_heads=8, head_dim=128, bf16):
- 每 token 每层 KV: 2×8×128×2 = **4096 bytes**
- 每 token 全部 64 层 KV: 64×4096 = **256 KB**
- 128 token 全部层: 128×256KB = **32 MB**
- 1024 token 全部层: 1024×256KB = **256 MB**

### 绘图

```bash
# 用测试结果（results/ 目录下的 JSON）
python3 plot_comm_comparison.py

# 用嵌入的参考数据（无需实际测试）
python3 plot_comm_comparison.py --use-embedded
```

生成 4 张子图：
1. 吞吐 (GB/s) vs 消息大小 — 所有配置
2. 延迟 (μs) vs 消息大小
3. KV 传输时间 vs token 数 (Qwen3-32B 64层)
4. NVLink/RDMA 带宽倍率

### 参考图

![NVLink vs RDMA comparison](./charts/mooncake_rdma_vs_nvlink.png)
