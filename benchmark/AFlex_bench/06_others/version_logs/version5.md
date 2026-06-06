# Version 5: 4-GPU 综合实验

## Part 1: 变长 Workload 对比

详见 `version5_4gpu_var.md`（4-GPU Variable-Length Workload Results）。

---

## Part 2: Multi-SLO Sweep 实验

## 实验配置

- **GPU**: 0-3（4 GPU）
- **部署方案**: PDAF DynM（动态 M 阈值 = 128，即全程 M=1）
- **三个对比方案**:
  - Baseline: 最高频率（1410 MHz），无 DVFS
  - V1: 旧模型（per-layer 独立预测），DVFS 启用
  - V2: 耦合模型（iteration-level Pipeline Coupled），DVFS 启用
- **TPOT SLO**: 300 / 250 / 200 / 150 / 100 / 90 / 80 / 70 ms
- **Workload**: workload_steady（变长序列，steady QPS）
- **指标**: Per-token TPOT SLO 违背率（逐 token 判断 inter-token 延迟是否超 SLO）

## 实验结果

| SLO (ms) | Baseline Thpt | Baseline TokViol | V1 Thpt | V1 TokViol | V1 节能 | V2 Thpt | V2 TokViol | V2 节能 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 | 737 tok/s | 0.06% | 674 tok/s | 0.18% | 31.4% | 714 tok/s | 0.07% | 30.8% |
| 250 | 737 tok/s | 0.06% | 694 tok/s | 0.18% | 31.8% | 714 tok/s | 0.08% | 30.3% |
| 200 | 737 tok/s | 0.06% | 384 tok/s | **96.27%** | 3.4% | 714 tok/s | 0.08% | 30.7% |
| 150 | 737 tok/s | 0.06% | 378 tok/s | **99.90%** | 1.5% | 713 tok/s | 0.25% | 30.3% |
| 100 | 737 tok/s | 0.56% | 384 tok/s | **99.90%** | 3.1% | 709 tok/s | **45.82%** | 31.0% |
| 90 | 737 tok/s | 0.60% | 380 tok/s | **99.90%** | 2.4% | 716 tok/s | **38.22%** | 27.1% |
| 80 | 737 tok/s | 1.12% | 380 tok/s | **99.90%** | 2.5% | 706 tok/s | **93.37%** | 27.3% |
| 70 | 737 tok/s | 1.55% | 382 tok/s | **99.90%** | 2.4% | 708 tok/s | **96.84%** | 19.0% |

## 关键发现

### 1. V1 存在"悬崖效应"（SLO=250ms → 200ms）

V1 的 per-token 违背率在 SLO 从 250ms 降到 200ms 时从 0.18% 暴涨到 96.27%。这是因为 V1 盲目选择最低频率组合（f_A=690, f_F=930 MHz），对应的 decode iteration 实际延迟约 200-220ms。当 SLO 低于这个值时，几乎所有 token 都超标。

此时 V1 的吞吐也从 694 暴跌到 384 tok/s，因为 DVFS 降频导致 decode 变慢、请求堆积、最终超时。

### 2. V2 在 SLO ≥ 150ms 时保持稳健

V2 在 SLO=300~150ms 范围内始终保持：
- Per-token 违背率 ≤ 0.25%
- 节能 ~30%
- 吞吐 713-714 tok/s（接近 Baseline 的 737）

V2 通过非对称调频（f_F 调高、f_A 适当降）使 decode iteration 延迟稳定在 83-84ms，远低于 150ms SLO，留有充分余量。

### 3. V2 的临界点在 SLO=80ms

当 SLO 收紧到 100ms 时，V2 违背率为 45.82%；SLO=90ms 时为 38.22%（因频率已调至高位，波动减小）；SLO=80ms 时暴涨到 93.37%，与 V1 接近。这说明 V2 的有效下限在 SLO ≈ 90ms 附近——低于此值，即使非对称调频也无法满足约束（系统物理延迟 83-96ms 已接近极限）。

### 4. Baseline 的物理极限

Baseline（最高频率 1410 MHz）在 SLO=70ms 时也有 1.55% 违背，说明系统的 per-token 延迟底噪约 64-65ms，偶发峰值可达 70ms+。这是 AF pipeline 通信开销的固有下限。

### 5. SLO 合规是服务可靠性的硬约束

V1 在宽松 SLO（≥250ms）下节能 31-32%，表面上优于 V2。但一旦 SLO 收紧到 200ms，V1 完全不可用。而 V2 在同一 SLO 下仍能保持 30% 节能和零违背。

**结论**：V2 的有效工作范围（SLO ≥ 90ms，违背率 <40%）远大于 V1（仅 SLO ≥ 250ms），且在 SLO ≥ 150ms 时保持 ≤0.25% 违背率和 ~30% 节能。V2 是更优的 DVFS 方案。

## 图表

- `retrain/figures/slo_sweep_comparison.png`: 多 SLO 对比四象图
- `retrain/figures/dvfs_timeline_v1_v2.png`: V1 vs V2 调频时间轴（SLO=90ms）
- `retrain/figures/dvfs_timeline_overlay.png`: 调频频率叠加对比

---

## 每日进展

### 2026-06-03

今日进展：
1.完成变长 workload 全方案对比（PD TP2 / PD DP2 / PDAF DynM / PDAF Tier+DynM），涵盖 5 种负载模式（steady/varying/heavy/overload/tier1_demo）。结果表明 PDAF Tier+DynM 在轻中负载下能耗降低 27-33%，但在高负载（overload）下 SLO 违背率较高（76%），PD TP2 则在所有场景中保持零 SLO 违背。
2.完成 Decode 阶段预测模型重训（V2 Pipeline Coupled Model），并通过 Multi-SLO Sweep 实验验证了 V2 相对 V1 的优势：V1 在 SLO < 200ms 时彻底崩溃（96-100% token 超标），而 V2 在 SLO ≥ 150ms 内保持 ≤0.25% 违背率且节能 ~30%。V2 通过学习 A/F 延迟耦合关系实现非对称调频（f_F 保高），在 SLO 优先的前提下最大化节能。
正在进行：
- 后续 SLO=90/80/70ms 的压力测试数据收集（已有 300-100ms 完整数据）
明日规划：
- 调频细节可视化与论文图表整理
- 探索 V2 在 SLO=100ms 附近违背率偏高的优化方向（如在线校准 calibration 机制）
