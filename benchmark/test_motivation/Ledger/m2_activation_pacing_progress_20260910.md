# M2 首次激活节奏：执行进度

2026-09-10。本轮按“修正基线 -> 单因素暂缓干预 -> 资源归因 -> 更强 batching”顺序推进。**48 组/3840 请求的主分析已完成，未发现固定首次激活限额的额外收益。** 完整结果、两张图与结论边界见[本轮分析](m2_activation_pacing_results_20260910.md)。一次监测超时后按整会话规则补跑，补跑服务退出后 GPU 6/7 已释放。

## 已完成

1. 确认本地、远端 scheduler 的原始 SHA-256 均为 `768c839c9b5772447491aff04edfaac9d0f316d4c9ada2bccfbb8f8c1740a58b`，已默认启用修正后的 slot 计数；本轮没有退回 legacy gate。
2. 增加默认关闭、实验 opt-in 且 idle-only 的 `tenant_new_activation_limit`。它只限制每轮首次激活数量，继续 chunk 复用 slot，不扣新额度；达到额度不设置资源耗尽标志。CPU 测试执行实际候选扫描循环，确认剩余候选尚未初始化/分配，不因人为暂缓造成永久 full 状态。
3. 冻结[实验协议](m2_activation_pacing_protocol_20260910.md)。四个 batching 配置、额度 0/1/4、两种到达顺序、两次独立重启，共计划 48 组筛选、3840 个测量请求；每会话额外预热排除。正式运行前先做 6 组 smoke、120 个请求。
4. 实现[运行脚本](run_m2_activation_pacing.py)与[分析脚本](analyze_m2_activation_pacing.py)。每对固定 token IDs、输出长度与预设到达，验证容量/readback、时钟、请求完整性、trace 丢失、资源回收与 legacy slot 反例。报告全部对照及长短请求代价，不自动挑选赢家或宣称 PASS。
5. 准备阶段远端 50 项实验/分析/slot 回归通过，本地同一组测试曾有 19 项因缺 aiohttp 跳过，远端没有跳过。测量完成后，远端重新执行 50 项加 6 项新合并回归，共 56 项全部通过；另 10 项 tracker/policy 组件测试在隐藏 CUDA 设备的进程中通过，有 17 条配置、弃用及无 GPU quantization 警告，无失败。
6. 准备阶段 `--dry-run` 成功，确认 48 组/3840 请求、TP2、正确 graph/chunk/资源参数；当时本地与远端五个代码/协议文件 SHA-256 一致。后来本地可选 GPU 参数改动予以保留，不覆盖冻结远端。13 个测量源码哈希在原运行、补跑和最终合并检查中相同；没有安装新的依赖。
7. 完成 6 组 smoke / 120 请求及主分析的 48 组 / 3840 请求，保留请求与轨迹全部通过校验。原运行和补跑合计 54 组预热排除；51 个完整测量 artifacts 中按整会话规则排除 3 个，原始数据全部保留。
8. 生成完整配对分析和两张图，逐图检查标签、数据与布局。8 个 k=1 组合的平均 JCT 退化 1.05%--16.23%；k=4 未发现可重复正向收益。Mixed-1K / short-first 的 +2.09% 出现在限额完全未触发的组合，不归因于策略。

## 新的源码观察

SGLang 已有 `prefill_max_requests`、`min_free_slots_delay` 与 `PrefillDelayer`，所以“现有系统不会主动暂缓或不考虑请求数”不是成立的 motivation。

本轮新开关只是区分首次激活和已有 slot 复用的诊断工具，不是论文算法贡献。若筛选出机会，应先校准并比较这些简单原生机制，再进入完整 Sarathi-Serve/FairBatching 等移植。现有 trace 的 VTC counter gap 未限定双方持续排队，也包含 lift，不能据此宣布公平改善；正式公平验证还需要队列事件。

## 执行状态

此前 GPU 6 被其他 MoE routing-quality 实验占用，启动前读数为 60322 MiB、17% utilization；GPU 7 为 1 MiB、0%。占用进程 PID 3981208 的父进程运行 `/workspace/MoE_sglang/.../collect_paired_quality.py`，不是本轮 runner。

此前尝试启动 smoke 时，`ensure_available` 在创建结果目录、启动服务和发送请求之前拒绝运行。用户继续后重新检查，两卡均为 1 MiB、0%，随后在 GPU 6/7 启动。没有停止其他进程，没有调整时钟/功率，没有改用其他 GPU。

`smoke01` 六组全部完成，120 请求，完整性错误为零。额度 1 在 long-first / short-first 分别触发 11 / 10 次 pacing，额度 0 / 4 未触发；峰值 active 均为 8。smoke 数据只证明功能与负载条件，不混入筛选性能。

`screen01` 完成 39 个 artifacts 后，GPU 状态查询超过 10s，监测器抛错并清理本轮服务。按补跑前记录的[整会话规则](m2_activation_pacing_recovery_20260910.md)，`recovery01` 完整重跑 Separate-4K / rep2，并补齐 Mixed-1K / rep2，共 12 组。原运行中断会话已完成的 3 组只留档，不参加主分析；不是根据性能排除。

`assembled01` 保留原运行 36 组和补跑 12 组，记录选中/排除文件的原始来源与 SHA-256。`analysis01` 完整分析成功，48 组/3840 请求；没有在原始 `screen01` 补写成功标记。3774 个双卡采样时点中，高负载 SM 为 1350--1410MHz，实际提交最大迟到 47.78ms，低于冻结 50ms 门限。所有保留运行结束后 active=0。最终服务退出后 GPU 6/7 均为 1MiB、0% utilization。

## 尚未执行

- 自然池容量下的负载爬坡、已定位 slot 缺陷的发生频率，以及 state 与 cap-only 的外部有效性验证。本轮未筛出 pacing 的正向收益，不能直接进入该策略的收益归因。
- 原生延迟/批大小机制的校准对照、文献级 batching、CUDA graphs 与独立 held-out 确认。
- 精确 backlogged-window 公平性与 steady-state SLO goodput 验证。

## 历史执行入口

以下是容器 `/workspace/sglang` 中的执行记录，所列目录已存在，不能原样覆盖重跑。脚本会先检查双卡和端口；任一被占用即退出。

```bash
.venv/bin/python benchmark/test_motivation/Ledger/run_m2_activation_pacing.py --stage smoke --output benchmark/test_motivation/Ledger/results/m2_activation_pacing_20260910/smoke01
```

smoke 通过后启动原始筛选，此次运行后来因监测超时中断：

```bash
.venv/bin/python benchmark/test_motivation/Ledger/run_m2_activation_pacing.py --stage screen --output benchmark/test_motivation/Ledger/results/m2_activation_pacing_20260910/screen01
```

补跑及溯源合并：

```bash
.venv/bin/python benchmark/test_motivation/Ledger/run_m2_activation_pacing.py --stage screen --sessions 1 --rep-start 2 --modes separate-4096 mixed-1024 --output benchmark/test_motivation/Ledger/results/m2_activation_pacing_20260910/recovery01
.venv/bin/python benchmark/test_motivation/Ledger/assemble_m2_activation_pacing.py benchmark/test_motivation/Ledger/results/m2_activation_pacing_20260910/screen01 benchmark/test_motivation/Ledger/results/m2_activation_pacing_20260910/recovery01 --output benchmark/test_motivation/Ledger/results/m2_activation_pacing_20260910/assembled01
```

分析组合后的完整矩阵：

```bash
.venv/bin/python benchmark/test_motivation/Ledger/analyze_m2_activation_pacing.py benchmark/test_motivation/Ledger/results/m2_activation_pacing_20260910/assembled01 --output benchmark/test_motivation/Ledger/results/m2_activation_pacing_20260910/analysis01
```

本记录没有把旧 slot bug 修正的 8.37% 平均 JCT 改善当成本轮新干预的收益。
