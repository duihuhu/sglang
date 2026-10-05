# M2 池管理对照：已验证什么，尚未验证什么

2026-09-16，远端 `/workspace/sglang` 的 CPU/meta 预检。**没有新增 GPU 性能实验，没有证明统一池无效，也没有证明它已消除服务损失。**

当前采用记录：[offline02/report.md](results/m2_pool_comparison_20260916/offline02/report.md)、[完整 JSON 与源码哈希](results/m2_pool_comparison_20260916/offline02/preflight.json)。`offline01` 保留为开发期记录，其静态主 cache 列尚未包含一个 MLA padding token；`offline02` 补齐此项，不混合两版数值。

## 1. 核实了三个实现事实

### 共享 cache 字节，不会自动解除请求上限

调用当前源码的 `KVCacheConfigurator.resolve_max_num_reqs`，在相同上下文和 radix-off 条件下得到：

| 条件 | 初始化 state 参数 | KV tokens：离线初值 | request cap：原生逻辑离线解析 |
|---|---:|---:|---:|
| 原静态池 | 485 | 1,458,643 | 485 |
| 重划静态池 | 607 | 1,130,211 | 607 |
| 统一池，配套提高 state 参数 | 607 | 1,130,211 | 607 |
| 统一池，只开开关 | 485 | 1,458,643 | 485 |

另外，统一池保留 state485 时，即便显式请求 `max-running-requests=607`，原生 resolver 仍返回485。上述是当前代码路径的配置语义，不是新服务会话的实测容量。

因此，“统一池打开后仍排队”至少有两种不同解释：共享字节真正不足，或者预分配 request table/请求上限仍然较小。测试必须分开检查。这个配置事实本身不等于 SGLang 的安全性错误，也不意味着请求表必须无限增长。

### 主 cache 预算相同，不等于完整 HBM 相同

真实模型配置与当前原生布局给出 TP2 每 rank：state=21,708,800 bytes/request，MLA KV=8,064 bytes/token；20层 KDA、7层 MLA，BF16 conv/KV、FP32 temporal。这不是外部 TP8 模型的54MB/27KB估计。

| 条件 | 主 backing 含已知 padding，bytes/rank | 请求索引表，bytes/rank |
|---|---:|---:|
| 原静态池485 | 22,312,982,016 | 2,038,439,520 |
| 重划静态池607 | 22,312,979,968 | 2,550,146,560 |
| 统一池607 | 22,291,271,168 | 2,550,146,560 |
| 统一池485负对照 | 22,291,273,216 | 2,038,439,520 |

这张表是源码驱动的离线计量，不是完整 HBM 实测。重新分池的主 backing 只减少2048 bytes，但请求表增加511,707,040 bytes，约488 MiB。统一池的初始化公式又与静态池的 dummy state 计量不同，不能忽略其约21.7MB差异。

表中没有模型权重、全部 workspace、allocator 辅助分配、Torch 保留块和非 Torch 分配。它只说明下一轮需要匹配什么，不能宣称总 HBM 已匹配。

### 当前 backend guard 与真实形状的统一池视图检查通过

在隐藏 GPU 的 serving 环境中，当前 Triton backend 通过统一池/page-major 参数检查。用真实层数、形状、dtype 和约22.29GB逻辑大小，在 `meta` device 构建了原生统一池，验证 raw、KDA temporal/conv、7层 MLA views 只共享一个 storage，不重复计费；没有实际申请这些 GPU/CPU cache 字节。

这排除了本次配置的已知参数 guard 拒绝，以及这组形状的视图构造错误；**没有测试实际 Triton 算子、动态 allocator/压缩、输出正确性或运行性能。** 虚拟 state/KV slots 的各自最大值也不能同时用满。

## 2. 对 M2 故事线的影响

应把旧结论收窄为：**自动静态配置下有 state/request 方向的瓶颈，而 KV 仍有余量；已有共享池和静态重划可能修复其中的分配问题，尚未完成其对照。**

不能直接写成“hybrid 天生有两个不可替代的硬池，所以现有调度必然损失性能”。Fixed state tax 是真实的资源成本，但硬分池闲置、请求上限限制和执行干扰不是同一个机制。

新的 observation 是：在当前实现中，**共享字节容量和可准入请求数是两个独立需要配置/核实的量**。新的实验 insight 是：先用已有内存管理与正确上限排除可修复部分，再验证剩余准入损失。它目前是实验控制和归因要求，不作为已成立的新论文贡献。

## 3. 下一步的完成条件

按[新协议](m2_pool_comparison_protocol_20260916.md)继续：

1. G1：真实模型启动、输出正确性、逐 rank 完整 HBM 账本、实际请求 cap；据字节审计冻结匹配预算，而不是据延迟调参。
2. G2：同一607请求工作集，静态/重划/统一池配合 Separate4K、Mixed4K，必要的共同cap与同池cap485/607对照；报告绝对JCT及tail/makespan/state-time，不只看模式差距。
3. 若已有池管理已解决主要损失，保留部署 insight、缩小 M2；若仍存在损失，再分离计算/压缩/准入因素，验证有目标约束的可避免收益。

本轮远端新增测试13/13通过，包括原生cap截断、显式cap、KV约束、padding、CPU/meta共享storage去重等。本地纯算术7项通过，6项因无torch跳过；远端补齐全部测试。GPU隐藏引出的量化后端不可用/无CUDA警告及Torch弃用警告不影响通过结果。

检查时物理GPU6/7分别约87%/88%利用率，显存约5157/7493MiB，仍有其他任务。本轮没有启动模型、停止他人进程、修改时钟/功率或使用GPU0--5。G1/G2等待6/7可用后执行；当前不自动轮询或占用其他GPU。

复现离线检查，不需要 GPU：

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m unittest discover -s benchmark/test_motivation/Ledger -p test_m2_pool_comparison.py -v
CUDA_VISIBLE_DEVICES='' .venv/bin/python benchmark/test_motivation/Ledger/preflight_m2_pool_comparison.py --out benchmark/test_motivation/Ledger/results/m2_pool_comparison_20260916/offline03
```

输出目录须为未存在的新目录；脚本拒绝覆盖已有记录。
