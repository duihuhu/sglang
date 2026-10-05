# M2 mediation 内存口径补充：主 cache budget 与全部分配不同

2026-09-16，在正式校准结果分析前审计；不修改冻结测量协议、代码或条件。

Inspect readback：自然池 `[485,1458643,485]`；宽池 `[728,804471,728]`，分别为state slots、KV tokens、effective request slots。mem-fraction-static均0.85，dtype与backend不变。

从本机模型config与实际初始化路径：20个KDA层，TP2后每rank16heads，head_dim128；conv `(3,6144)`，BF16；temporal `(16,128,128)`，FP32。每rank每state的主体字节为 `(3*6144*2 + 16*128*128*4)*20 = 21708800`。7个MLA层，BF16，KV维度512+64，对应每token每rank `7*576*2 = 8064` bytes。包含state padding1后：

| 配置 | State主体+KV逻辑容量预算，bytes/rank |
|---|---:|
| Natural | `(485+1)*21708800 + 1458643*8064 = 22312973952` |
| Wide | `(728+1)*21708800 + 804471*8064 = 22312969344` |

两者只差4608bytes，说明主要cache预算的确是从KV重新分配给state，非单纯增加这两类cache之和。启动日志与之相符：自然conv/ssm/KV约0.33/9.49/10.95GiB，宽池约0.50/14.24/6.04GiB。不是引用Kimi-K3 TP8的54MB/27KB数字。

但 **不能据此称全部HBM分配严格相同**。`ReqToTokenPool` 在GPU分配 `(effective_requests+1, context_len+extra)` 的int32索引表；context_len=1048576，request485 ->728会增加约0.95GiB的这张表。其他allocator bookkeeping也可能变化。宽池启动后available_gpu_mem约8.85GiB，自然先前约9.76GiB，与额外开销方向一致，但NVML/available差值不能精确归因给一张表。

因此：

- A/B是同显存比例、近等主cache预算的池配置对照，含辅助分配及跨服务会话差异，不是全部HBM字节严格匹配的纯state干预。
- B/C在同一个宽池服务会话内，只改已有runtime cap485/607，request/KV/state池完全不变。这才是本轮最清楚的并发上限中介干预。
- 不将A/B可能出现的收益宣称为严格同总字节预算下的新算法收益。若未来需要该结论，必须另外控制request表/辅助buffer或显式核对全部分配，不能事后换一套预算口径。

源码依据：`configs/mamba_utils.py` 的 `KimiLinearStateShape.create` 与 `BaseLinearStateParams.mamba_cache_per_req`；`configs/kimi_linear.py` 的 `mamba2_cache_params`；`mem_cache/kv_cache_configurator.py` 的ratio/显式state预算扣除及`resolve_max_num_reqs`；`mem_cache/memory_pool.py` 的 `ReqToTokenPool.__init__`。本轮不修改上述实现。
