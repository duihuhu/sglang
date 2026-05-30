  AFlex 两层控制详解（修正版）                                                                                   

---

  零、初始化：AF TP 配置从哪里来                                                                                 

  系统启动时，AF 的 TP 通过两种路径确定：                                                                        

  路径 A — Startup ILP（论文方案）：                                                                             

  Tier 1 ILP 根据 G、预估 λ、默认负载分桶求解初始配置 → 输出 k_P, k_D, tp_PA, tp_PF, tp_DA, tp_DF, f̄_* → 启动 AFD
   实例 → Tier 2 从 server_args 读取 tp_a, tp_f 初始化。

  路径 B — 手动指定（当前开发调试用）：                                                                          

  --tp-size 4               # 基础 TP  
  --afd-attn-tp 2            # A 侧 TP  
  --afd-ffn-tp 4             # F 侧 TP  
  --afd-dvfs-enabled  
  --afd-energy-model-dir /path/  
  --afd-ttft-slo-ms 5000     # TTFT SLO，默认 5000ms  
  --afd-tpot-slo-us 50000    # TPOT SLO，默认 50000us                                                            

  初始化代码路径（scheduler.py）：                                                                               

# __init__ (line 520-526)

  self._cur_f_a = 1410          # 从最高频起步，安全  
  self._cur_f_f = 1410  
  if server_args.afd_dvfs_enabled:  
      self._init_afd_dvfs(server_args)                                                                           

# _init_afd_dvfs (line 1540-1558)

  predictor = AFProfilePredictor(model_dir)           # 加载 .pkl 模型  
  tp_a = server_args.afd_attn_tp or server_args.tp_size  # A 侧 TP  
  tp_f = server_args.afd_ffn_tp or server_args.tp_size  # F 侧 TP  
  self._af_dvfs_ctrl = AFDVFSController(  
      predictor=predictor,  
      num_layers=self.model_config.num_hidden_layers,  # 动态读取，如 Qwen3-32B=64  
      tp_a=tp_a,            # 异构 TP 支持  
      tp_f=tp_f,  
  )                                                                                                              

  tp_a 和 tp_f 在启动时确定后运行时不变（模块 C 降级后）。Tier 1 重规划需要改 TP 时，走 drain-then-switch +  
  重启。                                                                                                         

---

  一、Tier 1：联合资源规划 (ILP)

  1.1 目标函数    

  minimize:  Σ_{c ∈ {P,D}}  k_c × [ E_A(tp_cA, f̄_cA, wl) + E_F(tp_cF, f̄_cF, wl) + E_bubble(M) ]                  

  逐项拆解：                                                                                                     

- k_c：Prefill 或 Decode 的 AF 流水线对数。每对含 1 组 A GPU + 1 组 F GPU。                                    
- E_A(tp_cA, f̄_cA, wl)：在给定 TP、频率、负载下，单层 Attention 能耗（mJ），直接从 Profile 表查。
- E_F(...)：同上，FFN 单层能耗。                                                                               
- E_bubble(M)：流水线气泡能耗。

  为什么是"单层"？ Profile 表测的就是单层。总能耗 = 单层 × L × 迭代数，迭代数取决于请求量。Tier 1  
  只需比较单位能耗即可。                                                                                         

  E_bubble 公式：                                                                                                

  E_bubble = P_idle(f_wait) × max(0, t_slow - t_fast) × (M-1)/M                                                  

- f_wait：完成更快那方的频率（谁先完成谁等，等的过程中按自己的频率耗电）                                       
- t_slow - t_fast：等待时间                                                                                    
- (M-1)/M：M=1 时无 bubble（串行执行无互相等待），M=2 时 bubble 减半，M=3 时减到 2/3

  为什么放在目标函数而非约束里？ E_bubble 做连续惩罚——10% 失衡有小代价，50% 失衡有大代价。ILP 自动倾向选 A/F  
  延迟接近的配置。如果用硬约束 |t_A - t_F| ≤ threshold，可能把所有可行解砍掉。                                   

  1.2 约束条件

  关于 AF 配对的物理拓扑（非 ILP 变量，但在生成候选配置时考虑）：
  - A/F 配对跨节点部署：节点内通过 NVLink 通信，节点间通过 RDMA（IB/RoCE）传输 hidden states。
    Decode 阶段传输量小（seq_len=1, t_AF_comm ≈ 37-42us via NVLink），跨节点 RDMA 开销在可接受范围。
    Prefill 大 batch 长序列时传输量大（如 seq_len=4096, bs=4: ~965us via NVLink），跨节点时需纳入 SLO 预算。
  - 配对后避免剩余孤立 GPU（如 7 卡给 P、1 卡剩余无法配对）。

  多节点拓扑对 ILP 的额外约束（当前简化处理，后续扩展）：
  - 每节点 GPU 数 ≤ 8（A800 SXM 单节点上限）。ILP 输出的 k_P×tp_P 个 GPU
    需能分配到有限数量的节点上。若所有 AF 对跨节点部署，需要 2×N_node 个节点。
  - 论文 v1 采用同构假设：所有 AF 对均为跨节点配对（A 在节点 i，F 在节点 j），
    每个节点内部署同类型 GPU。节点容量约束通过 k_c × tp_c ≤ G_node × N_nodes_c 体现。
  - 异构拓扑（部分同节点 NVLink + 部分跨节点 RDMA）的联合优化留作后续扩展。

  约束 (1)：GPU 总量                                                                                             

  k_P × (tp_PA + tp_PF) + k_D × (tp_DA + tp_DF) ≤ G                                                              

  一条流水线需 tp_A + tp_F 个 GPU。AF 架构下 A 和 F 空间分离，不能共享 GPU。配对比 k_PA = k_PF = k_P  
  是硬件约束——流水线需要 1:1 对接。                                                                              

  约束 (2)(3)(4)：变量域                                                                                         

  tp ∈ {1, 2, 4, 8}     # num_kv_heads=8 整除约束（Qwen3-32B）  
  k_P, k_D ∈ Z⁺          # 至少 1 对  
  f̄ ∈ {210, 450, 690, 930, 1170, 1410}  # A800 supported_sm_clocks                                               

  频率离散是硬件限制，不是设计选择。SetGpuLockedClocks 只能接受 NVML 列表中的值。                                

  约束 (5)(6)：延迟 SLO（保守 M=1 上界）                                                                         

  (5) Prefill:  t_PA(r) + t_PF(r) + t_comm(r, M) ≤ TTFT_SLO / L  
  (6) Decode:   t_DA(r) + t_DF(r) + t_comm(r, M) ≤ TPOT_SLO / L                                                        


  // t_comm 待测：当前使用 NVLink P2P 实测值（第 1 节表格）。跨节点 RDMA（IB/RoCE）
  // 通信开销待 bench_af_comm.py 实测。预计 Decode（seq_len=1）RDMA 开销仍
  // 在 ~tens of us 量级；Prefill 大 batch 长序列可能到 ~ms 量级。
  // t_comm 非全局常数，应按请求类型桶 r 分别计算：t_comm(r) = latency_overhead + tensor_size(r) / bandwidth。
  为什么用 M=1（串行）而非 M>1（pipeline）？                                                                     

  这是整个设计最重要的保守假设。M=1 时 t_layer = t_A + t_F + t_comm，M=2 时 t_layer = max(t_A, t_F) +  
  t_comm/2。串行公式永远 ≥ pipeline 公式（上界）。

  Tier 1 是静态规划——几分钟前做的决定要保证未来几分钟内不管实际 M 是多少、batch 怎么变，SLO  
  都不违反。用最坏情况做可行性检查。

  以 tp=4, il=1024, bs=4, 1410MHz 为例：                                                                         

  M=1 (ILP 保守上界): t_layer = 1011 + 4132 + 283 = 5426us  
  M=2 (运行时实际):   t_layer = max(1011, 4132) + 283/2 = 4274us  
  保守比: 5426 / 4274 ≈ 1.27  →  27% slack 留给 Tier 2                                                           

  注意：27% 来自这个特定配置（F/A ≈ 4），当 F/A 接近 1 时 slack 接近 50%。论文引用时应标明具体配置。             

  为什么 Prefill 和 Decode 都 /L？                                                                               

  t_PA + t_PF + t_comm 是单层延迟。TTFT_SLO 是整次 Prefill 的总预算（如 500ms），TPOT_SLO 是生成一个 token  
  的总预算（如 50ms）。两者都需经过全部 L 层，所以要除以 L 才能与单层延迟对齐。

  r 是什么？ 不是每个请求，是负载分桶的 P90 代表值。4 个 Prefill 桶 → 4 条约束 (5)。P90  
  比均值保守（覆盖长尾），比最大值高效（不过度浪费）。

  约束 (7)(8)：吞吐容量                                                                                          

  (7) k_P × Thpt_pair_P(M) ≥ (1+α) × λ  
  (8) k_D × Thpt_pair_D(M) ≥ (1+α) × N_active                                                                    

  Thpt_pair 的两种模式：                                                                                         

- M=1：Thpt_pair = 1 / (t_A + t_F + t_comm) —— A 和 F 串行，一个 batch 的总时间 = 三者之和                     
- M>1：Thpt_pair ≈ 1 / max(t_A, t_F) —— pipeline 重叠后，吞吐卡在慢的那一方

  (1+α)：α=0.1~0.2，容量裕度。负载监控的 λ 是统计均值，实际有波动。                                              

  N_active：活跃 decode 请求数。在 continuous batching 中，每个活跃请求每 iteration 生成一个 token，所需吞吐 ∝  
  N_active。这是近似——精确 Decode 吞吐模型需要离散事件模拟。                                                     

  约束 (9)：A/F 延迟平衡（剪枝用）                                                                               

  |t_A - t_F| ≤ β × max(t_A, t_F)      β ∈ [0.5, 0.8]                                                            

  与 E_bubble 的关系：E_bubble 在目标函数里做连续惩罚，约束 (9) 做硬剪枝。失衡超过 β=0.8 的配置被直接排除，不进入
   ILP 搜索。这减少搜索空间约 60%，且不丢最优解（因为 bubble 已经大到无法补偿）。                                

  约束 (10)-(13)：显存（按 P/D × A/F 四池分别约束）                                                              

  (10) Mem_PA = W_attn/tp_PA + KV_cache(bs_max_P, il_max_P, tp_PA)                    ≤ GPU_MEM (80GB)  
  (11) Mem_DA = W_attn/tp_DA + KV_peak_D(tp_DA)                                      ≤ GPU_MEM  
  (12) Mem_PF = W_ffn/tp_PF  + activation_buffer_P                                   ≤ GPU_MEM  
  (13) Mem_DF = W_ffn/tp_DF  + activation_buffer_D                                   ≤ GPU_MEM                  

  为什么要区分 P/D？                                                                                             

- Prefill-A：KV cache 在一次 forward 中生成后即传给 Decode 侧（PD 分离），  
峰值 = bs_max_P × il_max_P × KV_per_token / tp_PA，是静态可算的。                                           
- Decode-A：KV cache 是动态的——每个活跃请求每步增长 1 token，请求完成后释放。  
在 continuous batching 稳态下，同时有 N_active 个请求驻留，每个请求的  
KV cache 长度 = il + 已生成的 ol（不断增长直到完成）。                                                       
- tp_PA 和 tp_DA 可能不同（ILP 允许异构 TP），所以必须分开约束。                                               
- F 实例无 KV cache 但 FFN 权重大（~59GB for Qwen3-32B），activation buffer 在 P/D  
间也有差异（Prefill 的 activation 与 seq_len 成正比，Decode 的 activation 较小）。

  Decode-A 的 KV cache 动态过程与峰值估计：                                                                      

  Decode batch 不是静态的：每个 iteration 有请求完成（释放 KV cache）、有新请求从  
  Prefill 完成后加入（带入新的 KV cache）。KV cache 总占用在持续波动。                                           

  Tier 1 是静态规划，无法追踪逐步动态。用稳态峰值估计：                                                         

  KV_peak_D = N_active × avg_seq_total × KV_per_token / tp_DA                                                    

- N_active：稳态下同时活跃的 Decode 请求数（由 Little's Law 近似：N_active ≈ λ × avg_ol，在稳态下成立；Tier 1 是分钟级规划，稳态假设合理）                      
- avg_seq_total：活跃请求的平均总序列长度 = avg(il + ol_current)，取 P99 做保守估计                            
- 为什么不用 bs_max_D × (il_max + ol_max)？那是绝对最坏情况（所有请求同时在最长  
序列长度），过于保守。稳态下请求处于生成过程的不同阶段，有的刚开始（KV 小），  
有的快结束（KV 大），有的已完成释放。P99 稳态估计更贴近实际峰值。

  运行时显存保护（sglang 已有机制 + Tier 2 扩展方向）：                                                           

  Tier 1 的显存估计是静态规划用的，实际运行时 KV cache 的动态波动由 sglang scheduler  
  已有的 retract 机制处理（非 AFlex 设计）：                                                                     

1. 每个 Decode iteration 前调用 check_decode_mem() 检查 KV pool 剩余空间                                       
2. 空间不足 → retract_decode()：按策略逐个踢出请求（优先踢已生成最多 output 的，                               
因为它们已消耗最多计算资源但也占最多 KV cache），释放其 KV cache，被踢请求回到                              
waiting queue 重新 Prefill                                                                                  
3. 极端情况：踢到只剩 1 个请求仍不够 → abort 该请求

  当前 Tier 2 DVFS 不感知显存——只根据 (bs, il, ol, slack) 选频率。这意味着显存紧张时  
  Tier 2 可能选了低频节能，反而延长了请求驻留时间，加剧 KV cache 压力，最终触发更多  
  retract（浪费已完成的计算）。                                                                                  

  Tier 2 显存感知扩展（以下为设计探索，非当前实现范围；推荐方向 A）：                                                                              

  方向 A — 显存感知的频率决策（推荐，增量最小）：  
  在 Tier 2 频率选择中加入 KV cache 占用率作为输入。当占用率超过阈值（如 85%）时，  
  强制升频加速当前请求完成，更快释放 KV cache。  
  优点：改动最小（只需在 af_dvfs_controller 的 pick 函数中加一个条件），不改变 scheduler  
  的 retract 逻辑，与现有架构完全兼容。  
  实现：kv_util = 1 - available_size / total_size；if kv_util > 0.85: fallback (1410, 1410)。（阈值 85% 留出 15% 空间，足以容纳一个典型最大序列请求的 KV cache 进入，避免在等待准入期间 OOM）  
  效果：主动升频比被动 retract 更优——retract 意味着浪费已完成的 Prefill + 部分 Decode  
  计算，而升频只是临时多耗一点能量。                                                                             

  方向 B — 显存感知的准入控制：  
  Tier 2 根据当前 KV cache 占用 + 新请求预计的 KV 需求，决定是否接收新请求进入 Decode  
  batch。类似 throttLL'eM 的 Scoreboard 机制。  
  缺点：需要生成长度预测模型（估计新请求会占多少 KV cache），增加系统复杂度。且 sglang  
  已有 max_running_requests 限制和 retract 机制，准入控制的增量收益有限。                                        

  方向 C — 联合优化频率 + batch size：  
  当显存紧张时，Tier 2 主动缩小 batch（暂缓接收新请求）+ 升频加速现有请求完成，而不是  
  等到 OOM 才 retract。  
  缺点：Tier 2 需要介入 scheduler 的 batch 组装逻辑，跨模块耦合严重，实现复杂度高。                             

  推荐方向 A：它是唯一不需要改变 scheduler 逻辑、不需要预测模型、不增加跨模块耦合的  
  方案。核心洞察是"升频加速释放 > 被动 retract 浪费计算"，用 O(1) 的条件判断换取  
  显著减少 retract 次数。方向 B/C 作为 Future Work 保留。                                  

  Qwen3-32B（GQA, bf16, L=64, num_kv_heads=8, head_dim=128）：                                                           

  KV_cache/token = 2 × L × (num_kv_heads/tp) × head_dim × 2 bytes ≈ 0.25 MB / tp  
  tp=1, bs_max_D=256, seq_max_D=4096: KV_cache ≈ 250GB → OOM → 被迫选 tp ≥ 4                                           

  1.3 搜索空间                                                                                                   

  名义空间：k_P/k_D ~8 种 × tp 4^4 × freq 6^4 ≈ 21M 组合。                                                       

  剪枝链：                                                                                                       

1. 资源约束 (1)：排除总 GPU 超限组合
2. 显存约束 (10)-(13)：排除 OOM 组合                                                                            
3. SLO 约束 (5)(6)：排除低频不可行组合
4. 平衡约束 (9)：排除极端失衡组合                                                                              
5. Pareto 剪枝：每池 24 种配置只保留 Pareto 前沿（没有被其他配置同时在延迟和能耗上支配的）

  最终可行解：500-5000，ILP 秒级可解。                                                                           

  1.4 重规划触发                                                                                                 

  4 个 Monitoring 指标（每 10-30s 采样），任一在连续 2+ 窗口超阈值：                                             

  metric_1: SLO_violation_rate > 1%     → 当前配置跟不上负载  
  metric_2: |A_util - F_util| > 0.2    → A/F 配比失衡  
  metric_3: |P_util - D_util| > 0.2    → P/D 配比失衡  
  metric_4: KL(当前分布 || 参考分布) > thr → 负载特征变化                                                        

  重要区分：metric_2（A/F  
  利用率失衡）需要区分"主动降频导致的低利用率"和"负载不足导致的低利用率"。实践中可以加一个条件：如果 Tier 2  
  当前正在用低频且 SLO 未违反 → 不触发重规划（这是主动节能，不是配置问题）。                                     

  "连续多个窗口"避免单窗口统计噪声导致频繁重规划。                                                               

  1.5 过渡策略                                                                                                   

  三种场景：                                                                                                     

1. 仅频率变化（最常见）：直接切频 ~6ms，无请求中断
2. TP 变化：drain-then-switch——停止接收新请求到旧实例 → 等待活跃请求完成 → 激活预创建的 shadow 实例            
3. P/D 重分配：先缩再扩

  过渡期 Tier 2 自动升满频率补偿临时容量下降。                                                                   

---

  二、Tier 2：算子级 DVFS                                                                                        

  2.1 为什么需要 Tier 2

  Tier 1 用保守 M=1 模型有 ~27%（该数字以 tp=4, il=1024, bs=4 配置为例，实际值随 F/A 比变化）slack，且用负载分桶 
  P90 代表值——实际 batch 可能更小（更多 slack）或更大（需紧急升频）。Tier 2 用精确 pipeline 公式在毫秒级利用这些 
  slack。                                                                                                        

  2.2 频率切换约束                                                                                               

  实测 A800-80GB SXM, SetGpuLockedClocks：P50 ~4.5ms, avg ~6ms。多 GPU 因 nvidia.ko 全局锁串行化（N 卡 ×  
  6ms）。AF 分离下每池独立调频，单次只涉 1-2 卡，开销 6-12ms。

  关键约束：不能在 A/F stage 之间切频。64 层 × 6ms/层 = 384ms 额外开销，完全不可行。调频在 scheduler 层、batch开始前做一次。

  多节点部署优势：A/F 分属不同节点时，两个节点的 nvidia.ko 锁互不影响——可同时切频，
  总开销保持在 ~6ms 而非串行的 ~12ms。这对 Decode 决策窗口是利好（更短的窗口即可满足 10% 开销约束）。


  2.3 Prefill DVFS                                                                                               

  输入: bs, il, slack_us = min(deadline - elapsed), M, remaining_layers (默认 L)  
  输出: (f_A, f_F)                                                                                               

1. for (f_a, f_f) in all 36 combos:                                                                            
  t_layer = max(M_PA.lat(f_a), M_PF.lat(f_f)) + t_comm / M   # 精确 pipeline 公式                           
  if t_layer × remaining_layers > slack_us: continue          # 不满足 SLO                                  
  total_e = (M_PA.energy(f_a) + M_PF.energy(f_f)) × remaining_layers                                        
  记录 (f_a, f_f, total_e)                                                                                  
2. if 没有可行组合:                                                                                            
  fallback → (1410, 1410), _stats_fallback += 1                                                             
3. return argmin total_e

  slack 的 fallback：如果请求缺少 api_server_dispatch_time（时间戳丢失），elapsed 视为 0，slack 保持为完整的  
  slo_us——等于不约束频率，Tier 2 以最大空间追求节能。

  为什么是 min(slack) 而不是平均？ 同一 batch 的所有请求共享一次 forward pass，频率必须统一。最紧的 deadline  
  约束整个 batch。这是保守但安全的设计——宁可多耗能，不违反任何请求的 SLO。

  为什么不用 MPC？ BiScale 用 MPC 是因为 Prefill 只有一个频率旋钮。AF 分离提供两个空间自由度（f_A,  
  f_F），单步穷举的搜索空间已经够大，不需要在时间维上做预测。

  2.4 Decode DVFS                                                                                                

  2.4.1 窗口大小                                                                                                 

  W = max(10, int(10 × T_SWITCH_US / t_iter_avg_us + 0.5))  其中 T_SWITCH_US = 6000 (6ms 切频开销)                                                            

  代码使用四舍五入（int(x+0.5)），与 ceil 绝大多数情况等价。                                                     

- bs=16, t_iter≈1.5ms → W = int(60/1.5 + 0.5) = 40 iterations                                                  
- bs=64, t_iter≈3ms → W = int(60/3 + 0.5) = 20 iterations
- bs=4, t_iter≈0.8ms → W = int(60/0.8 + 0.5) = 75 iterations

  10× 乘数确保切频开销 ≤ 10% 窗口时间。                                                                          

  2.4.2 三种触发条件                                                                                             

  def should_reevaluate_decode(current_bs, current_tpot_us, slo_tpot_us) -> int:                                 

```
  # cond_1: 窗口到期                                                                                         
  if iters_since_decision >= window_size:                                                                    
      return REEVAL_WINDOW_EXPIRED                                                                           
                                                                                                             
  # cond_2: batch 大小显著变化                                                                               
  if abs(current_bs - last_bs) / last_bs > 0.3:                                                              
      return REEVAL_BS_CHANGE                                                                                
                                                                                                             
  # cond_3: 延迟逼近 SLO（紧急）                                                                             
  if current_tpot_us > 0 and slo_tpot_us > 0                                                                 
     and current_tpot_us > slo_tpot_us * 0.9:                                                                
      return REEVAL_SLO_URGENT                                                                               
                                                                                                             
  return REEVAL_NONE                                                                                         
                                                                                                             
```

- WINDOW_EXPIRED：保底。即使负载不变，定期检查是否有更好频率                                                   
- BS_CHANGE (>30%)：DF 的频率敏感性随 bs 显著变化。30% 而非 10%——频率一档是 240MHz，小幅 bs 波动不值得重评估
- SLO_URGENT：安全阀。不等窗口到期，TPOT 打过 90% SLO 立即升频

  tick 时序说明：tick_decode_iteration() 在 should_reevaluate_decode() 之前调用，所以判断时 iters_since_decision 
  已经 +1 了。窗口到期的实际时刻比直觉上早一个 iteration。这不会产生错误——只是意味着"窗口到期"的判断时机是"第 W  
  个 iteration 结束时"，而非"第 W+1 个 iteration 开始时"。                                                       

  2.4.3 频率选择（修正版）                                                                                       

  def select_freq_decode(bs, il, ol, slo_tpot_us, M, reeval_reason):                                             

```
  # w_remaining 取决于触发原因                                                                               
  if reeval_reason == REEVAL_WINDOW_EXPIRED:                                                                 
      w_remaining = window_size       # 新窗口 → 完整窗口长度                                                
  else:                                                                                                      
      w_remaining = window_size - iters_since_decision  # 中断窗口 → 剩余迭代                                
                                                                                                             
  # 按能耗升序排列                                                                                           
  candidates = [(E(f_a,f_f), f_a, f_f) for all 36].sort()                                                    
                                                                                                             
  for e, f_a, f_f in candidates:                                                                             
      t_layer = max(M_DA.lat(f_a), M_DF.lat(f_f)) + t_comm / M                                               
      if t_layer × num_layers > slo_tpot_us:      # 不满足 SLO                                               
          continue                                                                                           
                                                                                                             
      switched = _should_switch(f_new, f_cur, bs, il, ol, w_remaining)                                       
      # 注意：以下 3 行在循环内无条件执行，不会 continue 到下一个候选                                        
      _update_decode_state(f_a, f_f, bs, switched)                                                           
      return DVFSDecision(                                                                                   
          f_a=f_a, f_f=f_f,                                                                                  
          energy_mj=e × num_layers,                                                                          
          latency_us=t_layer × num_layers,                                                                   
          switched=switched,                                                                                 
      )                                                                                                      
                                                                                                             
  # fallback: 所有候选都不满足 SLO                                                                           
  self._stats_fallback += 1                                                                                  
  self._stats_switch_up += 1                                                                                 
  self._update_decode_state(F_MAX, F_MAX, bs, switched=True)                                                 
  return DVFSDecision(f_a=F_MAX, f_f=F_MAX, switched=True)                                                   
                                                                                                             
```

  关键行为（修正后的描述）：第一个满足 SLO 的候选直接返回，不管惰性判断的结果。switched=True → scheduler  
  执行硬件切频；switched=False → 维持当前频率，但控制器内部更新 last_bs、重置  
  iters_since_decision，记录"这个候选是最优的，但不值得切"。不会 continue 到下一个候选。                         

  为什么这是正确的？ candidates 是按能耗升序的。第一个满足 SLO 的就是最低能耗可行解。如果它不值得切（savings <  
  1800mJ），下一个更高能耗的候选更不值得切——savings 只会更小。所以停在第一个可行候选是最优策略。

  注意 fallback 路径：如果所有候选都不满足 SLO，fallback 路径调用 _update_decode_state(F_MAX, F_MAX, bs,  
  switched=True)。这是唯一一次调用——之前的 _update_decode_state 只在找到可行候选时调用（循环内），如果循环全部
  continue 了，不会进入循环体。                                                                                  

  2.4.4 惰性切换                                                                                                 

  def _should_switch(f_new, f_cur, bs, il, ol, w_remaining):                                                     

```
  # 条件 1: 频率有变化                                                                                       
  if f_a_new == f_a_cur and f_f_new == f_f_cur:                                                              
      return False                                                                                           
                                                                                                             
  # 条件 2: 变化至少一档 (240MHz)                                                                            
  if abs(f_a_new - f_a_cur) + abs(f_f_new - f_f_cur) < 240:                                                  
      return False                                                                                           
                                                                                                             
  # 条件 3: 省的电 > 切频成本                                                                                
  e_cur = _layer_energy(f_cur, bs, il, ol)    # 单层 mJ                                                      
  e_new = _layer_energy(f_new, bs, il, ol)    # 单层 mJ                                                      
  savings = (e_cur - e_new) × num_layers × max(w_remaining, 1)                                               
  return savings > 1800  # mJ                                                                                
                                                                                                             
```

  1800mJ 来源：P_avg × t_switch ≈ 300W × 6ms = 1.8J = 1800mJ。                                                   

  为什么 savings 乘 num_layers？ e_cur 和 e_new 是单层能耗。num_layers × w_remaining iterations = 全模型总节能。 

  正例（bs=16, f_cur=(1410,1410)→f_new=(210,1170), w_remaining=40）：                                            

  e_cur/层 = 60.4 mJ, e_new/层 = 48.4 mJ  
  savings = (60.4 - 48.4) × 64 × 40 = 30,720 mJ >> 1800 mJ → 切 ✅                                               

  反例（bs=4, f_cur=(930,930)→f_new=(690,930), w_remaining=5）：                                                 

  e_cur/层 = 16.5 mJ, e_new/层 = 16.1 mJ  
  savings = (16.5 - 16.1) × 64 × 5 = 128 mJ < 1800 mJ → 不切 ❌                                                  

  ▎ 注：以上能耗数字为示意值（手工估算），用于说明算法行为，非精确 Profile 数据。                                

  2.5 分布式语义                                                                                                 

  _apply_freq 不是由一个中心化控制器同时设置 A 和 F 的频率。在 AFD 架构中，Attn 节点和 FFN  
  节点是独立进程。每个进程的 scheduler 调用 _apply_freq 时：

  def _apply_freq(f_a, f_f):  
      from sglang.srt.layers.afd import afd_is_attn  
      target_f = f_a if afd_is_attn() else f_f    # 每个进程只设自己的  
      self._dvfs_hw.lock_sm_clock(target_f)                                                                      

  A 进程只设 f_a，F 进程只设 f_f。两者运行相同的 select_freq_* 算法（相同模型 + 相同输入 →  
  确定性输出），所以选取的 (f_a, f_f) 对是一致的。当前没有显式校验，依赖确定性保证。                             

---

  三、Tier 1 和 Tier 2 的协同

  3.1 时序全景    

  T=0s     启动 ILP → 初始配置 → 启动 AFD → Tier 2 从 f̄ 开始调频

  T=0~N    稳态  
           │  每 ~1ms: decode iteration  
           │    tick → should_reevaluate? → 大多是 NONE  
           │    每 ~60 iters: WINDOW_EXPIRED → 重选频率  
           │  每 ~100ms: prefill batch  
           │    select_freq_prefill → 重选频率  
           │  每 10-30s: Monitoring 采样 → 正常，不触发                                                          

  T=N      负载突变（如 λ 翻倍）  
           ├─ Tier 2 毫秒级: SLO_URGENT → 升满频率  
           ├─ 若频率不够: 准入控制（排队）  
           └─ Monitoring 2 窗口确认 → 触发 Tier 1 重规划                                                         

  T=N+20s  Tier 1 新配置 → 过渡 → Tier 2 在新配置上调频                                                          

  3.2 分工本质                                                                                                   

  ┌──────────┬─────────────────────┬───────────────────────────────┐  
  │          │       Tier 1        │            Tier 2             │
  ├──────────┼─────────────────────┼───────────────────────────────┤  
  │ 时间尺度 │ 分钟                │ 毫秒                          │
  ├──────────┼─────────────────────┼───────────────────────────────┤  
  │ 决策变量 │ 10 个离散           │ 2 个离散（36 组合）           │  
  ├──────────┼─────────────────────┼───────────────────────────────┤  
  │ 延迟模型 │ M=1 保守上界        │ M≥1 精确 pipeline             │  
  ├──────────┼─────────────────────┼───────────────────────────────┤  
  │ 负载模型 │ 统计分桶 P90        │ 实际 batch                    │
  ├──────────┼─────────────────────┼───────────────────────────────┤  
  │ 设计哲学 │ "保证绝对安全"      │ "在安全边界的 slack 内尽量省" │
  ├──────────┼─────────────────────┼───────────────────────────────┤  
  │ 失效模式 │ 配置保守 → 多余 GPU │ 频率过低 → SLO_URGENT 升频    │
  └──────────┴─────────────────────┴───────────────────────────────┘                                             

  3.3 协同例子                                                                                                   
  3.3 协同例子
  Tier 1 配了 f̄_DA=1410, f̄_DF=930。运行时 bs=8。

  Tier 2 从 (1410, 930) 出发，首次重评估：

  候选 (210, 1410): t_layer = max(222, 780) + 20 = 800us × 64 = 51.2ms ≈ TPOT_SLO
    通过！e = 58.4 mJ/层，当前 (1410,1410) e = 60.4 mJ/层
    savings = (60.4 - 58.4) × 64 × 60 = 7680 mJ > 1800 → 切 ✅

  Tier 2 把 f_DA 从 Tier 1 的 1410 降到 210——因为 runtime 的 M=2 pipeline 公式下 max(t_A, t_F) 让 DA 的降频不影响总延迟。这是 Tier
   1 用 M=1 保守模型无法发现的。

  ▎ 注：t_DA=222us、e_DA 等值为示意值，用于说明算法行为。

  当 Tier 2 持续偏离基线：Tier 2 连续 10 分钟用 f_DA=210（而非 f̄_DA=1410），Monitoring 检测到 DA 利用率异常低。但如果是 Tier 2
  主动降频且 SLO 未违反，应抑制重规划——这是正常的节能行为。只有 SLO 违反率升高或 A/F 利用率失衡伴随 SLO 压力时，才触发 Tier
  1。具体逻辑在 WorkloadMonitor 实现时加入。

  3.4 为什么没有动态 TP 也能保证 SLA

  频率提供 ~50% 吞吐弹性（f_min → f_max），准入控制处理过载，重启处理结构性变化。动态 TP
  是锦上添花（让过渡更平滑），不是雪中送炭。三层防线：

  ┌───────────┬──────────────────────┬─────────────┬──────────────────────────────┐
  │ 负载变化  │         手段         │  响应时间   │             SLA              │
  ├───────────┼──────────────────────┼─────────────┼──────────────────────────────┤
  │ ±20% 波动 │ Tier 2 调频          │ ~~6ms        │ ✅ 不违反                    │
  ├───────────┼──────────────────────┼─────────────┼──────────────────────────────┤
  │ +30~~50%   │ 升满 + 准入控制      │ ~6ms + 排队 │ ✅ 排队不丢                  │
  ├───────────┼──────────────────────┼─────────────┼──────────────────────────────┤
  │ 持续翻倍  │ Tier 1 重规划 + 重启 │ 分钟        │ ⚠️  过渡期靠频率+接入控制兜底 │
  └───────────┴──────────────────────┴─────────────┴──────────────────────────────┘