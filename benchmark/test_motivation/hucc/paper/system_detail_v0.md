  AFlex 两层控制系统详解                                                                                         

---

  零、初始化：AF TP 配置从哪里来                                                                                 

  在深入 Tier 1/2 之前，先搞清楚系统启动时 AF 的 TP 是怎么确定的。

  0.1 两种初始化路径                                                                                             

  路径 A — Startup ILP（推荐，论文方案）：                                                                       

  启动参数:  
    --tp-size 8              # 总共 8 GPU  
    --afd-dvfs-enabled        # 启用 Tier 2 调频  
    --afd-energy-model-dir /path/to/models/                                                                      

  内部流程:  
    1. Tier 1 ILP 求解:  
         输入: G=8, λ=估计值, workload_bins=默认分布  
         输出: k_P, k_D, tp_PA, tp_PF, tp_DA, tp_DF, f̄_PA..f̄_DF  
    2. 按 ILP 输出启动 AFD 实例  
    3. Tier 2 初始化时从 server_args 读取 tp_a, tp_f                                                             

  路径 B — 手动指定（当前实现，用于开发调试）：                                                                  

  --tp-size 4               # 基础 TP（用于非 AFD 模式）  
  --afd-attn-tp 2            # A 侧 TP（AFD 模式下覆盖 tp-size）  
  --afd-ffn-tp 4             # F 侧 TP  
  --afd-dvfs-enabled  
  --afd-energy-model-dir /path/  
  --afd-ttft-slo-ms 5000     # TTFT SLO 预算  
  --afd-tpot-slo-us 50000    # TPOT SLO 预算（单 token）                                                         

  0.2 初始化代码路径                                                                                             

# scheduler.py **init** (line 520-526)

  self._cur_f_a = 1410          # 初始频率 = 最高（安全起步）  
  self._cur_f_f = 1410  
  if server_args.afd_dvfs_enabled:  
      self._init_afd_dvfs(server_args)                                                                           

# _init_afd_dvfs (line 1540-1555)

  predictor = AFProfilePredictor(model_dir)           # 加载 .pkl 模型  
  tp_a = server_args.afd_attn_tp or server_args.tp_size  # A 侧 TP  
  tp_f = server_args.afd_ffn_tp or server_args.tp_size  # F 侧 TP  
  self._af_dvfs_ctrl = AFDVFSController(  
      predictor=predictor,  
      num_layers=64,        # Qwen3-32B  
      tp_a=tp_a,            # 异构 TP 支持  
      tp_f=tp_f,  
  )                                                                                                              

  关键点：tp_a 和 tp_f 在启动时确定，运行时不变（模块 C 降级后）。Tier 1 如果以后实现，它的输出会覆盖  
  --afd-attn-tp / --afd-ffn-tp 的值，或直接作为启动参数传入。

---

  一、Tier 1：联合资源规划 (ILP)

  1.1 为什么需要 Tier 1

  Tier 2 能做的事情有边界——它只能调频率，不能改变"几块 GPU 跑 Attention、几块跑 FFN、用多大 TP"。Tier 1  
  的职责就是在分钟尺度上，根据负载画像，确定这些结构性配置，使 Tier 2 有足够大且合理的调频空间。                 

  1.2 目标函数                                                                                                   

  minimize:  Σ_{c ∈ {P,D}}  k_c × [ E_A(tp_cA, f̄_cA, wl) + E_F(tp_cF, f̄_cF, wl) + E_bubble(M) ]                  

  逐项拆解：                                                                                                     

- k_c：Prefill（或 Decode）有多少对 AF 实例。一对 = 一条流水线。                                               
- E_A(tp_cA, f̄_cA, wl)：在给定 TP、频率、负载下，单层 Attention 能耗，直接从 Profile 表查（单位 mJ）。
- E_F(...)：同上，FFN 单层能耗。                                                                               
- E_bubble(M)：流水线气泡能耗，下面详述。

  为什么是"单层"而不是"整个模型"？ Profile 表测的就是单层延迟和单层能耗。总能耗 = 单层 × 64 层 ×  
  迭代数，但迭代数是动态的（取决于请求数量）。Tier 1  
  只需要比较单位能耗——哪个配置的每层能耗更低，总能耗就成比例更低。                                               

  E_bubble 为什么在目标函数里？                                                                                  

  E_bubble = P_idle(f_wait) × max(0, t_slow - t_fast) × (M-1)/M                                                  

  当 A 和 F 延迟不同时，快的一方算完了要等慢的。等待期间的 GPU 空转功耗就是 bubble。注意：                       

- f_wait 是等的那一方的频率（不是被等的那一方）。谁先完成谁等，等的过程按自己的频率耗电。                      
- (M-1)/M：M 个 microbatch 中，第一个启动到最后一个排空，有 M-1 个 microbatch 存在等待。
- M=1 时 (M-1)/M=0，无 bubble（因为 A 和 F 完全串行，没有同时运行就没有互相等待）。

  E_bubble 放在目标函数里的设计意图：ILP 会自动倾向选 A/F 延迟接近的配置（bubble 小），而不是事后惩罚。比如 tp=4,
   il=1024, bs=4 时 F/A=4.09x，bubble 巨大，ILP 自然会避免。                                                     

  1.3 约束逐条解析                                                                                               

  约束 (1)：资源总量                                                                                             

  k_P × (tp_PA + tp_PF) + k_D × (tp_DA + tp_DF) ≤ G                                                              

  每条 AF 流水线占用 tp_A + tp_F 个 GPU。为什么是加号不是乘号？因为 AF 架构下 A 和 F 是空间上分离的，一块 GPU  
  不能同时跑 Attention 和 FFN。                                                                                  

  举例：G=16。如果 k_P=2, tp_PA=2, tp_PF=3（一对占用 5 GPU，两对 10 GPU），那么留给 Decode 的最多是 6 GPU。      

  约束 (2)(3)(4)：变量域                                                                                         

  tp ∈ {1, 2, 4, 8}     # TP 只能选 2 的幂，且受 num_kv_heads=8 整除约束  
  k_P, k_D ∈ Z⁺          # 至少 1 对  
  f̄ ∈ {210, 450, 690, 930, 1170, 1410}  # A800 支持的 SM 频率                                                    

  为什么频率离散？ SetGpuLockedClocks 只能设 NVML supported_sm_clocks 列表中的值。这不是设计选择，是硬件限制。   

  约束 (5)(6)：延迟 SLO                                                                                          

  Prefill:  t_PA(r) + t_PF(r) + t_comm ≤ TTFT_SLO / L  
  Decode:   t_DA(r) + t_DF(r) + t_comm ≤ TPOT_SLO / L                                                            

  三个关键设计决策：                                                                                             

1. 为什么用 M=1 的串行公式而不用 M>1 的 pipeline 公式？

  这是整个设计最重要的保守假设。M=1 时 t_layer = t_A + t_F + t_comm，M>1 时 t_layer = max(t_A, t_F) +  
  t_comm/M。串行公式总是 ≥ pipeline 公式（上界）。

  Tier 1 是静态规划——它在几分钟前做的决定要保证未来几分钟内"不管实际 M 取多少、不管 batch 怎么变，SLO  
  都不会违反"。用最坏情况（M=1）做可行性检查是最安全的。

  Tier 2 在运行时知道实际的 M 和 batch 特征，用精确公式去发现真实的 slack，补回 Tier 1 的保守性。                

1. 为什么 Prefill 和 Decode 都 /L？

  TTFT_SLO 是整个 Prefill 请求的延迟预算（如 500ms）。t_PA + t_PF + t_comm 是单层延迟。单层预算 = 总预算 / 层数。

  同理，TPOT_SLO 是生成一个 token 的延迟预算（如 50ms）。一个 token 经过 64 层，单层预算 = TPOT_SLO / 64。       

1. ∀r ∈ R_prefill 的 r 是什么？

  不是每个请求，是"请求类型桶"的代表值（见 1.4 负载分桶）。假设 Prefill 分了 4 个桶，那就生成 4 条约束  
  (5)。桶的代表值取 P90——比均值保守（覆盖长尾），比最大值高效（不过度浪费）。

  约束 (7)(8)：吞吐容量                                                                                          

  k_P × Thpt_pair_P(M) ≥ (1+α) × λ  
  k_D × Thpt_pair_D(M) ≥ (1+α) × N_active                                                                        

  Thpt_pair 的两种模式：                                                                                         

- M=1：Thpt_pair = 1 / (t_A + t_F + t_comm)。A 和 F 串行，一个 iteration 的时间 = 三者之和。                   
- M>1：Thpt_pair ≈ 1 / max(t_A, t_F)。pipeline 重叠后，吞吐卡在慢的那一方。

  为什么不是 1/t_A + 1/t_F？ 因为 A 和 F 是串在一条流水线上的，A 的输出是 F  
  的输入，不能独立并行。一条线只有一个瓶颈——较慢的那一方。                                                       

  (1+α) 的物理含义：α=0.1~0.2，是容量裕度。负载监控窗口的 λ 是统计均值，实际有波动。留 10-20% buffer  
  防止统计误差导致容量不足。

  N_active 为什么代替吞吐？ 在 continuous batching 中，Decode 的吞吐 = 每秒处理的 iteration 数 ×  
  batch_size。每个活跃 decode 请求每 iteration 生成一个 token，所以所需吞吐 ∝ N_active。这是个近似——精确的 Decode
   吞吐模型需要离散事件模拟，不在 Tier 1 的精度范围内。                                                          

  约束 (9)：A/F 延迟平衡                                                                                         

  |t_A - t_F| ≤ β × max(t_A, t_F)      β ∈ [0.5, 0.8]                                                            

  这是剪枝约束，不是硬可行性约束。                                                                               

  和 E_bubble 的关系：E_bubble 在目标函数里做连续惩罚（10% 失衡 → 小代价，50% 失衡 → 大代价，ILP  
  会自动避免）。约束 (9) 是硬剪枝——失衡超过 β=0.8 的配置直接被排除，不等 ILP 评估，减少搜索空间。

  β 为什么是 0.5~~0.8 而不是 1.0？ β=1.0 意味允许 A 是 F 的 2 倍或 F 是 A 的 2 倍。对 β=0.8，如果 A 是 1ms，F~~  
  ~~必须在 0.2~~1.8ms 之间。超过这个范围说明 pipeline 效率极低，不如换 TP 配置改变 A/F 延迟比。

  约束 (10)(11)：显存                                                                                            

  Mem_A = W_attn/tp + 2×L×(num_kv_heads/tp)×head_dim×bs_max×seq_max×2B ≤ GPU_MEM  
  Mem_F = W_ffn/tp  + activation_buffer ≤ GPU_MEM                                                                

  A 和 F 显存需求的结构性差异：                                                                                  

  A 实例有 KV cache。Qwen3-32B（bf16, 64 层, 32 kv_heads, head_dim=128, L=64）：                                 

  KV_cache/token = 2 × 64 × (32/tp) × 128 × 2 bytes ≈ 1 MB / tp                                                  

  tp=1, bs_max=256, seq_max=4096：KV_cache = 256×4096×1MB ≈ 1TB → OOM。所以 ILP 会被迫选 tp≥4。                  

  F 实例没有 KV cache，但 FFN 权重大（~59GB for gate+up+down）。tp=1 时 59GB < 80GB 可以放下，但加上 activation  
  buffer 就紧张了。tp=2 时 29.5GB 就很安全。

  为什么显存约束是必要的？ 小 TP + 大 batch → KV cache 爆炸。如果不约束，ILP 可能选 tp=1（A latency  
  最低→频率可以最低→最省电），但运行时直接 OOM。

  1.4 搜索空间分析                                                                                               

  名义空间：                                                                                                     

  k_P, k_D: ~8 种取值各（1 到 G/2）  
  tp_PA, tp_PF, tp_DA, tp_DF: 4 种各 → 4^4 = 256  
  f_PA, f_PF, f_DA, f_DF: 6 种各 → 6^4 = 1296                                                                    

  名义总量：~8×8×256×1296 ≈ 21M                                                                                  

  剪枝链：                                                                                                       

1. 配对约束（k_PA=k_PF, k_DA=k_DF）：k 变量从 4 个降到 2 个。                                                  
2. 资源约束 (1)：k_P×(tp_PA+tp_PF) + k_D×(tp_DA+tp_DF) ≤ G，对 G=16，绝大多数 (k_P, k_D, tp) 组合被排除。
3. 显存约束 (10)(11)：小 TP + 大 bs 的组合被排除。                                                             
4. SLO 约束 (5)(6)：低频组合大部分被排除。如 f_PF=210MHz 对所有 il≥512 都是不可行的。                          
5. 平衡约束 (9)：极端失衡配置被剪枝。                                                                          
6. Pareto 剪枝：每个池的 4×6=24 种 (tp, freq) 配置，只保留 Pareto                                              
  前沿上的（没有被其他配置同时在延迟和能耗上支配的）。

  最终可行解：500-5000，ILP 秒级可解。                                                                           

  1.5 重规划触发                                                                                                 

  4 个 Monitoring 指标，任一在连续 2+ 窗口超阈值 → 触发 Tier 1：                                                 

  metric_1: SLO_violation_rate > 1%     → 当前配置跟不上负载，需更多资源  
  metric_2: |A_util - F_util| > 0.2    → A/F 配比失衡  
  metric_3: |P_util - D_util| > 0.2    → P/D 配比失衡  
  metric_4: KL(当前分布 || 参考分布) > thr  → 负载特征变化                                                       

  为什么需要"连续多个窗口"？ 避免单窗口的统计噪声导致频繁重规划——切一次配置的成本（drain + shadow +  
  重启）远高于频率微调。                                                                                         

---

  二、Tier 2：算子级 DVFS

  2.1 为什么 Tier 1 不够，还需要 Tier 2

  Tier 1 的保守 M=1 模型有 ~27% 的 slack 空间。而且 Tier 1 用的是负载分桶的 P90 代表值——实际 batch 可能比 P90  
  小（有更多 slack）或更大（需要紧急升频）。Tier 2 在毫秒级利用这些 slack 做精细节省。                           

  2.2 Prefill DVFS                                                                                               

  输入: bs, il, slack_us = min(deadline - elapsed), M, remaining_layers  
  输出: (f_A, f_F)                                                                                               

  第 1 步：计算 slack                                                                                            

  slack_us = min_{r∈B}(d_r - elapsed_r)                                                                          

# d_r = TTFT_SLO for request r

# elapsed_r = time since request arrived at API server

  这是最保守的算法——用 batch 中最紧的 deadline 约束整个 batch。所有请求共享同一个 forward  
  pass，不能单独给某个请求调频。宁可多耗电，不违反 SLO。                                                         

  第 2 步：遍历 36 种组合                                                                                        

  for f_a, f_f in [(210,210), (210,450), ..., (1410,1410)]:  # 36 种  
      t_layer = max(t_PA(f_a), t_PF(f_f)) + t_comm / M      # pipeline 公式  
      total_lat = t_layer × remaining_layers  
      if total_lat > slack_us:                               # 不满足 SLO → 跳过  
          continue  
      total_e = (e_PA(f_a) + e_PF(f_f)) × remaining_layers   # 总能耗  
      记录 (f_a, f_f, total_e)                                                                                   

  关键：Tier 2 用精确 pipeline 公式 max(t_A, t_F) + t_comm/M，而非 Tier 1 的保守串行公式。这 27%  
  的差异就是额外节能空间。                                                                                       

  第 3 步：选能耗最低的                                                                                          

  return argmin_(f_a,f_f) total_e                                                                                

  如果没有可行组合（所有 36 种都超 SLO）→ fallback 到 f_max (1410, 1410)，记录 _stats_fallback += 1。            

  为什么 Prefill 的算法比 Decode 简单？ Prefill 是一次性的——一个 batch  
  进来，算一遍，完事。不需要考虑"如果我不切频，这次的频率选择会影响未来 60 个 iteration"。Prefill
  也不需要惰性切频——每个 Prefill 请求开始时选一次频率即可，不存在"频繁切换"的问题。                              

  2.3 Decode DVFS                                                                                                

  Decode 是 continuous batching——batch 在几十到几千个 iteration 中持续存在，composition  
  缓慢变化。需要用窗口机制避免过于频繁的切频。

  2.3.1 窗口大小动态计算                                                                                         

  W = max(10, ceil(10 × 6000us / t_iter_avg_us))                                                                 

- t_iter_avg_us 是最近 iteration 的墙上时间（μs）                                                              
- 6000us 是切频开销                                                                                            
- 10× 确保切频开销 ≤ 10% 的窗口时间

  举例：                                                                                                         

- bs=16, t_iter≈1.5ms → W = max(10, 60/1.5) = 40 iterations                                                    
- bs=4, t_iter≈0.8ms → W = max(10, 60/0.8) = 75 iterations                                                     
- bs=64, t_iter≈3ms → W = max(10, 60/3) = 20 iterations

  2.3.2 三种重评估触发                                                                                           

  def should_reevaluate_decode(current_bs, current_tpot_us, slo_tpot_us) -> int:                                 

```
  # 触发 1: 定期检查（窗口到期）                                                                             
  if iters_since_decision >= window_size:                                                                    
      return REEVAL_WINDOW_EXPIRED                                                                           
                                                                                                             
  # 触发 2: batch 大小显著变化                                                                               
  if abs(current_bs - last_bs) / last_bs > 0.3:                                                              
      return REEVAL_BS_CHANGE                                                                                
                                                                                                             
  # 触发 3: 延迟逼近 SLO（紧急）                                                                             
  if current_tpot_us > slo_tpot_us × 0.9:                                                                    
      return REEVAL_SLO_URGENT                                                                               
                                                                                                             
  return REEVAL_NONE                                                                                         
                                                                                                             
```

  三种触发的设计逻辑：                                                                                           

- WINDOW_EXPIRED：保底机制。即使负载完全不变，也要周期性检查有没有更好的频率。                                 
- BS_CHANGE：DF 的频率敏感性随 bs 显著变化（小 bs→memory-bound，大 bs→compute-bound）。bs 变化 30%
  足以改变最优频率。30% 而不是 10% 是为了避免频繁触发——频率的一档是 240MHz，小幅 bs 变化不值得重评估。           
- SLO_URGENT：安全阀。不等到窗口到期，只要检测到 TPOT 超过 SLO 的 90% 就立即重新评估（大概率会升频）。

  2.3.3 频率选择                                                                                                 

# 第 1 步: 按能耗升序排列所有 36 种组合

  candidates = [(E(f_a,f_f), f_a, f_f) for all 36].sort()                                                        

# 第 2 步: 确定 w_remaining（剩余窗口迭代数）

  if reeval_reason == REEVAL_WINDOW_EXPIRED:  
      w_remaining = window_size       # 新窗口，全量  
  else:  
      w_remaining = window_size - iters_since_decision  # 中断当前窗口                                           

# 第 3 步: 从低能耗到高能耗，找第一个可行且值得切的

  for e, f_a, f_f in candidates:  
      t_layer = max(t_DA(f_a), t_DF(f_f)) + t_comm / M  
      if t_layer × 64 > slo_tpot_us:    # 不满足 SLO  
          continue  
      if _should_switch(f_new, f_cur, w_remaining):  # 惰性判断  
          return (f_a, f_f)              # 第一个通过 = 最低能耗  
  return (1410, 1410)                     # fallback                                                             

  为什么是"第一个通过 = 最优"？ candidates 是按能量升序排列的，第一个既能满足 SLO  
  又能通过惰性判断的，就是最低能耗可行解。                                                                       

  为什么 w_remaining 依赖触发原因？ WINDOW_EXPIRED 意味新窗口开始了，有完整的 window_size 个 iteration  
  去摊销切频成本。BS_CHANGE 或 SLO_URGENT 是中断当前窗口，只剩下 window_size - iters 个 iteration。如果只剩 3 个
  iteration，可能不值得切——省的电不够 cover 切频开销。                                                           

  2.3.4 惰性切换                                                                                                 

  def _should_switch(f_new, f_cur, bs, il, ol, w_remaining):  
      # 条件 1: 频率有变化  
      if f_a_new == f_a_cur and f_f_new == f_f_cur:  
          return False                                                                                           

```
  # 条件 2: 变化至少一档                                                                                     
  freq_delta = |f_a_new - f_a_cur| + |f_f_new - f_f_cur|                                                     
  if freq_delta < 240:  # F_STEP_MIN                                                                         
      return False                                                                                           
                                                                                                             
  # 条件 3: 省的电 > 切频成本                                                                                
  e_cur = predict_energy(f_cur, bs, il, ol)    # 单层 (mJ)                                                   
  e_new = predict_energy(f_new, bs, il, ol)    # 单层 (mJ)                                                   
  savings = (e_cur - e_new) × 64 × max(w_remaining, 1)                                                       
  return savings > 1800  # mJ                                                                                
                                                                                                             
```

  1800mJ 从哪里来：P_avg × t_switch ≈ 300W × 6ms = 1.8J = 1800mJ。                                               

  为什么 savings 乘 64：e_cur 和 e_new 是单层能耗。64 层 × w_remaining iterations = 总节省。                     

  数值例子：                                                                                                     

  场景：bs=16, f_cur=(1410,1410), f_new=(210,1170), w_remaining=40                                               

  e_cur/层 = 2.4 + 58 = 60.4 mJ  
  e_new/层 = 0.36 + 48 = 48.4 mJ  
  savings = (60.4 - 48.4) × 64 × 40 = 30,720 mJ >> 1800 mJ → 切！✅                                              

  场景：bs=4, f_cur=(930,930), f_new=(690,930), w_remaining=5                                                    

  e_cur/层 = 1.5 + 15 = 16.5 mJ  
  e_new/层 = 1.1 + 15 = 16.1 mJ
  savings = (16.5 - 16.1) × 64 × 5 = 128 mJ < 1800 mJ → 不切！❌                                                 

  第二个例子很关键——不是频率越低越好，如果切频省的电抵不过切频本身消耗的电，维持当前频率才是最优的。             

---

  三、Tier 1 和 Tier 2 的协同运行

  3.1 时序全景    

  T=0s   启动  
         ├─ ILP 求解 → 初始配置 (k_P, k_D, tp, f̄)  
         ├─ 启动 AFD 实例  
         └─ Tier 2 初始化，从 f̄ 开始                                                                             

  T=0~5min  稳态  
         │  每 ~1ms: 一个 decode iteration  
         │    ├─ tick_decode_iteration()  
         │    ├─ should_reevaluate_decode() → 大多返回 NONE  
         │    └─ 每 ~60 iterations: REEVAL_WINDOW_EXPIRED → 重选频率  
         │  
         │  每 ~100ms: 一个 prefill batch  
         │    └─ select_freq_prefill() → 重选频率  
         │  
         │  每 10-30s: Monitoring 采样  
         │    └─ 检查 4 个指标 → 全正常，不触发                                                                  

  T=5min  负载突变  
         ├─ Tier 2 毫秒级响应: SLO_URGENT → 升频  
         ├─ Monitoring 检测到 SLO 违反  
         └─ 连续 2 窗口确认 → 触发 Tier 1 重规划                                                                 

  T=5min+20s  Tier 1 重规划完成  
         ├─ 新配置下发  
         ├─ drain-then-switch（如果需要改 TP）  
         └─ Tier 2 在新配置上开始调频                                                                            

  3.2 两层分工的本质                                                                                             

  ┌──────────┬─────────────────────────┬───────────────────────────────────┐  
  │          │         Tier 1          │              Tier 2               │
  ├──────────┼─────────────────────────┼───────────────────────────────────┤  
  │ 时间尺度 │ 分钟                    │ 毫秒                              │
  ├──────────┼─────────────────────────┼───────────────────────────────────┤  
  │ 决策变量 │ k_P, k_D, tp_*, f̄_*     │ f_A, f_F                          │  
  ├──────────┼─────────────────────────┼───────────────────────────────────┤  
  │ 自由度   │ 10 个离散变量           │ 2 个离散变量（36 组合）           │  
  ├──────────┼─────────────────────────┼───────────────────────────────────┤  
  │ 延迟模型 │ M=1 保守上界            │ M≥1 精确 pipeline                 │
  ├──────────┼─────────────────────────┼───────────────────────────────────┤  
  │ 负载模型 │ 统计分桶 P90            │ 实际 batch 特征                   │
  ├──────────┼─────────────────────────┼───────────────────────────────────┤  
  │ 设计哲学 │ "保证绝对安全"          │ "在安全边界内尽量省"              │
  ├──────────┼─────────────────────────┼───────────────────────────────────┤  
  │ 失效模式 │ 配置过于保守 → 浪费 GPU │ 频率过低 → SLO 违反 → cond_3 升频 │
  └──────────┴─────────────────────────┴───────────────────────────────────┘                                     

  为什么不能只用 Tier 1？ 分钟级的统计窗口跟不上毫秒级的 batch 变化——实际 batch 从 4 跳到 64 只需要几百 ms，Tier 
  1 要几分钟后才反应。而且 Tier 1 的 M=1 保守模型会浪费 ~27% 的节能空间。

  为什么不能只用 Tier 2？ 频率调整只能提供 ~50% 的吞吐弹性。如果负载特征根本变了（比如用户从短 prompt  
  变成长文档），可能所有 36 种频率组合都无法满足 SLO——需要改变 TP 或增加副本数，这是 Tier 1 的职责。

  3.3 协同的具体例子                                                                                             

  场景：Tier 1 配了 f̄_DA=1410, f̄_DF=930。运行时实际 bs=8。                                                       

  Tier 2 从基线 (1410, 930) 开始，在第一次重评估时发现：                                                         

  候选 (f_DA, f_DF) = (210, 930):  
    t_layer = max(222, 1200) + 20 = 1220us × 64 = 78.1ms  
    TPOT_SLO = 50ms → 不通过，skip                                                                               

  候选 (690, 930):  
  为什么不能只用 Tier 2？ 频率调整只能提供 ~50% 的吞吐弹性。如果负载特征根本变了（比如用户从短 prompt 变成长文档），可能所有 36
  种频率组合都无法满足 SLO——需要改变 TP 或增加副本数，这是 Tier 1 的职责。

  3.3 协同的具体例子

  场景：Tier 1 配了 f̄_DA=1410, f̄_DF=930。运行时实际 bs=8。
  3.3 协同的具体例子

  场景：Tier 1 配了 f̄_DA=1410, f̄_DF=930。运行时实际 bs=8。

  Tier 2 从基线 (1410, 930) 开始，在第一次重评估时发现：

  候选 (f_DA, f_DF) = (210, 930):
    t_layer = max(222, 1200) + 20 = 1220us × 64 = 78.1ms
    TPOT_SLO = 50ms → 不通过，skip

  候选 (690, 930):
    t_layer = max(248, 1200) + 20 = 1220us × 64 = 78.1ms
    还是不通过——瓶颈在 DF

  候选 (210, 1410):
    t_layer = max(222, 780) + 20 = 800us × 64 = 51.2ms
    通过！e = 0.36 + 58 = 58.4 mJ/层

  候选 (690, 1410):
    t_layer = max(248, 780) + 20 = 800us × 64 = 51.2ms
    通过。e = 1.2 + 58 = 59.2 mJ/层

  当前 (1410, 1410): e = 2.4 + 58 = 60.4 mJ/层

  最低能耗可行 = (210, 1410)
  savings = (60.4 - 58.4) × 64 × 60 = 7680 mJ > 1800 mJ → 切！

  Tier 2 把 Tier 1 的 f_DA=1410 降到了 210，因为 runtime 发现 DA 的 memory-bound 特性让降频几乎无代价。这是 Tier 1 无法做到的——Tier 1 用 M=1 保守模型时 t_A + t_F
  之和会让这种极端降频看起来不可行，但 M>1 pipeline 下 max(t_A, t_F) 让 DA 的降频不影响总延迟。

  当 Tier 2 持续偏离 Tier 1 基线时：

  如果 Tier 2 连续 10 分钟都在用 f_DA=210（而非 f̄_DA=1410），Monitoring 的 metric_2（A/F 利用率失衡）会触发——DA 利用率低（太闲了），DF 利用率高（瓶颈）。Tier 1
  重规划时会发现可以给 DA 减 TP 或给 DF 加 TP，然后把省下的 GPU 分配出去。

※ recap: 你正在设计 AFlex 系统——AF 分离后用两层控制（Tier 1 ILP 资源规划 + Tier 2 算子级 DVFS）在满足 SLO 下最小化能耗。Tier 2 代码已完审无误，当前任务：继续推进 Tier 1 ILP
   的三个实现文件（profile_table、tier1_solver、workload_monitor）。