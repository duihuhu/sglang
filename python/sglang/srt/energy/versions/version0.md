# v0版本

## 1.修复了inputlen>16384时的报错bug。

#####

根因： 在 AFD 模式下，FFN 侧的 prefill 请求绕过了 bootstrap queue（scheduler.py:2624-2628），因此 req.disagg_kv_sender 保持为 None。当输入长度 
16384 触发 chunked prefill 时，process_prefill_chunk() 在 line 802 直接调用 self.send_kv_chunk(self.chunked_req)，没有像其他两处调用点（lines 606
和 647）那样加上 if not afd_is_ffn() 守卫。

## 2.修复了DF端多请求（>1）时CUDA device-side assert（索引越界）报错bug。

根因： 在 AFD DF 端，第一个迭代中 prepare_for_decode 将 running_batch 的 output_ids 置为 None。下一个迭代中，merge_batch 合并 running_batch（output_ids=None）和新 prebuilt_batch（output_ids=tensor）时，旧代码 if self.output_ids is not None: 条件为 False，跳过了合并，output_ids 保持 None。随后 prepare_for_decode 执行 self.input_ids = self.output_ids = None，导致 forward pass 因 CUDA 索引越界崩溃。
修复： 在 merge_batch（schedule_batch.py:2207-2219）中，当 self.output_ids 为 None 但 reqs 非空时，从 req.output_ids[-1] 重建 output_ids 张量，然后再与 other.output_ids 合并。

## 3. 修复了 AFD prefill/disaggregation 模式下 micro-batch 空张量导致 crash 的 bug以及下游 IndexError。

根因： `_split_seq_indices_m_way`（afd_overlap.py）在 `num_seqs == m` 时可能产生等于 `num_seqs` 的分割点（如 3 个 seq、M=3、extend_lens=[584,617,862] 时产生 `[2, 3]`），尾部 child range `[3, 3)` 为空。修复后返回 `[2]`（2 children），但 `AfdForwardBatchPreparer.prepare` 仍用 `m=3` 构建 boundaries 和迭代，导致 `boundaries_tok[i+1]` 越界 IndexError。

修复（3处）：

1. **`_split_seq_indices_m_way`（afd_overlap.py）** — 根因修复：
   - greedy 路径：追加 split 前检查 `i + 1 < num_seqs`，跳过末尾分割点
   - while 填充：`remaining <= 1` 时 break，且 `min(..., num_seqs - 1)` 防止生成 num_seqs
   - 非 greedy 路径：`if interval * (i + 1) < num_seqs` 过滤无效分割
   
2. **`AfdForwardBatchPreparer.prepare`（afd_overlap.py）** — `actual_m` 适配 + 防御：
   - 用 `actual_m = len(seq_indices) + 1` 替代 `m` 构建 boundaries 和迭代
   - `assert len(children_backends) >= actual_m` 放松断言
   - `if not batch.afd_split_seq_index` 同时捕获 `None` 和 `[]`（1 seq 时返回 []）
   
3. **`afd_prepare_overlap`（scheduler_afd_mixin.py）** — 一致性修复：
   - `batch_size < m` 时显式设置 `batch.afd_split_seq_index = None`，与 FFN 侧 `_prepare_afd_overlap` 行为一致，防止残留值跨 batch 传播

## 4. 修复了 AFD FFN 侧 `embed_tokens` CUDA 越界 crash（PF 和 DF 均修复）。

根因：在 AFD FFN 侧（包括 PF 和 DF），`embed_tokens(input_ids)` 的 `F.embedding` 会触发 `vectorized_gather_kernel` / `IndexKernel` 索引越界断言。所有线程同时失败说明 embedding weight 的 `ind_dim_size` 可能为 0 或极小值。由于 FFN 侧 `forward_afd_A.prepare_mlp` 的 UCX `recv_wait()` 会完全覆盖 hidden_states，embedding 结果实际上被立即丢弃，因此可以安全跳过。

修复（llama.py:420-426）：
- 在 `afd_is_ffn()` 时，用 `torch.zeros` 替代 `self.embed_tokens(input_ids)`
- 非 FFN 侧行为不变，仍然调用 `self.embed_tokens`

## 5. 修复了 DF 端 KV cache 内存泄漏 bug（token_to_kv_pool_allocator memory leak）。

根因：在 AFD DF 端，`prepare_for_prebuilt()`（decode_schedule_batch_mixin.py）中，FFN 路径通过 `alloc_for_extend()` 分配了大量的 KV cache token（整个请求序列），但 `kv_committed_len` 和 `kv_allocated_len` 没有被设置，保持为初始值 0。每次 decode step 中 `prepare_for_decode` 只会将它们加 1，但初始的 `extend_num_tokens` 分配从未被计入。当 `_afd_ffn_cleanup_all()` 调用 `release_kv_cache()` 时，`cache_finished_req` 只释放 `kv_committed_len` 个 token（仅覆盖 decode step 分配的部分），其余 token 永久泄漏，导致 `self_check_during_idle()` 触发 `ValueError: token_to_kv_pool_allocator memory leak detected!`。

修复（decode_schedule_batch_mixin.py:54-55）：
- 在 `prepare_for_prebuilt()` 的 FFN 路径中，将 `req.kv_committed_len` 和 `req.kv_allocated_len` 设置为 `len(req.fill_ids)`（即 `origin_input_ids + output_ids` 的总长度），匹配实际通过 `alloc_for_extend()` 分配的 token 数。之前使用 `seq_len`（少 1 个 token，当 `len(output_ids) >= 1` 时）会导致最后 1 个 token 永久泄漏。
