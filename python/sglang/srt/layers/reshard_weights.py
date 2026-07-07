#!/usr/bin/env python3
"""In-place weight reshard for TP scaling.

Core idea: Instead of killing the old PA and reloading model from disk (20-28s),
directly slice existing GPU weights and scatter to new GPUs (~1s).

For TP1→TP2:
  - GPU0 has full weight W [out, in] 
  - After: GPU0 keeps W[:out//2, :] (column-parallel) or W[:, :in//2] (row-parallel)
  - GPU1 receives the other half via NCCL/P2P

For TP_old → TP_new (expanding):
  - Each existing rank holds a shard of size S
  - New TP divides each shard further
  - Each old rank sends (TP_new/TP_old - 1) sub-shards to new ranks
"""

import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist


def get_tp_split_rules(model) -> Dict[str, Tuple[str, int]]:
    """Extract TP split rules from model layers.
    
    Returns dict: param_name -> ("column" | "row", split_dim)
    Column-parallel: split along dim 0 (output_dim)
    Row-parallel: split along dim 1 (input_dim)
    
    Works with sglang/vLLM linear layer classes:
    - ColumnParallelLinear, MergedColumnParallelLinear, QKVParallelLinear → column, dim 0
    - RowParallelLinear → row, dim 1
    - VocabParallelEmbedding → column, dim 0
    """
    rules = {}
    
    # Class names that indicate column-parallel (split output dim)
    COLUMN_CLASSES = {
        "ColumnParallelLinear", "MergedColumnParallelLinear",
        "QKVParallelLinear", "MergedColumnParallelRepeatedLinear",
        "ColumnParallelBatchedLinear",
    }
    ROW_CLASSES = {"RowParallelLinear"}
    # ParallelLMHead subclasses VocabParallelEmbedding but reports its own class
    # name; it must be split (column, dim 0) on reshard too, otherwise rank0 keeps
    # a full-vocab lm_head while joining ranks hold a sharded one, and the
    # vocab-parallel logits all_gather deadlocks on mismatched shapes.
    EMBED_CLASSES = {"VocabParallelEmbedding", "ParallelLMHead"}
    
    for name, module in model.named_modules():
        cls_name = type(module).__name__

        # Fused column-parallel weights ([q|k|v] or [gate|up] concatenated on
        # dim 0) must be split PER SEGMENT: each segment is independently divided
        # across the new TP ranks and re-concatenated. A naive single dim-0 split
        # crosses segment boundaries and produces numerically wrong (garbage)
        # weights after reshard. We record the full-tensor segment sizes so the
        # transfer code can shard each segment correctly.
        fused_segments = _get_fused_segment_sizes(module, cls_name)

        if cls_name in COLUMN_CLASSES or cls_name in EMBED_CLASSES:
            for pname, _ in module.named_parameters(recurse=False):
                full_name = f"{name}.{pname}" if name else pname
                if "weight" in pname:
                    if fused_segments is not None:
                        rules[full_name] = ("column_fused", 0, tuple(fused_segments))
                    else:
                        rules[full_name] = ("column", 0)
        elif cls_name in ROW_CLASSES:
            for pname, _ in module.named_parameters(recurse=False):
                full_name = f"{name}.{pname}" if name else pname
                if "weight" in pname:
                    rules[full_name] = ("row", 1)

    return rules


def _get_fused_segment_sizes(module, cls_name):
    """Return the full (pre-shard) dim-0 segment sizes of a fused column-parallel
    weight, or None if the module is not fused.

    - QKVParallelLinear: [q_size, k_size, v_size] using the TOTAL head counts.
    - MergedColumnParallelLinear: its output_sizes, scaled back to full (the
      module may have been built at tp>1, storing per-partition sizes).
    """
    if cls_name == "QKVParallelLinear":
        head_size = getattr(module, "head_size", None)
        total_num_heads = getattr(module, "total_num_heads", None)
        total_num_kv_heads = getattr(module, "total_num_kv_heads", None)
        if None in (head_size, total_num_heads, total_num_kv_heads):
            return None
        q = total_num_heads * head_size
        k = total_num_kv_heads * head_size
        v = total_num_kv_heads * head_size
        return [q, k, v]
    if cls_name in ("MergedColumnParallelLinear", "MergedColumnParallelRepeatedLinear"):
        output_sizes = getattr(module, "output_sizes", None)
        if not output_sizes:
            return None
        # output_sizes are full segment sizes (they are the constructor args and
        # are not divided by tp_size in sglang's MergedColumnParallelLinear).
        return list(output_sizes)
    return None


def reshard_shard_for_rank(full, rule, local_idx: int, new_tp: int):
    """Given a full (pre-shard) weight tensor and its split rule, return the
    contiguous shard belonging to `local_idx` under `new_tp`.

    Handles plain column (dim 0), fused column ([q|k|v] / [gate|up], sharded
    per segment then concatenated), and row (dim 1). Used by the in-place
    reshard transfer so fused weights are not split across segment boundaries.
    """
    import logging as _lg
    import torch

    kind = rule[0]
    if kind in ("column", "row"):
        dim = rule[1]
        if full.shape[dim] % new_tp != 0:
            _lg.getLogger("sglang").error(
                "[RESHARD] non-divisible %s dim=%d shape=%s new_tp=%d",
                kind, dim, tuple(full.shape), new_tp,
            )
        seg = full.shape[dim] // new_tp
        return full.narrow(dim, local_idx * seg, seg).contiguous()
    if kind == "column_fused":
        _, dim, segments = rule[0], rule[1], rule[2]
        pieces = []
        offset = 0
        for seg_size in segments:
            per = seg_size // new_tp
            start = offset + local_idx * per
            pieces.append(full.narrow(dim, start, per))
            offset += seg_size
        return torch.cat(pieces, dim=dim).contiguous()
    raise ValueError(f"unknown reshard rule kind: {kind}")


def reshard_subshard(shard, rule, sub_idx: int, factor: int):
    """Split an EXISTING (already TP-sharded) weight into `factor` finer
    sub-shards and return sub-shard `sub_idx`.

    Generalizes `reshard_shard_for_rank` to any old_tp: with old_tp==1 the input
    `shard` is the full tensor (equivalent to the single-jump helper); with
    old_tp>1 the input is what an active rank currently holds and is divided by
    `factor = new_tp/old_tp`.

    Contiguous layout (matches the single-jump helper and standard TP ordering):
    global segment g is held by new rank g. New rank j corresponds to
    old rank `j // factor` holding its sub-shard `sub_idx = j % factor`. Thus an
    active rank i keeps sub_idx=0 and sends sub_idx=s (s=1..factor-1) to the new
    physical rank `i*factor + s`.

    Fused weights ([q|k|v] / [gate|up]) are sub-split PER SEGMENT within the
    current shard, so segment boundaries are never crossed.
    """
    import logging as _lg
    import torch

    kind = rule[0]
    if kind in ("column", "row"):
        dim = rule[1]
        if shard.shape[dim] % factor != 0:
            _lg.getLogger("sglang").error(
                "[RESHARD] non-divisible %s dim=%d shape=%s factor=%d",
                kind, dim, tuple(shard.shape), factor,
            )
        seg = shard.shape[dim] // factor
        return shard.narrow(dim, sub_idx * seg, seg).contiguous()
    if kind == "column_fused":
        _, dim, full_segments = rule[0], rule[1], rule[2]
        total_full = sum(full_segments)
        # Infer old_tp from how much of the full tensor this shard represents.
        old_tp = max(1, total_full // shard.shape[dim])
        pieces = []
        offset = 0
        for seg_full in full_segments:
            cur_seg = seg_full // old_tp  # this segment's size in the current shard
            per = cur_seg // factor       # sub-shard size for one new rank
            start = offset + sub_idx * per
            pieces.append(shard.narrow(dim, start, per))
            offset += cur_seg
        return torch.cat(pieces, dim=dim).contiguous()
    raise ValueError(f"unknown reshard rule kind: {kind}")


def compute_reshard_plan(
    old_tp: int, new_tp: int, old_rank: int
) -> List[Tuple[int, int, int]]:
    """Compute which slices to send/keep for a given rank.
    
    For expanding TP (new_tp > old_tp), each old rank splits its shard
    into (new_tp / old_tp) pieces.
    
    Returns: list of (new_rank, slice_start_ratio_num, slice_end_ratio_num)
             where ratio is relative to current shard size (denominator = new_tp/old_tp)
    """
    assert new_tp > old_tp and new_tp % old_tp == 0
    factor = new_tp // old_tp  # how many new ranks per old rank
    
    plan = []
    for i in range(factor):
        new_rank = old_rank * factor + i
        plan.append((new_rank, i, factor))
    return plan


@torch.no_grad()
def reshard_weights_inplace(
    model: torch.nn.Module,
    old_tp: int,
    new_tp: int,
    old_rank: int,
    new_rank: int,
    process_group=None,
) -> Dict[str, torch.Tensor]:
    """Reshard model weights in-place for TP expansion.
    
    For the current rank (old_rank in old TP scheme, new_rank in new TP scheme):
    - Slice each weight tensor to get the portion for new_rank
    - Return dict of {param_name: new_shard_tensor}
    
    This is the LOCAL operation - just compute which slice this rank keeps.
    The SEND operation is separate (done via NCCL scatter).
    """
    assert new_tp > old_tp
    factor = new_tp // old_tp
    
    # Which sub-shard within the old shard does new_rank correspond to?
    local_idx = new_rank % factor  # 0..factor-1
    
    rules = get_tp_split_rules(model)
    new_weights = {}
    
    for name, param in model.named_parameters():
        if name in rules:
            split_type, split_dim = rules[name]
            # Current shard size along split_dim
            shard_size = param.shape[split_dim]
            # New sub-shard size
            new_shard_size = shard_size // factor
            start = local_idx * new_shard_size
            # Slice the tensor
            new_weights[name] = param.data.narrow(split_dim, start, new_shard_size).contiguous()
        else:
            # Non-TP parameters (layernorm, etc.) are replicated - keep as-is
            new_weights[name] = param.data.clone()
    
    return new_weights


@torch.no_grad()
def scatter_weights_to_new_ranks(
    model: torch.nn.Module,
    old_tp: int,
    new_tp: int,
    old_rank: int,
    new_ranks: List[int],
    device_map: Dict[int, int],  # new_rank -> gpu_id
    use_nccl: bool = True,
):
    """Scatter weight shards from old rank to new ranks.
    
    Args:
        model: The model with current (old_tp) weights
        old_tp: Current TP degree
        new_tp: Target TP degree
        old_rank: This rank's position in old TP
        new_ranks: List of new rank IDs that this old rank is responsible for
        device_map: Mapping from new_rank to target GPU device
        use_nccl: If True, use NCCL; otherwise use direct CUDA memcpy (same node)
    """
    factor = new_tp // old_tp
    rules = get_tp_split_rules(model)
    
    t0 = time.time()
    total_bytes = 0
    
    for name, param in model.named_parameters():
        if name in rules:
            split_type, split_dim = rules[name]
            shard_size = param.shape[split_dim]
            new_shard_size = shard_size // factor
            
            for i, new_rank in enumerate(new_ranks):
                if new_rank == old_rank * factor:
                    # This is our own sub-shard, skip sending
                    continue
                start = i * new_shard_size
                shard = param.data.narrow(split_dim, start, new_shard_size).contiguous()
                total_bytes += shard.numel() * shard.element_size()
                
                if use_nccl:
                    dist.send(shard, dst=new_rank)
                else:
                    # Direct CUDA P2P copy (same node, faster)
                    target_device = device_map[new_rank]
                    # Allocate on target and copy
                    target_tensor = torch.empty_like(shard, device=f"cuda:{target_device}")
                    target_tensor.copy_(shard)
    
    elapsed = time.time() - t0
    bw = total_bytes / elapsed / 1e9 if elapsed > 0 else 0
    print(f"[reshard] Scattered {total_bytes/1e9:.2f} GB in {elapsed:.3f}s ({bw:.1f} GB/s)")


# ─── CUDA Kernel for fast tensor slicing ───
# For very large tensors, a custom CUDA kernel can do the slice + copy in one pass
# avoiding intermediate allocations.

RESHARD_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Fast weight reshard: slice a 2D tensor along dim and copy to output
// This fuses the narrow + contiguous + copy into one kernel
__global__ void reshard_slice_kernel(
    const char* __restrict__ input,
    char* __restrict__ output,
    int64_t num_rows,
    int64_t input_stride_bytes,   // bytes per row in input
    int64_t output_stride_bytes,  // bytes per row in output
    int64_t offset_bytes,         // byte offset within each row (for dim=1 split)
    int64_t copy_bytes            // bytes to copy per row
) {
    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        const char* src = input + row * input_stride_bytes + offset_bytes;
        char* dst = output + row * output_stride_bytes;
        memcpy(dst, src, copy_bytes);
    }
}

// Slice a 2D weight tensor along specified dimension
// dim=0: slice rows (output[start:start+size, :])
// dim=1: slice columns (output[:, start:start+size])
torch::Tensor reshard_slice_2d(
    torch::Tensor input,
    int dim,
    int64_t start,
    int64_t size,
    c10::optional<torch::Tensor> output_opt
) {
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(dim == 0 || dim == 1, "dim must be 0 or 1");
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    
    int64_t rows = input.size(0);
    int64_t cols = input.size(1);
    int64_t elem_size = input.element_size();
    
    torch::Tensor output;
    if (dim == 0) {
        // Simple case: just narrow along rows (already contiguous sub-block)
        return input.narrow(0, start, size).contiguous();
    } else {
        // dim=1: need to gather columns from each row
        int64_t out_rows = rows;
        int64_t out_cols = size;
        
        if (output_opt.has_value()) {
            output = output_opt.value();
        } else {
            output = torch::empty({out_rows, out_cols}, input.options());
        }
        
        int64_t input_stride = cols * elem_size;
        int64_t output_stride = out_cols * elem_size;
        int64_t offset = start * elem_size;
        int64_t copy_size = size * elem_size;
        
        int threads = 256;
        int blocks = (out_rows + threads - 1) / threads;
        
        reshard_slice_kernel<<<blocks, threads>>>(
            (const char*)input.data_ptr(),
            (char*)output.data_ptr(),
            out_rows,
            input_stride,
            output_stride,
            offset,
            copy_size
        );
        
        return output;
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reshard_slice_2d", &reshard_slice_2d, "Fast 2D tensor slice for TP reshard");
}
"""


def build_reshard_cuda_extension():
    """JIT compile the CUDA reshard kernel."""
    from torch.utils.cpp_extension import load_inline
    
    module = load_inline(
        name="reshard_cuda",
        cpp_sources="",
        cuda_sources=[RESHARD_CUDA_SRC],
        functions=["reshard_slice_2d"],
        verbose=False,
    )
    return module


# ─── High-level reshard API ───

def live_reshard_tp_expand(
    model: torch.nn.Module,
    old_tp: int,
    new_tp: int,
    rank: int,
    world_size: int,
    process_group=None,
) -> float:
    """Perform live TP expansion reshard.
    
    This is called on ALL ranks (old and new) simultaneously.
    Old ranks: scatter their weight shards to new ranks
    New ranks: receive weight shards from old ranks
    
    Returns: elapsed time in seconds
    """
    factor = new_tp // old_tp
    t0 = time.time()
    
    rules = get_tp_split_rules(model)
    
    # Determine this rank's role
    old_rank = rank // factor if rank < new_tp else -1
    local_idx = rank % factor
    source_old_rank = rank // factor
    
    total_bytes = 0
    
    for name, param in model.named_parameters():
        if name in rules:
            split_type, split_dim = rules[name]
            
            if rank < old_tp:
                # We are an old rank with full (old) shard
                shard_size = param.shape[split_dim]
                new_shard_size = shard_size // factor
                
                # Keep our portion
                keep_start = local_idx * new_shard_size
                new_data = param.data.narrow(split_dim, keep_start, new_shard_size).contiguous()
                
                # Send other portions to new ranks
                for i in range(factor):
                    target_rank = rank * factor + i
                    if target_rank == rank:
                        continue
                    send_start = i * new_shard_size
                    send_data = param.data.narrow(split_dim, send_start, new_shard_size).contiguous()
                    if process_group:
                        dist.send(send_data, dst=target_rank, group=process_group)
                    total_bytes += send_data.numel() * send_data.element_size()
                
                # Update our weight in-place
                param.data = new_data
            else:
                # We are a new rank, receive from our source old rank
                shard_size = param.shape[split_dim] // factor  # expected size
                recv_buf = torch.empty(
                    *[shard_size if d == split_dim else s for d, s in enumerate(param.shape)],
                    dtype=param.dtype, device=param.device
                )
                if process_group:
                    dist.recv(recv_buf, src=source_old_rank, group=process_group)
                param.data = recv_buf
                total_bytes += recv_buf.numel() * recv_buf.element_size()
        else:
            # Non-TP params: replicated, no change needed
            pass
    
    elapsed = time.time() - t0
    bw = total_bytes / elapsed / 1e9 if elapsed > 0 else 0
    return elapsed
