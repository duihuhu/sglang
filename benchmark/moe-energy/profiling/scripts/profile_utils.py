#!/usr/bin/env python3
"""Self-contained Qwen3-MoE component latency/energy profiler helpers."""
from __future__ import annotations
import argparse, hashlib, json, multiprocessing, os, sys, time, traceback
from array import array
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
import numpy as np
import torch
SCRIPT_DIR = Path(__file__).resolve().parent
PROFILE_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "python"))
SCHEMA_VERSION = "qwen3-af-v2"
DEFAULT_MODEL = "/models/Qwen3-30B-A3B"
DEFAULT_FREQS = [210, 450, 690, 930, 1170, 1410]
PREFILL_LENGTHS = [64, 128, 256, 512, 1024, 2048, 4096]
PREFILL_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
DECODE_LENGTHS = [64, 128, 256, 512, 1024, 2048, 4096]
DECODE_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
# Expanded phase-specific kernel-EP matrix (2026-08-20).
KERNEL_EP_FREQS = [210, 450, 690, 930, 1170, 1410]
KERNEL_EP_PREFILL_LENGTHS = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
KERNEL_EP_PREFILL_BATCHES = [1]
KERNEL_EP_DECODE_LENGTHS = [64]
KERNEL_EP_DECODE_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
# CLI defaults remain a union; the distributed scheduler passes phase-specific axes.
KERNEL_EP_LENGTHS = KERNEL_EP_PREFILL_LENGTHS
KERNEL_EP_BATCHES = KERNEL_EP_DECODE_BATCHES
FORCED_ROUTING_CHOICES = (
    "natural", "balanced", "skewed_rank0", "middle_rank0",
    "hot_rank_0250", "hot_rank_03125", "hot_rank_0375", "hot_rank_04375",
    "hot_rank_0500", "hot_rank_05625", "hot_rank_0625", "hot_rank_06875",
    "hot_rank_0750", "hot_rank_08125", "hot_rank_0875", "hot_rank_09375",
    "hot_rank_1000", "hot_rank_exact_cold", "active_ranks_equal",
)


def kernel_shape_excluded(phase: str, world_size: int, length: int, batch: int) -> bool:
    """Return True for a documented, reproducible OOM boundary."""
    return phase == "prefill" and world_size == 2 and length >= 8192


def kernel_phase_axes(phase: str) -> tuple[list[int], list[int]]:
    if phase == "prefill":
        return KERNEL_EP_PREFILL_LENGTHS, KERNEL_EP_PREFILL_BATCHES
    if phase == "decode":
        return KERNEL_EP_DECODE_LENGTHS, KERNEL_EP_DECODE_BATCHES
    raise ValueError(f"unsupported kernel phase: {phase}")


def kernel_expected_shape_count(phase: str, world_size: int) -> int:
    lengths, batches = kernel_phase_axes(phase)
    return sum(
        not kernel_shape_excluded(phase, world_size, length, batch)
        for length in lengths
        for batch in batches
    )


def kernel_expected_rows(phase: str, world_size: int) -> int:
    return kernel_expected_shape_count(phase, world_size) * len(KERNEL_EP_FREQS)

def _visible_tokens():
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    return None if not raw else [x.strip() for x in raw.split(",") if x.strip()]

class NvmlController:
    """Lock one visible GPU's SM clock and read its NVML energy counter."""
    def __init__(self, logical_cuda_id: int):
        import pynvml
        self.nvml = pynvml; pynvml.nvmlInit()
        tokens = _visible_tokens(); token = str(logical_cuda_id) if tokens is None else tokens[logical_cuda_id]
        if token.isdigit():
            self.physical_id = int(token); self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.physical_id)
        elif token.startswith("GPU-"):
            self.physical_id = token; self.handle = pynvml.nvmlDeviceGetHandleByUUID(token.encode())
        else:
            raise RuntimeError(f"Unsupported CUDA_VISIBLE_DEVICES entry {token!r}; use physical indices or GPU UUIDs")
        memory_clocks = pynvml.nvmlDeviceGetSupportedMemoryClocks(self.handle)
        clocks = set()
        for memory_clock in memory_clocks:
            clocks.update(int(x) for x in pynvml.nvmlDeviceGetSupportedGraphicsClocks(self.handle, memory_clock))
        self.supported_clocks = sorted(clocks)
    def snap(self, requested: int) -> int:
        if not self.supported_clocks: raise RuntimeError("NVML returned no supported graphics clocks")
        return min(self.supported_clocks, key=lambda x: (abs(x - requested), x))
    def lock(self, mhz: int): self.nvml.nvmlDeviceSetGpuLockedClocks(self.handle, mhz, mhz)
    def unlock(self):
        try: self.nvml.nvmlDeviceResetGpuLockedClocks(self.handle)
        except Exception: pass
    def energy_mj(self): return int(self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle))
    def close(self):
        self.unlock()
        try: self.nvml.nvmlShutdown()
        except Exception: pass

class TreeCacheNamespace(SimpleNamespace):
    def supports_swa(self): return False
    def supports_mamba(self): return False
    def is_chunk_cache(self): return False
    def is_tree_cache(self): return True
    def evict(self, params): return None

def stable_seed(seed: int, *parts: Any) -> int:
    return int.from_bytes(hashlib.sha256(json.dumps([seed, *parts], sort_keys=True, default=str).encode()).digest()[:4], "little")

def row_key(row):
    return tuple(row.get(k) for k in ("phase", "component", "parallel_mode", "world_size", "length", "batch", "freq_mhz"))

def read_completed(path: Path):
    completed = set()
    if not path.exists(): return completed
    with path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
                if row.get("status") == "ok": completed.add(row_key(row))
            except (json.JSONDecodeError, TypeError): pass
    return completed

def append_row(f, row):
    f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"); f.flush(); os.fsync(f.fileno())

def dist_barrier(world):
    if world > 1: torch.distributed.barrier()

def gather_object(value, world):
    if world == 1: return [value]
    output = [None] * world if torch.distributed.get_rank() == 0 else None
    torch.distributed.gather_object(value, output, dst=0); return output

def profile_world(target: Callable[[], Any], ctrl, warmup, repeat, world, phase, min_measure_seconds=None):
    for _ in range(warmup):
        target()
    torch.cuda.synchronize()
    dist_barrier(world)
    probe_start = time.perf_counter()
    target()
    torch.cuda.synchronize()
    probe_seconds = time.perf_counter() - probe_start
    min_seconds = (3.0 if phase == "prefill" else 0.5) if min_measure_seconds is None else min_measure_seconds
    actual_repeat = max(repeat, int(min_seconds / max(probe_seconds, 1e-6)) + 1)
    repeat_tensor = torch.tensor(actual_repeat, device="cuda", dtype=torch.int64)
    if world > 1:
        torch.distributed.all_reduce(repeat_tensor, op=torch.distributed.ReduceOp.MAX)
    actual_repeat = int(repeat_tensor.item())
    dist_barrier(world)
    before = ctrl.energy_mj()
    start = time.perf_counter_ns()
    for _ in range(actual_repeat):
        target()
    torch.cuda.synchronize()
    elapsed_us = (time.perf_counter_ns() - start) / 1000 / actual_repeat
    after = ctrl.energy_mj()
    samples = gather_object(
        {
            "rank": torch.distributed.get_rank() if world > 1 else 0,
            "latency_us": elapsed_us,
            "energy_mj": (after - before) / actual_repeat,
        },
        world,
    )
    if samples is None:
        return None, None, None, None, actual_repeat
    samples.sort(key=lambda x: x["rank"])
    rank_latency = [round(float(x["latency_us"]), 6) for x in samples]
    rank_energy = [round(float(x["energy_mj"]), 6) for x in samples]
    return (
        max(rank_latency),
        rank_latency,
        rank_energy,
        sum(rank_energy),
        actual_repeat,
    )


def profile_moe_kernel_core(ctrl, experts, dispatch_output, warmup, repeat, world, phase):
    """Profile ``run_moe_core`` with block-level NVML energy sampling.

    NVML is read only once before and once after the whole repeat block. A
    barrier after the before-snapshots aligns block start; a barrier after all
    local CUDA work includes real straggler waiting in cluster energy. NVML and
    barrier boundary costs are therefore paid once and amortized by repeat.
    Latency excludes both barriers and is max-rank local block wall / repeat.
    """
    for _ in range(warmup):
        experts.run_moe_core(dispatch_output)
    torch.cuda.synchronize()
    dist_barrier(world)

    # Size the measurement block from an amortized local probe. A single call
    # is too sensitive to CPU scheduling and can under-size short decode blocks.
    probe_repeat = max(10, min(50, repeat))
    dist_barrier(world)
    probe_start = time.perf_counter()
    for _ in range(probe_repeat):
        experts.run_moe_core(dispatch_output)
    torch.cuda.synchronize()
    probe_seconds_per_call = (
        time.perf_counter() - probe_start
    ) / probe_repeat
    probe_tensor = torch.tensor(
        probe_seconds_per_call, device="cuda", dtype=torch.float64
    )
    if world > 1:
        torch.distributed.all_reduce(probe_tensor, op=torch.distributed.ReduceOp.MAX)
    probe_seconds_per_call = float(probe_tensor.item())
    min_seconds = 3.0
    actual_repeat = max(
        repeat, int(min_seconds / max(probe_seconds_per_call, 1e-6)) + 1
    )
    repeat_tensor = torch.tensor(actual_repeat, device="cuda", dtype=torch.int64)
    if world > 1:
        torch.distributed.all_reduce(repeat_tensor, op=torch.distributed.ReduceOp.MAX)
    actual_repeat = int(repeat_tensor.item())
    dist_barrier(world)

    # Snapshot first, then align all ranks. Snapshot serialization is outside
    # the work block and its one-time energy cost is amortized over the block.
    energy_before_mj = ctrl.energy_mj()
    dist_barrier(world)

    block_wall_start_ns = time.perf_counter_ns()
    cuda_start = torch.cuda.Event(enable_timing=True)
    cuda_end = torch.cuda.Event(enable_timing=True)
    cuda_start.record()
    for _ in range(actual_repeat):
        experts.run_moe_core(dispatch_output)
    cuda_end.record()
    torch.cuda.synchronize()
    local_block_wall_us = (time.perf_counter_ns() - block_wall_start_ns) / 1000.0

    # Fast ranks wait here for the true compute straggler. This wait remains in
    # the energy window, but not in latency_us.
    dist_barrier(world)
    sync_block_wall_us = (time.perf_counter_ns() - block_wall_start_ns) / 1000.0
    energy_after_mj = ctrl.energy_mj()

    latency_cuda_us = cuda_start.elapsed_time(cuda_end) * 1000.0 / actual_repeat
    latency_local_wall_us = local_block_wall_us / actual_repeat
    latency_sync_wall_us = sync_block_wall_us / actual_repeat
    energy_mj = (energy_after_mj - energy_before_mj) / actual_repeat
    samples = gather_object(
        {
            "rank": torch.distributed.get_rank() if world > 1 else 0,
            "latency_local_wall_us": latency_local_wall_us,
            "latency_sync_wall_us": latency_sync_wall_us,
            "latency_cuda_us": latency_cuda_us,
            "energy_mj": energy_mj,
        },
        world,
    )
    if samples is None:
        return None, None, None, None, actual_repeat, None, None
    samples.sort(key=lambda x: x["rank"])
    rank_local_wall_latency = [
        round(float(x["latency_local_wall_us"]), 6) for x in samples
    ]
    rank_sync_wall_latency = [
        round(float(x["latency_sync_wall_us"]), 6) for x in samples
    ]
    rank_cuda_latency = [round(float(x["latency_cuda_us"]), 6) for x in samples]
    rank_energy = [round(float(x["energy_mj"]), 6) for x in samples]
    return (
        max(rank_local_wall_latency),
        rank_local_wall_latency,
        rank_energy,
        sum(rank_energy),
        actual_repeat,
        rank_cuda_latency,
        rank_sync_wall_latency,
    )

def sync_any(flag, world):
    t = torch.tensor(int(flag), device="cuda", dtype=torch.int32)
    if world > 1: torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
    return bool(t.item())

def gather_dp_token_counts(local_counts, world, dp_size):
    """Gather per-DP-rank token counts, selecting one rank per attention TP group."""
    local_counts_gpu = torch.tensor(local_counts, device="cuda", dtype=torch.int64)
    world_counts_gpu = torch.empty(
        (world, len(local_counts)), device="cuda", dtype=torch.int64
    )
    if world > 1:
        torch.distributed.all_gather_into_tensor(
            world_counts_gpu.flatten(), local_counts_gpu
        )
    else:
        world_counts_gpu[0].copy_(local_counts_gpu)
    attn_tp_size = world // dp_size
    dp_counts_cpu = world_counts_gpu[::attn_tp_size].cpu()
    return tuple(dp_counts_cpu[:, i].tolist() for i in range(len(local_counts)))


def set_shape_dp_metadata(
    n_tokens, world, dp_size, is_extend_in_batch, global_num_tokens=None
):
    from sglang.srt.layers.dp_attention import DpPaddingMode, set_dp_buffer_len

    if global_num_tokens is None:
        (global_num_tokens,) = gather_dp_token_counts((n_tokens,), world, dp_size)
    else:
        global_num_tokens = list(global_num_tokens)
    attn_tp_size = world // dp_size
    dp_padding_mode = DpPaddingMode.get_dp_padding_mode(
        is_extend_in_batch, global_num_tokens
    )
    if dp_padding_mode.is_max_len():
        local_dp_buffer_len = max(global_num_tokens)
        global_num_tokens = [local_dp_buffer_len] * dp_size
    else:
        dp_rank = (torch.distributed.get_rank() if world > 1 else 0) // attn_tp_size
        local_dp_buffer_len = global_num_tokens[dp_rank]
    global_dp_buffer_len = sum(global_num_tokens)
    global_num_tokens_gpu = torch.tensor(
        global_num_tokens, device="cuda", dtype=torch.int64
    )
    set_dp_buffer_len(
        global_dp_buffer_len,
        local_dp_buffer_len,
        dp_padding_mode.is_max_len(),
        global_num_tokens,
        global_num_tokens_gpu,
    )

def set_batch_dp_metadata(batch, runner, is_extend_in_batch):
    if is_extend_in_batch:
        num_tokens = batch.extend_num_tokens
        num_tokens_for_logprob = sum(
            max(extend_len - logprob_start_len, 1)
            for logprob_start_len, extend_len in zip(
                batch.extend_logprob_start_lens, batch.extend_lens
            )
        )
    else:
        num_tokens = num_tokens_for_logprob = batch.batch_size()

    (
        batch.global_num_tokens,
        batch.global_num_tokens_for_logprob,
    ) = gather_dp_token_counts(
        (num_tokens, num_tokens_for_logprob),
        runner.server_args.tp_size,
        runner.server_args.dp_size,
    )
    set_shape_dp_metadata(
        num_tokens,
        runner.server_args.tp_size,
        runner.server_args.dp_size,
        is_extend_in_batch,
        batch.global_num_tokens,
    )

def _forced_balanced_style_topk_ids(
    token_ids: torch.Tensor,
    slots: torch.Tensor,
    top_k: int,
    world: int,
    num_experts: int,
) -> torch.Tensor:
    """Shared tensor path for forced routing; matches balanced arithmetic cost."""
    logits_idx = token_ids * top_k + slots
    rank_ids = logits_idx % world
    experts_per_rank = num_experts // world
    local_ids = (logits_idx // world) % experts_per_rank
    return rank_ids * experts_per_rank + local_ids


def forced_routing(moe, mode, world):
    if mode == "natural":
        return None
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    original_forward = moe.topk.forward
    top_k = moe.topk.topk_config.top_k
    skewed_expert0_ids_cache: dict[tuple, torch.Tensor] = {}

    def forward(hidden_states, router_logits, **kwargs):
        num_tokens, num_experts = router_logits.shape
        token_ids = torch.arange(num_tokens, device=router_logits.device).unsqueeze(1)
        slots = torch.arange(top_k, device=router_logits.device).unsqueeze(0)
        combined = _forced_balanced_style_topk_ids(
            token_ids, slots, top_k, world, num_experts
        )
        if mode == "balanced":
            topk_ids = combined.to(torch.int64)
        elif mode == "active_ranks_equal":
            if world not in (4, 8):
                raise ValueError("active-rank routing currently requires EP4 or EP8")
            del combined
            active = int(getattr(moe, "_forced_active_rank_count"))
            if not (1 <= active <= world):
                raise ValueError(f"unsupported active rank count: {active} for EP{world}")
            logits_idx = token_ids * top_k + slots
            rank_ids = logits_idx % active
            experts_per_rank = num_experts // world
            local_ids = (logits_idx // active) % experts_per_rank
            topk_ids = rank_ids * experts_per_rank + local_ids
        elif mode == "hot_rank_exact_cold":
            if world != 4:
                raise ValueError("exact cold-rank routing currently requires EP4")
            del combined
            cold = int(getattr(moe, "_forced_cold_assignments_per_rank"))
            logits_idx = token_ids * top_k + slots
            total = num_tokens * top_k
            hot = total - 3 * cold
            if hot < 0:
                raise ValueError(f"cold assignments {cold} exceed total {total}")
            rank_ids = torch.zeros_like(logits_idx)
            if cold > 0:
                rank_ids = torch.where(logits_idx >= hot, 1 + (logits_idx - hot) // cold, rank_ids)
            experts_per_rank = num_experts // world
            within_rank_idx = torch.where(logits_idx < hot, logits_idx, (logits_idx - hot) % max(cold, 1))
            local_ids = within_rank_idx % experts_per_rank
            topk_ids = rank_ids * experts_per_rank + local_ids
        elif mode.startswith("hot_rank_"):
            if world != 4:
                raise ValueError("hot_rank_fraction routing currently requires EP4")
            del combined
            # Exact cyclic rank shares. Within each rank, assignments remain
            # evenly striped over all local experts, isolating rank-level skew.
            rank_patterns = {
                "hot_rank_0250": (0, 1, 2, 3),
                # New intermediate points use a 48-assignment cycle.  The
                # remainder is exactly equal across ranks 1/2/3.
                "hot_rank_03125": (0,) * 15 + (1,) * 11 + (2,) * 11 + (3,) * 11,
                "hot_rank_0375": (0,) * 18 + (1,) * 10 + (2,) * 10 + (3,) * 10,
                "hot_rank_04375": (0,) * 21 + (1,) * 9 + (2,) * 9 + (3,) * 9,
                "hot_rank_0500": (0, 0, 0, 1, 2, 3),
                "hot_rank_05625": (0,) * 27 + (1,) * 7 + (2,) * 7 + (3,) * 7,
                "hot_rank_0625": (0,) * 30 + (1,) * 6 + (2,) * 6 + (3,) * 6,
                "hot_rank_06875": (0,) * 33 + (1,) * 5 + (2,) * 5 + (3,) * 5,
                "hot_rank_0750": (0,) * 9 + (1, 2, 3),
                "hot_rank_08125": (0,) * 39 + (1,) * 3 + (2,) * 3 + (3,) * 3,
                "hot_rank_0875": (0,) * 21 + (1, 2, 3),
                "hot_rank_09375": (0,) * 45 + (1, 2, 3),
                "hot_rank_1000": (0,),
            }
            pattern = torch.tensor(
                rank_patterns[mode], device=router_logits.device, dtype=torch.int64
            )
            logits_idx = token_ids * top_k + slots
            rank_ids = pattern[logits_idx % pattern.numel()]
            experts_per_rank = num_experts // world
            active_experts = getattr(moe, "_forced_active_experts_per_rank", experts_per_rank)
            if active_experts not in (1, 2, 8, experts_per_rank):
                raise ValueError(f"unsupported active experts per rank: {active_experts}")
            local_ids = (logits_idx // pattern.numel()) % active_experts
            topk_ids = rank_ids * experts_per_rank + local_ids
        elif mode == "middle_rank0":
            # All slots -> rank0's local experts 0..(E/world-1), evenly (~8/slot per expert at b32).
            del combined
            experts_per_rank = num_experts // world
            logits_idx = token_ids * top_k + slots
            topk_ids = (logits_idx % experts_per_rank).to(torch.int64)
        else:
            # skewed_rank0: all slots -> global expert 0 (cached zeros for topk timing parity).
            del combined
            cache_key = (router_logits.device, num_tokens, top_k)
            topk_ids = skewed_expert0_ids_cache.get(cache_key)
            if topk_ids is None:
                topk_ids = torch.zeros(
                    num_tokens,
                    top_k,
                    device=router_logits.device,
                    dtype=torch.int64,
                )
                skewed_expert0_ids_cache[cache_key] = topk_ids
        topk_weights = torch.full(
            (num_tokens, top_k),
            1.0 / top_k,
            device=router_logits.device,
            dtype=router_logits.dtype,
        )
        return StandardTopKOutput(topk_weights, topk_ids, router_logits)

    moe.topk.forward = forward
    return original_forward


def routing_summary(moe, hidden_states, world):
    with torch.no_grad():
        logits, _ = moe.gate(hidden_states); output = moe.topk(hidden_states, logits)
        if hasattr(output, "to_standard"): output = output.to_standard(getattr(moe, "layer_id", None))
        valid = output.topk_ids[output.topk_ids >= 0].to(torch.int64); n_experts = int(logits.shape[-1])
        counts = torch.bincount(valid, minlength=n_experts).cpu().tolist()
    gathered = gather_object(counts, world)
    if gathered is None: return None
    logical = gathered[0]
    mean = sum(logical) / max(1, len(logical))
    identical = all(rank == logical for rank in gathered[1:])
    return {"counts_per_rank": gathered, "logical_expert_counts": logical, "routing_identical_across_ranks": identical, "total_assignments": sum(logical), "active_experts": sum(x > 0 for x in logical), "max_to_mean": max(logical) / mean if mean else 0.0, "aggregation": "rank0 logical routing; per-rank copies retained for validation"}

def make_reqs(batch_size, input_len, rng):
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams
    ids = rng.integers(0, 10000, (batch_size, input_len), dtype=np.int32); sp = SamplingParams(temperature=0, max_new_tokens=1); reqs=[]
    for i in range(batch_size):
        req = Req(
            rid=str(i),
            origin_input_text="",
            origin_input_ids=array("q", ids[i].tolist()),
            sampling_params=sp,
        )
        req._refresh_fill_ids()
        req.set_extend_range(0, len(req.origin_input_ids))
        req.logprob_start_len = -1
        reqs.append(req)
    return reqs

def new_schedule_batch(reqs, runner):
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    cache=TreeCacheNamespace(page_size=runner.server_args.page_size, device=runner.device, token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator)
    return ScheduleBatch.init_new(reqs=reqs, req_to_token_pool=runner.req_to_token_pool, token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator, tree_cache=cache, model_config=runner.model_config, enable_overlap=False, spec_algorithm=SpeculativeAlgorithm.NONE)

def build_forward_batch(reqs, runner, phase):
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    batch=new_schedule_batch(reqs,runner); batch.prepare_for_extend(); batch.input_ids = batch.prefill_input_ids_cpu.to(batch.device, non_blocking=True); batch.prefill_input_ids_cpu = None
    if runner.server_args.enable_dp_attention:
        set_batch_dp_metadata(batch, runner, True)
    prefill_fb=ForwardBatch.init_new(batch, runner, return_hidden_states_before_norm=False)
    runner.attn_backend.init_forward_metadata(prefill_fb)
    if phase == "prefill": return prefill_fb
    from sglang.srt.layers.dp_attention import set_is_extend_in_batch
    from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
    set_is_extend_in_batch(True)
    try:
        with forward_context(ForwardContext(attn_backend=runner.attn_backend)):
            logits = runner.model.forward(
                prefill_fb.input_ids, prefill_fb.positions, prefill_fb
            )
    finally:
        set_is_extend_in_batch(False)
    batch.input_ids = runner.sample(logits, prefill_fb)
    batch.prepare_for_decode()
    if runner.server_args.enable_dp_attention:
        set_batch_dp_metadata(batch, runner, False)
    fb=ForwardBatch.init_new(batch, runner, return_hidden_states_before_norm=False); runner.attn_backend.init_forward_metadata(fb); return fb

def load_runner(server_args, port_args, gpu_id, rank):
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.utils import suppress_other_loggers
    suppress_other_loggers(); model_config=ModelConfig.from_server_args(server_args)
    at_rank,at_size,ad_rank,ad_size=compute_dp_attention_world_info(server_args.enable_dp_attention,rank,server_args.tp_size,server_args.dp_size,server_args.attn_cp_size)
    ps=ParallelState(tp_rank=rank,tp_size=server_args.tp_size,pp_rank=0,pp_size=1,dp_rank=None,dp_size=server_args.dp_size,attn_tp_rank=at_rank,attn_tp_size=at_size,attn_cp_rank=0,attn_cp_size=server_args.attn_cp_size,attn_dcp_rank=rank%server_args.dcp_size,attn_dcp_size=server_args.dcp_size,attn_dp_rank=ad_rank,attn_dp_size=ad_size,moe_ep_rank=rank//(server_args.tp_size//server_args.ep_size),moe_ep_size=server_args.ep_size,moe_dp_rank=None,moe_dp_size=server_args.moe_dp_size,gpu_id=gpu_id)
    runner=ModelRunner(model_config=model_config,mem_fraction_static=server_args.mem_fraction_static,gpu_id=gpu_id,ps=ps,nccl_port=port_args.nccl_port,server_args=server_args); runner.alloc_memory_pool(); runner.init_attention_backends(); runner.init_cuda_graphs(); dist_barrier(server_args.tp_size); return runner

def topology(args, server_args):
    world=server_args.tp_size
    return {"parallel_mode":args.parallel_mode,"world_size":world,"attn_tp":world,"moe_tp":world//server_args.ep_size,"moe_ep":server_args.ep_size}

def worker(server_args, port_args, args, gpu_id, rank):
    from sglang.srt.layers.dp_attention import set_is_extend_in_batch
    from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
    from sglang.srt.layers.moe import initialize_moe_config
    initialize_moe_config(server_args); runner=load_runner(server_args,port_args,gpu_id,rank); world=server_args.tp_size; set_is_extend_in_batch(args.phase=="prefill"); ctrl=NvmlController(gpu_id); layer=runner.model.model.layers[args.layer_id]; layout=topology(args,server_args); output=Path(args.output); completed=read_completed(output); out_f=None
    if rank==0: output.parent.mkdir(parents=True,exist_ok=True); out_f=output.open("a",buffering=1); print(f"{SCHEMA_VERSION}: resumed {len(completed)} rows from {output}")
    try:
        freqs=sorted({ctrl.snap(f) for f in args.freqs})
        for length in args.lengths:
            for batch_size in args.batch_sizes:
                base={"phase":args.phase,"component":args.component,**layout,"length":length,"batch":batch_size}; pending=[f for f in freqs if row_key({**base,"freq_mhz":f}) not in completed]
                if not pending: continue
                shape_tokens = length * batch_size
                default_safe_shape_tokens = (
                    524288
                    if args.phase == "prefill" and args.component == "A"
                    else 131072
                )
                configured_shape_limit = (
                    default_safe_shape_tokens
                    if args.shape_token_limit is None
                    else args.shape_token_limit
                )
                shape_limit = (
                    runner.max_total_num_tokens
                    if configured_shape_limit == 0
                    else min(runner.max_total_num_tokens, configured_shape_limit)
                )
                if shape_tokens > shape_limit:
                    if rank == 0:
                        print(
                            f"[skip capacity] length={length} batch={batch_size} "
                            f"tokens={shape_tokens} limit={shape_limit}"
                        )
                    continue
                runner.req_to_token_pool.clear(); runner.token_to_kv_pool_allocator.clear(); failed=False; fb=hidden=residual=positions=None; request_seed=stable_seed(args.seed,args.phase,length,batch_size,args.layer_id,"req"); hidden_seed=stable_seed(args.seed,args.phase,length,batch_size,args.layer_id,"hidden")
                try:
                    reqs=make_reqs(batch_size,length if args.phase=="prefill" else length-1,np.random.default_rng(request_seed)); fb=build_forward_batch(reqs,runner,args.phase); n_tokens=int(fb.seq_lens_sum) if args.phase=="prefill" else batch_size
                    gen=torch.Generator(device=runner.device).manual_seed(hidden_seed); hidden=torch.randn(n_tokens,runner.model_config.hidden_size,device=runner.device,dtype=torch.bfloat16,generator=gen); residual=hidden.clone(); positions=fb.positions
                except (torch.cuda.OutOfMemoryError,RuntimeError) as exc:
                    failed=True
                    if rank==0:
                        print(f"[skip setup] length={length} batch={batch_size}: {exc}")
                        traceback.print_exc()
                    torch.cuda.empty_cache()
                if sync_any(failed,world):
                    fb=hidden=residual=positions=reqs=None
                    runner.req_to_token_pool.clear()
                    runner.token_to_kv_pool_allocator.clear()
                    torch.cuda.empty_cache()
                    if args.stop_on_shape_failure:
                        if rank == 0:
                            print(
                                f"[stop length] setup failed at length={length} "
                                f"batch={batch_size}"
                            )
                        break
                    continue
                with torch.no_grad(), forward_context(ForwardContext(attn_backend=runner.attn_backend)):
                    routing=None
                    if args.component=="F":
                        if args.forced_active_experts_per_rank is not None:
                            layer.mlp._forced_active_experts_per_rank = args.forced_active_experts_per_rank
                        original_topk_forward = forced_routing(layer.mlp, args.forced_routing, world)
                        normalized,_=layer.post_attention_layernorm(hidden,residual); routing=routing_summary(layer.mlp,normalized,world)
                        if routing is not None:
                            routing["forced_mode"] = args.forced_routing
                        def target():
                            hs,_=layer.post_attention_layernorm(hidden,residual); return layer.mlp(hs,fb)
                    else:
                        def target():
                            hs,_=layer.input_layernorm(hidden,residual); return layer.self_attn(positions=positions,hidden_states=hs,forward_batch=fb)
                    for freq in pending:
                        ctrl.lock(freq); time.sleep(args.clock_settle_seconds); latency,rank_latency,rank_energy,total_energy,actual_repeat=profile_world(target,ctrl,args.warmup,args.repeat,world,args.phase,args.min_measure_seconds); ctrl.unlock()
                        if rank==0:
                            row={"schema_version":SCHEMA_VERSION,"status":"ok",**base,"freq_mhz":freq,"latency_us":latency,"latency_per_rank_us":rank_latency,"energy_per_rank_mj":rank_energy,"energy_total_mj":total_energy,"warmup":args.warmup,"requested_repeat":args.repeat,"actual_repeat":actual_repeat,"layer_id":args.layer_id,"model":server_args.model_path,"request_seed":request_seed,"hidden_seed":hidden_seed,"routing":routing}; append_row(out_f,row); completed.add(row_key(row)); print(f"{args.component} {args.phase} l={length} b={batch_size} f={freq}: {latency:.1f} us, {total_energy:.3f} mJ")
                del fb,hidden,residual,positions; torch.cuda.empty_cache()
    except KeyboardInterrupt:
        if rank==0: print("Interrupted; completed JSONL rows are durable")
    finally:
        if out_f: out_f.close()
        ctrl.close()
        if world>1:
            from sglang.srt.distributed.parallel_state import destroy_distributed_environment
            destroy_distributed_environment()

def add_profile_args(parser,phase):
    from sglang.srt.server_args import ServerArgs
    ServerArgs.add_cli_args(parser); parser.set_defaults(model_path=DEFAULT_MODEL,moe_runner_backend="triton",moe_a2a_backend="none",cuda_graph_backend_decode="disabled",cuda_graph_backend_prefill="disabled")
    parser.add_argument("--forced-routing", choices=list(FORCED_ROUTING_CHOICES), default="natural"); parser.add_argument("--component",choices=["A","F"],required=True); parser.add_argument("--parallel-mode",choices=["attn_tp","moe_tp","moe_ep"],required=True); parser.add_argument("--freqs",type=int,nargs="+",default=DEFAULT_FREQS); parser.add_argument("--lengths",type=int,nargs="+",default=PREFILL_LENGTHS if phase=="prefill" else DECODE_LENGTHS); parser.add_argument("--batch-sizes",type=int,nargs="+",default=PREFILL_BATCHES if phase=="prefill" else DECODE_BATCHES); parser.add_argument("--repeat",type=int,default=50); parser.add_argument("--warmup",type=int,default=10); parser.add_argument("--seed",type=int,default=42); parser.add_argument("--layer-id",type=int,default=0); parser.add_argument("--clock-settle-seconds",type=float,default=.05); parser.add_argument("--output",required=True); parser.add_argument("--local-world-size",type=int); parser.add_argument("--quick",action="store_true")
    parser.add_argument(
        "--min-measure-seconds",
        type=float,
        default=None,
        help="minimum accumulated measurement window per point",
    )
    parser.add_argument(
        "--forced-active-experts-per-rank",
        type=int,
        default=None,
        help="limit active local experts for hot-rank controlled routing",
    )
    parser.add_argument(
        "--shape-token-limit",
        type=int,
        default=None,
        help=(
            "per-shape token safety limit; 0 uses only runner.max_total_num_tokens "
            "(default: 524288 for prefill A, otherwise 131072)"
        ),
    )
    parser.add_argument(
        "--stop-on-shape-failure",
        action="store_true",
        help="after a synchronized setup failure, stop the current length",
    )
def validate_topology(parser,args,server_args):
    world=server_args.tp_size
    if args.component=="A" and args.parallel_mode!="attn_tp": parser.error("component A requires --parallel-mode attn_tp")
    if args.component=="F" and args.parallel_mode=="attn_tp": parser.error("component F requires --parallel-mode moe_tp or moe_ep")
    if args.parallel_mode in ("attn_tp","moe_tp") and server_args.ep_size!=1: parser.error("attn_tp/moe_tp requires --ep-size 1")
    if args.parallel_mode=="moe_ep" and server_args.ep_size<=1: parser.error("moe_ep requires --ep-size > 1")
    if args.parallel_mode=="moe_ep" and server_args.moe_a2a_backend in ("ampere_ep", "deepep"):
        if not server_args.enable_dp_attention or server_args.dp_size!=world:
            parser.error(f"{server_args.moe_a2a_backend} moe_ep requires --enable-dp-attention and --dp-size == --tp-size")
    if world%server_args.ep_size: parser.error("--tp-size must be divisible by --ep-size")
def profile_main(phase):
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.server_args import PortArgs,ServerArgs
    from sglang.srt.utils import maybe_reindex_device_id
    parser=argparse.ArgumentParser(description=f"Qwen3-30B-A3B {phase} component profiler"); add_profile_args(parser,phase); args=parser.parse_args(); args.phase=phase
    if phase=="decode" and any(x<2 for x in args.lengths): parser.error("decode lengths must be >= 2")
    if args.shape_token_limit is not None and args.shape_token_limit < 0: parser.error("--shape-token-limit must be >= 0")
    if args.quick: args.lengths=[64]; args.batch_sizes=[1]; args.freqs=[args.freqs[-1]]; args.warmup=min(args.warmup,2); args.repeat=min(args.repeat,5)
    server_args=ServerArgs.from_cli_args(args); validate_topology(parser,args,server_args); _set_envs_and_config(server_args)
    if server_args.nnodes!=1: parser.error("this profiler currently supports one node only")
    local_world=args.local_world_size or server_args.tp_size
    if local_world!=server_args.tp_size: parser.error("--local-world-size must equal --tp-size on one node")
    port_args=PortArgs.init_new(server_args)
    if local_world==1: worker(server_args,port_args,args,0,0); return
    procs=[]
    for rank in range(local_world):
        with maybe_reindex_device_id(rank) as gpu_id:
            proc=multiprocessing.Process(target=worker,args=(server_args,port_args,args,gpu_id,rank)); proc.start(); procs.append(proc)
    for proc in procs: proc.join()
    failed=[(p.pid,p.exitcode) for p in procs if p.exitcode]
    if failed: raise SystemExit(f"profiling workers failed: {failed}")
