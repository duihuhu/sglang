#!/usr/bin/env python3
"""
Decode Attention/FFN latency + energy profiling (fast KV cache setup).

Same goal as bench_decode_af.py, but uses a trick to speed up KV cache
preparation: instead of running output_len decode steps one by one, we
prefill (input_len + output_len - K) tokens at once, then only run K
decode steps to reach the target KV length. This is ~100x faster for
large output_len (e.g., 4096).

The final measurement is taken at KV length = input_len + output_len,
identical to the slow version.

Prerequisites:
    cd benchmark/test_motivation/dvfs && make

Usage:
    sudo python bench_decode_af_fast.py --model-path /models/Qwen/Qwen3-32B/ --quick
    sudo python bench_decode_af_fast.py --model-path /models/Qwen/Qwen3-32B/ --tp-size 1
    sudo python bench_decode_af_fast.py --model-path /models/Qwen/Qwen3-32B/ --tp-size 2
"""

import argparse
import logging
import multiprocessing
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

_script_dir = Path(__file__).resolve().parent
_sglang_python = _script_dir.parent.parent / "python"
if _sglang_python.exists():
    sys.path.insert(0, str(_sglang_python))
_so_path = _script_dir / "dvfs" / "libdvfs_ctrl.so"
if _so_path.exists():
    os.environ.setdefault("DVFS_CTRL_LIB", str(_so_path))

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.layers.dvfs import DVFSController
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    maybe_reindex_device_id,
    suppress_other_loggers,
)

DEFAULT_FREQS = [210, 450, 690, 930, 1170, 1410]
DEFAULT_INPUT_LENS = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
DEFAULT_OUTPUT_LENS = [64, 128, 256, 512, 1024, 2048, 4096]
DEFAULT_BATCH_SIZES = [1, 4, 8, 16, 32, 64, 128, 256]

# Number of real decode steps after the fast prefill.
# Must be >= warmup+1 to ensure KV cache is in decode mode before measurement.
DECODE_SETTLE_STEPS = 5


class TreeCacheNamespace(SimpleNamespace):
    def supports_swa(self):
        return False

    def supports_mamba(self):
        return False

    def is_chunk_cache(self):
        return False

    def is_tree_cache(self):
        return True

    def evict(self, params: EvictParams):
        pass


def load_model(server_args, port_args, gpu_id, tp_rank):
    suppress_other_loggers()
    model_config = ModelConfig.from_server_args(server_args)
    moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=gpu_id,
        tp_rank=tp_rank,
        tp_size=server_args.tp_size,
        moe_ep_rank=moe_ep_rank,
        moe_ep_size=server_args.ep_size,
        pp_rank=0,
        pp_size=1,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
    )
    if server_args.tp_size > 1:
        import torch.distributed as dist
        dist.barrier()
    return model_runner


def make_reqs(batch_size, prefill_len, max_output_len):
    """Create requests with prefill_len input tokens."""
    input_ids = np.random.randint(0, 10000, (batch_size, prefill_len), dtype=np.int32)
    sampling_params = SamplingParams(temperature=0, max_new_tokens=max_output_len)
    reqs = []
    for i in range(batch_size):
        req = Req(rid=i, origin_input_text="", origin_input_ids=list(input_ids[i]),
                  sampling_params=sampling_params)
        req.fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))
        reqs.append(req)
    return reqs


def run_prefill(reqs, model_runner):
    dummy_tree_cache = TreeCacheNamespace(
        page_size=model_runner.server_args.page_size,
        device=model_runner.device,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
    )
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=dummy_tree_cache,
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.prepare_for_extend()
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    logits_output = model_runner.forward(forward_batch).logits_output
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids, batch


def run_decode_steps(next_token_ids, batch, model_runner, n_steps):
    for _ in range(n_steps):
        batch.output_ids = next_token_ids
        batch.prepare_for_decode()
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
        logits_output = model_runner.forward(forward_batch).logits_output
        next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids, batch


def build_decode_forward_batch(next_token_ids, batch, model_runner):
    batch.output_ids = next_token_ids
    batch.prepare_for_decode()
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    model_runner.attn_backend.init_forward_metadata(forward_batch)
    return forward_batch


# ── TP-safe profiling helpers ────────────────────────────────────────────

MIN_MEASURE_TIME_S = 0.5


def _sync_skip(skip_local: bool, tp_size: int) -> bool:
    """All TP ranks agree on whether to skip. If ANY rank wants to skip, ALL skip.
    Uses the TP process group to avoid deadlocks with model-internal all-reduce."""
    if tp_size <= 1:
        return skip_local
    from sglang.srt.distributed.parallel_state import get_tp_group
    tp_group = get_tp_group().device_group
    flag = torch.tensor([1 if skip_local else 0], dtype=torch.int32, device="cuda")
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX, group=tp_group)
    return flag.item() > 0


def _sync_repeat_count(actual_repeat: int, tp_size: int) -> int:
    """All TP ranks agree on the same repeat count (use the max across ranks).
    Uses the TP process group to avoid deadlocks with model-internal all-reduce."""
    if tp_size <= 1:
        return actual_repeat
    from sglang.srt.distributed.parallel_state import get_tp_group
    tp_group = get_tp_group().device_group
    count = torch.tensor([actual_repeat], dtype=torch.int64, device="cuda")
    torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.MAX, group=tp_group)
    return count.item()


def profile_one(fn, ctrl, n_warmup, n_repeat, tp_size=1):
    for _ in range(n_warmup):
        fn()

    torch.cuda.synchronize()
    t_probe = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    t_single = time.perf_counter() - t_probe

    actual_repeat = max(n_repeat, int(MIN_MEASURE_TIME_S / max(t_single, 1e-6)) + 1)
    actual_repeat = _sync_repeat_count(actual_repeat, tp_size)

    torch.cuda.synchronize()
    e0 = ctrl.get_energy_mj()
    t0 = time.perf_counter()
    for _ in range(actual_repeat):
        fn()
    torch.cuda.synchronize()
    e1 = ctrl.get_energy_mj()
    t1 = time.perf_counter()
    return (t1 - t0) / actual_repeat * 1e6, (e1 - e0) / actual_repeat


# ── Main profiling worker ────────────────────────────────────────────────

def profiling_worker(server_args, port_args, bench_args, gpu_id, tp_rank):
    rank_print = print if tp_rank == 0 else lambda *_, **__: None
    tp_size = server_args.tp_size

    model_runner = load_model(server_args, port_args, gpu_id, tp_rank)
    ctrl = DVFSController(gpu_id)
    device = model_runner.device
    hidden_size = model_runner.model_config.hidden_size

    freqs = sorted(set(ctrl.snap_to_supported(f) for f in bench_args.freqs))
    input_lens = bench_args.input_lens
    output_lens = bench_args.output_lens
    batch_sizes = bench_args.batch_sizes
    n_warmup = bench_args.warmup
    n_repeat = bench_args.repeat

    rank_print(f"\n{'='*65}")
    rank_print(f" Decode A/F Profiling — FAST mode  (tp={tp_size})")
    rank_print(f"{'='*65}")
    rank_print(f"  Model:       {server_args.model_path}")
    rank_print(f"  GPU:         {gpu_id}  (TP rank {tp_rank})")
    rank_print(f"  Frequencies: {freqs} MHz")
    rank_print(f"  Input lens:  {input_lens}")
    rank_print(f"  Output lens: {output_lens}")
    rank_print(f"  Batch sizes: {batch_sizes}")
    rank_print(f"  Repeat:      {n_repeat},  Warmup: {n_warmup}")
    rank_print(f"  Settle steps: {DECODE_SETTLE_STEPS}")
    rank_print(f"  Max tokens:  {model_runner.max_total_num_tokens}")
    rank_print(f"{'='*65}")
    rank_print(f"  Strategy: prefill (il+ol-{DECODE_SETTLE_STEPS}) tokens, "
               f"then {DECODE_SETTLE_STEPS} decode steps → measure at KV=il+ol")
    rank_print(f"{'='*65}\n")

    out_path = _script_dir / bench_args.output
    completed = set()
    if tp_rank == 0 and out_path.exists():
        with open(out_path) as f:
            for line in f:
                if line.startswith("tp\t"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) >= 5:
                    completed.add(tuple(parts[:5]))
        rank_print(f"Resuming: {len(completed)} configs already done.")

    tp = tp_size
    configs = []
    total = 0
    for il in input_lens:
        for ol in output_lens:
            for bs in batch_sizes:
                total_kv = il + ol
                max_bs = model_runner.max_total_num_tokens // total_kv
                if bs > max_bs:
                    continue
                for freq in freqs:
                    total += 1
                    key = (str(tp), str(il), str(ol), str(freq), str(bs))
                    if key not in completed:
                        configs.append((il, ol, bs, freq))

    rank_print(f"Total: {total} configs ({len(configs)} remaining)\n")

    f_out = None
    if tp_rank == 0:
        write_header = not out_path.exists() or len(completed) == 0
        f_out = open(out_path, "a")
        if write_header:
            f_out.write("tp\tinput_len\toutput_len\tgpu_clock\tbatch_size\t"
                        "D_A_lat\tD_F_lat\tD_A_energy\tD_F_energy\n")
            f_out.flush()

    layer = model_runner.model.model.layers[0]
    done = len(completed)
    t_start = time.time()

    # Calibrated activation memory per token (set after first successful prefill).
    # Much more accurate than any formula — we measure the actual peak delta.
    activation_bytes_per_token = None

    # Group configs by (il, ol, bs) so we can reuse KV cache setup across freqs.
    grouped_configs = []
    for (il, ol, bs, freq) in configs:
        if not grouped_configs or grouped_configs[-1][0] != (il, ol, bs):
            grouped_configs.append(((il, ol, bs), [freq]))
        else:
            grouped_configs[-1][1].append(freq)

    try:
        for (il, ol, bs), freq_list in grouped_configs:
            model_runner.req_to_token_pool.clear()
            model_runner.token_to_kv_pool_allocator.clear()

            # ── Fast KV cache setup ──────────────────────────────────
            prefill_len = il + ol - DECODE_SETTLE_STEPS
            if prefill_len < 1:
                prefill_len = 1
            actual_decode_steps = (il + ol) - prefill_len - 1

            # ── Pre-flight: check capacity BEFORE any TP communication ──
            total_kv_tokens = (il + ol) * bs
            capacity_ok = total_kv_tokens <= model_runner.max_total_num_tokens
            # Only apply activation memory check for large prefills where OOM
            # is a real risk.  The calibrated activation_bytes_per_token is
            # measured from the *first* prefill and includes one-time overheads
            # (NCCL buffers, CUDA context growth, …), so it over-estimates for
            # small batches.  Skip the heuristic when total prefill tokens are
            # small (< 32k) — the OOM try/except will catch real failures.
            prefill_tokens = prefill_len * bs
            if capacity_ok and activation_bytes_per_token is not None and prefill_tokens >= 32768:
                torch.cuda.empty_cache()
                free_mem, _ = torch.cuda.mem_get_info(device)
                est_need = activation_bytes_per_token * prefill_tokens
                if free_mem < est_need * 1.2:
                    capacity_ok = False
                    rank_print(f"  [MEM-check] il={il} ol={ol} bs={bs} — "
                               f"free={free_mem/1e9:.2f}GB, "
                               f"est_need={est_need*1.2/1e9:.2f}GB "
                               f"(calib={activation_bytes_per_token:.0f} B/tok)")
            if not capacity_ok:
                torch.cuda.empty_cache()
            if _sync_skip(not capacity_ok, tp_size):
                rank_print(f"  [SKIP-capacity] il={il} ol={ol} bs={bs} — skipped "
                           f"({len(freq_list)} freq points)")
                done += len(freq_list)
                continue

            # ── Setup: prefill + decode settle (involves TP communication) ──
            setup_failed = False
            forward_batch = None
            hidden_states = None
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            mem_before = torch.cuda.memory_allocated(device)
            try:
                reqs = make_reqs(bs, prefill_len, actual_decode_steps + 1)
                next_token_ids, batch = run_prefill(reqs, model_runner)
                if actual_decode_steps > 0:
                    next_token_ids, batch = run_decode_steps(
                        next_token_ids, batch, model_runner, actual_decode_steps)
                forward_batch = build_decode_forward_batch(
                    next_token_ids, batch, model_runner)
                hidden_states = torch.randn(bs, hidden_size,
                                            device=device, dtype=torch.bfloat16)
                residual = hidden_states.clone()
                positions = forward_batch.positions
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                torch.cuda.empty_cache()
                setup_failed = True
                rank_print(f"  [OOM-setup-local] il={il} ol={ol} bs={bs} — {e}")

            # Calibrate activation_bytes_per_token from the first success.
            if not setup_failed and activation_bytes_per_token is None:
                peak = torch.cuda.max_memory_allocated(device)
                peak_delta = peak - mem_before
                n_tokens = prefill_len * bs
                if n_tokens > 0:
                    activation_bytes_per_token = peak_delta / n_tokens
                    rank_print(f"  [CALIBRATED] peak_delta={peak_delta/1e9:.2f}GB "
                               f"for {n_tokens} tokens → "
                               f"{activation_bytes_per_token:.0f} B/tok")

            if _sync_skip(setup_failed, tp_size):
                rank_print(f"  [OOM-setup] il={il} ol={ol} bs={bs} — skipped "
                           f"({len(freq_list)} freq points)")
                if forward_batch is not None:
                    del forward_batch
                if hidden_states is not None:
                    del hidden_states
                torch.cuda.empty_cache()
                done += len(freq_list)
                continue

            # ── Profile each frequency with the SAME KV cache ──
            for freq in freq_list:
                ret = ctrl.lock_sm_clock(freq)
                if ret != 0:
                    rank_print(f"  [ERROR] lock_sm_clock({freq}) failed (code={ret}). "
                               f"Need root? Try: sudo python ...")
                    break
                time.sleep(0.05)

                with torch.no_grad():
                    def _attn_with_norm():
                        hs, _ = layer.input_layernorm(
                            hidden_states, residual)
                        return layer.self_attn(
                            positions=positions,
                            hidden_states=hs,
                            forward_batch=forward_batch)

                    def _ffn_with_norm():
                        hs, _ = layer.post_attention_layernorm(
                            hidden_states, residual)
                        return layer.mlp(hs)

                    a_lat, a_energy = profile_one(
                        _attn_with_norm, ctrl, n_warmup, n_repeat, tp_size)
                    f_lat, f_energy = profile_one(
                        _ffn_with_norm, ctrl, n_warmup, n_repeat, tp_size)

                ctrl.unlock_sm_clock()

                if tp_rank == 0:
                    f_out.write(f"{tp}\t{il}\t{ol}\t{freq}\t{bs}\t"
                                f"{a_lat:.2f}\t{f_lat:.2f}\t"
                                f"{a_energy:.4f}\t{f_energy:.4f}\n")
                    f_out.flush()

                done += 1
                elapsed = time.time() - t_start
                progress = done - len(completed)
                remaining = total - done
                eta = elapsed / progress * remaining if progress > 0 else 0
                rank_print(f"  [{done}/{total}] il={il:>5} ol={ol:>4} bs={bs:>3} "
                           f"f={freq:>4}MHz | "
                           f"A: {a_lat:>9.1f}us {a_energy:>7.3f}mJ | "
                           f"F: {f_lat:>9.1f}us {f_energy:>7.3f}mJ | "
                           f"ETA: {eta/60:.1f}min")

            del hidden_states, residual, positions, forward_batch
            torch.cuda.empty_cache()

    except KeyboardInterrupt:
        rank_print("\n\n[Interrupted] Partial results saved.")
    finally:
        if f_out:
            f_out.close()
        ctrl.unlock_sm_clock()
        if tp_size > 1:
            from sglang.srt.distributed.parallel_state import destroy_distributed_environment
            destroy_distributed_environment()

    rank_print(f"\nDone. Results: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Decode A/F profiling (fast KV setup via prefill)")
    ServerArgs.add_cli_args(parser)
    parser.add_argument("--freqs", type=int, nargs="+", default=DEFAULT_FREQS)
    parser.add_argument("--input-lens", type=int, nargs="+", default=DEFAULT_INPUT_LENS)
    parser.add_argument("--output-lens", type=int, nargs="+", default=DEFAULT_OUTPUT_LENS)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=str, default="decode_data_v1.txt")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    server_args = ServerArgs.from_cli_args(args)

    bench_args = SimpleNamespace(
        freqs=args.freqs, input_lens=args.input_lens,
        output_lens=args.output_lens, batch_sizes=args.batch_sizes,
        repeat=args.repeat, warmup=args.warmup, output=args.output,
    )

    if args.quick:
        bench_args.input_lens = [512, 4096]
        bench_args.output_lens = [64, 512]
        bench_args.batch_sizes = [1, 16]
        bench_args.freqs = [bench_args.freqs[0], bench_args.freqs[-1]]
        bench_args.repeat = 20
        bench_args.warmup = 5

    _set_envs_and_config(server_args)
    server_args.disable_cuda_graph = True
    server_args.disable_cuda_graph_padding = True
    server_args.disable_piecewise_cuda_graph = True
    port_args = PortArgs.init_new(server_args)

    if server_args.tp_size == 1:
        profiling_worker(server_args, port_args, bench_args, 0, 0)
    else:
        workers = []
        for tp_rank in range(server_args.tp_size):
            with maybe_reindex_device_id(tp_rank) as gpu_id:
                proc = multiprocessing.Process(
                    target=profiling_worker,
                    args=(server_args, port_args, bench_args, gpu_id, tp_rank),
                )
                proc.start()
                workers.append(proc)
        for proc in workers:
            proc.join()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    main()
