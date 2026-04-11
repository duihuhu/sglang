#!/usr/bin/env python3
"""
Prefill Attention/FFN latency profiling using real SGLang model.

Loads the actual model via SGLang's ModelRunner (same kernels as production:
FlashInfer/FlashAttention, RoPE, paged KV cache, etc.), then independently
profiles layer.self_attn and layer.mlp latency only.

Based on sglang.bench_one_batch pattern — no server needed.

Prerequisites:
    Build NVML helper .so (SM lock + energy counter):
        bash-test/C_nvml_for_energy/build.sh

Usage:
    # Quick validation (tp=1, small subset)
    sudo python bench_prefill_af.py --model-path Qwen/Qwen3-32B --load-format dummy --quick

    # Full sweep tp=1
    sudo python bench_prefill_af.py --model-path Qwen/Qwen3-32B --tp-size 1

    # Full sweep tp=2 (uses 2 GPUs)
    sudo python bench_prefill_af.py --model-path Qwen/Qwen3-32B --tp-size 2

    # Custom SM lock points and batch sizes
    sudo python bench_prefill_af.py --model-path Qwen/Qwen3-32B \\
        --freqs 210 690 1410 --batch-sizes 1 8 --input-lens 512 4096

Note:
    ``gpu_id`` 传给 ``ModelRunner`` 的是逻辑 CUDA 序号；NVML 锁频/能耗使用
    ``CUDA_VISIBLE_DEVICES`` 解析出的物理 ``nvml_idx``（十进制设备号列表时）。
    Per-config: NVML ``SetGpuLockedClocks`` (SM MHz, ``--freqs``). Some drivers
    reject or ignore SM lock changes while a large CUDA context is active; this
    script syncs GPU, unlocks previous lock before retargeting, then sleeps
    ``BENCH_SM_LOCK_SETTLE_S`` seconds (default 0.2). If locks still fail, use
    ``--skip-nvml-sm-lock`` and set SM externally before each run, or one
    ``--freqs`` per process. Energy: NVML mJ delta / repeat. Root.
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

_nvml_energy_dir = _script_dir.parent / "C_nvml_for_energy"
if _nvml_energy_dir.is_dir():
    sys.path.insert(0, str(_nvml_energy_dir))
try:
    from nvml_energy import NvmlEnergy, NvmlEnergyGpu

    _NVML_ENERGY_AVAILABLE = True
except Exception:
    NvmlEnergy = None  # type: ignore[misc, assignment]
    NvmlEnergyGpu = None  # type: ignore[misc, assignment]
    _NVML_ENERGY_AVAILABLE = False

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    kill_process_tree,
    maybe_reindex_device_id,
    suppress_other_loggers,
)

DEFAULT_FREQS = [210, 1410]
DEFAULT_INPUT_LENS = [128]
DEFAULT_BATCH_SIZES = [1]


def _debug_a_input_enabled() -> bool:
    return os.getenv("SGLANG_DEBUG_A_INPUT", "0") == "1"


def _tensor_brief(x: torch.Tensor | None) -> str:
    if x is None:
        return "None"
    shape = tuple(x.shape)
    dtype = str(x.dtype)
    device = str(x.device)
    if x.numel() == 0:
        return f"shape={shape} dtype={dtype} device={device} empty"
    x32 = x.detach().to(torch.float32)
    mean = float(x32.mean().item())
    std = float(x32.std().item())
    return (
        f"shape={shape} dtype={dtype} device={device} "
        f"mean={mean:.6f} std={std:.6f}"
    )


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


def make_reqs(batch_size, input_len):
    input_ids = np.random.randint(0, 10000, (batch_size, input_len), dtype=np.int32)
    sampling_params = SamplingParams(temperature=0, max_new_tokens=1)
    reqs = []
    for i in range(batch_size):
        req = Req(rid=i, origin_input_text="", origin_input_ids=list(input_ids[i]),
                  sampling_params=sampling_params)
        req.fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))
        reqs.append(req)
    return reqs


def nvml_device_index_for_cuda_gpu_id(cuda_gpu_id: int) -> int:
    """将 ``ModelRunner`` 使用的逻辑 ``gpu_id`` 映射为 NVML 设备序号。

    NVML 的 ``device index`` 按物理 GPU 固定枚举；设置 ``CUDA_VISIBLE_DEVICES``
    后，逻辑 ``cuda:N`` 对应环境变量里**从左到右第 N 个**条目。条目为十进制
    整数字符串时，即为该卡在 NVML 下的序号（与 ``nvidia-smi -i`` 一致）。

    若条目为 UUID、PCI 字符串等非纯数字，无法可靠解析，则退回 ``cuda_gpu_id``
    （可能与真实物理卡不一致）；此时请改用数字形式的 ``CUDA_VISIBLE_DEVICES``。
    """
    raw = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if not raw:
        return cuda_gpu_id
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if cuda_gpu_id < 0 or cuda_gpu_id >= len(parts):
        return cuda_gpu_id
    token = parts[cuda_gpu_id]
    if token.isdigit():
        return int(token)
    return cuda_gpu_id


def build_prefill_forward_batch(reqs, model_runner):
    """Build a valid prefill ForwardBatch with KV cache allocation and metadata."""
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
    model_runner.attn_backend.init_forward_metadata(forward_batch)
    return forward_batch


def profile_one(
    fn,
    n_warmup,
    n_repeat,
    sample_hook=None,
    *,
    nvdev=None,
):
    """Run fn(); return (avg_latency_us, avg_energy_mj_per_repeat).

    Energy = (NVML total mJ at end − at start) / n_repeat on the timed loop only
    (hardware counter). If ``nvdev`` is None, energy is NaN.
    """
    for _ in range(n_warmup):
        fn()

    torch.cuda.synchronize()
    actual_repeat = max(n_repeat, 1)

    torch.cuda.synchronize()

    if nvdev is not None:
        e0 = nvdev.total_energy_mj()
    else:
        e0 = 0
    t0 = time.perf_counter()

    for _ in range(actual_repeat):
        fn()
    torch.cuda.synchronize()

    if nvdev is not None:
        e1 = nvdev.total_energy_mj()
        avg_e = (e1 - e0) / actual_repeat
    else:
        avg_e = float("nan")
    t1 = time.perf_counter()

    return (t1 - t0) / actual_repeat * 1e6, avg_e


def profiling_worker(server_args, port_args, bench_args, gpu_id, tp_rank):
    """Main profiling logic — runs in each TP worker."""
    rank_print = print if tp_rank == 0 else lambda *_, **__: None

    model_runner = load_model(server_args, port_args, gpu_id, tp_rank)
    device = model_runner.device
    if isinstance(device, str):
        device = torch.device(device)
    hidden_size = model_runner.model_config.hidden_size

    nvml_idx = nvml_device_index_for_cuda_gpu_id(gpu_id)
    cvd_visible = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()

    freqs = sorted(set(bench_args.freqs))
    input_lens = bench_args.input_lens
    batch_sizes = bench_args.batch_sizes
    n_warmup = bench_args.warmup
    n_repeat = bench_args.repeat

    rank_print(f"\n{'='*65}")
    rank_print(f" Prefill A/F Latency Profiling  (real SGLang model, tp={server_args.tp_size})")
    rank_print(f"{'='*65}")
    rank_print(f"  Model:       {server_args.model_path}")
    if cvd_visible:
        rank_print(
            f"  GPU:         cuda_id={gpu_id}  nvml_idx={nvml_idx}  (TP rank {tp_rank})"
        )
    else:
        rank_print(f"  GPU:         cuda_id={gpu_id}  (TP rank {tp_rank})")
    rank_print(f"  SM lock MHz (--freqs, min=max): {freqs}")
    rank_print(f"  Input lens:  {input_lens}")
    rank_print(f"  Batch sizes: {batch_sizes}")
    rank_print(f"  Repeat:      {n_repeat},  Warmup: {n_warmup}")
    rank_print(f"  Max tokens:  {model_runner.max_total_num_tokens}")
    rank_print(f"{'='*65}\n")

    if cvd_visible:
        toks = [p.strip() for p in cvd_visible.split(",") if p.strip()]
        if (
            gpu_id < len(toks)
            and not toks[gpu_id].isdigit()
            and nvml_idx == gpu_id
        ):
            rank_print(
                "[warn] CUDA_VISIBLE_DEVICES 中对应 cuda:%d 的项不是十进制整数字符串，"
                "NVML 仍使用索引 %d，可能与实际 GPU 不一致；建议用数字设备号。"
                % (gpu_id, nvml_idx)
            )

    out_path = _script_dir / bench_args.output
    completed = set()
    if tp_rank == 0 and bench_args.resume and out_path.exists():
        with open(out_path) as f:
            for line in f:
                if line.startswith("tp\t"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    completed.add(tuple(parts[:4]))
        rank_print(f"Resuming: {len(completed)} configs already done.")

    configs = []
    total = 0
    tp = server_args.tp_size
    for il in input_lens:
        for bs in batch_sizes:
            max_bs = model_runner.max_total_num_tokens // il
            if bs > max_bs:
                continue
            for freq in freqs:
                total += 1
                key = (str(tp), str(il), str(freq), str(bs))
                if key not in completed:
                    configs.append((il, bs, freq))

    rank_print(f"Total: {total} configs ({len(configs)} remaining)\n")
    if len(configs) == 0:
        rank_print(
            "No pending config to run. GPU clock will NOT be adjusted in this run. "
            "Use --no-resume or a new --output file to force rerun."
        )

    f_out = None
    if tp_rank == 0:
        if bench_args.resume:
            write_header = not out_path.exists() or len(completed) == 0
            f_out = open(out_path, "a")
        else:
            write_header = True
            f_out = open(out_path, "w")
        if write_header:
            f_out.write(
                "tp\tinput_len\tsm_lock_mhz\tbatch_size\t"
                "P_A_lat\tP_F_lat\t"
                "P_A_E_mj_per_rep\tP_F_E_mj_per_rep\n"
            )
            f_out.flush()
    sample_f = None
    sample_path = _script_dir / bench_args.samples_output
    if tp_rank == 0:
        if bench_args.resume:
            write_samples_header = not sample_path.exists() or len(completed) == 0
            sample_f = open(sample_path, "a")
        else:
            write_samples_header = True
            sample_f = open(sample_path, "w")
        if write_samples_header:
            sample_f.write("tp\tinput_len\tsm_lock_mhz\tbatch_size\top\titer\tlatency_us\n")
            sample_f.flush()

    layer = model_runner.model.model.layers[0]
    done = len(completed)
    t_start = time.time()
    a_input_logged = False
    last_sm_mhz = None

    def _run_config_loop(nvg: "NvmlEnergyGpu") -> None:
        nonlocal done, a_input_logged, last_sm_mhz
        for il, bs, freq in configs:

            model_runner.req_to_token_pool.clear()
            model_runner.token_to_kv_pool_allocator.clear()

            try:
                reqs = make_reqs(bs, il)
                forward_batch = build_prefill_forward_batch(reqs, model_runner)
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                torch.cuda.empty_cache()
                rank_print(f"  [OOM-alloc] il={il} bs={bs} — skipped")
                continue

            n_tokens = forward_batch.seq_lens_sum
            hidden_states = torch.randn(n_tokens, hidden_size,
                                        device=device, dtype=torch.bfloat16)
            residual = hidden_states.clone()
            positions = forward_batch.positions
            
            nvg.lock_sm_clock(freq)
            print('sleeping for 3 seconds')
            time.sleep(5)

            try:
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

                    a_lat, a_e_mj = profile_one(
                        _attn_with_norm,
                        n_warmup,
                        n_repeat,
                        sample_hook=(
                            (lambda i, lat, _tp=tp, _il=il, _freq=freq, _bs=bs:
                                sample_f.write(
                                    f"{_tp}\t{_il}\t{_freq}\t{_bs}\tA\t{i}\t{lat:.6f}\n"
                                ))
                            if (tp_rank == 0 and sample_f is not None)
                            else None
                        ),
                        nvdev=nvg,
                    )
                    f_lat, f_e_mj = profile_one(
                        _ffn_with_norm,
                        n_warmup,
                        n_repeat,
                        sample_hook=(
                            (lambda i, lat, _tp=tp, _il=il, _freq=freq, _bs=bs:
                                sample_f.write(
                                    f"{_tp}\t{_il}\t{_freq}\t{_bs}\tF\t{i}\t{lat:.6f}\n"
                                ))
                            if (tp_rank == 0 and sample_f is not None)
                            else None
                        ),
                        nvdev=nvg,
                    )
                    if tp_rank == 0 and sample_f is not None:
                        sample_f.flush()
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                rank_print(f"  [OOM-run] il={il} bs={bs} freq={freq} err={e} — skipped")
                continue
            except RuntimeError as e:
                torch.cuda.empty_cache()
                rank_print(
                    f"  [run-failed] il={il} bs={bs} freq={freq} err={repr(e)} — skipped"
                )
                continue

            if tp_rank == 0:
                f_out.write(
                    f"{tp}\t{il}\t{freq}\t{bs}\t"
                    f"{a_lat:.2f}\t{f_lat:.2f}\t"
                    f"{a_e_mj:.6f}\t{f_e_mj:.6f}\n"
                )
                f_out.flush()

            done += 1
            elapsed = time.time() - t_start
            eta = elapsed / (done - len(completed)) * (len(configs) - (done - len(completed))) if done > len(completed) else 0
            rank_print(
                f"  [{done}/{total}] il={il:>5} bs={bs:>2} sm={freq:>4}MHz | "
                f"A: {a_lat:>10.1f}us E={a_e_mj:.4f}mJ/rep | "
                f"F: {f_lat:>10.1f}us E={f_e_mj:.4f}mJ/rep | "
                f"ETA: {eta/60:.1f}min"
            )

            del hidden_states, positions, forward_batch
            torch.cuda.empty_cache()

    try:
        if len(configs) > 0 and not _NVML_ENERGY_AVAILABLE:
            rank_print(
                "ERROR: nvml_energy / libnvml_energy.so not available. "
                "Build: bash-test/C_nvml_for_energy/build.sh"
            )
        elif len(configs) > 0:
            with NvmlEnergy() as nv:
                nvg = nv.gpu(nvml_idx)
                try:
                    _run_config_loop(nvg)
                finally:
                    if not bench_args.skip_nvml_sm_lock:
                        try:
                            nvg.unlock_sm_clock()
                        except Exception as e:
                            rank_print(
                                f"[warn] nvml unlock SM nvml_idx={nvml_idx}: {e}"
                            )
    except KeyboardInterrupt:
        rank_print("\n\n[Interrupted] Partial results saved.")
    finally:
        if f_out:
            f_out.close()
        if sample_f:
            sample_f.close()
        if server_args.tp_size > 1:
            from sglang.srt.distributed.parallel_state import destroy_distributed_environment
            destroy_distributed_environment()

    rank_print(f"\nDone. Results: {out_path}")


def _worker_with_env(env, server_args, port_args, bench_args, tp_rank):
    """Set per-process CUDA_VISIBLE_DEVICES then run profiling_worker with gpu_id=0."""
    os.environ.update(env)
    profiling_worker(server_args, port_args, bench_args, 0, tp_rank)


def main():
    parser = argparse.ArgumentParser(description="Prefill A/F latency profiling")
    ServerArgs.add_cli_args(parser)
    parser.add_argument(
        "--freqs",
        type=int,
        nargs="+",
        default=DEFAULT_FREQS,
        help=(
            f"SM lock MHz (min=max each, NVML SetGpuLockedClocks; default {DEFAULT_FREQS}). "
            "Must be supported lock points on this GPU."
        ),
    )
    parser.add_argument("--input-lens", type=int, nargs="+", default=DEFAULT_INPUT_LENS)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--mem-clock",
        type=int,
        default=1593,
        help="Unused for SM-lock sweep (ignored); kept for backward-compatible CLI.",
    )
    parser.add_argument("--output", type=str, default="prefill_data_v1.txt")
    parser.add_argument(
        "--samples-output",
        type=str,
        default="prefill_data_v1_samples.txt",
        help="Per-repeat latency output file (tab-separated).",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from existing output file by skipping completed configs "
             "(default: True). Use --no-resume to rerun all configs.",
    )
    parser.add_argument(
        "--skip-nvml-sm-lock",
        action="store_true",
        help="不在进程内调用 NVML 锁 SM；请在本机先设好对应档位再跑（见 Note）。",
    )
    args = parser.parse_args()

    server_args = ServerArgs.from_cli_args(args)

    bench_args = SimpleNamespace(
        freqs=args.freqs, input_lens=args.input_lens,
        batch_sizes=args.batch_sizes, repeat=args.repeat,
        warmup=args.warmup, mem_clock=args.mem_clock, output=args.output,
        samples_output=args.samples_output,
        resume=args.resume,
        skip_nvml_sm_lock=args.skip_nvml_sm_lock,
    )

    if args.quick:
        bench_args.input_lens = [512, 4096]
        bench_args.batch_sizes = [1, 8]
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
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if cvd:
            phys_gpus = [g.strip() for g in cvd.split(",") if g.strip()]
        else:
            phys_gpus = [str(i) for i in range(server_args.tp_size)]
        if len(phys_gpus) < server_args.tp_size:
            raise ValueError(
                f"CUDA_VISIBLE_DEVICES only has {len(phys_gpus)} device(s), "
                f"but tp_size={server_args.tp_size} requires at least that many"
            )
        for tp_rank in range(server_args.tp_size):
            per_rank_cvd = phys_gpus[tp_rank]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = per_rank_cvd
            env["SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS"] = "1"
            proc = multiprocessing.Process(
                target=_worker_with_env,
                args=(env, server_args, port_args, bench_args, tp_rank),
            )
            proc.start()
            workers.append(proc)
        for proc in workers:
            proc.join()
        for proc in workers:
            if proc.is_alive():
                proc.terminate()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    main()