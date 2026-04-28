#!/usr/bin/env python3
# Copyright 2026
"""
不同 seq_len、batch_size、hidden 下的 **点对点 GPU 通信** 基准。

**仅支持 WORLD_SIZE==2**：rank0 ``dist.send(tensor, dst=1)``，rank1 ``dist.recv(tensor, src=0)``。

张量形状：[batch_size, seq_len, hidden_size]。计时 / 能耗：探针预估次数、计时段 for 前后各打一次时间；NVML 累计差分。

CSV 默认 ``comm_benchmark_results.csv``（列含 ``latency_us_mean`` 等）。

用法::

  torchrun --nproc_per_node=2 bash-test/comm_test/benchmark_comm_energy.py --seq-lens 128 --batch-sizes 1 --hidden-sizes 4096

固定迭代次数（便于扫参 / 复现实验）::

  torchrun --nproc_per_node=2 ... benchmark_comm_energy.py --fixed-comm-iters 2000 --fixed-idle-iters 2000 ...

按 ``batch×seq_len`` 反推迭代次数（总积 ``N = 10×8192×256``，``comm_n = N // (bs*seq_len)``）::

  torchrun --nproc_per_node=2 ... benchmark_comm_energy.py --default-total-seq-batch-product
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
import torch.distributed as dist

_COMM_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_OUT_CSV = os.path.join(_COMM_DIR, "comm_benchmark_results.csv")
_BASH_TEST = os.path.dirname(_COMM_DIR)
# 固定「seq_len×batch_size×迭代次数」总积时常用：10×8192×256
_DEFAULT_ITER_TOTAL_SEQ_BATCH = 10 * 8192 * 256
sys.path.insert(0, os.path.join(_BASH_TEST, "C_nvml_for_energy"))


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _p2p_once(tensor: torch.Tensor, rank: int) -> None:
    """严格双进程：0 → 1。"""
    if rank == 0:
        dist.send(tensor, dst=1)
    else:
        dist.recv(tensor, src=0)


@dataclass
class EnergyMeter:
    """读取 GPU 累计能耗（mJ）；失败时回退为功率采样积分。"""

    ok: bool
    dev_index: int
    _read: Callable[[], float]
    _shutdown: Optional[Callable[[], None]] = None

    @classmethod
    def create(cls, dev_index: int) -> "EnergyMeter":
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()

            def shutdown() -> None:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass

            handle = pynvml.nvmlDeviceGetHandleByIndex(dev_index)
            if hasattr(pynvml, "nvmlDeviceGetTotalEnergyConsumption"):
                def read_mj() -> float:
                    return float(pynvml.nvmlDeviceGetTotalEnergyConsumption(handle))

                return cls(True, dev_index, read_mj, shutdown)
        except Exception:
            pass

        try:
            from nvml_energy import NvmlEnergy  # type: ignore

            nv = NvmlEnergy()
            nv.__enter__()

            def read_mj() -> float:
                return float(nv.total_energy_mj(dev_index))

            def shutdown() -> None:
                try:
                    nv.__exit__(None, None, None)
                except Exception:
                    pass

            return cls(True, dev_index, read_mj, shutdown)
        except Exception:
            pass

        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(dev_index)

            def read_power_w() -> float:
                return float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0

            def shutdown() -> None:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass

            class _PowerIntegral:
                def __init__(self) -> None:
                    self._e_mj = 0.0
                    self._last_t = time.perf_counter()
                    self._last_p = read_power_w()

                def sample(self) -> float:
                    now = time.perf_counter()
                    p_now = read_power_w()
                    dt = now - self._last_t
                    self._e_mj += 0.5 * (self._last_p + p_now) * dt * 1000.0
                    self._last_t = now
                    self._last_p = p_now
                    return self._e_mj

            pit = _PowerIntegral()

            return cls(True, dev_index, pit.sample, shutdown)
        except Exception:
            def noop() -> float:
                return float("nan")

            return cls(False, dev_index, noop, None)

    def close(self) -> None:
        if self._shutdown:
            self._shutdown()

    @classmethod
    def create_noop(cls, dev_index: int) -> "EnergyMeter":
        return cls(False, dev_index, lambda: float("nan"), None)


def _trace_stderr(rank: int, world: int, enabled: bool, msg: str) -> None:
    if not enabled:
        return
    print(
        f"[trace rank {rank}/{world}] {time.perf_counter():.3f}s {msg}",
        file=sys.stderr,
        flush=True,
    )


def _smoke_p2p(local_rank: int, dtype: torch.dtype, rank: int) -> None:
    x = torch.ones(32, device=f"cuda:{local_rank}", dtype=dtype)
    _p2p_once(x, rank)
    torch.cuda.synchronize()
    dist.barrier()


def _broadcast_int_from_rank0(value: int, tensor: torch.Tensor, rank: int) -> int:
    """各 rank 使用相同整数；避免 probe 计时噪声导致 comm_n/idle_n 不一致而 NCCL 死锁。"""
    buf = torch.zeros(1, dtype=torch.long, device=tensor.device)
    if rank == 0:
        buf[0] = int(value)
    dist.broadcast(buf, src=0)
    return int(buf[0].item())


def _iters_for_target_duration(
    duration_s: float,
    seconds_per_iter: float,
    max_iters: int,
) -> Tuple[int, bool]:
    if (
        not math.isfinite(seconds_per_iter)
        or seconds_per_iter <= 0
        or not math.isfinite(duration_s)
        or duration_s <= 0
    ):
        return 1, False
    sec = max(seconds_per_iter, 1e-7)
    n_raw = int(round(duration_s / sec))
    n_raw = max(1, min(n_raw, 100_000_000))
    capped = n_raw > max_iters
    n = min(n_raw, max_iters)
    return max(1, n), capped


def _probe_p2p_seconds_per_iter(tensor: torch.Tensor, rank: int, probe_iters: int) -> float:
    dist.barrier()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(probe_iters):
        _p2p_once(tensor, rank)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / max(probe_iters, 1)


def _probe_barrier_seconds_per_iter(probe_iters: int) -> float:
    dist.barrier()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(probe_iters):
        dist.barrier()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / max(probe_iters, 1)


def _warmup_p2p(tensor: torch.Tensor, rank: int, steps: int) -> None:
    for _ in range(steps):
        _p2p_once(tensor, rank)
    torch.cuda.synchronize()
    dist.barrier()


def benchmark_one(
    tensor: torch.Tensor,
    duration_s: float,
    idle_duration_s: float,
    warmup: int,
    probe_iters: int,
    meter: EnergyMeter,
    *,
    rank: int = 0,
    world_size: int = 1,
    quiet: bool = False,
    trace_ranks: bool = False,
    case_label: str = "",
    max_iters_comm: int = 8000,
    max_iters_idle: int = 8000,
    fixed_comm_n: Optional[int] = None,
    fixed_idle_n: Optional[int] = None,
    iter_total_seq_batch: Optional[int] = None,
) -> dict:
    _trace_stderr(rank, world_size, trace_ranks, f"→ {case_label} ① warmup send/recv x{warmup}")
    _warmup_p2p(tensor, rank, warmup)
    _trace_stderr(rank, world_size, trace_ranks, f"← {case_label} ① warmup 结束")

    _trace_stderr(rank, world_size, trace_ranks, f"→ {case_label} ② probe send/recv")
    t_per = _probe_p2p_seconds_per_iter(tensor, rank, probe_iters)
    _trace_stderr(rank, world_size, trace_ranks, f"← {case_label} ② probe 结束")
    if fixed_comm_n is not None and fixed_comm_n > 0:
        comm_n = max(1, min(int(fixed_comm_n), max_iters_comm))
    elif iter_total_seq_batch is not None and iter_total_seq_batch > 0:
        bsz = int(tensor.shape[0])
        slen = int(tensor.shape[1])
        denom = bsz * slen
        raw = iter_total_seq_batch // denom if denom > 0 else 1
        comm_n = max(1, min(raw, max_iters_comm))
    else:
        comm_n, _ = _iters_for_target_duration(duration_s, t_per, max_iters_comm)
    comm_n = _broadcast_int_from_rank0(comm_n, tensor, rank)
    _trace_stderr(
        rank,
        world_size,
        trace_ranks,
        f"→ {case_label} ③ 计时段 send/recv x{comm_n}",
    )

    dist.barrier()
    torch.cuda.synchronize()
    e0 = meter._read()
    dist.barrier()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(comm_n):
        _p2p_once(tensor, rank)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    e1 = meter._read()
    comm_wall = t1 - t0
    comm_energy = e1 - e0 if meter.ok else float("nan")
    comm_e_per_iter = comm_energy / comm_n if comm_n and meter.ok else float("nan")
    _trace_stderr(rank, world_size, trace_ranks, f"← {case_label} ③ 计时段结束")

    t_per_barrier = _probe_barrier_seconds_per_iter(probe_iters)
    if fixed_idle_n is not None and fixed_idle_n > 0:
        idle_n = max(1, min(int(fixed_idle_n), max_iters_idle))
    elif iter_total_seq_batch is not None and iter_total_seq_batch > 0:
        idle_n = max(1, min(int(comm_n), max_iters_idle))
    else:
        idle_n, _ = _iters_for_target_duration(
            idle_duration_s, t_per_barrier, max_iters_idle
        )
    idle_n = _broadcast_int_from_rank0(idle_n, tensor, rank)

    dist.barrier()
    torch.cuda.synchronize()
    e2 = meter._read()
    dist.barrier()
    torch.cuda.synchronize()
    for _ in range(idle_n):
        dist.barrier()
    torch.cuda.synchronize()
    e3 = meter._read()
    idle_energy = e3 - e2 if meter.ok else float("nan")
    idle_e_per_iter = idle_energy / idle_n if idle_n and meter.ok else float("nan")

    nbytes = tensor.numel() * tensor.element_size()
    # 单向 send 字节量 / 墙钟（与 all_reduce 的 2× 不同）
    gbps = (nbytes * comm_n) / comm_wall / 1e9 if comm_wall > 0 else float("nan")
    lat_avg_us = (comm_wall / comm_n) * 1e6 if comm_n else float("nan")
    if not math.isnan(gbps):
        gbps = round(gbps, 2)
    if not math.isnan(lat_avg_us):
        lat_avg_us = round(lat_avg_us, 2)

    return {
        "latency_us_mean": lat_avg_us,
        "throughput_GB_s": gbps,
        "comm_energy_mj_per_iter": comm_e_per_iter,
        "idle_energy_mj_per_iter": idle_e_per_iter,
        "comm_iters": comm_n,
        "idle_iters": idle_n,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="点对点 dist.send(0→1) / dist.recv 通信与能耗基准（仅 WORLD_SIZE==2）。"
    )
    p.add_argument(
        "--tp",
        type=int,
        default=None,
        help="固定为 2（双进程）；省略时取 WORLD_SIZE，且必须为 2",
    )
    p.add_argument("--seq-lens", type=str, default="128,256", help="逗号分隔")
    p.add_argument("--batch-sizes", type=str, default="1,2", help="逗号分隔 batch size")
    p.add_argument("--hidden-sizes", type=str, default="5120", help="逗号分隔 hidden size")
    p.add_argument("--duration", type=float, default=3.0, help="send/recv 计时段目标秒数")
    p.add_argument(
        "--idle-duration",
        type=float,
        default=-1.0,
        help="空转 barrier 段秒数；<0 与 --duration 相同",
    )
    p.add_argument("--warmup", type=int, default=5, help="每种形状 warmup 次数")
    p.add_argument("--probe-iters", type=int, default=8, help="探针迭代次数")
    p.add_argument("--max-iters-comm", type=int, default=16000, help="send/recv 循环次数上限")
    p.add_argument("--max-iters-idle", type=int, default=16000, help="barrier 循环次数上限")
    p.add_argument(
        "--fixed-comm-iters",
        type=int,
        default=0,
        help=">0 时固定 send/recv 计时段迭代次数（仍受 --max-iters-comm 裁剪），不再用 --duration 推算",
    )
    p.add_argument(
        "--fixed-idle-iters",
        type=int,
        default=0,
        help=">0 时固定 barrier 段迭代次数（仍受 --max-iters-idle 裁剪），不再用 --idle-duration 推算",
    )
    p.add_argument(
        "--iter-total-seq-batch",
        type=int,
        default=0,
        metavar="N",
        help=">0 时：comm_n = min(--max-iters-comm, N // (batch_size * seq_len))；"
        "未设 --fixed-idle-iters 时 idle 次数与 comm_n 相同（仍受 --max-iters-idle 裁剪）。"
        f" 常用 N={_DEFAULT_ITER_TOTAL_SEQ_BATCH}（即 10×8192×256）。",
    )
    p.add_argument(
        "--default-total-seq-batch-product",
        action="store_true",
        help=f"等价于 --iter-total-seq-batch {_DEFAULT_ITER_TOTAL_SEQ_BATCH}（10×8192×256）；"
        "若同时显式传入正数 --iter-total-seq-batch，则以该显式值为准。",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=("float16", "bfloat16", "float32"),
    )
    p.add_argument("--out-csv", type=str, default=_DEFAULT_OUT_CSV, help="CSV 路径")
    p.add_argument("--no-out-csv", action="store_true", help="不写 CSV")
    p.add_argument("--quiet", action="store_true", help="少打印")
    p.add_argument("--trace-ranks", action="store_true", help="各 rank stderr 锚点")
    p.add_argument("--no-energy", action="store_true", help="跳过 NVML")
    args = p.parse_args()
    idle_d = args.idle_duration if args.idle_duration >= 0 else args.duration

    iter_total_sb = int(args.iter_total_seq_batch) if args.iter_total_seq_batch > 0 else 0
    if args.default_total_seq_batch_product and iter_total_sb <= 0:
        iter_total_sb = _DEFAULT_ITER_TOTAL_SEQ_BATCH
    iter_total_arg: Optional[int] = iter_total_sb if iter_total_sb > 0 else None

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))

    if args.tp is None:
        if "WORLD_SIZE" not in os.environ:
            p.error("未提供 --tp 且无 WORLD_SIZE；请用 torchrun 启动")
        args.tp = world_size

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    else:
        raise RuntimeError("需要 CUDA")

    if world_size != args.tp:
        raise RuntimeError(f"WORLD_SIZE={world_size} 与 --tp={args.tp} 不一致")
    if world_size != 2 or args.tp != 2:
        raise RuntimeError(
            "本脚本仅支持双进程 WORLD_SIZE==2、--tp==2（rank0 send→rank1 recv）。"
            " 请使用: torchrun --nproc_per_node=2 ... benchmark_comm_energy.py"
        )

    if args.trace_ranks:
        os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")

    dtype = getattr(torch, args.dtype)
    seq_lens = _parse_int_list(args.seq_lens)
    batch_sizes = _parse_int_list(args.batch_sizes)
    hidden_sizes = _parse_int_list(args.hidden_sizes)
    total_cases = len(seq_lens) * len(batch_sizes) * len(hidden_sizes)

    _trace_stderr(rank, world_size, args.trace_ranks, "init_process_group (NCCL)")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    dist.barrier()
    _trace_stderr(rank, world_size, args.trace_ranks, "init 完成")

    if args.no_energy:
        meter = EnergyMeter.create_noop(local_rank)
    else:
        # 多进程同时 nvmlInit 在部分驱动上会长时间互锁，后续 dist.barrier 表现为“卡住”
        meter: EnergyMeter
        for r in range(world_size):
            if rank == r:
                meter = EnergyMeter.create(local_rank)
            dist.barrier()
    dist.barrier()

    _trace_stderr(rank, world_size, args.trace_ranks, "smoke send/recv")
    _smoke_p2p(local_rank, dtype, rank)
    dist.barrier()

    dist.barrier()
    torch.cuda.synchronize()
    if rank == 0 and not meter.ok and not args.no_energy:
        print(
            "警告: 无法读取 NVML，能耗列为 nan",
            file=sys.stderr,
        )
    if rank == 0 and args.no_energy and not args.quiet:
        print("[benchmark_comm_energy] --no-energy，能耗为 nan", flush=True)
    if rank == 0 and not args.quiet:
        _mode = ""
        if iter_total_arg:
            _mode = f" | comm_n=N//(bs*seq), N={iter_total_arg}"
        print(
            f"[benchmark_comm_energy] 共 {total_cases} 组 | world=2 (0→1 P2P) | "
            f"comm={args.duration}s | idle={idle_d}s{_mode} | "
            f"seq={seq_lens} bs={batch_sizes} hidden={hidden_sizes}",
            flush=True,
        )
    dist.barrier()

    rows: List[dict] = []
    fieldnames = [
        "seq_len",
        "batch_size",
        "latency_us_mean",
        "throughput_GB_s",
    ]

    try:
        case_idx = 0
        for seq_len in seq_lens:
            for bs in batch_sizes:
                for hidden in hidden_sizes:
                    case_idx += 1
                    case_label = f"[{case_idx}/{total_cases} seq={seq_len} bs={bs} h={hidden}]"
                    dist.barrier()
                    torch.cuda.synchronize()
                    if rank == 0 and not args.quiet:
                        print(flush=True)
                    dist.barrier()
                    try:
                        t = torch.empty(
                            (bs, seq_len, hidden),
                            dtype=dtype,
                            device=f"cuda:{local_rank}",
                        )
                        t.uniform_(-0.5, 0.5)
                    except RuntimeError as e:
                        dist.barrier()
                        if rank == 0:
                            print(
                                f"跳过 OOM seq={seq_len} bs={bs} h={hidden}: {e}",
                                file=sys.stderr,
                            )
                        dist.barrier()
                        continue

                    fix_c = args.fixed_comm_iters if args.fixed_comm_iters > 0 else None
                    fix_i = args.fixed_idle_iters if args.fixed_idle_iters > 0 else None
                    stats = benchmark_one(
                        t,
                        duration_s=args.duration,
                        idle_duration_s=idle_d,
                        warmup=args.warmup,
                        probe_iters=max(1, args.probe_iters),
                        meter=meter,
                        rank=rank,
                        world_size=world_size,
                        quiet=args.quiet,
                        trace_ranks=args.trace_ranks,
                        case_label=case_label,
                        max_iters_comm=max(1, args.max_iters_comm),
                        max_iters_idle=max(1, args.max_iters_idle),
                        fixed_comm_n=fix_c,
                        fixed_idle_n=fix_i,
                        iter_total_seq_batch=iter_total_arg,
                    )
                    row = {
                        "seq_len": seq_len,
                        "batch_size": bs,
                        "latency_us_mean": stats["latency_us_mean"],
                        "throughput_GB_s": stats["throughput_GB_s"],
                    }
                    rows.append(row)
                    dist.barrier()
                    torch.cuda.synchronize()
                    if rank == 0 and not args.quiet:
                        print(
                            f"{case_label} 结果: "
                            f"lat≈{stats['latency_us_mean']:.2f} us  "
                            f"throughput={stats['throughput_GB_s']:.2f} GB/s  "
                            f"comm_E/iter={stats['comm_energy_mj_per_iter']:.4f} mJ  "
                            f"idle_E/iter={stats['idle_energy_mj_per_iter']:.4f} mJ",
                            flush=True,
                        )
                    elif rank == 0:
                        print(
                            f"[seq={seq_len} bs={bs} h={hidden}] "
                            f"lat≈{stats['latency_us_mean']:.2f} us  "
                            f"throughput={stats['throughput_GB_s']:.2f} GB/s  "
                            f"comm_E/iter={stats['comm_energy_mj_per_iter']:.4f} mJ  "
                            f"idle_E/iter={stats['idle_energy_mj_per_iter']:.4f} mJ",
                            flush=True,
                        )
                    dist.barrier()
                    del t
                    torch.cuda.empty_cache()
    finally:
        meter.close()
        if dist.is_initialized():
            dist.destroy_process_group()

    if rank == 0 and not args.no_out_csv and rows:
        out_path = os.path.abspath(args.out_csv)
        _out_dir = os.path.dirname(out_path)
        if _out_dir:
            os.makedirs(_out_dir, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k, "") for k in fieldnames})
        if not args.quiet:
            print(f"[benchmark_comm_energy] 结果已写入 CSV: {out_path}", flush=True)


if __name__ == "__main__":
    main()
