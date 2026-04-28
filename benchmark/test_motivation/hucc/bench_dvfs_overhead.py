#!/usr/bin/env python3
"""
Benchmark: GPU DVFS frequency switching overhead measurement.

Validates the ~6ms switching overhead assumption from Paper 1.
Uses C++ NVML bindings (libdvfs_ctrl.so) for minimal-overhead timing.

Supports two NVML frequency control modes:
  --locked   : SetGpuLockedClocks (hard lock, both up/down effective)  [recommended]
  (default)  : SetApplicationsClocks (soft hint, only up-scaling works)

Measures three dimensions:
  1. C++ API call latency
  2. Frequency settle latency (until actual clock matches target)
  3. Kernel-level impact (GEMM performance before/after switch)

Prerequisites:
  cd benchmark/test_motivation/dvfs && make

Usage:
    # Locked clocks mode (recommended — hard frequency control)
    sudo python bench_dvfs_overhead.py --locked --full --repeat 50

    # ApplicationsClocks mode (soft hint, for comparison)
    sudo python bench_dvfs_overhead.py --full --repeat 50

    # Quick check with locked clocks
    sudo python bench_dvfs_overhead.py --locked --quick

Note:
    Both APIs require root privileges or nvidia-persistenced.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

# add sglang python path
_script_dir = Path(__file__).resolve().parent
_sglang_python = _script_dir.parent.parent / "python"
if _sglang_python.exists():
    sys.path.insert(0, str(_sglang_python))

# set .so path if running from benchmark dir
_so_path = _script_dir / "dvfs" / "libdvfs_ctrl.so"
if _so_path.exists():
    os.environ.setdefault("DVFS_CTRL_LIB", str(_so_path))

import torch
from sglang.srt.layers.dvfs import DVFSController


# ── Helpers for dual-mode (locked vs app clocks) ────────────────────────

def set_freq(ctrl: DVFSController, mhz: int, use_locked: bool) -> int:
    """Set SM frequency using either locked clocks or application clocks."""
    if use_locked:
        return ctrl.lock_sm_clock(mhz)
    else:
        return ctrl.set_sm_clock(mhz)


def set_freq_timed(ctrl: DVFSController, mhz: int, use_locked: bool) -> int:
    """Set SM frequency (timed) — returns API latency in nanoseconds."""
    if use_locked:
        return ctrl.lock_sm_clock_timed(mhz)
    else:
        return ctrl.set_sm_clock_timed(mhz)


def reset_freq(ctrl: DVFSController, use_locked: bool) -> int:
    """Reset frequency to default."""
    if use_locked:
        return ctrl.unlock_sm_clock()
    else:
        return ctrl.reset()


def mode_label(use_locked: bool) -> str:
    return "SetGpuLockedClocks" if use_locked else "SetApplicationsClocks"


def check_prerequisites(gpu_idx: int, use_locked: bool) -> DVFSController:
    """Check that NVML and GPU are available, frequency control works."""
    try:
        ctrl = DVFSController(gpu_idx)
    except Exception as e:
        print(f"[ERROR] Cannot initialize DVFSController for GPU {gpu_idx}: {e}")
        print("Make sure libdvfs_ctrl.so is built:")
        print("  cd benchmark/test_motivation/dvfs && make")
        sys.exit(1)

    # check if we can set frequency
    try:
        max_sm = ctrl.get_max_sm_clock()
        ret = set_freq(ctrl, max_sm, use_locked)
        if ret != 0:
            raise RuntimeError(f"set freq returned {ret}")
        reset_freq(ctrl, use_locked)
        print(f"[OK] Frequency control available on GPU {gpu_idx} "
              f"(mode: {mode_label(use_locked)})")
    except Exception as e:
        print(f"[ERROR] Cannot set GPU frequency: {e}")
        print("This typically requires root privileges or nvidia-persistenced.")
        print("Try: sudo python bench_dvfs_overhead.py")
        sys.exit(1)

    return ctrl


def print_table(headers, rows, title=None):
    """Print a formatted ASCII table."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    hdr = "|" + "|".join(
        f" {h:<{col_widths[i]}} " for i, h in enumerate(headers)
    ) + "|"

    if title:
        print(f"\n{'=' * len(sep)}")
        print(f" {title}")
        print(f"{'=' * len(sep)}")
    print(sep)
    print(hdr)
    print(sep)
    for row in rows:
        line = "|" + "|".join(
            f" {str(v):<{col_widths[i]}} " for i, v in enumerate(row)
        ) + "|"
        print(line)
    print(sep)


# ── Test 1: C++ API call latency ────────────────────────────────────────

def bench_api_overhead(ctrl: DVFSController, freq_candidates, n_warmup, n_repeat,
                       use_locked: bool):
    """Measure raw NVML API call latency using C++ high_resolution_clock."""
    label = mode_label(use_locked)
    print(f"\n[1/3] Measuring C++ API call latency ({label})...")
    print(f"      Candidates: {freq_candidates} MHz")
    print(f"      Warmup: {n_warmup}, Repeat: {n_repeat}")

    results = []
    for from_f in freq_candidates:
        for to_f in freq_candidates:
            if from_f == to_f:
                continue

            api_ns_list = []
            for trial in range(n_warmup + n_repeat):
                set_freq(ctrl, from_f, use_locked)
                time.sleep(0.02)

                # C++ timed call
                elapsed_ns = set_freq_timed(ctrl, to_f, use_locked)

                if trial >= n_warmup:
                    api_ns_list.append(elapsed_ns)

            api_us = [ns / 1000.0 for ns in api_ns_list]
            api_us.sort()
            avg = sum(api_us) / len(api_us)
            p50 = api_us[len(api_us) // 2]
            p99 = api_us[int(len(api_us) * 0.99)]
            mn, mx = min(api_us), max(api_us)

            results.append({
                "from_mhz": from_f, "to_mhz": to_f,
                "avg_us": round(avg, 1), "p50_us": round(p50, 1),
                "p99_us": round(p99, 1),
                "min_us": round(mn, 1), "max_us": round(mx, 1),
            })

    reset_freq(ctrl, use_locked)

    headers = ["From(MHz)", "To(MHz)", "Avg(us)", "P50(us)", "P99(us)",
               "Min(us)", "Max(us)"]
    rows = [[r["from_mhz"], r["to_mhz"], r["avg_us"], r["p50_us"],
             r["p99_us"], r["min_us"], r["max_us"]] for r in results]
    print_table(headers, rows, f"C++ {label} API Latency")

    return results


# ── Test 2: Frequency settle time ───────────────────────────────────────

def bench_settle_time(ctrl: DVFSController, freq_candidates, n_warmup, n_repeat,
                      use_locked: bool):
    """Measure time for actual GPU clock to reach target after API call."""
    label = mode_label(use_locked)
    print(f"\n[2/3] Measuring frequency settle time ({label})...")

    results = []
    for from_f in freq_candidates:
        for to_f in freq_candidates:
            if from_f == to_f:
                continue

            settle_times = []
            settled_flags = []

            for trial in range(n_warmup + n_repeat):
                set_freq(ctrl, from_f, use_locked)
                time.sleep(0.02)

                set_freq(ctrl, to_f, use_locked)
                t_start = time.perf_counter()

                # busy-poll actual frequency (time.sleep has ms-level granularity)
                settled = False
                deadline = t_start + 0.1  # 100ms timeout
                while time.perf_counter() < deadline:
                    actual = ctrl.get_sm_clock()
                    # tolerance: actual may not exactly match target due to
                    # GPU boost behavior, accept within 15 MHz
                    if abs(actual - to_f) <= 15:
                        settled = True
                        break

                t_end = time.perf_counter()
                settle_us = (t_end - t_start) * 1e6

                if trial >= n_warmup:
                    settle_times.append(settle_us)
                    settled_flags.append(settled)

            avg = sum(settle_times) / len(settle_times)
            p50 = sorted(settle_times)[len(settle_times) // 2]
            p99 = sorted(settle_times)[int(len(settle_times) * 0.99)]
            rate = sum(settled_flags) / len(settled_flags) * 100

            results.append({
                "from_mhz": from_f, "to_mhz": to_f,
                "avg_us": round(avg, 1), "p50_us": round(p50, 1),
                "p99_us": round(p99, 1), "settle_rate": round(rate, 1),
            })

    reset_freq(ctrl, use_locked)

    headers = ["From(MHz)", "To(MHz)", "Avg(us)", "P50(us)", "P99(us)", "Settled%"]
    rows = [[r["from_mhz"], r["to_mhz"], r["avg_us"], r["p50_us"],
             r["p99_us"], r["settle_rate"]] for r in results]
    print_table(headers, rows, f"Frequency Settle Time ({label})")

    return results


# ── Test 3: Kernel-level impact ─────────────────────────────────────────

def bench_kernel_impact(ctrl: DVFSController, freq_candidates,
                        n_warmup, n_repeat, kernel_size, gpu_idx,
                        use_locked: bool):
    """Measure GEMM kernel performance impact of frequency switching."""
    label = mode_label(use_locked)
    print(f"\n[3/3] Measuring kernel-level impact ({label}, GEMM {kernel_size}x{kernel_size})...")

    device = torch.device(f"cuda:{gpu_idx}")
    a = torch.randn(kernel_size, kernel_size, device=device, dtype=torch.float16)
    b = torch.randn(kernel_size, kernel_size, device=device, dtype=torch.float16)

    def run_gemm(n=1):
        times = []
        for _ in range(n):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter_ns()
            _ = torch.mm(a, b)
            torch.cuda.synchronize(device)
            t1 = time.perf_counter_ns()
            times.append((t1 - t0) / 1000.0)
        return times

    # 3a: absolute kernel time at each frequency
    print(f"\n  [3a] Kernel time at each frequency:")
    freq_kernel_times = {}
    for freq in freq_candidates:
        set_freq(ctrl, freq, use_locked)
        time.sleep(0.05)
        run_gemm(5)  # warmup
        times = run_gemm(n_repeat)
        avg = sum(times) / len(times)
        freq_kernel_times[freq] = avg
        print(f"       {freq:>5} MHz: {avg:>10.1f} us")

    # 3b: switch + first kernel latency
    print(f"\n  [3b] Switch + first kernel latency:")
    results = []
    for from_f in [freq_candidates[0], freq_candidates[-1]]:
        for to_f in freq_candidates:
            if from_f == to_f:
                continue

            switch_plus_kernel = []
            first_kernel = []

            for trial in range(n_warmup + n_repeat):
                set_freq(ctrl, from_f, use_locked)
                time.sleep(0.02)
                run_gemm(3)

                # switch + immediately run kernel
                torch.cuda.synchronize(device)
                t0 = time.perf_counter_ns()
                set_freq(ctrl, to_f, use_locked)
                _ = torch.mm(a, b)
                torch.cuda.synchronize(device)
                t1 = time.perf_counter_ns()
                total_us = (t1 - t0) / 1000.0

                fk = run_gemm(1)[0]

                if trial >= n_warmup:
                    switch_plus_kernel.append(total_us)
                    first_kernel.append(fk)

            avg_total = sum(switch_plus_kernel) / len(switch_plus_kernel)
            avg_first = sum(first_kernel) / len(first_kernel)
            steady = freq_kernel_times.get(to_f, avg_first)
            overhead = avg_total - steady

            results.append({
                "from_mhz": from_f, "to_mhz": to_f,
                "switch_plus_kernel_us": round(avg_total, 1),
                "first_kernel_after_us": round(avg_first, 1),
                "steady_kernel_us": round(steady, 1),
                "estimated_overhead_us": round(overhead, 1),
            })

    reset_freq(ctrl, use_locked)

    headers = ["From(MHz)", "To(MHz)", "Switch+Kernel(us)",
               "1st After(us)", "Steady(us)", "Est.Overhead(us)"]
    rows = [[r["from_mhz"], r["to_mhz"], r["switch_plus_kernel_us"],
             r["first_kernel_after_us"], r["steady_kernel_us"],
             r["estimated_overhead_us"]] for r in results]
    print_table(headers, rows, f"Kernel Impact ({label})")

    return results, freq_kernel_times


# ── Bonus: Continuous switching simulation ──────────────────────────────

def bench_continuous_switching(ctrl: DVFSController, freq_low, freq_high,
                               n_iterations, gpu_idx, kernel_size,
                               use_locked: bool):
    """Simulate per-iteration DVFS: alternate freq every iteration."""
    label = mode_label(use_locked)
    print(f"\n[Bonus] Continuous switching simulation ({label}, {n_iterations} iters)...")
    print(f"        Alternating {freq_low} MHz <-> {freq_high} MHz")

    device = torch.device(f"cuda:{gpu_idx}")
    a = torch.randn(kernel_size, kernel_size, device=device, dtype=torch.float16)
    b = torch.randn(kernel_size, kernel_size, device=device, dtype=torch.float16)

    def timed_gemm():
        torch.cuda.synchronize(device)
        t0 = time.perf_counter_ns()
        _ = torch.mm(a, b)
        torch.cuda.synchronize(device)
        return (time.perf_counter_ns() - t0) / 1000.0

    # baseline: fixed high freq
    set_freq(ctrl, freq_high, use_locked)
    time.sleep(0.05)
    baseline = [timed_gemm() for _ in range(n_iterations)]

    # with switching: no settle wait — intentionally measures the real
    # per-iteration overhead when frequency is changed every iteration
    freqs = [freq_low, freq_high]
    switched = []
    for i in range(n_iterations):
        set_freq(ctrl, freqs[i % 2], use_locked)
        switched.append(timed_gemm())

    reset_freq(ctrl, use_locked)

    avg_base = sum(baseline) / len(baseline)
    avg_switch = sum(switched) / len(switched)
    overhead_pct = (avg_switch - avg_base) / avg_base * 100

    print(f"  Baseline (fixed {freq_high} MHz):  {avg_base:.1f} us/iter")
    print(f"  With switching:                  {avg_switch:.1f} us/iter")
    print(f"  Overhead:                        {overhead_pct:+.1f}%")

    return {
        "baseline_us": round(avg_base, 1),
        "switching_us": round(avg_switch, 1),
        "overhead_pct": round(overhead_pct, 1),
    }


# ── Multi-GPU parallel vs serial switching ──────────────────────────────

def bench_multi_gpu(freq_candidates, n_warmup, n_repeat):
    """Compare serial vs thread-parallel vs process-parallel switching."""
    from sglang.srt.layers.dvfs import DVFSManager

    mgr = DVFSManager()
    n_gpus = len(mgr.controllers)
    if n_gpus < 2:
        print(f"\n[Multi-GPU] Only {n_gpus} GPU detected, skipping.")
        return

    print(f"\n[Multi-GPU] Testing serial vs thread-parallel vs process-parallel "
          f"on {n_gpus} GPUs...")

    freq_low, freq_high = freq_candidates[0], freq_candidates[-1]

    # ── Serial switching ────────────────────────────────────────────────
    serial_times = []
    for trial in range(n_warmup + n_repeat):
        mgr.lock_sm_clock_all(freq_low)
        time.sleep(0.02)

        t0 = time.perf_counter_ns()
        mgr.lock_sm_clock_all(freq_high)
        t1 = time.perf_counter_ns()

        if trial >= n_warmup:
            serial_times.append((t1 - t0) / 1000.0)

    # ── Thread-parallel switching ───────────────────────────────────────
    thread_times = []
    thread_per_gpu = []
    for trial in range(n_warmup + n_repeat):
        mgr.lock_sm_clock_all(freq_low)
        time.sleep(0.02)

        t0 = time.perf_counter_ns()
        per_gpu_ns = mgr.lock_sm_clock_all_parallel(freq_high)
        t1 = time.perf_counter_ns()

        if trial >= n_warmup:
            thread_times.append((t1 - t0) / 1000.0)
            thread_per_gpu.append(per_gpu_ns)

    mgr.unlock_all()

    # ── Process-parallel switching ──────────────────────────────────────
    import multiprocessing as mp
    so_path = os.environ.get("DVFS_CTRL_LIB", "")
    if not so_path:
        # try to find it
        candidate = Path(__file__).resolve().parent / "dvfs" / "libdvfs_ctrl.so"
        if candidate.exists():
            so_path = str(candidate)

    proc_times = []
    proc_per_gpu = []
    ctx = mp.get_context("spawn")
    # create pool once, reuse for all trials
    with ctx.Pool(processes=n_gpus, initializer=_worker_init,
                  initargs=(so_path,)) as pool:
        for trial in range(n_warmup + n_repeat):
            mgr.lock_sm_clock_all(freq_low)
            time.sleep(0.02)

            t0 = time.perf_counter_ns()
            per_gpu_ns = _multiprocess_lock(pool, n_gpus, freq_high)
            t1 = time.perf_counter_ns()

            if trial >= n_warmup:
                proc_times.append((t1 - t0) / 1000.0)
                proc_per_gpu.append(per_gpu_ns)

    mgr.unlock_all()

    # ── Results ─────────────────────────────────────────────────────────
    def stats(times):
        avg = sum(times) / len(times)
        p50 = sorted(times)[len(times) // 2]
        mx = max(times)
        return round(avg, 1), round(p50, 1), round(mx, 1)

    s_avg, s_p50, s_max = stats(serial_times)
    t_avg, t_p50, t_max = stats(thread_times)
    p_avg, p_p50, p_max = stats(proc_times)

    headers = ["Method", "GPUs", "Avg(us)", "P50(us)", "Max(us)"]
    rows = [
        ["Serial", n_gpus, s_avg, s_p50, s_max],
        ["Thread-parallel", n_gpus, t_avg, t_p50, t_max],
        ["Process-parallel", n_gpus, p_avg, p_p50, p_max],
    ]
    print_table(headers, rows, f"Multi-GPU Switching: {freq_low} → {freq_high} MHz")

    # per-GPU latency for thread mode
    if thread_per_gpu:
        _print_per_gpu(thread_per_gpu, "Per-GPU Latency (Thread-Parallel)")

    # per-GPU latency for process mode
    if proc_per_gpu:
        _print_per_gpu(proc_per_gpu, "Per-GPU Latency (Process-Parallel)")

    print(f"\n  Serial:           {s_avg/1000:.1f} ms avg for {n_gpus} GPUs")
    print(f"  Thread-parallel:  {t_avg/1000:.1f} ms avg for {n_gpus} GPUs "
          f"({s_avg/t_avg:.1f}x vs serial)")
    print(f"  Process-parallel: {p_avg/1000:.1f} ms avg for {n_gpus} GPUs "
          f"({s_avg/p_avg:.1f}x vs serial)")


def _print_per_gpu(per_gpu_list, title):
    """Print per-GPU latency table from a list of {gpu_idx: ns} dicts."""
    gpu_avgs = {}
    for d in per_gpu_list:
        for idx, ns in d.items():
            gpu_avgs.setdefault(idx, []).append(ns / 1000.0)

    headers = ["GPU", "Avg(us)", "P50(us)", "Max(us)"]
    rows = []
    for idx in sorted(gpu_avgs.keys()):
        vals = sorted(gpu_avgs[idx])
        avg = sum(vals) / len(vals)
        p50 = vals[len(vals) // 2]
        mx = max(vals)
        rows.append([idx, round(avg, 1), round(p50, 1), round(mx, 1)])
    print_table(headers, rows, title)


def _worker_init(so_path):
    """Per-process initializer: load .so and init NVML once."""
    import os
    os.environ["DVFS_CTRL_LIB"] = so_path
    from sglang.srt.layers.dvfs import _load_lib
    _load_lib()


def _worker_lock_gpu(args):
    """Worker function for process-parallel switching.

    Each worker is a separate process with its own NVML context,
    so there is no global lock contention.
    The returned ns is the C++-measured SetGpuLockedClocks latency only
    (does not include process scheduling overhead).
    """
    gpu_idx, freq_mhz = args
    from sglang.srt.layers.dvfs import DVFSController
    ctrl = DVFSController(gpu_idx)
    ns = ctrl.lock_sm_clock_timed(freq_mhz)
    return gpu_idx, ns


def _multiprocess_lock(pool, n_gpus, freq_mhz):
    """Lock all GPUs in parallel using a pre-created process pool."""
    results = pool.map(_worker_lock_gpu, [(i, freq_mhz) for i in range(n_gpus)])
    return {idx: ns for idx, ns in results}


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GPU DVFS frequency switching overhead"
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--freqs", type=int, nargs="+", default=None,
                        help="Frequency candidates in MHz (default: paper {210,540,870,1200})")
    parser.add_argument("--full", action="store_true",
                        help="Run all benchmarks including kernel impact")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer pairs, fewer repeats")
    parser.add_argument("--kernel-size", type=int, default=4096)
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to CSV file")
    parser.add_argument("--continuous", type=int, default=0,
                        help="Run continuous switching simulation for N iterations")
    parser.add_argument("--locked", action="store_true",
                        help="Use SetGpuLockedClocks (hard lock) instead of "
                             "SetApplicationsClocks (soft hint)")
    parser.add_argument("--multi-gpu", action="store_true",
                        help="Run multi-GPU parallel vs serial switching test "
                             "(requires --locked, uses all available GPUs)")
    args = parser.parse_args()

    if args.quick:
        args.repeat = 10
        args.warmup = 2

    use_locked = args.locked
    label = mode_label(use_locked)

    print("=" * 60)
    print(" GPU DVFS Frequency Switching Overhead Benchmark")
    print(f" Mode: {label}")
    print(" (C++ NVML bindings via libdvfs_ctrl.so)")
    print("=" * 60)

    ctrl = check_prerequisites(args.gpu, use_locked)

    # show GPU info
    info = ctrl.info()
    print(f"\nGPU {args.gpu}:")
    print(f"  SM clock:  {info['sm_clock_mhz']} MHz (max {info['max_sm_clock_mhz']})")
    print(f"  Mem clock: {info['mem_clock_mhz']} MHz (max {info['max_mem_clock_mhz']})")
    print(f"  Power:     {info['power_w']} W / {info['power_limit_w']} W")
    print(f"  Temp:      {info['temperature_c']} C")

    supported = ctrl.supported_sm_clocks
    print(f"  Supported SM clocks ({len(supported)} values):")
    print(f"    Min: {min(supported)} MHz, Max: {max(supported)} MHz")
    if len(supported) <= 20:
        print(f"    All: {supported}")

    # determine test frequencies
    if args.freqs:
        freq_candidates = sorted(set(ctrl.snap_to_supported(f) for f in args.freqs))
    else:
        paper_freqs = [210, 540, 870, 1200]
        freq_candidates = sorted(set(ctrl.snap_to_supported(f) for f in paper_freqs))

    print(f"\nTest frequencies: {freq_candidates} MHz")
    if args.quick:
        freq_candidates = [freq_candidates[0], freq_candidates[-1]]
        print(f"  (quick mode: {freq_candidates})")
    print(f"Repeat: {args.repeat}, Warmup: {args.warmup}")

    all_results = {}

    # Test 1
    api_results = bench_api_overhead(
        ctrl, freq_candidates, args.warmup, args.repeat, use_locked
    )
    all_results["api_overhead"] = api_results

    # Test 2
    settle_results = bench_settle_time(
        ctrl, freq_candidates, args.warmup, args.repeat, use_locked
    )
    all_results["settle_time"] = settle_results

    # Test 3
    if args.full:
        kernel_results, freq_kernel_times = bench_kernel_impact(
            ctrl, freq_candidates, args.warmup, args.repeat,
            args.kernel_size, args.gpu, use_locked
        )
        all_results["kernel_impact"] = kernel_results
        all_results["freq_kernel_times"] = freq_kernel_times

    # Bonus
    if args.continuous > 0:
        cont = bench_continuous_switching(
            ctrl, freq_candidates[0], freq_candidates[-1],
            args.continuous, args.gpu, args.kernel_size, use_locked
        )
        all_results["continuous_switching"] = cont

    # ── Multi-GPU test ──────────────────────────────────────────────────
    if args.multi_gpu and use_locked:
        bench_multi_gpu(freq_candidates, args.warmup, args.repeat)

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f" Summary ({label})")
    print("=" * 60)

    api_avg = sum(r["avg_us"] for r in api_results) / len(api_results)
    api_max = max(r["max_us"] for r in api_results)
    settle_avg = sum(r["avg_us"] for r in settle_results) / len(settle_results)
    settle_p99_max = max(r["p99_us"] for r in settle_results)

    print(f"  C++ API latency:      avg {api_avg:.0f} us, max {api_max:.0f} us")
    print(f"  Frequency settle:     avg {settle_avg:.0f} us, P99 max {settle_p99_max:.0f} us")
    total_avg_ms = (api_avg + settle_avg) / 1000
    total_worst_ms = (api_max + settle_p99_max) / 1000
    print(f"  Total switch overhead: ~{total_avg_ms:.2f} ms (avg)")
    print(f"                         ~{total_worst_ms:.2f} ms (worst case)")

    paper_ms = 6.0
    if total_worst_ms <= paper_ms:
        print(f"\n  [PASS] Worst case {total_worst_ms:.2f} ms <= paper assumption {paper_ms} ms")
    else:
        print(f"\n  [INFO] Worst case {total_worst_ms:.2f} ms > paper assumption {paper_ms} ms")
        print(f"         Consider adjusting paper assumption or using lazy switching.")

    # save CSV
    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["mode", "test", "from_mhz", "to_mhz", "metric", "value"])
            for test_name, results in all_results.items():
                if isinstance(results, list):
                    for r in results:
                        if isinstance(r, dict):
                            for k, v in r.items():
                                if k not in ("from_mhz", "to_mhz"):
                                    writer.writerow([label, test_name,
                                                     r.get("from_mhz", ""),
                                                     r.get("to_mhz", ""), k, v])
                elif isinstance(results, dict):
                    for k, v in results.items():
                        writer.writerow([label, test_name, "", "", k, v])
        print(f"\n  Results saved to {args.output}")

    reset_freq(ctrl, use_locked)
    print(f"\n  GPU clocks reset to default.")


if __name__ == "__main__":
    main()
