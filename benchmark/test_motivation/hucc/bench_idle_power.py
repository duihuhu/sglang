#!/usr/bin/env python3
"""
Benchmark: GPU idle power / energy baseline at each frequency.

Measures idle power draw at each locked SM frequency by reading the
NVML hardware energy counter over a quiet interval (no CUDA kernels).

Prerequisites:
    cd benchmark/test_motivation/dvfs && make

Usage:
    sudo python bench_idle_power.py
    sudo python bench_idle_power.py --gpu 0 --duration 20
    sudo python bench_idle_power.py --gpu 0 1 2 3 --output idle_power.txt
    sudo python bench_idle_power.py --rounds 3   # multiple rounds for std

Output columns:
    gpu_idx  gpu_clock  idle_power_W  power_std_W  energy_mj  duration_s
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_sglang_python = _script_dir.parent.parent / "python"
if _sglang_python.exists():
    sys.path.insert(0, str(_sglang_python))

_so_path = _script_dir / "dvfs" / "libdvfs_ctrl.so"
if _so_path.exists():
    os.environ.setdefault("DVFS_CTRL_LIB", str(_so_path))

from sglang.srt.layers.dvfs import DVFSController  # noqa: E402

DEFAULT_FREQS = [210, 450, 690, 930, 1170, 1410]


def measure_idle_power(ctrl, freq, duration, settle=2.0):
    """Lock to freq, wait settle seconds, then measure for duration seconds."""
    ctrl.lock_sm_clock(freq)
    time.sleep(settle)

    e0 = ctrl.get_energy_mj()
    t0 = time.perf_counter()
    time.sleep(duration)
    e1 = ctrl.get_energy_mj()
    t1 = time.perf_counter()

    dt = t1 - t0
    de = e1 - e0
    power_w = de / dt / 1000.0
    return power_w, de, dt


def main():
    parser = argparse.ArgumentParser(
        description="Measure GPU idle power at each frequency")
    parser.add_argument("--gpu", type=int, nargs="+", default=[0],
                        help="GPU indices (default: [0])")
    parser.add_argument("--freqs", type=int, nargs="+", default=None,
                        help=f"Frequencies in MHz (default: {DEFAULT_FREQS})")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Seconds to sample at each frequency (default: 30)")
    parser.add_argument("--rounds", type=int, default=1,
                        help="Number of measurement rounds per freq (default: 1)")
    parser.add_argument("--output", type=str, default="idle_power.txt",
                        help="Output TSV file (default: idle_power.txt)")
    args = parser.parse_args()

    gpu_indices = args.gpu
    controllers = {}
    for idx in gpu_indices:
        controllers[idx] = DVFSController(idx)

    # use first GPU to snap frequencies
    ref_ctrl = controllers[gpu_indices[0]]
    freqs = args.freqs or DEFAULT_FREQS
    freqs = sorted(set(ref_ctrl.snap_to_supported(f) for f in freqs))

    print(f"{'=' * 60}")
    print(f" GPU Idle Power Baseline Benchmark")
    print(f"{'=' * 60}")
    print(f"  GPUs:       {gpu_indices}")
    print(f"  Frequencies: {freqs} MHz")
    print(f"  Duration:   {args.duration}s per measurement")
    print(f"  Rounds:     {args.rounds}")
    print(f"{'=' * 60}\n")

    results = []

    for gpu_idx in gpu_indices:
        ctrl = controllers[gpu_idx]
        print(f"--- GPU {gpu_idx} ---")

        for freq in freqs:
            powers = []
            for r in range(args.rounds):
                power_w, de, dt = measure_idle_power(
                    ctrl, freq, args.duration)
                powers.append(power_w)
                if args.rounds > 1:
                    print(f"  [{r + 1}/{args.rounds}] {freq:>5} MHz: "
                          f"{power_w:.2f} W")

            mean_w = sum(powers) / len(powers)
            if len(powers) > 1:
                std_w = math.sqrt(
                    sum((p - mean_w) ** 2 for p in powers) / (len(powers) - 1))
            else:
                std_w = 0.0

            total_energy = de  # last round's energy (for reference)
            total_duration = dt

            results.append({
                "gpu_idx": gpu_idx,
                "gpu_clock": freq,
                "idle_power_W": round(mean_w, 2),
                "power_std_W": round(std_w, 3),
                "energy_mj": round(total_energy, 1),
                "duration_s": round(total_duration, 2),
            })

            std_str = f" ± {std_w:.3f}" if args.rounds > 1 else ""
            print(f"  {freq:>5} MHz: {mean_w:.2f} W{std_str}")

        ctrl.unlock_sm_clock()
        print()

    out_path = _script_dir / args.output
    with open(out_path, "w") as f:
        f.write("gpu_idx\tgpu_clock\tidle_power_W\tpower_std_W\t"
                "energy_mj\tduration_s\n")
        for r in results:
            f.write(f"{r['gpu_idx']}\t{r['gpu_clock']}\t{r['idle_power_W']}\t"
                    f"{r['power_std_W']}\t{r['energy_mj']}\t{r['duration_s']}\n")

    print(f"Results saved to {out_path}")
    print("All GPU clocks restored to default.")


if __name__ == "__main__":
    main()
