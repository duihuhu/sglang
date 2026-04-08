#!/usr/bin/env python3
"""
Benchmark: GPU idle power / energy baseline at each frequency.

Measures idle power draw at each locked SM frequency by reading the
NVML hardware energy counter over a quiet interval (no CUDA kernels).

Prerequisites:
    cd benchmark/test_motivation/dvfs && make

Usage:
    sudo python bench_idle_power.py
    sudo python bench_idle_power.py --gpu 0 --duration 20 --output idle_power.txt
"""

import argparse
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

from sglang.srt.layers.dvfs import DVFSController

DEFAULT_FREQS = [210, 450, 690, 930, 1170, 1410]


def main():
    parser = argparse.ArgumentParser(description="Measure GPU idle power at each frequency")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument("--freqs", type=int, nargs="+", default=None,
                        help=f"Frequencies in MHz (default: {DEFAULT_FREQS})")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Seconds to sample at each frequency (default: 30)")
    parser.add_argument("--output", type=str, default="idle_power.txt",
                        help="Output TSV file (default: idle_power.txt)")
    args = parser.parse_args()

    ctrl = DVFSController(args.gpu)
    freqs = args.freqs or DEFAULT_FREQS
    freqs = sorted(set(ctrl.snap_to_supported(f) for f in freqs))

    print(f"GPU {args.gpu}: {ctrl.info()['sm_clock_mhz']} MHz")
    print(f"Frequencies to test: {freqs}")
    print(f"Duration per freq: {args.duration}s")
    print()

    out_path = _script_dir / args.output
    results = []

    for i, freq in enumerate(freqs):
        ctrl.lock_sm_clock(freq)
        time.sleep(2.0)

        e0 = ctrl.get_energy_mj()
        t0 = time.perf_counter()
        time.sleep(args.duration)
        e1 = ctrl.get_energy_mj()
        t1 = time.perf_counter()

        dt = t1 - t0
        de = e1 - e0
        power_w = de / dt / 1000.0

        results.append({
            "gpu_clock": freq,
            "idle_power_W": round(power_w, 2),
            "energy_mj": de,
            "duration_s": round(dt, 2),
        })

        print(f"  [{i+1}/{len(freqs)}] {freq:>5} MHz: {power_w:.2f} W  "
              f"({de:.0f} mJ / {dt:.1f} s)")

    ctrl.unlock_sm_clock()

    with open(out_path, "w") as f:
        f.write("gpu_clock\tidle_power_W\tenergy_mj\tduration_s\n")
        for r in results:
            f.write(f"{r['gpu_clock']}\t{r['idle_power_W']}\t{r['energy_mj']}\t{r['duration_s']}\n")

    print(f"\nResults saved to {out_path}")
    print("GPU clocks restored to default.")


if __name__ == "__main__":
    main()
