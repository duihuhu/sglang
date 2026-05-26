#!/usr/bin/env python3
"""Monitor GPU frequency and power during benchmark — shows Tier2 DVFS in action.

Runs in background during the benchmark, sampling GPU state every 0.5s.
Produces a CSV log and optional terminal plot showing frequency transitions.

Usage:
    python monitor_dvfs.py --duration 300 --output freq_trace.csv
    python monitor_dvfs.py --duration 300 --output freq_trace.csv --plot
"""

import argparse
import csv
import sys
import time
from pathlib import Path

GPU_INDICES = [0, 1, 2, 3]
GPU_ROLES = {0: "DF", 1: "DA", 2: "PF", 3: "PA"}


def sample_gpu_state(gpu_indices: list[int]) -> list[dict]:
    """Sample frequency, power, utilization for each GPU."""
    try:
        import pynvml
        pynvml.nvmlInit()
        samples = []
        for idx in gpu_indices:
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            freq = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            samples.append({
                "gpu": idx,
                "role": GPU_ROLES.get(idx, f"GPU{idx}"),
                "freq_mhz": freq,
                "power_w": power_mw / 1000.0,
                "util_pct": util.gpu,
                "energy_mj": energy,
            })
        pynvml.nvmlShutdown()
        return samples
    except Exception as e:
        print(f"NVML error: {e}", file=sys.stderr)
        return []


def main():
    parser = argparse.ArgumentParser(description="Monitor GPU DVFS during benchmark")
    parser.add_argument("--duration", type=float, default=300,
                        help="Monitoring duration in seconds")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="Sampling interval in seconds")
    parser.add_argument("--output", type=str, default="freq_trace.csv",
                        help="Output CSV file")
    parser.add_argument("--plot", action="store_true",
                        help="Print live frequency to terminal")
    parser.add_argument("--gpu-indices", type=str, default="0,1,2,3",
                        help="Comma-separated GPU indices to monitor")
    args = parser.parse_args()

    gpu_indices = [int(x) for x in args.gpu_indices.split(",")]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["time_s", "gpu", "role", "freq_mhz", "power_w", "util_pct", "energy_mj"]

    print(f"Monitoring GPUs {gpu_indices} for {args.duration}s → {out_path}")
    print(f"Roles: {GPU_ROLES}")
    print()

    with open(out_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        t_start = time.monotonic()
        sample_count = 0

        while (time.monotonic() - t_start) < args.duration:
            elapsed = time.monotonic() - t_start
            samples = sample_gpu_state(gpu_indices)

            for s in samples:
                row = {"time_s": f"{elapsed:.2f}", **s}
                writer.writerow(row)

            if args.plot and samples:
                freq_str = " | ".join(
                    f"{s['role']}:{s['freq_mhz']:4d}MHz {s['power_w']:5.1f}W"
                    for s in samples
                )
                print(f"\r[{elapsed:6.1f}s] {freq_str}", end="", flush=True)

            sample_count += 1
            time.sleep(args.interval)

    print(f"\n\nDone: {sample_count} samples written to {out_path}")

    # Print summary
    print("\n--- Frequency Distribution ---")
    import csv as csv_mod
    with open(out_path) as f:
        reader = csv_mod.DictReader(f)
        freq_by_role: dict[str, list[int]] = {}
        for row in reader:
            role = row["role"]
            freq = int(row["freq_mhz"])
            freq_by_role.setdefault(role, []).append(freq)

    for role, freqs in sorted(freq_by_role.items()):
        if not freqs:
            continue
        from collections import Counter
        dist = Counter(freqs)
        total = len(freqs)
        print(f"  {role}:")
        for f, cnt in sorted(dist.items()):
            pct = cnt / total * 100
            bar = "█" * int(pct / 2)
            print(f"    {f:5d} MHz: {cnt:4d} ({pct:5.1f}%) {bar}")


if __name__ == "__main__":
    main()
