#!/usr/bin/env python3
"""Verify each visible GPU locks to requested SM clock (no torch/distributed)."""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from profile_utils import NvmlController  # noqa: E402


def worker(gpu_id: int, freq: int) -> dict:
    ctrl = NvmlController(gpu_id)
    requested = ctrl.snap(freq)
    before = int(ctrl.nvml.nvmlDeviceGetClockInfo(ctrl.handle, ctrl.nvml.NVML_CLOCK_GRAPHICS))
    ctrl.lock(requested)
    time.sleep(0.05)
    after = int(ctrl.nvml.nvmlDeviceGetClockInfo(ctrl.handle, ctrl.nvml.NVML_CLOCK_GRAPHICS))
    ctrl.unlock()
    ctrl.close()
    return {
        "logical_cuda_id": gpu_id,
        "physical_id": ctrl.physical_id,
        "requested_mhz": requested,
        "clock_before_mhz": before,
        "clock_after_mhz": after,
        "locked_ok": after == requested,
    }


def main() -> None:
    freq = int(sys.argv[1]) if len(sys.argv) > 1 else 930
    world = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    with mp.Pool(world) as pool:
        rows = pool.starmap(worker, [(i, freq) for i in range(world)])
    print(json.dumps(rows, indent=2))
    if not all(r["locked_ok"] for r in rows):
        raise SystemExit("some GPUs failed to lock")


if __name__ == "__main__":
    main()
