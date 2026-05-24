"""
GPU DVFS (Dynamic Voltage and Frequency Scaling) control module.

Python ctypes wrapper around libdvfs_ctrl.so (C++ NVML bindings).
Supports two frequency control modes:
  - SetApplicationsClocks (soft hint, only up-scaling effective)
  - SetGpuLockedClocks    (hard lock, both up/down effective)  [recommended]

Architecture:
    Python (ctypes) → libdvfs_ctrl.so (C++) → libnvidia-ml.so (NVML C API)

This avoids Python-level pynvml overhead for minimal switching latency.

Build the .so first:
    cd benchmark/test_motivation/dvfs && make

Usage:
    from sglang.srt.layers.dvfs import DVFSController

    ctrl = DVFSController(device_index=0)
    print(ctrl.supported_sm_clocks)    # available SM clock frequencies

    # Locked clocks (recommended — hard frequency control)
    ctrl.lock_sm_clock(1200)           # force GPU to run at 1200 MHz
    print(ctrl.get_sm_clock())         # read current SM clock
    ctrl.unlock_sm_clock()             # restore auto-boost

    # Application clocks (soft hint)
    ctrl.set_sm_clock(1200)            # suggest 1200 MHz to driver
    ctrl.reset()                       # reset to default

    # Timed calls (returns API latency in nanoseconds)
    latency_ns = ctrl.lock_sm_clock_timed(540)
    print(f"Switch took {latency_ns / 1000:.1f} us")
"""

from __future__ import annotations

import atexit
import ctypes
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── .so loading ─────────────────────────────────────────────────────────

_lib: Optional[ctypes.CDLL] = None
_lib_initialized = False


def _find_so() -> str:
    """Locate libdvfs_ctrl.so by searching known paths."""
    candidates = []

    # 1. env var override
    env_path = os.environ.get("DVFS_CTRL_LIB")
    if env_path:
        candidates.append(env_path)

    # 2. relative to this file: ../../../../benchmark/test_motivation/dvfs/
    this_dir = Path(__file__).resolve().parent
    candidates.append(
        str(this_dir / ".." / ".." / ".." / ".." / "benchmark"
            / "test_motivation" / "dvfs" / "libdvfs_ctrl.so")
    )

    # 3. relative to this file: ../../../../benchmark/test_motivation/hucc/dvfs/
    candidates.append(
        str(this_dir / ".." / ".." / ".." / ".." / "benchmark"
            / "test_motivation" / "hucc" / "dvfs" / "libdvfs_ctrl.so")
    )

    # 4. relative to cwd
    candidates.append("benchmark/test_motivation/dvfs/libdvfs_ctrl.so")
    candidates.append("benchmark/test_motivation/hucc/dvfs/libdvfs_ctrl.so")

    # 5. system library path
    candidates.append("libdvfs_ctrl.so")

    for p in candidates:
        resolved = Path(p).resolve()
        if resolved.exists():
            return str(resolved)

    # last resort: let ctypes search LD_LIBRARY_PATH / system paths
    return "libdvfs_ctrl.so"


def _load_lib() -> ctypes.CDLL:
    """Load and initialize the DVFS control library."""
    global _lib, _lib_initialized
    if _lib is not None:
        return _lib

    so_path = _find_so()
    logger.info(f"Loading DVFS control library from {so_path}")
    _lib = ctypes.CDLL(so_path)

    # declare function signatures
    _lib.dvfs_init.restype = ctypes.c_int
    _lib.dvfs_init.argtypes = []

    _lib.dvfs_shutdown.restype = ctypes.c_int
    _lib.dvfs_shutdown.argtypes = []

    _lib.dvfs_get_device_count.restype = ctypes.c_int
    _lib.dvfs_get_device_count.argtypes = []

    _lib.dvfs_get_device_name.restype = ctypes.c_int
    _lib.dvfs_get_device_name.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int
    ]

    _lib.dvfs_get_sm_clock.restype = ctypes.c_int
    _lib.dvfs_get_sm_clock.argtypes = [ctypes.c_int]

    _lib.dvfs_get_mem_clock.restype = ctypes.c_int
    _lib.dvfs_get_mem_clock.argtypes = [ctypes.c_int]

    _lib.dvfs_get_max_sm_clock.restype = ctypes.c_int
    _lib.dvfs_get_max_sm_clock.argtypes = [ctypes.c_int]

    _lib.dvfs_get_max_mem_clock.restype = ctypes.c_int
    _lib.dvfs_get_max_mem_clock.argtypes = [ctypes.c_int]

    _lib.dvfs_get_power_mw.restype = ctypes.c_int
    _lib.dvfs_get_power_mw.argtypes = [ctypes.c_int]

    _lib.dvfs_get_power_limit_mw.restype = ctypes.c_int
    _lib.dvfs_get_power_limit_mw.argtypes = [ctypes.c_int]

    _lib.dvfs_get_temperature.restype = ctypes.c_int
    _lib.dvfs_get_temperature.argtypes = [ctypes.c_int]

    _lib.dvfs_get_supported_mem_clocks.restype = ctypes.c_int
    _lib.dvfs_get_supported_mem_clocks.argtypes = [
        ctypes.c_int, ctypes.POINTER(ctypes.c_uint), ctypes.c_int
    ]

    _lib.dvfs_get_supported_sm_clocks.restype = ctypes.c_int
    _lib.dvfs_get_supported_sm_clocks.argtypes = [
        ctypes.c_int, ctypes.c_uint,
        ctypes.POINTER(ctypes.c_uint), ctypes.c_int
    ]

    _lib.dvfs_set_app_clocks.restype = ctypes.c_int
    _lib.dvfs_set_app_clocks.argtypes = [
        ctypes.c_int, ctypes.c_uint, ctypes.c_uint
    ]

    _lib.dvfs_set_sm_clock.restype = ctypes.c_int
    _lib.dvfs_set_sm_clock.argtypes = [ctypes.c_int, ctypes.c_uint]

    _lib.dvfs_reset_clocks.restype = ctypes.c_int
    _lib.dvfs_reset_clocks.argtypes = [ctypes.c_int]

    # ── Locked Clocks (SetGpuLockedClocks) ──
    _lib.dvfs_lock_sm_clock.restype = ctypes.c_int
    _lib.dvfs_lock_sm_clock.argtypes = [ctypes.c_int, ctypes.c_uint]

    _lib.dvfs_lock_sm_clock_range.restype = ctypes.c_int
    _lib.dvfs_lock_sm_clock_range.argtypes = [
        ctypes.c_int, ctypes.c_uint, ctypes.c_uint
    ]

    _lib.dvfs_unlock_sm_clock.restype = ctypes.c_int
    _lib.dvfs_unlock_sm_clock.argtypes = [ctypes.c_int]

    _lib.dvfs_lock_sm_clock_timed.restype = ctypes.c_int
    _lib.dvfs_lock_sm_clock_timed.argtypes = [
        ctypes.c_int, ctypes.c_uint, ctypes.POINTER(ctypes.c_longlong)
    ]

    # ── Timed ApplicationsClocks ──

    _lib.dvfs_set_sm_clock_timed.restype = ctypes.c_int
    _lib.dvfs_set_sm_clock_timed.argtypes = [
        ctypes.c_int, ctypes.c_uint, ctypes.POINTER(ctypes.c_longlong)
    ]

    _lib.dvfs_set_app_clocks_timed.restype = ctypes.c_int
    _lib.dvfs_set_app_clocks_timed.argtypes = [
        ctypes.c_int, ctypes.c_uint, ctypes.c_uint,
        ctypes.POINTER(ctypes.c_longlong)
    ]

    # ── Energy consumption ──
    _lib.dvfs_get_energy_mj.restype = ctypes.c_longlong
    _lib.dvfs_get_energy_mj.argtypes = [ctypes.c_int]

    # initialize NVML
    ret = _lib.dvfs_init()
    if ret != 0:
        raise RuntimeError(f"dvfs_init() failed with error code {ret}")
    _lib_initialized = True
    atexit.register(_shutdown)

    return _lib


def _shutdown():
    global _lib, _lib_initialized
    if _lib is not None and _lib_initialized:
        _lib.dvfs_shutdown()
        _lib_initialized = False


# ── DVFSController ──────────────────────────────────────────────────────


@dataclass
class DVFSController:
    """Per-GPU DVFS controller using C++ NVML bindings (libdvfs_ctrl.so)."""

    device_index: int
    _lib: ctypes.CDLL = field(init=False, repr=False, default=None)
    _supported_sm: Optional[List[int]] = field(init=False, repr=False, default=None)
    _supported_mem: Optional[List[int]] = field(init=False, repr=False, default=None)
    _max_mem_clock: Optional[int] = field(init=False, repr=False, default=None)

    def __post_init__(self):
        self._lib = _load_lib()
        # cache max memory clock (used for set_sm_clock)
        self._max_mem_clock = self._lib.dvfs_get_max_mem_clock(self.device_index)
        if self._max_mem_clock < 0:
            raise RuntimeError(
                f"Cannot query max memory clock for GPU {self.device_index}"
            )
        # log device info
        name_buf = ctypes.create_string_buffer(256)
        self._lib.dvfs_get_device_name(self.device_index, name_buf, 256)
        name = name_buf.value.decode("utf-8", errors="replace")
        logger.info(
            f"DVFSController GPU {self.device_index}: {name}, "
            f"max_mem={self._max_mem_clock} MHz"
        )

    # ── Query ───────────────────────────────────────────────────────────

    @property
    def supported_sm_clocks(self) -> List[int]:
        """Sorted list of supported SM clock frequencies (MHz)."""
        if self._supported_sm is None:
            buf = (ctypes.c_uint * 256)()
            count = self._lib.dvfs_get_supported_sm_clocks(
                self.device_index, self._max_mem_clock, buf, 256
            )
            if count < 0:
                self._supported_sm = []
            else:
                self._supported_sm = sorted(set(int(buf[i]) for i in range(count)))
        return self._supported_sm

    @property
    def supported_mem_clocks(self) -> List[int]:
        """Sorted list of supported memory clock frequencies (MHz)."""
        if self._supported_mem is None:
            buf = (ctypes.c_uint * 64)()
            count = self._lib.dvfs_get_supported_mem_clocks(
                self.device_index, buf, 64
            )
            if count < 0:
                self._supported_mem = []
            else:
                self._supported_mem = sorted(set(int(buf[i]) for i in range(count)))
        return self._supported_mem

    def get_sm_clock(self) -> int:
        """Current SM clock in MHz."""
        return self._lib.dvfs_get_sm_clock(self.device_index)

    def get_mem_clock(self) -> int:
        """Current memory clock in MHz."""
        return self._lib.dvfs_get_mem_clock(self.device_index)

    def get_max_sm_clock(self) -> int:
        """Max SM clock in MHz."""
        return self._lib.dvfs_get_max_sm_clock(self.device_index)

    def get_power(self) -> float:
        """Current power draw in watts."""
        mw = self._lib.dvfs_get_power_mw(self.device_index)
        return mw / 1000.0 if mw >= 0 else -1.0

    def get_power_limit(self) -> float:
        """Power limit in watts."""
        mw = self._lib.dvfs_get_power_limit_mw(self.device_index)
        return mw / 1000.0 if mw >= 0 else -1.0

    def get_temperature(self) -> int:
        """GPU temperature in Celsius."""
        return self._lib.dvfs_get_temperature(self.device_index)

    def get_energy_mj(self) -> int:
        """Cumulative energy consumption in millijoules since driver load.

        Uses the hardware energy counter (nvmlDeviceGetTotalEnergyConsumption),
        which accumulates continuously at high resolution.

        To measure a workload's energy:
            e0 = ctrl.get_energy_mj()
            # ... run workload ...
            e1 = ctrl.get_energy_mj()
            energy_mj = e1 - e0
        """
        return self._lib.dvfs_get_energy_mj(self.device_index)

    # ── Control ─────────────────────────────────────────────────────────

    def set_sm_clock(self, sm_mhz: int) -> int:
        """Set SM clock frequency (MHz). Uses max memory clock.

        Returns 0 on success, NVML error code on failure.
        """
        return self._lib.dvfs_set_sm_clock(self.device_index, sm_mhz)

    def set_app_clocks(self, mem_mhz: int, sm_mhz: int) -> int:
        """Set both memory and SM clock frequencies (MHz).

        Returns 0 on success, NVML error code on failure.
        """
        return self._lib.dvfs_set_app_clocks(self.device_index, mem_mhz, sm_mhz)

    def reset(self) -> int:
        """Reset application clocks to default.

        Returns 0 on success, NVML error code on failure.
        """
        return self._lib.dvfs_reset_clocks(self.device_index)

    # ── Locked Clocks (SetGpuLockedClocks — hard frequency control) ─────

    def lock_sm_clock(self, sm_mhz: int) -> int:
        """Force-lock SM clock to an exact frequency (MHz).

        Uses nvmlDeviceSetGpuLockedClocks(min=sm_mhz, max=sm_mhz).
        Unlike set_sm_clock (ApplicationsClocks), this is a hard constraint:
        both up-scaling and down-scaling take effect immediately.

        Equivalent to: nvidia-smi -lgc <sm_mhz>,<sm_mhz>

        Returns 0 on success, NVML error code on failure.
        """
        return self._lib.dvfs_lock_sm_clock(self.device_index, sm_mhz)

    def lock_sm_clock_range(self, min_mhz: int, max_mhz: int) -> int:
        """Lock SM clock to a range [min_mhz, max_mhz].

        Equivalent to: nvidia-smi -lgc <min_mhz>,<max_mhz>

        Returns 0 on success, NVML error code on failure.
        """
        return self._lib.dvfs_lock_sm_clock_range(
            self.device_index, min_mhz, max_mhz
        )

    def unlock_sm_clock(self) -> int:
        """Remove locked clock constraint, restore auto-boost.

        Equivalent to: nvidia-smi -rgc

        Returns 0 on success, NVML error code on failure.
        """
        return self._lib.dvfs_unlock_sm_clock(self.device_index)

    def lock_sm_clock_timed(self, sm_mhz: int) -> int:
        """Lock SM clock and return C++-measured API latency in nanoseconds.

        Timing covers only the nvmlDeviceSetGpuLockedClocks() call.
        """
        elapsed_ns = ctypes.c_longlong(0)
        ret = self._lib.dvfs_lock_sm_clock_timed(
            self.device_index, sm_mhz, ctypes.byref(elapsed_ns)
        )
        if ret != 0:
            raise RuntimeError(
                f"dvfs_lock_sm_clock_timed failed with error code {ret}"
            )
        return elapsed_ns.value

    # ── Timed control (for overhead measurement) ────────────────────────

    def set_sm_clock_timed(self, sm_mhz: int) -> int:
        """Set SM clock and return C++-measured API latency in nanoseconds.

        The timing is done in C++ using std::chrono::high_resolution_clock,
        measuring only the nvmlDeviceSetApplicationsClocks() call itself.
        """
        elapsed_ns = ctypes.c_longlong(0)
        ret = self._lib.dvfs_set_sm_clock_timed(
            self.device_index, sm_mhz, ctypes.byref(elapsed_ns)
        )
        if ret != 0:
            raise RuntimeError(
                f"dvfs_set_sm_clock_timed failed with error code {ret}"
            )
        return elapsed_ns.value

    def set_app_clocks_timed(self, mem_mhz: int, sm_mhz: int) -> int:
        """Set app clocks and return C++-measured API latency in nanoseconds."""
        elapsed_ns = ctypes.c_longlong(0)
        ret = self._lib.dvfs_set_app_clocks_timed(
            self.device_index, mem_mhz, sm_mhz, ctypes.byref(elapsed_ns)
        )
        if ret != 0:
            raise RuntimeError(
                f"dvfs_set_app_clocks_timed failed with error code {ret}"
            )
        return elapsed_ns.value

    # ── Convenience ─────────────────────────────────────────────────────

    def snap_to_supported(self, target_mhz: int) -> int:
        """Find the closest supported SM clock to target_mhz."""
        clocks = self.supported_sm_clocks
        if not clocks:
            return target_mhz
        return min(clocks, key=lambda c: abs(c - target_mhz))

    @contextmanager
    def locked(self, sm_mhz: int):
        """Context manager: lock frequency on enter, unlock on exit.

        Uses SetGpuLockedClocks for hard frequency control.
        """
        self.lock_sm_clock(sm_mhz)
        try:
            yield self
        finally:
            self.unlock_sm_clock()

    def info(self) -> dict:
        """Summary dict of current GPU DVFS state."""
        return {
            "device_index": self.device_index,
            "sm_clock_mhz": self.get_sm_clock(),
            "mem_clock_mhz": self.get_mem_clock(),
            "max_sm_clock_mhz": self.get_max_sm_clock(),
            "max_mem_clock_mhz": self._max_mem_clock,
            "power_w": round(self.get_power(), 1),
            "power_limit_w": round(self.get_power_limit(), 1),
            "temperature_c": self.get_temperature(),
            "num_supported_sm_clocks": len(self.supported_sm_clocks),
        }


# ── Multi-GPU helper ────────────────────────────────────────────────────


class DVFSManager:
    """Manage DVFS controllers for multiple GPUs."""

    def __init__(self, device_indices: Optional[List[int]] = None):
        lib = _load_lib()
        if device_indices is None:
            n = lib.dvfs_get_device_count()
            device_indices = list(range(n))
        self.controllers = {i: DVFSController(i) for i in device_indices}

    def set_sm_clock_all(self, sm_mhz: int) -> None:
        """Set application clocks on all GPUs (serial)."""
        for ctrl in self.controllers.values():
            ctrl.set_sm_clock(sm_mhz)

    def lock_sm_clock_all(self, sm_mhz: int) -> None:
        """Force-lock SM clock on all GPUs (serial)."""
        for ctrl in self.controllers.values():
            ctrl.lock_sm_clock(sm_mhz)

    def lock_sm_clock_all_parallel(self, sm_mhz: int) -> dict[int, int]:
        """Force-lock SM clock on all GPUs in parallel using threads.

        Returns dict of {gpu_idx: latency_ns} for each GPU.
        """
        from concurrent.futures import ThreadPoolExecutor

        results = {}

        def _lock(idx, ctrl):
            ns = ctrl.lock_sm_clock_timed(sm_mhz)
            return idx, ns

        with ThreadPoolExecutor(max_workers=len(self.controllers)) as pool:
            futures = [pool.submit(_lock, i, c) for i, c in self.controllers.items()]
            for f in futures:
                idx, ns = f.result()
                results[idx] = ns

        return results

    def unlock_all(self) -> None:
        """Unlock SM clocks on all GPUs."""
        for ctrl in self.controllers.values():
            ctrl.unlock_sm_clock()

    def reset_all(self) -> None:
        """Reset application clocks on all GPUs."""
        for ctrl in self.controllers.values():
            ctrl.reset()

    def get_power_all(self) -> dict[int, float]:
        return {i: ctrl.get_power() for i, ctrl in self.controllers.items()}

    def info_all(self) -> dict[int, dict]:
        return {i: ctrl.info() for i, ctrl in self.controllers.items()}
