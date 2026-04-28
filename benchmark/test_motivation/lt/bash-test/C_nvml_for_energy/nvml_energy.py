"""
ctypes 封装 libnvml_energy.so：应用时钟（-ac/-rac）、SM 锁频（-lgc 类）、累计能耗（mJ）。
需先 ./build.sh。
"""

from __future__ import annotations

import ctypes
import os

_NVML_SUCCESS = 0

_dir = os.path.dirname(os.path.abspath(__file__))
_lib_path = os.environ.get("NVML_ENERGY_SO", os.path.join(_dir, "libnvml_energy.so"))
_lib = ctypes.CDLL(_lib_path)

_lib.nvml_energy_init.argtypes = []
_lib.nvml_energy_init.restype = ctypes.c_int

_lib.nvml_energy_shutdown.argtypes = []
_lib.nvml_energy_shutdown.restype = None

_lib.nvml_energy_strerror.argtypes = [ctypes.c_int]
_lib.nvml_energy_strerror.restype = ctypes.c_char_p

_lib.nvml_clk_set_applications_mhz.argtypes = [
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
]
_lib.nvml_clk_set_applications_mhz.restype = ctypes.c_int

_lib.nvml_clk_reset_applications.argtypes = [ctypes.c_uint]
_lib.nvml_clk_reset_applications.restype = ctypes.c_int

_lib.nvml_clk_set_gpu_locked_mhz.argtypes = [
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
]
_lib.nvml_clk_set_gpu_locked_mhz.restype = ctypes.c_int

_lib.nvml_clk_reset_gpu_locked.argtypes = [ctypes.c_uint]
_lib.nvml_clk_reset_gpu_locked.restype = ctypes.c_int

_lib.dvfs_lock_sm_clock.argtypes = [ctypes.c_int, ctypes.c_uint]
_lib.dvfs_lock_sm_clock.restype = ctypes.c_int

_lib.dvfs_unlock_sm_clock.argtypes = [ctypes.c_int]
_lib.dvfs_unlock_sm_clock.restype = ctypes.c_int

_lib.nvml_energy_device_count.argtypes = [ctypes.POINTER(ctypes.c_uint)]
_lib.nvml_energy_device_count.restype = ctypes.c_int

_lib.nvml_energy_get_total_mj.argtypes = [
    ctypes.c_uint,
    ctypes.POINTER(ctypes.c_ulonglong),
]
_lib.nvml_energy_get_total_mj.restype = ctypes.c_int


def _check(ret: int, what: str) -> None:
    if ret == _NVML_SUCCESS:
        return
    if ret == -1:
        raise RuntimeError(f"{what}: -1 (e.g. gpu_idx < 0)")
    msg = _lib.nvml_energy_strerror(ret)
    s = msg.decode("utf-8", "replace") if msg else str(ret)
    raise RuntimeError(f"{what}: NVML error {ret} ({s})")


class NvmlEnergyGpu:
    """固定 ``gpu_index`` 的视图：锁频/能耗调用不再重复传卡号（对齐每卡一个控制器的用法）。"""

    __slots__ = ("_nv", "_gpu_index")

    def __init__(self, nv: "NvmlEnergy", gpu_index: int) -> None:
        self._nv = nv
        self._gpu_index = int(gpu_index)

    @property
    def gpu_index(self) -> int:
        return self._gpu_index

    def lock_sm_clock(self, sm_clock_mhz: int) -> None:
        self._nv.lock_sm_clock(self._gpu_index, sm_clock_mhz)

    def unlock_sm_clock(self) -> None:
        self._nv.unlock_sm_clock(self._gpu_index)

    def total_energy_mj(self) -> int:
        return self._nv.total_energy_mj(self._gpu_index)

    def set_applications_clocks_mhz(self, mem_mhz: int, graphics_mhz: int) -> None:
        self._nv.set_applications_clocks_mhz(
            self._gpu_index, mem_mhz, graphics_mhz
        )

    def reset_applications_clocks(self) -> None:
        self._nv.reset_applications_clocks(self._gpu_index)

    def set_gpu_locked_mhz(self, min_mhz: int, max_mhz: int | None = None) -> None:
        self._nv.set_gpu_locked_mhz(self._gpu_index, min_mhz, max_mhz)

    def reset_gpu_locked(self) -> None:
        self._nv.reset_gpu_locked(self._gpu_index)


class NvmlEnergy:
    """nvmlInit / shutdown + 应用时钟 + SM 锁频 + 累计能耗（mJ）。"""

    def __init__(self) -> None:
        self._inited = False

    def gpu(self, gpu_index: int) -> NvmlEnergyGpu:
        return NvmlEnergyGpu(self, gpu_index)

    def __enter__(self) -> "NvmlEnergy":
        r = _lib.nvml_energy_init()
        _check(r, "nvml_energy_init")
        self._inited = True
        return self

    def __exit__(self, *args: object) -> None:
        if self._inited:
            _lib.nvml_energy_shutdown()
            self._inited = False

    def set_applications_clocks_mhz(
        self, gpu_index: int, mem_mhz: int, graphics_mhz: int
    ) -> None:
        """等价 nvidia-smi -i <gpu> -ac <mem>,<graphics>。"""
        r = _lib.nvml_clk_set_applications_mhz(
            ctypes.c_uint(gpu_index),
            ctypes.c_uint(mem_mhz),
            ctypes.c_uint(graphics_mhz),
        )
        _check(r, f"nvml_clk_set_applications_mhz(gpu={gpu_index})")

    def reset_applications_clocks(self, gpu_index: int) -> None:
        """等价 nvidia-smi -i <gpu> -rac。"""
        r = _lib.nvml_clk_reset_applications(ctypes.c_uint(gpu_index))
        _check(r, f"nvml_clk_reset_applications(gpu={gpu_index})")

    def lock_sm_clock(self, gpu_index: int, sm_clock_mhz: int) -> None:
        """单档锁 SM（C: dvfs_lock_sm_clock）。"""
        r = _lib.dvfs_lock_sm_clock(ctypes.c_int(gpu_index), ctypes.c_uint(sm_clock_mhz))
        _check(r, f"dvfs_lock_sm_clock(gpu={gpu_index}, sm={sm_clock_mhz})")

    def unlock_sm_clock(self, gpu_index: int) -> None:
        """清除 SM 锁（C: dvfs_unlock_sm_clock）。"""
        r = _lib.dvfs_unlock_sm_clock(ctypes.c_int(gpu_index))
        _check(r, f"dvfs_unlock_sm_clock(gpu={gpu_index})")

    def set_gpu_locked_mhz(
        self, gpu_index: int, min_mhz: int, max_mhz: int | None = None
    ) -> None:
        """锁 SM 到 [min,max] MHz；默认 max=min（底层 nvml_clk_set_gpu_locked_mhz）。"""
        if max_mhz is None:
            max_mhz = min_mhz
        r = _lib.nvml_clk_set_gpu_locked_mhz(
            ctypes.c_uint(gpu_index),
            ctypes.c_uint(min_mhz),
            ctypes.c_uint(max_mhz),
        )
        _check(r, f"nvml_clk_set_gpu_locked_mhz(gpu={gpu_index})")

    def reset_gpu_locked(self, gpu_index: int) -> None:
        r = _lib.nvml_clk_reset_gpu_locked(ctypes.c_uint(gpu_index))
        _check(r, f"nvml_clk_reset_gpu_locked(gpu={gpu_index})")

    def device_count(self) -> int:
        n = ctypes.c_uint(0)
        r = _lib.nvml_energy_device_count(ctypes.byref(n))
        _check(r, "nvml_energy_device_count")
        return int(n.value)

    def total_energy_mj(self, gpu_index: int) -> int:
        """自驱动加载以来累计能耗（mJ），Volta+。"""
        v = ctypes.c_ulonglong(0)
        r = _lib.nvml_energy_get_total_mj(ctypes.c_uint(gpu_index), ctypes.byref(v))
        _check(r, f"nvml_energy_get_total_mj(gpu={gpu_index})")
        return int(v.value)


def read_all_energy_mj() -> list[int]:
    with NvmlEnergy() as nv:
        c = nv.device_count()
        return [nv.total_energy_mj(i) for i in range(c)]


if __name__ == "__main__":
    import sys

    try:
        with NvmlEnergy() as nv:
            n = nv.device_count()
            print(f"GPUs: {n}")
            for i in range(n):
                mj = nv.total_energy_mj(i)
                print(f"  [{i}] total_energy_mj={mj}")
    except RuntimeError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
