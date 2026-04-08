#!/usr/bin/env python3
"""
调频 + 单算子能耗粗测（示例脚本）

流程：
1. 将 GPU 图形时钟锁到较低频率（默认 210 MHz），在后台线程持续运行示例算子（默认大矩阵 GEMM），
   主线程在固定时间窗口内对功率采样并积分得到能耗（焦耳）。
2. 再将 GPU 锁到较高频率（默认 1410 MHz），重复测量。

依赖：
- NVIDIA 驱动；需要能执行 `nvidia-smi -lgc`（部分环境需 root / 管理员权限）。
- 推荐：`pip install torch pynvml`（或 `nvidia-ml-py`）。无 pynvml 时会用 nvidia-smi 轮询，精度较差。

能耗估算：对采样到的功率 P(t) 做离散积分（梯形法），单位焦耳 (J)。

注意：
- 不同 GPU 支持的锁频范围不同；若 210/1410 不被支持，nvidia-smi 会报错，请按 `nvidia-smi -q -d SUPPORTED_CLOCKS` 调整。
- 本脚本测的是「整卡在负载窗口内的近似能耗」，严格单算子隔离需配合 Nsight / CUPTI 等工具。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple


def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def run_nvidia_smi_lgc(min_mhz: int, max_mhz: int, gpu_index: int) -> None:
    """锁定图形时钟到 [min_mhz, max_mhz]（通常写成相同值）。"""
    nv = _which("nvidia-smi")
    if not nv:
        raise RuntimeError("未找到 nvidia-smi，请确认已安装 NVIDIA 驱动。")
    cmd = [nv, "-i", str(gpu_index), "-lgc", f"{min_mhz},{max_mhz}"]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def run_nvidia_smi_reset_clocks(gpu_index: int) -> None:
    """恢复 GPU 时钟策略（若驱动支持）。"""
    nv = _which("nvidia-smi")
    if not nv:
        return
    subprocess.run(
        [nv, "-i", str(gpu_index), "-rgc"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def try_import_nvml():
    try:
        import pynvml  # type: ignore

        return pynvml
    except Exception:
        return None


@dataclass
class PowerSample:
    t_s: float
    power_w: float


class PowerSampler:
    """封装 NVML 或 nvidia-smi 读功率，避免重复 init/shutdown。"""

    def __init__(self, gpu_index: int) -> None:
        self.gpu_index = gpu_index
        self._pynvml = try_import_nvml()
        self._handle = None
        if self._pynvml is not None:
            print(f"nvmlInit")
            self._pynvml.nvmlInit()
            self._handle = self._pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

    def read_w(self) -> float:
        if self._handle is not None:
            return float(self._pynvml.nvmlDeviceGetPowerUsage(self._handle)) / 1000.0
        print(f"nvidia-smi")
        nv = _which("nvidia-smi")
        out = subprocess.check_output(
            [
                nv,
                "-i",
                str(self.gpu_index),
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        return float(out.strip().splitlines()[0].strip())

    def close(self) -> None:
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass


def measure_energy_joules_under_load(
    sampler: PowerSampler,
    duration_s: float,
    sample_interval_s: float,
    idle_warmup_s: float,
    workload: Callable[[], None],
) -> Tuple[float, List[PowerSample]]:
    """
    后台线程持续执行 workload，主线程在 duration_s 内采样功率并梯形积分得到能耗 (J)。
    """
    stop = threading.Event()

    def _worker() -> None:
        while not stop.is_set():
            workload()

    th = threading.Thread(target=_worker, daemon=True)
    th.start()
    try:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < idle_warmup_s:
            _ = sampler.read_w()
            time.sleep(sample_interval_s)

        samples: List[PowerSample] = []
        energy_j = 0.0
        t_start = time.perf_counter()
        last_t = t_start
        last_p = sampler.read_w()

        while time.perf_counter() - t_start < duration_s:
            time.sleep(sample_interval_s)
            now = time.perf_counter()
            p = sampler.read_w()
            dt = now - last_t
            energy_j += 0.5 * (last_p + p) * dt
            samples.append(PowerSample(t_s=now, power_w=p))
            last_t, last_p = now, p
        return energy_j, samples
    finally:
        stop.set()
        th.join(timeout=5.0)


def default_torch_op(gpu_index: int, mat_n: int) -> Callable[[], None]:
    import torch

    torch.cuda.set_device(gpu_index)
    device = torch.device(f"cuda:{gpu_index}")
    a = torch.randn(mat_n, mat_n, device=device, dtype=torch.float16)
    b = torch.randn(mat_n, mat_n, device=device, dtype=torch.float16)

    def _run() -> None:
        _ = torch.matmul(a, b)
        torch.cuda.synchronize()

    return _run


def main() -> None:
    parser = argparse.ArgumentParser(description="调频后测量单算子近似能耗（功率积分）")
    parser.add_argument("--gpu", type=int, default=0, help="GPU 索引")
    parser.add_argument("--low-mhz", type=int, default=810, help="第一次锁定的图形时钟 (MHz)")
    parser.add_argument("--high-mhz", type=int, default=1410, help="第二次锁定的图形时钟 (MHz)")
    parser.add_argument(
        "--mat-n",
        type=int,
        default=8192,
        help="示例 GEMM 方阵维度，越大单次 kernel 越长、负载越稳",
    )
    parser.add_argument(
        "--measure-window-s",
        type=float,
        default=3.0,
        help="每次锁频后，负载下功率积分窗口长度（秒）",
    )
    parser.add_argument(
        "--sample-interval-ms",
        type=float,
        default=10.0,
        help="功率采样间隔（毫秒）",
    )
    parser.add_argument(
        "--warmup-ms",
        type=float,
        default=200.0,
        help="开始积分前预热采样（毫秒）",
    )
    parser.add_argument(
        "--no-reset-clocks",
        action="store_true",
        help="结束后不执行 nvidia-smi -rgc（默认会尝试恢复）",
    )
    args = parser.parse_args()

    if not _which("nvidia-smi"):
        print("错误：需要 nvidia-smi。", file=sys.stderr)
        sys.exit(1)

    try:
        import torch

        if not torch.cuda.is_available():
            print("错误：需要 PyTorch CUDA 版本且本机有可用的 CUDA GPU。", file=sys.stderr)
            sys.exit(1)
    except ImportError:
        print("错误：需要安装 torch：`pip install torch`", file=sys.stderr)
        sys.exit(1)

    sample_interval_s = max(0.001, args.sample_interval_ms / 1000.0)
    warmup_s = max(0.0, args.warmup_ms / 1000.0)

    op = default_torch_op(args.gpu, args.mat_n)
    sampler = PowerSampler(args.gpu)

    def one_stage(label: str, clock_mhz: int) -> float:
        print(f"\n=== {label}: 锁定图形时钟约 {clock_mhz} MHz ===")
        run_nvidia_smi_lgc(clock_mhz, clock_mhz, args.gpu)
        time.sleep(0.5)

        import torch

        torch.cuda.synchronize()
        energy_j, _ = measure_energy_joules_under_load(
            sampler=sampler,
            duration_s=args.measure_window_s,
            sample_interval_s=sample_interval_s,
            idle_warmup_s=warmup_s,
            workload=op,
        )
        torch.cuda.synchronize()
        print(f"近似能耗（负载窗口内功率积分）: {energy_j:.4f} J")
        return energy_j

    try:
        e_low = one_stage("阶段1（低频）", args.low_mhz)
        e_high = one_stage("阶段2（高频）", args.high_mhz)
        print("\n=== 汇总 ===")
        print(f"低频 ({args.low_mhz} MHz) 能耗 ≈ {e_low:.4f} J")
        print(f"高频 ({args.high_mhz} MHz) 能耗 ≈ {e_high:.4f} J")
        if e_low > 1e-9:
            print(f"高频 / 低频 能耗比 ≈ {e_high / e_low:.3f}")
    finally:
        sampler.close()
        if not args.no_reset_clocks:
            run_nvidia_smi_reset_clocks(args.gpu)


if __name__ == "__main__":
    main()
