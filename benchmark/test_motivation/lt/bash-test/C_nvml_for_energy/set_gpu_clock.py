#!/usr/bin/env python3
"""
通过 libnvml_energy.so（nvml_energy）调 GPU：

  - SM 锁频：--lock-sm（等价 nvidia-smi -lgc 思路）/ --reset-lock
  - 应用时钟：--mem + --graphics（-ac）/ --reset（-rac）

--query 使用 nvidia-smi。使用前: ./build.sh
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from nvml_energy import NvmlEnergy


def _query_clocks_smi(gpu: int) -> tuple[str, str, str]:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=clocks.current.sm,clocks.current.graphics,clocks.current.memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    parts = [x.strip() for x in out.split(",")]
    if len(parts) != 3:
        raise RuntimeError(f"unexpected nvidia-smi output: {out!r}")
    return parts[0], parts[1], parts[2]


def _resolve_gpus(gpu_arg: str) -> list[int]:
    """解析 -i 参数：'all' 返回全部 GPU 索引，否则返回单个索引。"""
    if gpu_arg.lower() == "all":
        with NvmlEnergy() as nv:
            return list(range(nv.device_count()))
    try:
        return [int(gpu_arg)]
    except ValueError:
        raise ValueError(f"无效的 GPU 参数: {gpu_arg!r}，需为整数或 'all'")


def main() -> int:
    p = argparse.ArgumentParser(
        description="NVML：SM 锁频 (-lgc 类) 或 应用时钟 (-ac/-rac)。"
    )
    p.add_argument(
        "-i", "--gpu", type=str, default="0", metavar="N",
        help="GPU 索引，或 'all' 表示所有 GPU",
    )
    p.add_argument(
        "--lock-sm",
        type=int,
        default=None,
        metavar="MHZ",
        help="锁 SM 为该 MHz（固定一档，min=max）",
    )
    p.add_argument(
        "--reset-lock",
        action="store_true",
        help="清除 SM 锁频（ResetGpuLockedClocks）",
    )
    p.add_argument("--mem", type=int, default=None, metavar="MHZ", help="-ac 显存 MHz")
    p.add_argument("--graphics", type=int, default=None, metavar="MHZ", help="-ac graphics MHz")
    p.add_argument("--reset", action="store_true", help="复位应用时钟（-rac）")
    p.add_argument("--query", action="store_true", help="nvidia-smi 查询 sm/graphics/mem")
    args = p.parse_args()

    mode_ac = args.mem is not None and args.graphics is not None
    mode_ac_partial = (args.mem is not None) ^ (args.graphics is not None)
    if mode_ac_partial:
        p.error("使用 -ac 需同时指定 --mem 与 --graphics")

    modes = [
        bool(args.query),
        bool(args.reset),
        bool(args.reset_lock),
        args.lock_sm is not None,
        mode_ac,
    ]
    if sum(modes) != 1:
        p.error(
            "请指定其一：--query | --reset | --reset-lock | --lock-sm ... | (--mem 与 --graphics)"
        )

    gpus = _resolve_gpus(args.gpu)

    if args.query:
        for g in gpus:
            sm, gr, mem = _query_clocks_smi(g)
            print(f"GPU {g} (nvidia-smi): sm={sm} graphics={gr} mem={mem} MHz")
        return 0

    if args.reset:
        with NvmlEnergy() as nv:
            for g in gpus:
                nv.reset_applications_clocks(g)
                print(f"GPU {g}: 已 -rac")
        return 0

    if args.reset_lock:
        with NvmlEnergy() as nv:
            for g in gpus:
                nv.reset_gpu_locked(g)
                print(f"GPU {g}: 已清除 SM 锁频")
        return 0

    if args.lock_sm is not None:
        mhz = args.lock_sm
        with NvmlEnergy() as nv:
            for g in gpus:
                nv.set_gpu_locked_mhz(g, mhz)
                sm, gr, mem = _query_clocks_smi(g)
                print(
                    f"GPU {g}: SM locked {mhz} MHz "
                    f"(nvidia-smi: sm={sm} graphics={gr} mem={mem} MHz)"
                )
        return 0

    with NvmlEnergy() as nv:
        for g in gpus:
            nv.set_applications_clocks_mhz(g, args.mem, args.graphics)
            sm, gr, mem = _query_clocks_smi(g)
            print(
                f"GPU {g}: 已 -ac {args.mem},{args.graphics} "
                f"(nvidia-smi: sm={sm} graphics={gr} mem={mem} MHz)"
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError, FileNotFoundError) as e:
        print(e, file=sys.stderr)
        raise SystemExit(1)
