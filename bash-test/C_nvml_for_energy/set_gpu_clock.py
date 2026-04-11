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


def main() -> int:
    p = argparse.ArgumentParser(
        description="NVML：SM 锁频 (-lgc 类) 或 应用时钟 (-ac/-rac)。"
    )
    p.add_argument("-i", "--gpu", type=int, default=0, metavar="N", help="GPU 索引")
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

    if args.query:
        sm, gr, mem = _query_clocks_smi(args.gpu)
        print(f"GPU {args.gpu} (nvidia-smi): sm={sm} graphics={gr} mem={mem} MHz")
        return 0

    if args.reset:
        with NvmlEnergy() as nv:
            nv.reset_applications_clocks(args.gpu)
        print(f"GPU {args.gpu}: 已 -rac")
        return 0

    if args.reset_lock:
        with NvmlEnergy() as nv:
            nv.unlock_sm_clock(args.gpu)
        print(f"GPU {args.gpu}: 已清除 SM 锁频")
        return 0

    if args.lock_sm is not None:
        mhz = args.lock_sm
        with NvmlEnergy() as nv:
            nv.lock_sm_clock(args.gpu, mhz)
        sm, gr, mem = _query_clocks_smi(args.gpu)
        print(
            f"GPU {args.gpu}: SM locked {mhz} MHz "
            f"(nvidia-smi: sm={sm} graphics={gr} mem={mem} MHz)"
        )
        return 0

    with NvmlEnergy() as nv:
        nv.set_applications_clocks_mhz(args.gpu, args.mem, args.graphics)
    sm, gr, mem = _query_clocks_smi(args.gpu)
    print(
        f"GPU {args.gpu}: 已 -ac {args.mem},{args.graphics} "
        f"(nvidia-smi: sm={sm} graphics={gr} mem={mem} MHz)"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError, FileNotFoundError) as e:
        print(e, file=sys.stderr)
        raise SystemExit(1)
