#!/usr/bin/env python3
"""
批量跑 tp=1,2,4,8 下的 bench_prefill_af.py，合并结果。

用法:
    python run_tp_sweep.py
    python run_tp_sweep.py --tp-sizes 1 2 4 8 --freqs 210 1410
    python run_tp_sweep.py --model-path /mnt/nvme1/models/Qwen/Qwen3-32B/
    python run_tp_sweep.py --quick
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_SCRIPT = SCRIPT_DIR / "bench_prefill_af.py"
DEFAULT_MODEL = "/mnt/nvme1/models/Qwen/Qwen3-32B/"


def run_one_tp(tp: int, model_path: str, extra_args: list[str],
               output_file: str) -> bool:
    cmd = [
        sys.executable, str(BENCH_SCRIPT),
        "--model-path", model_path,
        "--tp-size", str(tp),
        "--output", output_file,
        *extra_args,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(tp))

    print(f"\n{'='*60}")
    print(f"  Prefill A/F  tp={tp}  CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
    print(f"  Output: {output_file}")
    print(f"{'='*60}")
    print(f"  CMD: {' '.join(cmd)}\n")

    ret = subprocess.run(cmd, env=env, cwd=str(SCRIPT_DIR))
    if ret.returncode != 0:
        print(f"  [ERROR] tp={tp} 返回码 {ret.returncode}", file=sys.stderr)
        return False
    return True


def main():
    p = argparse.ArgumentParser(
        description="tp 扫描: bench_prefill_af.py × tp=1,2,4,8")
    p.add_argument("--model-path", type=str, default=DEFAULT_MODEL)
    p.add_argument("--tp-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--freqs", type=int, nargs="+", default=[210, 450, 690, 930, 1170, 1410],
                    help="SM 频率列表 (MHz)，不指定则用默认值 [210, 450, 690, 930, 1170, 1410]")
    p.add_argument("--input-lens", type=int, nargs="+", default=None)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=None)
    p.add_argument("--repeat", type=int, default=None)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--output", type=str, default="prefill_data_v1.txt",
                    help="输出文件名（所有 tp 共用同一个文件）")
    args = p.parse_args()

    extra_args: list[str] = []
    if args.freqs:
        extra_args += ["--freqs"] + [str(f) for f in args.freqs]
    if args.input_lens:
        extra_args += ["--input-lens"] + [str(x) for x in args.input_lens]
    if args.batch_sizes:
        extra_args += ["--batch-sizes"] + [str(x) for x in args.batch_sizes]
    if args.repeat is not None:
        extra_args += ["--repeat", str(args.repeat)]
    if args.warmup is not None:
        extra_args += ["--warmup", str(args.warmup)]
    if args.quick:
        extra_args += ["--quick"]
    out_path = SCRIPT_DIR / args.output
    for i, tp in enumerate(args.tp_sizes):
        tp_extra = extra_args + (["--append"] if i > 0 else [])
        ok = run_one_tp(tp, args.model_path, tp_extra, args.output)
        if not ok:
            print(f"  tp={tp} 失败，继续下一个 tp...\n")

    print(f"\n{'='*60}")
    print(f"  所有 tp 测试完成!")
    print(f"  结果: {out_path}")
    print(f"{'='*60}\n")

    if out_path.exists():
        with open(out_path) as f:
            lines = f.read().strip().splitlines()
        if len(lines) > 1:
            print(lines[0])
            print("-" * len(lines[0]))
            for line in lines[1:]:
                print(line)


if __name__ == "__main__":
    main()
