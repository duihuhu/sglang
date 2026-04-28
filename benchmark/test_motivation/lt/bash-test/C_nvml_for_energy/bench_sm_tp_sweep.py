#!/usr/bin/env python3
"""
批量测试不同 SM 频率 × TP 组合下的 Prefill latency（多次测试取均值）。

使用 nvml_energy.py 锁 SM 频率，调用 sglang.bench_one_batch 获取 prefill latency。

用法:
    python bench_sm_tp_sweep.py
    python bench_sm_tp_sweep.py --sm-clocks 210 1410 --tp-sizes 1 2 4 8
    python bench_sm_tp_sweep.py --repeat 5 --output results.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nvml_energy import NvmlEnergy

MODEL_PATH = "/mnt/nvme1/models/Qwen/Qwen3-32B/"
BATCH_SIZE = 1
INPUT_LEN = 128
OUTPUT_LEN = 1

_PREFILL_RE = re.compile(
    r"Prefill\.\s+latency:\s+([\d.]+)\s*s,\s*throughput:\s+([\d.]+)\s*token/s"
)


def lock_sm_all(nv: NvmlEnergy, sm_mhz: int, gpu_count: int) -> None:
    for g in range(gpu_count):
        nv.set_gpu_locked_mhz(g, sm_mhz)
    print(f"  所有 GPU ({gpu_count} 张) SM 已锁定到 {sm_mhz} MHz")


def reset_sm_all(nv: NvmlEnergy, gpu_count: int) -> None:
    for g in range(gpu_count):
        nv.reset_gpu_locked(g)
    print(f"  所有 GPU ({gpu_count} 张) SM 锁频已清除")


def cuda_visible_devices(tp: int) -> str:
    return ",".join(str(i) for i in range(tp))


def run_bench(tp: int) -> tuple[float, float]:
    """运行 bench_one_batch，返回 (prefill_latency_s, prefill_throughput)。"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices(tp)

    cmd = [
        sys.executable, "-m", "sglang.bench_one_batch",
        "--model-path", MODEL_PATH,
        "--tp-size", str(tp),
        "--batch-size", str(BATCH_SIZE),
        "--input-len", str(INPUT_LEN),
        "--output-len", str(OUTPUT_LEN),
        "--disable-cuda-graph",
        "--disable-overlap-schedule",
        "--chunked-prefill-size", "-1",
        "--mem-fraction-static", "0.9",
    ]

    print(f"    CMD: CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} "
          f"{' '.join(cmd[1:])}")

    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)

    if proc.returncode != 0:
        print(f"    [ERROR] 返回码 {proc.returncode}", file=sys.stderr)
        if proc.stderr:
            for line in proc.stderr.strip().splitlines()[-10:]:
                print(f"      stderr: {line}", file=sys.stderr)
        return float("nan"), float("nan")

    output = proc.stdout + "\n" + proc.stderr
    matches = _PREFILL_RE.findall(output)
    if not matches:
        print("    [WARN] 未匹配到 Prefill latency，输出末尾:", file=sys.stderr)
        for line in output.strip().splitlines()[-10:]:
            print(f"      {line}", file=sys.stderr)
        return float("nan"), float("nan")

    lat_s, thr_s = matches[-1]
    return float(lat_s), float(thr_s)


def main() -> int:
    p = argparse.ArgumentParser(description="SM 频率 × TP Prefill latency 扫描")
    p.add_argument("--sm-clocks", type=int, nargs="+", default=[210, 1410],
                    metavar="MHZ", help="SM 频率列表 (MHz)")
    p.add_argument("--tp-sizes", type=int, nargs="+", default=[8],
                    metavar="N", help="TP 并行度列表")
    p.add_argument("--repeat", type=int, default=5,
                    help="每个 (tp, sm) 组合重复测试次数 (默认 3)")
    p.add_argument("--output", type=str, default=None, metavar="CSV",
                    help="结果保存到 CSV 文件（含每次明细）")
    p.add_argument("--settle-sec", type=float, default=2.0,
                    help="锁频后等待稳定秒数")
    args = p.parse_args()

    raw_results: list[dict] = []

    with NvmlEnergy() as nv:
        gpu_count = nv.device_count()
        print(f"检测到 {gpu_count} 张 GPU, repeat={args.repeat}\n")

        for tp in args.tp_sizes:
            if tp > gpu_count:
                print(f"  [SKIP] tp={tp} > GPU 数量 {gpu_count}")
                continue
            print(f"======== TP = {tp} ========")

            for sm_mhz in args.sm_clocks:
                print(f"  ---- SM = {sm_mhz} MHz ----")
                lock_sm_all(nv, sm_mhz, gpu_count)
                time.sleep(args.settle_sec)

                for i in range(args.repeat):
                    print(f"    [iter {i+1}/{args.repeat}]")
                    latency, throughput = run_bench(tp)
                    print(f"      Prefill latency = {latency:.5f} s, "
                          f"throughput = {throughput:.2f} token/s")
                    raw_results.append({
                        "tp": tp,
                        "sm_mhz": sm_mhz,
                        "iter": i + 1,
                        "prefill_latency_s": latency,
                        "prefill_throughput": throughput,
                    })
                print()

            reset_sm_all(nv, gpu_count)
            print()

    # --- 汇总均值 ---
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for r in raw_results:
        grouped[(r["tp"], r["sm_mhz"])].append(r)

    avg_results: list[dict] = []
    for (tp, sm_mhz), runs in grouped.items():
        lats = [r["prefill_latency_s"] for r in runs if not math.isnan(r["prefill_latency_s"])]
        thrs = [r["prefill_throughput"] for r in runs if not math.isnan(r["prefill_throughput"])]
        avg_results.append({
            "tp": tp,
            "sm_mhz": sm_mhz,
            "n": len(lats),
            "avg_latency_s": sum(lats) / len(lats) if lats else float("nan"),
            "avg_throughput": sum(thrs) / len(thrs) if thrs else float("nan"),
        })

    # --- 每次明细 ---
    print("=" * 72)
    print(f"{'TP':>4} {'SM(MHz)':>10} {'Iter':>5} "
          f"{'Prefill Latency(s)':>20} {'Throughput(tok/s)':>20}")
    print("-" * 72)
    for r in raw_results:
        print(f"{r['tp']:>4} {r['sm_mhz']:>10} {r['iter']:>5} "
              f"{r['prefill_latency_s']:>20.5f} {r['prefill_throughput']:>20.2f}")
    print("=" * 72)

    # --- 均值汇总 ---
    print(f"\n{'='*60}")
    print(f"{'TP':>4} {'SM(MHz)':>10} {'N':>4} "
          f"{'Avg Latency(s)':>18} {'Avg Thrpt(tok/s)':>18}")
    print(f"{'-'*60}")
    for r in avg_results:
        print(f"{r['tp']:>4} {r['sm_mhz']:>10} {r['n']:>4} "
              f"{r['avg_latency_s']:>18.5f} {r['avg_throughput']:>18.2f}")
    print(f"{'='*60}")

    if args.output:
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "tp", "sm_mhz", "iter",
                "prefill_latency_s", "prefill_throughput",
            ])
            w.writeheader()
            w.writerows(raw_results)
        print(f"\n明细结果已保存到 {args.output}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n中断，尝试清除 SM 锁频...")
        with NvmlEnergy() as nv:
            reset_sm_all(nv, nv.device_count())
        raise SystemExit(130)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        raise SystemExit(1)
