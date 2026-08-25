#!/usr/bin/env python3
"""Run kernel-only EP matrix locally (same span as data/EP/*.txt)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROFILE_ROOT = HERE.parent
REPO_ROOT = HERE.parents[3]
RAW_ROOT = PROFILE_ROOT / "data" / "kernel-EP" / "raw"
KERNEL_EP_ROOT = PROFILE_ROOT / "data" / "kernel-EP"
EXPORTER = HERE / "export_kernel_ep_pf_df.py"

ROUTING_MODES = ("balanced", "middle_rank0", "skewed_rank0")
PHASES = ("prefill", "decode")
SIZES = (2, 4, 8)
KERNEL_EP_LENGTHS = 4
KERNEL_EP_BATCHES = 4
KERNEL_EP_FREQS = 4
EXPECTED_ROWS_PER_JOB = KERNEL_EP_LENGTHS * KERNEL_EP_BATCHES * KERNEL_EP_FREQS
SKIP_ROW_THRESHOLD = max(40, int(EXPECTED_ROWS_PER_JOB * 0.9))
DEFAULT_MODEL = "/models/Qwen3-30B-A3B"
DEFAULT_VISIBLE_GPUS = "4,5,6,7"


def stem(phase: str, routing: str, size: int) -> str:
    return f"kernel_{phase}_{routing}_ws{size}"


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.open() if line.strip())


def parse_visible_gpus(text: str) -> list[int]:
    gpus = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not gpus:
        raise ValueError("empty --visible-gpus")
    return gpus


def run_job(
    phase: str,
    routing: str,
    size: int,
    visible_gpus: list[int],
    model_path: str,
    nccl_port: int,
    quick: bool,
) -> int:
    script = HERE / f"bench_kernel_{phase}_af.py"
    output = RAW_ROOT / f"{stem(phase, routing, size)}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    if len(visible_gpus) < size:
        print(
            f"SKIP {output.name}: need {size} GPUs, pool has {len(visible_gpus)} "
            f"({visible_gpus})",
            flush=True,
        )
        return 0
    cmd = [
        sys.executable,
        str(script),
        "--model-path",
        model_path,
        "--parallel-mode",
        "moe_ep",
        "--tp-size",
        str(size),
        "--ep-size",
        str(size),
        "--moe-runner-backend",
        "triton",
        "--moe-a2a-backend",
        "none",
        "--cuda-graph-backend-decode",
        "disabled",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--nccl-port",
        str(nccl_port),
        "--forced-routing",
        routing,
        "--output",
        str(output),
        "--local-world-size",
        str(size),
    ]
    if quick:
        cmd.append("--quick")
    # Map logical ranks 0..size-1 onto the tail of the visible GPU pool.
    slot = visible_gpus[-size:]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in slot)
    env["PYTHONPATH"] = str(REPO_ROOT / "python")
    env.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    print(
        f"RUN physical_gpus={slot} port={nccl_port} -> {output.name}",
        flush=True,
    )
    result = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT))
    return result.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument(
        "--visible-gpus",
        default=DEFAULT_VISIBLE_GPUS,
        help="physical GPU indices to use (comma-separated); jobs take the last N of this pool",
    )
    parser.add_argument("--base-nccl-port", type=int, default=29500)
    parser.add_argument(
        "--routing-modes",
        nargs="+",
        default=list(ROUTING_MODES),
        choices=ROUTING_MODES,
    )
    parser.add_argument("--phases", nargs="+", default=list(PHASES), choices=PHASES)
    parser.add_argument("--sizes", nargs="+", type=int, default=list(SIZES))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="log failures and keep going instead of aborting the matrix",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    visible_gpus = parse_visible_gpus(args.visible_gpus)
    failures: list[str] = []

    if not args.export_only:
        pending: list[tuple[str, str, int]] = []
        for phase in args.phases:
            for routing in args.routing_modes:
                for size in args.sizes:
                    pending.append((phase, routing, size))

        port = args.base_nccl_port
        for phase, routing, size in pending:
            output = RAW_ROOT / f"{stem(phase, routing, size)}.jsonl"
            rows = count_rows(output)
            if rows >= SKIP_ROW_THRESHOLD:
                print(f"SKIP-EXISTING {output.name} rows={rows}", flush=True)
                continue
            if size > len(visible_gpus):
                print(
                    f"SKIP-CAPACITY {output.name}: ws={size} > pool={len(visible_gpus)}",
                    flush=True,
                )
                continue
            port += 1
            rc = run_job(
                phase, routing, size, visible_gpus, args.model_path, port, args.quick
            )
            if rc != 0:
                msg = f"FAILED {phase} {routing} ws={size} rc={rc}"
                print(msg, flush=True)
                failures.append(msg)
                if not args.continue_on_error:
                    return rc

    export = subprocess.run(
        [sys.executable, str(EXPORTER)],
        cwd=str(REPO_ROOT),
    )
    if export.returncode != 0:
        return export.returncode
    print(f"Done. TSV under {KERNEL_EP_ROOT}", flush=True)
    if failures:
        print(f"Failures ({len(failures)}):", flush=True)
        for line in failures:
            print(f"  {line}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
