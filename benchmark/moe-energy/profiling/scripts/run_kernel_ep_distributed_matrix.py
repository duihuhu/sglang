#!/usr/bin/env python3
"""Schedule kernel-only EP profiling across nodes (parallel GPU packing)."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROFILE_ROOT = HERE.parent
REPO_ROOT = HERE.parents[3]
KERNEL_RAW_ROOT = PROFILE_ROOT / "data" / "kernel-EP" / "raw"
KERNEL_EP_ROOT = PROFILE_ROOT / "data" / "kernel-EP"
KERNEL_EXPORTER = HERE / "export_kernel_ep_pf_df.py"
KERNEL_RAW_SUBDIR = "benchmark/moe-energy/profiling/data/kernel-EP/raw"

# Import scheduler primitives from the matrix runner (same docker/ssh model).
from run_distributed_profile_matrix import (  # noqa: E402
    DEFAULT_NODES,
    Job,
    Node,
    Scheduler,
    filter_skip_existing,
    parse_nodes,
)

# Host scheduler has no torch; mirror the phase-specific matrix constants.
KERNEL_ROUTING_MODES = ("balanced", "middle_rank0", "skewed_rank0")
KERNEL_PHASES = ("prefill", "decode")
KERNEL_SIZES = (2, 4, 8)
KERNEL_FREQS = (210, 450, 690, 930, 1170, 1410)
KERNEL_PREFILL_LENGTHS = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
KERNEL_PREFILL_BATCHES = (1,)
KERNEL_DECODE_LENGTHS = (64,)
KERNEL_DECODE_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


def _kernel_phase_axes(phase: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if phase == "prefill":
        return KERNEL_PREFILL_LENGTHS, KERNEL_PREFILL_BATCHES
    return KERNEL_DECODE_LENGTHS, KERNEL_DECODE_BATCHES


def _kernel_shape_excluded(phase: str, world_size: int, length: int, batch: int) -> bool:
    return phase == "prefill" and world_size == 2 and length >= 8192


def _kernel_expected_rows(phase: str, world_size: int) -> int:
    lengths, batches = _kernel_phase_axes(phase)
    shapes = sum(
        not _kernel_shape_excluded(phase, world_size, length, batch)
        for length in lengths
        for batch in batches
    )
    return shapes * len(KERNEL_FREQS)

def kernel_stem(phase: str, routing: str, size: int) -> str:
    return f"kernel_{phase}_{routing}_ws{size}"


def make_kernel_jobs(
    phases: tuple[str, ...],
    routing_modes: tuple[str, ...],
    sizes: tuple[int, ...],
) -> list[Job]:
    jobs: list[Job] = []
    for phase in phases:
        for routing in routing_modes:
            for size in sizes:
                jobs.append(
                    Job(
                        phase=phase,
                        label="K-EP",
                        component="K",
                        mode="moe_ep",
                        size=size,
                        stem=kernel_stem(phase, routing, size),
                        routing_mode=routing,
                        expected_rows=_kernel_expected_rows(phase, size),
                    )
                )
    return jobs


class KernelScheduler(Scheduler):
    def benchmark_command(self, job: Job, output: str, port: int) -> list[str]:
        if job.label != "K-EP":
            return super().benchmark_command(job, output, port)
        script = f"benchmark/moe-energy/profiling/scripts/bench_kernel_{job.phase}_af.py"
        cmd = [
            "python3",
            script,
            "--model-path",
            self.args.model_path,
            "--parallel-mode",
            "moe_ep",
            "--tp-size",
            str(job.size),
            "--ep-size",
            str(job.size),
            "--moe-runner-backend",
            "triton",
            "--moe-a2a-backend",
            "none",
            "--cuda-graph-backend-decode",
            "disabled",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--nccl-port",
            str(port),
            "--forced-routing",
            job.routing_mode or "balanced",
            "--output",
            output,
            "--local-world-size",
            str(job.size),
            "--lengths",
            *[str(x) for x in _kernel_phase_axes(job.phase)[0]],
            "--batch-sizes",
            *[str(x) for x in _kernel_phase_axes(job.phase)[1]],
            "--freqs",
            *[str(x) for x in KERNEL_FREQS],
        ]
        if self.args.quick:
            cmd.append("--quick")
        if self.args.extra_args.strip():
            import shlex

            cmd.extend(shlex.split(self.args.extra_args))
        return cmd

    def export_compat(self) -> None:
        command = [
            sys.executable,
            str(KERNEL_EXPORTER),
            "--raw-root",
            str(self.raw_root),
            "--output-dir",
            str(KERNEL_EP_ROOT),
        ]
        result = subprocess.run(command, text=True, capture_output=True)
        if result.stdout.strip():
            print(result.stdout.strip(), flush=True)
        if result.returncode:
            print(
                f"KERNEL-EXPORT-WARNING rc={result.returncode}: {result.stderr.strip()}",
                file=sys.stderr,
                flush=True,
            )

    def job_in_progress(self, job: Job, exclude_node: str | None = None) -> bool:
        if job.label != "K-EP":
            return super().job_in_progress(job, exclude_node=exclude_node)
        needle = f"{job.stem}.jsonl"
        for state in self.states.values():
            if exclude_node and state.node.name == exclude_node:
                continue
            import shlex

            try:
                result = self.run_on_node(
                    state.node,
                    f"pgrep -af {shlex.quote(needle)} || true",
                    capture=True,
                )
            except subprocess.SubprocessError:
                continue
            for line in result.stdout.splitlines():
                if needle in line and "bench_kernel" in line:
                    return True
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=["run", "status"], default="run")
    parser.add_argument("--model-path", default="/models/Qwen3-30B-A3B")
    parser.add_argument("--container", default="moe-energy")
    parser.add_argument("--container-repo", default="/workspace/sglang-source/sglang")
    parser.add_argument("--host-repo", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=KERNEL_RAW_ROOT,
        help="host manifest + rsync target for per-node JSONL",
    )
    parser.add_argument(
        "--container-output-root",
        default=KERNEL_RAW_SUBDIR,
    )
    parser.add_argument("--nodes", nargs="+", default=["node3", "node4"])
    parser.add_argument("--node-host", action="append", default=[], metavar="NAME=HOST")
    parser.add_argument(
        "--routing-modes",
        nargs="+",
        default=list(KERNEL_ROUTING_MODES),
        choices=KERNEL_ROUTING_MODES,
    )
    parser.add_argument("--phases", nargs="+", choices=list(KERNEL_PHASES), default=list(KERNEL_PHASES))
    parser.add_argument("--sizes", nargs="+", type=int, choices=list(KERNEL_SIZES), default=list(KERNEL_SIZES))
    parser.add_argument(
        "--extra-args",
        default="--shape-token-limit 0 --max-total-tokens 600000 --warmup 10 --repeat 50",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-existing-min-rows", type=int, default=58)
    parser.add_argument("--retry-failed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-job-attempts", type=int, default=10)
    parser.add_argument("--base-nccl-port", type=int, default=29700)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--control-timeout", type=float, default=30.0)
    parser.add_argument("--terminate-timeout", type=float, default=10.0)
    parser.add_argument("--prefer-node1-progress", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action="store_true")
    # Unused but required by Scheduler base
    parser.add_argument("--components", nargs="+", default=["K-EP"])
    parser.add_argument("--ep-a2a-backend", default="none")
    parser.add_argument("--ep-max-dispatch-tokens", type=int, default=131072)
    parser.add_argument("--no-export-compat", action="store_true")
    parser.add_argument("--shard-matrix", action="store_true")
    return parser.parse_args()


def kernel_dry_run(args: argparse.Namespace, nodes: list[Node], jobs: list[Job]) -> None:
    import shlex
    from run_distributed_profile_matrix import NodeState

    states = {node.name: NodeState(node) for node in nodes}
    helper = KernelScheduler(args, nodes, [])
    pending = sorted(jobs, key=lambda j: (-j.size, j.phase, j.mode, j.routing_mode or ""))
    wave = 1
    while pending:
        assigned = []
        for job in list(pending):
            candidates = [states[job.pinned_node]] if job.pinned_node else list(states.values())
            candidates.sort(key=lambda s: (sum(s.used), nodes.index(s.node)))
            for state in candidates:
                gpus = state.allocate(job.size)
                if gpus is not None:
                    output_root = Path(args.container_output_root)
                    if not output_root.is_absolute():
                        output_root = Path(args.container_repo) / output_root
                    output = str(output_root / state.node.name / f"{job.stem}.jsonl")
                    port = args.base_nccl_port + gpus[0]
                    cmd = helper.benchmark_command(job, output, port)
                    assigned.append((job, state, gpus, cmd))
                    pending.remove(job)
                    print(
                        f"DRY-RUN wave={wave} {job.job_id} node={state.node.name} "
                        f"host={state.node.host} gpus={gpus} port={port}\n  "
                        f"{shlex.join(cmd)}"
                    )
                    break
        if not assigned:
            raise SystemExit("no selected node can fit the pending jobs")
        for _, state, gpus, _ in assigned:
            state.release(gpus)
        wave += 1
    print(f"planned {len(jobs)} jobs across {wave - 1} packing wave(s)")


def main() -> int:
    args = parse_args()
    if not args.node_host:
        args.node_host = [
            f"node3={DEFAULT_NODES['node3']}",
            f"node4={DEFAULT_NODES['node4']}",
        ]
    nodes = parse_nodes(args)
    jobs = make_kernel_jobs(tuple(args.phases), tuple(args.routing_modes), tuple(args.sizes))
    if args.skip_existing:
        jobs = filter_skip_existing(
            jobs, args.raw_root, None, args.skip_existing_min_rows
        )
    if args.no_export:
        args.no_export_compat = True

    if args.command == "status":
        from run_distributed_profile_matrix import status

        return status(args, nodes, jobs)

    if args.dry_run:
        kernel_dry_run(args, nodes, jobs)
        return 0

    scheduler = KernelScheduler(args, nodes, jobs)
    rc = scheduler.run()
    if not args.no_export_compat:
        scheduler.export_compat()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
