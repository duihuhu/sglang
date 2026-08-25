#!/usr/bin/env python3
"""Refresh data/EP: F-EP forced routing on node3+node4 (balanced/middle/skewed)."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROFILE_ROOT = HERE.parent
REPO_ROOT = HERE.parents[3]
EP_RAW_ROOT = PROFILE_ROOT / "data" / "EP" / "raw_v2"
EP_OUT_ROOT = PROFILE_ROOT / "data" / "EP"
EP_EXPORTER = HERE / "export_ep_routing_pf_df.py"
EP_RAW_SUBDIR = "benchmark/moe-energy/profiling/data/EP/raw_v2"

from run_distributed_profile_matrix import (  # noqa: E402
    DEFAULT_NODES,
    Job,
    Node,
    Scheduler,
    filter_skip_existing,
    parse_nodes,
)

from export_ep_routing_pf_df import EP_ROUTING_MODES  # noqa: E402

EP_PHASES = ("prefill", "decode")
EP_SIZES = (2, 4, 8)
EP_LENGTHS = (64, 256, 512, 1024)
EP_BATCHES = (1, 8, 32, 1024)
EP_FREQS = (210, 690, 930, 1410)
EXPECTED_ROWS_PER_JOB = len(EP_LENGTHS) * len(EP_BATCHES) * len(EP_FREQS)


def ep_stem(phase: str, routing: str, size: int) -> str:
    return f"qwen3_{phase}_moe_ep_{routing}_ws{size}"


def make_ep_refresh_jobs(
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
                        label="F-EP",
                        component="F",
                        mode="moe_ep",
                        size=size,
                        stem=ep_stem(phase, routing, size),
                        routing_mode=routing,
                        expected_rows=EXPECTED_ROWS_PER_JOB,
                    )
                )
    return jobs


class EpRefreshScheduler(Scheduler):
    def benchmark_command(self, job: Job, output: str, port: int) -> list[str]:
        if job.label != "F-EP":
            return super().benchmark_command(job, output, port)
        script = f"benchmark/moe-energy/profiling/scripts/bench_{job.phase}_af.py"
        cmd = [
            "python3",
            script,
            "--model-path",
            self.args.model_path,
            "--component",
            "F",
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
            *[str(x) for x in EP_LENGTHS],
            "--batch-sizes",
            *[str(x) for x in EP_BATCHES],
            "--freqs",
            *[str(x) for x in EP_FREQS],
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
            str(EP_EXPORTER),
            "--raw-root",
            str(self.raw_root),
            "--output-dir",
            str(EP_OUT_ROOT),
        ]
        result = subprocess.run(command, text=True, capture_output=True)
        if result.stdout.strip():
            print(result.stdout.strip(), flush=True)
        if result.returncode:
            print(
                f"EP-EXPORT-WARNING rc={result.returncode}: {result.stderr.strip()}",
                file=sys.stderr,
                flush=True,
            )

    def job_in_progress(self, job: Job, exclude_node: str | None = None) -> bool:
        if job.label != "F-EP":
            return super().job_in_progress(job, exclude_node=exclude_node)
        needle = f"{job.stem}.jsonl"
        import shlex

        for state in self.states.values():
            if exclude_node and state.node.name == exclude_node:
                continue
            try:
                result = self.run_on_node(
                    state.node,
                    f"pgrep -af {shlex.quote(needle)} || true",
                    capture=True,
                )
            except subprocess.SubprocessError:
                continue
            for line in result.stdout.splitlines():
                if needle in line and "bench_" in line and "_af.py" in line:
                    return True
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=["run", "status"], default="run")
    parser.add_argument("--model-path", default="/models/Qwen3-30B-A3B")
    parser.add_argument("--container", default="moe-energy")
    parser.add_argument("--container-repo", default="/workspace/sglang-source/sglang")
    parser.add_argument("--host-repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--raw-root", type=Path, default=EP_RAW_ROOT)
    parser.add_argument("--container-output-root", default=EP_RAW_SUBDIR)
    parser.add_argument("--nodes", nargs="+", default=["node3", "node4"])
    parser.add_argument("--node-host", action="append", default=[], metavar="NAME=HOST")
    parser.add_argument(
        "--routing-modes",
        nargs="+",
        default=list(EP_ROUTING_MODES),
        choices=EP_ROUTING_MODES,
    )
    parser.add_argument("--phases", nargs="+", default=list(EP_PHASES), choices=EP_PHASES)
    parser.add_argument("--sizes", nargs="+", type=int, default=list(EP_SIZES))
    parser.add_argument(
        "--extra-args",
        default=(
            "--shape-token-limit 0 --max-total-tokens 600000 "
            "--warmup 10 --repeat 50 --max-running-requests 4096"
        ),
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-existing-min-rows", type=int, default=58)
    parser.add_argument("--retry-failed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-job-attempts", type=int, default=10)
    parser.add_argument("--base-nccl-port", type=int, default=29800)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--control-timeout", type=float, default=30.0)
    parser.add_argument("--terminate-timeout", type=float, default=10.0)
    parser.add_argument("--prefer-node1-progress", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--components", nargs="+", default=["F-EP"])
    parser.add_argument("--ep-a2a-backend", default="none")
    parser.add_argument("--ep-max-dispatch-tokens", type=int, default=131072)
    parser.add_argument("--no-export-compat", action="store_true")
    parser.add_argument("--shard-matrix", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.node_host:
        args.node_host = [
            f"node3={DEFAULT_NODES['node3']}",
            f"node4={DEFAULT_NODES['node4']}",
        ]
    nodes = parse_nodes(args)
    jobs = make_ep_refresh_jobs(tuple(args.phases), tuple(args.routing_modes), tuple(args.sizes))
    if args.skip_existing:
        jobs = filter_skip_existing(jobs, args.raw_root, None, args.skip_existing_min_rows)

    if args.command == "status":
        from run_distributed_profile_matrix import status

        return status(args, nodes, jobs)

    if args.dry_run:
        from run_distributed_profile_matrix import NodeState
        import shlex

        states = {node.name: NodeState(node) for node in nodes}
        helper = EpRefreshScheduler(args, nodes, [])
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
                            f"gpus={gpus} port={port}\n  {shlex.join(cmd)}"
                        )
                        break
            if not assigned:
                raise SystemExit("no selected node can fit the pending jobs")
            for _, state, gpus, _ in assigned:
                state.release(gpus)
            wave += 1
        print(f"planned {len(make_ep_refresh_jobs(tuple(args.phases), tuple(args.routing_modes), tuple(args.sizes)))} jobs across {wave - 1} packing wave(s)")
        return 0

    scheduler = EpRefreshScheduler(args, nodes, jobs)
    rc = scheduler.run()
    if not args.no_export_compat:
        scheduler.export_compat()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
