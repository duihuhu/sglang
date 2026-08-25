#!/usr/bin/env python3
"""Dynamically schedule the Qwen3 profiling matrix across 8-GPU nodes."""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
PROFILE_ROOT = HERE.parent
REPO_ROOT = HERE.parents[3]
HOST_DATA_ROOT = PROFILE_ROOT / "data"
RAW_V2_ROOT = HOST_DATA_ROOT / "raw_v2"
COMPAT_DIR = HOST_DATA_ROOT / "compat"
EXPORTER = HERE / "export_compat_matrix.py"
DEFAULT_REPO = "/workspace/sglang-source/sglang"
DEFAULT_NODES = {
    "node1": "local",
    "node2": "10.252.129.35",
    "node3": "10.252.129.34",
    "node4": "10.252.129.33",
}
EP_ROUTING_MODES = ("balanced", "middle_rank0", "skewed_rank0")
COMPONENTS = {"A": ("A", "attn_tp"), "F-TP": ("F", "moe_tp"), "F-EP": ("F", "moe_ep")}
DEFAULT_FREQ_COUNT = 6
# Import matrix defaults from profile_utils without pulling torch/sglang at import time.
PREFILL_LENGTHS = [64, 128, 256, 512, 1024, 2048, 4096]
PREFILL_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
PREFILL_BATCHES_CAP = [1, 2, 4, 8, 16, 32, 64, 128, 256]
DECODE_LENGTHS = [64, 128, 256, 512, 1024, 2048, 4096]
DECODE_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
DECODE_BATCHES_CAP = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def count_rows(path: Path) -> int:
    try:
        with path.open() as f:
            return sum(bool(line.strip()) for line in f)
    except FileNotFoundError:
        return 0


def job_existing_rows(job: Job, raw_root: Path, nodes: list[Node] | None = None) -> int:
    if nodes:
        paths = [raw_root / node.name / f"{job.stem}.jsonl" for node in nodes]
    else:
        paths = sorted(raw_root.glob(f"node*/{job.stem}.jsonl"))
    if not paths:
        return 0
    return max(count_rows(path) for path in paths)


def parse_cuda_visible_devices(line: str) -> list[int]:
    match = re.search(r"CUDA_VISIBLE_DEVICES=([0-9,]+)", line)
    if not match:
        return []
    return [int(item) for item in match.group(1).split(",") if item]


def filter_skip_existing(
    jobs: list[Job], raw_root: Path, nodes: list[Node] | None, min_rows: int,
) -> list[Job]:
    kept = []
    for job in jobs:
        threshold = job.skip_threshold(min_rows)
        rows = job_existing_rows(job, raw_root, nodes)
        if rows >= threshold:
            print(
                f"SKIP-EXISTING {job.job_id} rows={rows} threshold={threshold}",
                flush=True,
            )
            continue
        kept.append(job)
    return kept


def matrix_shard_specs(phase: str) -> list[tuple[str, tuple[int, ...], tuple[int, ...]]]:
    lengths = PREFILL_LENGTHS if phase == "prefill" else DECODE_LENGTHS
    batches = PREFILL_BATCHES if phase == "prefill" else DECODE_BATCHES
    batches_cap = PREFILL_BATCHES_CAP if phase == "prefill" else DECODE_BATCHES_CAP
    return [
        ("l64-256", tuple(lengths[:3]), tuple(batches)),
        ("l512-1024", tuple(lengths[3:5]), tuple(batches)),
        ("l2048", (lengths[5],), tuple(batches_cap)),
        ("l4096", (lengths[6],), tuple(batches_cap)),
    ]


@dataclass(frozen=True)
class Node:
    name: str
    host: str

    @property
    def local(self) -> bool:
        return self.host == "local"


@dataclass(frozen=True)
class Job:
    phase: str
    label: str
    component: str
    mode: str
    size: int
    stem: str
    routing_mode: str | None = None
    pinned_node: str | None = None
    shard_label: str | None = None
    lengths: tuple[int, ...] | None = None
    batch_sizes: tuple[int, ...] | None = None
    expected_rows: int | None = None

    @property
    def job_id(self) -> str:
        base = f"{self.phase}-{self.mode}"
        if self.routing_mode:
            base = f"{base}-{self.routing_mode}"
        base = f"{base}-ws{self.size}"
        if self.shard_label:
            return f"{base}-{self.shard_label}"
        return base

    def skip_threshold(self, default_min: int) -> int:
        if self.expected_rows is None:
            return default_min
        return max(40, int(self.expected_rows * 0.98))


@dataclass
class Running:
    job: Job
    node: Node
    gpus: list[int]
    port: int
    tag: str
    process: subprocess.Popen
    started_at: str
    rows_before: int
    output: str
    log: str
    command: list[str]


@dataclass
class NodeState:
    node: Node
    used: list[bool] = field(default_factory=lambda: [False] * 8)

    def allocate(self, size: int) -> list[int] | None:
        for start in range(0, len(self.used) - size + 1):
            if not any(self.used[start : start + size]):
                for gpu in range(start, start + size):
                    self.used[gpu] = True
                return list(range(start, start + size))
        return None

    def release(self, gpus: list[int]) -> None:
        for gpu in gpus:
            self.used[gpu] = False


class Scheduler:
    def __init__(self, args: argparse.Namespace, nodes: list[Node], jobs: list[Job]):
        self.args = args
        self.nodes = nodes
        self.states = {node.name: NodeState(node) for node in nodes}
        self.pending = sorted(
            jobs,
            key=lambda j: (
                -j.size,
                j.phase,
                j.mode,
                j.routing_mode or "",
                j.shard_label or "",
            ),
        )
        self.running: list[Running] = []
        self.run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        self.raw_root = args.raw_root
        self.container_output_root = PurePosixPath(args.container_output_root)
        if not self.container_output_root.is_absolute():
            self.container_output_root = PurePosixPath(args.container_repo) / self.container_output_root
        self.manifest = self.raw_root / "manifest.jsonl"
        self.interrupted = False
        self.job_attempts: dict[str, int] = {}

    def data_dir(self, node: Node) -> PurePosixPath:
        return self.container_output_root / node.name

    def host_node1_output(self, job: Job) -> Path:
        return self.raw_root / "node1" / f"{job.stem}.jsonl"

    def remote_argv(self, node: Node, argv: list[str]) -> list[str]:
        docker = ["docker", "exec", self.args.container, *argv]
        if node.local:
            return docker
        return ["ssh", node.host, shlex.join(docker)]

    def run_on_node(self, node: Node, script: str, *, capture: bool = False):
        argv = self.remote_argv(node, ["bash", "-lc", script])
        return subprocess.run(argv, text=True, capture_output=capture, timeout=self.args.control_timeout)

    def remote_rows(self, node: Node, output: str) -> int:
        if node.name == "node1":
            return count_rows(self.raw_root / node.name / Path(output).name)
        code = "from pathlib import Path; p=Path(" + repr(output) + "); print(sum(bool(x.strip()) for x in p.open()) if p.exists() else 0)"
        try:
            result = self.run_on_node(node, f"python3 -c {shlex.quote(code)}", capture=True)
            return int(result.stdout.strip()) if result.returncode == 0 else -1
        except (subprocess.SubprocessError, ValueError):
            return -1

    def append_manifest(self, record: dict) -> None:
        self.manifest.parent.mkdir(parents=True, exist_ok=True)
        with self.manifest.open("a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def paths(self, job: Job, node: Node) -> tuple[str, str]:
        data = self.data_dir(node)
        return str(data / f"{job.stem}.jsonl"), str(data / "logs" / f"{job.stem}.log")

    def benchmark_command(self, job: Job, output: str, port: int) -> list[str]:
        ep = job.size if job.mode == "moe_ep" else 1
        ep_a2a_backend = self.args.ep_a2a_backend if job.mode == "moe_ep" else "none"
        cmd = [
            "python3", f"benchmark/moe-energy/profiling/scripts/bench_{job.phase}_af.py",
            "--model-path", self.args.model_path,
            "--component", job.component,
            "--parallel-mode", job.mode,
            "--tp-size", str(job.size),
            "--ep-size", str(ep),
            "--moe-runner-backend", "triton",
            "--moe-a2a-backend", ep_a2a_backend,
            "--cuda-graph-backend-decode", "disabled",
            "--cuda-graph-backend-prefill", "disabled",
            "--nccl-port", str(port),
            "--output", output,
        ]
        if ep_a2a_backend in ("ampere_ep", "deepep"):
            cmd.extend(["--enable-dp-attention", "--dp-size", str(job.size)])
            if ep_a2a_backend == "deepep":
                cmd.extend(
                    [
                        "--deepep-mode",
                        "low_latency",
                        "--deepep-dispatcher-output-dtype",
                        "bf16",
                    ]
                )
        if job.mode == "moe_ep" and job.routing_mode:
            cmd.extend(["--forced-routing", job.routing_mode])
        if job.lengths:
            cmd.extend(["--lengths", *[str(item) for item in job.lengths]])
        if job.batch_sizes:
            cmd.extend(["--batch-sizes", *[str(item) for item in job.batch_sizes]])
        if self.args.quick:
            cmd.append("--quick")
        cmd.extend(shlex.split(self.args.extra_args))
        return cmd

    def launch(self, job: Job, node: Node, gpus: list[int]) -> Running:
        output, log = self.paths(job, node)
        port = self.args.base_nccl_port + gpus[0]
        tag = f"sgl-profile-{self.run_id}-{job.job_id}"
        cmd = self.benchmark_command(job, output, port)
        quoted_cmd = shlex.join(cmd)
        launch_env = [
            f"CUDA_VISIBLE_DEVICES={','.join(map(str, gpus))}",
            "FLASHINFER_DISABLE_VERSION_CHECK=1",
            f"SGLANG_PROFILE_TAG={tag}",
        ]
        if (
            job.mode == "moe_ep"
            and self.args.ep_a2a_backend == "ampere_ep"
        ):
            launch_env.append(
                "SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK="
                f"{self.args.ep_max_dispatch_tokens}"
            )
        quoted_env = " ".join(shlex.quote(item) for item in launch_env)
        wrapper = (
            "trap 'kill -TERM -- -$child 2>/dev/null; wait $child 2>/dev/null' TERM INT HUP; "
            f"setsid env {quoted_env} "
            f"{quoted_cmd} >> {shlex.quote(log)} 2>&1 & child=$!; wait $child"
        )
        setup = (
            f"cd {shlex.quote(self.args.container_repo)} && "
            f"mkdir -p {shlex.quote(str(PurePosixPath(output).parent))} {shlex.quote(str(PurePosixPath(log).parent))} && "
            f"bash -lc {shlex.quote(wrapper)} {shlex.quote(tag)}"
        )
        argv = self.remote_argv(node, ["bash", "-lc", setup])
        before = self.remote_rows(node, output)
        started = utcnow()
        process = subprocess.Popen(argv, start_new_session=True)
        item = Running(job, node, gpus, port, tag, process, started, before, output, log, argv)
        self.append_manifest({
            "event": "start", "status": "running", "run_id": self.run_id, "attempt_id": tag,
            "started_at": started, "node": node.name, "host": node.host, "gpus": gpus,
            "nccl_port": port, "phase": job.phase, "component": job.label,
            "parallel_mode": job.mode, "world_size": job.size, "output": output,
            "log": log, "rows_before": before, "command": argv,
        })
        return item

    def eligible_states(self, job: Job) -> list[NodeState]:
        states = list(self.states.values())
        if job.pinned_node:
            states = [s for s in states if s.node.name == job.pinned_node]
        return sorted(states, key=lambda s: (sum(s.used), self.nodes.index(s.node)))

    def sync_gpu_occupancy(self) -> None:
        for state in self.states.values():
            occupied = {
                gpu
                for item in self.running
                if item.node.name == state.node.name
                for gpu in item.gpus
            }
            try:
                result = self.run_on_node(
                    state.node,
                    "pgrep -af 'SGLANG_PROFILE_TAG=sgl-profile-' || true",
                    capture=True,
                )
                for line in result.stdout.splitlines():
                    if "pgrep -af" in line:
                        continue
                    occupied.update(parse_cuda_visible_devices(line))
            except subprocess.SubprocessError:
                pass
            state.used = [gpu in occupied for gpu in range(len(state.used))]

    def queue_retry(self, job: Job, reason: str) -> None:
        attempts = self.job_attempts.get(job.job_id, 0) + 1
        if attempts > self.args.max_job_attempts:
            print(
                f"RETRY-EXHAUSTED {job.job_id} attempts={attempts} reason={reason}",
                flush=True,
            )
            return
        self.job_attempts[job.job_id] = attempts
        self.pending.append(job)
        self.pending.sort(
            key=lambda j: (
                -j.size,
                j.phase,
                j.mode,
                j.routing_mode or "",
                j.shard_label or "",
            )
        )
        print(f"RETRY-QUEUED {job.job_id} attempt={attempts} reason={reason}", flush=True)

    def job_in_progress(self, job: Job, exclude_node: str | None = None) -> bool:
        needle = f"{job.stem}.jsonl"
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
                if needle in line and "bench_" in line:
                    return True
        return False

    def job_is_complete(self, job: Job) -> bool:
        if not self.args.skip_existing:
            return False
        rows = job_existing_rows(job, self.raw_root, None)
        return rows >= job.skip_threshold(self.args.skip_existing_min_rows)

    def fill(self) -> bool:
        self.sync_gpu_occupancy()
        launched = False
        for job in list(self.pending):
            if self.job_is_complete(job):
                print(
                    f"SKIP-COMPLETE {job.job_id} rows>={job.skip_threshold(self.args.skip_existing_min_rows)}",
                    flush=True,
                )
                self.pending.remove(job)
                continue
            for state in self.eligible_states(job):
                if self.job_in_progress(job, exclude_node=state.node.name):
                    continue
                gpus = state.allocate(job.size)
                if gpus is None:
                    continue
                try:
                    running = self.launch(job, state.node, gpus)
                except Exception as exc:
                    state.release(gpus)
                    print(f"LAUNCH-ERROR {job.job_id} node={state.node.name}: {exc}", flush=True)
                    self.append_manifest({
                        "event": "done", "status": "launch-failed", "run_id": self.run_id,
                        "attempt_id": f"{self.run_id}-{job.job_id}", "ended_at": utcnow(),
                        "node": state.node.name, "phase": job.phase, "parallel_mode": job.mode,
                        "world_size": job.size, "error": repr(exc),
                    })
                    self.queue_retry(job, f"launch-failed: {exc!r}")
                    self.pending.remove(job)
                    launched = True
                    break
                self.running.append(running)
                self.pending.remove(job)
                launched = True
                print(
                    f"START {job.job_id} node={state.node.name} gpus={gpus} port={running.port} "
                    f"pending={len(self.pending)} running={len(self.running)}",
                    flush=True,
                )
                break
        return launched

    def host_path_for_container_output(self, output: str) -> Path:
        try:
            relative = PurePosixPath(output).relative_to(
                PurePosixPath(self.args.container_repo)
            )
        except ValueError as exc:
            raise ValueError(
                f"container output {output!r} is outside --container-repo "
                f"{self.args.container_repo!r}"
            ) from exc
        return self.args.host_repo / Path(*relative.parts)

    def fetch_remote_output(self, item: Running) -> None:
        if item.node.local:
            return
        local_dir = self.raw_root / item.node.name
        local_dir.mkdir(parents=True, exist_ok=True)
        remote_host_path = self.host_path_for_container_output(item.output)
        result = subprocess.run(
            ["rsync", "-a", f"{item.node.host}:{remote_host_path}", str(local_dir / Path(item.output).name)],
            text=True,
            capture_output=True,
        )
        if result.returncode:
            print(
                f"FETCH-WARNING node={item.node.name} output={item.output}: {result.stderr.strip()}",
                file=sys.stderr,
                flush=True,
            )

    def export_compat(self) -> None:
        command = [sys.executable, str(EXPORTER), "--raw-root", str(self.raw_root), "--output-dir", str(COMPAT_DIR)]
        result = subprocess.run(command, text=True, capture_output=True)
        if result.stdout.strip():
            print(result.stdout.strip(), flush=True)
        if result.returncode:
            print(f"EXPORT-WARNING rc={result.returncode}: {result.stderr.strip()}", file=sys.stderr, flush=True)

    def reap(self) -> bool:
        completed = False
        for item in list(self.running):
            rc = item.process.poll()
            if rc is None:
                continue
            completed = True
            self.running.remove(item)
            self.states[item.node.name].release(item.gpus)
            after = self.remote_rows(item.node, item.output)
            status = "ok" if rc == 0 else "failed"
            ended = utcnow()
            self.append_manifest({
                "event": "done", "status": status, "run_id": self.run_id,
                "attempt_id": item.tag, "started_at": item.started_at, "ended_at": ended,
                "returncode": rc, "node": item.node.name, "host": item.node.host,
                "gpus": item.gpus, "nccl_port": item.port, "phase": item.job.phase,
                "component": item.job.label, "parallel_mode": item.job.mode,
                "world_size": item.job.size, "output": item.output, "log": item.log,
                "rows_before": item.rows_before, "rows_after": after,
            })
            self.fetch_remote_output(item)
            if not self.args.no_export_compat:
                self.export_compat()
            print(
                f"DONE  {item.job.job_id} node={item.node.name} gpus={item.gpus} rc={rc} "
                f"rows={item.rows_before}->{after} pending={len(self.pending)} running={len(self.running)}",
                flush=True,
            )
            if status == "failed" and after <= item.rows_before and self.args.retry_failed:
                self.queue_retry(item.job, f"rc={rc} rows={item.rows_before}->{after}")
        return completed

    def stop_all(self) -> None:
        if not self.running:
            return
        print(f"INTERRUPT terminating {len(self.running)} scheduler-owned task(s)", flush=True)
        for item in self.running:
            try:
                os.killpg(item.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for item in self.running:
            pattern = f"[{item.tag[0]}]{item.tag[1:]}"
            cleanup = f"pkill -TERM -f -- {shlex.quote(pattern)} || true"
            try:
                self.run_on_node(item.node, cleanup)
            except subprocess.SubprocessError:
                pass
        deadline = time.monotonic() + self.args.terminate_timeout
        while time.monotonic() < deadline and any(x.process.poll() is None for x in self.running):
            time.sleep(0.2)
        for item in self.running:
            if item.process.poll() is None:
                try:
                    os.killpg(item.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def occupancy(self) -> str:
        return " ".join(f"{name}:{sum(s.used)}/8" for name, s in self.states.items())

    def run(self) -> int:
        print(f"run_id={self.run_id} jobs={len(self.pending)} nodes={','.join(self.states)}", flush=True)
        try:
            while self.pending or self.running:
                launched = self.fill()
                reaped = self.reap()
                if self.pending or self.running:
                    print(
                        f"STATE gpu={self.occupancy()} pending={len(self.pending)} running={len(self.running)}",
                        flush=True,
                    )
                if not launched and not reaped:
                    time.sleep(self.args.poll_interval)
        except KeyboardInterrupt:
            self.interrupted = True
            self.stop_all()
            return 130
        return 0


def parse_nodes(args: argparse.Namespace) -> list[Node]:
    hosts = dict(DEFAULT_NODES)
    for override in args.node_host:
        if "=" not in override:
            raise SystemExit(f"--node-host expects NAME=HOST, got {override!r}")
        name, host = override.split("=", 1)
        hosts[name] = host
    unknown = set(args.nodes) - set(hosts)
    if unknown:
        raise SystemExit(f"unknown --nodes: {sorted(unknown)}; add each with --node-host NAME=HOST")
    return [Node(name, hosts[name]) for name in args.nodes]


def make_jobs(args: argparse.Namespace, selected_nodes: list[Node]) -> list[Job]:
    node1_selected = any(n.name == "node1" for n in selected_nodes)
    jobs = []
    routing_modes = list(args.routing_modes) if args.routing_modes else []
    for phase in args.phases:
        for label in args.components:
            component, mode = COMPONENTS[label]
            routing_list = routing_modes if mode == "moe_ep" and routing_modes else [None]
            for size in args.sizes:
                for routing_mode in routing_list:
                    routing_suffix = f"_{routing_mode}" if routing_mode else ""
                    base_stem = f"qwen3_{phase}_{mode}{routing_suffix}_ws{size}"
                    pinned = None
                    if (
                        args.prefer_node1_progress
                        and node1_selected
                        and count_rows(args.raw_root / "node1" / f"{base_stem}.jsonl") > 0
                    ):
                        pinned = "node1"
                    shard_specs = matrix_shard_specs(phase) if args.shard_matrix else []
                    if shard_specs:
                        for shard_label, lengths, batches in shard_specs:
                            expected = len(lengths) * len(batches) * DEFAULT_FREQ_COUNT
                            stem = f"{base_stem}_shard-{shard_label}"
                            jobs.append(
                                Job(
                                    phase,
                                    label,
                                    component,
                                    mode,
                                    size,
                                    stem,
                                    routing_mode=routing_mode,
                                    pinned_node=pinned,
                                    shard_label=shard_label,
                                    lengths=lengths,
                                    batch_sizes=batches,
                                    expected_rows=expected,
                                )
                            )
                    else:
                        jobs.append(
                            Job(
                                phase,
                                label,
                                component,
                                mode,
                                size,
                                base_stem,
                                routing_mode=routing_mode,
                                pinned_node=pinned,
                            )
                        )
    return jobs


def dry_run(args: argparse.Namespace, nodes: list[Node], jobs: list[Job]) -> None:
    states = {node.name: NodeState(node) for node in nodes}
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
                    output_root = PurePosixPath(args.container_output_root)
                    if not output_root.is_absolute():
                        output_root = PurePosixPath(args.container_repo) / output_root
                    output = output_root / state.node.name
                    output = str(output / f"{job.stem}.jsonl")
                    port = args.base_nccl_port + gpus[0]
                    cmd = Scheduler(args, nodes, []).benchmark_command(job, output, port)
                    assigned.append((job, state, gpus, cmd))
                    pending.remove(job)
                    print(
                        f"DRY-RUN wave={wave} {job.job_id} node={state.node.name} host={state.node.host} "
                        f"gpus={gpus} port={port} pinned={bool(job.pinned_node)}\n  {shlex.join(cmd)}"
                    )
                    break
        if not assigned:
            raise SystemExit("no selected node can fit the pending jobs")
        for _, state, gpus, _ in assigned:
            state.release(gpus)
        wave += 1
    print(f"planned {len(jobs)} jobs across {wave - 1} packing wave(s)")


def load_active_manifest(path: Path) -> dict[str, dict]:
    active = {}
    if not path.exists():
        return active
    with path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            attempt = row.get("attempt_id")
            if not attempt:
                continue
            if row.get("event") == "start":
                active[attempt] = row
            elif row.get("event") == "done":
                active.pop(attempt, None)
    return active


def status(args: argparse.Namespace, nodes: list[Node], jobs: list[Job]) -> int:
    manifest = args.raw_root / "manifest.jsonl"
    active = load_active_manifest(manifest)
    print(f"manifest={manifest.resolve()} active_attempts={len(active)}")
    helper = Scheduler(args, nodes, [])
    for node in nodes:
        owned = [row for row in active.values() if row.get("node") == node.name]
        gpu_map = sorted((gpu, row["attempt_id"]) for row in owned for gpu in row.get("gpus", []))
        try:
            result = helper.run_on_node(node, "pgrep -af 'sgl-profile-' || true", capture=True)
            live = [line for line in result.stdout.splitlines() if "bash -lc pgrep" not in line]
        except subprocess.SubprocessError as exc:
            live = [f"status query failed: {exc}"]
        print(f"NODE {node.name} host={node.host} manifest_running={len(owned)} gpus={gpu_map or 'free'}")
        for line in live:
            print(f"  PROCESS {line}")
        for job in jobs:
            output, _ = helper.paths(job, node)
            rows = helper.remote_rows(node, output)
            if rows > 0 or any(row.get("output") == output for row in owned):
                print(f"  ROWS {job.job_id}={rows} output={output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=["run", "status"], default="run")
    parser.add_argument("--model-path", default="/models/Qwen3-30B-A3B")
    parser.add_argument("--container", default="moe-energy")
    parser.add_argument("--container-repo", default=DEFAULT_REPO)
    parser.add_argument("--host-repo", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--raw-root", type=Path, default=RAW_V2_ROOT,
        help="host path used for manifests and collected raw JSONL files",
    )
    parser.add_argument(
        "--container-output-root",
        default="benchmark/moe-energy/profiling/data/raw_v2",
        help="container output root (relative paths are resolved under --container-repo)",
    )
    parser.add_argument("--nodes", nargs="+", default=list(DEFAULT_NODES))
    parser.add_argument("--node-host", action="append", default=[], metavar="NAME=HOST")
    parser.add_argument("--phases", nargs="+", choices=["prefill", "decode"], default=["prefill", "decode"])
    parser.add_argument("--components", nargs="+", choices=list(COMPONENTS), default=list(COMPONENTS))
    parser.add_argument("--sizes", nargs="+", type=int, choices=[2, 4, 8], default=[2, 4, 8])
    parser.add_argument(
        "--routing-modes",
        nargs="+",
        choices=EP_ROUTING_MODES,
        default=[],
        help="forced EP routing modes; each F-EP job is duplicated per mode",
    )
    parser.add_argument("--extra-args", default="--disable-custom-all-reduce")
    parser.add_argument(
        "--ep-max-dispatch-tokens", type=int, default=131072,
        help="FlashInfer ampere_ep max dispatch tokens per rank",
    )
    parser.add_argument(
        "--ep-a2a-backend", choices=["none", "ampere_ep", "deepep"], default="none",
        help="MoE EP all-to-all backend; A and F-TP always use none",
    )
    parser.add_argument(
        "--no-export-compat", action="store_true",
        help="do not run the legacy compatibility exporter after each job",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--shard-matrix",
        action="store_true",
        help="split each job into length/batch sub-matrices for higher parallelism",
    )
    parser.add_argument("--prefer-node1-progress", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="skip jobs that already have >= --skip-existing-min-rows JSONL rows on any node",
    )
    parser.add_argument("--skip-existing-min-rows", type=int, default=400)
    parser.add_argument(
        "--retry-failed", action=argparse.BooleanOptionalAction, default=True,
        help="re-queue jobs that exit non-zero without adding JSONL rows",
    )
    parser.add_argument(
        "--max-job-attempts", type=int, default=10,
        help="max launches per job_id when --retry-failed is enabled",
    )
    parser.add_argument("--base-nccl-port", type=int, default=29500)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--control-timeout", type=float, default=15.0)
    parser.add_argument("--terminate-timeout", type=float, default=10.0)
    args = parser.parse_args()
    if args.ep_max_dispatch_tokens < 1:
        parser.error("--ep-max-dispatch-tokens must be positive")
    if args.max_job_attempts < 1:
        parser.error("--max-job-attempts must be positive")
    return args


def main() -> int:
    args = parse_args()
    nodes = parse_nodes(args)
    jobs = make_jobs(args, nodes)
    if args.skip_existing:
        if args.skip_existing_min_rows < 1:
            raise SystemExit("--skip-existing-min-rows must be positive")
        jobs = filter_skip_existing(jobs, args.raw_root, None, args.skip_existing_min_rows)
    if args.command == "status":
        return status(args, nodes, jobs)
    if args.dry_run:
        dry_run(args, nodes, jobs)
        return 0
    return Scheduler(args, nodes, jobs).run()


if __name__ == "__main__":
    raise SystemExit(main())
