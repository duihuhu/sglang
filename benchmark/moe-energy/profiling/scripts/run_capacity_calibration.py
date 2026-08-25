#!/usr/bin/env python3
"""Calibrate Qwen3 A/F capacity boundaries with isolated distributed probes."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable

HERE = Path(__file__).resolve().parent
PROFILE_ROOT = HERE.parent
REPO_ROOT = HERE.parents[3]
DATA_ROOT = PROFILE_ROOT / "data"
RAW_V2_ROOT = DATA_ROOT / "raw_v2"
CAPACITY_ROOT = DATA_ROOT / "capacity"
DEFAULT_REPO = "/workspace/sglang-source/sglang"
DEFAULT_NODES = {
    "node1": "local",
    "node3": "10.252.129.34",
    "node4": "10.252.129.33",
}
COMPONENTS = {
    "A": ("A", "attn_tp"),
    "F-TP": ("F", "moe_tp"),
    "F-EP": ("F", "moe_ep"),
}
LENGTHS = [64, 128, 256, 512, 1024, 2048, 4096]
PROBE_FREQ = 1410
BACKFILL_FREQS = [210, 450, 690, 930, 1170]
SCHEMA_VERSION = "qwen3-af-v2"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, order=True)
class Axis:
    phase: str
    label: str
    component: str
    mode: str
    size: int
    length: int

    @property
    def key(self) -> str:
        return (
            f"{self.phase}:{self.label}:ws{self.size}:length{self.length}"
        )

    def fields(self) -> dict:
        return {
            "chain": self.key,
            "phase": self.phase,
            "component": self.label,
            "raw_component": self.component,
            "parallel_mode": self.mode,
            "world_size": self.size,
            "length": self.length,
        }


@dataclass(frozen=True)
class Task:
    axis: Axis
    batch: int
    freq: int
    kind: str

    @property
    def key(self) -> tuple[Axis, int, int]:
        return (self.axis, self.batch, self.freq)

    @property
    def task_id(self) -> str:
        return (
            f"{self.kind}-{self.axis.phase}-{self.axis.mode}-ws"
            f"{self.axis.size}-l{self.axis.length}-b{self.batch}-f{self.freq}"
        )


@dataclass(frozen=True)
class Node:
    name: str
    host: str

    @property
    def local(self) -> bool:
        return self.host == "local"


@dataclass
class NodeState:
    node: Node
    used: list[bool] = field(default_factory=lambda: [False] * 8)

    def allocate(self, size: int) -> list[int] | None:
        for start in range(0, len(self.used) - size + 1):
            if any(self.used[start : start + size]):
                continue
            for gpu in range(start, start + size):
                self.used[gpu] = True
            return list(range(start, start + size))
        return None

    def release(self, gpus: Iterable[int]) -> None:
        for gpu in gpus:
            self.used[gpu] = False


@dataclass
class Running:
    task: Task
    node: Node
    gpus: list[int]
    port: int
    tag: str
    process: subprocess.Popen
    started_at: str
    started_monotonic: float
    output: str
    log: str
    command: list[str]


def task_sort_key(task: Task) -> tuple:
    return (
        0 if task.kind == "probe" else 1,
        -task.axis.size,
        task.axis.phase,
        task.axis.mode,
        task.axis.length,
        task.batch,
        task.freq,
    )


def next_power_of_two(value: int) -> int:
    if value < 1:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


def ep_max_dispatch_tokens(task: Task, args: argparse.Namespace) -> int:
    required = task.axis.length * task.batch
    configured = args.ep_max_dispatch_tokens
    dispatch_tokens = configured or next_power_of_two(required)
    if dispatch_tokens < required:
        raise ValueError(
            f"--ep-max-dispatch-tokens={dispatch_tokens} is smaller than "
            f"probe shape tokens={required} for {task.task_id}"
        )
    if args.ep_max_dispatch_tokens_cap:
        dispatch_tokens = min(dispatch_tokens, args.ep_max_dispatch_tokens_cap)
        if dispatch_tokens < required:
            raise ValueError(
                f"--ep-max-dispatch-tokens-cap={dispatch_tokens} is smaller "
                f"than probe shape tokens={required} for {task.task_id}"
            )
    return dispatch_tokens


def next_probe_batch(
    successful_batches: Iterable[int], cap: int
) -> tuple[int | None, bool]:
    """Return the next power of two, or censored=True if the cap blocks it."""
    batches = list(successful_batches)
    if not batches:
        return (1, False) if cap >= 1 else (None, True)
    candidate = 1
    maximum = max(batches)
    while candidate <= maximum:
        candidate *= 2
    if candidate > cap:
        return None, True
    return candidate, False


def advance_probe(
    task: Task,
    *,
    succeeded: bool,
    cap: int,
    known_freqs: Iterable[int],
    probe_freq: int = PROBE_FREQ,
    backfill_freqs: Iterable[int] = BACKFILL_FREQS,
) -> tuple[list[Task], str | None]:
    """Pure chain transition used by the scheduler and CPU tests."""
    if not succeeded:
        return [], "boundary"
    known = set(known_freqs)
    followups = [
        Task(task.axis, task.batch, freq, "backfill")
        for freq in backfill_freqs
        if freq not in known and freq != probe_freq
    ]
    next_batch = task.batch * 2
    if next_batch > cap:
        return followups, "censored"
    followups.append(Task(task.axis, next_batch, probe_freq, "probe"))
    return followups, None


def pack_once(
    tasks: Iterable[Task], nodes: Iterable[Node]
) -> list[tuple[Task, Node, list[int]]]:
    """Pack one scheduling wave without mutating caller-owned state."""
    states = [NodeState(node) for node in nodes]
    assigned: list[tuple[Task, Node, list[int]]] = []
    for task in sorted(tasks, key=task_sort_key):
        for state in sorted(states, key=lambda item: sum(item.used)):
            gpus = state.allocate(task.axis.size)
            if gpus is not None:
                assigned.append((task, state.node, gpus))
                break
    return assigned


def _row_axis(row: dict) -> Axis | None:
    pair = (row.get("component"), row.get("parallel_mode"))
    labels = {
        ("A", "attn_tp"): "A",
        ("F", "moe_tp"): "F-TP",
        ("F", "moe_ep"): "F-EP",
    }
    label = labels.get(pair)
    try:
        if label is None:
            return None
        world = int(row["world_size"])
        if pair == ("A", "attn_tp") and int(row["attn_tp"]) != world:
            return None
        if pair == ("F", "moe_tp") and (
            int(row["moe_tp"]) != world or int(row["moe_ep"]) != 1
        ):
            return None
        if pair == ("F", "moe_ep") and (
            int(row["moe_ep"]) != world or int(row["moe_tp"]) != 1
        ):
            return None
        return Axis(
            str(row["phase"]),
            label,
            str(row["component"]),
            str(row["parallel_mode"]),
            world,
            int(row["length"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def load_successes(
    roots: Iterable[Path], *, exclude_capacity: bool = False
) -> set[tuple[Axis, int, int]]:
    successes: set[tuple[Axis, int, int]] = set()
    visited: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.jsonl")):
            resolved = path.resolve()
            if resolved in visited or path.name == "manifest.jsonl":
                continue
            visited.add(resolved)
            try:
                relative_parts = path.relative_to(root).parts
            except ValueError:
                relative_parts = path.parts
            if exclude_capacity and "capacity" in relative_parts:
                continue
            try:
                source = path.open()
            except OSError:
                continue
            with source:
                for line in source:
                    try:
                        row = json.loads(line)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if (
                        row.get("schema_version") != SCHEMA_VERSION
                        or row.get("status") != "ok"
                    ):
                        continue
                    axis = _row_axis(row)
                    try:
                        if axis is not None:
                            successes.add(
                                (axis, int(row["batch"]), int(row["freq_mhz"]))
                            )
                    except (KeyError, TypeError, ValueError):
                        continue
    return successes


def read_manifest(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.exists():
        return records
    with path.open() as source:
        for line in source:
            try:
                records.append(json.loads(line))
            except (json.JSONDecodeError, TypeError):
                continue
    return records


def manifest_state(
    records: Iterable[dict],
) -> tuple[dict[str, dict], dict[str, dict], set[tuple[str, int, int]]]:
    active: dict[str, dict] = {}
    terminal: dict[str, dict] = {}
    completed: set[tuple[str, int, int]] = set()
    for row in records:
        attempt = row.get("attempt_id")
        if row.get("event") == "start" and attempt:
            active[attempt] = row
        elif row.get("event") == "done" and attempt:
            active.pop(attempt, None)
            if row.get("status") == "ok":
                try:
                    completed.add(
                        (str(row["chain"]), int(row["batch"]), int(row["freq_mhz"]))
                    )
                except (KeyError, TypeError, ValueError):
                    pass
            if row.get("boundary"):
                terminal[str(row.get("chain"))] = row
        elif row.get("event") == "chain" and row.get("status") == "censored":
            terminal[str(row.get("chain"))] = row
    return active, terminal, completed


def axis_batches(
    successes: Iterable[tuple[Axis, int, int]], axis: Axis
) -> set[int]:
    return {batch for point_axis, batch, _ in successes if point_axis == axis}


def axis_freqs(
    successes: Iterable[tuple[Axis, int, int]], axis: Axis, batch: int
) -> set[int]:
    return {
        freq
        for point_axis, point_batch, freq in successes
        if point_axis == axis and point_batch == batch
    }


def build_initial_plan(
    axes: Iterable[Axis],
    baseline: set[tuple[Axis, int, int]],
    all_successes: set[tuple[Axis, int, int]],
    terminal: dict[str, dict],
    completed: set[tuple[str, int, int]],
    caps: dict[str, int],
    *,
    probe_freq: int = PROBE_FREQ,
    backfill_freqs: Iterable[int] = BACKFILL_FREQS,
) -> tuple[list[Task], list[tuple[Axis, int | None, int | None]], list[Axis]]:
    tasks: list[Task] = []
    starts: list[tuple[Axis, int | None, int | None]] = []
    censored: list[Axis] = []
    targets = [probe_freq, *backfill_freqs]
    for axis in axes:
        baseline_batches = axis_batches(baseline, axis)
        successful_batches = axis_batches(all_successes, axis)
        baseline_max = max(baseline_batches) if baseline_batches else None

        for batch in sorted(successful_batches):
            if baseline_max is not None and batch <= baseline_max:
                continue
            known = axis_freqs(all_successes, axis, batch)
            known.update(
                freq
                for chain, done_batch, freq in completed
                if chain == axis.key and done_batch == batch
            )
            tasks.extend(
                Task(axis, batch, freq, "backfill")
                for freq in targets
                if freq not in known
            )

        terminal_row = terminal.get(axis.key)
        reopen_censored = bool(
            terminal_row
            and terminal_row.get("status") == "censored"
            and caps[axis.phase] > int(terminal_row.get("host_batch_cap", 0))
        )
        if terminal_row and not reopen_censored:
            starts.append((axis, baseline_max, None))
            continue

        next_batch, is_censored = next_probe_batch(
            successful_batches, caps[axis.phase]
        )
        starts.append((axis, baseline_max, next_batch))
        if is_censored:
            censored.append(axis)
        elif next_batch is not None:
            tasks.append(Task(axis, next_batch, probe_freq, "probe"))

    unique = {task.key: task for task in tasks}
    return sorted(unique.values(), key=task_sort_key), starts, censored


class CapacityScheduler:
    def __init__(
        self,
        args: argparse.Namespace,
        nodes: list[Node],
        tasks: list[Task],
        successes: set[tuple[Axis, int, int]],
        completed: set[tuple[str, int, int]],
        stale_active: dict[str, dict],
    ):
        self.args = args
        self.nodes = nodes
        self.states = {node.name: NodeState(node) for node in nodes}
        self.pending: list[Task] = []
        self.queued: set[tuple[Axis, int, int]] = set()
        self.running: list[Running] = []
        self.successes = successes
        self.completed = completed
        self.stale_active = stale_active
        self.run_id = (
            f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-"
            f"{uuid.uuid4().hex[:8]}"
        )
        self.container_capacity_root = PurePosixPath(args.container_capacity_root)
        if not self.container_capacity_root.is_absolute():
            self.container_capacity_root = PurePosixPath(args.container_repo) / self.container_capacity_root
        self.manifest = args.capacity_root / "manifest.jsonl"
        self.interrupted = False
        for task in tasks:
            self.enqueue(task)

    def enqueue(self, task: Task) -> None:
        if task.key in self.queued:
            return
        if (task.axis.key, task.batch, task.freq) in self.completed:
            return
        self.pending.append(task)
        self.queued.add(task.key)

    def append_manifest(self, record: dict) -> None:
        self.manifest.parent.mkdir(parents=True, exist_ok=True)
        with self.manifest.open("a") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())

    def remote_argv(self, node: Node, argv: list[str]) -> list[str]:
        docker = ["docker", "exec", self.args.container, *argv]
        if node.local:
            return docker
        return ["ssh", node.host, shlex.join(docker)]

    def run_on_node(
        self,
        node: Node,
        script: str,
        *,
        capture: bool = False,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            self.remote_argv(node, ["bash", "-lc", script]),
            text=True,
            capture_output=capture,
            timeout=self.args.control_timeout if timeout is None else timeout,
        )

    def paths(self, task: Task, node: Node, tag: str) -> tuple[str, str]:
        root = self.container_capacity_root / "raw_v2" / node.name
        stem = f"{task.task_id}-{tag.rsplit('-', 1)[-1]}"
        return str(root / f"{stem}.jsonl"), str(root / "logs" / f"{stem}.log")

    def benchmark_command(
        self, task: Task, output: str, port: int
    ) -> list[str]:
        ep_size = task.axis.size if task.axis.mode == "moe_ep" else 1
        ep_a2a_backend = (
            self.args.ep_a2a_backend if task.axis.mode == "moe_ep" else "none"
        )
        max_running_requests = task.batch
        max_total_tokens = task.axis.length * task.batch
        if ep_a2a_backend in ("ampere_ep", "deepep"):
            # ServerArgs values are global under DP attention and are divided
            # among attention-DP workers when each local pool is constructed.
            # Keep the profiling row's batch local while sizing the global
            # limits so every worker can materialize that complete shape.
            dp_size = task.axis.size
            max_running_requests = max(8 * dp_size, task.batch * dp_size)
            max_total_tokens *= dp_size
        command = [
            "python3",
            (
                "benchmark/moe-energy/profiling/scripts/"
                f"bench_{task.axis.phase}_af.py"
            ),
            "--model-path",
            self.args.model_path,
            "--component",
            task.axis.component,
            "--parallel-mode",
            task.axis.mode,
            "--tp-size",
            str(task.axis.size),
            "--ep-size",
            str(ep_size),
            "--moe-runner-backend",
            "triton",
            "--moe-a2a-backend",
            ep_a2a_backend,
            "--lengths",
            str(task.axis.length),
            "--batch-sizes",
            str(task.batch),
            "--freqs",
            str(task.freq),
            "--shape-token-limit",
            "0",
            "--max-total-tokens",
            str(max_total_tokens),
            "--max-running-requests",
            str(max_running_requests),
            "--stop-on-shape-failure",
            "--dist-timeout",
            str(self.args.dist_timeout),
            "--nccl-port",
            str(port),
            "--output",
            output,
        ]
        if ep_a2a_backend in ("ampere_ep", "deepep"):
            command.extend(
                ["--enable-dp-attention", "--dp-size", str(task.axis.size)]
            )
            if ep_a2a_backend == "deepep":
                command.extend(
                    [
                        "--deepep-mode",
                        "low_latency",
                        "--deepep-dispatcher-output-dtype",
                        "bf16",
                    ]
                )
        command.extend(shlex.split(self.args.extra_args))
        return command

    def launch(self, task: Task, node: Node, gpus: list[int]) -> Running:
        node_index = self.nodes.index(node)
        port = self.args.base_nccl_port + node_index * 100 + gpus[0]
        tag = f"sgl-capacity-{self.run_id}-{uuid.uuid4().hex[:10]}"
        output, log = self.paths(task, node, tag)
        command = self.benchmark_command(task, output, port)
        launch_env = [
            f"CUDA_VISIBLE_DEVICES={','.join(map(str, gpus))}",
            "FLASHINFER_DISABLE_VERSION_CHECK=1",
            f"SGLANG_PROFILE_TAG={tag}",
        ]
        if (
            task.axis.mode == "moe_ep"
            and self.args.ep_a2a_backend == "ampere_ep"
        ):
            launch_env.append(
                "SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK="
                f"{ep_max_dispatch_tokens(task, self.args)}"
            )
        quoted_env = " ".join(shlex.quote(item) for item in launch_env)
        pidfile = f"/tmp/sgl-capacity/{tag}.pid"
        wrapper = (
            f"mkdir -p /tmp/sgl-capacity; "
            f"trap 'if test -s {shlex.quote(pidfile)}; then "
            f"p=$(cat {shlex.quote(pidfile)}); kill -TERM -- -$p "
            "2>/dev/null || true; fi' TERM INT HUP; "
            f"setsid env {quoted_env} "
            f"{shlex.join(command)} >> {shlex.quote(log)} 2>&1 & "
            f"child=$!; printf '%s\\n' \"$child\" > {shlex.quote(pidfile)}; "
            "wait \"$child\""
        )
        setup = (
            f"cd {shlex.quote(self.args.container_repo)} && "
            f"mkdir -p {shlex.quote(str(PurePosixPath(output).parent))} "
            f"{shlex.quote(str(PurePosixPath(log).parent))} && "
            f"bash -lc {shlex.quote(wrapper)} {shlex.quote(tag)}"
        )
        argv = self.remote_argv(node, ["bash", "-lc", setup])
        started_at = utcnow()
        process = subprocess.Popen(
            argv,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        running = Running(
            task,
            node,
            gpus,
            port,
            tag,
            process,
            started_at,
            time.monotonic(),
            output,
            log,
            argv,
        )
        self.append_manifest(
            {
                "event": "start",
                "status": "running",
                "run_id": self.run_id,
                "attempt_id": tag,
                "task_kind": task.kind,
                "started_at": started_at,
                "node": node.name,
                "host": node.host,
                "gpus": gpus,
                "nccl_port": port,
                **task.axis.fields(),
                "batch": task.batch,
                "freq_mhz": task.freq,
                "output": output,
                "log": log,
                "command": argv,
            }
        )
        return running

    def cleanup_tag(self, node: Node, tag: str) -> bool:
        pidfile = f"/tmp/sgl-capacity/{tag}.pid"
        script = (
            f"pidfile={shlex.quote(pidfile)}; "
            "test -s \"$pidfile\" || exit 0; "
            "p=$(cat \"$pidfile\"); "
            "kill -TERM -- -\"$p\" 2>/dev/null || true; "
            f"deadline=$((SECONDS+{max(1, int(self.args.terminate_timeout))})); "
            "while kill -0 -- -\"$p\" 2>/dev/null && "
            "test \"$SECONDS\" -lt \"$deadline\"; do sleep 1; done; "
            "kill -KILL -- -\"$p\" 2>/dev/null || true; sleep 1; "
            "if kill -0 -- -\"$p\" 2>/dev/null; then exit 1; fi; "
            "rm -f \"$pidfile\""
        )
        try:
            return (
                self.run_on_node(
                    node,
                    script,
                    timeout=max(
                        self.args.control_timeout,
                        self.args.terminate_timeout + 10,
                    ),
                ).returncode
                == 0
            )
        except subprocess.SubprocessError:
            return False

    def kill_host_wrapper(self, item: Running) -> None:
        if item.process.poll() is not None:
            return
        try:
            os.killpg(item.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            item.process.wait(timeout=self.args.terminate_timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(item.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def _host_path(self, container_path: str) -> Path:
        try:
            relative = PurePosixPath(container_path).relative_to(
                PurePosixPath(self.args.container_repo)
            )
        except ValueError as exc:
            raise ValueError(
                f"container output {container_path!r} is outside --container-repo "
                f"{self.args.container_repo!r}"
            ) from exc
        return self.args.host_repo / Path(*relative.parts)

    def fetch_file(self, item: Running, remote_path: str, local_path: Path) -> bool:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        source = self._host_path(remote_path)
        if item.node.local:
            if not source.exists():
                return False
            if source.resolve() != local_path.resolve():
                shutil.copy2(source, local_path)
            return True
        try:
            result = subprocess.run(
                ["rsync", "-a", f"{item.node.host}:{source}", str(local_path)],
                text=True,
                capture_output=True,
                timeout=self.args.control_timeout,
            )
        except subprocess.TimeoutExpired:
            print(
                f"FETCH-WARNING node={item.node.name} path={remote_path}: "
                "rsync timed out",
                file=sys.stderr,
                flush=True,
            )
            return False
        if result.returncode:
            print(
                f"FETCH-WARNING node={item.node.name} path={remote_path}: "
                f"{result.stderr.strip()}",
                file=sys.stderr,
                flush=True,
            )
            return False
        return True

    def collect_output(
        self, item: Running
    ) -> tuple[Path, Path, int, list[dict]]:
        central = (
            self.args.capacity_root
            / "raw_v2"
            / item.node.name
            / Path(item.output).name
        )
        self.fetch_file(item, item.output, central)
        central_log = (
            self.args.capacity_root
            / "raw_v2"
            / item.node.name
            / "logs"
            / Path(item.log).name
        )
        self.fetch_file(item, item.log, central_log)
        promoted = (
            self.args.formal_raw_root
            / "capacity"
            / item.node.name
            / central.name
        )
        rows: list[dict] = []
        if central.exists():
            with central.open() as source:
                for line in source:
                    try:
                        row = json.loads(line)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    axis = _row_axis(row)
                    try:
                        matches = (
                            row.get("status") == "ok"
                            and axis == item.task.axis
                            and int(row.get("batch", -1)) == item.task.batch
                        )
                    except (TypeError, ValueError):
                        matches = False
                    if matches:
                        rows.append(row)
        return central, promoted, len(rows), rows

    def record_censored(self, task: Task) -> None:
        self.append_manifest(
            {
                "event": "chain",
                "status": "censored",
                "run_id": self.run_id,
                "ended_at": utcnow(),
                **task.axis.fields(),
                "last_success_batch": task.batch,
                "host_batch_cap": self.args.caps[task.axis.phase],
                "censored": True,
            }
        )

    def finish(self, item: Running, forced_status: str | None = None) -> None:
        cleanup_ok = self.cleanup_tag(item.node, item.tag)
        self.kill_host_wrapper(item)
        central, promoted, row_count, rows = self.collect_output(item)
        rc = item.process.poll()
        if forced_status is not None:
            status = forced_status
        elif rc not in (0, None):
            status = "failed"
        elif row_count == 0:
            status = "no-data"
        else:
            status = "ok"
        if not cleanup_ok:
            status = "cleanup-failed"
        if status == "ok":
            promoted.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(central, promoted)

        self.running.remove(item)
        self.queued.discard(item.task.key)
        if cleanup_ok:
            self.states[item.node.name].release(item.gpus)
        else:
            print(
                f"CRITICAL {item.tag}: process group could not be cleared; "
                f"GPUs {item.gpus} remain quarantined",
                file=sys.stderr,
                flush=True,
            )

        actual_freqs: set[int] = set()
        for row in rows:
            try:
                freq = int(row["freq_mhz"])
            except (KeyError, TypeError, ValueError):
                continue
            actual_freqs.add(freq)
            if status == "ok":
                self.successes.add((item.task.axis, item.task.batch, freq))
        if status == "ok":
            self.completed.add(
                (item.task.axis.key, item.task.batch, item.task.freq)
            )

        boundary = item.task.kind == "probe" and status in {
            "failed",
            "no-data",
            "timeout",
            "cleanup-failed",
        }
        self.append_manifest(
            {
                "event": "done",
                "status": status,
                "run_id": self.run_id,
                "attempt_id": item.tag,
                "task_kind": item.task.kind,
                "started_at": item.started_at,
                "ended_at": utcnow(),
                "returncode": rc,
                "timeout": status == "timeout",
                "boundary": boundary,
                "cleanup_ok": cleanup_ok,
                "rows": row_count,
                "actual_freqs": sorted(actual_freqs),
                "node": item.node.name,
                "host": item.node.host,
                "gpus": item.gpus,
                "nccl_port": item.port,
                **item.task.axis.fields(),
                "batch": item.task.batch,
                "freq_mhz": item.task.freq,
                "output": item.output,
                "log": item.log,
            }
        )

        if item.task.kind == "probe":
            known = axis_freqs(
                self.successes, item.task.axis, item.task.batch
            )
            known.update(
                freq
                for chain, batch, freq in self.completed
                if chain == item.task.axis.key and batch == item.task.batch
            )
            followups, terminal = advance_probe(
                item.task,
                succeeded=status == "ok",
                cap=self.args.caps[item.task.axis.phase],
                known_freqs=known,
                probe_freq=self.args.probe_freq,
                backfill_freqs=self.args.backfill_freqs,
            )
            for task in followups:
                self.enqueue(task)
            if terminal == "censored":
                self.record_censored(item.task)

        print(
            f"DONE {item.task.task_id} node={item.node.name} "
            f"gpus={item.gpus} status={status} rows={row_count} "
            f"pending={len(self.pending)} running={len(self.running)}",
            flush=True,
        )

    def recover_stale(self) -> None:
        if not self.stale_active:
            return
        by_name = {node.name: node for node in self.nodes}
        for attempt, row in self.stale_active.items():
            node = by_name.get(str(row.get("node")))
            cleanup_ok = bool(node and self.cleanup_tag(node, attempt))
            self.append_manifest(
                {
                    **row,
                    "event": "done",
                    "status": "recovered-stale",
                    "ended_at": utcnow(),
                    "cleanup_ok": cleanup_ok,
                    "boundary": False,
                }
            )
            if not cleanup_ok:
                raise RuntimeError(
                    f"could not clean stale attempt {attempt}; refusing to schedule"
                )

    def fill(self) -> bool:
        launched = False
        for task in sorted(list(self.pending), key=task_sort_key):
            for state in sorted(
                self.states.values(),
                key=lambda item: (sum(item.used), self.nodes.index(item.node)),
            ):
                gpus = state.allocate(task.axis.size)
                if gpus is None:
                    continue
                try:
                    running = self.launch(task, state.node, gpus)
                except Exception as exc:
                    state.release(gpus)
                    self.pending.remove(task)
                    self.queued.discard(task.key)
                    boundary = task.kind == "probe"
                    self.append_manifest(
                        {
                            "event": "done",
                            "status": "launch-failed",
                            "run_id": self.run_id,
                            "attempt_id": f"{self.run_id}-{uuid.uuid4().hex[:8]}",
                            "task_kind": task.kind,
                            "ended_at": utcnow(),
                            "node": state.node.name,
                            **task.axis.fields(),
                            "batch": task.batch,
                            "freq_mhz": task.freq,
                            "boundary": boundary,
                            "error": repr(exc),
                        }
                    )
                    print(
                        f"LAUNCH-ERROR {task.task_id} "
                        f"node={state.node.name}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    launched = True
                    break
                self.pending.remove(task)
                self.running.append(running)
                launched = True
                print(
                    f"START {task.task_id} node={state.node.name} "
                    f"gpus={gpus} port={running.port}",
                    flush=True,
                )
                break
        return launched

    def reap(self) -> bool:
        completed_any = False
        now = time.monotonic()
        for item in list(self.running):
            if item.process.poll() is not None:
                self.finish(item)
                completed_any = True
            elif now - item.started_monotonic >= self.args.point_timeout:
                self.finish(item, "timeout")
                completed_any = True
        return completed_any

    def stop_all(self) -> None:
        for item in list(self.running):
            self.finish(item, "interrupted")

    def occupancy(self) -> str:
        return " ".join(
            f"{name}:{sum(state.used)}/8"
            for name, state in self.states.items()
        )

    def run(self) -> int:
        self.recover_stale()
        print(
            f"run_id={self.run_id} tasks={len(self.pending)} "
            f"nodes={','.join(self.states)}",
            flush=True,
        )
        try:
            while self.pending or self.running:
                launched = self.fill()
                reaped = self.reap()
                if self.pending or self.running:
                    print(
                        f"STATE gpu={self.occupancy()} "
                        f"pending={len(self.pending)} "
                        f"running={len(self.running)}",
                        flush=True,
                    )
                if not launched and not reaped:
                    time.sleep(self.args.poll_interval)
        except KeyboardInterrupt:
            self.interrupted = True
            self.stop_all()
            return 130
        return 0


def make_axes(args: argparse.Namespace) -> list[Axis]:
    return [
        Axis(phase, label, *COMPONENTS[label], size, length)
        for phase in args.phases
        for label in args.components
        for size in args.sizes
        for length in args.lengths
    ]


def parse_nodes(args: argparse.Namespace) -> list[Node]:
    hosts = dict(DEFAULT_NODES)
    for override in args.node_host:
        if "=" not in override:
            raise SystemExit(
                f"--node-host expects NAME=HOST, got {override!r}"
            )
        name, host = override.split("=", 1)
        hosts[name] = host
    unknown = set(args.nodes) - set(hosts)
    if unknown:
        raise SystemExit(
            f"unknown nodes {sorted(unknown)}; use --node-host NAME=HOST"
        )
    return [Node(name, hosts[name]) for name in args.nodes]


def print_dry_run(
    starts: Iterable[tuple[Axis, int | None, int | None]],
    terminal: dict[str, dict],
    censored: Iterable[Axis],
    tasks: list[Task],
    nodes: list[Node],
) -> None:
    start_rows = list(starts)
    censored_keys = {axis.key for axis in censored}
    for axis, baseline_max, next_batch in start_rows:
        if axis.key in terminal:
            state = terminal[axis.key].get("status", "terminal")
            point = state
        elif axis.key in censored_keys:
            point = "censored"
        else:
            point = str(next_batch)
        print(
            f"CHAIN {axis.key} baseline_max={baseline_max} "
            f"next_probe={point}"
        )
    wave = pack_once(tasks, nodes)
    print(f"INITIAL-PACK tasks={len(wave)} gpus={sum(len(x[2]) for x in wave)}")
    for task, node, gpus in wave:
        print(
            f"  {task.task_id} node={node.name} host={node.host} gpus={gpus}"
        )
    print(
        f"DRY-RUN chains={len(start_rows)} pending_points={len(tasks)} "
        f"censored={len(censored_keys)}"
    )


def status(
    args: argparse.Namespace,
    baseline: set[tuple[Axis, int, int]],
    all_successes: set[tuple[Axis, int, int]],
) -> int:
    records = read_manifest(args.capacity_root / "manifest.jsonl")
    active, terminal, _ = manifest_state(records)
    boundaries = [
        row for row in terminal.values() if row.get("boundary")
    ]
    censored = [
        row for row in terminal.values() if row.get("status") == "censored"
    ]
    probe_points = {
        (row.get("chain"), row.get("batch"))
        for row in records
        if row.get("event") == "done"
        and row.get("task_kind") == "probe"
        and row.get("status") == "ok"
    }
    baseline_shapes = {(axis.key, batch) for axis, batch, _ in baseline}
    expanded = {
        (axis.key, batch) for axis, batch, _ in all_successes
    } - baseline_shapes
    print(
        f"capacity-status active={len(active)} boundaries={len(boundaries)} "
        f"successful_probe_points={len(probe_points)} "
        f"expanded_points={len(expanded)} censored={len(censored)}"
    )
    for row in active.values():
        print(
            f"ACTIVE {row.get('chain')} batch={row.get('batch')} "
            f"freq={row.get('freq_mhz')} node={row.get('node')} "
            f"gpus={row.get('gpus')} kind={row.get('task_kind')}"
        )
    for row in sorted(boundaries, key=lambda item: str(item.get("chain"))):
        print(
            f"BOUNDARY {row.get('chain')} first_failed_batch={row.get('batch')} "
            f"status={row.get('status')} timeout={row.get('timeout', False)}"
        )
    for row in sorted(censored, key=lambda item: str(item.get("chain"))):
        print(
            f"CENSORED {row.get('chain')} "
            f"last_success_batch={row.get('last_success_batch')} "
            f"cap={row.get('host_batch_cap')}"
        )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", nargs="?", choices=["run", "status"], default="run"
    )
    parser.add_argument("--model-path", default="/models/Qwen3-30B-A3B")
    parser.add_argument("--container", default="moe-energy")
    parser.add_argument("--container-repo", default=DEFAULT_REPO)
    parser.add_argument("--host-repo", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--container-capacity-root",
        default="benchmark/moe-energy/profiling/data/capacity",
        help="container capacity output root (relative paths resolve under --container-repo)",
    )
    parser.add_argument("--nodes", nargs="+", default=list(DEFAULT_NODES))
    parser.add_argument(
        "--node-host", action="append", default=[], metavar="NAME=HOST"
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=["prefill", "decode"],
        default=["prefill", "decode"],
    )
    parser.add_argument(
        "--components",
        nargs="+",
        choices=list(COMPONENTS),
        default=list(COMPONENTS),
    )
    parser.add_argument(
        "--sizes", nargs="+", type=int, choices=[2, 4, 8], default=[2, 4, 8]
    )
    parser.add_argument("--lengths", nargs="+", type=int, default=LENGTHS)
    parser.add_argument("--probe-freq", type=int, default=PROBE_FREQ)
    parser.add_argument(
        "--backfill-freqs", nargs="+", type=int, default=BACKFILL_FREQS
    )
    parser.add_argument("--prefill-batch-cap", type=int, default=4096)
    parser.add_argument("--decode-batch-cap", type=int, default=16384)
    parser.add_argument("--point-timeout", type=float, default=15 * 60)
    parser.add_argument("--dist-timeout", type=int, default=120)
    parser.add_argument("--base-nccl-port", type=int, default=29600)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--control-timeout", type=float, default=30.0)
    parser.add_argument("--terminate-timeout", type=float, default=10.0)
    parser.add_argument(
        "--extra-args", default="--disable-custom-all-reduce"
    )
    parser.add_argument(
        "--ep-max-dispatch-tokens",
        type=int,
        default=0,
        help=(
            "fixed FlashInfer ampere_ep dispatch workspace size; 0 chooses "
            "the next power of two at least as large as length * batch"
        ),
    )
    parser.add_argument(
        "--ep-max-dispatch-tokens-cap",
        type=int,
        default=0,
        help=(
            "optional global cap for the dynamic/fixed dispatch workspace; "
            "0 disables"
        ),
    )
    parser.add_argument(
        "--ep-a2a-backend", choices=["none", "ampere_ep", "deepep"], default="none",
        help="MoE EP all-to-all backend; A and F-TP always use none",
    )
    parser.add_argument("--capacity-root", type=Path, default=CAPACITY_ROOT)
    parser.add_argument(
        "--formal-raw-root", type=Path, default=RAW_V2_ROOT
    )
    parser.add_argument(
        "--baseline-raw-root", type=Path, action="append", default=None
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.ep_max_dispatch_tokens < 0 or args.ep_max_dispatch_tokens_cap < 0:
        parser.error("EP max dispatch token settings must be non-negative")
    if (
        args.ep_max_dispatch_tokens
        and args.ep_max_dispatch_tokens_cap
        and args.ep_max_dispatch_tokens > args.ep_max_dispatch_tokens_cap
    ):
        parser.error(
            "--ep-max-dispatch-tokens must not exceed "
            "--ep-max-dispatch-tokens-cap"
        )
    if args.prefill_batch_cap < 1 or args.decode_batch_cap < 1:
        parser.error("batch caps must be positive")
    if args.point_timeout <= 0 or args.dist_timeout <= 0:
        parser.error("timeouts must be positive")
    if args.probe_freq in args.backfill_freqs:
        parser.error("--probe-freq must not appear in --backfill-freqs")
    args.caps = {
        "prefill": args.prefill_batch_cap,
        "decode": args.decode_batch_cap,
    }
    if args.baseline_raw_root is None:
        args.baseline_raw_root = [args.formal_raw_root]
    return args


def main() -> int:
    args = parse_args()
    nodes = parse_nodes(args)
    axes = make_axes(args)
    baseline = load_successes(
        args.baseline_raw_root, exclude_capacity=True
    )
    capacity_successes = load_successes(
        [args.formal_raw_root / "capacity"]
    )
    all_successes = baseline | capacity_successes
    records = read_manifest(args.capacity_root / "manifest.jsonl")
    active, terminal, completed = manifest_state(records)

    if args.command == "status":
        return status(args, baseline, all_successes)

    tasks, starts, initially_censored = build_initial_plan(
        axes,
        baseline,
        all_successes,
        terminal,
        completed,
        args.caps,
        probe_freq=args.probe_freq,
        backfill_freqs=args.backfill_freqs,
    )
    if args.dry_run:
        print_dry_run(starts, terminal, initially_censored, tasks, nodes)
        return 0

    args.capacity_root.mkdir(parents=True, exist_ok=True)
    lock_path = args.capacity_root / ".scheduler.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(
                f"ERROR: another capacity scheduler holds {lock_path}",
                file=sys.stderr,
            )
            return 2
        scheduler = CapacityScheduler(
            args, nodes, tasks, all_successes, completed, active
        )
        for axis in initially_censored:
            last_success = max(axis_batches(all_successes, axis), default=None)
            scheduler.append_manifest(
                {
                    "event": "chain",
                    "status": "censored",
                    "run_id": scheduler.run_id,
                    "ended_at": utcnow(),
                    **axis.fields(),
                    "last_success_batch": last_success,
                    "host_batch_cap": args.caps[axis.phase],
                    "censored": True,
                }
            )
        return scheduler.run()


if __name__ == "__main__":
    raise SystemExit(main())
