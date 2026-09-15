from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from enum import Enum
import json, shlex, subprocess, time
from pathlib import Path
from typing import Callable

@dataclass
class ProcessSpec:
    role: str
    node: str
    host: str
    gpus: list[int]
    port: int
    command: str
    health_path: str | None = "/health"
    bootstrap_port: int | None = None
    nccl_port: int | None = None
    internal_ports: list[int] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    ready_log_patterns: list[str] = field(default_factory=list)
    ready_timeout_s: float | None = None
    startup_stage: int = 0
    log_path: str = ""
    pid_path: str = ""
    ready_command: str | None = None

    def __post_init__(self):
        if not self.log_path or not self.pid_path:
            safe_node = "".join(c if c.isalnum() or c in "_.-" else "_" for c in self.node)
            safe_role = "".join(c if c.isalnum() or c in "_.-" else "_" for c in self.role)
            base = "/tmp/aflex_bench/dryrun"
            stem = f"{safe_node}_{safe_role}_{self.port}"
            self.log_path = self.log_path or f"{base}/{stem}.log"
            self.pid_path = self.pid_path or f"{base}/{stem}.pid"

class RoutingPolicy(str, Enum):
    SINGLE = "single"
    CLIENT_ROUND_ROBIN = "client_round_robin"


@dataclass(frozen=True)
class DeploymentHandle:
    endpoints: tuple[str, ...]
    routing_policy: RoutingPolicy = RoutingPolicy.SINGLE

    @property
    def endpoint(self) -> str:
        return self.endpoints[0]

    def endpoint_for_request(self, request_index: int) -> str:
        if self.routing_policy == RoutingPolicy.SINGLE:
            return self.endpoint
        return self.endpoints[request_index % len(self.endpoints)]


@dataclass
class DeploymentPlan:
    architecture: str
    processes: list[ProcessSpec]
    endpoints: list[str] | str
    cleanup: list[tuple[str, str]] = field(default_factory=list)
    unlock: list[tuple[str, str]] = field(default_factory=list)
    cluster: dict = field(default_factory=dict)
    routing_policy: RoutingPolicy = RoutingPolicy.SINGLE
    runtime_options: dict = field(default_factory=dict)
    pre_actions: list[tuple[str, str]] = field(default_factory=list)
    run_tag: str = "dryrun"

    def __post_init__(self):
        if isinstance(self.endpoints, str):
            self.endpoints = [self.endpoints]
        else:
            self.endpoints = list(self.endpoints)
        if isinstance(self.routing_policy, str):
            self.routing_policy = RoutingPolicy(self.routing_policy)

    @property
    def endpoint(self) -> str:
        """Compatibility accessor for single-endpoint callers."""
        return self.endpoints[0]

    @property
    def handle(self) -> DeploymentHandle:
        return DeploymentHandle(tuple(self.endpoints), self.routing_policy)

    def used_gpu_map(self) -> dict[str, list[int]]:
        """Return the exact node/GPU allocation used by server processes."""
        used = {}
        for process in self.processes:
            if process.gpus:
                used.setdefault(process.node, set()).update(process.gpus)
        return {node: sorted(gpus) for node, gpus in used.items()}

    def validate(self):
        if not self.endpoints:
            raise ValueError("deployment requires at least one endpoint")
        if self.routing_policy == RoutingPolicy.SINGLE and len(self.endpoints) != 1:
            raise ValueError("single routing policy requires exactly one endpoint")
        ports = []
        for process in self.processes:
            if isinstance(process.startup_stage, bool) or not isinstance(process.startup_stage, int) or process.startup_stage < 0:
                raise ValueError("startup_stage must be a non-negative integer")
            ports.append((process.host, process.port, "http"))
            if process.bootstrap_port is not None:
                ports.append((process.host, process.bootstrap_port, "bootstrap"))
            if process.nccl_port is not None:
                ports.append((process.host, process.nccl_port, "nccl"))
            ports.extend((process.host, port, "internal") for port in process.internal_ports)
        keys = [(host, port) for host, port, _ in ports]
        if len(keys) != len(set(keys)):
            raise ValueError("HTTP/bootstrap/NCCL/UCX/sched port collision on the same host")
        used = [(p.node, g) for p in self.processes for g in p.gpus]
        if len(used) != len(set(used)):
            raise ValueError("GPU over-allocation")
        if self.architecture == "af":
            if not {p.role for p in self.processes}.issubset({"A", "F", "AF_ROUTER", "AF_COORDINATOR"}) or not {"A", "F"}.issubset({p.role for p in self.processes}):
                raise ValueError("AF-only requires A/F replicas and an optional top router")
            joined = " ".join(p.command for p in self.processes)
            if ("--disaggregation-mode" in joined or "--disaggregation-transfer-backend" in joined
                    or "--pd-disaggregation" in joined):
                raise ValueError("AF-only leaked PD semantics")

    def manifest(self) -> dict:
        return {"architecture": self.architecture, "endpoint": self.endpoint,
                "endpoints": self.endpoints,
                "routing_policy": self.routing_policy.value,
                "processes": [asdict(p) for p in self.processes],
                "cleanup": self.cleanup, "unlock": self.unlock,
                "cluster": self.cluster, "container": self.cluster.get("container"),
                "runtime_options": self.runtime_options,
                "afd_pool_coordinator": self.runtime_options.get(
                    "afd_pool_coordinator"
                ),
                "afd_pool_edges": self.runtime_options.get("afd_pool_edges", []),
                "pre_actions": self.pre_actions, "run_tag": self.run_tag}

class RemoteExecutor:
    def __init__(self, container="moe-energy", dry_run=False, timeout=60,
                 emit: Callable[[str], None]=print):
        self.container = container
        self.dry_run = dry_run
        self.timeout = timeout
        self.emit = emit

    def argv(self, host: str, cmd: str):
        if host in {"127.0.0.1", "localhost", "::1"}:
            return ["docker", "exec", self.container, "bash", "-lc", cmd]
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host,
                f"docker exec {shlex.quote(self.container)} bash -lc {shlex.quote(cmd)}"]

    def run(self, host: str, cmd: str, check=True, quiet=False, timeout=None):
        argv = self.argv(host, cmd)
        if not quiet:
            self.emit(("DRY-RUN " if self.dry_run else "RUN ") + shlex.join(argv))
        if self.dry_run:
            return subprocess.CompletedProcess(argv, 0, "", "")
        return subprocess.run(argv, text=True, capture_output=True,
                              timeout=self.timeout if timeout is None else timeout,
                              check=check)

    def host_run(self, host: str, cmd: str, check=True, quiet=False, timeout=None):
        if host in {"127.0.0.1", "localhost", "::1"}:
            argv = ["bash", "-lc", cmd]
        else:
            argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host,
                    f"bash -lc {shlex.quote(cmd)}"]
        if not quiet:
            self.emit(("DRY-RUN " if self.dry_run else "HOST RUN ") + shlex.join(argv))
        if self.dry_run:
            return subprocess.CompletedProcess(argv, 0, "", "")
        return subprocess.run(argv, text=True, capture_output=True,
                              timeout=self.timeout if timeout is None else timeout,
                              check=check)

    def participating_hosts(self, plan: DeploymentPlan):
        return list(dict.fromkeys(spec.host for spec in plan.processes))

    def deep_cleanup(self, plan: DeploymentPlan):
        command = plan.cluster.get("host_deep_cleanup_cmd")
        if not command:
            return
        strict = plan.cluster.get("host_deep_cleanup_strict", True)
        for host in self.participating_hosts(plan):
            result = self.host_run(host, command, check=False, timeout=180)
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "").strip()
                suffix = f": {detail}" if detail else ""
                message = f"host deep cleanup failed on {host} (rc={result.returncode}){suffix}"
                if strict:
                    raise RuntimeError(message)
                self.emit(f"BEST-EFFORT {message}")

    def verify_gpu_compute_apps_zero(self, plan: DeploymentPlan):
        """Reject compute applications only on GPUs allocated by this plan."""
        hosts = {node["name"]: node["host"] for node in plan.cluster.get("nodes", [])}
        hosts.update({spec.node: spec.host for spec in plan.processes})
        for node, gpus in plan.used_gpu_map().items():
            host = hosts[node]
            indices = " ".join(map(str, gpus))
            command = (
                f"busy=; for gpu in {indices}; do "
                "apps=$(nvidia-smi -i $gpu --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null); "
                "compact=$(printf '%s' \"$apps\" | tr -d '[:space:]'); "
                "if test -n \"$compact\"; then "
                "printf 'GPU %s busy: %s\\n' \"$gpu\" \"$apps\"; busy=1; fi; "
                "done; test -z \"$busy\""
            )
            result = self.host_run(host, command, check=False, quiet=True)
            if result.returncode != 0:
                detail = (result.stdout or result.stderr or "").strip()
                raise RuntimeError(
                    f"planned GPU compute applications remain on {host}: {detail}"
                )


    def _plan_ports(self, plan: DeploymentPlan):
        by_host = {}
        for spec in plan.processes:
            ports = [spec.port, spec.bootstrap_port, spec.nccl_port, *spec.internal_ports]
            by_host.setdefault(spec.host, set()).update(port for port in ports if port is not None)
        return by_host

    def verify_ports_free(self, plan: DeploymentPlan):
        """Fail before launch if cleanup left any planned TCP listener behind."""
        for host, ports in self._plan_ports(plan).items():
            values = " ".join(str(port) for port in sorted(ports))
            command = (f"busy=; for p in {values}; do "
                       "lines=$(ss -ltnpH \"sport = :$p\" 2>/dev/null || true); "
                       "pids=$(fuser ${p}/tcp 2>/dev/null || true); "
                       "if test -n \"$lines$pids\"; then "
                       "printf 'port %s busy: %s %s\\n' \"$p\" \"$lines\" \"$pids\"; busy=1; fi; "
                       "done; test -z \"$busy\"")
            scopes = [("container", self.run)]
            if plan.cluster.get("host_network", False):
                scopes.insert(0, ("host", self.host_run))
            for scope, runner in scopes:
                result = runner(host, command, check=False, quiet=True)
                if result.returncode != 0:
                    detail = (result.stdout or result.stderr or "").strip()
                    raise RuntimeError(
                        f"planned port still occupied before launch on {host} "
                        f"({scope}): {detail or values}"
                    )

    def preclean_plan(self, plan: DeploymentPlan):
        run_dirs = {}
        for spec in plan.processes:
            if spec.log_path:
                run_dirs.setdefault(spec.host, str(Path(spec.log_path).parent))
        by_host = {}
        for spec in plan.processes:
            by_host.setdefault(spec.host, set()).add(spec.port)
            if spec.bootstrap_port is not None:
                by_host[spec.host].add(spec.bootstrap_port)
            if spec.nccl_port is not None:
                by_host[spec.host].add(spec.nccl_port)
            by_host[spec.host].update(spec.internal_ports)
        for host, ports in by_host.items():
            values = " ".join(str(port) for port in sorted(ports))
            if plan.cluster.get("coexistence_mode", False):
                run_dir = run_dirs.get(host)
                if run_dir:
                    self.run(host, f"rm -rf -- {shlex.quote(run_dir)} && mkdir -p -- {shlex.quote(run_dir)}", check=False)
                continue
            command = (f"for p in {values}; do "
                       "pids=$(fuser ${p}/tcp 2>/dev/null || true); "
                       "if test -z \"$pids\"; then pids=$(ss -ltnpH \"sport = :$p\" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u | tr '\\n' ' '); fi; "
                       "test -z \"$pids\" || kill -TERM $pids 2>/dev/null || true; "
                       "done; sleep 1; "
                       f"for p in {values}; do "
                       "pids=$(fuser ${p}/tcp 2>/dev/null || true); "
                       "if test -z \"$pids\"; then pids=$(ss -ltnpH \"sport = :$p\" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u | tr '\\n' ' '); fi; "
                       "test -z \"$pids\" || kill -KILL $pids 2>/dev/null || true; done")
            if plan.cluster.get("host_network", False):
                self.host_run(host, command, check=False)
            run_dir = run_dirs.get(host)
            if run_dir:
                command = (f"rm -rf -- {shlex.quote(run_dir)} && "
                           f"mkdir -p -- {shlex.quote(run_dir)}; " + command)
            self.run(host, command, check=False)

    def launch(self, spec: ProcessSpec):
        self.run(spec.host, spec.command)

    def verify_alive(self, spec: ProcessSpec):
        pid_file = spec.pid_path
        command = (f"test -s {pid_file} && pid=$(cat {pid_file}) && "
                   "kill -0 -- -$pid 2>/dev/null")
        result = self.run(spec.host, command, check=False, quiet=True)
        if result.returncode != 0:
            raise RuntimeError(f"launch failed: {spec.role} {spec.host}, pid not alive")

    def verify_gpu_process_binding(self, spec: ProcessSpec, timeout_s=180, interval_s=1):
        """Require a run-token-owned compute process on every planned GPU."""
        if not spec.gpus:
            return
        token = shlex.quote(str(Path(spec.pid_path).parent.name))
        role = shlex.quote(spec.role)
        expected_visible = shlex.quote(",".join(map(str, spec.gpus)))
        process_scan = (
            "owned=; candidates=; ancestry=; "
            "launcher_pgid=$(ps -o pgid= -p $launcher 2>/dev/null | tr -d ' '); "
            "test -n \"$launcher_pgid\" && test \"$launcher_pgid\" = \"$launcher\" || "
            "{ echo 'AFLEX_BARRIER launcher is not process-group leader pid='$launcher' pgid='$launcher_pgid; exit 21; }; "
            "group_pids=$(ps -eo pid=,pgid= | awk -v pgid=\"$launcher_pgid\" '$2 == pgid {print $1}'); "
            "for pid in $group_pids; do proc=/proc/$pid; test -r $proc/environ || continue; "
            "env=$(tr '\\0' '\\n' < $proc/environ 2>/dev/null || true); "
            f"printf '%s\\n' \"$env\" | grep -Fxq AFLEX_RUN_ID={token} || continue; "
            f"printf '%s\\n' \"$env\" | grep -Fxq AFLEX_COMPONENT_ROLE={role} || continue; "
            f"printf '%s\\n' \"$env\" | grep -Fxq AFLEX_COMPONENT_PORT={int(spec.port)} || continue; "
            "candidates=\"$candidates $pid\"; owned=\"$owned $pid\"; "
            "ppid=$(awk '/^PPid:/{print $2}' $proc/status 2>/dev/null); sid=$(ps -o sid= -p $pid 2>/dev/null | tr -d ' '); ancestry=\"$ancestry pid=$pid,ppid=$ppid,sid=$sid,pgid=$launcher_pgid\"; done; "
        )
        checks = []
        for gpu in spec.gpus:
            checks.append(
                f"gpu={int(gpu)}; uuid=$(nvidia-smi -i $gpu --query-gpu=uuid --format=csv,noheader,nounits 2>/dev/null); "
                "nvml=$(nvidia-smi -i $gpu --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null); "
                "test -n \"$uuid\" && test -n \"$owned\" || { "
                f"echo 'AFLEX_BARRIER expected run={token} role={role} port={int(spec.port)} gpu='$gpu' uuid='$uuid' visible={expected_visible}' "
                f"'pid_file={shlex.quote(spec.pid_path)} launcher='$launcher' nvml_pid='$nvml' pid_namespace_note=NVML_may_report_host_PID ancestry='$ancestry' token_candidates='$candidates' owned='$owned; exit 22; }}"
            )
        command = (
            f"test -s {shlex.quote(spec.pid_path)} || {{ echo 'AFLEX_BARRIER missing pid file {shlex.quote(spec.pid_path)}'; exit 20; }}; "
            f"launcher=$(cat {shlex.quote(spec.pid_path)}); kill -0 -- -$launcher 2>/dev/null || {{ echo 'AFLEX_BARRIER launcher dead pid='$launcher; exit 21; }}; "
            + process_scan
            + "; ".join(checks)
        )
        if self.dry_run:
            return
        deadline = time.monotonic() + float(timeout_s)
        result = None
        while time.monotonic() < deadline:
            result = self.run(spec.host, command, check=False, quiet=True)
            if result.returncode == 0:
                return
            if result.returncode in {20, 21}:
                break
            time.sleep(interval_s)
        detail = ((result.stdout or "") + (result.stderr or "")).strip() if result else ""
        raise RuntimeError(
            f"startup GPU ownership barrier failed: {spec.role} {spec.host} GPUs={spec.gpus} "
            f"rc={getattr(result, 'returncode', None)} diagnostic={detail}"
        )

    def _startup_status(self, spec: ProcessSpec):
        pid_file = spec.pid_path
        log_file = spec.log_path
        fatal = ("address already in use|bind(ing)? failed|failed to bind|"
                 "cannot assign requested address|traceback \(most recent call last\)")
        command = (f"test -s {pid_file} || {{ echo 'missing pid file'; exit 20; }}; "
                   f"pid=$(cat {pid_file}); "
                   "kill -0 -- -$pid 2>/dev/null || { echo 'process group exited'; exit 21; }; "
                   f"if test -f {log_file} && tail -n 80 {log_file} | "
                   f"grep -Eqi {shlex.quote(fatal)}; then "
                   f"echo 'fatal startup log:'; tail -n 80 {log_file}; exit 22; fi")
        return self.run(spec.host, command, check=False, quiet=True)

    def verify_port_owner(self, spec: ProcessSpec):
        """Verify the new process group, including router children, owns its port."""
        pid_file = spec.pid_path
        command = (f"pid=$(cat {pid_file}); pgid=$(ps -o pgid= -p $pid | tr -d ' '); "
                   f"line=$(ss -ltnpH \"sport = :{spec.port}\" 2>/dev/null); "
                   "test -n \"$line\" || { echo 'no listener'; exit 1; }; "
                   "owners=$(printf '%s' \"$line\" | grep -oE 'pid=[0-9]+' | cut -d= -f2); "
                   "test -n \"$owners\" || { echo \"listener owner unavailable: $line\"; exit 1; }; "
                   "for owner in $owners; do opgid=$(ps -o pgid= -p $owner | tr -d ' '); "
                   "test -n \"$pgid\" && test \"$opgid\" = \"$pgid\" && exit 0; done; "
                   "echo \"listener not owned by process group $pgid: $line\"; exit 1")
        result = self.run(spec.host, command, check=False, quiet=True)
        if result.returncode != 0:
            detail = (result.stdout or result.stderr or "").strip()
            raise RuntimeError(f"launch port ownership failed: {spec.role} "
                               f"{spec.host}:{spec.port}: {detail}")

    def wait_log_patterns(self, spec: ProcessSpec, patterns=None, timeout_s=None, interval_s=2):
        patterns = list(spec.ready_log_patterns if patterns is None else patterns)
        if not patterns:
            return
        timeout_s = float(spec.ready_timeout_s if timeout_s is None and spec.ready_timeout_s is not None
                          else timeout_s if timeout_s is not None else 600)
        if self.dry_run:
            self.emit(f"DRY-RUN READY-LOG {spec.host}:{spec.log_path} patterns={patterns!r}")
            return
        deadline = time.monotonic() + timeout_s
        quoted = " ".join(shlex.quote(pattern) for pattern in patterns)
        command = (f"test -f {shlex.quote(spec.log_path)} || exit 1; "
                   f"for pattern in {quoted}; do "
                   f'grep -Fq -- "$pattern" {shlex.quote(spec.log_path)} || exit 1; done')
        while time.monotonic() < deadline:
            startup = self._startup_status(spec)
            if startup.returncode != 0:
                detail = (startup.stdout or startup.stderr or "").strip()
                raise RuntimeError(f"startup failed while waiting for ready marker: "
                                   f"{spec.role} {spec.host}:{spec.port}: {detail}")
            result = self.run(spec.host, command, check=False, quiet=True)
            if result.returncode == 0:
                return
            time.sleep(interval_s)
        raise TimeoutError(f"ready log timeout: {spec.role} {spec.host}:{spec.port}; "
                           f"patterns={patterns!r}")

    def wait_health(self, spec: ProcessSpec, timeout_s=600, interval_s=2):
        if spec.ready_command is None and spec.health_path is None:
            return
        if spec.ready_command is not None:
            command = spec.ready_command
            description = f"command for {spec.role} {spec.host}:{spec.port}"
        else:
            command = (f"curl -fsS --max-time 2 http://{spec.host}:{spec.port}"
                       f"{spec.health_path} >/dev/null")
            description = f"http://{spec.host}:{spec.port}{spec.health_path}"
        if self.dry_run:
            self.emit(f"DRY-RUN HEALTH {description}")
            return
        deadline = time.monotonic() + timeout_s
        result = None
        while time.monotonic() < deadline:
            startup = self._startup_status(spec)
            if startup.returncode != 0:
                detail = (startup.stdout or startup.stderr or "").strip()
                raise RuntimeError(f"startup failed: {spec.role} {spec.host}:"
                                   f"{spec.port}: {detail}")
            result = self.run(spec.host, command, check=False, quiet=True)
            if result.returncode == 0:
                self.verify_port_owner(spec)
                return
            time.sleep(interval_s)
        detail = ""
        if result is not None:
            output = (result.stderr or result.stdout or "").strip()
            detail = f"; last returncode={result.returncode}"
            if output:
                detail += f"; output={output}"
        message = f"health timeout: {spec.role} {spec.host}:{spec.port}{detail}"
        self.emit(f"HEALTH FAILED {message}")
        raise TimeoutError(message)

    def warmup(self, endpoint: str, requests=1, timeout_s=180, interval_s=2,
               curl_timeout_s=120, max_new_tokens=4, input_len=8):
        host_port = endpoint.split("//", 1)[-1]
        host = host_port.rsplit(":", 1)[0]
        payload = shlex.quote(json.dumps({"input_ids": [1000] * int(input_len),
            "sampling_params": {"max_new_tokens": int(max_new_tokens), "temperature": 0.0}}))
        command = (f"for i in $(seq 1 {int(requests)}); do "
                   f"response=$(curl -sS --max-time {float(curl_timeout_s):g} -w '\\n%{{http_code}}' "
                   "-H 'Content-Type: application/json' "
                   f"-d {payload} {endpoint.rstrip('/')}/generate); "
                   "curl_rc=$?; status=${response##*$'\\n'}; "
                   "body=${response%$'\\n'*}; "
                   "printf '%s\\nHTTP status: %s' \"$body\" \"$status\"; "
                   "test $curl_rc -eq 0 && test \"$status\" = 200 || exit 1; "
                   "done")
        deadline = time.monotonic() + timeout_s
        outer_timeout_s = float(curl_timeout_s) + 10
        result = None
        timeout_error = None
        while True:
            try:
                result = self.run(host, command, check=False, quiet=True,
                                  timeout=outer_timeout_s)
                timeout_error = None
                if result.returncode == 0:
                    return
            except subprocess.TimeoutExpired as exc:
                result = None
                timeout_error = exc
            if time.monotonic() >= deadline:
                break
            time.sleep(interval_s)

        if timeout_error is not None:
            outputs = []
            for value in (timeout_error.stdout, timeout_error.stderr):
                if isinstance(value, bytes):
                    value = value.decode(errors="replace")
                if value and value.strip():
                    outputs.append(value.strip())
            detail = f"; output={' | '.join(outputs)}" if outputs else ""
            message = (f"warmup timeout: {endpoint}; last attempt exceeded "
                       f"outer timeout={timeout_error.timeout}s{detail}")
        else:
            outputs = [value.strip() for value in (result.stdout, result.stderr)
                       if value and value.strip()]
            detail = f"; output={' | '.join(outputs)}" if outputs else ""
            message = (f"warmup timeout: {endpoint}; last returncode="
                       f"{result.returncode}{detail}")
        self.emit(f"WARMUP FAILED {message}")
        raise TimeoutError(message)

    def collect_logs(self, plan: DeploymentPlan, output_dir: Path):
        if self.dry_run:
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, spec in enumerate(plan.processes):
            remote = spec.log_path
            result = self.run(spec.host, f"test -f {remote} && cat {remote} || true", check=False)
            (output_dir / f"{i:02d}_{spec.node}_{spec.role}_{spec.port}.log").write_text(result.stdout or "")

    def cleanup_plan(self, plan: DeploymentPlan):
        for host, cmd in reversed(plan.cleanup):
            self.run(host, cmd, check=False)
        for host, cmd in reversed(plan.unlock):
            self.run(host, cmd, check=False)

def execute_lifecycle(plan, executor, body, artifact_dir: Path | None=None,
                      health_timeout_s=None, warmup_requests=1,
                      runtime_options=None):
    plan.validate()
    if executor.dry_run:
        raise ValueError("execute_lifecycle cannot execute in dry-run mode")
    options = dict(getattr(plan, "runtime_options", {}) or {})
    options.update(runtime_options or {})
    if options.get("debug_fast_fail"):
        presets = {"health_timeout_s": 90, "warmup_timeout_s": 30,
                   "warmup_curl_timeout_s": 20, "request_timeout_s": 30,
                   "warmup_max_new_tokens": 2}
        options = {**presets, **options}
    health_timeout_s = float(health_timeout_s if health_timeout_s is not None
                             else options.get("health_timeout_s", 600))
    warmup_timeout_s = float(options.get("warmup_timeout_s", 180))
    warmup_curl_timeout_s = float(options.get("warmup_curl_timeout_s", 120))
    warmup_max_new_tokens = int(options.get("warmup_max_new_tokens", 4))
    warmup_input_len = int(options.get("warmup_input_len", 8))
    warmup_parallel = bool(options.get("warmup_parallel", False))
    wait_after_warmup_s = float(options.get("wait_after_warmup_s", 0))
    skip_warmup = bool(options.get("skip_warmup", False))
    launched = []
    try:
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "deployment_manifest.json").write_text(
                json.dumps(plan.manifest(), indent=2) + "\n")
        executor.deep_cleanup(plan)
        executor.preclean_plan(plan)
        executor.verify_ports_free(plan)
        executor.verify_gpu_compute_apps_zero(plan)
        for host, command in plan.pre_actions:
            executor.run(host, command)
        def run_parallel(stage_processes, operation):
            with ThreadPoolExecutor(max_workers=len(stage_processes)) as pool:
                futures = [pool.submit(operation, process) for process in stage_processes]
                try:
                    for future in as_completed(futures):
                        future.result()
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise

        def warm(endpoint):
            executor.warmup(endpoint, warmup_requests,
                            timeout_s=warmup_timeout_s,
                            curl_timeout_s=warmup_curl_timeout_s,
                            max_new_tokens=warmup_max_new_tokens,
                            input_len=warmup_input_len)

        def wait_ready(process):
            executor.verify_alive(process)
            executor.wait_health(process, health_timeout_s)
            if process.ready_log_patterns:
                executor.wait_log_patterns(process)
            if process.metadata.get("activation_warmup") and not skip_warmup:
                warm(f"http://{process.host}:{process.port}")

        stages = sorted({process.startup_stage for process in plan.processes})
        for stage in stages:
            stage_processes = [process for process in plan.processes
                               if process.startup_stage == stage]
            launched.extend(stage_processes)
            run_parallel(stage_processes, executor.launch)
            run_parallel(stage_processes, executor.verify_gpu_process_binding)
            run_parallel(stage_processes, wait_ready)
        gpu_processes = [process for process in plan.processes if process.gpus]
        if gpu_processes:
            run_parallel(gpu_processes, executor.verify_gpu_process_binding)
        if not skip_warmup:
            endpoints = plan.endpoints
            if warmup_parallel and len(endpoints) > 1:
                with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
                    futures = [pool.submit(warm, endpoint) for endpoint in endpoints]
                    for future in as_completed(futures):
                        future.result()
            else:
                for endpoint in endpoints:
                    warm(endpoint)
            if wait_after_warmup_s > 0:
                time.sleep(wait_after_warmup_s)
        argument = plan.endpoint if plan.routing_policy == RoutingPolicy.SINGLE else plan.handle
        return body(argument)
    finally:
        try:
            if artifact_dir is not None and launched:
                executor.collect_logs(plan, artifact_dir / "logs")
        finally:
            try:
                executor.deep_cleanup(plan)
            finally:
                executor.cleanup_plan(plan)
                executor.verify_gpu_compute_apps_zero(plan)
