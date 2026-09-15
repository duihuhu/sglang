from __future__ import annotations

from dataclasses import dataclass
import shlex
import re

from .base import DeploymentPlan, ProcessSpec

PY = "/usr/bin/python3"

SERVER_EXTRA_ENV_ALLOWLIST = frozenset({
    "AFD_NULL_DEBUG",
    "CUDA_LAUNCH_BLOCKING",
})


def normalize_extra_env(extra_env):
    """Validate point-provided server debug environment and quote its values."""
    if extra_env is None:
        return ""
    if not isinstance(extra_env, dict):
        raise ValueError("extra_env must be a dictionary")
    unknown = set(extra_env) - SERVER_EXTRA_ENV_ALLOWLIST
    if unknown:
        raise ValueError(
            f"extra_env contains unsupported variables: {', '.join(sorted(unknown))}"
        )
    assignments = []
    for key, value in extra_env.items():
        if not isinstance(value, (str, int, bool)):
            raise ValueError(f"extra_env value for {key} must be a string, integer, or boolean")
        normalized = "1" if value is True else "0" if value is False else str(value)
        assignments.append(f"{key}={shlex.quote(normalized)}")
    return " ".join(assignments)


def node_map(cluster):
    return {n["name"]: n for n in cluster["nodes"]}


def _pythonpath_export(python_source):
    if not python_source:
        return ""
    source = shlex.quote(python_source)
    return f"export PYTHONPATH={source}:${{PYTHONPATH:-}}; "


def _safe_component(value):
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"unsafe run path component: {value!r}")
    return value


def process_paths(run_tag, node, role, port):
    run_tag = _safe_component(run_tag)
    node = _safe_component(node)
    role = _safe_component(role)
    base = f"/tmp/aflex_bench/{run_tag}"
    stem = f"{node}_{role}_{int(port)}"
    return f"{base}/{stem}.log", f"{base}/{stem}.pid"


def fingerprint_command(python_source, log_path):
    source = python_source or ""
    qwen = f"{source.rstrip('/')}/sglang/srt/models/qwen3.py" if source else ""
    script = (
        "import hashlib,os,sglang; "
        "p=" + repr(qwen) + " or os.path.join(os.path.dirname(sglang.__file__),'srt','models','qwen3.py'); "
        "print('AFLEX_SOURCE sglang_file='+str(sglang.__file__)); "
        "print('AFLEX_SOURCE qwen3_path='+p); "
        "print('AFLEX_SOURCE qwen3_sha256='+"
        "(hashlib.sha256(open(p,'rb').read()).hexdigest() if p and os.path.isfile(p) else 'missing')); "
        "print('AFLEX_SOURCE qwen3_mtime='+"
        "(str(os.path.getmtime(p)) if p and os.path.isfile(p) else 'missing')); "
        "print('AFLEX_SOURCE qwen3_marker='+"
        "('|'.join(line.strip() for line in open(p,errors='replace') if 'AFD' in line)[:1000] "
        "if p and os.path.isfile(p) else 'missing'))"
    )
    return f"{PY} -c {shlex.quote(script)} > {shlex.quote(log_path)} 2>&1"


def command(model, tp, host, port, gpus, log_path, pid_path, extra="", env="",
            visible_gpus=None, python_source=None, run_tag="dryrun", role="SERVER"):
    visible = ",".join(map(str, visible_gpus if visible_gpus is not None else gpus))
    environment = " ".join(x for x in [
        "SGLANG_DISABLE_REQUEST_LOGGING=true",
        "FLASHINFER_DISABLE_VERSION_CHECK=1", env,
        f"AFLEX_RUN_ID={shlex.quote(_safe_component(run_tag))}",
        f"AFLEX_COMPONENT_ROLE={shlex.quote(_safe_component(role))}",
        f"AFLEX_COMPONENT_PORT={int(port)}",
        f"CUDA_VISIBLE_DEVICES={visible}"] if x)
    prefix = _pythonpath_export(python_source)
    fingerprint = fingerprint_command(python_source, log_path)
    launch = (f"export {environment}; setsid prlimit --memlock=unlimited:unlimited "
              f"{PY} -m sglang.launch_server --model-path {model} --tp {tp} "
              f"--host {host} --port {port} --disable-cuda-graph "
              "--disable-piecewise-cuda-graph --disable-radix-cache "
              f"--skip-server-warmup {extra} >> {shlex.quote(log_path)} 2>&1 "
              f"< /dev/null & echo $! > {shlex.quote(pid_path)}")
    return prefix + fingerprint + "; " + launch


def make_server(role, node, refs, port, model, tp, extra="", env="",
                bootstrap_port=None, visible_gpus=None, health_path="/health",
                startup_stage=0, python_source=None, run_tag="dryrun"):
    gpus = [x.gpu for x in refs]
    log_path, pid_path = process_paths(run_tag, node["name"], role, port)
    return ProcessSpec(
        role, node["name"], node["host"], gpus, port,
        command(model, tp, node["host"], port, gpus, log_path, pid_path,
                extra, env, visible_gpus, python_source, run_tag, role),
        startup_stage=startup_stage, health_path=health_path,
        bootstrap_port=bootstrap_port, log_path=log_path, pid_path=pid_path,
    )


def make_router(role, owner, port, args, startup_stage=0, python_source=None,
                run_tag="dryrun"):
    log_path, pid_path = process_paths(run_tag, owner["name"], role, port)
    prefix = _pythonpath_export(python_source)
    fingerprint = fingerprint_command(python_source, log_path)
    identity = (f"export AFLEX_RUN_ID={shlex.quote(run_tag)} "
                f"AFLEX_COMPONENT_ROLE={shlex.quote(role)} "
                f"AFLEX_COMPONENT_PORT={int(port)}")
    launch = (prefix + fingerprint + "; " + identity + "; " +
              f"setsid {PY} -m sglang_router.launch_router {args} "
              f"--host {owner['host']} --port {port} >> {shlex.quote(log_path)} "
              f"2>&1 < /dev/null & echo $! > {shlex.quote(pid_path)}")
    return ProcessSpec(role, owner["name"], owner["host"], [], port, launch,
                       startup_stage=startup_stage, log_path=log_path,
                       pid_path=pid_path)


@dataclass(frozen=True)
class PlanContext:
    cluster: dict

    @property
    def run_tag(self):
        return self.cluster.get("_run_tag", "dryrun")

    @property
    def python_source(self):
        return self.cluster.get("python_source")

    def server(self, *args, **kwargs):
        return make_server(*args, **kwargs, python_source=self.python_source, run_tag=self.run_tag)

    def router(self, *args, **kwargs):
        return make_router(*args, **kwargs, python_source=self.python_source, run_tag=self.run_tag)

    def finish(self, architecture, processes, endpoint, routing_policy="single"):
        return final_plan(architecture, processes, endpoint, self.cluster,
                          routing_policy, self.run_tag)


def final_plan(architecture, processes, endpoint, cluster, routing_policy="single", run_tag="dryrun"):
    by_host = {}
    for process in processes:
        by_host.setdefault(process.host, []).append(process)
    cleanup = []
    unlock = []
    for host, host_processes in by_host.items():
        pid_files = " ".join(shlex.quote(p.pid_path) for p in host_processes)
        token = shlex.quote(_safe_component(run_tag))
        cleanup.append((host,
            f"token={token}; for f in {pid_files}; do "
            "if test -s $f; then pid=$(cat $f); "
            "owned=; for proc in /proc/[0-9]*; do p=${proc#/proc/}; "
            "test -r $proc/environ || continue; "
            "tr '\\0' '\\n' < $proc/environ 2>/dev/null | grep -Fxq AFLEX_RUN_ID=$token || continue; "
            "pgid=$(ps -o pgid= -p $p 2>/dev/null | tr -d ' '); test \"$pgid\" = \"$pid\" && owned=1 && break; done; "
            "if test -n \"$owned\"; then kill -TERM -- -$pid 2>/dev/null || true; "
            "for i in $(seq 1 20); do kill -0 -- -$pid 2>/dev/null || break; sleep 0.25; done; "
            "kill -KILL -- -$pid 2>/dev/null || true; fi; fi; rm -f $f; done"))
        # In coexistence mode a GPU-wide clock reset can alter another owner's job.
        gpus = sorted({gpu for process in host_processes for gpu in process.gpus})
        if gpus and not cluster.get("coexistence_mode", False):
            indices = ",".join(map(str, gpus))
            unlock.append((host, f"nvidia-smi -i {indices} --reset-gpu-clocks >/dev/null 2>&1 || true"))
    cluster_info = {
        "container": cluster.get("container"),
        "container_root": cluster.get("container_root"),
        "host_root": cluster.get("host_root"),
        "python_source": cluster.get("python_source"),
        "host_deep_cleanup_cmd": cluster.get("host_deep_cleanup_cmd"),
        "host_deep_cleanup_strict": cluster.get("host_deep_cleanup_strict", True),
        "host_network": cluster.get("host_network", False),
        "coexistence_mode": cluster.get("coexistence_mode", False),
        "nodes": [{"name": node["name"], "host": node["host"]}
                  for node in cluster.get("nodes", [])],
    }
    plan = DeploymentPlan(architecture, processes, endpoint, cleanup, unlock,
                          cluster_info, routing_policy, run_tag=run_tag)
    plan.validate()
    return plan
