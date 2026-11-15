"""Explicit transport and ownership-scoped process lifecycle."""

import json
import shlex
import subprocess
import time
import uuid

NODE1_IP = NODE2_IP = "UNCONFIGURED"
MODEL = PYTHON = LOG_C = ENERGY_MODEL_DIR_V1 = "UNCONFIGURED"
ROUTER_PORT = 0
GPU_NIC = {}
CONFIG = None
DRY = False
COMMANDS = []
OWNED = []
RUN_ID = uuid.uuid4().hex
_LOCAL_NODE1 = False
NODE_EXTRA_IP = ""
COMMON_BENCH_SERVER_FLAGS = "--mem-fraction-static 0.85 --disable-cuda-graph --disable-piecewise-cuda-graph --skip-server-warmup --disable-radix-cache "


def _is_local(host):
    return False


def _ssh(host, cmd):
    node = next(n for n in CONFIG["nodes"] if n["host"] == host)
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        f"{node['ssh_user']}@{host}",
        cmd,
    ]


def execute(host, shell, **kwargs):
    node = next(n for n in CONFIG["nodes"] if n["host"] == host)
    inner = (
        "cd "
        + shlex.quote(CONFIG["paths"]["repo"])
        + " && export DVFS_CTRL_LIB="
        + shlex.quote(CONFIG["paths"]["dvfs_library"])
        + " && "
        + shell
    )
    argv = _ssh(
        host, shlex.join(["docker", "exec", node["container"], "bash", "-lc", inner])
    )
    COMMANDS.append({"host": host, "shell": shell, "argv": argv})
    if DRY:
        return subprocess.CompletedProcess(argv, 0, "{}", "")
    return subprocess.run(argv, timeout=kwargs.pop("timeout", 120), **kwargs)


def dexec_on_host(host, shell):
    # Legacy launchers emit one background command per call. Strip only their
    # redirection and launch it in an owned session instead.
    if shell.rstrip().endswith("&"):
        body = shell.rsplit(" > ", 1)[0].strip().replace("setsid ", "")
        import re

        if "sglang_router.launch_router" in body:
            metric_port = CONFIG["ports"]["router"] + 100 + len(COMMANDS)
            body += f" --prometheus-port {metric_port}"
        match = re.search(r"--port (\d+)", body)
        if match:
            ensure_ports_free(host, [int(match.group(1))])
        name = f"process_{len(COMMANDS)}"
        log = f"{LOG_C}/{RUN_ID}_{name}.log"
        token = f"AFLEX_BENCH_OWNER={RUN_ID}"
        wrapper = f"exec env {token} bash -lc {shlex.quote(body)}"
        launch = f"mkdir -p {shlex.quote(LOG_C)}; setsid bash -lc {shlex.quote(wrapper)} > {shlex.quote(log)} 2>&1 < /dev/null & echo $!"
        ret = execute(host, launch, check=True, capture_output=True, text=True)
        if not DRY:
            pid = int(ret.stdout.strip().splitlines()[-1])
            OWNED.append((host, pid))
    else:
        execute(host, shell, check=True)


def dexec_local(shell):
    dexec_on_host(NODE1_IP, shell)


def dexec_remote(shell):
    dexec_on_host(NODE2_IP, shell)


def _launch(host, cmd, logname):
    dexec_on_host(host, cmd + f" > {LOG_C}/{logname}.log 2>&1 < /dev/null &")


def cleanup_all():
    import inspect

    from .process_cleanup import cleanup_session

    errors = []
    options = CONFIG.get("cleanup", {})
    for host, pid in reversed(OWNED[:]):
        code = (
            inspect.getsource(cleanup_session)
            + "\n"
            + (
                f"cleanup_session({pid!r}, {RUN_ID!r}, "
                f"term_timeout={options.get('term_timeout_s', 15)!r}, "
                f"force={options.get('force_kill', False)!r}, "
                f"kill_timeout={options.get('kill_timeout_s', 5)!r})"
            )
        )
        try:
            execute(
                host,
                f"{shlex.quote(PYTHON)} -c {shlex.quote(code)}",
                check=True,
                timeout=options.get("term_timeout_s", 15)
                + options.get("kill_timeout_s", 5)
                + 30,
            )
            # The helper returns only after all live session members have exited.
            OWNED.remove((host, pid))
        except Exception as exc:
            errors.append(f"{host} session {pid}: {exc}")
    if errors:
        raise RuntimeError("owned process cleanup failed: " + "; ".join(errors))


def unlock_scope(scope):
    errors = []
    for host, gpus in scope.items():
        if not gpus:
            continue
        code = f"from sglang.srt.layers.dvfs import unlock_gpus; assert unlock_gpus({list(gpus)!r}) == 0"
        try:
            execute(host, f"{shlex.quote(PYTHON)} -c {shlex.quote(code)}", check=True)
        except Exception as exc:
            errors.append(f"{host}: {exc}")
    if errors:
        raise RuntimeError("GPU unlock failed: " + "; ".join(errors))


def finalize_scope(scope):
    errors = []
    for action in (cleanup_all, lambda: unlock_scope(scope)):
        try:
            action()
        except Exception as exc:
            errors.append(str(exc))
    if errors:
        raise RuntimeError("; ".join(errors))


def ensure_ports_free(host, ports, retries=1):
    code = "import socket; " + "; ".join(
        f"s=socket.socket(); s.bind(('0.0.0.0',{int(p)})); s.close()" for p in ports
    )
    execute(host, f"{shlex.quote(PYTHON)} -c {shlex.quote(code)}", check=True)


def wait_health(host, port, timeout=600, check_model_info=False):
    if DRY:
        return True
    import urllib.request

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://{host}:{port}/"
                + ("get_model_info" if check_model_info else "health"),
                timeout=10,
            ) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def test_generate(url):
    if DRY:
        return True
    import urllib.request

    body = json.dumps(
        {
            "text": "Hello, explain quantum computing:",
            "sampling_params": {"max_new_tokens": 16, "temperature": 0.0},
        }
    ).encode()
    with urllib.request.urlopen(
        urllib.request.Request(
            url + "/generate", body, {"Content-Type": "application/json"}
        ),
        timeout=240,
    ) as response:
        return "text" in json.load(response)


def get_energy_on_host(host, gpus):
    if not gpus:
        return {}
    code = (
        "import pynvml,json; pynvml.nvmlInit(); "
        f"print(json.dumps({{i:pynvml.nvmlDeviceGetTotalEnergyConsumption(pynvml.nvmlDeviceGetHandleByIndex(i)) for i in {gpus!r}}})); "
        "pynvml.nvmlShutdown()"
    )
    ret = execute(
        host,
        f"{shlex.quote(PYTHON)} -c {shlex.quote(code)}",
        check=True,
        capture_output=True,
        text=True,
    )
    return {int(k): v for k, v in json.loads(ret.stdout).items()}


def get_energy_local(gpus):
    return get_energy_on_host(NODE1_IP, gpus)


def get_energy_remote(gpus):
    return get_energy_on_host(NODE2_IP, gpus)


def lock_freq_map_on_host(host, mapping):
    code = f"from sglang.srt.layers.dvfs import lock_gpu_freq_map; assert lock_gpu_freq_map({mapping!r}) == 0"
    execute(host, f"{shlex.quote(PYTHON)} -c {shlex.quote(code)}", check=True)


def lock_freq_both(gpus, freq=None):
    for host in (NODE1_IP, NODE2_IP):
        lock_freq_map_on_host(host, {g: freq or MAX_GPU_FREQ for g in gpus})


def unlock_freq_both(gpus):
    unlock_scope({host: gpus for host in (NODE1_IP, NODE2_IP)})


def write_ib_json(mapping, node2_mapping=None):
    for host in (NODE1_IP, NODE2_IP):
        node = next(n for n in CONFIG["nodes"] if n["host"] == host)
        execute(host, "mkdir -p " + shlex.quote(LOG_C), check=True)
        payload = json.dumps(node["gpu_nic"])
        code = (
            f"from pathlib import Path; Path({IB_JSON_FILE!r}).write_text({payload!r})"
        )
        execute(host, f"{shlex.quote(PYTHON)} -c {shlex.quote(code)}", check=True)


def remote_preflight():
    """Read-only checks on both nodes, before any clock or service changes."""
    manifest = CONFIG["preflight"]
    paths = CONFIG["paths"]
    code = "\n".join(
        [
            "import os,sys,importlib.util,ctypes,json",
            "from pathlib import Path",
            "assert sys.version_info >= (3,10), 'Python >=3.10 required'",
            "modules = ['sglang', 'sglang.launch_server', 'sglang.srt.layers.dvfs', 'sglang.srt.energy.af_profile_predictor', 'sglang_router', 'sglang_router.launch_router', 'pynvml', 'aiohttp']",
            "for name in modules:",
            "    assert importlib.util.find_spec(name) is not None, 'missing module: ' + name",
            f"assert Path({paths['model']!r}).is_dir(), 'missing model directory'",
            f"assert Path({paths['model']!r}, 'config.json').is_file(), 'missing model config.json'",
            f"libraries = {manifest['native_libraries']!r}",
            "for name in libraries:",
            "    assert Path(name).is_file() and os.access(name, os.R_OK), 'missing native library: ' + name",
            "    ctypes.CDLL(name)",
            f"root = Path({paths['energy_model']!r})",
            f"required = {manifest['prediction_files']!r}",
            "for name in required:",
            "    path = root / name",
            "    assert path.is_file() and os.access(path, os.R_OK) and path.stat().st_size > 0, 'missing prediction file: ' + str(path)",
            "print(json.dumps({'status':'REMOTE_READONLY_OK'}))",
        ]
    )
    errors = []
    for node in CONFIG["nodes"]:
        try:
            execute(
                node["host"],
                f"PYTHONDONTWRITEBYTECODE=1 {shlex.quote(PYTHON)} -B -c {shlex.quote(code)}",
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception as exc:
            errors.append(f"{node['host']}: {exc}")
    if errors:
        raise RuntimeError("remote read-only preflight failed: " + "; ".join(errors))
