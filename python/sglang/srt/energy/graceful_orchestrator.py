#!/usr/bin/env python3
"""Tier1 Graceful Reload Orchestrator — drain-then-switch for zero-loss reshard.

Instead of kill-all-restart, this orchestrator:
  1. Identifies which modules have TP changes
  2. Tells router to drain those modules (stop new traffic)
  3. In parallel, starts new modules on new GPUs
  4. Waits for old modules to become idle (configurable timeout)
  5. After drain (or timeout+abort), kills old modules
  6. Tells router to activate new modules
  7. Writes ready signal

Usage (called by scheduler, not manually):
    python -m sglang.srt.energy.graceful_orchestrator --config /path/to/config.json
"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [GracefulReload] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("graceful_orchestrator")

PYTHON = os.environ.get("SGLANG_PYTHON", sys.executable)

# Import write_signal from reload_signal (same package)
try:
    from sglang.srt.energy.reload_signal import write_signal as _write_signal
except ImportError:
    # Fallback for direct module loading (e.g. in tests)
    _reload_signal_path = os.path.join(os.path.dirname(__file__), "reload_signal.py")
    if os.path.exists(_reload_signal_path):
        import importlib.util
        _spec = importlib.util.spec_from_file_location("_reload_signal", _reload_signal_path)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _write_signal = _mod.write_signal
    else:
        def _write_signal(path, status, extra=None):
            import json as _json
            from pathlib import Path as _Path
            data = {"status": status, "timestamp": time.time()}
            if extra:
                data.update(extra)
            _Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                _json.dump(data, f)


def _http_post_json(url: str, data: dict, timeout: float = 10.0) -> dict:
    """POST JSON to a URL and return parsed response."""
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _http_get_json(url: str, timeout: float = 10.0) -> dict:
    """GET a URL and return parsed JSON response."""
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def _get_pids_on_port(port: int) -> list[str]:
    result = subprocess.run(
        ["ss", "-tlnp", f"sport = :{port}"],
        capture_output=True, text=True,
    )
    return list(set(re.findall(r'pid=(\d+)', result.stdout)))


def _kill_port(port: int):
    for pid in _get_pids_on_port(port):
        subprocess.run(["kill", "-9", pid], capture_output=True)


def _graceful_kill_port(port: int, grace_timeout: float = 5.0):
    """Send SIGTERM first, then SIGKILL after grace_timeout."""
    pids = _get_pids_on_port(port)
    for pid in pids:
        subprocess.run(["kill", "-15", pid], capture_output=True)
    if not pids:
        return
    deadline = time.monotonic() + grace_timeout
    while time.monotonic() < deadline:
        remaining = _get_pids_on_port(port)
        if not remaining:
            return
        time.sleep(0.5)
    for pid in _get_pids_on_port(port):
        subprocess.run(["kill", "-9", pid], capture_output=True)


def _wait_port_ready(host: str, port: int, timeout: float = 300.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            s = socket.create_connection((host, port), timeout=2)
            s.close()
            return True
        except Exception:
            time.sleep(2)
    return False


def _wait_health(url: str, timeout: float = 180.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=5)
            return True
        except Exception:
            time.sleep(3)
    return False


def _check_idle(url: str) -> tuple[bool, int]:
    """Check if a module is idle. Returns (is_idle, inflight_count)."""
    try:
        data = _http_get_json(f"{url}/is_idle", timeout=5)
        return data.get("idle", False), data.get("inflight", -1)
    except Exception:
        return True, 0


def _abort_module(url: str):
    """Abort all inflight requests on a module via pause_generation."""
    try:
        _http_post_json(f"{url}/pause_generation", {"mode": "abort"}, timeout=10)
    except Exception as e:
        log.warning("Failed to abort module %s: %s", url, e)


def _reset_gpu_clocks(gpu_indices: list[int]):
    for idx in gpu_indices:
        subprocess.run(["nvidia-smi", "-rgc", "-i", str(idx)], capture_output=True)


def _diff_modules(old_solution: dict, new_solution: dict) -> list[str]:
    """Return list of module names whose TP changed."""
    changed = []
    for name in ("PA", "PF", "DA", "DF"):
        key = f"tp_{name.lower()}"
        if old_solution.get(key) != new_solution.get(key):
            changed.append(name)
    return changed


def _find_modules_by_names(modules: list[dict], names: list[str]) -> list[dict]:
    """Filter module configs by name."""
    name_set = set(n.upper() for n in names)
    return [m for m in modules if m["name"].upper() in name_set]


def _start_module(mod: dict, solution: dict, server_cfg: dict) -> subprocess.Popen:
    """Start a single AFD module process. Returns the Popen handle."""
    model_path = server_cfg["model_path"]
    bootstrap_port = server_cfg["bootstrap_port"]
    ib_device = server_cfg["ib_device"]
    extra_args = server_cfg.get("extra_args", [])
    dvfs_args = server_cfg.get("dvfs_args", [])
    tier1_args = server_cfg.get("tier1_args", [])
    log_dir = Path(server_cfg.get("log_dir", "/tmp/tier1_reload_logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    name = mod["name"]
    perspective = mod["perspective"]
    disagg_mode = mod["disagg_mode"]
    port = mod["port"]
    visible_gpus = mod["visible_gpus"]
    base_gpu_id = mod["base_gpu_id"]
    ucx_base = mod["ucx_base_port"]
    sched_port = mod["sched_port"]
    nvml_idx = mod["nvml_device_index"]
    ffn_host = mod.get("ffn_host")
    is_pa = mod.get("is_pa", False)

    tp_key = f"tp_{name.lower()}"
    tp = solution.get(tp_key, 1)

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = visible_gpus
    env['SGLANG_DISABLE_REQUEST_LOGGING'] = 'true'
    env['UCX_LOG_LEVEL'] = 'fatal'
    env['AFD_UCX_TLS'] = 'rc,tcp,cuda_copy,cuda_ipc'
    env['SGLANG_DISAGGREGATION_THREAD_POOL_SIZE'] = '128'
    env['SGLANG_DISAGGREGATION_QUEUE_SIZE'] = '32'
    env['SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE'] = '0'
    env['AFD_UCX_BASE_PORT'] = str(ucx_base)
    env['AFD_SCHED_PORT'] = str(sched_port)
    env['SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT'] = '600'
    env['SGLANG_DISAGGREGATION_WAITING_TIMEOUT'] = '600'
    env['AFD_NVML_DEVICE_INDEX'] = str(nvml_idx)
    env['AFD_IPC_SYNC_MODE'] = 'ipc_event'
    env['AFD_IPC_PEER_DEVICE'] = str(mod.get("peer_device", 1 - base_gpu_id))
    if ffn_host:
        env['AFD_UCX_FFN_HOST'] = ffn_host

    cmd = [
        PYTHON, '-m', 'sglang.launch_server',
        '--model-path', model_path,
        '--tp', str(tp),
        '--host', '127.0.0.1', '--port', str(port),
        '--afd-perspective', perspective,
        '--afd-comm-backend', 'ipc_cpp',
        '--mem-fraction-static', str(server_cfg.get("mem_fraction", "0.85")),
        '--base-gpu-id', str(base_gpu_id),
        '--disaggregation-mode', disagg_mode,
        '--disaggregation-transfer-backend', 'mooncake',
        '--disaggregation-bootstrap-port', str(bootstrap_port),
        '--disaggregation-ib-device', ib_device,
    ] + extra_args + dvfs_args

    if is_pa and tier1_args:
        cmd += tier1_args

    fh = open(log_dir / f"graceful_{name}.log", 'w')
    p = subprocess.Popen(
        cmd, env=env, stdout=fh, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log.info("  Started %s (tp=%d, port=%d, pid=%d)", name, tp, port, p.pid)
    return p


def run_graceful_reload(config: dict) -> bool:
    """Execute the graceful drain-then-switch reload.

    Returns True on success, False on failure.
    """
    signal_path = config["signal_path"]
    old_solution = config["old_solution"]
    new_solution = config["solution"]
    server_cfg = config["server_config"]
    drain_timeout = config.get("drain_timeout_s", 30.0)
    router_port = server_cfg["router_port"]
    router_url = f"http://127.0.0.1:{router_port}"
    reload_start = time.time()

    # Step 0: Determine which modules changed
    changed_names = _diff_modules(old_solution, new_solution)
    if not changed_names:
        log.info("No TP changes detected, nothing to reload.")
        _write_signal(signal_path, "ready", {"reload_duration_s": 0})
        return True

    changed_modules = _find_modules_by_names(server_cfg["modules"], changed_names)
    changed_ports = [m["port"] for m in changed_modules]

    log.info("=" * 60)
    log.info("GRACEFUL RELOAD: modules=%s", changed_names)
    log.info("  Old TP: PA=%d PF=%d DA=%d DF=%d",
             old_solution.get("tp_pa", 1), old_solution.get("tp_pf", 1),
             old_solution.get("tp_da", 1), old_solution.get("tp_df", 1))
    log.info("  New TP: PA=%d PF=%d DA=%d DF=%d",
             new_solution["tp_pa"], new_solution["tp_pf"],
             new_solution["tp_da"], new_solution["tp_df"])
    log.info("=" * 60)

    # Step 1: Signal draining
    _write_signal(signal_path, "draining", {
        "changed_modules": changed_names,
        "drain_timeout_s": drain_timeout,
    })

    # Step 2: Tell router to drain changed modules
    drain_prefill_urls = []
    drain_decode_urls = []
    for mod in changed_modules:
        url = f"http://127.0.0.1:{mod['port']}"
        if mod["disagg_mode"] == "prefill":
            drain_prefill_urls.append(url)
        else:
            drain_decode_urls.append(url)

    try:
        _http_post_json(f"{router_url}/admin/drain_module", {
            "prefill_urls": drain_prefill_urls,
            "decode_urls": drain_decode_urls,
        })
        log.info("Router drain requested: prefill=%s decode=%s",
                 drain_prefill_urls, drain_decode_urls)
    except Exception as e:
        log.error("Failed to drain router: %s", e)
        _write_signal(signal_path, "error", {"error": f"drain_failed: {e}"})
        return False

    # Step 3: Signal starting + start new modules in parallel
    _write_signal(signal_path, "starting", {"changed_modules": changed_names})

    # Determine GPU indices to reset for changed modules
    changed_gpu_indices = []
    for mod in changed_modules:
        changed_gpu_indices.append(mod["nvml_device_index"])

    # Start new modules in background threads
    new_procs = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        for mod in changed_modules:
            fut = executor.submit(_start_module, mod, new_solution, server_cfg)
            futures[fut] = mod

        # Step 4: While new modules start, drain old modules
        log.info("Waiting for old modules to drain (timeout=%ds)...", drain_timeout)
        drain_deadline = time.monotonic() + drain_timeout
        old_module_urls = [f"http://127.0.0.1:{p}" for p in changed_ports]

        while time.monotonic() < drain_deadline:
            all_idle = True
            for url in old_module_urls:
                idle, inflight = _check_idle(url)
                if not idle:
                    all_idle = False
                    log.info("  %s: still has %d inflight requests", url, inflight)
                    break
            if all_idle:
                log.info("All old modules drained successfully.")
                break
            time.sleep(1.0)
        else:
            # Timeout: abort remaining requests
            log.warning("Drain timeout reached. Aborting remaining requests...")
            for url in old_module_urls:
                _abort_module(url)
            time.sleep(2)

        # Collect new module start results
        for fut in as_completed(futures):
            mod = futures[fut]
            try:
                proc = fut.result()
                new_procs.append((mod, proc))
            except Exception as e:
                log.error("Failed to start module %s: %s", mod["name"], e)
                _write_signal(signal_path, "error",
                              {"error": f"start_failed: {mod['name']}: {e}"})
                return False

    # Step 5: Kill old modules gracefully
    log.info("Killing old modules on ports %s...", changed_ports)
    _reset_gpu_clocks(changed_gpu_indices)
    for port in changed_ports:
        _graceful_kill_port(port, grace_timeout=5.0)
    time.sleep(2)

    # Step 6: Wait for new modules to be healthy
    log.info("Waiting for new modules to become healthy...")
    for mod, proc in new_procs:
        port = mod["port"]
        if not _wait_port_ready('127.0.0.1', port, timeout=300):
            log.error("Module %s failed to start on port %d", mod["name"], port)
            _write_signal(signal_path, "error",
                          {"error": f"health_failed: {mod['name']}"})
            return False
        log.info("  %s healthy on port %d", mod["name"], port)

    # Step 7: Tell router to activate new modules
    _write_signal(signal_path, "switching")

    add_prefill = []
    add_decode = []
    for mod in changed_modules:
        url = f"http://127.0.0.1:{mod['port']}"
        bootstrap = server_cfg.get("bootstrap_port")
        if mod["disagg_mode"] == "prefill":
            add_prefill.append([url, bootstrap])
        else:
            add_decode.append(url)

    try:
        _http_post_json(f"{router_url}/admin/activate_module", {
            "add_prefill_urls": add_prefill,
            "add_decode_urls": add_decode,
            "remove_prefill_urls": drain_prefill_urls,
            "remove_decode_urls": drain_decode_urls,
        })
        log.info("Router activated new modules.")
    except Exception as e:
        log.error("Failed to activate router: %s", e)
        _write_signal(signal_path, "error", {"error": f"activate_failed: {e}"})
        return False

    # Step 8: Verify router health
    if not _wait_health(f"{router_url}/health", timeout=30):
        log.error("Router health check failed after activation")
        _write_signal(signal_path, "error", {"error": "router_health_failed"})
        return False

    # Step 9: Write ready signal
    reload_duration = time.time() - reload_start
    _write_signal(signal_path, "ready", {
        "reload_duration_s": round(reload_duration, 1),
        "changed_modules": changed_names,
        "tp_pa": new_solution["tp_pa"],
        "tp_pf": new_solution["tp_pf"],
        "tp_da": new_solution["tp_da"],
        "tp_df": new_solution["tp_df"],
        "f_pa": new_solution.get("f_pa"),
        "f_pf": new_solution.get("f_pf"),
        "f_da": new_solution.get("f_da"),
        "f_df": new_solution.get("f_df"),
    })
    log.info("Graceful reload complete in %.1fs (changed: %s)",
             reload_duration, changed_names)
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Tier1 Graceful Reload Orchestrator")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to reload config JSON")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    log.info("=" * 60)
    log.info("TIER1 GRACEFUL RELOAD ORCHESTRATOR STARTED")
    log.info("Config: %s", args.config)
    log.info("=" * 60)

    success = run_graceful_reload(config)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
