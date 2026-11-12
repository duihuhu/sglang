#!/usr/bin/env python3
"""Tier1 Reload Orchestrator — kill all servers and restart with new TP config.

This script is spawned as an independent process by the PA scheduler when
Tier1 detects a TP change. It:
  1. Writes reload signal (status=reloading)
  2. Kills all 4 server processes + router
  3. Waits for ports to free, resets GPU clocks
  4. Restarts all servers with new TP configuration
  5. Waits for health checks to pass
  6. Writes reload signal (status=ready)

Usage (called by scheduler, not manually):
    python -m sglang.srt.energy.reload_orchestrator --config /path/to/reload_config.json
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Tier1Reload] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("reload_orchestrator")

PYTHON = os.environ.get("SGLANG_PYTHON", sys.executable)


def _get_pids_on_port(port: int) -> list[str]:
    result = subprocess.run(
        ["ss", "-tlnp", f"sport = :{port}"],
        capture_output=True, text=True,
    )
    return list(set(re.findall(r'pid=(\d+)', result.stdout)))


def _kill_port(port: int):
    for pid in _get_pids_on_port(port):
        subprocess.run(["kill", "-9", pid], capture_output=True)


def _wait_port_free(ports: list[int], timeout: float = 60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        busy = []
        out = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True).stdout
        for p in ports:
            if f":{p}" in out:
                busy.append(p)
        if not busy:
            return True
        for p in busy:
            _kill_port(p)
        time.sleep(2)
    return False


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


def _reset_gpu_clocks(gpu_indices: list[int]):
    for idx in gpu_indices:
        subprocess.run(["nvidia-smi", "-rgc", "-i", str(idx)], capture_output=True)


def _write_signal(signal_path: str, status: str, extra: dict = None):
    data = {"status": status, "timestamp": time.time()}
    if extra:
        data.update(extra)
    Path(signal_path).parent.mkdir(parents=True, exist_ok=True)
    with open(signal_path, "w") as f:
        json.dump(data, f)
    log.info("Signal written: status=%s → %s", status, signal_path)


def run_reload(config: dict):
    """Execute the full reload sequence."""
    signal_path = config["signal_path"]
    solution = config["solution"]
    server_cfg = config["server_config"]
    all_ports = server_cfg["all_ports"]
    gpu_indices = server_cfg["gpu_indices"]
    reload_start = time.time()

    # Step 1: Write reloading signal
    _write_signal(signal_path, "reloading", {
        "new_tp_pa": solution["tp_pa"],
        "new_tp_pf": solution["tp_pf"],
        "new_tp_da": solution["tp_da"],
        "new_tp_df": solution["tp_df"],
    })

    # Step 2: Kill all servers
    log.info("Killing all servers on ports %s...", all_ports)
    for port in all_ports:
        _kill_port(port)
    time.sleep(3)

    # Step 3: Wait for ports to free
    if not _wait_port_free(all_ports, timeout=30):
        log.error("Ports not freed after 30s, force killing...")
        for port in all_ports:
            _kill_port(port)
        time.sleep(5)

    # Step 4: Reset GPU clocks
    _reset_gpu_clocks(gpu_indices)
    log.info("GPU clocks reset on %s", gpu_indices)

    # Step 5: Restart all servers with new TP
    log.info("Restarting servers with new config: tp_pa=%d tp_pf=%d tp_da=%d tp_df=%d",
             solution["tp_pa"], solution["tp_pf"], solution["tp_da"], solution["tp_df"])

    procs = _start_all_servers(server_cfg, solution)
    if procs is None:
        log.error("Failed to restart servers")
        _write_signal(signal_path, "error", {"error": "restart_failed"})
        return False

    # Step 6: Wait for health
    router_port = server_cfg["router_port"]
    if not _wait_health(f"http://127.0.0.1:{router_port}/health", timeout=180):
        log.error("Router health check failed")
        _write_signal(signal_path, "error", {"error": "health_check_failed"})
        return False

    # Step 7: Write ready signal
    reload_duration = time.time() - reload_start
    _write_signal(signal_path, "ready", {
        "reload_duration_s": round(reload_duration, 1),
        "tp_pa": solution["tp_pa"],
        "tp_pf": solution["tp_pf"],
        "tp_da": solution["tp_da"],
        "tp_df": solution["tp_df"],
        "f_pa": solution["f_pa"],
        "f_pf": solution["f_pf"],
        "f_da": solution["f_da"],
        "f_df": solution["f_df"],
    })
    log.info("Reload complete in %.1fs", reload_duration)
    return True


def _start_all_servers(server_cfg: dict, solution: dict):
    """Start DF → DA → PF → PA → router with new TP config."""
    model_path = server_cfg["model_path"]
    bootstrap_port = server_cfg["bootstrap_port"]
    ib_device = server_cfg["ib_device"]
    extra_args = server_cfg.get("extra_args", [])
    dvfs_args = server_cfg.get("dvfs_args", [])
    tier1_args = server_cfg.get("tier1_args", [])
    log_dir = Path(server_cfg.get("log_dir", "/tmp/tier1_reload_logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    modules = server_cfg["modules"]
    procs = []

    env_base = os.environ.copy()
    env_base['SGLANG_DISABLE_REQUEST_LOGGING'] = 'true'
    env_base['UCX_LOG_LEVEL'] = 'fatal'
    env_base['AFD_UCX_TLS'] = 'rc,tcp,cuda_copy,cuda_ipc'
    env_base['SGLANG_DISAGGREGATION_THREAD_POOL_SIZE'] = '128'
    env_base['SGLANG_DISAGGREGATION_QUEUE_SIZE'] = '32'
    env_base['SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE'] = '0'

    for mod in modules:
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

        env = env_base.copy()
        env['CUDA_VISIBLE_DEVICES'] = visible_gpus
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

        fh = open(log_dir / f"reload_{name}.log", 'w')
        p = subprocess.Popen(
            cmd, env=env, stdout=fh, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        procs.append((name, p, fh, port))
        log.info("  Started %s (tp=%d, port=%d)", name, tp, port)
        time.sleep(2)

        if not _wait_port_ready('127.0.0.1', port, timeout=300):
            log.error("  %s failed to start (port %d)", name, port)
            return None

    # Start router
    router_cfg = server_cfg.get("router", {})
    if router_cfg.get("enabled", True):
        router_port = server_cfg["router_port"]
        pa_port = router_cfg["prefill_port"]
        da_port = router_cfg["decode_port"]
        cmd = [
            PYTHON, '-m', 'sglang_router.launch_router',
            '--pd-disaggregation', '--mini-lb',
            '--prefill', f'http://127.0.0.1:{pa_port}',
            '--decode', f'http://127.0.0.1:{da_port}',
            '--host', '127.0.0.1', '--port', str(router_port),
        ]
        rf = open(log_dir / "reload_router.log", 'w')
        rp = subprocess.Popen(
            cmd, env=os.environ.copy(), stdout=rf,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
        procs.append(('router', rp, rf, router_port))
        log.info("  Started router (port=%d)", router_port)

    return procs


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Tier1 Reload Orchestrator")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to reload config JSON")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    log.info("=" * 60)
    log.info("TIER1 RELOAD ORCHESTRATOR STARTED")
    log.info("Config: %s", args.config)
    log.info("=" * 60)

    success = run_reload(config)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
