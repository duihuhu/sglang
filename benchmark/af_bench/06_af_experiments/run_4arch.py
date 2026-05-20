#!/usr/bin/env python3
"""
Test 4 serving architectures and compare performance.

Architectures:
  1. 原生 (Vanilla)      — single server, tp=4, no disaggregation
  2. PD分离               — prefill server + decode server + router
  3. AF分离               — attn server + ffn server (UCX), no PD
  4. PD+AF分离             — DF+DA+PF+PA + router (current af_launcher.py)

Usage:
  python run_4arch.py [--max-requests 100] [--concurrency 50]
"""

import json, logging, os, socket, subprocess, sys, time, re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("run_4arch")

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
_PYTHON = "/workspace/env/af-test/bin/python"
_CONFIG_PATH = _REPO / "python" / "sglang" / "srt" / "energy" / "af_launch_config.json"
_LAUNCHER = _REPO / "python" / "sglang" / "srt" / "energy" / "af_launcher.py"
_BENCHMARK = _HERE / "benchmark_replay.py"
_LOG_DIR = _REPO / "af_launch_logs"
_MODEL = "/models/Qwen/Qwen3-32B/"
_GPU_INDICES = [0, 1, 4, 5]

_PORT_TIMEOUT = 180
_BENCH_TIMEOUT = 600

# ── Helpers ────────────────────────────────────────────────────────────────

def _kill_all():
    """Kill everything that might conflict."""
    _my_pid = os.getpid()
    for port in [50000, 50010, 50011, 50020, 50021]:
        _kill_port(port)
    for t in ["sglang.launch_server", "sglang_router", "sglang::", "af_launcher"]:
        subprocess.run(["pkill", "-9", "-f", t], capture_output=True, timeout=10)
    # Kill UCX and Mooncake processes, but NOT ourselves (our cmdline may
    # contain "ucx" from --afd-comm-backend, which pkill -f would match).
    for t in ["ucx", "mooncake"]:
        r = subprocess.run(["pgrep", "-f", t], capture_output=True, text=True, timeout=5)
        for pid in r.stdout.strip().split():
            if pid and int(pid) != _my_pid:
                subprocess.run(["kill", "-9", pid], capture_output=True, timeout=5)
    # Try to clean up RDMA state
    subprocess.run(["rdma", "link", "delete", "mlx5_4/1"], capture_output=True, timeout=5)
    time.sleep(2)
    # Reset GPU clocks
    for g in _GPU_INDICES:
        subprocess.run(["nvidia-smi", "-i", str(g), "-rgc"], capture_output=True, timeout=10)

def _kill_port(port):
    try:
        r = subprocess.run(["ss", "-tlnp", f"sport = :{port}"], capture_output=True, text=True, timeout=10)
        for m in re.finditer(r'pid=(\d+)', r.stdout):
            subprocess.run(["kill", "-9", m.group(1)], capture_output=True, timeout=10)
    except: pass

def _wait_port(host, port, timeout=_PORT_TIMEOUT):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except: time.sleep(2)
    return False

def _wait_health(url, timeout=180):
    import urllib.request
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=5)
            return True
        except: time.sleep(3)
    return False

def _run_benchmark(url, scenario_label, dump_file, max_requests=100, concurrency=50,
                   dataset="azure", sample_input_len=1024, sample_output_len=128,
                   sample_qps=1.0, seed=42, warmup=0):
    """Run benchmark_replay.py and return the dump data."""
    cmd = [
        _PYTHON, str(_BENCHMARK),
        "--url", url,
        "--speedup", "1.0",
        "--max-requests", str(max_requests),
        "--concurrency", str(concurrency),
        "--timeout", str(_BENCH_TIMEOUT),
        "--monitor-energy",
        "--gpu-indices", ",".join(str(g) for g in _GPU_INDICES),
        "--scenario-label", scenario_label,
        "--dump", dump_file,
        "--dataset", dataset,
    ]
    if warmup > 0:
        cmd += ["--warmup", str(warmup)]
    if dataset == "sample":
        cmd += [
            "--sample-input-len", str(sample_input_len),
            "--sample-output-len", str(sample_output_len),
            "--sample-qps", str(sample_qps),
            "--seed", str(seed),
        ]
    else:
        cmd += ["--trace", str(_HERE / "AzureLLMInferenceTrace_conv_1week.csv")]
    logger.info("Benchmark: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(_HERE))
    if os.path.exists(dump_file):
        with open(dump_file) as f:
            return json.load(f)
    return None

def _log_proc(proc, tag, log_dir):
    """Write a process's stdout/err to a log file."""
    path = log_dir / f"{tag}.log"
    with open(path, "wb") as f:
        stdout, stderr = proc.communicate()
        if stdout: f.write(stdout)
        if stderr: f.write(stderr)

def _check_processes(procs):
    """Check if any process has exited."""
    for name, proc in procs:
        rc = proc.poll()
        if rc is not None:
            logger.error("%s exited early with code %d", name, rc)
            return False
    return True

# ── Architecture 1: 原生sglang (vanilla, tp=4) ─────────────────────────────

def run_native(log_dir, max_requests=100, concurrency=50, timeout_s=600,
               dataset="azure", sample_input_len=1024, sample_output_len=128,
               sample_qps=1.0, seed=42, warmup=0):
    """Single server with tp=4, no disaggregation."""
    logger.info("=" * 60)
    logger.info("  ARCH 1/4: 原生sglang (vanilla, tp=4)")
    logger.info("=" * 60)

    _kill_all()

    port = 50000
    log_file = log_dir / "native_server.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in _GPU_INDICES)
    env["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"

    cmd = [
        _PYTHON, "-m", "sglang.launch_server",
        "--model-path", _MODEL,
        "--tp", "4",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--mem-fraction-static", "0.7",
        "--enable-metrics",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--skip-server-warmup",
    ]

    logger.info("Launching: %s", " ".join(cmd))
    fh = open(log_file, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)

    ok = _wait_port("127.0.0.1", port, timeout=300)
    if not ok:
        logger.error("native server failed to start on port %d", port)
        proc.terminate()
        return None

    # Health check
    if not _wait_health(f"http://127.0.0.1:{port}/health", timeout=60):
        logger.error("native server health check failed")
        proc.terminate()
        return None

    logger.info("native server ready on port %d", port)

    dump_file = str(log_dir / "results_native.json")
    data = _run_benchmark(f"http://127.0.0.1:{port}", "原生", dump_file,
                          max_requests, concurrency,
                          dataset=dataset, sample_input_len=sample_input_len,
                          sample_output_len=sample_output_len,
                          sample_qps=sample_qps, seed=seed, warmup=warmup)

    proc.terminate()
    proc.wait(timeout=30)
    return data

# ── Architecture 2: PD分离 ────────────────────────────────────────────────

def run_pd_only(log_dir, max_requests=100, concurrency=50, timeout_s=600,
                dataset="azure", sample_input_len=1024, sample_output_len=128,
                sample_qps=1.0, seed=42, warmup=0):
    """Prefill-decode disaggregation with mooncake."""
    logger.info("=" * 60)
    logger.info("  ARCH 2/4: PD分离 (prefill + decode + router)")
    logger.info("=" * 60)

    _kill_all()

    ib_dev = "mlx5_4"
    bootstrap_port = 18999
    prefill_port = 50010
    decode_port = 50020
    router_port = 50000

    # ── Prefill server (GPU 2,3, tp=2) ──
    prefill_log = log_dir / "pd_prefill.log"
    env_prefill = os.environ.copy()
    env_prefill["CUDA_VISIBLE_DEVICES"] = "2,3"
    env_prefill["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"

    prefill_cmd = [
        _PYTHON, "-m", "sglang.launch_server",
        "--model-path", _MODEL,
        "--tp", "2",
        "--host", "127.0.0.1",
        "--port", str(prefill_port),
        "--enable-metrics",
        "--disaggregation-mode", "prefill",
        "--disaggregation-transfer-backend", "mooncake",
        "--disaggregation-bootstrap-port", str(bootstrap_port),
        "--disaggregation-ib-device", ib_dev,
        "--mem-fraction-static", "0.7",
    ]

    logger.info("Prefill: %s", " ".join(prefill_cmd))
    pf_fh = open(prefill_log, "w")
    pf_proc = subprocess.Popen(prefill_cmd, env=env_prefill,
                               stdout=pf_fh, stderr=subprocess.STDOUT,
                               start_new_session=True)

    if not _wait_port("127.0.0.1", prefill_port):
        logger.error("PD prefill failed to start")
        pf_proc.terminate()
        return None
    logger.info("PD prefill ready on port %d", prefill_port)

    # ── Decode server (GPU 0,1, tp=2) ──
    decode_log = log_dir / "pd_decode.log"
    env_decode = os.environ.copy()
    env_decode["CUDA_VISIBLE_DEVICES"] = "0,1"
    env_decode["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"

    decode_cmd = [
        _PYTHON, "-m", "sglang.launch_server",
        "--model-path", _MODEL,
        "--tp", "2",
        "--host", "127.0.0.1",
        "--port", str(decode_port),
        "--enable-metrics",
        "--disaggregation-mode", "decode",
        "--disaggregation-transfer-backend", "mooncake",
        "--disaggregation-bootstrap-port", str(bootstrap_port),
        "--disaggregation-ib-device", ib_dev,
        "--mem-fraction-static", "0.7",
    ]

    logger.info("Decode: %s", " ".join(decode_cmd))
    dd_fh = open(decode_log, "w")
    dd_proc = subprocess.Popen(decode_cmd, env=env_decode,
                               stdout=dd_fh, stderr=subprocess.STDOUT,
                               start_new_session=True)

    if not _wait_port("127.0.0.1", decode_port):
        logger.error("PD decode failed to start")
        pf_proc.terminate()
        dd_proc.terminate()
        return None
    logger.info("PD decode ready on port %d", decode_port)

    # ── Router ──
    router_log = log_dir / "pd_router.log"
    env_router = os.environ.copy()
    router_cmd = [
        _PYTHON, "-m", "sglang_router.launch_router",
        "--pd-disaggregation",
        "--mini-lb",
        "--prefill", f"http://127.0.0.1:{prefill_port}",
        "--decode", f"http://127.0.0.1:{decode_port}",
        "--host", "127.0.0.1",
        "--port", str(router_port),
    ]
    logger.info("Router: %s", " ".join(router_cmd))
    rt_fh = open(router_log, "w")
    rt_proc = subprocess.Popen(router_cmd, env=env_router,
                               stdout=rt_fh, stderr=subprocess.STDOUT,
                               start_new_session=True)

    if not _wait_health(f"http://127.0.0.1:{router_port}/health", timeout=120):
        logger.error("PD router health check failed")
        pf_proc.terminate(); dd_proc.terminate(); rt_proc.terminate()
        return None
    logger.info("PD router ready on port %d", router_port)

    # ── Benchmark ──
    dump_file = str(log_dir / "results_pd.json")
    data = _run_benchmark(f"http://127.0.0.1:{router_port}", "PD分离", dump_file,
                          max_requests, concurrency,
                          dataset=dataset, sample_input_len=sample_input_len,
                          sample_output_len=sample_output_len,
                          sample_qps=sample_qps, seed=seed, warmup=warmup)

    pf_proc.terminate(); dd_proc.terminate(); rt_proc.terminate()
    for p in [pf_proc, dd_proc, rt_proc]:
        try: p.wait(timeout=10)
        except: pass
    return data

# ── Architecture 3: AF分离 ────────────────────────────────────────────────

def run_af_only(log_dir, max_requests=100, concurrency=50, timeout_s=600,
                dataset="azure", sample_input_len=1024, sample_output_len=128,
                sample_qps=1.0, seed=42, warmup=0):
    """Attention-FFN disaggregation without PD."""
    logger.info("=" * 60)
    logger.info("  ARCH 3/4: AF分离 (attn + ffn via UCX, no PD)")
    logger.info("=" * 60)

    _kill_all()

    attn_port = 50010
    ffn_port = 50011

    ucx_base = 25100
    sched_port = 65300

    # ── FFN server (GPU 0,2, tp=2) ──
    ffn_log = log_dir / "af_ffn.log"
    env_ffn = os.environ.copy()
    env_ffn["CUDA_VISIBLE_DEVICES"] = "0,2"
    env_ffn["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
    env_ffn["AFD_UCX_BASE_PORT"] = str(ucx_base)
    env_ffn["AFD_SCHED_PORT"] = str(sched_port)
    env_ffn["AFD_UCX_TLS"] = "rc,tcp,cuda_copy,cuda_ipc"
    env_ffn["UCX_LOG_LEVEL"] = "fatal"
    env_ffn["UCX_WARN_UNUSED_ENV_VARS"] = "n"

    ffn_cmd = [
        _PYTHON, "-m", "sglang.launch_server",
        "--model-path", _MODEL,
        "--tp", "2",
        "--host", "127.0.0.1",
        "--port", str(ffn_port),
        "--afd-perspective", "ffn",
        "--afd-comm-backend", "ucx",
        "--mem-fraction-static", "0.7",
        "--skip-server-warmup",
    ]

    logger.info("FFN: %s", " ".join(ffn_cmd))
    ffn_fh = open(ffn_log, "w")
    ffn_proc = subprocess.Popen(ffn_cmd, env=env_ffn,
                                stdout=ffn_fh, stderr=subprocess.STDOUT,
                                start_new_session=True)

    if not _wait_port("127.0.0.1", ffn_port):
        logger.error("AF FFN failed to start")
        ffn_proc.terminate()
        return None
    logger.info("AF FFN ready on port %d", ffn_port)

    # ── Attn server (GPU 1,3, tp=2) ──
    attn_log = log_dir / "af_attn.log"
    env_attn = os.environ.copy()
    env_attn["CUDA_VISIBLE_DEVICES"] = "1,3"
    env_attn["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
    env_attn["AFD_UCX_BASE_PORT"] = str(ucx_base)
    env_attn["AFD_SCHED_PORT"] = str(sched_port)
    env_attn["AFD_UCX_TLS"] = "rc,tcp,cuda_copy,cuda_ipc"
    env_attn["AFD_UCX_FFN_HOST"] = "127.0.0.1"
    env_attn["UCX_LOG_LEVEL"] = "fatal"
    env_attn["UCX_WARN_UNUSED_ENV_VARS"] = "n"

    attn_cmd = [
        _PYTHON, "-m", "sglang.launch_server",
        "--model-path", _MODEL,
        "--tp", "2",
        "--host", "127.0.0.1",
        "--port", str(attn_port),
        "--enable-metrics",
        "--afd-perspective", "attn",
        "--afd-comm-backend", "ucx",
        "--mem-fraction-static", "0.7",
        "--skip-server-warmup",
    ]

    logger.info("Attn: %s", " ".join(attn_cmd))
    attn_fh = open(attn_log, "w")
    # Disable generation-based health check (AFD pipeline stalls on single-token requests)
    env_attn["SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION"] = "false"
    attn_proc = subprocess.Popen(attn_cmd, env=env_attn,
                                 stdout=attn_fh, stderr=subprocess.STDOUT,
                                 start_new_session=True)

    if not _wait_port("127.0.0.1", attn_port):
        logger.error("AF Attn failed to start")
        ffn_proc.terminate()
        attn_proc.terminate()
        return None
    logger.info("AF Attn ready on port %d", attn_port)

    # ── Health check on attn server ──
    if not _wait_health(f"http://127.0.0.1:{attn_port}/health", timeout=60):
        logger.error("AF attn server health check failed")
        ffn_proc.terminate()
        attn_proc.terminate()
        return None

    # ── Benchmark (send to attn server directly) ──
    dump_file = str(log_dir / "results_af.json")
    data = _run_benchmark(f"http://127.0.0.1:{attn_port}", "AF分离", dump_file,
                          max_requests, concurrency,
                          dataset=dataset, sample_input_len=sample_input_len,
                          sample_output_len=sample_output_len,
                          sample_qps=sample_qps, seed=seed, warmup=warmup)

    ffn_proc.terminate(); attn_proc.terminate()
    for p in [ffn_proc, attn_proc]:
        try: p.wait(timeout=10)
        except: pass
    return data

# ── Architecture 4: PD+AF分离 ─────────────────────────────────────────────

def run_pd_af(log_dir, max_requests=100, concurrency=50, timeout_s=600,
              dataset="azure", sample_input_len=1024, sample_output_len=128,
              sample_qps=1.0, seed=42, warmup=0,
              afd_comm_backend="ucx", afd_ipc_sync_send="0", afd_micro_batch=3,
              pdaf_max_running_requests=16,
              pdaf_prefill_max_running=8,
              pdaf_chunked_prefill_size=4096):
    """Full PD+AF disaggregation via af_launcher.py.

    Args:
        afd_comm_backend: \"ucx\" or \"ipc\"
        afd_ipc_sync_send: for IPC backend, \"0\"=async (bg worker), \"1\"=sync (comm_stream)
        afd_micro_batch: micro-batch size (1 or 3)
    """
    arch_label = f"PD+AF分离 ({afd_comm_backend}"
    if afd_comm_backend == "ipc":
        arch_label += f", {'async' if afd_ipc_sync_send == '0' else 'sync'})"
    else:
        arch_label += ")"
    logger.info("=" * 60)
    logger.info("  ARCH 4/4: %s (DF+DA+PF+PA + router)", arch_label)
    logger.info("=" * 60)
    _kill_all()

    # Create a clean config without DVFS/Tier1 for fair architecture comparison
    with open(_CONFIG_PATH) as f:
        pdaf_cfg = json.load(f)
    pdaf_cfg["afd"]["dvfs_enabled"] = False
    pdaf_cfg["tier1"]["enable_tier1_pa"] = False
    pdaf_cfg["tier1"]["start_with_workload"] = False
    # Set AFD backend
    pdaf_cfg["afd"]["comm_backend"] = afd_comm_backend
    pdaf_cfg["afd"]["ipc_sync_send"] = afd_ipc_sync_send
    # Increase mem_fraction_static for tp=1 modules (each GPU has full 64GB weights)
    pdaf_cfg["model"]["mem_fraction_static"] = 0.93

    # Add per-module extra CLI args
    for mod in pdaf_cfg["modules"]:
        mod["extra_cli_args"] = [
            "--skip-server-warmup",
            "--disable-cuda-graph",
            "--disable-piecewise-cuda-graph",
        ]
    # Decode modules: enable micro-batch and limit batch size
    for mod in pdaf_cfg["modules"]:
        if mod["name"] in ("DF", "DA"):
            mod["extra_cli_args"] += [
                "--afd-micro-batch", str(afd_micro_batch),
                "--max-running-requests", str(pdaf_max_running_requests),
            ]
    # Prefill modules: optionally cap concurrent prefill batch size + chunked prefill
    # to prevent OOM on long-input bursts (e.g. IL=4096 × 100 concurrent).
    # Disabled by default — enabling these with AFD disagg prefill can cause
    # KV bootstrap transfer deadlocks (UCX protocol interleaving).
    if pdaf_prefill_max_running is not None or pdaf_chunked_prefill_size is not None:
        for mod in pdaf_cfg["modules"]:
            if mod["name"] in ("PF", "PA"):
                if pdaf_prefill_max_running is not None:
                    mod["extra_cli_args"] += [
                        "--max-running-requests", str(pdaf_prefill_max_running),
                    ]
                if pdaf_chunked_prefill_size is not None:
                    mod["extra_cli_args"] += [
                        "--chunked-prefill-size", str(pdaf_chunked_prefill_size),
                    ]
    # Use unique UCX base ports to avoid residual RDMA state from AF-only run
    for mod in pdaf_cfg["modules"]:
        if mod["name"] in ("DF", "DA"):
            mod["ucx_base_port"] = 25200
        elif mod["name"] in ("PF", "PA"):
            mod["ucx_base_port"] = 25100
    tmp_cfg = log_dir / "config_pdaf_clean.json"
    with open(tmp_cfg, "w") as f:
        json.dump(pdaf_cfg, f, indent=2)

    launcher_cmd = [_PYTHON, str(_LAUNCHER), "--config", str(tmp_cfg)]

    logger.info("Launcher: %s", " ".join(launcher_cmd))
    launcher_log = log_dir / "pdaf_launcher.log"
    with open(launcher_log, "w") as f:
        launcher_proc = subprocess.Popen(
            launcher_cmd, stdout=f, stderr=subprocess.STDOUT,
            start_new_session=True, cwd=str(_REPO),
        )

    # Wait for router
    router_ok = _wait_health("http://127.0.0.1:50000/health", timeout=300)

    try:
        launcher_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        launcher_proc.terminate()
        launcher_proc.wait(timeout=5)

    # Read launcher output
    with open(launcher_log) as f:
        output = f.read()

    if not router_ok:
        logger.error("PD+AF router did not become ready")
        _kill_all()
        return None

    if "All modules started successfully" not in output and "failed to start" in output:
        logger.error("PD+AF launcher reported failures")
        _kill_all()
        return None

    logger.info("PD+AF all modules ready")

    dump_file = str(log_dir / f"results_pdaf_{afd_comm_backend}.json")
    data = _run_benchmark(f"http://127.0.0.1:50000", arch_label, dump_file,
                          max_requests, concurrency,
                          dataset=dataset, sample_input_len=sample_input_len,
                          sample_output_len=sample_output_len,
                          sample_qps=sample_qps, seed=seed, warmup=warmup)

    _kill_all()
    return data

# ── Metrics extraction ─────────────────────────────────────────────────────

def extract_metrics(data):
    results = data.get("results", [])
    ok = [r for r in results if r.get("success")]
    if not ok:
        return {}

    ttft = [r["ttft_ms"] for r in ok if r.get("ttft_ms", 0) > 0]
    tpot = [r["tpot_ms"] for r in ok if r.get("tpot_ms", 0) > 0]
    total_in = sum(r.get("input_tokens", 0) for r in ok)
    total_out = sum(r.get("output_tokens", 0) for r in ok)
    wall = data.get("wall_duration_s", 1)
    energy_mj = data.get("energy_mj_delta", {})
    total_energy_j = sum(energy_mj.values()) / 1000 if energy_mj else 0

    return {
        "mean_ttft_ms": round(sum(ttft)/len(ttft), 2) if ttft else None,
        "p50_ttft_ms": round(sorted(ttft)[len(ttft)//2], 2) if ttft else None,
        "mean_tpot_ms": round(sum(tpot)/len(tpot), 2) if tpot else None,
        "output_throughput_tok_s": round(total_out / wall, 1),
        "input_throughput_tok_s": round(total_in / wall, 1),
        "total_energy_j": round(total_energy_j, 1),
        "wall_duration_s": round(wall, 1),
        "success_rate_pct": round(len(ok)/len(results)*100, 1),
        "total_requests": len(results),
        "succeeded": len(ok),
    }

def print_table(archs, all_metrics):
    print("\n")
    print("=" * 130)
    print("  COMPARISON: 4 Serving Architectures")
    print("=" * 130)
    header = f"  {'Metric':<30}"
    for a in archs:
        header += f" {a:<22}"
    print(header)
    print("  " + "-" * (30 + 23 * len(archs)))

    rows = [
        ("mean_ttft_ms", "Mean TTFT (ms)"),
        ("p50_ttft_ms", "P50 TTFT (ms)"),
        ("mean_tpot_ms", "Mean TPOT (ms)"),
        ("output_throughput_tok_s", "Output Throughput (tok/s)"),
        ("input_throughput_tok_s", "Input Throughput (tok/s)"),
        ("total_energy_j", "Total Energy (J)"),
        ("wall_duration_s", "Wall Duration (s)"),
        ("success_rate_pct", "Success Rate (%)"),
    ]
    for key, label in rows:
        line = f"  {label:<30}"
        for a in archs:
            m = all_metrics.get(a)
            if m is None or m.get(key) is None:
                line += f" {'N/A':<22}"
            elif isinstance(m[key], float):
                line += f" {m[key]:<22.2f}"
            else:
                line += f" {str(m[key]):<22}"
        print(line)
    print("=" * 130)

# ── Main ───────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test 4 serving architectures")
    parser.add_argument("--max-requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--only", type=str, default=None,
                        help="Run only one arch: native|pd|af|pdaf")
    parser.add_argument("--dataset", default="azure", choices=["azure", "sample"],
                        help="Dataset source: 'azure' for CSV trace, 'sample' for synthetic")
    parser.add_argument("--sample-input-len", type=int, default=1024,
                        help="Fixed input length for sample dataset")
    parser.add_argument("--sample-output-len", type=int, default=128,
                        help="Fixed output length for sample dataset")
    parser.add_argument("--sample-qps", type=float, default=1.0,
                        help="Target QPS for sample dataset")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sample dataset")
    parser.add_argument("--warmup", type=int, default=0,
                        help="Send N warmup requests before benchmark")
    parser.add_argument("--afd-comm-backend", type=str, default="ucx",
                        choices=["ucx", "ipc"],
                        help="AFD communication backend for PD+AF arch")
    parser.add_argument("--afd-ipc-sync-send", type=str, default="0",
                        choices=["0", "1"],
                        help="IPC sync mode: 0=async(bg worker), 1=sync(comm_stream)")
    parser.add_argument("--afd-micro-batch", type=int, default=3,
                        help="AFD micro-batch size")
    parser.add_argument("--pdaf-max-running-requests", type=int, default=16,
                        help="max-running-requests for PD+AF DF/DA modules")
    parser.add_argument("--pdaf-prefill-max-running", type=int, default=None,
                        help="max-running-requests for PD+AF PF/PA prefill modules (default: no limit)")
    parser.add_argument("--pdaf-chunked-prefill-size", type=int, default=None,
                        help="chunked-prefill-size for PD+AF PF/PA prefill modules (default: disabled)")
    args = parser.parse_args()

    log_dir = _LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    arch_map = {
        "native": ("原生sglang", run_native),
        "pd": ("PD分离", run_pd_only),
        "af": ("AF分离", run_af_only),
        "pdaf": ("PD+AF分离", run_pd_af),
    }

    if args.only:
        order = [x.strip() for x in args.only.split(",")]
    else:
        order = ["native", "pd", "af", "pdaf"]

    all_metrics = {}

    for key in order:
        label, func = arch_map[key]
        logger.info("\n\n")
        logger.info("#" * 60)
        logger.info("# Starting: %s", label)
        logger.info("#" * 60)

        try:
            kwargs = {}
            if func == run_pd_af:
                kwargs = dict(afd_comm_backend=args.afd_comm_backend,
                              afd_ipc_sync_send=args.afd_ipc_sync_send,
                              afd_micro_batch=args.afd_micro_batch,
                              pdaf_max_running_requests=args.pdaf_max_running_requests,
                              pdaf_prefill_max_running=args.pdaf_prefill_max_running,
                              pdaf_chunked_prefill_size=args.pdaf_chunked_prefill_size)
            data = func(log_dir, args.max_requests, args.concurrency, args.timeout,
                        dataset=args.dataset, sample_input_len=args.sample_input_len,
                        sample_output_len=args.sample_output_len,
                        sample_qps=args.sample_qps, seed=args.seed, warmup=args.warmup,
                        **kwargs)
        except Exception as e:
            logger.error("Arch %s failed: %s", label, e)
            _kill_all()
            data = None

        if data is not None:
            all_metrics[label] = extract_metrics(data)
        else:
            all_metrics[label] = None

        # Brief pause between architectures
        if key != order[-1]:
            logger.info("Cool-down pause...")
            time.sleep(5)

    # Print comparison
    arch_labels = [arch_map[k][0] for k in order]
    print_table(arch_labels, all_metrics)

    # Save JSON
    out = {}
    for label in arch_labels:
        out[label] = all_metrics.get(label)
    comp_path = log_dir / "arch_comparison.json"
    with open(comp_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Comparison saved to %s", comp_path)

if __name__ == "__main__":
    main()
