#!/usr/bin/env python3
"""
Run 4 AFlex scenarios and compare performance & energy.

Scenarios:
  1. Tier1 + Tier2   — ILP planning + runtime DVFS
  2. Tier1 only       — ILP planning, no runtime DVFS
  3. Tier2 only       — runtime DVFS only (no ILP)
  4. Neither          — baseline (all at 1410 MHz)

For each scenario:
  - Kill any running server processes
  - Patch config, launch servers via af_launcher.py
  - Wait for router readiness
  - Run benchmark_replay.py with energy monitoring
  - Kill servers
  - Collect results

Usage:
  python run_af_scenarios.py [--max-requests 100] [--concurrency 50]
"""

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("af_scenarios")

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent  # workspace/sglang
_CONFIG_PATH = _REPO_ROOT / "python" / "sglang" / "srt" / "energy" / "af_launch_config.json"
_LAUNCHER = _REPO_ROOT / "python" / "sglang" / "srt" / "energy" / "af_launcher.py"
_BENCHMARK = _HERE / "benchmark_replay.py"
_LOG_DIR = _REPO_ROOT / "af_launch_logs"

GPU_INDICES = [0, 1, 2, 7]


# ── Helpers ────────────────────────────────────────────────────────────────


def _kill_servers():
    """Kill all server processes including the Rust router, without killing ourselves."""
    logger.info("Killing server processes ...")

    # 1. Kill the router process by port (Rust binary won't match pkill -f sglang_router)
    _kill_process_on_port(50000)
    _kill_process_on_port(50010)
    _kill_process_on_port(50011)
    _kill_process_on_port(50020)
    _kill_process_on_port(50021)

    # 2. Kill known server processes (Python + Rust binary)
    targets = [
        "sglang.launch_server",
        "sglang_router.launch_router",
        "sglang::scheduler",
        "sglang::router",
        "af_launcher.py",
    ]
    for t in targets:
        subprocess.run(["pkill", "-9", "-f", t], capture_output=True, timeout=10)

    # 3. Reset GPU clocks
    for g in GPU_INDICES:
        subprocess.run(["nvidia-smi", "-i", str(g), "-rgc"],
                       capture_output=True, text=True, timeout=10)

    # 4. Wait and verify router port is free (retry up to 10s)
    for _ in range(10):
        _check_port_free(50000, "Router")
        result = subprocess.run(
            ["ss", "-tlnp", "sport = :50000"],
            capture_output=True, text=True, timeout=5,
        )
        if result.stdout.strip() and "LISTEN" in result.stdout:
            # Kill by PID from ss output
            import re
            for match in re.finditer(r'pid=(\d+)', result.stdout):
                subprocess.run(["kill", "-9", match.group(1)], capture_output=True, timeout=5)
            time.sleep(1)
        else:
            break


def _kill_process_on_port(port: int):
    """Kill whatever process is listening on `port` using ss."""
    try:
        result = subprocess.run(
            ["ss", "-tlnp", f"sport = :{port}"],
            capture_output=True, text=True, timeout=10,
        )
        # ss output: LISTEN 0 2048 127.0.0.1:50000 0.0.0.0:* users:(("sglang::router",pid=2912583,fd=13))
        import re
        for match in re.finditer(r'pid=(\d+)', result.stdout):
            pid = match.group(1)
            subprocess.run(["kill", "-9", pid], capture_output=True, timeout=10)
    except Exception:
        pass


def _check_port_free(port: int, name: str):
    """Warn if a port is still in use after killing."""
    try:
        result = subprocess.run(
            ["ss", "-tlnp", f"sport = :{port}"],
            capture_output=True, text=True, timeout=10,
        )
        if result.stdout.strip() and "LISTEN" in result.stdout:
            logger.warning("%s still alive on port %d", name, port)
    except Exception:
        pass


def _wait_for_url(url: str, timeout: int = 180) -> bool:
    """Wait until an HTTP endpoint responds, with TCP fallback."""
    import socket
    import urllib.request
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 50000
    path = parsed.path or "/"

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        # Try TCP connect first
        try:
            with socket.create_connection((host, port), timeout=2):
                # TCP works; now try HTTP
                try:
                    urllib.request.urlopen(url, timeout=2)
                    return True
                except Exception:
                    # TCP ok but HTTP failed — wait a bit more
                    time.sleep(2)
                    continue
        except Exception:
            time.sleep(2)
    return False


def _run_launcher(config_path: str) -> bool:
    """Run af_launcher.py and wait for servers to become ready."""
    cmd = [sys.executable, str(_LAUNCHER), "--config", config_path]

    # Check if Tier1 is enabled in the config
    with open(config_path) as f:
        cfg = json.load(f)
    if cfg.get("tier1", {}).get("start_with_workload", False):
        cmd.append("--start-with-workload")

    logger.info("Running: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(_REPO_ROOT),
    )

    # Wait for router to be ready (main success indicator)
    router_ok = _wait_for_url("http://localhost:50000/health", timeout=300)

    # Kill the launcher if still running (it should exit after launching)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.terminate()
        proc.wait(timeout=5)

    # Log the launcher output
    output = proc.stdout.read().decode("utf-8", errors="replace") if proc.stdout else ""
    for line in output.splitlines():
        print(f"    {line}")

    if not router_ok:
        logger.error("Router did not become ready")
        return False

    if "All modules started successfully" in output:
        return True
    if "failed to start" in output:
        logger.warning("Launcher reported failures but router is alive — continuing")
        return True
    if proc.returncode != 0:
        logger.warning("Launcher returned code %d, but router is alive", proc.returncode)

    return True  # router is alive, good enough


def _patch_config(base_cfg: dict, scenario: str) -> dict:
    """Return a copy of base_cfg modified for `scenario`."""
    cfg = json.loads(json.dumps(base_cfg))  # deep copy

    if scenario == "Tier1+Tier2":
        cfg["tier1"]["enable_tier1_pa"] = True
        cfg["tier1"]["start_with_workload"] = True
        cfg["afd"]["dvfs_enabled"] = True
    elif scenario == "Tier1-only":
        cfg["tier1"]["enable_tier1_pa"] = True
        cfg["tier1"]["start_with_workload"] = True
        cfg["afd"]["dvfs_enabled"] = False
    elif scenario == "Tier2-only":
        cfg["tier1"]["enable_tier1_pa"] = False
        cfg["tier1"]["start_with_workload"] = False
        cfg["afd"]["dvfs_enabled"] = True
    elif scenario == "Neither":
        cfg["tier1"]["enable_tier1_pa"] = False
        cfg["tier1"]["start_with_workload"] = False
        cfg["afd"]["dvfs_enabled"] = False
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return cfg


def _run_one_scenario(scenario: str, base_cfg: dict, trace_csv: str,
                      max_requests: int, concurrency: int,
                      speedup: float, timeout_s: int):
    """Run a single scenario and return the results dict from the dump file."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("  SCENARIO: %s", scenario)
    logger.info("=" * 70)

    # 1. Kill leftovers
    _kill_servers()

    # 2. Patch config and write temp file
    cfg = _patch_config(base_cfg, scenario)
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    tmp_cfg = str(_LOG_DIR / f"config_{scenario}.json")
    with open(tmp_cfg, "w") as f:
        json.dump(cfg, f, indent=2)

    # 3. Launch servers
    ok = _run_launcher(tmp_cfg)
    if not ok:
        logger.error("Server launch failed for %s, skipping", scenario)
        _kill_servers()
        return None

    # 4. Wait for router
    logger.info("Waiting for router at http://localhost:50000 ...")
    if not _wait_for_url("http://localhost:50000/health", timeout=120):
        logger.error("Router did not become ready within 120s")
        _kill_servers()
        return None
    logger.info("Router is ready.")

    # 5. Run benchmark
    logger.info("Running benchmark ...")
    dump_file = str(_LOG_DIR / f"results_{scenario}.json")
    bench_cmd = [
        sys.executable, str(_BENCHMARK),
        "--trace", trace_csv,
        "--url", "http://localhost:50000",
        "--speedup", str(speedup),
        "--max-requests", str(max_requests),
        "--concurrency", str(concurrency),
        "--timeout", str(timeout_s),
        "--monitor-energy",
        "--gpu-indices", ",".join(str(g) for g in GPU_INDICES),
        "--scenario-label", scenario,
        "--dump", dump_file,
    ]
    logger.info("Benchmark cmd: %s", " ".join(bench_cmd))
    result = subprocess.run(bench_cmd, cwd=str(_HERE))

    # 6. Kill servers
    _kill_servers()

    if result.returncode != 0:
        logger.warning("Benchmark exited with code %d for %s", result.returncode, scenario)

    # 7. Read dump
    if os.path.exists(dump_file):
        with open(dump_file) as f:
            data = json.load(f)
        return data
    return None


# ── Comparison table ──────────────────────────────────────────────────────


def _extract_metrics(data: dict) -> dict:
    """Compute summary metrics from a benchmark dump dict."""
    results = data.get("results", [])
    if not results:
        return {}

    ok = [r for r in results if r.get("success")]
    if not ok:
        return {}

    ttft_vals = [r["ttft_ms"] for r in ok if r.get("ttft_ms", 0) > 0]
    tpot_vals = [r["tpot_ms"] for r in ok if r.get("tpot_ms", 0) > 0]
    total_in = sum(r.get("input_tokens", 0) for r in ok)
    total_out = sum(r.get("output_tokens", 0) for r in ok)
    total = len(results)
    succeeded = len(ok)
    wall = data.get("wall_duration_s", 1)
    energy_mj = data.get("energy_mj_delta", {})
    total_energy_j = sum(energy_mj.values()) / 1000 if energy_mj else 0

    return {
        "mean_ttft_ms": round(sum(ttft_vals) / len(ttft_vals), 2) if ttft_vals else None,
        "p50_ttft_ms": round(sorted(ttft_vals)[len(ttft_vals) // 2], 2) if ttft_vals else None,
        "mean_tpot_ms": round(sum(tpot_vals) / len(tpot_vals), 2) if tpot_vals else None,
        "input_throughput_tok_s": round(total_in / wall, 1),
        "output_throughput_tok_s": round(total_out / wall, 1),
        "total_energy_j": round(total_energy_j, 1),
        "wall_duration_s": round(wall, 1),
        "success_rate_pct": round(succeeded / total * 100, 1) if total else 0,
        "total_requests": total,
        "succeeded": succeeded,
    }


def _print_comparison(scenarios: list, all_metrics: dict):
    """Print a formatted comparison table."""
    print("\n")
    print("=" * 120)
    print("  COMPARISON: 4 AFlex Scenarios")
    print("=" * 120)

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

    header = f"  {'Metric':<30}"
    for sc in scenarios:
        header += f" {sc:<20}"
    print(header)
    print("  " + "-" * (30 + 21 * len(scenarios)))

    for key, label in rows:
        line = f"  {label:<30}"
        for sc in scenarios:
            m = all_metrics.get(sc)
            if m is None or m.get(key) is None:
                line += f" {'N/A':<20}"
            elif isinstance(m[key], float):
                line += f" {m[key]:<20.2f}"
            else:
                line += f" {str(m[key]):<20}"
        print(line)

    print("=" * 120)


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Run 4 AFlex scenarios (Tier1/Tier2 combinations) and compare.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--trace", default=str(_HERE / "AzureLLMInferenceTrace_conv_1week.csv"))
    parser.add_argument("--max-requests", type=int, default=100, help="Requests per scenario")
    parser.add_argument("--concurrency", type=int, default=50, help="Concurrent requests")
    parser.add_argument("--speedup", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=600, help="Per-request timeout (s)")
    args = parser.parse_args()

    # Load base config
    with open(_CONFIG_PATH) as f:
        base_cfg = json.load(f)

    logger.info("Base config: %s", _CONFIG_PATH)
    logger.info("Trace: %s", args.trace)
    logger.info("Max requests/scenario: %d", args.max_requests)
    logger.info("Concurrency: %d", args.concurrency)

    scenarios = ["Tier1+Tier2", "Tier1-only", "Tier2-only", "Neither"]
    all_metrics = {}

    for sc in scenarios:
        data = _run_one_scenario(
            sc, base_cfg, args.trace,
            args.max_requests, args.concurrency,
            args.speedup, args.timeout,
        )
        if data is not None:
            all_metrics[sc] = _extract_metrics(data)
        else:
            all_metrics[sc] = None

    # Print comparison
    _print_comparison(scenarios, all_metrics)

    # Save comparison JSON
    out = {}
    for sc in scenarios:
        out[sc] = all_metrics.get(sc)
    comp_path = _LOG_DIR / "scenario_comparison.json"
    with open(comp_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Comparison saved to %s", comp_path)


if __name__ == "__main__":
    main()
