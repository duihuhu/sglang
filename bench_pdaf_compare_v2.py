#!/usr/bin/env python3
"""PD+AF M=1 benchmark: IPC vs UCX comparison (v2 - separate log dirs).

Fixes from v1:
- Separate log directories per backend so IPC logs are not overwritten
- Extra wait after router ready to ensure IPC handshakes complete
- Error detail capture for debugging
"""

import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("pdaf_compare")

_REPO = Path("/workspace/sglang")
_CONFIG_PATH = _REPO / "python" / "sglang" / "srt" / "energy" / "af_launch_config.json"
_LAUNCHER = _REPO / "python" / "sglang" / "srt" / "energy" / "af_launcher.py"
_BENCHMARK = _REPO / "benchmark" / "test_motivation" / "AzurePublicDataset" / "benchmark_replay.py"
_LOG_DIR = _REPO / "af_launch_logs"
_PYTHON = "/workspace/env/af-test/bin/python"
_GPU_INDICES = [0, 1, 2, 3]


def kill_all():
    for port in [50000, 50010, 50011, 50020, 50021]:
        try:
            r = subprocess.run(["ss", "-tlnp", f"sport = :{port}"], capture_output=True, text=True, timeout=10)
            for m in re.finditer(r'pid=(\d+)', r.stdout):
                subprocess.run(["kill", "-9", m.group(1)], capture_output=True, timeout=10)
        except Exception:
            pass
    for t in ["sglang.launch_server", "sglang_router", "af_launcher", "ucx", "mooncake"]:
        subprocess.run(["pkill", "-9", "-f", t], capture_output=True, timeout=10)
    subprocess.run(["rdma", "link", "delete", "mlx5_4/1"], capture_output=True, timeout=5)
    time.sleep(5)


def run_pdaf_bench(backend: str, max_requests=100, qps=2.0) -> dict:
    logger.info("=" * 60)
    logger.info("  PD+AF M=1 with %s backend", backend.upper())
    logger.info("=" * 60)

    kill_all()

    # Build config for this backend
    with open(_CONFIG_PATH) as f:
        cfg = json.load(f)

    cfg["afd"]["comm_backend"] = backend
    cfg["afd"]["dvfs_enabled"] = False
    cfg["tier1"]["enable_tier1_pa"] = False
    cfg["tier1"]["start_with_workload"] = False
    cfg["model"]["mem_fraction_static"] = 0.93

    for mod in cfg["modules"]:
        mod["extra_cli_args"] = [
            "--skip-server-warmup",
            "--disable-cuda-graph",
            "--disable-piecewise-cuda-graph",
        ]
        if mod["name"] in ("DF", "DA"):
            mod["extra_cli_args"] += [
                "--afd-micro-batch", "1",
                "--disable-overlap-schedule",
                "--max-running-requests", "16",
            ]

    # Use SEPARATE log dir per backend so logs are not overwritten
    server_log_dir = _LOG_DIR / f"logs_{backend}"
    cfg["logs"]["dir"] = str(server_log_dir)

    # For IPC, clean up stale SHM/socket files
    if backend == "ipc":
        for p in range(1000):
            for f in [f"/dev/shm/afd_ipc_flags_{p}", f"/tmp/afd_ipc_{p}.sock"]:
                try:
                    os.unlink(f)
                except Exception:
                    pass

    tmp_cfg = _LOG_DIR / f"config_pdaf_{backend}_v2.json"
    with open(tmp_cfg, "w") as f:
        json.dump(cfg, f, indent=2)

    # Clean log dir
    if server_log_dir.exists():
        shutil.rmtree(server_log_dir)
    server_log_dir.mkdir(parents=True, exist_ok=True)

    # Launch PD+AF
    launcher_cmd = [_PYTHON, str(_LAUNCHER), "--config", str(tmp_cfg)]
    logger.info("Launching: %s", " ".join(launcher_cmd))
    launcher_log = _LOG_DIR / f"launcher_{backend}_v2.log"
    with open(launcher_log, "w") as f:
        launcher_proc = subprocess.Popen(
            launcher_cmd, stdout=f, stderr=subprocess.STDOUT,
            start_new_session=True, cwd=str(_REPO),
        )

    # Wait for router
    deadline = time.monotonic() + 300
    router_ok = False
    while time.monotonic() < deadline:
        try:
            import urllib.request
            urllib.request.urlopen("http://127.0.0.1:50000/health", timeout=5)
            router_ok = True
            break
        except Exception:
            time.sleep(3)

    if not router_ok:
        logger.error("PD+AF router not ready for backend %s", backend)
        kill_all()
        return None

    # Extra wait for IPC handshake to complete (happens lazily on first AFD op)
    logger.info("Router ready, waiting 10s for IPC handshake stabilization...")
    time.sleep(10)

    logger.info("PD+AF %s all modules ready", backend)

    # Run benchmark
    dump_file = str(_LOG_DIR / f"results_pdaf_{backend}_v2.json")
    bench_cmd = [
        _PYTHON, str(_BENCHMARK),
        "--url", "http://127.0.0.1:50000",
        "--speedup", "1.0",
        "--max-requests", str(max_requests),
        "--concurrency", "50",
        "--timeout", "600",
        "--monitor-energy",
        "--gpu-indices", ",".join(str(g) for g in _GPU_INDICES),
        "--scenario-label", f"PD+AF_{backend}",
        "--dump", dump_file,
        "--dataset", "sample",
        "--sample-input-len", "1024",
        "--sample-output-len", "128",
        "--sample-qps", str(qps),
        "--seed", "42",
    ]
    logger.info("Benchmark: %s", " ".join(bench_cmd))
    bench_result = subprocess.run(bench_cmd, cwd=str(_REPO))

    # Give time for events to flush
    time.sleep(5)

    kill_all()

    # Load benchmark results
    bench_data = None
    if os.path.exists(dump_file):
        with open(dump_file) as f:
            bench_data = json.load(f)

    # Parse AFD_HOST_EVENTS from server logs
    decode_events = {}
    for mod_name in ["DA", "DF"]:
        log_path = server_log_dir / f"{mod_name}.log"
        events = []
        if log_path.exists():
            with open(log_path) as f:
                for line in f:
                    if "[AFD_HOST_EVENTS]" in line:
                        m = re.search(r'events=(\[.*\])$', line)
                        if m:
                            try:
                                events.extend(json.loads(m.group(1)))
                            except json.JSONDecodeError:
                                pass
        if events:
            decode_events[mod_name] = parse_events(events)
            logger.info("  Parsed %d events from %s/%s", len(events), backend, mod_name)

    # Also check for IPC handshake messages
    ipc_status = {}
    for mod_name in ["DA", "DF", "PA", "PF"]:
        log_path = server_log_dir / f"{mod_name}.log"
        if log_path.exists():
            with open(log_path) as f:
                for line in f:
                    if "IPC" in line and ("handshake" in line or "error" in line.lower() or "failed" in line.lower()):
                        ipc_status.setdefault(mod_name, []).append(line.strip())

    return {
        "bench": bench_data,
        "events": decode_events,
        "ipc_status": ipc_status,
    }


def parse_events(events):
    """Parse AFD_HOST_EVENTS into per-layer timing dict (accumulate all iterations)."""
    layers = {}

    for ev in events:
        layer = ev.get("layer", -1)
        if layer < 0:
            continue
        if layer not in layers:
            layers[layer] = {}

        event_type = ev.get("event")
        stage = ev.get("stage", "")

        if event_type == "send_end" and stage == "A":
            layers[layer]["send_dur_us"] = ev.get("send_dur_us", 0)
        elif event_type == "recv_end" and stage == "A":
            layers[layer]["recv_a_dur_us"] = ev.get("recv_dur_us", 0)
        elif event_type == "recv_end" and stage == "F":
            layers[layer]["recv_f_dur_us"] = ev.get("recv_dur_us", 0)
        elif event_type == "send_end" and stage == "F":
            layers[layer]["send_f_dur_us"] = ev.get("send_dur_us", 0)

    return layers


def compute_stats(events_data, mod="DA"):
    """Compute aggregate stats from parsed events."""
    if events_data is None or events_data.get(mod) is None:
        return None

    layers = events_data[mod]
    if not layers:
        return None

    send_durs = []
    recv_f_durs = []
    recv_a_durs = []
    send_f_durs = []

    for lid in sorted(layers.keys()):
        l = layers[lid]
        if "send_dur_us" in l:
            send_durs.append(l["send_dur_us"])
        if "recv_f_dur_us" in l:
            recv_f_durs.append(l["recv_f_dur_us"])
        if "recv_a_dur_us" in l:
            recv_a_durs.append(l["recv_a_dur_us"])
        if "send_f_dur_us" in l:
            send_f_durs.append(l["send_f_dur_us"])

    def fmt(vals):
        if not vals:
            return {"avg": 0, "min": 0, "max": 0, "count": 0}
        return {"avg": sum(vals)/len(vals), "min": min(vals), "max": max(vals), "count": len(vals)}

    return {
        "DA_send": fmt(send_durs),
        "DA_recv": fmt(recv_f_durs),
        "FFN_recv": fmt(recv_a_durs),
        "FFN_send": fmt(send_f_durs),
    }


def extract_bench_metrics(data):
    """Extract key metrics from benchmark dump."""
    if data is None:
        return {}
    results = data.get("results", [])
    ok = [r for r in results if r.get("success")]
    if not ok:
        # Capture sample errors
        errs = results[:3] if results else []
        return {"error": "no successful requests", "total": len(results),
                "sample_errors": errs}

    ttft = [r["ttft_ms"] for r in ok if r.get("ttft_ms", 0) > 0]
    tpot = [r["tpot_ms"] for r in ok if r.get("tpot_ms", 0) > 0]
    total_out = sum(r.get("output_tokens", 0) for r in ok)
    total_in = sum(r.get("input_tokens", 0) for r in ok)
    wall = data.get("wall_duration_s", 1)
    energy_mj = data.get("energy_mj_delta", {})
    total_energy_j = sum(energy_mj.values()) / 1000 if energy_mj else 0

    return {
        "mean_ttft_ms": round(sum(ttft)/len(ttft), 2) if ttft else None,
        "mean_tpot_ms": round(sum(tpot)/len(tpot), 2) if tpot else None,
        "p50_ttft_ms": round(sorted(ttft)[len(ttft)//2], 2) if ttft else None,
        "p50_tpot_ms": round(sorted(tpot)[len(tpot)//2], 2) if tpot else None,
        "output_tok_s": round(total_out / wall, 1),
        "input_tok_s": round(total_in / wall, 1),
        "total_energy_j": round(total_energy_j, 1),
        "wall_duration_s": round(wall, 1),
        "success_rate_pct": round(len(ok)/len(results)*100, 1),
        "total_requests": len(results),
        "succeeded": len(ok),
    }


if __name__ == "__main__":
    results = {}
    max_requests = 100
    qps = 2.0

    for backend in ["ipc", "ucx"]:
        data = run_pdaf_bench(backend, max_requests, qps)
        results[backend] = data

    # ── Print results ──
    print("\n" + "=" * 100)
    print("  PD+AF M=1: IPC vs UCX Comparison")
    print("  Test: {} requests at {} QPS (sample dataset, il=1024, ol=128)".format(max_requests, qps))
    print("=" * 100)

    # Metrics table
    print(f"\n  {'Metric':<30} {'IPC':<22} {'UCX':<22}")
    print(f"  {'-'*74}")

    for backend in ["ipc", "ucx"]:
        if results.get(backend) and results[backend].get("bench"):
            m = extract_bench_metrics(results[backend]["bench"])
            results[backend]["metrics"] = m

    metrics_keys = [
        ("mean_ttft_ms", "Mean TTFT (ms)"),
        ("mean_tpot_ms", "Mean TPOT (ms)"),
        ("p50_ttft_ms", "P50 TTFT (ms)"),
        ("p50_tpot_ms", "P50 TPOT (ms)"),
        ("output_tok_s", "Output Throughput (tok/s)"),
        ("success_rate_pct", "Success Rate (%)"),
        ("wall_duration_s", "Wall Duration (s)"),
        ("total_energy_j", "Total Energy (J)"),
    ]
    for key, label in metrics_keys:
        line = f"  {label:<30}"
        for backend in ["ipc", "ucx"]:
            m = results.get(backend, {}).get("metrics", {})
            val = m.get(key)
            if val is None:
                line += f" {'N/A':<22}"
            elif isinstance(val, float):
                line += f" {val:<22.2f}"
            else:
                line += f" {str(val):<22}"
        print(line)

    # IPC status / errors
    for backend in ["ipc", "ucx"]:
        ipc_st = results.get(backend, {}).get("ipc_status", {})
        if ipc_st:
            print(f"\n  --- {backend.upper()} IPC handshake messages ---")
            for mod_name, msgs in ipc_st.items():
                for msg in msgs:
                    print(f"  [{mod_name}] {msg}")
        err = results.get(backend, {}).get("metrics", {}).get("error")
        if err:
            print(f"\n  {backend.upper()} ERROR: {err}")
            samples = results[backend]["metrics"].get("sample_errors", [])
            for se in samples:
                print(f"    Sample: {json.dumps(se)}")

    # AFD_HOST_EVENTS breakdown (DA decode server)
    print(f"\n  --- AFD_HOST_EVENTS Breakdown (DA decode server, per-layer timing) ---")

    for backend in ["ipc", "ucx"]:
        if results.get(backend) and results[backend].get("events"):
            stats = compute_stats(results[backend]["events"], "DA")
            if stats:
                results[backend]["da_stats"] = stats
        if results.get(backend) and results[backend].get("events"):
            stats = compute_stats(results[backend]["events"], "DF")
            if stats:
                results[backend]["df_stats"] = stats

    ipc_da = results.get("ipc", {}).get("da_stats")
    ucx_da = results.get("ucx", {}).get("da_stats")
    ipc_df = results.get("ipc", {}).get("df_stats")
    ucx_df = results.get("ucx", {}).get("df_stats")

    if ipc_da and ucx_da:
        print(f"\n  {'Metric':<32} {'IPC(us)':<12} {'UCX(us)':<12} {'Ratio':<10}")
        print(f"  {'-'*66}")

        for cat, label in [("DA_send", "DA→DF send (DA side)"),
                            ("DA_recv", "DA←DF recv (DA side)")]:
            ipc_avg = ipc_da.get(cat, {}).get("avg", 0)
            ucx_avg = ucx_da.get(cat, {}).get("avg", 0)
            ratio = ucx_avg / ipc_avg if ipc_avg > 0 else float('nan')
            print(f"  {label:<32} {ipc_avg:<12.1f} {ucx_avg:<12.1f} {ratio:<10.2f}x")

        ipc_total = ipc_da["DA_send"]["avg"] + ipc_da["DA_recv"]["avg"]
        ucx_total = ucx_da["DA_send"]["avg"] + ucx_da["DA_recv"]["avg"]
        print(f"\n  DA-side comm per decode step:")
        print(f"    IPC: {ipc_total:.1f}us  |  UCX: {ucx_total:.1f}us  |  Ratio (IPC/UCX): {ipc_total/ucx_total:.2f}x" if ucx_total > 0 else f"    IPC: {ipc_total:.1f}us  |  UCX: N/A")

    if ipc_df and ucx_df:
        print(f"\n  --- AFD_HOST_EVENTS Breakdown (DF decode server) ---")
        print(f"  {'Metric':<32} {'IPC(us)':<12} {'UCX(us)':<12} {'Ratio':<10}")
        print(f"  {'-'*66}")
        for cat, label in [("FFN_recv", "DF←DA recv (DF side)"),
                            ("FFN_send", "DF→DA send (DF side)")]:
            ipc_avg = ipc_df.get(cat, {}).get("avg", 0)
            ucx_avg = ucx_df.get(cat, {}).get("avg", 0)
            ratio = ucx_avg / ipc_avg if ipc_avg > 0 else float('nan')
            print(f"  {label:<32} {ipc_avg:<12.1f} {ucx_avg:<12.1f} {ratio:<10.2f}x")

    # Also print how many events were parsed
    for backend in ["ipc", "ucx"]:
        events = results.get(backend, {}).get("events", {})
        for mod in ["DA", "DF"]:
            if mod in events:
                print(f"  [{backend}] {mod} events: {len(events[mod])} layers with data")

    print("=" * 100)

    # Save full results
    results_path = _LOG_DIR / "comparison_ipc_vs_ucx_v2.json"
    # Strip non-serializable data
    save_results = {}
    for bk in ["ipc", "ucx"]:
        save_results[bk] = {
            "metrics": results[bk].get("metrics"),
            "da_stats": results[bk].get("da_stats"),
            "df_stats": results[bk].get("df_stats"),
        }
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")
