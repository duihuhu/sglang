#!/usr/bin/env python3
"""AFD Full Comparison Benchmark: M=1/3 × UCX/IPC × Optimized/Non-optimized.

Measures TPOT, output throughput, TTFT across 8 configurations:
  1. M=1, UCX, non-optimized
  2. M=1, UCX, optimized
  3. M=1, IPC, non-optimized
  4. M=1, IPC, optimized
  5. M=3, UCX, non-optimized
  6. M=3, UCX, optimized
  7. M=3, IPC, non-optimized
  8. M=3, IPC, optimized

"Optimized" = CUDA graph + overlap schedule enabled.
"Non-optimized" = CUDA graph disabled + overlap schedule disabled.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("afd_full_compare")

_REPO = Path("/workspace/sglang")
_CONFIG_PATH = _REPO / "python" / "sglang" / "srt" / "energy" / "af_launch_config.json"
_LAUNCHER = _REPO / "python" / "sglang" / "srt" / "energy" / "af_launcher.py"
_BENCHMARK = _REPO / "benchmark" / "test_motivation" / "AzurePublicDataset" / "benchmark_replay.py"
_LOG_DIR = _REPO / "af_launch_logs" / "full_compare"
_PYTHON = "/workspace/env/af-test/bin/python"
_GPU_INDICES = [0, 1, 4, 5]

# Benchmark parameters
MAX_REQUESTS = 120
QPS = 8.0
SAMPLE_INPUT_LEN = 512
SAMPLE_OUTPUT_LEN = 128


def kill_all():
    for port in [50000, 50010, 50011, 50020, 50021]:
        try:
            r = subprocess.run(
                ["ss", "-tlnp", f"sport = :{port}"],
                capture_output=True, text=True, timeout=10,
            )
            for m in re.finditer(r"pid=(\d+)", r.stdout):
                subprocess.run(["kill", "-9", m.group(1)], capture_output=True, timeout=10)
        except Exception:
            pass
    for t in ["sglang.launch_server", "sglang_router", "af_launcher", "ucx", "mooncake"]:
        subprocess.run(["pkill", "-9", "-f", t], capture_output=True, timeout=10)
    time.sleep(5)


def clean_ipc_artifacts():
    for p in range(1000):
        for f in [f"/dev/shm/afd_ipc_flags_{p}", f"/tmp/afd_ipc_{p}.sock"]:
            try:
                os.unlink(f)
            except Exception:
                pass


def wait_for_router(timeout=300) -> bool:
    import urllib.request
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen("http://127.0.0.1:50000/health", timeout=5)
            return True
        except Exception:
            time.sleep(3)
    return False


def run_single_config(
    micro_batch: int,
    backend: str,
    optimized: bool,
    run_label: str,
) -> dict:
    """Run one configuration and return metrics."""
    opt_str = "optimized" if optimized else "non-optimized"
    logger.info("=" * 70)
    logger.info("  Config: M=%d, %s, %s  [%s]", micro_batch, backend.upper(), opt_str, run_label)
    logger.info("=" * 70)

    kill_all()
    if backend == "ipc":
        clean_ipc_artifacts()

    with open(_CONFIG_PATH) as f:
        cfg = json.load(f)

    cfg["afd"]["comm_backend"] = backend
    cfg["afd"]["dvfs_enabled"] = False
    cfg["tier1"]["enable_tier1_pa"] = False
    cfg["tier1"]["start_with_workload"] = False
    cfg["model"]["mem_fraction_static"] = 0.93

    for mod in cfg["modules"]:
        extra = ["--skip-server-warmup"]
        if not optimized:
            extra += ["--disable-cuda-graph", "--disable-piecewise-cuda-graph"]
        if mod["name"] in ("DA", "DF"):
            extra += ["--afd-micro-batch", str(micro_batch)]
            if not optimized:
                extra += ["--disable-overlap-schedule"]
            extra += ["--max-running-requests", "16"]
        mod["extra_cli_args"] = extra

    server_log_dir = _LOG_DIR / f"logs_{run_label}"
    cfg["logs"]["dir"] = str(server_log_dir)

    tmp_cfg = _LOG_DIR / f"config_{run_label}.json"
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(tmp_cfg, "w") as f:
        json.dump(cfg, f, indent=2)

    if server_log_dir.exists():
        shutil.rmtree(server_log_dir)
    server_log_dir.mkdir(parents=True, exist_ok=True)

    launcher_cmd = [_PYTHON, str(_LAUNCHER), "--config", str(tmp_cfg)]
    logger.info("Launching: %s", " ".join(launcher_cmd))
    launcher_log = _LOG_DIR / f"launcher_{run_label}.log"
    with open(launcher_log, "w") as lf:
        launcher_proc = subprocess.Popen(
            launcher_cmd, stdout=lf, stderr=subprocess.STDOUT,
            start_new_session=True, cwd=str(_REPO),
        )

    if not wait_for_router():
        logger.error("Router not ready for config %s", run_label)
        kill_all()
        return {"error": "router_timeout", "label": run_label}

    logger.info("Router ready, waiting 15s for handshake stabilization...")
    time.sleep(15)

    dump_file = str(_LOG_DIR / f"results_{run_label}.json")
    bench_cmd = [
        _PYTHON, str(_BENCHMARK),
        "--url", "http://127.0.0.1:50000",
        "--speedup", "1.0",
        "--max-requests", str(MAX_REQUESTS),
        "--concurrency", "50",
        "--timeout", "300",
        "--monitor-energy",
        "--gpu-indices", ",".join(str(g) for g in _GPU_INDICES),
        "--scenario-label", run_label,
        "--dump", dump_file,
        "--dataset", "sample",
        "--sample-input-len", str(SAMPLE_INPUT_LEN),
        "--sample-output-len", str(SAMPLE_OUTPUT_LEN),
        "--sample-qps", str(QPS),
        "--seed", "42",
    ]
    logger.info("Benchmark: %s", " ".join(bench_cmd))
    subprocess.run(bench_cmd, cwd=str(_REPO))
    time.sleep(5)
    kill_all()

    bench_data = None
    if os.path.exists(dump_file):
        with open(dump_file) as f:
            bench_data = json.load(f)

    metrics = extract_bench_metrics(bench_data)
    metrics["label"] = run_label
    metrics["micro_batch"] = micro_batch
    metrics["backend"] = backend
    metrics["optimized"] = optimized

    afd_stats = parse_afd_events(server_log_dir)
    metrics["afd_stats"] = afd_stats

    return metrics


def extract_bench_metrics(data) -> dict:
    if data is None:
        return {"error": "no_data"}
    results = data.get("results", [])
    ok = [r for r in results if r.get("success")]
    if not ok:
        return {"error": "no_successful_requests", "total": len(results)}

    ttft = [r["ttft_ms"] for r in ok if r.get("ttft_ms", 0) > 0]
    tpot = [r["tpot_ms"] for r in ok if r.get("tpot_ms", 0) > 0]
    total_out = sum(r.get("output_tokens", 0) for r in ok)
    wall = data.get("wall_duration_s", 1)
    energy_mj = data.get("energy_mj_delta", {})
    total_energy_j = sum(energy_mj.values()) / 1000 if energy_mj else 0

    return {
        "mean_ttft_ms": round(sum(ttft) / len(ttft), 2) if ttft else None,
        "mean_tpot_ms": round(sum(tpot) / len(tpot), 2) if tpot else None,
        "p50_tpot_ms": round(sorted(tpot)[len(tpot) // 2], 2) if tpot else None,
        "p99_tpot_ms": round(sorted(tpot)[int(len(tpot) * 0.99)], 2) if tpot else None,
        "output_tok_s": round(total_out / wall, 1),
        "total_energy_j": round(total_energy_j, 1),
        "wall_duration_s": round(wall, 1),
        "success_rate_pct": round(len(ok) / len(results) * 100, 1),
        "succeeded": len(ok),
        "total_requests": len(results),
    }


def parse_afd_events(log_dir: Path) -> dict:
    stats = {}
    for mod_name in ["DA", "DF"]:
        log_path = log_dir / f"{mod_name}.log"
        if not log_path.exists():
            continue
        send_durs = []
        recv_durs = []
        with open(log_path) as f:
            for line in f:
                if "[AFD_HOST_EVENTS]" in line:
                    m = re.search(r"events=(\[.*\])$", line)
                    if m:
                        try:
                            events = json.loads(m.group(1))
                            for ev in events:
                                if ev.get("event") == "send_end":
                                    send_durs.append(ev.get("send_dur_us", 0))
                                elif ev.get("event") == "recv_end":
                                    recv_durs.append(ev.get("recv_dur_us", 0))
                        except json.JSONDecodeError:
                            pass
        if send_durs or recv_durs:
            stats[mod_name] = {
                "avg_send_us": round(sum(send_durs) / len(send_durs), 1) if send_durs else 0,
                "avg_recv_us": round(sum(recv_durs) / len(recv_durs), 1) if recv_durs else 0,
                "n_send": len(send_durs),
                "n_recv": len(recv_durs),
            }
    return stats


def print_results_table(all_results: list[dict]):
    print("\n" + "=" * 110)
    print("  AFD Full Comparison: M=1/3 × UCX/IPC × Optimized/Non-optimized")
    print(f"  Test: {MAX_REQUESTS} requests @ {QPS} QPS (il={SAMPLE_INPUT_LEN}, ol={SAMPLE_OUTPUT_LEN})")
    print("=" * 110)

    header = (
        f"  {'Config':<32} {'TPOT(ms)':<12} {'P50 TPOT':<12} "
        f"{'Throughput':<14} {'TTFT(ms)':<12} {'Success%':<10} {'Energy(J)':<10}"
    )
    print(f"\n{header}")
    print(f"  {'-' * 104}")

    for r in all_results:
        if r.get("error"):
            print(f"  {r.get('label', '?'):<32} ERROR: {r['error']}")
            continue
        label = r.get("label", "?")
        tpot = r.get("mean_tpot_ms")
        p50 = r.get("p50_tpot_ms")
        thru = r.get("output_tok_s")
        ttft = r.get("mean_ttft_ms")
        succ = r.get("success_rate_pct")
        energy = r.get("total_energy_j")

        tpot_s = f"{tpot:.2f}" if tpot else "N/A"
        p50_s = f"{p50:.2f}" if p50 else "N/A"
        thru_s = f"{thru:.1f} tok/s" if thru else "N/A"
        ttft_s = f"{ttft:.2f}" if ttft else "N/A"
        succ_s = f"{succ:.1f}" if succ else "N/A"
        energy_s = f"{energy:.1f}" if energy else "N/A"

        print(f"  {label:<32} {tpot_s:<12} {p50_s:<12} {thru_s:<14} {ttft_s:<12} {succ_s:<10} {energy_s:<10}")

    # Comparison section
    print(f"\n  --- Relative Comparisons ---")
    grouped = {}
    for r in all_results:
        if r.get("error"):
            continue
        key = (r["micro_batch"], r["backend"])
        grouped.setdefault(key, {})[r["optimized"]] = r

    # Optimization speedup
    print(f"\n  Optimization Speedup (TPOT reduction):")
    for (m, bk), variants in sorted(grouped.items()):
        if True in variants and False in variants:
            base = variants[False].get("mean_tpot_ms")
            opt = variants[True].get("mean_tpot_ms")
            if base and opt and base > 0:
                speedup = base / opt
                reduction = (1 - opt / base) * 100
                print(f"    M={m} {bk.upper()}: {base:.2f}ms → {opt:.2f}ms ({reduction:+.1f}%, {speedup:.2f}x)")

    # IPC vs UCX comparison
    print(f"\n  IPC vs UCX (same M, same optimization):")
    for m in [1, 3]:
        for opt in [False, True]:
            opt_s = "opt" if opt else "non-opt"
            ipc_r = next((r for r in all_results if r.get("micro_batch") == m and r.get("backend") == "ipc" and r.get("optimized") == opt and not r.get("error")), None)
            ucx_r = next((r for r in all_results if r.get("micro_batch") == m and r.get("backend") == "ucx" and r.get("optimized") == opt and not r.get("error")), None)
            if ipc_r and ucx_r:
                ipc_tpot = ipc_r.get("mean_tpot_ms", 0)
                ucx_tpot = ucx_r.get("mean_tpot_ms", 0)
                if ucx_tpot > 0:
                    ratio = ipc_tpot / ucx_tpot
                    print(f"    M={m} {opt_s}: IPC={ipc_tpot:.2f}ms, UCX={ucx_tpot:.2f}ms (IPC/UCX={ratio:.2f}x)")

    # M=1 vs M=3 comparison
    print(f"\n  M=1 vs M=3 (same backend, same optimization):")
    for bk in ["ucx", "ipc"]:
        for opt in [False, True]:
            opt_s = "opt" if opt else "non-opt"
            m1_r = next((r for r in all_results if r.get("micro_batch") == 1 and r.get("backend") == bk and r.get("optimized") == opt and not r.get("error")), None)
            m3_r = next((r for r in all_results if r.get("micro_batch") == 3 and r.get("backend") == bk and r.get("optimized") == opt and not r.get("error")), None)
            if m1_r and m3_r:
                m1_tpot = m1_r.get("mean_tpot_ms", 0)
                m3_tpot = m3_r.get("mean_tpot_ms", 0)
                if m1_tpot > 0:
                    speedup = m1_tpot / m3_tpot
                    print(f"    {bk.upper()} {opt_s}: M=1={m1_tpot:.2f}ms, M=3={m3_tpot:.2f}ms (M=3 speedup={speedup:.2f}x)")

    # AFD communication stats
    print(f"\n  --- AFD Communication Latency (DA side, avg per layer) ---")
    print(f"  {'Config':<32} {'Send(us)':<12} {'Recv(us)':<12} {'Total(us)':<12}")
    print(f"  {'-' * 68}")
    for r in all_results:
        if r.get("error"):
            continue
        da_stats = r.get("afd_stats", {}).get("DA")
        if da_stats:
            send = da_stats.get("avg_send_us", 0)
            recv = da_stats.get("avg_recv_us", 0)
            total = send + recv
            print(f"  {r['label']:<32} {send:<12.1f} {recv:<12.1f} {total:<12.1f}")

    print("=" * 110)


if __name__ == "__main__":
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    configs = [
        (1, "ucx", False, "M1_UCX_noopt"),
        (1, "ucx", True, "M1_UCX_opt"),
        (1, "ipc", False, "M1_IPC_noopt"),
        (1, "ipc", True, "M1_IPC_opt"),
        (3, "ucx", False, "M3_UCX_noopt"),
        (3, "ucx", True, "M3_UCX_opt"),
        # IPC+M=3 skipped: deadlocks in PD mode with concurrent requests
    ]

    all_results = []
    for micro_batch, backend, optimized, label in configs:
        result = run_single_config(micro_batch, backend, optimized, label)
        all_results.append(result)
        logger.info("Completed %s: %s", label, json.dumps(
            {k: v for k, v in result.items() if k != "afd_stats"}, default=str
        ))

    print_results_table(all_results)

    results_path = _LOG_DIR / "full_comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull results saved to {results_path}")
