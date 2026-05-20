#!/usr/bin/env python3
"""Verify --afd-async-schedule and compare with static schedule M=3.

Runs standalone AFD (no PD): DA on GPU 5, DF on GPU 4.
Captures AFD_TIMELINE for both static and async schedules.
"""

import concurrent.futures
import json
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("async_verify")

PYTHON = "/workspace/env/af-test/bin/python3"
MODEL = "/models/Qwen/Qwen3-32B/"

DA_PORT = 50020
DF_PORT = 50021

UCX_BASE_PORT = 25400
UCX_BASE_PORT_ASYNC = 25700
SCHED_PORT = 65432
SCHED_PORT_ASYNC = 65433

GPU_DF = 4
GPU_DA = 5

_log_procs = []


def kill_all():
    for t in ["sglang.launch_server", "sglang_router"]:
        subprocess.run(["pkill", "-9", "-f", t], capture_output=True)
    time.sleep(3)


def wait_port(port, timeout=420):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            s = socket.create_connection(("127.0.0.1", port), timeout=2)
            s.close()
            return True
        except Exception:
            time.sleep(2)
    return False


def run_config(label, async_schedule=False):
    """Start DA+DF with M=3, run requests, capture timeline."""
    global _log_procs
    _log_procs = []

    kill_all()

    da_log = f"/tmp/gantt_DA_{label}.log"
    df_log = f"/tmp/gantt_DF_{label}.log"
    for f in [da_log, df_log]:
        if os.path.exists(f):
            os.remove(f)

    env_base = os.environ.copy()
    env_base["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
    env_base["AFD_DETAILED_TIMING"] = "1"
    env_base["AFD_UCX_TLS"] = "rc,tcp,cuda_copy,cuda_ipc"
    env_base["UCX_LOG_LEVEL"] = "fatal"
    env_base["UCX_WARN_UNUSED_ENV_VARS"] = "n"
    env_base["SGLANG_LOG_LEVEL"] = "error"
    env_base["AFD_UCX_BASE_PORT"] = str(UCX_BASE_PORT_ASYNC if async_schedule else UCX_BASE_PORT)
    env_base["AFD_SCHED_PORT"] = str(SCHED_PORT_ASYNC if async_schedule else SCHED_PORT)
    env_base["AFD_UCX_TIMEOUT"] = "180"

    extra = [
        "--skip-server-warmup",
        "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
        "--afd-micro-batch", "3",
        "--max-running-requests", "64",
        "--mem-fraction-static", "0.85",
    ]
    if async_schedule:
        extra.append("--afd-async-schedule")

    # Use ZMQ for async schedule (UCX per-mb channel has connection
    # ordering issues); UCX for static schedule baseline.
    comm_backend = "ucx"

    # Start DF
    df_env = env_base.copy()
    df_env["CUDA_VISIBLE_DEVICES"] = str(GPU_DF)
    df_cmd = [PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--tp", "1",
        "--host", "127.0.0.1", "--port", str(DF_PORT),
        "--afd-perspective", "ffn", "--afd-comm-backend", comm_backend,
    ] + extra

    log.info("[%s] Starting DF on GPU %d (async=%s)", label, GPU_DF, async_schedule)
    df_fh = open(df_log, "w")
    df_proc = subprocess.Popen(df_cmd, stdout=df_fh, stderr=subprocess.STDOUT,
                               env=df_env, start_new_session=True)
    _log_procs.append(("DF", df_proc, df_fh))

    # Start DA
    da_env = env_base.copy()
    da_env["CUDA_VISIBLE_DEVICES"] = str(GPU_DA)
    da_cmd = [PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--tp", "1",
        "--host", "127.0.0.1", "--port", str(DA_PORT),
        "--afd-perspective", "attn", "--afd-comm-backend", comm_backend,
    ] + extra

    log.info("[%s] Starting DA on GPU %d", label, GPU_DA)
    da_fh = open(da_log, "w")
    da_proc = subprocess.Popen(da_cmd, stdout=da_fh, stderr=subprocess.STDOUT,
                               env=da_env, start_new_session=True)
    _log_procs.append(("DA", da_proc, da_fh))

    # Wait for both servers
    if not wait_port(DF_PORT):
        log.error("[%s] DF did not start", label)
        shutdown()
        return None
    log.info("[%s] DF ready", label)

    if not wait_port(DA_PORT):
        log.error("[%s] DA did not start", label)
        shutdown()
        return None
    log.info("[%s] DA ready", label)

    time.sleep(15)

    # Warmup
    import requests as req
    warm_payloads = [
        {"text": f"Hi {i}:", "sampling_params": {"max_new_tokens": 16, "temperature": 0.0}}
        for i in range(16)
    ]
    log.info("[%s] Warmup (%d requests)...", label, len(warm_payloads))

    def _send(payload):
        try:
            req.post(f"http://127.0.0.1:{DA_PORT}/generate", json=payload, timeout=120)
        except Exception:
            pass

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(_send, warm_payloads))
    time.sleep(5)
    log.info("[%s] Warmup done", label)

    # Benchmark
    bench_payloads = [
        {"text": f"Tell me something interesting about the number {i}:",
         "sampling_params": {"max_new_tokens": 64, "temperature": 0.0}}
        for i in range(32)
    ]
    log.info("[%s] Running benchmark (%d requests)...", label, len(bench_payloads))
    t_start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(_send, bench_payloads))
    t_end = time.monotonic()
    time.sleep(15)
    log.info("[%s] Benchmark done in %.1fs", label, t_end - t_start)

    shutdown()

    # Parse timeline
    timeline_data = parse_timeline(da_log, df_log, label)
    return timeline_data


def shutdown():
    global _log_procs
    for name, proc, fh in _log_procs:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
    time.sleep(8)
    for name, proc, fh in _log_procs:
        try:
            proc.kill()
        except Exception:
            pass
        try:
            fh.close()
        except Exception:
            pass
    _log_procs = []
    time.sleep(10)
    kill_all()
    time.sleep(5)


def parse_timeline(da_log, df_log, label):
    """Parse AFD_TIMELINE from logs and compute per-layer metrics."""
    results = {"label": label, "da": None, "df": None}

    for side, fpath in [("da", da_log), ("df", df_log)]:
        try:
            with open(fpath) as f:
                content = f.read()
        except Exception:
            continue

        timelines = re.findall(r"\[AFD_TIMELINE\].*?timeline=(\[.*?\])\s*$", content, re.MULTILINE)
        log.info("[%s] %s: found %d AFD_TIMELINE entries", label, side.upper(), len(timelines))

        if not timelines:
            continue

        # Use the last timeline (most stable, post-warmup)
        tl = json.loads(timelines[-1])
        results[side] = tl

    return results


def analyze_and_compare(static_data, async_data):
    """Analyze and compare static vs async schedule timelines."""
    print("\n" + "=" * 90)
    print("  AFD M=3 Schedule Comparison: Static vs Async (Event-Driven)")
    print("=" * 90)

    for label, data in [("STATIC", static_data), ("ASYNC", async_data)]:
        if not data or not data.get("da"):
            print(f"\n  [{label}] No DA timeline data available")
            continue

        da_tl = data["da"]
        layers_1_5 = [e for e in da_tl if 1 <= e["layer"] <= 5]
        if not layers_1_5:
            print(f"\n  [{label}] No layer 1-5 data")
            continue

        t0 = layers_1_5[0]["t_start_ms"]

        print(f"\n  --- [{label}] DA Timeline (Layers 1-5) ---")
        print(f"  {'Step':<5} {'Stage':<14} {'Layer':<6} {'MB':<4} {'Start':<10} {'End':<10} {'Dur(ms)':<8}")
        print(f"  {'-' * 60}")

        for e in layers_1_5[:30]:
            stage = "A-stage" if "STAGE_A" in e["stage"] else "F-stage"
            s = e["t_start_ms"] - t0
            en = e["t_end_ms"] - t0
            print(f"  {e['step']:<5} {stage:<14} L{e['layer']:<5} mb{e['mb']:<3} {s:<10.3f} {en:<10.3f} {e['dur_ms']:<8.3f}")

        # Per-layer cycle time
        print(f"\n  [{label}] Per-layer cycle time:")
        for layer in range(1, 6):
            events = [e for e in layers_1_5 if e["layer"] == layer]
            if not events:
                break
            layer_start = min(e["t_start_ms"] for e in events) - t0
            layer_end = max(e["t_end_ms"] for e in events) - t0
            print(f"    Layer {layer}: {layer_end - layer_start:.3f}ms")

        # Check if mb0 advances before mb1/mb2 return
        print(f"\n  [{label}] Does mb0 advance to next layer before mb1/mb2?")
        for layer in range(1, 5):
            a_next = [e for e in da_tl if e["layer"] == layer + 1 and "STAGE_A" in e["stage"] and e["mb"] == 0]
            f_cur_mb2 = [e for e in da_tl if e["layer"] == layer and "STAGE_F" in e["stage"] and e["mb"] == 2]
            if a_next and f_cur_mb2:
                a_start = a_next[0]["t_start_ms"] - t0
                f2_end = f_cur_mb2[0]["t_end_ms"] - t0
                if a_start < f2_end:
                    print(f"    Layer {layer}→{layer+1}: YES! A(L{layer+1},mb0) starts at {a_start:.3f}ms, F(L{layer},mb2) ends at {f2_end:.3f}ms")
                else:
                    print(f"    Layer {layer}→{layer+1}: NO. A(L{layer+1},mb0) starts at {a_start:.3f}ms, F(L{layer},mb2) ends at {f2_end:.3f}ms (gap={a_start-f2_end:.3f}ms)")

    # Summary comparison
    if static_data and static_data.get("da") and async_data and async_data.get("da"):
        print("\n  --- COMPARISON SUMMARY ---")
        for label, data in [("STATIC", static_data), ("ASYNC", async_data)]:
            da_tl = data["da"]
            layers = [e for e in da_tl if 1 <= e["layer"] <= 5]
            if not layers:
                continue
            t0 = layers[0]["t_start_ms"]
            total_time = max(e["t_end_ms"] for e in layers) - t0
            n_layers = len(set(e["layer"] for e in layers))
            avg_cycle = total_time / n_layers if n_layers > 0 else 0
            print(f"    {label}: {n_layers} layers in {total_time:.3f}ms, avg {avg_cycle:.3f}ms/layer")

    print("\n" + "=" * 90)


def main():
    kill_all()

    # Phase 1: Static schedule M=3
    log.info("=" * 60)
    log.info("PHASE 1: Static schedule M=3")
    log.info("=" * 60)
    static_data = run_config("m3_static", async_schedule=False)

    # Phase 2: Async schedule M=3
    log.info("=" * 60)
    log.info("PHASE 2: Async schedule M=3")
    log.info("=" * 60)
    async_data = run_config("m3_async", async_schedule=True)

    # Analysis
    analyze_and_compare(static_data, async_data)

    # Save results
    results = {
        "static": {"label": "m3_static", "has_data": bool(static_data and static_data.get("da"))},
        "async": {"label": "m3_async", "has_data": bool(async_data and async_data.get("da"))},
    }
    out_path = "/tmp/async_schedule_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved to %s", out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
