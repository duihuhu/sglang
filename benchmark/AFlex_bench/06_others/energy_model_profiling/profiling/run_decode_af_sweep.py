#!/usr/bin/env python3
"""Profile Decode iteration TPOT and energy for (freq_a, freq_f, bs) combinations.

Strategy: Send bs requests with very short input (il=32, fast prefill) and long
output (ol=512), so all requests quickly enter decode and stay there for many
iterations. Parse DA log for steady-state iterations (peak running_req).

Measures per-iteration: total_ms, pipeline_ms, drain_ms, and per-GPU energy.
Also measures user-perceived TPOT from end-to-end latency.

Output: TSV with columns:
  tp, freq_a, freq_f, batch_size, input_len, output_len,
  TPOT_ms, total_iter_ms, pipeline_ms, drain_ms,
  A_energy_per_iter_mj, F_energy_per_iter_mj, peak_running_req, n_steady_iters

Usage:
    /workspace/env/sglang-tier/bin/python run_decode_af_sweep.py
    /workspace/env/sglang-tier/bin/python run_decode_af_sweep.py --freq 930 1170 1410 --bs 4 8 16 32 64
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("decode_af_sweep")

PYTHON = "/workspace/env/sglang-tier/bin/python"
MODEL = "/models/Qwen/Qwen3-32B/"
HERE = Path(__file__).resolve().parent

GPU_PA, GPU_PF, GPU_DA, GPU_DF = 7, 6, 5, 4
PA_PORT, PF_PORT = 51010, 51011
DA_PORT, DF_PORT = 51020, 51021
ROUTER_PORT = 51000
BOOTSTRAP_PORT = 29999
UCX_P, UCX_D = 26200, 26300
SCHED_P, SCHED_D = 66400, 66500
ALL_PORTS = [ROUTER_PORT, PA_PORT, PF_PORT, DA_PORT, DF_PORT]

FREQS = [450, 690, 930, 1170, 1410]
DEFAULT_BS = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128]
# Short input for fast prefill, long output for many decode iterations
INPUT_LEN = 32
OUTPUT_LEN = 512


def _get_pids_on_port(port):
    result = subprocess.run(["ss", "-tlnp", f"sport = :{port}"], capture_output=True, text=True)
    return list(set(re.findall(r'pid=(\d+)', result.stdout)))


def kill_servers():
    for port in ALL_PORTS:
        for pid in _get_pids_on_port(port):
            subprocess.run(["kill", "-9", pid], capture_output=True)
    time.sleep(3)
    for idx in [GPU_DF, GPU_DA, GPU_PF, GPU_PA]:
        subprocess.run(["nvidia-smi", "-rgc", "-i", str(idx)], capture_output=True)


def wait_port(host, port, timeout=300):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            s = socket.create_connection((host, port), timeout=2); s.close(); return True
        except:
            time.sleep(2)
    return False


def wait_health(url, timeout=180):
    import urllib.request
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=5); return True
        except:
            time.sleep(3)
    return False


def set_gpu_freq(freq_mhz: int, gpu_indices: list[int]):
    for idx in gpu_indices:
        subprocess.run(["nvidia-smi", "-lgc", str(freq_mhz), "-i", str(idx)], capture_output=True)
    time.sleep(0.3)


def reset_gpu_freq(gpu_indices: list[int]):
    for idx in gpu_indices:
        subprocess.run(["nvidia-smi", "-rgc", "-i", str(idx)], capture_output=True)


def get_gpu_energy_mj(gpu_indices: list[int]) -> dict[int, float]:
    try:
        import pynvml
        pynvml.nvmlInit()
        result = {}
        for idx in gpu_indices:
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            result[idx] = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        pynvml.nvmlShutdown()
        return result
    except Exception:
        return {idx: 0 for idx in gpu_indices}


def start_server():
    """Start full PD+AF server."""
    kill_servers()
    time.sleep(3)
    procs = []
    env_base = os.environ.copy()
    env_base['SGLANG_DISABLE_REQUEST_LOGGING'] = 'true'
    env_base['UCX_LOG_LEVEL'] = 'fatal'
    env_base['AFD_UCX_TLS'] = 'rc,tcp,cuda_copy,cuda_ipc'
    env_base['SGLANG_DISAGGREGATION_THREAD_POOL_SIZE'] = '128'
    env_base['SGLANG_DISAGGREGATION_QUEUE_SIZE'] = '32'
    env_base['SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE'] = '0'
    env_base['AFD_ASYNC_PIPELINE'] = '1'

    extra = [
        '--skip-server-warmup', '--disable-cuda-graph', '--disable-piecewise-cuda-graph',
        '--afd-micro-batch', '2', '--afd-disagg-interleave-poll',
        '--disable-radix-cache', '--num-reserved-decode-tokens', '32',
        '--afd-async-pipeline',
    ]

    def _start(name, perspective, disagg_mode, port, ucx_base, sched_port,
               visible_gpus, base_gpu_id, peer_device, nvml_idx, ffn_host=None):
        env = env_base.copy()
        env['CUDA_VISIBLE_DEVICES'] = visible_gpus
        env['AFD_UCX_BASE_PORT'] = str(ucx_base)
        env['AFD_SCHED_PORT'] = str(sched_port)
        env['SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT'] = '600'
        env['SGLANG_DISAGGREGATION_WAITING_TIMEOUT'] = '600'
        env['AFD_NVML_DEVICE_INDEX'] = str(nvml_idx)
        env['AFD_IPC_SYNC_MODE'] = 'ipc_event'
        env['AFD_IPC_PEER_DEVICE'] = str(peer_device)
        if ffn_host:
            env['AFD_UCX_FFN_HOST'] = ffn_host
        cmd = [PYTHON, '-m', 'sglang.launch_server',
            '--model-path', MODEL, '--tp', '1',
            '--host', '127.0.0.1', '--port', str(port),
            '--afd-perspective', perspective, '--afd-comm-backend', 'ipc_cpp',
            '--mem-fraction-static', '0.85', '--base-gpu-id', str(base_gpu_id),
            '--disaggregation-mode', disagg_mode,
            '--disaggregation-transfer-backend', 'mooncake',
            '--disaggregation-bootstrap-port', str(BOOTSTRAP_PORT),
            '--disaggregation-ib-device', 'mlx5_4'] + extra
        fh = open(HERE / f'logs/{name}.log', 'w')
        p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        procs.append((name, p, fh))

    (HERE / "logs").mkdir(parents=True, exist_ok=True)

    p_vis = f'{GPU_PF},{GPU_PA}'
    d_vis = f'{GPU_DF},{GPU_DA}'

    log.info("Starting PD+AF server...")

    # Prefill side (always at max freq for fast prefill)
    _start('pf', 'ffn', 'prefill', PF_PORT, UCX_P, SCHED_P, p_vis, 0, 1, GPU_PF)
    time.sleep(2)
    _start('pa', 'attn', 'prefill', PA_PORT, UCX_P, SCHED_P, p_vis, 1, 0, GPU_PA, ffn_host='127.0.0.1')
    if not wait_port('127.0.0.1', PA_PORT) or not wait_port('127.0.0.1', PF_PORT):
        log.error("Prefill failed"); return None

    # Decode side
    _start('df', 'ffn', 'decode', DF_PORT, UCX_D, SCHED_D, d_vis, 0, 1, GPU_DF)
    time.sleep(2)
    _start('da', 'attn', 'decode', DA_PORT, UCX_D, SCHED_D, d_vis, 1, 0, GPU_DA, ffn_host='127.0.0.1')
    if not wait_port('127.0.0.1', DA_PORT) or not wait_port('127.0.0.1', DF_PORT):
        log.error("Decode failed"); return None

    # Lock prefill to max freq
    set_gpu_freq(1410, [GPU_PF, GPU_PA])

    # Router
    time.sleep(5)
    router_cmd = [PYTHON, '-m', 'sglang_router.launch_router',
        '--pd-disaggregation', '--mini-lb',
        '--prefill', f'http://127.0.0.1:{PA_PORT}', '--decode', f'http://127.0.0.1:{DA_PORT}',
        '--host', '127.0.0.1', '--port', str(ROUTER_PORT)]
    rf = open(HERE / 'logs/router.log', 'w')
    rp = subprocess.Popen(router_cmd, stdout=rf, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(('router', rp, rf))

    if not wait_health(f'http://127.0.0.1:{ROUTER_PORT}/health', 120):
        log.error("Router failed"); return None

    log.info("Server ready!")
    return procs


def send_requests_async(url: str, bs: int, input_len: int, output_len: int, timeout: float = 600):
    """Send bs concurrent requests, return futures."""
    import concurrent.futures
    import requests as req_lib

    prompt = "Hello " * (input_len // 2)
    payload = {"text": prompt, "sampling_params": {"max_new_tokens": output_len, "temperature": 0.0}}

    def _send(i):
        t0 = time.perf_counter()
        try:
            resp = req_lib.post(f"{url}/generate", json=payload, timeout=timeout)
            t1 = time.perf_counter()
            if resp.status_code == 200:
                return {"success": True, "latency_s": t1 - t0, "idx": i}
        except Exception as e:
            t1 = time.perf_counter()
            return {"success": False, "latency_s": t1 - t0, "idx": i, "error": str(e)}
        return {"success": False, "latency_s": t1 - t0, "idx": i}

    with concurrent.futures.ThreadPoolExecutor(max_workers=bs) as ex:
        futures = [ex.submit(_send, i) for i in range(bs)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    return results


def parse_da_log_steady(da_log: Path, offset: int) -> list[dict]:
    """Parse DA log for decode iterations. Returns all entries with running_req info."""
    entries = []
    if not da_log.exists():
        return entries

    with open(da_log) as f:
        f.seek(offset)
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.search(
            r'AFD_FWD_OVERHEAD.*total=([0-9.]+)ms.*pipeline=([0-9.]+)ms.*drain=([0-9.]+)ms',
            line)
        if m:
            entry = {
                "total_ms": float(m.group(1)),
                "pipeline_ms": float(m.group(2)),
                "drain_ms": float(m.group(3)),
                "running_req": 0,
            }
            # Next line usually has Decode batch info
            if i + 1 < len(lines):
                m2 = re.search(r'#running-req: (\d+)', lines[i + 1])
                if m2:
                    entry["running_req"] = int(m2.group(1))
            entries.append(entry)
        i += 1

    return entries


def profile_one_config(url: str, freq_a: int, freq_f: int, bs: int,
                       input_len: int, output_len: int) -> dict | None:
    """Profile one (freq_a, freq_f, bs) config.

    1. Lock DA=freq_a, DF=freq_f
    2. Send bs requests (short il=32, long ol=512)
    3. Wait for all to complete
    4. Parse DA log: find peak running_req period (steady state)
    5. Compute TPOT from e2e latency and iteration metrics from log
    """
    # Lock decode GPUs
    set_gpu_freq(freq_a, [GPU_DA])
    set_gpu_freq(freq_f, [GPU_DF])
    time.sleep(0.5)

    # Record DA log position
    da_log = HERE / "logs" / "da.log"
    da_log_offset = da_log.stat().st_size if da_log.exists() else 0

    # Measure energy before
    energy_before = get_gpu_energy_mj([GPU_DA, GPU_DF])
    t_start = time.perf_counter()

    # Send all requests concurrently
    results = send_requests_async(url, bs, input_len, output_len, timeout=600)
    success = sum(1 for r in results if r.get("success"))

    t_end = time.perf_counter()
    energy_after = get_gpu_energy_mj([GPU_DA, GPU_DF])
    duration_s = t_end - t_start

    if success < max(1, bs * 0.5):
        log.warning("    FAILED: %d/%d succeeded (freq_a=%d freq_f=%d bs=%d)",
                    success, bs, freq_a, freq_f, bs)
        return None

    time.sleep(1)

    # Parse DA log
    entries = parse_da_log_steady(da_log, da_log_offset)
    if not entries:
        log.warning("    No DA log entries found")
        return None

    # Find steady-state: peak running_req period
    max_running = max(e["running_req"] for e in entries)
    # Use entries where running_req >= 50% of peak
    threshold = max(max_running * 0.5, 1)
    steady = [e for e in entries if e["running_req"] >= threshold]

    if len(steady) < 3:
        # Fallback: use all decode entries
        steady = [e for e in entries if e["running_req"] > 0]

    if not steady:
        log.warning("    No steady-state entries")
        return None

    # Compute iteration metrics from steady state
    avg_total_ms = np.mean([e["total_ms"] for e in steady])
    avg_pipeline_ms = np.mean([e["pipeline_ms"] for e in steady])
    avg_drain_ms = np.mean([e["drain_ms"] for e in steady])
    peak_running = max_running

    # TPOT from end-to-end latency (user-perceived)
    latencies = [r["latency_s"] for r in results if r.get("success")]
    avg_e2e_s = np.mean(latencies)
    tpot_ms = avg_e2e_s / output_len * 1000

    # Energy per iteration
    n_total_iters = len([e for e in entries if e["running_req"] > 0])
    energy_a = energy_after.get(GPU_DA, 0) - energy_before.get(GPU_DA, 0)
    energy_f = energy_after.get(GPU_DF, 0) - energy_before.get(GPU_DF, 0)
    a_energy_per_iter = energy_a / max(n_total_iters, 1)
    f_energy_per_iter = energy_f / max(n_total_iters, 1)

    return {
        "tp": 1,
        "freq_a": freq_a,
        "freq_f": freq_f,
        "batch_size": bs,
        "input_len": input_len,
        "output_len": output_len,
        "TPOT_ms": round(tpot_ms, 2),
        "total_iter_ms": round(avg_total_ms, 2),
        "pipeline_ms": round(avg_pipeline_ms, 2),
        "drain_ms": round(avg_drain_ms, 2),
        "A_energy_per_iter_mj": round(a_energy_per_iter, 2),
        "F_energy_per_iter_mj": round(f_energy_per_iter, 2),
        "peak_running_req": peak_running,
        "n_steady_iters": len(steady),
        "n_total_iters": n_total_iters,
        "duration_s": round(duration_s, 2),
        "success_rate": round(success / bs, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Profile Decode TPOT for (freq_a, freq_f, bs) combinations")
    parser.add_argument("--freq", type=int, nargs="+", default=FREQS,
                        help="Frequency values to sweep (both A and F)")
    parser.add_argument("--bs", type=int, nargs="+", default=DEFAULT_BS)
    parser.add_argument("--il", type=int, default=INPUT_LEN)
    parser.add_argument("--ol", type=int, default=OUTPUT_LEN)
    parser.add_argument("--output", type=str, default=str(HERE / "data_decode_af/decode_af_sweep.tsv"))
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate all (freq_a, freq_f) combinations
    freq_pairs = [(fa, ff) for fa in args.freq for ff in args.freq]
    total = len(freq_pairs) * len(args.bs)

    log.info("=" * 60)
    log.info("  DECODE AF SWEEP (TPOT + Energy)")
    log.info("  Freq pairs: %d (%d × %d)", len(freq_pairs), len(args.freq), len(args.freq))
    log.info("  Batch sizes: %s", args.bs)
    log.info("  Input len: %d (short for fast prefill)", args.il)
    log.info("  Output len: %d (long for many decode iters)", args.ol)
    log.info("  Total configs: %d", total)
    log.info("  Output: %s", out_path)
    log.info("=" * 60)

    # Start server
    procs = start_server()
    if procs is None:
        log.error("Failed to start server")
        return

    url = f"http://127.0.0.1:{ROUTER_PORT}"

    # Warmup
    log.info("Warming up...")
    send_requests_async(url, 4, 32, 16)
    time.sleep(3)

    results = []
    count = 0

    try:
        for freq_a, freq_f in freq_pairs:
            for bs in args.bs:
                count += 1
                log.info("  [%d/%d] freq_a=%d freq_f=%d bs=%d",
                         count, total, freq_a, freq_f, bs)

                r = profile_one_config(url, freq_a, freq_f, bs, args.il, args.ol)
                if r:
                    results.append(r)
                    log.info("    OK: TPOT=%.1fms iter=%.1fms drain=%.1fms peak_rr=%d n_steady=%d",
                             r["TPOT_ms"], r["total_iter_ms"], r["drain_ms"],
                             r["peak_running_req"], r["n_steady_iters"])

                # Incremental save every 10 results
                if results and len(results) % 10 == 0:
                    import pandas as pd
                    pd.DataFrame(results).to_csv(out_path, sep="\t", index=False)

                time.sleep(0.5)

    except KeyboardInterrupt:
        log.info("Interrupted!")
    finally:
        # Final save
        if results:
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv(out_path, sep="\t", index=False)
            log.info("Saved %d results to %s", len(df), out_path)

        # Cleanup
        log.info("Cleaning up...")
        reset_gpu_freq([GPU_DA, GPU_DF, GPU_PA, GPU_PF])
        for _, p, fh in procs:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except:
                pass
        kill_servers()

    log.info("Done! %d/%d configs successful.", len(results), total)


if __name__ == "__main__":
    main()
