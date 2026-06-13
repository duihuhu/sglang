#!/usr/bin/env python3
"""
Decode Pipeline Coupled Profiling — TP=2 version.

Extends the original bench_decode_pipeline.py to profile TP=2 configurations,
covering the input_len/batch_size ranges seen in Azure Code and Conv traces.

Architecture (8 GPUs):
    PF(ffn, TP=2) on GPU 0,1  +  PA(attn, TP=2) on GPU 2,3  (prefill, max freq)
    DF(ffn, TP=2) on GPU 4,5  +  DA(attn, TP=2) on GPU 6,7  (decode, freq swept)

Output format (TSV, same as v1):
    tp  M  f_A  f_F  input_len  batch_size  iter_lat_us  DA_energy_mj  DF_energy_mj

Note: DA_energy_mj = sum of both DA GPUs per iteration
      DF_energy_mj = sum of both DF GPUs per iteration

Usage:
    /workspace/env/sglang-tier/bin/python bench_decode_pipeline_tp2.py --quick
    /workspace/env/sglang-tier/bin/python bench_decode_pipeline_tp2.py
"""

import argparse
import concurrent.futures
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests as req

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bench_pipeline_tp2")

PYTHON = "/workspace/env/sglang-tier/bin/python"
MODEL = "/models/Qwen/Qwen3-32B/"
SCRIPT_DIR = Path(__file__).resolve().parent

# Sweep grid — covers Azure Code (IL up to ~7700) and Conv (IL up to ~7800) traces
# Skip 210MHz (never selected by DVFS in practice for TP=2)
DEFAULT_FREQS = [450, 690, 930, 1170, 1410]
DEFAULT_INPUT_LENS = [128, 512, 1024, 2048, 4096]
DEFAULT_BATCH_SIZES = [1, 4, 8, 16, 32, 64, 128, 256]
DEFAULT_M_VALUES = [1, 2]
GEN_TOKENS = 128

# GPU layout — TP=2 per component
P_CVD = "0,1,2,3"  # prefill: PF(0,1) + PA(2,3)
D_CVD = "4,5,6,7"  # decode:  DF(4,5) + DA(6,7)
DA_NVML_GPUS = [6, 7]
DF_NVML_GPUS = [4, 5]
TP = 2

# Ports
PF_PORT = 55010
PA_PORT = 55011
DF_PORT = 55020
DA_PORT = 55021
ROUTER_PORT = 55000
BOOTSTRAP_PORT = 55099
UCX_P = 55100
UCX_D = 55200
SCHED_P = 55301
SCHED_D = 55302
ALL_PORTS = [PF_PORT, PA_PORT, DF_PORT, DA_PORT, ROUTER_PORT]


class DVFSHelper:
    def __init__(self):
        sys.path.insert(0, str(SCRIPT_DIR.parent.parent / "python"))
        from sglang.srt.layers.dvfs import DVFSController
        self._cls = DVFSController
        self._ctrls = {}

    def _ctrl(self, gpu_id):
        if gpu_id not in self._ctrls:
            self._ctrls[gpu_id] = self._cls(gpu_id)
        return self._ctrls[gpu_id]

    def lock(self, gpu_id, freq):
        return self._ctrl(gpu_id).lock_sm_clock(freq)

    def unlock(self, gpu_id):
        self._ctrl(gpu_id).unlock_sm_clock()

    def energy_mj(self, gpu_id):
        return self._ctrl(gpu_id).get_energy_mj()

    def lock_prefill_max(self):
        for g in [0, 1, 2, 3]:
            self.lock(g, 1410)

    def lock_decode(self, f_a, f_f):
        for g in DA_NVML_GPUS:
            self.lock(g, f_a)
        for g in DF_NVML_GPUS:
            self.lock(g, f_f)

    def unlock_all(self):
        for g in range(8):
            self.unlock(g)

    def da_energy_mj(self):
        return sum(self.energy_mj(g) for g in DA_NVML_GPUS)

    def df_energy_mj(self):
        return sum(self.energy_mj(g) for g in DF_NVML_GPUS)


def kill_all_servers():
    for port in ALL_PORTS + [BOOTSTRAP_PORT]:
        r = subprocess.run(["ss", "-tlnp", f"sport = :{port}"],
                           capture_output=True, text=True)
        for m in re.finditer(r'pid=(\d+)', r.stdout):
            subprocess.run(["kill", "-9", m.group(1)], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "sglang.launch_server"],
                   capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "sglang_router"],
                   capture_output=True)
    time.sleep(5)
    # Wait for ports to actually free up
    for _ in range(20):
        all_free = True
        for port in ALL_PORTS:
            r = subprocess.run(["ss", "-tlnp", f"sport = :{port}"],
                               capture_output=True, text=True)
            if "LISTEN" in r.stdout:
                all_free = False
                for m2 in re.finditer(r'pid=(\d+)', r.stdout):
                    subprocess.run(["kill", "-9", m2.group(1)], capture_output=True)
                break
        if all_free:
            break
        time.sleep(1)
    else:
        log.warning("Some ports still in use after cleanup attempts")


def wait_health(port, timeout=300):
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2)
                s.connect(("127.0.0.1", port))
                break
        except (ConnectionRefusedError, OSError):
            time.sleep(2)
    else:
        return False
    while time.time() < deadline:
        try:
            r = req.get(f"http://127.0.0.1:{port}/health", timeout=5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def start_pdaf_stack(M: int, log_dir: Path):
    """Start full 8-process PDAF stack (TP=2) + router."""
    log_dir.mkdir(parents=True, exist_ok=True)
    procs = []

    env_base = os.environ.copy()
    env_base["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
    env_base["SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION"] = "false"
    env_base["UCX_LOG_LEVEL"] = "fatal"
    env_base["AFD_UCX_TLS"] = "rc,tcp,cuda_copy,cuda_ipc"
    env_base["SGLANG_DISAGGREGATION_THREAD_POOL_SIZE"] = "128"
    env_base["AFD_ASYNC_PIPELINE"] = "1"

    common = [
        "--model-path", MODEL, "--tp", str(TP),
        "--host", "127.0.0.1",
        "--afd-comm-backend", "ipc_cpp",
        "--afd-micro-batch", str(M),
        "--mem-fraction-static", "0.85",
        "--max-running-requests", "512",
        "--skip-server-warmup",
        "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
        "--afd-disagg-interleave-poll",
        "--disable-radix-cache",
        "--num-reserved-decode-tokens", "512",
        "--disaggregation-transfer-backend", "mooncake",
        "--disaggregation-bootstrap-port", str(BOOTSTRAP_PORT),
        "--disaggregation-ib-device", "mlx5_4",
    ]

    def _env(cvd, ucx_base, sched_port, peer_device, nvml_indices, ffn_host=None):
        e = env_base.copy()
        e["CUDA_VISIBLE_DEVICES"] = cvd
        e["AFD_UCX_BASE_PORT"] = str(ucx_base)
        e["AFD_SCHED_PORT"] = str(sched_port)
        e["AFD_IPC_SYNC_MODE"] = "ipc_event"
        e["AFD_IPC_PEER_DEVICE"] = str(peer_device)
        e["AFD_NVML_DEVICE_INDICES"] = nvml_indices
        e["AFD_NVML_DEVICE_INDEX"] = nvml_indices.split(",")[0]
        e["SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT"] = "600"
        e["SGLANG_DISAGGREGATION_WAITING_TIMEOUT"] = "600"
        if ffn_host:
            e["AFD_UCX_FFN_HOST"] = ffn_host
        return e

    def _cmd(port, perspective, disagg, base_gpu_id, attn_tp=None, ffn_tp=None):
        cmd = [PYTHON, "-m", "sglang.launch_server",
               "--port", str(port),
               "--afd-perspective", perspective,
               "--disaggregation-mode", disagg,
               "--base-gpu-id", str(base_gpu_id)] + common
        if attn_tp is not None:
            cmd += ["--afd-attn-tp", str(attn_tp)]
        if ffn_tp is not None:
            cmd += ["--afd-ffn-tp", str(ffn_tp)]
        return cmd

    def _start(name, cmd, env):
        fh = open(log_dir / f"{name}.log", "w")
        p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                             start_new_session=True)
        procs.append((name, p, fh))

    # Prefill: PF(ffn, TP=2, base=0) on GPU 0,1; PA(attn, TP=2, base=2) on GPU 2,3
    _start("pf", _cmd(PF_PORT, "ffn", "prefill", 0),
           _env(P_CVD, UCX_P, SCHED_P, peer_device=2, nvml_indices="0,1"))
    time.sleep(2)
    _start("pa", _cmd(PA_PORT, "attn", "prefill", 2),
           _env(P_CVD, UCX_P, SCHED_P, peer_device=0, nvml_indices="2,3",
                ffn_host="127.0.0.1"))
    time.sleep(2)

    # Decode: DF(ffn, TP=2, base=0) on GPU 4,5; DA(attn, TP=2, base=2) on GPU 6,7
    _start("df", _cmd(DF_PORT, "ffn", "decode", 0, attn_tp=TP),
           _env(D_CVD, UCX_D, SCHED_D, peer_device=2, nvml_indices="4,5"))
    time.sleep(5)
    _start("da", _cmd(DA_PORT, "attn", "decode", 2, ffn_tp=TP),
           _env(D_CVD, UCX_D, SCHED_D, peer_device=0, nvml_indices="6,7",
                ffn_host="127.0.0.1"))

    log.info("Waiting for all 4 processes to become healthy (TP=2)...")
    for port, name in [(PF_PORT, "PF"), (PA_PORT, "PA"),
                       (DF_PORT, "DF"), (DA_PORT, "DA")]:
        if not wait_health(port, 600):
            log.error("%s (port %d) failed to start", name, port)
            _cleanup(procs)
            return None
        log.info("  %s ready (port %d)", name, port)

    router_cmd = [PYTHON, "-m", "sglang_router.launch_router",
                  "--pd-disaggregation", "--mini-lb",
                  "--prefill", f"http://127.0.0.1:{PA_PORT}",
                  "--decode", f"http://127.0.0.1:{DA_PORT}",
                  "--host", "127.0.0.1", "--port", str(ROUTER_PORT)]
    rf = open(log_dir / "router.log", "w")
    rp = subprocess.Popen(router_cmd, env=os.environ.copy(), stdout=rf,
                          stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(("router", rp, rf))

    if not wait_health(ROUTER_PORT, 60):
        log.error("Router failed to start")
        _cleanup(procs)
        return None

    log.info("PDAF TP=2 stack ready (M=%d): router at %d", M, ROUTER_PORT)
    return procs


def _cleanup(procs):
    for name, p, fh in procs:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        fh.close()
    time.sleep(3)


MAX_CONFIG_TIME_S = 180


def measure_one_config(dvfs: DVFSHelper, bs: int, il: int, n_warmup: int = 1):
    """Send bs concurrent requests with GEN_TOKENS output.
    Returns (iter_lat_us, da_energy_mj, df_energy_mj) or None on failure.
    Energy values are summed across both GPUs per component.
    """
    url = f"http://127.0.0.1:{ROUTER_PORT}/generate"
    prompt = "Hello " * (il // 2)

    payload = {
        "text": prompt,
        "sampling_params": {"max_new_tokens": GEN_TOKENS, "temperature": 0.0},
    }

    warmup_payload = {
        "text": prompt,
        "sampling_params": {"max_new_tokens": 16, "temperature": 0.0},
    }
    for _ in range(n_warmup):
        try:
            req.post(url, json=warmup_payload, timeout=60)
        except Exception:
            pass
    time.sleep(0.3)

    e_da_start = dvfs.da_energy_mj()
    e_df_start = dvfs.df_energy_mj()
    t_start = time.perf_counter()

    per_req_timeout = min(MAX_CONFIG_TIME_S, max(90, bs * 3))

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(bs, 64)) as pool:
        futures = [pool.submit(req.post, url, json=payload, timeout=per_req_timeout)
                   for _ in range(bs)]
        results = []
        try:
            for f in concurrent.futures.as_completed(futures, timeout=MAX_CONFIG_TIME_S):
                try:
                    resp = f.result()
                    results.append(resp.json())
                except Exception:
                    pass
        except concurrent.futures.TimeoutError:
            log.warning("    Watchdog: config timed out after %ds", MAX_CONFIG_TIME_S)
            for f in futures:
                f.cancel()

    t_end = time.perf_counter()
    e_da_end = dvfs.da_energy_mj()
    e_df_end = dvfs.df_energy_mj()

    if not results:
        return None

    total_time_s = t_end - t_start
    n_iters = GEN_TOKENS
    iter_lat_us = (total_time_s / n_iters) * 1e6
    da_energy_mj = (e_da_end - e_da_start) / n_iters
    df_energy_mj = (e_df_end - e_df_start) / n_iters

    return (iter_lat_us, da_energy_mj, df_energy_mj)


def check_server_alive() -> bool:
    try:
        r = req.get(f"http://127.0.0.1:{ROUTER_PORT}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def should_skip_config(il, bs, f_a, f_f):
    """Skip configs that will certainly timeout or OOM for TP=2."""
    if il >= 4096 and bs >= 256:
        return True
    if il >= 4096 and bs >= 128 and (f_a < 690 or f_f < 690):
        return True
    # Large bs + low freq combos timeout even at smaller IL
    if il >= 1024 and bs >= 256 and (f_a < 690 or f_f < 690):
        return True
    if il >= 2048 and bs >= 256:
        return True
    return False


def run_sweep(args):
    dvfs = DVFSHelper()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)

    completed = set()
    if out_path.exists() and os.path.getsize(out_path) > 0:
        with open(out_path) as f:
            for line in f:
                if line.startswith("tp\t"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) >= 6:
                    completed.add(tuple(parts[:6]))
        log.info("Resuming: %d data points already done.", len(completed))

    write_header = not out_path.exists() or os.path.getsize(out_path) == 0
    f_out = open(out_path, "a")
    if write_header:
        f_out.write("tp\tM\tf_A\tf_F\tinput_len\tbatch_size\t"
                    "iter_lat_us\tDA_energy_mj\tDF_energy_mj\n")
        f_out.flush()

    by_m = defaultdict(list)
    for M in args.m_values:
        for il in args.input_lens:
            for bs in args.batch_sizes:
                for f_a in args.freqs:
                    for f_f in args.freqs:
                        if should_skip_config(il, bs, f_a, f_f):
                            continue
                        key = (str(TP), str(M), str(f_a), str(f_f), str(il), str(bs))
                        if key not in completed:
                            by_m[M].append((f_a, f_f, il, bs))

    total_configs = sum(len(v) for v in by_m.values())
    log.info("Total configs remaining: %d", total_configs)

    done_configs = 0
    t_sweep_start = time.time()
    MAX_CONSEC_FAILURES = 3

    for M in sorted(by_m.keys()):
        m_configs = by_m[M]
        if not m_configs:
            continue

        log.info("=" * 60)
        log.info("Phase M=%d: %d configs (TP=2)", M, len(m_configs))
        log.info("=" * 60)

        kill_all_servers()
        dvfs.lock_prefill_max()

        procs = start_pdaf_stack(M, log_dir / f"M{M}")
        if procs is None:
            log.error("Failed to start stack for M=%d, skipping", M)
            dvfs.unlock_all()
            continue

        # Warmup at max freq
        dvfs.lock_decode(1410, 1410)
        time.sleep(0.2)
        measure_one_config(dvfs, bs=4, il=128, n_warmup=3)
        log.info("Warmup done for M=%d", M)

        consec_failures = 0

        for cfg_idx, (f_a, f_f, il, bs) in enumerate(m_configs):
            if consec_failures >= MAX_CONSEC_FAILURES:
                log.warning("  %d consecutive failures, restarting stack...",
                            consec_failures)
                dvfs.unlock_all()
                _cleanup(procs)
                kill_all_servers()
                time.sleep(5)
                dvfs.lock_prefill_max()
                procs = start_pdaf_stack(M, log_dir / f"M{M}_restart")
                if procs is None:
                    log.error("  Restart failed, skipping rest of M=%d", M)
                    break
                dvfs.lock_decode(1410, 1410)
                measure_one_config(dvfs, bs=4, il=128, n_warmup=3)
                log.info("  Stack restarted successfully")
                consec_failures = 0

            dvfs.lock_decode(f_a, f_f)
            time.sleep(0.05)

            try:
                result = measure_one_config(dvfs, bs, il, n_warmup=0)

                if result is None:
                    log.warning("  FAILED: M=%d f_A=%d f_F=%d il=%d bs=%d",
                                M, f_a, f_f, il, bs)
                    consec_failures += 1
                    if not check_server_alive():
                        log.warning("  Server appears dead!")
                        consec_failures = MAX_CONSEC_FAILURES
                    continue

                consec_failures = 0
                iter_lat, da_e, df_e = result
                f_out.write(f"{TP}\t{M}\t{f_a}\t{f_f}\t{il}\t{bs}\t"
                            f"{iter_lat:.2f}\t{da_e:.2f}\t{df_e:.2f}\n")
                f_out.flush()
                done_configs += 1

                elapsed = time.time() - t_sweep_start
                rate = done_configs / max(elapsed, 1)
                eta_min = (total_configs - done_configs) / max(rate, 0.001) / 60

                log.info(
                    "  [%d/%d] M=%d f_A=%d f_F=%d il=%d bs=%d "
                    "→ iter=%.0fus DA=%.1fmJ DF=%.1fmJ [ETA %.0fmin]",
                    done_configs, total_configs, M, f_a, f_f, il, bs,
                    iter_lat, da_e, df_e, eta_min)

            except Exception as e:
                log.warning("  ERROR: M=%d f_A=%d f_F=%d il=%d bs=%d: %s",
                            M, f_a, f_f, il, bs, e)
                consec_failures += 1

        dvfs.unlock_all()
        _cleanup(procs)
        kill_all_servers()

    f_out.close()
    dvfs.unlock_all()
    log.info("=" * 60)
    log.info("Sweep complete! Results: %s", out_path)
    log.info("Total measured: %d configs in %.1f min",
             done_configs, (time.time() - t_sweep_start) / 60)


def main():
    parser = argparse.ArgumentParser(
        description="Decode pipeline coupled profiling (TP=2)")
    parser.add_argument("--output", default=str(
        SCRIPT_DIR / "paper" / "decode_pipeline_tp2.txt"))
    parser.add_argument("--log-dir", default=str(
        SCRIPT_DIR / "logs" / "decode_pipeline_tp2"))
    parser.add_argument("--freqs", type=int, nargs="+", default=DEFAULT_FREQS)
    parser.add_argument("--input-lens", type=int, nargs="+", default=DEFAULT_INPUT_LENS)
    parser.add_argument("--batch-sizes", type=int, nargs="+",
                        default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--m-values", type=int, nargs="+", default=DEFAULT_M_VALUES)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.freqs = [690, 1170, 1410]
        args.input_lens = [512, 2048]
        args.batch_sizes = [4, 32, 128]
        args.m_values = [1, 2]

    n_configs = (len(args.freqs) ** 2 * len(args.input_lens) *
                 len(args.batch_sizes) * len(args.m_values))

    log.info("Decode Pipeline Coupled Profiling — TP=2")
    log.info("  Freqs: %s (%d^2 = %d combos)", args.freqs,
             len(args.freqs), len(args.freqs) ** 2)
    log.info("  IL: %s, GEN_TOKENS: %d (fixed)", args.input_lens, GEN_TOKENS)
    log.info("  BS: %s, M: %s", args.batch_sizes, args.m_values)
    log.info("  Total configs (before skip): %d", n_configs)
    log.info("  Est. time: %.1f hours (@ ~20s/config)", n_configs * 20 / 3600)

    run_sweep(args)


if __name__ == "__main__":
    main()
