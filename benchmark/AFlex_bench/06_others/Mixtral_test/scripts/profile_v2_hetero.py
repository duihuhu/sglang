"""Collect V2 pipeline decode data for heterogeneous DA(TP1)+DF(TP2).

Deploy PDAF: PA(TP1)+PF(TP2)+DA(TP1)+DF(TP2) on 6 GPUs.
Then sweep f_A (on DA gpu) x f_F (on DF gpus) to collect V2 pipeline data.
Uses same deploy logic as run_micro_bench.py start_pdaf.
"""
from __future__ import annotations

import concurrent.futures
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("v2_hetero")

PYTHON = "/workspace/env/sglang-test/bin/python"
MODEL = "/models/Mixtral/Mixtral-8x7B/"
DATA_DIR = Path("/workspace/sglang/benchmark/AFlex_bench/06_others/Mixtral_test/energy_model/data")

# 6 GPU PDAF: P on GPU 0,1,2 (PF TP2 on 0,1 + PA TP1 on 2)
#              D on GPU 3,4,5 (DF TP2 on 3,4 + DA TP1 on 5)
P_CVD = "0,1,2"
D_CVD = "3,4,5"
TP_FFN = 2
TP_ATTN = 1

# Physical GPU IDs for freq control
DA_PHYS_GPU = 5    # last GPU in D_CVD
DF_PHYS_GPUS = [3, 4]  # first tp_ffn GPUs in D_CVD
PA_PHYS_GPU = 2
PF_PHYS_GPUS = [0, 1]
ALL_PHYS_GPUS = [0, 1, 2, 3, 4, 5]

PA_PORT = 42010
PF_PORT = 42011
DA_PORT = 42020
DF_PORT = 42021
ROUTER_PORT = 42000

GPU_CLOCKS = [210, 450, 690, 930, 1170, 1410]
V2_INPUT_LENS = [128, 512, 1024, 2048, 4096]
V2_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
M_VALUES = [1, 2]

procs = []

def lock_freq(gpus, freq):
    for g in gpus:
        subprocess.run(["nvidia-smi", "-i", str(g), "-lgc", str(freq)], capture_output=True)

def unlock_freq(gpus):
    for g in gpus:
        subprocess.run(["nvidia-smi", "-i", str(g), "-rgc"], capture_output=True)

def kill_port(port):
    r = subprocess.run(["ss", "-tlnp", f"sport = :{port}"], capture_output=True, text=True)
    for m in re.finditer(r"pid=(\d+)", r.stdout):
        try:
            os.kill(int(m.group(1)), 9)
        except OSError:
            pass

def kill_all():
    for port in [PA_PORT, PF_PORT, DA_PORT, DF_PORT, ROUTER_PORT]:
        kill_port(port)
    for p in procs:
        try:
            p.kill()
        except:
            pass
    procs.clear()
    time.sleep(3)

def wait_health(port, timeout=600, check_model_info=False):
    for _ in range(timeout // 2):
        try:
            if check_model_info:
                r = requests.get(f"http://127.0.0.1:{port}/get_model_info", timeout=3)
            else:
                r = requests.get(f"http://127.0.0.1:{port}/health", timeout=3)
            if r.status_code == 200:
                return True
        except:
            pass
        time.sleep(2)
    return False

def start_pdaf():
    """Deploy PDAF with heterogeneous TP: PA(TP1)+PF(TP2) | DA(TP1)+DF(TP2)."""
    kill_all()
    
    env_base = os.environ.copy()
    env_base["AFD_UCX_TLS"] = "rc,tcp,cuda_copy,cuda_ipc"
    env_base["SGLANG_DISAGGREGATION_THREAD_POOL_SIZE"] = "128"
    env_base["AFD_ASYNC_PIPELINE"] = "1"
    
    common = ["--model-path", MODEL,
              "--host", "127.0.0.1",
              "--afd-comm-backend", "ipc_cpp",
              "--afd-micro-batch", "2",
              "--afd-dynamic-micro-batch",
              "--mem-fraction-static", "0.85",
              "--max-running-requests", "512",
              "--skip-server-warmup",
              "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
              "--afd-disagg-interleave-poll",
              "--disable-radix-cache",
              "--num-reserved-decode-tokens", "512",
              "--disaggregation-transfer-backend", "mooncake",
              "--disaggregation-bootstrap-port", "49999",
              "--disaggregation-ib-device", "mlx5_4",
              "--enable-metrics"]

    ucx_p, ucx_d = 28200, 28300
    sched_p, sched_d = 68400, 68500

    def _env(cvd, ucx_base, sched_port, peer_device, nvml_idx, ffn_host=None):
        e = env_base.copy()
        e["CUDA_VISIBLE_DEVICES"] = cvd
        e["AFD_UCX_BASE_PORT"] = str(ucx_base)
        e["AFD_SCHED_PORT"] = str(sched_port)
        e["AFD_IPC_SYNC_MODE"] = "ipc_event"
        e["AFD_IPC_PEER_DEVICE"] = str(peer_device)
        e["AFD_NVML_DEVICE_INDICES"] = str(nvml_idx)
        e["AFD_NVML_DEVICE_INDEX"] = str(nvml_idx).split(",")[0]
        e["SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT"] = "600"
        e["SGLANG_DISAGGREGATION_WAITING_TIMEOUT"] = "600"
        if ffn_host:
            e["AFD_UCX_FFN_HOST"] = ffn_host
        return e

    def _cmd(port, tp, perspective, disagg, base_gpu_id):
        return [PYTHON, "-m", "sglang.launch_server",
                "--port", str(port), "--tp", str(tp),
                "--afd-perspective", perspective,
                "--disaggregation-mode", disagg,
                "--base-gpu-id", str(base_gpu_id)] + common

    # GPU layout in CVD: [FFN GPUs (0..tp_ffn-1) | Attn GPUs (tp_ffn..)]
    p_ffn_nvml = ",".join(str(g) for g in PF_PHYS_GPUS)
    p_attn_nvml = str(PA_PHYS_GPU)
    d_ffn_nvml = ",".join(str(g) for g in DF_PHYS_GPUS)
    d_attn_nvml = str(DA_PHYS_GPU)

    # PF (FFN TP2, base=0)
    log.info("Starting PF (TP2, phys GPU %s)...", PF_PHYS_GPUS)
    p = subprocess.Popen(_cmd(PF_PORT, TP_FFN, "ffn", "prefill", 0),
                         env=_env(P_CVD, ucx_p, sched_p, peer_device=TP_FFN,
                                  nvml_idx=p_ffn_nvml),
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    procs.append(p)
    time.sleep(8)

    # PA (Attn TP1, base=tp_ffn)
    log.info("Starting PA (TP1, phys GPU %d)...", PA_PHYS_GPU)
    p = subprocess.Popen(_cmd(PA_PORT, TP_ATTN, "attn", "prefill", TP_FFN),
                         env=_env(P_CVD, ucx_p, sched_p, peer_device=0,
                                  nvml_idx=p_attn_nvml, ffn_host="127.0.0.1"),
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    procs.append(p)
    time.sleep(8)

    # DF (FFN TP2, base=0)
    log.info("Starting DF (TP2, phys GPU %s)...", DF_PHYS_GPUS)
    p = subprocess.Popen(_cmd(DF_PORT, TP_FFN, "ffn", "decode", 0),
                         env=_env(D_CVD, ucx_d, sched_d, peer_device=TP_FFN,
                                  nvml_idx=d_ffn_nvml),
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    procs.append(p)
    time.sleep(10)

    # DA (Attn TP1, base=tp_ffn)
    log.info("Starting DA (TP1, phys GPU %d)...", DA_PHYS_GPU)
    p = subprocess.Popen(_cmd(DA_PORT, TP_ATTN, "attn", "decode", TP_FFN),
                         env=_env(D_CVD, ucx_d, sched_d, peer_device=0,
                                  nvml_idx=d_attn_nvml, ffn_host="127.0.0.1"),
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    procs.append(p)

    # Wait for all 4 components
    checks = [(PA_PORT, "PA", False), (PF_PORT, "PF", True),
              (DA_PORT, "DA", False), (DF_PORT, "DF", True)]
    for port, name, use_model in checks:
        if not wait_health(port, 600, check_model_info=use_model):
            log.error("%s failed (port %d)!", name, port)
            return False
        log.info("  %s ready", name)

    # Router
    cmd_r = [PYTHON, "-m", "sglang_router.launch_router",
             "--pd-disaggregation", "--mini-lb",
             "--prefill", f"http://127.0.0.1:{PA_PORT}",
             "--decode", f"http://127.0.0.1:{DA_PORT}",
             "--host", "127.0.0.1", "--port", str(ROUTER_PORT)]
    p = subprocess.Popen(cmd_r, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    procs.append(p)
    if not wait_health(ROUTER_PORT, 60):
        log.error("Router failed!")
        return False

    log.info("PDAF DA(TP1)+DF(TP2) ready!")
    return True


def generate_sync(prompt, max_tokens=64):
    return requests.post(f"http://127.0.0.1:{ROUTER_PORT}/generate",
                         json={"text": prompt,
                               "sampling_params": {"max_new_tokens": max_tokens, "temperature": 0}},
                         timeout=600).json()


def main():
    import resource
    resource.setrlimit(resource.RLIMIT_MEMLOCK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not start_pdaf():
        log.error("Failed to deploy, aborting")
        return

    # Warmup
    generate_sync("Hello world", 10)
    log.info("Warmup done")

    v2_file = DATA_DIR / "decode_pipeline_da1_df2.txt"
    with open(v2_file, "w") as f:
        f.write("tp_a\ttp_f\tM\tf_A\tf_F\tinput_len\tbatch_size\titer_lat_us\tDA_energy_mj\tDF_energy_mj\n")

        total_combos = len(M_VALUES) * len(GPU_CLOCKS) * len(GPU_CLOCKS)
        done = 0
        for M in M_VALUES:
            for f_a in GPU_CLOCKS:
                for f_f in GPU_CLOCKS:
                    done += 1
                    lock_freq([DA_PHYS_GPU], f_a)
                    lock_freq(DF_PHYS_GPUS, f_f)
                    # Prefill side at max freq (not profiling P)
                    lock_freq([PA_PHYS_GPU] + PF_PHYS_GPUS, 1410)
                    time.sleep(0.3)

                    for il in V2_INPUT_LENS:
                        for bs in V2_BATCH_SIZES:
                            if il * bs > 512 * 256:
                                continue
                            actual_bs = bs * M
                            try:
                                prompt = "x " * il
                                ol = 64

                                t_start = time.time()
                                with concurrent.futures.ThreadPoolExecutor(max_workers=min(actual_bs, 64)) as pool:
                                    futures = [pool.submit(generate_sync, prompt, ol) for _ in range(actual_bs)]
                                    [fut.result() for fut in futures]
                                t_end = time.time()

                                iter_lat_us = (t_end - t_start) * 1e6 / ol

                                # Energy: read power during a short burst
                                # Simpler: use (power * time) approach
                                da_power = float(subprocess.run(
                                    ["nvidia-smi", "-i", str(DA_PHYS_GPU), "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True).stdout.strip())
                                df_power = sum(float(subprocess.run(
                                    ["nvidia-smi", "-i", str(g), "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True).stdout.strip()) for g in DF_PHYS_GPUS)

                                duration_s = (t_end - t_start) / ol
                                da_energy_mj = da_power * duration_s * 1000
                                df_energy_mj = df_power * duration_s * 1000

                                f.write(f"1\t2\t{M}\t{f_a}\t{f_f}\t{il}\t{bs}\t{iter_lat_us:.2f}\t{da_energy_mj:.2f}\t{df_energy_mj:.2f}\n")
                                f.flush()

                                if bs == 1 and il == 128:
                                    log.info("[%d/%d] M=%d fA=%d fF=%d il=%d bs=%d -> %.0fus",
                                             done, total_combos, M, f_a, f_f, il, bs, iter_lat_us)
                            except Exception as e:
                                log.error("  M=%d fA=%d fF=%d il=%d bs=%d FAILED: %s",
                                          M, f_a, f_f, il, bs, e)

    unlock_freq(ALL_PHYS_GPUS)
    kill_all()

    lines = sum(1 for _ in open(v2_file)) - 1
    log.info("Done! %d data points -> %s", lines, v2_file)


if __name__ == "__main__":
    main()
