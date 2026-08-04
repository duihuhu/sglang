#!/usr/bin/env python3
"""Benchmark Mixtral-8x7B (MoE): AFlex + MegaScale across QPS=2/4/6/8/12/16.

Uses Tier1 solver-derived configs for the MoE model on node1+node2.
Mixtral requires minimum TP=2 (model ~93GB doesn't fit single GPU).

For each QPS:
  - AFlex: solver frequency assignment (DVFS enabled)
  - MegaScale: same topology, all GPUs locked at 1410MHz (no DVFS)

Each QPS restarts the service to ensure clean state.
"""
from __future__ import annotations

import argparse, asyncio, json, logging, os, shlex, subprocess, sys, time
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("bench_moe")

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NODE1_IP = "10.252.129.36"
NODE2_IP = "10.252.129.35"
CONTAINER = "operator_test"
PYTHON = "/usr/bin/python3"
MODEL = "/models/Mixtral-8x7B/"
LOG_C = "/workspace/sglang/benchmark/AFlex_bench/multi_node/logs"
CLEANUP = "/workspace/sglang/benchmark/AFlex_bench/multi_node/scripts/cleanup_node.sh"
IB_JSON_FILE = "/tmp/ib_scal_map.json"
ENERGY_MODEL_DIR_V1 = ("/workspace/sglang/benchmark/AFlex_bench/"
                       "energy_model/Mixtral-8x7B/models_v1")
WORKLOAD_DIR = HERE.parent / "workloads"

TTFT_SLO_MS = 5000.0
TPOT_SLO_MS = 300.0
MAX_FREQ = 1410
GPUS_PER_NODE = 8
NUM_LAYERS = 32

GPU_NIC = {0: "mlx5_0", 1: "mlx5_0", 2: "mlx5_1", 3: "mlx5_1",
           4: "mlx5_4", 5: "mlx5_4", 6: "mlx5_5", 7: "mlx5_5"}

ROUTER_PORT = 44000
SUB_ROUTER_BASE = 45000
DECODE_PORT_BASE = 43020
PREFILL_PORT_BASE = 43200
NCCL_PORT_BASE = 37300
MAX_RUN_S = 400


@dataclass
class MoEConfig:
    name: str
    qps: int
    k_p: int; k_d: int
    tp_pa: int; tp_pf: int
    tp_da: int; tp_df: int
    f_pa: int; f_pf: int
    f_da: int; f_df: int
    tier: bool = True


# Solver-derived configs for conv dataset (Mixtral, TP>=2, max_pair=4)
# Note: 210MHz causes CUDA assert (NaN in multinomial) for Mixtral MoE.
# Minimum safe frequency is 450MHz.
# Layout: 1P(TP2 interleaved) on node1 + 1D(TP2 interleaved) on node2 = 8 GPU
AFLEX_CONV = [
    MoEConfig("aflex_conv_q2", 2, k_p=1, k_d=1, tp_pa=2, tp_pf=2, tp_da=2, tp_df=2,
              f_pa=450, f_pf=1410, f_da=450, f_df=450),
    MoEConfig("aflex_conv_q4", 4, k_p=1, k_d=1, tp_pa=2, tp_pf=2, tp_da=2, tp_df=2,
              f_pa=450, f_pf=1410, f_da=450, f_df=450),
    MoEConfig("aflex_conv_q6", 6, k_p=1, k_d=1, tp_pa=2, tp_pf=2, tp_da=2, tp_df=2,
              f_pa=690, f_pf=1410, f_da=450, f_df=450),
    MoEConfig("aflex_conv_q8", 8, k_p=1, k_d=1, tp_pa=2, tp_pf=2, tp_da=2, tp_df=2,
              f_pa=690, f_pf=1410, f_da=690, f_df=690),
    MoEConfig("aflex_conv_q12", 12, k_p=1, k_d=1, tp_pa=2, tp_pf=2, tp_da=2, tp_df=2,
              f_pa=930, f_pf=1410, f_da=690, f_df=690),
    MoEConfig("aflex_conv_q16", 16, k_p=1, k_d=1, tp_pa=2, tp_pf=2, tp_da=2, tp_df=2,
              f_pa=1410, f_pf=1410, f_da=930, f_df=930),
]

# Solver-derived configs for code dataset
AFLEX_CODE = [
    MoEConfig("aflex_code_q2", 2, k_p=1, k_d=1, tp_pa=2, tp_pf=2, tp_da=2, tp_df=2,
              f_pa=930, f_pf=450, f_da=450, f_df=450),
    MoEConfig("aflex_code_q4", 4, k_p=1, k_d=1, tp_pa=2, tp_pf=2, tp_da=2, tp_df=2,
              f_pa=930, f_pf=450, f_da=450, f_df=450),
    MoEConfig("aflex_code_q6", 6, k_p=1, k_d=1, tp_pa=2, tp_pf=2, tp_da=2, tp_df=2,
              f_pa=930, f_pf=690, f_da=450, f_df=450),
    MoEConfig("aflex_code_q8", 8, k_p=1, k_d=1, tp_pa=2, tp_pf=2, tp_da=2, tp_df=2,
              f_pa=930, f_pf=930, f_da=450, f_df=450),
    MoEConfig("aflex_code_q12", 12, k_p=1, k_d=1, tp_pa=2, tp_pf=2, tp_da=2, tp_df=2,
              f_pa=1170, f_pf=1170, f_da=690, f_df=690),
    MoEConfig("aflex_code_q16", 16, k_p=1, k_d=1, tp_pa=2, tp_pf=2, tp_da=2, tp_df=2,
              f_pa=1410, f_pf=1410, f_da=930, f_df=930),
]


def _make_mega(aflex_cfgs: list[MoEConfig]) -> list[MoEConfig]:
    """Create MegaScale configs: same topology, all freq locked to MAX."""
    out = []
    for c in aflex_cfgs:
        name = c.name.replace("aflex_", "mega_")
        out.append(MoEConfig(
            name, c.qps, c.k_p, c.k_d, c.tp_pa, c.tp_pf, c.tp_da, c.tp_df,
            MAX_FREQ, MAX_FREQ, MAX_FREQ, MAX_FREQ, tier=False))
    return out


MEGA_CONV = _make_mega(AFLEX_CONV)
MEGA_CODE = _make_mega(AFLEX_CODE)


# ============================================================
# Infrastructure helpers
# ============================================================

def _ssh(host, cmd):
    return ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", host, cmd]


def dexec_local(shell_cmd):
    subprocess.run(["docker", "exec", CONTAINER, "bash", "-lc", shell_cmd], check=False)


def dexec_remote(shell_cmd):
    inner = f"docker exec {CONTAINER} bash -lc {shlex.quote(shell_cmd)}"
    subprocess.run(_ssh(NODE2_IP, inner), check=False)


def cleanup_all():
    dexec_local(f"bash {CLEANUP}")
    subprocess.run(_ssh(NODE2_IP,
                        f"docker exec {CONTAINER} bash -lc 'bash {CLEANUP}'"),
                   check=False)
    time.sleep(5)


def wait_health(host, port, timeout=600, check_model_info=False):
    import requests
    deadline = time.time() + timeout
    ep = "get_model_info" if check_model_info else "health"
    while time.time() < deadline:
        try:
            r = requests.get(f"http://{host}:{port}/{ep}", timeout=10)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def get_energy_local(gpus):
    try:
        import pynvml
        pynvml.nvmlInit()
        res = {i: pynvml.nvmlDeviceGetTotalEnergyConsumption(
            pynvml.nvmlDeviceGetHandleByIndex(i)) for i in gpus}
        pynvml.nvmlShutdown()
        return res
    except Exception as e:
        log.warning("local NVML energy read failed: %s", e)
        return {i: 0 for i in gpus}


def get_energy_remote(gpus):
    idx_csv = ",".join(str(i) for i in gpus)
    pycode = (
        "import pynvml,json;pynvml.nvmlInit();"
        f"idxs=[int(x) for x in '{idx_csv}'.split(',')];"
        "print(json.dumps({i:pynvml.nvmlDeviceGetTotalEnergyConsumption("
        "pynvml.nvmlDeviceGetHandleByIndex(i)) for i in idxs}));"
        "pynvml.nvmlShutdown()"
    )
    inner = f"docker exec {CONTAINER} bash -lc {shlex.quote(f'{PYTHON} -c {shlex.quote(pycode)}')}"
    try:
        out = subprocess.run(_ssh(NODE2_IP, inner), capture_output=True,
                             text=True, timeout=30)
        line = [l for l in out.stdout.strip().splitlines() if l.startswith("{")]
        return {int(k): v for k, v in json.loads(line[-1]).items()} if line else \
               {i: 0 for i in gpus}
    except Exception as e:
        log.warning("remote NVML energy read failed: %s", e)
        return {i: 0 for i in gpus}


def lock_freq_on_host(host, gpus, freq):
    cmd = ";".join(f"nvidia-smi -i {i} --lock-gpu-clocks={freq},{freq}"
                   for i in gpus) + ";true"
    if host == NODE1_IP:
        dexec_local(cmd)
    else:
        dexec_remote(cmd)


def lock_freq_map_on_host(host, gpu_freq: dict):
    cmd = ";".join(f"nvidia-smi -i {g} --lock-gpu-clocks={f},{f}"
                   for g, f in gpu_freq.items()) + ";true"
    if host == NODE1_IP:
        dexec_local(cmd)
    else:
        dexec_remote(cmd)


def unlock_freq_both(gpus):
    cmd = ";".join(f"nvidia-smi -i {i} --reset-gpu-clocks" for i in gpus) + ";true"
    dexec_local(cmd)
    dexec_remote(cmd)


def write_ib_json(mapping):
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(mapping, f)
        tmp = f.name
    subprocess.run(["docker", "cp", tmp, f"{CONTAINER}:{IB_JSON_FILE}"], check=False)
    subprocess.run(["scp", "-o", "StrictHostKeyChecking=no", "-q",
                    tmp, f"{NODE2_IP}:{IB_JSON_FILE}"], check=False)
    subprocess.run(_ssh(NODE2_IP,
                        f"docker cp {IB_JSON_FILE} {CONTAINER}:{IB_JSON_FILE}"),
                   check=False)
    os.unlink(tmp)


def test_generate(url):
    import requests
    try:
        r = requests.post(f"{url}/generate",
                          json={"text": "Hello", "sampling_params": {"max_new_tokens": 8}},
                          timeout=120)
        return r.status_code == 200
    except Exception as e:
        log.error("test_generate failed: %s", e)
        return False


# ============================================================
# GPU Allocation (simplified for interleaved)
# ============================================================


# ============================================================
# Deploy AFD (interleaved layout, matching run_moe_macro.py)
# ============================================================

def _build_env_interleaved(role, host, attn_gpus, ffn_gpus, is_prefill):
    """Build env string for interleaved AFD layout (like run_moe_macro.py).
    
    All GPUs visible, peer offset +/-1 for neighbor pairing.
    SGLANG_HOST_IP is set explicitly to avoid Docker meta-interface (198.18.0.x)
    being picked up by get_local_ip_auto() on node1.
    """
    cvd = "0,1,2,3,4,5,6,7"
    ucx = "28200" if is_prefill else "28300"
    sched = "68400" if is_prefill else "68500"
    base = (
        f"export SGLANG_HOST_IP={host} "
        f"SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal "
        "AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc "
        "SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=128 AFD_ASYNC_PIPELINE=1 "
        "AFD_IPC_SYNC_MODE=ipc_event "
        "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 "
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600 "
        "SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600 "
        f"CUDA_VISIBLE_DEVICES={cvd} "
    )
    if role in ("PF", "DF"):
        nvml = ",".join(str(g) for g in ffn_gpus)
        return (base + f"AFD_UCX_BASE_PORT={ucx} AFD_SCHED_PORT={sched} "
                f"AFD_IPC_PEER_OFFSET=-1 "
                f"AFD_NVML_DEVICE_INDICES={nvml} AFD_NVML_DEVICE_INDEX={ffn_gpus[0]};")
    else:
        nvml = ",".join(str(g) for g in attn_gpus)
        return (base + f"AFD_UCX_BASE_PORT={ucx} AFD_SCHED_PORT={sched} "
                f"AFD_IPC_PEER_OFFSET=1 "
                f"AFD_NVML_DEVICE_INDICES={nvml} AFD_NVML_DEVICE_INDEX={attn_gpus[0]} "
                "AFD_UCX_FFN_HOST=127.0.0.1;")


def _build_afd_flags(tp, tier, gpu_step=2):
    flags = (
        f"--model-path {MODEL} --tp {tp} --gpu-id-step {gpu_step} "
        "--afd-comm-backend ipc_cpp --afd-micro-batch 2 "
        "--afd-dynamic-micro-batch "
        "--mem-fraction-static 0.85 --max-running-requests 512 "
        "--skip-server-warmup --watchdog-timeout 600 "
        "--disable-cuda-graph --disable-piecewise-cuda-graph "
        "--afd-disagg-interleave-poll --disable-radix-cache "
        "--num-reserved-decode-tokens 512 "
        "--disaggregation-transfer-backend mooncake "
        f"--disaggregation-ib-device {IB_JSON_FILE} --enable-metrics "
    )
    if tier:
        flags += (
            "--afd-dvfs-enabled "
            f"--afd-energy-model-dir {ENERGY_MODEL_DIR_V1} "
            f"--afd-ttft-slo-ms {int(TTFT_SLO_MS)} "
            f"--afd-tpot-slo-us {int(TPOT_SLO_MS * 1000)} "
            "--afd-dvfs-decode-compositional --afd-dvfs-idle-lock "
        )
    return flags


def _launch(host, cmd, logname):
    prefix = "setsid prlimit --memlock=unlimited:unlimited "
    full = cmd.replace(f"{PYTHON} -m", f"{prefix}{PYTHON} -m", 1)
    full = f"{full} > {LOG_C}/{logname}.log 2>&1 < /dev/null &"
    if host == NODE1_IP:
        dexec_local(full)
    else:
        dexec_remote(full)


def deploy(cfg: MoEConfig):
    """Deploy Mixtral PDAF using interleaved layout (tp=2, step=2).
    
    For TP=2: attn=[4,6], ffn=[5,7] per node (interleaved pairs).
    k_p prefill pairs on first k_p/2 nodes, k_d decode on remaining.
    """
    tp = cfg.tp_pa  # TP=2 for Mixtral (same for attn/ffn)
    gpu_step = 2
    attn_gpus = [4, 6]  # interleaved for TP=2
    ffn_gpus = [5, 7]
    attn_base = 4
    ffn_base = 5

    total_gpu = cfg.k_p * (cfg.tp_pa + cfg.tp_pf) + cfg.k_d * (cfg.tp_da + cfg.tp_df)
    log.info("=" * 60)
    log.info("DEPLOY %s: %dGPU (TP=%d interleaved, k_p=%d k_d=%d) tier=%s",
             cfg.name, total_gpu, tp, cfg.k_p, cfg.k_d, cfg.tier)

    # Frequency locking - different for prefill (node1) and decode (node2)
    if cfg.tier:
        # Node1 = Prefill: attn GPUs at f_pa, ffn GPUs at f_pf
        fmap_p = {g: cfg.f_pa for g in attn_gpus}
        fmap_p.update({g: cfg.f_pf for g in ffn_gpus})
        lock_freq_map_on_host(NODE1_IP, fmap_p)
        # Node2 = Decode: attn GPUs at f_da, ffn GPUs at f_df
        fmap_d = {g: cfg.f_da for g in attn_gpus}
        fmap_d.update({g: cfg.f_df for g in ffn_gpus})
        lock_freq_map_on_host(NODE2_IP, fmap_d)

    ib_map = {str(g): GPU_NIC[g] for g in range(GPUS_PER_NODE)}
    write_ib_json(ib_map)

    cf = _build_afd_flags(tp, cfg.tier, gpu_step)
    BS_PORT = 49999

    # Deploy based on k_p and k_d
    # Simple strategy: P instances on both nodes, D on node1
    hosts_for_p = []
    if cfg.k_p == 1:
        hosts_for_p = [NODE1_IP]
    elif cfg.k_p == 2:
        hosts_for_p = [NODE1_IP, NODE2_IP]
    elif cfg.k_p == 3:
        hosts_for_p = [NODE1_IP, NODE2_IP, NODE1_IP]
    else:
        hosts_for_p = [NODE1_IP, NODE2_IP] * (cfg.k_p // 2)
        if cfg.k_p % 2:
            hosts_for_p.append(NODE1_IP)

    hosts_for_d = []
    if cfg.k_d == 1:
        hosts_for_d = [NODE2_IP if cfg.k_p <= 1 else NODE1_IP]
    else:
        hosts_for_d = [NODE1_IP, NODE2_IP][:cfg.k_d]

    # For simplicity with interleaved layout: all P and D share same GPU layout
    # P on one node, D on other node
    # Rethink: with k_p=1, k_d=1, put P on node1, D on node2
    if cfg.k_p == 1 and cfg.k_d == 1:
        p_hosts = [NODE1_IP]
        d_hosts = [NODE2_IP]
    elif cfg.k_p == 2 and cfg.k_d == 1:
        p_hosts = [NODE1_IP, NODE2_IP]
        d_hosts = [NODE1_IP]  # share node1 with P0 (different GPUs via interleaved)
        # Actually for 12 GPU: 2P*4 + 1D*4 = 12 GPU across 2 nodes
        # Better: P0 on node1, P1 on node2, D on node2 (they use different GPU sets)
        # But interleaved means they use SAME GPUs (0-7)! So can't colocate.
        # With k_p=2: need 2 nodes for P, 1 node for D = 3 nodes!
        # Since we only have 2 nodes: must time-share or use different GPU sets
        # For Mixtral TP=2 interleaved: each instance uses GPUs 4,5,6,7
        # So we CAN'T run 2P + 1D on 2 nodes with interleaved layout!
        # Solution: use contiguous layout for multi-instance, or adjust GPU allocation
        pass
    elif cfg.k_p == 3 and cfg.k_d == 1:
        p_hosts = [NODE1_IP, NODE2_IP, NODE1_IP]
        d_hosts = [NODE2_IP]

    # Actually let me simplify: for k_p=1,k_d=1 use the proven interleaved layout
    # For k_p>1, we need a different approach

    log.info("  Launching PF on %s ...", NODE1_IP.split('.')[-1])
    env_pf = _build_env_interleaved("PF", NODE1_IP, attn_gpus, ffn_gpus, is_prefill=True)
    _launch(NODE1_IP,
            f"{env_pf} {PYTHON} -m sglang.launch_server --host {NODE1_IP} "
            f"--port 42011 --afd-perspective ffn --disaggregation-mode prefill "
            f"--base-gpu-id {ffn_base} --disaggregation-bootstrap-port {BS_PORT} {cf}",
            f"{cfg.name}_pf")
    time.sleep(6)

    log.info("  Launching PA on %s ...", NODE1_IP.split('.')[-1])
    env_pa = _build_env_interleaved("PA", NODE1_IP, attn_gpus, ffn_gpus, is_prefill=True)
    _launch(NODE1_IP,
            f"{env_pa} {PYTHON} -m sglang.launch_server --host {NODE1_IP} "
            f"--port 42010 --afd-perspective attn --disaggregation-mode prefill "
            f"--base-gpu-id {attn_base} --disaggregation-bootstrap-port {BS_PORT} "
            f"--nccl-port 34100 {cf}",
            f"{cfg.name}_pa")
    time.sleep(3)

    log.info("  Launching DF on %s ...", NODE2_IP.split('.')[-1])
    env_df = _build_env_interleaved("DF", NODE2_IP, attn_gpus, ffn_gpus, is_prefill=False)
    _launch(NODE2_IP,
            f"{env_df} {PYTHON} -m sglang.launch_server --host {NODE2_IP} "
            f"--port 42021 --afd-perspective ffn --disaggregation-mode decode "
            f"--base-gpu-id {ffn_base} --disaggregation-bootstrap-port {BS_PORT} "
            f"--nccl-port 34200 {cf}",
            f"{cfg.name}_df")
    time.sleep(6)

    log.info("  Launching DA on %s ...", NODE2_IP.split('.')[-1])
    env_da = _build_env_interleaved("DA", NODE2_IP, attn_gpus, ffn_gpus, is_prefill=False)
    _launch(NODE2_IP,
            f"{env_da} {PYTHON} -m sglang.launch_server --host {NODE2_IP} "
            f"--port 42020 --afd-perspective attn --disaggregation-mode decode "
            f"--base-gpu-id {attn_base} --disaggregation-bootstrap-port {BS_PORT} "
            f"--nccl-port 34300 {cf}",
            f"{cfg.name}_da")
    time.sleep(3)

    # Health checks (PF and DF check model_info, PA and DA just health)
    if not wait_health(NODE1_IP, 42011, 600, check_model_info=True):
        log.error("PF health failed"); return None
    log.info("  PF ready")
    if not wait_health(NODE1_IP, 42010, 600, check_model_info=False):
        log.error("PA health failed"); return None
    log.info("  PA ready")
    if not wait_health(NODE2_IP, 42021, 600, check_model_info=True):
        log.error("DF health failed"); return None
    log.info("  DF ready")
    if not wait_health(NODE2_IP, 42020, 600, check_model_info=False):
        log.error("DA health failed"); return None
    log.info("  DA ready")

    # Router (with bootstrap port for PD KV transfer)
    BS_PORT = 49999
    rc = (f"setsid {PYTHON} -m sglang_router.launch_router "
          f"--pd-disaggregation --mini-lb "
          f"--prefill http://{NODE1_IP}:42010 {BS_PORT} "
          f"--decode http://{NODE2_IP}:42020 "
          f"--host {NODE1_IP} --port {ROUTER_PORT} "
          f"> {LOG_C}/{cfg.name}_router.log 2>&1 < /dev/null &")
    dexec_local(rc)
    if not wait_health(NODE1_IP, ROUTER_PORT, 60):
        log.error("Router health failed"); return None

    log.info("  Deploy complete: http://%s:%d", NODE1_IP, ROUTER_PORT)
    return [f"http://{NODE1_IP}:{ROUTER_PORT}"]


# ============================================================
# Workload runner
# ============================================================

import aiohttp
import numpy as np


async def send_one(session, url, req, base_time, results):
    delay = req["arrival_time_s"] - (time.monotonic() - base_time)
    if delay > 0:
        await asyncio.sleep(delay)
    payload = {"text": "x" * req["input_len"],
               "sampling_params": {"max_new_tokens": req["output_len"],
               "temperature": 0, "ignore_eos": True},
               "stream": True}
    t0 = time.monotonic()
    ttft = None
    tokens = 0
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                results.append({"success": False, "error": f"HTTP {resp.status}"})
                return
            async for line in resp.content:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(b"data:"):
                    if ttft is None:
                        ttft = time.monotonic() - t0
                    tokens += 1
    except asyncio.CancelledError:
        return
    except Exception as e:
        results.append({"success": False, "error": str(e)})
        return
    total_time = time.monotonic() - t0
    tpot = ((total_time - ttft) / max(1, tokens - 1)) * 1000 if tokens > 1 else 0
    results.append({
        "success": True,
        "ttft_ms": (ttft or 0) * 1000,
        "ttft_proc_ms": (ttft or 0) * 1000,
        "tpot_ms": tpot,
        "completion_tokens": req["output_len"],
        "total_time_s": total_time,
    })


async def _run_workload_rr(reqs, n1_gpus, n2_gpus, max_run_s=400):
    e1s = get_energy_local(n1_gpus)
    e2s = get_energy_remote(n2_gpus)
    results = []
    timeout = aiohttp.ClientTimeout(total=max_run_s + 60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        base_time = time.monotonic()
        tasks = [asyncio.create_task(
            send_one(session, r["_target_url"], r, base_time, results))
            for r in reqs]
        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=max_run_s)
            except asyncio.TimeoutError:
                log.warning("Timed out after %ds", max_run_s)
    duration_s = time.monotonic() - base_time
    e1e = get_energy_local(n1_gpus)
    e2e = get_energy_remote(n2_gpus)
    energy_n1_j = sum((e1e.get(i, 0) - e1s.get(i, 0)) / 1000.0 for i in n1_gpus)
    energy_n2_j = sum((e2e.get(i, 0) - e2s.get(i, 0)) / 1000.0 for i in n2_gpus)
    total_energy_j = energy_n1_j + energy_n2_j
    ok = [r for r in results if r.get("success")]
    fail = [r for r in results if not r.get("success")]
    if not ok:
        return {"status": "FAIL", "failed": len(fail)}
    ttfts = [r["ttft_proc_ms"] for r in ok if r.get("ttft_proc_ms", 0) > 0]
    tpots = [r["tpot_ms"] for r in ok if r["tpot_ms"] > 0]
    total_tokens = sum(r.get("completion_tokens", 0) for r in ok)
    throughput = total_tokens / duration_s if duration_s > 0 else 0
    n_ttft_viol = sum(1 for v in ttfts if v > TTFT_SLO_MS)
    n_tpot_viol = sum(1 for r in ok if r["tpot_ms"] > TPOT_SLO_MS)
    slo_rate = ((n_ttft_viol + n_tpot_viol + len(fail))
                / len(results) * 100 if results else 0)
    return {
        "status": "PASS", "duration_s": round(duration_s, 1),
        "total_requests": len(reqs), "successful": len(ok), "failed": len(fail),
        "total_tokens": total_tokens,
        "throughput_tok_s": round(throughput, 1),
        "ttft_proc_avg_ms": round(float(np.mean(ttfts)), 1) if ttfts else 0,
        "ttft_proc_p50_ms": round(float(np.percentile(ttfts, 50)), 1) if ttfts else 0,
        "ttft_proc_p99_ms": round(float(np.percentile(ttfts, 99)), 1) if ttfts else 0,
        "tpot_avg_ms": round(float(np.mean(tpots)), 1) if tpots else 0,
        "tpot_p50_ms": round(float(np.percentile(tpots, 50)), 1) if tpots else 0,
        "tpot_p99_ms": round(float(np.percentile(tpots, 99)), 1) if tpots else 0,
        "energy_node1_j": round(energy_n1_j, 1),
        "energy_node2_j": round(energy_n2_j, 1),
        "total_energy_j": round(total_energy_j, 1),
        "energy_per_token_mj": (round(total_energy_j * 1000 / total_tokens, 2)
                                if total_tokens else 0),
        "slo_violation_rate": round(slo_rate, 1),
        "ttft_violations": n_ttft_viol, "tpot_violations": n_tpot_viol,
    }


def run_benchmark(urls, cfg: MoEConfig, dataset: str):
    wl_file = WORKLOAD_DIR / f"macro_{dataset}_qps{cfg.qps}.jsonl"
    if not wl_file.exists():
        return {"status": "NO_WORKLOAD"}
    with open(wl_file) as f:
        reqs = [json.loads(line) for line in f]
    last_arrival = max((r["arrival_time_s"] for r in reqs), default=0)
    run_s = int(min(max(MAX_RUN_S, last_arrival + 150), 900))
    log.info("Workload: %d reqs, run_window=%ds, %d sub-routers",
             len(reqs), run_s, len(urls))
    for i, r in enumerate(reqs):
        r["_target_url"] = urls[i % len(urls)] + "/generate"
    gpus = list(range(GPUS_PER_NODE))
    summary = asyncio.run(_run_workload_rr(reqs, gpus, gpus, run_s))
    if isinstance(summary, dict) and summary.get("status") == "PASS":
        summary["config"] = {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}
    return summary


# ============================================================
# Main loop
# ============================================================

def run_one(cfg: MoEConfig, dataset: str, all_results: dict):
    gpu_count = cfg.k_p * (cfg.tp_pa + cfg.tp_pf) + cfg.k_d * (cfg.tp_da + cfg.tp_df)
    is_mega = not cfg.tier
    label = "MEGASCALE" if is_mega else "AFLEX"
    log.info("\n" + "#" * 72)
    log.info("%s: %s (QPS=%d, %dGPU, dataset=%s)", label, cfg.name, cfg.qps, gpu_count, dataset)
    log.info("#" * 72)

    cleanup_all()
    time.sleep(10)

    url = deploy(cfg)
    if url is None:
        all_results[cfg.name] = {"status": "DEPLOY_FAILED"}
        return

    if is_mega:
        lock_freq_on_host(NODE1_IP, [4, 5, 6, 7], MAX_FREQ)
        lock_freq_on_host(NODE2_IP, [4, 5, 6, 7], MAX_FREQ)
        log.info("  Locked GPUs 4-7 to %dMHz on both nodes (MegaScale)", MAX_FREQ)

    if not test_generate(url[0]):
        log.error("WARMUP_FAILED for %s", cfg.name)
        all_results[cfg.name] = {"status": "WARMUP_FAILED"}
        cleanup_all()
        return

    time.sleep(3)
    summary = run_benchmark(url, cfg, dataset)
    all_results[cfg.name] = summary

    if isinstance(summary, dict) and summary.get("status") == "PASS":
        log.info("PASS: QPS=%d GPU=%d thpt=%.1f TTFT=%.1fms TPOT=%.1fms E/tok=%.1fmJ",
                 cfg.qps, gpu_count,
                 summary["throughput_tok_s"], summary["ttft_proc_avg_ms"],
                 summary["tpot_avg_ms"], summary.get("energy_per_token_mj", 0))
    else:
        log.error("FAIL: %s -> %s", cfg.name, summary)

    if is_mega:
        unlock_freq_both([4, 5, 6, 7])
    cleanup_all()
    time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["conv", "code", "both"], default="both")
    parser.add_argument("--mode", choices=["aflex", "mega", "both"], default="both")
    parser.add_argument("--qps-list", default="2,4,6,8,12,16")
    args = parser.parse_args()

    qps_targets = [int(x) for x in args.qps_list.split(",")]
    datasets = ["conv", "code"] if args.dataset == "both" else [args.dataset]

    for dataset in datasets:
        all_results = {}
        aflex_cfgs = AFLEX_CONV if dataset == "conv" else AFLEX_CODE
        mega_cfgs = MEGA_CONV if dataset == "conv" else MEGA_CODE

        if args.mode in ("aflex", "both"):
            log.info("=" * 72)
            log.info("  PHASE 1: AFlex (DVFS) - %s dataset", dataset)
            log.info("=" * 72)
            for cfg in aflex_cfgs:
                if cfg.qps in qps_targets:
                    run_one(cfg, dataset, all_results)

        if args.mode in ("mega", "both"):
            log.info("=" * 72)
            log.info("  PHASE 2: MegaScale (locked %dMHz) - %s dataset", MAX_FREQ, dataset)
            log.info("=" * 72)
            for cfg in mega_cfgs:
                if cfg.qps in qps_targets:
                    run_one(cfg, dataset, all_results)

        ts = time.strftime("%Y%m%d_%H%M%S")
        out = RESULTS_DIR / f"moe_{dataset}_{args.mode}_{ts}.json"
        meta = {
            "model": "Mixtral-8x7B",
            "benchmark": f"moe_{dataset}_aflex_mega",
            "dataset": dataset,
            "ttft_slo_ms": TTFT_SLO_MS,
            "tpot_slo_ms": TPOT_SLO_MS,
            "nodes": f"{NODE1_IP} + {NODE2_IP}",
        }
        out.write_text(json.dumps({"meta": meta, "results": all_results}, indent=2))
        log.info("Saved %s", out)

        print(f"\n{'='*72}")
        print(f"MoE {dataset.upper()} RESULTS")
        print(f"{'='*72}")
        for cfg in aflex_cfgs + mega_cfgs:
            if cfg.qps not in qps_targets:
                continue
            r = all_results.get(cfg.name, {})
            gpu_count = cfg.k_p * (cfg.tp_pa + cfg.tp_pf) + cfg.k_d * (cfg.tp_da + cfg.tp_df)
            if r.get("status") == "PASS":
                print(f"  {cfg.name:20s} QPS={cfg.qps:2d} GPU={gpu_count:2d}: "
                      f"thpt={r['throughput_tok_s']:.1f} TTFT={r['ttft_proc_avg_ms']:.1f}ms "
                      f"TPOT={r['tpot_avg_ms']:.1f}ms E/tok={r.get('energy_per_token_mj',0):.1f}mJ")
            else:
                print(f"  {cfg.name:20s} QPS={cfg.qps:2d} GPU={gpu_count:2d}: "
                      f"{r.get('status', 'UNKNOWN')}")


if __name__ == "__main__":
    main()

