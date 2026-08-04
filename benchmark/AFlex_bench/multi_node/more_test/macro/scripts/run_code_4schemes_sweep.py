"""
Run all 4 schemes for Code dataset, QPS = 2, 4, 8, 16. Restart per QPS.
  - SGLang: 16xTP1 (8/node), round-robin, locked 1410MHz
  - DynamoLLM: 16xTP1 (8/node), round-robin, DVFS unified
  - DistServe: 2P(TP4) + 4D(TP2) = 16GPU, locked 1410MHz
  - BiScale: 2P(TP4) + 4D(TP2) = 16GPU, DVFS biscale
"""
import sys, os, time, json, asyncio
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/mnt/workspace/lt/sglang/python')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import run_macro_benchmark as RMB

RMB.NODE1_IP = os.environ.get("MN_NODE1_IP", RMB.NODE1_IP)
RMB.NODE2_IP = os.environ.get("MN_NODE2_IP", RMB.NODE2_IP)

NODE1 = RMB.NODE1_IP
NODE2 = RMB.NODE2_IP
MODEL = RMB.MODEL
PYTHON = RMB.PYTHON
LOG_C = RMB.LOG_C
FLAGS = RMB.COMMON_BENCH_SERVER_FLAGS
GPU_NIC = RMB.GPU_NIC
ROUTER_PORT = RMB.ROUTER_PORT

ENERGY_MODEL_DIR_V1 = ("/workspace/sglang/benchmark/AFlex_bench/energy_model/"
                       "Qwen3-32B/models_v1")
BISCALE_TTFT_SLO_MS = 2000
BISCALE_TPOT_SLO_MS = 100


def _native_dvfs_flags() -> str:
    """DynamoLLM unified DVFS; honors RMB.TTFT_SLO_MS set by run_benchmark."""
    ttft = getattr(RMB, "TTFT_SLO_MS", 5000)
    tpot = getattr(RMB, "TPOT_SLO_MS", 300)
    return (
        f" --dvfs-enabled "
        f"--dvfs-energy-model-dir {ENERGY_MODEL_DIR_V1} "
        f"--dvfs-ttft-slo-ms {int(ttft)} "
        f"--dvfs-tpot-slo-us {int(tpot * 1000)}"
    )


def _biscale_dvfs_flags() -> str:
    ttft = getattr(RMB, "TTFT_SLO_MS", BISCALE_TTFT_SLO_MS)
    tpot = getattr(RMB, "TPOT_SLO_MS", BISCALE_TPOT_SLO_MS)
    return (
        f" --dvfs-enabled "
        f"--dvfs-energy-model-dir {ENERGY_MODEL_DIR_V1} "
        f"--dvfs-ttft-slo-ms {int(ttft)} "
        f"--dvfs-tpot-slo-us {int(tpot * 1000)} "
        f"--dvfs-policy biscale"
    )


def _lock_dvfs_baseline():
    """1410 MHz baseline for DVFS (avoid hardware-default ~1140 MHz)."""
    RMB.lock_freq_both(ALL_GPUS, 1410)

WORKLOAD_DIR = Path("/mnt/workspace/lt/sglang/benchmark/AFlex_bench/multi_node/more_test/macro/data/workloads")
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DVFS_LOG_DIR = "/workspace/sglang/benchmark/AFlex_bench/multi_node/logs"

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("code_4scheme_sweep")

ALL_GPUS = list(range(8))
P_GROUPS = [[0, 1, 2, 3], [4, 5, 6, 7]]
D_GROUPS = [[0, 1], [2, 3], [4, 5], [6, 7]]


def deploy_native_tp1(scheme: str):
    """Deploy 16xTP1, round-robin router. scheme='sglang' or 'dynamollm'."""
    is_dynamo = (scheme == "dynamollm")
    log.info("=" * 60)
    log.info("Deploying %s: 16xTP1 (8/node), round-robin", scheme.upper())
    log.info("=" * 60)

    RMB.cleanup_all()
    time.sleep(8)

    if is_dynamo:
        pass  # DVFS baseline applied after servers are healthy.
    else:
        RMB.lock_freq_both(ALL_GPUS, 1410)
        log.info("  Locked all GPUs to 1410MHz")

    insts = []
    idx = 0
    for host in (NODE1, NODE2):
        for gpu in ALL_GPUS:
            port = 53200 + idx * 10
            insts.append({"host": host, "gpu": gpu, "port": port, "idx": idx})
            idx += 1

    for inst in insts:
        gpu = inst["gpu"]
        host = inst["host"]
        port = inst["port"]
        i = inst["idx"]
        csv = str(gpu)
        node_label = "n1" if host == NODE1 else "n2"
        dvfs_flags = _native_dvfs_flags() if is_dynamo else ""
        env = (f"export SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal "
               f"CUDA_VISIBLE_DEVICES={csv} AFD_NVML_DEVICE_INDEX={gpu} "
               f"AFD_NVML_DEVICE_INDICES={csv}; ")
        cmd = (f"{env}setsid prlimit --memlock=unlimited:unlimited {PYTHON} -m sglang.launch_server "
               f"--model-path {MODEL} --tp 1 --host {host} --port {port} "
               f"--nccl-port {33300 + i * 10} {FLAGS}{dvfs_flags} "
               f"> {LOG_C}/{scheme}_{node_label}_g{gpu}.log 2>&1 < /dev/null &")
        if host == NODE1:
            RMB.dexec_local(cmd)
        else:
            RMB.dexec_remote(cmd)
        if i % 4 == 3:
            time.sleep(2)

    log.info("  Waiting for all 16 instances...")
    time.sleep(10)
    for inst in insts:
        if not RMB.wait_health(inst["host"], inst["port"], 400):
            log.error("  Instance %s:%d FAILED!", inst["host"], inst["port"])
            return None
    log.info("  All 16 instances healthy!")

    worker_urls = " ".join(f"http://{inst['host']}:{inst['port']}" for inst in insts)
    rc = (f"setsid {PYTHON} -m sglang_router.launch_router "
          f"--host {NODE1} --port {ROUTER_PORT} --policy round_robin "
          f"--worker-urls {worker_urls} > {LOG_C}/router.log 2>&1 < /dev/null &")
    RMB.dexec_local(rc)

    if not RMB.wait_health(NODE1, ROUTER_PORT, 60):
        log.error("  Router FAILED!")
        return None

    if is_dynamo:
        _lock_dvfs_baseline()
        log.info("  Locked GPUs to 1410MHz DVFS baseline")

    url = f"http://{NODE1}:{ROUTER_PORT}"
    if not RMB.test_generate(url):
        log.error("  Warmup FAILED!")
        return None
    log.info("  Deploy OK: %s", url)
    return url


def deploy_pd(scheme: str):
    """Deploy 2P(TP4)+4D(TP2)=16GPU. scheme='distserve' or 'biscale'."""
    is_biscale = (scheme == "biscale")
    log.info("=" * 60)
    log.info("Deploying %s: 2P(TP4)+4D(TP2)=16GPU", scheme.upper())
    log.info("=" * 60)

    RMB.cleanup_all()
    time.sleep(8)

    if is_biscale:
        RMB.unlock_freq_both(ALL_GPUS)
        log.info("  Unlocked all GPUs (DVFS biscale)")
    else:
        RMB.lock_freq_both(ALL_GPUS, 1410)
        log.info("  Locked all GPUs to 1410MHz")

    p_insts = []
    d_insts = []

    for i, gpus in enumerate(P_GROUPS):
        p_port = 53100 + i * 10
        bs_port = 49100 + i
        nic = GPU_NIC[gpus[0]]
        csv = ",".join(str(g) for g in gpus)
        dvfs_log = f"{DVFS_LOG_DIR}/{scheme}_p{i}_gpu{gpus[0]}.jsonl"
        dvfs_flags = _biscale_dvfs_flags() if is_biscale else ""
        env = (f"export SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal "
               f"CUDA_VISIBLE_DEVICES={csv} AFD_NVML_DEVICE_INDEX={gpus[0]} "
               f"AFD_NVML_DEVICE_INDICES={csv} SGLANG_HOST_IP={NODE1} "
               f"AFD_DVFS_DECISION_LOG='{dvfs_log}'; ")
        extra = (f"--disaggregation-mode prefill "
                 f"--disaggregation-transfer-backend mooncake "
                 f"--disaggregation-bootstrap-port {bs_port} "
                 f"--disaggregation-ib-device {nic} ")
        cmd = (f"{env}setsid prlimit --memlock=unlimited:unlimited {PYTHON} -m sglang.launch_server "
               f"--model-path {MODEL} --tp 4 --host {NODE1} --port {p_port} "
               f"--nccl-port {34000 + i * 10} {FLAGS}{extra}{dvfs_flags} "
               f"> {LOG_C}/{scheme}_p{i}.log 2>&1 < /dev/null &")
        RMB.dexec_local(cmd)
        p_insts.append({"port": p_port, "bs_port": bs_port, "gpus": gpus, "idx": i})
        log.info("  Started P%d: TP4 GPUs=%s port=%d", i, gpus, p_port)
        time.sleep(3)

    for i, gpus in enumerate(D_GROUPS):
        d_port = 53150 + i * 10
        nic = GPU_NIC[gpus[0]]
        csv = ",".join(str(g) for g in gpus)
        dvfs_log = f"{DVFS_LOG_DIR}/{scheme}_d{i}_gpu{gpus[0]}.jsonl"
        dvfs_flags = _biscale_dvfs_flags() if is_biscale else ""
        env = (f"export SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal "
               f"CUDA_VISIBLE_DEVICES={csv} AFD_NVML_DEVICE_INDEX={gpus[0]} "
               f"AFD_NVML_DEVICE_INDICES={csv} SGLANG_HOST_IP={NODE2} "
               f"AFD_DVFS_DECISION_LOG='{dvfs_log}'; ")
        extra = (f"--disaggregation-mode decode "
                 f"--disaggregation-transfer-backend mooncake "
                 f"--disaggregation-bootstrap-port {p_insts[0]['bs_port']} "
                 f"--disaggregation-ib-device {nic} ")
        cmd = (f"{env}setsid prlimit --memlock=unlimited:unlimited {PYTHON} -m sglang.launch_server "
               f"--model-path {MODEL} --tp 2 --host {NODE2} --port {d_port} "
               f"--nccl-port {34050 + i * 10} {FLAGS}{extra}{dvfs_flags} "
               f"> {LOG_C}/{scheme}_d{i}.log 2>&1 < /dev/null &")
        RMB.dexec_remote(cmd)
        d_insts.append({"port": d_port, "gpus": gpus, "idx": i})
        log.info("  Started D%d: TP2 GPUs=%s port=%d", i, gpus, d_port)
        time.sleep(3)

    log.info("  Waiting for servers...")
    for inst in p_insts:
        if not RMB.wait_health(NODE1, inst["port"], 400):
            log.error("  P%d FAILED!", inst["idx"])
            return None
    for inst in d_insts:
        if not RMB.wait_health(NODE2, inst["port"], 400):
            log.error("  D%d FAILED!", inst["idx"])
            return None
    log.info("  All P/D servers healthy!")

    rc_parts = [f"setsid {PYTHON} -m sglang_router.launch_router",
                "--pd-disaggregation", f"--host {NODE1} --port {ROUTER_PORT}"]
    for inst in p_insts:
        rc_parts.append(f"--prefill http://{NODE1}:{inst['port']} {inst['bs_port']}")
    for inst in d_insts:
        rc_parts.append(f"--decode http://{NODE2}:{inst['port']}")
    rc = " ".join(rc_parts) + f" > {LOG_C}/router.log 2>&1 < /dev/null &"
    RMB.dexec_local(rc)

    if not RMB.wait_health(NODE1, ROUTER_PORT, 60):
        log.error("  Router FAILED!")
        return None

    url = f"http://{NODE1}:{ROUTER_PORT}"
    if not RMB.test_generate(url):
        log.error("  Warmup FAILED!")
        return None
    log.info("  Deploy OK: %s", url)
    return url


def run_benchmark(url, dataset, qps):
    wl_file = WORKLOAD_DIR / f"macro_{dataset}_qps{qps}.jsonl"
    if not wl_file.exists():
        log.error("  Workload not found: %s", wl_file)
        return None

    with open(wl_file) as f:
        reqs = [json.loads(l) for l in f]

    last_arrival = max((r["arrival_time_s"] for r in reqs), default=0)
    run_s = int(min(max(400, last_arrival + 150), 900))
    log.info("  %s QPS=%d (%d reqs, run_window=%ds)", dataset, qps, len(reqs), run_s)

    n1_gpus = ALL_GPUS
    n2_gpus = ALL_GPUS
    summary = asyncio.run(RMB.run_workload(reqs, url + "/generate", n1_gpus, n2_gpus, run_s))

    if summary.get("status") == "PASS":
        e_per_tok = summary["total_energy_j"] / summary["total_tokens"] if summary.get("total_tokens") else 0
        log.info("  PASS: Thpt=%.1f | TTFT=%.1f | TPOT=%.1f | E/tok=%.2f J",
                 summary["throughput_tok_s"],
                 summary["ttft_proc_p50_ms"],
                 summary["tpot_p50_ms"],
                 e_per_tok)
    else:
        log.warning("  FAIL: %s", summary.get("status", "UNKNOWN"))
    return summary


def main():
    dataset = "code"
    qps_list = [2, 4, 8, 16]
    # Order: SGLang, DynamoLLM, DistServe, BiScale
    schemes = [
        ("sglang", deploy_native_tp1),
        ("dynamollm", deploy_native_tp1),
        ("distserve", deploy_pd),
        ("biscale", deploy_pd),
    ]

    all_results = {}

    for scheme_name, deploy_fn in schemes:
        all_results[scheme_name] = {}
        for qps in qps_list:
            log.info("\n" + "#" * 70)
            log.info("## %s | code QPS=%d", scheme_name.upper(), qps)
            log.info("#" * 70)

            url = deploy_fn(scheme_name)
            if url is None:
                log.error("  DEPLOY FAILED for %s QPS=%d", scheme_name, qps)
                all_results[scheme_name][qps] = {"status": "DEPLOY_FAILED"}
                continue

            summary = run_benchmark(url, dataset, qps)
            all_results[scheme_name][qps] = summary if summary else {"status": "BENCH_FAILED"}

    # Save combined results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = RESULTS_DIR / f"code_4schemes_16gpu_{ts}.json"
    out = {
        "config": "SGLang/DynamoLLM=16xTP1; DistServe/BiScale=2P(TP4)+4D(TP2)=16GPU",
        "dataset": dataset,
        "qps_list": qps_list,
        "schemes": [s[0] for s in schemes],
        "results": {}
    }
    for scheme_name, _ in schemes:
        for qps in qps_list:
            key = f"{scheme_name}_q{qps}"
            out["results"][key] = all_results[scheme_name][qps]

    out_file.write_text(json.dumps(out, indent=2, default=str))
    log.info("\n" + "=" * 70)
    log.info("ALL DONE! Results saved to: %s", out_file.name)
    log.info("=" * 70)

    # Print final table
    log.info("\nFINAL SUMMARY TABLE (code dataset):")
    log.info("%-12s %5s %8s %8s %8s %8s", "Scheme", "QPS", "Thpt", "TTFT", "TPOT", "E/tok")
    log.info("-" * 60)
    for scheme_name, _ in schemes:
        for qps in qps_list:
            res = all_results[scheme_name][qps]
            if isinstance(res, dict) and res.get("status") == "PASS":
                e = res["total_energy_j"] / res["total_tokens"] if res.get("total_tokens") else 0
                log.info("%-12s %5d %8.1f %8.1f %8.1f %8.2f",
                         scheme_name, qps,
                         res["throughput_tok_s"],
                         res["ttft_proc_p50_ms"],
                         res["tpot_p50_ms"],
                         e)
            else:
                status = res.get("status", "?") if isinstance(res, dict) else "?"
                log.info("%-12s %5d %8s", scheme_name, qps, status)


if __name__ == "__main__":
    main()
