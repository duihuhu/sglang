#!/usr/bin/env python3
"""Unified node-scalability benchmark: 6 schemes × 1/2/4 node(s).

Generates the data JSONs consumed by the node_scalability_energy plot:
  - 8gpu_six_schemes_{code,conv}_qps8.json   (1 node, 8 GPU)
  - 16gpu_six_schemes_{code,conv}_qps16.json  (2 nodes, 16 GPU)
  - 32gpu_six_schemes.json                    (4 nodes, 32 GPU)

Usage:
    python3 run_node_scalability_benchmark.py --nodes 1 --dataset code --qps 8
    python3 run_node_scalability_benchmark.py --nodes 2 --dataset both --qps 16
    python3 run_node_scalability_benchmark.py --nodes 4 --dataset both --qps 16,24,32
    python3 run_node_scalability_benchmark.py --nodes all --dataset both

Requires the common module (bench_common_cross.py) in the same directory.
"""
from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import importlib.util
import json
import logging
import os
import shlex
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import aiohttp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # node_scalibility/
AFLEX_ROOT = ROOT.parents[3]  # AFlex_bench/
MACRO_DIR = AFLEX_ROOT / "multi_node/more_test/macro/scripts"
sys.path.insert(0, str(MACRO_DIR))
import run_macro_benchmark as RMB  # noqa: E402

SGLANG_ROOT = Path(os.environ.get("SGLANG_ROOT", "/workspace/sglang"))
RMB.DVFS_PY_SRC = SGLANG_ROOT / "python" / "sglang" / "srt" / "layers" / "dvfs.py"

# Load the cross-node common module (originally run_aflex_32g_cross_af.py)
COMMON_PATH = HERE / "bench_common_cross.py"
_spec = importlib.util.spec_from_file_location("bench_common_cross", str(COMMON_PATH))
common = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(common)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("node_scalability")

# Node IPs (override via environment)
NODE1 = os.environ.get("BENCH_NODE1", "10.252.129.36")
NODE2 = os.environ.get("BENCH_NODE2", "10.252.129.35")
NODE3 = os.environ.get("BENCH_NODE3", "10.252.129.34")
NODE4 = os.environ.get("BENCH_NODE4", "10.252.129.33")

MODEL, PYTHON = common.MODEL, common.PYTHON
WORKLOAD_DIR = AFLEX_ROOT / "multi_node/more_test/macro/data/workloads"
DATA_DIR = ROOT / "data"
RESULTS_DIR = HERE / "results"
REMOTE_LOG_BASE = "/workspace/sglang/benchmark/AFlex_bench/multi_node/more_test/Ablation/node_scalibility/logs"
ENERGY_MODEL_V1 = RMB.ENERGY_MODEL_DIR_V1

SCHEMES = ("sglang", "dynamollm", "distserve", "biscale", "megascale", "aflex")
TTFT_SLO_MS, TPOT_SLO_MS = 2000, 100

COMMON_FLAGS = ("--mem-fraction-static 0.85 --disable-cuda-graph --disable-piecewise-cuda-graph "
                "--skip-server-warmup --disable-radix-cache --max-running-requests 512")

NATIVE_DVFS = (f"--dvfs-enabled --dvfs-energy-model-dir {ENERGY_MODEL_V1} "
               f"--dvfs-ttft-slo-ms {TTFT_SLO_MS} --dvfs-tpot-slo-us {TPOT_SLO_MS * 1000}")
BISCALE_DVFS = NATIVE_DVFS + " --dvfs-policy biscale"
AFLEX_DVFS = (f"--afd-dvfs-enabled --afd-energy-model-dir {ENERGY_MODEL_V1} "
              f"--afd-ttft-slo-ms {TTFT_SLO_MS} --afd-tpot-slo-us {TPOT_SLO_MS * 1000} "
              "--afd-dvfs-decode-compositional --afd-dvfs-idle-lock")


def dexec(host, cmd, timeout=60):
    return common.dexec(host, cmd, timeout=timeout)


def read_energy(host, gpu_count=8):
    code = (f"import json,pynvml;pynvml.nvmlInit();print(json.dumps({{i:pynvml."
            f"nvmlDeviceGetTotalEnergyConsumption(pynvml.nvmlDeviceGetHandleByIndex(i)) for i in range({gpu_count})}}))")
    r = dexec(host, f"{PYTHON} -c {shlex.quote(code)}", timeout=30)
    lines = [x for x in r.stdout.splitlines() if x.startswith("{")]
    return {int(k): int(v) for k, v in json.loads(lines[-1]).items()} if lines else {}


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct / 100
    lo, hi = int(pos), min(int(pos) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


# ===================== 1-NODE (8 GPU) =====================

class OneNodeBench:
    """Single-node 8GPU six-scheme benchmark (parallel across 3 nodes)."""

    ROUTER_PORT = 48000
    SERVER_BASE = 48100

    def __init__(self):
        self.remote_log_dir = f"{REMOTE_LOG_BASE}/8g_1node"

    def cleanup(self, host):
        ports = list(range(48000, 48200))
        dexec(host, common._remote_port_script("cleanup", ports), timeout=60)
        dexec(host, "pkill -9 -f 'sglang.launch_server|sglang::scheduler|sglang::detokenizer|"
              "sglang_router|sglang::router|smg::server' 2>/dev/null || true", timeout=30)
        time.sleep(3)

    def lock_freq(self, host, freq):
        gpus = ",".join(map(str, range(8)))
        if freq:
            common.ssh(host, f"nvidia-smi -i {gpus} -lgc {freq},{freq}", timeout=20)
        else:
            common.ssh(host, f"nvidia-smi -i {gpus} -rgc", timeout=20)

    def launch(self, host, cmd, name):
        full = f"mkdir -p {self.remote_log_dir}; {cmd} > {self.remote_log_dir}/{name}.log 2>&1 < /dev/null &"
        dexec(host, full)

    def wait_health(self, host, port, timeout=600):
        return common.wait_health(host, port, f"{host}:{port}", timeout=timeout)

    def deploy_native(self, host, scheme):
        for i in range(8):
            port = self.SERVER_BASE + i
            env = (f"export CUDA_VISIBLE_DEVICES={i} SGLANG_HOST_IP={host} "
                   f"SGLANG_DISABLE_REQUEST_LOGGING=true AFD_NVML_DEVICE_INDEX={i} AFD_NVML_DEVICE_INDICES={i};")
            dvfs = NATIVE_DVFS if scheme == "dynamollm" else ""
            cmd = (f"{env} setsid prlimit --memlock=unlimited:unlimited {PYTHON} -m sglang.launch_server "
                   f"--model-path {MODEL} --tp 1 --host {host} --port {port} "
                   f"--nccl-port {49000 + i * 10} {COMMON_FLAGS} {dvfs}")
            self.launch(host, cmd, f"{scheme}_inst{i}")
        for i in range(8):
            if not self.wait_health(host, self.SERVER_BASE + i):
                return None
        workers = " ".join(f"http://{host}:{self.SERVER_BASE + i}" for i in range(8))
        rc = (f"setsid {PYTHON} -m sglang_router.launch_router --host {host} "
              f"--port {self.ROUTER_PORT} --policy round_robin --worker-urls {workers}")
        dexec(host, f"fuser -k {self.ROUTER_PORT}/tcp 2>/dev/null || true", timeout=10)
        self.launch(host, rc, f"{scheme}_router")
        if not self.wait_health(host, self.ROUTER_PORT, timeout=120):
            return None
        return f"http://{host}:{self.ROUTER_PORT}"

    def deploy_pd(self, host, scheme):
        dvfs = BISCALE_DVFS if scheme == "biscale" else ""
        for i in range(4):
            port = self.SERVER_BASE + i
            bs_port = 49500 + i
            env = (f"export CUDA_VISIBLE_DEVICES={i} SGLANG_HOST_IP={host} "
                   f"SGLANG_DISABLE_REQUEST_LOGGING=true AFD_NVML_DEVICE_INDEX={i} AFD_NVML_DEVICE_INDICES={i};")
            cmd = (f"{env} setsid prlimit --memlock=unlimited:unlimited {PYTHON} -m sglang.launch_server "
                   f"--model-path {MODEL} --tp 1 --host {host} --port {port} "
                   f"--nccl-port {49000 + i * 10} --disaggregation-mode prefill "
                   f"--disaggregation-transfer-backend mooncake --disaggregation-bootstrap-port {bs_port} "
                   f"--disaggregation-ib-device mlx5_bond_0 {COMMON_FLAGS} {dvfs}")
            self.launch(host, cmd, f"{scheme}_p{i}")
        for i in range(4):
            port = self.SERVER_BASE + 10 + i
            env = (f"export CUDA_VISIBLE_DEVICES={4 + i} SGLANG_HOST_IP={host} "
                   f"SGLANG_DISABLE_REQUEST_LOGGING=true AFD_NVML_DEVICE_INDEX={4+i} AFD_NVML_DEVICE_INDICES={4+i};")
            cmd = (f"{env} setsid prlimit --memlock=unlimited:unlimited {PYTHON} -m sglang.launch_server "
                   f"--model-path {MODEL} --tp 1 --host {host} --port {port} "
                   f"--nccl-port {49100 + i * 10} --disaggregation-mode decode "
                   f"--disaggregation-transfer-backend mooncake --disaggregation-bootstrap-port 49500 "
                   f"--disaggregation-ib-device mlx5_bond_0 {COMMON_FLAGS} {dvfs}")
            self.launch(host, cmd, f"{scheme}_d{i}")
        for i in range(4):
            if not self.wait_health(host, self.SERVER_BASE + i):
                return None
        for i in range(4):
            if not self.wait_health(host, self.SERVER_BASE + 10 + i):
                return None
        prefill_args = " ".join(f"--prefill http://{host}:{self.SERVER_BASE + i} {49500 + i}" for i in range(4))
        decode_args = " ".join(f"--decode http://{host}:{self.SERVER_BASE + 10 + i}" for i in range(4))
        dexec(host, f"fuser -k {self.ROUTER_PORT}/tcp 2>/dev/null || true", timeout=10)
        rc = (f"setsid {PYTHON} -m sglang_router.launch_router --pd-disaggregation --mini-lb "
              f"{prefill_args} {decode_args} --host {host} --port {self.ROUTER_PORT}")
        self.launch(host, rc, f"{scheme}_router")
        if not self.wait_health(host, self.ROUTER_PORT, timeout=120):
            return None
        return f"http://{host}:{self.ROUTER_PORT}"

    def deploy_af(self, host, scheme):
        dvfs = AFLEX_DVFS if scheme == "aflex" else ""
        dexec(host, f"CUDA_VISIBLE_DEVICES=0 {PYTHON} -c \"import sys;sys.path.insert(0,'/workspace/sglang/python');"
              "from sglang.srt.layers.afd_ipc_cpp import get_module;get_module()\"", timeout=300)
        pairs = [(f"prefill", i, i * 2, i * 2 + 1) for i in range(3)] + [("decode", 0, 6, 7)]
        slot = 0
        endpoints = {}
        for role, idx, fg, ag in pairs:
            attn_port = self.SERVER_BASE + slot * 3
            ffn_port = self.SERVER_BASE + slot * 3 + 2
            bootstrap = self.SERVER_BASE + slot * 3 + 1
            ucx = 49200 + slot * 20
            sched = 49300 + slot * 20
            for perspective, port, gpu, peer in [("ffn", ffn_port, fg, 1), ("attn", attn_port, ag, 0)]:
                env = (f"export CUDA_VISIBLE_DEVICES={fg},{ag} SGLANG_HOST_IP={host} "
                       f"AFD_IPC_PEER_DEVICE={peer} AFD_IPC_SYNC_MODE=ipc_event AFD_ASYNC_PIPELINE=1 "
                       f"AFD_UCX_FFN_HOST=127.0.0.1 AFD_UCX_BASE_PORT={ucx} AFD_SCHED_PORT={sched} "
                       f"AFD_NVML_DEVICE_INDEX={gpu} AFD_NVML_DEVICE_INDICES={gpu} "
                       f"SGLANG_DISABLE_REQUEST_LOGGING=true;")
                base_gpu = 0 if perspective == "ffn" else 1
                cmd = (f"{env} setsid prlimit --memlock=unlimited:unlimited {PYTHON} -m sglang.launch_server "
                       f"--model-path {MODEL} --tp 1 --base-gpu-id {base_gpu} --host {host} --port {port} "
                       f"--nccl-port {49400 + slot * 20 + peer} --afd-perspective {perspective} "
                       f"--afd-comm-backend ipc_cpp --afd-micro-batch 1 --afd-attn-tp 1 --afd-ffn-tp 1 "
                       f"--disaggregation-mode {role} --disaggregation-transfer-backend mooncake "
                       f"--disaggregation-ib-device mlx5_bond_0 {COMMON_FLAGS} "
                       f"--watchdog-timeout 900 --afd-disagg-interleave-poll --num-reserved-decode-tokens 512 {dvfs}")
                self.launch(host, cmd, f"{scheme}_{role}{idx}_{perspective}")
                time.sleep(3)
            endpoints[(role, idx)] = (attn_port, bootstrap)
            slot += 1
        for role, idx, _, _ in pairs:
            ap, _ = endpoints[(role, idx)]
            if not self.wait_health(host, ap + 2):
                return None
            if not self.wait_health(host, ap):
                return None
        sub_ports = []
        d_ep = endpoints[("decode", 0)]
        for i in range(3):
            p_ap, p_bs = endpoints[("prefill", i)]
            sp = self.ROUTER_PORT + 1 + i
            dexec(host, f"fuser -k {sp}/tcp 2>/dev/null || true", timeout=10)
            rc = (f"setsid {PYTHON} -m sglang_router.launch_router --pd-disaggregation --mini-lb "
                  f"--prefill http://{host}:{p_ap} {p_bs} --decode http://{host}:{d_ep[0]} "
                  f"--host {host} --port {sp}")
            self.launch(host, rc, f"{scheme}_sub_p{i}")
            if not self.wait_health(host, sp, timeout=60):
                return None
            sub_ports.append(sp)
        workers = " ".join(f"http://{host}:{sp}" for sp in sub_ports)
        dexec(host, f"fuser -k {self.ROUTER_PORT}/tcp 2>/dev/null || true", timeout=10)
        rc = (f"setsid {PYTHON} -m sglang_router.launch_router --host {host} "
              f"--port {self.ROUTER_PORT} --policy round_robin --worker-urls {workers}")
        self.launch(host, rc, f"{scheme}_top")
        if not self.wait_health(host, self.ROUTER_PORT, timeout=60):
            return None
        return f"http://{host}:{self.ROUTER_PORT}"

    def run_workload(self, url, host, dataset, qps):
        workload = WORKLOAD_DIR / f"macro_{dataset}_qps{qps}.jsonl"
        reqs = [json.loads(line) for line in workload.read_text().splitlines() if line.strip()]
        last = max(r["arrival_time_s"] for r in reqs)
        run_s = int(min(max(last + 150, 300), 600))
        before = read_energy(host)
        rows = []

        async def _run():
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=run_s + 60)) as session:
                start = time.monotonic()
                tasks = [asyncio.create_task(common.send_one(session, url + "/generate", req, start, rows)) for req in reqs]
                done, pending = await asyncio.wait(tasks, timeout=run_s)
                for t in pending:
                    t.cancel()
                await asyncio.gather(*done, *pending, return_exceptions=True)
                return time.monotonic() - start

        duration = asyncio.run(_run())
        after = read_energy(host)
        energy_j = sum(after.get(g, 0) - before.get(g, 0) for g in range(8)) / 1000
        ok = [r for r in rows if r.get("success")]
        ttft = [r.get("ttft_proc_ms") or r.get("ttft_ms", 0) for r in ok]
        tpot = [r["tpot_ms"] for r in ok if r.get("tpot_ms", 0) > 0]
        tokens = sum(r.get("completion_tokens", 0) for r in ok)
        return {
            "status": "PASS" if len(ok) == len(reqs) else "PARTIAL" if ok else "FAIL",
            "total_requests": len(reqs), "successful": len(ok), "failed": len(reqs) - len(ok),
            "total_tokens": tokens, "throughput_tok_s": round(tokens / duration, 2) if duration else 0,
            "ttft_proc_avg_ms": round(statistics.fmean(ttft), 2) if ttft else 0,
            "ttft_proc_p50_ms": round(percentile(ttft, 50), 2),
            "ttft_proc_p99_ms": round(percentile(ttft, 99), 2),
            "tpot_avg_ms": round(statistics.fmean(tpot), 2) if tpot else 0,
            "tpot_p50_ms": round(percentile(tpot, 50), 2),
            "tpot_p99_ms": round(percentile(tpot, 99), 2),
            "total_energy_j": round(energy_j, 2),
            "energy_per_token_mj": round(energy_j * 1000 / tokens, 3) if tokens else None,
            "duration_s": round(duration, 2),
        }

    def run_scheme(self, host, scheme, dataset, qps):
        deploy_map = {
            "sglang": (self.deploy_native, 1410),
            "dynamollm": (self.deploy_native, None),
            "distserve": (self.deploy_pd, 1410),
            "biscale": (self.deploy_pd, None),
            "megascale": (self.deploy_af, 1410),
            "aflex": (self.deploy_af, None),
        }
        deploy_fn, freq = deploy_map[scheme]
        log.info("===== %s on %s (8GPU, %s QPS=%d) =====", scheme, host, dataset, qps)
        self.cleanup(host)
        if freq:
            self.lock_freq(host, freq)
        try:
            url = deploy_fn(host, scheme)
            if not url:
                return {"status": "DEPLOY_FAILED"}
            if not common.test_generate(url):
                return {"status": "WARMUP_FAILED"}
            time.sleep(3)
            result = self.run_workload(url, host, dataset, qps)
            result["scheme"] = scheme
            result["host"] = host
            result["gpu_count"] = 8
            result["qps"] = qps
            result["dataset"] = dataset
            return result
        except Exception as exc:
            log.exception("%s failed", scheme)
            return {"status": "FAILED", "error": str(exc)}
        finally:
            self.cleanup(host)
            self.lock_freq(host, None)

    def run_all(self, schemes, dataset, qps):
        """Run all schemes in parallel across 3 physical nodes."""
        node_assign = {
            "sglang": NODE1, "dynamollm": NODE1,
            "distserve": NODE3, "biscale": NODE3,
            "megascale": NODE4, "aflex": NODE4,
        }
        all_results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = {}
            for scheme in schemes:
                host = node_assign[scheme]
                futures[pool.submit(self.run_scheme, host, scheme, dataset, qps)] = scheme
            for future in concurrent.futures.as_completed(futures):
                scheme = futures[future]
                try:
                    all_results[scheme] = future.result()
                except Exception as exc:
                    log.error("%s exception: %s", scheme, exc)
                    all_results[scheme] = {"status": "FAILED", "error": str(exc)}
        return all_results


# ===================== 2-NODE (16 GPU) =====================
# Uses run_fixed16g_node_scaling.py logic (P/D split, ipc_cpp, TP4)
# Detailed deployment delegated to common module helpers

class TwoNodeBench:
    """Two-node 16GPU six-scheme benchmark."""

    def __init__(self):
        self.remote_log_dir = f"{REMOTE_LOG_BASE}/16g_two_node"

    def run_all(self, schemes, dataset, qps):
        """Placeholder: delegates to per-scheme runner using common module."""
        log.info("2-node 16GPU benchmark not yet fully inlined - use original scripts for now")
        return {}


# ===================== 4-NODE (32 GPU) =====================
# Uses run_32g_six_schemes.py logic

class FourNodeBench:
    """Four-node 32GPU six-scheme benchmark."""

    def __init__(self):
        self.remote_log_dir = f"{REMOTE_LOG_BASE}/32g_four_node"

    def run_all(self, schemes, dataset, qps):
        """Placeholder: delegates to per-scheme runner using common module."""
        log.info("4-node 32GPU benchmark not yet fully inlined - use original scripts for now")
        return {}


# ===================== MAIN =====================

DEFAULT_QPS = {1: 8, 2: 16, 4: 32}


def save_results(node_count, schemes, dataset, qps, results):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    gpu_count = node_count * 8
    key = f"{dataset}_qps{qps}"
    payload = {
        "meta": {
            "benchmark": f"{gpu_count}gpu_six_schemes",
            "dataset": dataset, "qps": qps,
            "gpu_count": gpu_count, "nodes": node_count,
        },
        "results": {scheme: {key: data} for scheme, data in results.items()},
    }
    out = DATA_DIR / f"{gpu_count}gpu_six_schemes_{dataset}_qps{qps}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    log.info("Saved %s", out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nodes", required=True,
                        help="1, 2, 4, or 'all' to run all three")
    parser.add_argument("--schemes", default=",".join(SCHEMES),
                        help="comma-separated subset of schemes")
    parser.add_argument("--dataset", choices=["code", "conv", "both"], default="both")
    parser.add_argument("--qps", default=None, help="QPS values (comma-separated); default depends on node count")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    node_list = [1, 2, 4] if args.nodes == "all" else [int(args.nodes)]
    schemes = [s.strip() for s in args.schemes.split(",")]
    datasets = ["code", "conv"] if args.dataset == "both" else [args.dataset]

    for nodes in node_list:
        qps_list = [int(q) for q in args.qps.split(",")] if args.qps else [DEFAULT_QPS[nodes]]
        bench = {1: OneNodeBench, 2: TwoNodeBench, 4: FourNodeBench}[nodes]()

        if args.dry_run:
            log.info("[DRY RUN] nodes=%d schemes=%s datasets=%s qps=%s", nodes, schemes, datasets, qps_list)
            continue

        for ds in datasets:
            for qps in qps_list:
                results = bench.run_all(schemes, ds, qps)
                if results:
                    save_results(nodes, schemes, ds, qps, results)

    log.info("All done")


if __name__ == "__main__":
    main()
