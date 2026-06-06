#!/usr/bin/env python3
"""Quick validation: TTFT processing field + dynamic M on 4 GPUs (0-3).

Launches AF M=2 with --afd-dynamic-micro-batch on GPU 0-3 (PA/PF TP1 + DA/DF TP1),
sends a few requests, and checks that:
1. ttft_processing_ms is populated in results
2. Server starts without errors with dynamic_mb flag
"""
import asyncio
import json
import os
import signal
import subprocess
import sys
import time

import aiohttp

PYTHON = "/workspace/env/sglang-tier/bin/python"
MODEL = "/models/Qwen/Qwen3-32B/"
BASE_PORT = 54000

sys.path.insert(0, os.path.dirname(__file__))
import run_fixed_qps_bench as B


def start_af_4gpu():
    """Start AF on GPU 0-3: PF(TP1,GPU0) PA(TP1,GPU1) DF(TP1,GPU2) DA(TP1,GPU3)."""
    env = os.environ.copy()
    env["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
    env["UCX_LOG_LEVEL"] = "fatal"
    env["AFD_UCX_TLS"] = "rc,tcp,cuda_copy,cuda_ipc"
    env["AFD_ASYNC_PIPELINE"] = "1"
    env["AFD_IPC_SYNC_MODE"] = "ipc_event"
    env["SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT"] = "300"
    env["SGLANG_DISAGGREGATION_WAITING_TIMEOUT"] = "300"

    procs = []

    def launch(name, port, persp, disagg, gpu, base_gpu_id, peer_device,
               extra_env=None, extra_args=None):
        e = env.copy()
        e["CUDA_VISIBLE_DEVICES"] = str(gpu)
        e["AFD_UCX_BASE_PORT"] = "18000" if disagg == "prefill" else "19000"
        e["AFD_SCHED_PORT"] = "17000" if disagg == "prefill" else "17100"
        e["AFD_IPC_PEER_DEVICE"] = str(peer_device)
        e["AFD_NVML_DEVICE_INDICES"] = str(gpu)
        e["AFD_NVML_DEVICE_INDEX"] = str(gpu)
        if extra_env:
            e.update(extra_env)

        cmd = [PYTHON, "-m", "sglang.launch_server",
               "--model-path", MODEL, "--tp", "1",
               "--host", "127.0.0.1", "--port", str(port),
               "--afd-perspective", persp,
               "--disaggregation-mode", disagg,
               "--base-gpu-id", str(base_gpu_id),
               "--afd-comm-backend", "ipc_cpp",
               "--mem-fraction-static", "0.85",
               "--disaggregation-transfer-backend", "mooncake",
               "--disaggregation-bootstrap-port", "49999",
               "--disaggregation-ib-device", "mlx5_4",
               "--skip-server-warmup",
               "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
               "--afd-disagg-interleave-poll",
               "--disable-radix-cache",
               "--afd-micro-batch", "2",
               "--afd-dynamic-micro-batch"]
        if extra_args:
            cmd += extra_args

        log_f = open(f"/tmp/test_dynm_{name}.log", "w")
        p = subprocess.Popen(cmd, env=e, stdout=log_f, stderr=subprocess.STDOUT)
        procs.append((name, p, log_f))
        print(f"  Started {name} (PID={p.pid}, GPU={gpu}, port={port})")

    # PF (ffn, prefill, GPU0)
    launch("pf", BASE_PORT + 11, "ffn", "prefill", 0, 0, 1)
    time.sleep(2)
    # PA (attn, prefill, GPU1)
    launch("pa", BASE_PORT + 10, "attn", "prefill", 1, 0, 0,
           extra_env={"AFD_UCX_FFN_HOST": "127.0.0.1"})
    time.sleep(2)
    # DF (ffn, decode, GPU2)
    launch("df", BASE_PORT + 21, "ffn", "decode", 2, 0, 1,
           extra_args=["--afd-attn-tp", "1"])
    time.sleep(2)
    # DA (attn, decode, GPU3)
    launch("da", BASE_PORT + 20, "attn", "decode", 3, 0, 0,
           extra_env={"AFD_UCX_FFN_HOST": "127.0.0.1"},
           extra_args=["--afd-ffn-tp", "1"])

    return procs


def cleanup(procs):
    for name, p, f in procs:
        p.terminate()
    time.sleep(2)
    for name, p, f in procs:
        p.kill()
        f.close()


async def test_request(url):
    """Send a single request and check for ttft_processing_ms."""
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        payload = {
            "text": "Hello " * 64,
            "sampling_params": {"max_new_tokens": 32, "temperature": 0.0},
            "stream": True,
        }
        t0 = time.perf_counter()
        first_token_time = None
        token_count = 0
        last_meta = None

        async with session.post(f"{url}/generate", json=payload) as resp:
            async for line in resp.content:
                text = line.decode().strip()
                if not text or text.startswith(":"):
                    continue
                if text.startswith("data:"):
                    text = text[5:].strip()
                if text == "[DONE]":
                    break
                try:
                    chunk = json.loads(text)
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    token_count += 1
                    if isinstance(chunk, dict) and "meta_info" in chunk:
                        last_meta = chunk["meta_info"]
                except json.JSONDecodeError:
                    pass

        ttft_e2e = (first_token_time - t0) * 1000 if first_token_time else 0
        ttft_proc = last_meta.get("time_to_first_token_processing", 0) * 1000 if last_meta else 0

        print(f"\n  Tokens: {token_count}")
        print(f"  TTFT (e2e, includes queue): {ttft_e2e:.1f} ms")
        print(f"  TTFT (processing, no queue): {ttft_proc:.1f} ms")
        print(f"  Queue wait: {ttft_e2e - ttft_proc:.1f} ms")

        if last_meta:
            print(f"  meta_info keys: {list(last_meta.keys())[:10]}")

        return ttft_proc > 0


def main():
    print("=" * 60)
    print("Test: TTFT processing field + dynamic M (4 GPU)")
    print("=" * 60)

    procs = start_af_4gpu()

    # Wait for router
    print("\nWaiting for servers to be ready...")
    router_url = f"http://127.0.0.1:{BASE_PORT}"

    # Start router
    router_env = os.environ.copy()
    router_cmd = [PYTHON, "-m", "sglang_router.launch_router",
                  "--port", str(BASE_PORT),
                  "--worker-urls",
                  f"http://127.0.0.1:{BASE_PORT+10}",
                  f"http://127.0.0.1:{BASE_PORT+20}"]
    router_log = open("/tmp/test_dynm_router.log", "w")
    router_p = subprocess.Popen(router_cmd, env=router_env,
                                stdout=router_log, stderr=subprocess.STDOUT)
    procs.append(("router", router_p, router_log))

    # Wait for all ports
    for port in [BASE_PORT + 10, BASE_PORT + 11, BASE_PORT + 20, BASE_PORT + 21]:
        if not B.wait_port("127.0.0.1", port, 300):
            print(f"FAIL: port {port} never came up")
            cleanup(procs)
            return 1

    time.sleep(5)
    if not B.wait_port("127.0.0.1", BASE_PORT, 60):
        print("FAIL: router never came up")
        cleanup(procs)
        return 1

    # Warmup
    print("Warming up...")
    B.warmup(router_url)

    # Test
    print("\nSending test request...")
    has_proc_ttft = asyncio.run(test_request(router_url))

    if has_proc_ttft:
        print("\n✓ PASS: ttft_processing_ms is populated")
    else:
        print("\n✗ FAIL: ttft_processing_ms is NOT populated")

    cleanup(procs)
    return 0 if has_proc_ttft else 1


if __name__ == "__main__":
    sys.exit(main())
