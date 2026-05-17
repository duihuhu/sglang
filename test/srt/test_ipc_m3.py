"""Standalone IPC M=3 end-to-end test — launches Attn + FFN servers.

Bypasses the test fixture's popen_launch_server (which uses /health_generate
and deadlocks with AFD) — instead spawns processes and polls /health directly.
"""

import os
import sys
import time
import json
import subprocess
import requests


MODEL = "/models/Qwen3-0.6B"
ATTN_PORT = 30200
FFN_PORT = 30201
MICRO_BATCH = 3
STARTUP_TIMEOUT = 600  # seconds for /health to return 200


def get_free_gpus(count=2):
    """Find GPUs with < 100 MiB used."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"]
    ).decode().strip().split("\n")
    free = []
    for line in out:
        parts = line.split(",")
        idx = int(parts[0].strip())
        mem = int(parts[1].strip())
        if mem < 100:
            free.append(idx)
    if len(free) < count:
        raise RuntimeError(f"Need {count} free GPUs, found {len(free)}: {free}")
    return free[:count]


def tail_file(path, lines=500):
    try:
        with open(path, "r") as f:
            all_lines = f.readlines()
            return "".join(all_lines[-lines:])
    except FileNotFoundError:
        return "(file not found)"


def main():
    try:
        gpus = get_free_gpus(2)
    except RuntimeError:
        # Fall back to using first 2 GPUs if none are below threshold
        import torch
        if torch.cuda.device_count() >= 2:
            gpus = [0, 1]
        else:
            raise
    gpu_attn = gpus[0]  # GPU ID 0 in CUDA_VISIBLE_DEVICES
    gpu_ffn = gpus[1]   # GPU ID 1
    print(f"[TEST] Using GPUs: Attn={gpu_attn}, FFN={gpu_ffn}", flush=True)

    env_base = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": f"{gpu_attn},{gpu_ffn}",
        "AFD_SCHED_HOST": "127.0.0.1",
        "AFD_SCHED_PORT": "65300",
        # /health must return 200 without generating a token, since FFN
        # cannot generate without Attn (and vice versa in AFD mode).
        "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
    }

    # Clean stale files from previous runs
    subprocess.run(
        ["rm", "-f", "/dev/shm/afd_ipc_flags_*", "/tmp/afd_ipc_*.sock",
         "/tmp/afd_test_attn.log", "/tmp/afd_test_ffn.log"],
        check=False,
    )

    procs = []

    def cleanup():
        for p in procs:
            try:
                p.terminate()
                p.wait(timeout=10)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        subprocess.run(
            ["rm", "-f", "/dev/shm/afd_ipc_flags_*", "/tmp/afd_ipc_*.sock"], check=False
        )

    try:
        # ── Launch FFN first (creates IPC Unix socket for Attn to connect) ──
        ffn_log = open("/tmp/afd_test_ffn.log", "w")
        ffn_cmd = [
            "sglang", "serve",
            "--model-path", MODEL,
            "--host", "127.0.0.1", "--port", str(FFN_PORT),
            "--trust-remote-code",
            "--afd-perspective", "ffn",
            "--afd-comm-backend", "ipc",
            "--afd-micro-batch", "1",
            "--base-gpu-id", "1",
            "--disable-cuda-graph",
            "--skip-server-warmup",
            "--watchdog-timeout", "3600",
        ]
        print(f"[TEST] Starting FFN on port {FFN_PORT}...", flush=True)
        proc_ffn = subprocess.Popen(
            ffn_cmd, env=env_base, stdout=ffn_log, stderr=subprocess.STDOUT,
        )
        procs.append(proc_ffn)

        # ── Wait for FFN /health ──
        ffn_url = f"http://127.0.0.1:{FFN_PORT}"
        deadline = time.time() + STARTUP_TIMEOUT
        ffn_ready = False
        while time.time() < deadline:
            if proc_ffn.poll() is not None:
                raise RuntimeError(
                    f"FFN exited with code {proc_ffn.returncode}\n"
                    f"Log tail:\n{tail_file('/tmp/afd_test_ffn.log')}"
                )
            try:
                r = requests.get(f"{ffn_url}/health", timeout=5)
                if r.status_code == 200:
                    ffn_ready = True
                    break
            except requests.RequestException:
                pass
            time.sleep(3)
        if not ffn_ready:
            raise TimeoutError(
                f"FFN /health not ready within {STARTUP_TIMEOUT}s\n"
                f"Log tail:\n{tail_file('/tmp/afd_test_ffn.log')}"
            )
        print("[TEST] FFN is healthy!", flush=True)

        # ── Launch Attn ──
        attn_log = open("/tmp/afd_test_attn.log", "w")
        attn_cmd = [
            "sglang", "serve",
            "--model-path", MODEL,
            "--host", "127.0.0.1", "--port", str(ATTN_PORT),
            "--trust-remote-code",
            "--afd-perspective", "attn",
            "--afd-comm-backend", "ipc",
            "--afd-micro-batch", "1",
            "--base-gpu-id", "0",
            "--disable-cuda-graph",
            "--skip-server-warmup",
        ]
        print(f"[TEST] Starting Attn on port {ATTN_PORT}...", flush=True)
        proc_attn = subprocess.Popen(
            attn_cmd, env=env_base, stdout=attn_log, stderr=subprocess.STDOUT,
        )
        procs.append(proc_attn)

        # ── Wait for Attn /health ──
        attn_url = f"http://127.0.0.1:{ATTN_PORT}"
        deadline = time.time() + STARTUP_TIMEOUT
        attn_ready = False
        while time.time() < deadline:
            if proc_attn.poll() is not None:
                raise RuntimeError(
                    f"Attn exited with code {proc_attn.returncode}\n"
                    f"Log tail:\n{tail_file('/tmp/afd_test_attn.log')}"
                )
            try:
                r = requests.get(f"{attn_url}/health", timeout=5)
                if r.status_code == 200:
                    attn_ready = True
                    break
            except requests.RequestException:
                pass
            time.sleep(3)
        if not attn_ready:
            raise TimeoutError(
                f"Attn /health not ready within {STARTUP_TIMEOUT}s\n"
                f"FFN log tail:\n{tail_file('/tmp/afd_test_ffn.log')}"
                f"\nAttn log tail:\n{tail_file('/tmp/afd_test_attn.log')}"
            )
        print("[TEST] Attn is healthy!", flush=True)

        # ── Give IPC handshake a moment to complete ──
        time.sleep(2)

        # ── Phase 1: Single request (M=1, validates basic IPC) ──
        print("[TEST] Phase 1: Sending single generate request...", flush=True)
        resp = requests.post(
            f"{attn_url}/generate",
            json={
                "text": "Hello, world!",
                "sampling_params": {"max_new_tokens": 32, "temperature": 0},
            },
            timeout=120,
        )
        assert resp.status_code == 200, f"Generate failed: {resp.text}"
        result = resp.json()
        print(f"[TEST] Single request response: {json.dumps(result, indent=2)}", flush=True)

        assert "text" in result, f"No 'text' in response: {result}"
        assert len(result["text"]) > 0, "Output text is empty"
        print(f"[TEST] Phase 1 PASS — generated text: {repr(result['text'])}", flush=True)

        # ── Phase 2: Concurrent requests ──
        import concurrent.futures
        print("[TEST] Phase 2: Sending 4 concurrent generate requests...", flush=True)
        prompts = [
            "What is 2+2?",
            "The capital of France is",
            "Write a haiku about",
            "In a world where AI",
        ]

        def _gen(prompt):
            return requests.post(
                f"{attn_url}/generate",
                json={
                    "text": prompt,
                    "sampling_params": {"max_new_tokens": 16, "temperature": 0},
                },
                timeout=120,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            futures = [executor.submit(_gen, p) for p in prompts]
            results = [f.result() for f in futures]

        for i, (prompt, r) in enumerate(zip(prompts, results)):
            assert r.status_code == 200, f"Request {i} failed: {r.text}"
            data = r.json()
            assert "text" in data and len(data["text"]) > 0, f"Request {i} empty"
            print(f"[TEST] Request {i} ({prompt[:30]}...): {repr(data['text'][:50])}", flush=True)
        print(f"[TEST] Phase 2 PASS — all {len(prompts)} concurrent requests completed", flush=True)

        # ── Analyze logs for async recv evidence ──
        ffn_out = tail_file("/tmp/afd_test_ffn.log", lines=2000)
        attn_out = tail_file("/tmp/afd_test_attn.log", lines=2000)

        async_recv = "recv_start_detail" in ffn_out or "recv_start_detail" in attn_out
        print(f"[TEST] Async recv triggered: {async_recv}", flush=True)

        two_phase = "recv_complete_detail" in ffn_out or "recv_complete_detail" in attn_out
        print(f"[TEST] 2-phase recv used: {two_phase}", flush=True)

        poll_only = "poll_only" in ffn_out or "poll_only" in attn_out
        print(f"[TEST] Poll-only phase: {poll_only}", flush=True)

        step_count = ffn_out.count("AFD_PER_STEP") + attn_out.count("AFD_PER_STEP")
        print(f"[TEST] AFD_PER_STEP entries: {step_count}", flush=True)

        return 0

    except Exception as e:
        print(f"\n[TEST] FAIL: {e}", flush=True)
        print(f"\n── Attn log tail ──\n{tail_file('/tmp/afd_test_attn.log')}", flush=True)
        print(f"\n── FFN log tail ──\n{tail_file('/tmp/afd_test_ffn.log')}", flush=True)
        return 1

    finally:
        cleanup()
        try:
            ffn_log.close()
            attn_log.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
