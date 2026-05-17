#!/usr/bin/env python3
"""AF-only smoke test for --afd-async-schedule (M=3, IPC backend).

Launches:
  DA on GPU 0,1 (TP=2)  → port 30100
  DF on GPU 2,3 (TP=2)  → port 30101
Both processes share AFD_SCHED_PORT=65300.

Then sends a single chat-completion request to DA and prints the result
plus the benchmark timing breakdown.

Run from /workspace/sglang:
  /workspace/env/af-test/bin/python bench_afd_async_smoke.py
"""

import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request

REPO = "/workspace/sglang"
PYTHON = "/workspace/env/af-test/bin/python"
MODEL = "/models/Qwen/Qwen3-32B/"
MICRO_BATCH = 3
COMM_BACKEND = "ipc"

# IPC requires both processes to share the SAME CUDA_VISIBLE_DEVICES so
# cudaMemcpyPeer can cross between them.  We use physical GPUs 0,1,2,3:
#   DA → physical 0,1 (base_gpu_id=0, TP=2 over CUDA index 0,1)
#   DF → physical 2,3 (base_gpu_id=2, TP=2 over CUDA index 2,3)
# Both processes set CUDA_VISIBLE_DEVICES="0,1,2,3".
SHARED_GPUS = "0,1,2,3"
DA_BASE_GPU = 0  # DA TP0 → CUDA 0, DA TP1 → CUDA 1
DF_BASE_GPU = 2  # DF TP0 → CUDA 2, DF TP1 → CUDA 3
# IPC peer offset: peer_idx = local_idx + offset.  This makes each TP
# rank pair with its matching peer rank automatically.
DA_PEER_OFFSET = +2  # DA TP0(0) ↔ DF TP0(2); DA TP1(1) ↔ DF TP1(3)
DF_PEER_OFFSET = -2  # DF TP0(2) ↔ DA TP0(0); DF TP1(3) ↔ DA TP1(1)

TP = 2

DA_PORT = 30100
DF_PORT = 30101
SCHED_PORT = 65300

LOG_DIR = "/tmp/afd_async_smoke"
os.makedirs(LOG_DIR, exist_ok=True)


def free_port(p: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", p))
        except OSError:
            return False
    return True


def cleanup_old():
    """Kill leftover python processes and tmp socket files."""
    subprocess.call(["pkill", "-9", "-f", "sglang.launch_server"],
                    stderr=subprocess.DEVNULL)
    time.sleep(2)
    for f in os.listdir("/tmp"):
        if f.startswith("afd_ipc"):
            try:
                os.unlink(f"/tmp/{f}")
            except Exception:
                pass
    for f in os.listdir("/dev/shm"):
        if f.startswith("afd_ipc"):
            try:
                os.unlink(f"/dev/shm/{f}")
            except Exception:
                pass


def launch(role: str, base_gpu: int, peer_offset: int,
           port: int) -> subprocess.Popen:
    log = os.path.join(LOG_DIR, f"{role}.log")
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": SHARED_GPUS,
        "AFD_SCHED_HOST": "127.0.0.1",
        "AFD_SCHED_PORT": str(SCHED_PORT),
        "AFD_DETAILED_TIMING": "1",
        "AFD_IPC_PEER_OFFSET": str(peer_offset),
    }
    cmd = [
        PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL,
        "--tp", str(TP),
        "--base-gpu-id", str(base_gpu),
        "--host", "0.0.0.0",
        "--port", str(port),
        "--afd-perspective", role,
        "--afd-comm-backend", COMM_BACKEND,
        "--afd-micro-batch", str(MICRO_BATCH),
        "--afd-async-schedule",
        "--mem-fraction-static", "0.85",
        "--disable-overlap-schedule",
        "--disable-cuda-graph",
        "--trust-remote-code",
        "--skip-server-warmup",
    ]
    print(f"[smoke] launching {role} base_gpu={base_gpu} peer_offset={peer_offset:+d} -> port {port}")
    print(f"  log: {log}")
    print(f"  cmd: {' '.join(cmd)}")
    f = open(log, "w")
    proc = subprocess.Popen(
        cmd, stdout=f, stderr=subprocess.STDOUT, env=env,
        preexec_fn=os.setsid,
    )
    return proc


def wait_ready(url: str, proc: subprocess.Popen,
               timeout: float = 600.0) -> bool:
    print(f"[smoke] waiting for {url} (up to {timeout}s) ...")
    deadline = time.time() + timeout
    last_print = 0
    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"  process exited early with code {proc.returncode}")
            return False
        try:
            urllib.request.urlopen(f"{url}/health", timeout=3)
            print(f"  {url} ready")
            return True
        except Exception:
            pass
        if time.time() - last_print > 15:
            print(f"  ... still waiting ({int(time.time() - (deadline - timeout))}s elapsed)")
            last_print = time.time()
        time.sleep(2)
    print(f"  timed out waiting for {url}")
    return False


def main():
    cleanup_old()

    if not (free_port(DA_PORT) and free_port(DF_PORT)):
        print("ports busy, trying cleanup again")
        cleanup_old()

    # Start FFN first (it owns the IPC listener).
    df = launch("ffn", DF_BASE_GPU, DF_PEER_OFFSET, DF_PORT)
    time.sleep(3)
    da = launch("attn", DA_BASE_GPU, DA_PEER_OFFSET, DA_PORT)

    try:
        if not wait_ready(f"http://127.0.0.1:{DA_PORT}", da, timeout=600):
            raise RuntimeError("DA did not come up")
        # DF doesn't always answer /health (it's the peer); just verify the
        # process is alive.
        if df.poll() is not None:
            raise RuntimeError(f"DF exited (code {df.returncode})")

        print("[smoke] both up, sending one request")
        import json
        body = json.dumps({
            "text": "Once upon a time",
            "sampling_params": {"max_new_tokens": 32, "temperature": 0.0},
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{DA_PORT}/generate",
            data=body, headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        resp = urllib.request.urlopen(req, timeout=120).read().decode()
        t1 = time.time()
        print(f"[smoke] response in {t1 - t0:.2f}s:\n{resp[:500]}")

        # Quick perf hint: send a small batch, measure throughput.
        prompts = ["The quick brown fox" for _ in range(8)]
        t0 = time.time()
        out_tokens = 0
        import concurrent.futures
        def one(p):
            b = json.dumps({
                "text": p,
                "sampling_params": {"max_new_tokens": 64, "temperature": 0.0},
            }).encode()
            r = urllib.request.Request(
                f"http://127.0.0.1:{DA_PORT}/generate",
                data=b, headers={"Content-Type": "application/json"},
            )
            d = urllib.request.urlopen(r, timeout=120).read().decode()
            return d
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(one, prompts))
        t1 = time.time()
        for r in results[:2]:
            print(f"[batch sample] {r[:200]}")
        # Approx output tokens (count whitespace-tokens; rough)
        for r in results:
            try:
                d = json.loads(r)
                txt = d.get("text", "")
                out_tokens += len(txt.split())
            except Exception:
                pass
        print(f"[smoke] batch=8 wall={t1-t0:.2f}s out_words≈{out_tokens} "
              f"≈{out_tokens / (t1-t0):.1f} words/s")
    finally:
        print("[smoke] cleaning up")
        for p in [da, df]:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                pass
        time.sleep(2)
        for p in [da, df]:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:
                pass
        cleanup_old()


if __name__ == "__main__":
    main()
