#!/usr/bin/env python3
"""
Benchmark PDAF decode TPOT: Baseline vs FusedPipeline (C++ fused send_recv).

Deploys PDAF (TP=4 DA + TP=4 DF interleaved) on node4 (single node, 8 GPU),
runs single-token decode, measures TPOT with two configs:
  1. Baseline (Plan B): AFD_FUSED_PIPELINE=0 (current optimized Python loop)
  2. FusedPipeline:      AFD_FUSED_PIPELINE=1 (C++ fused send_recv per layer)
"""

import subprocess
import time
import json
import sys
import os
import urllib.request

NODE = "10.252.129.34"  # node3 for deployment
MODEL = "/models/Qwen3-32B"
PYTHON = "/usr/bin/python3"
LOG_DIR = "/workspace/sglang/benchmark/AFlex_bench/fused_pipeline_bench/logs"
RESULT_DIR = "/mnt/workspace/lt/sglang/benchmark/AFlex_bench/fused_pipeline_bench"

DA_PORT = 53100
DF_PORT = 53200


def dexec(cmd, timeout=30):
    """Execute command in operator_test container on NODE."""
    r = subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", NODE,
         f"docker exec operator_test bash -c \"{cmd}\""],
        capture_output=True, text=True, timeout=timeout
    )
    return r.stdout + r.stderr


def dexec_script(script_content, script_name="run_tmp.sh", timeout=30):
    """Write a script to node and execute it (avoids quoting hell)."""
    script_path = f"/workspace/sglang/benchmark/AFlex_bench/fused_pipeline_bench/{script_name}"
    local_path = f"/mnt/workspace/lt/sglang/benchmark/AFlex_bench/fused_pipeline_bench/{script_name}"
    with open(local_path, "w") as f:
        f.write("#!/bin/bash\n" + script_content)
    os.chmod(local_path, 0o755)
    r = subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", NODE,
         f"docker exec operator_test bash {script_path}"],
        capture_output=True, text=True, timeout=timeout
    )
    return r.stdout + r.stderr


def kill_all():
    dexec("pkill -9 -f sglang 2>/dev/null || true")
    dexec("pkill -9 -f python.*sglang 2>/dev/null || true")
    time.sleep(3)


def wait_health(port, timeout=400):
    url = f"http://{NODE}:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(url, timeout=5)
            if resp.status == 200:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def deploy_pdaf(fused_pipeline: bool):
    """Deploy TP=4 PDAF on node (DA on GPU 0,2,4,6; DF on GPU 1,3,5,7)."""
    kill_all()
    time.sleep(2)

    fused_flag = "1" if fused_pipeline else "0"
    config_name = "fused" if fused_pipeline else "baseline"

    dexec(f"mkdir -p {LOG_DIR}")

    # Write launch script for DF
    df_script = f"""#!/bin/bash
export SGLANG_DISABLE_REQUEST_LOGGING=true
export UCX_LOG_LEVEL=fatal
export AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc
export SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=128
export AFD_ASYNC_PIPELINE=1
export AFD_IPC_SYNC_MODE=ipc_event
export AFD_IPC_CPP=1
export AFD_FUSED_PIPELINE={fused_flag}
export AFD_GPU_ONLY_IPC=0
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export AFD_IPC_PEER_OFFSET=-1
export AFD_NVML_DEVICE_INDICES=1,3,5,7
export AFD_NVML_DEVICE_INDEX=1
export AFD_SCHED_PORT=68500
export AFD_UCX_BASE_PORT=28300

setsid {PYTHON} -m sglang.launch_server --host {NODE} \\
    --port {DF_PORT} --afd-perspective ffn \\
    --nccl-port 34050 \\
    --base-gpu-id 1 \\
    --model-path {MODEL} --tp 4 --gpu-id-step 2 \\
    --afd-comm-backend ipc_cpp \\
    --afd-micro-batch 1 --mem-fraction-static 0.85 \\
    --max-running-requests 512 --skip-server-warmup \\
    --watchdog-timeout 600 \\
    --disable-cuda-graph --disable-piecewise-cuda-graph \\
    --disable-radix-cache \\
    --num-reserved-decode-tokens 512 \\
    > {LOG_DIR}/df_{config_name}.log 2>&1 < /dev/null &
echo "DF launched (pid=$!)"
"""

    # Write launch script for DA
    da_script = f"""#!/bin/bash
export SGLANG_DISABLE_REQUEST_LOGGING=true
export UCX_LOG_LEVEL=fatal
export AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc
export SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=128
export AFD_ASYNC_PIPELINE=1
export AFD_IPC_SYNC_MODE=ipc_event
export AFD_IPC_CPP=1
export AFD_FUSED_PIPELINE={fused_flag}
export AFD_GPU_ONLY_IPC=0
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export AFD_IPC_PEER_OFFSET=1
export AFD_NVML_DEVICE_INDICES=0,2,4,6
export AFD_NVML_DEVICE_INDEX=0
export AFD_SCHED_PORT=68500
export AFD_UCX_BASE_PORT=28300
export AFD_UCX_FFN_HOST=127.0.0.1

setsid {PYTHON} -m sglang.launch_server --host {NODE} \\
    --port {DA_PORT} --afd-perspective attn \\
    --nccl-port 34000 \\
    --base-gpu-id 0 \\
    --model-path {MODEL} --tp 4 --gpu-id-step 2 \\
    --afd-comm-backend ipc_cpp \\
    --afd-micro-batch 1 --mem-fraction-static 0.85 \\
    --max-running-requests 512 --skip-server-warmup \\
    --watchdog-timeout 600 \\
    --disable-cuda-graph --disable-piecewise-cuda-graph \\
    --disable-radix-cache \\
    --num-reserved-decode-tokens 512 \\
    > {LOG_DIR}/da_{config_name}.log 2>&1 < /dev/null &
echo "DA launched (pid=$!)"
"""

    print(f"  Launching DF (fused={fused_flag})...")
    result = dexec_script(df_script, f"launch_df_{config_name}.sh", timeout=30)
    print(f"    {result.strip()}")
    time.sleep(8)

    print(f"  Launching DA (fused={fused_flag})...")
    result = dexec_script(da_script, f"launch_da_{config_name}.sh", timeout=30)
    print(f"    {result.strip()}")

    print("  Waiting for DA to be ready...")
    if not wait_health(DA_PORT, timeout=400):
        print("  ERROR: DA server did not start! Checking logs...")
        log = dexec(f"tail -30 {LOG_DIR}/da_{config_name}.log", timeout=15)
        print(log[:2000])
        return False
    print("  DA ready!")
    return True


def benchmark_decode(num_warmup=5, num_requests=20):
    """Send decode-heavy requests (short prompt, long generation) and measure timing."""
    url = f"http://{NODE}:{DA_PORT}/v1/completions"
    prompt = "Explain the quicksort algorithm step by step in detail:"
    results = []

    for i in range(num_warmup + num_requests):
        payload = json.dumps({
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": 64,
            "temperature": 0,
            "stream": False,
        }).encode()

        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"})
        try:
            t0 = time.time()
            resp = urllib.request.urlopen(req, timeout=120)
            data = json.loads(resp.read())
            t1 = time.time()

            ct = data["usage"]["completion_tokens"]
            pt = data["usage"]["prompt_tokens"]
            total_ms = (t1 - t0) * 1000

            if ct > 1 and i >= num_warmup:
                tpot_ms = total_ms / ct
                results.append({
                    "total_ms": total_ms,
                    "completion_tokens": ct,
                    "prompt_tokens": pt,
                    "tpot_approx_ms": tpot_ms,
                })
                if i < num_warmup + 5:
                    print(f"    req {i-num_warmup}: {ct} tokens, total={total_ms:.0f}ms, "
                          f"tpot~{tpot_ms:.1f}ms")
        except Exception as e:
            print(f"    req {i} failed: {e}")
            if i >= num_warmup:
                time.sleep(5)

    return results


def benchmark_streaming(num_warmup=3, num_requests=15):
    """Streaming decode to get accurate per-token timing."""
    url = f"http://{NODE}:{DA_PORT}/v1/completions"
    prompt = "Write a detailed comparison of merge sort and quicksort:"
    results = []

    for i in range(num_warmup + num_requests):
        payload = json.dumps({
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": 32,
            "temperature": 0,
            "stream": True,
        }).encode()

        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"})
        try:
            resp = urllib.request.urlopen(req, timeout=120)
            token_times = []
            t_start = time.time()

            for line in resp:
                line = line.decode().strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    token_times.append(time.time())

            if len(token_times) >= 3 and i >= num_warmup:
                ttft = (token_times[0] - t_start) * 1000
                inter_token = [(token_times[j] - token_times[j-1]) * 1000
                               for j in range(1, len(token_times))]
                avg_tpot = sum(inter_token) / len(inter_token)
                p50_tpot = sorted(inter_token)[len(inter_token)//2]
                results.append({
                    "ttft_ms": ttft,
                    "avg_tpot_ms": avg_tpot,
                    "p50_tpot_ms": p50_tpot,
                    "num_tokens": len(token_times),
                })
                if i < num_warmup + 5:
                    print(f"    req {i-num_warmup}: {len(token_times)} tokens, "
                          f"TTFT={ttft:.0f}ms, TPOT_avg={avg_tpot:.1f}ms, "
                          f"TPOT_p50={p50_tpot:.1f}ms")
        except Exception as e:
            print(f"    streaming req {i} failed: {e}")

    return results


def summarize(results_ns, results_s):
    """Compute summary stats."""
    summary = {}
    if results_ns:
        tpots = [r["tpot_approx_ms"] for r in results_ns]
        summary["non_streaming"] = {
            "avg_tpot_ms": sum(tpots) / len(tpots),
            "min_tpot_ms": min(tpots),
            "max_tpot_ms": max(tpots),
            "p50_tpot_ms": sorted(tpots)[len(tpots)//2],
            "n": len(tpots),
        }
    if results_s:
        tpots = [r["avg_tpot_ms"] for r in results_s]
        ttfts = [r["ttft_ms"] for r in results_s]
        summary["streaming"] = {
            "avg_tpot_ms": sum(tpots) / len(tpots),
            "min_tpot_ms": min(tpots),
            "max_tpot_ms": max(tpots),
            "p50_tpot_ms": sorted(tpots)[len(tpots)//2],
            "avg_ttft_ms": sum(ttfts) / len(ttfts),
            "n": len(tpots),
        }
    return summary


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    all_results = {}
    configs = [
        ("Baseline (Plan B)", False),
        ("FusedPipeline (C++)", True),
    ]

    for name, fused in configs:
        print(f"\n{'='*60}")
        print(f"  Config: {name}")
        print(f"{'='*60}")

        if not deploy_pdaf(fused):
            print(f"  FAILED to deploy {name}")
            all_results[name] = {"error": "deployment failed"}
            kill_all()
            time.sleep(5)
            continue

        time.sleep(10)

        print(f"\n  --- Non-streaming benchmark ({name}) ---")
        results_ns = benchmark_decode(num_warmup=5, num_requests=20)

        print(f"\n  --- Streaming benchmark ({name}) ---")
        results_s = benchmark_streaming(num_warmup=3, num_requests=15)

        summary = summarize(results_ns, results_s)
        all_results[name] = {
            "summary": summary,
            "non_streaming_raw": results_ns,
            "streaming_raw": results_s,
        }

        if "non_streaming" in summary:
            ns = summary["non_streaming"]
            print(f"\n  [Non-streaming] Avg TPOT: {ns['avg_tpot_ms']:.1f}ms "
                  f"(p50={ns['p50_tpot_ms']:.1f}, min={ns['min_tpot_ms']:.1f}, "
                  f"max={ns['max_tpot_ms']:.1f})")
        if "streaming" in summary:
            s = summary["streaming"]
            print(f"  [Streaming] Avg TPOT: {s['avg_tpot_ms']:.1f}ms "
                  f"(p50={s['p50_tpot_ms']:.1f}), TTFT: {s['avg_ttft_ms']:.1f}ms")

        kill_all()
        time.sleep(5)

    # Final summary
    print(f"\n{'='*60}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Config':<30s} | {'TPOT(ns) avg':>12s} | {'TPOT(s) avg':>12s} | {'TTFT avg':>10s}")
    print(f"  {'-'*30}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")
    for name, res in all_results.items():
        if "error" in res:
            print(f"  {name:<30s} | {'ERROR':>12s} |")
            continue
        s = res.get("summary", {})
        ns_tpot = f"{s['non_streaming']['avg_tpot_ms']:.1f}ms" if "non_streaming" in s else "N/A"
        s_tpot = f"{s['streaming']['avg_tpot_ms']:.1f}ms" if "streaming" in s else "N/A"
        s_ttft = f"{s['streaming']['avg_ttft_ms']:.1f}ms" if "streaming" in s else "N/A"
        print(f"  {name:<30s} | {ns_tpot:>12s} | {s_tpot:>12s} | {s_ttft:>10s}")

    # Compute improvement
    names = list(all_results.keys())
    if len(names) >= 2 and "summary" in all_results[names[0]] and "summary" in all_results[names[1]]:
        b = all_results[names[0]].get("summary", {})
        f = all_results[names[1]].get("summary", {})
        if "streaming" in b and "streaming" in f:
            improvement = b["streaming"]["avg_tpot_ms"] - f["streaming"]["avg_tpot_ms"]
            pct = improvement / b["streaming"]["avg_tpot_ms"] * 100
            print(f"\n  Improvement: {improvement:.1f}ms ({pct:.1f}%)")

    # Save
    out_path = os.path.join(RESULT_DIR, "results.json")
    with open(out_path, "w") as fp:
        json.dump(all_results, fp, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
