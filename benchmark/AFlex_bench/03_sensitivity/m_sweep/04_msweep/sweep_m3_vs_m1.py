#!/usr/bin/env python3
"""Sweep M=3 vs M=1 with working configurations.

Known working: qps=4, max_running=64, IL=1024, OL=128
Strategy: Keep qps=4 max_running=64 (known stable), vary parameters that
affect the compute/comm ratio to find where M=3 wins.

Theory: M=3 wins when compute_per_microbatch > comm_roundtrip (~2ms).
With batch=64, M=3 gives 21 tokens/microbatch.
Attn compute for 21 tokens with seq_len=1024: ~0.2ms (too short).
Attn compute for 21 tokens with seq_len=4096: ~0.8ms (still short).

Alternative: just increase max_running to get bigger batches.
But max_running>64 causes UCX port conflicts.

Real fix: we need max_running=192+ for M=3 to have 64 tokens/microbatch.
Let's try max_running=64,128,192 with careful port management.
"""
import subprocess, sys, os, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_4ARCH = os.path.join(HERE, "run_4arch.py")
PYTHON = "/workspace/env/af-test/bin/python"
LOG_DIR = "/tmp/sweep_m3_v4"
os.makedirs(LOG_DIR, exist_ok=True)

# Test configs: (qps, max_running, num_requests, IL, OL)
CONFIGS = [
    # Baseline: known working
    (4, 64, 100, 1024, 128),
    # Higher QPS to saturate faster (still max_running=64)
    (8, 64, 100, 1024, 128),
    # Higher QPS + longer output (more decode steps = more time at full batch)
    (8, 64, 100, 1024, 256),
    # Even higher QPS
    (16, 64, 100, 1024, 128),
]

CONCURRENCY = 200
TIMEOUT = 600


def clean_env():
    os.system("pkill -9 -f 'sglang.launch_server' 2>/dev/null")
    os.system("pkill -9 -f 'sglang_router' 2>/dev/null")
    os.system("pkill -9 -f 'af_launcher' 2>/dev/null")
    os.system("rm -f /dev/shm/cuda.shm.* 2>/dev/null")
    time.sleep(10)


def run_test(micro_batch, qps, max_running, num_requests, il, ol):
    """Run a single test and return metrics dict."""
    clean_env()
    label = f"M{micro_batch}_qps{qps}_mr{max_running}_il{il}_ol{ol}"
    log_path = os.path.join(LOG_DIR, f"{label}.log")

    cmd = [
        PYTHON, RUN_4ARCH,
        "--only", "pdaf",
        "--dataset", "sample",
        "--max-requests", str(num_requests),
        "--sample-qps", str(qps),
        "--sample-input-len", str(il),
        "--sample-output-len", str(ol),
        "--concurrency", str(CONCURRENCY),
        "--timeout", str(TIMEOUT),
        "--afd-comm-backend", "ucx",
        "--afd-micro-batch", str(micro_batch),
        "--pdaf-max-running-requests", str(max_running),
    ]

    print(f"  [{label}] ...", end="", flush=True)
    t_start = time.time()
    with open(log_path, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=HERE)
        try:
            proc.wait(timeout=TIMEOUT + 180)
        except subprocess.TimeoutExpired:
            proc.kill()
            print(f" TIMEOUT")
            return None
    elapsed = time.time() - t_start

    result_file = "/workspace/sglang/af_launch_logs/results_pdaf_ucx.json"
    if not os.path.exists(result_file):
        print(f" NO RESULTS ({elapsed:.0f}s)")
        return None

    with open(result_file) as f:
        data = json.load(f)

    rs = data.get("results", [])
    total = len(rs)
    ok = sum(1 for r in rs if r.get("success"))

    if ok == 0:
        print(f" FAILED 0/{total} ({elapsed:.0f}s)")
        return {"ok": 0, "total": total, "output_tps": 0, "tpot_ms": 0}

    successful = [r for r in rs if r.get("success")]
    tpots = []
    for r in successful:
        meta = r.get("metadata", {})
        if meta.get("decode_tpot_avg_s"):
            tpots.append(meta["decode_tpot_avg_s"] * 1000)
    avg_tpot = sum(tpots) / len(tpots) if tpots else 0

    total_out = sum(r.get("metadata", {}).get("completion_tokens", ol) for r in successful)
    wall = data.get("wall_duration_s", elapsed)
    output_tps = total_out / wall if wall > 0 else 0

    print(f" {ok}/{total} ok, {output_tps:.1f} tok/s, TPOT={avg_tpot:.1f}ms ({wall:.0f}s)")
    return {"ok": ok, "total": total, "output_tps": output_tps, "tpot_ms": avg_tpot, "wall_s": wall}


def main():
    print("=" * 70)
    print("  SWEEP v4: M=3 vs M=1 (stable configs)")
    print("=" * 70)

    results = []

    for qps, max_running, num_req, il, ol in CONFIGS:
        print(f"\n--- qps={qps} mr={max_running} il={il} ol={ol} n={num_req} ---")

        r1 = run_test(1, qps, max_running, num_req, il, ol)
        r3 = run_test(3, qps, max_running, num_req, il, ol)

        entry = {"qps": qps, "max_running": max_running, "il": il, "ol": ol, "M1": r1, "M3": r3}
        results.append(entry)

        if r1 and r3 and r1.get("output_tps", 0) > 0 and r3.get("output_tps", 0) > 0:
            ratio = r3["output_tps"] / r1["output_tps"]
            print(f"  => Ratio M3/M1 = {ratio:.3f}x")
            if ratio > 1.0:
                print(f"\n  *** M=3 WINS! ***")
                break

    # Summary
    print(f"\n{'='*70}")
    print(f"  {'Config':<35} {'M1 tok/s':<12} {'M3 tok/s':<12} {'Ratio':<8}")
    print(f"  {'-'*65}")
    for e in results:
        cfg = f"qps={e['qps']} mr={e['max_running']} il={e['il']} ol={e['ol']}"
        r1 = e.get("M1") or {}
        r3 = e.get("M3") or {}
        m1 = f"{r1['output_tps']:.1f}" if r1.get("ok") else "FAIL"
        m3 = f"{r3['output_tps']:.1f}" if r3.get("ok") else "FAIL"
        ratio = ""
        if r1.get("output_tps", 0) > 0 and r3.get("output_tps", 0) > 0:
            ratio = f"{r3['output_tps']/r1['output_tps']:.3f}"
        print(f"  {cfg:<35} {m1:<12} {m3:<12} {ratio:<8}")

    out_path = os.path.join(LOG_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
