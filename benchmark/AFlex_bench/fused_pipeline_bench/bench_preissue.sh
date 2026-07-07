#!/bin/bash
# Benchmark M=1 fast_path with pre-issue recv optimization
# Usage: bash bench_preissue.sh
set -e

NODE3="10.252.129.34"
LOGDIR="/workspace/sglang/benchmark/AFlex_bench/fused_pipeline_bench/logs"
BENCH_DIR="/workspace/sglang/benchmark/AFlex_bench/fused_pipeline_bench"

echo "=== Cleaning up old processes ==="
ssh $NODE3 "docker exec operator_test bash -c 'pkill -f sglang.launch_server || true; sleep 3'"

echo "=== Starting PDAF M=1 with pre-issue recv (code-level change) ==="
echo "--- Launching DF (FFN) ---"
ssh $NODE3 "docker exec -d operator_test bash -c 'cd /workspace/sglang && bash $BENCH_DIR/launch_df.sh preissue'"
sleep 3
echo "--- Launching DA (ATTN) ---"
ssh $NODE3 "docker exec -d operator_test bash -c 'cd /workspace/sglang && bash $BENCH_DIR/launch_da.sh preissue'"

echo "=== Waiting for servers to be ready ==="
for i in $(seq 1 120); do
    if ssh $NODE3 "docker exec operator_test bash -c 'curl -s http://10.252.129.34:53100/health'" 2>/dev/null | grep -q "ok"; then
        echo "DA ready after ${i}s"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "ERROR: DA not ready after 120s"
        ssh $NODE3 "docker exec operator_test tail -50 $LOGDIR/da_preissue.log"
        exit 1
    fi
    sleep 1
done

# Also check DF
for i in $(seq 1 30); do
    if ssh $NODE3 "docker exec operator_test bash -c 'curl -s http://10.252.129.34:53200/health'" 2>/dev/null | grep -q "ok"; then
        echo "DF ready"
        break
    fi
    sleep 1
done

echo "=== Warmup (5 requests) ==="
for i in $(seq 1 5); do
    ssh $NODE3 "docker exec operator_test bash -c 'curl -s http://10.252.129.34:53100/v1/completions -H \"Content-Type: application/json\" -d \"{\\\"model\\\":\\\"Qwen3-32B\\\",\\\"prompt\\\":\\\"Hello\\\",\\\"max_tokens\\\":16,\\\"temperature\\\":0}\" | python3 -c \"import sys,json; r=json.load(sys.stdin); print(r.get(\\\"choices\\\", [{}])[0].get(\\\"text\\\",\\\"\\\")[:40])\"'" 2>/dev/null
done

echo ""
echo "=== Measuring TPOT (10 sequential requests, 128 output tokens each) ==="
ssh $NODE3 "docker exec operator_test bash -c '
python3 -u -c \"
import requests, time, json

url = \\\"http://10.252.129.34:53100/v1/completions\\\"
results = []
for i in range(10):
    payload = {
        \\\"model\\\": \\\"Qwen3-32B\\\",
        \\\"prompt\\\": \\\"Write a detailed essay about artificial intelligence and its impact on society. Discuss the key areas where AI\\\",
        \\\"max_tokens\\\": 128,
        \\\"temperature\\\": 0,
        \\\"ignore_eos\\\": True,
    }
    t0 = time.time()
    resp = requests.post(url, json=payload)
    t1 = time.time()
    data = resp.json()
    usage = data.get(\\\"usage\\\", {})
    completion_tokens = usage.get(\\\"completion_tokens\\\", 128)
    prompt_tokens = usage.get(\\\"prompt_tokens\\\", 0)
    total_time = t1 - t0
    # Estimate TTFT ~ prompt_tokens * 0.5ms, rest is decode
    tpot = (total_time * 1000) / max(completion_tokens, 1)
    results.append(tpot)
    print(f\\\"  req {i}: {total_time*1000:.1f}ms total, {completion_tokens} tokens, TPOT={tpot:.1f}ms\\\")

# Skip first 2 as warmup
valid = results[2:]
avg_tpot = sum(valid) / len(valid)
print(f\\\"\\\")
print(f\\\"=== M=1 Pre-issue TPOT (avg of {len(valid)} reqs): {avg_tpot:.1f} ms ==\\\")
print(f\\\"    min={min(valid):.1f} max={max(valid):.1f} ms\\\")
\"'" 

echo ""
echo "=== Checking server-side pipeline time ==="
ssh $NODE3 "docker exec operator_test bash -c 'grep -o \"pipeline_time_ms=[0-9.]*\" $LOGDIR/da_preissue.log | tail -10'"

echo ""
echo "=== Done ==="
