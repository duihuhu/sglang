#!/usr/bin/env bash
# TP4 with --max-running-requests 6144, single point C=6144
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1

MODEL=/models/Qwen3-30B-A3B
HOST=127.0.0.1
PORT=30000
RESULT_DIR="${RESULT_DIR:-$MOE_DATA_DIR/attn_moe_matrix/tp4_maxrun6144_c6144}"
MAX_RUNNING="${MAX_RUNNING_REQUESTS:-6144}"
C="${CONCURRENCY:-6144}"

mkdir -p "$RESULT_DIR"
LOG="$RESULT_DIR/run.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== TP4 max_running_requests=${MAX_RUNNING} C=${C} started $(date -Is) ==="

pkill -9 -f "python3 -m sglang.launch_server.*--port ${PORT} " 2>/dev/null || true
sleep 3

SERVER_LOG="$RESULT_DIR/tp4_server.log"
python3 -m sglang.launch_server \
  --model-path "$MODEL" --host 0.0.0.0 --port "$PORT" \
  --moe-runner-backend triton --mem-fraction-static 0.85 --disable-cuda-graph \
  --tp-size 4 --dp-size 1 --ep-size 1 --moe-a2a-backend none \
  --max-running-requests "$MAX_RUNNING" \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 90); do
  if curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
    echo "server ready pid=$SERVER_PID"
    break
  fi
  sleep 10
done
curl -sf "http://${HOST}:${PORT}/health" >/dev/null || { tail -30 "$SERVER_LOG"; exit 1; }

grep -E "max_total_num_tokens=|max_running_requests=" "$SERVER_LOG" | tail -3

ulimit -n 1048576 || true
BENCH_LOG="$RESULT_DIR/tp4_c${C}.bench.log"
JSON="$RESULT_DIR/tp4_c${C}.json"
python3 -m sglang.benchmark.serving \
  --backend sglang --host "$HOST" --port "$PORT" --model "$MODEL" \
  --dataset-name random-ids --random-input-len 32 --random-output-len 256 \
  --random-range-ratio 1 --tokenize-prompt --request-rate inf \
  --warmup-requests 0 --seed 42 --disable-tqdm \
  --num-prompts "$C" --max-concurrency "$C" \
  --output-file "$JSON" \
  > "$BENCH_LOG" 2>&1

kill -9 "$SERVER_PID" 2>/dev/null || true
sleep 3

python3 - "$JSON" "$SERVER_LOG" <<'PY'
import json, re, sys
from pathlib import Path

bench = json.loads(Path(sys.argv[1]).read_text().splitlines()[-1])
log = Path(sys.argv[2]).read_text(errors="replace")
runs, queues = [], []
for line in log.splitlines():
    m = re.search(r"#running-req: (\d+), #queue-req: (\d+)", line)
    if m:
        runs.append(int(m.group(1))); queues.append(int(m.group(2)))
print("=== benchmark ===")
print(f"completed={bench['completed']} out_tps={bench['output_throughput']:.1f}")
print(f"TTFT mean={bench['mean_ttft_ms']:.0f} median={bench['median_ttft_ms']:.0f} p90={bench['p90_ttft_ms']:.0f} p99={bench['p99_ttft_ms']:.0f}")
print(f"TPOT mean={bench['mean_tpot_ms']:.1f} median={bench['median_tpot_ms']:.1f}")
print(f"E2E mean={bench['mean_e2e_latency_ms']:.0f} dur={bench['duration']:.1f}s")
if runs:
    print("=== scheduler peaks ===")
    print(f"max_running={max(runs)} max_queue={max(queues)} mean_queue={sum(queues)/len(queues):.0f}")
PY

echo "Results: $RESULT_DIR"
