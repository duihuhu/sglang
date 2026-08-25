#!/usr/bin/env bash
# Attn TP4 + MoE EP4（无 DP Attention）单点/阶梯压测
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
PORT="${PORT:-30000}"
RESULT_DIR="${RESULT_DIR:-$MOE_DATA_DIR/attn_moe_matrix/attn_tp_moe_ep}"
CONCURRENCY_LIST="${CONCURRENCY_LIST:-512 2048 6144}"

mkdir -p "$RESULT_DIR"
exec > >(tee -a "$RESULT_DIR/run.log") 2>&1
echo "=== Attn TP4 + MoE EP4 (tp=4 dp=1 ep=4, no dp-attention) ==="
echo "RESULT_DIR=$RESULT_DIR CONCURRENCY_LIST=$CONCURRENCY_LIST"

pkill -9 -f "python3 -m sglang.launch_server.*--port ${PORT} " 2>/dev/null || true
sleep 3

SERVER_LOG="$RESULT_DIR/server.log"
python3 -m sglang.launch_server \
  --model-path "$MODEL" --host 0.0.0.0 --port "$PORT" \
  --moe-runner-backend triton --mem-fraction-static 0.85 --disable-cuda-graph \
  --tp-size 4 --dp-size 1 --ep-size 4 --moe-a2a-backend none \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 90); do
  curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1 && break
  sleep 10
done
curl -sf "http://${HOST}:${PORT}/health" >/dev/null || { tail -30 "$SERVER_LOG"; exit 1; }
grep -E "max_total_num_tokens=|max_running_requests=|enable_dp_attention" "$SERVER_LOG" | tail -3

for C in $CONCURRENCY_LIST; do
  echo "=== C=${C} ==="
  ulimit -n 1048576 || true
  python3 -m sglang.benchmark.serving \
    --backend sglang --host "$HOST" --port "$PORT" --model "$MODEL" \
    --dataset-name random-ids --random-input-len 32 --random-output-len 256 \
    --random-range-ratio 1 --tokenize-prompt --request-rate inf \
    --warmup-requests 0 --seed 42 --disable-tqdm \
    --num-prompts "$C" --max-concurrency "$C" \
    --output-file "$RESULT_DIR/c${C}.json" \
    > "$RESULT_DIR/c${C}.bench.log" 2>&1
  curl -sf "http://${HOST}:${PORT}/health" >/dev/null || break
  sleep 3
done

kill -9 "$SERVER_PID" 2>/dev/null || true

python3 - "$RESULT_DIR" <<'PY'
import json, re, sys
from pathlib import Path
d = Path(sys.argv[1])
print(f"\n{'C':>6} {'out/s':>9} {'TTFT':>8} {'medTTFT':>8} {'p90TTFT':>9} {'TPOT':>8} {'dur':>7}")
print("-" * 65)
for p in sorted(d.glob("c*.json"), key=lambda x: int(x.stem[1:])):
    j = json.loads(p.read_text().splitlines()[-1])
    c = int(p.stem[1:])
    print(f"{c:>6} {j['output_throughput']:>9.1f} {j['mean_ttft_ms']:>8.0f} {j['median_ttft_ms']:>8.0f} {j['p90_ttft_ms']:>9.0f} {j['mean_tpot_ms']:>8.1f} {j['duration']:>7.1f}")
log = (d/"server.log").read_text(errors="replace")
runs, queues = [], []
for line in log.splitlines():
    m = re.search(r"#running-req: (\d+), #queue-req: (\d+)", line)
    if m: runs.append(int(m.group(1))); queues.append(int(m.group(2)))
if runs:
    print(f"\nscheduler: max_running={max(runs)} max_queue={max(queues)}")
PY
echo "Done: $RESULT_DIR"
