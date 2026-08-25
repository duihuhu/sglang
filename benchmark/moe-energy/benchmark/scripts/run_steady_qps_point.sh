#!/usr/bin/env bash
# 单 QPS 点稳态压测：独立起服 → 压测 → 停服
set -euo pipefail

CONFIG="${CONFIG:?CONFIG required}"
QPS="${QPS:?QPS required}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?required}"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1

MODEL=/models/Qwen3-30B-A3B
HOST=127.0.0.1
PORT="${PORT:?required}"
RESULT_DIR="${RESULT_DIR:?required}"
NCCL_PORT="${NCCL_PORT:?NCCL_PORT required}"
DIST_INIT_PORT="${DIST_INIT_PORT:?DIST_INIT_PORT required}"
NUM_PROMPTS="${NUM_PROMPTS:-8000}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-20}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONITOR_PY="$SCRIPT_DIR/monitor_batch.py"

mkdir -p "$RESULT_DIR"
exec > >(tee -a "$RESULT_DIR/run.log") 2>&1
echo "=== steady QPS point CONFIG=$CONFIG QPS=$QPS started $(date -Is) ==="
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES PORT=$PORT NCCL_PORT=$NCCL_PORT DIST_INIT_PORT=$DIST_INIT_PORT NUM_PROMPTS=$NUM_PROMPTS"

case "$CONFIG" in
  attn_tp_moe_tp)
    EXTRA=(--tp-size 4 --dp-size 1 --ep-size 1 --moe-a2a-backend none) ;;
  attn_tp_moe_ep)
    EXTRA=(--tp-size 4 --dp-size 1 --ep-size 4 --moe-a2a-backend none) ;;
  attn_dp_moe_tp)
    EXTRA=(--tp-size 4 --dp-size 4 --ep-size 1 --enable-dp-attention --moe-a2a-backend none) ;;
  attn_dp_moe_ep)
    EXTRA=(--tp-size 4 --dp-size 4 --ep-size 4 --enable-dp-attention --moe-a2a-backend none) ;;
  *) echo "unknown CONFIG=$CONFIG"; exit 1 ;;
esac

SERVER_LOG="$RESULT_DIR/server.log"
python3 -m sglang.launch_server \
  --model-path "$MODEL" --host 0.0.0.0 --port "$PORT" \
  --nccl-port "$NCCL_PORT" \
  --dist-init-addr "127.0.0.1:${DIST_INIT_PORT}" \
  --moe-runner-backend triton --mem-fraction-static 0.85 --disable-cuda-graph \
  "${EXTRA[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 90); do
  curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1 && break
  sleep 10
done
curl -sf "http://${HOST}:${PORT}/health" >/dev/null || { tail -30 "$SERVER_LOG"; exit 1; }
grep -E "max_total_num_tokens=|max_running_requests=" "$SERVER_LOG" | tail -2

ulimit -n 1048576 || true
STOP_FILE="$RESULT_DIR/.monitor_stop"
BATCH_LOG="$RESULT_DIR/batch_samples.jsonl"
rm -f "$STOP_FILE" "$BATCH_LOG"
python3 "$MONITOR_PY" --host "$HOST" --port "$PORT" --interval 2 \
  --output "$BATCH_LOG" --stop-file "$STOP_FILE" &
MONITOR_PID=$!
sleep 1

python3 -m sglang.benchmark.serving \
  --backend sglang --host "$HOST" --port "$PORT" --model "$MODEL" \
  --dataset-name random-ids --random-input-len 32 --random-output-len 256 \
  --random-range-ratio 1 --tokenize-prompt \
  --request-rate "$QPS" --num-prompts "$NUM_PROMPTS" \
  --warmup-requests "$WARMUP_REQUESTS" --seed 42 --disable-tqdm \
  --output-file "$RESULT_DIR/result.jsonl" \
  > "$RESULT_DIR/bench.log" 2>&1

touch "$STOP_FILE"
wait "$MONITOR_PID" 2>/dev/null || true

kill -9 "$SERVER_PID" 2>/dev/null || true
echo "=== steady QPS point CONFIG=$CONFIG QPS=$QPS done $(date -Is) ==="
