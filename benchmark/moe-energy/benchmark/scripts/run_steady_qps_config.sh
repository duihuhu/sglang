#!/usr/bin/env bash
# 单配置稳态 QPS 压测（泊松到达，不设 max-concurrency）
set -euo pipefail

CONFIG="${CONFIG:?CONFIG required: attn_tp_moe_tp|attn_tp_moe_ep|attn_dp_moe_tp|attn_dp_moe_ep}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?required}"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1

MODEL=/models/Qwen3-30B-A3B
HOST=127.0.0.1
PORT="${PORT:?required}"
RESULT_DIR="${RESULT_DIR:?required}"
QPS_LIST="${QPS_LIST:-30 40 50}"
NUM_PROMPTS="${NUM_PROMPTS:-8000}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-20}"

mkdir -p "$RESULT_DIR"
exec > >(tee -a "$RESULT_DIR/run.log") 2>&1
echo "=== steady QPS CONFIG=$CONFIG started $(date -Is) ==="
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES PORT=$PORT"
echo "QPS_LIST=$QPS_LIST NUM_PROMPTS=$NUM_PROMPTS WARMUP=$WARMUP_REQUESTS"

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

pkill -9 -f "python3 -m sglang.launch_server.*--port ${PORT} " 2>/dev/null || true
sleep 3

SERVER_LOG="$RESULT_DIR/server.log"
python3 -m sglang.launch_server \
  --model-path "$MODEL" --host 0.0.0.0 --port "$PORT" \
  --moe-runner-backend triton --mem-fraction-static 0.85 --disable-cuda-graph \
  "${EXTRA[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 90); do
  curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1 && break
  sleep 10
done
curl -sf "http://${HOST}:${PORT}/health" >/dev/null || { tail -30 "$SERVER_LOG"; exit 1; }
grep -E "max_total_num_tokens=|max_running_requests=" "$SERVER_LOG" | tail -2

for QPS in $QPS_LIST; do
  echo "=== QPS=${QPS} ==="
  ulimit -n 1048576 || true
  python3 -m sglang.benchmark.serving \
    --backend sglang --host "$HOST" --port "$PORT" --model "$MODEL" \
    --dataset-name random-ids --random-input-len 32 --random-output-len 256 \
    --random-range-ratio 1 --tokenize-prompt \
    --request-rate "$QPS" --num-prompts "$NUM_PROMPTS" \
    --warmup-requests "$WARMUP_REQUESTS" --seed 42 --disable-tqdm \
    --output-file "$RESULT_DIR/qps${QPS}.jsonl" \
    > "$RESULT_DIR/qps${QPS}.bench.log" 2>&1
  curl -sf "http://${HOST}:${PORT}/health" >/dev/null || { echo "health fail after QPS=$QPS"; break; }
  sleep 5
done

kill -9 "$SERVER_PID" 2>/dev/null || true
echo "=== steady QPS CONFIG=$CONFIG done $(date -Is) ==="
