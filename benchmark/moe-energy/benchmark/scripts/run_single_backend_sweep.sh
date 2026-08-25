#!/usr/bin/env bash
# 单 backend 阶梯压测（供并行 orchestrator 调用）
set -euo pipefail

BACKEND="${BACKEND:?BACKEND required (tp4|ampere_ep)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES required}"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1

MODEL=/models/Qwen3-30B-A3B
HOST=127.0.0.1
PORT="${PORT:?PORT required}"
RESULT_DIR="${RESULT_DIR:?RESULT_DIR required}"
GPU_LIST="${GPU_LIST:-$CUDA_VISIBLE_DEVICES}"

CONCURRENCY_LIST="${CONCURRENCY_LIST:-512 1024 2048 4096 6144}"
NVLINK_CONCURRENCY="${NVLINK_CONCURRENCY:-2048}"

mkdir -p "$RESULT_DIR"
WORKER_LOG="$RESULT_DIR/${BACKEND}_worker.log"
exec > >(tee -a "$WORKER_LOG") 2>&1

echo "=== ${BACKEND} sweep started at $(date -Is) ==="
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES PORT=$PORT RESULT_DIR=$RESULT_DIR"

COMMON_SERVER=(
  --model-path "$MODEL" --host 0.0.0.0 --port "$PORT"
  --moe-runner-backend triton --mem-fraction-static 0.85 --disable-cuda-graph
)

COMMON_BENCH=(
  --backend sglang --host "$HOST" --port "$PORT" --model "$MODEL"
  --dataset-name random-ids --random-input-len 32 --random-output-len 256
  --random-range-ratio 1 --tokenize-prompt --request-rate inf
  --warmup-requests 0 --seed 42 --disable-tqdm
)

stop_server() {
  pkill -9 -f "python3 -m sglang.launch_server.*--port ${PORT} " 2>/dev/null || true
  sleep 5
}

start_server() {
  local log="$RESULT_DIR/${BACKEND}_server.log"
  stop_server
  : > "$log"
  if [[ "$BACKEND" == "tp4" ]]; then
    nohup python3 -m sglang.launch_server \
      "${COMMON_SERVER[@]}" \
      --tp-size 4 --dp-size 1 --ep-size 1 --moe-a2a-backend none \
      >> "$log" 2>&1 &
  elif [[ "$BACKEND" == "ampere_ep" ]]; then
    nohup python3 -m sglang.launch_server \
      "${COMMON_SERVER[@]}" \
      --tp-size 4 --dp-size 4 --ep-size 4 --enable-dp-attention \
      --moe-a2a-backend ampere_ep \
      >> "$log" 2>&1 &
  else
    echo "unknown backend: $BACKEND" >&2
    exit 1
  fi
  for _ in $(seq 1 90); do
    if curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
      echo "[${BACKEND}] server ready on port ${PORT}"
      return 0
    fi
    sleep 10
  done
  echo "[${BACKEND}] server failed to start on port ${PORT}" >&2
  tail -30 "$log" >&2 || true
  exit 1
}

snapshot_nvlink() {
  local out="$1"
  nvidia-smi -i "$GPU_LIST" nvlink -gt d > "$out" 2>/dev/null \
    || nvidia-smi -i "$GPU_LIST" nvlink --status > "$out" 2>/dev/null \
    || true
}

run_bench() {
  local C="$1" tag="${2:-}"
  local suffix="${tag:+_${tag}}"
  local json="$RESULT_DIR/${BACKEND}_c${C}${suffix}.json"
  local log="$RESULT_DIR/${BACKEND}_c${C}${suffix}.bench.log"
  echo "=== ${BACKEND} C=${C}${suffix} ==="
  ulimit -n 1048576 || true
  python3 -m sglang.benchmark.serving \
    "${COMMON_BENCH[@]}" \
    --num-prompts "$C" --max-concurrency "$C" \
    --output-file "$json" \
    > "$log" 2>&1
  local code=$?
  echo "$code" > "$RESULT_DIR/${BACKEND}_c${C}${suffix}.exit"
  return $code
}

run_nvlink_case() {
  local C="$1"
  local before="$RESULT_DIR/${BACKEND}_c${C}_nvlink_before.txt"
  local after="$RESULT_DIR/${BACKEND}_c${C}_nvlink_after.txt"
  snapshot_nvlink "$before"
  run_bench "$C" "nvlink"
  snapshot_nvlink "$after"
}

parse_server_log() {
  python3 - "$RESULT_DIR/${BACKEND}_server.log" "$RESULT_DIR/${BACKEND}_scheduler.json" <<'PY'
import json, re, sys
from pathlib import Path
log_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
text = log_path.read_text(errors="replace") if log_path.exists() else ""
runs, queues, usage = [], [], []
for line in text.splitlines():
    m = re.search(r"#running-req: (\d+)", line)
    if m: runs.append(int(m.group(1)))
    m = re.search(r"#queue-req: (\d+)", line)
    if m: queues.append(int(m.group(1)))
    m = re.search(r"token usage: ([0-9.]+)", line)
    if m: usage.append(float(m.group(1)))
out = {
    "max_running_req": max(runs) if runs else 0,
    "max_queue_req": max(queues) if queues else 0,
    "max_token_usage": max(usage) if usage else 0.0,
    "mean_running_req": sum(runs)/len(runs) if runs else 0.0,
    "mean_queue_req": sum(queues)/len(queues) if queues else 0.0,
    "cuda_graph_false_count": text.count("cuda graph: False"),
    "oom_count": text.count("CUDA out of memory"),
    "traceback_count": text.count("Traceback"),
}
out_path.write_text(json.dumps(out, indent=2))
print(json.dumps(out))
PY
}

start_server
for C in $CONCURRENCY_LIST; do
  if [[ "$C" == "$NVLINK_CONCURRENCY" ]]; then
    run_nvlink_case "$C"
  else
    run_bench "$C"
  fi
  curl -sf "http://${HOST}:${PORT}/health" >/dev/null || {
    echo "health check failed after ${BACKEND} C=${C}" >&2
    break
  }
  sleep 3
done
parse_server_log
stop_server
echo "=== ${BACKEND} sweep finished at $(date -Is) ==="
