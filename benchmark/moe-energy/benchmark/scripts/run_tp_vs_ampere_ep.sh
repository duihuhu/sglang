#!/usr/bin/env bash
# TP4 vs Ampere EP4 对照压测（仅两种部署，不含标准 EP none）
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
RESULT_DIR="${RESULT_DIR:-$MOE_DATA_DIR/tp_vs_ampere/run_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RESULT_DIR"
MASTER_LOG="$RESULT_DIR/master.log"
exec > >(tee -a "$MASTER_LOG") 2>&1
echo "=== TP4 vs ampere_ep benchmark started at $(date -Is) ==="
echo "RESULT_DIR=$RESULT_DIR"

# 基础阶梯 + 一个高并发点；可通过环境变量覆盖
CONCURRENCY_LIST="${CONCURRENCY_LIST:-512 1024 2048 4096 6144}"
NVLINK_CONCURRENCY="${NVLINK_CONCURRENCY:-2048}"

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
  pkill -9 -f "python3 -m sglang.launch_server" 2>/dev/null || true
  sleep 5
}

start_server() {
  local backend="$1"
  local log="$RESULT_DIR/${backend}_server.log"
  stop_server
  : > "$log"
  if [[ "$backend" == "tp4" ]]; then
    nohup python3 -m sglang.launch_server \
      "${COMMON_SERVER[@]}" \
      --tp-size 4 --dp-size 1 --ep-size 1 --moe-a2a-backend none \
      >> "$log" 2>&1 &
  elif [[ "$backend" == "ampere_ep" ]]; then
    nohup python3 -m sglang.launch_server \
      "${COMMON_SERVER[@]}" \
      --tp-size 4 --dp-size 4 --ep-size 4 --enable-dp-attention \
      --moe-a2a-backend ampere_ep \
      >> "$log" 2>&1 &
  else
    echo "unknown backend: $backend" >&2; exit 1
  fi
  for _ in $(seq 1 90); do
    if curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
      echo "[${backend}] server ready"
      return 0
    fi
    sleep 10
  done
  echo "[${backend}] server failed to start" >&2
  tail -30 "$log" >&2 || true
  exit 1
}

snapshot_nvlink() {
  local out="$1"
  nvidia-smi nvlink -gt d > "$out" 2>/dev/null || nvidia-smi nvlink --status > "$out" 2>/dev/null || true
}

run_bench() {
  local backend="$1" C="$2" tag="${3:-}"
  local suffix="${tag:+_${tag}}"
  local json="$RESULT_DIR/${backend}_c${C}${suffix}.json"
  local log="$RESULT_DIR/${backend}_c${C}${suffix}.bench.log"
  echo "=== ${backend} C=${C}${suffix} ==="
  ulimit -n 1048576 || true
  python3 -m sglang.benchmark.serving \
    "${COMMON_BENCH[@]}" \
    --num-prompts "$C" --max-concurrency "$C" \
    --output-file "$json" \
    > "$log" 2>&1
  local code=$?
  echo "$code" > "$RESULT_DIR/${backend}_c${C}${suffix}.exit"
  return $code
}

run_nvlink_case() {
  local backend="$1" C="$2"
  local before="$RESULT_DIR/${backend}_c${C}_nvlink_before.txt"
  local after="$RESULT_DIR/${backend}_c${C}_nvlink_after.txt"
  snapshot_nvlink "$before"
  run_bench "$backend" "$C" "nvlink"
  snapshot_nvlink "$after"
}

parse_server_log() {
  local backend="$1"
  python3 - "$RESULT_DIR/${backend}_server.log" "$RESULT_DIR/${backend}_scheduler.json" <<'PY'
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

for backend in tp4 ampere_ep; do
  start_server "$backend"
  for C in $CONCURRENCY_LIST; do
    if [[ "$C" == "$NVLINK_CONCURRENCY" ]]; then
      run_nvlink_case "$backend" "$C"
    else
      run_bench "$backend" "$C"
    fi
    curl -sf "http://${HOST}:${PORT}/health" >/dev/null || {
      echo "health check failed after ${backend} C=${C}" >&2
      break
    }
    sleep 3
  done
  parse_server_log "$backend"
done

SUMMARY_PY="$SCRIPT_DIR/summarize_tp_vs_ampere.py"
python3 "$SUMMARY_PY" "$RESULT_DIR"
echo "Results: $RESULT_DIR"
