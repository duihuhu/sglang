#!/usr/bin/env bash
# 同一部署方式，8 卡并行扫多个 QPS 点（每点 4 卡独立服务）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

CONFIG="${CONFIG:-attn_dp_moe_ep}"
BASE="${RESULT_DIR:?RESULT_DIR required}"
QPS_WAVE1="${QPS_WAVE1:-30 40}"
QPS_WAVE2="${QPS_WAVE2:-50}"
NUM_PROMPTS="${NUM_PROMPTS:-8000}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-20}"
POINT_SH="$SCRIPT_DIR/run_steady_qps_point.sh"

mkdir -p "$BASE"
MASTER="$BASE/master.log"
exec > >(tee -a "$MASTER") 2>&1
echo "=== Same-config parallel steady QPS $(date -Is) ==="
echo "CONFIG=$CONFIG BASE=$BASE WAVE1=$QPS_WAVE1 WAVE2=$QPS_WAVE2"

for port in 30000 30001; do
  pkill -9 -f "python3 -m sglang.launch_server.*--port ${port} " 2>/dev/null || true
done
pkill -9 -f "python3 -m sglang.launch_server" 2>/dev/null || true
pkill -9 -f "run_steady_qps_point.sh" 2>/dev/null || true
pkill -9 -f "run_steady_qps_config.sh" 2>/dev/null || true
pkill -9 -f benchmark.serving 2>/dev/null || true
sleep 5

run_point() {
  local qps=$1 gpus=$2 port=$3 nccl_port=$4 dist_port=$5
  local dir="$BASE/qps${qps}"
  mkdir -p "$dir"
  CONFIG="$CONFIG" QPS="$qps" CUDA_VISIBLE_DEVICES="$gpus" PORT="$port" \
    NCCL_PORT="$nccl_port" DIST_INIT_PORT="$dist_port" RESULT_DIR="$dir" \
    NUM_PROMPTS="$NUM_PROMPTS" WARMUP_REQUESTS="$WARMUP_REQUESTS" \
    bash "$POINT_SH"
}

run_wave() {
  local -a pids=()
  local -a qsps=()
  local idx=0
  for qps in $*; do
    qsps+=("$qps")
    if (( idx == 0 )); then
      run_point "$qps" "0,1,2,3" 30000 29500 29700 &
    else
      sleep 10
      run_point "$qps" "4,5,6,7" 30001 29600 29800 &
    fi
    pids+=($!)
    idx=$((idx + 1))
  done
  local fail=0
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "FAIL qps=${qsps[$i]}"
      fail=1
    fi
  done
  return $fail
}

echo "--- wave1: $QPS_WAVE1 ---"
run_wave $QPS_WAVE1

if [[ -n "$QPS_WAVE2" ]]; then
  echo "--- wave2: $QPS_WAVE2 ---"
  run_wave $QPS_WAVE2
fi

SUMMARY_PY="$SCRIPT_DIR/summarize_steady_qps.py"
python3 "$SUMMARY_PY" "$BASE"
echo "Results: $BASE"
