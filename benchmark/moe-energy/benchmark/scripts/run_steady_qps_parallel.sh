#!/usr/bin/env bash
# 8 卡并行稳态 QPS 压测：Attn DP+MoE EP vs Attn TP+MoE TP
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

BASE="${RESULT_DIR:-$MOE_DATA_DIR/steady_qps}"
QPS_LIST="${QPS_LIST:-30 40 50}"
NUM_PROMPTS="${NUM_PROMPTS:-8000}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-20}"
CONFIG_SH="$SCRIPT_DIR/run_steady_qps_config.sh"

mkdir -p "$BASE"
MASTER="$BASE/master.log"
exec > >(tee -a "$MASTER") 2>&1
echo "=== Steady QPS benchmark $(date -Is) ==="
echo "BASE=$BASE QPS_LIST=$QPS_LIST NUM_PROMPTS=$NUM_PROMPTS"

for port in 30000 30001; do
  pkill -9 -f "python3 -m sglang.launch_server.*--port ${port} " 2>/dev/null || true
done
sleep 3

run_pair() {
  local cfg1=$1 gpu1=$2 port1=$3 cfg2=$4 gpu2=$5 port2=$6
  echo "--- parallel: $cfg1 (GPU $gpu1) + $cfg2 (GPU $gpu2) ---"
  CONFIG=$cfg1 CUDA_VISIBLE_DEVICES=$gpu1 PORT=$port1 RESULT_DIR="$BASE/$cfg1" \
    QPS_LIST="$QPS_LIST" NUM_PROMPTS="$NUM_PROMPTS" WARMUP_REQUESTS="$WARMUP_REQUESTS" \
    bash "$CONFIG_SH" &
  local p1=$!
  CONFIG=$cfg2 CUDA_VISIBLE_DEVICES=$gpu2 PORT=$port2 RESULT_DIR="$BASE/$cfg2" \
    QPS_LIST="$QPS_LIST" NUM_PROMPTS="$NUM_PROMPTS" WARMUP_REQUESTS="$WARMUP_REQUESTS" \
    bash "$CONFIG_SH" &
  local p2=$!
  wait $p1; local e1=$?
  wait $p2; local e2=$?
  echo "parallel done: $cfg1 exit=$e1 $cfg2 exit=$e2"
  [[ $e1 -eq 0 && $e2 -eq 0 ]]
}

# Best steady config vs single-scheduler baseline
run_pair attn_dp_moe_ep 0,1,2,3 30000 attn_tp_moe_tp 4,5,6,7 30001

SUMMARY_PY="$SCRIPT_DIR/summarize_steady_qps.py"
python3 "$SUMMARY_PY" "$BASE"
echo "Results: $BASE"
