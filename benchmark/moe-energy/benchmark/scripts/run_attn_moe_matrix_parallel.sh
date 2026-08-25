#!/usr/bin/env bash
# 8 卡并行跑 Attn×MoE 四宫格（两波，每波 2 配置）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

BASE="${RESULT_DIR:-$MOE_DATA_DIR/attn_moe_matrix}"
CONCURRENCY_LIST="${CONCURRENCY_LIST:-512 1024 2048 4096 6144}"
MATRIX_SH="$SCRIPT_DIR/run_matrix_config.sh"

mkdir -p "$BASE"
MASTER="$BASE/master.log"
exec > >(tee -a "$MASTER") 2>&1
echo "=== Attn×MoE matrix benchmark $(date -Is) ==="
echo "BASE=$BASE CONCURRENCY_LIST=$CONCURRENCY_LIST"

for port in 30000 30001; do
  pkill -9 -f "python3 -m sglang.launch_server.*--port ${port} " 2>/dev/null || true
done
sleep 3

run_pair() {
  local cfg1=$1 gpu1=$2 port1=$3 cfg2=$4 gpu2=$5 port2=$6
  echo "--- wave: $cfg1 (GPU $gpu1) + $cfg2 (GPU $gpu2) ---"
  CONFIG=$cfg1 CUDA_VISIBLE_DEVICES=$gpu1 PORT=$port1 RESULT_DIR="$BASE/$cfg1" \
    CONCURRENCY_LIST="$CONCURRENCY_LIST" bash "$MATRIX_SH" &
  local p1=$!
  CONFIG=$cfg2 CUDA_VISIBLE_DEVICES=$gpu2 PORT=$port2 RESULT_DIR="$BASE/$cfg2" \
    CONCURRENCY_LIST="$CONCURRENCY_LIST" bash "$MATRIX_SH" &
  local p2=$!
  wait $p1; local e1=$?
  wait $p2; local e2=$?
  echo "wave done: $cfg1 exit=$e1 $cfg2 exit=$e2"
  [[ $e1 -eq 0 && $e2 -eq 0 ]]
}

# Wave 1: 缺失的 attn_dp_moe_tp + 补全 attn_tp_moe_ep
run_pair attn_dp_moe_tp 0,1,2,3 30000 attn_tp_moe_ep 4,5,6,7 30001

# Wave 2: attn_tp_moe_tp + attn_dp_moe_ep
run_pair attn_tp_moe_tp 0,1,2,3 30000 attn_dp_moe_ep 4,5,6,7 30001

SUMMARY_PY="$SCRIPT_DIR/summarize_attn_moe_matrix.py"
python3 "$SUMMARY_PY" "$BASE"
echo "Results: $BASE"
