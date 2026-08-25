#!/usr/bin/env bash
# 四宫格稳态 QPS 全量测试（每配置 8 卡并行扫 QPS 30/40/50）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

BASE="${RESULT_DIR:-$MOE_DATA_DIR/steady_qps_matrix}"
QPS_WAVE1="${QPS_WAVE1:-30 40}"
QPS_WAVE2="${QPS_WAVE2:-50}"
NUM_PROMPTS="${NUM_PROMPTS:-8000}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-20}"
PARALLEL_SH="$SCRIPT_DIR/run_steady_qps_same_config_parallel.sh"
SUMMARY_PY="$SCRIPT_DIR/summarize_steady_qps_matrix.py"

mkdir -p "$BASE"
MASTER="$BASE/master.log"
exec > >(tee -a "$MASTER") 2>&1
echo "=== Steady QPS matrix $(date -Is) ==="
echo "BASE=$BASE"

CONFIGS=(
  attn_tp_moe_tp
  attn_tp_moe_ep
  attn_dp_moe_tp
  attn_dp_moe_ep
)

for cfg in "${CONFIGS[@]}"; do
  echo "======== config=$cfg ========"
  CONFIG="$cfg" RESULT_DIR="$BASE/$cfg" \
    QPS_WAVE1="$QPS_WAVE1" QPS_WAVE2="$QPS_WAVE2" \
    NUM_PROMPTS="$NUM_PROMPTS" WARMUP_REQUESTS="$WARMUP_REQUESTS" \
    bash "$PARALLEL_SH"
done

python3 "$SUMMARY_PY" "$BASE"
echo "Results: $BASE"
