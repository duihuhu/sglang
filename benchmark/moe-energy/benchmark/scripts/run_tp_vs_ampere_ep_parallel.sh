#!/usr/bin/env bash
# TP4 (GPU 0-3) 与 ampere_ep (GPU 4-7) 并行压测
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

RESULT_DIR="${RESULT_DIR:-$MOE_DATA_DIR/no_cuda_sweep/run_$(date +%Y%m%d_%H%M%S)}"
CONCURRENCY_LIST="${CONCURRENCY_LIST:-512 1024 2048 4096 6144}"
NVLINK_CONCURRENCY="${NVLINK_CONCURRENCY:-2048}"

mkdir -p "$RESULT_DIR"
MASTER_LOG="$RESULT_DIR/master.log"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "=== Parallel TP4 vs ampere_ep started at $(date -Is) ==="
echo "RESULT_DIR=$RESULT_DIR"
echo "TP4:      CUDA_VISIBLE_DEVICES=0,1,2,3  PORT=30000"
echo "ampere_ep: CUDA_VISIBLE_DEVICES=4,5,6,7  PORT=30001"

# 仅清理本脚本使用的端口，避免误杀其他服务
for port in 30000 30001; do
  pkill -9 -f "python3 -m sglang.launch_server.*--port ${port} " 2>/dev/null || true
done
sleep 3

export CONCURRENCY_LIST NVLINK_CONCURRENCY RESULT_DIR

BACKEND=tp4 CUDA_VISIBLE_DEVICES=0,1,2,3 GPU_LIST=0,1,2,3 PORT=30000 \
  bash "$SCRIPT_DIR/run_single_backend_sweep.sh" &
PID_TP=$!

BACKEND=ampere_ep CUDA_VISIBLE_DEVICES=4,5,6,7 GPU_LIST=4,5,6,7 PORT=30001 \
  bash "$SCRIPT_DIR/run_single_backend_sweep.sh" &
PID_EP=$!

wait "$PID_TP"
TP_EXIT=$?
wait "$PID_EP"
EP_EXIT=$?

SUMMARY_PY="$SCRIPT_DIR/summarize_tp_vs_ampere.py"
python3 "$SUMMARY_PY" "$RESULT_DIR"

echo "tp4 worker exit=$TP_EXIT ampere_ep worker exit=$EP_EXIT"
echo "Results: $RESULT_DIR"
[[ "$TP_EXIT" -eq 0 && "$EP_EXIT" -eq 0 ]]
