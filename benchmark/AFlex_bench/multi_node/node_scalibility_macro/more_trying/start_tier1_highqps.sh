#!/bin/bash
# Pause 6-scheme, run Tier1 fixed-layout high-QPS validation.
set -euo pipefail

DIR="/mnt/workspace/lt/sglang/benchmark/AFlex_bench/multi_node/node_scalibility_macro/more_trying"
LOG="/mnt/workspace/lt/sglang/benchmark/AFlex_bench/multi_node/logs/tier1_highqps_validate.log"
QPS_LIST="${1:-8,12,16}"

cd "$DIR"
touch PAUSE_BENCH.flag

echo "[$(date)] Stopping watchdog / 6scheme..."
pkill -f bench_watchdog.py || true
pkill -f orchestrate_overnight.py || true
pkill -f run_best_pdaf_6scheme_benchmark.py || true
sleep 3
cd ..
python3 -c "import run_macro_benchmark as R; R.cleanup_all()" || true
sleep 5

echo "[$(date)] Starting Tier1 high-QPS validate (fixed QPS4 layout) QPS=$QPS_LIST..."
cd "$DIR"
nohup env MN_NODE1_IP=10.252.129.34 MN_NODE2_IP=10.252.129.33 \
  python3 -u run_tier1_highqps_validate.py --qps-list "$QPS_LIST" --resume \
  > "$LOG" 2>&1 &
echo "[$(date)] PID=$! log=$LOG"

# Wait in background and resume 6-scheme when done
nohup bash -c "
  while pgrep -f run_tier1_highqps_validate.py >/dev/null 2>&1; do sleep 30; done
  bash $DIR/resume_6scheme.sh
  echo \"[\$(date)] 6-scheme resumed after high-QPS validate\" >> $LOG
" > /dev/null 2>&1 &
