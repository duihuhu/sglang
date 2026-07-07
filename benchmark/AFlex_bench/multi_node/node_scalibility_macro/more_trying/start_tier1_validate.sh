#!/bin/bash
# Pause 6-scheme benchmark, run Tier1 energy validation, then resume.
set -euo pipefail

DIR="/mnt/workspace/lt/sglang/benchmark/AFlex_bench/multi_node/node_scalibility_macro/more_trying"
LOG="/mnt/workspace/lt/sglang/benchmark/AFlex_bench/multi_node/logs/tier1_energy_validate.log"
QPS="${1:-4}"

cd "$DIR"
touch PAUSE_BENCH.flag

echo "[$(date)] Stopping watchdog / orchestrate / 6scheme..."
pkill -f bench_watchdog.py || true
pkill -f orchestrate_overnight.py || true
pkill -f run_best_pdaf_6scheme_benchmark.py || true
sleep 3
python3 -c "import sys; sys.path.insert(0,'..'); import run_macro_benchmark as R; R.cleanup_all()" || true
sleep 5

echo "[$(date)] Starting Tier1 validate QPS=$QPS..."
nohup env MN_NODE1_IP=10.252.129.34 MN_NODE2_IP=10.252.129.33 \
  python3 -u run_tier1_energy_validate.py --qps "$QPS" --resume \
  > "$LOG" 2>&1 &
echo "[$(date)] PID=$! log=$LOG"
