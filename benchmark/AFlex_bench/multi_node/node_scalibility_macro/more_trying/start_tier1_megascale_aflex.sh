#!/bin/bash
# Tier1-layout MegaScale vs AFlex: 6 datasets x 6 QPS (2,4,6,8,12,16)
set -euo pipefail

DIR="/mnt/workspace/lt/sglang/benchmark/AFlex_bench/multi_node/node_scalibility_macro/more_trying"
LOG="/mnt/workspace/lt/sglang/benchmark/AFlex_bench/multi_node/logs/tier1_megascale_aflex.log"

cd "$DIR"
rm -f PAUSE_BENCH.flag

echo "[$(date)] Stopping other benchmarks..."
pkill -f bench_watchdog.py || true
pkill -f run_7scheme_6dataset_benchmark.py || true
pkill -f run_best_pdaf_6scheme_benchmark.py || true
pkill -f run_tier1_megascale_aflex_benchmark.py || true
pkill -f orchestrate_overnight.py || true
sleep 3
cd ..
python3 -c "import run_macro_benchmark as R; R.cleanup_all()" || true
sleep 5

QPS_LIST="${QPS_LIST:-2,4,6,8,12,16}"
echo "[$(date)] Starting Tier1 MegaScale vs AFlex (QPS=$QPS_LIST)..."
cd "$DIR"
nohup env MN_NODE1_IP=10.252.129.34 MN_NODE2_IP=10.252.129.33 \
  python3 -u run_tier1_megascale_aflex_benchmark.py --resume --qps-list "$QPS_LIST" \
  > "$LOG" 2>&1 &
echo "[$(date)] PID=$! log=$LOG"

nohup env MN_NODE1_IP=10.252.129.34 MN_NODE2_IP=10.252.129.33 \
  python3 -u bench_watchdog.py >> /mnt/workspace/lt/sglang/benchmark/AFlex_bench/multi_node/logs/bench_watchdog.log 2>&1 &
echo "[$(date)] Watchdog restarted"
