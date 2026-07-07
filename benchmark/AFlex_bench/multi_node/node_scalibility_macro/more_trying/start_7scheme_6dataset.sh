#!/bin/bash
# Stop other benchmarks and run 7-scheme x 6-dataset suite.
set -euo pipefail

DIR="/mnt/workspace/lt/sglang/benchmark/AFlex_bench/multi_node/node_scalibility_macro/more_trying"
LOG="/mnt/workspace/lt/sglang/benchmark/AFlex_bench/multi_node/logs/7scheme_6dataset.log"

cd "$DIR"
rm -f PAUSE_BENCH.flag

echo "[$(date)] Stopping other benchmarks..."
pkill -f bench_watchdog.py || true
pkill -f run_best_pdaf_6scheme_benchmark.py || true
pkill -f run_7scheme_6dataset_benchmark.py || true
pkill -f orchestrate_overnight.py || true
sleep 3
cd ..
python3 -c "import run_macro_benchmark as R; R.cleanup_all()" || true
sleep 5

QPS_LIST="${QPS_LIST:-2,4,6,8,12,16}"
echo "[$(date)] Starting 7-scheme x 6-dataset benchmark (QPS=$QPS_LIST, order=dataset)..."
cd "$DIR"
nohup env MN_NODE1_IP=10.252.129.34 MN_NODE2_IP=10.252.129.33 \
  python3 -u run_7scheme_6dataset_benchmark.py --resume --qps-list "$QPS_LIST" --order dataset \
  > "$LOG" 2>&1 &
echo "[$(date)] PID=$! log=$LOG"

nohup env MN_NODE1_IP=10.252.129.34 MN_NODE2_IP=10.252.129.33 \
  python3 -u bench_watchdog.py >> /mnt/workspace/lt/sglang/benchmark/AFlex_bench/multi_node/logs/bench_watchdog.log 2>&1 &
echo "[$(date)] Watchdog restarted"
