#!/bin/bash
# Resume 6-scheme benchmark after Tier1 validation.
set -euo pipefail

DIR="/mnt/workspace/lt/sglang/benchmark/AFlex_bench/multi_node/node_scalibility_macro/more_trying"
LOG_BENCH="/mnt/workspace/lt/sglang/benchmark/AFlex_bench/multi_node/logs/best_pdaf_6scheme.log"
LOG_WD="/mnt/workspace/lt/sglang/benchmark/AFlex_bench/multi_node/logs/bench_watchdog.log"

cd "$DIR"
rm -f PAUSE_BENCH.flag

echo "[$(date)] Restarting 6-scheme benchmark..."
nohup env MN_NODE1_IP=10.252.129.34 MN_NODE2_IP=10.252.129.33 \
  python3 -u run_best_pdaf_6scheme_benchmark.py --resume \
  > "$LOG_BENCH" 2>&1 &

echo "[$(date)] Restarting watchdog..."
nohup env MN_NODE1_IP=10.252.129.34 MN_NODE2_IP=10.252.129.33 \
  python3 -u bench_watchdog.py > "$LOG_WD" 2>&1 &

echo "[$(date)] Done."
