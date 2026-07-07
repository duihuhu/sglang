#!/bin/bash
# Clean restart: 6 schemes x 6 datasets x QPS 2/8/16
set -euo pipefail

DIR="/mnt/workspace/lt/sglang/benchmark/AFlex_bench/multi_node/node_scalibility_macro/more_trying"
LOG="/mnt/workspace/lt/sglang/benchmark/AFlex_bench/multi_node/logs/6scheme_6dataset.log"

cd "$DIR"

echo "[$(date)] Stopping all benchmarks..."
pkill -f bench_watchdog.py || true
pkill -f run_6scheme_6dataset_benchmark.py || true
pkill -f run_7scheme_6dataset_benchmark.py || true
pkill -f run_tier1_megascale_aflex_benchmark.py || true
pkill -f start_tier1_megascale_aflex_resume.sh || true
pkill -f orchestrate_full_qps6.sh || true
sleep 3
cd ..
python3 -c "import run_macro_benchmark as R; R.cleanup_all()" 2>/dev/null || true
sleep 5

echo "[$(date)] Removing old benchmark results and charts..."
cd "$DIR"
rm -f results/7scheme_6dataset_*.json
rm -f results/tier1_megascale_aflex_*.json
rm -f results/6scheme_6dataset_*.json
rm -f charts/*.png charts/*.pdf 2>/dev/null || true

# Keep tier1_*_solutions.json (ILP inputs), remove old validation runs
rm -f results/tier1_energy_validate_*.json results/tier1_highqps_validate_*.json

QPS_LIST="${QPS_LIST:-2,8,16}"
echo "[$(date)] Starting clean 6-scheme benchmark QPS=$QPS_LIST ..."
nohup env MN_NODE1_IP=10.252.129.34 MN_NODE2_IP=10.252.129.33 \
  python3 -u run_6scheme_6dataset_benchmark.py --qps-list "$QPS_LIST" \
  > "$LOG" 2>&1 &
echo "[$(date)] PID=$! log=$LOG"

nohup env MN_NODE1_IP=10.252.129.34 MN_NODE2_IP=10.252.129.33 \
  python3 -u bench_watchdog.py >> /mnt/workspace/lt/sglang/benchmark/AFlex_bench/multi_node/logs/bench_watchdog.log 2>&1 &
