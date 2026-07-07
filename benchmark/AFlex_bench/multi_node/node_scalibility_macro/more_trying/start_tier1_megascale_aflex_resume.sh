#!/bin/bash
# Resume Tier1 MegaScale/AFlex missing points (after deploy fix).
# Waits for any other benchmark using the GPUs, then runs.
set -euo pipefail

DIR="/mnt/workspace/lt/sglang/benchmark/AFlex_bench/multi_node/node_scalibility_macro/more_trying"
LOG="/mnt/workspace/lt/sglang/benchmark/AFlex_bench/multi_node/logs/tier1_megascale_aflex_resume.log"

cd "$DIR"
export MN_NODE1_IP="${MN_NODE1_IP:-10.252.129.34}"
export MN_NODE2_IP="${MN_NODE2_IP:-10.252.129.33}"
QPS_LIST="${QPS_LIST:-2,4,6,8,12,16}"

echo "[$(date)] Waiting for other benchmarks to finish..."
while pgrep -f "run_7scheme_6dataset_benchmark.py" >/dev/null 2>&1; do
  sleep 30
done
echo "[$(date)] GPU free — starting tier1 MegaScale/AFlex resume"

nohup env MN_NODE1_IP="$MN_NODE1_IP" MN_NODE2_IP="$MN_NODE2_IP" \
  python3 -u run_tier1_megascale_aflex_benchmark.py --resume --qps-list "$QPS_LIST" \
  > "$LOG" 2>&1 &
echo "[$(date)] PID=$! log=$LOG"
