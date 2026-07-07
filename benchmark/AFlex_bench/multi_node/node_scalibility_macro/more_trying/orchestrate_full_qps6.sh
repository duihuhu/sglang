#!/bin/bash
# Full QPS6 benchmark pipeline (sequential, no GPU conflict):
#   Phase 1: Tier1-layout MegaScale vs AFlex (if not already running/done)
#   Phase 2: Other 5 schemes (SGLang, DynamoLLM, DistServe, BiScale, AFlex-Tier1)
set -euo pipefail

DIR="/mnt/workspace/lt/sglang/benchmark/AFlex_bench/multi_node/node_scalibility_macro/more_trying"
LOG_DIR="/mnt/workspace/lt/sglang/benchmark/AFlex_bench/multi_node/logs"
MS_LOG="$LOG_DIR/tier1_megascale_aflex.log"
OTHER_LOG="$LOG_DIR/7scheme_other_qps6.log"
ORCH_LOG="$LOG_DIR/orchestrate_full_qps6.log"

QPS_LIST="${QPS_LIST:-2,4,6,8,12,16}"
OTHER_SCHEMES=(
  native_tp1_baseline
  native_tp1_tier
  pd_hetero_baseline
  pd_hetero_tier_biscale
  tier1_ilp_tier2
)

exec > >(tee -a "$ORCH_LOG") 2>&1

cd "$DIR"
export MN_NODE1_IP="${MN_NODE1_IP:-10.252.129.34}"
export MN_NODE2_IP="${MN_NODE2_IP:-10.252.129.33}"

echo "[$(date)] orchestrate_full_qps6 started | QPS=$QPS_LIST"

# --- Phase 1: MegaScale + AFlex (Tier1 per-QPS layout) ---
if pgrep -f run_tier1_megascale_aflex_benchmark.py >/dev/null 2>&1; then
  echo "[$(date)] Phase 1 running — waiting for tier1 MegaScale/AFlex ..."
  while pgrep -f run_tier1_megascale_aflex_benchmark.py >/dev/null 2>&1; do
    sleep 30
  done
  echo "[$(date)] Phase 1 finished"
elif ! ls results/tier1_megascale_aflex_final_*.json >/dev/null 2>&1; then
  echo "[$(date)] Phase 1 not running and no final — starting tier1 MegaScale/AFlex"
  nohup env MN_NODE1_IP="$MN_NODE1_IP" MN_NODE2_IP="$MN_NODE2_IP" \
    python3 -u run_tier1_megascale_aflex_benchmark.py --resume --qps-list "$QPS_LIST" \
    > "$MS_LOG" 2>&1 &
  echo "[$(date)] Phase 1 PID=$!"
  while pgrep -f run_tier1_megascale_aflex_benchmark.py >/dev/null 2>&1; do
    sleep 30
  done
  echo "[$(date)] Phase 1 finished"
else
  echo "[$(date)] Phase 1 already complete (final exists) — skip"
fi

# --- Phase 2: other 5 schemes ---
echo "[$(date)] Phase 2: other schemes ${OTHER_SCHEMES[*]}"
cd "$DIR"
nohup env MN_NODE1_IP="$MN_NODE1_IP" MN_NODE2_IP="$MN_NODE2_IP" \
  python3 -u run_7scheme_6dataset_benchmark.py \
  --resume --order dataset --qps-list "$QPS_LIST" \
  --schemes "${OTHER_SCHEMES[@]}" \
  > "$OTHER_LOG" 2>&1 &
echo "[$(date)] Phase 2 PID=$! log=$OTHER_LOG"

while pgrep -f run_7scheme_6dataset_benchmark.py >/dev/null 2>&1; do
  sleep 30
done

echo "[$(date)] orchestrate_full_qps6 DONE"
