#!/bin/bash
# Run 16-card PDAF smoke for all node1-4 pairs (6 combinations).
set -u

IB_DEV="${1:-mlx5_bond_0}"
TP="${2:-4}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEANUP="/mnt/workspace/lt/sglang/benchmark/AFlex_bench/multi_node/scripts/cleanup_node.sh"
LOG="/mnt/workspace/lt/sglang/benchmark/AFlex_bench/multi_node/logs/smoke_all_pairs_$(date +%Y%m%d_%H%M%S).log"

declare -A NODES=(
  [node1]=10.252.129.36
  [node2]=10.252.129.35
  [node3]=10.252.129.34
  [node4]=10.252.129.33
)

PAIRS=(
  "node1 node2"
  "node1 node3"
  "node1 node4"
  "node2 node3"
  "node2 node4"
  "node3 node4"
)

exec > >(tee -a "$LOG") 2>&1
echo "=== smoke_pdaf_all_pairs start $(date) ib=$IB_DEV tp=$TP ==="
echo "Log: $LOG"
echo "Health: timeout=${SMOKE_HEALTH_TIMEOUT:-180}s interval=${SMOKE_HEALTH_INTERVAL:-2}s"
echo "Policy: strictly 2 nodes active per pair; cleanup ALL 4 nodes before each pair"

cleanup_all() {
  for n in node1 node2 node3 node4; do
    ip="${NODES[$n]}"
    ssh -o BatchMode=yes "$ip" "docker exec operator_test bash $CLEANUP" >/dev/null 2>&1 || true
  done
  sleep 3
}

verify_all_idle() {
  local bad=0
  for n in node1 node2 node3 node4; do
    ip="${NODES[$n]}"
    running=$(ssh -o BatchMode=yes "$ip" \
      'docker exec operator_test bash -lc "pgrep launch_server >/dev/null 2>&1 && echo yes || echo no"' 2>/dev/null | tr -d '[:space:]')
    if [ "$running" = "yes" ]; then
      echo "  $n ($ip): launch_server still running"
      bad=1
    fi
  done
  [ "$bad" = "0" ] && echo "  all 4 nodes idle" && return 0
  return 1
}

PASS=0
FAIL=0
declare -a FAILED

for pair in "${PAIRS[@]}"; do
  read -r P D <<< "$pair"
  PIP="${NODES[$P]}"
  DIP="${NODES[$D]}"
  echo ""
  echo "########################################"
  echo "# Pair: $P (prefill) + $D (decode)  $(date +%H:%M:%S)"
  echo "########################################"
  echo "=== cleanup all 4 nodes before pair ==="
  cleanup_all
  verify_all_idle || { echo "FAIL: stray processes before $P+$D"; exit 1; }
  t0=$(date +%s)
  if PREFILL_IP="$PIP" DECODE_IP="$DIP" bash "$DIR/smoke_pdaf_pair.sh" "$IB_DEV" "$TP"; then
    PASS=$((PASS + 1))
    echo ">>> $P+$D: PASS ($(( $(date +%s) - t0 ))s)"
  else
    FAIL=$((FAIL + 1))
    FAILED+=("$P+$D")
    echo ">>> $P+$D: FAIL ($(( $(date +%s) - t0 ))s)"
  fi
  echo "=== cleanup all 4 nodes after pair ==="
  cleanup_all
  verify_all_idle || echo "WARN: cleanup incomplete after $P+$D"
  sleep 3
done

echo ""
echo "=== SUMMARY $(date) ==="
echo "PASS: $PASS / ${#PAIRS[@]}"
echo "FAIL: $FAIL / ${#PAIRS[@]}"
if [ "$FAIL" -gt 0 ]; then
  echo "Failed pairs: ${FAILED[*]}"
  exit 1
fi
echo "=== ALL PAIRS PASS ==="
