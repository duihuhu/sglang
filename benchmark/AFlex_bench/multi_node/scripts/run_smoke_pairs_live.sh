#!/bin/bash
# Run 6 pairwise 16-card PDAF smokes with aggressive cleanup + live status (5s).
set -u

IB_DEV="${1:-mlx5_bond_0}"
TP="${2:-4}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$DIR/../logs/smoke_pairs_live_$(date +%Y%m%d_%H%M%S).log"
SMOKE_HEALTH_TIMEOUT="${SMOKE_HEALTH_TIMEOUT:-120}"
SMOKE_HEALTH_INTERVAL="${SMOKE_HEALTH_INTERVAL:-2}"
export SMOKE_HEALTH_TIMEOUT SMOKE_HEALTH_INTERVAL

declare -A NODES=([node1]=10.252.129.36 [node2]=10.252.129.35 [node3]=10.252.129.34 [node4]=10.252.129.33)
PAIRS=("node1 node2" "node1 node3" "node1 node4" "node2 node3" "node2 node4" "node3 node4")

node_busy() {
  local ip=$1
  ssh -o BatchMode=yes "$ip" "docker exec operator_test bash -lc '
    ss -tln 2>/dev/null | grep -qE \":4200[0-9]|:4201[0-9]|:4202[0-9]|:49999\"
  '" 2>/dev/null
}

active_nodes() {
  local out=""
  for n in node1 node2 node3 node4; do
    node_busy "${NODES[$n]}" && out+="$n "
  done
  echo "${out:-(none)}"
}

exec > >(tee -a "$LOG") 2>&1
echo "=== smoke_pairs_live $(date) ib=$IB_DEV tp=$TP ==="
echo "Log: $LOG  health=${SMOKE_HEALTH_TIMEOUT}s/${SMOKE_HEALTH_INTERVAL}s"

bash "$DIR/cleanup_all_nodes.sh" || { echo "ABORT: cleanup failed"; exit 1; }

PASS=0; FAIL=0; declare -a FAILED=()

for pair in "${PAIRS[@]}"; do
  read -r P D <<< "$pair"
  PIP="${NODES[$P]}"; DIP="${NODES[$D]}"
  echo ""
  echo "========== $P+$D  prefill=$PIP decode=$DIP  $(date +%H:%M:%S) =========="
  bash "$DIR/cleanup_all_nodes.sh" || { echo "SKIP $P+$D: cleanup failed"; FAIL=$((FAIL+1)); FAILED+=("$P+$D"); continue; }

  t0=$(date +%s)
  PREFILL_IP="$PIP" DECODE_IP="$DIP" bash "$DIR/smoke_pdaf_pair.sh" "$IB_DEV" "$TP" &
  pid=$!
  last=""
  while kill -0 "$pid" 2>/dev/null; do
    act=$(active_nodes)
    line=$(tail -1 "$LOG" 2>/dev/null | tr -d '\r')
    msg="[$(date +%H:%M:%S)] running $P+$D active=[$act] $line"
    [ "$msg" != "$last" ] && echo "$msg"
    last="$msg"
    sleep 5
  done
  wait "$pid"; rc=$?
  dt=$(( $(date +%s) - t0 ))
  bash "$DIR/cleanup_all_nodes.sh" >/dev/null 2>&1 || true
  act=$(active_nodes)
  if [ "$rc" -eq 0 ]; then
    PASS=$((PASS+1)); echo ">>> $P+$D: PASS (${dt}s)  active_after=[$act]"
  else
    FAIL=$((FAIL+1)); FAILED+=("$P+$D"); echo ">>> $P+$D: FAIL (${dt}s)  active_after=[$act]"
  fi
  [ "$act" != "(none)" ] && echo "WARN: stray processes after $P+$D, force cleanup" && bash "$DIR/cleanup_all_nodes.sh"
done

echo ""
echo "=== SUMMARY $(date) ==="
echo "PASS: $PASS / ${#PAIRS[@]}"
echo "FAIL: $FAIL / ${#PAIRS[@]}"
[ "$FAIL" -gt 0 ] && echo "Failed: ${FAILED[*]}" && exit 1
echo "=== ALL 6 PAIRS PASS ==="
