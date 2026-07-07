#!/bin/bash
# Aggressive cleanup on all 4 benchmark nodes (run from node1 host).
set -u

NODES=(10.252.129.36 10.252.129.35 10.252.129.34 10.252.129.33)
CONTAINER="operator_test"
INNER="/workspace/sglang/benchmark/AFlex_bench/multi_node/scripts/cleanup_node.sh"

echo "=== cleanup_all_nodes $(date '+%F %T') ==="
for ip in "${NODES[@]}"; do
  echo -n "  $ip: "
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$ip" "docker exec $CONTAINER bash $INNER" >/dev/null 2>&1 || true
  # Extra: kill scheduler/router orphans that cleanup_node may miss
  ssh -o BatchMode=yes "$ip" "docker exec $CONTAINER bash -lc '
    pkill -9 -f sglang.launch_server 2>/dev/null
    pkill -9 -f launch_router 2>/dev/null
    pkill -9 -x sglang::router 2>/dev/null
    pkill -9 -f sglang::schedul 2>/dev/null
    pkill -9 -f sglang_router 2>/dev/null
    for p in \$(seq 42000 42029) 49999 39411 39421 39431 39441; do
      fuser -k \${p}/tcp 2>/dev/null
    done
    true
  '" >/dev/null 2>&1 || true
  busy=$(ssh -o BatchMode=yes "$ip" "docker exec $CONTAINER bash -lc '
    ss -tln 2>/dev/null | grep -E \":4200[0-9]|:4201[0-9]|:4202[0-9]|:49999\" | wc -l
  '" 2>/dev/null | tr -d '[:space:]')
  procs=$(ssh -o BatchMode=yes "$ip" "docker exec $CONTAINER bash -lc '
    n=0
    pgrep -f launch_server >/dev/null 2>&1 && n=1
    pgrep -f launch_router >/dev/null 2>&1 && n=1
    pgrep -x sglang::router >/dev/null 2>&1 && n=1
    pgrep -f sglang::schedul >/dev/null 2>&1 && n=1
    echo \$n
  '" 2>/dev/null | tr -d '[:space:]')
  echo "ports=$busy bench_procs=$procs"
done
sleep 2
echo "=== verify idle ==="
ok=1
for ip in "${NODES[@]}"; do
  r=$(ssh -o BatchMode=yes "$ip" "docker exec $CONTAINER bash -lc '
    ss -tln 2>/dev/null | grep -qE \":4200[0-9]|:4201[0-9]|:4202[0-9]|:49999\" && echo busy || echo idle
  '" 2>/dev/null | tr -d '[:space:]')
  [ "$r" != "idle" ] && ok=0 && echo "  WARN $ip ports still busy"
done
[ "$ok" = "1" ] && echo "  all nodes idle" && exit 0
exit 1
