#!/bin/bash
# Cross-node PD (prefill-decode) smoke test.
# Goal: validate that mooncake KV transfer works across the two nodes and
# determine the correct --disaggregation-ib-device.
#
# Topology:
#   node1 (10.252.129.36) GPU0 -> prefill,  host = node1 IP
#   node2 (10.252.129.35) GPU0 -> decode,   host = node2 IP
#   node1 PD router -> prefill(node1) + decode(node2)
#
# Usage:
#   bash smoke_pd_xnode.sh mlx5_bond_0     # try RoCE bond device (default)
#   bash smoke_pd_xnode.sh mlx5_4          # try plain IB device
set -u

IB_DEV="${1:-mlx5_bond_0}"
NODE1_IP="10.252.129.36"
NODE2_IP="10.252.129.35"
CONTAINER="operator_test"
PYTHON="/usr/bin/python3"
MODEL="/models/Qwen3-32B/"

P_PORT=53100
D_PORT=53101
BS_PORT=49100
ROUTER_PORT=42000

remote() { ssh -o StrictHostKeyChecking=no -o BatchMode=yes "$NODE2_IP" "$@"; }
dexec_local() { docker exec "$CONTAINER" bash -lc "$1"; }
dexec_remote() { remote "docker exec $CONTAINER bash -lc \"$1\""; }

echo "=== [0] cleanup both nodes ==="
dexec_local "pkill -9 -f sglang.launch_server; pkill -9 -f launch_router; true" >/dev/null 2>&1
dexec_remote "pkill -9 -f sglang.launch_server; pkill -9 -f launch_router; true" >/dev/null 2>&1
sleep 4

COMMON_ENV="export SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal;"
P_FLAGS="--model-path $MODEL --tp 1 --mem-fraction-static 0.85 \
  --disable-cuda-graph --disable-piecewise-cuda-graph --skip-server-warmup \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port $BS_PORT --disaggregation-ib-device $IB_DEV"

LOGDIR=/workspace/sglang/benchmark/AFlex_bench/multi_node/logs
echo "=== [1] start prefill on node1 GPU0 (host=$NODE1_IP, ib=$IB_DEV) ==="
dexec_local "$COMMON_ENV export CUDA_VISIBLE_DEVICES=0; \
  setsid prlimit --memlock=unlimited:unlimited $PYTHON -m sglang.launch_server \
  --host $NODE1_IP --port $P_PORT --nccl-port 34000 \
  --disaggregation-mode prefill $P_FLAGS \
  > $LOGDIR/smoke_prefill.log 2>&1 < /dev/null &"

echo "=== [2] start decode on node2 GPU0 (host=$NODE2_IP, ib=$IB_DEV) ==="
dexec_remote "$COMMON_ENV export CUDA_VISIBLE_DEVICES=0; \
  setsid prlimit --memlock=unlimited:unlimited $PYTHON -m sglang.launch_server \
  --host $NODE2_IP --port $D_PORT --nccl-port 34001 \
  --disaggregation-mode decode $P_FLAGS \
  > $LOGDIR/smoke_decode.log 2>&1 < /dev/null &"

echo "=== [3] wait for health ==="
wait_health() { # host port
  for i in $(seq 1 100); do
    if curl -sf "http://$1:$2/health" >/dev/null 2>&1; then echo "  $1:$2 healthy"; return 0; fi
    sleep 3
  done
  echo "  $1:$2 TIMEOUT"; return 1
}
wait_health "$NODE1_IP" "$P_PORT" || { echo "prefill failed"; tail -30 /mnt/workspace/lt/sglang/benchmark/AFlex_bench/multi_node/logs/smoke_prefill.log; exit 1; }
wait_health "$NODE2_IP" "$D_PORT" || { echo "decode failed"; exit 1; }

echo "=== [4] start PD router on node1 ==="
dexec_local "$PYTHON -m sglang_router.launch_router --pd-disaggregation --mini-lb \
  --prefill http://$NODE1_IP:$P_PORT $BS_PORT --decode http://$NODE2_IP:$D_PORT \
  --host $NODE1_IP --port $ROUTER_PORT \
  > $LOGDIR/smoke_router.log 2>&1 < /dev/null &"
wait_health "$NODE1_IP" "$ROUTER_PORT" || { echo "router failed"; exit 1; }

echo "=== [5] send test request through router ==="
RESP=$(curl -s "http://$NODE1_IP:$ROUTER_PORT/generate" \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":16,"temperature":0.0}}')
echo "Response: $RESP"
echo "$RESP" | grep -q '"text"' && echo "=== SMOKE PASS (ib=$IB_DEV) ===" || { echo "=== SMOKE FAIL (ib=$IB_DEV) ==="; echo "--- decode log tail ---"; tail -25 /mnt/workspace/lt/sglang/benchmark/AFlex_bench/multi_node/logs/smoke_decode.log; }

echo "=== [6] cleanup ==="
dexec_local "pkill -9 -f sglang.launch_server; pkill -9 -f launch_router; true" >/dev/null 2>&1
dexec_remote "pkill -9 -f sglang.launch_server; pkill -9 -f launch_router; true" >/dev/null 2>&1
