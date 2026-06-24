#!/bin/bash
# Cross-node PDAF (4PA + 4PF + 4DA + 4DF) smoke test on 16 GPUs.
#
# Topology:
#   node1 (10.252.129.36) = Prefill side, 8 GPUs, CVD=0..7
#       PF (FFN,  prefill) base_gpu_id=0 -> device 0,1,2,3  (tp=4)
#       PA (Attn, prefill) base_gpu_id=4 -> device 4,5,6,7  (tp=4)
#       PA<->PF talk via intra-node CUDA IPC (ipc_cpp), full NVLink.
#   node2 (10.252.129.35) = Decode side, 8 GPUs, CVD=0..7
#       DF (FFN,  decode)  base_gpu_id=0 -> device 0,1,2,3  (tp=4)
#       DA (Attn, decode)  base_gpu_id=4 -> device 4,5,6,7  (tp=4)
#       DA<->DF talk via intra-node CUDA IPC.
#   Cross-node: P->D KV transfer via mooncake RDMA over mlx5_bond_0.
#               router(node1) -> PA(node1) + DA(node2) over HTTP.
set -u

IB_DEV="${1:-mlx5_bond_0}"
TP="${2:-4}"
NODE1_IP="10.252.129.36"
NODE2_IP="10.252.129.35"
CONTAINER="operator_test"
PYTHON="/usr/bin/python3"
MODEL="/models/Qwen3-32B/"
LOGDIR="/workspace/sglang/benchmark/AFlex_bench/multi_node/logs"
HLOGDIR="/mnt/workspace/lt/sglang/benchmark/AFlex_bench/multi_node/logs"

PA_PORT=42010; PF_PORT=42011
DA_PORT=42020; DF_PORT=42021
ROUTER_PORT=42000
BS_PORT=49999

remote() { ssh -o StrictHostKeyChecking=no -o BatchMode=yes "$NODE2_IP" "$@"; }
dl() { docker exec "$CONTAINER" bash -lc "$1"; }
dr() { remote "docker exec $CONTAINER bash -lc \"$1\""; }

# Robust cleanup via synced script (router renames to sglang::router; kill by port).
CLEANUP="/workspace/sglang/benchmark/AFlex_bench/multi_node/scripts/cleanup_node.sh"
clean_local()  { dl "bash $CLEANUP" >/dev/null 2>&1; }
clean_remote() { dr "bash $CLEANUP" >/dev/null 2>&1; }

COMMON="--model-path $MODEL --tp $TP --afd-comm-backend ipc_cpp \
--afd-micro-batch 2 --afd-dynamic-micro-batch --mem-fraction-static 0.85 \
--max-running-requests 512 --skip-server-warmup \
--disable-cuda-graph --disable-piecewise-cuda-graph \
--afd-disagg-interleave-poll --disable-radix-cache \
--num-reserved-decode-tokens 512 \
--disaggregation-transfer-backend mooncake \
--disaggregation-bootstrap-port $BS_PORT \
--disaggregation-ib-device $IB_DEV --enable-metrics"

# Shared AFD env. FFN_HOST stays 127.0.0.1 because Attn<->FFN is intra-node.
AENV="export SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal \
AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=128 \
AFD_ASYNC_PIPELINE=1 AFD_IPC_SYNC_MODE=ipc_event \
SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600;"

echo "=== [0] cleanup both nodes ==="
clean_local
clean_remote
sleep 5

CVD="0,1,2,3,4,5,6,7"
FFN_NVML="0,1,2,3"
ATTN_NVML="4,5,6,7"

# ---- node1: Prefill side (PF base=0, PA base=tp) ----
echo "=== [1] node1 PF (ffn, prefill, base=0, dev 0-3) ==="
dl "$AENV export CUDA_VISIBLE_DEVICES=$CVD AFD_UCX_BASE_PORT=28200 AFD_SCHED_PORT=68400 \
AFD_IPC_PEER_OFFSET=$TP AFD_NVML_DEVICE_INDICES=$FFN_NVML AFD_NVML_DEVICE_INDEX=0; \
setsid prlimit --memlock=unlimited:unlimited $PYTHON -m sglang.launch_server \
--host $NODE1_IP --port $PF_PORT --afd-perspective ffn --disaggregation-mode prefill \
--base-gpu-id 0 $COMMON > $LOGDIR/pf.log 2>&1 < /dev/null &"
sleep 6
echo "=== [2] node1 PA (attn, prefill, base=$TP, dev 4-7) ==="
dl "$AENV export CUDA_VISIBLE_DEVICES=$CVD AFD_UCX_BASE_PORT=28200 AFD_SCHED_PORT=68400 \
AFD_IPC_PEER_OFFSET=-$TP AFD_NVML_DEVICE_INDICES=$ATTN_NVML AFD_NVML_DEVICE_INDEX=4 \
AFD_UCX_FFN_HOST=127.0.0.1; \
setsid prlimit --memlock=unlimited:unlimited $PYTHON -m sglang.launch_server \
--host $NODE1_IP --port $PA_PORT --afd-perspective attn --disaggregation-mode prefill \
--base-gpu-id $TP $COMMON > $LOGDIR/pa.log 2>&1 < /dev/null &"

# ---- node2: Decode side (DF base=0, DA base=tp) ----
echo "=== [3] node2 DF (ffn, decode, base=0, dev 0-3) ==="
dr "$AENV export CUDA_VISIBLE_DEVICES=$CVD AFD_UCX_BASE_PORT=28300 AFD_SCHED_PORT=68500 \
AFD_IPC_PEER_OFFSET=$TP AFD_NVML_DEVICE_INDICES=$FFN_NVML AFD_NVML_DEVICE_INDEX=0; \
setsid prlimit --memlock=unlimited:unlimited $PYTHON -m sglang.launch_server \
--host $NODE2_IP --port $DF_PORT --afd-perspective ffn --disaggregation-mode decode \
--base-gpu-id 0 $COMMON > $LOGDIR/df.log 2>&1 < /dev/null &"
sleep 8
echo "=== [4] node2 DA (attn, decode, base=$TP, dev 4-7) ==="
dr "$AENV export CUDA_VISIBLE_DEVICES=$CVD AFD_UCX_BASE_PORT=28300 AFD_SCHED_PORT=68500 \
AFD_IPC_PEER_OFFSET=-$TP AFD_NVML_DEVICE_INDICES=$ATTN_NVML AFD_NVML_DEVICE_INDEX=4 \
AFD_UCX_FFN_HOST=127.0.0.1; \
setsid prlimit --memlock=unlimited:unlimited $PYTHON -m sglang.launch_server \
--host $NODE2_IP --port $DA_PORT --afd-perspective attn --disaggregation-mode decode \
--base-gpu-id $TP $COMMON > $LOGDIR/da.log 2>&1 < /dev/null &"

echo "=== [5] wait for health ==="
wh() { # host port [model_info]
  ep="health"; [ "${3:-}" = "1" ] && ep="get_model_info"
  for i in $(seq 1 200); do
    if curl -sf "http://$1:$2/$ep" >/dev/null 2>&1; then echo "  $1:$2 ($ep) ready"; return 0; fi
    sleep 3
  done
  echo "  $1:$2 TIMEOUT"; return 1
}
wh "$NODE1_IP" "$PF_PORT" 1 || { echo "PF failed"; tail -40 $HLOGDIR/pf.log; exit 1; }
wh "$NODE1_IP" "$PA_PORT"   || { echo "PA failed"; tail -40 $HLOGDIR/pa.log; exit 1; }
wh "$NODE2_IP" "$DF_PORT" 1 || { echo "DF failed"; ssh $NODE2_IP "tail -40 $HLOGDIR/df.log"; exit 1; }
wh "$NODE2_IP" "$DA_PORT"   || { echo "DA failed"; ssh $NODE2_IP "tail -40 $HLOGDIR/da.log"; exit 1; }

echo "=== [6] start router on node1 (PA=prefill, DA=decode) ==="
dl "$PYTHON -m sglang_router.launch_router --pd-disaggregation --mini-lb \
--prefill http://$NODE1_IP:$PA_PORT --decode http://$NODE2_IP:$DA_PORT \
--host $NODE1_IP --port $ROUTER_PORT > $LOGDIR/router.log 2>&1 < /dev/null &"
wh "$NODE1_IP" "$ROUTER_PORT" || { echo "router failed"; tail -30 $HLOGDIR/router.log; exit 1; }

echo "=== [7] send test request ==="
RESP=$(curl -s "http://$NODE1_IP:$ROUTER_PORT/generate" \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":16,"temperature":0.0}}')
echo "Response: $RESP"
echo "$RESP" | grep -q '"text"' && echo "=== PDAF SMOKE PASS (ib=$IB_DEV tp=$TP) ===" \
  || { echo "=== PDAF SMOKE FAIL ==="; \
       echo "--- router log ---"; tail -25 $HLOGDIR/router.log; \
       echo "--- PA log ---";     tail -35 $HLOGDIR/pa.log; \
       echo "--- PF log ---";     tail -15 $HLOGDIR/pf.log; \
       echo "--- DA log ---";     ssh $NODE2_IP "tail -35 $HLOGDIR/da.log"; \
       echo "--- DF log ---";     ssh $NODE2_IP "tail -15 $HLOGDIR/df.log"; }

echo "=== [8] cleanup ==="
sleep 2
clean_local
clean_remote
dl "pkill -9 -f sglang.launch_server; pkill -9 -f launch_router; true" >/dev/null 2>&1
dr "pkill -9 -f sglang.launch_server; pkill -9 -f launch_router; true" >/dev/null 2>&1
