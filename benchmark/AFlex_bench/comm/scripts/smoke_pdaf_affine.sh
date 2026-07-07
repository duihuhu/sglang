#!/bin/bash
# Cross-node PDAF smoke test with INTERLEAVED GPU layout (NIC-affine).
#
# Topology (affine):
#   PA/DA: base_gpu_id=0, gpu_id_step=2, tp=4 → GPU 0,2,4,6
#          Each rank has PXB-affine NIC: GPU0→mlx5_0, GPU2→mlx5_1, GPU4→mlx5_4, GPU6→mlx5_5
#   PF/DF: base_gpu_id=1, gpu_id_step=2, tp=4 → GPU 1,3,5,7
#          Each rank neighbors its paired PA/DA rank (offset=1)
#   AFD_IPC_PEER_OFFSET = ±1 (PA rank i on GPU 2i, PF rank i on GPU 2i+1)
#
#   node1 (10.252.129.36) = Prefill side (PA + PF)
#   node2 (10.252.129.35) = Decode side (DA + DF)
#   Cross-node: PA→DA KV transfer via mooncake RDMA, using per-GPU NIC affinity.
#
# Usage (on node1 host):
#   bash smoke_pdaf_affine.sh [ib_json_file] [tp]
#   bash smoke_pdaf_affine.sh   # uses default JSON mapping

set -u

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

# NIC affinity JSON: GPU_id -> IB device (PXB relationship from nvidia-smi topo)
IB_JSON='{"0":"mlx5_0","1":"mlx5_0","2":"mlx5_1","3":"mlx5_1","4":"mlx5_4","5":"mlx5_4","6":"mlx5_5","7":"mlx5_5"}'
IB_JSON_FILE="/tmp/ib_affine_map.json"

remote() { ssh -o StrictHostKeyChecking=no -o BatchMode=yes "$NODE2_IP" "$@"; }
dl() { docker exec "$CONTAINER" bash -lc "$1"; }
dr() { remote "docker exec $CONTAINER bash -lc \"$1\""; }

CLEANUP="/workspace/sglang/benchmark/AFlex_bench/multi_node/scripts/cleanup_node.sh"
clean_local()  { dl "bash $CLEANUP" >/dev/null 2>&1; }
clean_remote() { dr "bash $CLEANUP" >/dev/null 2>&1; }

# Interleaved layout: PA uses GPU 0,2,4,6 (step=2); PF uses GPU 1,3,5,7 (step=2)
# IPC offset: PA(even) <-> PF(odd) = offset ±1
PA_BASE=0; PF_BASE=1
GPU_STEP=2
IPC_OFFSET=1  # PF offset = +1, PA offset = -1

# NVML device indices for interleaved layout
PA_NVML="0,2,4,6"
PF_NVML="1,3,5,7"

COMMON="--model-path $MODEL --tp $TP --gpu-id-step $GPU_STEP \
--afd-comm-backend ipc_cpp \
--afd-micro-batch 2 --afd-dynamic-micro-batch --mem-fraction-static 0.85 \
--max-running-requests 512 --skip-server-warmup \
--disable-cuda-graph --disable-piecewise-cuda-graph \
--afd-disagg-interleave-poll --disable-radix-cache \
--num-reserved-decode-tokens 512 \
--disaggregation-transfer-backend mooncake \
--disaggregation-bootstrap-port $BS_PORT \
--disaggregation-ib-device $IB_JSON_FILE --enable-metrics"

AENV="export SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal \
AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=128 \
AFD_ASYNC_PIPELINE=1 AFD_IPC_SYNC_MODE=ipc_event \
SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600 \
SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0;"

echo "=== PDAF Affine Smoke Test (interleaved GPU layout) ==="
echo "  PA/DA GPUs: 0,2,4,6 (step=2, base=0)"
echo "  PF/DF GPUs: 1,3,5,7 (step=2, base=1)"
echo "  IPC offset: ±1"
echo "  IB device mapping: $IB_JSON"
echo ""

echo "=== [0] cleanup both nodes ==="
clean_local
clean_remote
sleep 5

# Write IB JSON mapping file to both containers
echo "$IB_JSON" > /tmp/ib_affine_map.json
docker cp /tmp/ib_affine_map.json "$CONTAINER:$IB_JSON_FILE"
scp -o StrictHostKeyChecking=no -q /tmp/ib_affine_map.json "$NODE2_IP:/tmp/ib_affine_map.json"
remote "docker cp /tmp/ib_affine_map.json $CONTAINER:$IB_JSON_FILE"

CVD="0,1,2,3,4,5,6,7"

# ---- node1: Prefill side ----
echo "=== [1] node1 PF (ffn, prefill, base=1, step=2, GPU 1,3,5,7) ==="
dl "$AENV export CUDA_VISIBLE_DEVICES=$CVD AFD_UCX_BASE_PORT=28200 AFD_SCHED_PORT=68400 \
AFD_IPC_PEER_OFFSET=-$IPC_OFFSET AFD_NVML_DEVICE_INDICES=$PF_NVML AFD_NVML_DEVICE_INDEX=1; \
setsid prlimit --memlock=unlimited:unlimited $PYTHON -m sglang.launch_server \
--host $NODE1_IP --port $PF_PORT --afd-perspective ffn --disaggregation-mode prefill \
--base-gpu-id $PF_BASE $COMMON > $LOGDIR/pf_affine.log 2>&1 < /dev/null &"
sleep 6

echo "=== [2] node1 PA (attn, prefill, base=0, step=2, GPU 0,2,4,6) ==="
dl "$AENV export CUDA_VISIBLE_DEVICES=$CVD AFD_UCX_BASE_PORT=28200 AFD_SCHED_PORT=68400 \
AFD_IPC_PEER_OFFSET=$IPC_OFFSET AFD_NVML_DEVICE_INDICES=$PA_NVML AFD_NVML_DEVICE_INDEX=0 \
AFD_UCX_FFN_HOST=127.0.0.1; \
setsid prlimit --memlock=unlimited:unlimited $PYTHON -m sglang.launch_server \
--host $NODE1_IP --port $PA_PORT --afd-perspective attn --disaggregation-mode prefill \
--base-gpu-id $PA_BASE $COMMON > $LOGDIR/pa_affine.log 2>&1 < /dev/null &"

# ---- node2: Decode side ----
echo "=== [3] node2 DF (ffn, decode, base=1, step=2, GPU 1,3,5,7) ==="
dr "$AENV export CUDA_VISIBLE_DEVICES=$CVD AFD_UCX_BASE_PORT=28300 AFD_SCHED_PORT=68500 \
AFD_IPC_PEER_OFFSET=-$IPC_OFFSET AFD_NVML_DEVICE_INDICES=$PF_NVML AFD_NVML_DEVICE_INDEX=1; \
setsid prlimit --memlock=unlimited:unlimited $PYTHON -m sglang.launch_server \
--host $NODE2_IP --port $DF_PORT --afd-perspective ffn --disaggregation-mode decode \
--base-gpu-id $PF_BASE $COMMON > $LOGDIR/df_affine.log 2>&1 < /dev/null &"
sleep 8

echo "=== [4] node2 DA (attn, decode, base=0, step=2, GPU 0,2,4,6) ==="
dr "$AENV export CUDA_VISIBLE_DEVICES=$CVD AFD_UCX_BASE_PORT=28300 AFD_SCHED_PORT=68500 \
AFD_IPC_PEER_OFFSET=$IPC_OFFSET AFD_NVML_DEVICE_INDICES=$PA_NVML AFD_NVML_DEVICE_INDEX=0 \
AFD_UCX_FFN_HOST=127.0.0.1; \
setsid prlimit --memlock=unlimited:unlimited $PYTHON -m sglang.launch_server \
--host $NODE2_IP --port $DA_PORT --afd-perspective attn --disaggregation-mode decode \
--base-gpu-id $PA_BASE $COMMON > $LOGDIR/da_affine.log 2>&1 < /dev/null &"

echo "=== [5] wait for health ==="
wh() {
  ep="health"; [ "${3:-}" = "1" ] && ep="get_model_info"
  for i in $(seq 1 200); do
    if curl -sf "http://$1:$2/$ep" >/dev/null 2>&1; then echo "  $1:$2 ($ep) ready"; return 0; fi
    sleep 3
  done
  echo "  $1:$2 TIMEOUT"; return 1
}
wh "$NODE1_IP" "$PF_PORT" 1 || { echo "PF failed"; tail -40 $HLOGDIR/pf_affine.log; exit 1; }
wh "$NODE1_IP" "$PA_PORT"   || { echo "PA failed"; tail -40 $HLOGDIR/pa_affine.log; exit 1; }
wh "$NODE2_IP" "$DF_PORT" 1 || { echo "DF failed"; ssh $NODE2_IP "tail -40 $HLOGDIR/df_affine.log"; exit 1; }
wh "$NODE2_IP" "$DA_PORT"   || { echo "DA failed"; ssh $NODE2_IP "tail -40 $HLOGDIR/da_affine.log"; exit 1; }

echo "=== [6] start router ==="
dl "$PYTHON -m sglang_router.launch_router --pd-disaggregation --mini-lb \
--prefill http://$NODE1_IP:$PA_PORT --decode http://$NODE2_IP:$DA_PORT \
--host $NODE1_IP --port $ROUTER_PORT > $LOGDIR/router_affine.log 2>&1 < /dev/null &"
wh "$NODE1_IP" "$ROUTER_PORT" || { echo "router failed"; tail -30 $HLOGDIR/router_affine.log; exit 1; }

echo "=== [7] send test request ==="
RESP=$(curl -s "http://$NODE1_IP:$ROUTER_PORT/generate" \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":16,"temperature":0.0}}')
echo "Response: $RESP"
echo "$RESP" | grep -q '"text"' && echo "=== PDAF AFFINE SMOKE PASS (tp=$TP, interleaved GPU) ===" \
  || { echo "=== PDAF AFFINE SMOKE FAIL ==="; \
       echo "--- router ---"; tail -25 $HLOGDIR/router_affine.log; \
       echo "--- PA ---";     tail -35 $HLOGDIR/pa_affine.log; \
       echo "--- PF ---";     tail -15 $HLOGDIR/pf_affine.log; \
       echo "--- DA ---";     ssh $NODE2_IP "tail -35 $HLOGDIR/da_affine.log"; \
       echo "--- DF ---";     ssh $NODE2_IP "tail -15 $HLOGDIR/df_affine.log"; }

echo "=== [8] cleanup ==="
sleep 2
clean_local
clean_remote
dl "pkill -9 -f sglang.launch_server; pkill -9 -f launch_router; true" >/dev/null 2>&1
dr "pkill -9 -f sglang.launch_server; pkill -9 -f launch_router; true" >/dev/null 2>&1
