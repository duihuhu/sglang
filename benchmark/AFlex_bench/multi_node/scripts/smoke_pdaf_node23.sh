#!/bin/bash
# 16-card PDAF smoke: node2 (prefill) + node3 (decode)
set -u

IB_DEV="${1:-mlx5_bond_0}"
TP="${2:-4}"
PREFILL_IP="10.252.129.35"   # node2
DECODE_IP="10.252.129.34"    # node3
CONTAINER="operator_test"
PYTHON="/usr/bin/python3"
MODEL="/models/Qwen3-32B/"
LOGDIR="/workspace/sglang/benchmark/AFlex_bench/multi_node/logs"
HLOGDIR="/mnt/workspace/lt/sglang/benchmark/AFlex_bench/multi_node/logs"

PA_PORT=42010; PF_PORT=42011
DA_PORT=42020; DF_PORT=42021
ROUTER_PORT=42000
BS_PORT=49999

dp() { ssh -o StrictHostKeyChecking=no -o BatchMode=yes "$PREFILL_IP" "$@"; }
dd() { ssh -o StrictHostKeyChecking=no -o BatchMode=yes "$DECODE_IP" "$@"; }
dpp() { dp "docker exec $CONTAINER bash -lc \"$1\""; }
dpd() { dd "docker exec $CONTAINER bash -lc \"$1\""; }

CLEANUP="/workspace/sglang/benchmark/AFlex_bench/multi_node/scripts/cleanup_node.sh"

echo "=== [0] cleanup node2 + node3 ==="
dp "docker exec $CONTAINER bash $CLEANUP" >/dev/null 2>&1
dd "docker exec $CONTAINER bash $CLEANUP" >/dev/null 2>&1
sleep 5

COMMON="--model-path $MODEL --tp $TP --afd-comm-backend ipc_cpp \
--afd-micro-batch 2 --afd-dynamic-micro-batch --mem-fraction-static 0.85 \
--max-running-requests 512 --skip-server-warmup \
--disable-cuda-graph --disable-piecewise-cuda-graph \
--afd-disagg-interleave-poll --disable-radix-cache \
--num-reserved-decode-tokens 512 \
--disaggregation-transfer-backend mooncake \
--disaggregation-bootstrap-port $BS_PORT \
--disaggregation-ib-device $IB_DEV --enable-metrics"

AENV="export SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal \
AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=128 \
AFD_ASYNC_PIPELINE=1 AFD_IPC_SYNC_MODE=ipc_event \
SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600;"

CVD="0,1,2,3,4,5,6,7"
FFN_NVML="0,1,2,3"
ATTN_NVML="4,5,6,7"

echo "=== [1] node2 PF (prefill ffn) ==="
dpp "$AENV export CUDA_VISIBLE_DEVICES=$CVD AFD_UCX_BASE_PORT=28200 AFD_SCHED_PORT=68400 \
AFD_IPC_PEER_OFFSET=$TP AFD_NVML_DEVICE_INDICES=$FFN_NVML AFD_NVML_DEVICE_INDEX=0; \
setsid prlimit --memlock=unlimited:unlimited $PYTHON -m sglang.launch_server \
--host $PREFILL_IP --port $PF_PORT --afd-perspective ffn --disaggregation-mode prefill \
--base-gpu-id 0 $COMMON > $LOGDIR/pf.log 2>&1 < /dev/null &"
sleep 6

echo "=== [2] node2 PA (prefill attn) ==="
dpp "$AENV export CUDA_VISIBLE_DEVICES=$CVD AFD_UCX_BASE_PORT=28200 AFD_SCHED_PORT=68400 \
AFD_IPC_PEER_OFFSET=-$TP AFD_NVML_DEVICE_INDICES=$ATTN_NVML AFD_NVML_DEVICE_INDEX=4 \
AFD_UCX_FFN_HOST=127.0.0.1; \
setsid prlimit --memlock=unlimited:unlimited $PYTHON -m sglang.launch_server \
--host $PREFILL_IP --port $PA_PORT --afd-perspective attn --disaggregation-mode prefill \
--base-gpu-id $TP $COMMON > $LOGDIR/pa.log 2>&1 < /dev/null &"

echo "=== [3] node3 DF (decode ffn) ==="
dpd "$AENV export CUDA_VISIBLE_DEVICES=$CVD AFD_UCX_BASE_PORT=28300 AFD_SCHED_PORT=68500 \
AFD_IPC_PEER_OFFSET=$TP AFD_NVML_DEVICE_INDICES=$FFN_NVML AFD_NVML_DEVICE_INDEX=0; \
setsid prlimit --memlock=unlimited:unlimited $PYTHON -m sglang.launch_server \
--host $DECODE_IP --port $DF_PORT --afd-perspective ffn --disaggregation-mode decode \
--base-gpu-id 0 $COMMON > $LOGDIR/df.log 2>&1 < /dev/null &"
sleep 8

echo "=== [4] node3 DA (decode attn) ==="
dpd "$AENV export CUDA_VISIBLE_DEVICES=$CVD AFD_UCX_BASE_PORT=28300 AFD_SCHED_PORT=68500 \
AFD_IPC_PEER_OFFSET=-$TP AFD_NVML_DEVICE_INDICES=$ATTN_NVML AFD_NVML_DEVICE_INDEX=4 \
AFD_UCX_FFN_HOST=127.0.0.1; \
setsid prlimit --memlock=unlimited:unlimited $PYTHON -m sglang.launch_server \
--host $DECODE_IP --port $DA_PORT --afd-perspective attn --disaggregation-mode decode \
--base-gpu-id $TP $COMMON > $LOGDIR/da.log 2>&1 < /dev/null &"

wh() {
  ep="health"; [ "${3:-}" = "1" ] && ep="get_model_info"
  for i in $(seq 1 200); do
    if curl -sf "http://$1:$2/$ep" >/dev/null 2>&1; then echo "  $1:$2 ($ep) ready"; return 0; fi
    sleep 3
  done
  echo "  $1:$2 TIMEOUT"; return 1
}

echo "=== [5] wait for health ==="
wh "$PREFILL_IP" "$PF_PORT" 1 || { echo "PF failed"; ssh $PREFILL_IP "tail -40 $HLOGDIR/pf.log"; exit 1; }
wh "$PREFILL_IP" "$PA_PORT"   || { echo "PA failed"; ssh $PREFILL_IP "tail -40 $HLOGDIR/pa.log"; exit 1; }
wh "$DECODE_IP" "$DF_PORT" 1  || { echo "DF failed"; ssh $DECODE_IP "tail -40 $HLOGDIR/df.log"; exit 1; }
wh "$DECODE_IP" "$DA_PORT"    || { echo "DA failed"; ssh $DECODE_IP "tail -40 $HLOGDIR/da.log"; exit 1; }

echo "=== [6] router on node2 ==="
dpp "$PYTHON -m sglang_router.launch_router --pd-disaggregation --mini-lb \
--prefill http://$PREFILL_IP:$PA_PORT --decode http://$DECODE_IP:$DA_PORT \
--host $PREFILL_IP --port $ROUTER_PORT > $LOGDIR/router.log 2>&1 < /dev/null &"
wh "$PREFILL_IP" "$ROUTER_PORT" || { echo "router failed"; ssh $PREFILL_IP "tail -30 $HLOGDIR/router.log"; exit 1; }

echo "=== [7] test request ==="
RESP=$(curl -s "http://$PREFILL_IP:$ROUTER_PORT/generate" \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":16,"temperature":0.0}}')
echo "Response: $RESP"
if echo "$RESP" | grep -q '"text"'; then
  echo "=== PDAF NODE23 PASS (prefill=node2 decode=node3 ib=$IB_DEV tp=$TP) ==="
else
  echo "=== PDAF NODE23 FAIL ==="
  ssh $PREFILL_IP "tail -25 $HLOGDIR/router.log; tail -20 $HLOGDIR/pa.log"
  ssh $DECODE_IP "tail -25 $HLOGDIR/da.log; tail -15 $HLOGDIR/df.log"
  exit 1
fi

echo "=== [8] cleanup ==="
sleep 2
dp "docker exec $CONTAINER bash $CLEANUP" >/dev/null 2>&1
dd "docker exec $CONTAINER bash $CLEANUP" >/dev/null 2>&1
