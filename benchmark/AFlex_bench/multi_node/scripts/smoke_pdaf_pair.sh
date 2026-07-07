#!/bin/bash
# 16-card PDAF smoke for arbitrary prefill/decode node pair.
# Usage: PREFILL_IP=... DECODE_IP=... bash smoke_pdaf_pair.sh [IB_DEV] [TP]
#    or: bash smoke_pdaf_pair.sh <PREFILL_IP> <DECODE_IP> [IB_DEV] [TP]
set -u

if [ $# -ge 2 ] && [[ "${1:-}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  PREFILL_IP="$1"
  DECODE_IP="$2"
  IB_DEV="${3:-mlx5_bond_0}"
  TP="${4:-4}"
else
  IB_DEV="${1:-mlx5_bond_0}"
  TP="${2:-4}"
fi

: "${PREFILL_IP:?PREFILL_IP required}"
: "${DECODE_IP:?DECODE_IP required}"

CONTAINER="operator_test"
PYTHON="/usr/bin/python3"
MODEL="/models/Qwen3-32B/"
LOGDIR="/workspace/sglang/benchmark/AFlex_bench/multi_node/logs"
HLOGDIR="/mnt/workspace/lt/sglang/benchmark/AFlex_bench/multi_node/logs"
TAG="${PREFILL_IP##*.}_${DECODE_IP##*.}"
HEALTH_TIMEOUT="${SMOKE_HEALTH_TIMEOUT:-180}"
HEALTH_INTERVAL="${SMOKE_HEALTH_INTERVAL:-2}"

PA_PORT=42010; PF_PORT=42011
DA_PORT=42020; DF_PORT=42021
ROUTER_PORT=42000
BS_PORT=49999

dp() { ssh -o StrictHostKeyChecking=no -o BatchMode=yes "$PREFILL_IP" "$@"; }
dd() { ssh -o StrictHostKeyChecking=no -o BatchMode=yes "$DECODE_IP" "$@"; }
dpp() { dp "docker exec $CONTAINER bash -lc \"$1\""; }
dpd() { dd "docker exec $CONTAINER bash -lc \"$1\""; }

CLEANUP="/workspace/sglang/benchmark/AFlex_bench/multi_node/scripts/cleanup_node.sh"

ports_busy() {
  local ip=$1
  ssh -o BatchMode=yes "$ip" "docker exec $CONTAINER bash -lc '
    for p in 42000 42010 42011 42020 42021 49999; do
      ss -tln sport = :\$p 2>/dev/null | grep -q LISTEN && echo busy:\$p
    done'" 2>/dev/null
}

cleanup_nodes() {
  dp "docker exec $CONTAINER bash $CLEANUP" >/dev/null 2>&1 || true
  dd "docker exec $CONTAINER bash $CLEANUP" >/dev/null 2>&1 || true
  sleep 3
}

# Always cleanup both nodes on exit (success or fail).
trap cleanup_nodes EXIT

wait_svc() {
  local name=$1 host=$2 port=$3 logfile=$4 model_info=${5:-0}
  local ep="health"
  [ "$model_info" = "1" ] && ep="get_model_info"
  local max=$((HEALTH_TIMEOUT / HEALTH_INTERVAL))
  for i in $(seq 1 "$max"); do
    if curl -sf "http://$host:$port/$ep" >/dev/null 2>&1; then
      echo "  [$name] $host:$port ready (${i}x${HEALTH_INTERVAL}s)"
      return 0
    fi
    if ssh -o BatchMode=yes "$host" "test -f $HLOGDIR/$(basename "$logfile") && grep -qE 'EADDRINUSE|Traceback|Fatal error|CUDA out of memory' $HLOGDIR/$(basename "$logfile")" 2>/dev/null; then
      echo "  [$name] $host:$port FAILED (error in log)"
      ssh -o BatchMode=yes "$host" "tail -20 $HLOGDIR/$(basename "$logfile")"
      return 1
    fi
    printf "\r  [%s] waiting %s:%s ... %ds   " "$name" "$host" "$port" "$((i * HEALTH_INTERVAL))"
    sleep "$HEALTH_INTERVAL"
  done
  echo ""
  echo "  [$name] $host:$port TIMEOUT (${HEALTH_TIMEOUT}s)"
  ssh -o BatchMode=yes "$host" "tail -30 $HLOGDIR/$(basename "$logfile")" 2>/dev/null || true
  return 1
}

echo "=== PDAF pair smoke: prefill=$PREFILL_IP decode=$DECODE_IP ib=$IB_DEV tp=$TP ==="

echo "=== [0] cleanup + verify ports ==="
for attempt in 1 2 3; do
  cleanup_nodes
  busy_p=$(ports_busy "$PREFILL_IP")
  busy_d=$(ports_busy "$DECODE_IP")
  if [ -z "$busy_p" ] && [ -z "$busy_d" ]; then
    echo "  ports free on both nodes"
    break
  fi
  echo "  attempt $attempt: busy prefill=[$busy_p] decode=[$busy_d], retry cleanup..."
  [ "$attempt" = "3" ] && { echo "FAIL: ports still busy"; exit 1; }
done

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

echo "=== [1] prefill PF ==="
dpp "$AENV export CUDA_VISIBLE_DEVICES=$CVD AFD_UCX_BASE_PORT=28200 AFD_SCHED_PORT=68400 \
AFD_IPC_PEER_OFFSET=$TP AFD_NVML_DEVICE_INDICES=$FFN_NVML AFD_NVML_DEVICE_INDEX=0; \
setsid prlimit --memlock=unlimited:unlimited $PYTHON -m sglang.launch_server \
--host $PREFILL_IP --port $PF_PORT --afd-perspective ffn --disaggregation-mode prefill \
--base-gpu-id 0 $COMMON > $LOGDIR/pf_${TAG}.log 2>&1 < /dev/null &"
wait_svc PF "$PREFILL_IP" "$PF_PORT" "pf_${TAG}.log" 1 || exit 1

echo "=== [2] prefill PA ==="
dpp "$AENV export CUDA_VISIBLE_DEVICES=$CVD AFD_UCX_BASE_PORT=28200 AFD_SCHED_PORT=68400 \
AFD_IPC_PEER_OFFSET=-$TP AFD_NVML_DEVICE_INDICES=$ATTN_NVML AFD_NVML_DEVICE_INDEX=4 \
AFD_UCX_FFN_HOST=127.0.0.1; \
setsid prlimit --memlock=unlimited:unlimited $PYTHON -m sglang.launch_server \
--host $PREFILL_IP --port $PA_PORT --afd-perspective attn --disaggregation-mode prefill \
--base-gpu-id $TP $COMMON > $LOGDIR/pa_${TAG}.log 2>&1 < /dev/null &"
wait_svc PA "$PREFILL_IP" "$PA_PORT" "pa_${TAG}.log" 0 || exit 1

echo "=== [3] decode DF ==="
dpd "$AENV export CUDA_VISIBLE_DEVICES=$CVD AFD_UCX_BASE_PORT=28300 AFD_SCHED_PORT=68500 \
AFD_IPC_PEER_OFFSET=$TP AFD_NVML_DEVICE_INDICES=$FFN_NVML AFD_NVML_DEVICE_INDEX=0; \
setsid prlimit --memlock=unlimited:unlimited $PYTHON -m sglang.launch_server \
--host $DECODE_IP --port $DF_PORT --afd-perspective ffn --disaggregation-mode decode \
--base-gpu-id 0 $COMMON > $LOGDIR/df_${TAG}.log 2>&1 < /dev/null &"
wait_svc DF "$DECODE_IP" "$DF_PORT" "df_${TAG}.log" 1 || exit 1

echo "=== [4] decode DA ==="
dpd "$AENV export CUDA_VISIBLE_DEVICES=$CVD AFD_UCX_BASE_PORT=28300 AFD_SCHED_PORT=68500 \
AFD_IPC_PEER_OFFSET=-$TP AFD_NVML_DEVICE_INDICES=$ATTN_NVML AFD_NVML_DEVICE_INDEX=4 \
AFD_UCX_FFN_HOST=127.0.0.1; \
setsid prlimit --memlock=unlimited:unlimited $PYTHON -m sglang.launch_server \
--host $DECODE_IP --port $DA_PORT --afd-perspective attn --disaggregation-mode decode \
--base-gpu-id $TP $COMMON > $LOGDIR/da_${TAG}.log 2>&1 < /dev/null &"
wait_svc DA "$DECODE_IP" "$DA_PORT" "da_${TAG}.log" 0 || exit 1

echo "=== [5] router on prefill node ==="
dpp "$PYTHON -m sglang_router.launch_router --pd-disaggregation --mini-lb \
--prefill http://$PREFILL_IP:$PA_PORT --decode http://$DECODE_IP:$DA_PORT \
--host $PREFILL_IP --port $ROUTER_PORT > $LOGDIR/router_${TAG}.log 2>&1 < /dev/null &"
wait_svc ROUTER "$PREFILL_IP" "$ROUTER_PORT" "router_${TAG}.log" 0 || exit 1

echo "=== [6] test request ==="
RESP=$(curl -s --max-time 60 "http://$PREFILL_IP:$ROUTER_PORT/generate" \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":16,"temperature":0.0}}')
echo "Response: $RESP"
if echo "$RESP" | grep -q '"text"'; then
  echo "=== PDAF PAIR PASS (prefill=$PREFILL_IP decode=$DECODE_IP ib=$IB_DEV tp=$TP) ==="
else
  echo "=== PDAF PAIR FAIL (prefill=$PREFILL_IP decode=$DECODE_IP) ==="
  ssh "$PREFILL_IP" "tail -25 $HLOGDIR/router_${TAG}.log; tail -20 $HLOGDIR/pa_${TAG}.log"
  ssh "$DECODE_IP" "tail -25 $HLOGDIR/da_${TAG}.log; tail -15 $HLOGDIR/df_${TAG}.log"
  exit 1
fi

# trap EXIT handles cleanup
