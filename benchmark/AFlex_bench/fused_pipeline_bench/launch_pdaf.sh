#!/bin/bash
# Launch PDAF baseline (AFD_FUSED_PIPELINE=0) on node3
# DA on GPU 0,2,4,6; DF on GPU 1,3,5,7
set -e

export SGLANG_DISABLE_REQUEST_LOGGING=true
export UCX_LOG_LEVEL=fatal
export AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc
export SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=128
export AFD_ASYNC_PIPELINE=1
export AFD_IPC_SYNC_MODE=ipc_event
export AFD_IPC_CPP=1
export AFD_FUSED_PIPELINE=${AFD_FUSED_PIPELINE:-0}
export AFD_GPU_ONLY_IPC=0
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL="/models/Qwen3-32B"
LOGDIR="/workspace/sglang/benchmark/AFlex_bench/fused_pipeline_bench/logs"
mkdir -p $LOGDIR

COMMON="--model-path $MODEL --tp 4 --gpu-id-step 2 \
    --afd-comm-backend ipc_cpp \
    --afd-micro-batch 1 --mem-fraction-static 0.85 \
    --max-running-requests 64 --skip-server-warmup \
    --watchdog-timeout 600 \
    --disable-cuda-graph --disable-piecewise-cuda-graph \
    --disable-radix-cache \
    --num-reserved-decode-tokens 512"

MODE=${1:-baseline}  # "baseline" or "fused"

echo "=== Launching PDAF ($MODE, AFD_FUSED_PIPELINE=$AFD_FUSED_PIPELINE) ==="

# Kill existing
pkill -9 -f sglang 2>/dev/null || true
sleep 3

# DF (FFN) first
export AFD_IPC_PEER_OFFSET=-1
export AFD_NVML_DEVICE_INDICES=1,3,5,7
export AFD_NVML_DEVICE_INDEX=1
export AFD_SCHED_PORT=68500
export AFD_UCX_BASE_PORT=28300

python3 -m sglang.launch_server --host 10.252.129.34 \
    --port 53200 --afd-perspective ffn \
    --nccl-port 34050 --base-gpu-id 1 \
    $COMMON > $LOGDIR/df_${MODE}.log 2>&1 &
DF_PID=$!
echo "DF PID=$DF_PID"
sleep 8

# DA (ATTN)
export AFD_IPC_PEER_OFFSET=1
export AFD_NVML_DEVICE_INDICES=0,2,4,6
export AFD_NVML_DEVICE_INDEX=0
export AFD_UCX_FFN_HOST=127.0.0.1

python3 -m sglang.launch_server --host 10.252.129.34 \
    --port 53100 --afd-perspective attn \
    --nccl-port 34000 --base-gpu-id 0 \
    $COMMON > $LOGDIR/da_${MODE}.log 2>&1 &
DA_PID=$!
echo "DA PID=$DA_PID"

echo "Waiting for DA health..."
for i in $(seq 1 80); do
    if curl -sf http://10.252.129.34:53100/health > /dev/null 2>&1; then
        echo "DA ready! (after ${i}x5s)"
        exit 0
    fi
    sleep 5
done
echo "DA failed to start!"
tail -20 $LOGDIR/da_${MODE}.log
exit 1
