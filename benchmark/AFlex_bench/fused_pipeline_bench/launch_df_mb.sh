#!/bin/bash
# Launch DF (FFN) with configurable micro-batch
export SGLANG_DISABLE_REQUEST_LOGGING=true
export UCX_LOG_LEVEL=fatal
export AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc
export SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=128
export AFD_ASYNC_PIPELINE=1
export AFD_IPC_SYNC_MODE=ipc_event
export AFD_IPC_CPP=1
export AFD_FUSED_PIPELINE=0
export AFD_GPU_ONLY_IPC=0
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export AFD_IPC_PEER_OFFSET=-1
export AFD_NVML_DEVICE_INDICES=1,3,5,7
export AFD_NVML_DEVICE_INDEX=1
export AFD_SCHED_PORT=68500
export AFD_UCX_BASE_PORT=28300

MB=${AFD_MICRO_BATCH:-2}
LOGDIR="/workspace/sglang/benchmark/AFlex_bench/fused_pipeline_bench/logs"
mkdir -p $LOGDIR
MODE=${1:-mb2}

MB_FLAGS="--afd-micro-batch $MB"
if [ "$MB" -gt 1 ]; then
    MB_FLAGS="$MB_FLAGS --afd-dynamic-micro-batch"
fi

exec python3 -m sglang.launch_server --host 10.252.129.34 \
    --port 53200 --afd-perspective ffn \
    --nccl-port 34050 --base-gpu-id 1 \
    --model-path /models/Qwen3-32B --tp 4 --gpu-id-step 2 \
    --afd-comm-backend ipc_cpp \
    $MB_FLAGS --mem-fraction-static 0.85 \
    --max-running-requests 64 --skip-server-warmup \
    --watchdog-timeout 600 \
    --disable-cuda-graph --disable-piecewise-cuda-graph \
    --disable-radix-cache \
    --num-reserved-decode-tokens 512 \
    > $LOGDIR/df_${MODE}.log 2>&1
