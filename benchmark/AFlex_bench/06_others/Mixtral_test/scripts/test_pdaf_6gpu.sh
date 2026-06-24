#!/bin/bash
# Test PDAF heterogeneous deployment for Mixtral-8x7B
# PA (TP=1, GPU 2) + PF (TP=2, GPU 3,4) + DA (TP=1, GPU 5) + DF (TP=2, GPU 6,7)
# Total: 6 GPUs (2-7), avoids GPU 0,1 (used for profiling)

set -e

PYTHON=/workspace/env/sglang-test/bin/python
MODEL=/models/Mixtral/Mixtral-8x7B/

# Ports
PA_PORT=50010
PF_PORT=50011
DA_PORT=50020
DF_PORT=50021
ROUTER_PORT=50000
BOOTSTRAP_PORT=49999

# Common args
COMMON="--model-path $MODEL \
  --host 127.0.0.1 \
  --afd-comm-backend ipc_cpp \
  --afd-micro-batch 2 \
  --afd-dynamic-micro-batch \
  --mem-fraction-static 0.85 \
  --max-running-requests 512 \
  --skip-server-warmup \
  --disable-cuda-graph --disable-piecewise-cuda-graph \
  --afd-disagg-interleave-poll \
  --disable-radix-cache \
  --num-reserved-decode-tokens 512 \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port $BOOTSTRAP_PORT \
  --disaggregation-ib-device mlx5_4 \
  --enable-metrics"

export AFD_UCX_TLS="rc,tcp,cuda_copy,cuda_ipc"
export SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=128
export AFD_ASYNC_PIPELINE=1
export AFD_IPC_SYNC_MODE=ipc_event
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600

echo "=== Starting PDAF 6-GPU heterogeneous (PA:TP1 + PF:TP2 + DA:TP1 + DF:TP2) ==="

# --- Prefill side: GPU 2,3,4 ---
# PF (FFN, TP=2, base_gpu_id=1 -> uses logical GPU 1,2 = physical GPU 3,4)
echo "[1/4] Starting PF (FFN, TP=2, GPU 3,4)..."
CUDA_VISIBLE_DEVICES=2,3,4 \
AFD_UCX_BASE_PORT=28200 \
AFD_SCHED_PORT=68400 \
AFD_IPC_PEER_DEVICE=0 \
AFD_NVML_DEVICE_INDICES=3,4 \
AFD_NVML_DEVICE_INDEX=3 \
$PYTHON -m sglang.launch_server \
  --port $PF_PORT \
  --tp 2 \
  --base-gpu-id 1 \
  --afd-perspective ffn \
  --disaggregation-mode prefill \
  $COMMON > /workspace/sglang/benchmark/AFlex_bench/06_others/Mixtral_test/logs/pdaf_pf.log 2>&1 &
PF_PID=$!
echo "  PF PID=$PF_PID"
sleep 8

# PA (Attn, TP=1, base_gpu_id=0 -> uses logical GPU 0 = physical GPU 2)
echo "[2/4] Starting PA (Attn, TP=1, GPU 2)..."
CUDA_VISIBLE_DEVICES=2,3,4 \
AFD_UCX_BASE_PORT=28200 \
AFD_SCHED_PORT=68400 \
AFD_IPC_PEER_DEVICE=1 \
AFD_NVML_DEVICE_INDICES=2 \
AFD_NVML_DEVICE_INDEX=2 \
AFD_UCX_FFN_HOST=127.0.0.1 \
$PYTHON -m sglang.launch_server \
  --port $PA_PORT \
  --tp 1 \
  --base-gpu-id 0 \
  --afd-perspective attn \
  --disaggregation-mode prefill \
  $COMMON > /workspace/sglang/benchmark/AFlex_bench/06_others/Mixtral_test/logs/pdaf_pa.log 2>&1 &
PA_PID=$!
echo "  PA PID=$PA_PID"
sleep 8

# --- Decode side: GPU 5,6,7 ---
# DF (FFN, TP=2, base_gpu_id=1 -> uses logical GPU 1,2 = physical GPU 6,7)
echo "[3/4] Starting DF (FFN, TP=2, GPU 6,7)..."
CUDA_VISIBLE_DEVICES=5,6,7 \
AFD_UCX_BASE_PORT=28300 \
AFD_SCHED_PORT=68500 \
AFD_IPC_PEER_DEVICE=0 \
AFD_NVML_DEVICE_INDICES=6,7 \
AFD_NVML_DEVICE_INDEX=6 \
$PYTHON -m sglang.launch_server \
  --port $DF_PORT \
  --tp 2 \
  --base-gpu-id 1 \
  --afd-perspective ffn \
  --disaggregation-mode decode \
  $COMMON > /workspace/sglang/benchmark/AFlex_bench/06_others/Mixtral_test/logs/pdaf_df.log 2>&1 &
DF_PID=$!
echo "  DF PID=$DF_PID"
sleep 10

# DA (Attn, TP=1, base_gpu_id=0 -> uses logical GPU 0 = physical GPU 5)
echo "[4/4] Starting DA (Attn, TP=1, GPU 5)..."
CUDA_VISIBLE_DEVICES=5,6,7 \
AFD_UCX_BASE_PORT=28300 \
AFD_SCHED_PORT=68500 \
AFD_IPC_PEER_DEVICE=1 \
AFD_NVML_DEVICE_INDICES=5 \
AFD_NVML_DEVICE_INDEX=5 \
AFD_UCX_FFN_HOST=127.0.0.1 \
$PYTHON -m sglang.launch_server \
  --port $DA_PORT \
  --tp 1 \
  --base-gpu-id 0 \
  --afd-perspective attn \
  --disaggregation-mode decode \
  $COMMON > /workspace/sglang/benchmark/AFlex_bench/06_others/Mixtral_test/logs/pdaf_da.log 2>&1 &
DA_PID=$!
echo "  DA PID=$DA_PID"

echo ""
echo "All 4 components launched. PIDs: PA=$PA_PID PF=$PF_PID DA=$DA_PID DF=$DF_PID"
echo "Waiting for health checks..."
echo "Monitor: tail -f /workspace/sglang/benchmark/AFlex_bench/06_others/Mixtral_test/logs/pdaf_*.log"
