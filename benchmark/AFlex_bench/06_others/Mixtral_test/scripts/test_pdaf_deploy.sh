#!/bin/bash
# Test PDAF IPC deployment for Mixtral-8x22B
# Layout: PF(TP=4,GPU0-3) + PA(TP=4,GPU0-3) | DF(TP=4,GPU4-7) + DA(TP=4,GPU4-7)
set -e

PYTHON=/workspace/env/sglang-tier/bin/python
MODEL=/models/Mixtral/
LOG_DIR=/workspace/sglang-tier/benchmark/AFlex_bench/06_others/Mixtral_test/logs

export SGLANG_DISABLE_REQUEST_LOGGING=true
export UCX_LOG_LEVEL=fatal
export AFD_UCX_TLS="rc,tcp,cuda_copy,cuda_ipc"
export SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=128
export AFD_ASYNC_PIPELINE=1

echo "=== Starting PDAF 4P+4D for Mixtral-8x22B ==="

# PF (FFN, TP=4, Prefill, GPU0-3)
echo "[1/4] Starting PF (FFN, Prefill)..."
CUDA_VISIBLE_DEVICES=0,1,2,3 \
AFD_UCX_BASE_PORT=28200 AFD_SCHED_PORT=68400 \
AFD_IPC_SYNC_MODE=ipc_event AFD_IPC_PEER_DEVICE=0 \
AFD_NVML_DEVICE_INDICES="0,1,2,3" AFD_NVML_DEVICE_INDEX=0 \
SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600 \
SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600 \
$PYTHON -m sglang.launch_server \
    --model-path $MODEL --tp 4 \
    --host 127.0.0.1 --port 42011 \
    --afd-perspective ffn --disaggregation-mode prefill \
    --base-gpu-id 0 --afd-comm-backend ipc_cpp \
    --afd-micro-batch 2 --afd-dynamic-micro-batch \
    --mem-fraction-static 0.85 --max-running-requests 512 \
    --skip-server-warmup --disable-cuda-graph --disable-piecewise-cuda-graph \
    --afd-disagg-interleave-poll --disable-radix-cache \
    --num-reserved-decode-tokens 512 \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-bootstrap-port 49999 \
    --disaggregation-ib-device mlx5_4 \
    --enable-return-routed-experts --enable-metrics \
    > $LOG_DIR/pdaf_pf.log 2>&1 &
PF_PID=$!
sleep 5

# PA (Attn, TP=4, Prefill, GPU0-3, shares with PF)
echo "[2/4] Starting PA (Attn, Prefill)..."
CUDA_VISIBLE_DEVICES=0,1,2,3 \
AFD_UCX_BASE_PORT=28200 AFD_SCHED_PORT=68400 \
AFD_IPC_SYNC_MODE=ipc_event AFD_IPC_PEER_DEVICE=0 \
AFD_NVML_DEVICE_INDICES="0,1,2,3" AFD_NVML_DEVICE_INDEX=0 \
AFD_UCX_FFN_HOST=127.0.0.1 \
SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600 \
SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600 \
$PYTHON -m sglang.launch_server \
    --model-path $MODEL --tp 4 \
    --host 127.0.0.1 --port 42010 \
    --afd-perspective attn --disaggregation-mode prefill \
    --base-gpu-id 0 --afd-comm-backend ipc_cpp \
    --afd-micro-batch 2 --afd-dynamic-micro-batch \
    --mem-fraction-static 0.85 --max-running-requests 512 \
    --skip-server-warmup --disable-cuda-graph --disable-piecewise-cuda-graph \
    --afd-disagg-interleave-poll --disable-radix-cache \
    --num-reserved-decode-tokens 512 \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-bootstrap-port 49999 \
    --disaggregation-ib-device mlx5_4 \
    --enable-return-routed-experts --enable-metrics \
    > $LOG_DIR/pdaf_pa.log 2>&1 &
PA_PID=$!
sleep 5

# DF (FFN, TP=4, Decode, GPU4-7)
echo "[3/4] Starting DF (FFN, Decode)..."
CUDA_VISIBLE_DEVICES=4,5,6,7 \
AFD_UCX_BASE_PORT=28300 AFD_SCHED_PORT=68500 \
AFD_IPC_SYNC_MODE=ipc_event AFD_IPC_PEER_DEVICE=0 \
AFD_NVML_DEVICE_INDICES="4,5,6,7" AFD_NVML_DEVICE_INDEX=4 \
SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600 \
SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600 \
$PYTHON -m sglang.launch_server \
    --model-path $MODEL --tp 4 \
    --host 127.0.0.1 --port 42021 \
    --afd-perspective ffn --disaggregation-mode decode \
    --base-gpu-id 0 --afd-comm-backend ipc_cpp \
    --afd-micro-batch 2 --afd-dynamic-micro-batch \
    --mem-fraction-static 0.85 --max-running-requests 512 \
    --skip-server-warmup --disable-cuda-graph --disable-piecewise-cuda-graph \
    --afd-disagg-interleave-poll --disable-radix-cache \
    --num-reserved-decode-tokens 512 \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-bootstrap-port 49999 \
    --disaggregation-ib-device mlx5_4 \
    --enable-return-routed-experts --enable-metrics \
    > $LOG_DIR/pdaf_df.log 2>&1 &
DF_PID=$!
sleep 8

# DA (Attn, TP=4, Decode, GPU4-7, shares with DF)
echo "[4/4] Starting DA (Attn, Decode)..."
CUDA_VISIBLE_DEVICES=4,5,6,7 \
AFD_UCX_BASE_PORT=28300 AFD_SCHED_PORT=68500 \
AFD_IPC_SYNC_MODE=ipc_event AFD_IPC_PEER_DEVICE=0 \
AFD_NVML_DEVICE_INDICES="4,5,6,7" AFD_NVML_DEVICE_INDEX=4 \
AFD_UCX_FFN_HOST=127.0.0.1 \
SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600 \
SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600 \
$PYTHON -m sglang.launch_server \
    --model-path $MODEL --tp 4 \
    --host 127.0.0.1 --port 42020 \
    --afd-perspective attn --disaggregation-mode decode \
    --base-gpu-id 0 --afd-comm-backend ipc_cpp \
    --afd-micro-batch 2 --afd-dynamic-micro-batch \
    --mem-fraction-static 0.85 --max-running-requests 512 \
    --skip-server-warmup --disable-cuda-graph --disable-piecewise-cuda-graph \
    --afd-disagg-interleave-poll --disable-radix-cache \
    --num-reserved-decode-tokens 512 \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-bootstrap-port 49999 \
    --disaggregation-ib-device mlx5_4 \
    --enable-return-routed-experts --enable-metrics \
    > $LOG_DIR/pdaf_da.log 2>&1 &
DA_PID=$!

echo "All 4 components launched. PIDs: PF=$PF_PID PA=$PA_PID DF=$DF_PID DA=$DA_PID"
echo "Waiting for servers to be ready..."

# Wait for PA health (Attn perspective uses /health)
for i in $(seq 1 60); do
    if curl -s http://127.0.0.1:42010/health 2>/dev/null | grep -q "200\|ok"; then
        echo "PA ready after ${i}x5s"
        break
    fi
    if ! kill -0 $PA_PID 2>/dev/null; then
        echo "PA DIED!"
        tail -10 $LOG_DIR/pdaf_pa.log
        exit 1
    fi
    sleep 5
done

# Wait for DA health
for i in $(seq 1 60); do
    if curl -s http://127.0.0.1:42020/health 2>/dev/null | grep -q "200\|ok"; then
        echo "DA ready after ${i}x5s"
        break
    fi
    if ! kill -0 $DA_PID 2>/dev/null; then
        echo "DA DIED!"
        tail -10 $LOG_DIR/pdaf_da.log
        exit 1
    fi
    sleep 5
done

# Check PF/DF via /get_model_info
for port in 42011 42021; do
    for i in $(seq 1 10); do
        if curl -s http://127.0.0.1:$port/get_model_info 2>/dev/null | grep -q "model"; then
            echo "Port $port (FFN) ready"
            break
        fi
        sleep 3
    done
done

# Start router
echo "Starting router..."
$PYTHON -m sglang_router.launch_router \
    --pd-disaggregation --mini-lb \
    --prefill http://127.0.0.1:42010 \
    --decode http://127.0.0.1:42020 \
    --host 127.0.0.1 --port 42000 \
    > $LOG_DIR/pdaf_router.log 2>&1 &
ROUTER_PID=$!
sleep 5

# Test
echo "Testing generation..."
curl -s http://127.0.0.1:42000/generate \
  -H "Content-Type: application/json" \
  -d '{"text":"Explain MoE:","sampling_params":{"max_new_tokens":20,"temperature":0.0}}' | head -c 300
echo ""
echo "=== PDAF deployment test complete ==="
