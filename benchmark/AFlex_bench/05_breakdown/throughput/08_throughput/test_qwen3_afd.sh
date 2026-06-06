#!/usr/bin/env bash
# Quick test: start AF分离 with Qwen3-32B and send one request
set -e

MODEL="/models/Qwen/Qwen3-32B/"
LOG_DIR="/workspace/sglang/af_launch_logs/logs/test/qwen3_af"
PYTHON="/workspace/env/af-test/bin/python"

mkdir -p "$LOG_DIR"

cleanup() {
    echo "=== Cleaning up ==="
    pkill -9 -f "sglang.*afd-perspective" 2>/dev/null || true
    pkill -9 -f ucx 2>/dev/null || true
    sleep 2
}

# Kill any existing servers
cleanup

# Start FFN server (GPU 2, tp=1)
echo "=== Starting FFN server (GPU 2) ==="
CUDA_VISIBLE_DEVICES=2 \
AFD_UCX_BASE_PORT=25000 \
AFD_SCHED_PORT=65300 \
AFD_NVML_DEVICE_INDEX=2 \
UCX_LOG_LEVEL=fatal \
UCX_WARN_UNUSED_ENV_VARS=n \
AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc \
nohup $PYTHON -m sglang.launch_server \
  --model-path $MODEL --tp 1 \
  --host 127.0.0.1 --port 50011 \
  --afd-perspective ffn --afd-comm-backend ucx \
  --mem-fraction-static 0.7 \
  --skip-server-warmup --afd-micro-batch 1 \
  --disable-cuda-graph --disable-piecewise-cuda-graph \
  > "$LOG_DIR/ffn.log" 2>&1 &
FFN_PID=$!
echo "FFN PID=$FFN_PID"

# Wait for FFN to be ready (port 50011)
echo "=== Waiting for FFN port 50011 ==="
for i in $(seq 1 120); do
    if python3 -c "import socket; s=socket.create_connection(('127.0.0.1',50011), timeout=1); s.close(); print('ready')" 2>/dev/null; then
        echo "FFN ready after ${i}s"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "FFN FAILED to start! Log:"
        tail -20 "$LOG_DIR/ffn.log"
        cleanup
        exit 1
    fi
    sleep 1
done

# Start Attn server (GPU 3, tp=1)
echo "=== Starting Attn server (GPU 3) ==="
CUDA_VISIBLE_DEVICES=3 \
AFD_UCX_BASE_PORT=25000 \
AFD_SCHED_PORT=65300 \
AFD_NVML_DEVICE_INDEX=3 \
UCX_LOG_LEVEL=fatal \
UCX_WARN_UNUSED_ENV_VARS=n \
AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc \
SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=false \
nohup $PYTHON -m sglang.launch_server \
  --model-path $MODEL --tp 1 \
  --host 127.0.0.1 --port 50010 \
  --afd-perspective attn --afd-comm-backend ucx \
  --mem-fraction-static 0.7 \
  --enable-metrics \
  --skip-server-warmup --afd-micro-batch 1 \
  --disable-cuda-graph --disable-piecewise-cuda-graph \
  > "$LOG_DIR/attn.log" 2>&1 &
ATTN_PID=$!
echo "Attn PID=$ATTN_PID"

# Wait for Attn to be ready (port 50010)
echo "=== Waiting for Attn port 50010 ==="
for i in $(seq 1 120); do
    if python3 -c "import socket; s=socket.create_connection(('127.0.0.1',50010), timeout=1); s.close(); print('ready')" 2>/dev/null; then
        echo "Attn ready after ${i}s"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "Attn FAILED to start! Log:"
        tail -20 "$LOG_DIR/attn.log"
        cleanup
        exit 1
    fi
    sleep 1
done

# Wait a bit for UCX connection
sleep 15

# Check both logs for errors
echo "=== Checking for errors ==="
grep -i "error\|exception\|traceback" "$LOG_DIR/ffn.log" || echo "FFN: no errors"
grep -i "error\|exception\|traceback" "$LOG_DIR/attn.log" || echo "Attn: no errors"

# Send test request
echo "=== Sending test request ==="
curl -s http://127.0.0.1:50010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello, what is 1+1?"}],
    "max_tokens": 10
  }' 2>&1

echo ""
echo "=== Test complete ==="

# Check for errors after request
sleep 5
grep -i "error\|exception\|traceback" "$LOG_DIR/ffn.log" || echo "FFN post-request: no errors"
grep -i "error\|exception\|traceback" "$LOG_DIR/attn.log" || echo "Attn post-request: no errors"

# Cleanup
cleanup
echo "=== Done ==="
