#!/bin/bash
# Quick benchmark for pipeline timing breakdown (M=3, 50 requests, short output)
set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${HERE}/throughput_logs"
mkdir -p "${LOG_DIR}"
PYTHON="$(which python)"
MODEL="/models/Qwen/Qwen3-32B/"
BENCHMARK="${HERE}/../benchmark_replay.py"

# Kill existing
pkill -9 -f "sglang.launch_server" 2>/dev/null || true
pkill -9 -f "sglang_router" 2>/dev/null || true
sleep 3

ENV_BASE="SGLANG_DISABLE_REQUEST_LOGGING=true AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc UCX_LOG_LEVEL=fatal"

EXTRA="--skip-server-warmup --disable-cuda-graph --disable-piecewise-cuda-graph --afd-micro-batch 3 --max-running-requests 128 --mem-fraction-static 0.88"

# --- PF (GPU 2) ---
CUDA_VISIBLE_DEVICES=2 AFD_UCX_BASE_PORT=25100 AFD_SCHED_PORT=65300 \
  env ${ENV_BASE} \
  ${PYTHON} -m sglang.launch_server \
    --model-path ${MODEL} --tp 1 \
    --host 127.0.0.1 --port 50011 \
    --afd-perspective ffn --afd-comm-backend ucx \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-bootstrap-port 18999 \
    --disaggregation-ib-device mlx5_4 \
    ${EXTRA} \
    > "${LOG_DIR}/quick_m3_pf.log" 2>&1 &

# --- PA (GPU 3) ---
CUDA_VISIBLE_DEVICES=3 AFD_UCX_BASE_PORT=25100 AFD_SCHED_PORT=65300 AFD_UCX_FFN_HOST=127.0.0.1 \
  env ${ENV_BASE} \
  ${PYTHON} -m sglang.launch_server \
    --model-path ${MODEL} --tp 1 \
    --host 127.0.0.1 --port 50010 \
    --enable-metrics \
    --afd-perspective attn --afd-comm-backend ucx \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-bootstrap-port 18999 \
    --disaggregation-ib-device mlx5_4 \
    ${EXTRA} \
    > "${LOG_DIR}/quick_m3_pa.log" 2>&1 &

echo "Waiting for PA/PF to start..."
for port in 50010 50011; do
  for i in $(seq 1 150); do
    if python -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1',${port})); s.close()" 2>/dev/null; then
      echo "Port ${port} ready"
      break
    fi
    sleep 2
  done
done

# --- DF (GPU 0) ---
CUDA_VISIBLE_DEVICES=0 AFD_UCX_BASE_PORT=25200 AFD_SCHED_PORT=65400 \
  env ${ENV_BASE} \
  ${PYTHON} -m sglang.launch_server \
    --model-path ${MODEL} --tp 1 \
    --host 127.0.0.1 --port 50021 \
    --afd-perspective ffn --afd-comm-backend ucx \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-bootstrap-port 18999 \
    --disaggregation-ib-device mlx5_4 \
    ${EXTRA} \
    > "${LOG_DIR}/quick_m3_df.log" 2>&1 &

# --- DA (GPU 1) ---
CUDA_VISIBLE_DEVICES=1 AFD_UCX_BASE_PORT=25200 AFD_SCHED_PORT=65400 AFD_UCX_FFN_HOST=127.0.0.1 \
  env ${ENV_BASE} \
  ${PYTHON} -m sglang.launch_server \
    --model-path ${MODEL} --tp 1 \
    --host 127.0.0.1 --port 50020 \
    --enable-metrics \
    --afd-perspective attn --afd-comm-backend ucx \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-bootstrap-port 18999 \
    --disaggregation-ib-device mlx5_4 \
    ${EXTRA} \
    > "${LOG_DIR}/quick_m3_da.log" 2>&1 &

echo "Waiting for DA/DF to start..."
for port in 50020 50021; do
  for i in $(seq 1 150); do
    if python -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1',${port})); s.close()" 2>/dev/null; then
      echo "Port ${port} ready"
      break
    fi
    sleep 2
  done
done

# --- Router ---
${PYTHON} -m sglang_router.launch_router \
  --pd-disaggregation --mini-lb \
  --prefill http://127.0.0.1:50010 \
  --decode http://127.0.0.1:50020 \
  --host 127.0.0.1 --port 50001 \
  > "${LOG_DIR}/quick_m3_router.log" 2>&1 &

echo "Waiting for router health..."
for i in $(seq 1 60); do
  if curl -sf http://127.0.0.1:50001/health >/dev/null 2>&1; then
    echo "Router ready"
    break
  fi
  sleep 3
done

# Warmup
echo "Warming up..."
for i in $(seq 0 15); do
  curl -sf -X POST http://127.0.0.1:50001/generate \
    -H "Content-Type: application/json" \
    -d "{\"text\": \"Hello ${i}:\", \"sampling_params\": {\"max_new_tokens\": 16, \"temperature\": 0.0}}" >/dev/null 2>&1 || true
done
sleep 5

# Benchmark
echo "Running benchmark..."
${PYTHON} "${BENCHMARK}" \
  --url http://127.0.0.1:50001 \
  --dataset sample \
  --max-requests 50 \
  --concurrency 128 \
  --sample-input-len 512 \
  --sample-output-len 64 \
  --sample-qps 2 \
  --timeout 600 \
  --dump "${LOG_DIR}/quick_m3_results.json" \
  --scenario-label "quick_m3_qps2"

echo "Done! Results in ${LOG_DIR}/"
