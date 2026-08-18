#!/usr/bin/env bash
# Run one clean, continuous FlowServe LooGLE replay through Central I/O.
set -euo pipefail

CONTAINER="${CONTAINER:-latticekv-scheduler-runtime-20260729}"
BENCH="/workspace/bench/official_sglang_flowserve_calibration"
RUN="/workspace/bench/run_dynamic_600gb"
PRESSURE_TICK_S="${PRESSURE_TICK_S:-0.25}"
PRESSURE_MAX_TRANSFER_GIB="${PRESSURE_MAX_TRANSFER_GIB:-50}"

HOT_MODEL_ID="${HOT_MODEL_ID:-hot-qwen3-8b}"
HOT_MODEL_PATH="${HOT_MODEL_PATH:-/models/Qwen3-8B}"
HOT_SERVED_MODEL="${HOT_SERVED_MODEL:-Qwen3-8B}"
COLD_B_MODEL_ID="${COLD_B_MODEL_ID:-cold-qwen3-14b}"
COLD_B_MODEL_PATH="${COLD_B_MODEL_PATH:-/models/Qwen3-14B}"
COLD_B_SERVED_MODEL="${COLD_B_SERVED_MODEL:-Qwen3-14B}"
COLD_C_MODEL_ID="${COLD_C_MODEL_ID:-cold-llama3-8b}"
COLD_C_MODEL_PATH="${COLD_C_MODEL_PATH:-/models/Llama-3.1-8B-Instruct}"
COLD_C_SERVED_MODEL="${COLD_C_SERVED_MODEL:-Llama-3.1-8B-Instruct}"
# SGLang accepts this environment value in binary GiB, while the validated
# sweep point is --hicache-size=200 decimal GB. Use the exact conversion so
# the run begins at the intended static 200/200/200 point rather than asking
# Central I/O for a meaningless tail grow during startup.
INITIAL_QUOTA_GIB="${INITIAL_QUOTA_GIB:-186.2645149230957}"

exec_in() {
  docker exec "$CONTAINER" bash -lc "$1"
}

wait_for() {
  local name="$1"
  local command="$2"
  local limit="${3:-900}"
  local started
  started=$(date +%s)
  while ! exec_in "$command" >/dev/null 2>&1; do
    if (( $(date +%s) - started > limit )); then
      echo "timed out waiting for ${name}" >&2
      exit 1
    fi
    sleep 2
  done
  echo "ready: ${name}"
}

# Do not accidentally replay with the old scheduled-time-only hook.
exec_in "grep -q admission_time_s ${BENCH}/test_type/open_loop_sglang.py"
wait_for "Central agent" "test -S ${RUN}/central.sock" 900

start_server() {
  local gpu="$1" model_id="$2" model_path="$3" served="$4" port="$5" log="$6"
  docker exec -d "$CONTAINER" bash -lc \
    "CUDA_VISIBLE_DEVICES=${gpu} \
     SGLANG_CENTRAL_IO_SOCKET=${RUN}/central.sock \
     SGLANG_CENTRAL_IO_MODEL_ID=${model_id} \
     SGLANG_CENTRAL_IO_INITIAL_GIB=${INITIAL_QUOTA_GIB} \
     SGLANG_CENTRAL_IO_LEASE_MODE=page \
     SGLANG_CENTRAL_IO_QUOTA_BATCH_GIB=50 \
     SGLANG_CENTRAL_IO_AGENT_DEVICE_ID=${gpu} \
     SGLANG_DEBUG_REQ_TO_TOKEN_DEVICE=1 \
     PYTHONPATH=/workspace/sglang/python \
     python3 -m sglang.launch_server \
       --model-path ${model_path} --served-model-name ${served} \
       --host 127.0.0.1 --port ${port} --base-gpu-id 0 --trust-remote-code \
       --context-length 40960 --mem-fraction-static 0.8 \
       --enable-metrics --enable-cache-report --enable-hierarchical-cache \
       --hicache-size 300 --hicache-write-policy write_through \
       --hicache-io-backend kernel --hicache-mem-layout page_first \
       --disable-hicache-numa-detect > ${RUN}/${log} 2>&1"
}

start_server 0 "$HOT_MODEL_ID" "$HOT_MODEL_PATH" "$HOT_SERVED_MODEL" 33100 hot.server.log
start_server 1 "$COLD_B_MODEL_ID" "$COLD_B_MODEL_PATH" "$COLD_B_SERVED_MODEL" 33101 cold14.server.log
start_server 2 "$COLD_C_MODEL_ID" "$COLD_C_MODEL_PATH" "$COLD_C_SERVED_MODEL" 33102 coldllama.server.log
wait_for "hot server" "curl -sf http://127.0.0.1:33100/health" 900
wait_for "cold14 server" "curl -sf http://127.0.0.1:33101/health" 900
wait_for "coldllama server" "curl -sf http://127.0.0.1:33102/health" 900

exec_in "rm -rf ${RUN}/barrier; mkdir -p ${RUN}/barrier; rm -f ${RUN}/*.events.jsonl ${RUN}/*.jsonl ${RUN}/*.summary.json ${RUN}/*.client.log ${RUN}/scheduler.log ${RUN}/pressure-scheduler.log"

start_client() {
  local endpoint="$1" served="$2" tokenizer="$3" rate="$4" requests="$5" participant="$6" model_id="$7" output="$8"
  docker exec -d "$CONTAINER" bash -lc \
    "cd ${BENCH} && FLOWSERVE_LOOGLE_DOC_TOKENS=-1 FLOWSERVE_MAX_TOTAL_TOKENS=40960 FLOWSERVE_DISABLE_CACHE=1 PYTHONPATH=/workspace/sglang/python:. \
     python3 main_sglang.py --endpoint ${endpoint}/v1 --served-model ${served} --tokenizer-path ${tokenizer} \
       --sharegpt-json data/longdep_qa.jsonl --loogle-json data/longdep_qa.jsonl --dataset LooGLE --test-type open \
       --request-rate ${rate} --num-requests ${requests} --warmup-trace-seconds 60 \
       --phase-barrier-dir ${RUN}/barrier --phase-participant ${participant} \
       --scheduler-event-jsonl ${RUN}/${output}.events.jsonl --scheduler-model-id ${model_id} \
       --output-jsonl ${RUN}/${output}.jsonl --output-summary ${RUN}/${output}.summary.json > ${RUN}/${output}.client.log 2>&1"
}

start_client http://127.0.0.1:33100 "$HOT_SERVED_MODEL" "$HOT_MODEL_PATH" 0.35 448 hot "$HOT_MODEL_ID" hot
start_client http://127.0.0.1:33101 "$COLD_B_SERVED_MODEL" "$COLD_B_MODEL_PATH" 0.15 96 cold14 "$COLD_B_MODEL_ID" cold14
start_client http://127.0.0.1:33102 "$COLD_C_SERVED_MODEL" "$COLD_C_MODEL_PATH" 0.15 96 coldllama "$COLD_C_MODEL_ID" coldllama
wait_for "hot H completion" "test -e ${RUN}/barrier/hot.ready" 3600
wait_for "cold14 H completion" "test -e ${RUN}/barrier/cold14.ready" 3600
wait_for "coldllama H completion" "test -e ${RUN}/barrier/coldllama.ready" 3600

docker exec -d "$CONTAINER" bash -lc \
  "PYTHONPATH=/workspace/sglang/python \
   python3 /workspace/sglang/python/sglang/srt/mem_cache/run_pressure_scheduler.py \
     --socket-path ${RUN}/central.sock \
     --max-transfer-gib ${PRESSURE_MAX_TRANSFER_GIB} \
     --interval-s ${PRESSURE_TICK_S} \
     --output-jsonl ${RUN}/pressure-scheduler.audit.jsonl \
     > ${RUN}/pressure-scheduler.log 2>&1"
sleep 2
exec_in "test -s ${RUN}/pressure-scheduler.audit.jsonl"
exec_in "touch ${RUN}/barrier/release"
echo "W released"
