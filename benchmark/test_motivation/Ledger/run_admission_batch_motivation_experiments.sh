#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-formal}"
REPO_DIR="${2:-/workspace/sglang}"
RESULT_DIR="${3:-${REPO_DIR}/benchmark/test_motivation/Ledger/results/admission_batch_20260828}"
PYTHON_BIN="${REPO_DIR}/.venv/bin/python"
DRIVER="${REPO_DIR}/benchmark/test_motivation/Ledger/validate_admission_batch_motivation.py"
ANALYZER="${REPO_DIR}/benchmark/test_motivation/Ledger/analyze_admission_batch_motivation.py"
PROFILE="${REPO_DIR}/benchmark/test_motivation/Ledger/results/vtc_state_tax_20260826/profiled_vtc.json"
LOG_DIR="${RESULT_DIR}/server_logs"
SERVER_PID=""

CELLS=(
  "low-state-auto 48 196608 auto"
  "high-state-capped 480 196608 9"
  "high-state-auto 480 196608 auto"
)

mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

stop_server() {
  pkill -TERM -f '[s]glang.launch_server' 2>/dev/null || true
  for _ in $(seq 1 90); do
    if ! pgrep -f '[s]glang.launch_server' >/dev/null; then
      SERVER_PID=""
      return
    fi
    sleep 1
  done
  return 1
}

wait_ready() {
  local log_file="$1"
  for _ in $(seq 1 600); do
    if [[ -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      tail -n 120 "${log_file}" >&2
      return 1
    fi
    if curl -fsS --max-time 2 http://127.0.0.1:30000/health >/dev/null 2>&1; then
      return
    fi
    sleep 1
  done
  return 1
}

start_server() {
  local label="$1" state_cap="$2" kv_cap="$3" max_running="$4"
  local log_file="${LOG_DIR}/${label}.log"
  local running_args=()
  if [[ "${max_running}" != "auto" ]]; then
    running_args=(--max-running-requests "${max_running}")
  fi
  stop_server
  nvidia-smi -i 6,7 --query-gpu=index,name,uuid,memory.total \
    --format=csv,noheader >"${RESULT_DIR}/hardware-gpu6-7.txt"
  nohup env CUDA_VISIBLE_DEVICES="${CUDA_DEVICES:-6,7}" \
    SGLANG_ENABLE_TENANT_RESOURCE_TRACKING=1 \
    "${PYTHON_BIN}" -m sglang.launch_server \
      --model-path /models/Kimi-Linear-48B-A3B \
      --tokenizer-path /models/Kimi-Linear-48B-A3B \
      --tp-size 2 --trust-remote-code --host 0.0.0.0 --port 30000 \
      --mem-fraction-static 0.85 --schedule-policy fcfs \
      --max-mamba-cache-size "${state_cap}" --max-total-tokens "${kv_cap}" \
      "${running_args[@]}" >"${log_file}" 2>&1 &
  SERVER_PID=$!
  wait_ready "${log_file}"
}

restore_server() {
  local status=$?
  trap - EXIT INT TERM
  start_server restored-full-fcfs 480 393216 auto || true
  exit "${status}"
}

trap restore_server EXIT INT TERM

profile_value() {
  local key="$1"
  "${PYTHON_BIN}" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["raw_scheduler_weights"][sys.argv[2]])' \
    "${PROFILE}" "${key}"
}

run_case() {
  local run_id="$1" mode="$2" cell="$3" experiment_stage="$4"
  local composition="$5" runtime_policy="$6" policy_label="$7"
  shift 7
  "${PYTHON_BIN}" "${DRIVER}" \
    --mode "${mode}" --run-id "${run_id}" \
    --output "${RESULT_DIR}/${run_id}.json" \
    --cell-id "${cell}" --experiment-stage "${experiment_stage}" \
    --composition "${composition}" --runtime-policy "${runtime_policy}" \
    --policy-label "${policy_label}" --physical-gpus 6,7 \
    "$@" >"${RESULT_DIR}/${run_id}.stdout"
}

analyze() {
  "${PYTHON_BIN}" "${ANALYZER}" "${RESULT_DIR}" \
    --output "${RESULT_DIR}/admission_batch_motivation_report.md" \
    --analysis-output "${RESULT_DIR}/analysis.json"
}

if [[ "${STAGE}" == "analyze" ]]; then
  trap - EXIT INT TERM
  analyze
  exit 0
fi

if [[ "${STAGE}" == "smoke" ]]; then
  start_server smoke-high-capped 480 196608 9
  run_case ab-smoke partition smoke smoke mixed vtc token-vtc \
    --waves 1 --wave-interval-s 1.5 --split-factor 16 --wave-order split-first
  "${PYTHON_BIN}" -c \
    'import json,sys; x=json.load(open(sys.argv[1])); assert x["scheduler_resource_trace"]; assert x["scheduler_batch_trace"]; assert not x["scheduler_batch_trace"]["dropped_events"]' \
    "${RESULT_DIR}/ab-smoke.json"
  exit 0
fi

if [[ "${STAGE}" != "formal" ]]; then
  echo "unknown stage: ${STAGE}" >&2
  exit 2
fi

PROFILE_REQUEST="$(profile_value request_weight)"
PROFILE_INPUT="$(profile_value input_weight)"
PROFILE_INPUT2="$(profile_value input_quadratic_weight)"
PROFILE_OUTPUT="$(profile_value output_weight)"
PROFILE_IO="$(profile_value input_output_weight)"
PROFILE_OUTPUT2="$(profile_value output_quadratic_weight)"

for spec in "${CELLS[@]}"; do
  read -r cell state_cap kv_cap max_running <<<"${spec}"
  start_server "formal-${cell}" "${state_cap}" "${kv_cap}" "${max_running}"
  for rep in 1 2 3; do
    wave_order="split-first"
    if [[ "${rep}" == "2" ]]; then
      wave_order="packed-first"
    fi
    run_case "ab-${cell}-confirm-fcfs-split-only-rep${rep}" \
      partition "${cell}" confirm split-only fcfs fcfs-split-only \
      --waves 8 --wave-interval-s 1.5 --split-factor 16 --wave-order "${wave_order}"
    run_case "ab-${cell}-confirm-fcfs-packed-only-rep${rep}" \
      partition "${cell}" confirm packed-only fcfs fcfs-packed-only \
      --waves 8 --wave-interval-s 1.5 --split-factor 16 --wave-order "${wave_order}"
    run_case "ab-${cell}-confirm-token-vtc-rep${rep}" \
      partition "${cell}" confirm mixed vtc token-vtc \
      --waves 8 --wave-interval-s 1.5 --split-factor 16 --wave-order "${wave_order}" \
      --request-weight 0 --input-weight 1 --output-weight 2
    run_case "ab-${cell}-confirm-profiled-vtc-rep${rep}" \
      partition "${cell}" confirm mixed vtc profiled-vtc \
      --waves 8 --wave-interval-s 1.5 --split-factor 16 --wave-order "${wave_order}" \
      --request-weight "${PROFILE_REQUEST}" --input-weight "${PROFILE_INPUT}" \
      --input-quadratic-weight "${PROFILE_INPUT2}" --output-weight "${PROFILE_OUTPUT}" \
      --input-output-weight "${PROFILE_IO}" --output-quadratic-weight "${PROFILE_OUTPUT2}"
    run_case "ab-${cell}-confirm-instant-share-rep${rep}" \
      partition "${cell}" confirm mixed instant instant-share \
      --waves 8 --wave-interval-s 1.5 --split-factor 16 --wave-order "${wave_order}"
  done
done

start_server formal-pulse-capped4 480 196608 4
for rep in 1 2 3; do
  run_case "ab-pulse-capped4-pulse-fcfs-incumbent-only-rep${rep}" \
    pulse pulse-capped4 pulse incumbent-only fcfs fcfs-incumbent-only
  run_case "ab-pulse-capped4-pulse-fcfs-pulser-only-rep${rep}" \
    pulse pulse-capped4 pulse pulser-only fcfs fcfs-pulser-only
  run_case "ab-pulse-capped4-pulse-token-vtc-rep${rep}" \
    pulse pulse-capped4 pulse mixed vtc token-vtc \
    --request-weight 0 --input-weight 1 --output-weight 2
  run_case "ab-pulse-capped4-pulse-instant-share-rep${rep}" \
    pulse pulse-capped4 pulse mixed instant instant-share
done

analyze
