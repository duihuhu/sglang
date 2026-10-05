#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-calibrate}"
REPO_DIR="${2:-/workspace/sglang}"
RESULT_DIR="${3:-${REPO_DIR}/benchmark/test_motivation/Ledger/results/lifecycle_fairness_oracle_20260825}"
PYTHON_BIN="${REPO_DIR}/.venv/bin/python"
DRIVER="${REPO_DIR}/benchmark/test_motivation/Ledger/validate_lifecycle_gate_motivation.py"
ANALYZER="${REPO_DIR}/benchmark/test_motivation/Ledger/analyze_lifecycle_fairness_oracle.py"
LOG_DIR="${RESULT_DIR}/server_logs"
SELECTION="${RESULT_DIR}/selection.json"
SERVER_PID=""

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
  local label="$1"
  local state_cap="$2"
  local kv_cap="$3"
  local log_file="${LOG_DIR}/${label}.log"
  stop_server
  nohup env CUDA_VISIBLE_DEVICES="${CUDA_DEVICES:-6,7}" \
    SGLANG_ENABLE_TENANT_RESOURCE_TRACKING=1 \
    "${PYTHON_BIN}" -m sglang.launch_server \
      --model-path /models/Kimi-Linear-48B-A3B \
      --tokenizer-path /models/Kimi-Linear-48B-A3B \
      --tp-size 2 --trust-remote-code --host 0.0.0.0 --port 30000 \
      --mem-fraction-static 0.85 --schedule-policy fcfs \
      --max-mamba-cache-size "${state_cap}" --max-total-tokens "${kv_cap}" \
      >"${log_file}" 2>&1 &
  SERVER_PID=$!
  wait_ready "${log_file}"
}

restore_server() {
  local status=$?
  trap - EXIT INT TERM
  echo "restoring full-capacity FCFS server"
  start_server restored-full-fcfs 480 393216 || true
  exit "${status}"
}
trap restore_server EXIT INT TERM

run_one() {
  local output="$1"
  shift
  if [[ -s "${output}" ]]; then
    echo "skip ${output}"
    return
  fi
  echo "START $(basename "${output}" .json)"
  "${PYTHON_BIN}" "${DRIVER}" --mode fairness-sustained \
    --output "${output}" --waves 8 --wave-interval-s 1.5 \
    --slo-seconds 30 "$@" >"${output%.json}.stdout"
  echo "DONE $(basename "${output}" .json)"
}

run_smoke() {
  start_server fairness-smoke 48 196608
  local policy run_id path
  for policy in vtc lifecycle; do
    run_id="smoke-fairness-${policy}"
    path="${RESULT_DIR}/${run_id}.json"
    rm -f "${path}" "${path%.json}.stdout"
    echo "START ${run_id}"
    if [[ "${policy}" == lifecycle ]]; then
      "${PYTHON_BIN}" "${DRIVER}" --mode fairness-sustained \
        --output "${path}" --run-id "${run_id}" --composition mixed \
        --runtime-policy lifecycle --lifecycle-state-weight 1 \
        --lifecycle-kv-weight 1 --waves 2 --wave-interval-s 1.5 \
        --slo-seconds 30 >"${path%.json}.stdout"
    else
      "${PYTHON_BIN}" "${DRIVER}" --mode fairness-sustained \
        --output "${path}" --run-id "${run_id}" --composition mixed \
        --runtime-policy vtc --request-weight 0 --waves 2 \
        --wave-interval-s 1.5 --slo-seconds 30 >"${path%.json}.stdout"
    fi
    echo "DONE ${run_id}"
  done
}

run_isolation() {
  local stage="$1"
  local rep="$2"
  local composition run_id path
  for composition in split-only packed-only; do
    run_id="fair-${stage}-fcfs-${composition}-rep${rep}"
    path="${RESULT_DIR}/${run_id}.json"
    run_one "${path}" --run-id "${run_id}" --experiment-stage "${stage}" \
      --runtime-policy fcfs --composition "${composition}"
  done
}

run_calibration() {
  start_server fairness-calibration 48 196608
  run_isolation calibration 1
  local run_id path weight
  run_id="fair-calibration-token-vtc-rep1"
  path="${RESULT_DIR}/${run_id}.json"
  run_one "${path}" --run-id "${run_id}" --experiment-stage calibration \
    --runtime-policy vtc --request-weight 0 --composition mixed

  for weight in 1024 4096 8192 12288 16384 32768 65536; do
    run_id="fair-calibration-scalar-r${weight}-rep1"
    path="${RESULT_DIR}/${run_id}.json"
    run_one "${path}" --run-id "${run_id}" --experiment-stage calibration \
      --runtime-policy vtc --request-weight "${weight}" --composition mixed
  done

  for weight in 0 0.25 0.5 1 2 4 8 16; do
    run_id="fair-calibration-lifecycle-s${weight}-k1-rep1"
    path="${RESULT_DIR}/${run_id}.json"
    run_one "${path}" --run-id "${run_id}" --experiment-stage calibration \
      --runtime-policy lifecycle --lifecycle-state-weight "${weight}" \
      --lifecycle-kv-weight 1 --composition mixed
  done

  "${PYTHON_BIN}" "${ANALYZER}" "${RESULT_DIR}" \
    --output "${RESULT_DIR}/calibration_report.md" \
    --selection-output "${SELECTION}"
}

selection_value() {
  local key="$1"
  "${PYTHON_BIN}" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' \
    "${SELECTION}" "${key}"
}

run_confirmation() {
  if [[ ! -s "${SELECTION}" ]]; then
    echo "missing ${SELECTION}; run calibrate first" >&2
    exit 2
  fi
  start_server fairness-confirm 48 196608
  local scalar_weight lifecycle_state_weight lifecycle_kv_weight
  scalar_weight="$(selection_value scalar_request_weight)"
  lifecycle_state_weight="$(selection_value lifecycle_state_weight)"
  lifecycle_kv_weight="$(selection_value lifecycle_kv_weight)"
  local rep run_id path
  for rep in 1 2 3; do
    run_isolation confirm "${rep}"

    run_id="fair-confirm-token-vtc-rep${rep}"
    path="${RESULT_DIR}/${run_id}.json"
    run_one "${path}" --run-id "${run_id}" --experiment-stage confirm \
      --runtime-policy vtc --request-weight 0 --composition mixed

    run_id="fair-confirm-scalar-r${scalar_weight}-rep${rep}"
    path="${RESULT_DIR}/${run_id}.json"
    run_one "${path}" --run-id "${run_id}" --experiment-stage confirm \
      --runtime-policy vtc --request-weight "${scalar_weight}" --composition mixed

    run_id="fair-confirm-lifecycle-s${lifecycle_state_weight}-k${lifecycle_kv_weight}-rep${rep}"
    path="${RESULT_DIR}/${run_id}.json"
    run_one "${path}" --run-id "${run_id}" --experiment-stage confirm \
      --runtime-policy lifecycle \
      --lifecycle-state-weight "${lifecycle_state_weight}" \
      --lifecycle-kv-weight "${lifecycle_kv_weight}" --composition mixed
  done

  "${PYTHON_BIN}" "${ANALYZER}" "${RESULT_DIR}" \
    --output "${RESULT_DIR}/lifecycle_fairness_oracle_report.md" \
    --selection-output "${SELECTION}"
}

case "${STAGE}" in
  smoke) run_smoke ;;
  calibrate) run_calibration ;;
  confirm) run_confirmation ;;
  *)
    echo "usage: $0 [smoke|calibrate|confirm]" >&2
    exit 2
    ;;
esac
