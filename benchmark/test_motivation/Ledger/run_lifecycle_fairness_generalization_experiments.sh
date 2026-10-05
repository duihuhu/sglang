#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-screen}"
REPO_DIR="${2:-/workspace/sglang}"
RESULT_DIR="${3:-${REPO_DIR}/benchmark/test_motivation/Ledger/results/lifecycle_fairness_generalization_20260825}"
PYTHON_BIN="${REPO_DIR}/.venv/bin/python"
DRIVER="${REPO_DIR}/benchmark/test_motivation/Ledger/validate_lifecycle_gate_motivation.py"
ANALYZER="${REPO_DIR}/benchmark/test_motivation/Ledger/analyze_lifecycle_fairness_generalization.py"
SELECTION="${RESULT_DIR}/selection.json"
LOG_DIR="${RESULT_DIR}/server_logs"
SERVER_PID=""

CELLS=(
  "base 16 16384 512 1.5 48 196608"
  "split4 4 16384 512 1.5 48 196608"
  "split32 32 16384 512 1.5 48 196608"
  "output1k 16 8192 1024 1.5 48 196608"
  "output2k 16 4096 2048 1.5 48 196608"
  "arrival1 16 16384 512 1.0 48 196608"
  "arrival3 16 16384 512 3.0 48 196608"
  "state96 16 16384 512 1.5 96 196608"
  "kv393 16 16384 512 1.5 48 393216"
  "relaxed 16 16384 512 1.5 96 393216"
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
  local label="$1"
  local state_cap="$2"
  local kv_cap="$3"
  local log_file="${LOG_DIR}/${label}.log"
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
  local cell="$1" split="$2" prompt="$3" output_tokens="$4"
  local interval="$5" waves="$6" stage="$7" rep="$8" label="$9"
  shift 9
  local run_id="gen-${cell}-${stage}-${label}-rep${rep}"
  local output="${RESULT_DIR}/${run_id}.json"
  if [[ -s "${output}" ]]; then
    echo "skip ${run_id}"
    return
  fi
  echo "START ${run_id}"
  "${PYTHON_BIN}" "${DRIVER}" --mode fairness-sustained \
    --run-id "${run_id}" --output "${output}" --cell-id "${cell}" \
    --physical-gpus 6,7 --experiment-stage "${stage}" \
    --composition mixed --waves "${waves}" --wave-interval-s "${interval}" \
    --packed-prompt "${prompt}" --packed-output "${output_tokens}" \
    --split-factor "${split}" --slo-seconds 60 "$@" \
    >"${output%.json}.stdout"
  echo "DONE ${run_id}"
}

run_isolation() {
  local cell="$1" split="$2" prompt="$3" output_tokens="$4"
  local interval="$5" waves="$6" stage="$7" rep="$8"
  local composition run_id output
  for composition in split-only packed-only; do
    run_id="gen-${cell}-${stage}-fcfs-${composition}-rep${rep}"
    output="${RESULT_DIR}/${run_id}.json"
    if [[ -s "${output}" ]]; then
      echo "skip ${run_id}"
      continue
    fi
    echo "START ${run_id}"
    "${PYTHON_BIN}" "${DRIVER}" --mode fairness-sustained \
      --run-id "${run_id}" --output "${output}" --cell-id "${cell}" \
      --physical-gpus 6,7 --experiment-stage "${stage}" \
      --runtime-policy fcfs --composition "${composition}" --waves "${waves}" \
      --wave-interval-s "${interval}" --packed-prompt "${prompt}" \
      --packed-output "${output_tokens}" --split-factor "${split}" \
      --slo-seconds 60 >"${output%.json}.stdout"
    echo "DONE ${run_id}"
  done
}

selection_value() {
  local cell="$1" key="$2"
  "${PYTHON_BIN}" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]][sys.argv[3]])' \
    "${SELECTION}" "${cell}" "${key}"
}

run_screen() {
  local spec cell split prompt output_tokens interval state_cap kv_cap weight
  for spec in "${CELLS[@]}"; do
    read -r cell split prompt output_tokens interval state_cap kv_cap <<<"${spec}"
    start_server "screen-${cell}" "${state_cap}" "${kv_cap}"
    run_isolation "${cell}" "${split}" "${prompt}" "${output_tokens}" \
      "${interval}" 4 screen 1
    run_one "${cell}" "${split}" "${prompt}" "${output_tokens}" \
      "${interval}" 4 screen 1 fcfs --runtime-policy fcfs
    run_one "${cell}" "${split}" "${prompt}" "${output_tokens}" \
      "${interval}" 4 screen 1 request-vtc --runtime-policy vtc \
      --request-weight 1 --input-weight 0 --output-weight 0
    run_one "${cell}" "${split}" "${prompt}" "${output_tokens}" \
      "${interval}" 4 screen 1 token-vtc --runtime-policy vtc
    for weight in 1024 4096 8192 16384 32768 65536; do
      run_one "${cell}" "${split}" "${prompt}" "${output_tokens}" \
        "${interval}" 4 screen 1 "request-tax-r${weight}" \
        --runtime-policy vtc --request-weight "${weight}"
    done
    run_one "${cell}" "${split}" "${prompt}" "${output_tokens}" \
      "${interval}" 4 screen 1 instant --runtime-policy instant
    run_one "${cell}" "${split}" "${prompt}" "${output_tokens}" \
      "${interval}" 4 screen 1 vector --runtime-policy vector
    for weight in 0 0.25 0.5 1 2 4 8; do
      run_one "${cell}" "${split}" "${prompt}" "${output_tokens}" \
        "${interval}" 4 screen 1 "lifecycle-s${weight}-k1" \
        --runtime-policy lifecycle --lifecycle-state-weight "${weight}" \
        --lifecycle-kv-weight 1
    done
  done
  "${PYTHON_BIN}" "${ANALYZER}" "${RESULT_DIR}" \
    --output "${RESULT_DIR}/screening_report.md" \
    --selection-output "${SELECTION}"
}

run_selected_request_tax() {
  local selected="$1" cell="$2" split="$3" prompt="$4" output_tokens="$5"
  local interval="$6" rep="$7"
  if [[ "${selected}" == "token-vtc" ]]; then
    return
  fi
  local tail="${selected#request-tax-r}"
  local weight="${tail%%-i*}"
  run_one "${cell}" "${split}" "${prompt}" "${output_tokens}" \
    "${interval}" 8 confirm "${rep}" "selected-request-tax-r${weight}" \
    --runtime-policy vtc --request-weight "${weight}"
}

run_selected_lifecycle() {
  local selected="$1" cell="$2" split="$3" prompt="$4" output_tokens="$5"
  local interval="$6" rep="$7"
  if [[ "${selected}" == "lifecycle-s0.25-k1" ]]; then
    return
  fi
  local tail="${selected#lifecycle-s}"
  local state_weight="${tail%%-k*}"
  local kv_weight="${tail##*-k}"
  run_one "${cell}" "${split}" "${prompt}" "${output_tokens}" \
    "${interval}" 8 confirm "${rep}" "oracle-lifecycle-s${state_weight}-k${kv_weight}" \
    --runtime-policy lifecycle --lifecycle-state-weight "${state_weight}" \
    --lifecycle-kv-weight "${kv_weight}"
}

run_confirm() {
  if [[ ! -s "${SELECTION}" ]]; then
    echo "missing ${SELECTION}; run screen first" >&2
    exit 2
  fi
  local spec cell split prompt output_tokens interval state_cap kv_cap rep
  local request_tax lifecycle_oracle
  for spec in "${CELLS[@]}"; do
    read -r cell split prompt output_tokens interval state_cap kv_cap <<<"${spec}"
    start_server "confirm-${cell}" "${state_cap}" "${kv_cap}"
    request_tax="$(selection_value "${cell}" request_tax)"
    lifecycle_oracle="$(selection_value "${cell}" lifecycle_oracle)"
    for rep in 1 2 3; do
      run_isolation "${cell}" "${split}" "${prompt}" "${output_tokens}" \
        "${interval}" 8 confirm "${rep}"
      run_one "${cell}" "${split}" "${prompt}" "${output_tokens}" \
        "${interval}" 8 confirm "${rep}" fcfs --runtime-policy fcfs
      run_one "${cell}" "${split}" "${prompt}" "${output_tokens}" \
        "${interval}" 8 confirm "${rep}" request-vtc --runtime-policy vtc \
        --request-weight 1 --input-weight 0 --output-weight 0
      run_one "${cell}" "${split}" "${prompt}" "${output_tokens}" \
        "${interval}" 8 confirm "${rep}" token-vtc --runtime-policy vtc
      run_selected_request_tax "${request_tax}" "${cell}" "${split}" \
        "${prompt}" "${output_tokens}" "${interval}" "${rep}"
      run_one "${cell}" "${split}" "${prompt}" "${output_tokens}" \
        "${interval}" 8 confirm "${rep}" instant --runtime-policy instant
      run_one "${cell}" "${split}" "${prompt}" "${output_tokens}" \
        "${interval}" 8 confirm "${rep}" vector --runtime-policy vector
      run_one "${cell}" "${split}" "${prompt}" "${output_tokens}" \
        "${interval}" 8 confirm "${rep}" lifecycle-s0.25-k1 \
        --runtime-policy lifecycle --lifecycle-state-weight 0.25 \
        --lifecycle-kv-weight 1
      run_selected_lifecycle "${lifecycle_oracle}" "${cell}" "${split}" \
        "${prompt}" "${output_tokens}" "${interval}" "${rep}"
    done
  done
  "${PYTHON_BIN}" "${ANALYZER}" "${RESULT_DIR}" \
    --output "${RESULT_DIR}/lifecycle_fairness_generalization_report.md" \
    --selection-output "${SELECTION}"
}

run_kill_confirm() {
  local spec cell split prompt output_tokens interval state_cap kv_cap rep
  for spec in "${CELLS[@]}"; do
    read -r cell split prompt output_tokens interval state_cap kv_cap <<<"${spec}"
    start_server "kill-${cell}" "${state_cap}" "${kv_cap}"
    for rep in 1 2 3; do
      run_isolation "${cell}" "${split}" "${prompt}" "${output_tokens}" \
        "${interval}" 8 kill "${rep}"
      run_one "${cell}" "${split}" "${prompt}" "${output_tokens}" \
        "${interval}" 8 kill "${rep}" token-vtc --runtime-policy vtc
      run_one "${cell}" "${split}" "${prompt}" "${output_tokens}" \
        "${interval}" 8 kill "${rep}" instant --runtime-policy instant
      run_one "${cell}" "${split}" "${prompt}" "${output_tokens}" \
        "${interval}" 8 kill "${rep}" lifecycle-s0.25-k1 \
        --runtime-policy lifecycle --lifecycle-state-weight 0.25 \
        --lifecycle-kv-weight 1
    done
  done
  "${PYTHON_BIN}" "${ANALYZER}" "${RESULT_DIR}" \
    --output "${RESULT_DIR}/lifecycle_fairness_generalization_report.md" \
    --selection-output "${SELECTION}"
}

case "${STAGE}" in
  screen) run_screen ;;
  confirm) run_confirm ;;
  kill) run_kill_confirm ;;
  all)
    run_screen
    run_confirm
    run_kill_confirm
    ;;
  *)
    echo "usage: $0 [screen|confirm|kill|all]" >&2
    exit 2
    ;;
esac
