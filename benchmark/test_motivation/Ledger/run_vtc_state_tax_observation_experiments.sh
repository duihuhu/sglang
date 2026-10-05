#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-profile}"
REPO_DIR="${2:-/workspace/sglang}"
RESULT_DIR="${3:-${REPO_DIR}/benchmark/test_motivation/Ledger/results/vtc_state_tax_20260826}"
PYTHON_BIN="${REPO_DIR}/.venv/bin/python"
DRIVER="${REPO_DIR}/benchmark/test_motivation/Ledger/validate_vtc_state_tax_motivation.py"
FITTER="${REPO_DIR}/benchmark/test_motivation/Ledger/fit_vtc_profile.py"
ANALYZER="${REPO_DIR}/benchmark/test_motivation/Ledger/analyze_vtc_state_tax_observation.py"
PROFILE="${RESULT_DIR}/profiled_vtc.json"
SELECTION="${RESULT_DIR}/selection.json"
LOG_DIR="${RESULT_DIR}/server_logs"
SERVER_PID=""

CELLS=(
  "f1-base 1 1.5 48 196608"
  "f4-base 4 1.5 48 196608"
  "f16-base 16 1.5 48 196608"
  "f32-base 32 1.5 48 196608"
  "f16-state-relaxed 16 1.5 480 196608"
  "f16-kv-relaxed 16 1.5 48 393216"
  "f16-full 16 1.5 480 393216"
  "f16-fast 16 1.0 48 196608"
  "f16-slow 16 3.0 48 196608"
)
TAXES=(1024 4096 8192 16384 32768 65536)

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
  local label="$1" state_cap="$2" kv_cap="$3"
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

profile_value() {
  local key="$1"
  "${PYTHON_BIN}" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["raw_scheduler_weights"][sys.argv[2]])' \
    "${PROFILE}" "${key}"
}

selection_value() {
  local cell="$1"
  "${PYTHON_BIN}" -c \
    'import json,sys; x=json.load(open(sys.argv[1])); print(x["global_request_tax"] if sys.argv[2]=="global" else x["per_cell_request_tax"][sys.argv[2]])' \
    "${SELECTION}" "${cell}"
}

tax_from_label() {
  local label="$1"
  if [[ "${label}" == "token-vtc" ]]; then
    echo 0
  else
    local tail="${label#request-tax-r}"
    echo "${tail}"
  fi
}

load_profile_weights() {
  PROFILE_REQUEST="$(profile_value request_weight)"
  PROFILE_INPUT="$(profile_value input_weight)"
  PROFILE_INPUT_QUAD="$(profile_value input_quadratic_weight)"
  PROFILE_OUTPUT="$(profile_value output_weight)"
  PROFILE_INTERACTION="$(profile_value input_output_weight)"
  PROFILE_OUTPUT_QUAD="$(profile_value output_quadratic_weight)"
}

run_profile_one() {
  local prompt="$1" output_tokens="$2" rep="$3"
  local run_id="st-profile-p${prompt}-o${output_tokens}-rep${rep}"
  local output="${RESULT_DIR}/${run_id}.json"
  if [[ -s "${output}" ]]; then
    echo "skip ${run_id}"
    return
  fi
  echo "START ${run_id}"
  "${PYTHON_BIN}" "${DRIVER}" --mode profile --run-id "${run_id}" \
    --output "${output}" --experiment-stage profile --physical-gpus 6,7 \
    --profile-prompt "${prompt}" --profile-output "${output_tokens}" \
    --runtime-policy fcfs --policy-label profile-fcfs \
    >"${output%.json}.stdout"
  echo "DONE ${run_id}"
}

run_partition_one() {
  local cell="$1" split="$2" interval="$3" waves="$4" stage="$5"
  local rep="$6" label="$7" runtime_policy="$8" composition="$9"
  shift 9
  local order="split-first"
  if [[ "${rep}" == "2" ]]; then
    order="packed-first"
  fi
  local run_id="st-${cell}-${stage}-${label}-rep${rep}"
  local output="${RESULT_DIR}/${run_id}.json"
  if [[ -s "${output}" ]]; then
    echo "skip ${run_id}"
    return
  fi
  echo "START ${run_id}"
  "${PYTHON_BIN}" "${DRIVER}" --mode partition --run-id "${run_id}" \
    --output "${output}" --experiment-stage "${stage}" --cell-id "${cell}" \
    --physical-gpus 6,7 --composition "${composition}" --waves "${waves}" \
    --wave-interval-s "${interval}" --split-factor "${split}" \
    --packed-prompt 16384 --packed-output 512 --wave-order "${order}" \
    --runtime-policy "${runtime_policy}" --policy-label "${label}" "$@" \
    >"${output%.json}.stdout"
  echo "DONE ${run_id}"
}

run_isolation() {
  local cell="$1" split="$2" interval="$3" waves="$4" stage="$5" rep="$6"
  local composition
  for composition in split-only packed-only; do
    run_partition_one "${cell}" "${split}" "${interval}" "${waves}" \
      "${stage}" "${rep}" "fcfs-${composition}" fcfs "${composition}"
  done
}

profiled_args() {
  PROFILE_ARGS=(
    --request-weight "${PROFILE_REQUEST}"
    --input-weight "${PROFILE_INPUT}"
    --input-quadratic-weight "${PROFILE_INPUT_QUAD}"
    --output-weight "${PROFILE_OUTPUT}"
    --input-output-weight "${PROFILE_INTERACTION}"
    --output-quadratic-weight "${PROFILE_OUTPUT_QUAD}"
  )
}

run_profile() {
  local prompt output_tokens rep
  start_server profile 480 393216
  for prompt in 1024 4096 16384; do
    for output_tokens in 32 128 512; do
      for rep in 1 2 3; do
        run_profile_one "${prompt}" "${output_tokens}" "${rep}"
      done
    done
  done
  "${PYTHON_BIN}" "${FITTER}" "${RESULT_DIR}" --output "${PROFILE}"
}

run_screen() {
  if [[ ! -s "${PROFILE}" ]]; then
    echo "missing ${PROFILE}; run profile first" >&2
    exit 2
  fi
  load_profile_weights
  profiled_args
  local spec cell split interval state_cap kv_cap tax
  for spec in "${CELLS[@]}"; do
    read -r cell split interval state_cap kv_cap <<<"${spec}"
    start_server "screen-${cell}" "${state_cap}" "${kv_cap}"
    run_isolation "${cell}" "${split}" "${interval}" 4 screen 1
    run_partition_one "${cell}" "${split}" "${interval}" 4 screen 1 fcfs fcfs mixed
    run_partition_one "${cell}" "${split}" "${interval}" 4 screen 1 request-vtc vtc mixed \
      --request-weight 1 --input-weight 0 --output-weight 0
    run_partition_one "${cell}" "${split}" "${interval}" 4 screen 1 token-vtc vtc mixed \
      --request-weight 0 --input-weight 1 --output-weight 2
    for tax in "${TAXES[@]}"; do
      run_partition_one "${cell}" "${split}" "${interval}" 4 screen 1 \
        "request-tax-r${tax}" vtc mixed --request-weight "${tax}" \
        --input-weight 1 --output-weight 2
    done
    run_partition_one "${cell}" "${split}" "${interval}" 4 screen 1 \
      profiled-vtc vtc mixed "${PROFILE_ARGS[@]}"
    run_partition_one "${cell}" "${split}" "${interval}" 4 screen 1 \
      instant-share instant mixed
  done
  "${PYTHON_BIN}" "${ANALYZER}" "${RESULT_DIR}" \
    --output "${RESULT_DIR}/screening_report.md" \
    --selection-output "${SELECTION}" \
    --analysis-output "${RESULT_DIR}/screening_analysis.json"
}

run_confirm() {
  if [[ ! -s "${PROFILE}" || ! -s "${SELECTION}" ]]; then
    echo "missing profile or selection; run profile and screen first" >&2
    exit 2
  fi
  load_profile_weights
  profiled_args
  local global_label global_tax spec cell split interval state_cap kv_cap
  local oracle_label oracle_tax rep
  global_label="$(selection_value global)"
  global_tax="$(tax_from_label "${global_label}")"
  for spec in "${CELLS[@]}"; do
    read -r cell split interval state_cap kv_cap <<<"${spec}"
    start_server "confirm-${cell}" "${state_cap}" "${kv_cap}"
    oracle_label="$(selection_value "${cell}")"
    oracle_tax="$(tax_from_label "${oracle_label}")"
    for rep in 1 2 3; do
      run_isolation "${cell}" "${split}" "${interval}" 8 confirm "${rep}"
      run_partition_one "${cell}" "${split}" "${interval}" 8 confirm "${rep}" fcfs fcfs mixed
      run_partition_one "${cell}" "${split}" "${interval}" 8 confirm "${rep}" request-vtc vtc mixed \
        --request-weight 1 --input-weight 0 --output-weight 0
      run_partition_one "${cell}" "${split}" "${interval}" 8 confirm "${rep}" token-vtc vtc mixed \
        --request-weight 0 --input-weight 1 --output-weight 2
      run_partition_one "${cell}" "${split}" "${interval}" 8 confirm "${rep}" \
        profiled-vtc vtc mixed "${PROFILE_ARGS[@]}"
      run_partition_one "${cell}" "${split}" "${interval}" 8 confirm "${rep}" \
        "global-${global_label}" vtc mixed --request-weight "${global_tax}" \
        --input-weight 1 --output-weight 2
      run_partition_one "${cell}" "${split}" "${interval}" 8 confirm "${rep}" \
        "oracle-${oracle_label}" vtc mixed --request-weight "${oracle_tax}" \
        --input-weight 1 --output-weight 2
      run_partition_one "${cell}" "${split}" "${interval}" 8 confirm "${rep}" \
        instant-share instant mixed
    done
  done
  "${PYTHON_BIN}" "${ANALYZER}" "${RESULT_DIR}" \
    --output "${RESULT_DIR}/vtc_state_tax_observation_report.md" \
    --selection-output "${SELECTION}" \
    --analysis-output "${RESULT_DIR}/analysis.json"
}

case "${STAGE}" in
  profile) run_profile ;;
  screen) run_screen ;;
  confirm) run_confirm ;;
  all)
    run_profile
    run_screen
    run_confirm
    ;;
  *)
    echo "usage: $0 [profile|screen|confirm|all]" >&2
    exit 2
    ;;
esac
