#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-calibrate}"
REPO_DIR="${2:-/workspace/sglang}"
RESULT_DIR="${3:-${REPO_DIR}/benchmark/test_motivation/Ledger/results/dynamic_cost_20260821}"
PYTHON_BIN="${REPO_DIR}/.venv/bin/python"
CUDA_DEVICES="${CUDA_DEVICES:-6,7}"
DRIVER="${REPO_DIR}/benchmark/test_motivation/Ledger/validate_dynamic_cost_motivation.py"
LOG_DIR="${RESULT_DIR}/server_logs"
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
      tail -n 100 "${log_file}" >&2
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
  shift
  local log_file="${LOG_DIR}/${label}.log"
  stop_server
  nohup env CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
    SGLANG_ENABLE_TENANT_RESOURCE_TRACKING=1 \
    "${PYTHON_BIN}" -m sglang.launch_server \
      --model-path /models/Kimi-Linear-48B-A3B \
      --tokenizer-path /models/Kimi-Linear-48B-A3B \
      --tp-size 2 --trust-remote-code --host 0.0.0.0 --port 30000 \
      --mem-fraction-static 0.85 --schedule-policy fcfs "$@" \
      >"${log_file}" 2>&1 &
  SERVER_PID=$!
  wait_ready "${log_file}"
}

restore_server() {
  local status=$?
  trap - EXIT INT TERM
  echo "restoring full-capacity FCFS server"
  start_server restored-full-fcfs || true
  exit "${status}"
}
trap restore_server EXIT INT TERM

run_one() {
  local run_id="$1"
  shift
  local output="${RESULT_DIR}/${run_id}.json"
  if [[ -s "${output}" ]]; then
    echo "skip ${run_id}"
    return
  fi
  echo "run ${run_id}"
  "${PYTHON_BIN}" "${DRIVER}" --mode phase --run-id "${run_id}" \
    --output "${output}" "$@" >"${output%.json}.stdout"
}

start_server phase-s480-k196608 \
  --max-mamba-cache-size 480 --max-total-tokens 196608

case "${STAGE}" in
  calibrate)
    read -r -a weights <<<"${SCALAR_WEIGHTS:-0 1024 4096 16384 65536 262144}"
    for pattern in state kv; do
      run_one "phase-cal-${pattern}-fcfs-r1" --phase-pattern "${pattern}" --policy fcfs
      for weight in "${weights[@]}"; do
        run_one "phase-cal-${pattern}-scalar-w${weight}-r1" \
          --phase-pattern "${pattern}" --policy scalar --scalar-weight "${weight}"
      done
      run_one "phase-cal-${pattern}-vector-r1" --phase-pattern "${pattern}" --policy vector
    done
    ;;
  screen)
    read -r -a weights <<<"${SCALAR_WEIGHTS:-0 1024 4096 16384 65536 262144}"
    run_one phase-switch-fcfs-r1 --phase-pattern switch --policy fcfs
    for weight in "${weights[@]}"; do
      run_one "phase-switch-scalar-w${weight}-r1" \
        --phase-pattern switch --policy scalar --scalar-weight "${weight}"
    done
    run_one phase-switch-vector-r1 --phase-pattern switch --policy vector
    run_one phase-switch-dynamic-r1 --phase-pattern switch \
      --policy dynamic-scalar --phase-weights "${PHASE_WEIGHTS:-65536,0,65536}"
    ;;
  confirm)
    read -r -a weights <<<"${FIXED_WEIGHTS:-0 65536}"
    reps="${REPS:-3}"
    for ((rep = 1; rep <= reps; rep++)); do
      run_one "phase-confirm-fcfs-r${rep}" --phase-pattern switch --policy fcfs
      for weight in "${weights[@]}"; do
        run_one "phase-confirm-scalar-w${weight}-r${rep}" \
          --phase-pattern switch --policy scalar --scalar-weight "${weight}"
      done
      run_one "phase-confirm-vector-r${rep}" --phase-pattern switch --policy vector
      run_one "phase-confirm-dynamic-r${rep}" --phase-pattern switch \
        --policy dynamic-scalar --phase-weights "${PHASE_WEIGHTS:-65536,0,65536}"
    done
    ;;
  *)
    echo "usage: $0 [calibrate|screen|confirm]" >&2
    exit 2
    ;;
esac
