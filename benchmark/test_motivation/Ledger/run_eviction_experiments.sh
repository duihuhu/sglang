#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-/workspace/sglang}"
RESULT_DIR="${2:-${REPO_DIR}/benchmark/test_motivation/Ledger/results/dynamic_cost_20260821}"
REPS="${REPS:-3}"
PYTHON_BIN="${REPO_DIR}/.venv/bin/python"
CUDA_DEVICES="${CUDA_DEVICES:-6,7}"
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
  echo "timed out stopping SGLang" >&2
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
  echo "timed out waiting for SGLang" >&2
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

echo "starting controlled low-KV server"
start_server eviction-s480-k196608 \
  --max-mamba-cache-size 480 --max-total-tokens 196608

cd "${REPO_DIR}"
REPS="${REPS}" RESULT_DIR="${RESULT_DIR#${REPO_DIR}/}" \
  bash benchmark/test_motivation/Ledger/run_dynamic_cost_experiments.sh eviction
