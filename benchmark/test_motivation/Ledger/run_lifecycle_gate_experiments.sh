#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-all}"
REPO_DIR="${2:-/workspace/sglang}"
RESULT_DIR="${3:-${REPO_DIR}/benchmark/test_motivation/Ledger/results/lifecycle_gates_20260825}"
PYTHON_BIN="${REPO_DIR}/.venv/bin/python"
CUDA_DEVICES="${CUDA_DEVICES:-6,7}"
DYNAMIC_DRIVER="${REPO_DIR}/benchmark/test_motivation/Ledger/validate_dynamic_cost_motivation.py"
GATE_DRIVER="${REPO_DIR}/benchmark/test_motivation/Ledger/validate_lifecycle_gate_motivation.py"
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
      tail -n 120 "${log_file}" >&2
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
  local policy="$2"
  local state_cap="$3"
  local kv_cap="$4"
  local log_file="${LOG_DIR}/${label}.log"
  stop_server
  nohup env CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
    SGLANG_ENABLE_TENANT_RESOURCE_TRACKING=1 \
    "${PYTHON_BIN}" -m sglang.launch_server \
      --model-path /models/Kimi-Linear-48B-A3B \
      --tokenizer-path /models/Kimi-Linear-48B-A3B \
      --tp-size 2 --trust-remote-code --host 0.0.0.0 --port 30000 \
      --mem-fraction-static 0.85 --schedule-policy "${policy}" \
      --max-mamba-cache-size "${state_cap}" --max-total-tokens "${kv_cap}" \
      >"${log_file}" 2>&1 &
  SERVER_PID=$!
  wait_ready "${log_file}"
}

restore_server() {
  local status=$?
  trap - EXIT INT TERM
  echo "restoring full-capacity FCFS server"
  start_server restored-full-fcfs fcfs 480 393216 || true
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
  "${PYTHON_BIN}" "$@" --output "${output}" >"${output%.json}.stdout"
  echo "DONE $(basename "${output}" .json)"
}

run_smoke() {
  start_server smoke-fcfs fcfs 96 196608
  local run_id="smoke-lifecycle-gates"
  local path="${RESULT_DIR}/${run_id}.json"
  rm -f "${path}" "${path%.json}.stdout"
  run_one "${path}" "${GATE_DRIVER}" --mode state-pair \
    --run-id "${run_id}" --regime smoke --order interleave --pair-count 2
}

run_m1() {
  start_server m1-full-fcfs fcfs 480 393216
  local rep prompt output reuse tag run_id path
  for rep in $(seq 1 "${M1_REPS:-2}"); do
    for prompt in 1024 4096 16384; do
      for output in 32 128 512; do
        for reuse in 0 0.5 0.9; do
          tag="p${prompt}-o${output}-r${reuse/./}"
          run_id="m1-${tag}-rep${rep}"
          path="${RESULT_DIR}/${run_id}.json"
          run_one "${path}" "${DYNAMIC_DRIVER}" \
            --mode cost --run-id "${run_id}" --prompt-len "${prompt}" \
            --output-len "${output}" --count 8 --reuse-fraction "${reuse}"
        done
      done
    done
  done
}

run_m2() {
  local config regime state_cap kv_cap rep order run_id path
  for config in state-bound:48:393216 kv-bound:480:196608 compute-bound:480:393216; do
    IFS=: read -r regime state_cap kv_cap <<<"${config}"
    start_server "m2-${regime}" fcfs "${state_cap}" "${kv_cap}"
    for rep in $(seq 1 "${M2_REPS:-2}"); do
      for order in decode-first prefill-first interleave; do
        run_id="m2-${regime}-${order}-rep${rep}"
        path="${RESULT_DIR}/${run_id}.json"
        run_one "${path}" "${GATE_DRIVER}" --mode state-pair \
          --run-id "${run_id}" --regime "${regime}" --order "${order}"
      done
    done
  done
}

run_m3() {
  local policy runtime rep order run_id path
  for policy in fcfs lpm; do
    start_server "m3-${policy}" "${policy}" 96 196608
    runtime=fcfs
    [[ "${policy}" == lpm ]] && runtime=server
    for rep in $(seq 1 "${M3_REPS:-2}"); do
      for order in pressure-first reuse-first interleave; do
        run_id="m3-${policy}-${order}-rep${rep}"
        path="${RESULT_DIR}/${run_id}.json"
        run_one "${path}" "${GATE_DRIVER}" --mode joint \
          --run-id "${run_id}" --regime "${policy}" --runtime-policy "${runtime}" \
          --order "${order}"
      done
    done
  done
}

run_m4() {
  start_server m4-fairness fcfs 96 196608
  local rep policy composition run_id path
  for rep in $(seq 1 "${M4_REPS:-3}"); do
    for composition in split-only packed-only; do
      run_id="m4-fcfs-${composition}-rep${rep}"
      path="${RESULT_DIR}/${run_id}.json"
      run_one "${path}" "${GATE_DRIVER}" --mode fairness \
        --run-id "${run_id}" --runtime-policy fcfs --composition "${composition}"
    done
    for policy in fcfs vtc; do
      run_id="m4-${policy}-mixed-rep${rep}"
      path="${RESULT_DIR}/${run_id}.json"
      run_one "${path}" "${GATE_DRIVER}" --mode fairness \
        --run-id "${run_id}" --runtime-policy "${policy}" --composition mixed
    done
  done
}

run_m5() {
  start_server m5-balanced-fcfs fcfs 96 196608
  local order rep tag run_id path
  local orders=(
    decode,prefill,reuse,medium decode,prefill,medium,reuse
    decode,reuse,prefill,medium decode,reuse,medium,prefill
    decode,medium,prefill,reuse decode,medium,reuse,prefill
    prefill,decode,reuse,medium prefill,decode,medium,reuse
    prefill,reuse,decode,medium prefill,reuse,medium,decode
    prefill,medium,decode,reuse prefill,medium,reuse,decode
    reuse,decode,prefill,medium reuse,decode,medium,prefill
    reuse,prefill,decode,medium reuse,prefill,medium,decode
    reuse,medium,decode,prefill reuse,medium,prefill,decode
    medium,decode,prefill,reuse medium,decode,reuse,prefill
    medium,prefill,decode,reuse medium,prefill,reuse,decode
    medium,reuse,decode,prefill medium,reuse,prefill,decode
  )
  for rep in $(seq 1 "${M5_REPS:-1}"); do
    for order in "${orders[@]}"; do
      tag="${order//,/-}"
      run_id="m5-${tag}-rep${rep}"
      path="${RESULT_DIR}/${run_id}.json"
      run_one "${path}" "${GATE_DRIVER}" --mode permutation \
        --run-id "${run_id}" --regime balanced --order "${order}"
    done
  done
}

case "${STAGE}" in
  smoke) run_smoke ;;
  m1) run_m1 ;;
  m2) run_m2 ;;
  m3) run_m3 ;;
  m4) run_m4 ;;
  m5) run_m5 ;;
  all)
    run_m1
    run_m2
    run_m3
    run_m4
    run_m5
    ;;
  *)
    echo "usage: $0 [smoke|m1|m2|m3|m4|m5|all]" >&2
    exit 2
    ;;
esac
