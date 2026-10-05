#!/usr/bin/env bash
set -euo pipefail

repo_dir=${1:-/workspace/sglang}
output_dir=${2:-${repo_dir}/benchmark/test_motivation/Ledger/results/capacity_scalar_20260821}
repetitions=${3:-1}
stage=${4:-screen}
capacity_filter=${CAPACITY_FILTER:-}
python_bin=${repo_dir}/.venv/bin/python
cuda_devices=${CUDA_DEVICES:-6,7}
harness=${repo_dir}/benchmark/test_motivation/Ledger/validate_hybrid_motivation.py
server_log_dir=${output_dir}/server_logs
server_pid=""

if [[ "${stage}" != "calibrate" && "${stage}" != "screen" ]]; then
  echo "stage must be calibrate or screen, got ${stage}" >&2
  exit 2
fi

mkdir -p "${output_dir}" "${server_log_dir}"

stop_server() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill -TERM "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
  server_pid=""
  pkill -TERM -f '[s]glang.launch_server' 2>/dev/null || true
  for _ in $(seq 1 60); do
    if ! pgrep -f '[s]glang.launch_server' >/dev/null; then
      return
    fi
    sleep 1
  done
  echo "Timed out stopping SGLang" >&2
  return 1
}

wait_ready() {
  local log_file=$1
  for _ in $(seq 1 600); do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "SGLang exited during startup; inspect ${log_file}" >&2
      tail -n 80 "${log_file}" >&2
      return 1
    fi
    if curl -fsS --max-time 2 http://127.0.0.1:30000/health \
      >/dev/null 2>&1; then
      return
    fi
    sleep 1
  done
  echo "Timed out waiting for SGLang; inspect ${log_file}" >&2
  return 1
}

start_server() {
  local capacity_label=$1
  local state_cap=$2
  local kv_cap=$3
  local log_file=${server_log_dir}/${capacity_label}.log
  stop_server
  echo "START SERVER ${capacity_label}: state=${state_cap}, kv=${kv_cap}"
  nohup env CUDA_VISIBLE_DEVICES="${cuda_devices}" \
    SGLANG_ENABLE_TENANT_RESOURCE_TRACKING=1 \
    "${python_bin}" -m sglang.launch_server \
      --model-path /models/Kimi-Linear-48B-A3B \
      --tokenizer-path /models/Kimi-Linear-48B-A3B \
      --tp-size 2 \
      --trust-remote-code \
      --host 0.0.0.0 \
      --port 30000 \
      --mem-fraction-static 0.85 \
      --max-mamba-cache-size "${state_cap}" \
      --max-total-tokens "${kv_cap}" \
      --schedule-policy fcfs \
      >"${log_file}" 2>&1 &
  server_pid=$!
  wait_ready "${log_file}"
}

set_policy() {
  local policy=$1
  local weight=$2
  local policy_code
  case "${policy}" in
    fcfs) policy_code=0 ;;
    tenant-vtc) policy_code=1 ;;
    tenant-vector-debt) policy_code=2 ;;
    *) echo "Unknown tenant policy: ${policy}" >&2; return 2 ;;
  esac
  local payload
  payload=$(printf \
    '{"server_args":{"tenant_schedule_policy_code":%s,"tenant_vtc_request_weight":%s}}' \
    "${policy_code}" "${weight}")
  local response
  response=$(curl -fsS -X POST http://127.0.0.1:30000/set_internal_state \
    -H 'Content-Type: application/json' -d "${payload}")
  if [[ "${response}" != *'true'* ]]; then
    echo "Policy update failed: ${response}" >&2
    return 1
  fi
}

run_one() {
  local capacity_label=$1
  local policy_label=$2
  local schedule_policy=$3
  local weight=$4
  local repetition=$5
  local run_id="cross-${capacity_label}-${policy_label}-r${repetition}"
  local output=${output_dir}/${run_id}.json
  if [[ -s "${output}" ]]; then
    echo "SKIP ${run_id}"
    return
  fi
  set_policy "${schedule_policy}" "${weight}"
  echo "START ${run_id}"
  "${python_bin}" "${harness}" \
    --mode fairness-sustained \
    --order state-first \
    --run-id "${run_id}" \
    --output "${output}" \
    --capacity-label "${capacity_label}" \
    --policy-label "${policy_label}" \
    --scalar-weight "${weight}" \
    --waves 12 \
    --state-per-wave 16 \
    --kv-per-wave 1 \
    --wave-interval-ms 750 \
    --arrival-gap-ms 1 \
    --slo-seconds 20 \
    >"${output%.json}.stdout"
  echo "DONE ${run_id}"
}

restore_server() {
  local status=$?
  trap - EXIT INT TERM
  stop_server || true
  echo "RESTORE baseline FCFS server"
  nohup env CUDA_VISIBLE_DEVICES="${cuda_devices}" \
    SGLANG_ENABLE_TENANT_RESOURCE_TRACKING=1 \
    "${python_bin}" -m sglang.launch_server \
      --model-path /models/Kimi-Linear-48B-A3B \
      --tokenizer-path /models/Kimi-Linear-48B-A3B \
      --tp-size 2 \
      --trust-remote-code \
      --host 0.0.0.0 \
      --port 30000 \
      --mem-fraction-static 0.85 \
      --schedule-policy fcfs \
      >"${server_log_dir}/restored-fcfs.log" 2>&1 &
  server_pid=$!
  wait_ready "${server_log_dir}/restored-fcfs.log" || true
  exit "${status}"
}
trap restore_server EXIT INT TERM

capacity_rows=(
  "s240-k196608 240 196608"
  "s240-k393216 240 393216"
  "s240-k786432 240 786432"
  "s480-k196608 480 196608"
  "s480-k393216 480 393216"
  "s480-k786432 480 786432"
  "s720-k196608 720 196608"
  "s720-k393216 720 393216"
  "s720-k786432 720 786432"
)
read -r -a scalar_weights <<<"${SCALAR_WEIGHTS:-0 1024 2048 4096 8192 16384 32768}"

for row in "${capacity_rows[@]}"; do
  read -r capacity_label state_cap kv_cap <<<"${row}"
  if [[ -n "${capacity_filter}" && "${capacity_label}" != "${capacity_filter}" ]]; then
    continue
  fi
  start_server "${capacity_label}" "${state_cap}" "${kv_cap}"
  for repetition in $(seq 1 "${repetitions}"); do
    run_one "${capacity_label}" fcfs fcfs 0 "${repetition}"
    if [[ "${stage}" == "calibrate" ]]; then
      continue
    fi
    for weight in "${scalar_weights[@]}"; do
      run_one "${capacity_label}" "scalar-w${weight}" tenant-vtc "${weight}" "${repetition}"
    done
    run_one "${capacity_label}" vector tenant-vector-debt 0 "${repetition}"
  done
done
