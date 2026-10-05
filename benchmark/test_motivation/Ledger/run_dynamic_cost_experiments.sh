#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-all}"
REPS="${REPS:-3}"
BASE_URL="${BASE_URL:-http://127.0.0.1:30000}"
RESULT_DIR="${RESULT_DIR:-benchmark/test_motivation/Ledger/results/dynamic_cost_20260821}"
DRIVER="benchmark/test_motivation/Ledger/validate_dynamic_cost_motivation.py"

mkdir -p "${RESULT_DIR}"

run_one() {
  local output="$1"
  shift
  if [[ -s "${output}" ]]; then
    echo "skip ${output}"
    return
  fi
  python3 "${DRIVER}" --base-url "${BASE_URL}" --output "${output}" "$@"
}

run_cost() {
  local conditions=(
    "1024 32 1 0"
    "1024 512 1 0"
    "4096 32 1 0"
    "16384 32 1 0"
    "16384 512 1 0"
    "1024 32 16 0"
    "1024 512 16 0"
    "16384 32 8 0"
    "16384 32 8 0.9"
    "16384 512 8 0.9"
  )
  local rep condition prompt output count reuse tag run_id path
  for ((rep = 1; rep <= REPS; rep++)); do
    for condition in "${conditions[@]}"; do
      read -r prompt output count reuse <<<"${condition}"
      tag="p${prompt}_o${output}_n${count}_r${reuse/./}"
      run_id="cost-${tag}-rep${rep}"
      path="${RESULT_DIR}/${run_id}.json"
      run_one "${path}" --mode cost --run-id "${run_id}" \
        --prompt-len "${prompt}" --output-len "${output}" \
        --count "${count}" --reuse-fraction "${reuse}"
    done
  done
}

run_pairs() {
  local rep pair order run_id path
  for ((rep = 1; rep <= REPS; rep++)); do
    for pair in state-kv lifetime reuse; do
      for order in left-first right-first interleave; do
        run_id="pair-${pair}-${order}-rep${rep}"
        path="${RESULT_DIR}/${run_id}.json"
        run_one "${path}" --mode pair --run-id "${run_id}" \
          --pair "${pair}" --order "${order}"
      done
    done
  done
}

run_batch() {
  local rep candidate run_id path
  for ((rep = 1; rep <= REPS; rep++)); do
    for candidate in none short-unique long-unique long-reused; do
      run_id="batch-${candidate}-rep${rep}"
      path="${RESULT_DIR}/${run_id}.json"
      run_one "${path}" --mode batch --run-id "${run_id}" \
        --candidate "${candidate}"
    done
  done
}

run_eviction() {
  local rep pressure run_id path
  for ((rep = 1; rep <= REPS; rep++)); do
    for pressure in none state kv; do
      run_id="eviction-${pressure}-rep${rep}"
      path="${RESULT_DIR}/${run_id}.json"
      run_one "${path}" --mode eviction --run-id "${run_id}" \
        --pressure "${pressure}"
    done
  done
}

case "${STAGE}" in
  cost) run_cost ;;
  pair) run_pairs ;;
  batch) run_batch ;;
  eviction) run_eviction ;;
  all)
    run_cost
    run_pairs
    run_batch
    run_eviction
    ;;
  *)
    echo "usage: $0 [cost|pair|batch|eviction|all]" >&2
    exit 2
    ;;
esac
