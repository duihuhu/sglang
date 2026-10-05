#!/usr/bin/env bash
set -euo pipefail

repo_dir=${1:-/workspace/sglang}
output_dir=${2:-${repo_dir}/benchmark/test_motivation/Ledger/results/exact_20260820}
repetitions=${3:-3}
python_bin=${repo_dir}/.venv/bin/python
harness=${repo_dir}/benchmark/test_motivation/Ledger/validate_hybrid_motivation.py

mkdir -p "${output_dir}"

run_condition() {
  local condition=$1
  local repetition=$2
  local order=$3
  local state_count=$4
  local kv_count=$5
  local run_id="exact-${condition}-r${repetition}"
  local output="${output_dir}/${run_id}.json"

  echo "START ${run_id}"
  "${python_bin}" "${harness}" \
    --mode fairness-pair \
    --order "${order}" \
    --run-id "${run_id}" \
    --output "${output}" \
    --state-count "${state_count}" \
    --kv-count "${kv_count}" \
    >"${output%.json}.stdout"
  echo "DONE ${run_id}"
}

for repetition in $(seq 1 "${repetitions}"); do
  run_condition state-isolated "${repetition}" state-first 96 0
  run_condition kv-isolated "${repetition}" kv-first 0 6
  run_condition state-first "${repetition}" state-first 96 6
  run_condition kv-first "${repetition}" kv-first 96 6
  run_condition interleave "${repetition}" interleave 96 6
done
