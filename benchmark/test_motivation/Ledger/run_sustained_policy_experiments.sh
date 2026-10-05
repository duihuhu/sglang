#!/usr/bin/env bash
set -euo pipefail

policy_label=$1
repo_dir=${2:-/workspace/sglang}
output_dir=${3:-${repo_dir}/benchmark/test_motivation/Ledger/results/policy_20260820}
repetitions=${4:-3}
python_bin=${repo_dir}/.venv/bin/python
harness=${repo_dir}/benchmark/test_motivation/Ledger/validate_hybrid_motivation.py

mkdir -p "${output_dir}"

for repetition in $(seq 1 "${repetitions}"); do
  run_id="policy-${policy_label}-r${repetition}"
  output="${output_dir}/${run_id}.json"
  echo "START ${run_id}"
  "${python_bin}" "${harness}" \
    --mode fairness-sustained \
    --order state-first \
    --run-id "${run_id}" \
    --output "${output}" \
    --waves 12 \
    --state-per-wave 16 \
    --kv-per-wave 1 \
    --wave-interval-ms 750 \
    --arrival-gap-ms 1 \
    --slo-seconds 20 \
    >"${output%.json}.stdout"
  echo "DONE ${run_id}"
done
