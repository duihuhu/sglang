#!/usr/bin/env bash
# Kernel-only EP matrix on node3 + node4 in parallel (same span as data/EP/*.txt).
set -euo pipefail

REPO_HOST="/mnt/workspace/lt/sglang-source/sglang"
REPO_CONTAINER="/workspace/sglang-source/sglang"
RAW_SUBDIR="benchmark/moe-energy/profiling/data/kernel-EP/raw"

cd "${REPO_HOST}"

SYNC_FILES=(
  benchmark/moe-energy/profiling/scripts/profile_utils.py
  benchmark/moe-energy/profiling/scripts/bench_kernel_ep.py
  benchmark/moe-energy/profiling/scripts/bench_kernel_decode_af.py
  benchmark/moe-energy/profiling/scripts/bench_kernel_prefill_af.py
  benchmark/moe-energy/profiling/scripts/export_kernel_ep_pf_df.py
  benchmark/moe-energy/profiling/scripts/run_kernel_ep_distributed_matrix.py
)

for host in 10.252.129.34 10.252.129.33; do
  rsync -a "${SYNC_FILES[@]}" \
    "${host}:${REPO_HOST}/benchmark/moe-energy/profiling/scripts/"
done

EXTRA_ARGS="--shape-token-limit 0 --max-total-tokens 600000 --warmup 10 --repeat 50 --max-running-requests 4096"

python3 benchmark/moe-energy/profiling/scripts/run_kernel_ep_distributed_matrix.py run \
  --nodes node3 node4 \
  --node-host node3=10.252.129.34 \
  --node-host node4=10.252.129.33 \
  --phases prefill decode \
  --sizes 2 4 8 \
  --routing-modes balanced middle_rank0 skewed_rank0 \
  --skip-existing \
  --skip-existing-min-rows 58 \
  --container moe-energy \
  --container-repo "${REPO_CONTAINER}" \
  --host-repo "${REPO_HOST}" \
  --raw-root "${REPO_HOST}/${RAW_SUBDIR}" \
  --container-output-root "${RAW_SUBDIR}" \
  --extra-args "${EXTRA_ARGS}"
