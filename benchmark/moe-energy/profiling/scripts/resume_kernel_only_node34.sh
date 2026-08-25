#!/usr/bin/env bash
# Resume kernel-EP gaps only (no data/EP full-stack refresh).
set -euo pipefail

REPO_HOST="/mnt/workspace/lt/sglang-source/sglang"
REPO_CONTAINER="/workspace/sglang-source/sglang"
LOG_DIR="${REPO_HOST}/benchmark/moe-energy/profiling/data"
KERNEL_RAW="${LOG_DIR}/kernel-EP/raw"

cd "${REPO_HOST}"

SYNC_FILES=(
  benchmark/moe-energy/profiling/scripts/profile_utils.py
  benchmark/moe-energy/profiling/scripts/bench_kernel_ep.py
  benchmark/moe-energy/profiling/scripts/bench_kernel_decode_af.py
  benchmark/moe-energy/profiling/scripts/bench_kernel_prefill_af.py
  benchmark/moe-energy/profiling/scripts/export_kernel_ep_pf_df.py
  benchmark/moe-energy/profiling/scripts/run_kernel_ep_distributed_matrix.py
  benchmark/moe-energy/profiling/scripts/run_distributed_profile_matrix.py
)

for host in 10.252.129.34 10.252.129.33; do
  rsync -a "${SYNC_FILES[@]}" \
    "${host}:${REPO_HOST}/benchmark/moe-energy/profiling/scripts/"
done

KERNEL_EXTRA="--shape-token-limit 0 --max-total-tokens 600000 --warmup 10 --repeat 50 --max-running-requests 4096"

echo "===== $(date -Is) KERNEL-EP ONLY RESUME (prefill ws2 gaps) ====="
python3 benchmark/moe-energy/profiling/scripts/run_kernel_ep_distributed_matrix.py run \
  --nodes node3 node4 \
  --node-host node3=10.252.129.34 \
  --node-host node4=10.252.129.33 \
  --phases prefill \
  --sizes 2 \
  --routing-modes balanced middle_rank0 skewed_rank0 \
  --skip-existing \
  --skip-existing-min-rows 58 \
  --base-nccl-port 29750 \
  --container moe-energy \
  --container-repo "${REPO_CONTAINER}" \
  --host-repo "${REPO_HOST}" \
  --raw-root "${KERNEL_RAW}" \
  --container-output-root benchmark/moe-energy/profiling/data/kernel-EP/raw \
  --extra-args "${KERNEL_EXTRA}"

python3 benchmark/moe-energy/profiling/scripts/export_kernel_ep_pf_df.py \
  --raw-root "${KERNEL_RAW}" \
  --output-dir "${LOG_DIR}/kernel-EP"

echo "===== $(date -Is) KERNEL-EP DONE ====="
