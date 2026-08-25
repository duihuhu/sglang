#!/usr/bin/env bash
# Wait for kernel-EP matrix on node3/node4, then refresh data/EP (F-EP, 3 routings).
set -euo pipefail

REPO_HOST="/mnt/workspace/lt/sglang-source/sglang"
REPO_CONTAINER="/workspace/sglang-source/sglang"
LOG_DIR="${REPO_HOST}/benchmark/moe-energy/profiling/data"
KERNEL_LOG="${LOG_DIR}/kernel-EP/node34_launch.log"
EP_LOG="${LOG_DIR}/EP/refresh_node34.log"

cd "${REPO_HOST}"

echo "Waiting for kernel-EP scheduler to finish..."
while pgrep -f "run_kernel_ep_distributed_matrix.py run" >/dev/null 2>&1; do
  sleep 30
  tail -1 "${KERNEL_LOG}" 2>/dev/null || true
done
echo "Kernel-EP scheduler exited."

SYNC_FILES=(
  benchmark/moe-energy/profiling/scripts/profile_utils.py
  benchmark/moe-energy/profiling/scripts/export_ep_routing_pf_df.py
  benchmark/moe-energy/profiling/scripts/run_ep_refresh_distributed.py
  benchmark/moe-energy/profiling/scripts/run_distributed_profile_matrix.py
)

for host in 10.252.129.34 10.252.129.33; do
  rsync -a "${SYNC_FILES[@]}" \
    "${host}:${REPO_HOST}/benchmark/moe-energy/profiling/scripts/"
done

python3 benchmark/moe-energy/profiling/scripts/run_ep_refresh_distributed.py run \
  --nodes node3 node4 \
  --node-host node3=10.252.129.34 \
  --node-host node4=10.252.129.33 \
  --phases prefill decode \
  --sizes 2 4 8 \
  --routing-modes balanced middle_rank0 skewed_rank0 \
  --container moe-energy \
  --container-repo "${REPO_CONTAINER}" \
  --host-repo "${REPO_HOST}" \
  --raw-root "${REPO_HOST}/benchmark/moe-energy/profiling/data/EP/raw_v2" \
  --container-output-root benchmark/moe-energy/profiling/data/EP/raw_v2 \
  2>&1 | tee "${EP_LOG}"

echo "EP refresh complete. TSV under ${REPO_HOST}/benchmark/moe-energy/profiling/data/EP/"
