#!/usr/bin/env bash
# Full kernel-EP refresh on node1-4 (32 GPUs), no skip-existing.
set -euo pipefail

REPO_HOST="/mnt/workspace/lt/sglang-source/sglang"
REPO_CONTAINER="/workspace/sglang-source/sglang"
DATA_DIR="${REPO_HOST}/benchmark/moe-energy/profiling/data/kernel-EP"
RAW_SUBDIR="benchmark/moe-energy/profiling/data/kernel-EP/raw"
RAW_HOST="${REPO_HOST}/${RAW_SUBDIR}"
LOG_DIR="${REPO_HOST}/benchmark/moe-energy/profiling/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/kernel_ep_full_refresh_node1234_${STAMP}.log"

cd "${REPO_HOST}"
mkdir -p "${LOG_DIR}"

SYNC_FILES=(
  benchmark/moe-energy/profiling/scripts/profile_utils.py
  benchmark/moe-energy/profiling/scripts/bench_kernel_ep.py
  benchmark/moe-energy/profiling/scripts/bench_kernel_decode_af.py
  benchmark/moe-energy/profiling/scripts/bench_kernel_prefill_af.py
  benchmark/moe-energy/profiling/scripts/export_kernel_ep_pf_df.py
  benchmark/moe-energy/profiling/scripts/run_kernel_ep_distributed_matrix.py
  benchmark/moe-energy/profiling/scripts/run_distributed_profile_matrix.py
)

for host in 10.252.129.35 10.252.129.34 10.252.129.33; do
  rsync -a "${SYNC_FILES[@]}" "${host}:${REPO_HOST}/benchmark/moe-energy/profiling/scripts/"
done

# Stop stale kernel profilers on all four nodes.
docker exec moe-energy bash -lc "pkill -TERM -f 'benchmark/moe-energy/profiling/scripts/bench_kernel_' || true; pkill -TERM -f 'sgl-profile-' || true; sleep 2; pkill -KILL -f 'benchmark/moe-energy/profiling/scripts/bench_kernel_' || true; pkill -KILL -f 'multiprocessing.spawn' || true" || true
for host in 10.252.129.35 10.252.129.34 10.252.129.33; do
  ssh -o BatchMode=yes "${host}" \
    "docker exec moe-energy bash -lc \"pkill -TERM -f 'benchmark/moe-energy/profiling/scripts/bench_kernel_' || true; pkill -TERM -f 'sgl-profile-' || true; sleep 2; pkill -KILL -f 'benchmark/moe-energy/profiling/scripts/bench_kernel_' || true; pkill -KILL -f 'multiprocessing.spawn' || true\" || true"
done

# Wipe formal results locally and in each node's host/container view.
rm -rf "${RAW_HOST}" "${DATA_DIR}/pic"
rm -f "${DATA_DIR}"/PF-*.txt "${DATA_DIR}"/DF-*.txt
mkdir -p "${RAW_HOST}"/{node1,node2,node3,node4}

for spec in \
  '10.252.129.35 node2' \
  '10.252.129.34 node3' \
  '10.252.129.33 node4'; do
  read -r host node <<<"${spec}"
  ssh -o BatchMode=yes "${host}" \
    "rm -rf '${RAW_HOST:?}/${node}' && mkdir -p '${RAW_HOST}/${node}'; docker exec moe-energy bash -lc \"rm -rf '${REPO_CONTAINER}/${RAW_SUBDIR}/${node}' && mkdir -p '${REPO_CONTAINER}/${RAW_SUBDIR}/${node}'\""
done

echo "wiped formal kernel-EP raw + TSV + pic; kept doc.md and validation anchors" | tee -a "${LOG_FILE}"
echo "=== kernel-EP node1-4 full refresh ${STAMP} ===" | tee -a "${LOG_FILE}"

EXTRA_ARGS="--shape-token-limit 0 --max-total-tokens 600000 --warmup 10 --repeat 50 --max-running-requests 4096"
python3 benchmark/moe-energy/profiling/scripts/run_kernel_ep_distributed_matrix.py run \
  --nodes node1 node2 node3 node4 \
  --node-host node2=10.252.129.35 \
  --node-host node3=10.252.129.34 \
  --node-host node4=10.252.129.33 \
  --phases prefill decode \
  --sizes 2 4 8 \
  --routing-modes balanced middle_rank0 skewed_rank0 \
  --container moe-energy \
  --container-repo "${REPO_CONTAINER}" \
  --host-repo "${REPO_HOST}" \
  --raw-root "${RAW_HOST}" \
  --container-output-root "${RAW_SUBDIR}" \
  --extra-args "${EXTRA_ARGS}" \
  2>&1 | tee -a "${LOG_FILE}"

python3 benchmark/moe-energy/profiling/scripts/export_kernel_ep_pf_df.py \
  --raw-root "${RAW_HOST}" \
  --output-dir "${DATA_DIR}" \
  2>&1 | tee -a "${LOG_FILE}"

echo "=== done ${STAMP} ===" | tee -a "${LOG_FILE}"
