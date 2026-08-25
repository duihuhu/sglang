#!/usr/bin/env bash
# Full kernel-EP matrix refresh on node3 + node4 (no --skip-existing).
# Matrix skips / OOM boundaries: data/kernel-EP/doc.md
set -euo pipefail

REPO_HOST="/mnt/workspace/lt/sglang-source/sglang"
REPO_CONTAINER="/workspace/sglang-source/sglang"
RAW_SUBDIR="benchmark/moe-energy/profiling/data/kernel-EP/raw"
LOG_DIR="${REPO_HOST}/benchmark/moe-energy/profiling/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/kernel_ep_full_refresh_${STAMP}.log"

cd "${REPO_HOST}"
mkdir -p "${LOG_DIR}"

RAW_HOST="${REPO_HOST}/${RAW_SUBDIR}"
DATA_DIR="${REPO_HOST}/benchmark/moe-energy/profiling/data/kernel-EP"

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

# Kill stale workers and wipe prior jsonl on both nodes (NFS + docker view).
for host in 10.252.129.34 10.252.129.33; do
  ssh -o BatchMode=yes "${host}" bash -s <<EOF
set -euo pipefail
docker exec moe-energy bash -lc 'pkill -f bench_kernel || true; pkill -f run_kernel_ep || true' || true
rm -f "${RAW_HOST}/node3/kernel_"*.jsonl "${RAW_HOST}/node4/kernel_"*.jsonl
docker exec moe-energy bash -lc 'rm -f "${REPO_CONTAINER}/${RAW_SUBDIR}/node3/kernel_"*.jsonl "${REPO_CONTAINER}/${RAW_SUBDIR}/node4/kernel_"*.jsonl' || true
EOF
done

rm -f "${DATA_DIR}"/PF-*.txt "${DATA_DIR}"/DF-*.txt
rm -f "${RAW_HOST}/node3/kernel_"*.jsonl "${RAW_HOST}/node4/kernel_"*.jsonl
rm -f "${RAW_HOST}/manifest.jsonl"
mkdir -p "${RAW_HOST}/node3" "${RAW_HOST}/node4"

echo "wiped old kernel-EP raw + TSV on node3/node4 and controller" | tee -a "${LOG_FILE}"

EXTRA_ARGS="--shape-token-limit 0 --max-total-tokens 600000 --warmup 10 --repeat 50 --max-running-requests 4096"

echo "=== kernel-EP full refresh ${STAMP} ===" | tee -a "${LOG_FILE}"

python3 benchmark/moe-energy/profiling/scripts/run_kernel_ep_distributed_matrix.py run \
  --nodes node3 node4 \
  --node-host node3=10.252.129.34 \
  --node-host node4=10.252.129.33 \
  --phases prefill decode \
  --sizes 2 4 8 \
  --routing-modes balanced middle_rank0 skewed_rank0 \
  --container moe-energy \
  --container-repo "${REPO_CONTAINER}" \
  --host-repo "${REPO_HOST}" \
  --raw-root "${REPO_HOST}/${RAW_SUBDIR}" \
  --container-output-root "${RAW_SUBDIR}" \
  --extra-args "${EXTRA_ARGS}" \
  2>&1 | tee -a "${LOG_FILE}"

python3 benchmark/moe-energy/profiling/scripts/export_kernel_ep_pf_df.py \
  --raw-root "${REPO_HOST}/benchmark/moe-energy/profiling/data/kernel-EP/raw" \
  --output-dir "${REPO_HOST}/benchmark/moe-energy/profiling/data/kernel-EP" \
  2>&1 | tee -a "${LOG_FILE}"

echo "=== done ${STAMP} ===" | tee -a "${LOG_FILE}"
