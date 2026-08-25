#!/usr/bin/env bash
# Tail supplement: ~162 missing high-batch points on node2/node3/node4 only.
set -euo pipefail

REPO_HOST="/mnt/workspace/lt/sglang-source/sglang"
REPO_CONTAINER="/workspace/sglang-source/sglang"
RAW_SUBDIR="benchmark/moe-energy/profiling/data/ep_routing_none_v2/raw_v2"
EXPORT_SUBDIR="benchmark/moe-energy/profiling/data/ep_routing_none_v2/export"

cd "${REPO_HOST}"

# node4 may lag on NFS; sync profiling scripts before launch.
for host in 10.252.129.33; do
  rsync -a benchmark/moe-energy/profiling/scripts/profile_utils.py \
    benchmark/moe-energy/profiling/scripts/run_distributed_profile_matrix.py \
    "${host}:${REPO_HOST}/benchmark/moe-energy/profiling/scripts/"
done

COMMON=(
  --nodes node2 node3 node4
  --node-host node2=10.252.129.35
  --node-host node3=10.252.129.34
  --node-host node4=10.252.129.33
  --components F-EP
  --ep-a2a-backend none
  --skip-existing
  --no-export-compat
  --container moe-energy
  --container-repo "${REPO_CONTAINER}"
  --host-repo "${REPO_HOST}"
  --raw-root "${REPO_HOST}/${RAW_SUBDIR}"
  --container-output-root "${RAW_SUBDIR}"
)

run_matrix() {
  local label="$1"
  local extra_args="$2"
  shift 2
  echo "===== ${label} ====="
  python3 benchmark/moe-energy/profiling/scripts/run_distributed_profile_matrix.py run \
    "${COMMON[@]}" --extra-args "${extra_args}" "$@"
}

PREFILL_EXTRA="--disable-custom-all-reduce --shape-token-limit 0 --max-total-tokens 600000 --warmup 10 --repeat 50 --max-running-requests 1024"
DECODE_EXTRA="--disable-custom-all-reduce --shape-token-limit 0 --max-total-tokens 600000 --warmup 10 --repeat 50 --max-running-requests 4096"

# ws2 prefill tail: batch=1024 and long-context large batches (resume monolithic JSONL).
run_matrix "prefill ws2 tail" "${PREFILL_EXTRA}" \
  --phases prefill \
  --sizes 2 \
  --routing-modes balanced skewed_rank0 \
  --skip-existing-min-rows 420

# ws8 prefill balanced only: len=512 batch=1024.
run_matrix "prefill ws8 balanced tail" "${PREFILL_EXTRA}" \
  --phases prefill \
  --sizes 8 \
  --routing-modes balanced \
  --skip-existing-min-rows 420

# ws2 decode tail: high batch shapes (resume monolithic JSONL).
run_matrix "decode ws2 tail" "${DECODE_EXTRA}" \
  --phases decode \
  --sizes 2 \
  --routing-modes balanced skewed_rank0 \
  --skip-existing-min-rows 456

python3 benchmark/moe-energy/profiling/scripts/export_ep_routing_pf_df.py \
  --raw-root "${REPO_HOST}/${RAW_SUBDIR}" \
  --output-dir "${REPO_HOST}/${EXPORT_SUBDIR}"
