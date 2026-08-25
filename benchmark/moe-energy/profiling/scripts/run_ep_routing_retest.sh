#!/usr/bin/env bash
# EP forced-routing retest (none backend) on node2/node3 (node1 reserved for manual testing).
set -euo pipefail

REPO_HOST="/mnt/workspace/lt/sglang-source/sglang"
REPO_CONTAINER="/workspace/sglang-source/sglang"
RAW_SUBDIR="benchmark/moe-energy/profiling/data/ep_routing_none_v2/raw_v2"
EXPORT_SUBDIR="benchmark/moe-energy/profiling/data/ep_routing_none_v2/export"

cd "${REPO_HOST}"

EXTRA_ARGS=(
  --disable-custom-all-reduce
  --shape-token-limit 0
  --max-total-tokens 600000
  --max-running-requests 512
  --warmup 10
  --repeat 50
)

python3 benchmark/moe-energy/profiling/scripts/run_distributed_profile_matrix.py run \
  --nodes node2 node3 \
  --node-host node2=10.252.129.35 \
  --node-host node3=10.252.129.34 \
  --phases prefill decode \
  --components F-EP \
  --sizes 2 4 8 \
  --routing-modes balanced skewed_rank0 \
  --ep-a2a-backend none \
  --skip-existing \
  --shard-matrix \
  --no-export-compat \
  --container moe-energy \
  --container-repo "${REPO_CONTAINER}" \
  --host-repo "${REPO_HOST}" \
  --raw-root "${REPO_HOST}/${RAW_SUBDIR}" \
  --container-output-root "${RAW_SUBDIR}" \
  --extra-args "$(printf '%s ' "${EXTRA_ARGS[@]}")"

python3 benchmark/moe-energy/profiling/scripts/export_ep_routing_pf_df.py \
  --raw-root "${REPO_HOST}/${RAW_SUBDIR}" \
  --output-dir "${REPO_HOST}/${EXPORT_SUBDIR}"
