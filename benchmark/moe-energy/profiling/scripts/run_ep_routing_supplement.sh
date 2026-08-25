#!/usr/bin/env bash
# Supplement missing EP forced-routing points (none backend) on node1-4 (32 GPUs).
set -euo pipefail

REPO_HOST="/mnt/workspace/lt/sglang-source/sglang"
REPO_CONTAINER="/workspace/sglang-source/sglang"
RAW_SUBDIR="benchmark/moe-energy/profiling/data/ep_routing_none_v2/raw_v2"
EXPORT_SUBDIR="benchmark/moe-energy/profiling/data/ep_routing_none_v2/export"
LOG_DIR="${REPO_HOST}/benchmark/moe-energy/profiling/data/ep_routing_none_v2"

cd "${REPO_HOST}"

BASE_ARGS=(
  --disable-custom-all-reduce
  --shape-token-limit 0
  --max-total-tokens 600000
  --warmup 10
  --repeat 50
)
PREFILL_EXTRA=(
  "${BASE_ARGS[@]}"
  --max-running-requests 1024
)
DECODE_EXTRA=(
  "${BASE_ARGS[@]}"
  --max-running-requests 4096
)

COMMON=(
  --nodes node1 node2 node3 node4
  --node-host node2=10.252.129.35
  --node-host node3=10.252.129.34
  --node-host node4=10.252.129.33
  --components F-EP
  --routing-modes balanced skewed_rank0
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

# ws2 prefill: finish partial length shards (resume JSONL per shard).
run_matrix "prefill ws2 shards" "${PREFILL_EXTRA[*]}" \
  --phases prefill \
  --sizes 2 \
  --shard-matrix

# ws4/ws8 prefill: resume monolithic JSONL for high-batch tail only.
run_matrix "prefill ws4/ws8 monolithic" "${PREFILL_EXTRA[*]}" \
  --phases prefill \
  --sizes 4 8 \
  --skip-existing-min-rows 417

# all decode sizes: resume monolithic JSONL for high-batch tail only.
run_matrix "decode ws2/4/8 monolithic" "${DECODE_EXTRA[*]}" \
  --phases decode \
  --sizes 2 4 8 \
  --skip-existing-min-rows 447

python3 benchmark/moe-energy/profiling/scripts/export_ep_routing_pf_df.py \
  --raw-root "${REPO_HOST}/${RAW_SUBDIR}" \
  --output-dir "${REPO_HOST}/${EXPORT_SUBDIR}"
