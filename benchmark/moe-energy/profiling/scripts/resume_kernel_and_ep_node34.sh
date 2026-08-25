#!/usr/bin/env bash
# Resume kernel-EP gaps then refresh data/EP on node3 + node4.
set -euo pipefail

REPO_HOST="/mnt/workspace/lt/sglang-source/sglang"
REPO_CONTAINER="/workspace/sglang-source/sglang"
LOG_DIR="${REPO_HOST}/benchmark/moe-energy/profiling/data"
KERNEL_RAW="${LOG_DIR}/kernel-EP/raw"
EP_RAW="${LOG_DIR}/EP/raw_v2"
ORCH_LOG="${LOG_DIR}/resume_node34_orchestrator.log"

cd "${REPO_HOST}"

SYNC_FILES=(
  benchmark/moe-energy/profiling/scripts/profile_utils.py
  benchmark/moe-energy/profiling/scripts/bench_kernel_ep.py
  benchmark/moe-energy/profiling/scripts/bench_kernel_decode_af.py
  benchmark/moe-energy/profiling/scripts/bench_kernel_prefill_af.py
  benchmark/moe-energy/profiling/scripts/bench_decode_af.py
  benchmark/moe-energy/profiling/scripts/bench_prefill_af.py
  benchmark/moe-energy/profiling/scripts/export_kernel_ep_pf_df.py
  benchmark/moe-energy/profiling/scripts/export_ep_routing_pf_df.py
  benchmark/moe-energy/profiling/scripts/run_kernel_ep_distributed_matrix.py
  benchmark/moe-energy/profiling/scripts/run_ep_refresh_distributed.py
  benchmark/moe-energy/profiling/scripts/run_distributed_profile_matrix.py
)

for host in 10.252.129.34 10.252.129.33; do
  rsync -a "${SYNC_FILES[@]}" \
    "${host}:${REPO_HOST}/benchmark/moe-energy/profiling/scripts/"
done

# Stale local orphans (not node3/node4) should not affect export.
rm -f "${KERNEL_RAW}"/kernel_*.jsonl 2>/dev/null || true

KERNEL_EXTRA="--shape-token-limit 0 --max-total-tokens 600000 --warmup 10 --repeat 50 --max-running-requests 4096"
EP_EXTRA="--shape-token-limit 0 --max-total-tokens 600000 --warmup 10 --repeat 50 --max-running-requests 4096"

echo "===== $(date -Is) KERNEL-EP RESUME ====="
python3 benchmark/moe-energy/profiling/scripts/run_kernel_ep_distributed_matrix.py run \
  --nodes node3 node4 \
  --node-host node3=10.252.129.34 \
  --node-host node4=10.252.129.33 \
  --phases prefill decode \
  --sizes 2 4 8 \
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

echo "===== $(date -Is) data/EP F-EP REFRESH ====="
python3 benchmark/moe-energy/profiling/scripts/run_ep_refresh_distributed.py run \
  --nodes node3 node4 \
  --node-host node3=10.252.129.34 \
  --node-host node4=10.252.129.33 \
  --phases prefill decode \
  --sizes 2 4 8 \
  --routing-modes balanced middle_rank0 skewed_rank0 \
  --skip-existing \
  --skip-existing-min-rows 58 \
  --base-nccl-port 29850 \
  --container moe-energy \
  --container-repo "${REPO_CONTAINER}" \
  --host-repo "${REPO_HOST}" \
  --raw-root "${EP_RAW}" \
  --container-output-root benchmark/moe-energy/profiling/data/EP/raw_v2 \
  --extra-args "${EP_EXTRA}"

python3 benchmark/moe-energy/profiling/scripts/export_ep_routing_pf_df.py \
  --raw-root "${EP_RAW}" \
  --output-dir "${LOG_DIR}/EP"

echo "===== $(date -Is) ALL DONE ====="
