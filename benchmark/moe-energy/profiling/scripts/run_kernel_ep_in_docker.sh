#!/usr/bin/env bash
# Run kernel-only EP matrix inside the moe-energy container.
set -euo pipefail
CONTAINER="${CONTAINER:-moe-energy}"
REPO_IN_CONTAINER="${REPO_IN_CONTAINER:-/workspace/sglang-source/sglang}"
MODEL_PATH="${MODEL_PATH:-/models/Qwen3-30B-A3B}"
VISIBLE_GPUS="${VISIBLE_GPUS:-4,5,6,7}"
QUICK="${QUICK:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
EXTRA_ARGS=()
if [[ "${QUICK}" == "1" ]]; then
  EXTRA_ARGS+=(--quick)
fi
if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
  EXTRA_ARGS+=(--continue-on-error)
fi

docker exec "${CONTAINER}" bash -lc "
  set -euo pipefail
  export FLASHINFER_DISABLE_VERSION_CHECK=1
  cd '${REPO_IN_CONTAINER}'
  python3 benchmark/moe-energy/profiling/scripts/run_kernel_ep_matrix.py \
    --model-path '${MODEL_PATH}' \
    --visible-gpus '${VISIBLE_GPUS}' \
    ${EXTRA_ARGS[*]:-}
"
