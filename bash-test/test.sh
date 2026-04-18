#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   1) 前台调试（默认，能直接看到 job_worker 日志）:
#      bash bash-test/test.sh
#   2) 后台跑：
#      RUN_BG=1 bash bash-test/test.sh

REPO_DIR="/workspace/benchmark/sglang-main"
CONFIG_PATH="${CONFIG_PATH:-bash-test/pd_batch_config_d.json}"
RUN_BG="${RUN_BG:-0}"
LOG_PATH="${LOG_PATH:-bash-test/test_d.log}"
DRIVER_LOG_PATH="${DRIVER_LOG_PATH:-bash-test/test_pd_driver_d.log}"

CMD="/workspace/env/sglang-main/bin/python bash-test/batch_pd_nvtx_test.py --keep-processed --config ${CONFIG_PATH}"

if [[ "${RUN_BG}" == "1" ]]; then
  nohup bash -lc "cd ${REPO_DIR} && ${CMD} > ${LOG_PATH} 2>&1" > "${DRIVER_LOG_PATH}" 2>&1 &
  echo "[started] background pid=$!, config=${CONFIG_PATH}"
  echo "[log] ${LOG_PATH}"
  echo "[driver_log] ${DRIVER_LOG_PATH}"
else
  cd "${REPO_DIR}"
  ${CMD} 2>&1 | tee "${LOG_PATH}"
fi
