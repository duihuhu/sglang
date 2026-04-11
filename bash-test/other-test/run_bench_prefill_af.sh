#!/usr/bin/env bash
# bench_prefill_af 启动包装：先编 NVML .so，再 sudo 跑（SM 锁频需 root）。
# 用法:
#   ./run_bench_prefill_af.sh
#   MODEL_PATH=/path/to/model ./run_bench_prefill_af.sh --no-resume
# 环境变量:
#   BENCH_SM_LOCK_SETTLE_S  锁 SM 后等待秒数（默认 0.2）
# 其它模式见脚本末尾注释，或直接用 python3 bench_prefill_af.py --help
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
NVML_DIR="${REPO_ROOT}/bash-test/C_nvml_for_energy"

"${NVML_DIR}/build.sh"

export BENCH_SM_LOCK_SETTLE_S="${BENCH_SM_LOCK_SETTLE_S:-0.2}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-32B}"
TP_SIZE="${TP_SIZE:-1}"

cd "${SCRIPT_DIR}"

exec sudo -E env "PATH=${PATH}" \
  python3 bench_prefill_af.py \
  --model-path "${MODEL_PATH}" \
  --load-format dummy \
  --tp-size "${TP_SIZE}" \
  --quick \
  "$@"
