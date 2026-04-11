#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# CUDA NVML header; system libnvidia-ml for actual driver NVML
CUDA_ROOT="${CUDA_HOME:-/usr/local/cuda}"
NVML_INC="${CUDA_ROOT}/include"

if [[ ! -f "${NVML_INC}/nvml.h" ]]; then
  echo "nvml.h not found under ${NVML_INC}; set CUDA_HOME." >&2
  exit 1
fi

g++ -std=c++17 -O2 -fPIC -shared \
  -I"${NVML_INC}" \
  dvfs_ctrl.cpp \
  -o libnvml_energy.so \
  -L/usr/lib/x86_64-linux-gnu \
  -lnvidia-ml

echo "Built: ${DIR}/libnvml_energy.so"
ls -l libnvml_energy.so
