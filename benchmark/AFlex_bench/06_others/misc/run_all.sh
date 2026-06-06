#!/bin/bash
# run_all.sh — Tier2 DVFS benchmark (PD+AF C++ IPC, GPU 0-3)
#
# Usage:
#   bash run_all.sh
#   bash run_all.sh --skip-baseline
#   bash run_all.sh --skip-dvfs

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/workspace/env/sglang-tier/bin/python"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "  Tier2 DVFS Energy Benchmark"
echo "  Mode: PD+AF M=1 (C++ IPC backend)"
echo "  GPUs: 0,1,2,3 (A800-SXM4-80GB)"
echo "  Model: /models/Qwen3-0.6B"
echo "=============================================="
echo ""

# Step 1: Generate workloads
echo "[Step 1] Generating workload traces..."
$PYTHON gen_workload.py --output-dir workloads --mode all
echo ""

# Step 2: Run A/B benchmark
echo "[Step 2] Running A/B benchmark (baseline vs DVFS)..."
$PYTHON run_tier2_bench.py \
    --workload workloads/workload_varying.jsonl \
    --output-dir results \
    "$@"

echo ""
echo "[Done] Results in results/"
