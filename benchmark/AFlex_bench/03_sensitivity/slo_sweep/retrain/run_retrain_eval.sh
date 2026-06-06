#!/bin/bash
# Retrain Model Evaluation: Compare V1 (old) vs V2 (coupled) DVFS models
# in a real PDAF Tier workload test on 4 GPUs (4-7).
#
# Tests:
#   1. PDAF DynM + Tier (V1 model) — old per-layer independent models
#   2. PDAF DynM + Tier (V2 model) — new coupled iteration-level models
#   3. PDAF DynM (max freq, no DVFS) — baseline
#
# For each, run the same workload and compare:
#   - TTFT, TPOT, Throughput
#   - Total Energy, Energy per token
#   - SLO violation rate
#   - DVFS decision logs (frequency choices)
#
# Usage:
#   bash run_retrain_eval.sh
#   bash run_retrain_eval.sh --workload steady  # specific workload
set -e

PYTHON=/workspace/env/sglang-tier/bin/python
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH_SCRIPT="$BENCH_DIR/scripts/bench/run_4gpu_deploy_bench.py"
WL_DIR="$BENCH_DIR/workloads"
OUT_BASE="$SCRIPT_DIR/results"
LOG_BASE="$SCRIPT_DIR/logs"

# Model directories
V1_MODEL_DIR="/workspace/sglang/benchmark/test_motivation/energy_models"
V2_MODEL_DIR="$SCRIPT_DIR/models_v2"

mkdir -p "$OUT_BASE" "$LOG_BASE"

# Parse args
WORKLOAD="${1:-steady}"
case "$WORKLOAD" in
    steady)  WL_FILE="$WL_DIR/workload_steady.jsonl" ;;
    varying) WL_FILE="$WL_DIR/workload_varying.jsonl" ;;
    heavy)   WL_FILE="$WL_DIR/workload_heavy.jsonl" ;;
    *)       WL_FILE="$WORKLOAD" ;;
esac

echo "=========================================="
echo " Retrain Model Evaluation"
echo " Workload: $WL_FILE"
echo " V1 models: $V1_MODEL_DIR"
echo " V2 models: $V2_MODEL_DIR"
echo " Started: $(date)"
echo "=========================================="

# --- Test 1: Baseline (max freq, no DVFS) ---
echo ""
echo ">>> [1/3] PDAF DynM Baseline (max freq, no DVFS)"
$PYTHON "$BENCH_SCRIPT" \
    --deploys pdaf_4g_dyn \
    --workloads "$WL_FILE" \
    --freq max \
    --tpot-slo-ms 90 \
    --output-dir "$OUT_BASE/baseline/json" \
    --log-dir "$LOG_BASE/baseline" \
    --max-run-s 600 \
    --force
echo ">>> Baseline done at $(date)"

# --- Test 2: Tier with V1 model ---
echo ""
echo ">>> [2/3] PDAF DynM + Tier (V1 old model)"
# Temporarily override ENERGY_MODEL_DIR for V1
export SGLANG_ENERGY_MODEL_DIR="$V1_MODEL_DIR"
$PYTHON "$BENCH_SCRIPT" \
    --deploys pdaf_4g_dyn_tier \
    --workloads "$WL_FILE" \
    --freq auto \
    --tpot-slo-ms 90 \
    --output-dir "$OUT_BASE/tier_v1/json" \
    --log-dir "$LOG_BASE/tier_v1" \
    --max-run-s 600 \
    --force
echo ">>> Tier V1 done at $(date)"

# --- Test 3: Tier with V2 coupled model ---
echo ""
echo ">>> [3/3] PDAF DynM + Tier (V2 coupled model)"
export SGLANG_ENERGY_MODEL_DIR="$V2_MODEL_DIR"
$PYTHON "$BENCH_SCRIPT" \
    --deploys pdaf_4g_dyn_tier \
    --workloads "$WL_FILE" \
    --freq auto \
    --tpot-slo-ms 90 \
    --output-dir "$OUT_BASE/tier_v2/json" \
    --log-dir "$LOG_BASE/tier_v2" \
    --max-run-s 600 \
    --force
echo ">>> Tier V2 done at $(date)"

echo ""
echo "=========================================="
echo " All tests completed at $(date)"
echo " Results in: $OUT_BASE/"
echo " DVFS logs in: $LOG_BASE/"
echo "=========================================="

# Run analysis
echo ""
echo ">>> Running analysis..."
$PYTHON "$SCRIPT_DIR/analyze_retrain_eval.py" \
    --results-dir "$OUT_BASE" \
    --logs-dir "$LOG_BASE"
