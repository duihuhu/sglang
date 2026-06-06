#!/bin/bash
# Continue SLO sweep: only run SLO=90/80/70 (300-100 already done)
set -e

PYTHON=/workspace/env/sglang-tier/bin/python
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH_SCRIPT="$BENCH_DIR/scripts/bench/run_4gpu_deploy_bench.py"
WL_DIR="$BENCH_DIR/workloads"
OUT_BASE="$SCRIPT_DIR/results_slo_sweep"
LOG_BASE="$SCRIPT_DIR/logs_slo_sweep"

V1_MODEL_DIR="/workspace/sglang/benchmark/test_motivation/energy_models"
V2_MODEL_DIR="$SCRIPT_DIR/models_v2"

WL_FILE="$WL_DIR/workload_steady.jsonl"

SLOS=(90 80 70)

mkdir -p "$OUT_BASE" "$LOG_BASE"

echo "=========================================="
echo " SLO Sweep (continued): ${SLOS[*]} ms"
echo " Started: $(date)"
echo "=========================================="

for SLO in "${SLOS[@]}"; do
    echo ""
    echo "############################################################"
    echo "# TPOT SLO = ${SLO} ms"
    echo "############################################################"

    echo ">>> [SLO=${SLO}ms] Baseline (max freq)"
    $PYTHON "$BENCH_SCRIPT" \
        --deploys pdaf_4g_dyn \
        --workloads "$WL_FILE" \
        --freq max \
        --tpot-slo-ms "$SLO" \
        --output-dir "$OUT_BASE/slo_${SLO}/baseline/json" \
        --log-dir "$LOG_BASE/slo_${SLO}/baseline" \
        --max-run-s 600 \
        --force
    echo ">>> [SLO=${SLO}ms] Baseline done at $(date)"

    echo ">>> [SLO=${SLO}ms] V1 (old model)"
    export SGLANG_ENERGY_MODEL_DIR="$V1_MODEL_DIR"
    $PYTHON "$BENCH_SCRIPT" \
        --deploys pdaf_4g_dyn_tier \
        --workloads "$WL_FILE" \
        --freq auto \
        --tpot-slo-ms "$SLO" \
        --output-dir "$OUT_BASE/slo_${SLO}/tier_v1/json" \
        --log-dir "$LOG_BASE/slo_${SLO}/tier_v1" \
        --max-run-s 600 \
        --force
    echo ">>> [SLO=${SLO}ms] V1 done at $(date)"

    echo ">>> [SLO=${SLO}ms] V2 (coupled model)"
    export SGLANG_ENERGY_MODEL_DIR="$V2_MODEL_DIR"
    $PYTHON "$BENCH_SCRIPT" \
        --deploys pdaf_4g_dyn_tier \
        --workloads "$WL_FILE" \
        --freq auto \
        --tpot-slo-ms "$SLO" \
        --output-dir "$OUT_BASE/slo_${SLO}/tier_v2/json" \
        --log-dir "$LOG_BASE/slo_${SLO}/tier_v2" \
        --max-run-s 600 \
        --force
    echo ">>> [SLO=${SLO}ms] V2 done at $(date)"
done

echo ""
echo "=========================================="
echo " SLO sweep (90/80/70) completed at $(date)"
echo "=========================================="
