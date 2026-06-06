#!/bin/bash
# TTFT SLO sweep: measure Prefill DVFS behavior under tightening TTFT constraints
# Uses GPU 4-7. V2 model only. TPOT SLO kept at 150ms (safe for V2).
set -e

PYTHON=/workspace/env/sglang-tier/bin/python
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH_SCRIPT="$BENCH_DIR/scripts/bench/run_4gpu_deploy_bench.py"
WL_DIR="$BENCH_DIR/workloads"
OUT_BASE="$SCRIPT_DIR/results_ttft_sweep"
LOG_BASE="$SCRIPT_DIR/logs_ttft_sweep"

V2_MODEL_DIR="$SCRIPT_DIR/models_v2"
export SGLANG_ENERGY_MODEL_DIR="$V2_MODEL_DIR"

WL_FILE="$WL_DIR/workload_steady.jsonl"

TTFT_SLOS=(5000 2000 1000 500 300 200)
TPOT_SLO=150

mkdir -p "$OUT_BASE" "$LOG_BASE"

echo "=========================================="
echo " TTFT SLO Sweep (Prefill DVFS): V2 model"
echo " TTFT SLOs: ${TTFT_SLOS[*]} ms"
echo " TPOT SLO: ${TPOT_SLO} ms (fixed)"
echo " GPU: 4-7"
echo " Started: $(date)"
echo "=========================================="

for TTFT in "${TTFT_SLOS[@]}"; do
    echo ""
    echo "############################################################"
    echo "# TTFT SLO = ${TTFT} ms (TPOT=${TPOT_SLO}ms)"
    echo "############################################################"

    echo ">>> [TTFT=${TTFT}ms] V2 Tier"
    $PYTHON "$BENCH_SCRIPT" \
        --deploys pdaf_4g_dyn_tier \
        --workloads "$WL_FILE" \
        --freq auto \
        --tpot-slo-ms "$TPOT_SLO" \
        --ttft-slo-ms "$TTFT" \
        --output-dir "$OUT_BASE/ttft_${TTFT}/json" \
        --log-dir "$LOG_BASE/ttft_${TTFT}" \
        --max-run-s 600 \
        --force
    echo ">>> [TTFT=${TTFT}ms] done at $(date)"
done

# Also run baseline (max freq) once for reference
echo ""
echo "############################################################"
echo "# Baseline (max freq, TTFT SLO irrelevant)"
echo "############################################################"
$PYTHON "$BENCH_SCRIPT" \
    --deploys pdaf_4g_dyn \
    --workloads "$WL_FILE" \
    --freq max \
    --tpot-slo-ms "$TPOT_SLO" \
    --ttft-slo-ms 5000 \
    --output-dir "$OUT_BASE/baseline/json" \
    --log-dir "$LOG_BASE/baseline" \
    --max-run-s 600 \
    --force
echo ">>> Baseline done at $(date)"

echo ""
echo "=========================================="
echo " TTFT SLO sweep completed at $(date)"
echo " Results: $OUT_BASE/"
echo " DVFS logs: $LOG_BASE/"
echo "=========================================="
