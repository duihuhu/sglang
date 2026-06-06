#!/bin/bash
# 8-GPU variable-length: full SLO sweep (TTFT: 2000, 2500, 3000; TPOT: 150)
# 4 schemes x 5 workloads x 3 SLO settings = 60 tests
set -e

PYTHON=/workspace/env/sglang-tier/bin/python
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WL_DIR="/workspace/sglang-tier/benchmark/energy_bench/workloads"
OUT_BASE="$SCRIPT_DIR/results/8gpu_var"
LOG_BASE="$SCRIPT_DIR/logs/8gpu_var"

export SGLANG_ENERGY_MODEL_DIR="/workspace/sglang-tier/benchmark/energy_bench/retrain/models_v2"

ALL_WL="$WL_DIR/workload_steady.jsonl,$WL_DIR/workload_varying.jsonl,$WL_DIR/workload_heavy.jsonl,$WL_DIR/workload_overload.jsonl,$WL_DIR/workload_tier1_demo.jsonl"

TPOT_SLO=150

echo "=========================================="
echo " 8-GPU Full SLO Sweep (TTFT: 2000/2500/3000)"
echo " Started: $(date)"
echo "=========================================="

for TTFT_SLO in 2000 2500 3000; do
    echo ""
    echo "=========================================="
    echo " TTFT SLO = ${TTFT_SLO}ms, TPOT SLO = ${TPOT_SLO}ms"
    echo " Started: $(date)"
    echo "=========================================="

    SUFFIX="slo${TTFT_SLO}"

    # Scheme 1: PDAF DynM (max freq)
    echo ">>> PDAF DynM (max) — TTFT_SLO=${TTFT_SLO}"
    $PYTHON "$SCRIPT_DIR/run_8gpu_deploy_bench.py" \
        --deploys pdaf_8g_dyn \
        --workloads "$ALL_WL" \
        --freq max \
        --ttft-slo-ms $TTFT_SLO \
        --tpot-slo-ms $TPOT_SLO \
        --output-dir "$OUT_BASE/pdaf_dyn_${SUFFIX}/json" \
        --log-dir "$LOG_BASE/pdaf_dyn_${SUFFIX}" \
        --max-run-s 600 \
        --force

    # Scheme 2: PDAF DynM + DVFS V2
    echo ">>> PDAF DynM + DVFS V2 — TTFT_SLO=${TTFT_SLO}"
    $PYTHON "$SCRIPT_DIR/run_8gpu_deploy_bench.py" \
        --deploys pdaf_8g_dyn_tier \
        --workloads "$ALL_WL" \
        --freq auto \
        --ttft-slo-ms $TTFT_SLO \
        --tpot-slo-ms $TPOT_SLO \
        --output-dir "$OUT_BASE/pdaf_dyn_tier_${SUFFIX}/json" \
        --log-dir "$LOG_BASE/pdaf_dyn_tier_${SUFFIX}" \
        --max-run-s 600 \
        --force

    # Scheme 3: PD DP4 (4x 1P1D disagg, max freq)
    echo ">>> PD DP4 (4x1P1D, max) — TTFT_SLO=${TTFT_SLO}"
    $PYTHON "$SCRIPT_DIR/run_8gpu_deploy_bench.py" \
        --deploys pd_dp4 \
        --workloads "$ALL_WL" \
        --freq max \
        --ttft-slo-ms $TTFT_SLO \
        --tpot-slo-ms $TPOT_SLO \
        --output-dir "$OUT_BASE/pd_dp4_${SUFFIX}/json" \
        --log-dir "$LOG_BASE/pd_dp4_${SUFFIX}" \
        --max-run-s 600 \
        --force

    # Scheme 4: Native TP=8 (max freq)
    echo ">>> Native TP=8 (max) — TTFT_SLO=${TTFT_SLO}"
    $PYTHON "$SCRIPT_DIR/run_8gpu_deploy_bench.py" \
        --deploys native_tp8 \
        --workloads "$ALL_WL" \
        --freq max \
        --ttft-slo-ms $TTFT_SLO \
        --tpot-slo-ms $TPOT_SLO \
        --output-dir "$OUT_BASE/native_tp8_${SUFFIX}/json" \
        --log-dir "$LOG_BASE/native_tp8_${SUFFIX}" \
        --max-run-s 600 \
        --force

    echo ">>> All done for TTFT_SLO=${TTFT_SLO} at $(date)"
done

echo ""
echo "=========================================="
echo " Full SLO sweep completed at $(date)"
echo "=========================================="
