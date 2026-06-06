#!/bin/bash
# 4-GPU sweep V2: variable-length workloads, 4 schemes, GPUs 4-7
# TTFT SLO uses pure processing time (excludes queue wait)
set -e

PYTHON=/workspace/env/sglang-tier/bin/python
export SGLANG_ENERGY_MODEL_DIR="/workspace/sglang-tier/benchmark/energy_bench/retrain/models_v2"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WL_DIR="/workspace/sglang-tier/benchmark/energy_bench/workloads"
OUT_BASE="$SCRIPT_DIR/results/4gpu_var_v2"
LOG_BASE="$SCRIPT_DIR/logs/4gpu_var_v2"

mkdir -p "$OUT_BASE" "$LOG_BASE"

ALL_WL="$WL_DIR/workload_steady.jsonl,$WL_DIR/workload_varying.jsonl,$WL_DIR/workload_heavy.jsonl,$WL_DIR/workload_overload.jsonl,$WL_DIR/workload_tier1_demo.jsonl"

TTFT_SLO=2000
TPOT_SLO=150

echo "=========================================="
echo " 4-GPU Var-Len Sweep V2 (GPUs 4-7)"
echo " TTFT SLO=${TTFT_SLO}ms (pure processing, no queue)"
echo " TPOT SLO=${TPOT_SLO}ms"
echo " Schemes: PDAF DynM, PDAF Tier+DynM, PD DP2, Native DP4"
echo " Started: $(date)"
echo "=========================================="

# Scheme 1: PDAF Dynamic M (max freq)
echo ""
echo ">>> [1/4] PDAF DynM (max_freq)"
$PYTHON "$SCRIPT_DIR/run_4gpu_deploy_bench.py" \
    --deploys pdaf_4g_dyn \
    --workloads "$ALL_WL" \
    --freq max \
    --ttft-slo-ms $TTFT_SLO \
    --tpot-slo-ms $TPOT_SLO \
    --output-dir "$OUT_BASE/pdaf_dyn/json" \
    --log-dir "$LOG_BASE/pdaf_dyn" \
    --max-run-s 600 \
    --force
echo ">>> PDAF DynM done at $(date)"

# Scheme 2: PDAF Tier + Dynamic M (DVFS)
echo ""
echo ">>> [2/4] PDAF Tier+DynM (DVFS)"
$PYTHON "$SCRIPT_DIR/run_4gpu_deploy_bench.py" \
    --deploys pdaf_4g_dyn_tier \
    --workloads "$ALL_WL" \
    --freq auto \
    --ttft-slo-ms $TTFT_SLO \
    --tpot-slo-ms $TPOT_SLO \
    --output-dir "$OUT_BASE/pdaf_dyn_tier/json" \
    --log-dir "$LOG_BASE/pdaf_dyn_tier" \
    --max-run-s 600 \
    --force
echo ">>> PDAF Tier+DynM done at $(date)"

# Scheme 3: PD DP2 (PD-separated 1P1D, TP=1)
echo ""
echo ">>> [3/4] PD DP2 (1P1D, max_freq)"
$PYTHON "$SCRIPT_DIR/run_4gpu_deploy_bench.py" \
    --deploys pd_dp2_disagg \
    --workloads "$ALL_WL" \
    --freq max \
    --ttft-slo-ms $TTFT_SLO \
    --tpot-slo-ms $TPOT_SLO \
    --output-dir "$OUT_BASE/pd_dp2/json" \
    --log-dir "$LOG_BASE/pd_dp2" \
    --max-run-s 600 \
    --force
echo ">>> PD DP2 done at $(date)"

# Scheme 4: Native DP4 (4x TP=1 full instances)
echo ""
echo ">>> [4/4] Native DP4 (max_freq)"
$PYTHON "$SCRIPT_DIR/run_4gpu_deploy_bench.py" \
    --deploys native_dp4 \
    --workloads "$ALL_WL" \
    --freq max \
    --ttft-slo-ms $TTFT_SLO \
    --tpot-slo-ms $TPOT_SLO \
    --output-dir "$OUT_BASE/native_dp4/json" \
    --log-dir "$LOG_BASE/native_dp4" \
    --max-run-s 600 \
    --force
echo ">>> Native DP4 done at $(date)"

echo ""
echo "=========================================="
echo " All V2 tests complete at $(date)"
echo "=========================================="
