#!/bin/bash
# 8-GPU variable-length workload sweep: 4 schemes, 5 workloads
# Schemes:
#   1. pdaf_8g_dyn       - PDAF(TP=2) + DynM (threshold=128), max freq
#   2. pdaf_8g_dyn_tier  - PDAF(TP=2) + DynM (threshold=128) + DVFS V2
#   3. pd_dp4            - PD DP=4 (4x TP=2 full instances), max freq
#   4. native_tp8        - Native TP=8 (no disaggregation), max freq
#
# SLO: TTFT=2000ms, TPOT=150ms
# Workloads: steady, varying, heavy, overload, tier1_demo
set -e

PYTHON=/workspace/env/sglang-tier/bin/python
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WL_DIR="/workspace/sglang-tier/benchmark/energy_bench/workloads"
OUT_BASE="$SCRIPT_DIR/results/8gpu_var"
LOG_BASE="$SCRIPT_DIR/logs/8gpu_var"

# V2 energy model
export SGLANG_ENERGY_MODEL_DIR="/workspace/sglang-tier/benchmark/energy_bench/retrain/models_v2"

mkdir -p "$OUT_BASE" "$LOG_BASE"

ALL_WL="$WL_DIR/workload_steady.jsonl,$WL_DIR/workload_varying.jsonl,$WL_DIR/workload_heavy.jsonl,$WL_DIR/workload_overload.jsonl,$WL_DIR/workload_tier1_demo.jsonl"

TTFT_SLO=2000
TPOT_SLO=150

echo "=========================================="
echo " 8-GPU Var-Len Sweep (All 8 GPUs)"
echo " SLO: TTFT=${TTFT_SLO}ms, TPOT=${TPOT_SLO}ms"
echo " Started: $(date)"
echo "=========================================="

# Scheme 1: PDAF DynM (max freq)
echo ""
echo ">>> [1/4] PDAF(TP=2) DynM thr=128 (max_freq)"
$PYTHON "$SCRIPT_DIR/run_8gpu_deploy_bench.py" \
    --deploys pdaf_8g_dyn \
    --workloads "$ALL_WL" \
    --freq max \
    --ttft-slo-ms $TTFT_SLO \
    --tpot-slo-ms $TPOT_SLO \
    --output-dir "$OUT_BASE/pdaf_dyn/json" \
    --log-dir "$LOG_BASE/pdaf_dyn" \
    --max-run-s 600 \
    --force
echo ">>> PDAF DynM done at $(date)"

# Scheme 2: PDAF DynM + DVFS V2
echo ""
echo ">>> [2/4] PDAF(TP=2) DynM thr=128 + DVFS V2"
$PYTHON "$SCRIPT_DIR/run_8gpu_deploy_bench.py" \
    --deploys pdaf_8g_dyn_tier \
    --workloads "$ALL_WL" \
    --freq auto \
    --ttft-slo-ms $TTFT_SLO \
    --tpot-slo-ms $TPOT_SLO \
    --output-dir "$OUT_BASE/pdaf_dyn_tier/json" \
    --log-dir "$LOG_BASE/pdaf_dyn_tier" \
    --max-run-s 600 \
    --force
echo ">>> PDAF DynM+DVFS done at $(date)"

# Scheme 3: PD DP4 (max freq)
echo ""
echo ">>> [3/4] PD DP4 (max_freq)"
$PYTHON "$SCRIPT_DIR/run_8gpu_deploy_bench.py" \
    --deploys pd_dp4 \
    --workloads "$ALL_WL" \
    --freq max \
    --ttft-slo-ms $TTFT_SLO \
    --tpot-slo-ms $TPOT_SLO \
    --output-dir "$OUT_BASE/pd_dp4/json" \
    --log-dir "$LOG_BASE/pd_dp4" \
    --max-run-s 600 \
    --force
echo ">>> PD DP4 done at $(date)"

# Scheme 4: Native TP=8 (max freq)
echo ""
echo ">>> [4/4] Native TP=8 (max_freq)"
$PYTHON "$SCRIPT_DIR/run_8gpu_deploy_bench.py" \
    --deploys native_tp8 \
    --workloads "$ALL_WL" \
    --freq max \
    --ttft-slo-ms $TTFT_SLO \
    --tpot-slo-ms $TPOT_SLO \
    --output-dir "$OUT_BASE/native_tp8/json" \
    --log-dir "$LOG_BASE/native_tp8" \
    --max-run-s 600 \
    --force
echo ">>> Native TP=8 done at $(date)"

echo ""
echo "=========================================="
echo " 8-GPU Var-Len sweep completed at $(date)"
echo "=========================================="
