#!/bin/bash
# 4-GPU sweep: variable-length workloads, all 4 schemes, GPUs 4-7
set -e

PYTHON=/workspace/env/sglang-tier/bin/python
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WL_DIR="/workspace/sglang-tier/benchmark/energy_bench/workloads"
OUT_BASE="$SCRIPT_DIR/results/4gpu_var"
LOG_BASE="$SCRIPT_DIR/logs/4gpu_var"

mkdir -p "$OUT_BASE" "$LOG_BASE"

# Variable-length workloads (all have arrival_time_s, mixed il/ol)
ALL_WL="$WL_DIR/workload_steady.jsonl,$WL_DIR/workload_varying.jsonl,$WL_DIR/workload_heavy.jsonl,$WL_DIR/workload_overload.jsonl,$WL_DIR/workload_tier1_demo.jsonl"

echo "=========================================="
echo " 4-GPU Var-Len Sweep (GPUs 4-7)"
echo " Started: $(date)"
echo "=========================================="

# Scheme 1: PD TP2
echo ""
echo ">>> [1/4] PD TP2 (max_freq)"
$PYTHON "$SCRIPT_DIR/run_4gpu_deploy_bench.py" \
    --deploys pd_tp2 \
    --workloads "$ALL_WL" \
    --freq max \
    --output-dir "$OUT_BASE/pd_tp2/json" \
    --log-dir "$LOG_BASE/pd_tp2" \
    --max-run-s 600 \
    --force
echo ">>> PD TP2 done at $(date)"

# Scheme 2: PD DP2
echo ""
echo ">>> [2/4] PD DP2 (max_freq)"
$PYTHON "$SCRIPT_DIR/run_4gpu_deploy_bench.py" \
    --deploys pd_dp2 \
    --workloads "$ALL_WL" \
    --freq max \
    --output-dir "$OUT_BASE/pd_dp2/json" \
    --log-dir "$LOG_BASE/pd_dp2" \
    --max-run-s 600 \
    --force
echo ">>> PD DP2 done at $(date)"

# Scheme 3: PDAF Dynamic M
echo ""
echo ">>> [3/4] PDAF DynM (max_freq)"
$PYTHON "$SCRIPT_DIR/run_4gpu_deploy_bench.py" \
    --deploys pdaf_4g_dyn \
    --workloads "$ALL_WL" \
    --freq max \
    --output-dir "$OUT_BASE/pdaf_dyn/json" \
    --log-dir "$LOG_BASE/pdaf_dyn" \
    --max-run-s 600 \
    --force
echo ">>> PDAF DynM done at $(date)"

# Scheme 4: PDAF Tier + Dynamic M
echo ""
echo ">>> [4/4] PDAF Tier+DynM (auto_freq/DVFS)"
$PYTHON "$SCRIPT_DIR/run_4gpu_deploy_bench.py" \
    --deploys pdaf_4g_dyn_tier \
    --workloads "$ALL_WL" \
    --freq auto \
    --output-dir "$OUT_BASE/pdaf_dyn_tier/json" \
    --log-dir "$LOG_BASE/pdaf_dyn_tier" \
    --max-run-s 600 \
    --force
echo ">>> PDAF Tier+DynM done at $(date)"

echo ""
echo "=========================================="
echo " 4-GPU Var-Len sweep completed at $(date)"
echo "=========================================="
