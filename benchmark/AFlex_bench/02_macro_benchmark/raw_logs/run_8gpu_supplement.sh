#!/bin/bash
# Minimal re-run: only tests that haven't been done or were done incorrectly
#
# Already done (TTFT=2000, correct):
#   - pdaf_8g_dyn (max freq) -> results/8gpu_var/pdaf_dyn/json/
#   - pdaf_8g_dyn_tier (DVFS V2) -> results/8gpu_var/pdaf_dyn_tier/json/
#   - native_tp8 (max freq) -> results/8gpu_var/native_tp8/json/
#
# Need to redo (TTFT=2000):
#   - pd_dp4 (was wrong config, now fixed to 4x 1P1D disagg)
#
# New tests (TTFT=2500, 3000):
#   - Only DVFS scheme behavior changes; others just get new SLO stats
#   - But we run all to get consistent SLO violation rates
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
echo " 8-GPU补测 + SLO Sweep"
echo " Started: $(date)"
echo "=========================================="

# ---- Part 1: Fix PD DP4 for TTFT=2000 ----
echo ""
echo "=== [Part 1] PD DP4 (4x1P1D) TTFT=2000 (redo) ==="
$PYTHON "$SCRIPT_DIR/run_8gpu_deploy_bench.py" \
    --deploys pd_dp4 \
    --workloads "$ALL_WL" \
    --freq max \
    --ttft-slo-ms 2000 \
    --tpot-slo-ms $TPOT_SLO \
    --output-dir "$OUT_BASE/pd_dp4_slo2000/json" \
    --log-dir "$LOG_BASE/pd_dp4_slo2000" \
    --max-run-s 600 \
    --force
echo ">>> PD DP4 (TTFT=2000) done at $(date)"

# ---- Part 2: TTFT=2500 sweep ----
echo ""
echo "=== [Part 2] TTFT=2500 sweep ==="

echo ">>> PDAF DynM (max) — TTFT=2500"
$PYTHON "$SCRIPT_DIR/run_8gpu_deploy_bench.py" \
    --deploys pdaf_8g_dyn \
    --workloads "$ALL_WL" \
    --freq max \
    --ttft-slo-ms 2500 \
    --tpot-slo-ms $TPOT_SLO \
    --output-dir "$OUT_BASE/pdaf_dyn_slo2500/json" \
    --log-dir "$LOG_BASE/pdaf_dyn_slo2500" \
    --max-run-s 600 \
    --force

echo ">>> PDAF DynM+DVFS — TTFT=2500"
$PYTHON "$SCRIPT_DIR/run_8gpu_deploy_bench.py" \
    --deploys pdaf_8g_dyn_tier \
    --workloads "$ALL_WL" \
    --freq auto \
    --ttft-slo-ms 2500 \
    --tpot-slo-ms $TPOT_SLO \
    --output-dir "$OUT_BASE/pdaf_dyn_tier_slo2500/json" \
    --log-dir "$LOG_BASE/pdaf_dyn_tier_slo2500" \
    --max-run-s 600 \
    --force

echo ">>> PD DP4 (4x1P1D) — TTFT=2500"
$PYTHON "$SCRIPT_DIR/run_8gpu_deploy_bench.py" \
    --deploys pd_dp4 \
    --workloads "$ALL_WL" \
    --freq max \
    --ttft-slo-ms 2500 \
    --tpot-slo-ms $TPOT_SLO \
    --output-dir "$OUT_BASE/pd_dp4_slo2500/json" \
    --log-dir "$LOG_BASE/pd_dp4_slo2500" \
    --max-run-s 600 \
    --force

echo ">>> Native TP=8 — TTFT=2500"
$PYTHON "$SCRIPT_DIR/run_8gpu_deploy_bench.py" \
    --deploys native_tp8 \
    --workloads "$ALL_WL" \
    --freq max \
    --ttft-slo-ms 2500 \
    --tpot-slo-ms $TPOT_SLO \
    --output-dir "$OUT_BASE/native_tp8_slo2500/json" \
    --log-dir "$LOG_BASE/native_tp8_slo2500" \
    --max-run-s 600 \
    --force

echo ">>> TTFT=2500 all done at $(date)"

# ---- Part 3: TTFT=3000 sweep ----
echo ""
echo "=== [Part 3] TTFT=3000 sweep ==="

echo ">>> PDAF DynM (max) — TTFT=3000"
$PYTHON "$SCRIPT_DIR/run_8gpu_deploy_bench.py" \
    --deploys pdaf_8g_dyn \
    --workloads "$ALL_WL" \
    --freq max \
    --ttft-slo-ms 3000 \
    --tpot-slo-ms $TPOT_SLO \
    --output-dir "$OUT_BASE/pdaf_dyn_slo3000/json" \
    --log-dir "$LOG_BASE/pdaf_dyn_slo3000" \
    --max-run-s 600 \
    --force

echo ">>> PDAF DynM+DVFS — TTFT=3000"
$PYTHON "$SCRIPT_DIR/run_8gpu_deploy_bench.py" \
    --deploys pdaf_8g_dyn_tier \
    --workloads "$ALL_WL" \
    --freq auto \
    --ttft-slo-ms 3000 \
    --tpot-slo-ms $TPOT_SLO \
    --output-dir "$OUT_BASE/pdaf_dyn_tier_slo3000/json" \
    --log-dir "$LOG_BASE/pdaf_dyn_tier_slo3000" \
    --max-run-s 600 \
    --force

echo ">>> PD DP4 (4x1P1D) — TTFT=3000"
$PYTHON "$SCRIPT_DIR/run_8gpu_deploy_bench.py" \
    --deploys pd_dp4 \
    --workloads "$ALL_WL" \
    --freq max \
    --ttft-slo-ms 3000 \
    --tpot-slo-ms $TPOT_SLO \
    --output-dir "$OUT_BASE/pd_dp4_slo3000/json" \
    --log-dir "$LOG_BASE/pd_dp4_slo3000" \
    --max-run-s 600 \
    --force

echo ">>> Native TP=8 — TTFT=3000"
$PYTHON "$SCRIPT_DIR/run_8gpu_deploy_bench.py" \
    --deploys native_tp8 \
    --workloads "$ALL_WL" \
    --freq max \
    --ttft-slo-ms 3000 \
    --tpot-slo-ms $TPOT_SLO \
    --output-dir "$OUT_BASE/native_tp8_slo3000/json" \
    --log-dir "$LOG_BASE/native_tp8_slo3000" \
    --max-run-s 600 \
    --force

echo ">>> TTFT=3000 all done at $(date)"

echo ""
echo "=========================================="
echo " Full supplemental sweep completed at $(date)"
echo "=========================================="
