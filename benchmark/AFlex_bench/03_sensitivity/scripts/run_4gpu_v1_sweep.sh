#!/bin/bash
# 4-GPU sweep: all 4 schemes using GPUs 4-7
set -e

PYTHON=/workspace/env/sglang-tier/bin/python
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WL_DIR="/workspace/sglang-tier/benchmark/energy_bench/workloads"
OUT_BASE="$SCRIPT_DIR/results/4gpu_v1"
LOG_BASE="$SCRIPT_DIR/logs/4gpu_v1"

mkdir -p "$OUT_BASE" "$LOG_BASE"

WL_128="$WL_DIR/fixed_il128_ol1024_qps1.jsonl,$WL_DIR/fixed_il128_ol1024_qps2.jsonl,$WL_DIR/fixed_il128_ol1024_qps4.jsonl,$WL_DIR/fixed_il128_ol1024_qps6.jsonl,$WL_DIR/fixed_il128_ol1024_qps8.jsonl,$WL_DIR/fixed_il128_ol1024_qps10.jsonl,$WL_DIR/fixed_il128_ol1024_qps12.jsonl,$WL_DIR/fixed_il128_ol1024_qps15.jsonl,$WL_DIR/fixed_il128_ol1024_qps20.jsonl"
WL_512="$WL_DIR/fixed_il512_ol256_qps1.jsonl,$WL_DIR/fixed_il512_ol256_qps2.jsonl,$WL_DIR/fixed_il512_ol256_qps4.jsonl,$WL_DIR/fixed_il512_ol256_qps6.jsonl,$WL_DIR/fixed_il512_ol256_qps8.jsonl,$WL_DIR/fixed_il512_ol256_qps10.jsonl,$WL_DIR/fixed_il512_ol256_qps12.jsonl,$WL_DIR/fixed_il512_ol256_qps15.jsonl"
WL_2048="$WL_DIR/fixed_il2048_ol64_qps1.jsonl,$WL_DIR/fixed_il2048_ol64_qps2.jsonl,$WL_DIR/fixed_il2048_ol64_qps3.jsonl,$WL_DIR/fixed_il2048_ol64_qps4.jsonl,$WL_DIR/fixed_il2048_ol64_qps6.jsonl,$WL_DIR/fixed_il2048_ol64_qps8.jsonl,$WL_DIR/fixed_il2048_ol64_qps10.jsonl"
WL_4096="$WL_DIR/fixed_il4096_ol64_qps1.jsonl,$WL_DIR/fixed_il4096_ol64_qps2.jsonl,$WL_DIR/fixed_il4096_ol64_qps3.jsonl,$WL_DIR/fixed_il4096_ol64_qps4.jsonl,$WL_DIR/fixed_il4096_ol64_qps5.jsonl,$WL_DIR/fixed_il4096_ol64_qps6.jsonl"

ALL_WL="$WL_128,$WL_512,$WL_2048,$WL_4096"

echo "=========================================="
echo " 4-GPU Sweep (GPUs 4-7): All Schemes"
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
    --stop-on-collapse \
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
    --stop-on-collapse \
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
    --stop-on-collapse \
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
    --stop-on-collapse \
    --force
echo ">>> PDAF Tier+DynM done at $(date)"

echo ""
echo "=========================================="
echo " 4-GPU sweep completed at $(date)"
echo "=========================================="
