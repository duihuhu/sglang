#!/bin/bash
# 8-GPU v2 sweep: remaining 3 schemes (PD TP4 already done)
# DP4 = 4x TP=2 full instances (no PD disagg), round-robin
# PDAF DynM = AF with dynamic micro-batch, max_freq
# PDAF Tier DynM = AF with dynamic micro-batch + DVFS
set -e

PYTHON=/workspace/env/sglang-tier/bin/python
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WL_DIR="/workspace/sglang-tier/benchmark/energy_bench/workloads"
OUT_BASE="$SCRIPT_DIR/results/8gpu_v2"
LOG_BASE="$SCRIPT_DIR/logs/8gpu_v2"

WL_128="$WL_DIR/fixed_il128_ol1024_qps1.jsonl,$WL_DIR/fixed_il128_ol1024_qps2.jsonl,$WL_DIR/fixed_il128_ol1024_qps4.jsonl,$WL_DIR/fixed_il128_ol1024_qps6.jsonl,$WL_DIR/fixed_il128_ol1024_qps8.jsonl,$WL_DIR/fixed_il128_ol1024_qps10.jsonl,$WL_DIR/fixed_il128_ol1024_qps12.jsonl,$WL_DIR/fixed_il128_ol1024_qps15.jsonl,$WL_DIR/fixed_il128_ol1024_qps20.jsonl"
WL_512="$WL_DIR/fixed_il512_ol256_qps1.jsonl,$WL_DIR/fixed_il512_ol256_qps2.jsonl,$WL_DIR/fixed_il512_ol256_qps4.jsonl,$WL_DIR/fixed_il512_ol256_qps6.jsonl,$WL_DIR/fixed_il512_ol256_qps8.jsonl,$WL_DIR/fixed_il512_ol256_qps10.jsonl,$WL_DIR/fixed_il512_ol256_qps12.jsonl,$WL_DIR/fixed_il512_ol256_qps15.jsonl"
WL_2048="$WL_DIR/fixed_il2048_ol64_qps1.jsonl,$WL_DIR/fixed_il2048_ol64_qps2.jsonl,$WL_DIR/fixed_il2048_ol64_qps3.jsonl,$WL_DIR/fixed_il2048_ol64_qps4.jsonl,$WL_DIR/fixed_il2048_ol64_qps6.jsonl,$WL_DIR/fixed_il2048_ol64_qps8.jsonl,$WL_DIR/fixed_il2048_ol64_qps10.jsonl"
WL_4096="$WL_DIR/fixed_il4096_ol64_qps1.jsonl,$WL_DIR/fixed_il4096_ol64_qps2.jsonl,$WL_DIR/fixed_il4096_ol64_qps3.jsonl,$WL_DIR/fixed_il4096_ol64_qps4.jsonl,$WL_DIR/fixed_il4096_ol64_qps5.jsonl,$WL_DIR/fixed_il4096_ol64_qps6.jsonl"

ALL_WL="$WL_128,$WL_512,$WL_2048,$WL_4096"

echo "=========================================="
echo " 8-GPU v2 Sweep (remaining 3 schemes)"
echo " Started: $(date)"
echo "=========================================="

# Scheme 2: DP=4 (4x TP=2 full instances, max_freq)
echo ""
echo ">>> [2/4] DP=4 (4x TP=2, max_freq)"
$PYTHON "$SCRIPT_DIR/run_8gpu_deploy_bench.py" \
    --deploys pd_dp4 \
    --workloads "$ALL_WL" \
    --freq max \
    --output-dir "$OUT_BASE/pd_dp4/json" \
    --log-dir "$LOG_BASE/pd_dp4" \
    --max-run-s 600 \
    --stop-on-collapse \
    --force
echo ">>> DP4 done at $(date)"

# Scheme 3: PDAF Dynamic M (max_freq)
echo ""
echo ">>> [3/4] PDAF Dynamic M (max_freq)"
$PYTHON "$SCRIPT_DIR/run_8gpu_deploy_bench.py" \
    --deploys pdaf_8g_dyn \
    --workloads "$ALL_WL" \
    --freq max \
    --output-dir "$OUT_BASE/pdaf_dyn/json" \
    --log-dir "$LOG_BASE/pdaf_dyn" \
    --max-run-s 600 \
    --stop-on-collapse \
    --force
echo ">>> PDAF DynM done at $(date)"

# Scheme 4: PDAF Tier + Dynamic M (auto_freq/DVFS)
echo ""
echo ">>> [4/4] PDAF Tier + Dynamic M (auto_freq/DVFS)"
$PYTHON "$SCRIPT_DIR/run_8gpu_deploy_bench.py" \
    --deploys pdaf_8g_dyn_tier \
    --workloads "$ALL_WL" \
    --freq auto \
    --output-dir "$OUT_BASE/pdaf_dyn_tier/json" \
    --log-dir "$LOG_BASE/pdaf_dyn_tier" \
    --max-run-s 600 \
    --stop-on-collapse \
    --force
echo ">>> PDAF Tier DynM done at $(date)"

echo ""
echo "=========================================="
echo " All remaining schemes completed at $(date)"
echo "=========================================="
