#!/bin/bash
# 8-GPU v2 sweep: 4 deployment schemes × 4 workloads × increasing QPS
# TTFT now uses processing time (excludes queue wait)
# Non-tier schemes use max_freq; tier scheme uses auto (DVFS)
set -e

PYTHON=/workspace/env/sglang-tier/bin/python
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WL_DIR="/workspace/sglang-tier/benchmark/energy_bench/workloads"
OUT_BASE="$SCRIPT_DIR/results/8gpu_v2"
LOG_BASE="$SCRIPT_DIR/logs/8gpu_v2"

# All workloads (sorted by QPS within each type)
WL_128="$WL_DIR/fixed_il128_ol1024_qps1.jsonl,$WL_DIR/fixed_il128_ol1024_qps2.jsonl,$WL_DIR/fixed_il128_ol1024_qps4.jsonl,$WL_DIR/fixed_il128_ol1024_qps6.jsonl,$WL_DIR/fixed_il128_ol1024_qps8.jsonl,$WL_DIR/fixed_il128_ol1024_qps10.jsonl,$WL_DIR/fixed_il128_ol1024_qps12.jsonl,$WL_DIR/fixed_il128_ol1024_qps15.jsonl,$WL_DIR/fixed_il128_ol1024_qps20.jsonl"
WL_512="$WL_DIR/fixed_il512_ol256_qps1.jsonl,$WL_DIR/fixed_il512_ol256_qps2.jsonl,$WL_DIR/fixed_il512_ol256_qps4.jsonl,$WL_DIR/fixed_il512_ol256_qps6.jsonl,$WL_DIR/fixed_il512_ol256_qps8.jsonl,$WL_DIR/fixed_il512_ol256_qps10.jsonl,$WL_DIR/fixed_il512_ol256_qps12.jsonl,$WL_DIR/fixed_il512_ol256_qps15.jsonl"
WL_2048="$WL_DIR/fixed_il2048_ol64_qps1.jsonl,$WL_DIR/fixed_il2048_ol64_qps2.jsonl,$WL_DIR/fixed_il2048_ol64_qps3.jsonl,$WL_DIR/fixed_il2048_ol64_qps4.jsonl,$WL_DIR/fixed_il2048_ol64_qps6.jsonl,$WL_DIR/fixed_il2048_ol64_qps8.jsonl,$WL_DIR/fixed_il2048_ol64_qps10.jsonl"
WL_4096="$WL_DIR/fixed_il4096_ol64_qps1.jsonl,$WL_DIR/fixed_il4096_ol64_qps2.jsonl,$WL_DIR/fixed_il4096_ol64_qps3.jsonl,$WL_DIR/fixed_il4096_ol64_qps4.jsonl,$WL_DIR/fixed_il4096_ol64_qps5.jsonl,$WL_DIR/fixed_il4096_ol64_qps6.jsonl"

ALL_WL="$WL_128,$WL_512,$WL_2048,$WL_4096"

echo "=========================================="
echo " 8-GPU v2 Sweep (TTFT=processing, DynM)"
echo " Started: $(date)"
echo "=========================================="

# Scheme 1: PD TP4 (max_freq)
echo ""
echo ">>> [1/4] PD TP4 (max_freq)"
$PYTHON "$SCRIPT_DIR/run_8gpu_deploy_bench.py" \
    --deploys pd_p4d4 \
    --workloads "$ALL_WL" \
    --freq max \
    --output-dir "$OUT_BASE/pd_tp4/json" \
    --log-dir "$LOG_BASE/pd_tp4" \
    --max-run-s 600 \
    --stop-on-collapse \
    --force
echo ">>> PD TP4 done at $(date)"

# Scheme 2: PD DP4 (max_freq)
echo ""
echo ">>> [2/4] PD DP4 (max_freq)"
$PYTHON "$SCRIPT_DIR/run_8gpu_deploy_bench.py" \
    --deploys pd_dp4 \
    --workloads "$ALL_WL" \
    --freq max \
    --output-dir "$OUT_BASE/pd_dp4/json" \
    --log-dir "$LOG_BASE/pd_dp4" \
    --max-run-s 600 \
    --stop-on-collapse \
    --force
echo ">>> PD DP4 done at $(date)"

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
echo " All 4 schemes completed at $(date)"
echo "=========================================="
