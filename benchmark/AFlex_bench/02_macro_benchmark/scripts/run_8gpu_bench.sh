#!/bin/bash
# 8-GPU full-scale benchmark: 5 topologies × 4 workload types × increasing QPS.
# Runs all 140 test points sequentially with early-stop on SLO collapse.
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="/workspace/env/sglang-tier/bin/python"
WL_DIR="/workspace/sglang-tier/benchmark/energy_bench/workloads"

# Build workload list: 4 length combos × multiple QPS each
WORKLOADS=""

# il128_ol1024 (decode-heavy)
for q in 2 4 6 8 10 12 15 20; do
    WORKLOADS="${WORKLOADS:+$WORKLOADS,}${WL_DIR}/fixed_il128_ol1024_qps${q}.jsonl"
done

# il512_ol256 (balanced)
for q in 2 4 6 8 10 12 15; do
    WORKLOADS="${WORKLOADS:+$WORKLOADS,}${WL_DIR}/fixed_il512_ol256_qps${q}.jsonl"
done

# il2048_ol64 (prefill-heavy)
for q in 1 2 3 4 6 8 10; do
    WORKLOADS="${WORKLOADS:+$WORKLOADS,}${WL_DIR}/fixed_il2048_ol64_qps${q}.jsonl"
done

# il4096_ol64 (extreme prefill)
for q in 1 2 3 4 5 6; do
    WORKLOADS="${WORKLOADS:+$WORKLOADS,}${WL_DIR}/fixed_il4096_ol64_qps${q}.jsonl"
done

echo "[$(date '+%H:%M:%S')] Starting 8-GPU full benchmark..."
echo "  Deploys: pd_p4d4, pdaf_8g_m1, pdaf_8g_m2, pdaf_8g_m1_tier, pdaf_8g_m2_tier"
echo "  Workloads: 4 length combos × increasing QPS = 28 QPS points"
echo "  Total runs: up to 140 (with early-stop on SLO collapse)"
echo ""

mkdir -p logs/_sweep results/8gpu/json results/8gpu/figures

$PYTHON run_8gpu_deploy_bench.py \
    --deploys "pd_p4d4,pdaf_8g_m1,pdaf_8g_m2,pdaf_8g_m1_tier,pdaf_8g_m2_tier" \
    --workloads "$WORKLOADS" \
    --freq auto \
    --max-run-s 600 \
    --output-dir results/8gpu/json \
    --log-dir logs/8gpu \
    --stop-on-collapse \
    2>&1 | tee logs/_sweep/8gpu_full.log

echo ""
echo "[$(date '+%H:%M:%S')] 8-GPU benchmark complete."
