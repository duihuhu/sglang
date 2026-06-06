#!/bin/bash
# Run pdaf_m1 and pdaf_m1_tier benchmarks on GPU0-3.
# Waits until GPU0 and GPU1 are free (no compute processes) before starting.
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="/workspace/env/sglang-tier/bin/python"
WORKLOAD_DIR="/workspace/sglang-tier/benchmark/energy_bench/workloads"
WORKLOADS="${WORKLOAD_DIR}/fixed_il512_ol128_qps2.jsonl,${WORKLOAD_DIR}/fixed_il512_ol128_qps4.jsonl,${WORKLOAD_DIR}/fixed_il512_ol128_qps6.jsonl,${WORKLOAD_DIR}/fixed_il512_ol128_qps8.jsonl"

echo "[$(date '+%H:%M:%S')] Waiting for GPU0 and GPU1 to be free..."

while true; do
    # Check if GPU0 and GPU1 have any compute processes
    procs_0=$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]' || true)
    procs_1=$(nvidia-smi --id=1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]' || true)

    if [ "$procs_0" -eq 0 ] && [ "$procs_1" -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] GPU0 and GPU1 are free. Starting M=1 benchmarks..."
        break
    fi
    echo "[$(date '+%H:%M:%S')] GPU0 has ${procs_0} proc(s), GPU1 has ${procs_1} proc(s). Retrying in 30s..."
    sleep 30
done

# Phase 1: bare AF M=1
echo ""
echo "=========================================="
echo "  Phase 1: pdaf_m1 (bare AF, M=1)"
echo "=========================================="
$PYTHON run_deploy_bench.py \
    --deploys pdaf_m1 \
    --workloads "$WORKLOADS" \
    --freq auto \
    --max-run-s 360 \
    --output-dir results/deploy/json \
    --log-dir logs/deploy \
    2>&1 | tee logs/_sweep/deploy_m1_bare.log

# Phase 2: AF M=1 + Tier1/DVFS
echo ""
echo "=========================================="
echo "  Phase 2: pdaf_m1_tier (AF M=1 + DVFS)"
echo "=========================================="
$PYTHON run_deploy_bench.py \
    --deploys pdaf_m1_tier \
    --workloads "$WORKLOADS" \
    --freq auto \
    --max-run-s 360 \
    --output-dir results/deploy/json \
    --log-dir logs/deploy \
    2>&1 | tee logs/_sweep/deploy_m1_tier.log

echo ""
echo "[$(date '+%H:%M:%S')] All M=1 benchmarks complete."
