#!/bin/bash
# Run all macro-benchmark combinations sequentially.
# Priority order: PDAF first in each group.
# Groups: 8GPU no-tier -> 8GPU tier -> 4GPU no-tier -> 4GPU tier

set -e
PYTHON=/workspace/env/sglang-tier/bin/python
SCRIPT=/workspace/sglang-tier/benchmark/AFlex_bench/retesting/scripts/run_macro_bench.py
cd /workspace/sglang-tier/benchmark/AFlex_bench/retesting

echo "====== [1/4] 8GPU no-tier (pdaf -> native_dp -> pd_dp) ======"
echo "SKIPPED - already running separately"

echo ""
echo "====== [2/4] 8GPU + Tier (pdaf -> native_dp -> pd_dp) ======"
$PYTHON $SCRIPT --ngpu 8 --deploy pdaf,native_dp,pd_dp --tier --max-run-s 600 2>&1 | tee macro_benchmark/8gpu/run_8gpu_tier.log

echo ""
echo "====== [3/4] 4GPU no-tier (pdaf -> native_dp -> pd_dp) ======"
$PYTHON $SCRIPT --ngpu 4 --deploy pdaf,native_dp,pd_dp --max-run-s 600 2>&1 | tee macro_benchmark/4gpu/run_4gpu.log

echo ""
echo "====== [4/4] 4GPU + Tier (pdaf -> native_dp -> pd_dp) ======"
$PYTHON $SCRIPT --ngpu 4 --deploy pdaf,native_dp,pd_dp --tier --max-run-s 600 2>&1 | tee macro_benchmark/4gpu/run_4gpu_tier.log

echo ""
echo "====== ALL MACRO BENCHMARKS COMPLETE ======"
