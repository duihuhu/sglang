#!/bin/bash
set -e

PYTHON=/workspace/env/sglang-test/bin/python
BASE=/workspace/sglang/benchmark/AFlex_bench/06_others/Mixtral_test
LOG_DIR=$BASE/logs

# Step 1: Wait for TP4 prefill profiling
echo "$(date) Waiting for TP4 prefill profiling (PID $1)..."
while kill -0 $1 2>/dev/null; do sleep 30; done
echo "$(date) TP4 prefill done!"

# Step 2: Run TP4 decode profiling
echo "$(date) Starting TP4 decode profiling..."
cd $BASE/scripts
$PYTHON profile_tp4.py --gpus 0,1,2,3 --phase decode \
  2>&1 | tee $LOG_DIR/profile_tp4_decode.log
echo "$(date) TP4 decode done!"

echo "$(date) TP4 profiling complete. Ready for heterogeneous PDAF deployment."
