#!/bin/bash
# Wait for V1 decode profiling to finish
V1_PID=438710
echo "Waiting for V1 decode profiling (PID=$V1_PID) to finish..."
while kill -0 $V1_PID 2>/dev/null; do
    sleep 30
done
echo "V1 done. Starting V2 pipeline profiling..."

cd /workspace/sglang/benchmark/AFlex_bench/06_others/Mixtral_test/scripts
/workspace/env/sglang-test/bin/python profile_energy.py --gpus 0,1 --phase v2 \
  > /workspace/sglang/benchmark/AFlex_bench/06_others/Mixtral_test/logs/profile_v2_pipeline.log 2>&1

echo "V2 pipeline profiling done!"
