#!/bin/bash
# Wait for baseline to finish, then run Tier extended tests

PYTHON=/workspace/env/sglang-test/bin/python
BASE=/workspace/sglang/benchmark/AFlex_bench/06_others/Mixtral_test
LOG_DIR=$BASE/logs

echo "$(date) Waiting for baseline extended (PID $1) to finish..."
while kill -0 $1 2>/dev/null; do
    sleep 30
done
echo "$(date) Baseline done! Starting Tier extended..."

cd $BASE/scripts
$PYTHON run_micro_bench.py \
  --ngpu 8 --gpus 0,1,2,3,4,5,6,7 \
  --deploy all \
  --scenario chatbot,qa,rag,summary \
  --qps 1,2,3,4,5,6,7,8,9 \
  --tier \
  --max-run-s 300 \
  2>&1 | tee $LOG_DIR/bench_8gpu_ext_tier.log

echo "$(date) ALL EXTENDED TESTS DONE!"
