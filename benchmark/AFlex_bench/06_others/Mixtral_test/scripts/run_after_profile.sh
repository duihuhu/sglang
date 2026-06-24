#!/bin/bash
set -e

PYTHON=/workspace/env/sglang-test/bin/python
BASE=/workspace/sglang/benchmark/AFlex_bench/06_others/Mixtral_test
RETRAIN=/workspace/sglang/benchmark/AFlex_bench/03_sensitivity/slo_sweep/retrain
LOG_DIR=$BASE/logs

echo "$(date) Waiting for parallel V2 profiling to finish (PID $1)..."
while kill -0 $1 2>/dev/null; do
    sleep 30
done
echo "$(date) V2 profiling done!"

# ============================================================
# Step 1: Train energy models (V1 + V2)
# ============================================================
echo "$(date) === Training V1 energy model ==="
cd $RETRAIN
$PYTHON energy_model_v1.py \
  --prefill $BASE/energy_model/data/prefill_data_v1.txt \
  --decode $BASE/energy_model/data/decode_data_v1.txt \
  --output-dir $BASE/energy_model/models_v1 \
  2>&1 | tee $LOG_DIR/train_v1.log

echo "$(date) === Training V2 energy model ==="
$PYTHON energy_model_v2.py \
  --data $BASE/energy_model/data/decode_pipeline_v1.txt \
  --output-dir $BASE/energy_model/models_v2 \
  2>&1 | tee $LOG_DIR/train_v2.log

echo "$(date) Models trained!"

# ============================================================
# Step 2: 8-GPU Baseline benchmark (Native/PD/PDAF)
# ============================================================
echo "$(date) === 8-GPU Baseline Benchmark ==="
cd $BASE/scripts
$PYTHON run_micro_bench.py \
  --ngpu 8 --gpus 0,1,2,3,4,5,6,7 \
  --deploy all \
  --scenario chatbot,qa,rag,summary \
  --qps 1,3,5 \
  --max-run-s 400 \
  2>&1 | tee $LOG_DIR/bench_8gpu_baseline.log

echo "$(date) Baseline done!"

# ============================================================
# Step 3: 8-GPU Tier benchmark (Native/PD/PDAF + DVFS)
# ============================================================
echo "$(date) === 8-GPU Tier Benchmark ==="
$PYTHON run_micro_bench.py \
  --ngpu 8 --gpus 0,1,2,3,4,5,6,7 \
  --deploy all \
  --scenario chatbot,qa,rag,summary \
  --qps 1,3,5 \
  --tier \
  --max-run-s 400 \
  2>&1 | tee $LOG_DIR/bench_8gpu_tier.log

echo "$(date) ALL DONE!"
