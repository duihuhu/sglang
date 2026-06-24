#!/bin/bash
# Wait for V2 hetero profiling to finish, then:
# 1. Train V2 energy model with new DA1+DF2 data
# 2. Run hetero PDAF benchmark (baseline + tier)

set -e

PYTHON="/workspace/env/sglang-test/bin/python"
SCRIPTS_DIR="/workspace/sglang/benchmark/AFlex_bench/06_others/Mixtral_test/scripts"
LOG_DIR="/workspace/sglang/benchmark/AFlex_bench/06_others/Mixtral_test/logs"
DATA_DIR="/workspace/sglang/benchmark/AFlex_bench/06_others/Mixtral_test/energy_model/data"
MODEL_DIR="/workspace/sglang/benchmark/AFlex_bench/06_others/Mixtral_test/energy_model/models"
TRAIN_SCRIPT_DIR="/workspace/sglang/benchmark/AFlex_bench/03_sensitivity/slo_sweep/retrain"

mkdir -p "$LOG_DIR" "$MODEL_DIR"

echo "[$(date)] Waiting for V2 hetero profiling to complete..."
while pgrep -f "profile_v2_hetero.py" > /dev/null 2>&1; do
    sleep 60
done
echo "[$(date)] V2 hetero profiling done!"
echo "[$(date)] Data points: $(( $(wc -l < $DATA_DIR/decode_pipeline_da1_df2.txt) - 1 ))"

# Step 1: Prepare data (add 'tp' column for compatibility)
echo "[$(date)] Preparing decode data..."
$PYTHON -c "
import pandas as pd
df = pd.read_csv('$DATA_DIR/decode_pipeline_da1_df2.txt', sep='\t')
# energy_model_v2.py expects 'tp' column
df['tp'] = 1  # DA is TP1 (the pipeline reference)
df.to_csv('$DATA_DIR/decode_pipeline_hetero_compat.txt', sep='\t', index=False)
print(f'Prepared {len(df)} rows')
"

# Step 2: Train V2 models
echo "[$(date)] Training V2 energy models..."
cd "$TRAIN_SCRIPT_DIR"
$PYTHON energy_model_v2.py \
    --data-dir "$DATA_DIR" \
    --decode-data "decode_pipeline_hetero_compat.txt" \
    --output-dir "$MODEL_DIR" \
    --skip-cv \
    2>&1 | tee "$LOG_DIR/train_v2_hetero.log"

echo "[$(date)] V2 model training complete!"

# Step 3: Run hetero PDAF benchmark
echo "[$(date)] Starting heterogeneous PDAF benchmark..."
cd "$SCRIPTS_DIR"
$PYTHON run_hetero_pdaf_bench.py \
    2>&1 | tee "$LOG_DIR/hetero_pdaf_bench.log"

echo "[$(date)] All done!"
