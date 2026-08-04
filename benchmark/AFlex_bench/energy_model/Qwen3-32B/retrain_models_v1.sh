#!/usr/bin/env bash
# Retrain Qwen3-32B V1 energy models (LUT + LinearReg + GBDT) from v1_layer_profile.
# Includes full batch_size grid (bs=1..256); fixes Prefill energy LUT gap at bs=256.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$ROOT/../../../../.." && pwd)"
DATA_DIR="$ROOT/data/v1_layer_profile"
OUT_DIR="$ROOT/models_v1"

python3 "$REPO/benchmark/test_motivation/energy_model.py" \
  --prefill "$DATA_DIR/prefill_data_v1.txt" \
  --decode "$DATA_DIR/decode_data_v1.txt" \
  --output-dir "$OUT_DIR" \
  --folds 5

echo "Models written to $OUT_DIR"
