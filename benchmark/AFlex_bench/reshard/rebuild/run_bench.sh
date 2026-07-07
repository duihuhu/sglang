#!/bin/bash
# Run SGLang TP=8 cold-start breakdown benchmark on node2 container.
# Execute from node1 host:
#   ssh 10.252.129.35 "docker exec operator_test bash /workspace/sglang/benchmark/AFlex_bench/reshard/rebuild/run_bench.sh"
#
# Or directly inside node2 container:
#   cd /workspace/sglang/benchmark/AFlex_bench/reshard/rebuild && bash run_bench.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo " SGLang TP=8 Cold-Start Breakdown Benchmark"
echo " Node: $(hostname)"
echo " Time: $(date)"
echo " GPUs: $(nvidia-smi -L | wc -l) visible"
echo "============================================"

# Kill any lingering sglang processes
pkill -f "sglang.launch_server" 2>/dev/null || true
sleep 3

# Ensure output directories exist
mkdir -p results logs

# Run the benchmark (3 trials)
python3 bench_instance_startup.py \
    --model-path /models/Qwen3-32B \
    --tp 8 \
    --port 39900 \
    --nccl-port 39910 \
    --repeats 3 \
    --timeout-s 300 \
    --output results/startup_breakdown_tp8.json

echo ""
echo "Done! Results in: $SCRIPT_DIR/results/startup_breakdown_tp8.json"
