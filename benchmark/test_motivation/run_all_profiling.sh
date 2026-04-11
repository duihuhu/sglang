#!/bin/bash
# =============================================================================
# Phase 1: 全量 Profiling 数据采集
#
# 模型: Qwen3-32B (/models/Qwen/Qwen3-32B/)
# GPU:  A800-80GB SXM x 8
# 频率: {210, 450, 690, 930, 1170, 1410} MHz
#
# 用法:
#   chmod +x run_all_profiling.sh
#   sudo ./run_all_profiling.sh          # 跑全部 (tp=1,2,4,8 + idle + comm)
#   sudo ./run_all_profiling.sh quick    # 快速验证模式
#   sudo ./run_all_profiling.sh prefill  # 只跑 prefill
#   sudo ./run_all_profiling.sh decode   # 只跑 decode
# =============================================================================

set -e

MODEL_PATH="/models/Qwen/Qwen3-32B/"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-all}"  # all, quick, prefill, decode, idle, comm

echo "============================================================"
echo " Phase 1 Profiling — Qwen3-32B on A800-80GB"
echo " Mode: $MODE"
echo " Model: $MODEL_PATH"
echo " Working dir: $SCRIPT_DIR"
echo "============================================================"
echo ""

# ── Step 0: 编译 DVFS 库 ─────────────────────────────────────────────
echo "[Step 0] Building libdvfs_ctrl.so ..."
cd dvfs && make && cd ..
echo ""

# ── Step 1: 空闲功耗基线 ─────────────────────────────────────────────
if [[ "$MODE" == "all" || "$MODE" == "quick" || "$MODE" == "idle" ]]; then
    echo "============================================================"
    echo " [Step 1] Idle Power Baseline"
    echo "============================================================"
    python bench_idle_power.py --gpu 0
    echo ""
fi

# ── Step 2: Prefill A/F Profiling ────────────────────────────────────
if [[ "$MODE" == "all" || "$MODE" == "prefill" ]]; then
    echo "============================================================"
    echo " [Step 2] Prefill A/F Profiling"
    echo "============================================================"

    # tp=1: 模型 ~61GB, 剩余 ~19GB 给 KV cache
    echo "--- Prefill tp=1 ---"
    python bench_prefill_af.py \
        --model-path "$MODEL_PATH" \
        --tp-size 1 \
        --output prefill_data_v1_tp1.txt
    echo ""

    # tp=2: 模型 ~30.5GB/GPU, 剩余 ~49.5GB
    echo "--- Prefill tp=2 ---"
    python bench_prefill_af.py \
        --model-path "$MODEL_PATH" \
        --tp-size 2 \
        --output prefill_data_v1_tp2.txt
    echo ""

    # tp=4: 模型 ~15.3GB/GPU, 剩余 ~64.7GB
    echo "--- Prefill tp=4 ---"
    python bench_prefill_af.py \
        --model-path "$MODEL_PATH" \
        --tp-size 4 \
        --output prefill_data_v1_tp4.txt
    echo ""

    # tp=8: 模型 ~7.6GB/GPU, 剩余 ~72.4GB
    echo "--- Prefill tp=8 ---"
    python bench_prefill_af.py \
        --model-path "$MODEL_PATH" \
        --tp-size 8 \
        --output prefill_data_v1_tp8.txt
    echo ""

    # 合并所有 tp 的结果
    echo "--- Merging prefill results ---"
    head -1 prefill_data_v1_tp1.txt > prefill_data_v1.txt
    for f in prefill_data_v1_tp1.txt prefill_data_v1_tp2.txt prefill_data_v1_tp4.txt prefill_data_v1_tp8.txt; do
        [ -f "$f" ] && tail -n +2 "$f" >> prefill_data_v1.txt
    done
    echo "Merged → prefill_data_v1.txt ($(wc -l < prefill_data_v1.txt) lines)"
    echo ""
fi

# ── Step 3: Decode A/F Profiling ─────────────────────────────────────
if [[ "$MODE" == "all" || "$MODE" == "decode" ]]; then
    echo "============================================================"
    echo " [Step 3] Decode A/F Profiling"
    echo "============================================================"

    # tp=1
    echo "--- Decode tp=1 ---"
    python bench_decode_af_fast.py \
        --model-path "$MODEL_PATH" \
        --tp-size 1 \
        --output decode_data_v1_tp1.txt
    echo ""

    # tp=2
    echo "--- Decode tp=2 ---"
    python bench_decode_af_fast.py \
        --model-path "$MODEL_PATH" \
        --tp-size 2 \
        --output decode_data_v1_tp2.txt
    echo ""

    # tp=4
    echo "--- Decode tp=4 ---"
    python bench_decode_af_fast.py \
        --model-path "$MODEL_PATH" \
        --tp-size 4 \
        --output decode_data_v1_tp4.txt
    echo ""

    # tp=8
    echo "--- Decode tp=8 ---"
    python bench_decode_af_fast.py \
        --model-path "$MODEL_PATH" \
        --tp-size 8 \
        --output decode_data_v1_tp8.txt
    echo ""

    # 合并
    echo "--- Merging decode results ---"
    head -1 decode_data_v1_tp1.txt > decode_data_v1.txt
    for f in decode_data_v1_tp1.txt decode_data_v1_tp2.txt decode_data_v1_tp4.txt decode_data_v1_tp8.txt; do
        [ -f "$f" ] && tail -n +2 "$f" >> decode_data_v1.txt
    done
    echo "Merged → decode_data_v1.txt ($(wc -l < decode_data_v1.txt) lines)"
    echo ""
fi

# ── Step 4: AF 通信开销 ──────────────────────────────────────────────
if [[ "$MODE" == "all" || "$MODE" == "comm" ]]; then
    echo "============================================================"
    echo " [Step 4] AF Communication Overhead"
    echo "============================================================"
    torchrun --nproc_per_node=2 bench_af_comm.py
    echo ""
fi

# ── Quick 模式 ───────────────────────────────────────────────────────
if [[ "$MODE" == "quick" ]]; then
    echo "============================================================"
    echo " [Quick] Prefill + Decode validation (tp=1 only)"
    echo "============================================================"

    echo "--- Prefill quick ---"
    python bench_prefill_af.py \
        --model-path "$MODEL_PATH" \
        --tp-size 1 \
        --quick \
        --output prefill_data_v1_quick.txt
    echo ""

    echo "--- Decode quick ---"
    python bench_decode_af_fast.py \
        --model-path "$MODEL_PATH" \
        --tp-size 1 \
        --quick \
        --output decode_data_v1_quick.txt
    echo ""
fi

# ── bench_one_batch 对比验证 ─────────────────────────────────────────
if [[ "$MODE" == "validate" ]]; then
    echo "============================================================"
    echo " [Validate] bench_one_batch comparison (f=210MHz, il=512, bs=1)"
    echo "============================================================"

    nvidia-smi -i 0 -lgc 210,210
    echo "GPU 0 locked to 210 MHz"

    python -m sglang.bench_one_batch \
        --model-path "$MODEL_PATH" \
        --tp-size 1 \
        --batch-size 1 \
        --input-len 512 \
        --output-len 1 \
        --disable-cuda-graph \
        --disable-piecewise-cuda-graph

    nvidia-smi -i 0 -rgc
    echo "GPU 0 clocks restored"
    echo ""
    echo "Compare: your profiling data (il=512, bs=1, f=210) single layer A+F"
    echo "  × 64 layers ≈ bench_one_batch prefill latency"
fi

# ── 完成 ─────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " All done!"
echo " Results in: $SCRIPT_DIR"
echo "   prefill_data_v1.txt  — Prefill A/F latency + energy"
echo "   decode_data_v1.txt   — Decode A/F latency + energy"
echo "   idle_power.txt       — Idle power baseline"
echo "   af_comm_overhead.txt — AF communication overhead"
echo "============================================================"
