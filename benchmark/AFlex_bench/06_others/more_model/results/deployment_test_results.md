# Qwen3-30B-A3B (MoE) Deployment Test Results

## Model Info
- Model: Qwen3-30B-A3B (MoE, 128 experts, 8 active per token)
- Total params: ~30B, Active params: ~3B per token
- Model size on disk: 57GB
- Min TP requirement: TP=2 (per-GPU memory ~72GB with full expert weights)

## Deployment Configurations (8x A800-80GB)

| Config | Topology | GPUs Used | Tier DVFS |
|--------|----------|-----------|-----------|
| native_tp2 | Single instance TP=2 | 2 (GPU 0,1) | No |
| native_tp2_tier | Single instance TP=2 | 2 (GPU 0,1) | Yes |
| pd_tp2 | P-TP2 (GPU 0,1) + D-TP2 (GPU 2,3) | 4 | No |
| pd_tp2_tier | P-TP2 (GPU 0,1) + D-TP2 (GPU 2,3) | 4 | Yes |
| pdaf_tp2 | PA/PF/DA/DF each TP=2 | 8 | No |
| pdaf_tp2_tier | PA/PF/DA/DF each TP=2 | 8 | Yes |

## Generation Test Results (32 tokens, temperature=0)

| Config | TTFT (ms) | E2E (s) | Status |
|--------|-----------|---------|--------|
| native_tp2 | 407.1 | 2.30 | PASS |
| native_tp2_tier | 436.5 | 2.45 | PASS |
| pd_tp2 | 348.9 | 3.73 | PASS |
| pd_tp2_tier | 359.1 | 3.92 | PASS |
| pdaf_tp2 | 310.1 | 4.58 | PASS |
| pdaf_tp2_tier | 301.4 | 4.93 | PASS |

## Bug Fix Applied
- File: `python/sglang/srt/models/qwen3_moe.py`
- Issue: `NameError: name 'AFDDecoderLayerMixin' is not defined` in `forward_afd_A` and `forward_afd_F`
- Cause: Local import in `__init__` not accessible from other methods
- Fix: Added `from sglang.srt.layers.afd_mixin import AFDDecoderLayerMixin` in both methods

## Notes
- MoE model requires full expert weight loading on all GPUs (no expert parallelism in this config)
- PDAF FFN perspective servers have a health check limitation: `/health` endpoint
  returns timeout because event_loop_afd doesn't send detokenizer heartbeats.
  Use `/get_model_info` for liveness check instead.
- Memory: ~28GB model weights per GPU (TP=2), ~50GB free for KV cache after loading
- Disk I/O: Loading 4x TP=2 instances simultaneously takes ~20-30s due to I/O contention
