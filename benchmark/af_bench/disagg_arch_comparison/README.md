# Disaggregation Architecture Comparison

Quick 6-config comparison of disaggregation architectures on GPUs 4-7.

## Configurations

| Config | GPUs | Disagg | AF M | GPU mapping |
|--------|------|--------|------|-------------|
| `native_tp1` | 1 | None | — | GPU 4 |
| `pd_only` | 2 | PD | — | Prefill=4, Decode=5 |
| `af_m1` | 2 | AF | 1 | Attn=4, FFN=5 |
| `af_m3_opt` | 2 | AF | 3 (interleaved) | Attn=4, FFN=5 |
| `pdaf_m1` | 4 | PD+AF | 1 | DF=4, DA=5, PF=6, PA=7 |
| `pdaf_m3_opt` | 4 | PD+AF | 3 (interleaved) | DF=4, DA=5, PF=6, PA=7 |

## Parameters

- Model: Qwen3-32B, input=512, output=128, req=400
- Client concurrency: 256 (all configs), QPS scaled by GPU count (16/32/32/32/64/64)
- max_running_requests: 256 (native/PD/AF), 96 (PD+AF — limited by decode-FFN GPU memory)
- UCX RDMA, CUDA graph off

## Run

```bash
python benchmark/af_bench/disagg_arch_comparison/run_comparison.py
```

Output: `results.json` + server logs in `logs/`.
