# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SGLang is a high-performance serving framework for LLMs and multimodal models. It provides RadixAttention prefix caching, zero-overhead CPU scheduling, prefill-decode disaggregation, speculative decoding, continuous batching, paged attention, and broad model/hardware support.

## Commands

### Install (development)

```bash
# Main Python package (editable)
pip install -e python/.

# With test dependencies
pip install -e "python/.[test]"

# sgl-kernel (AOT CUDA kernels — from sgl-kernel/ directory)
cd sgl-kernel && make install
```

### Run the server

```bash
# Launch LLM server (primary entry point)
sglang serve <model_path> --host 0.0.0.0 --port 30000

# Launch diffusion model server
sglang serve wan --model-path <model_path>

# Generate (diffusion/multimodal)
sglang generate <model_path> --prompt "..."
```

### Linting & formatting

```bash
# Pre-commit (isort, ruff, black, clang-format, codespell, nbstripout)
pre-commit run --all-files

# Python only
isort . && black .
ruff check --select=F401,F821 --fix .
```

### Testing

```bash
# Single test file
python -m pytest test/srt/test_afd_basic.py -xvs

# A single test case
python -m pytest test/srt/test_afd_basic.py::TestAFDBasicM3::test_single_request -xvs

# sgl-kernel tests
cd sgl-kernel && make test

# Run CI suite with registration
python test/srt/run_suite.py
```

Tests use `CustomTestCase` (from `sglang.test.test_utils`) as the base class. CI tests are registered via `CIRegistry` in `sglang.test.ci.ci_register`. Retriable failures (accuracy, latency, throughput) are distinguished from non-retriable ones (SyntaxError, ImportError, OOM) in `ci_utils.py`.

Tests live under `test/srt/` (not the repo root). AFD-related tests are in `test/srt/test_afd_basic.py` and `test/srt/test_afd_fixture.py`. Test configs (model args, env vars) are in `test/srt/configs/`.

## Architecture

The repo has three main packages and two independent sub-projects:

### `python/sglang/` — Main Python package

**CLI** (`cli/`): `sglang serve` auto-detects LLM vs. diffusion models. For LLMs, it calls `launch_server.run_server()`.

**Frontend Language** (`lang/`): A Python DSL for programmatic LLM interaction (`interpreter.py`, `ir.py`, `tracer.py`). Not required for the server — `sglang serve` uses the runtime engine directly.

**Runtime Engine** (`srt/`) — the core inference engine:

| Subsystem | Key Files | Purpose |
|---|---|---|
| Entrypoints | `entrypoints/http_server.py`, `entrypoints/openai/` | FastAPI HTTP server, OpenAI-compatible API |
| Scheduler | `managers/scheduler.py` + many mixins | Token-level batching, RadixAttention cache, request dispatch. Key mixins: `scheduler_afd_mixin.py` (AFD event loop), `scheduler_pp_mixin.py` (pipeline parallelism), `scheduler_dp_attn_mixin.py` (data-parallel attention), `scheduler_profiler_mixin.py`, `scheduler_runtime_checker_mixin.py`, `scheduler_update_weights_mixin.py`, `scheduler_output_processor_mixin.py` |
| Model executor | `model_executor/model_runner.py` | CUDA graph capture, forward pass orchestration |
| Models | `models/` | ~160+ model implementations (one file per model family) |
| Attention | `layers/attention/` | 20+ backends (FlashAttention, FlashInfer, Triton, MLA variants, etc.) |
| Quantization | `layers/quantization/` | FP8, FP4, INT4, AWQ, GPTQ, GGUF, MXFP4 |
| Memory/Cache | `mem_cache/` | RadixCache, chunk cache, prefix caching, HiCache, memory pool |
| Distributed | `distributed/` | Communication ops, device communicators, parallel state |
| Disaggregation | `disaggregation/` | Prefill-decode split: encode/decode servers, MoonCake/Mori/Nixl/fake backends |
| AFD (A/F Disagg) | `layers/afd.py`, `layers/afd_mixin.py`, `layers/afd_type.py`, `managers/scheduler_afd_mixin.py` | Attention-FFN operator-level disaggregation: `AFDPerspective` (ATTN/FFN), `AFDCommunicator` (ZMQ/StepMesh/UCX backends), `AFDProxyAttention`/`AFDProxyMLP`, `AFDDecoderLayerMixin`, `AFDWeightFilter`, heterogeneous TP (tp_A ≠ tp_F), microbatch pipeline (M=1,2,3) |
| DVFS | `layers/dvfs.py` | GPU frequency control via NVML `SetGpuLockedClocks`: `DVFSController` (per-GPU lock/unlock/reset), `DVFSManager` (node-wide), supports energy counter query |
| Speculative | `speculative/` | Eagle v1/v2, ngram speculative decoding |
| Constrained | `constrained/` | Structured JSON/grammar outputs (xgrammar, outlines, llguidance) |
| LoRA | `lora/` | Multi-LoRA batching with eviction policy |
| Compilation | `compilation/` | torch.compile, piecewise CUDA graph, inductor passes |
| Function call | `function_call/` | Model-specific function-call detection/parsing |

### `sgl-kernel/` — AOT CUDA/C++ kernel library

Built with scikit-build-core + CMake. Source in `csrc/`, Python bindings in `python/sgl_kernel/`. Covers: attention, MoE, gemm, quantization, allreduce, speculative, grammar, kvcacheio, mamba, elementwise, spatial ops. Built separately from the main package (`pip install sglang-kernel` or `make install` from `sgl-kernel/`).

### `python/sglang/jit_kernel/` — JIT-compiled kernels

Triton/CUDA kernels compiled at runtime: flash attention, MoE, ngram embedding. These are in the main Python package (not the AOT kernel library).

### `sgl-model-gateway/` — Rust-based proxy/gateway

Independent Rust project (`Cargo.toml`). gRPC client, MCP support, routing policies, WASM, tool/reasoning parsing.

## Key patterns

- **Model implementations** (`srt/models/`): Each model file defines a `*ForCausalLM` class (e.g., `LlamaForCausalLM`) inheriting from a base. Models use composition: attention layers, MoE layers, etc. are mixins from `srt/layers/`.
- **Server lifecycle**: `launch_server.py` → `scheduler.py` (main loop: schedule → run forward pass via model executor → sample → return tokens).
- **Disaggregation (PD)**: Prefill server (`disaggregation/encode.py`) and decode server (`disaggregation/decode.py`) communicate via MoonCake/Nixl transfer engines.
- **CI**: PR tests defined in `.github/workflows/`. Test registration via `CIRegistry` in `sglang/test/ci/ci_register.py`. Use `CustomTestCase` from `sglang.test.test_utils` for all test classes.

## Environment

- Python >= 3.10, CUDA 12.9, PyTorch 2.9.1, flashinfer 0.6.6, transformers 5.3.0
- `uv` is the preferred package manager (see `[[tool.uv.index]]` and `[tool.uv.sources]` in pyproject.toml)
- Build variants: CPU (`pyproject_cpu.toml`), NPU, XPU, ROCm, MUSA — each with separate pyproject config

## Research / Benchmark sub-projects

`benchmark/test_motivation/` contains energy profiling and AF-disaggregation motivation analysis:
- `bench_prefill_af.py` / `bench_decode_af.py` — Profile A/F latency + energy (NVML counters) across TP/freq/bs/seq_len grids
- `bench_dvfs_overhead.py` — Measure GPU frequency switching overhead
- `dvfs/` — C++ NVML wrappers (`libdvfs_ctrl.so`) for `SetGpuLockedClocks`
- `analyze_decode_v1.py` / `analyze_prefill_v1.py` — Energy saving analysis: AF-disaggregated vs unified frequency tuning
- `energy_model.py` — Train GBDT/LinearReg latency and energy predictors from profile data
- `prepare_trace.py` — Process Azure LLM traces for trace-driven evaluation
- Profile data: `decode_data_v1.txt` (7530 rows), `prefill_data_v1.txt` (906 rows)
