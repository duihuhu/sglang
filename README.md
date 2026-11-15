# AFlex

AFlex is an energy-efficient LLM serving system built on [SGLang](https://github.com/sgl-project/sglang).
It disaggregates Prefill / Decode (P/D) and Attention / FFN (A/F), combining operator-level prediction, global resource allocation, and local DVFS to optimize energy under GPU budgets and latency constraints.
This repository retains the SGLang serving stack; the four modules below map the design to its implementation.

## 1. Operator-Level Profiling and Prediction

Operator-level profiling measures A/F latency and energy for Prefill / Decode. Predictors provide estimates for global planning and local frequency selection.
Profiles must cover TP, frequency, batch size, and sequence length on the actual hardware; trained models do not guarantee portability across GPUs.

Source map:
- [profiling/](python/sglang/srt/energy/profiling/): Prefill / Decode profiling and data normalization.
- [prediction/](python/sglang/srt/energy/prediction/): Prediction model definitions, training, and loading.
- [af_profile_predictor.py](python/sglang/srt/energy/af_profile_predictor.py): Runtime prediction interface, with exact LUT lookup before learned-model fallback.
- [profile_table.py](python/sglang/srt/energy/profile_table.py): Latency and energy estimates for the planner.

The profiling-to-training workflow has three steps. Training inputs, units, and runtime features must remain consistent; see [prediction/README.md](python/sglang/srt/energy/prediction/README.md) for detailed usage.
The examples use user-provided models and writable external paths. Profiling requires CUDA/NVML and clock-control permissions; `--quick` checks the workflow but does not replace full profiling.
Ordinary `python -m sglang...` commands, including `--help`, require the complete SGLang environment and its base dependencies.

1. Profile Prefill / Decode on the target GPUs:

```bash
python -m sglang.srt.energy.profiling.bench_prefill_af --model-path /external/model --tp-size 1 --output /external/prefill_raw.tsv --quick
python -m sglang.srt.energy.profiling.bench_decode_af --model-path /external/model --tp-size 1 --output /external/decode_raw.tsv --quick
```

2. Normalize headers while preserving latency in microseconds and per-layer energy in mJ for the rank-0 GPU. Energy is not summed across TP ranks: do not scale it by TP or by 1,000.

```bash
python -m sglang.srt.energy.profiling.normalize --phase prefill --input /external/prefill_raw.tsv --output /external/prefill.tsv
python -m sglang.srt.energy.profiling.normalize --phase decode --input /external/decode_raw.tsv --output /external/decode.tsv
```

3. Train predictors from the normalized data with five-fold cross-validation:

```bash
python -m sglang.srt.energy.prediction.train --prefill /external/prefill.tsv --decode /external/decode.tsv --output-dir /external/models --folds 5
```

Train models on the target hardware and supply them through local paths. Load pickle files only from trusted sources.

## 2. Global Resource Controller

The global controller selects replica counts, TP, and initial frequencies for four pools: PA, PF, DA, and DF. It also provides monitoring and reconfiguration components.
The planner uses candidate enumeration, constraint pruning, and Pareto filtering grouped by GPU count to compare energy under GPU-budget, SLO, memory, and capacity constraints; it does not rely on an ILP solver.

Source map:
- [tier1_solver.py](python/sglang/srt/energy/tier1_solver.py): `Tier1Solver`, constraints, and plan search.
- [af_launcher.py](python/sglang/srt/energy/af_launcher.py): Config-driven launch of DF, DA, PF, PA, and routers, with optional initial planning before launch.
- [workload_collector.py](python/sglang/srt/energy/workload_collector.py) / [workload_monitor.py](python/sglang/srt/energy/workload_monitor.py): Workload statistics and hysteresis-based replanning triggers.
- [reload_orchestrator.py](python/sglang/srt/energy/reload_orchestrator.py) / [reload_signal.py](python/sglang/srt/energy/reload_signal.py): Reload coordination.
- [afd_component_reshard.py](python/sglang/srt/reshard/afd_component_reshard.py): Component reshard requests, epoch validation, and phase state machine.
- [afd_component_runtime.py](python/sglang/srt/reshard/afd_component_runtime.py) / [afd_component_adapter.py](python/sglang/srt/reshard/afd_component_adapter.py): Control-plane and scheduler execution boundaries.
- [afd_component_weight_staging.py](python/sglang/srt/reshard/afd_component_weight_staging.py) / [afd_weight_reshard.py](python/sglang/srt/reshard/afd_weight_reshard.py): Weight preparation, migration, and activation.

Monitoring, planning, and reconfiguration are optional components, not a complete closed loop automatically enabled in the default macrobenchmarks.
Component resharding currently requires single-node Qwen3 dense, PP=DP=1, M=1, disabled CUDA Graphs, and enough prestarted max-world ranks.
Supported transitions are TP2 to TP4 for A/F and TP4 to TP1 for Attention only; FFN may retain its original TP.
Arbitrary bidirectional TP scaling and cross-node live migration are not supported. Actual commits must also pass model, weight-layout, communication, and runtime-capability checks.
A successful control-plane dry-run or validation does not mean weights have migrated.

## 3. AFD Data Plane + Interleaved A/F Pipeline

The AFD data plane places decoder Attention and FFN operators in separate processes and supports heterogeneous A/F TP.
Communication options include single-node CUDA IPC (`ipc` / `ipc_cpp`) and UCX (requiring UCX-Py). Configure communication and ports for the deployment topology.

Source map:
- [layers/afd.py](python/sglang/srt/layers/afd.py): Communication, proxy operators, and pipeline scheduling; `AFDStageScheduleGenerator.attn_stage_interleaved` generates interleaved Attention stages.
- [layers/afd_mixin.py](python/sglang/srt/layers/afd_mixin.py): Decoder-layer integration.
- [managers/scheduler_afd_mixin.py](python/sglang/srt/managers/scheduler_afd_mixin.py): A/F event loops, microbatch splitting, and scheduler coordination.
- [batch_overlap/afd_overlap.py](python/sglang/srt/batch_overlap/afd_overlap.py): AFD-aware CPU/GPU overlap scheduling.

`--afd-micro-batch` sets M. Interleaved pipelining and CPU/GPU overlap operate at different levels.
`--afd-dynamic-micro-batch` switches between M=1 and the configured M using a Decode batch-size threshold; it is not a profile-based argmin search.
Historical dense macrobenchmarks used M=1 and therefore do not establish that interleaved microbatch pipelining ran by default.

## 4. Local DVFS Controller

The local controller selects and applies A/F frequencies before batch execution.
Prefill uses a processing-time budget. Decode reevaluates over windows, considering batch changes, SLO pressure, and switching costs to avoid unproductive frequency changes.

Source map:
- [af_dvfs_controller.py](python/sglang/srt/energy/af_dvfs_controller.py): Prefill frequency selection, Decode windows, and lazy switching policy.
- [managers/scheduler.py](python/sglang/srt/managers/scheduler.py): Pre-execution integration, budget calculation, and hardware frequency control.
- [layers/dvfs.py](python/sglang/srt/layers/dvfs.py): Native-library wrapper, clock locking, unlocking, and reset.
- [native/dvfs/build.py](python/sglang/srt/energy/native/dvfs/build.py): Standalone build entry for the native NVML library.

`_compute_prefill_slack` computes the remaining budget from the start of Prefill processing. It excludes queueing time and is not an end-to-end deadline measured from request arrival.
Hardware frequency control uses NVML `SetGpuLockedClocks` to lock SM clocks and requires NVIDIA drivers, NVML, supported frequencies, and clock-control permissions.
Calibrate both frequency candidates and prediction models for the actual GPUs.

## Directory Layout

```text
python/sglang/
  cli/                          # Service CLI
  srt/
    energy/
      profiling/                # Operator profiling and normalization
      prediction/               # Prediction models and training
      native/dvfs/              # Packaged NVML source and build entry
      af_profile_predictor.py    # Runtime prediction
      profile_table.py          # Planner estimates
      tier1_solver.py           # Enumeration and Pareto search
      af_launcher.py            # Config-driven launcher
      workload_monitor.py       # Replanning triggers
      af_dvfs_controller.py      # Local DVFS policy
    layers/afd.py               # A/F data plane and interleaved stages
    layers/afd_mixin.py         # Decoder integration
    layers/dvfs.py              # NVML wrapper
    managers/scheduler.py       # Runtime DVFS integration
    managers/scheduler_afd_mixin.py
    batch_overlap/afd_overlap.py
    reshard/                    # Component TP reshard
benchmark/test_aflex/           # Standalone end-to-end harness
sgl-kernel/                     # AOT CUDA kernels
test/srt/                      # Runtime and CPU interface tests
```

`profiling/` and `prediction/` reside under `python/sglang/srt/energy/`, not at the repository root.
Historical `benchmark/AFlex_bench/` and `benchmark/test_motivation/` remain ignored and unchanged; they are not public run entry points.
The release includes source, configuration, and small synthetic examples only, not collected profiles, `.pkl` models, raw traces, logs, or experimental results.

## Installation and Build

Python >= 3.10 is recommended. The [dependency configuration](python/pyproject.toml) targets PyTorch 2.9.1 and CUDA 12.9 components.
Prepare compatible drivers and a CUDA environment first, then run from the repository root:

```bash
# Install the runtime and AFlex prediction dependencies
uv pip install -e "python/.[aflex]"

# Optional: build local AOT kernels instead of using the packaged wheel
(cd sgl-kernel && make install)

# Build the native NVML wrapper outside the source tree
python -m sglang.srt.energy.native.dvfs.build --output /external/libdvfs_ctrl.so --cuda-home /usr/local/cuda
export DVFS_CTRL_LIB=/external/libdvfs_ctrl.so
```

The native build requires a C++ compiler, `nvml.h`, and the NVIDIA management library. Building does not itself load NVML or operate GPUs.
`/external` denotes user-provided writable storage. At runtime, link against the real driver library; do not add CUDA stubs to the runtime library search path.

## Serving CLI and Key Options

First verify the model environment with ordinary SGLang serving; these commands do not enable A/F or DVFS:

```bash
sglang serve /path/to/model --host 0.0.0.0 --port 30000
python -m sglang.launch_server --help
```

A/F requires paired Attention / FFN processes, GPUs, communication settings, and ports. P/D additionally requires matching transfer and routing configuration.
Update the model, devices, and local paths in the [minimal launch configuration](python/sglang/srt/energy/af_launch_config_minimal.json) before running:

```bash
python -m sglang.srt.energy.af_launcher --config python/sglang/srt/energy/af_launch_config_minimal.json
```

Key options are defined in [server_args.py](python/sglang/srt/server_args.py):
- A/F: `--afd-perspective {attn,ffn}`, `--afd-attn-tp`, `--afd-ffn-tp`, and `--afd-comm-backend`.
- IPC: `--afd-ipc-per-rank` applies only to `ipc_cpp` with homogeneous TP.
- Pipeline: `--afd-micro-batch`, `--afd-dynamic-micro-batch`, and `--afd-dynamic-mb-threshold`; `--afd-enable-overlap-schedule` explicitly enables AFD-aware overlap.
- DVFS: `--afd-dvfs-enabled` requires an A/F role and `--afd-energy-model-dir`.
- Constraints: `--afd-ttft-slo-ms` is in milliseconds and `--afd-tpot-slo-us` in microseconds; see Section 4 for Prefill budget semantics.
- Global control: `--enable-tier1-pa` and `--tier1-initial-solution` must match the launcher configuration.
- Resharding: The Attention HTTP coordinator uses `--enable-afd-component-reshard`; participants use `--enable-afd-component-reshard-participant`.
  Also configure `--afd-component-max-tp`, `--afd-reshard-stage-id {prefill,decode}`, `--afd-reshard-pair-id`, and `--disable-cuda-graph`, subject to the limits in Section 2.

## Standalone End-to-End Evaluation

See [benchmark/test_aflex/README.md](benchmark/test_aflex/README.md) for entry points and deployment boundaries.
First check the small synthetic configuration on CPU. These commands do not use SSH or Docker, operate GPUs, or write experimental results:

```bash
python benchmark/test_aflex/run_benchmark.py --help
python benchmark/test_aflex/run_benchmark.py preflight --config benchmark/test_aflex/examples/config.json
python benchmark/test_aflex/run_benchmark.py dry-run --config benchmark/test_aflex/examples/config.json
```

Example IPs are reserved documentation addresses and cannot be deployed directly. Passing preflight does not establish remote model, port, or RDMA availability.
Set `paths.dvfs_library` to the native library's absolute path inside the container and list the same path in `preflight.native_libraries`. The harness exports the corresponding `DVFS_CTRL_LIB` for all remote commands and launched processes; see the evaluation directory documentation.
Actual runs require user-configured clusters, containers, GPUs/NICs, model and predictor paths, exclusive use of the selected GPUs, and the explicit execution flag:

```bash
python benchmark/test_aflex/run_benchmark.py run --config /external/cluster.json --execute --output /external/results/run01.json
```

The harness accepts user-provided feasible plans; it does not automatically run an offline optimizer or select the optimal plan.
Keep profiles, training assets, and results in external storage. Synthetic examples validate interfaces only and do not establish paper-level performance or energy savings.
