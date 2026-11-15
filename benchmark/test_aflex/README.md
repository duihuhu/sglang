# AFlex End-to-End Macrobenchmark

This directory independently preserves the original macrobenchmark's real deployment implementation without importing or running the original ignored experiment directory. It originates from `macro_bench/aflex.py`, `deployments.py`, `fabric_affinity.py`, and the indexed SSE sender in `bench_common.py` used by the original `run_benchmark.py`; the repository license is retained. Only source code, configuration, and two small synthetic requests are published; original traces, profiles, model predictors, pkl files, and results are not included.

## Supported Schemes

- `aflex_tier1`: decode-first, FFN-first GPU allocation, heterogeneous Attention/FFN TP, IPC C++, Mooncake PD, and multiple Prefill subrouters; the client uses weighted round-robin based on `max(tp_pa,tp_pf)` for each pair; planned frequency initialization and runtime compositional DVFS.
- `native_tp1_baseline` / `native_tp1_tier`: eight TP1 workers on each of two nodes, round-robin routing, and fixed frequencies or DynamoLLM unified DVFS.
- `pd_hetero_baseline` / `pd_hetero_tier_biscale`: 2×TP4 Prefill and 4×TP2 Decode, Mooncake, and fixed frequencies or BiScale DVFS.

These baselines are comparison policies implemented within the repository's SGLang runtime, not official external Dynamo/DistServe deployments. MegaScale, plotting, frequency timelines, RDMA/NVLink counters, and four-node parallelism are not currently included.

## Configuration and Inputs

The IPs in `examples/config.json` are documentation-reserved addresses and cannot be used directly for real experiments. You must configure all container paths, SSH users, container names, models, energy predictors, GPU/NIC mappings, Mooncake rank IP bases, ports, frequencies, decision and evaluation SLOs, and request and plan files for each test point.

Local `workload` / `plan` paths are relative to the configuration file; container paths must be absolute and contain no shell special characters. To preserve the original baseline comparison semantics, the physical GPU pool on each node must currently be `0..7`; AFlex uses only the subset actually allocated. GPU/NIC mappings may differ by node. Adding the physical GPU ID to the last octet of `mooncake_ip_base` must yield a genuinely usable RDMA rank IP; an administrator must configure the network environment in advance.

You must configure `paths.dvfs_library` (the example is `/assets/libdvfs_ctrl.so`) and include the same path in `preflight.native_libraries`. This path must be available inside the containers on both nodes and must be absolute with no shell special characters. After entering the configured repo, the harness exports `DVFS_CTRL_LIB` for all remote commands; service startup and child processes inherit this variable, without relying on environment variables preset in the local terminal or container. Local configuration validation fails immediately if the path is missing or absent from the manifest.

Request input is JSONL, with each line containing at least positive integers `input_len` and `output_len`, and a nonnegative, nondecreasing `arrival_time_s`. Optional `use_input_ids:true` repeats token ID 1000 to generate inputs of exact length; the default retains the original script's `"x" * input_len`, whose character count is not guaranteed to equal the tokenizer's token count.

An AFlex plan file is a JSON object with the fields shown in `examples/plan.json`: `name,k_p,k_d,tp_pa,tp_pf,tp_da,tp_df,f_pa,f_pf,f_da,f_df,tier`. Frequencies are in MHz; optional `prefill_specs` is an array of exactly `k_p` objects, each containing `tp_pa,tp_pf,f_pa,f_pf`, to override the TP/frequencies of each Prefill pair. Configuration, rendering, deployment, and failure cleanup scopes all use the same parser. No offline optimizer runs here; users must supply feasible plans. The small synthetic plan does not imply any performance-optimality claim.

## CPU-Safe Entry Points

Run from the repository root:

```bash
python benchmark/test_aflex/run_benchmark.py --help
python benchmark/test_aflex/run_benchmark.py preflight --config benchmark/test_aflex/examples/config.json
python benchmark/test_aflex/run_benchmark.py dry-run --config benchmark/test_aflex/examples/config.json
python -m pytest test/srt/test_aflex_benchmark.py -q
```

`preflight` validates local configuration, requests, and plans and renders through the real deployment functions; `dry-run` outputs complete command JSON. Both disable actual transport, HTTP, NVML, and waiting; they do not use SSH or docker, operate GPUs, or write result files. Installing torch, CUDA, or pynvml is not required. Offline preflight does not check remote models/predictors, port availability, RDMA connectivity, container dependencies, or GPU models; `OFFLINE_OK` does not mean the remote deployment is runnable. In contrast, before any frequency locking or service deployment, `run --execute` performs read-only remote preflight on both nodes: Python version, runtime/router modules, configured native shared libraries and their loading, the model directory and config.json, and the predictor file manifest (existence, readability, and nonempty files). Remote preflight failures are aggregated by node and terminate the run without unlocking or modifying GPUs.

## Preparing an Azure Trace

```bash
python benchmark/test_aflex/prepare_trace.py --input /external/AzureLLMInferenceTrace_code.csv --output /external/macro_code_qps2.jsonl --qps 2 --n-requests 200 --seed 42
```

This follows the original macro workload preparation script's conversion semantics: read the first 500000 rows of `ContextTokens,GeneratedTokens`, filter nonpositive lengths, cap them at 4096/1024 respectively, sample using the seed and then shuffle, and use `seed+qps` to generate exponential intervals, accumulate them, and retain four decimal places. `TIMESTAMP` is not used to generate Poisson arrivals; this is neither a direct replay of Azure's original timestamps nor the grid-snapping conversion used in motivation experiments. Output defaults to the original macrobenchmark's text input semantics; existing output files are not overwritten.

## Real Runs (Explicit Authorization Flag Required)

```bash
python benchmark/test_aflex/run_benchmark.py run --config /external/cluster.json --execute --output /external/results/run01.json
```

First deploy a runtime matching this repository, sglang_router, the IPC JIT build environment, Mooncake, NVML Python bindings, and energy model assets. `preflight.native_libraries` must list absolute paths to required shared libraries inside the containers, and `preflight.prediction_files` must list all required prediction files relative to energy_model (the example uses the default V1 predictor combination); neither list may be empty. The experiment user maintains these manifests; pkl files are not deserialized. `ports.bootstrap+i` is the explicit bootstrap port for the i-th PA; it is checked for availability before startup, and the same value is passed to the service and subrouter. The configured repo should already be installed in the container's Python environment. This tool does not copy or rewrite remote runtime files. SSH verifies hosts using the system known_hosts and does not automatically bypass host verification. Runs require frequency-locking privileges and exclusive access to the configured GPUs. Each test point is redeployed; after health checks and warmup, concurrent SSE requests are issued according to arrival times. The total request window follows the original `min(max(400,last_arrival+150),900)` second rule, after which results are recorded.

Full success requires receiving `finish_reason.type=length` and a reported completion token count matching the target; chunk counts are not token counts. Statistics include TTFT and TPOT percentiles, throughput, per-request results, timeouts/failures, and the request-level rate of violating either SLO. TTFT preferentially uses server-side processing-only time, falling back to client TTFT if that field is absent. Energy is the difference in NVML cumulative millijoule counters for the GPUs actually allocated; missing, nonpositive, or regressing counters raise errors and are never silently replaced with zero. The mJ/token denominator is the input plus output lengths of successful requests; text input length fields may only represent character budgets, so papers should state the input mode explicitly. Deployment and warmup are excluded from energy measured over this request window.

## Safety and Limitations

Only processes from this run are terminated: each service uses a separate session and a random `AFLEX_BENCH_OWNER`, with its PID retained. Before cleanup, the token in `/proc/PID/environ` and the session ID are checked, then TERM is sent to that process group and surviving members are awaited. If the leader has disappeared, token-owned descendants in the same session are scanned; signals are refused if any member cannot be verified. The default wait is 15 seconds; on timeout, ownership records are retained and an error is raised. With `cleanup.force_kill:true`, ownership is reverified after the TERM timeout before KILL is sent, followed by another wait for exit. Cleanup is repeatable, already-exited sessions count as successful, and records are deleted only after exit is confirmed. There is no broad pkill, killing by port, or invocation of old cleanup scripts. An occupied service port causes failure without killing existing listeners. Frequency locking applies only to configured/allocated GPUs, and frequencies are reset at the end; arbitrary custom frequency settings from before the experiment are not restored, so exclusive access to these GPUs is required.

Forcefully terminating the orchestrator, losing SSH connectivity, processes ignoring TERM (without safe force-kill enabled), or processes escaping the managed session may require an administrator to intervene manually using the logs and ownership token; no cross-run global cleanup is provided. Logs are written to the configured remote_output and results to the specified external output file; they are not automatically packaged into the source directory. Automatic resume, optimal-plan selection, asset downloads, and paper-figure generation are not currently supported. Real GPU, SSH, and docker workflows have not been executed or validated in this delivery environment; CPU unit tests establish only configuration, command rendering, and statistics logic.
