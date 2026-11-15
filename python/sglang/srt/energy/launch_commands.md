# Launch Commands

## External Resource Configuration (Read Before First Launch)

The public `benchmark/test_aflex` tools do not include model weights, profile data, or trained prediction models.
Both launcher JSON files are templates: copy them to local configuration files and change `model.path` from `null`
to your own model directory or model ID; GPUs, NICs, and model geometry parameters must match the actual deployment.
Relative paths are resolved against the launching process's current working directory; absolute paths are recommended for external resources.

When enabling DVFS or Tier 1, set `afd.energy_model_dir` to the external prediction model directory;
when enabling Tier 1 (including `--start-with-workload`), you must also set:
- `tier1.prefill_data_path`: `/path/to/aflex-assets/prefill_data_v1.txt`
- `tier1.decode_data_path`: `/path/to/aflex-assets/decode_data_v1.txt`
- `afd.energy_model_dir`: `/path/to/aflex-assets/energy_models`

These resources have no implicit fallback paths within the repository. If required configuration or files are missing, the launcher reports an error before starting child processes.
Minimal mode requires only an external model, not profiles or prediction models. Prediction models contain pickle files; load only trusted assets.
Replace the configuration paths in the launcher commands below with completed local copies; full mode also requires enabling the corresponding switches in those copies.
Likewise, replace `/path/to/...` in manual commands with actual external paths.

## 1. Using af_launcher.py (Recommended)

Supports full configuration + Tier 1 pre-solving + automatic GPU allocation + frequency locking.

### Full Mode (DVFS + Tier 1 + Pre-solving)

```bash
python python/sglang/srt/energy/af_launcher.py \
    --config python/sglang/srt/energy/af_launch_config.json \
    --start-with-workload
```

Startup order:**DF → DA → PF → PA → Router**

Module ports:
| Module | Role | GPU | Port | Description |
|---|---|---|---|---|
| DF | Decode-FFN | GPU 0 | 50021 | FFN-side decode |
| DA | Decode-Attn | GPU 1 | 50020 | Attention-side decode, UCX FFN host=127.0.0.1 |
| PF | Prefill-FFN | GPU 2 | 50011 | FFN-side prefill |
| PA | Prefill-Attn | GPU 3 | 50010 | Attention-side prefill + Tier 1 monitoring |
| Router | Routing | — | 50000 | PD routing + mini load balancing |

### Minimal Mode (No DVFS, No Tier 1)

```bash
python python/sglang/srt/energy/af_launcher.py \
    --config python/sglang/srt/energy/af_launch_config_minimal.json
```

### Without Pre-solving (Tier 1 Dynamic Monitoring Enabled)

```bash
python python/sglang/srt/energy/af_launcher.py \
    --config python/sglang/srt/energy/af_launch_config.json
# Enable tier1.enable_tier1_pa in the local config; the solver initializes lazily on re-plan.
```

## 2. Manual Launch (For Debugging)

### Environment Variables

```bash
# UCX communication
export AFD_UCX_BASE_PORT=<base_port>        # DF/DA: 25100, PF/PA: 25000
export AFD_UCX_FFN_HOST=<ffn_ip>            # Required on the Attn side: FFN host IP
export AFD_UCX_TLS="rc,tcp,cuda_copy,cuda_ipc"
export UCX_LOG_LEVEL=fatal
export UCX_WARN_UNUSED_ENV_VARS=n

# Scheduling
export AFD_SCHED_PORT=<sched_port>           # decode: 65400, prefill: 65300

# GPU
export CUDA_VISIBLE_DEVICES=<gpu_index>      # Single GPU
export AFD_NVML_DEVICE_INDEX=<nvml_index>    # Physical NVML GPU index (not CUDA index)
export AFD_ATTN_GPU_INDICES="<csv>"          # PA only: GPU index list for Attn modules
export AFD_FFN_GPU_INDICES="<csv>"           # PA only: GPU index list for FFN modules

# Logging
export SGLANG_DISABLE_REQUEST_LOGGING=true
```

### Decode-Attn (DA)

```bash
python -m sglang.launch_server \
    --model-path /path/to/model \
    --tp 1 \
    --port 50020 \
    --afd-perspective attn \
    --afd-comm-backend ucx \
    --disaggregation-mode decode \
    --disaggregation-bootstrap-port 18999 \
    --disaggregation-ib-device mlx5_4 \
    --mem-fraction-static 0.7 \
    --afd-dvfs-enabled \
    --afd-energy-model-dir /path/to/aflex-assets/energy_models \
    --tier1-stats-path af_launch_logs/decode_stats.json
```

Environment variables:
```bash
CUDA_VISIBLE_DEVICES=1
AFD_UCX_BASE_PORT=25100
AFD_SCHED_PORT=65400
AFD_UCX_FFN_HOST=127.0.0.1
AFD_NVML_DEVICE_INDEX=1
```

### Decode-FFN (DF)

```bash
python -m sglang.launch_server \
    --model-path /path/to/model \
    --tp 1 \
    --port 50021 \
    --afd-perspective ffn \
    --afd-comm-backend ucx \
    --disaggregation-mode decode \
    --disaggregation-bootstrap-port 18999 \
    --disaggregation-ib-device mlx5_4 \
    --mem-fraction-static 0.7 \
    --tier1-stats-path af_launch_logs/decode_stats.json
```

Environment variables:
```bash
CUDA_VISIBLE_DEVICES=0
AFD_UCX_BASE_PORT=25100
AFD_SCHED_PORT=65400
AFD_NVML_DEVICE_INDEX=0
```

### Prefill-Attn (PA)

```bash
python -m sglang.launch_server \
    --model-path /path/to/model \
    --tp 1 \
    --port 50010 \
    --afd-perspective attn \
    --afd-comm-backend ucx \
    --disaggregation-mode prefill \
    --disaggregation-bootstrap-port 18999 \
    --disaggregation-ib-device mlx5_4 \
    --mem-fraction-static 0.7 \
    --afd-dvfs-enabled \
    --afd-energy-model-dir /path/to/aflex-assets/energy_models \
    --enable-tier1-pa \
    --tier1-gpu-count 4 \
    --tier1-lambda-prefill 10.0 \
    --tier1-n-active-decode 32 \
    --tier1-il-rep-p 1024 \
    --tier1-bs-avg-p 8 \
    --tier1-il-rep-d 512 \
    --tier1-ol-rep-d 256 \
    --tier1-bs-avg-d 16 \
    --tier1-monitor-window-s 30.0 \
    --tier1-prefill-data-path /path/to/aflex-assets/prefill_data_v1.txt \
    --tier1-decode-data-path /path/to/aflex-assets/decode_data_v1.txt \
    --tier1-stats-path af_launch_logs/decode_stats.json
```

Environment variables:
```bash
CUDA_VISIBLE_DEVICES=3
AFD_UCX_BASE_PORT=25000
AFD_SCHED_PORT=65300
AFD_UCX_FFN_HOST=127.0.0.1
AFD_NVML_DEVICE_INDEX=3
AFD_ATTN_GPU_INDICES="1,3"      # DA + PA
AFD_FFN_GPU_INDICES="0,2"       # DF + PF
```

### Prefill-FFN (PF)

```bash
python -m sglang.launch_server \
    --model-path /path/to/model \
    --tp 1 \
    --port 50011 \
    --afd-perspective ffn \
    --afd-comm-backend ucx \
    --disaggregation-mode prefill \
    --disaggregation-bootstrap-port 18999 \
    --disaggregation-ib-device mlx5_4 \
    --mem-fraction-static 0.7 \
    --tier1-stats-path af_launch_logs/decode_stats.json
```

Environment variables:
```bash
CUDA_VISIBLE_DEVICES=2
AFD_UCX_BASE_PORT=25000
AFD_SCHED_PORT=65300
AFD_NVML_DEVICE_INDEX=2
```

### Router

```bash
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --mini-lb \
    --prefill http://127.0.0.1:50010 \
    --decode http://127.0.0.1:50020 \
    --host 127.0.0.1 \
    --port 50000
```

## 3. Log Files

Default log directory:`af_launch_logs/`

| File | Contents |
|---|---|
| `PA.log` | Prefill-Attn server log |
| `PF.log` | Prefill-FFN server log |
| `DA.log` | Decode-Attn server log |
| `DF.log` | Decode-FFN server log |
| `router.log` | Router server log |
| `decode_stats.json` | Decode timing statistics written by DA (read by PA Tier 1 monitoring) |
| `tier1_initial_solution.json` | Tier 1 configuration produced by pre-solving (tp/f/k) |

The log level is controlled by `logs.level` in the configuration and defaults to `info`.

## 4. Sending Requests

Send through the router (recommended) or directly to PA/DA:

```bash
# Through the router
curl http://127.0.0.1:50000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 128
    }'

# Directly to prefill
curl http://127.0.0.1:50010/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 128
    }'
```

## 5. Stopping

```bash
# Stop all child processes (started by af_launcher.py)
pkill -f "sglang.launch_server"
pkill -f "launch_router"
```
