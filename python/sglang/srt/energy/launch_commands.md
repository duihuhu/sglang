# 启动命令

## 1. 使用 af_launcher.py（推荐）

支持完整配置 + Tier 1 预求解 + 自动 GPU 分配 + 频率锁定。

### 完整模式（DVFS + Tier 1 + 预求解）

```bash
python python/sglang/srt/energy/af_launcher.py \
    --config python/sglang/srt/energy/af_launch_config.json \
    --start-with-workload
```

启动顺序：**DF → DA → PF → PA → Router**

各模块端口：
| 模块 | 角色 | GPU | 端口 | 描述 |
|---|---|---|---|---|
| DF | Decode-FFN | GPU 0 | 50021 | FFN 侧 decode |
| DA | Decode-Attn | GPU 1 | 50020 | Attention 侧 decode, UCX FFN host=127.0.0.1 |
| PF | Prefill-FFN | GPU 2 | 50011 | FFN 侧 prefill |
| PA | Prefill-Attn | GPU 3 | 50010 | Attention 侧 prefill + Tier 1 监控 |
| Router | 路由 | — | 50000 | PD 路由 + 最小负载均衡 |

### 最小模式（无 DVFS，无 Tier 1）

```bash
python python/sglang/srt/energy/af_launcher.py \
    --config python/sglang/srt/energy/af_launch_config_minimal.json
```

### 无预求解（PA 启动时运行 solver）

```bash
python python/sglang/srt/energy/af_launcher.py \
    --config python/sglang/srt/energy/af_launch_config.json
# 不传 --start-with-workload, PA 会在 __init__ 中调用 solver
```

## 2. 手动启动（调试用）

### 环境变量

```bash
# UCX 通信
export AFD_UCX_BASE_PORT=<base_port>        # DF/DA: 25100, PF/PA: 25000
export AFD_UCX_FFN_HOST=<ffn_ip>            # Attn 侧需要: FFN 主机 IP
export AFD_UCX_TLS="rc,tcp,cuda_copy,cuda_ipc"
export UCX_LOG_LEVEL=fatal
export UCX_WARN_UNUSED_ENV_VARS=n

# 调度
export AFD_SCHED_PORT=<sched_port>           # decode: 65400, prefill: 65300

# GPU
export CUDA_VISIBLE_DEVICES=<gpu_index>      # 单个 GPU
export AFD_NVML_DEVICE_INDEX=<nvml_index>    # 物理 NVML GPU index（非 CUDA index）
export AFD_ATTN_GPU_INDICES="<csv>"          # PA 专用: Attn 模块的 GPU index 列表
export AFD_FFN_GPU_INDICES="<csv>"           # PA 专用: FFN 模块的 GPU index 列表

# 日志
export SGLANG_DISABLE_REQUEST_LOGGING=true
```

### Decode-Attn (DA)

```bash
python -m sglang.launch_server \
    --model-path /models/llama3.1-8 \
    --tp 1 \
    --port 50020 \
    --afd-perspective attn \
    --afd-comm-backend ucx \
    --disaggregation-mode decode \
    --disaggregation-bootstrap-port 18999 \
    --disaggregation-ib-device mlx5_4 \
    --mem-fraction-static 0.7 \
    --afd-dvfs-enabled \
    --afd-energy-model-dir /workspace/sglang/benchmark/test_motivation/energy_models \
    --tier1-stats-path af_launch_logs/decode_stats.json
```

环境变量：
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
    --model-path /models/llama3.1-8 \
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

环境变量：
```bash
CUDA_VISIBLE_DEVICES=0
AFD_UCX_BASE_PORT=25100
AFD_SCHED_PORT=65400
AFD_NVML_DEVICE_INDEX=0
```

### Prefill-Attn (PA)

```bash
python -m sglang.launch_server \
    --model-path /models/llama3.1-8 \
    --tp 1 \
    --port 50010 \
    --afd-perspective attn \
    --afd-comm-backend ucx \
    --disaggregation-mode prefill \
    --disaggregation-bootstrap-port 18999 \
    --disaggregation-ib-device mlx5_4 \
    --mem-fraction-static 0.7 \
    --afd-dvfs-enabled \
    --afd-energy-model-dir /workspace/sglang/benchmark/test_motivation/energy_models \
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
    --tier1-prefill-data-path benchmark/test_motivation/hucc/paper/prefill_data_v1.txt \
    --tier1-decode-data-path benchmark/test_motivation/hucc/paper/decode_data_v1.txt \
    --tier1-stats-path af_launch_logs/decode_stats.json
```

环境变量：
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
    --model-path /models/llama3.1-8 \
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

环境变量：
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

## 3. 日志文件

默认日志目录：`af_launch_logs/`

| 文件 | 内容 |
|---|---|
| `PA.log` | Prefill-Attn 服务器日志 |
| `PF.log` | Prefill-FFN 服务器日志 |
| `DA.log` | Decode-Attn 服务器日志 |
| `DF.log` | Decode-FFN 服务器日志 |
| `router.log` | 路由服务器日志 |
| `decode_stats.json` | DA 写入的 decode 计时统计（给 PA Tier 1 监控读取） |
| `tier1_initial_solution.json` | 预求解产生的 Tier 1 配置（tp/f/k） |

日志级别通过配置 `logs.level` 控制，默认 `info`。

## 4. 发送请求

通过 router（推荐）或直接发送到 PA/DA：

```bash
# 通过 router
curl http://127.0.0.1:50000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 128
    }'

# 直接到 prefill
curl http://127.0.0.1:50010/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 128
    }'
```

## 5. 停止

```bash
# 停止所有子进程（af_launcher.py 启动的）
pkill -f "sglang.launch_server"
pkill -f "launch_router"
```
