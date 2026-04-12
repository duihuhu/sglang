# Adapted from qwen2.py
import json
import logging
import os
import time
import csv
import atexit
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.qwen2 import Qwen2MLP as Qwen3MLP
from sglang.srt.models.qwen2 import Qwen2Model
from sglang.srt.models.qwen2 import (
    _sync_bench_window_active,
    _sync_internal_op_bench_enabled,
    _sync_qwen3_loop_bench_enabled,
)
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, is_cuda, is_npu

Qwen3Config = None

logger = logging.getLogger(__name__)
_is_cuda = is_cuda()
_is_npu = is_npu()
_latency_lock = Lock()
_latency_values: Dict[str, List[float]] = {"TTFT": [], "A": [], "F": []}
_energy_values_uj: Dict[str, List[float]] = {"A": [], "F": []}
_nvml_lock = Lock()

# Per-job bench config read from the JSON phase file.
# When the file changes (mtime), accumulators are cleared so each job starts fresh.
_bench_job_cfg: Dict[str, object] = {}
_bench_job_cfg_phase: str = ""
_bench_job_cfg_file_mtime: float = 0.0


def _refresh_bench_job_config() -> None:
    """Read the phase/config file.  If JSON, update per-job config and phase.
    If plain text, update phase only (backward compat).
    On config change, reset latency/energy accumulators."""
    global _bench_job_cfg, _bench_job_cfg_phase, _bench_job_cfg_file_mtime

    phase_file = os.getenv("SGLANG_BENCH_PHASE_FILE", "").strip()
    if not phase_file:
        return

    try:
        mt = os.path.getmtime(phase_file)
    except OSError:
        return
    if mt == _bench_job_cfg_file_mtime:
        return

    _bench_job_cfg_file_mtime = mt
    try:
        with open(phase_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except Exception:
        return
    if not content:
        return

    try:
        data = json.loads(content)
        if isinstance(data, dict):
            _bench_job_cfg = data
            _bench_job_cfg_phase = str(data.get("phase", "none")).strip().lower()
            with _latency_lock:
                for arr in _latency_values.values():
                    arr.clear()
                for arr in _energy_values_uj.values():
                    arr.clear()
            return
    except (json.JSONDecodeError, ValueError):
        pass

    # Plain text — just the phase string.
    _bench_job_cfg_phase = content.lower()


def _bench_cfg_int(key: str, default: int) -> int:
    val = _bench_job_cfg.get(key)
    if val is not None:
        try:
            return int(val)
        except (ValueError, TypeError):
            pass
    env_map = {"input_len": "SGLANG_BENCH_INPUT_LEN", "batch_size": "SGLANG_BENCH_BATCH_SIZE"}
    env_name = env_map.get(key)
    if env_name:
        return _env_int(env_name, default)
    return default


def _bench_cfg_str(key: str, default: str = "") -> str:
    val = _bench_job_cfg.get(key)
    if val is not None:
        return str(val)
    env_map = {
        "tp": "SGLANG_BENCH_TP",
        "sm_clock": "SGLANG_BENCH_SM_CLOCK",
        "csv_path": "SGLANG_TTFT_AF_CSV_PATH",
    }
    env_name = env_map.get(key)
    if env_name:
        return os.getenv(env_name, default)
    return default
_nvml_pynvml = None
_nvml_handle = None
_nvml_init_attempted = False


if _is_npu:
    from sgl_kernel_npu.norm.split_qkv_rmsnorm_rope import split_qkv_rmsnorm_rope

    from sglang.srt.hardware_backend.npu.cmo import get_cmo_stream, wait_cmo_stream


def _get_af_profile_repeat() -> int:
    try:
        v = int(os.getenv("SGLANG_SYNC_BENCH_NUM_ITERS", "1"))
    except Exception:
        v = 1
    return max(v, 1)


def _get_af_profile_warmup() -> int:
    try:
        v = int(os.getenv("SGLANG_SYNC_BENCH_WARMUP_ITERS", "0"))
    except Exception:
        v = 0
    return max(v, 0)


def _try_init_nvml_for_current_device():
    global _nvml_pynvml, _nvml_handle, _nvml_init_attempted
    if not torch.cuda.is_available():
        return None, None
    with _nvml_lock:
        if _nvml_init_attempted:
            return _nvml_pynvml, _nvml_handle
        if _nvml_pynvml is not None and _nvml_handle is not None:
            return _nvml_pynvml, _nvml_handle
        _nvml_init_attempted = True
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            # Logical cuda:0 may map to a different physical GPU when CUDA_VISIBLE_DEVICES is set.
            # Match NVML handle to the same device as torch via UUID (see matmul_power_bench NVML usage).
            device_idx = int(torch.cuda.current_device())
            uuid_str = str(torch.cuda.get_device_properties(device_idx).uuid)
            if not uuid_str.startswith("GPU-"):
                uuid_str = "GPU-" + uuid_str
            handle = pynvml.nvmlDeviceGetHandleByUUID(uuid_str.encode("utf-8"))
            _nvml_pynvml = pynvml
            _nvml_handle = handle
            return _nvml_pynvml, _nvml_handle
        except Exception:
            _nvml_pynvml = None
            _nvml_handle = None
            return None, None


# Defer NVML binding to first use: import-time torch device may not match server worker yet.


def _read_nvml_power_w(pynvml_mod, handle) -> Optional[float]:
    try:
        return float(pynvml_mod.nvmlDeviceGetPowerUsage(handle)) / 1000.0
    except Exception:
        return None


def _read_nvml_total_energy_mj(pynvml_mod, handle) -> Optional[float]:
    try:
        return float(pynvml_mod.nvmlDeviceGetTotalEnergyConsumption(handle))
    except Exception:
        return None


def _debug_energy_enabled() -> bool:
    return os.getenv("SGLANG_DEBUG_ENERGY", "0") == "1"


def _bench_energy_enabled() -> bool:
    # Default on when unset (e.g. ad-hoc runs); batch script sets explicitly from config.
    return os.getenv("SGLANG_BENCH_ENERGY", "1") == "1"


_MIN_MEASURE_TIME_S = 3


def _profile_one_like_prefill(fn, n_warmup: int, n_repeat: int) -> Tuple[float, Optional[float], int]:
    """Same timing style as bench_prefill_af.profile_one().

    Probes single-call latency after warmup, then ensures the measurement
    window is at least _MIN_MEASURE_TIME_S so that NVML energy counters
    (updated every ~20-100 ms) produce non-zero readings.

    When TP > 1, actual_repeat is broadcast from rank 0 so all workers
    call fn() the same number of times (required by NCCL collectives
    inside fn).

    Returns (avg_latency_us, energy_per_op_uj, actual_repeat).
    """
    for _ in range(n_warmup):
        fn()

    torch.cuda.synchronize()
    t_probe = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    t_single = time.perf_counter() - t_probe

    actual_repeat = max(n_repeat, int(_MIN_MEASURE_TIME_S / max(t_single, 1e-6)) + 1)

    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        ar = torch.tensor([actual_repeat], dtype=torch.long, device="cuda")
        torch.distributed.broadcast(ar, src=0)
        actual_repeat = int(ar.item())

    measure_energy = _bench_energy_enabled()
    pynvml_mod, nvml_handle = (None, None)
    if measure_energy:
        pynvml_mod, nvml_handle = _try_init_nvml_for_current_device()

    tp_size = (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1
    )

    start_energy_mj = None
    if measure_energy and pynvml_mod is not None and nvml_handle is not None:
        start_energy_mj = _read_nvml_total_energy_mj(pynvml_mod, nvml_handle)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(actual_repeat):
        fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    end_energy_mj = None
    if measure_energy and pynvml_mod is not None and nvml_handle is not None:
        end_energy_mj = _read_nvml_total_energy_mj(pynvml_mod, nvml_handle)

    elapsed_s = t1 - t0
    avg_latency_us = elapsed_s / actual_repeat * 1e6

    energy_per_op_uj: Optional[float] = None
    if measure_energy and start_energy_mj is not None and end_energy_mj is not None:
        counter_delta_mj = float(end_energy_mj) - float(start_energy_mj)
        if counter_delta_mj > 0:
            local_e_uj = counter_delta_mj / actual_repeat * 1000.0
            if tp_size > 1:
                e_tensor = torch.tensor([local_e_uj], dtype=torch.double, device="cuda")
                torch.distributed.all_reduce(e_tensor, op=torch.distributed.ReduceOp.SUM)
                energy_per_op_uj = float(e_tensor.item())
            else:
                energy_per_op_uj = local_e_uj

    if _debug_energy_enabled():
        print(
            "[sglang-energy] "
            f"elapsed_s={elapsed_s:.6f} actual_repeat={actual_repeat} "
            f"start_mj={start_energy_mj} end_mj={end_energy_mj} "
            f"energy_per_op_uj={energy_per_op_uj} tp_size={tp_size}"
        )

    return avg_latency_us, energy_per_op_uj, actual_repeat


def _debug_a_input_enabled() -> bool:
    return os.getenv("SGLANG_DEBUG_A_INPUT", "0") == "1"


def _latency_csv_enabled() -> bool:
    return os.getenv("SGLANG_TTFT_AF_CSV_ENABLE", "0") == "1"


def _latency_csv_path() -> str:
    p = os.getenv("SGLANG_TTFT_AF_CSV_PATH", "").strip()
    if p:
        return p
    return "/tmp/sglang_ttft_af_latency.csv"


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, str(default))).strip())
    except Exception:
        return default


def _bench_phase() -> str:
    """Runtime phase: 'ttft', 'af', 'both', or 'none'.
    Reads the JSON/plain-text phase file (also refreshes per-job config)."""
    _refresh_bench_job_config()
    if _bench_job_cfg_phase in {"ttft", "af", "both", "none"}:
        return _bench_job_cfg_phase
    stage_on = os.getenv("SGLANG_SYNC_STAGE_BENCH", "0") == "1"
    internal_on = os.getenv("SGLANG_SYNC_INTERNAL_OP_BENCH", "0") == "1"
    if stage_on and internal_on:
        return "both"
    if stage_on:
        return "ttft"
    if internal_on:
        return "af"
    return "none"


def _ttft_window_match(input_ids: Optional[torch.Tensor]) -> bool:
    expected_input_len = _bench_cfg_int("input_len", 0)
    expected_batch_size = _bench_cfg_int("batch_size", 1)
    if expected_input_len <= 0:
        return True
    expected_tokens = expected_input_len * max(expected_batch_size, 1)
    if input_ids is None:
        return False
    return int(input_ids.numel()) == expected_tokens


def _af_window_match(forward_batch: Optional[ForwardBatch]) -> bool:
    expected_input_len = _bench_cfg_int("input_len", 0)
    expected_batch_size = _bench_cfg_int("batch_size", 1)
    if expected_input_len <= 0:
        return True
    expected_tokens = expected_input_len * max(expected_batch_size, 1)
    if forward_batch is None:
        return False
    try:
        actual_tokens = int(forward_batch.seq_lens_sum)
    except Exception:
        return False
    return actual_tokens == expected_tokens


def _record_latency(op: str, latency_us: float, energy_uj: Optional[float] = None) -> None:
    if op not in _latency_values:
        return
    with _latency_lock:
        _latency_values[op].append(float(latency_us))
        if (
            op in _energy_values_uj
            and energy_uj is not None
            and isinstance(energy_uj, (int, float))
        ):
            _energy_values_uj[op].append(float(energy_uj))
    # Flush incrementally so results are persisted even if process is terminated.
    _dump_latency_csv()


def _dump_latency_csv() -> None:
    if not _latency_csv_enabled():
        return
    cfg_csv = _bench_cfg_str("csv_path", "").strip()
    path = cfg_csv if cfg_csv else _latency_csv_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with _latency_lock:
            ttft_arr = list(_latency_values["TTFT"])
            a_arr = list(_latency_values["A"])
            f_arr = list(_latency_values["F"])
            a_energy_arr = list(_energy_values_uj["A"])
            f_energy_arr = list(_energy_values_uj["F"])
        base_meta = {
            "tp": _bench_cfg_str("tp"),
            "input_len": _bench_cfg_str("input_len"),
            "sm_clock": _bench_cfg_str("sm_clock"),
            "batch_size": _bench_cfg_str("batch_size"),
        }
        row = {
            **base_meta,
            "ttft_avg_us": (sum(ttft_arr) / len(ttft_arr) if ttft_arr else ""),
            "a_avg_us": (sum(a_arr) / len(a_arr) if a_arr else ""),
            "f_avg_us": (sum(f_arr) / len(f_arr) if f_arr else ""),
            "a_avg_energy_uj": (sum(a_energy_arr) / len(a_energy_arr) if a_energy_arr else ""),
            "f_avg_energy_uj": (sum(f_energy_arr) / len(f_energy_arr) if f_energy_arr else ""),
            "ttft_count": len(ttft_arr),
            "a_count": len(a_arr),
            "f_count": len(f_arr),
            "a_energy_count": len(a_energy_arr),
            "f_energy_count": len(f_energy_arr),
        }
        tmp = path + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            fieldnames = list(row.keys())
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerow(row)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

        samples_path = (
            path[:-4] + "_samples.csv" if path.lower().endswith(".csv") else path + "_samples.csv"
        )
        samples_tmp = samples_path + ".tmp"
        with open(samples_tmp, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "tp",
                "input_len",
                "sm_clock",
                "batch_size",
                "op",
                "sample_idx",
                "latency_us",
                "energy_uj",
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for op_name, arr, earr in (
                ("TTFT", ttft_arr, []),
                ("A", a_arr, a_energy_arr),
                ("F", f_arr, f_energy_arr),
            ):
                for idx, val in enumerate(arr, start=1):
                    e_val = earr[idx - 1] if idx - 1 < len(earr) else ""
                    w.writerow(
                        {
                            **base_meta,
                            "op": op_name,
                            "sample_idx": idx,
                            "latency_us": val,
                            "energy_uj": e_val,
                        }
                    )
            f.flush()
            os.fsync(f.fileno())
        os.replace(samples_tmp, samples_path)
    except Exception:
        pass


@atexit.register
def _dump_latency_csv_on_exit() -> None:
    _dump_latency_csv()


def _tensor_brief(x: Optional[torch.Tensor]) -> str:
    if x is None:
        return "None"
    shape = tuple(x.shape)
    dtype = str(x.dtype)
    device = str(x.device)
    if x.numel() == 0:
        return f"shape={shape} dtype={dtype} device={device} empty"
    x32 = x.detach().to(torch.float32)
    mean = float(x32.mean().item())
    std = float(x32.std().item())
    return (
        f"shape={shape} dtype={dtype} device={device} "
        f"mean={mean:.6f} std={std:.6f}"
    )


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 1000000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        head_dim: Optional[int] = None,
        max_position_embeddings: int = 32768,
        quant_config: Optional[QuantizationConfig] = None,
        rms_norm_eps: float = None,
        attention_bias: bool = False,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.tp_rank = get_tensor_model_parallel_rank()

        norm_kwargs = (
            dict(
                weight_dtype=torch.float32,
                cast_x_before_out_mul=True,
            )
            if get_global_server_args().rl_on_policy_target is not None
            else {}
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, **norm_kwargs)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, **norm_kwargs)

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )
        self.alt_stream = alt_stream

    def forward_prepare_native(
        self, positions, hidden_states, forward_batch: ForwardBatch
    ):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = apply_qk_norm(
            q=q,
            k=k,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            head_dim=self.head_dim,
            alt_stream=self.alt_stream,
        )
        q, k = self.rotary_emb(positions, q, k)
        return q, k, v

    def forward_prepare_npu(self, positions, hidden_states, forward_batch):
        qkv, _ = self.qkv_proj(hidden_states)

        if self.attn.layer_id == forward_batch.token_to_kv_pool.start_layer:
            self.rotary_emb.get_cos_sin_with_position(positions)

        q, k, v = split_qkv_rmsnorm_rope(
            qkv,
            self.rotary_emb.position_sin,
            self.rotary_emb.position_cos,
            self.q_size,
            self.kv_size,
            self.head_dim,
            eps=self.q_norm.variance_epsilon,
            q_weight=self.q_norm.weight,
            k_weight=self.k_norm.weight,
            q_bias=getattr(self.q_norm, "bias", None),
            k_bias=getattr(self.k_norm, "bias", None),
        )
        return q, k, v

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # print(f"positions: {positions.shape}")
        # print(f"hidden_states: {hidden_states.shape}")
        if get_global_server_args().rl_on_policy_target is not None:
            hidden_states = hidden_states.bfloat16()

        if not _is_npu or forward_batch.forward_mode.is_extend():
            q, k, v = self.forward_prepare_native(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
        else:
            q, k, v = self.forward_prepare_npu(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        if get_global_server_args().rl_on_policy_target is not None:
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)

        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
        head_dim = getattr(config, "head_dim", None)
        self.self_attn = Qwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
            prefix=add_prefix("self_attn", prefix),
            alt_stream=alt_stream,
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

        norm_kwargs = (
            dict(
                weight_dtype=torch.float32,
                cast_x_before_out_mul=True,
                override_orig_dtype=torch.float32,
                fp32_residual=True,
            )
            if get_global_server_args().rl_on_policy_target is not None
            else {}
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, **norm_kwargs
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, **norm_kwargs
        )

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=False,
            is_previous_layer_sparse=False,
            is_next_layer_sparse=False,
        )
        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
        )
        self.layer_id = layer_id
        self._debug_a_input_printed = False

    with torch.no_grad():
        def _run_a_block(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            forward_batch: ForwardBatch,
            residual: Optional[torch.Tensor],
            post_residual_addition: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            hidden_states, residual = self.layer_communicator.prepare_attn(
                hidden_states,
                residual,
                forward_batch,
                post_residual_addition=post_residual_addition,
            )
            if hidden_states.shape[0] != 0:
                hidden_states = self.self_attn(
                    positions=positions,
                    hidden_states=hidden_states,
                    forward_batch=forward_batch,
                )
            return hidden_states, residual
            # hs = self.input_layernorm(
            #     hidden_states, residual)

            # hidden_states = self.self_attn(
            #     positions=positions,
            #     hidden_states=hs,
            #     forward_batch=forward_batch)

            # return hidden_states, residual
                

    with torch.no_grad():
        def _run_f_block(
            self,
            hidden_states: torch.Tensor,
            forward_batch: ForwardBatch,
            residual: Optional[torch.Tensor],
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            hidden_states, residual = self.layer_communicator.prepare_mlp(
                hidden_states,
                residual,
                forward_batch,
                cache=(
                    [self.mlp.gate_up_proj.weight, self.mlp.down_proj.weight]
                    if _is_npu
                    and not get_global_server_args().enable_piecewise_cuda_graph
                    and (
                        hasattr(self.mlp.gate_up_proj, "weight")
                        and hasattr(self.mlp.down_proj, "weight")
                    )
                    else None
                ),
            )
            # hs = self.post_attention_layernorm(hidden_states, residual)
            hidden_states = self.mlp(hidden_states, forward_batch=forward_batch)
            return hidden_states, residual

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _phase = _bench_phase()
        _ttft_mode = _phase == "ttft"
        _internal_bench = (
            (os.getenv("SGLANG_SYNC_INTERNAL_OP_BENCH", "0") == "1")
            and (not _ttft_mode)
            and (_phase in {"af", "both"})
            and forward_batch.forward_mode.is_extend()
            and _af_window_match(forward_batch)
        )
        if _internal_bench and self.layer_id > 0:
            # AF-only benchmark mode: only run layer0 and skip following layers.
            return hidden_states, residual
        # Only benchmark layer 0; this value is already averaged by repeat runs.
        _use_profile_wrap = _internal_bench and self.layer_id == 0
        if _debug_a_input_enabled() and (not self._debug_a_input_printed):
            try:
                fwd_mode = str(forward_batch.forward_mode)
            except Exception:
                fwd_mode = "unknown"
            print(
                f"[qwen3][A-input] layer={self.layer_id} forward_mode={fwd_mode} "
                f"positions={_tensor_brief(positions)} "
                f"hidden_states={_tensor_brief(hidden_states)} "
                f"residual={_tensor_brief(residual)} "
                f"post_residual_addition={_tensor_brief(post_residual_addition)}"
            )
            self._debug_a_input_printed = True

        # A block = input_layernorm + self_attn
        if _use_profile_wrap:
            n_warmup = _get_af_profile_warmup()
            n_repeat = _get_af_profile_repeat()

            # n_tokens = forward_batch.seq_lens_sum
            # hidden_states = torch.randn(n_tokens, self.hidden_size,
            #                             device='cuda', dtype=torch.bfloat16)
            # residual = hidden_states.clone()
            # positions = forward_batch.positions

            def _a_once():
                self._run_a_block(
                    positions=positions,
                    hidden_states=hidden_states,
                    forward_batch=forward_batch,
                    residual=residual,
                    post_residual_addition=post_residual_addition,
                )

            a_lat_us, a_energy_uj, a_actual_repeat = _profile_one_like_prefill(_a_once, n_warmup, n_repeat)
            e_str = f"{a_energy_uj:.2f}" if a_energy_uj is not None else "NA"
            print(
                f"[sync-op-bench] A_l{self.layer_id} avg_us={a_lat_us:.2f} "
                f"energy_uj={e_str} (warmup={n_warmup}, repeat={a_actual_repeat})"
            )
            _record_latency("A", a_lat_us, a_energy_uj)
            hidden_states, residual = self._run_a_block(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                residual=residual,
                post_residual_addition=post_residual_addition,
            )
        else:
            hidden_states, residual = self._run_a_block(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                residual=residual,
                post_residual_addition=post_residual_addition,
            )

        # F block = post_attention_layernorm + mlp
        if _use_profile_wrap:
            n_warmup = _get_af_profile_warmup()
            n_repeat = _get_af_profile_repeat()

            # n_tokens = forward_batch.seq_lens_sum
            # hidden_states = torch.randn(n_tokens, self.hidden_size,
            #                             device='cuda', dtype=torch.bfloat16)
            # residual = hidden_states.clone()
            # positions = forward_batch.positions

            def _f_once():
                self._run_f_block(
                    hidden_states=hidden_states,
                    forward_batch=forward_batch,
                    residual=residual,
                )

            f_lat_us, f_energy_uj, f_actual_repeat = _profile_one_like_prefill(_f_once, n_warmup, n_repeat)
            e_str = f"{f_energy_uj:.2f}" if f_energy_uj is not None else "NA"
            print(
                f"[sync-op-bench] F_l{self.layer_id} avg_us={f_lat_us:.2f} "
                f"energy_uj={e_str} (warmup={n_warmup}, repeat={f_actual_repeat})"
            )
            _record_latency("F", f_lat_us, f_energy_uj)
            hidden_states, residual = self._run_f_block(
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                residual=residual,
            )
        else:
            hidden_states, residual = self._run_f_block(
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                residual=residual,
            )

        if _is_npu and get_cmo_stream():
            wait_cmo_stream()
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )
        return hidden_states, residual


class Qwen3Model(Qwen2Model):
    def __init__(
        self,
        config: Qwen3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        alt_stream = torch.cuda.Stream() if _is_cuda else None
        super().__init__(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            decoder_layer_type=Qwen3DecoderLayer,
            alt_stream=alt_stream,
        )


class Qwen3ForCausalLM(nn.Module):
    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = Qwen3Model(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        # handle the lm head on different pp ranks
        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            # ranks other than the last rank will have a placeholder layer
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

        # For EAGLE3 support
        self.capture_aux_hidden_states = False

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        input_ids_shape = tuple(input_ids.shape) if input_ids is not None else None
        positions_shape = tuple(positions.shape) if positions is not None else None
        input_embeds_shape = (
            tuple(input_embeds.shape) if input_embeds is not None else None
        )
        logger.info(
            "[model_input] input_ids.shape=%s positions.shape=%s input_embeds.shape=%s",
            input_ids_shape,
            positions_shape,
            input_embeds_shape,
        )

        collect_ttft = (
            (_bench_phase() in {"ttft", "both"})
            and (os.getenv("SGLANG_SYNC_STAGE_BENCH", "0") == "1")
            and forward_batch.forward_mode.is_extend()
            and _ttft_window_match(input_ids)
        )
        if collect_ttft:
            torch.cuda.synchronize()
            _t0 = time.perf_counter()

        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            if not get_embedding:
                out = self.logits_processor(
                    input_ids,
                    hidden_states,
                    self.lm_head,
                    forward_batch,
                    aux_hidden_states,
                )
            else:
                out = self.pooler(hidden_states, forward_batch)
        else:
            out = hidden_states

        if collect_ttft:
            torch.cuda.synchronize()
            _ttft_us = (time.perf_counter() - _t0) * 1e6
            _record_latency("TTFT", _ttft_us)

        return out

    @torch.no_grad()
    def forward_split_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        split_interval: Tuple[int, int],  # [start, end) 0-based
        input_embeds: torch.Tensor = None,
    ):
        start, end = split_interval
        # embed
        if start == 0:
            if input_embeds is None:
                forward_batch.hidden_states = self.model.embed_tokens(input_ids)
            else:
                forward_batch.hidden_states = input_embeds
        # decoder layer
        for i in range(start, end):
            layer = self.model.layers[i]
            forward_batch.hidden_states, forward_batch.residual = layer(
                positions,
                forward_batch.hidden_states,
                forward_batch,
                forward_batch.residual,
            )

        if end == self.model.config.num_hidden_layers:
            # norm
            hidden_states, _ = self.model.norm(
                forward_batch.hidden_states, forward_batch.residual
            )
            forward_batch.hidden_states = hidden_states
            # logits process
            result = self.logits_processor(
                input_ids, forward_batch.hidden_states, self.lm_head, forward_batch
            )
        else:
            result = None

        return result

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if not name.startswith("model.") and (
                name.startswith("layers.")
                or name.startswith("embed_tokens.")
                or name.startswith("norm.")
            ):
                name = add_prefix(name, "model")

            if name == "model.embed_tokens.weight":
                if self.pp_group.is_last_rank and self.config.tie_word_embeddings:
                    if "lm_head.weight" in params_dict:
                        param = params_dict["lm_head.weight"]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)

            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                else:
                    logger.warning(f"Parameter {name} not found in params_dict")

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if not self.pp_group.is_last_rank:
            return

        self.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [
                2,
                num_layers // 2,
                num_layers - 3,
            ]  # Specific layers for EAGLE3 support
        else:
            self.model.layers_to_capture = [val + 1 for val in layer_ids]


EntryClass = Qwen3ForCausalLM
