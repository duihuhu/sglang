"""Load rank0 TP weights in an isolated process for in-place reshard memory parity.

The parent scheduler process keeps a long-lived CUDA context whose caching
allocator retains peak buffers from earlier TP degrees.  Loading weights in a
short-lived child process and importing the final shards via CUDA IPC lets the
parent adopt a compact layout closer to cold ``sglang serve --tp N``.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import pickle
import socket
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _rank0_subprocess_load_worker(conn, config_bytes: bytes) -> None:
    """Child entry: fresh CUDA context, disk load at (tp_rank=0, tp_size=new_tp).

    Loads raw safetensor weights, fuses q/k/v→qkv_proj and gate/up→gate_up_proj
    to match sglang's internal parameter names, then TP-shards for rank0.
    """
    try:
        cfg: Dict[str, Any] = pickle.loads(config_bytes)
        gpu_id = int(cfg["gpu_id"])
        new_tp = int(cfg["new_tp"])
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        for key in (
            "SGLANG_INPLACE_RESHARD_MAX_TP",
            "SGLANG_INPLACE_RESHARD_ACTIVE_TP",
        ):
            os.environ.pop(key, None)

        import torch

        torch.cuda.set_device(0)

        from safetensors import safe_open
        from pathlib import Path

        model_path = Path(cfg["model_path"])
        shard_files = sorted(model_path.glob("*.safetensors"))
        if not shard_files:
            raise FileNotFoundError(f"No safetensors in {model_path}")

        from sglang.srt.layers.reshard_weights import (
            reshard_shard_for_rank,
        )

        fused_rules = _build_fused_tp_rules(model_path)

        raw: Dict[str, torch.Tensor] = {}
        for sf in shard_files:
            with safe_open(str(sf), framework="pt", device="cpu") as f:
                for name in f.keys():
                    raw[name] = f.get_tensor(name)

        fused = _fuse_safetensor_weights(raw)
        del raw

        exported = {}
        for name, tensor in fused.items():
            rule = fused_rules.get(name)
            if rule is not None and new_tp > 1:
                tensor = reshard_shard_for_rank(tensor, rule, 0, new_tp)
            exported[name] = tensor.contiguous()
            del tensor
        del fused

        conn.send(("ok", exported))
    except Exception as exc:
        logger.exception("rank0 subprocess weight load failed")
        conn.send(("error", repr(exc)))
    finally:
        conn.close()


def _build_fused_tp_rules(model_path) -> Dict[str, Any]:
    """Build TP split rules keyed by fused sglang parameter names."""
    import json
    from pathlib import Path

    p = Path(model_path)
    config_file = p / "config.json"
    config = {}
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)

    num_heads = config.get("num_attention_heads", 32)
    num_kv_heads = config.get("num_key_value_heads", num_heads)
    hidden_size = config.get("hidden_size", 4096)
    head_dim = config.get("head_dim", hidden_size // num_heads)
    intermediate_size = config.get("intermediate_size", hidden_size * 4)

    q_size = num_heads * head_dim
    k_size = num_kv_heads * head_dim
    v_size = num_kv_heads * head_dim

    rules: Dict[str, Any] = {}
    num_layers = config.get("num_hidden_layers", 64)
    for i in range(num_layers):
        prefix = f"model.layers.{i}"
        rules[f"{prefix}.self_attn.qkv_proj.weight"] = (
            "column_fused", 0, (q_size, k_size, v_size)
        )
        rules[f"{prefix}.mlp.gate_up_proj.weight"] = (
            "column_fused", 0, (intermediate_size, intermediate_size)
        )
        rules[f"{prefix}.self_attn.o_proj.weight"] = ("row", 1)
        rules[f"{prefix}.mlp.down_proj.weight"] = ("row", 1)
    rules["model.embed_tokens.weight"] = ("column", 0)
    rules["lm_head.weight"] = ("column", 0)
    return rules


def _fuse_safetensor_weights(raw: Dict[str, "torch.Tensor"]) -> Dict[str, "torch.Tensor"]:
    """Convert HF safetensor param names to sglang fused names.

    Fusions:
    - q_proj + k_proj + v_proj → qkv_proj (cat dim=0)
    - gate_proj + up_proj → gate_up_proj (cat dim=0)
    """
    import re
    import torch

    fused: Dict[str, torch.Tensor] = {}
    consumed = set()

    layer_pattern = re.compile(
        r"(model\.layers\.\d+\.self_attn)\.(q_proj|k_proj|v_proj)(\.weight|\.bias)"
    )
    mlp_pattern = re.compile(
        r"(model\.layers\.\d+\.mlp)\.(gate_proj|up_proj)(\.weight|\.bias)"
    )

    layer_groups: Dict[str, Dict[str, "torch.Tensor"]] = {}
    mlp_groups: Dict[str, Dict[str, "torch.Tensor"]] = {}

    for name, tensor in raw.items():
        m = layer_pattern.match(name)
        if m:
            prefix, proj, suffix = m.group(1), m.group(2), m.group(3)
            key = prefix + suffix
            layer_groups.setdefault(key, {})[proj] = tensor
            consumed.add(name)
            continue
        m = mlp_pattern.match(name)
        if m:
            prefix, proj, suffix = m.group(1), m.group(2), m.group(3)
            key = prefix + suffix
            mlp_groups.setdefault(key, {})[proj] = tensor
            consumed.add(name)
            continue

    for key, parts in layer_groups.items():
        if "q_proj" in parts and "k_proj" in parts and "v_proj" in parts:
            fused_name = key.replace(".weight", ".qkv_proj.weight").replace(
                ".bias", ".qkv_proj.bias"
            )
            suffix = ".weight" if key.endswith(".weight") else ".bias"
            prefix = key[: -len(suffix)]
            fused_name = prefix + ".qkv_proj" + suffix
            fused[fused_name] = torch.cat(
                [parts["q_proj"], parts["k_proj"], parts["v_proj"]], dim=0
            )
        else:
            for proj, t in parts.items():
                orig = key.replace(".weight", f".{proj}.weight").replace(
                    ".bias", f".{proj}.bias"
                )
                suffix = ".weight" if key.endswith(".weight") else ".bias"
                prefix = key[: -len(suffix)]
                fused[prefix + f".{proj}" + suffix] = t

    for key, parts in mlp_groups.items():
        if "gate_proj" in parts and "up_proj" in parts:
            suffix = ".weight" if key.endswith(".weight") else ".bias"
            prefix = key[: -len(suffix)]
            fused_name = prefix + ".gate_up_proj" + suffix
            fused[fused_name] = torch.cat(
                [parts["gate_proj"], parts["up_proj"]], dim=0
            )
        else:
            for proj, t in parts.items():
                suffix = ".weight" if key.endswith(".weight") else ".bias"
                prefix = key[: -len(suffix)]
                fused[prefix + f".{proj}" + suffix] = t

    for name, tensor in raw.items():
        if name not in consumed:
            fused[name] = tensor

    return fused


def load_rank0_weights_via_subprocess(config: Dict[str, Any], *, timeout_s: float = 600.0):
    """Spawn an isolated loader and return serialized parameter tensors."""
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_rank0_subprocess_load_worker,
        args=(child_conn, pickle.dumps(config)),
        name="inplace-reshard-rank0-loader",
        daemon=True,
    )
    proc.start()
    child_conn.close()
    if not parent_conn.poll(timeout_s):
        proc.terminate()
        proc.join(timeout=5)
        raise TimeoutError(
            f"rank0 subprocess weight load timed out after {timeout_s:.0f}s"
        )
    status, payload = parent_conn.recv()
    proc.join(timeout=30)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
    if status != "ok":
        raise RuntimeError(f"rank0 subprocess weight load failed: {payload}")
    return payload


def build_subprocess_reload_config(runner, new_tp: int) -> Dict[str, Any]:
    from sglang.srt.model_loader.loader import LoadFormat

    original = runner.server_args.load_format
    load_fmt = original if original != LoadFormat.DUMMY else LoadFormat.AUTO
    return {
        "gpu_id": int(runner.gpu_id),
        "new_tp": int(new_tp),
        "dist_port": _pick_free_port(),
        "model_path": runner.server_args.model_path,
        "trust_remote_code": bool(runner.server_args.trust_remote_code),
        "revision": runner.server_args.revision,
        "context_length": runner.server_args.context_length,
        "dtype": runner.server_args.dtype,
        "quantization": runner.server_args.quantization,
        "load_format": load_fmt,
        "download_dir": runner.server_args.download_dir,
    }


def import_subprocess_weights(
    runner,
    exported: Dict[str, Any],
    new_tp: int,
) -> None:
    """Replace rank0 model weights with tensors imported from the child loader."""
    import gc as _gc

    import torch

    from sglang.srt.model_loader.loader import LoadConfig, LoadFormat
    from sglang.srt.utils import MultiprocessingSerializer

    hooks = getattr(runner, "pyt_hooks", None)
    if hooks is not None:
        runner.pyt_hooks = None
        del hooks

    # Instead of deleting the model and re-creating it (which causes fragmentation),
    # directly overwrite existing parameter data in-place from CPU tensors.
    # This reuses the SAME GPU memory addresses and avoids allocator churn.
    import gc as _gc

    local_device = torch.device(runner.device, runner.gpu_id)

    missing = []
    loaded = 0
    for name, param in runner.model.named_parameters():
        if name not in exported:
            missing.append(name)
            continue
        cpu_tensor = exported[name]
        # Reshape param storage to match exported tensor if sizes differ
        if param.data.shape != cpu_tensor.shape:
            param.data = torch.empty(
                cpu_tensor.shape, dtype=cpu_tensor.dtype, device=local_device
            )
        param.data.copy_(cpu_tensor)
        loaded += 1
        del cpu_tensor
    if missing:
        logger.warning(
            "subprocess reload: %d params not in export (kept as-is): %s",
            len(missing),
            missing[:5],
        )
    exported.clear()
    _gc.collect()
    torch.cuda.synchronize()

    runner.tp_size = new_tp
    runner.server_args.tp_size = new_tp
    runner._update_model_tp_metadata(new_tp)
    runner._finalize_inplace_reshard_weight_storage()
