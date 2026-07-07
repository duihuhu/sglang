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
    """Child entry: fresh CUDA context, disk load at (tp_rank=0, tp_size=new_tp)."""
    try:
        cfg: Dict[str, Any] = pickle.loads(config_bytes)
        gpu_id = int(cfg["gpu_id"])
        new_tp = int(cfg["new_tp"])
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # Child is a standalone loader (world_size=1); inherited in-place reshard
        # env from the parent server breaks initialize_model_parallel assertions.
        for key in (
            "SGLANG_INPLACE_RESHARD_MAX_TP",
            "SGLANG_INPLACE_RESHARD_ACTIVE_TP",
        ):
            os.environ.pop(key, None)

        import torch

        torch.cuda.set_device(0)

        port = int(cfg["dist_port"])
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        os.environ.setdefault("LOCAL_RANK", "0")

        from sglang.srt.distributed.parallel_state import (
            init_distributed_environment,
            initialize_model_parallel,
        )

        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            local_rank=0,
            backend="gloo",
        )
        initialize_model_parallel(tensor_model_parallel_size=new_tp)

        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.model_executor.model_runner import DeviceConfig
        from sglang.srt.model_loader import get_model_loader
        from sglang.srt.model_loader.loader import LoadConfig, LoadFormat

        model_config = ModelConfig(
            model_path=cfg["model_path"],
            trust_remote_code=bool(cfg.get("trust_remote_code", True)),
            revision=cfg.get("revision"),
            context_length=cfg.get("context_length"),
            dtype=cfg.get("dtype", "auto"),
            quantization=cfg.get("quantization"),
        )
        load_format = cfg.get("load_format", LoadFormat.AUTO)
        if load_format == LoadFormat.DUMMY:
            load_format = LoadFormat.AUTO
        load_config = LoadConfig(
            load_format=load_format,
            download_dir=cfg.get("download_dir"),
            tp_rank=0,
        )
        loader = get_model_loader(load_config=load_config, model_config=model_config)
        model = loader.load_model(
            model_config=model_config,
            device_config=DeviceConfig("cuda", 0),
        )

        from sglang.srt.utils import MultiprocessingSerializer

        exported = {
            name: MultiprocessingSerializer.serialize(param.data.detach())
            for name, param in model.named_parameters()
        }
        del model
        torch.cuda.synchronize()

        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()

        conn.send(("ok", exported))
    except Exception as exc:
        logger.exception("rank0 subprocess weight load failed")
        conn.send(("error", repr(exc)))
    finally:
        conn.close()


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
    stale_model = runner.model
    runner.model = None
    if hasattr(runner, "loader"):
        runner.loader = None
    if stale_model is not None:
        for param in stale_model.parameters():
            if param.data.is_cuda and param.data.numel() > 0:
                param.data = torch.empty(0, device=param.device, dtype=param.dtype)
    del stale_model
    for _ in range(5):
        _gc.collect()
        torch.cuda.synchronize()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

    runner.tp_size = new_tp
    runner.server_args.tp_size = new_tp
    original = runner.server_args.load_format
    runner.server_args.load_format = LoadFormat.DUMMY
    runner._skip_load_model_barrier_once = True
    try:
        runner.load_model()
    finally:
        runner.server_args.load_format = original

    local_device = torch.device(runner.device, runner.gpu_id)
    for name, param in runner.model.named_parameters():
        if name not in exported:
            raise RuntimeError(f"subprocess reload missing parameter {name}")
        tensor = MultiprocessingSerializer.deserialize(exported[name])
        param.data = tensor.to(local_device, non_blocking=False).contiguous()
        del tensor
    exported.clear()
    _gc.collect()
    torch.cuda.synchronize()
    if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()

    runner._update_model_tp_metadata(new_tp)
    runner._finalize_inplace_reshard_weight_storage()
