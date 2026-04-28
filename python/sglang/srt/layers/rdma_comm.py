# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0

"""UCX-Py RDMA tensor communicator for AFD Attn-FFN communication.

Replaces StepMesh (fserver_lib) with native RDMA support via UCX-Py.
Automatically selects the best available transport:
  - IB RDMA + GPU-direct (when nvidia_peermem loaded)
  - CUDA IPC / NVLink (same-node GPU pairs)
  - TCP fallback (always available)

Supports:
  - 1:1 homogeneous TP (K=1, attn_tp == ffn_tp == 1)
  - N:M heterogeneous TP via NIC-aware grouping:
      K NIC representatives do RDMA (each sends 1/K of data),
      then NVLink all_gather/broadcast to distribute to all ranks.
  - Total cross-node RDMA traffic = 2NH per layer (independent of K).
"""

import asyncio
import logging
import os
import threading
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

# Must be set before UCX C library initializes (before any `import ucp`)
os.environ.setdefault("UCX_LOG_LEVEL", "fatal")
os.environ.setdefault("UCX_WARN_UNUSED_ENV_VARS", "n")
os.environ.setdefault("UCX_MEMTYPE_CACHE", "n")

import numpy as np
import torch
import torch.distributed as dist

from abc import ABC, abstractmethod

try:
    from sglang.srt.layers.afd_type import AFDPerspective
except ImportError:
    from enum import Enum

    class AFDPerspective(Enum):
        AFD_PERSPECTIVE_ATTN = "attn"
        AFD_PERSPECTIVE_FFN = "ffn"


class _FifoTensorCommunicatorBase(ABC):
    """Lightweight base matching afd.FifoTensorCommunicator interface."""

    @abstractmethod
    def recv_tensor(self) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def send_tensor(self, x: torch.Tensor):
        raise NotImplementedError

logger = logging.getLogger(__name__)

_EAGER_INIT = False


# ---- Dtype mapping ----

_DTYPE_TO_INT = {
    torch.float16: 0, torch.bfloat16: 1, torch.float32: 2,
    torch.float64: 3, torch.int32: 4, torch.int64: 5,
}
_INT_TO_DTYPE = {v: k for k, v in _DTYPE_TO_INT.items()}
_META_SLOTS = 8


def _encode_meta(tensor: torch.Tensor) -> np.ndarray:
    """Encode tensor shape and dtype into a fixed-size int64 array."""
    meta = np.zeros(_META_SLOTS, dtype=np.int64)
    ndim = tensor.ndim
    if ndim > _META_SLOTS - 2:
        raise ValueError(f"Tensor ndim {ndim} exceeds metadata capacity {_META_SLOTS - 2}")
    meta[0] = ndim
    for i, s in enumerate(tensor.shape):
        meta[i + 1] = s
    dtype_code = _DTYPE_TO_INT.get(tensor.dtype)
    if dtype_code is None:
        raise ValueError(
            f"Unsupported dtype {tensor.dtype} for UCX tensor transfer. "
            f"Supported: {list(_DTYPE_TO_INT.keys())}"
        )
    meta[ndim + 1] = dtype_code
    return meta


def _decode_meta(meta: np.ndarray) -> Tuple[tuple, torch.dtype]:
    ndim = int(meta[0])
    shape = tuple(int(meta[i + 1]) for i in range(ndim))
    dtype_code = int(meta[ndim + 1])
    dtype = _INT_TO_DTYPE.get(dtype_code)
    if dtype is None:
        raise ValueError(
            f"Unknown dtype code {dtype_code} in UCX tensor metadata. "
            f"Known codes: {list(_INT_TO_DTYPE.keys())}"
        )
    return shape, dtype


# ---- Async-to-sync bridge ----


class _AsyncBridge:
    """Persistent SelectorEventLoop in a background thread.

    Uses SelectorEventLoop (not uvloop) to avoid conflicts with
    UCX-Py's BlockingMode progress callbacks under uvloop.
    """

    def __init__(self):
        self._ready = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="ucx-event-loop"
        )
        self._thread.start()
        if not self._ready.wait(timeout=10):
            raise RuntimeError("UCX event loop thread failed to start")

    def _run_loop(self):
        self._loop = asyncio.SelectorEventLoop()
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    def run(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def stop(self):
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)


# ---- Buffer pool ----


class _BufferPool:
    """Reusable GPU tensor pool with optional warmup pre-allocation.

    After warmup, get/put operate on pre-allocated buffers with zero
    torch.empty() calls on the hot path. Without warmup, falls back
    to lazy allocation (backward compatible).
    """

    def __init__(self, max_free: int = 30):
        self._max_free = max_free
        self._pools: Dict[Tuple[tuple, torch.dtype], Deque[torch.Tensor]] = {}

    def warmup(self, shape: tuple, dtype: torch.dtype, device: str, count: int):
        """Pre-allocate count buffers for the given shape/dtype."""
        key = (shape, dtype)
        pool = self._pools.setdefault(key, deque())
        for _ in range(count):
            if len(pool) < self._max_free:
                pool.append(torch.empty(shape, dtype=dtype, device=device))
        logger.info(
            "BufferPool warmup: shape=%s dtype=%s count=%d (pool size=%d)",
            shape, dtype, count, len(pool),
        )

    def get(self, shape: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        key = (shape, dtype)
        pool = self._pools.get(key)
        if pool and len(pool) > 0:
            return pool.popleft()
        return torch.empty(shape, dtype=dtype, device=device)

    def put(self, tensor: torch.Tensor):
        key = (tuple(tensor.shape), tensor.dtype)
        pool = self._pools.setdefault(key, deque())
        if len(pool) < self._max_free:
            pool.append(tensor)


# ---- NIC auto-detection ----


def _detect_num_nic_groups(local_tp: int) -> int:
    """Auto-detect number of NIC groups from GPU-NIC PCI topology.

    Parses nvidia-smi topo to find which GPUs share a NIC (PXB connection).
    Only counts high-speed NICs (>= 100 Gb/s). Returns K such that K
    divides local_tp, defaulting to 1 if detection fails.

    The detection uses CUDA_VISIBLE_DEVICES to determine which physical
    GPUs are in use, then groups them by their closest high-speed NIC.
    """
    try:
        import subprocess
        import re

        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            gpu_ids = [int(x.strip()) for x in visible.split(",")]
        else:
            gpu_ids = list(range(local_tp))

        topo_result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True, text=True, timeout=5,
        )
        if topo_result.returncode != 0:
            return 1

        topo_lines = topo_result.stdout.strip().split("\n")

        header = None
        for line in topo_lines:
            if line.startswith("\t") and "GPU0" in line:
                header = line.strip().split("\t")
                break
        if header is None:
            return 1

        nic_columns = {}
        for idx, col in enumerate(header):
            if col.startswith("NIC"):
                nic_columns[col] = idx

        gpu_nic_map = {}
        for line in topo_lines:
            match = re.match(r"^GPU(\d+)\t", line)
            if not match:
                continue
            gpu_id = int(match.group(1))
            if gpu_id not in gpu_ids:
                continue
            cols = line.split("\t")
            best_nic = None
            for nic_name, col_idx in nic_columns.items():
                if col_idx < len(cols) and cols[col_idx].strip() == "PXB":
                    best_nic = nic_name
                    break
            if best_nic is None:
                for nic_name, col_idx in nic_columns.items():
                    if col_idx < len(cols) and cols[col_idx].strip() in ("PIX", "PHB"):
                        best_nic = nic_name
                        break
            if best_nic:
                gpu_nic_map[gpu_id] = best_nic

        if not gpu_nic_map:
            return 1

        nic_groups = {}
        for gpu_id in gpu_ids[:local_tp]:
            nic = gpu_nic_map.get(gpu_id, "unknown")
            nic_groups.setdefault(nic, []).append(gpu_id)

        K = len(nic_groups)
        if "unknown" in nic_groups:
            K = max(1, K - 1)

        while K > 1 and local_tp % K != 0:
            K -= 1

        logger.info(
            "NIC auto-detect: gpu_ids=%s, nic_groups=%s, K=%d",
            gpu_ids[:local_tp], dict(nic_groups), K,
        )
        return max(1, K)

    except Exception as e:
        logger.debug("NIC auto-detection failed: %s, defaulting to K=1", e)
        return 1


# ---- Point-to-point UCX communicator ----


class _UcxP2PCommunicator:
    """Low-level 1:1 UCX endpoint for RDMA between a pair of NIC representatives."""

    def __init__(self, is_ffn: bool, local_rank: int, peer_ffn_rank: int,
                 base_port: int, ffn_host: str, timeout: int,
                 bridge: _AsyncBridge, pool: _BufferPool, device: str):
        self._is_ffn = is_ffn
        self._local_rank = local_rank
        self._peer_ffn_rank = peer_ffn_rank
        self._base_port = base_port
        self._ffn_host = ffn_host
        self._timeout = timeout
        self._bridge = bridge
        self._pool = pool
        self._device = device
        self._endpoint = None
        self._listener = None
        self._connected = threading.Event()
        self._last_recv_buf: Optional[torch.Tensor] = None

    def connect(self):
        self._bridge.run(self._init_connection())
        if not self._connected.wait(timeout=self._timeout):
            raise TimeoutError(
                f"UCX P2P connection not established within {self._timeout}s"
            )

    async def _init_connection(self):
        import ucp

        port = self._base_port + self._peer_ffn_rank

        if self._is_ffn:
            logger.info("FFN rank %d: listening on port %d", self._local_rank, port)

            async def _on_connect(ep):
                rank_buf = np.empty(1, dtype=np.int64)
                await ep.recv(rank_buf)
                self._endpoint = ep
                self._connected.set()
                logger.info("FFN rank %d: peer connected (id=%d)",
                            self._local_rank, int(rank_buf[0]))

            self._listener = ucp.create_listener(_on_connect, port)
        else:
            max_retries = self._timeout * 2
            for attempt in range(max_retries):
                try:
                    self._endpoint = await ucp.create_endpoint(
                        self._ffn_host, port
                    )
                    rank_buf = np.array([self._local_rank], dtype=np.int64)
                    await self._endpoint.send(rank_buf)
                    self._connected.set()
                    logger.info(
                        "Attn rank %d: connected to FFN at %s:%d",
                        self._local_rank, self._ffn_host, port,
                    )
                    return
                except Exception as e:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(0.5)
                    else:
                        raise ConnectionError(
                            f"Attn rank {self._local_rank} connect failed: {e}"
                        ) from e

    def send(self, x: torch.Tensor):
        """Synchronous send: blocks until RDMA completes."""
        torch.cuda.current_stream().synchronize()
        self._bridge.run(self._async_send(x))

    def send_nonblocking(self, x: torch.Tensor):
        """Fire-and-forget send: submits to bridge, returns a Future.

        Caller must ensure GPU data is ready (e.g., via CUDA event)
        before the bridge thread reads the tensor.
        """
        return asyncio.run_coroutine_threadsafe(
            self._async_send(x), self._bridge._loop
        )

    def send_wait(self, future):
        """Wait for a nonblocking send to complete."""
        if future is not None:
            future.result()

    def recv(self) -> torch.Tensor:
        return self._bridge.run(self._async_recv())

    async def _async_send(self, x: torch.Tensor):
        meta = _encode_meta(x)
        await self._endpoint.send(meta)
        await self._endpoint.send(x.contiguous())

    async def _async_recv(self) -> torch.Tensor:
        if self._last_recv_buf is not None:
            self._pool.put(self._last_recv_buf)
        meta = np.empty(_META_SLOTS, dtype=np.int64)
        await self._endpoint.recv(meta)
        shape, dtype = _decode_meta(meta)
        buf = self._pool.get(shape, dtype, self._device)
        await self._endpoint.recv(buf)
        self._last_recv_buf = buf
        return buf

    def close(self):
        if self._endpoint is not None:
            try:
                self._bridge.run(self._endpoint.close())
            except Exception:
                pass
            self._endpoint = None
        if self._listener is not None:
            self._listener.close()
            self._listener = None


# ---- NIC-aware N:M communicator ----


class UcxTensorCommunicator(_FifoTensorCommunicatorBase):
    """NIC-aware UCX tensor communicator for AFD N:M heterogeneous TP.

    Architecture:
      K = number of NICs (env AFD_UCX_NUM_NICS, default 1).
      Each side's TP ranks are divided into K NIC groups.
      One representative per group does RDMA (sending/receiving 1/K of data).
      After RDMA, NVLink all_gather/broadcast reconstructs the full tensor
      on every rank.

    Cross-node RDMA traffic per layer = 2NH (independent of K and TP sizes).

    For K=1 (single NIC):
      rank 0 sends/receives full tensor via RDMA,
      then NVLink broadcast to all other ranks.

    For K>1:
      K representative pairs exchange 1/K data each via RDMA,
      then NVLink all_gather across local TP group + stride dedup.

    Environment variables:
      AFD_UCX_NUM_NICS   : number of NICs to use (default 1, must divide local_tp)
      AFD_UCX_BASE_PORT  : base port for FFN listeners (default 25000)
      AFD_UCX_FFN_HOST   : FFN node address (default "127.0.0.1")
      AFD_UCX_TLS        : UCX transport list (default "rc,tcp,cuda_copy,cuda_ipc")
      AFD_UCX_TIMEOUT    : connection timeout in seconds (default 60)
    """

    def __init__(self, afd_perspective: AFDPerspective):
        super().__init__()
        self._perspective = afd_perspective
        self._is_ffn = afd_perspective == AFDPerspective.AFD_PERSPECTIVE_FFN

        self._base_port = int(os.environ.get("AFD_UCX_BASE_PORT", "25000"))
        self._ffn_host = os.environ.get("AFD_UCX_FFN_HOST", "127.0.0.1")
        self._timeout = int(os.environ.get("AFD_UCX_TIMEOUT", "60"))
        self._device = f"cuda:{torch.cuda.current_device()}"

        self._local_rank = self._get_local_rank()
        self._resolve_config()

        self._setup_ucx_env()
        self._bridge = _AsyncBridge()
        self._pool = _BufferPool()
        self._tp_group = None
        self._skip_warmup = getattr(__import__(__name__), '_EAGER_INIT', False)

        self._p2p: Optional[_UcxP2PCommunicator] = None
        self._pending_num_tokens: deque = deque()

        logger.info(
            "UcxTensorCommunicator init: perspective=%s, rank=%d, "
            "local_tp=%d, K=%d, nic_group=%d, ranks_per_group=%d, "
            "is_representative=%s, device=%s",
            afd_perspective, self._local_rank, self._local_tp,
            self._K, self._nic_group, self._ranks_per_group,
            self._is_rep, self._device,
        )

        if self._is_rep:
            self._p2p = _UcxP2PCommunicator(
                is_ffn=self._is_ffn,
                local_rank=self._local_rank,
                peer_ffn_rank=self._nic_group,
                base_port=self._base_port,
                ffn_host=self._ffn_host,
                timeout=self._timeout,
                bridge=self._bridge,
                pool=self._pool,
                device=self._device,
            )
            self._p2p.connect()
        else:
            logger.info(
                "Rank %d: non-representative, will get data via NVLink",
                self._local_rank,
            )

        self._warmup_buffer_pool()
        logger.info("UcxTensorCommunicator: ready (K=%d)", self._K)

    def _warmup_buffer_pool(self):
        """Pre-allocate recv buffers based on model config if available."""
        if getattr(self, "_skip_warmup", False):
            logger.info("UcxTensorCommunicator: skipping buffer pool warmup (eager init)")
            return
        try:
            server_args = None
            try:
                from sglang.srt.server_args import get_global_server_args
                server_args = get_global_server_args()
            except Exception:
                pass

            hidden_size = getattr(server_args, "hidden_size", None)
            if hidden_size is None and server_args is not None:
                json_config = getattr(server_args, "json_config", None)
                if json_config:
                    hidden_size = json_config.get("hidden_size")

            micro_batch = int(os.environ.get(
                "AFD_MICRO_BATCH",
                str(getattr(server_args, "afd_micro_batch", 3)),
            ))

            if hidden_size is not None:
                buf_count = micro_batch + 2
                for max_tokens in [1, 32, 128]:
                    shape = (max_tokens, hidden_size)
                    self._pool.warmup(shape, torch.bfloat16, self._device, buf_count)
                    self._pool.warmup(shape, torch.float16, self._device, buf_count)
        except Exception as e:
            logger.debug("BufferPool warmup skipped: %s", e)

    def _resolve_config(self):
        server_args = None
        try:
            from sglang.srt.server_args import get_global_server_args
            server_args = get_global_server_args()
        except Exception:
            pass

        self._local_tp = getattr(server_args, "tp_size", 1) if server_args else 1
        self._local_tp = int(os.environ.get("AFD_LOCAL_TP", str(self._local_tp)))

        explicit_k = os.environ.get("AFD_UCX_NUM_NICS")
        if explicit_k is not None:
            K = int(explicit_k)
        elif self._local_tp > 1:
            K = _detect_num_nic_groups(self._local_tp)
        else:
            K = 1

        K = min(K, self._local_tp)

        if self._local_tp > 1 and K > 1 and self._local_tp % K != 0:
            old_K = K
            while K > 1 and self._local_tp % K != 0:
                K -= 1
            logger.warning(
                "NIC groups K=%d does not divide local_tp=%d, adjusted to K=%d",
                old_K, self._local_tp, K,
            )

        self._K = K
        self._ranks_per_group = self._local_tp // self._K if self._K > 0 else 1

        if self._ranks_per_group > 0:
            self._nic_group = self._local_rank // self._ranks_per_group
            self._nic_group = min(self._nic_group, self._K - 1)
            self._is_rep = (self._local_rank % self._ranks_per_group == 0)
        else:
            self._nic_group = 0
            self._is_rep = True

        if self._local_tp <= 1:
            self._K = 1
            self._ranks_per_group = 1
            self._nic_group = 0
            self._is_rep = True

    def _get_tp_group(self):
        if self._tp_group is None:
            try:
                from sglang.srt.distributed import get_tp_group
                self._tp_group = get_tp_group()
            except Exception:
                pass
        return self._tp_group

    @staticmethod
    def _get_local_rank() -> int:
        if os.environ.get("LOCAL_RANK") is not None:
            return int(os.environ["LOCAL_RANK"])
        try:
            if dist.is_initialized():
                server_args = None
                try:
                    from sglang.srt.server_args import get_global_server_args
                    server_args = get_global_server_args()
                except Exception:
                    pass
                tp = getattr(server_args, "tp_size", 1) if server_args else 1
                return dist.get_rank() % tp
        except Exception:
            pass
        return 0

    def _setup_ucx_env(self):
        os.environ.setdefault("UCX_RNDV_THRESH", "8192")
        os.environ.setdefault("UCX_RNDV_SCHEME", "put_zcopy")
        os.environ.setdefault("UCX_ZCOPY_THRESH", "8192")
        ucx_tls = os.environ.get("AFD_UCX_TLS", "rc,tcp,cuda_copy,cuda_ipc")
        os.environ.setdefault("UCX_TLS", ucx_tls)

    # ---- send_tensor ----

    def send_tensor(self, x: torch.Tensor):
        """Synchronous send (blocks until RDMA completes)."""
        if self._local_tp <= 1:
            self._p2p.send(x)
            return

        if self._K == 1:
            if self._is_rep:
                self._p2p.send(x)
            return

        num_tokens = x.shape[0]
        self._pending_num_tokens.append(num_tokens)

        if self._is_rep:
            chunk = (num_tokens + self._K - 1) // self._K
            start = self._nic_group * chunk
            end = min(start + chunk, num_tokens)
            shard = x[start:end].contiguous()
            if shard.shape[0] < chunk:
                pad = torch.zeros(
                    chunk - shard.shape[0], *x.shape[1:],
                    dtype=x.dtype, device=x.device,
                )
                shard = torch.cat([shard, pad], dim=0)
            self._p2p.send(shard)

    def send_tensor_nonblocking(self, x: torch.Tensor):
        """Non-blocking send: returns immediately, RDMA completes in background.

        Caller must ensure GPU data is ready before calling (e.g., via
        CUDA event synchronization). Use fence() to wait for completion.
        """
        self._last_send_future = None

        if self._local_tp <= 1:
            self._last_send_future = self._p2p.send_nonblocking(x)
            return

        if self._K == 1:
            if self._is_rep:
                self._last_send_future = self._p2p.send_nonblocking(x)
            return

        num_tokens = x.shape[0]
        self._pending_num_tokens.append(num_tokens)

        if self._is_rep:
            chunk = (num_tokens + self._K - 1) // self._K
            start = self._nic_group * chunk
            end = min(start + chunk, num_tokens)
            shard = x[start:end].contiguous()
            if shard.shape[0] < chunk:
                pad = torch.zeros(
                    chunk - shard.shape[0], *x.shape[1:],
                    dtype=x.dtype, device=x.device,
                )
                shard = torch.cat([shard, pad], dim=0)
            self._last_send_future = self._p2p.send_nonblocking(shard)

    def fence(self):
        """Wait for all in-flight nonblocking sends to complete."""
        future = getattr(self, "_last_send_future", None)
        if future is not None:
            self._p2p.send_wait(future)
            self._last_send_future = None

    # ---- recv_tensor ----

    def recv_tensor(self) -> torch.Tensor:
        if self._local_tp <= 1:
            return self._p2p.recv()

        tp_group = self._get_tp_group()

        if self._K == 1:
            return self._recv_k1(tp_group)
        else:
            return self._recv_k_multi(tp_group)

    def _recv_k1(self, tp_group) -> torch.Tensor:
        """K=1: rank 0 does RDMA recv, then NVLink broadcast to all."""
        if self._is_rep:
            tensor = self._p2p.recv()
        else:
            tensor = None

        if tp_group is not None and self._local_tp > 1:
            if self._is_rep:
                shape_dtype = torch.tensor(
                    list(tensor.shape) + [_DTYPE_TO_INT.get(tensor.dtype, 2)],
                    dtype=torch.long, device=self._device,
                )
                ndim_t = torch.tensor([tensor.ndim], dtype=torch.long, device=self._device)
            else:
                ndim_t = torch.empty(1, dtype=torch.long, device=self._device)

            dist.broadcast(ndim_t, src=tp_group.ranks[0], group=tp_group.device_group)
            ndim = int(ndim_t.item())

            if self._is_rep:
                shape_dtype = torch.tensor(
                    list(tensor.shape) + [_DTYPE_TO_INT.get(tensor.dtype, 2)],
                    dtype=torch.long, device=self._device,
                )
            else:
                shape_dtype = torch.empty(ndim + 1, dtype=torch.long, device=self._device)

            dist.broadcast(shape_dtype, src=tp_group.ranks[0], group=tp_group.device_group)

            if not self._is_rep:
                shape = tuple(int(shape_dtype[i].item()) for i in range(ndim))
                dtype = _INT_TO_DTYPE.get(int(shape_dtype[ndim].item()), torch.float32)
                tensor = torch.empty(shape, dtype=dtype, device=self._device)

            dist.broadcast(tensor, src=tp_group.ranks[0], group=tp_group.device_group)

        return tensor

    def _recv_k_multi(self, tp_group) -> torch.Tensor:
        """K>1: K representatives RDMA recv, then all_gather + stride dedup."""
        if self._is_rep:
            my_shard = self._p2p.recv()
        else:
            my_shard = None

        if tp_group is None or self._local_tp <= 1:
            return my_shard

        root_rank = tp_group.ranks[0]

        if self._is_rep and self._nic_group == 0:
            shard_info = torch.tensor(
                list(my_shard.shape) + [_DTYPE_TO_INT.get(my_shard.dtype, 2)],
                dtype=torch.long, device=self._device,
            )
            ndim_t = torch.tensor([my_shard.ndim], dtype=torch.long, device=self._device)
        else:
            ndim_t = torch.empty(1, dtype=torch.long, device=self._device)

        dist.broadcast(ndim_t, src=root_rank, group=tp_group.device_group)
        ndim = int(ndim_t.item())

        if self._is_rep and self._nic_group == 0:
            shard_info = torch.tensor(
                list(my_shard.shape) + [_DTYPE_TO_INT.get(my_shard.dtype, 2)],
                dtype=torch.long, device=self._device,
            )
        else:
            shard_info = torch.empty(ndim + 1, dtype=torch.long, device=self._device)

        dist.broadcast(shard_info, src=root_rank, group=tp_group.device_group)

        shard_shape = tuple(int(shard_info[i].item()) for i in range(ndim))
        shard_dtype = _INT_TO_DTYPE.get(int(shard_info[ndim].item()), torch.float32)

        if not self._is_rep:
            my_shard = torch.zeros(shard_shape, dtype=shard_dtype, device=self._device)

        gathered = [torch.empty_like(my_shard) for _ in range(self._local_tp)]
        dist.all_gather(gathered, my_shard, group=tp_group.device_group)

        unique = [gathered[i * self._ranks_per_group] for i in range(self._K)]
        full = torch.cat(unique, dim=0)

        if self._pending_num_tokens:
            original_tokens = self._pending_num_tokens.popleft()
            if full.shape[0] > original_tokens:
                full = full[:original_tokens]

        return full

    # ---- Cleanup ----

    def close(self):
        if self._p2p is not None:
            self._p2p.close()
            self._p2p = None
        self._bridge.stop()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
