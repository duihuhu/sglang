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
# UCX_MEMTYPE_CACHE: 'y' lets UCX auto-detect CUDA buffers for cuda_copy transport.
# For GPU-Direct (bypassing UCX-Py's Array CUDA check), we need 'n' because our
# replaced UCX libraries lack libucm_cuda.so, and memtype cache with 'y' but no
# CUDA hooks causes rendezvous to stall on GPU buffers passed as ctypes.
if os.environ.get("AFD_UCX_GPU_DIRECT", "0") == "1":
    os.environ["UCX_MEMTYPE_CACHE"] = "n"
else:
    os.environ.setdefault("UCX_MEMTYPE_CACHE", "y")

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


def _cross_node_experimental_enabled() -> bool:
    return os.environ.get("AFD_CROSS_NODE_EXPERIMENTAL", "0") == "1"


# ---- Dtype mapping ----

_DTYPE_TO_INT = {
    torch.float16: 0, torch.bfloat16: 1, torch.float32: 2,
    torch.float64: 3, torch.int32: 4, torch.int64: 5,
}
_INT_TO_DTYPE = {v: k for k, v in _DTYPE_TO_INT.items()}
_META_SLOTS = 8


def _as_uint8_numpy_view(tensor: torch.Tensor) -> np.ndarray:
    """Return a zero-copy byte view of a contiguous CPU tensor."""
    if tensor.device.type != "cpu":
        raise ValueError("A NumPy byte view requires a CPU tensor")
    if not tensor.is_contiguous():
        raise ValueError("A NumPy byte view requires a contiguous tensor")
    # NumPy does not support every torch dtype (notably bfloat16). UCX only
    # needs the underlying bytes, so reinterpret them before entering NumPy.
    # The ndarray keeps the torch uint8 view (and therefore its storage) alive.
    return tensor.view(torch.uint8).numpy()


def _encode_meta(tensor: torch.Tensor, original_num_tokens: int = 0) -> np.ndarray:
    """Encode tensor shape, dtype, and original token count into a fixed-size int64 array.

    Layout: [ndim, shape[0], ..., shape[ndim-1], dtype_code, original_num_tokens, ...]
    """
    meta = np.zeros(_META_SLOTS, dtype=np.int64)
    ndim = tensor.ndim
    if ndim > _META_SLOTS - 3:
        raise ValueError(f"Tensor ndim {ndim} exceeds metadata capacity {_META_SLOTS - 3}")
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
    meta[ndim + 2] = original_num_tokens
    return meta


def _decode_meta(meta: np.ndarray) -> Tuple[tuple, torch.dtype, int]:
    """Decode metadata array into (shape, dtype, original_num_tokens)."""
    ndim = int(meta[0])
    shape = tuple(int(meta[i + 1]) for i in range(ndim))
    dtype_code = int(meta[ndim + 1])
    dtype = _INT_TO_DTYPE.get(dtype_code)
    if dtype is None:
        raise ValueError(
            f"Unknown dtype code {dtype_code} in UCX tensor metadata. "
            f"Known codes: {list(_INT_TO_DTYPE.keys())}"
        )
    original_num_tokens = int(meta[ndim + 2])
    return shape, dtype, original_num_tokens


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
        # Initialize CUDA context in this thread for GPU-direct UCX transfers
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.current_device()
        except Exception:
            pass
        self._loop = asyncio.SelectorEventLoop()
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    def run(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def submit_callable(self, fn):
        """Submit a synchronous callable to run on the bridge thread.

        The callable is wrapped in a coroutine and scheduled on the event loop.
        Returns a concurrent.futures.Future that resolves when fn() completes.
        """
        async def _wrapper():
            return fn()
        return asyncio.run_coroutine_threadsafe(_wrapper(), self._loop)

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
        self._send_lock: Optional[asyncio.Lock] = None  # created lazily on bridge loop
        self._cross_node_experimental = _cross_node_experimental_enabled()
        self._send_meta_signature = None
        self._recv_meta_signature = None
        self._send_host_signature = None
        self._send_host_wire = None
        self._send_host_buffer = None
        self._recv_buffer = None
        self._recv_host_buffer = None
        self._recv_host_wire = None
        self._host_staging = (
            self._cross_node_experimental
            and os.environ.get("AFD_UCX_HOST_STAGING", "0") == "1"
        )
        self._pinned_staging = (
            self._cross_node_experimental
            and os.environ.get("AFD_UCX_PINNED_STAGING", "0") == "1"
        )
        self._gpu_direct = (
            self._cross_node_experimental
            and os.environ.get("AFD_UCX_GPU_DIRECT", "0") == "1"
            and not self._host_staging
        )
        if self._gpu_direct:
            import ctypes as _ctypes
            self._ctypes = _ctypes
            logger.info(
                "UCX GPU-Direct RDMA enabled for rank %d peer group %d: "
                "CUDA payloads sent/received directly via nvidia_peermem",
                self._local_rank,
                self._peer_ffn_rank,
            )
        if self._host_staging:
            logger.info(
                "UCX host staging enabled for rank %d peer group %d: "
                "CUDA payloads will use %s CPU wire buffers",
                self._local_rank,
                self._peer_ffn_rank,
                "pinned" if self._pinned_staging else "pageable NumPy",
            )

    def connect(self):
        self._bridge.run(self._init_connection())
        if not self._connected.wait(timeout=self._timeout):
            raise TimeoutError(
                f"UCX P2P connection not established within {self._timeout}s"
            )

    def start_listen(self):
        """FFN only: start listener without waiting for peer connection."""
        if not self._is_ffn:
            raise RuntimeError("start_listen() is only for FFN side")
        self._bridge.run(self._init_connection())

    def wait_connected(self, timeout: Optional[float] = None):
        """Wait for peer to connect (after start_listen or connect)."""
        t = timeout if timeout is not None else self._timeout
        if not self._connected.wait(timeout=t):
            raise TimeoutError(
                f"UCX P2P connection not established within {t}s"
            )

    async def _init_connection(self):
        logger.info(
            "UCX init: MEMTYPE_REG=%s RNDV_THRESH=%s RNDV_FRAG=%s MEMTYPE_CACHE=%s TLS=%s GPU_DIRECT=%s",
            os.environ.get("UCX_MEMTYPE_REG_WHOLE_ALLOC_TYPES", "?"),
            os.environ.get("UCX_RNDV_THRESH", "?"),
            os.environ.get("UCX_RNDV_FRAG_MEM_TYPE", "?"),
            os.environ.get("UCX_MEMTYPE_CACHE", "?"),
            os.environ.get("UCX_TLS", "?"),
            os.environ.get("AFD_UCX_GPU_DIRECT", "?"),
        )
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

    def reset_metadata_cache(self):
        """Reset per-forward metadata cache on both peers.

        AFD tensors keep the same shape/dtype across transformer layers.  The
        first transfer of a forward exchanges metadata; subsequent layers send
        only the CUDA payload.  Both peers call this at the forward boundary.
        """
        if self._cross_node_experimental:
            self._send_meta_signature = None
            self._recv_meta_signature = None
            self._recv_buffer = None
            self._recv_host_buffer = None
            self._recv_host_wire = None

    def _allocate_host_wire(self, nbytes: int) -> np.ndarray:
        """Allocate the exact uint8 buffer passed to UCX.

        Pageable NumPy memory is the safe default: UCX never sees memory owned
        by PyTorch's pinned allocator. Pinned staging remains an explicit
        opt-in for installations where that allocator/UCX combination works.
        """
        if not self._pinned_staging:
            return np.empty(nbytes, dtype=np.uint8)
        pinned = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        return _as_uint8_numpy_view(pinned)

    def _gpu_direct_buf(self, tensor: torch.Tensor):
        """Wrap a CUDA tensor's data_ptr as a ctypes char array.

        UCX-Py's Array.__cinit__ sees this as a generic host buffer (no
        __cuda_array_interface__), bypassing the cuda_support check.
        UCX C library's memory type detection (libucm with --with-cuda)
        recognizes the address as CUDA memory and uses GPU-Direct RDMA
        via nvidia_peermem for the actual transfer.
        """
        nbytes = tensor.numel() * tensor.element_size()
        return (self._ctypes.c_char * nbytes).from_address(tensor.data_ptr())

    @staticmethod
    def _wire_tensor(
        wire: np.ndarray, shape: tuple, dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.from_numpy(wire).view(dtype).reshape(shape)

    def _prepare_send_buffer(
        self, x: torch.Tensor, *, cache_host_wire: bool = False
    ):
        """Prepare a UCX payload on the caller thread.

        The single cached host wire is safe only for synchronous ``send``:
        ``bridge.run(_async_send)`` does not return until UCX has completed the
        transfer. Nonblocking callers must use an independent wire so an
        in-flight payload cannot be overwritten by the next D2H copy.
        """
        x_contig = x if x.is_contiguous() else x.contiguous()
        if not self._host_staging:
            return x_contig

        shape = tuple(x_contig.shape)
        dtype = x_contig.dtype
        nbytes = x_contig.numel() * x_contig.element_size()
        signature = (shape, dtype, nbytes, self._pinned_staging)
        if cache_host_wire:
            if self._send_host_signature != signature:
                self._send_host_wire = self._allocate_host_wire(nbytes)
                self._send_host_buffer = self._wire_tensor(
                    self._send_host_wire, shape, dtype
                )
                self._send_host_signature = signature
            wire = self._send_host_wire
            host = self._send_host_buffer
        else:
            wire = self._allocate_host_wire(nbytes)
            host = self._wire_tensor(wire, shape, dtype)

        host.copy_(x_contig, non_blocking=x_contig.device.type == "cuda")
        if x_contig.device.type == "cuda":
            # This synchronization both orders the producer and completes D2H;
            # no separate pre-copy synchronization is needed for host staging.
            torch.cuda.current_stream(x_contig.device).synchronize()
        return wire, shape, dtype

    def send(self, x: torch.Tensor, original_num_tokens: int = 0):
        """Synchronous send; blocks until transfer completes."""
        if not self._cross_node_experimental:
            torch.cuda.current_stream().synchronize()
            send_buffer = x.contiguous().cpu()
        else:
            if self._host_staging:
                send_buffer = self._prepare_send_buffer(x, cache_host_wire=True)
            else:
                if x.device.type == "cuda":
                    # GPU-direct UCX must not observe an unfinished producer.
                    torch.cuda.current_stream(x.device).synchronize()
                send_buffer = self._prepare_send_buffer(x)
        self._bridge.run(self._async_send(send_buffer, original_num_tokens))

    def send_nonblocking(self, x: torch.Tensor, original_num_tokens: int = 0,
                          _prof_layer: int = -1, _prof_mb: int = -1):
        """Fire-and-forget send: submits to bridge, returns a Future.

        Caller must ensure GPU data is ready (e.g., via CUDA event)
        before the bridge thread reads the tensor.
        """
        if _prof_layer < 0:
            try:
                from sglang.srt.layers.afd_mixin import _afd_ctx
                _prof_layer = _afd_ctx.get("layer", -1)
                _prof_mb = _afd_ctx.get("mb", -1)
            except Exception:
                pass
        send_buffer = (
            self._prepare_send_buffer(x, cache_host_wire=False)
            if self._cross_node_experimental
            else x
        )
        return asyncio.run_coroutine_threadsafe(
            self._async_send(send_buffer, original_num_tokens, _prof_layer, _prof_mb),
            self._bridge._loop,
        )

    def send_wait(self, future):
        """Wait for a nonblocking send to complete."""
        if future is not None:
            future.result()

    def send_nonblocking_stream_ordered(self, x: torch.Tensor,
                                         comm_stream,
                                         original_num_tokens: int = 0,
                                         _prof_layer: int = -1,
                                         _prof_mb: int = -1):
        """Stream-ordered fire-and-forget send.

        comm_stream already has a wait_event queued for the compute kernel.
        We do comm_stream.synchronize() on the CALLER thread (not the bridge
        event loop) to avoid blocking the asyncio loop.  Since wait_event is
        already queued, this sync is near-zero cost (~5μs) once the compute
        kernel finishes.  Then we submit the pure-async UCX send to the bridge.
        """
        import time as _time
        try:
            from sglang.srt.layers.afd_mixin import _afd_host_events as _host_ev
        except Exception:
            _host_ev = None

        # Sync on caller thread — does NOT block the bridge event loop
        t0 = _time.time()
        comm_stream.synchronize()  # ~5μs: confirms wait_event resolved
        t1 = _time.time()
        if _host_ev is not None:
            _host_ev.append({
                "ts_ms": round(t0 * 1000, 3),
                "role": "UCX_INNER", "layer": _prof_layer, "mb": _prof_mb,
                "event": "async_send_brk",
                "stream_sync_us": round((t1 - t0) * 1e6, 1),
            })

        # GPU data is now guaranteed ready. D2H staging stays on this caller thread.
        send_buffer = (
            self._prepare_send_buffer(x, cache_host_wire=False)
            if self._cross_node_experimental
            else x
        )
        return asyncio.run_coroutine_threadsafe(
            self._async_send(send_buffer, original_num_tokens, _prof_layer, _prof_mb),
            self._bridge._loop,
        )

    def recv(self) -> Tuple[torch.Tensor, int]:
        """Receive a tensor, copying staged host data to CUDA on this thread."""
        received, original_num_tokens = self._bridge.run(self._async_recv())
        if not self._cross_node_experimental:
            return received.to(self._device), original_num_tokens
        if not self._host_staging:
            return received, original_num_tokens
        shape, dtype = tuple(received.shape), received.dtype
        if (
            self._recv_buffer is None
            or tuple(self._recv_buffer.shape) != shape
            or self._recv_buffer.dtype != dtype
        ):
            self._recv_buffer = torch.empty(
                shape, dtype=dtype, device=self._device
            )
        self._recv_buffer.copy_(received, non_blocking=False)
        return self._recv_buffer, original_num_tokens

    async def _async_send(self, x: torch.Tensor, original_num_tokens: int = 0,
                          _prof_layer: int = -1, _prof_mb: int = -1):
        # Lazily create the lock on the event loop thread
        if self._send_lock is None:
            self._send_lock = asyncio.Lock()
        async with self._send_lock:
            t0 = time.time()
            if not self._cross_node_experimental:
                meta = _encode_meta(x, original_num_tokens)
                await self._endpoint.send(meta)
                t1 = time.time()
                x_contig = x.contiguous()
                await self._endpoint.send(_as_uint8_numpy_view(x_contig))
                t2 = time.time()
                return
            if isinstance(x, tuple):
                payload, shape, dtype = x
                signature = (shape, dtype, original_num_tokens)
                meta_tensor = self._wire_tensor(payload, shape, dtype)
            else:
                x_contig = x if x.is_contiguous() else x.contiguous()
                signature = (tuple(x_contig.shape), x_contig.dtype, original_num_tokens)
                meta_tensor = x_contig
                if x_contig.device.type == "cpu":
                    payload = _as_uint8_numpy_view(x_contig)
                elif self._gpu_direct:
                    payload = self._gpu_direct_buf(x_contig)
                else:
                    payload = x_contig
            meta_changed = signature != self._send_meta_signature
            # Always send a 1-byte flag indicating whether meta follows.
            flag = np.array([1 if meta_changed else 0], dtype=np.uint8)
            await self._endpoint.send(flag)
            if meta_changed:
                meta = _encode_meta(meta_tensor, original_num_tokens)
                await self._endpoint.send(meta)
                self._send_meta_signature = signature
            t1 = time.time()
            t2 = t1
            if self._gpu_direct and not isinstance(payload, np.ndarray):
                logger.info(
                    "GPU-Direct send: type=%s nbytes=%d ptr=0x%x",
                    type(payload).__name__, len(payload),
                    self._ctypes.addressof(payload),
                )
            await self._endpoint.send(payload)
            t3 = time.time()
            try:
                from sglang.srt.layers.afd_mixin import _afd_host_events
                _afd_host_events.append({
                    "ts_ms": round(t0 * 1000, 3),
                    "role": "UCX_INNER", "layer": _prof_layer, "mb": _prof_mb,
                    "event": "async_send_breakdown",
                    "encode_meta_us": round((t1 - t0) * 1e6, 1),
                    "send_meta_us": round((t2 - t1) * 1e6, 1),
                    "send_data_us": round((t3 - t2) * 1e6, 1),
                    "total_us": round((t3 - t0) * 1e6, 1),
                })
            except Exception:
                pass

    async def _async_recv(self) -> Tuple[torch.Tensor, int]:
        # Recv lock is needed because meta+data are two separate messages
        # that must be received atomically (in order) on the same endpoint.
        if not hasattr(self, "_recv_lock_async") or self._recv_lock_async is None:
            self._recv_lock_async = asyncio.Lock()
        async with self._recv_lock_async:
            t0 = time.time()
            if not self._cross_node_experimental:
                meta = np.empty(_META_SLOTS, dtype=np.int64)
                await self._endpoint.recv(meta)
                shape, dtype, original_num_tokens = _decode_meta(meta)
                nbytes = int(np.prod(shape)) * torch.empty((), dtype=dtype).element_size()
                buf_np = np.empty(nbytes, dtype=np.uint8)
                await self._endpoint.recv(buf_np)
                return torch.frombuffer(buf_np, dtype=dtype).reshape(shape), original_num_tokens
            # Receive the 1-byte flag indicating whether meta follows.
            flag_buf = np.empty(1, dtype=np.uint8)
            await self._endpoint.recv(flag_buf)
            has_meta = flag_buf[0] != 0
            if has_meta:
                meta = np.empty(_META_SLOTS, dtype=np.int64)
                await self._endpoint.recv(meta)
                shape, dtype, original_num_tokens = _decode_meta(meta)
                self._recv_meta_signature = (shape, dtype, original_num_tokens)
            else:
                if self._recv_meta_signature is None:
                    raise RuntimeError(
                        "UCX received metadata flag=0 before any cached metadata"
                    )
                shape, dtype, original_num_tokens = self._recv_meta_signature
            if self._host_staging:
                nbytes = (
                    int(np.prod(shape))
                    * torch.empty((), dtype=dtype).element_size()
                )
                if (
                    self._recv_host_wire is None
                    or self._recv_host_wire.nbytes != nbytes
                    or tuple(self._recv_host_buffer.shape) != shape
                    or self._recv_host_buffer.dtype != dtype
                ):
                    self._recv_host_wire = self._allocate_host_wire(nbytes)
                    self._recv_host_buffer = self._wire_tensor(
                        self._recv_host_wire, shape, dtype
                    )
                recv_buffer = self._recv_host_buffer
            else:
                if (
                    self._recv_buffer is None
                    or tuple(self._recv_buffer.shape) != shape
                    or self._recv_buffer.dtype != dtype
                ):
                    self._recv_buffer = torch.empty(
                        shape, dtype=dtype, device=self._device
                    )
                recv_buffer = self._recv_buffer
            t1 = time.time()
            if self._host_staging:
                payload = self._recv_host_wire
            elif self._gpu_direct:
                payload = self._gpu_direct_buf(recv_buffer)
            else:
                payload = recv_buffer
            await self._endpoint.recv(payload)
            t2 = time.time()
            # Return a stable tensor object. The next receive occurs only after
            # this layer has consumed it in the M=1 path.
            received_buffer = recv_buffer
            try:
                from sglang.srt.layers.afd_mixin import _afd_host_events
                _afd_host_events.append({
                    "ts_ms": round(t0 * 1000, 3),
                    "role": "UCX_INNER", "layer": -1, "mb": -1,
                    "event": "async_recv_breakdown",
                    "recv_meta_us": round((t1 - t0) * 1e6, 1),
                    "recv_data_us": round((t2 - t1) * 1e6, 1),
                    "total_us": round((t2 - t0) * 1e6, 1),
                })
            except Exception:
                pass
            return received_buffer, original_num_tokens

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
      AFD_UCX_HOST_STAGING: stage CUDA payloads through host buffers (default 0)
      AFD_UCX_PINNED_STAGING: use pinned instead of pageable NumPy staging (default 0)
      AFD_UCX_GPU_DIRECT: bypass UCX-Py CUDA check and send GPU buffers directly
          via nvidia_peermem GPU-Direct RDMA (default 0; requires UCX --with-cuda)
      AFD_UCX_SPLIT_RECV : overlap K=1 RDMA receive with compute, then broadcast
          from the main TP thread (default 0; requires GPU-Direct)
    """

    def __init__(self, afd_perspective: AFDPerspective,
                 mb_id: Optional[int] = None,
                 defer_connect: bool = False):
        super().__init__()
        self._perspective = afd_perspective
        self._is_ffn = afd_perspective == AFDPerspective.AFD_PERSPECTIVE_FFN

        self._base_port = int(os.environ.get("AFD_UCX_BASE_PORT", "25000"))
        # mb_id-based stride avoids port collisions when multiple
        # UcxTensorCommunicators coexist in the same process (used by
        # --afd-async-schedule).  None means "legacy single comm" — leave
        # the base port untouched so existing fixtures still work.
        self._mb_id: Optional[int] = mb_id
        if self._mb_id is not None:
            # 1024-port stride is far larger than any realistic
            # peer_ffn_rank fanout.
            self._base_port += (self._mb_id + 1) * 1024
        self._ffn_host = os.environ.get("AFD_UCX_FFN_HOST", "127.0.0.1")
        self._timeout = int(os.environ.get("AFD_UCX_TIMEOUT", "60"))
        self._device = f"cuda:{torch.cuda.current_device()}"

        self._local_rank = self._get_local_rank()
        self._resolve_config()
        self._split_phase_recv = (
            os.environ.get("AFD_UCX_SPLIT_RECV", "0") == "1"
            and os.environ.get("AFD_UCX_GPU_DIRECT", "0") == "1"
            and self._K == 1
        )

        self._setup_ucx_env()
        self._bridge = _AsyncBridge()
        self._pool = _BufferPool()
        self._tp_group = None
        self._skip_warmup = getattr(__import__(__name__), '_EAGER_INIT', False)

        self._p2p: Optional[_UcxP2PCommunicator] = None

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
            if not defer_connect:
                self._p2p.connect()
        else:
            logger.info(
                "Rank %d: non-representative, will get data via NVLink",
                self._local_rank,
            )

        if not defer_connect:
            self._warmup_buffer_pool()
            logger.info("UcxTensorCommunicator: ready (K=%d)", self._K)

    def start_listen(self):
        """FFN only: bind listener ports without waiting for Attn to connect.

        Call wait_connected() later to block until the peer arrives.
        """
        if not self._is_ffn:
            raise RuntimeError("start_listen() is only for FFN side")
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
            self._p2p.start_listen()

    def wait_connected(self, timeout=None):
        """Block until peer connects (after start_listen)."""
        if self._p2p is not None:
            self._p2p.wait_connected(timeout)
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
        os.environ.setdefault(
            "UCX_RNDV_SCHEME",
            "get_zcopy" if _cross_node_experimental_enabled() else "put_zcopy",
        )
        os.environ.setdefault("UCX_ZCOPY_THRESH", "8192")
        ucx_tls = os.environ.get("AFD_UCX_TLS", "rc,tcp,cuda_copy,cuda_ipc")
        os.environ.setdefault("UCX_TLS", ucx_tls)
        if os.environ.get("AFD_UCX_GPU_DIRECT", "0") == "1":
            os.environ.setdefault("UCX_MEMTYPE_REG_WHOLE_ALLOC_TYPES", "cuda")
            os.environ["UCX_RNDV_FRAG_MEM_TYPE"] = "host"
            # GPU-Direct RDMA requires access to physical NICs (not bond devices).
            # Setting NET_DEVICES=all lets UCX select the NIC closest to each GPU.
            os.environ.setdefault("UCX_NET_DEVICES", "all")

    def reset_metadata_cache(self):
        """Start a new AFD forward with one metadata exchange."""
        if _cross_node_experimental_enabled() and self._p2p is not None:
            self._p2p.reset_metadata_cache()
        # Split-phase K=1 receive buffers are indexed by async ring slot.
        # Keep allocations across forwards, but metadata is exchanged per recv
        # because dynamic microbatches can have different shapes.
        if not hasattr(self, "_k1_recv_bufs"):
            self._k1_recv_bufs = {}

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
            self._p2p.send(shard, original_num_tokens=num_tokens)

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
            self._last_send_future = self._p2p.send_nonblocking(
                shard, original_num_tokens=num_tokens
            )

    def fence(self):
        """Wait for all in-flight nonblocking sends to complete."""
        future = getattr(self, "_last_send_future", None)
        if future is not None:
            self._p2p.send_wait(future)
            self._last_send_future = None

    def send_tensor_nonblocking_stream_ordered(self, x: torch.Tensor,
                                                comm_stream, _prof_layer=-1,
                                                _prof_mb=-1):
        """Stream-ordered nonblocking send: comm_stream already has wait_event queued.

        The bridge thread does a lightweight comm_stream.synchronize() (near-zero
        cost since wait_event resolves as soon as compute kernel finishes) then
        fires the UCX send.  This eliminates the 38-288μs CPU event.synchronize()
        overhead from the old daemon-thread approach.
        """
        self._last_send_future = None

        if self._local_tp <= 1:
            self._last_send_future = self._p2p.send_nonblocking_stream_ordered(
                x, comm_stream, _prof_layer=_prof_layer, _prof_mb=_prof_mb
            )
            return

        if self._K == 1:
            if self._is_rep:
                self._last_send_future = self._p2p.send_nonblocking_stream_ordered(
                    x, comm_stream, _prof_layer=_prof_layer, _prof_mb=_prof_mb
                )
            return

        num_tokens = x.shape[0]

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
            self._last_send_future = self._p2p.send_nonblocking_stream_ordered(
                shard, comm_stream, original_num_tokens=num_tokens,
                _prof_layer=_prof_layer, _prof_mb=_prof_mb
            )

    # ---- recv_tensor ----

    def recv_tensor(self) -> torch.Tensor:
        if self._local_tp <= 1:
            tensor, _orig = self._p2p.recv()
            return tensor

        tp_group = self._get_tp_group()

        if self._K == 1:
            return self._recv_k1(tp_group)
        else:
            return self._recv_k_multi(tp_group)

    def recv_rdma_only(self) -> Optional[torch.Tensor]:
        """Receive the K=1 RDMA payload without entering a TP collective.

        This method is safe to run on a background thread. The representative
        returns the received CUDA tensor; non-representatives return ``None``.
        All ranks must later call :meth:`recv_broadcast` from their main thread.
        """
        if self._K != 1:
            raise RuntimeError("recv_rdma_only is only valid when K=1")
        if self._local_tp <= 1 or self._is_rep:
            tensor, _orig = self._p2p.recv()
            return tensor
        return None

    def recv_broadcast(
        self, tensor: Optional[torch.Tensor], slot: int = 0
    ) -> torch.Tensor:
        """Broadcast a completed K=1 RDMA receive to all local TP ranks.

        Dynamic microbatch scheduling may reuse a ring slot for a different
        shape at any layer, so shape/dtype metadata is packed into one fixed
        collective for every receive. This is one metadata collective instead
        of the previous two (ndim then shape/dtype).
        """
        if self._K != 1:
            raise RuntimeError("recv_broadcast is only valid when K=1")
        if self._local_tp <= 1:
            if tensor is None:
                raise RuntimeError("K=1 representative received no tensor")
            return tensor

        tp_group = self._get_tp_group()
        if tp_group is None:
            if tensor is None:
                raise RuntimeError("TP group unavailable on non-representative rank")
            return tensor

        max_ndim = 8
        if self._is_rep:
            if tensor is None:
                raise RuntimeError("K=1 representative received no tensor")
            if tensor.ndim > max_ndim:
                raise ValueError(
                    f"AFD tensor ndim={tensor.ndim} exceeds metadata capacity {max_ndim}"
                )
            metadata = torch.zeros(
                max_ndim + 2, dtype=torch.long, device=self._device
            )
            metadata[0] = tensor.ndim
            metadata[1 : 1 + tensor.ndim] = torch.tensor(
                tensor.shape, dtype=torch.long, device=self._device
            )
            metadata[-1] = _DTYPE_TO_INT.get(tensor.dtype, 2)
        else:
            metadata = torch.empty(
                max_ndim + 2, dtype=torch.long, device=self._device
            )
        dist.broadcast(metadata, src=tp_group.ranks[0], group=tp_group.device_group)
        ndim = int(metadata[0].item())
        shape = tuple(int(metadata[1 + i].item()) for i in range(ndim))
        dtype = _INT_TO_DTYPE.get(int(metadata[-1].item()), torch.float32)

        if not self._is_rep:
            buffers = getattr(self, "_k1_recv_bufs", None)
            if buffers is None:
                buffers = self._k1_recv_bufs = {}
            tensor = buffers.get(slot)
            if tensor is None or tuple(tensor.shape) != shape or tensor.dtype != dtype:
                tensor = torch.empty(shape, dtype=dtype, device=self._device)
                buffers[slot] = tensor

        dist.broadcast(tensor, src=tp_group.ranks[0], group=tp_group.device_group)
        return tensor

    def _recv_k1(self, tp_group) -> torch.Tensor:
        """K=1: rank 0 does RDMA recv, then NVLink broadcast to all.

        Optimization: after the first recv establishes shape/dtype, subsequent
        recvs skip the ndim+shape_dtype broadcasts (2 out of 3 NCCL calls).
        The shape is cached on all ranks from the first exchange.
        """
        if self._is_rep:
            tensor, _orig = self._p2p.recv()
        else:
            tensor = None

        if tp_group is None or self._local_tp <= 1:
            return tensor

        # Fast path: shape is cached from previous recv in this forward
        cached = getattr(self, "_k1_cached_shape", None)
        if cached is not None:
            c_shape, c_dtype = cached
            if not self._is_rep:
                if self._k1_recv_buf is None or self._k1_recv_buf.shape != c_shape:
                    self._k1_recv_buf = torch.empty(
                        c_shape, dtype=c_dtype, device=self._device
                    )
                tensor = self._k1_recv_buf
            dist.broadcast(tensor, src=tp_group.ranks[0], group=tp_group.device_group)
            return tensor

        # First recv: full metadata exchange
        if self._is_rep:
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
        else:
            shape = tuple(tensor.shape)
            dtype = tensor.dtype

        dist.broadcast(tensor, src=tp_group.ranks[0], group=tp_group.device_group)

        # Cache for subsequent layers in this forward
        self._k1_cached_shape = (shape, dtype)
        self._k1_recv_buf = None
        return tensor

    def _recv_k_multi(self, tp_group) -> torch.Tensor:
        """K>1: K representatives RDMA recv, then all_gather + stride dedup."""
        original_num_tokens = 0
        if self._is_rep:
            my_shard, original_num_tokens = self._p2p.recv()
        else:
            my_shard = None

        if tp_group is None or self._local_tp <= 1:
            return my_shard

        root_rank = tp_group.ranks[0]

        if self._is_rep and self._nic_group == 0:
            shard_info = torch.tensor(
                list(my_shard.shape) + [_DTYPE_TO_INT.get(my_shard.dtype, 2), original_num_tokens],
                dtype=torch.long, device=self._device,
            )
            ndim_t = torch.tensor([my_shard.ndim], dtype=torch.long, device=self._device)
        else:
            ndim_t = torch.empty(1, dtype=torch.long, device=self._device)

        dist.broadcast(ndim_t, src=root_rank, group=tp_group.device_group)
        ndim = int(ndim_t.item())

        if self._is_rep and self._nic_group == 0:
            shard_info = torch.tensor(
                list(my_shard.shape) + [_DTYPE_TO_INT.get(my_shard.dtype, 2), original_num_tokens],
                dtype=torch.long, device=self._device,
            )
        else:
            shard_info = torch.empty(ndim + 2, dtype=torch.long, device=self._device)

        dist.broadcast(shard_info, src=root_rank, group=tp_group.device_group)

        shard_shape = tuple(int(shard_info[i].item()) for i in range(ndim))
        shard_dtype = _INT_TO_DTYPE.get(int(shard_info[ndim].item()), torch.float32)
        original_tokens = int(shard_info[ndim + 1].item())

        if not self._is_rep:
            my_shard = torch.zeros(shard_shape, dtype=shard_dtype, device=self._device)

        gathered = [torch.empty_like(my_shard) for _ in range(self._local_tp)]
        dist.all_gather(gathered, my_shard, group=tp_group.device_group)

        unique = [gathered[i * self._ranks_per_group] for i in range(self._K)]
        full = torch.cat(unique, dim=0)

        if original_tokens > 0 and full.shape[0] > original_tokens:
            full = full[:original_tokens]

        return full

    # ---- Cleanup ----

    def close(self):
        if self._p2p is not None:
            self._p2p.close()
            self._p2p = None
        self._bridge.stop()

    # ---- Hot Reconnect (for live reshard without PF restart) ----

    def reconnect(self, timeout: Optional[float] = None):
        """Reconnect UCX endpoint without restarting peer (PF stays alive).

        For ATTN side: closes old endpoint, creates a new one to the same FFN.
        For FFN side: closes old endpoint, re-listens for new ATTN connection.

        This enables live PA reshard without killing PF.
        Typical latency: ~50ms (endpoint create + handshake).
        """
        t = timeout if timeout is not None else self._timeout
        if self._p2p is None:
            logger.warning("reconnect: no P2P communicator, nothing to do")
            return

        logger.info("UCX reconnect: closing old endpoint...")
        self._p2p.close()
        self._p2p._connected.clear()

        logger.info("UCX reconnect: re-establishing connection...")
        if self._is_ffn:
            self._p2p.start_listen()
        else:
            self._p2p.connect()

        self._p2p.wait_connected(t)
        self._warmup_buffer_pool()
        logger.info("UCX reconnect: done")

    # Aliases for compatibility with AFD code paths that call stream-ordered variants
    send_stream_ordered = send_tensor
    recv_stream_ordered = recv_tensor

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
