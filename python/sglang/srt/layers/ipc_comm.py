"""CUDA IPC communicator for single-node A↔F over NVLink.

Replaces UCX-Py with direct CUDA IPC handles + POSIX shared memory flags.
No Python event loop, no coroutines, no select() syscall.

Architecture per rank pair (DA rank i ↔ DF rank i):
  - DA allocates send_buf GPU, exports IPC handle → DF imports as peer tensor
  - DF allocates send_buf GPU, exports IPC handle → DA imports as peer tensor
  - POSIX shared memory (/dev/shm) carries per-slot flags + message sizes:
      offsets 0-31:   flag_a2f[0..3] (4 × uint64)
      offsets 32-63:  flag_f2a[0..3] (4 × uint64)
      offsets 64-95:  size_a2f[0..3] (4 × uint64)
      offsets 96-127: size_f2a[0..3] (4 × uint64)
  - Data transfer: cross-device cudaMemcpyPeer via NVLink
  - Synchronization: volatile uint64 load in spin-loop (~ns)

Design decisions:
  - Per-slot flags (RING_SIZE=4) enable M>1 pipelining: sender can post up to
    3 messages ahead of receiver without blocking.
  - 2-phase recv (recv_poll + recv_complete): bg thread only does flag polling
    (no GPU ops), avoiding the 10-40ms bg-thread event.synchronize() bottleneck.
    Main thread does the fast GPU copy + event sync (~80us).
  - Sends use comm_stream path (Path 2) — blocks main thread for ~105us which
    is acceptable given 400-550us GPU compute per layer.
  - msg_size in SHM allows single-copy recv (no separate header pre-copy + sync).
"""

import os
import queue
import struct
import pickle
import socket
import time
from contextlib import nullcontext
import mmap
import logging
import threading
from typing import Optional, Tuple

import torch
import numpy as np

from sglang.srt.layers.afd_type import AFDPerspective
from sglang.srt.layers.rdma_comm import _encode_meta, _decode_meta, _META_SLOTS

logger = logging.getLogger(__name__)

# SHM layout: 4 slots × 2 directions × (flag + size) × uint64 = 128 bytes
_SHM_FLAGS_A2F = 0    # offset  0: flag_a2f[0..3] (4 × uint64)
_SHM_FLAGS_F2A = 32   # offset 32: flag_f2a[0..3] (4 × uint64)
_SHM_SIZES_A2F = 64   # offset 64: size_a2f[0..3] (4 × uint64)
_SHM_SIZES_F2A = 96   # offset 96: size_f2a[0..3] (4 × uint64)
_SHM_TOTAL = 128

# Legacy constants kept for reference
_SHM_FLAG_A2F = 0
_SHM_FLAG_F2A = 8
_SHM_SIZE_A2F = 16
_SHM_SIZE_F2A = 24

# Dtype encoding for SHM metadata (stream-ordered path)
_DTYPE_TO_INT = {
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.float32: 3,
    torch.float64: 4,
    torch.int32: 5,
    torch.int64: 6,
    torch.int8: 7,
    torch.uint8: 8,
}
_INT_TO_DTYPE = {v: k for k, v in _DTYPE_TO_INT.items()}


class IpcTensorCommunicator:
    """CUDA IPC communicator for single-node A↔F over NVLink.

    Uses cross-device cudaMemcpyPeer (not cached loads) for data transfer
    to avoid GPU L2 cache coherence issues with IPC-mapped peer memory.

    Handles N:M heterogeneous TP: each rank pair (DA rank i ↔ DF rank i)
    establishes its own IPC channel.

    Per-slot SHM flags enable M>1 pipelining: sender can post multiple
    messages before receiver drains them (up to RING_SIZE-1 ahead).
    """

    HEADER_BYTES = 64  # 8 × int64 for metadata
    RING_SIZE = 4      # supports M ≤ 3 with one slot headroom

    def __init__(self, perspective: AFDPerspective, mb_id: Optional[int] = None):
        self.is_ffn = perspective == AFDPerspective.AFD_PERSPECTIVE_FFN
        tag = "FFN" if self.is_ffn else "ATTN"
        # mb_id distinguishes per-micro-batch communicator instances when
        # multiple AFDCommunicators coexist in the same process (used by
        # --afd-async-schedule).  None means "legacy single comm" — its
        # SHM/socket paths use no _mb suffix to remain backward-compatible
        # with existing IPC test fixtures.
        self.mb_id: Optional[int] = mb_id

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        self._local_device = torch.device(f"cuda:{torch.cuda.current_device()}")

        # Determine peer device. For 2-GPU AFD, peer = 1 - current.
        # For TP>1 with all GPUs visible, peer is computed from a signed
        # offset: peer = local + AFD_IPC_PEER_OFFSET.  E.g. with ATTN
        # base_gpu_id=0 and FFN base_gpu_id=2 (TP=2 each), use offset=+2
        # on the ATTN side and -2 on the FFN side so each TP rank pairs
        # with the matching peer rank.
        # AFD_IPC_PEER_DEVICE (absolute index) is still honoured for
        # backwards compat with TP=1 PD+AF setups.
        _peer_offset = os.environ.get("AFD_IPC_PEER_OFFSET")
        _peer_env = os.environ.get("AFD_IPC_PEER_DEVICE")
        if _peer_offset is not None:
            self._peer_device_idx = (
                torch.cuda.current_device() + int(_peer_offset)
            )
        elif _peer_env is not None:
            self._peer_device_idx = int(_peer_env)
        else:
            self._peer_device_idx = (
                1 if torch.cuda.current_device() == 0 else 0
            )
        self._peer_device = torch.device(f"cuda:{self._peer_device_idx}")
        logger.info(
            "[IPC] local=%s peer=%s (peer_idx=%d)",
            self._local_device, self._peer_device, self._peer_device_idx,
        )

        if torch.cuda.can_device_access_peer(
            torch.cuda.current_device(), self._peer_device_idx
        ):
            try:
                # Enable peer access via CUDA runtime for NVLink direct transfers
                import ctypes
                cuda_rt = ctypes.CDLL("libcudart.so")
                # cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags)
                ret = cuda_rt.cudaDeviceEnablePeerAccess(
                    ctypes.c_int(self._peer_device_idx), ctypes.c_uint(0)
                )
                if ret == 0:
                    logger.info(
                        "[IPC] peer access %d->%d enabled (NVLink direct)",
                        torch.cuda.current_device(), self._peer_device_idx,
                    )
                elif ret == 704:  # cudaErrorPeerAccessAlreadyEnabled
                    logger.info(
                        "[IPC] peer access %d->%d already enabled",
                        torch.cuda.current_device(), self._peer_device_idx,
                    )
                else:
                    logger.warning(
                        "[IPC] cudaDeviceEnablePeerAccess returned %d", ret
                    )
            except Exception as e:
                logger.warning("[IPC] failed to enable peer access: %s", e)
        else:
            logger.info(
                "[IPC] peer access %d->%d not available, cross-device copy may be slow",
                torch.cuda.current_device(),
                self._peer_device_idx,
            )

        # Max message size: 256 MB handles prefill
        self._max_msg_size = 256 * 1024 * 1024

        # One contiguous send pool shared via IPC, sliced into per-slot views.
        # This fixes the bug where only slot 0 was exported — the receiver
        # always read from slot 0 even when the sender wrote to slot 1/2/3.
        self._send_pool = torch.empty(
            self.RING_SIZE * self._max_msg_size,
            dtype=torch.uint8, device=self._local_device,
        )
        self._send_buf = [
            self._send_pool[i * self._max_msg_size : (i + 1) * self._max_msg_size]
            for i in range(self.RING_SIZE)
        ]
        self._send_event = [torch.cuda.Event() for _ in range(self.RING_SIZE)]

        # Stream-ordered IPC events (for potential future cross-process sync)
        self._send_ipc_event = [
            torch.cuda.Event() for _ in range(self.RING_SIZE)
        ]

        # Recv buffers are local only (not shared).
        self._recv_buf = []
        self._recv_event = []
        for _ in range(self.RING_SIZE):
            self._recv_buf.append(
                torch.empty(self._max_msg_size, dtype=torch.uint8, device=self._local_device)
            )
            self._recv_event.append(torch.cuda.Event())

        # Ring buffer indices
        self._send_slot: int = 0
        self._recv_slot: int = 0
        self._recv_lock = threading.Lock()  # serializes recv_poll slot assignment

        # Export the entire send pool (all slots) via IPC
        self._send_info = self._send_pool.untyped_storage()._share_cuda_()
        logger.info(
            "[IPC] mb=%s tag=%s send_pool addr=0x%x dev=%s",
            mb_id, tag, self._send_pool.data_ptr(), self._local_device,
        )

        # Resolve rank to avoid SHM/socket collisions
        rank = 0
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                rank = dist.get_rank()
        except Exception:
            pass
        if rank == 0:
            sched_port = int(os.environ.get("AFD_SCHED_PORT", "0"))
            if sched_port > 0:
                rank = sched_port % 1000  # decode 65400→400, prefill 65300→300
        self._rank = rank

        # Setup SHM flags + msg_size (per-slot layout)
        self._setup_shm(rank)
        # For AsyncTensorCommunicator.send_async compatibility
        self._last_send_future = None

        # Pre-create a CUDA stream for bg thread use. New Python threads
        # pay 5-25ms CUDA context initialization on first GPU op; reusing
        # a stream from the main thread avoids this.
        self._bg_stream = torch.cuda.Stream(device=self._local_device)

        # Persistent background worker thread for nonblocking sends.
        # Spawning a new thread per send (as AsyncTensorCommunicator does)
        # incurs CUDA context init (5-25ms) on each new thread's first
        # GPU op. A single persistent thread pays this once and reuses it.
        # AFD_IPC_SYNC_SEND=1 disables the worker (forces comm_stream path).
        self._send_queue = queue.Queue()
        self._send_worker_running = True
        self._send_worker_thread = threading.Thread(
            target=self._send_worker, daemon=True,
            name=f"ipc-send-{tag.lower()}",
        )
        self._send_worker_thread.start()
        # v4 fix: Do NOT expose send_tensor_nonblocking or send_tensor_stream.
        # IPC uses the simple comm_stream path in AsyncTensorCommunicator:
        #   comm_stream.wait_event(compute_event)  ← GPU-level dependency
        #   self.inner.send_tensor(x)              ← copy + sync + flag on comm_stream
        # Main thread cost: ~105μs (copy + event sync + flag write).
        # This is acceptable given 400-550μs GPU compute per layer.
        self._send_worker_running = False
        self._send_queue.put(None)  # wake worker to exit
        self._send_worker_thread.join(timeout=5)
        logger.info("[IPC %s] Using comm_stream path (send_tensor)", tag)

        # Lazy handshake
        self._ready = threading.Event()
        self._peer_send_buf = None
        self._exchange_thread = threading.Thread(
            target=self._exchange_background,
            args=(tag, rank),
            daemon=True,
        )
        self._exchange_thread.start()

        logger.info(
            "[IPC %s] rank=%d mb=%s local=%s peer=%s RING_SIZE=%d, "
            "handshake started (background)",
            tag, rank, self.mb_id, self._local_device, self._peer_device,
            self.RING_SIZE,
        )

    def _setup_shm(self, rank: int):
        """Create POSIX shared memory for per-slot flag + msg_size sync.

        First process uses O_CREAT|O_EXCL to create the SHM; second process
        just opens the existing mapping. Both sides zero the SHM to prevent
        stale flags from a previous run.
        """
        suffix = f"_{rank}" if self.mb_id is None else f"_{rank}_mb{self.mb_id}"
        self._shm_path = f"/dev/shm/afd_ipc_flags{suffix}"
        try:
            fd = os.open(self._shm_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            os.ftruncate(fd, _SHM_TOTAL)
        except FileExistsError:
            fd = os.open(self._shm_path, os.O_RDWR)
        self._shm = mmap.mmap(
            fd, _SHM_TOTAL, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE
        )
        os.close(fd)
        # Always zero — handles stale data from previous runs.
        self._shm[:_SHM_TOTAL] = b'\x00' * _SHM_TOTAL

        # Register SHM as CUDA-mapped host memory so GPU can write flags directly.
        import ctypes
        self._shm_buf = (ctypes.c_char * _SHM_TOTAL).from_buffer(self._shm)
        self._shm_host_ptr = ctypes.addressof(self._shm_buf)
        self._libcudart = ctypes.CDLL('libcudart.so')
        cudaHostRegisterMapped = 2
        cudaHostRegisterPortable = 1
        ret = self._libcudart.cudaHostRegister(
            ctypes.c_void_p(self._shm_host_ptr),
            ctypes.c_size_t(_SHM_TOTAL),
            ctypes.c_uint(cudaHostRegisterMapped | cudaHostRegisterPortable)
        )
        if ret != 0:
            logger.warning("[IPC] cudaHostRegister failed (ret=%d), GPU flag write disabled", ret)
            self._shm_dev_ptr = None
        else:
            dev_ptr = ctypes.c_void_p()
            ret = self._libcudart.cudaHostGetDevicePointer(
                ctypes.byref(dev_ptr),
                ctypes.c_void_p(self._shm_host_ptr),
                ctypes.c_uint(0)
            )
            if ret != 0:
                logger.warning("[IPC] cudaHostGetDevicePointer failed (ret=%d)", ret)
                self._shm_dev_ptr = None
            else:
                self._shm_dev_ptr = dev_ptr.value
                logger.info("[IPC] SHM registered for GPU flag write: host=0x%x dev=0x%x",
                           self._shm_host_ptr, self._shm_dev_ptr)

    def _exchange_background(self, tag: str, rank: int):
        """Exchange IPC handles and import peer buffer."""
        try:
            torch.cuda.set_device(self._local_device.index)
            logger.info("[IPC %s] rank=%d mb=%s exchanging handles...",
                        tag, rank, self.mb_id)
            self._peer_send_info = self._exchange_handles(rank)

            # Fix device_id mismatch: the peer's device_id is local to their
            # process. We need to remap it to our peer_device_idx.
            # _share_cuda_() returns (device_id, handle, size, offset, ...)
            if isinstance(self._peer_send_info, tuple):
                peer_info_list = list(self._peer_send_info)
                peer_info_list[0] = self._peer_device_idx
                self._peer_send_info = tuple(peer_info_list)

            torch.cuda.set_device(self._peer_device_idx)
            peer_storage = torch.storage.UntypedStorage._new_shared_cuda(
                *self._peer_send_info
            )
            _peer_send_pool = torch.tensor(
                peer_storage, dtype=torch.uint8, device=self._peer_device
            ).reshape(-1)[:self.RING_SIZE * self._max_msg_size]
            self._peer_send_buf = [
                _peer_send_pool[i * self._max_msg_size : (i + 1) * self._max_msg_size]
                for i in range(self.RING_SIZE)
            ]
            self._peer_ipc_event = []  # Not used in current implementation
            torch.cuda.set_device(self._local_device.index)

            # Initialize P2P signal flags for GPU-only communication
            self.setup_p2p_signal(self._peer_device_idx)

            self._ready.set()
            logger.info(
                "[IPC %s] rank=%d mb=%s handshake complete, "
                "peer_send_buf[0] device=%s addr=0x%x",
                tag, rank, self.mb_id, self._peer_send_buf[0].device,
                self._peer_send_buf[0].data_ptr(),
            )
        except Exception as e:
            logger.error("[IPC %s] rank=%d mb=%s handshake failed: %s",
                         tag, rank, self.mb_id, e)

    def _wait_ready(self):
        """Block until handshake with peer completes."""
        if not self._ready.is_set():
            logger.info("[IPC] rank=%d waiting for peer handshake...", self._rank)
            self._ready.wait()

    def _exchange_handles(self, rank: int):
        """Exchange IPC handles with peer via Unix socket.

        FFN listens, receives then sends. Attn connects, sends then receives.
        """
        suffix = f"_{rank}" if self.mb_id is None else f"_{rank}_mb{self.mb_id}"
        sock_path = f"/tmp/afd_ipc{suffix}.sock"

        if self.is_ffn:
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                os.unlink(sock_path)
            except FileNotFoundError:
                pass
            server.bind(sock_path)
            server.listen(1)
            conn, addr = server.accept()

            data_len = struct.unpack("!I", conn.recv(4))[0]
            data = conn.recv(data_len, socket.MSG_WAITALL)
            peer_info = pickle.loads(data)

            my_data = pickle.dumps(self._send_info)
            conn.send(struct.pack("!I", len(my_data)) + my_data)

            conn.close()
            server.close()
            os.unlink(sock_path)
            return peer_info

        else:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            connected = False
            deadline = time.time() + 300
            attempt = 0
            while time.time() < deadline:
                try:
                    sock.connect(sock_path)
                    connected = True
                    break
                except (ConnectionRefusedError, FileNotFoundError):
                    attempt += 1
                    if attempt % 100 == 0 and attempt > 0:
                        logger.info(
                            "[IPC] rank %d waiting for FFN socket... (%ds)",
                            rank, int(time.time() - deadline + 300),
                        )
                    time.sleep(0.05)

            if not connected:
                raise ConnectionError(
                    f"Attn rank {rank} cannot connect to FFN at {sock_path}"
                )

            my_data = pickle.dumps(self._send_info)
            sock.send(struct.pack("!I", len(my_data)) + my_data)

            data_len = struct.unpack("!I", sock.recv(4))[0]
            data = sock.recv(data_len, socket.MSG_WAITALL)
            peer_info = pickle.loads(data)

            sock.close()
            return peer_info

    # ── Per-slot SHM helpers ──

    def _send_flag_off(self, slot: int) -> int:
        """Offset for this process's outbound flag for the given slot."""
        base = _SHM_FLAGS_F2A if self.is_ffn else _SHM_FLAGS_A2F
        return base + slot * 8

    def _send_size_off(self, slot: int) -> int:
        """Offset for this process's outbound msg size for the given slot."""
        base = _SHM_SIZES_F2A if self.is_ffn else _SHM_SIZES_A2F
        return base + slot * 8

    def _recv_flag_off(self, slot: int) -> int:
        """Offset for this process's inbound flag for the given slot."""
        base = _SHM_FLAGS_A2F if self.is_ffn else _SHM_FLAGS_F2A
        return base + slot * 8

    def _recv_size_off(self, slot: int) -> int:
        """Offset for this process's inbound msg size for the given slot."""
        base = _SHM_SIZES_A2F if self.is_ffn else _SHM_SIZES_F2A
        return base + slot * 8

    def _read_u64(self, offset: int) -> int:
        return struct.unpack_from("Q", self._shm, offset)[0]

    def _write_u64(self, offset: int, value: int):
        struct.pack_into("Q", self._shm, offset, value)

    def _send_worker(self):
        """Persistent background thread for nonblocking sends.

        Initializes CUDA once then loops, consuming send requests from the
        queue. Each request's compute_event was already synchronized by the
        temporary bg thread in AsyncTensorCommunicator.send_async, so x is
        ready to copy when we pop it.
        """
        torch.cuda.set_device(self._local_device.index)
        while True:
            x = self._send_queue.get()
            if x is None:  # shutdown sentinel
                break
            try:
                self._send_tensor_impl(x, self._bg_stream)
            except Exception:
                logger.exception("[IPC send_worker] send failed")

    def _flag_writer_loop(self):
        """Unused — kept for potential future send_tensor_stream optimization."""
        pass

    # ── Public interface ──

    def send_tensor(self, x: torch.Tensor):
        """Synchronous send called from main thread (Path 2/3).

        Delegates to _send_tensor_impl after ensuring the handshake is complete.
        When called inside `with torch.cuda.stream(comm_stream):` from
        AsyncTensorCommunicator.send_async(), the copy runs on comm_stream
        with GPU-level wait_event ordering. Cost: ~105μs (copy + event sync + flag write).
        """
        self._wait_ready()
        self._send_tensor_impl(x)  # uses current stream (Path 2/3)

    def _send_tensor_impl(self, x: torch.Tensor, stream=None):
        """Core send logic: copy to send_buf, signal peer via SHM flag.

        When called with stream=self._bg_stream (Path 1, bg thread), avoids
        the 5-25ms CUDA context initialization penalty that new Python threads
        pay on their first GPU operation.

        When called without stream (Path 2/3, main thread), uses the current
        CUDA stream.
        """
        # Ensure correct CUDA device in bg thread context (Path 1).
        # Python threads default to device 0, but the communicator's
        # send_buf may be on a different device (e.g. device 1 for FFN).
        torch.cuda.set_device(self._local_device.index)

        t_impl_start = time.perf_counter()
        x_cont = x.contiguous() if not x.is_contiguous() else x
        t_contig = time.perf_counter()
        data_bytes = x_cont.numel() * x_cont.element_size()
        total_bytes = self.HEADER_BYTES + data_bytes

        if total_bytes > self._max_msg_size:
            raise RuntimeError(
                f"Message too large: {total_bytes} > {self._max_msg_size}"
            )

        slot = self._send_slot
        flag_off = self._send_flag_off(slot)
        size_off = self._send_size_off(slot)

        # Wait for peer to have consumed this slot
        t_wait0 = time.perf_counter()
        while self._read_u64(flag_off) != 0:
            pass
        t_wait1 = time.perf_counter()

        send_buf = self._send_buf[slot]
        meta_np = _encode_meta(x_cont)

        # Write header + data to send_buf directly on the current stream.
        # Then synchronize to ensure data is written before signaling peer.
        meta_np_view = meta_np.view(np.uint8)
        send_buf[:64].copy_(
            torch.from_numpy(meta_np_view).to(self._local_device)
        )
        send_buf[64:total_bytes].copy_(x_cont.view(torch.uint8).flatten())

        # Record event and synchronize to ensure copy is complete
        self._send_event[slot].record()
        self._send_event[slot].synchronize()
        t_sync = time.perf_counter()

        # Write flag from CPU (peer polls this)
        self._write_u64(size_off, total_bytes)
        self._write_u64(flag_off, 1)
        t_flag_write = time.perf_counter()

        wait_us = (t_wait1 - t_wait0) * 1e6
        sync_us = (t_sync - t_wait1) * 1e6

        # Detailed profiling
        try:
            from sglang.srt.layers.afd_mixin import _afd_host_events
            _afd_host_events.append({
                "ts_ms": round(t_impl_start * 1000, 3),
                "role": "IPC_INNER", "layer": -1, "mb": -1,
                "event": "send_breakdown",
                "contiguous_us": round((t_contig - t_impl_start) * 1e6, 1),
                "slot_wait_us": round(wait_us, 1),
                "gpu_copy_us": round((t_sync - t_wait1) * 1e6 - wait_us if wait_us < 1 else (t_sync - t_wait1) * 1e6, 1),
                "event_sync_us": round((t_sync - t_wait1) * 1e6, 1),
                "flag_write_us": round((t_flag_write - t_sync) * 1e6, 1),
                "total_us": round((t_flag_write - t_impl_start) * 1e6, 1),
                "data_bytes": data_bytes,
            })
        except Exception:
            pass

        if wait_us > 500 or sync_us > 500:
            logger.info(
                "[IPC send_detail] rank=%d mb=%s slot=%d wait=%.0fus "
                "copy_sync=%.0fus bytes=%d",
                self._rank, self.mb_id, slot, wait_us, sync_us, data_bytes,
            )

        self._send_slot = (slot + 1) % self.RING_SIZE

    def send_stream_ordered(self, x: torch.Tensor):
        """Optimized send: skip event.synchronize() on send side.

        Instead of sync-ing on CPU to confirm copy is done before setting flag,
        we record an event and set the flag immediately. The recv side will
        do a stream.synchronize() after its copy to ensure data integrity.

        Uses per-(shape, dtype) caching to eliminate metadata encoding overhead
        on repeated calls with the same tensor shape (common in decode).
        """
        self._wait_ready()
        x_cont = x.contiguous() if not x.is_contiguous() else x

        # --- Fast-path cache lookup ---
        cache_key = (x_cont.shape, x_cont.dtype)
        if not hasattr(self, '_send_cache'):
            self._send_cache = {}
        cache = self._send_cache.get(cache_key)
        if cache is None:
            # First call with this shape: pre-compute everything
            data_bytes = x_cont.numel() * x_cont.element_size()
            total_bytes = self.HEADER_BYTES + data_bytes
            if total_bytes > self._max_msg_size:
                raise RuntimeError(f"Message too large: {total_bytes} > {self._max_msg_size}")
            meta_np = _encode_meta(x_cont)
            meta_gpu = torch.from_numpy(meta_np.view(np.uint8)).to(
                self._local_device
            ).clone()  # persistent GPU copy
            cache = (total_bytes, meta_gpu)
            self._send_cache[cache_key] = cache
        total_bytes, meta_gpu = cache

        slot = self._send_slot
        send_buf = self._send_buf[slot]
        flag_off = self._send_flag_off(slot)
        size_off = self._send_size_off(slot)

        # Wait for peer to have consumed this slot
        while self._read_u64(flag_off) != 0:
            pass

        # HOT PATH: 2 GPU copies only, no Python allocations
        send_buf[:64].copy_(meta_gpu)  # pre-computed metadata (GPU-to-GPU, ~5us)
        send_buf[64:total_bytes].copy_(x_cont.view(torch.uint8).flatten())

        # Sync current stream to ensure GPU copy is visible before setting flag
        torch.cuda.current_stream().synchronize()

        # Write size + flag to SHM (CPU, after GPU copy confirmed done)
        self._write_u64(size_off, total_bytes)
        self._write_u64(flag_off, 1)

        self._send_slot = (slot + 1) % self.RING_SIZE

    def send_gpu_signal(self, x: torch.Tensor):
        """GPU-only send: data copy + __threadfence_system + GPU writes SHM flag.

        CPU never blocks. GPU ensures data visible before writing flag.
        Uses cudaMemcpyAsync to write flag to mapped SHM (same as send_zero_sync
        but with __threadfence_system for correctness).
        """
        from sglang.srt.layers.p2p_signal import signal

        self._wait_ready()
        x_cont = x.contiguous() if not x.is_contiguous() else x

        # --- Fast-path cache lookup ---
        cache_key = (x_cont.shape, x_cont.dtype)
        if not hasattr(self, '_send_cache'):
            self._send_cache = {}
        cache = self._send_cache.get(cache_key)
        if cache is None:
            data_bytes = x_cont.numel() * x_cont.element_size()
            total_bytes = self.HEADER_BYTES + data_bytes
            if total_bytes > self._max_msg_size:
                raise RuntimeError(f"Message too large: {total_bytes} > {self._max_msg_size}")
            meta_np = _encode_meta(x_cont)
            meta_gpu = torch.from_numpy(meta_np.view(np.uint8)).to(
                self._local_device
            ).clone()
            # Pre-allocate flag tensor: [total_bytes, 1] for cudaMemcpyAsync
            flag_gpu = torch.tensor(
                [total_bytes, 1], dtype=torch.int64, device=self._local_device
            )
            cache = (total_bytes, meta_gpu, flag_gpu)
            self._send_cache[cache_key] = cache
        total_bytes, meta_gpu, flag_gpu = cache

        slot = self._send_slot
        send_buf = self._send_buf[slot]
        flag_off = self._send_flag_off(slot)
        size_off = self._send_size_off(slot)

        # Wait for peer to have consumed this slot (CPU poll SHM)
        while self._read_u64(flag_off) != 0:
            pass

        # Copy data to send_buf on current stream (ordered after Attn/MLP kernel)
        send_buf[:64].copy_(meta_gpu)
        send_buf[64:total_bytes].copy_(x_cont.view(torch.uint8).flatten())

        # GPU writes flag to mapped SHM via cudaMemcpyAsync
        # __threadfence_system is implicit in cudaMemcpyAsync (DtoH ensures ordering)
        import ctypes
        dst_addr = self._shm_dev_ptr + size_off
        self._libcudart.cudaMemcpyAsync(
            ctypes.c_void_p(dst_addr),
            ctypes.c_void_p(flag_gpu.data_ptr()),
            ctypes.c_size_t(16),  # 2 x int64 = [size, flag]
            ctypes.c_int(1),  # cudaMemcpyDeviceToHost (mapped host)
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        )
        # CPU returns immediately! No synchronize!
        self._send_slot = (slot + 1) % self.RING_SIZE

    def recv_gpu_wait(self) -> torch.Tensor:
        """Hybrid recv: CPU polls SHM flag, then enqueue P2P copy without sync.

        Send side uses cudaMemcpyAsync to write flag to mapped SHM (no CPU sync).
        Recv side:
        1. CPU polls SHM flag (fast, ~77us once sender's GPU writes it)
        2. Enqueue P2P copy from peer's send_buf (no synchronize!)
        3. Return view — downstream kernel on same stream auto-waits for copy
        4. Clear SHM flag so sender can reuse slot

        Key insight: once CPU sees flag=1 in SHM, sender's GPU has completed
        both data copy AND cudaMemcpyAsync (same stream ordering). So P2P copy
        from peer's send_buf is safe without additional sync.
        """
        self._wait_ready()

        with self._recv_lock:
            slot = self._recv_slot
            self._recv_slot = (slot + 1) % self.RING_SIZE

        flag_off = self._recv_flag_off(slot)
        size_off = self._recv_size_off(slot)

        # CPU polls SHM flag (set by sender's GPU via cudaMemcpyAsync)
        while self._read_u64(flag_off) != 1:
            pass

        total_bytes = self._read_u64(size_off)
        recv_buf = self._recv_buf[slot]

        # --- Fast-path: use cached shape ---
        if not hasattr(self, '_recv_cache'):
            self._recv_cache = {}
        cache = self._recv_cache.get(total_bytes)
        if cache is not None:
            shape, dtype, data_offset = cache
            # Enqueue P2P copy — NO synchronize!
            # Downstream kernel on same stream will wait for copy to finish
            recv_buf[:total_bytes].copy_(self._peer_send_buf[slot][:total_bytes])
            # Clear SHM flag (CPU write, safe — GPU already finished writing data)
            self._write_u64(flag_off, 0)
            # Return view — no clone needed (ring buffer protects data)
            return recv_buf[data_offset:total_bytes].view(dtype).reshape(shape)

        # COLD PATH: first recv — decode metadata (needs sync for .cpu())
        recv_buf[:total_bytes].copy_(self._peer_send_buf[slot][:total_bytes])
        torch.cuda.current_stream().synchronize()
        header_np = recv_buf[:64].cpu().numpy().view(np.int64)
        shape, dtype, _ = _decode_meta(header_np)
        self._recv_cache[total_bytes] = (shape, dtype, 64)
        self._write_u64(flag_off, 0)
        return recv_buf[64:total_bytes].view(dtype).reshape(shape).contiguous()

    def setup_p2p_signal(self, peer_device: int):
        """Initialize P2P signal flags for GPU-only communication.

        Uses the LAST 8 bytes of each send_buf slot as the signal flag.
        Since peer already has IPC access to our send_buf (via _peer_send_buf),
        both sides can see the same flag without additional IPC setup.

        Convention:
        - send_buf layout: [64B header | data | ... | 8B flag at offset max_msg_size-8]
        - Sender writes flag=1 after data copy + __threadfence_system()
        - Receiver polls peer_send_buf[slot][flag_offset] via P2P volatile read

        Args:
            peer_device: the peer GPU device index (for logging only)
        """
        self._p2p_flag_offset = self._max_msg_size - 8
        # Pre-create flag value tensors on local device
        self._p2p_flag_one = torch.ones(1, dtype=torch.int64, device=self._local_device)
        self._p2p_flag_zero = torch.zeros(1, dtype=torch.int64, device=self._local_device)
        self._p2p_total_bytes = None
        # Clear flags in all send_buf slots
        for slot in range(self.RING_SIZE):
            self._send_buf[slot][self._p2p_flag_offset:self._p2p_flag_offset+8].zero_()
        logger.info("[IPC] P2P signal initialized: flag_offset=%d, peer=cuda:%d",
                   self._p2p_flag_offset, peer_device)

    def recv_zero_sync(self) -> torch.Tensor:
        """Zero-sync recv: no synchronize(), no clone().

        Polls SHM flag (set by sender's GPU via cudaMemcpyAsync).
        When flag=1, sender's data is guaranteed complete (same-stream ordering).
        Enqueues P2P copy on current stream — downstream kernel (MLP/Attn)
        is on same stream, so GPU auto-orders: copy finishes before kernel starts.
        Returns a view of recv_buf (no clone needed — ring buffer slot won't be
        reused until next layer's recv, by which time MLP has consumed the data).

        Requires: pre-launch cache populated (first call goes through cold path).
        """
        self._wait_ready()

        with self._recv_lock:
            slot = self._recv_slot
            self._recv_slot = (slot + 1) % self.RING_SIZE

        flag_off = self._recv_flag_off(slot)
        size_off = self._recv_size_off(slot)

        # Poll until sender's GPU writes flag=1 to mapped SHM
        while self._read_u64(flag_off) != 1:
            pass

        total_bytes = self._read_u64(size_off)
        recv_buf = self._recv_buf[slot]

        # --- Fast-path: use cached shape ---
        if not hasattr(self, '_recv_cache'):
            self._recv_cache = {}
        cache = self._recv_cache.get(total_bytes)
        if cache is not None:
            shape, dtype, data_offset = cache
            # Enqueue P2P copy — NO synchronize!
            # Downstream kernel on same stream will wait for copy to finish
            recv_buf[:total_bytes].copy_(self._peer_send_buf[slot][:total_bytes])
            # Return view — no clone needed (ring buffer protects data)
            result = recv_buf[data_offset:total_bytes].view(dtype).reshape(shape)
            # Clear flag so sender can reuse slot
            self._write_u64(flag_off, 0)
            return result

        # COLD PATH: first recv — decode metadata (needs sync)
        recv_buf[:total_bytes].copy_(self._peer_send_buf[slot][:total_bytes])
        torch.cuda.current_stream().synchronize()
        header_np = recv_buf[:64].cpu().numpy().view(np.int64)
        shape, dtype, _ = _decode_meta(header_np)
        self._recv_cache[total_bytes] = (shape, dtype, 64)
        result = recv_buf[64:total_bytes].view(dtype).reshape(shape).contiguous()
        self._write_u64(flag_off, 0)
        return result
        """Optimized recv: no send-side sync needed.

        The send side skips event.synchronize() and sets the flag immediately
        after recording the event. We poll the flag, then do cudaMemcpyPeer
        which implicitly waits for the peer's copy to complete (CUDA guarantees
        ordering on the same device's memory).

        Uses shape caching: after first recv, assumes same shape on subsequent
        calls (common in decode). Skips metadata decode + stream.synchronize()
        on hot path.
        """
        self._wait_ready()

        with self._recv_lock:
            slot = self._recv_slot
            self._recv_slot = (slot + 1) % self.RING_SIZE

        flag_off = self._recv_flag_off(slot)
        size_off = self._recv_size_off(slot)

        # Poll until data ready (sender sets flag after recording event)
        while self._read_u64(flag_off) != 1:
            pass

        total_bytes = self._read_u64(size_off)
        recv_buf = self._recv_buf[slot]

        # --- Fast-path: use cached shape if total_bytes matches ---
        if not hasattr(self, '_recv_cache'):
            self._recv_cache = {}
        cache = self._recv_cache.get(total_bytes)
        if cache is not None:
            shape, dtype, data_offset = cache
            # HOT PATH: single GPU copy, no metadata decode
            recv_buf[:total_bytes].copy_(self._peer_send_buf[slot][:total_bytes])
            torch.cuda.current_stream().synchronize()
            result = recv_buf[data_offset:total_bytes].view(dtype).reshape(shape).clone()
            self._write_u64(flag_off, 0)
            return result

        # COLD PATH: first time seeing this size — decode metadata
        recv_buf[:total_bytes].copy_(self._peer_send_buf[slot][:total_bytes])
        torch.cuda.current_stream().synchronize()
        header_np = recv_buf[:64].cpu().numpy().view(np.int64)
        shape, dtype, _ = _decode_meta(header_np)
        # Cache for future calls
        self._recv_cache[total_bytes] = (shape, dtype, 64)
        result = recv_buf[64:total_bytes].view(dtype).reshape(shape).contiguous()
        self._write_u64(flag_off, 0)
        return result

    def recv_tensor(self) -> torch.Tensor:
        """Synchronous recv: poll flag, single peer copy, decode, return.

        Reads msg_size from SHM first (CPU memory — no GPU sync needed),
        then copies header+data in ONE cudaMemcpyPeer.

        Atomically claims a recv slot under lock so that concurrent
        recv_poll (bg thread) cannot steal the same slot while we
        are inside GPU ops (which release the GIL).
        """
        self._wait_ready()

        # Atomically claim the slot so concurrent recv_poll bg threads
        # get a distinct slot even when GPU ops ahead release the GIL.
        with self._recv_lock:
            slot = self._recv_slot
            self._recv_slot = (slot + 1) % self.RING_SIZE

        flag_off = self._recv_flag_off(slot)
        size_off = self._recv_size_off(slot)

        # Poll until data ready
        t_flag0 = time.perf_counter()
        while self._read_u64(flag_off) != 1:
            pass
        t_flag1 = time.perf_counter()

        # Read total message size from SHM (CPU memory — no sync needed)
        total_bytes = self._read_u64(size_off)

        recv_buf = self._recv_buf[slot]

        # Single cross-device cudaMemcpyPeer for header+data
        # Copy from the correct per-slot peer buffer (not just slot 0)
        recv_buf[:total_bytes].copy_(self._peer_send_buf[slot][:total_bytes])

        self._recv_event[slot].record()
        self._recv_event[slot].synchronize()

        # Ack AFTER GPU copy — sender must not overwrite peer buffer
        # while we are reading from it.
        self._write_u64(flag_off, 0)
        t_copy_done = time.perf_counter()

        # Decode metadata from local recv buffer (safe path: CPU copy first)
        header_np = recv_buf[:64].cpu().numpy().view(np.int64)
        shape, dtype, original_num_tokens = _decode_meta(header_np)

        if len(shape) == 0:
            logger.error(
                "[IPC recv] CORRUPTED METADATA: rank=%d mb=%s slot=%d "
                "total_bytes=%d header_np=%s recv_buf[:8]=%s "
                "peer_send_buf[slot] addr=0x%x",
                self._rank, self.mb_id, slot, total_bytes,
                header_np.tolist(),
                recv_buf[:8].cpu().numpy().tolist(),
                self._peer_send_buf[slot].data_ptr(),
            )

        # Clone to decouple from recv_buf (caller may hold reference across layers)
        data_bytes = total_bytes - self.HEADER_BYTES
        result = recv_buf[64:total_bytes].view(dtype).reshape(shape).contiguous()
        t_decode_done = time.perf_counter()

        flag_wait_us = (t_flag1 - t_flag0) * 1e6
        copy_sync_us = (t_copy_done - t_flag1) * 1e6

        # Detailed profiling
        try:
            from sglang.srt.layers.afd_mixin import _afd_host_events
            _afd_host_events.append({
                "ts_ms": round(t_flag0 * 1000, 3),
                "role": "IPC_INNER", "layer": -1, "mb": -1,
                "event": "recv_breakdown",
                "flag_poll_us": round(flag_wait_us, 1),
                "memcpy_peer_sync_us": round(copy_sync_us, 1),
                "decode_us": round((t_decode_done - t_copy_done) * 1e6, 1),
                "total_us": round((t_decode_done - t_flag0) * 1e6, 1),
                "data_bytes": int(data_bytes),
            })
        except Exception:
            pass

        if flag_wait_us > 1000 or copy_sync_us > 2000:
            logger.info(
                "[IPC recv] rank=%d flag_wait=%.0fus copy_sync=%.0fus bytes=%d",
                self._rank, flag_wait_us, copy_sync_us, data_bytes,
            )

        return result

    # --- Async pipeline stream-ordered API ---

    def send_on_stream(self, x: torch.Tensor, stream: torch.cuda.Stream) -> torch.cuda.Event:
        """Stream-ordered send for async pipeline.

        Enqueues data copy on the given stream, records a completion event,
        then sets SHM flag after stream sync. Returns the event that fires
        when the send buffer is populated and visible to the peer.

        Unlike send_stream_ordered (which uses current stream), this allows
        the caller to specify which stream to use, enabling multi-stream
        pipeline overlap.
        """
        self._wait_ready()
        x_cont = x.contiguous() if not x.is_contiguous() else x

        cache_key = (x_cont.shape, x_cont.dtype)
        if not hasattr(self, '_send_cache'):
            self._send_cache = {}
        cache = self._send_cache.get(cache_key)
        if cache is None:
            data_bytes = x_cont.numel() * x_cont.element_size()
            total_bytes = self.HEADER_BYTES + data_bytes
            if total_bytes > self._max_msg_size:
                raise RuntimeError(
                    f"send_on_stream: msg too large: {total_bytes} > {self._max_msg_size}"
                )
            meta_np = _encode_meta(x_cont)
            meta_gpu = torch.from_numpy(meta_np.view(np.uint8)).to(
                self._local_device
            ).clone()
            cache = (total_bytes, meta_gpu)
            self._send_cache[cache_key] = cache
        total_bytes, meta_gpu = cache

        slot = self._send_slot
        send_buf = self._send_buf[slot]
        flag_off = self._send_flag_off(slot)
        size_off = self._send_size_off(slot)

        while self._read_u64(flag_off) != 0:
            pass

        with torch.cuda.stream(stream):
            send_buf[:64].copy_(meta_gpu)
            send_buf[64:total_bytes].copy_(x_cont.view(torch.uint8).flatten())

        ev = torch.cuda.Event()
        ev.record(stream)
        ev.synchronize()

        self._write_u64(size_off, total_bytes)
        self._write_u64(flag_off, 1)
        self._send_slot = (slot + 1) % self.RING_SIZE
        return ev

    def recv_poll_and_enqueue(
        self, stream: torch.cuda.Stream
    ) -> Tuple[torch.Tensor, torch.cuda.Event]:
        """Non-blocking recv for async pipeline.

        CPU polls SHM flag (fast spin, ~77us in ipc_event mode), then
        enqueues P2P copy on the given stream. Returns (tensor, event) where
        the event fires when data is ready for consumption on that stream.

        The caller should use stream.wait_event(event) on the compute stream
        before using the tensor, establishing a GPU-only dependency.
        """
        self._wait_ready()

        with self._recv_lock:
            slot = self._recv_slot
            self._recv_slot = (slot + 1) % self.RING_SIZE

        flag_off = self._recv_flag_off(slot)
        size_off = self._recv_size_off(slot)

        while self._read_u64(flag_off) != 1:
            pass

        total_bytes = self._read_u64(size_off)
        recv_buf = self._recv_buf[slot]

        if not hasattr(self, '_recv_cache'):
            self._recv_cache = {}
        cache = self._recv_cache.get(total_bytes)

        with torch.cuda.stream(stream):
            recv_buf[:total_bytes].copy_(self._peer_send_buf[slot][:total_bytes])

        if cache is not None:
            shape, dtype, data_offset = cache
            result = recv_buf[data_offset:total_bytes].view(dtype).reshape(shape)
        else:
            ev_tmp = torch.cuda.Event()
            ev_tmp.record(stream)
            ev_tmp.synchronize()
            header_np = recv_buf[:64].cpu().numpy().view(np.int64)
            shape, dtype, _ = _decode_meta(header_np)
            self._recv_cache[total_bytes] = (shape, dtype, 64)
            result = recv_buf[64:total_bytes].view(dtype).reshape(shape)

        ev = torch.cuda.Event()
        ev.record(stream)
        self._write_u64(flag_off, 0)
        return result, ev

    def fence(self):
        """Wait for all pending nonblocking sends to complete.

        Blocks until the persistent send worker drains its queue, then
        waits for the final GPU operations to complete on this device.
        """
        # Drain the worker queue
        while self._send_queue.qsize() > 0 and self._send_worker_thread.is_alive():
            time.sleep(0.0001)  # 100us backoff
        # Final GPU sync: ensure all operations on this device are done
        if self._send_worker_running and self._send_worker_thread.is_alive():
            torch.cuda.synchronize(self._local_device)

    def close(self):
        """Cleanup IPC resources."""
        if hasattr(self, "_shm"):
            self._shm.close()
        if hasattr(self, "_shm_path"):
            try:
                os.unlink(self._shm_path)
            except (FileNotFoundError, PermissionError):
                pass
        if hasattr(self, "_send_queue") and self._send_worker_running:
            self._send_worker_running = False
            self._send_queue.put(None)  # shutdown sentinel
            if self._send_worker_thread.is_alive():
                self._send_worker_thread.join(timeout=5)
