"""High-performance C++ IPC communicator for AF disaggregation.

Drop-in replacement for IpcTensorCommunicator that eliminates Python overhead
in the communication hot path by delegating to a C++ library (libafd_ipc).

Key improvements over Python IPC:
1. Zero Python object allocation in hot path (no numpy, no struct.pack)
2. CUDA IPC Event for pure GPU synchronization (no CPU synchronize())
3. Pre-cached metadata: first send/recv caches shape/dtype, subsequent calls
   skip all metadata encoding/decoding
4. High-priority CUDA stream for communication

Expected performance:
- Python IPC: ~560us/layer (150us Python + 60us event.query + 34us P2P + 27us sync)
- C++ IPC (CPU_FLAG): ~100us/layer (34us P2P + 50us CPU flag + 16us overhead)
- C++ IPC (IPC_EVENT): ~50us/layer (34us P2P + 10us GPU event wait + 6us overhead)
"""

import os
import logging
import threading
import time
from typing import Optional

import torch

from sglang.srt.layers.afd_type import AFDPerspective

logger = logging.getLogger(__name__)


class CppIpcTensorCommunicator:
    """C++ IPC communicator — drop-in replacement for IpcTensorCommunicator.

    Uses the compiled afd_ipc_cpp extension for all hot-path operations.
    Falls back to Python IPC if the C++ extension is unavailable.

    Implements FifoTensorCommunicator interface (send_tensor / recv_tensor).

    Sync modes:
    - "ipc_event": CUDA IPC Event (recommended, ~50us/layer, pure GPU sync)
    - "gpu_signal": GPU-side signal/wait kernel (~30us/layer, but cross-process
                    volatile read has L2 coherence issues on some topologies)
    - "cpu_flag": CPU polls SHM flag (~100us/layer, most compatible)
    """

    RING_SIZE = 4

    def __init__(self, perspective: AFDPerspective, mb_id: Optional[int] = None):
        self.is_ffn = perspective == AFDPerspective.AFD_PERSPECTIVE_FFN
        self.mb_id = mb_id
        tag = "FFN" if self.is_ffn else "ATTN"

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        self._local_device = torch.cuda.current_device()

        # Determine peer device (same logic as Python IPC)
        _peer_offset = os.environ.get("AFD_IPC_PEER_OFFSET")
        _peer_env = os.environ.get("AFD_IPC_PEER_DEVICE")
        if _peer_offset is not None:
            self._peer_device = self._local_device + int(_peer_offset)
        elif _peer_env is not None:
            self._peer_device = int(_peer_env)
        else:
            self._peer_device = 1 if self._local_device == 0 else 0

        # Determine rank
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
                rank = sched_port % 1000
        self._rank = rank

        # Sync mode from environment (default: ipc_event)
        sync_mode = os.environ.get("AFD_IPC_SYNC_MODE", "ipc_event")

        # Load C++ extension
        from sglang.srt.layers.afd_ipc_cpp import get_module
        self._cpp_mod = get_module()

        # Create C++ communicator
        self._comm = self._cpp_mod.AfdIpcComm(
            self.is_ffn,
            self._local_device,
            self._peer_device,
            self._rank,
            mb_id if mb_id is not None else -1,
            sync_mode,
        )

        # Background handshake
        self._ready = threading.Event()
        self._exchange_thread = threading.Thread(
            target=self._handshake_background,
            daemon=True,
        )
        self._exchange_thread.start()

        # For AsyncTensorCommunicator compatibility
        self._last_send_future = None

        logger.info(
            "[CppIPC %s] rank=%d mb=%s local=cuda:%d peer=cuda:%d sync=%s",
            tag, rank, mb_id, self._local_device, self._peer_device, sync_mode,
        )

    def _handshake_background(self):
        """Run handshake in background thread."""
        try:
            self._comm.handshake()
            self._ready.set()
            logger.info("[CppIPC] handshake complete (rank=%d)", self._rank)
        except Exception as e:
            logger.error("[CppIPC] handshake failed: %s", e)

    def _wait_ready(self):
        """Block until handshake completes."""
        if not self._ready.is_set():
            self._ready.wait()

    def send_tensor(self, x: torch.Tensor):
        """Send tensor to peer. Hot path after first call."""
        self._wait_ready()
        if not hasattr(self, '_send_count'):
            self._send_count = 0
        self._send_count += 1
        if self._send_count <= 3:
            logger.info(
                "[CppIPC] send_tensor #%d: shape=%s dtype=%s device=%s bytes=%d",
                self._send_count, list(x.shape), x.dtype, x.device,
                x.numel() * x.element_size(),
            )
        self._comm.send_tensor(x)

    # Aliases for compatibility with existing code paths
    send_stream_ordered = send_tensor
    send_gpu_signal = send_tensor

    def recv_tensor(self) -> torch.Tensor:
        """Receive tensor from peer. Hot path after first call."""
        self._wait_ready()
        if not hasattr(self, '_recv_count'):
            self._recv_count = 0
        self._recv_count += 1
        result = self._comm.recv_tensor()
        if self._recv_count <= 3:
            logger.info(
                "[CppIPC] recv_tensor #%d: shape=%s dtype=%s device=%s",
                self._recv_count, list(result.shape), result.dtype, result.device,
            )
        return result

    # Aliases for compatibility
    recv_zero_sync = recv_tensor
    recv_gpu_wait = recv_tensor

    def send_tensor_gpu_only(self, x: torch.Tensor):
        """GPU-only send: no CPU blocking. Uses GPU signal kernel.

        CPU only enqueues CUDA ops and returns immediately.
        Requires gpu_signal mode or peer_signal_flags to be set up.
        """
        self._wait_ready()
        self._comm.send_tensor_gpu(x)

    def recv_tensor_gpu_only(self) -> torch.Tensor:
        """GPU-only recv: no CPU blocking. Uses GPU wait kernel.

        CPU only enqueues wait_kernel + memcpy and returns immediately.
        Requires prior recv_tensor() call to cache shape metadata.
        """
        self._wait_ready()
        return self._comm.recv_tensor_gpu()

    def reset_cache(self):
        """Reset metadata cache (call when tensor shape changes)."""
        self._comm.reset_cache()

    def fence(self):
        """Wait for all pending operations to complete."""
        torch.cuda.synchronize(self._local_device)

    def close(self):
        """Cleanup resources."""
        pass  # C++ destructor handles cleanup

    @property
    def sync_mode(self) -> str:
        return self._comm.sync_mode()


def create_ipc_communicator(
    perspective: AFDPerspective,
    mb_id: Optional[int] = None,
    use_cpp: Optional[bool] = None,
):
    """Factory function: create the best available IPC communicator.

    Args:
        perspective: ATTN or FFN side
        mb_id: microbatch ID
        use_cpp: Force C++ (True) or Python (False) backend.
                 None = auto-detect (prefer C++ if available).
    """
    if use_cpp is None:
        use_cpp = os.environ.get("AFD_IPC_CPP", "1") == "1"

    if use_cpp:
        try:
            return CppIpcTensorCommunicator(perspective, mb_id)
        except Exception as e:
            logger.warning(
                "[IPC] C++ backend unavailable (%s), falling back to Python", e
            )

    # Fallback to Python implementation
    from sglang.srt.layers.ipc_comm import IpcTensorCommunicator
    return IpcTensorCommunicator(perspective, mb_id)
