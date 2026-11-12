"""NCCL P2P communicator for single-node A↔F over NVLink.

Uses a dedicated ProcessGroupNCCL for stream-ordered P2P communication.
Key design: NO header transmission — shape is negotiated once at first call,
then all subsequent transfers use fixed-size buffers. This eliminates the
need for any CPU-GPU synchronization on the data path.

Result: ~30-40us per transfer (same as NCCL AllReduce), vs 170us with IPC.
"""

import logging
import os
import time
import threading
from datetime import timedelta

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class NcclP2pTensorCommunicator:
    """NCCL P2P communicator — zero CPU sync after first call.

    First send/recv: exchanges shape info (with sync).
    All subsequent: fixed-size ncclSend/ncclRecv, fully stream-ordered.
    """

    def __init__(
        self,
        is_ffn: bool,
        local_device: torch.device,
        nccl_port: int,
        mb_id=None,
    ):
        self.is_ffn = is_ffn
        self._local_device = local_device
        self._nccl_port = nccl_port
        self.mb_id = mb_id
        self.use_compute_stream = True

        self._rank = 0 if is_ffn else 1
        self._peer_rank = 1 if is_ffn else 0

        self._pg = None
        self._ready = threading.Event()

        # Fixed-size buffer (allocated after first call reveals the shape)
        self._recv_buf = None
        self._shape = None
        self._dtype = None
        self._numel = None
        self._shape_negotiated = False

        self._init_thread = threading.Thread(target=self._init_nccl, daemon=True)
        self._init_thread.start()

    def _init_nccl(self):
        """Create a dedicated ProcessGroupNCCL via TCPStore."""
        try:
            torch.cuda.set_device(self._local_device.index)
            logger.info(
                "[NCCL_P2P] rank=%d device=%s port=%d initializing...",
                self._rank, self._local_device, self._nccl_port,
            )

            is_master = (self._rank == 0)
            store = dist.TCPStore(
                host_name="127.0.0.1",
                port=self._nccl_port,
                world_size=2,
                is_master=is_master,
                timeout=timedelta(seconds=120),
            )

            self._pg = dist.ProcessGroupNCCL(store, self._rank, 2)

            self._ready.set()
            logger.info("[NCCL_P2P] rank=%d ready (peer=%d)", self._rank, self._peer_rank)
        except Exception as e:
            logger.error("[NCCL_P2P] rank=%d init failed: %s", self._rank, e)
            import traceback
            traceback.print_exc()

    def _wait_ready(self):
        if not self._ready.is_set():
            self._ready.wait(timeout=300)
            if not self._ready.is_set():
                raise RuntimeError("NCCL P2P communicator init timeout")

    def send_tensor(self, x: torch.Tensor):
        """Send tensor via NCCL P2P.

        After shape negotiation, this is a single ncclSend — fully stream-ordered,
        no CPU-GPU sync. The .wait() only waits for the NCCL kernel to be
        enqueued (not for data transfer to complete).
        """
        self._wait_ready()
        x_cont = x.contiguous() if not x.is_contiguous() else x

        # First call: negotiate shape
        if not self._shape_negotiated:
            self._shape = x_cont.shape
            self._dtype = x_cont.dtype
            self._numel = x_cont.numel()
            # Send shape info as a small tensor
            meta = torch.tensor(
                [x_cont.ndim, x_cont.numel()] + list(x_cont.shape) + [0] * (4 - x_cont.ndim),
                dtype=torch.int64, device=self._local_device
            )
            self._pg.send([meta], self._peer_rank, 0).wait()
            self._shape_negotiated = True
            logger.info(
                "[NCCL_P2P] rank=%d shape negotiated: %s dtype=%s numel=%d",
                self._rank, self._shape, self._dtype, self._numel,
            )

        # Send data (stream-ordered)
        work = self._pg.send([x_cont], self._peer_rank, 0)
        work.wait()

    def recv_tensor(self) -> torch.Tensor:
        """Receive tensor via NCCL P2P.

        After shape negotiation, this is a single ncclRecv into a pre-allocated
        buffer — fully stream-ordered, no CPU-GPU sync needed.
        """
        self._wait_ready()

        # First call: receive shape info
        if not self._shape_negotiated:
            meta = torch.zeros(6, dtype=torch.int64, device=self._local_device)
            self._pg.recv([meta], self._peer_rank, 0).wait()
            torch.cuda.current_stream().synchronize()

            ndim = int(meta[0].item())
            numel = int(meta[1].item())
            shape = tuple(int(meta[2 + i].item()) for i in range(ndim))
            self._shape = shape
            self._numel = numel
            self._dtype = torch.bfloat16  # AF always uses bf16
            self._recv_buf = torch.empty(numel, dtype=self._dtype, device=self._local_device)
            self._shape_negotiated = True
            logger.info(
                "[NCCL_P2P] rank=%d shape negotiated: %s numel=%d",
                self._rank, self._shape, self._numel,
            )

        # Recv data into pre-allocated buffer (stream-ordered)
        work = self._pg.recv([self._recv_buf], self._peer_rank, 0)
        work.wait()

        return self._recv_buf.reshape(self._shape)
