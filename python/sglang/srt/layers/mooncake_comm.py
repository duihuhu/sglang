"""Mooncake RDMA tensor transport for homogeneous AFD tensor parallelism.

Each TP rank owns an independent engine/session, registered-buffer ring, and ZMQ
control channel. Attention rank ``i`` communicates only with FFN rank ``i``.
"""

import logging
import os
import threading
import time
from collections.abc import Callable

import torch
import zmq
from sglang.srt.layers.afd import FifoTensorCommunicator
from sglang.srt.layers.afd_type import AFDPerspective

logger = logging.getLogger(__name__)


class MooncakeTensorCommunicator(FifoTensorCommunicator):
    """Bidirectional FIFO transport backed by Mooncake ``transfer_sync``."""

    def __init__(
        self,
        afd_perspective: AFDPerspective,
        *,
        engine=None,
        peer_host: str | None = None,
        channel: int = 0,
        tp_rank: int = 0,
        tp_size: int = 1,
        ring_slots: int | None = None,
        buffer_bytes: int | None = None,
        timeout_ms: int | None = None,
        attn_control_port: int | None = None,
        ffn_control_port: int | None = None,
        tensor_allocator: Callable[[int], torch.Tensor] | None = None,
    ):
        if channel < 0:
            raise ValueError("Mooncake AFD channel must be non-negative")
        if tp_size <= 0 or tp_rank < 0 or tp_rank >= tp_size:
            raise ValueError(
                f"Invalid Mooncake TP rank/size: tp_rank={tp_rank}, tp_size={tp_size}"
            )
        self.perspective = afd_perspective
        self.channel = int(channel)
        self.tp_rank = int(tp_rank)
        self.tp_size = int(tp_size)
        self.timeout_ms = int(
            timeout_ms or os.getenv("AFD_MOONCAKE_TIMEOUT_MS", "60000")
        )
        self.ring_slots = int(ring_slots or os.getenv("AFD_MOONCAKE_RING_SLOTS", "2"))
        self.buffer_bytes = int(
            buffer_bytes or os.getenv("AFD_MOONCAKE_BUFFER_BYTES", str(4 << 20))
        )
        if self.timeout_ms <= 0 or self.ring_slots <= 0 or self.buffer_bytes <= 0:
            raise ValueError(
                "Mooncake timeout, ring slots, and buffer bytes must be positive"
            )
        self.peer_host = peer_host or os.getenv("AFD_MOONCAKE_PEER_HOST", "127.0.0.1")
        self._device = None
        if torch.cuda.is_available():
            if self.tp_rank >= torch.cuda.device_count():
                raise RuntimeError(
                    "Mooncake TP rank has no matching local CUDA device: "
                    f"tp_rank={self.tp_rank}, device_count={torch.cuda.device_count()}"
                )
            # Communicators can be initialized from helper threads, whose CUDA
            # current-device state defaults to cuda:0. Pin every rank-local
            # engine and registered buffer to the TP rank's model device.
            self._device = torch.device("cuda", self.tp_rank)
        control_base = os.getenv("AFD_MOONCAKE_CONTROL_BASE")
        attn_port = int(
            attn_control_port
            or control_base
            or os.getenv("AFD_MOONCAKE_ATTN_CONTROL_PORT", "63000")
        )
        ffn_port = int(
            ffn_control_port
            or control_base
            or os.getenv("AFD_MOONCAKE_FFN_CONTROL_PORT", "62000")
        )
        port_offset = self.tp_rank + self.channel * self.tp_size
        if afd_perspective == AFDPerspective.AFD_PERSPECTIVE_ATTN:
            self._local_port, self._peer_port = (
                ffn_port + port_offset,
                attn_port + port_offset,
            )
        else:
            self._local_port, self._peer_port = (
                attn_port + port_offset,
                ffn_port + port_offset,
            )

        self.engine = engine or self._get_or_init_engine()
        self._allocator = tensor_allocator or self._cuda_allocator
        self._send_buffers = [
            self._allocator(self.buffer_bytes) for _ in range(self.ring_slots)
        ]
        self._recv_buffers = [
            self._allocator(self.buffer_bytes) for _ in range(self.ring_slots)
        ]
        self._send_caps = [self.buffer_bytes] * self.ring_slots
        self._recv_caps = [self.buffer_bytes] * self.ring_slots
        self._registered: dict[int, int] = {}
        self._closed = False
        self._peer_closed = False
        self._send_seq = 0
        self._recv_seq = 0
        self._acked = set()
        self._ready: dict[int, dict] = {}
        self._resize_responses: dict[int, dict] = {}
        self._next_request_id = 0
        self._cv = threading.Condition()
        self._control_send_lock = threading.Lock()
        self._fifo_send_lock = threading.Lock()
        self._fifo_recv_lock = threading.Lock()
        self._stop = threading.Event()
        try:
            for tensor, capacity in zip(
                self._send_buffers + self._recv_buffers,
                self._send_caps + self._recv_caps,
            ):
                self._register(tensor, capacity)
            self._context = zmq.Context()
            self._pull = self._context.socket(zmq.PULL)
            self._push = self._context.socket(zmq.PUSH)
            for sock in (self._pull, self._push):
                sock.setsockopt(zmq.LINGER, 0)
            self._pull.setsockopt(zmq.RCVTIMEO, min(self.timeout_ms, 100))
            self._push.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
            self._push.setsockopt(zmq.IMMEDIATE, 1)
            self._pull.bind(f"tcp://*:{self._local_port}")
            self._push.connect(f"tcp://{self.peer_host}:{self._peer_port}")
            self._exchange_hello()
            self._control_thread = threading.Thread(
                target=self._control_loop,
                name=f"afd-mooncake-control-tp{self.tp_rank}-ch{self.channel}",
                daemon=True,
            )
            self._control_thread.start()
        except Exception:
            self.close()
            raise

    @property
    def local_control_port(self) -> int:
        return self._local_port

    @property
    def peer_control_port(self) -> int:
        return self._peer_port

    def _cuda_allocator(self, nbytes: int) -> torch.Tensor:
        if self._device is None:
            raise RuntimeError(
                "Mooncake AFD requires CUDA; inject tensor_allocator for CPU tests"
            )
        return torch.empty(nbytes, dtype=torch.uint8, device=self._device)

    def _get_or_init_engine(self):
        from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
            get_mooncake_transfer_engine,
            init_mooncake_transfer_engine,
        )

        engine = get_mooncake_transfer_engine()
        if engine is not None:
            return engine
        from sglang.srt.utils.network import get_local_ip_auto

        gpu_id = self._device.index if self._device is not None else 0
        return init_mooncake_transfer_engine(
            hostname=get_local_ip_auto(),
            gpu_id=gpu_id,
            ib_device=os.getenv("MOONCAKE_IB_DEVICE"),
        )

    @staticmethod
    def _ptr(tensor: torch.Tensor) -> int:
        return int(tensor.data_ptr())

    def _register(self, tensor: torch.Tensor, capacity: int) -> None:
        ptr = self._ptr(tensor)
        ret = self.engine.register(ptr, capacity)
        if ret not in (None, 0):
            raise RuntimeError(
                f"Mooncake memory registration failed: ptr={ptr}, bytes={capacity}, ret={ret}"
            )
        self._registered[ptr] = capacity

    def _deregister(self, tensor: torch.Tensor) -> None:
        ptr = self._ptr(tensor)
        if ptr in self._registered:
            ret = self.engine.deregister(ptr)
            if ret not in (None, 0):
                raise RuntimeError(
                    f"Mooncake memory deregistration failed: ptr={ptr}, ret={ret}"
                )
            self._registered.pop(ptr, None)

    def _send_control(self, message: dict) -> None:
        if self._closed:
            raise RuntimeError("Mooncake communicator is closed")
        try:
            with self._control_send_lock:
                self._push.send_pyobj(message)
        except zmq.Again as exc:
            raise TimeoutError("Mooncake control send timed out") from exc

    def _recv_control_direct(self) -> dict:
        deadline = time.monotonic() + self.timeout_ms / 1000
        while True:
            try:
                return self._pull.recv_pyobj()
            except zmq.Again:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Mooncake control receive timed out")

    def _exchange_hello(self) -> None:
        self._send_control(
            {
                "type": "HELLO",
                "version": 1,
                "session_id": self.engine.get_session_id(),
                "channel": self.channel,
                "tp_rank": self.tp_rank,
                "tp_size": self.tp_size,
                "recv_slots": [
                    {"slot": i, "ptr": self._ptr(buf), "capacity": self._recv_caps[i]}
                    for i, buf in enumerate(self._recv_buffers)
                ],
            }
        )
        peer = self._recv_control_direct()
        if peer.get("type") != "HELLO" or peer.get("version") != 1:
            raise RuntimeError(f"Invalid Mooncake HELLO: {peer!r}")
        if peer.get("channel") != self.channel:
            raise RuntimeError(f"Mooncake channel mismatch: {peer!r}")
        if peer.get("tp_size") != self.tp_size or peer.get("tp_rank") != self.tp_rank:
            raise ValueError(
                "Mooncake rank-per-rank TP mismatch: "
                f"local rank/size={self.tp_rank}/{self.tp_size}, "
                f"peer rank/size={peer.get('tp_rank')}/{peer.get('tp_size')}"
            )
        slots = peer.get("recv_slots", [])
        if len(slots) != self.ring_slots:
            raise RuntimeError(
                "Mooncake peers configured with different ring slot counts"
            )
        self._peer_session_id = peer["session_id"]
        self._remote_slots = {int(item["slot"]): dict(item) for item in slots}

    def _control_loop(self) -> None:
        while not self._stop.is_set():
            try:
                message = self._pull.recv_pyobj()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                if not self._stop.is_set():
                    logger.exception("Mooncake AFD control loop failed")
                return
            try:
                kind = message.get("type")
                if kind == "READY":
                    with self._cv:
                        self._ready[int(message["seq"])] = message
                        self._cv.notify_all()
                elif kind == "ACK":
                    with self._cv:
                        self._acked.add(int(message["seq"]))
                        self._cv.notify_all()
                elif kind == "RESIZE_REQUEST":
                    self._handle_resize_request(message)
                elif kind == "RESIZE_RESPONSE":
                    with self._cv:
                        self._resize_responses[int(message["request_id"])] = message
                        self._cv.notify_all()
                elif kind == "CLOSE":
                    with self._cv:
                        self._peer_closed = True
                        self._cv.notify_all()
                    return
                else:
                    raise RuntimeError(f"Unknown Mooncake control message: {message!r}")
            except Exception as exc:
                logger.exception("Mooncake AFD control message failed: %r", message)
                with self._cv:
                    self._control_error = exc
                    self._cv.notify_all()
                return

    def _wait_for(self, predicate, description: str):
        deadline = time.monotonic() + self.timeout_ms / 1000
        with self._cv:
            while not predicate():
                error = getattr(self, "_control_error", None)
                if error is not None:
                    raise RuntimeError("Mooncake control plane failed") from error
                if self._peer_closed:
                    raise RuntimeError("Mooncake peer closed")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Mooncake timed out waiting for {description}")
                self._cv.wait(remaining)

    def _replace_buffer(self, buffers, capacities, slot: int, required: int) -> None:
        old = buffers[slot]
        new_capacity = max(required, capacities[slot] * 2)
        new = self._allocator(new_capacity)
        self._register(new, new_capacity)
        buffers[slot] = new
        capacities[slot] = new_capacity
        self._deregister(old)

    def _handle_resize_request(self, message: dict) -> None:
        slot, required = int(message["slot"]), int(message["required"])
        if slot < 0 or slot >= self.ring_slots or required <= 0:
            raise ValueError(f"Invalid Mooncake resize request: {message!r}")
        if required > self._recv_caps[slot]:
            self._replace_buffer(self._recv_buffers, self._recv_caps, slot, required)
        self._send_control(
            {
                "type": "RESIZE_RESPONSE",
                "request_id": int(message["request_id"]),
                "slot": slot,
                "ptr": self._ptr(self._recv_buffers[slot]),
                "capacity": self._recv_caps[slot],
            }
        )

    def _ensure_remote_capacity(self, slot: int, required: int) -> None:
        if required <= int(self._remote_slots[slot]["capacity"]):
            return
        request_id = self._next_request_id
        self._next_request_id += 1
        self._send_control(
            {
                "type": "RESIZE_REQUEST",
                "request_id": request_id,
                "slot": slot,
                "required": required,
            }
        )
        self._wait_for(
            lambda: request_id in self._resize_responses,
            f"resize response {request_id}",
        )
        with self._cv:
            response = self._resize_responses.pop(request_id)
        self._remote_slots[slot] = {
            "slot": slot,
            "ptr": int(response["ptr"]),
            "capacity": int(response["capacity"]),
        }

    def send_tensor(self, x: torch.Tensor):
        with self._fifo_send_lock:
            if self._closed:
                raise RuntimeError("Mooncake communicator is closed")
            x = x.detach().contiguous()
            nbytes = x.numel() * x.element_size()
            seq, slot = self._send_seq, self._send_seq % self.ring_slots
            if seq >= self.ring_slots:
                previous = seq - self.ring_slots
                self._wait_for(lambda: previous in self._acked, f"ACK seq={previous}")
                with self._cv:
                    self._acked.discard(previous)
            self._ensure_remote_capacity(slot, nbytes)
            if nbytes > self._send_caps[slot]:
                self._replace_buffer(self._send_buffers, self._send_caps, slot, nbytes)
            send_bytes = self._send_buffers[slot][:nbytes]
            send_bytes.copy_(x.view(torch.uint8).reshape(-1))
            if x.is_cuda:
                torch.cuda.current_stream(x.device).synchronize()
            remote = self._remote_slots[slot]
            ret = self.engine.transfer_sync(
                self._peer_session_id, self._ptr(send_bytes), int(remote["ptr"]), nbytes
            )
            if ret < 0:
                raise RuntimeError(
                    f"Mooncake transfer_sync failed for seq={seq}, ret={ret}"
                )
            self._send_control(
                {
                    "type": "READY",
                    "seq": seq,
                    "slot": slot,
                    "shape": list(x.shape),
                    "dtype": str(x.dtype).removeprefix("torch."),
                    "nbytes": nbytes,
                }
            )
            self._send_seq += 1

    def recv_tensor(self) -> torch.Tensor:
        with self._fifo_recv_lock:
            seq = self._recv_seq
            self._wait_for(lambda: seq in self._ready, f"READY seq={seq}")
            with self._cv:
                message = self._ready.pop(seq)
            slot, nbytes = int(message["slot"]), int(message["nbytes"])
            if slot != seq % self.ring_slots or nbytes > self._recv_caps[slot]:
                raise RuntimeError(f"Invalid Mooncake READY: {message!r}")
            dtype = getattr(torch, message["dtype"], None)
            if not isinstance(dtype, torch.dtype):
                raise RuntimeError(
                    f"Unsupported Mooncake tensor dtype: {message['dtype']}"
                )
            element_size = torch.empty((), dtype=dtype).element_size()
            if nbytes % element_size:
                raise RuntimeError(f"Misaligned Mooncake payload: {message!r}")
            result = (
                self._recv_buffers[slot][:nbytes]
                .view(dtype)
                .reshape(message["shape"])
                .clone()
            )
            if result.is_cuda:
                torch.cuda.current_stream(result.device).synchronize()
            self._send_control({"type": "ACK", "seq": seq, "slot": slot})
            self._recv_seq += 1
            return result

    def close(self):
        if getattr(self, "_closed", False):
            return
        if hasattr(self, "_push"):
            try:
                self._send_control({"type": "CLOSE"})
            except Exception:
                pass
        self._closed = True
        if hasattr(self, "_stop"):
            self._stop.set()
        if hasattr(self, "_cv"):
            with self._cv:
                self._cv.notify_all()
        thread = getattr(self, "_control_thread", None)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(min(self.timeout_ms / 1000, 1.0), 0.1))
        for name in ("_pull", "_push"):
            sock = getattr(self, name, None)
            if sock is not None:
                try:
                    sock.close(linger=0)
                except Exception:
                    pass
        context = getattr(self, "_context", None)
        if context is not None:
            try:
                context.term()
            except Exception:
                pass
        for tensor in getattr(self, "_send_buffers", []) + getattr(
            self, "_recv_buffers", []
        ):
            try:
                self._deregister(tensor)
            except Exception:
                logger.exception("Mooncake AFD deregistration failed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
