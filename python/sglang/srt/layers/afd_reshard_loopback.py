# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Correctness-first same-GPU, cross-process transport for colocated A/F.

The two A/F processes have independent CUDA contexts and independent torch
process groups.  This backend therefore uses a private pair of TCP control
connections and receiver-owned POSIX shared-memory slots.  CUDA tensors are
staged through pinned host memory (D2H -> shared memory -> pinned host -> H2D).
It is a real cross-process transport, but deliberately not a performance path:
each transfer has two CUDA copies, two host copies, and an acknowledgement.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import struct
import threading
import time
from multiprocessing import shared_memory
from typing import Dict, Optional

import torch

from sglang.srt.layers.afd_type import AFDPerspective

logger = logging.getLogger(__name__)
_HEADER = struct.Struct("!I")
_DTYPE_TO_NAME = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.bool: "bool",
}
_NAME_TO_DTYPE = {name: dtype for dtype, name in _DTYPE_TO_NAME.items()}


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("AFD loopback peer closed the control connection")
        chunks.extend(chunk)
    return bytes(chunks)


def _send_frame(sock: socket.socket, value: Dict[str, object]) -> None:
    payload = json.dumps(value, separators=(",", ":")).encode("utf-8")
    sock.sendall(_HEADER.pack(len(payload)) + payload)


def _recv_frame(sock: socket.socket) -> Dict[str, object]:
    (size,) = _HEADER.unpack(_recv_exact(sock, _HEADER.size))
    if size > 1024 * 1024:
        raise RuntimeError(f"AFD loopback control frame is too large: {size}")
    return json.loads(_recv_exact(sock, size))


def _create_listener(host: str, port: int, timeout_s: float) -> socket.socket:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, port))
    listener.listen(1)
    listener.settimeout(timeout_s)
    return listener


def _accept(listener: socket.socket, timeout_s: float) -> socket.socket:
    try:
        conn, _ = listener.accept()
    finally:
        listener.close()
    conn.settimeout(timeout_s)
    return conn


def _connect(host: str, port: int, timeout_s: float) -> socket.socket:
    deadline = time.monotonic() + timeout_s
    last_error: Optional[OSError] = None
    while time.monotonic() < deadline:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(min(2.0, max(0.1, deadline - time.monotonic())))
        try:
            sock.connect((host, port))
            sock.settimeout(timeout_s)
            return sock
        except OSError as exc:
            last_error = exc
            sock.close()
            time.sleep(0.05)
    raise TimeoutError(f"timed out connecting AFD loopback {host}:{port}: {last_error}")


class _Direction:
    """One sender/receiver direction with a receiver-owned shared-memory slot."""

    def __init__(self, *, sender: bool, timeout_s: float, capacity: int):
        self.sender = sender
        self.timeout_s = timeout_s
        self.capacity = capacity
        self._sequence = 0
        self._lock = threading.Lock()
        self._closed = False
        self._owner = not sender
        self._shm = (
            shared_memory.SharedMemory(create=True, size=capacity)
            if self._owner
            else None
        )
        self._sock: Optional[socket.socket] = None

    def connect(self, host: str, port: int) -> None:
        if not self.sender or self._sock is not None:
            raise RuntimeError("invalid AFD loopback sender initialization")
        self._sock = _connect(host, port, self.timeout_s)

    def accept(self, listener: socket.socket) -> None:
        if self.sender or self._sock is not None:
            raise RuntimeError("invalid AFD loopback receiver initialization")
        self._sock = _accept(listener, self.timeout_s)
        _send_frame(
            self._sock,
            {"kind": "hello", "shm_name": self._shm.name, "capacity": self.capacity},
        )

    def finish_connect(self) -> None:
        if not self.sender or self._sock is None:
            raise RuntimeError("AFD loopback sender is not connected")
        hello = _recv_frame(self._sock)
        if hello.get("kind") != "hello":
            raise RuntimeError(f"invalid AFD loopback handshake: {hello}")
        self.capacity = int(hello["capacity"])
        self._shm = shared_memory.SharedMemory(name=str(hello["shm_name"]))

    def send(self, tensor: torch.Tensor) -> None:
        if not self.sender:
            raise RuntimeError("send called on an AFD loopback receive direction")
        if not tensor.is_cuda:
            raise ValueError("AFD loopback only accepts CUDA tensors")
        if tensor.dtype not in _DTYPE_TO_NAME:
            raise ValueError(f"unsupported AFD loopback dtype: {tensor.dtype}")
        value = tensor.detach().contiguous()
        nbytes = value.numel() * value.element_size()
        if nbytes > self.capacity:
            raise RuntimeError(
                f"AFD loopback tensor needs {nbytes} bytes, shared slot has {self.capacity}; "
                "increase AFD_RESHARD_LOOPBACK_CAPACITY_MB"
            )
        with self._lock:
            host = torch.empty(value.shape, dtype=value.dtype, device="cpu", pin_memory=True)
            host.copy_(value, non_blocking=True)
            torch.cuda.current_stream(value.device).synchronize()
            raw = host.view(torch.uint8).reshape(-1)
            self._shm.buf[:nbytes] = raw.numpy().tobytes()
            _send_frame(
                self._sock,
                {
                    "kind": "tensor",
                    "sequence": self._sequence,
                    "shape": list(value.shape),
                    "dtype": _DTYPE_TO_NAME[value.dtype],
                    "nbytes": nbytes,
                },
            )
            ack = _recv_frame(self._sock)
            if ack != {"kind": "ack", "sequence": self._sequence}:
                raise RuntimeError(f"invalid AFD loopback acknowledgement: {ack}")
            self._sequence += 1

    def recv(self, device: torch.device) -> torch.Tensor:
        if self.sender:
            raise RuntimeError("recv called on an AFD loopback send direction")
        with self._lock:
            meta = _recv_frame(self._sock)
            if meta.get("kind") != "tensor":
                raise RuntimeError(f"expected AFD loopback tensor, got {meta}")
            sequence = int(meta["sequence"])
            if sequence != self._sequence:
                raise RuntimeError(
                    f"AFD loopback sequence mismatch: {sequence} != {self._sequence}"
                )
            dtype = _NAME_TO_DTYPE.get(str(meta["dtype"]))
            if dtype is None:
                raise RuntimeError(f"unsupported peer dtype: {meta['dtype']}")
            shape = tuple(int(item) for item in meta["shape"])
            nbytes = int(meta["nbytes"])
            expected = torch.empty((), dtype=dtype).element_size()
            for dim in shape:
                expected *= dim
            if expected != nbytes or nbytes > self.capacity:
                raise RuntimeError(
                    f"invalid AFD loopback metadata: shape={shape}, nbytes={nbytes}"
                )
            host = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
            source = torch.frombuffer(self._shm.buf, dtype=torch.uint8, count=nbytes)
            host.view(torch.uint8).reshape(-1).copy_(source)
            output = host.to(device=device, non_blocking=True)
            torch.cuda.current_stream(device).synchronize()
            _send_frame(self._sock, {"kind": "ack", "sequence": sequence})
            self._sequence += 1
            return output

    def send_fence(self, epoch: int) -> None:
        if not self.sender:
            raise RuntimeError("send_fence called on receive direction")
        with self._lock:
            _send_frame(
                self._sock,
                {"kind": "fence", "epoch": epoch, "sequence": self._sequence},
            )
            ack = _recv_frame(self._sock)
            if ack != {
                "kind": "fence_ack",
                "epoch": epoch,
                "sequence": self._sequence,
            }:
                raise RuntimeError(f"invalid AFD loopback fence acknowledgement: {ack}")

    def recv_fence(self, epoch: int) -> None:
        if self.sender:
            raise RuntimeError("recv_fence called on send direction")
        with self._lock:
            value = _recv_frame(self._sock)
            expected = {"kind": "fence", "epoch": epoch, "sequence": self._sequence}
            if value != expected:
                raise RuntimeError(f"invalid AFD loopback fence: {value}, expected {expected}")
            _send_frame(
                self._sock,
                {"kind": "fence_ack", "epoch": epoch, "sequence": self._sequence},
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._sock.close()
        if self._shm is not None:
            self._shm.close()
            if self._owner:
                try:
                    self._shm.unlink()
                except FileNotFoundError:
                    pass


class AFDReshardLoopbackTensorCommunicator:
    """Full-duplex same-device A/F communicator using two isolated channels."""

    cross_process = True
    supports_concurrent_recv = False
    performance_note = "pinned D2H + POSIX SHM host copy + pinned H2D; correctness-first"

    def __init__(
        self,
        perspective: AFDPerspective,
        *,
        local_device: Optional[torch.device] = None,
        host: Optional[str] = None,
        base_port: Optional[int] = None,
        timeout_s: Optional[float] = None,
        capacity_bytes: Optional[int] = None,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("afd_reshard_loopback requires CUDA")
        self.is_ffn = perspective == AFDPerspective.AFD_PERSPECTIVE_FFN
        self._device = local_device or torch.device("cuda", torch.cuda.current_device())
        self._host = host or os.getenv("AFD_RESHARD_LOOPBACK_HOST", "127.0.0.1")
        self._base_port = int(
            base_port
            if base_port is not None
            else os.getenv("AFD_RESHARD_LOOPBACK_PORT", "29620")
        )
        self._timeout_s = float(
            timeout_s
            if timeout_s is not None
            else os.getenv("AFD_RESHARD_LOOPBACK_TIMEOUT_S", "120")
        )
        self._capacity = int(
            capacity_bytes
            if capacity_bytes is not None
            else int(os.getenv("AFD_RESHARD_LOOPBACK_CAPACITY_MB", "256"))
            * 1024
            * 1024
        )
        if not (0 < self._base_port < 65535) or self._base_port + 1 > 65535:
            raise ValueError(f"invalid AFD loopback base port: {self._base_port}")
        # A->F uses base_port; F->A uses base_port+1.  Both roles must publish
        # their incoming listener before either waits for a hello.  Otherwise each
        # process can block in its first _Direction constructor and never create
        # the listener needed by the peer's second direction.
        incoming_port = self._base_port if self.is_ffn else self._base_port + 1
        outgoing_port = self._base_port + 1 if self.is_ffn else self._base_port
        listener = _create_listener(self._host, incoming_port, self._timeout_s)
        self._incoming = _Direction(
            sender=False, timeout_s=self._timeout_s, capacity=self._capacity
        )
        self._outgoing = _Direction(
            sender=True, timeout_s=self._timeout_s, capacity=self._capacity
        )
        self._closed = False
        try:
            self._outgoing.connect(self._host, outgoing_port)
            self._incoming.accept(listener)
            self._outgoing.finish_connect()
        except BaseException:
            listener.close()
            self.close()
            raise
        logger.warning(
            "[AFD_RESHARD_LOOPBACK] role=%s device=%s ports=%d/%d capacity=%d; %s",
            "FFN" if self.is_ffn else "ATTN",
            self._device,
            self._base_port,
            self._base_port + 1,
            self._capacity,
            self.performance_note,
        )

    def send_tensor(self, tensor: torch.Tensor) -> None:
        self._outgoing.send(tensor)

    def recv_tensor(self) -> torch.Tensor:
        return self._incoming.recv(self._device)

    send_stream_ordered = send_tensor
    recv_stream_ordered = recv_tensor

    def fence(self, epoch: int = 0) -> None:
        """Drain both directions and verify an epoch-scoped peer fence."""
        if self.is_ffn:
            self._incoming.recv_fence(epoch)
            self._outgoing.send_fence(epoch)
        else:
            self._outgoing.send_fence(epoch)
            self._incoming.recv_fence(epoch)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._outgoing.close()
        self._incoming.close()

    cleanup = close

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
