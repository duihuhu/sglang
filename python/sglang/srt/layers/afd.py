# AF disaggregation

# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import itertools
import logging
import multiprocessing
import os
import queue
import subprocess
import threading
import time
import traceback
from abc import ABC, abstractmethod
from collections import deque
from enum import Enum, auto
from functools import cache
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.distributed as dist
import zmq
from torch import nn

from sglang.srt.layers.afd_type import AFDPerspective
from sglang.srt.layers.communicator import (
    CommunicateContext,
    CommunicateSummableTensorPairFn,
    LayerCommunicator,
    ScatterMode,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import get_global_server_args

logger = logging.getLogger(__name__)

_zmq_bound_endpoints = set()
_zmq_bound_endpoints_lock = threading.Lock()


def _cross_node_experimental_enabled() -> bool:
    return os.environ.get("AFD_CROSS_NODE_EXPERIMENTAL", "0") == "1"


def _zmq_double_buffer_enabled() -> bool:
    return _zmq_sharding_enabled() and os.environ.get("AFD_ZMQ_DOUBLE_BUFFER", "0") == "1"


def _zmq_sharding_enabled() -> bool:
    """Enable TP-sharded ZMQ only behind the cross-node master switch."""
    return (
        _cross_node_experimental_enabled()
        and os.environ.get("AFD_ZMQ_SHARDING", "0") == "1"
    )


# --------------- Stage types and scheduling ---------------


class AFDForwardStage(Enum):
    AFD_FORWARD_STAGE_A = auto()
    AFD_FORWARD_STAGE_F = auto()


# G6 optimization: use NamedTuple instead of dict for stage I/O
class StageIO(NamedTuple):
    hidden_states: torch.Tensor
    residual: Optional[torch.Tensor]


class AFDStageScheduleGenerator:
    Schedule = List[Tuple[AFDForwardStage, int, int]]

    @staticmethod
    def ffn_stage(num_layers: int, m_stage: int) -> "AFDStageScheduleGenerator.Schedule":
        schedule = []
        for layer_id, m in itertools.product(range(num_layers), range(m_stage)):
            schedule.append((AFDForwardStage.AFD_FORWARD_STAGE_A, layer_id, m))
            schedule.append((AFDForwardStage.AFD_FORWARD_STAGE_F, layer_id, m))
        return schedule

    @staticmethod
    def attn_stage(
        num_layers: int, m_stage: int
    ) -> "AFDStageScheduleGenerator.Schedule":
        # Batch schedule: all A-stages for a layer, then all F-stages.
        # This maximizes overlap: while DA does A(L,0)..A(L,M-1), DF is
        # computing MLP for L-1 results. By the time DA reaches F(L-1,0),
        # DF has likely finished and sent back all results.
        #
        # Schedule for M=3, 3 layers:
        #   A(0,0) A(0,1) A(0,2) | F(0,0) F(0,1) F(0,2) | A(1,0) A(1,1) A(1,2) | ...
        schedule = []
        if num_layers == 1:
            return [
                (AFDForwardStage.AFD_FORWARD_STAGE_A, 0, m) for m in range(m_stage)
            ] + [
                (AFDForwardStage.AFD_FORWARD_STAGE_F, 0, m) for m in range(m_stage)
            ]
        # Layer 0: only A-stages (no F(-1) exists)
        for m in range(m_stage):
            schedule.append((AFDForwardStage.AFD_FORWARD_STAGE_A, 0, m))
        # Layers 1..N-1: batch-F(prev) then batch-A(cur)
        for layer_id in range(1, num_layers):
            # All F-stages for previous layer (recv from DF)
            for m in range(m_stage):
                schedule.append(
                    (AFDForwardStage.AFD_FORWARD_STAGE_F, layer_id - 1, m)
                )
            # All A-stages for current layer
            for m in range(m_stage):
                schedule.append((AFDForwardStage.AFD_FORWARD_STAGE_A, layer_id, m))
        # Final F-stages for last layer
        for m in range(m_stage):
            schedule.append(
                (AFDForwardStage.AFD_FORWARD_STAGE_F, num_layers - 1, m)
            )
        return schedule

    @staticmethod
    def attn_stage_interleaved(
        num_layers: int, m_stage: int
    ) -> "AFDStageScheduleGenerator.Schedule":
        """Interleaved schedule: mb0 advances to next layer as soon as its
        F-stage completes, without waiting for mb1/mb2.

        For M=3, 3 layers the schedule is:
          A(0,0) A(0,1) A(0,2)
          F(0,0) A(1,0) F(0,1) A(1,1) F(0,2) A(1,2)
          F(1,0) A(2,0) F(1,1) A(2,1) F(1,2) A(2,2)
          F(2,0) F(2,1) F(2,2)

        This ensures each mb advances to the next layer immediately after
        receiving its FFN result, achieving compute-communication decoupling.
        The FIFO channel guarantees correct ordering since both DA and DF
        process mbs in the same 0,1,2 order within each layer.
        """
        schedule = []
        if m_stage == 1:
            # M=1: no interleaving possible, same as batch schedule
            for layer_id in range(num_layers):
                schedule.append((AFDForwardStage.AFD_FORWARD_STAGE_A, layer_id, 0))
                schedule.append((AFDForwardStage.AFD_FORWARD_STAGE_F, layer_id, 0))
            return schedule

        if num_layers == 1:
            return [
                (AFDForwardStage.AFD_FORWARD_STAGE_A, 0, m) for m in range(m_stage)
            ] + [
                (AFDForwardStage.AFD_FORWARD_STAGE_F, 0, m) for m in range(m_stage)
            ]

        # Layer 0: only A-stages
        for m in range(m_stage):
            schedule.append((AFDForwardStage.AFD_FORWARD_STAGE_A, 0, m))

        # Layers 1..N-1: interleave F(prev,m) with A(cur,m)
        for layer_id in range(1, num_layers):
            for m in range(m_stage):
                # Receive FFN result for previous layer, mb=m
                schedule.append(
                    (AFDForwardStage.AFD_FORWARD_STAGE_F, layer_id - 1, m)
                )
                # Immediately compute Attn for current layer, mb=m
                schedule.append(
                    (AFDForwardStage.AFD_FORWARD_STAGE_A, layer_id, m)
                )

        # Final F-stages for last layer
        for m in range(m_stage):
            schedule.append(
                (AFDForwardStage.AFD_FORWARD_STAGE_F, num_layers - 1, m)
            )
        return schedule


# --------------- Tensor communicators ---------------


class FifoTensorCommunicator(ABC):
    @abstractmethod
    def recv_tensor(self) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def send_tensor(self, x: torch.Tensor):
        raise NotImplementedError


class ZMQSimpleTensorCommunicator(FifoTensorCommunicator):
    """ZMQ tensor communicator with zero-copy optimization (C1) and timeouts (E5)."""

    # E5: configurable timeout in milliseconds
    SOCKET_TIMEOUT_MS = int(os.getenv("AFD_ZMQ_TIMEOUT_MS", "60000"))

    def __init__(self, afd_perspective: AFDPerspective):
        super().__init__()
        if _cross_node_experimental_enabled():
            logger.warning(
                "Constructing AFD ZMQ communicator pid=%d thread=%s perspective=%s "
                "experimental_singletons_id=%s stack=%s",
                os.getpid(),
                threading.current_thread().name,
                afd_perspective,
                id(globals().get("_experimental_tensor_communicators")),
                " | ".join(traceback.format_stack(limit=8)[:-1]).replace("\n", " "),
            )
        self.zmq_context = zmq.Context()

        self.start_lport = (
            self._get_ffn_port()
            if afd_perspective == AFDPerspective.AFD_PERSPECTIVE_ATTN
            else self._get_attn_port()
        )
        self.start_dport = (
            self._get_attn_port()
            if afd_perspective == AFDPerspective.AFD_PERSPECTIVE_ATTN
            else self._get_ffn_port()
        )
        self._cuda_device = None
        self._afd_perspective = afd_perspective
        self._diag_send_count = 0
        self._diag_recv_count = 0
        self._handshake_thread = None
        self._handshake_rep = None
        # C1: pre-allocated pinned memory for staging
        self._pinned_send_buf: Optional[torch.Tensor] = None
        self._pinned_recv_buf: Optional[torch.Tensor] = None
        # C2: separate CUDA stream for async transfers
        if torch.cuda.is_available():
            self._comm_stream = torch.cuda.Stream()
        else:
            self._comm_stream = None

        if _cross_node_experimental_enabled():
            self._get_pull_socket()
            self._get_push_socket()
            self._perform_ready_handshake()

    @staticmethod
    def _get_ffn_port() -> int:
        return int(os.getenv("AFD_FFN_BASE_PORT", "40000"))

    @staticmethod
    def _get_attn_port() -> int:
        return int(os.getenv("AFD_ATTN_BASE_PORT", "50000"))

    def _get_lport(self) -> int:
        return self.start_lport + 1 + self._rank()

    def _get_dport(self) -> int:
        return self.start_dport + 1 + self._rank()

    @staticmethod
    def _get_ffn_handshake_port() -> int:
        return int(os.getenv("AFD_ZMQ_FFN_HANDSHAKE_BASE_PORT", "60000"))

    @staticmethod
    def _get_attn_handshake_port() -> int:
        return int(os.getenv("AFD_ZMQ_ATTN_HANDSHAKE_BASE_PORT", "61000"))

    def _get_handshake_lport(self) -> int:
        base = (self._get_ffn_handshake_port() if self._afd_perspective == AFDPerspective.AFD_PERSPECTIVE_ATTN else self._get_attn_handshake_port())
        return base + 1 + self._rank()

    def _get_handshake_dport(self) -> int:
        base = (self._get_attn_handshake_port() if self._afd_perspective == AFDPerspective.AFD_PERSPECTIVE_ATTN else self._get_ffn_handshake_port())
        return base + 1 + self._rank()

    def _rank(self) -> int:
        """Return the node-local TP rank used for peer/port pairing.

        The scheduler's initialized distributed rank is authoritative. Some
        launch paths do not export LOCAL_RANK to forked scheduler processes.
        """
        env_rank = os.environ.get("LOCAL_RANK")
        if dist.is_initialized():
            local_tp = int(os.environ.get("AFD_LOCAL_TP", "0"))
            if local_tp <= 0:
                local_tp = int(getattr(get_global_server_args(), "tp_size", 1))
            rank = dist.get_rank() % local_tp
            if env_rank is not None and int(env_rank) != rank:
                logger.warning(
                    "Ignoring mismatched LOCAL_RANK=%s; distributed local TP rank=%d "
                    "(global_rank=%d local_tp=%d)",
                    env_rank, rank, dist.get_rank(), local_tp,
                )
            return rank
        if env_rank is not None:
            return int(env_rank)
        return 0

    @staticmethod
    def _port_owner_diagnostic(port: int) -> str:
        commands = (["ss", "-H", "-ltnp", f"sport = :{port}"],
                    ["fuser", "-v", "-n", "tcp", str(port)])
        output = []
        for command in commands:
            try:
                result = subprocess.run(command, text=True, capture_output=True,
                                        timeout=2, check=False)
                text = " ".join((result.stdout + result.stderr).split())
                if text:
                    output.append(f"{command[0]}: {text}")
            except (FileNotFoundError, subprocess.SubprocessError):
                pass
        return "; ".join(output) or "owner unavailable"

    def _bind_pull_socket(self, socket: zmq.Socket, endpoint: str) -> None:
        retries = (int(os.getenv("AFD_ZMQ_BIND_RETRIES", "3"))
                   if _cross_node_experimental_enabled() else 1)
        delay = float(os.getenv("AFD_ZMQ_BIND_RETRY_DELAY_S", "0.25"))
        with _zmq_bound_endpoints_lock:
            if endpoint in _zmq_bound_endpoints:
                raise RuntimeError(
                    f"AFD ZMQ duplicate communicator in pid={os.getpid()}: "
                    f"endpoint {endpoint} is already bound by this process"
                )
        for attempt in range(1, retries + 1):
            try:
                socket.bind(endpoint)
                with _zmq_bound_endpoints_lock:
                    if endpoint in _zmq_bound_endpoints:
                        socket.unbind(endpoint)
                        raise RuntimeError(
                            f"AFD ZMQ duplicate communicator in pid={os.getpid()}: "
                            f"endpoint {endpoint} was concurrently bound"
                        )
                    _zmq_bound_endpoints.add(endpoint)
                return
            except zmq.ZMQError as exc:
                if exc.errno != zmq.EADDRINUSE:
                    raise
                owner = self._port_owner_diagnostic(self._get_lport())
                logger.error(
                    "AFD ZMQ bind EADDRINUSE: endpoint=%s perspective=%s "
                    "rank=%d pid=%d attempt=%d/%d owner=%s",
                    endpoint, self._afd_perspective, self._rank(), os.getpid(),
                    attempt, retries, owner,
                )
                with _zmq_bound_endpoints_lock:
                    duplicate = endpoint in _zmq_bound_endpoints
                if duplicate or attempt == retries:
                    raise
                time.sleep(delay)

    def _perform_ready_handshake(self) -> None:
        peer_host = os.getenv("AFD_ZMQ_PEER_HOST", "127.0.0.1")
        timeout_ms = int(os.getenv("AFD_ZMQ_HANDSHAKE_TIMEOUT_MS", str(self.SOCKET_TIMEOUT_MS)))
        bound = threading.Event()
        errors = []

        def respond():
            try:
                rep = self.zmq_context.socket(zmq.REP)
                self._handshake_rep = rep
                rep.setsockopt(zmq.LINGER, 0)
                rep.setsockopt(zmq.RCVTIMEO, timeout_ms)
                rep.setsockopt(zmq.SNDTIMEO, timeout_ms)
                rep.bind(f"tcp://*:{self._get_handshake_lport()}")
                bound.set()
                request = rep.recv_pyobj()
                rep.send_pyobj({"ack": True, "rank": self._rank(), "request": request})
            except Exception as exc:
                errors.append(exc)
                bound.set()

        self._handshake_thread = threading.Thread(target=respond, name=f"afd-zmq-ready-{self._rank()}", daemon=True)
        self._handshake_thread.start()
        if not bound.wait(max(timeout_ms, 1) / 1000):
            raise TimeoutError("AFD ZMQ handshake listener bind timed out")
        if errors:
            raise RuntimeError(f"AFD ZMQ handshake listener failed: {errors[0]}")
        req = self.zmq_context.socket(zmq.REQ)
        req.setsockopt(zmq.LINGER, 0); req.setsockopt(zmq.IMMEDIATE, 1)
        req.setsockopt(zmq.SNDTIMEO, timeout_ms); req.setsockopt(zmq.RCVTIMEO, timeout_ms)
        endpoint = f"tcp://{peer_host}:{self._get_handshake_dport()}"
        started = time.monotonic()
        try:
            req.connect(endpoint)
            req.send_pyobj({"ready": True, "rank": self._rank()})
            ack = req.recv_pyobj()
            if not ack.get("ack"):
                raise RuntimeError(f"invalid ACK: {ack!r}")
            self._handshake_thread.join(max(timeout_ms, 1) / 1000)
            if self._handshake_thread.is_alive():
                raise TimeoutError("AFD ZMQ inbound handshake timed out")
            if errors:
                raise RuntimeError(f"AFD ZMQ inbound handshake failed: {errors[0]}")
        except zmq.Again as exc:
            raise TimeoutError(f"AFD ZMQ handshake timed out connecting {endpoint}") from exc
        finally:
            req.close(linger=0)
        logger.info("AFD ZMQ handshake ready: perspective=%s rank=%d pull=%d push=%s:%d handshake=%d<->%s:%d elapsed_ms=%.1f", self._afd_perspective, self._rank(), self._get_lport(), peer_host, self._get_dport(), self._get_handshake_lport(), peer_host, self._get_handshake_dport(), (time.monotonic() - started) * 1000)

    def _get_cuda_device(self) -> torch.device:
        if self._cuda_device is None:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available.")
            self._cuda_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        return self._cuda_device

    @cache
    def _get_push_socket(self) -> zmq.Socket:
        socket = self.zmq_context.socket(zmq.PUSH)
        if _cross_node_experimental_enabled():
            socket.setsockopt(zmq.IMMEDIATE, 1)
            socket.setsockopt(zmq.LINGER, 0)
        # E5: set send timeout
        if self.SOCKET_TIMEOUT_MS > 0:
            socket.setsockopt(zmq.SNDTIMEO, self.SOCKET_TIMEOUT_MS)
        if _cross_node_experimental_enabled():
            peer_host = os.getenv("AFD_ZMQ_PEER_HOST", "127.0.0.1")
            socket.connect(f"tcp://{peer_host}:{self._get_dport()}")
        else:
            socket.connect(f"tcp://localhost:{self._get_dport()}")
        return socket

    @cache
    def _get_pull_socket(self) -> zmq.Socket:
        socket = self.zmq_context.socket(zmq.PULL)
        # E5: set receive timeout
        if self.SOCKET_TIMEOUT_MS > 0:
            socket.setsockopt(zmq.RCVTIMEO, self.SOCKET_TIMEOUT_MS)
        endpoint = f"tcp://*:{self._get_lport()}"
        self._bind_pull_socket(socket, endpoint)
        return socket

    def close(self):
        """Release ZMQ sockets and context."""
        for getter in (self._get_push_socket, self._get_pull_socket):
            try:
                sock = getter()
                sock.close(linger=0)
            except Exception:
                pass
        endpoint = f"tcp://*:{self._get_lport()}"
        with _zmq_bound_endpoints_lock:
            _zmq_bound_endpoints.discard(endpoint)
        if self._handshake_rep is not None:
            try:
                self._handshake_rep.close(linger=0)
            except Exception:
                pass
        try:
            self.zmq_context.term()
        except Exception:
            pass

    def _ensure_pinned_buf(self, tensor: torch.Tensor, is_send: bool):
        """C1: ensure pinned memory buffer is large enough."""
        nbytes = tensor.nelement() * tensor.element_size()
        buf = self._pinned_send_buf if is_send else self._pinned_recv_buf
        if buf is None or buf.nbytes < nbytes:
            buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
            if is_send:
                self._pinned_send_buf = buf
            else:
                self._pinned_recv_buf = buf
        return buf

    def recv_tensor(self) -> torch.Tensor:
        socket = self._get_pull_socket()
        started = time.monotonic()
        seq = self._diag_recv_count
        if _cross_node_experimental_enabled() and seq < 8:
            logger.info("AFD ZMQ recv begin: direction=peer->local rank=%d port=%d seq=%d", self._rank(), self._get_lport(), seq)
        # C1: receive raw bytes then reconstruct tensor
        try:
            metadata = socket.recv_pyobj()
            data = socket.recv(copy=False)
        except zmq.Again:
            raise TimeoutError("AFD ZMQ recv timed out — peer may be dead")
        raw = bytearray(data)
        if len(raw) == 0:
            buf = torch.empty(metadata["shape"], dtype=metadata["dtype"])
        else:
            buf = torch.frombuffer(raw, dtype=metadata["dtype"]).reshape(
                metadata["shape"]
            )
        result = buf.to(self._get_cuda_device(), non_blocking=True)
        self._diag_recv_count += 1
        if _cross_node_experimental_enabled() and seq < 8:
            logger.info("AFD ZMQ recv complete: direction=peer->local rank=%d port=%d seq=%d shape=%s nbytes=%d elapsed_ms=%.1f", self._rank(), self._get_lport(), seq, list(result.shape), len(raw), (time.monotonic() - started) * 1000)
        return result

    def send_tensor(self, x: torch.Tensor):
        socket = self._get_push_socket()
        started = time.monotonic()
        seq = self._diag_send_count
        # C1: send metadata + raw bytes (avoid pickle)
        cpu_tensor = x.detach().contiguous().cpu()
        metadata = {"shape": list(cpu_tensor.shape), "dtype": cpu_tensor.dtype}
        nbytes = cpu_tensor.nelement() * cpu_tensor.element_size()
        if _cross_node_experimental_enabled() and seq < 8:
            logger.info("AFD ZMQ send begin: direction=local->peer rank=%d port=%d seq=%d shape=%s nbytes=%d", self._rank(), self._get_dport(), seq, metadata["shape"], nbytes)
        try:
            socket.send_pyobj(metadata, zmq.SNDMORE)
            if _cross_node_experimental_enabled():
                # Send only the logical payload, excluding unused storage capacity.
                raw_bytes = cpu_tensor.view(torch.uint8).reshape(-1).numpy().tobytes()
                socket.send(raw_bytes)
            else:
                # Preserve the original AFD wire behavior for existing tests.
                socket.send(bytes(cpu_tensor.untyped_storage()))
        except zmq.Again:
            raise TimeoutError("AFD ZMQ send timed out — peer may be dead")
        self._diag_send_count += 1
        if _cross_node_experimental_enabled() and seq < 8:
            logger.info("AFD ZMQ send complete: direction=local->peer rank=%d port=%d seq=%d shape=%s nbytes=%d elapsed_ms=%.1f", self._rank(), self._get_dport(), seq, metadata["shape"], nbytes, (time.monotonic() - started) * 1000)


class StepMeshTensorCache:
    __slots__ = ("push_tensor", "pull_tensor", "push_key", "pull_key", "h")

    def __init__(self):
        self.push_tensor = None
        self.pull_tensor = None
        self.push_key = 0
        self.pull_key = 0
        self.h = None


def _stepmesh_scheduler_process():
    os.environ["DMLC_ROLE"] = "scheduler"
    logger.info(
        "StepMesh scheduler: DMLC_PS_ROOT_URI=%s", os.environ["DMLC_NODE_HOST"]
    )
    import fserver_lib as f

    f.init()
    logger.info("StepMesh scheduler init done.")
    while True:
        time.sleep(10000)


class StepMeshTensorCommunicator(FifoTensorCommunicator):
    """RDMA-based tensor communicator via StepMesh (fserver_lib).

    Supports N:M sharded communication for heterogeneous TP (TP_A != TP_F):
    - A->F: each Attn rank pushes 1/TP_A token shard (broadcast to all FFN);
            FFN concatenates shards from all Workers.
    - F->A: each FFN rank responds its 1/TP_F shard to each Worker via
            separate pull_tensors; Attn concatenates all pull buffers.
    - StepMesh's push_pull natively supports multiple pull_tensors per
      Server, so each Server writes to its own buffer (no data overwrite).

    F9 Grouped StepMesh (--afd-grouped-stepmesh):
    Splits the global N:M StepMesh into gcd(TP_A, TP_F) independent groups,
    each with workers_per_group Workers and servers_per_group Servers on
    separate ports/schedulers. After group communication, NVLink all_gather
    + stride deduplication reconstructs the full tensor.
    Cross-node traffic: NH * (TP_A + TP_F) / gcd (vs NH * (TP_A + TP_F) ungrouped).
    """

    MAX_FREE_BUFFERS = 30

    def __init__(self, afd_perspective: AFDPerspective):
        self.perspective = afd_perspective
        super().__init__()

        # Resolve TP configuration
        server_args = get_global_server_args()
        local_tp = server_args.tp_size
        self.attn_tp = getattr(server_args, "afd_attn_tp", None) or local_tp
        self.ffn_tp = getattr(server_args, "afd_ffn_tp", None) or local_tp
        self.local_rank = dist.get_rank() % local_tp if dist.is_initialized() else 0
        self._heterogeneous = self.attn_tp != self.ffn_tp

        # F9: GCD-based grouped StepMesh
        self._grouped = (
            getattr(server_args, "afd_grouped_stepmesh", False)
            and self._heterogeneous
        )
        if self._grouped:
            from math import gcd

            g = gcd(self.attn_tp, self.ffn_tp)
            self.num_groups = g
            self.workers_per_group = self.attn_tp // g
            self.servers_per_group = self.ffn_tp // g
            if afd_is_attn():
                self.group_id = self.local_rank // self.workers_per_group
                self.intra_group_rank = self.local_rank % self.workers_per_group
            else:
                self.group_id = self.local_rank // self.servers_per_group
                self.intra_group_rank = self.local_rank % self.servers_per_group
            logger.info(
                "Grouped StepMesh: %d groups, %dW:%dS per group, "
                "group_id=%d, intra_rank=%d",
                self.num_groups,
                self.workers_per_group,
                self.servers_per_group,
                self.group_id,
                self.intra_group_rank,
            )

        import fserver_lib as f

        self._start_stepmesh_scheduler()
        time.sleep(10)

        logger.info("%s init...", os.environ["DMLC_ROLE"])
        f.init()
        logger.info("%s init done.", os.environ["DMLC_ROLE"])

        self.worker_num = int(os.environ["DMLC_NUM_WORKER"])
        self.key = 0
        self.gpu = torch.cuda.current_device()
        self.f = f

        self.comm_ids: deque = deque()
        self.waits: deque = deque()
        self.free_tensors: Dict[torch.Size, deque] = {}
        self.register_buf: Dict[int, torch.Tensor] = {}
        self._worker_count_per_recv: deque = deque()
        self._tp_group = None

    def _get_tp_group(self):
        if self._tp_group is None:
            from sglang.srt.distributed import get_tp_group

            self._tp_group = get_tp_group()
        return self._tp_group

    @staticmethod
    def _env_def(env: str, v: str):
        if os.environ.get(env) is None:
            os.environ[env] = v

    @staticmethod
    def _get_node_ip():
        if os.environ.get("DMLC_NODE_HOST") is not None:
            return
        import psutil

        interface_name = os.environ.get("MLC_INTERFACE")
        interfaces = psutil.net_if_addrs()
        if interface_name not in interfaces:
            logger.warning("Invalid MLC_INTERFACE %s", interface_name)
            return
        for addr in interfaces[interface_name]:
            if addr.family == 2:  # socket.AF_INET
                os.environ["DMLC_NODE_HOST"] = addr.address
                break

    def _start_stepmesh_scheduler(self):
        self._get_node_ip()
        gpu = str(torch.cuda.current_device())

        self._env_def("DMLC_NODE_RANK", "0")
        self._env_def("DMLC_GROUP_SIZE", "1")
        self._env_def("DMLC_ENABLE_RDMA", "ibverbs")
        self._env_def("STEPMESH_GPU", gpu)

        if self._grouped:
            # F9: per-group DMLC config — must force-set (not _env_def) because
            # each group needs distinct NUM_WORKER/NUM_SERVER/PORT values,
            # unlike the non-grouped path where all ranks share the same config.
            base_port = int(os.environ.get("DMLC_PS_ROOT_PORT", "8123"))
            os.environ["DMLC_NUM_WORKER"] = str(self.workers_per_group)
            os.environ["DMLC_NUM_SERVER"] = str(self.servers_per_group)
            os.environ["DMLC_PS_ROOT_PORT"] = str(base_port + self.group_id)
        else:
            self._env_def("DMLC_NUM_WORKER", str(self.attn_tp))
            self._env_def("DMLC_NUM_SERVER", str(self.ffn_tp))
            self._env_def("DMLC_PS_ROOT_PORT", "8123")

        if afd_is_attn():
            os.environ["DMLC_ROLE"] = "worker"
        else:
            os.environ["DMLC_ROLE"] = "server"

        if os.environ["DMLC_ROLE"] != "worker":
            return
        if os.environ.get("DMLC_NODE_RANK") != "0":
            return

        if self._grouped:
            # F9: each group's first Worker starts a scheduler
            if self.intra_group_rank != 0:
                return
        else:
            if os.environ["STEPMESH_GPU"] != "0":
                return
            if os.environ.get("STEPMESH_SCHEDULER_STARTED") == "1":
                return
            os.environ["STEPMESH_SCHEDULER_STARTED"] = "1"

        os.environ["DMLC_PS_ROOT_URI"] = os.environ["DMLC_NODE_HOST"]

        p = multiprocessing.Process(target=_stepmesh_scheduler_process)
        p.daemon = True
        p.start()
        logger.info(
            "StepMesh scheduler started (grouped=%s, group_id=%s, port=%s)",
            self._grouped,
            getattr(self, "group_id", None),
            os.environ.get("DMLC_PS_ROOT_PORT"),
        )

    def _get_or_create_free_deque(self, shape: torch.Size) -> deque:
        q = self.free_tensors.get(shape)
        if q is None:
            q = deque()
            self.free_tensors[shape] = q
        return q

    # --- A->F: Attn sends shard, FFN receives and concatenates ---

    def attn_send(self, x: torch.Tensor):
        if not self._heterogeneous:
            # Homogeneous: original 1:1 behavior
            free = self._get_or_create_free_deque(x.shape)
            if len(free) < self.MAX_FREE_BUFFERS:
                self.key += 2
                t = StepMeshTensorCache()
                t.push_tensor = torch.empty_like(x)
                t.pull_tensor = torch.empty_like(x)
                t.push_key = self.key
                t.pull_key = self.key + 1
            else:
                t = free.popleft()
            t.push_tensor.copy_(x)
            t.h = self.f.push_pull(
                [t.push_tensor], [t.push_key], [t.pull_tensor], [t.pull_key]
            )
            self.waits.append(t)
            return

        # Heterogeneous N:M sharded communication:
        # Push: only this rank's 1/TP_A token shard (broadcast to all FFN)
        # Pull: TP_F separate pull buffers (one per FFN Server, each receives a shard)
        num_tokens = x.shape[0]
        chunk_size = (num_tokens + self.attn_tp - 1) // self.attn_tp
        start = self.local_rank * chunk_size
        end_idx = min(start + chunk_size, num_tokens)
        push_shard = x[start:end_idx].contiguous()

        # F9: pad push shard to chunk_size for uniform all_gather sizing (grouped only)
        if self._grouped and push_shard.shape[0] < chunk_size:
            pad = torch.zeros(
                chunk_size - push_shard.shape[0], x.shape[1],
                dtype=x.dtype, device=x.device,
            )
            push_shard = torch.cat([push_shard, pad], dim=0)

        # F9: grouped path — pull from servers_per_group servers (not all TP_F)
        n_pull = self.servers_per_group if self._grouped else self.ffn_tp
        # In grouped mode, FFN processes padded tensor (attn_tp * chunk_size tokens),
        # so pull buffers must be sized for the padded output, not original num_tokens.
        effective_tokens = self.attn_tp * chunk_size if self._grouped else num_tokens
        f2a_chunk = (effective_tokens + self.ffn_tp - 1) // self.ffn_tp
        pull_tensors = []
        pull_keys = []
        self.key += 1 + n_pull
        push_key = self.key
        for s in range(n_pull):
            pull_buf = torch.empty(f2a_chunk, x.shape[1], dtype=x.dtype, device=x.device)
            pull_tensors.append(pull_buf)
            pull_keys.append(push_key + 1 + s)

        push_tensor_buf = torch.empty_like(push_shard)
        push_tensor_buf.copy_(push_shard)

        h = self.f.push_pull(
            [push_tensor_buf], [push_key],
            pull_tensors, pull_keys,
        )
        # Store handle + pull buffers + original num_tokens for attn_recv
        self.waits.append((h, pull_tensors, num_tokens))

    def ffn_recv(self) -> torch.Tensor:
        # get_batch returns batches in Worker rank order
        batches = self.f.get_batch()
        self._worker_count_per_recv.append(len(batches))

        if len(batches) == 1 and not self._heterogeneous:
            # Homogeneous: single Worker, original behavior
            x = batches[0][1][0]
            key = batches[0][2][0]
            self.comm_ids.append(batches[0][0])
            if self.register_buf.get(key) is None:
                y = torch.empty_like(x)
                self.f.register_recv_buffer(y, [0], [key])
                self.register_buf[key] = y
            return x.clone()

        # Heterogeneous: collect shards from all Workers (already in rank order)
        shards = []
        for batch in batches:
            self.comm_ids.append(batch[0])
            shards.append(batch[1][0])
            # Register recv buffer for each Worker's key
            key = batch[2][0]
            if self.register_buf.get(key) is None:
                y = torch.empty_like(batch[1][0])
                self.f.register_recv_buffer(y, [0], [key])
                self.register_buf[key] = y

        local_data = torch.cat(shards, dim=0)

        if not self._grouped:
            return local_data

        # F9: grouped path — local_data has workers_per_group shards (partial tensor).
        # attn_send pads all shards to uniform chunk_size, so all FFN ranks have
        # the same local_data size → all_gather requires no additional padding.
        # Truncation to original num_tokens is handled by attn_recv on the Attn side.
        tp_group = self._get_tp_group()
        gathered = [torch.empty_like(local_data) for _ in range(self.ffn_tp)]
        dist.all_gather(gathered, local_data, group=tp_group.device_group)

        # Stride-select: ranks within the same group have identical data
        stride = self.servers_per_group
        unique = [gathered[i] for i in range(0, self.ffn_tp, stride)]
        return torch.cat(unique, dim=0)

    # --- F->A: FFN responds its shard, Attn concatenates from pull buffers ---

    def ffn_send(self, x: torch.Tensor):
        n_workers = self._worker_count_per_recv.popleft()

        if n_workers == 1 and not self._heterogeneous:
            # Homogeneous: single Worker, original behavior
            free = self._get_or_create_free_deque(x.shape)
            if len(free) < self.MAX_FREE_BUFFERS:
                t = torch.empty_like(x)
            else:
                t = free.popleft()
            t.copy_(x)
            c = self.comm_ids.popleft()
            self.f.respond([t], c, True)
            free.append(t)
            return

        # Heterogeneous: each FFN rank responds its own shard to each Worker.
        # Each Worker's push_pull registered TP_F pull_keys. This FFN Server
        # responds with its shard matched by key.
        num_tokens = x.shape[0]
        chunk_size = (num_tokens + self.ffn_tp - 1) // self.ffn_tp
        ffn_rank = self.local_rank
        start = ffn_rank * chunk_size
        end_idx = min(start + chunk_size, num_tokens)
        my_shard = x[start:end_idx].contiguous()

        # G3 fix: pad last shard to chunk_size so respond size == pull_buf size
        if my_shard.shape[0] < chunk_size:
            pad = torch.zeros(
                chunk_size - my_shard.shape[0], x.shape[1],
                dtype=x.dtype, device=x.device,
            )
            my_shard = torch.cat([my_shard, pad], dim=0)

        for _ in range(n_workers):
            c = self.comm_ids.popleft()
            free = self._get_or_create_free_deque(my_shard.shape)
            if len(free) < self.MAX_FREE_BUFFERS:
                t = torch.empty_like(my_shard)
            else:
                t = free.popleft()
            t.copy_(my_shard)
            self.f.respond([t], c, True)
            free.append(t)

    def attn_recv(self) -> torch.Tensor:
        if not self._heterogeneous:
            # Homogeneous: original behavior
            t = self.waits.popleft()
            self.f.wait(t.h)
            self._get_or_create_free_deque(t.push_tensor.shape).append(t)
            return t.pull_tensor.clone()

        # Heterogeneous: pull buffers already have data from Servers
        h, pull_tensors, original_num_tokens = self.waits.popleft()
        self.f.wait(h)
        local_data = torch.cat(pull_tensors, dim=0)

        if not self._grouped:
            return local_data[:original_num_tokens]

        # F9: grouped path — local_data has servers_per_group pull shards (partial).
        # all_gather across full TP_A group, then stride-select to remove duplicates.
        tp_group = self._get_tp_group()
        gathered = [torch.empty_like(local_data) for _ in range(self.attn_tp)]
        dist.all_gather(gathered, local_data, group=tp_group.device_group)

        # Stride-select: ranks within the same group have identical data
        stride = self.workers_per_group
        unique = [gathered[i] for i in range(0, self.attn_tp, stride)]
        full = torch.cat(unique, dim=0)
        return full[:original_num_tokens]

    # --- Public interface ---

    def recv_tensor(self) -> torch.Tensor:
        if self.perspective == AFDPerspective.AFD_PERSPECTIVE_ATTN:
            return self.attn_recv()
        return self.ffn_recv()

    def send_tensor(self, x: torch.Tensor):
        if self.perspective == AFDPerspective.AFD_PERSPECTIVE_ATTN:
            self.attn_send(x)
        else:
            self.ffn_send(x)


# --------------- TP-aware sharded ZMQ communicator ---------------


def _shard_tensor_for_rank(
    x: torch.Tensor, rank: int, group_size: int
) -> Tuple[torch.Tensor, int]:
    """Split a replicated AFD boundary tensor on tokens and pad uniformly."""
    if x.ndim < 1:
        raise ValueError("AFD ZMQ sharding requires a tensor with a token dimension")
    if group_size < 1 or not 0 <= rank < group_size:
        raise ValueError(f"Invalid sharding rank/group: {rank}/{group_size}")
    original_num_tokens = x.shape[0]
    chunk = (original_num_tokens + group_size - 1) // group_size
    start = rank * chunk
    shard = x[start : min(start + chunk, original_num_tokens)].contiguous()
    if shard.shape[0] < chunk:
        pad = torch.zeros(
            chunk - shard.shape[0], *x.shape[1:], dtype=x.dtype, device=x.device
        )
        shard = torch.cat((shard, pad), dim=0)
    return shard, original_num_tokens


def _reassemble_tensor_shards(
    shards: List[torch.Tensor], original_num_tokens: int
) -> torch.Tensor:
    """Rank-order concatenate uniformly padded token shards and crop padding."""
    if not shards:
        raise ValueError("Cannot reassemble an empty shard list")
    return torch.cat(shards, dim=0)[:original_num_tokens]


class _ZMQBufferSlot:
    def __init__(self):
        self.buffer = None
        self.capacity = 0
        self.sequence = -1
        self.event = None
        self.metadata = None
        self.error = None
        self.free = threading.Event()
        self.free.set()

    def resize(self, nbytes):
        if self.buffer is None or self.capacity < nbytes:
            self.buffer = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
            self.capacity = nbytes
        return self.buffer

    def acquire(self, sequence):
        self.free.wait()
        if self.error:
            error, self.error = self.error, None
            raise error
        self.free.clear()
        self.sequence = sequence


class ShardedZMQTensorCommunicator(FifoTensorCommunicator):
    """Per-rank ZMQ sharding, optionally with two reusable pinned slots."""
    SLOT_COUNT = 2

    def __init__(self, afd_perspective, local_tp_size, local_tp_rank, inner_comm=None):
        super().__init__()
        self.local_tp_size, self.local_tp_rank = local_tp_size, local_tp_rank
        self.inner_comm = inner_comm or ZMQSimpleTensorCommunicator(afd_perspective)
        self._tp_group = None
        self._double_buffer = _zmq_double_buffer_enabled()
        self._send_seq = self._recv_seq = 0
        self._stats = {k: 0.0 for k in ("d2h_ms", "network_send_ms", "network_recv_ms", "h2d_ms", "messages")}
        self._stats_lock = threading.Lock()
        if self._double_buffer:
            self._stream = torch.cuda.Stream() if torch.cuda.is_available() else None
            self._send_slots = [_ZMQBufferSlot() for _ in range(2)]
            self._recv_slots = [_ZMQBufferSlot() for _ in range(2)]
            self._send_queue, self._recv_ready = queue.Queue(), queue.Queue(maxsize=2)
            self._closing = False
            self._sender = threading.Thread(target=self._send_worker, daemon=True)
            self._receiver = threading.Thread(target=self._recv_worker, daemon=True)
            self._sender.start(); self._receiver.start()
        logger.info("ShardedZMQ ready rank=%d K=%d double_buffer=%s", local_tp_rank, local_tp_size, self._double_buffer)

    def _get_tp_group(self):
        if self._tp_group is None:
            from sglang.srt.distributed import get_tp_group
            self._tp_group = get_tp_group()
        return self._tp_group

    def _add_stats(self, **values):
        with self._stats_lock:
            for key, value in values.items(): self._stats[key] += value
            if values.get("messages") and int(self._stats["messages"]) % 64 == 0:
                logger.info("AFD ZMQ transfer timing rank=%d %s", self.local_tp_rank, self._stats)

    def timing_stats(self):
        with self._stats_lock: return dict(self._stats)

    def _send_worker(self):
        socket = self.inner_comm._get_push_socket()
        while True:
            slot = self._send_queue.get()
            if slot is None: return
            try:
                t0 = time.perf_counter()
                if slot.event: slot.event.synchronize()
                t1 = time.perf_counter()
                socket.send_pyobj(slot.metadata, zmq.SNDMORE)
                socket.send(slot.buffer[:slot.metadata["nbytes"]].numpy(), copy=True)
                t2 = time.perf_counter()
                self._add_stats(d2h_ms=(t1-t0)*1000, network_send_ms=(t2-t1)*1000, messages=1)
            except Exception as exc: slot.error = exc
            finally: slot.free.set()

    def _recv_worker(self):
        socket = self.inner_comm._get_pull_socket(); expected = 0
        while not self._closing:
            slot = self._recv_slots[expected % 2]; slot.free.wait(); slot.free.clear()
            try:
                t0 = time.perf_counter(); metadata = socket.recv_pyobj(); data = socket.recv(copy=False); t1 = time.perf_counter()
                if metadata.get("sequence") != expected: raise RuntimeError(f"AFD ZMQ sequence mismatch: {metadata.get('sequence')} != {expected}")
                raw = bytearray(data); slot.resize(len(raw))[:len(raw)].copy_(torch.frombuffer(raw, dtype=torch.uint8))
                slot.metadata, slot.sequence = metadata, expected
                self._add_stats(network_recv_ms=(t1-t0)*1000)
                self._recv_ready.put(slot); expected += 1
            except Exception as exc:
                slot.error = exc; self._recv_ready.put(slot); return

    def send_tensor_nonblocking(self, x, compute_event=None):
        self.send_tensor(x)

    def send_tensor(self, x):
        shard, tokens = _shard_tensor_for_rank(x, self.local_tp_rank, self.local_tp_size)
        if not self._double_buffer:
            cpu = shard.detach().contiguous().cpu(); socket = self.inner_comm._get_push_socket()
            metadata = {"shape": list(cpu.shape), "dtype": cpu.dtype, "original_num_tokens": tokens, "group_size": self.local_tp_size, "shard_rank": self.local_tp_rank}
            socket.send_pyobj(metadata, zmq.SNDMORE); socket.send(cpu.view(torch.uint8).reshape(-1).numpy().tobytes()); return
        sequence = self._send_seq; slot = self._send_slots[sequence % 2]; slot.acquire(sequence)
        shard = shard.detach().contiguous(); nbytes = shard.nelement() * shard.element_size(); host = slot.resize(nbytes)
        slot.metadata = {"shape": list(shard.shape), "dtype": shard.dtype, "nbytes": nbytes, "original_num_tokens": tokens, "group_size": self.local_tp_size, "shard_rank": self.local_tp_rank, "sequence": sequence}
        if self._stream:
            self._stream.wait_stream(torch.cuda.current_stream(shard.device))
            with torch.cuda.stream(self._stream):
                host[:nbytes].copy_(shard.view(torch.uint8).reshape(-1), non_blocking=True)
                slot.event = torch.cuda.Event(); slot.event.record(self._stream)
        else:
            host[:nbytes].copy_(shard.view(torch.uint8).reshape(-1)); slot.event = None
        self._send_seq += 1; self._send_queue.put(slot)

    def recv_tensor(self):
        if not self._double_buffer:
            socket = self.inner_comm._get_pull_socket(); metadata = socket.recv_pyobj(); raw = bytearray(socket.recv(copy=False))
            shard = torch.frombuffer(raw, dtype=metadata["dtype"]).reshape(metadata["shape"]).to(self.inner_comm._get_cuda_device(), non_blocking=True)
        else:
            slot = self._recv_ready.get()
            if slot.error: error, slot.error = slot.error, None; raise error
            metadata = slot.metadata
            if slot.sequence != self._recv_seq: raise RuntimeError("AFD ZMQ local slot ordering violation")
            host = slot.buffer[:metadata["nbytes"]].view(metadata["dtype"]).reshape(metadata["shape"]); t0 = time.perf_counter()
            if self._stream:
                with torch.cuda.stream(self._stream):
                    shard = host.to(self.inner_comm._get_cuda_device(), non_blocking=True); event = torch.cuda.Event(); event.record(self._stream)
                event.synchronize()
            else: shard = host.clone()
            self._add_stats(h2d_ms=(time.perf_counter()-t0)*1000); self._recv_seq += 1; slot.free.set()
        if metadata.get("group_size") != self.local_tp_size or metadata.get("shard_rank") != self.local_tp_rank: raise RuntimeError("AFD ZMQ shard metadata mismatch")
        gathered = [torch.empty_like(shard) for _ in range(self.local_tp_size)]
        dist.all_gather(gathered, shard, group=self._get_tp_group().device_group)
        return _reassemble_tensor_shards(gathered, metadata["original_num_tokens"])

    def close(self):
        if self._double_buffer:
            for slot in self._send_slots: slot.free.wait()
            self._closing = True; self._send_queue.put(None); self._sender.join(timeout=5)
            logger.info("AFD ZMQ final transfer timing rank=%d %s", self.local_tp_rank, self.timing_stats())
        self.inner_comm.close()

    send_stream_ordered = send_tensor
    recv_stream_ordered = recv_tensor


# --------------- TP-aware communicator (rank-0 ZMQ + NVLink broadcast) ---------------


class BroadcastTensorCommunicator(FifoTensorCommunicator):
    """TP-aware communicator: rank 0 does ZMQ, other ranks get data via broadcast.

    Works for both homogeneous and heterogeneous TP (any N:M).
    After all-reduce each TP rank holds the identical full tensor, so only
    one rank needs to send/recv cross-node.  Other ranks reconstruct via
    NVLink broadcast (negligible latency).

    Cross-node traffic: 2NH per layer (independent of TP sizes).
    """

    def __init__(
        self,
        inner_comm: FifoTensorCommunicator,
        local_tp_size: int,
        local_tp_rank: int,
    ):
        super().__init__()
        self.inner_comm = inner_comm
        self.local_tp_size = local_tp_size
        self.local_tp_rank = local_tp_rank
        self._tp_group = None

    def _get_tp_group(self):
        if self._tp_group is None:
            from sglang.srt.distributed import get_tp_group

            self._tp_group = get_tp_group()
        return self._tp_group

    def send_tensor(self, x: torch.Tensor):
        if self.local_tp_rank == 0:
            self.inner_comm.send_tensor(x)

    def recv_tensor(self) -> torch.Tensor:
        if self.local_tp_rank == 0:
            tensor = self.inner_comm.recv_tensor()
        else:
            tensor = None

        if self.local_tp_size <= 1:
            return tensor

        tp_group = self._get_tp_group()

        # Broadcast shape + dtype so non-rank-0 can allocate
        if self.local_tp_rank == 0:
            shape_dtype = torch.tensor(
                [tensor.shape[0], tensor.shape[1], _dtype_to_int(tensor.dtype)],
                dtype=torch.long,
                device=tensor.device,
            )
        else:
            shape_dtype = torch.empty(3, dtype=torch.long, device=f"cuda:{torch.cuda.current_device()}")

        dist.broadcast(shape_dtype, src=tp_group.ranks[0], group=tp_group.device_group)

        if self.local_tp_rank != 0:
            tensor = torch.empty(
                int(shape_dtype[0].item()),
                int(shape_dtype[1].item()),
                dtype=_int_to_dtype(int(shape_dtype[2].item())),
                device=f"cuda:{torch.cuda.current_device()}",
            )

        dist.broadcast(tensor, src=tp_group.ranks[0], group=tp_group.device_group)
        return tensor

    # Aliases for compatibility with code paths that call stream-ordered variants
    send_stream_ordered = send_tensor
    recv_stream_ordered = recv_tensor


# dtype <-> int mapping for broadcasting tensor metadata
_DTYPE_MAP = {
    torch.float16: 0,
    torch.bfloat16: 1,
    torch.float32: 2,
    torch.float64: 3,
    torch.int32: 4,
    torch.int64: 5,
}
_INT_TO_DTYPE = {v: k for k, v in _DTYPE_MAP.items()}


def _dtype_to_int(dtype: torch.dtype) -> int:
    result = _DTYPE_MAP.get(dtype)
    if result is None:
        raise ValueError(
            f"Unsupported dtype {dtype} for AFD tensor transfer. "
            f"Supported: {list(_DTYPE_MAP.keys())}"
        )
    return result


def _int_to_dtype(i: int) -> torch.dtype:
    result = _INT_TO_DTYPE.get(i)
    if result is None:
        raise ValueError(
            f"Unknown dtype code {i} in AFD tensor metadata. "
            f"Known codes: {list(_INT_TO_DTYPE.keys())}"
        )
    return result


# --------------- Async communication wrapper (C2 optimization) ---------------


class AsyncTensorCommunicator:
    """Wraps a FifoTensorCommunicator to overlap communication with computation.

    send_async: queues the send on a background thread or CUDA stream;
        returns immediately.
    recv_start: launches a background daemon thread for the blocking
        inner.recv_tensor() call; returns immediately so the CPU pipeline
        loop can continue launching compute.
    recv_wait: joins the background thread, fences sends, and synchronizes
        the comm stream event so the received tensor is ready on the GPU.
    """

    _RING_SIZE = 3  # 3BO: support up to 3 concurrent recvs

    def __init__(
        self,
        inner: FifoTensorCommunicator,
        *,
        allow_background_recv: bool = False,
    ):
        self.inner = inner
        self._allow_background_recv = allow_background_recv
        self.comm_stream = (
            torch.cuda.Stream() if torch.cuda.is_available() else None
        )
        # Per-slot CUDA streams for truly parallel recv GPU transfers
        self._recv_streams: list = [
            torch.cuda.Stream() if torch.cuda.is_available() else None
            for _ in range(self._RING_SIZE)
        ]
        # 3BO ring buffer for concurrent recvs
        self._recv_ring: list = [None] * self._RING_SIZE
        self._recv_event_ring: list = [None] * self._RING_SIZE
        self._recv_threads: list = [None] * self._RING_SIZE
        self._recv_idx_write: int = 0
        self._recv_idx_read: int = 0
        self._pending_recv_count: int = 0
        # Sends — persistent sender daemon with queue + pre-allocated event pool
        self._send_thread: Optional[object] = None
        self._pending_sends: list = []  # legacy compat
        self._send_futures: list = []  # UCX-level send futures for fence
        self._send_queue: Optional[queue.Queue] = None
        self._sender_daemon: Optional[threading.Thread] = None
        self._sender_daemon_started = False
        # Pre-allocated CUDA event pool to avoid per-send object creation (~15μs)
        self._event_pool_size = 16
        self._event_pool: list = []
        self._event_pool_idx = 0
        self._available_events: queue.SimpleQueue = queue.SimpleQueue()
        if torch.cuda.is_available():
            self._event_pool = [
                torch.cuda.Event(enable_timing=False)
                for _ in range(self._event_pool_size)
            ]
            for event in self._event_pool:
                self._available_events.put(event)

    # Backward-compat properties for code that checks single-recv state.
    # With async recv, the ring slot may still be None while the background
    # thread is running — return the thread as a truthy sentinel so callers
    # that check "is not None" correctly detect a pending recv.
    @property
    def _pending_recv(self):
        if self._pending_recv_count > 0:
            idx = self._recv_idx_read
            thread = self._recv_threads[idx]
            if thread is not None and thread.is_alive():
                return thread  # truthy sentinel
            return self._recv_ring[idx]
        return None

    @property
    def _recv_event(self):
        return self._recv_event_ring[self._recv_idx_read] if self._pending_recv_count > 0 else None

    @torch.compiler.disable()
    def send_async(self, x: torch.Tensor):
        """Near-zero-cost send via pre-allocated event pool + persistent daemon.

        Main thread cost: ~3-5μs (reuse event from pool + queue.put).
        No Python object creation, no profiling overhead in hot path.
        """
        if self.comm_stream is not None and hasattr(self.inner, "send_tensor_nonblocking"):
            # Reuse only events released by the sender daemon. A modulo
            # index is unsafe when the producer gets more than 16 sends ahead:
            # re-recording an in-flight CUDA event changes what synchronize()
            # waits for and can stall or corrupt stream ordering.
            try:
                ev = self._available_events.get_nowait()
            except queue.Empty:
                ev = torch.cuda.Event(enable_timing=False)
            ev.record()  # record on current compute stream

            # Ensure persistent sender daemon is running
            if not self._sender_daemon_started:
                self._start_sender_daemon()

            # Fire-and-forget: enqueue (tensor, event)
            # No profiling in hot path — profiling moved to daemon thread
            self._send_queue.put_nowait((x, ev))
        elif self.comm_stream is not None and hasattr(self.inner, "send_tensor_stream"):
            # IPC path: submit GPU copy on comm_stream, flag_write deferred to daemon.
            # Main thread cost: ~5μs (record_event + wait_event + submit copy commands)
            ev = self._event_pool[self._event_pool_idx]
            self._event_pool_idx = (self._event_pool_idx + 1) % self._event_pool_size
            ev.record()
            self.comm_stream.wait_event(ev)
            self.inner.send_tensor_stream(x, self.comm_stream)
        elif self.comm_stream is not None:
            ev = self._event_pool[self._event_pool_idx]
            self._event_pool_idx = (self._event_pool_idx + 1) % self._event_pool_size
            ev.record()
            with torch.cuda.stream(self.comm_stream):
                self.comm_stream.wait_event(ev)
                self.inner.send_tensor(x)
        else:
            self.inner.send_tensor(x)

    def _start_sender_daemon(self):
        """Start the persistent sender daemon thread."""
        from sglang.srt.layers.afd_mixin import _afd_host_events

        self._send_queue = queue.Queue()

        def _sender_loop():
            while True:
                item = self._send_queue.get()
                if item is None:
                    self._send_queue.task_done()
                    break  # Shutdown signal
                x, compute_event = item
                compute_event.synchronize()
                # IPC needs compute_event for GPU-level stream ordering;
                # UCX ignores it (already CPU-synced above).
                try:
                    self.inner.send_tensor_nonblocking(x, compute_event=compute_event)
                except TypeError:
                    # Fallback for backends that don't accept compute_event
                    self.inner.send_tensor_nonblocking(x)
                future = getattr(self.inner, "_last_send_future", None)
                if future is not None:
                    # ctypes.from_address does not own the CUDA allocation.
                    # Capture x until UCX reports completion so PyTorch cannot
                    # recycle its storage while RDMA is still in flight.
                    future.add_done_callback(lambda _future, _x=x: None)
                    self._send_futures.append(future)
                self._available_events.put(compute_event)
                self._send_queue.task_done()

        self._sender_daemon = threading.Thread(
            target=_sender_loop, daemon=True, name="ucx-persistent-sender",
        )
        self._sender_daemon.start()
        self._sender_daemon_started = True

    @torch.compiler.disable()
    def recv_start(self):
        """3BO: enqueue a recv into the ring buffer.

        Runs the blocking inner.recv_tensor() in a background daemon thread so
        the CPU pipeline loop can continue.  GPU work is still issued on the
        comm_stream, isolated from the compute stream.

        Can be called up to RING_SIZE times before recv_wait drains slots.

        NOTE: For CppIPC backend, uses synchronous recv in the calling thread
        to avoid race conditions on the C++ recv_slot_ counter.
        """
        from sglang.srt.layers.afd_mixin import _afd_host_events, _afd_ctx

        idx = self._recv_idx_write
        _prof_layer = _afd_ctx.get("layer", -1)
        _prof_mb = _afd_ctx.get("mb", -1)

        def _do_recv_sync():
            """Synchronous recv - used for CppIPC to avoid recv_slot_ races."""
            t0 = time.time()
            tensor = self.inner.recv_tensor()
            t1 = time.time()
            _afd_host_events.append({
                "ts_ms": round(t0 * 1000, 3),
                "role": "UCX_PROFILE", "layer": _prof_layer, "mb": _prof_mb,
                "event": "recv_start_detail",
                "recv_inner_us": round((t1 - t0) * 1e6, 1),
                "phase": "sync_ipc",
            })
            self._recv_ring[idx] = tensor
            self._recv_event_ring[idx] = None

        # Backends can explicitly reject concurrent receives. Keep the legacy
        # _recv_lock signal for communicators that have not declared capability.
        supports_concurrent_recv = getattr(
            self.inner,
            "supports_concurrent_recv",
            not hasattr(self.inner, "_recv_lock"),
        )
        _use_sync_recv = not supports_concurrent_recv and not self._allow_background_recv
        if _use_sync_recv:
            _do_recv_sync()
            self._recv_threads[idx] = None
            self._recv_idx_write = (self._recv_idx_write + 1) % self._RING_SIZE
            self._pending_recv_count += 1
            return

        def _deferred_recv(_l=_prof_layer, _m=_prof_mb):
            t0 = time.time()
            if hasattr(self.inner, "recv_poll"):
                slot_info = self.inner.recv_poll()
                event = None
                t1 = time.time()
                _afd_host_events.append({
                    "ts_ms": round(t0 * 1000, 3),
                    "role": "UCX_PROFILE", "layer": _l, "mb": _m,
                    "event": "recv_start_detail",
                    "recv_inner_us": round((t1 - t0) * 1e6, 1),
                    "phase": "poll_only",
                })
                self._recv_ring[idx] = slot_info
                self._recv_event_ring[idx] = None
            elif (
                hasattr(self.inner, "recv_rdma_only")
                and getattr(self.inner, "_split_phase_recv", False)
                and getattr(self.inner, "_local_tp", 1) > 1
            ):
                # K=1 + TP>1: only RDMA recv in the background. Keep the
                # payload in this ring slot; all ranks enter NCCL broadcast
                # together from recv_wait on their main threads.
                try:
                    tensor = self.inner.recv_rdma_only()
                    result = ("rdma_done", tensor)
                except BaseException as exc:
                    result = ("rdma_error", exc)
                t1 = time.time()
                _afd_host_events.append({
                    "ts_ms": round(t0 * 1000, 3),
                    "role": "UCX_PROFILE", "layer": _l, "mb": _m,
                    "event": "recv_start_detail",
                    "recv_inner_us": round((t1 - t0) * 1e6, 1),
                    "phase": "rdma_only",
                })
                self._recv_ring[idx] = result
                self._recv_event_ring[idx] = None
            elif self.comm_stream is not None:
                # Use per-slot stream so multiple recvs can have their
                # GPU memcpy in parallel (no serialization on a single stream).
                slot_stream = self._recv_streams[idx]
                with torch.cuda.stream(slot_stream):
                    tensor = self.inner.recv_tensor()
                    event = slot_stream.record_event()
                t1 = time.time()
                _afd_host_events.append({
                    "ts_ms": round(t0 * 1000, 3),
                    "role": "UCX_PROFILE", "layer": _l, "mb": _m,
                    "event": "recv_start_detail",
                    "recv_inner_us": round((t1 - t0) * 1e6, 1),
                })
                self._recv_ring[idx] = tensor
                self._recv_event_ring[idx] = event
            else:
                tensor = self.inner.recv_tensor()
                event = None
                t1 = time.time()
                _afd_host_events.append({
                    "ts_ms": round(t0 * 1000, 3),
                    "role": "UCX_PROFILE", "layer": _l, "mb": _m,
                    "event": "recv_start_detail",
                    "recv_inner_us": round((t1 - t0) * 1e6, 1),
                })
                self._recv_ring[idx] = tensor
                self._recv_event_ring[idx] = event

        # Use a new thread for each recv (simple, no deadlock risk)
        thread = threading.Thread(
            target=_deferred_recv, daemon=True, name="ucx-deferred-recv",
        )
        thread.start()
        self._recv_threads[idx] = thread
        self._recv_idx_write = (self._recv_idx_write + 1) % self._RING_SIZE
        self._pending_recv_count += 1

    def _release_recv_slot(self, idx: int):
        self._recv_ring[idx] = None
        self._recv_event_ring[idx] = None
        self._recv_threads[idx] = None
        self._recv_idx_read = (self._recv_idx_read + 1) % self._RING_SIZE
        self._pending_recv_count -= 1

    @torch.compiler.disable()
    def recv_wait(self) -> torch.Tensor:
        """3BO: drain the oldest pending recv from the ring."""
        from sglang.srt.layers.afd_mixin import _afd_host_events, _afd_ctx

        if self._pending_recv_count == 0:
            raise RuntimeError("recv_wait called with no pending recv (drain mismatch)")
        idx = self._recv_idx_read
        _prof_layer = _afd_ctx.get("layer", -1)
        _prof_mb = _afd_ctx.get("mb", -1)

        # Wait for the background recv thread to finish
        thread = self._recv_threads[idx]
        t0 = time.time()
        if thread is not None:
            thread.join(timeout=30)
            if thread.is_alive():
                raise RuntimeError(
                    f"recv_wait: background recv thread timed out after 30s "
                    f"(slot={idx}, pending_count={self._pending_recv_count}). "
                    f"The remote node may be unresponsive."
                )
            self._recv_threads[idx] = None
        t1 = time.time()

        ev = self._recv_event_ring[idx]
        data = self._recv_ring[idx]

        # K=1 + TP>1 split phase: RDMA completed in the background;
        # broadcast is deliberately issued here by every TP rank.
        if isinstance(data, tuple) and data and data[0] == "rdma_error":
            self._release_recv_slot(idx)
            raise RuntimeError("background GPU-Direct receive failed") from data[1]
        if isinstance(data, tuple) and data and data[0] == "rdma_done":
            # A response cannot arrive before the peer consumed our request,
            # but drain the local submit queue so its tensor/event lifetime is
            # no longer owned by the sender daemon. The UCX future is already
            # complete in the normal request/response path.
            if self._send_queue is not None:
                self._send_queue.join()
            if hasattr(self.inner, "fence"):
                self.inner.fence()
            t_bcast = time.time()
            tensor = self.inner.recv_broadcast(data[1], slot=idx)
            t_bcast_end = time.time()
            _afd_host_events.append({
                "ts_ms": round(t_bcast * 1000, 3),
                "role": "UCX_PROFILE", "layer": _prof_layer, "mb": _prof_mb,
                "event": "recv_broadcast",
                "thread_wait_us": round((t1 - t0) * 1e6, 1),
                "broadcast_us": round((t_bcast_end - t_bcast) * 1e6, 1),
            })
            self._release_recv_slot(idx)
            return tensor

        # 2-phase IPC: bg thread polled flag only, main thread does GPU copy
        if hasattr(self.inner, "recv_complete") and isinstance(data, tuple):
            t_phase2 = time.time()
            slot, total_bytes = data
            if self.comm_stream is not None:
                with torch.cuda.stream(self.comm_stream):
                    tensor = self.inner.recv_complete(slot, total_bytes)
                    ev = self.comm_stream.record_event()
            else:
                tensor = self.inner.recv_complete(slot, total_bytes)
                ev = None
            t_phase2_end = time.time()
            _afd_host_events.append({
                "ts_ms": round(t_phase2 * 1000, 3),
                "role": "UCX_PROFILE", "layer": _prof_layer, "mb": _prof_mb,
                "event": "recv_complete_detail",
                "recv_complete_us": round((t_phase2_end - t_phase2) * 1e6, 1),
                "phase": "gpu_copy",
            })
        else:
            tensor = data

        if self._send_thread is not None:
            # Legacy: join per-send thread (not used with persistent daemon)
            self._send_thread.join(timeout=30)
            self._send_thread = None
        t2 = time.time()
        if hasattr(self.inner, "fence"):
            self.inner.fence()
        t3 = time.time()
        if ev is not None:
            ev.synchronize()
        t4 = time.time()
        _afd_host_events.append({
            "ts_ms": round(t0 * 1000, 3),
            "role": "UCX_PROFILE", "layer": _prof_layer, "mb": _prof_mb,
            "event": "recv_wait_detail",
            "recv_thread_join_us": round((t1 - t0) * 1e6, 1),
            "send_thread_join_us": round((t2 - t1) * 1e6, 1),
            "fence_us": round((t3 - t2) * 1e6, 1),
            "cuda_sync_us": round((t4 - t3) * 1e6, 1),
        })

        self._release_recv_slot(idx)
        return tensor

    @torch.compiler.disable()
    def drain_sends(self):
        """Wait for all in-flight sends to complete."""
        # With persistent sender daemon: drain the queue by waiting for it to empty
        if self._send_queue is not None:
            self._send_queue.join()  # blocks until all queued items are processed
        # Legacy: join any old-style per-send threads
        for t in self._pending_sends:
            try:
                t.join(timeout=30)
            except Exception:
                pass
        self._pending_sends.clear()
        self._send_thread = None
        # Wait for all captured UCX-level send Futures.
        for future in self._send_futures:
            try:
                future.result(timeout=30)
            except Exception:
                pass
        self._send_futures.clear()
        # Final fence to flush any remaining UCX operations.
        if hasattr(self.inner, "fence"):
            try:
                self.inner.fence()
            except Exception:
                pass

    @torch.compiler.disable()
    def drain_recvs(self):
        """3BO: discard any pending recvs. Call on error/reset."""
        for i in range(self._RING_SIZE):
            thread = self._recv_threads[i]
            if thread is not None and thread.is_alive():
                thread.join(timeout=5)
        self._recv_ring = [None] * self._RING_SIZE
        self._recv_event_ring = [None] * self._RING_SIZE
        self._recv_threads = [None] * self._RING_SIZE
        self._pending_recv_count = 0
        self._recv_idx_write = 0
        self._recv_idx_read = 0

    @torch.compiler.disable()
    def send_sync(self, x: torch.Tensor):
        self.inner.send_tensor(x)

    @torch.compiler.disable()
    def send_stream_ordered(self, x: torch.Tensor):
        """Stream-ordered send (no CPU sync). Falls back to send_tensor if not supported."""
        if hasattr(self.inner, 'send_stream_ordered'):
            self.inner.send_stream_ordered(x)
        else:
            self.inner.send_tensor(x)

    @torch.compiler.disable()
    def recv_stream_ordered(self) -> torch.Tensor:
        """Stream-ordered recv (minimal CPU sync). Falls back to recv_tensor if not supported."""
        if hasattr(self.inner, 'recv_stream_ordered'):
            return self.inner.recv_stream_ordered()
        else:
            return self.inner.recv_tensor()

    @torch.compiler.disable()
    def recv_sync(self) -> torch.Tensor:
        from sglang.srt.layers.afd_mixin import _afd_host_events, _afd_ctx

        _prof_layer = _afd_ctx.get("layer", -1)
        _prof_mb = _afd_ctx.get("mb", -1)
        t0 = time.time()
        result = self.inner.recv_tensor()
        t1 = time.time()
        _afd_host_events.append({
            "ts_ms": round(t1 * 1000, 3),
            "role": "UCX_PROFILE", "layer": _prof_layer, "mb": _prof_mb,
            "event": "recv_sync_detail",
            "ucx_recv_dur_us": round((t1 - t0) * 1e6, 1),
        })
        return result


# --------------- Global accessors ---------------


_async_communicator: Optional[AsyncTensorCommunicator] = None
_async_communicator_lock = threading.Lock()
# functools.cache is not single-flight: concurrent misses may both construct.
# Preserve the legacy path and lock only the cross-node experiment.
_experimental_tensor_communicators: Dict[AFDPerspective, FifoTensorCommunicator] = {}
_experimental_async_communicators: Dict[AFDPerspective, AsyncTensorCommunicator] = {}
_experimental_communicator_lock = threading.RLock()
_afd_peer_channel_pool = None
_afd_peer_channel_pool_lock = threading.Lock()

# Per-mb override: when the data-driven AsyncMbDriver is running, it sets
# this to the current mb's AsyncTensorCommunicator so prepare_mlp /
# postprocess_layer route send/recv through the right per-mb channel
# instead of the global singleton.  See afd_async_sched.AsyncMbDriver._step.
_per_mb_async_override: Optional[AsyncTensorCommunicator] = None


def get_afd_communicator():
    """Get the underlying tensor communicator (if AFD mode is active).

    Returns the raw FifoTensorCommunicator (e.g., UcxTensorCommunicator)
    or None if AFD is not configured. Used by live reshard for hot reconnect.
    """
    try:
        return get_tensor_communicator()
    except (RuntimeError, Exception):
        return None


def peek_async_communicator() -> Optional[AsyncTensorCommunicator]:
    """Return an already-created async communicator without building one.

    This accessor is for status/idle observation paths that must remain safe
    after ``reset_afd_communicators``. In particular, it never creates a tensor
    communicator, peer-channel pool, or transport connection.
    """
    if _per_mb_async_override is not None:
        return _per_mb_async_override

    server_args = get_global_server_args()
    if getattr(server_args, "afd_multi_pf_continuation", False):
        pool = _afd_peer_channel_pool
        if pool is None:
            return None
        perspective = get_afd_perspective()
        group_id = (
            0
            if perspective == AFDPerspective.AFD_PERSPECTIVE_ATTN
            else int(getattr(server_args, "afd_pf_group_id", 0))
        )
        try:
            return pool[group_id].async_comm
        except KeyError:
            return None

    if _cross_node_experimental_enabled():
        perspective = get_afd_perspective()
        if perspective is None:
            return None
        with _experimental_communicator_lock:
            return _experimental_async_communicators.get(perspective)

    return _async_communicator


def get_existing_async_communicator() -> Optional[AsyncTensorCommunicator]:
    """Compatibility alias for the side-effect-free async communicator lookup."""
    return peek_async_communicator()


def get_async_communicator() -> AsyncTensorCommunicator:
    if _per_mb_async_override is not None:
        return _per_mb_async_override
    server_args = get_global_server_args()
    if getattr(server_args, "afd_multi_pf_continuation", False):
        pool = get_afd_peer_channel_pool()
        perspective = get_afd_perspective()
        group_id = (
            0
            if perspective == AFDPerspective.AFD_PERSPECTIVE_ATTN
            else int(getattr(server_args, "afd_pf_group_id", 0))
        )
        return pool[group_id].async_comm
    if _cross_node_experimental_enabled():
        perspective = get_afd_perspective()
        if perspective is None:
            raise RuntimeError("AFD perspective is not set.")
        with _experimental_communicator_lock:
            comm = _experimental_async_communicators.get(perspective)
            if comm is None:
                comm = AsyncTensorCommunicator(get_tensor_communicator())
                _experimental_async_communicators[perspective] = comm
                logger.warning(
                    "Cached experimental AFD async communicator pid=%d thread=%s "
                    "perspective=%s singleton_id=%d inner_id=%d",
                    os.getpid(), threading.current_thread().name, perspective,
                    id(comm), id(comm.inner),
                )
            return comm

    global _async_communicator
    if _async_communicator is None:
        with _async_communicator_lock:
            if _async_communicator is None:
                _async_communicator = AsyncTensorCommunicator(
                    get_tensor_communicator()
                )
    return _async_communicator


def get_afd_peer_channel_pool():
    """Build the fixed group→channel registry for shared-PA mode."""
    global _afd_peer_channel_pool
    if _afd_peer_channel_pool is not None:
        return _afd_peer_channel_pool

    with _afd_peer_channel_pool_lock:
        if _afd_peer_channel_pool is not None:
            return _afd_peer_channel_pool

        from sglang.srt.layers.afd_multi_peer import (
            AFDPeerChannelPool,
            AFDPeerSpec,
        )
        from sglang.srt.layers.afd_ipc_cpp.communicator import (
            CppIpcTensorCommunicator,
        )

        server_args = get_global_server_args()
        perspective = get_afd_perspective()
        group_count = int(server_args.afd_pf_group_count)
        group_id = int(server_args.afd_pf_group_id)
        channel_base = int(server_args.afd_pf_channel_base)
        peer_devices = [
            int(item.strip())
            for item in server_args.afd_pf_peer_devices.split(",")
            if item.strip()
        ]

        if perspective == AFDPerspective.AFD_PERSPECTIVE_ATTN:
            specs = [
                AFDPeerSpec(
                    group_id=i,
                    channel_id=channel_base + i * 16,
                    peer_device=peer_devices[i],
                )
                for i in range(group_count)
            ]

            def comm_factory(role, *, peer_device, channel_id):
                return CppIpcTensorCommunicator(
                    role,
                    peer_device=peer_device,
                    channel_id=channel_id,
                )
        else:
            specs = [
                AFDPeerSpec(
                    group_id=group_id,
                    channel_id=channel_base + group_id * 16,
                    peer_device=peer_devices[0],
                )
            ]
            local_tp = server_args.tp_size
            local_tp_rank = (
                dist.get_rank() % local_tp if dist.is_initialized() else 0
            )

            def comm_factory(role, *, peer_device, channel_id):
                inner = (
                    CppIpcTensorCommunicator(
                        role,
                        peer_device=peer_device,
                        channel_id=channel_id,
                    )
                    if local_tp_rank == 0
                    else None
                )
                if local_tp <= 1:
                    return inner
                return BroadcastTensorCommunicator(
                    inner_comm=inner,
                    local_tp_size=local_tp,
                    local_tp_rank=local_tp_rank,
                )

        _afd_peer_channel_pool = AFDPeerChannelPool(
            perspective,
            specs,
            comm_factory=comm_factory,
        )
        return _afd_peer_channel_pool


def _create_tensor_communicator() -> FifoTensorCommunicator:
    perspective = get_afd_perspective()
    if perspective is None:
        raise RuntimeError("AFD perspective is not set.")

    server_args = get_global_server_args()
    comm_backend = getattr(server_args, "afd_comm_backend", None) or "auto"

    if comm_backend == "ucx" or (
        comm_backend == "auto" and os.environ.get("AFD_UCX_TLS")
    ):
        from sglang.srt.layers.rdma_comm import UcxTensorCommunicator

        return UcxTensorCommunicator(perspective)

    if comm_backend == "ipc":
        from sglang.srt.layers.ipc_comm import IpcTensorCommunicator

        return IpcTensorCommunicator(perspective)

    if comm_backend == "ipc_cpp":
        from sglang.srt.layers.afd_ipc_cpp.communicator import CppIpcTensorCommunicator

        local_tp = server_args.tp_size
        local_tp_rank = dist.get_rank() % local_tp if dist.is_initialized() else 0
        if getattr(server_args, "afd_ipc_per_rank", False):
            attn_tp = server_args.afd_attn_tp or local_tp
            ffn_tp = server_args.afd_ffn_tp or local_tp
            if attn_tp != local_tp or ffn_tp != local_tp:
                raise ValueError(
                    "--afd-ipc-per-rank requires homogeneous A/F TP: "
                    f"afd_attn_tp={attn_tp}, afd_ffn_tp={ffn_tp}, local_tp={local_tp}"
                )
            return CppIpcTensorCommunicator(
                perspective, channel_rank=local_tp_rank
            )
        if local_tp > 1:
            base_comm = CppIpcTensorCommunicator(perspective) if local_tp_rank == 0 else None
            return BroadcastTensorCommunicator(
                inner_comm=base_comm,
                local_tp_size=local_tp,
                local_tp_rank=local_tp_rank,
            )
        return CppIpcTensorCommunicator(perspective)

    if comm_backend == "afd_reshard_loopback":
        from sglang.srt.layers.afd_reshard_loopback import (
            AFDReshardLoopbackTensorCommunicator,
        )

        local_tp = server_args.tp_size
        local_tp_rank = dist.get_rank() % local_tp if dist.is_initialized() else 0
        base_comm = (
            AFDReshardLoopbackTensorCommunicator(perspective)
            if local_tp_rank == 0
            else None
        )
        if local_tp > 1:
            return BroadcastTensorCommunicator(
                inner_comm=base_comm,
                local_tp_size=local_tp,
                local_tp_rank=local_tp_rank,
            )
        return base_comm

    if comm_backend == "nccl_p2p":
        from sglang.srt.layers.nccl_p2p_comm import NcclP2pTensorCommunicator

        base_port = int(os.environ.get("AFD_NCCL_P2P_PORT", "29600"))
        # Use different ports for prefill and decode pairs
        disagg_mode = getattr(server_args, "disaggregation_mode", "prefill")
        if "decode" in str(disagg_mode):
            base_port += 1
        is_ffn = (perspective == AFDPerspective.AFD_PERSPECTIVE_FFN)
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        return NcclP2pTensorCommunicator(
            is_ffn=is_ffn,
            local_device=device,
            nccl_port=base_port,
        )

    if comm_backend == "stepmesh" or (
        comm_backend == "auto" and os.environ.get("MLC_INTERFACE")
    ):
        return StepMeshTensorCommunicator(perspective)

    # ZMQ baseline remains rank-0 + TP broadcast. Experimental sharding is
    # strictly double-gated and gives every local TP rank its own peer socket.
    local_tp = server_args.tp_size
    local_tp_rank = ZMQSimpleTensorCommunicator._rank(None)
    if local_tp > 1 and _zmq_sharding_enabled():
        return ShardedZMQTensorCommunicator(
            perspective, local_tp_size=local_tp, local_tp_rank=local_tp_rank
        )
    if local_tp > 1:
        base_comm = ZMQSimpleTensorCommunicator(perspective) if local_tp_rank == 0 else None
        return BroadcastTensorCommunicator(
            inner_comm=base_comm,
            local_tp_size=local_tp,
            local_tp_rank=local_tp_rank,
        )
    return ZMQSimpleTensorCommunicator(perspective)


@cache
def _get_legacy_tensor_communicator() -> FifoTensorCommunicator:
    return _create_tensor_communicator()


def get_tensor_communicator() -> FifoTensorCommunicator:
    """Return one communicator per process/perspective in experimental mode."""
    if not _cross_node_experimental_enabled():
        return _get_legacy_tensor_communicator()
    perspective = get_afd_perspective()
    if perspective is None:
        raise RuntimeError("AFD perspective is not set.")
    with _experimental_communicator_lock:
        comm = _experimental_tensor_communicators.get(perspective)
        if comm is None:
            comm = _create_tensor_communicator()
            _experimental_tensor_communicators[perspective] = comm
            logger.warning(
                "Cached experimental AFD tensor communicator pid=%d thread=%s "
                "perspective=%s singleton_id=%d cache_id=%d",
                os.getpid(), threading.current_thread().name, perspective,
                id(comm), id(_experimental_tensor_communicators),
            )
        return comm


def reset_afd_communicators() -> None:
    """Close and clear process-local communicators for lazy TP-aware rebuild."""
    global _async_communicator, _afd_peer_channel_pool, _per_mb_async_override
    with _experimental_communicator_lock:
        communicators = list(_experimental_tensor_communicators.values())
        _experimental_async_communicators.clear()
        _experimental_tensor_communicators.clear()
    legacy_comm = None
    if _get_legacy_tensor_communicator.cache_info().currsize:
        legacy_comm = _get_legacy_tensor_communicator()
    _get_legacy_tensor_communicator.cache_clear()
    with _async_communicator_lock:
        _async_communicator = None
        _per_mb_async_override = None
    with _afd_peer_channel_pool_lock:
        peer_pool = _afd_peer_channel_pool
        _afd_peer_channel_pool = None
    if peer_pool is not None:
        try:
            peer_pool.drain()
        except Exception:
            logger.exception("Failed to drain shared-PA channel pool")
    if legacy_comm is not None:
        communicators.append(legacy_comm)
    seen = set()
    for comm in communicators:
        if id(comm) in seen:
            continue
        seen.add(id(comm))
        close = getattr(comm, "close", None)
        if close is not None:
            close()


def get_afd_micro_batch() -> int:
    return getattr(get_global_server_args(), "afd_micro_batch", 3)


# Backward-compat alias
get_afd_mirco_batch = get_afd_micro_batch


def get_afd_perspective() -> Optional[AFDPerspective]:
    return getattr(get_global_server_args(), "afd_perspective", None)


def afd_is_ffn() -> bool:
    return get_afd_perspective() == AFDPerspective.AFD_PERSPECTIVE_FFN


def afd_is_attn() -> bool:
    return get_afd_perspective() == AFDPerspective.AFD_PERSPECTIVE_ATTN


# --------------- Model forward with AFD pipeline ---------------


def model_forward_afd_split_inputs(
    layers,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    input_data_scatter_mode: ScatterMode,
) -> List[Dict]:
    def _split_raw(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> List[Dict]:
        if forward_batch.afd_children is None:
            return [dict(
                hidden_states=hidden_states,
                residual=residual,
                positions=positions,
                forward_batch=forward_batch,
                afd_subbatch_index=0,
            )]
        result = []
        for idx, child_batch in enumerate(forward_batch.afd_children):
            token_slice = slice(*child_batch.afd_parent_token_range)
            result.append(
                dict(
                    hidden_states=hidden_states[token_slice],
                    residual=None if residual is None else residual[token_slice],
                    positions=positions[token_slice],
                    forward_batch=child_batch,
                    afd_subbatch_index=idx,
                )
            )
        return result

    layer_input_scatter_mode = layers[0].layer_scatter_modes.layer_input_mode
    afd_splitter_scatter_mode = ScatterMode.TP_ATTN_FULL
    context = CommunicateContext.init_new()

    hidden_states, residual = CommunicateSummableTensorPairFn.execute(
        hidden_states_input_mode=input_data_scatter_mode,
        residual_input_mode=input_data_scatter_mode,
        output_mode=afd_splitter_scatter_mode,
        hidden_states=hidden_states,
        residual=residual,
        forward_batch=forward_batch,
        context=context,
    )

    inputs_arr = _split_raw(hidden_states, residual, positions, forward_batch)

    def _post_transform(hidden_states, residual, forward_batch, **kwargs):
        hidden_states, residual = CommunicateSummableTensorPairFn.execute(
            hidden_states_input_mode=afd_splitter_scatter_mode,
            residual_input_mode=afd_splitter_scatter_mode,
            output_mode=layer_input_scatter_mode,
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
            context=context,
        )
        return dict(
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
            **kwargs,
        )

    return [_post_transform(**inp) for inp in inputs_arr]


def _log_afd_breakdown(
    timing_records: list,
    num_layers: int,
    m_stage: int,
    anchor_event: Optional[torch.cuda.Event] = None,
    last_event: Optional[torch.cuda.Event] = None,
    host_end: Optional[float] = None,
) -> None:
    """Aggregate AFD per-stage CUDA events and log per-node TPOT breakdown.

    Called after torch.cuda.synchronize() inside model_forward_afd() so all
    recorded events have elapsed_time available.

    If anchor_event/last_event/host_end are provided, computes wall-clock
    timestamps for each sub-stage event:
      wall_ms = host_end * 1000 + elapsed(sub_stage, last_event)

    This gives cross-GPU aligned timestamps (DA and DF share the same host clock).
    """
    from sglang.srt.layers.afd_mixin import _afd_timing_enabled

    if not _afd_timing_enabled or not timing_records:
        return

    perspective = "attn" if afd_is_attn() else "ffn"
    host_end_ms = host_end * 1000.0 if host_end is not None else None

    # Aggregate by stage and sub-stage
    agg = {
        "A_prep_attn_ms": 0.0,
        "A_attn_ms": 0.0,
        "A_prep_mlp_ms": 0.0,
        "F_mlp_ms": 0.0,
        "F_postprocess_ms": 0.0,
    }
    stage_counts = {"A": 0, "F": 0}

    for rec in timing_records:
        stage = rec["stage"]
        stage_counts[stage] += 1
        events = rec["events"]
        for key, (ev_start, ev_end) in events.items():
            if ev_start is not None and ev_end is not None:
                t_ms = ev_start.elapsed_time(ev_end)
                agg_key = f"{stage}_{key}_ms"
                if agg_key in agg:
                    agg[agg_key] += t_ms

    t_A_total = agg["A_prep_attn_ms"] + agg["A_attn_ms"] + agg["A_prep_mlp_ms"]
    t_F_total = agg["F_mlp_ms"] + agg["F_postprocess_ms"]
    t_total = t_A_total + t_F_total

    logger.error(
        f"[AFD_BREAKDOWN] perspective={perspective} "
        f"layers={num_layers} M={m_stage} "
        f"total={t_total:.1f}ms "
        f"| A_stage={t_A_total:.1f}ms "
        f"(prep_attn={agg['A_prep_attn_ms']:.1f}ms "
        f"attn={agg['A_attn_ms']:.1f}ms "
        f"prep_mlp={agg['A_prep_mlp_ms']:.1f}ms) "
        f"| F_stage={t_F_total:.1f}ms "
        f"(mlp={agg['F_mlp_ms']:.1f}ms "
        f"postprocess={agg['F_postprocess_ms']:.1f}ms) "
        f"| nA={stage_counts['A']} nF={stage_counts['F']}"
    )

    # Per-step timing output (one JSON array per forward pass)
    steps = []
    for rec in timing_records:
        step = {
            "stage": rec["stage"],
            "perspective": rec.get("perspective", perspective),
            "layer_id": rec.get("layer_id", -1),
            "mb": rec.get("mb", -1),
            "batch_size": rec.get("batch_size", -1),
        }
        for key, (ev_start, ev_end) in rec.get("events", {}).items():
            if ev_start is not None and ev_end is not None:
                step[f"{key}_ms"] = round(ev_start.elapsed_time(ev_end), 3)
                # Wall-clock alignment: offset from last_event
                if host_end_ms is not None and last_event is not None:
                    step[f"{key}_wall_start_ms"] = round(
                        host_end_ms - ev_start.elapsed_time(last_event), 3
                    )
                    step[f"{key}_wall_end_ms"] = round(
                        host_end_ms - ev_end.elapsed_time(last_event), 3
                    )
            else:
                step[f"{key}_ms"] = 0.0
        steps.append(step)
    import json as _json
    logger.error(
        f"[AFD_PER_STEP] perspective={perspective} "
        f"layers={num_layers} M={m_stage} "
        f"nsteps={len(steps)} "
        f"steps={_json.dumps(steps)}"
    )


def model_forward_afd(
    layers,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    input_data_scatter_mode: ScatterMode,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if hidden_states.shape[0] == 0:
        return hidden_states, residual

    from sglang.srt.layers.afd_mixin import (
        _afd_timing_records, _afd_timing_enabled, _afd_sched_ts,
        _afd_host_events, _afd_ctx,
    )

    _afd_sched_ts["forward_start"] = time.time()

    num_layers = len(layers)
    if forward_batch.afd_children is not None:
        m_stage = len(forward_batch.afd_children)
    else:
        m_stage = 1
    _afd_ctx["m_stage"] = m_stage

    _async_sched_enabled = bool(
        getattr(get_global_server_args(), "afd_async_schedule", False)
    )
    _multi_pf_enabled = bool(
        getattr(
            get_global_server_args(),
            "afd_multi_pf_continuation",
            False,
        )
        and afd_is_attn()
        and m_stage > 1
    )

    # Drain pending sends from previous forward is done at scheduler level
    # (before entering model_forward_afd) to maximize overlap.
    # Here we only drain recvs to clean up stale pre-issue state.
    try:
        comm = get_async_communicator()
        comm.drain_recvs()
        # Reset UCX metadata cache so both peers re-exchange shape/dtype
        # at the start of each forward pass (tensor shape changes between
        # requests with different batch sizes or sequence lengths).
        if hasattr(comm.inner, "reset_metadata_cache"):
            comm.inner.reset_metadata_cache()
    except Exception:
        pass

    # Reset timing records for this forward pass
    if _afd_timing_enabled:
        _afd_timing_records.clear()
    _afd_host_events.clear()

    # Detailed per-stage wall-clock timing (env-var gated)
    _detailed_timing_enabled = os.getenv("AFD_DETAILED_TIMING", "0") == "1"
    _detailed_timeline: list = []

    _t_split_start = time.time()
    input_arrs = model_forward_afd_split_inputs(
        layers=layers,
        hidden_states=hidden_states,
        residual=residual,
        positions=positions,
        forward_batch=forward_batch,
        input_data_scatter_mode=input_data_scatter_mode,
    )
    _t_split_end = time.time()

    # Data-driven scheduler branch (--afd-async-schedule).
    # With the interleaved schedule, we no longer need per-mb channels or
    # the AsyncMbDriver.  The interleaved schedule is selected above and
    # executed by the same pipeline loop below.  This branch is kept as
    # dead code for reference but disabled.
    if _multi_pf_enabled:
        from sglang.srt.layers.afd_async_sched import AsyncMbDriver

        channels = get_afd_peer_channel_pool()
        expected_groups = [
            child.afd_pf_group_id for child in forward_batch.afd_children
        ]
        if expected_groups != list(range(m_stage)):
            raise RuntimeError(
                "Shared-PA MVP requires contiguous lane/group mapping: "
                f"got groups={expected_groups}, m_stage={m_stage}"
            )
        driver = AsyncMbDriver(
            perspective=get_afd_perspective(),
            layers=layers,
            num_layers=num_layers,
            m_stage=m_stage,
            channels=channels,
            input_arrs=input_arrs,
            detailed_timeline=(
                _detailed_timeline if _detailed_timing_enabled else None
            ),
        )
        mbs_final = driver.run()

        try:
            channels.drain()
        except Exception:
            logger.exception("AsyncMbDriver: channel drain failed")

        _afd_sched_ts["forward_end"] = time.time()

        # ── Detailed per-stage timeline logging (mirrors legacy path) ──
        if _detailed_timing_enabled and _detailed_timeline:
            import json as _json
            perspective = "attn" if afd_is_attn() else "ffn"
            tl_json = _json.dumps(_detailed_timeline)
            logger.error(
                f"[AFD_TIMELINE] perspective={perspective} "
                f"layers={num_layers} M={m_stage} "
                f"total_steps={len(_detailed_timeline)} "
                f"timeline={tl_json}"
            )

        if len(mbs_final) == 1:
            return mbs_final[0].hidden_states, mbs_final[0].residual

        for i, m in enumerate(mbs_final):
            if m.hidden_states.numel() == 0:
                raise RuntimeError(
                    f"model_forward_afd (async): mb {i} hidden_states empty"
                )

        # Only sync in profiling mode; merge copies are stream-ordered.
        if _detailed_timing_enabled and torch.cuda.is_available():
            torch.cuda.synchronize()

        total_tokens = sum(m.hidden_states.shape[0] for m in mbs_final)
        hidden_dim = mbs_final[0].hidden_states.shape[1]
        dtype = mbs_final[0].hidden_states.dtype
        device = mbs_final[0].hidden_states.device

        if total_tokens <= 0:
            raise RuntimeError(
                f"model_forward_afd (async): total_tokens={total_tokens}"
            )

        merged_hidden = torch.empty(
            total_tokens, hidden_dim, dtype=dtype, device=device,
        )
        need_residual = afd_is_attn() and mbs_final[0].residual is not None
        merged_residual = (
            torch.empty(total_tokens, hidden_dim, dtype=dtype, device=device)
            if need_residual
            else None
        )
        offset = 0
        for m in mbs_final:
            n = m.hidden_states.shape[0]
            merged_hidden[offset : offset + n] = m.hidden_states
            if need_residual:
                merged_residual[offset : offset + n] = m.residual
            offset += n
        return merged_hidden, merged_residual

    # ── Cross-layer async pipeline (--afd-async-pipeline) ──────────────────
    _async_pipeline_enabled = (
        m_stage > 1
        and (
            bool(getattr(get_global_server_args(), "afd_async_pipeline", False))
            or os.environ.get("AFD_ASYNC_PIPELINE", "0") == "1"
        )
    )
    if _async_pipeline_enabled:
        from sglang.srt.layers.afd_async_pipeline import AsyncPipelineExecutor

        executor = AsyncPipelineExecutor(
            layers=layers,
            m_stage=m_stage,
            input_arrs=input_arrs,
            perspective=get_afd_perspective(),
        )
        mb_results = executor.run()

        _afd_sched_ts["forward_end"] = time.time()

        if len(mb_results) == 1:
            return mb_results[0][0], mb_results[0][1]

        total_tokens = sum(hs.shape[0] for hs, _ in mb_results)
        hidden_dim = mb_results[0][0].shape[1]
        dtype = mb_results[0][0].dtype
        device = mb_results[0][0].device

        merged_hidden = torch.empty(total_tokens, hidden_dim, dtype=dtype, device=device)
        need_residual = afd_is_attn() and mb_results[0][1] is not None
        merged_residual = (
            torch.empty(total_tokens, hidden_dim, dtype=dtype, device=device)
            if need_residual else None
        )
        offset = 0
        for hs, res in mb_results:
            n = hs.shape[0]
            merged_hidden[offset:offset + n] = hs
            if need_residual and res is not None:
                merged_residual[offset:offset + n] = res
            offset += n
        return merged_hidden, merged_residual

    # G6 optimization: use StageIO NamedTuple instead of dict
    stage_outputs: Dict[AFDForwardStage, deque] = {
        AFDForwardStage.AFD_FORWARD_STAGE_A: deque(),
        AFDForwardStage.AFD_FORWARD_STAGE_F: deque(),
    }
    stage_outputs[AFDForwardStage.AFD_FORWARD_STAGE_F].extend(
        StageIO(inp["hidden_states"], inp["residual"]) for inp in input_arrs
    )

    def forward_A(layer_id: int, micro_batch_idx: int):
        io = stage_outputs[AFDForwardStage.AFD_FORWARD_STAGE_F].popleft()
        hs, res = layers[layer_id].forward_afd_A(
            input_arrs[micro_batch_idx]["positions"],
            io.hidden_states,
            input_arrs[micro_batch_idx]["forward_batch"],
            io.residual,
        )
        if _afd_timing_records:
            _afd_timing_records[-1]["mb"] = micro_batch_idx
            _afd_timing_records[-1]["batch_size"] = io.hidden_states.shape[0]
        stage_outputs[AFDForwardStage.AFD_FORWARD_STAGE_A].append(StageIO(hs, res))

    def forward_F(layer_id: int, micro_batch_idx: int):
        io = stage_outputs[AFDForwardStage.AFD_FORWARD_STAGE_A].popleft()
        hs, res = layers[layer_id].forward_afd_F(
            io.hidden_states,
            input_arrs[micro_batch_idx]["forward_batch"],
            io.residual,
        )
        if _afd_timing_records:
            _afd_timing_records[-1]["mb"] = micro_batch_idx
            _afd_timing_records[-1]["batch_size"] = io.hidden_states.shape[0]
        stage_outputs[AFDForwardStage.AFD_FORWARD_STAGE_F].append(StageIO(hs, res))

    executors = {
        AFDForwardStage.AFD_FORWARD_STAGE_A: forward_A,
        AFDForwardStage.AFD_FORWARD_STAGE_F: forward_F,
    }

    pipeline = (
        AFDStageScheduleGenerator.attn_stage(num_layers, m_stage)
        if afd_is_attn()
        else AFDStageScheduleGenerator.ffn_stage(num_layers, m_stage)
    )

    # Use interleaved schedule if --afd-async-schedule is set and M>1.
    # This avoids the per-mb channel complexity while achieving the same
    # goal: mb0 advances to next layer immediately after its F-stage.
    if _async_sched_enabled and afd_is_attn() and m_stage > 1:
        pipeline = AFDStageScheduleGenerator.attn_stage_interleaved(
            num_layers, m_stage
        )

    # 3BO: with async recv (background-threaded UCX recv), pre-issue after
    # EVERY A-stage on the Attn node.  The background thread blocks on UCX
    # while the main pipeline loop continues launching compute, overlapping
    # DF→DA transfer with subsequent A-stage computation.
    #
    # On the FFN node: pre-issue after every F-stage for the next A-stage
    # so DA→DF transfer overlaps with FFN compute.
    #
    # NOTE: async recv is enabled for M>1 (micro-batch pipeline) where there
    # IS overlap opportunity between recv and the next micro-batch's compute.
    # For M=1, recv is synchronous (recv_sync) because:
    #   1. No overlap opportunity (only one micro-batch)
    #   2. The stream-ordered send path submits to the bridge event loop,
    #      and concurrent recv on the same bridge could cause ordering issues.
    #
    # IPC backend: uses per-slot SHM flags (RING_SIZE=4) and 2-phase recv
    # (bg-thread flag polling only, main-thread GPU copy) to avoid both
    # flag races and bg-thread event.synchronize() overhead.
    # Pre-issue is only valid when recv_start() is actually non-blocking.
    # CppIPC serializes recv_tensor() with _recv_lock, so recv_start() runs the
    # receive synchronously in the scheduler thread. Pre-issuing another receive
    # after A(L, mb1) would then wait for F(L, mb1) before the scheduler can
    # consume the already-ready F(L, mb0) and launch A(L+1, mb0), creating an
    # implicit layer barrier. For such backends, receive on demand in the
    # interleaved F stage instead.
    _recv_comm = get_async_communicator()
    _supports_nonblocking_recv = getattr(
        _recv_comm.inner,
        "supports_concurrent_recv",
        not hasattr(_recv_comm.inner, "_recv_lock"),
    )
    _async_recv_enabled = m_stage > 1 and _supports_nonblocking_recv
    _afd_pipe_logger = logging.getLogger("afd_pipeline")

    # === Precompute preissue schedule (eliminates O(n) sum() per iteration) ===
    _preissue_after: list = [False] * len(pipeline)
    if _async_recv_enabled:
        if afd_is_attn():
            # For attn node: preissue recv after each A-stage, up to RING_SIZE
            # pending at any time. Walk the schedule once to decide.
            _comm = get_async_communicator()
            _ring_size = _comm._RING_SIZE
            _pending_sim = 0  # simulated pending count
            for _pi, (_ps, *_pargs) in enumerate(pipeline):
                if _ps == AFDForwardStage.AFD_FORWARD_STAGE_F:
                    _pending_sim = max(0, _pending_sim - 1)
                elif _ps == AFDForwardStage.AFD_FORWARD_STAGE_A:
                    if _pending_sim < _ring_size:
                        # Check there's at least one F-stage after this point
                        _has_f_after = any(
                            s == AFDForwardStage.AFD_FORWARD_STAGE_F
                            for s, *_ in pipeline[_pi + 1:]
                        )
                        if _has_f_after:
                            _preissue_after[_pi] = True
                            _pending_sim += 1
        elif afd_is_ffn():
            # For FFN node: preissue after each F-stage if next is A-stage
            _comm = get_async_communicator()
            _ring_size = _comm._RING_SIZE
            _pending_sim = 0
            for _pi, (_ps, *_pargs) in enumerate(pipeline):
                if _ps == AFDForwardStage.AFD_FORWARD_STAGE_A:
                    _pending_sim = max(0, _pending_sim - 1)
                elif _ps == AFDForwardStage.AFD_FORWARD_STAGE_F:
                    if _pi + 1 < len(pipeline):
                        _next_s = pipeline[_pi + 1][0]
                        if _next_s == AFDForwardStage.AFD_FORWARD_STAGE_A and _pending_sim < _ring_size:
                            _preissue_after[_pi] = True
                            _pending_sim += 1

    _t_pipeline_start = time.time()

    # ═══ FAST PATH: M=1, no timing, no async recv ═══════════════════════
    # Eliminates ~36ms of Python overhead (128 iterations of dict lookups,
    # context updates, conditional checks, deque operations).
    # With batch-level round-robin, each batch targets a single PF group,
    # so we can safely use the fast path with channel override.
    _use_fast_path = (
        m_stage == 1
        and not _async_recv_enabled
        and not _detailed_timing_enabled
    )

    if _use_fast_path:
        # M=1 uses the proven per-layer AFD communicator path.  This preserves
        # model-specific residual/normalization semantics and keeps IPC protocol
        # ownership in AFDCommunicator instead of duplicating it here.
        hs = input_arrs[0]["hidden_states"]
        res = input_arrs[0]["residual"]
        pos = input_arrs[0]["positions"]
        fb = input_arrs[0]["forward_batch"]

        # NOTE: For multi-PF continuation (batch-level RR), route through
        # the correct channel for the target PF group.
        global _per_mb_async_override
        _fast_path_override = None
        if getattr(get_global_server_args(), "afd_multi_pf_continuation", False) and afd_is_attn():
            _target_group = getattr(fb, "afd_pf_group_id", 0)
            if _target_group != 0:
                pool = get_afd_peer_channel_pool()
                _fast_path_override = pool[_target_group].async_comm
                _per_mb_async_override = _fast_path_override
        try:
            _lp = os.environ.get("SGLANG_LAYER_PROFILE", "0") == "2"
            if _lp:
                _lp_a_times = []
                _lp_f_times = []
                torch.cuda.synchronize()
            for layer in layers:
                if _lp:
                    _t0 = time.time()
                hs, res = layer.forward_afd_A(pos, hs, fb, res)
                if _lp:
                    torch.cuda.synchronize()
                    _t1 = time.time()
                hs, res = layer.forward_afd_F(hs, fb, res)
                if _lp:
                    torch.cuda.synchronize()
                    _t2 = time.time()
                    _lp_a_times.append(_t1 - _t0)
                    _lp_f_times.append(_t2 - _t1)
            if _lp and _lp_a_times:
                _a_total = sum(_lp_a_times) * 1000
                _f_total = sum(_lp_f_times) * 1000
                _n = len(_lp_a_times)
                logger.info(
                    f"[AFD_FASTPATH_PROFILE] layers={_n} bs={hs.shape[0]} "
                    f"A_total={_a_total:.1f}ms F_total={_f_total:.1f}ms "
                    f"total={_a_total+_f_total:.1f}ms "
                    f"A_mean={_a_total/_n:.3f}ms F_mean={_f_total/_n:.3f}ms"
                )
        finally:
            if _fast_path_override is not None:
                _per_mb_async_override = None
        results = [StageIO(hs, res)]
    else:
        # In shared-PA multi-PF mode (batch-level RR), all requests in a batch
        # go to a single PF group. Route IPC through that group's channel.
        _pipeline_override = None
        if (
            getattr(get_global_server_args(), "afd_multi_pf_continuation", False)
            and afd_is_attn()
            and m_stage == 1
        ):
            _target_group = getattr(forward_batch, "afd_pf_group_id", 0)
            if _target_group != 0:
                pool = get_afd_peer_channel_pool()
                _pipeline_override = pool[_target_group].async_comm
                _per_mb_async_override = _pipeline_override

        # Reset CppIPC metadata cache since tensor shapes may differ from
        # previous iteration (which may have used a different channel).
        _reset_comm = get_async_communicator().inner
        if hasattr(_reset_comm, "reset_cache"):
            _reset_comm.reset_cache()
        for i, (stage_type, *args) in enumerate(pipeline):
            stage_name = stage_type.name
            layer_id = args[0] if args else -1
            mb_id = args[1] if len(args) > 1 else -1

            _afd_ctx["layer"] = layer_id
            _afd_ctx["mb"] = mb_id
            _afd_ctx["stage"] = stage_name

            t0 = time.perf_counter() if _detailed_timing_enabled else 0
            executors[stage_type](*args)
            t1 = time.perf_counter() if _detailed_timing_enabled else 0

            if _detailed_timing_enabled:
                _detailed_timeline.append({
                    "step": i,
                    "stage": stage_name,
                    "layer": layer_id,
                    "mb": mb_id,
                    "t_start_ms": t0 * 1000,
                    "t_end_ms": t1 * 1000,
                    "dur_ms": (t1 - t0) * 1000,
                })

            if _preissue_after[i]:
                comm = get_async_communicator()
                if comm._pending_recv_count < comm._RING_SIZE:
                    comm.recv_start()

        _wall_anchor_last_event = None
        _wall_anchor_host_end = None
        if _afd_timing_enabled and torch.cuda.is_available():
            _wall_anchor_last_event = torch.cuda.Event(enable_timing=True)
            _wall_anchor_last_event.record()

        try:
            results = [
                stage_outputs[AFDForwardStage.AFD_FORWARD_STAGE_F].popleft()
                for _ in range(m_stage)
            ]
        except IndexError:
            raise ValueError(
                "model_forward_afd: unexpected empty queue — potential implementation bug"
            )

        # Clear per-mb override set for multi-PF channel routing
        if _pipeline_override is not None:
            _per_mb_async_override = None

    _t_pipeline_end = time.time()

    # ── 3BO: drain pending recvs before returning ──────────────────────
    # drain_recvs prevents stale pre-issue state from leaking into
    # the next forward pass when m_stage changes (e.g. M=3 → M=1).
    # drain_sends is DEFERRED to the start of the next forward pass
    # so that the scheduler can overlap CPU work (process_batch_result,
    # recv_requests, get_next_batch) with the GPU finishing sends.
    try:
        comm = get_async_communicator()
        comm.drain_recvs()
    except Exception:
        pass

    _t_drain_end = time.time()

    # ── Scheduler wall-clock anchor for cross-GPU latency breakdown ─────
    _afd_sched_ts["forward_end"] = time.time()

    # Log forward overhead breakdown
    _fwd_overhead_ms = (_afd_sched_ts["forward_end"] - _afd_sched_ts["forward_start"]) * 1000
    _split_ms = (_t_split_end - _t_split_start) * 1000
    _pipeline_ms = (_t_pipeline_end - _t_pipeline_start) * 1000
    _drain_ms = (_t_drain_end - _t_pipeline_end) * 1000
    _pre_pipeline_ms = (_t_pipeline_start - _t_split_end) * 1000
    logger.info(
        f"[AFD_FWD_OVERHEAD] total={_fwd_overhead_ms:.1f}ms "
        f"split_inputs={_split_ms:.1f}ms "
        f"pre_pipeline={_pre_pipeline_ms:.1f}ms "
        f"pipeline={_pipeline_ms:.1f}ms "
        f"drain={_drain_ms:.1f}ms"
    )

    # ── TPOT breakdown logging ──────────────────────────────────────────
    if _afd_timing_enabled and _afd_timing_records:
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        _wall_anchor_host_end = time.time()
        _log_afd_breakdown(
            _afd_timing_records, num_layers, m_stage,
            last_event=_wall_anchor_last_event,
            host_end=_wall_anchor_host_end,
        )
        _afd_timing_records.clear()

        # Log scheduler-level timestamps for cross-GPU latency breakdown
        if _afd_sched_ts:
            import json as _json2
            ts_out = {k: round(v * 1000, 3) for k, v in _afd_sched_ts.items()}
            logger.error(
                f"[AFD_SCHED_TS] sched_timestamps={_json2.dumps(ts_out)}"
            )
            _afd_sched_ts.clear()

        # Log host wall-clock events for cross-GPU pipeline breakdown
        if _afd_host_events:
            import json as _json3
            perspective = "attn" if afd_is_attn() else "ffn"
            role = "DA" if afd_is_attn() else "DF"
            host_json = _json3.dumps(_afd_host_events)
            logger.error(
                f"[AFD_HOST_EVENTS] role={role} perspective={perspective} "
                f"num_events={len(_afd_host_events)} "
                f"events={host_json}"
            )
            _afd_host_events.clear()

    # ── Detailed per-stage timeline logging ─────────────────────────────
    if _detailed_timing_enabled and _detailed_timeline:
        import json as _json
        perspective = "attn" if afd_is_attn() else "ffn"
        tl_json = _json.dumps(_detailed_timeline)
        logger.error(
            f"[AFD_TIMELINE] perspective={perspective} "
            f"layers={num_layers} M={m_stage} "
            f"total_steps={len(_detailed_timeline)} "
            f"timeline={tl_json}"
        )

    # G5 optimization: pre-allocate output and copy slices instead of torch.cat
    if len(results) == 1:
        return results[0].hidden_states, results[0].residual

    # Validate results before merging
    for i, r in enumerate(results):
        if r.hidden_states.numel() == 0:
            raise RuntimeError(
                f"model_forward_afd: micro-batch {i} has empty hidden_states "
                f"(shape={r.hidden_states.shape}) — likely a communication or split mismatch"
            )

    # Detailed merge wall timing is benchmark-only and env-gated.  Keep the
    # default path free of synchronization/event overhead.
    _t_merge_start = time.perf_counter() if _detailed_timing_enabled else 0.0

    # Sync CUDA only in timing/profiling mode.  In production the merge
    # copies are stream-ordered and will execute after upstream kernels.
    if _detailed_timing_enabled and torch.cuda.is_available():
        torch.cuda.synchronize()

    total_tokens = sum(r.hidden_states.shape[0] for r in results)
    hidden_dim = results[0].hidden_states.shape[1]
    dtype = results[0].hidden_states.dtype
    device = results[0].hidden_states.device

    if total_tokens <= 0:
        raise RuntimeError(
            f"model_forward_afd: total_tokens={total_tokens} (expected > 0) "
            f"across {len(results)} micro-batches"
        )

    merged_hidden = torch.empty(total_tokens, hidden_dim, dtype=dtype, device=device)
    need_residual = afd_is_attn() and results[0].residual is not None
    merged_residual = (
        torch.empty(total_tokens, hidden_dim, dtype=dtype, device=device)
        if need_residual
        else None
    )

    offset = 0
    for r in results:
        n = r.hidden_states.shape[0]
        merged_hidden[offset : offset + n] = r.hidden_states
        if need_residual:
            merged_residual[offset : offset + n] = r.residual
        offset += n

    if _detailed_timing_enabled:
        # Synchronize only in profiling mode so host wall includes completion
        # of the output slice copies rather than launch time alone.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logger.error(
            f"[AFD_MERGE] perspective={'attn' if afd_is_attn() else 'ffn'} "
            f"M={m_stage} merge_ms={(time.perf_counter() - _t_merge_start) * 1000:.3f}"
        )

    return merged_hidden, merged_residual


# --------------- AFDCommunicator ---------------


class AFDCommunicator:
    """Wraps a LayerCommunicator and injects cross-node A<->F communication.

    Adapted for the new LayerCommunicator interface (extra optional kwargs in
    prepare_attn / prepare_mlp / postprocess_layer).
    """

    def __init__(
        self,
        layer_communicator: LayerCommunicator,
        perspective: AFDPerspective,
        layer_id: int,
    ):
        self.perspective = perspective
        self.layer_communicator = layer_communicator
        self.layer_id = layer_id

    # Expose layer_scatter_modes for external use
    @property
    def layer_scatter_modes(self):
        return self.layer_communicator.layer_scatter_modes

    @torch.compiler.disable()
    def prepare_attn(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ):
        if self.perspective == AFDPerspective.AFD_PERSPECTIVE_FFN:
            return hidden_states, residual
        return self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch, **kwargs
        )

    @torch.compiler.disable()
    def prepare_attn_and_capture_last_layer_outputs(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ):
        if self.perspective == AFDPerspective.AFD_PERSPECTIVE_FFN:
            return hidden_states, residual
        return self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
            hidden_states, residual, forward_batch, **kwargs
        )

    @torch.compiler.disable()
    def prepare_mlp(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ):
        from sglang.srt.layers.afd_mixin import _afd_host_events, _afd_ctx

        comm = get_async_communicator()
        if self.perspective == AFDPerspective.AFD_PERSPECTIVE_FFN:
            # 3BO: if recv was pre-issued by the pipeline, drain it.
            # Otherwise fall back to blocking sync recv (no thread overhead).
            t_recv_start = time.time()
            if comm._pending_recv is not None:
                hidden_states = comm.recv_wait()
            elif hasattr(comm, 'recv_stream_ordered') and os.environ.get("AFD_STREAM_ORDERED", "0") == "1":
                hidden_states = comm.recv_stream_ordered()
            else:
                hidden_states = comm.recv_sync()
            t_recv_end = time.time()
            _afd_host_events.append({
                "ts_ms": round(t_recv_start * 1000, 3),
                "role": "DF", "layer": self.layer_id, "mb": _afd_ctx["mb"],
                "event": "recv_start", "stage": "A",
            })
            _afd_host_events.append({
                "ts_ms": round(t_recv_end * 1000, 3),
                "role": "DF", "layer": self.layer_id, "mb": _afd_ctx["mb"],
                "event": "recv_end", "stage": "A",
                "recv_dur_us": round((t_recv_end - t_recv_start) * 1e6, 1),
            })
            return hidden_states, residual

        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch, **kwargs
        )
        # Queue A→F only when split-phase GPU-Direct is enabled. Other
        # backends retain their established stream-ordered behavior.
        if (
            getattr(comm.inner, "_split_phase_recv", False)
            and _afd_ctx.get("m_stage", 1) > 1
        ):
            comm.send_async(hidden_states)
        else:
            comm.inner.send_stream_ordered(hidden_states)
        return hidden_states, residual

    @torch.compiler.disable()
    def postprocess_layer(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        from sglang.srt.layers.afd_mixin import _afd_host_events, _afd_ctx

        comm = get_async_communicator()
        if self.perspective == AFDPerspective.AFD_PERSPECTIVE_FFN:
            # Queue F→A without blocking the FFN compute thread when the
            # matching split receive is enabled.
            if (
                getattr(comm.inner, "_split_phase_recv", False)
                and _afd_ctx.get("m_stage", 1) > 1
            ):
                comm.send_async(hidden_states)
            else:
                comm.inner.send_stream_ordered(hidden_states)
            return hidden_states, residual

        # R4: true overlap — recv_start was already issued by
        # postprocess_layer_start_recv if called from the pipeline;
        # otherwise fall back to blocking recv.
        t_recv_start = time.time()
        if comm._pending_recv is not None:
            hidden_states = comm.recv_wait()
        elif hasattr(comm.inner, 'recv_stream_ordered'):
            hidden_states = comm.inner.recv_stream_ordered()
        else:
            hidden_states = comm.recv_sync()
        # Ensure contiguity: IPC recv may produce views that downstream
        # .view() calls (e.g. rotary embedding) cannot handle.
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()
        t_recv_end = time.time()
        _afd_host_events.append({
            "ts_ms": round(t_recv_start * 1000, 3),
            "role": "DA", "layer": self.layer_id, "mb": _afd_ctx["mb"],
            "event": "recv_start", "stage": "F",
        })
        _afd_host_events.append({
            "ts_ms": round(t_recv_end * 1000, 3),
            "role": "DA", "layer": self.layer_id, "mb": _afd_ctx["mb"],
            "event": "recv_end", "stage": "F",
            "recv_dur_us": round((t_recv_end - t_recv_start) * 1e6, 1),
        })
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )
        return hidden_states, residual

    @torch.compiler.disable()
    def postprocess_layer_start_recv(self):
        """R4: Issue non-blocking recv_start for the F→A result.

        Called by the pipeline scheduler BEFORE starting the next microbatch's
        compute, so communication overlaps with the next stage's computation.
        """
        if self.perspective == AFDPerspective.AFD_PERSPECTIVE_ATTN:
            get_async_communicator().recv_start()

    # Delegate new LayerCommunicator methods
    def should_use_reduce_scatter(self, forward_batch: ForwardBatch) -> bool:
        if hasattr(self.layer_communicator, "should_use_reduce_scatter"):
            return self.layer_communicator.should_use_reduce_scatter(forward_batch)
        return False

    def should_fuse_mlp_allreduce_with_next_layer(
        self, forward_batch: ForwardBatch
    ) -> bool:
        if hasattr(self.layer_communicator, "should_fuse_mlp_allreduce_with_next_layer"):
            return self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        return False


# --------------- Proxy modules ---------------


class AFDProxyAttention(nn.Module):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> torch.Tensor:
        return hidden_states


class AFDProxyMLP(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        **kwargs,
    ) -> torch.Tensor:
        return hidden_states
