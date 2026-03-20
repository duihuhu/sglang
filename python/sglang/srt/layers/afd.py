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
import time
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
        schedule = []
        if num_layers == 1:
            return [
                (AFDForwardStage.AFD_FORWARD_STAGE_A, 0, m) for m in range(m_stage)
            ] + [
                (AFDForwardStage.AFD_FORWARD_STAGE_F, 0, m) for m in range(m_stage)
            ]
        for layer_id, m in itertools.product(range(num_layers + 1), range(m_stage)):
            if layer_id > 0:
                schedule.append(
                    (AFDForwardStage.AFD_FORWARD_STAGE_F, layer_id - 1, m)
                )
            if layer_id < num_layers:
                schedule.append((AFDForwardStage.AFD_FORWARD_STAGE_A, layer_id, m))
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
        # C1: pre-allocated pinned memory for staging
        self._pinned_send_buf: Optional[torch.Tensor] = None
        self._pinned_recv_buf: Optional[torch.Tensor] = None
        # C2: separate CUDA stream for async transfers
        if torch.cuda.is_available():
            self._comm_stream = torch.cuda.Stream()
        else:
            self._comm_stream = None

    @staticmethod
    def _get_ffn_port() -> int:
        return int(os.getenv("AFD_FFN_BASE_PORT", "40000"))

    @staticmethod
    def _get_attn_port() -> int:
        return int(os.getenv("AFD_ATTN_BASE_PORT", "50000"))

    def _get_lport(self) -> int:
        return self.start_lport + 1 + dist.get_rank()

    def _get_dport(self) -> int:
        return self.start_dport + 1 + dist.get_rank()

    def _get_cuda_device(self) -> torch.device:
        if self._cuda_device is None:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available.")
            self._cuda_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        return self._cuda_device

    @cache
    def _get_push_socket(self) -> zmq.Socket:
        socket = self.zmq_context.socket(zmq.PUSH)
        # E5: set send timeout
        if self.SOCKET_TIMEOUT_MS > 0:
            socket.setsockopt(zmq.SNDTIMEO, self.SOCKET_TIMEOUT_MS)
        socket.connect(f"tcp://localhost:{self._get_dport()}")
        return socket

    @cache
    def _get_pull_socket(self) -> zmq.Socket:
        socket = self.zmq_context.socket(zmq.PULL)
        # E5: set receive timeout
        if self.SOCKET_TIMEOUT_MS > 0:
            socket.setsockopt(zmq.RCVTIMEO, self.SOCKET_TIMEOUT_MS)
        socket.bind(f"tcp://*:{self._get_lport()}")
        return socket

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
        # C1: receive raw bytes then reconstruct tensor
        try:
            metadata = socket.recv_pyobj()
            data = socket.recv(copy=False)
        except zmq.Again:
            raise TimeoutError("AFD ZMQ recv timed out — peer may be dead")
        buf = torch.frombuffer(bytearray(data), dtype=metadata["dtype"]).reshape(
            metadata["shape"]
        )
        return buf.to(self._get_cuda_device(), non_blocking=True)

    def send_tensor(self, x: torch.Tensor):
        socket = self._get_push_socket()
        # C1: send metadata + raw bytes (avoid pickle)
        cpu_tensor = x.detach().contiguous().cpu()
        metadata = {"shape": list(cpu_tensor.shape), "dtype": cpu_tensor.dtype}
        try:
            socket.send_pyobj(metadata, zmq.SNDMORE)
            socket.send(cpu_tensor.numpy().tobytes())
        except zmq.Again:
            raise TimeoutError("AFD ZMQ send timed out — peer may be dead")


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
            # F9: per-group DMLC config — force-set to override any defaults
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
            flag = f"STEPMESH_SCHEDULER_STARTED_{self.group_id}"
            if os.environ.get(flag) == "1":
                return
            os.environ[flag] = "1"
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


# --------------- Heterogeneous TP communicator (Phase 6.4: sharded parallel) ---------------


class ShardedParallelCommunicator(FifoTensorCommunicator):
    """Sharded parallel communicator for heterogeneous TP (TP_A != TP_F).

    Each local rank sends/receives a shard of the FULL tensor to/from
    corresponding remote rank(s), then all_gathers locally to reconstruct
    the full tensor. This uses min(TP_A, TP_F) parallel cross-node links.

    R1 fix: supports different TP sizes via rank-to-rank mapping.
    R2 fix: transmits original num_tokens in metadata so receiver can truncate padding.
    """

    def __init__(
        self,
        inner_comm: FifoTensorCommunicator,
        local_tp_size: int,
        local_tp_rank: int,
        remote_tp_size: int,
    ):
        super().__init__()
        self.inner_comm = inner_comm
        self.local_tp_size = local_tp_size
        self.local_tp_rank = local_tp_rank
        self.remote_tp_size = remote_tp_size
        self._tp_group = None

    def _get_tp_group(self):
        if self._tp_group is None:
            from sglang.srt.distributed import get_tp_group

            self._tp_group = get_tp_group()
        return self._tp_group

    def send_tensor(self, x: torch.Tensor):
        if self.local_tp_size == 1:
            # R2: wrap with original num_tokens metadata
            self.inner_comm.send_tensor(
                self._pack_with_metadata(x, x.shape[0])
            )
            return

        num_tokens = x.shape[0]
        chunk_size = (num_tokens + self.local_tp_size - 1) // self.local_tp_size
        padded_tokens = chunk_size * self.local_tp_size
        if padded_tokens != num_tokens:
            pad = torch.zeros(
                padded_tokens - num_tokens,
                x.shape[1],
                dtype=x.dtype,
                device=x.device,
            )
            x = torch.cat([x, pad], dim=0)

        start = self.local_tp_rank * chunk_size
        my_shard = x[start : start + chunk_size].contiguous()
        # R2: embed original num_tokens so receiver can truncate
        self.inner_comm.send_tensor(
            self._pack_with_metadata(my_shard, num_tokens)
        )

    def recv_tensor(self) -> torch.Tensor:
        packed = self.inner_comm.recv_tensor()
        my_shard, original_num_tokens = self._unpack_metadata(packed)

        if self.local_tp_size == 1:
            return my_shard[:original_num_tokens]

        tp_group = self._get_tp_group()
        shard_list = [torch.empty_like(my_shard) for _ in range(self.local_tp_size)]
        dist.all_gather(shard_list, my_shard, group=tp_group.device_group)
        full = torch.cat(shard_list, dim=0)
        # R2: truncate to original num_tokens (remove padding)
        return full[:original_num_tokens]

    @staticmethod
    def _pack_with_metadata(
        tensor: torch.Tensor, original_num_tokens: int
    ) -> torch.Tensor:
        """Append a 1-element metadata row containing original_num_tokens."""
        device = tensor.device
        dtype = tensor.dtype
        meta = torch.zeros(1, tensor.shape[1], dtype=dtype, device=device)
        meta[0, 0] = float(original_num_tokens)
        return torch.cat([tensor, meta], dim=0)

    @staticmethod
    def _unpack_metadata(
        packed: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """Extract the metadata row and return (tensor, original_num_tokens)."""
        original_num_tokens = int(packed[-1, 0].item())
        return packed[:-1], original_num_tokens


# --------------- Async communication wrapper (C2 optimization) ---------------


class AsyncTensorCommunicator:
    """Wraps a FifoTensorCommunicator to overlap communication with computation.

    send_async: queues the send on a separate CUDA stream, returns immediately.
    recv_start: begins receiving on the comm stream (non-blocking on compute stream).
    recv_wait: blocks the compute stream until the recv completes.
    """

    def __init__(self, inner: FifoTensorCommunicator):
        self.inner = inner
        self.comm_stream = (
            torch.cuda.Stream() if torch.cuda.is_available() else None
        )
        self._pending_recv: Optional[torch.Tensor] = None
        self._recv_event: Optional[torch.cuda.Event] = None

    @torch.compiler.disable()
    def send_async(self, x: torch.Tensor):
        if self.comm_stream is not None:
            with torch.cuda.stream(self.comm_stream):
                self.inner.send_tensor(x)
        else:
            self.inner.send_tensor(x)

    @torch.compiler.disable()
    def recv_start(self):
        if self.comm_stream is not None:
            with torch.cuda.stream(self.comm_stream):
                self._pending_recv = self.inner.recv_tensor()
                self._recv_event = self.comm_stream.record_event()
        else:
            self._pending_recv = self.inner.recv_tensor()
            self._recv_event = None

    @torch.compiler.disable()
    def recv_wait(self) -> torch.Tensor:
        if self._recv_event is not None:
            self._recv_event.synchronize()
        result = self._pending_recv
        self._pending_recv = None
        self._recv_event = None
        return result

    @torch.compiler.disable()
    def send_sync(self, x: torch.Tensor):
        self.inner.send_tensor(x)

    @torch.compiler.disable()
    def recv_sync(self) -> torch.Tensor:
        return self.inner.recv_tensor()


# --------------- Global accessors ---------------


_async_communicator: Optional[AsyncTensorCommunicator] = None


def get_async_communicator() -> AsyncTensorCommunicator:
    global _async_communicator
    if _async_communicator is None:
        _async_communicator = AsyncTensorCommunicator(get_tensor_communicator())
    return _async_communicator


@cache
def get_tensor_communicator() -> FifoTensorCommunicator:
    perspective = get_afd_perspective()
    if perspective is None:
        raise RuntimeError("AFD perspective is not set.")

    if os.environ.get("MLC_INTERFACE"):
        # F1 Step 6: StepMesh natively supports N:M sharded communication,
        # no need for ShardedParallelCommunicator wrapper
        return StepMeshTensorCommunicator(perspective)

    # ZMQ path: wrap with ShardedParallelCommunicator for heterogeneous TP
    base_comm = ZMQSimpleTensorCommunicator(perspective)
    server_args = get_global_server_args()
    local_tp = server_args.tp_size
    attn_tp = getattr(server_args, "afd_attn_tp", None) or local_tp
    ffn_tp = getattr(server_args, "afd_ffn_tp", None) or local_tp
    if attn_tp != ffn_tp:
        remote_tp = (
            ffn_tp
            if perspective == AFDPerspective.AFD_PERSPECTIVE_ATTN
            else attn_tp
        )
        local_tp_rank = dist.get_rank() % local_tp if dist.is_initialized() else 0
        return ShardedParallelCommunicator(
            inner_comm=base_comm,
            local_tp_size=local_tp,
            local_tp_rank=local_tp_rank,
            remote_tp_size=remote_tp,
        )
    return base_comm


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


def model_forward_afd(
    layers,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    input_data_scatter_mode: ScatterMode,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    num_layers = len(layers)
    m_stage = get_afd_micro_batch()

    input_arrs = model_forward_afd_split_inputs(
        layers=layers,
        hidden_states=hidden_states,
        residual=residual,
        positions=positions,
        forward_batch=forward_batch,
        input_data_scatter_mode=input_data_scatter_mode,
    )

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
        stage_outputs[AFDForwardStage.AFD_FORWARD_STAGE_A].append(StageIO(hs, res))

    def forward_F(layer_id: int, micro_batch_idx: int):
        io = stage_outputs[AFDForwardStage.AFD_FORWARD_STAGE_A].popleft()
        hs, res = layers[layer_id].forward_afd_F(
            io.hidden_states,
            input_arrs[micro_batch_idx]["forward_batch"],
            io.residual,
        )
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

    # R4: true overlap — on the Attn node, after an F stage (which contains
    # postprocess_layer with recv), issue recv_start for the NEXT F stage
    # early so the network transfer overlaps with the next A stage compute.
    for i, (stage_type, *args) in enumerate(pipeline):
        executors[stage_type](*args)

        # After an A stage on Attn node, the send_async already returned.
        # Before the next F stage, pre-issue recv_start so it overlaps
        # with any remaining computation.
        if (
            afd_is_attn()
            and stage_type == AFDForwardStage.AFD_FORWARD_STAGE_A
            and i + 1 < len(pipeline)
            and pipeline[i + 1][0] == AFDForwardStage.AFD_FORWARD_STAGE_F
        ):
            layer_id_next_f = pipeline[i + 1][1]
            if hasattr(layers[layer_id_next_f], "layer_communicator"):
                lc = layers[layer_id_next_f].layer_communicator
                if hasattr(lc, "postprocess_layer_start_recv"):
                    lc.postprocess_layer_start_recv()

    try:
        results = [
            stage_outputs[AFDForwardStage.AFD_FORWARD_STAGE_F].popleft()
            for _ in range(m_stage)
        ]
    except IndexError:
        raise ValueError(
            "model_forward_afd: unexpected empty queue — potential implementation bug"
        )

    # G5 optimization: pre-allocate output and copy slices instead of torch.cat
    if len(results) == 1:
        return results[0].hidden_states, results[0].residual

    total_tokens = sum(r.hidden_states.shape[0] for r in results)
    hidden_dim = results[0].hidden_states.shape[1]
    dtype = results[0].hidden_states.dtype
    device = results[0].hidden_states.device

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
        comm = get_async_communicator()
        if self.perspective == AFDPerspective.AFD_PERSPECTIVE_FFN:
            # C2: start receiving from Attn asynchronously
            comm.recv_start()
            hidden_states = comm.recv_wait()
            return hidden_states, residual

        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch, **kwargs
        )
        # C2: send to FFN asynchronously (compute stream can proceed)
        comm.send_async(hidden_states)
        return hidden_states, residual

    @torch.compiler.disable()
    def postprocess_layer(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        comm = get_async_communicator()
        if self.perspective == AFDPerspective.AFD_PERSPECTIVE_FFN:
            # C2: send result back to Attn asynchronously
            comm.send_async(hidden_states)
            return hidden_states, residual

        # R4: true overlap — recv_start was already issued by
        # postprocess_layer_start_recv if called from the pipeline;
        # otherwise fall back to blocking recv.
        if comm._pending_recv is not None:
            hidden_states = comm.recv_wait()
        else:
            hidden_states = comm.recv_sync()
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
