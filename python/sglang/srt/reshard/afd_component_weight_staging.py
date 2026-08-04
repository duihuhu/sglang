# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Low-peak, standalone AFD component weight staging.

PREPARE never constructs a model on joining ranks.  Source rank zero publishes a
parameter descriptor over a dedicated control group, while parameter bytes move
one-at-a-time through a Gloo CPU group.  Every process retains only the target
shard belonging to that process.

Online Safety (Shrink / TPn->TP1)
---------------------------------
When ``target_tp < source_tp`` (currently only TP1 targets), the NCCL transport
uses a **CPU/Gloo fallback** instead of NCCL P2P on the staging ``data_group``.
Each source rank D2Hs its local CUDA shard to CPU (synchronizing only its own
copy, never holding the staging NCCL communicator or default CUDA stream), then
all max-world ranks participate in a sequential Gloo ``broadcast`` per source
shard through the independent ``control_group``.  Rank 0 assembles pieces into a
CPU-resident TP1 target; non-zero ranks return ``None``.  Replicated-shard
validation uses ``torch.equal`` on CPU tensors only.

This avoids deadlocks between the online A/F serving forward (which uses the
default CUDA stream and the serving NCCL communicator) and the staging NCCL
communicator that would otherwise require all ranks to enter P2P sends/recvs
simultaneously — impossible when some ranks are blocked in the serving forward.

Environment variables
---------------------
``AFD_RESHARD_FAKE_STAGING=1`` (default ``0``/off): TEST-ONLY fast mode. PREPARE
skips the real weight D2H/gather and builds a minimal empty ``no_op``/``fake``
shadow that reaches READY immediately, but still runs the descriptor broadcast so
the full fan-out/control/drain/commit/activate protocol path is exercised. Under
this flag ACTIVATE is a no-op (no process-group rebuild, no H2D) and the scheduler
skips refreshing serving groups so ``tp_size`` never diverges from the unchanged
real topology. Never enable in production — no weights actually move. All fake-mode
log lines are tagged ``[FAKE_STAGING]``.
"""

from __future__ import annotations

import gc
import logging
import os
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Sequence, Tuple

import torch

from sglang.srt.layers.afd_mixin import AFDWeightFilter
from sglang.srt.layers.afd_type import AFDPerspective
from sglang.srt.layers.reshard_weights import get_tp_split_rules
from sglang.srt.reshard.afd_weight_reshard import (
    ParameterManifest,
    ReshardManifest,
    ReshardValidationError,
    SplitRule,
    TransactionState,
)

logger = logging.getLogger(__name__)


class ComponentMaxWorldTransport(Protocol):
    rank: int
    world_size: int

    def broadcast_manifest(
        self, payload: Optional[Mapping[str, Any]]
    ) -> Mapping[str, Any]: ...
    def stage_target_shard(
        self,
        name: str,
        local_tensor: Optional[torch.Tensor],
        rule: SplitRule,
        source_tp: int,
        target_tp: int,
        target_rank: int,
    ) -> Optional[torch.Tensor]: ...
    def materialize_deferred_target(
        self,
        name: str,
        local_tensor: Optional[torch.Tensor],
        rule: SplitRule,
        source_tp: int,
        target_tp: int,
        target_rank: int,
    ) -> Optional[torch.Tensor]: ...
    def begin_operation_metrics(self, operation_id: str) -> None: ...
    def snapshot_operation_metrics(self) -> Mapping[str, float]: ...
    def consensus_error(self, error: Optional[str]) -> Optional[str]: ...
    def gather_timings(
        self, timings: Mapping[str, float]
    ) -> Sequence[Mapping[str, float]]: ...
    def barrier(self) -> None: ...


@dataclass
class ComponentStagingState:
    operation_id: str
    epoch: int
    source_tp: int
    target_tp: int
    manifest: ReshardManifest
    shadow: Dict[str, torch.Tensor]
    old_tensors: Dict[str, torch.Tensor]
    state: TransactionState = TransactionState.READY
    cpu_bytes: int = 0
    staged_gpu_bytes: int = 0
    deferred_local_bytes: int = 0
    deferred: Dict[str, SplitRule] = field(default_factory=dict)
    no_op: bool = False
    # Set only under AFD_RESHARD_FAKE_STAGING: the shadow carries no real
    # weights, so activate() must still rebuild groups but skip the H2D copy.
    fake: bool = False
    timings_s: Dict[str, float] = field(default_factory=dict)
    rank_timings_s: Tuple[Mapping[str, float], ...] = ()
    operation_transport: Optional[ComponentMaxWorldTransport] = field(
        default=None, repr=False
    )


def create_afd_component_staging_groups(dist, ranks, backend, timeout):
    """Collectively create dedicated staging groups in a deterministic order."""
    backend = str(backend).lower()
    if backend not in ("gloo", "nccl", "hybrid"):
        raise ValueError("AFD_COMPONENT_STAGING_BACKEND must be gloo, nccl, or hybrid")
    groups = {"gloo": None, "nccl": None, "control": None}
    # All ranks must execute these calls in this exact order. Hybrid eagerly
    # creates both data communicators; no operation is allowed to call new_group.
    if backend in ("gloo", "hybrid"):
        groups["gloo"] = dist.new_group(ranks=ranks, backend="gloo", timeout=timeout)
    if backend in ("nccl", "hybrid"):
        groups["nccl"] = dist.new_group(ranks=ranks, backend="nccl", timeout=timeout)
    groups["control"] = dist.new_group(ranks=ranks, backend="gloo", timeout=timeout)
    return groups


_RESHARD_METRIC_DEFAULTS = {
    "local_repartition_s": 0.0,
    "local_repartition_bytes": 0,
    "peer_transfer_s": 0.0,
    "peer_transfer_bytes": 0,
    "host_stage_d2h_s": 0.0,
    "host_stage_d2h_bytes": 0,
    # There is no checkpoint/host fallback loader in this implementation.
    "host_load_remaining_s": 0.0,
    "host_load_remaining_bytes": 0,
}


class _OperationMetricsMixin:
    """Operation-scoped, rank-local counters; one stager operation at a time."""

    def begin_operation_metrics(self, operation_id):
        self._metrics_lock = getattr(self, "_metrics_lock", threading.Lock())
        with self._metrics_lock:
            self._metrics_operation_id = str(operation_id)
            self._operation_metrics = dict(_RESHARD_METRIC_DEFAULTS)

    def _record_metric(self, seconds_key, elapsed, bytes_key, byte_count):
        lock = getattr(self, "_metrics_lock", None)
        if lock is None:
            self.begin_operation_metrics("")
            lock = self._metrics_lock
        with lock:
            self._operation_metrics[seconds_key] += float(elapsed)
            self._operation_metrics[bytes_key] += int(byte_count)

    def snapshot_operation_metrics(self):
        lock = getattr(self, "_metrics_lock", None)
        if lock is None:
            return dict(_RESHARD_METRIC_DEFAULTS)
        with lock:
            return dict(self._operation_metrics)


def _tensor_bytes(tensor):
    return 0 if tensor is None else int(tensor.numel() * tensor.element_size())


class TorchDistributedMaxWorldTransport(_OperationMetricsMixin):
    """Compatibility CPU transport over dedicated Gloo data/control groups."""

    def __init__(self, *, rank: int, world_size: int, data_group, control_group):
        import torch.distributed as dist

        if data_group is None or control_group is None:
            raise ValueError("dedicated max-world data/control groups are required")
        if (
            dist.get_world_size(data_group) != world_size
            or dist.get_world_size(control_group) != world_size
        ):
            raise ValueError("component groups must span max-world")
        if dist.get_backend(data_group) != "gloo":
            raise ValueError("component staging data group must use Gloo CPU transport")
        self.rank, self.world_size = int(rank), int(world_size)
        self.data_group, self.control_group = data_group, control_group
        self.backend_name = "gloo"

    def broadcast_manifest(self, payload):
        import torch.distributed as dist

        objects = [dict(payload) if self.rank == 0 and payload is not None else None]
        dist.broadcast_object_list(objects, src=0, group=self.control_group)
        if not isinstance(objects[0], dict):
            raise ReshardValidationError(
                "component parameter manifest was not broadcast"
            )
        return objects[0]

    def _collect_source_shards(self, name, local_tensor, source_tp):
        import torch.distributed as dist

        meta = None
        if self.rank < source_tp and local_tensor is not None:
            meta = (tuple(local_tensor.shape), str(local_tensor.dtype))
        metas = [None] * self.world_size
        dist.all_gather_object(metas, meta, group=self.control_group)
        source_meta = metas[:source_tp]
        if any(item is None for item in source_meta) or len(set(source_meta)) != 1:
            raise ReshardValidationError(f"{name}: source metadata mismatch")
        shape, dtype_name = source_meta[0]
        dtype = getattr(torch, dtype_name.removeprefix("torch."))
        shards = []
        for src in range(source_tp):
            if self.rank == src:
                # This is the only D2H allocation.  It is released after this
                # parameter is reshaped; no CUDA receive buffer is ever made.
                copy_started = time.perf_counter()
                buf = (
                    local_tensor.detach()
                    .to(device="cpu", copy=True, non_blocking=False)
                    .contiguous()
                )
                metric = (
                    ("host_stage_d2h_s", "host_stage_d2h_bytes")
                    if local_tensor.device.type == "cuda"
                    else ("local_repartition_s", "local_repartition_bytes")
                )
                self._record_metric(
                    metric[0], time.perf_counter() - copy_started,
                    metric[1], _tensor_bytes(buf),
                )
                try:
                    if torch.cuda.is_available():
                        buf = buf.pin_memory()
                except RuntimeError:
                    pass
            else:
                buf = torch.empty(shape, dtype=dtype, device="cpu")
            transfer_started = time.perf_counter()
            dist.broadcast(buf, src=src, group=self.data_group)
            self._record_metric(
                "peer_transfer_s", time.perf_counter() - transfer_started,
                "peer_transfer_bytes", _tensor_bytes(buf),
            )
            shards.append(buf)
        return tuple(shards)

    def stage_target_shard(
        self, name, local_tensor, rule, source_tp, target_tp, target_rank
    ):
        shards = self._collect_source_shards(name, local_tensor, source_tp)
        if target_tp > source_tp and self.rank < source_tp:
            return None
        if self.rank >= target_tp:
            return None
        repartition_started = time.perf_counter()
        result = _target_shard(shards, rule, target_tp, target_rank)
        self._record_metric(
            "local_repartition_s", time.perf_counter() - repartition_started,
            "local_repartition_bytes", _tensor_bytes(result),
        )
        return result

    def materialize_deferred_target(
        self, name, local_tensor, rule, source_tp, target_tp, target_rank
    ):
        shards = self._collect_source_shards(name, local_tensor, source_tp)
        return (
            _target_shard(shards, rule, target_tp, target_rank)
            if self.rank < source_tp
            else None
        )

    def consensus_error(self, error):
        import torch.distributed as dist

        errors = [None] * self.world_size
        dist.all_gather_object(errors, error, group=self.control_group)
        failures = [
            f"rank {rank}: {message}" for rank, message in enumerate(errors) if message
        ]
        return "; ".join(failures) if failures else None

    def gather_timings(self, timings):
        import torch.distributed as dist

        values = [None] * self.world_size
        dist.all_gather_object(values, dict(timings), group=self.control_group)
        return tuple(values)

    def barrier(self):
        import torch.distributed as dist

        dist.barrier(group=self.control_group)


class InMemoryMaxWorldTransport(_OperationMetricsMixin):
    """CPU test transport containing source shards for every parameter."""

    def __init__(
        self,
        rank: int,
        world_size: int,
        shards: Mapping[str, Sequence[torch.Tensor]],
        *,
        fail_parameter: Optional[str] = None,
    ):
        self.rank, self.world_size, self.shards = rank, world_size, shards
        self.fail_parameter = fail_parameter
        self._manifest = None

    def broadcast_manifest(self, payload):
        if payload is not None:
            self._manifest = dict(payload)
        if self._manifest is None:
            # Unit transports have no real rank zero; infer a descriptor from
            # the supplied source map when testing a joining rank in isolation.
            raise ReshardValidationError(
                "in-memory manifest requires rank-zero preparation"
            )
        return self._manifest

    def stage_target_shard(
        self, name, local_tensor, rule, source_tp, target_tp, target_rank
    ):
        if name == self.fail_parameter:
            raise RuntimeError(f"injected in-memory transport failure for {name}")
        started = time.perf_counter()
        out = tuple(t.detach().clone() for t in self.shards[name])
        if len(out) != source_tp:
            raise ReshardValidationError(f"{name}: source shard count mismatch")
        if target_tp > source_tp and self.rank < source_tp:
            return None
        result = (
            _target_shard(out, rule, target_tp, target_rank)
            if self.rank < target_tp else None
        )
        self._record_metric(
            "local_repartition_s", time.perf_counter() - started,
            "local_repartition_bytes", _tensor_bytes(result),
        )
        return result

    def materialize_deferred_target(
        self, name, local_tensor, rule, source_tp, target_tp, target_rank
    ):
        started = time.perf_counter()
        out = tuple(t.detach().clone() for t in self.shards[name])
        result = (
            _target_shard(out, rule, target_tp, target_rank)
            if self.rank < source_tp else None
        )
        self._record_metric(
            "local_repartition_s", time.perf_counter() - started,
            "local_repartition_bytes", _tensor_bytes(result),
        )
        return result

    def consensus_error(self, error):
        return error

    def gather_timings(self, timings):
        return (dict(timings),)

    def barrier(self):
        return None


class TorchDistributedNCCLMaxWorldTransport(_OperationMetricsMixin):
    """Hybrid max-world staging transport isolated from online collectives.

    Expansion keeps the direct NCCL/NVLink path. Shrink never enters the staging
    NCCL communicator: source shards are copied individually to CPU and broadcast
    over the dedicated staging Gloo control group. This prevents an online A/F
    forward from deadlocking with a concurrent, independently ordered NCCL P2P.
    """

    def __init__(
        self, *, rank: int, world_size: int, data_group, control_group, device
    ):
        import torch.distributed as dist

        if data_group is None or control_group is None:
            raise ValueError(
                "dedicated max-world NCCL data and Gloo control groups are required"
            )
        if (
            dist.get_backend(data_group) != "nccl"
            or dist.get_backend(control_group) != "gloo"
        ):
            raise ValueError("NCCL staging requires NCCL data and Gloo control groups")
        if (
            dist.get_world_size(data_group) != world_size
            or dist.get_world_size(control_group) != world_size
        ):
            raise ValueError("component staging groups must span max-world")
        self.rank, self.world_size = int(rank), int(world_size)
        self.data_group, self.control_group = data_group, control_group
        self.backend_name = "nccl"
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(device=self.device)
        # Warm the full max-world communicator once while every rank is present.
        # Later P2P batches may be sparse (for example only source rank1 sends to
        # joiners 2/3); without this collective rank0 can skip first-use NCCL ID
        # exchange and peers time out waiting for the communicator root.
        with torch.cuda.stream(self.stream):
            warmup = torch.zeros(1, dtype=torch.int32, device=self.device)
            dist.all_reduce(warmup, group=self.data_group)
        self.stream.synchronize()

    def broadcast_manifest(self, payload):
        import torch.distributed as dist

        objects = [dict(payload) if self.rank == 0 and payload is not None else None]
        dist.broadcast_object_list(objects, src=0, group=self.control_group)
        if not isinstance(objects[0], dict):
            raise ReshardValidationError(
                "component parameter manifest was not broadcast"
            )
        return objects[0]

    @staticmethod
    def _expansion_source(rule, source_tp, target_tp, target_rank):
        """Return the unique source for one expansion target.

        Replicated tensors retain one local copy on each existing source rank and
        distribute additional copies round-robin. Non-replicated tensors map each
        contiguous target block to its owning source shard.
        """
        if target_tp % source_tp:
            raise ReshardValidationError("target TP must be a source TP multiple")
        if rule.kind == "replicated":
            return target_rank % source_tp
        return target_rank // (target_tp // source_tp)

    @staticmethod
    def _expansion_piece(local, rule, source_tp, target_tp, target_rank):
        ratio = target_tp // source_tp
        subrank = target_rank % ratio
        if rule.kind == "replicated":
            return local.contiguous()
        dim = int(rule.dim)
        if rule.kind == "column_fused":
            offset, pieces = 0, []
            for full_size in rule.segments:
                local_size = full_size // source_tp
                piece_size = full_size // target_tp
                pieces.append(
                    local.narrow(dim, offset + subrank * piece_size, piece_size)
                )
                offset += local_size
            return torch.cat(pieces, dim=dim).contiguous()
        size = local.shape[dim] // ratio
        return local.narrow(dim, subrank * size, size).contiguous()

    @staticmethod
    def _adopt_local_expansion_piece(local, piece, replicated):
        if replicated:
            return piece, 0
        if piece.untyped_storage().data_ptr() == local.untyped_storage().data_ptr():
            result = piece.clone()
            return result, _tensor_bytes(result)
        return piece, _tensor_bytes(piece)

    def _stage_tp1_cpu(self, name, local_tensor, rule, source_tp, source_shape, dtype):
        """Serially assemble TP1 on rank zero using only staging Gloo collectives.

        The control group is dedicated to this stager (it is not a scheduler
        control group), and PREPARE has one thread per rank. Consequently metadata,
        these tensor broadcasts, and later object collectives execute in the same
        order on every max-world rank; no cross-thread collective interleaving is
        permitted. Each rank retains at most one CPU source buffer, while rank zero
        additionally owns the final CPU target.
        """
        import torch.distributed as dist

        if source_tp <= 1:
            raise ReshardValidationError(
                "CPU shrink requires source TP greater than one"
            )
        result = (
            _allocate_tp1_target(source_shape, dtype, "cpu", rule, source_tp)
            if self.rank == 0
            else None
        )
        for src in range(source_tp):
            if self.rank == src:
                if local_tensor is None:
                    raise ReshardValidationError(
                        f"{name}: source rank {src} has no tensor"
                    )
                # A blocking D2H copy waits only for this tensor's copy. It neither
                # records work on the staging NCCL stream nor synchronizes a loop of
                # default-stream operations.
                copy_started = time.perf_counter()
                cpu_shard = (
                    local_tensor.detach()
                    .to(device="cpu", copy=True, non_blocking=False)
                    .contiguous()
                )
                metric = (
                    ("host_stage_d2h_s", "host_stage_d2h_bytes")
                    if local_tensor.device.type == "cuda"
                    else ("local_repartition_s", "local_repartition_bytes")
                )
                self._record_metric(
                    metric[0], time.perf_counter() - copy_started,
                    metric[1], _tensor_bytes(cpu_shard),
                )
            else:
                cpu_shard = torch.empty(source_shape, dtype=dtype, device="cpu")
            # Reusing the dedicated staging control group is safe because this
            # prepare thread is its sole caller and all ranks use identical order.
            transfer_started = time.perf_counter()
            dist.broadcast(cpu_shard, src=src, group=self.control_group)
            self._record_metric(
                "peer_transfer_s", time.perf_counter() - transfer_started,
                "peer_transfer_bytes", _tensor_bytes(cpu_shard),
            )
            if self.rank == 0:
                repartition_started = time.perf_counter()
                _copy_source_shard_to_tp1_cpu(result, cpu_shard, rule, source_tp, src)
                self._record_metric(
                    "local_repartition_s", time.perf_counter() - repartition_started,
                    "local_repartition_bytes", _tensor_bytes(cpu_shard),
                )
            cpu_shard = None
        return result

    def stage_target_shard(
        self, name, local_tensor, rule, source_tp, target_tp, target_rank
    ):
        import torch.distributed as dist

        # Metadata uses the same dedicated Gloo group as the shrink tensor
        # broadcasts, preserving a single total collective order per parameter.
        meta = (
            (tuple(local_tensor.shape), str(local_tensor.dtype))
            if self.rank < source_tp and local_tensor is not None
            else None
        )
        metas = [None] * self.world_size
        dist.all_gather_object(metas, meta, group=self.control_group)
        if (
            any(item is None for item in metas[:source_tp])
            or len(set(metas[:source_tp])) != 1
        ):
            raise ReshardValidationError(f"{name}: source metadata mismatch")
        source_shape, dtype_name = metas[0]
        dtype = getattr(torch, dtype_name.removeprefix("torch."))

        if target_tp < source_tp:
            if target_tp != 1:
                raise ReshardValidationError(
                    "NCCL transport shrink currently supports target TP1 only"
                )
            # Online-safe fallback: do not touch data_group, the staging NCCL
            # stream, or any NCCL P2P API while serving forwards may be in flight.
            return self._stage_tp1_cpu(
                name, local_tensor, rule, source_tp, source_shape, dtype
            )

        if local_tensor is not None and local_tensor.device.type != "cuda":
            raise ReshardValidationError(
                f"{name}: NCCL expansion source tensor must be CUDA"
            )
        if target_tp % source_tp:
            raise ReshardValidationError("target TP must be a source TP multiple")
        result = None
        local_bytes = transfer_bytes = 0
        ops, refs = [], []
        local_started = time.perf_counter()
        with torch.cuda.stream(self.stream):
            for target in range(source_tp, target_tp):
                src = self._expansion_source(rule, source_tp, target_tp, target)
                if self.rank == src:
                    piece = self._expansion_piece(
                        local_tensor.detach(), rule, source_tp, target_tp, target
                    )
                    refs.append(piece)
                    local_bytes += _tensor_bytes(piece)
                    transfer_bytes += _tensor_bytes(piece)
                    ops.append(
                        dist.P2POp(dist.isend, piece, target, group=self.data_group)
                    )
                elif self.rank == target:
                    shape = list(source_shape)
                    if rule.kind != "replicated":
                        shape[int(rule.dim)] = _target_shape_from_local(
                            rule, source_shape, source_tp, target_tp
                        )
                    result = torch.empty(shape, dtype=dtype, device=self.device)
                    transfer_bytes += _tensor_bytes(result)
                    ops.append(
                        dist.P2POp(dist.irecv, result, src, group=self.data_group)
                    )
        # Complete all local slice/cat/contiguous construction before starting
        # the P2P interval. This makes the two wall-clock intervals disjoint.
        if local_bytes:
            self.stream.synchronize()
            self._record_metric(
                "local_repartition_s", time.perf_counter() - local_started,
                "local_repartition_bytes", local_bytes,
            )
        if ops:
            transfer_started = time.perf_counter()
            with torch.cuda.stream(self.stream):
                for work in dist.batch_isend_irecv(ops):
                    work.wait()
            self.stream.synchronize()
            self._record_metric(
                "peer_transfer_s", time.perf_counter() - transfer_started,
                "peer_transfer_bytes", transfer_bytes,
            )
        return result if self.rank < target_tp else None

    def materialize_deferred_target(
        self, name, local_tensor, rule, source_tp, target_tp, target_rank
    ):
        """Materialize active targets after drain with matched P2P ordering."""
        import torch.distributed as dist

        if target_tp <= source_tp or self.rank >= source_tp:
            return None
        result = None
        local_bytes = peer_bytes = 0
        ops, refs = [], []
        local_started = time.perf_counter()
        with torch.cuda.stream(self.stream):
            # Every source rank iterates all active target ranks in the same order.
            # This is required because target rank 1 may consume source rank 0's
            # second subshard for TP2->TP4 rather than transform its own old shard.
            for target in range(source_tp):
                src = self._expansion_source(rule, source_tp, target_tp, target)
                if self.rank == src:
                    piece = self._expansion_piece(
                        local_tensor.detach(), rule, source_tp, target_tp, target
                    )
                    if src == target:
                        if self.rank == target:
                            result, allocated_bytes = self._adopt_local_expansion_piece(
                                local_tensor, piece, rule.kind == "replicated"
                            )
                            local_bytes += allocated_bytes
                    else:
                        if (
                            piece.untyped_storage().data_ptr()
                            != local_tensor.untyped_storage().data_ptr()
                        ):
                            local_bytes += _tensor_bytes(piece)
                        refs.append(piece)
                        peer_bytes += _tensor_bytes(piece)
                        ops.append(
                            dist.P2POp(dist.isend, piece, target, group=self.data_group)
                        )
                elif self.rank == target:
                    shape = list(local_tensor.shape)
                    if rule.kind != "replicated":
                        shape[int(rule.dim)] = _target_shape_from_local(
                            rule, tuple(local_tensor.shape), source_tp, target_tp
                        )
                    result = torch.empty(
                        shape, dtype=local_tensor.dtype, device=self.device
                    )
                    peer_bytes += _tensor_bytes(result)
                    ops.append(
                        dist.P2POp(dist.irecv, result, src, group=self.data_group)
                    )
        # Resolve local piece/clone work first. P2P issue/wait begins only after
        # this synchronization, so no wall interval contributes to both metrics.
        if local_bytes:
            self.stream.synchronize()
            self._record_metric(
                "local_repartition_s", time.perf_counter() - local_started,
                "local_repartition_bytes", local_bytes,
            )
        if ops:
            peer_started = time.perf_counter()
            with torch.cuda.stream(self.stream):
                for work in dist.batch_isend_irecv(ops):
                    work.wait()
            self.stream.synchronize()
            self._record_metric(
                "peer_transfer_s", time.perf_counter() - peer_started,
                "peer_transfer_bytes", peer_bytes,
            )
        return result

    def consensus_error(self, error):
        import torch.distributed as dist

        errors = [None] * self.world_size
        dist.all_gather_object(errors, error, group=self.control_group)
        failures = [
            f"rank {rank}: {message}" for rank, message in enumerate(errors) if message
        ]
        return "; ".join(failures) if failures else None

    def gather_timings(self, timings):
        import torch.distributed as dist

        values = [None] * self.world_size
        dist.all_gather_object(values, dict(timings), group=self.control_group)
        return tuple(values)

    def barrier(self):
        import torch.distributed as dist

        dist.barrier(group=self.control_group)


class HybridMaxWorldTransport:
    """Operation-scoped routing across pre-created Gloo and NCCL transports."""

    backend_name = "hybrid"

    def __init__(self, gloo_transport, nccl_transport):
        if (gloo_transport.rank, gloo_transport.world_size) != (
            nccl_transport.rank,
            nccl_transport.world_size,
        ):
            raise ValueError("hybrid staging transports must span the same world")
        self.gloo_transport = gloo_transport
        self.nccl_transport = nccl_transport
        self.rank = gloo_transport.rank
        self.world_size = gloo_transport.world_size

    def select_operation_transport(self, source_tp, target_tp):
        # Shrink must stay entirely off NCCL while serving forward is live.
        # Equal-TP FFN operations are control-only, so Gloo is sufficient.
        if int(target_tp) <= int(source_tp):
            return self.gloo_transport
        return self.nccl_transport


def _target_shape_from_local(rule, source_shape, source_tp, target_tp):
    if rule.kind == "column_fused":
        return sum(int(segment) // target_tp for segment in rule.segments)
    dim = int(rule.dim)
    return int(source_shape[dim]) * source_tp // target_tp


def _rule_payload(rule):
    return tuple(rule) if rule is not None else None


def _allocate_tp1_target(source_shape, dtype, device, rule: SplitRule, source_tp: int):
    shape = list(source_shape)
    if rule.kind != "replicated":
        shape[int(rule.dim)] = _target_shape_from_local(
            rule, source_shape, source_tp, 1
        )
    return torch.empty(shape, dtype=dtype, device=device)


def _copy_source_shard_to_tp1(
    target: torch.Tensor,
    source: torch.Tensor,
    rule: SplitRule,
    source_tp: int,
    source_rank: int,
) -> None:
    """Copy one contiguous source shard into its TP1 destination positions."""
    if rule.kind == "replicated":
        if source_rank == 0:
            target.copy_(source)
        elif not torch.equal(target, source):
            raise ReshardValidationError("replicated source shards differ")
        return
    dim = int(rule.dim)
    if rule.kind == "column_fused":
        local_offset = target_offset = 0
        for full_size in rule.segments:
            local_size = int(full_size) // source_tp
            target.narrow(
                dim, target_offset + source_rank * local_size, local_size
            ).copy_(source.narrow(dim, local_offset, local_size))
            local_offset += local_size
            target_offset += int(full_size)
        return
    local_size = int(source.shape[dim])
    target.narrow(dim, source_rank * local_size, local_size).copy_(source)


def _copy_source_shard_to_tp1_cpu(
    target: torch.Tensor,
    source: torch.Tensor,
    rule: SplitRule,
    source_tp: int,
    source_rank: int,
) -> None:
    """CPU-only assembly of one source shard into TP1 target.

    Identical logic to _copy_source_shard_to_tp1, but explicitly requires both
    tensors to reside on CPU and performs replicated-shard validation via CPU
    torch.equal.  This avoids any GPU synchronization or NCCL dependency during
    the shrink path, which is critical for online safety when the A/F forward
    pass is concurrently using the default CUDA stream and NCCL communicator.
    """
    assert target.device.type == "cpu" and source.device.type == "cpu", (
        "_copy_source_shard_to_tp1_cpu requires CPU tensors"
    )
    if rule.kind == "replicated":
        if source_rank == 0:
            target.copy_(source)
        elif not torch.equal(target, source):
            raise ReshardValidationError("replicated source shards differ")
        return
    dim = int(rule.dim)
    if rule.kind == "column_fused":
        local_offset = target_offset = 0
        for full_size in rule.segments:
            local_size = int(full_size) // source_tp
            target.narrow(
                dim, target_offset + source_rank * local_size, local_size
            ).copy_(source.narrow(dim, local_offset, local_size))
            local_offset += local_size
            target_offset += int(full_size)
        return
    local_size = int(source.shape[dim])
    target.narrow(dim, source_rank * local_size, local_size).copy_(source)


def _assemble_tp1_target(source_shards, rule: SplitRule):
    """Pure CPU/CUDA helper mirroring serial NCCL TPn->TP1 assembly."""
    if not source_shards:
        raise ReshardValidationError("TP1 assembly requires source shards")
    first = source_shards[0]
    target = _allocate_tp1_target(
        tuple(first.shape), first.dtype, first.device, rule, len(source_shards)
    )
    for source_rank, source in enumerate(source_shards):
        _copy_source_shard_to_tp1(target, source, rule, len(source_shards), source_rank)
    return target


def _target_shard(source_shards, rule: SplitRule, target_tp: int, rank: int):
    """Compute one target shard without materializing every target rank."""
    if rule.kind == "replicated":
        if any(not torch.equal(source_shards[0], shard) for shard in source_shards[1:]):
            raise ReshardValidationError("replicated source shards differ")
        return source_shards[0].clone()
    dim = int(rule.dim)
    if rule.kind == "column_fused":
        old_tp, local_offset, pieces = len(source_shards), 0, []
        for full_size in rule.segments:
            local_size = full_size // old_tp
            full_segment = torch.cat(
                [s.narrow(dim, local_offset, local_size) for s in source_shards],
                dim=dim,
            )
            pieces.append(
                full_segment.narrow(
                    dim, rank * (full_size // target_tp), full_size // target_tp
                )
            )
            local_offset += local_size
        return torch.cat(pieces, dim=dim).contiguous()
    full = torch.cat(source_shards, dim=dim)
    size = full.shape[dim] // target_tp
    return full.narrow(dim, rank * size, size).contiguous()


class ModelRunnerAFDComponentStager:
    """Transactional standalone component shadows with bounded peak memory."""

    def __init__(
        self,
        runner: Any,
        perspective: AFDPerspective,
        max_tp: int,
        transport: ComponentMaxWorldTransport,
        *,
        max_cpu_staging_bytes: int = 16 * 1024**3,
        shadow_device: Optional[str] = None,
        group_refresh_callback: Optional[Callable[[int], None]] = None,
        topology_refresh_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        self.runner, self.perspective, self.max_tp = runner, perspective, int(max_tp)
        self.transport = transport
        self.max_cpu_staging_bytes = int(max_cpu_staging_bytes)
        self.group_refresh_callback = group_refresh_callback
        self.topology_refresh_callback = topology_refresh_callback
        requested = shadow_device or os.getenv("AFD_COMPONENT_SHADOW_DEVICE", "auto")
        if requested not in ("cpu", "cuda", "auto"):
            raise ValueError("AFD_COMPONENT_SHADOW_DEVICE must be cpu, cuda, or auto")
        self.shadow_device = requested
        self.prepared: Optional[ComponentStagingState] = None
        self._lock = threading.RLock()
        self._prepare_thread: Optional[threading.Thread] = None
        self._prepare_error: Optional[str] = None
        self._prepare_operation: Optional[str] = None
        self._cancel_requested = threading.Event()
        self.retired: list[Dict[str, torch.Tensor]] = []
        self.last_completed_status: Dict[str, Any] = {}

    def _target_tp(self, request):
        value = getattr(self.perspective, "value", str(self.perspective)).lower()
        return int(
            request["target_ffn_tp" if value.endswith("ffn") else "target_attn_tp"]
        )

    def _source_tp(self, request):
        value = getattr(self.perspective, "value", str(self.perspective)).lower()
        key = "expected_ffn_tp" if value.endswith("ffn") else "expected_attn_tp"
        return int(request.get(key, getattr(self.runner, "tp_size", 0)))

    def _is_ffn(self):
        return (
            str(getattr(self.perspective, "value", self.perspective))
            .lower()
            .endswith("ffn")
        )

    def _operation_transport(self, source_tp, target_tp):
        selector = getattr(self.transport, "select_operation_transport", None)
        return (
            selector(source_tp, target_tp) if selector is not None else self.transport
        )

    @staticmethod
    def _transport_name(transport):
        return str(getattr(transport, "backend_name", type(transport).__name__))

    @staticmethod
    def _topology(request, prefix):
        return (
            int(request[f"{prefix}_attn_tp"]),
            int(request[f"{prefix}_ffn_tp"]),
        )

    def _peer_topology_changed(self, request):
        expected_attn_tp, expected_ffn_tp = self._topology(request, "expected")
        target_attn_tp, target_ffn_tp = self._topology(request, "target")
        if self._is_ffn():
            return expected_attn_tp != target_attn_tp
        return expected_ffn_tp != target_ffn_tp

    def _refresh_peer_topology(self, request, *, reset_data_plane):
        target_attn_tp, target_ffn_tp = self._topology(request, "target")
        self.runner.afd_component_refresh_peer_topology(
            target_attn_tp,
            target_ffn_tp,
            reset_data_plane=reset_data_plane,
        )

    def safety_reason(self, request):
        source_tp, target_tp = self._source_tp(request), self._target_tp(request)
        rank = int(getattr(self.transport, "rank", -1))
        if self.transport.world_size != self.max_tp:
            return "component transport does not span launched max-world"
        if not (1 <= source_tp <= self.max_tp and 1 <= target_tp <= self.max_tp):
            return "component TP is outside launched max-world"
        if source_tp == target_tp and self._is_ffn():
            pass
        elif (source_tp, target_tp) not in ((2, 4), (4, 1)):
            return "MVP supports TP2->TP4 and attention TP4->TP1 only"
        if target_tp == 1 and self._is_ffn():
            return "TP4->TP1 is attention-only"
        if rank < 0 or rank >= self.max_tp:
            return "invalid max-world rank"
        if not getattr(
            self.runner, "_afd_component_collective_commands_enabled", False
        ):
            return "max-world scheduler command fan-out is not enabled"
        required_hooks = (
            "afd_component_rebuild_groups",
            "afd_component_refresh_runtime",
            "afd_component_refresh_peer_topology",
        )
        if any(not hasattr(self.runner, hook) for hook in required_hooks):
            return "component activation hooks are missing"
        return ""

    def _descriptor(self, request, source_tp, target_tp):
        model = getattr(self.runner, "model", None)
        if model is None:
            return None
        actual = {
            name: param
            for name, param in model.named_parameters()
            if AFDWeightFilter.should_load(name, self.perspective)
        }
        if not actual:
            raise RuntimeError("component owns no actual named parameters")
        rules = get_tp_split_rules(model)
        family = str(
            getattr(
                getattr(getattr(self.runner, "model_config", None), "hf_config", None),
                "model_type",
                "qwen3",
            )
        )
        return {
            "operation_id": str(request["operation_id"]),
            "epoch": int(request.get("epoch", request.get("expected_epoch", 0))),
            "model_family": family.lower(),
            "perspective": self.perspective.value,
            "source_tp": source_tp,
            "target_tp": target_tp,
            "parameters": tuple(
                (
                    name,
                    _rule_payload(rules.get(name)),
                    tuple(param.shape),
                    str(param.dtype),
                )
                for name, param in sorted(actual.items())
            ),
        }

    def _select_shadow_device(self, target_bytes, joining):
        if self.shadow_device == "cpu":
            return "cpu"
        if self.shadow_device == "cuda":
            return "cuda"
        if joining or not torch.cuda.is_available():
            return "cpu"
        reserve = int(
            os.getenv("AFD_COMPONENT_SHADOW_HEADROOM_BYTES", str(2 * 1024**3))
        )
        free_bytes, _ = torch.cuda.mem_get_info()
        return "cuda" if free_bytes >= target_bytes + reserve else "cpu"

    @torch.no_grad()
    def prepare(self, request):
        reason = self.safety_reason(request)
        if reason:
            raise RuntimeError(reason)
        if self.prepared is not None:
            raise RuntimeError("a component weight transition is already prepared")
        started = time.perf_counter()
        source_tp, target_tp = self._source_tp(request), self._target_tp(request)
        operation_transport = self._operation_transport(source_tp, target_tp)
        begin_metrics = getattr(operation_transport, "begin_operation_metrics", None)
        if begin_metrics is not None:
            begin_metrics(str(request["operation_id"]))
        rank = operation_transport.rank
        logger.info(
            "AFD component PREPARE start op=%s rank=%d TP%d->TP%d "
            "perspective=%s configured_backend=%s transport=%s",
            request.get("operation_id"),
            rank,
            source_tp,
            target_tp,
            self.perspective.value,
            self._transport_name(self.transport),
            self._transport_name(operation_transport),
        )
        # Fast test mode: skip the expensive real weight D2H/gather and build a
        # minimal no-op shadow placeholder that reaches READY immediately. This
        # still runs the descriptor broadcast (a small control-group collective)
        # so the full fan-out/control/drain/commit/activate protocol path is
        # exercised, but no real parameter bytes move. Gated STRICTLY behind
        # AFD_RESHARD_FAKE_STAGING=1; default OFF, never affects production.
        if os.getenv("AFD_RESHARD_FAKE_STAGING", "0") == "1":
            descriptor = operation_transport.broadcast_manifest(
                self._descriptor(request, source_tp, target_tp) if rank == 0 else None
            )
            manifest = ReshardManifest(
                descriptor["operation_id"],
                descriptor["epoch"],
                descriptor["model_family"],
                descriptor["perspective"],
                source_tp,
                target_tp,
                (),
                None,
            )
            manifest = ReshardManifest(
                **{**manifest.__dict__, "manifest_hash": manifest.computed_hash()}
            )
            elapsed = time.perf_counter() - started
            # no_op short-circuits activate() (no group rebuild, no H2D copy),
            # and fake=True lets the scheduler skip refreshing serving groups so
            # tp_size never diverges from the (unchanged) real process groups.
            self.prepared = ComponentStagingState(
                manifest.operation_id,
                manifest.epoch,
                source_tp,
                target_tp,
                manifest,
                {},
                {},
                no_op=True,
                fake=True,
                timings_s={"prepare_s": elapsed},
                operation_transport=operation_transport,
            )
            logger.warning(
                "AFD component PREPARE [FAKE_STAGING] no-op ready op=%s rank=%d "
                "TP%d->TP%d prepare=%.3fs (NO real weight movement; test-only)",
                manifest.operation_id,
                rank,
                source_tp,
                target_tp,
                elapsed,
            )
            return self.prepared
        if source_tp == target_tp and self._is_ffn():
            descriptor = operation_transport.broadcast_manifest(
                self._descriptor(request, source_tp, target_tp) if rank == 0 else None
            )
            manifest = ReshardManifest(
                descriptor["operation_id"],
                descriptor["epoch"],
                descriptor["model_family"],
                descriptor["perspective"],
                source_tp,
                target_tp,
                (),
                None,
            )
            manifest = ReshardManifest(
                **{**manifest.__dict__, "manifest_hash": manifest.computed_hash()}
            )
            elapsed = time.perf_counter() - started
            self.prepared = ComponentStagingState(
                manifest.operation_id,
                manifest.epoch,
                source_tp,
                target_tp,
                manifest,
                {},
                {},
                no_op=True,
                timings_s={"prepare_s": elapsed},
                operation_transport=operation_transport,
            )
            logger.info(
                "AFD component PREPARE no-op ready op=%s rank=%d prepare=%.3fs",
                manifest.operation_id,
                rank,
                elapsed,
            )
            return self.prepared

        local_error = None
        try:
            descriptor_started = time.perf_counter()
            descriptor = operation_transport.broadcast_manifest(
                self._descriptor(request, source_tp, target_tp) if rank == 0 else None
            )
            descriptor_s = time.perf_counter() - descriptor_started
            if (
                descriptor["source_tp"] != source_tp
                or descriptor["target_tp"] != target_tp
            ):
                raise ReshardValidationError("component descriptor TP mismatch")
            model = getattr(self.runner, "model", None)
            local_params = (
                dict(model.named_parameters())
                if model is not None and rank < source_tp
                else {}
            )
            entries, shadows, deferred, cpu_bytes = [], {}, {}, 0
            staged_gpu_bytes = deferred_local_bytes = 0
            joining = model is None
            parameters = descriptor["parameters"]
            progress_every = max(int(os.getenv("AFD_RESHARD_PROGRESS_EVERY", "25")), 1)
            staging_started = time.perf_counter()
            for index, (name, raw_rule, expected_shape, expected_dtype) in enumerate(
                parameters, 1
            ):
                local = (
                    local_params[name].data
                    if rank < source_tp and name in local_params
                    else None
                )
                if self._cancel_requested.is_set():
                    raise RuntimeError("component PREPARE cancelled")
                rule = SplitRule.from_tp_rule(raw_rule)
                target = operation_transport.stage_target_shard(
                    name, local, rule, source_tp, target_tp, rank
                )
                if local is not None and (
                    tuple(local.shape) != tuple(expected_shape)
                    or str(local.dtype) != expected_dtype
                ):
                    raise ReshardValidationError(
                        f"{name}: local source tensor disagrees with manifest"
                    )
                from sglang.srt.reshard.afd_weight_reshard import Ownership

                source_shapes = tuple(tuple(expected_shape) for _ in range(source_tp))
                if rule.kind == "replicated":
                    target_shape = tuple(expected_shape)
                else:
                    target_shape_list = list(expected_shape)
                    target_shape_list[int(rule.dim)] = _target_shape_from_local(
                        rule, expected_shape, source_tp, target_tp
                    )
                    target_shape = tuple(target_shape_list)
                classified = AFDWeightFilter.classify_strict(name)
                if classified is None:
                    raise ReshardValidationError(f"{name}: unknown parameter ownership")
                ownership = Ownership(classified)
                element_size = torch.empty(
                    (), dtype=getattr(torch, expected_dtype.removeprefix("torch."))
                ).element_size()
                source_bytes = source_tp * element_size
                for dimension in expected_shape:
                    source_bytes *= int(dimension)
                # The descriptor validates shape/dtype. Direct transport does not
                # perform a costly content checksum, and every rank must hash the
                # same manifest regardless of which source shard it owns.
                checksum = "metadata:" + "0" * 64
                entry = ParameterManifest(
                    name,
                    ownership,
                    rule,
                    expected_dtype,
                    source_shapes,
                    tuple(target_shape for _ in range(target_tp)),
                    source_bytes,
                    target_tp * element_size * __import__("math").prod(target_shape),
                    tuple(checksum for _ in range(source_tp)),
                )
                entries.append(entry)
                defer_active = target_tp > source_tp and isinstance(
                    operation_transport,
                    (TorchDistributedNCCLMaxWorldTransport, InMemoryMaxWorldTransport),
                )
                if rank < source_tp and defer_active:
                    if target is not None:
                        raise ReshardValidationError(
                            f"{name}: active expansion rank unexpectedly staged a shadow"
                        )
                    deferred[name] = rule
                    deferred_local_bytes += (
                        target_tp
                        * element_size
                        * __import__("math").prod(target_shape)
                        // target_tp
                    )
                elif rank < target_tp:
                    if target is None:
                        raise ReshardValidationError(
                            f"{name}: target shard was not received"
                        )
                    target_bytes = target.numel() * target.element_size()
                    gloo_shrink = target_tp < source_tp and isinstance(
                        operation_transport, TorchDistributedNCCLMaxWorldTransport
                    )
                    # The online-safe NCCL transport shrink path already returns a
                    # rank-zero CPU target. Keep it on CPU (even when CUDA shadows
                    # were requested) so PREPARE cannot reintroduce GPU work after
                    # the ordered Gloo assembly. Expansion retains existing policy.
                    device = (
                        "cpu"
                        if gloo_shrink
                        else (
                            "cuda"
                            if target.device.type == "cuda"
                            else self._select_shadow_device(target_bytes, joining)
                        )
                    )
                    if device == "cpu":
                        cpu_bytes += target_bytes
                        if cpu_bytes > self.max_cpu_staging_bytes:
                            raise RuntimeError(
                                f"bounded CPU staging exceeded: {cpu_bytes} > "
                                f"{self.max_cpu_staging_bytes} bytes"
                            )
                    else:
                        staged_gpu_bytes += target_bytes
                    shadows[name] = (
                        target
                        if target.device.type == device
                        else target.to(device=device, copy=True)
                    )
                if (
                    index == 1
                    or index % progress_every == 0
                    or index == len(parameters)
                ):
                    logger.info(
                        "AFD component PREPARE progress op=%s rank=%d %d/%d params "
                        "cpu_shadow=%.2fGiB staged_gpu=%.2fGiB deferred=%.2fGiB elapsed=%.2fs",
                        request.get("operation_id"),
                        rank,
                        index,
                        len(parameters),
                        cpu_bytes / 1024**3,
                        staged_gpu_bytes / 1024**3,
                        deferred_local_bytes / 1024**3,
                        time.perf_counter() - staging_started,
                    )
            base = ReshardManifest(
                descriptor["operation_id"],
                descriptor["epoch"],
                descriptor["model_family"],
                descriptor["perspective"],
                source_tp,
                target_tp,
                tuple(entries),
                None,
            )
            manifest = ReshardManifest(
                **{**base.__dict__, "manifest_hash": base.computed_hash()}
            )
            manifest.validate_hash()
            total_s = time.perf_counter() - started
            transport_metrics = getattr(
                operation_transport, "snapshot_operation_metrics",
                lambda: dict(_RESHARD_METRIC_DEFAULTS),
            )()
            timings = {
                **dict(_RESHARD_METRIC_DEFAULTS),
                **transport_metrics,
                "group_prepare_s": 0.0,
                "descriptor_s": descriptor_s,
                "parameter_staging_s": time.perf_counter() - staging_started,
                # Compatibility only: this is the enclosing staging wall time,
                # not a transfer-only metric. New analysis must use the mutually
                # defined local/peer/host fields above.
                "weight_transfer_s": time.perf_counter() - staging_started,
                "prepare_s": total_s,
                "staged_gpu_bytes": staged_gpu_bytes,
                "deferred_local_bytes": deferred_local_bytes,
            }
            self.prepared = ComponentStagingState(
                manifest.operation_id,
                manifest.epoch,
                source_tp,
                target_tp,
                manifest,
                shadows,
                {},
                cpu_bytes=cpu_bytes,
                staged_gpu_bytes=staged_gpu_bytes,
                deferred_local_bytes=deferred_local_bytes,
                deferred=deferred,
                timings_s=timings,
                operation_transport=operation_transport,
            )
            logger.info(
                "AFD component PREPARE ready op=%s rank=%d params=%d prepare=%.3fs "
                "descriptor=%.3fs staging=%.3fs staged_gpu=%.2fGiB deferred=%.2fGiB",
                manifest.operation_id,
                rank,
                len(entries),
                total_s,
                descriptor_s,
                timings["parameter_staging_s"],
                staged_gpu_bytes / 1024**3,
                deferred_local_bytes / 1024**3,
            )
            # --- Optimization: pre-create model shell on joining ranks during
            # PREPARE (background) so ACTIVATE skips the expensive load_model().
            # _ensure_inplace_reshard_dummy_model_loaded patches parallel_state
            # so the model is correctly TP-sharded for the target TP.
            if rank >= source_tp and rank < target_tp:
                shell_t0 = time.perf_counter()
                ensure = getattr(
                    self.runner, "afd_component_ensure_model_shell", None
                )
                if ensure is not None:
                    try:
                        ensure(target_tp)
                        timings["pre_ensure_shell_s"] = (
                            time.perf_counter() - shell_t0
                        )
                        logger.info(
                            "[AFD-reshard] op=%s PREPARE pre-ensure_shell "
                            "rank=%d target_tp=%d %.3fs",
                            manifest.operation_id,
                            rank,
                            target_tp,
                            timings["pre_ensure_shell_s"],
                        )
                    except Exception as shell_exc:
                        logger.warning(
                            "[AFD-reshard] op=%s PREPARE pre-ensure_shell "
                            "failed rank=%d: %s (will retry in ACTIVATE)",
                            manifest.operation_id,
                            rank,
                            shell_exc,
                        )
            # Pre-compute KV pool config hint so ACTIVATE can skip profiling.
            kv_hint_fn = getattr(
                self.runner, "afd_component_precompute_kv_hint", None
            )
            if kv_hint_fn is not None and rank < target_tp:
                kv_hint_t0 = time.perf_counter()
                try:
                    kv_hint_fn(target_tp)
                    timings["pre_kv_hint_s"] = time.perf_counter() - kv_hint_t0
                except Exception as kv_exc:
                    logger.warning(
                        "[AFD-reshard] op=%s PREPARE kv_hint failed rank=%d: %s",
                        manifest.operation_id,
                        rank,
                        kv_exc,
                    )
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        try:
            failure = operation_transport.consensus_error(local_error)
        except Exception as exc:
            self.abort(f"error consensus failed: {exc}")
            raise RuntimeError(
                f"component PREPARE consensus failed; local shadow aborted: {exc}"
            ) from exc
        if failure:
            self.abort(failure)
            raise RuntimeError(f"component PREPARE aborted on all ranks: {failure}")
        # All ranks enter this collective only after successful error consensus.
        # This ordering prevents a failed rank from waiting in consensus while a
        # successful rank waits in timing collection.
        self.prepared.rank_timings_s = tuple(
            operation_transport.gather_timings(self.prepared.timings_s)
        )
        return self.prepared

    def start_prepare(self, request):
        """Start PREPARE once and return without waiting for tensor movement."""
        operation_id = str(request["operation_id"])
        with self._lock:
            if self._prepare_thread is not None and self._prepare_thread.is_alive():
                if self._prepare_operation == operation_id:
                    return self.prepare_status(operation_id)
                raise RuntimeError("another component PREPARE is running")
            if self.prepared is not None:
                raise RuntimeError("a component weight transition is already prepared")
            self._prepare_error = None
            self._prepare_operation = operation_id
            self._cancel_requested.clear()
            payload = dict(request)
            self._prepare_thread = threading.Thread(
                target=self._run_prepare,
                args=(payload,),
                name=f"afd-component-stage-{self.perspective.value}-{operation_id}",
                daemon=True,
            )
            self._prepare_thread.start()
        return self.prepare_status(operation_id)

    def _run_prepare(self, request):
        try:
            gpu_id = getattr(self.runner, "gpu_id", None)
            if gpu_id is not None and gpu_id >= 0:
                torch.cuda.set_device(gpu_id)
            self.prepare(request)
        except Exception as exc:
            with self._lock:
                self._prepare_error = f"{type(exc).__name__}: {exc}"

    def prepare_status(self, operation_id=None):
        with self._lock:
            if operation_id is not None and self._prepare_operation not in (
                None,
                str(operation_id),
            ):
                return {
                    "state": "ERROR",
                    "error": "prepare operation mismatch",
                    "timings_s": {},
                }
            if self._prepare_error:
                state = "ERROR"
            elif self._prepare_thread is not None and self._prepare_thread.is_alive():
                # self.prepared is published before final timing consensus. Never
                # expose READY while the background thread can still own staging
                # control/NCCL groups; ACTIVATE must not overlap those collectives.
                state = "PREPARING"
            elif (
                self.prepared is not None
                and self.prepared.state == TransactionState.READY
            ):
                state = "READY"
            else:
                state = "IDLE"
            return {
                "state": state,
                "error": self._prepare_error,
                "timings_s": dict(getattr(self.prepared, "timings_s", {}) or {}),
                "rank_timings_s": list(
                    getattr(self.prepared, "rank_timings_s", ()) or ()
                ),
            }

    def cancel_prepare(self, reason="cancelled"):
        self._cancel_requested.set()
        with self._lock:
            if self._prepare_thread is None or not self._prepare_thread.is_alive():
                self.abort(reason)

    def _log_activate_phase(self, state, phase, started, **extra):
        elapsed = time.perf_counter() - started
        state.timings_s[f"activate_{phase}_s"] = elapsed
        fields = " ".join(f"{key}={value}" for key, value in extra.items())
        logger.info(
            "[AFD-reshard] op=%s stage=activate phase=%s perspective=%s "
            "rank=%d elapsed=%.3fs %s",
            state.operation_id,
            phase,
            self.perspective.value,
            self.transport.rank,
            elapsed,
            fields,
        )

    @torch.no_grad()
    def activate(self, request):
        activate_started = time.perf_counter()
        state = self.prepared
        operation_transport = (
            state.operation_transport if state is not None else self.transport
        )
        prepare_thread = self._prepare_thread
        if prepare_thread is not None and prepare_thread.is_alive():
            raise RuntimeError(
                "component background PREPARE still owns staging collectives"
            )
        if state is None or state.state != TransactionState.READY:
            raise RuntimeError("component shadow is not READY")
        if state.operation_id != str(request["operation_id"]):
            raise RuntimeError("prepared operation mismatch")
        if state.no_op:
            local_error = None
            try:
                # Fake staging deliberately exercises only the control protocol;
                # it must never mutate the real serving data plane.
                if not state.fake:
                    self._refresh_peer_topology(
                        request,
                        reset_data_plane=self._peer_topology_changed(request),
                    )
            except Exception as exc:
                local_error = f"{type(exc).__name__}: {exc}"
            failure = operation_transport.consensus_error(local_error)
            if failure:
                self.abort(failure)
                raise RuntimeError(
                    f"component ACTIVATE aborted on all ranks: {failure}"
                )
            state.state = TransactionState.COMMITTED
            # Keep no-op activation behind the same max-world success boundary:
            # peer communicator reset completes before any rank resumes serving.
            operation_transport.barrier()
            state.timings_s.update(dict(_RESHARD_METRIC_DEFAULTS))
            state.timings_s.setdefault("group_prepare_s", 0.0)
            state.timings_s["activate_s"] = time.perf_counter() - activate_started
            state.rank_timings_s = tuple(
                operation_transport.gather_timings(state.timings_s)
            )
            retire_started = time.perf_counter()
            _retire_state = state
            threading.Thread(
                target=self._retire_committed_state,
                name=f"afd-retire-{_retire_state.operation_id[:8]}",
                daemon=True,
            ).start()
            state.timings_s["retire_s"] = time.perf_counter() - retire_started
            self.last_completed_status = {
                "operation_id": state.operation_id,
                "timings_s": dict(state.timings_s),
                "rank_timings_s": [dict(item) for item in state.rank_timings_s],
            }
            return
        local_error = None
        try:
            phase_started = time.perf_counter()
            logger.info(
                "[AFD-reshard] op=%s stage=activate phase=barrier_before_begin "
                "perspective=%s rank=%d",
                state.operation_id,
                self.perspective.value,
                operation_transport.rank,
            )
            operation_transport.barrier()
            self._log_activate_phase(state, "barrier_before", phase_started)
            phase_started = time.perf_counter()
            logger.info(
                "[AFD-reshard] op=%s stage=activate phase=rebuild_groups_begin "
                "perspective=%s rank=%d target_tp=%d",
                state.operation_id,
                self.perspective.value,
                operation_transport.rank,
                state.target_tp,
            )
            self.runner.afd_component_rebuild_groups(state.target_tp)
            self._log_activate_phase(
                state, "rebuild_groups", phase_started, target_tp=state.target_tp
            )
            # Rebind every scheduler/worker serving-group reference while the
            # feature-local activation still owns the event-loop safe point.
            # No rank can return to request broadcast before the final barrier.
            if self.group_refresh_callback is not None:
                phase_started = time.perf_counter()
                self.group_refresh_callback(state.target_tp)
                self._log_activate_phase(
                    state, "refresh_scheduler_groups", phase_started
                )
            if operation_transport.rank < state.target_tp:
                joining = getattr(self.runner, "model", None) is None
                reserve = int(
                    os.getenv("AFD_COMPONENT_SHADOW_HEADROOM_BYTES", str(2 * 1024**3))
                )
                if torch.cuda.is_available():
                    free_bytes, _ = torch.cuda.mem_get_info()
                    if joining:
                        shell_headroom = int(
                            os.getenv(
                                "AFD_COMPONENT_JOINING_SHELL_HEADROOM_BYTES",
                                str(state.staged_gpu_bytes),
                            )
                        )
                        required = shell_headroom + reserve
                    else:
                        required = state.deferred_local_bytes + reserve
                    if free_bytes < required:
                        logger.warning(
                            "[AFD-reshard] op=%s headroom low: "
                            "free=%d required=%d joining=%s "
                            "staged_gpu=%d deferred=%d reserve=%d "
                            "(proceeding optimistically)",
                            state.operation_id,
                            free_bytes, required, joining,
                            state.staged_gpu_bytes,
                            state.deferred_local_bytes, reserve,
                        )
                if joining:
                    ensure = getattr(
                        self.runner, "afd_component_ensure_model_shell", None
                    )
                    if ensure is None:
                        raise RuntimeError("joining rank has no model shell hook")
                    phase_started = time.perf_counter()
                    logger.info(
                        "[AFD-reshard] op=%s stage=activate phase=ensure_shell_begin "
                        "perspective=%s rank=%d",
                        state.operation_id,
                        self.perspective.value,
                        operation_transport.rank,
                    )
                    ensure(state.target_tp)
                    self._log_activate_phase(state, "ensure_shell", phase_started)
                params = dict(self.runner.model.named_parameters())
                old = {}
                cpu_h2d_bytes = sum(
                    shadow.numel() * shadow.element_size()
                    for name, shadow in state.shadow.items()
                    if shadow.device.type == "cpu"
                    and name in params
                    and params[name].device.type == "cuda"
                )
                if not joining and cpu_h2d_bytes:
                    free_bytes, _ = torch.cuda.mem_get_info()
                    if free_bytes < cpu_h2d_bytes + reserve:
                        logger.warning(
                            "[AFD-reshard] op=%s active-rank H2D headroom low: "
                            "free=%d shadow=%d reserve=%d "
                            "(proceeding optimistically)",
                            state.operation_id,
                            free_bytes,
                            cpu_h2d_bytes,
                            reserve,
                        )
                if not joining and state.deferred and torch.cuda.is_available():
                    free_before, _ = torch.cuda.mem_get_info()
                    if free_before < state.deferred_local_bytes + reserve:
                        cache_trim_started = time.perf_counter()
                        torch.cuda.empty_cache()
                        cache_trim_elapsed = time.perf_counter() - cache_trim_started
                        free_after, _ = torch.cuda.mem_get_info()
                        state.timings_s["materialize_cache_trim_s"] = (
                            cache_trim_elapsed
                        )
                        logger.info(
                            "[AFD-reshard] op=%s active expansion deferred "
                            "materialization cache trim: free_before=%d "
                            "free_after=%d reclaimed=%d elapsed=%.3fs",
                            state.operation_id,
                            free_before,
                            free_after,
                            free_after - free_before,
                            cache_trim_elapsed,
                        )
                materialize_started = time.perf_counter()
                materialized_count = 0
                materialize_h2d_s = 0.0
                materialize_h2d_bytes = 0
                for entry in state.manifest.parameters:
                    name = entry.name
                    if name not in params:
                        raise RuntimeError(f"prepared parameter disappeared: {name}")
                    param = params[name]
                    if not joining:
                        old[name] = param.data
                    if name in state.deferred:
                        replacement = operation_transport.materialize_deferred_target(
                            name,
                            param.data,
                            state.deferred[name],
                            state.source_tp,
                            state.target_tp,
                            operation_transport.rank,
                        )
                        if replacement is None:
                            raise RuntimeError(
                                f"deferred active target was not materialized: {name}"
                            )
                        replacement = replacement.contiguous()
                    else:
                        shadow = state.shadow[name]
                        if joining:
                            # Drop dummy parameter storage before adopting the ready
                            # GPU shadow. The shell is still built all-at-once, so the
                            # preflight above reserves one shadow-sized lower bound.
                            param.data = torch.empty(
                                0, dtype=param.dtype, device=param.device
                            )
                        if shadow.device.type == "cpu" and param.device.type == "cuda":
                            h2d_started = time.perf_counter()
                            replacement = shadow.to(
                                device=self.runner.device, non_blocking=False
                            )
                            materialize_h2d_s += time.perf_counter() - h2d_started
                            materialize_h2d_bytes += _tensor_bytes(replacement)
                        else:
                            replacement = shadow
                    param.data = replacement
                    materialized_count += 1
                state.timings_s["materialize_h2d_s"] = materialize_h2d_s
                state.timings_s["materialize_h2d_bytes"] = materialize_h2d_bytes
                transport_metrics = getattr(
                    operation_transport, "snapshot_operation_metrics", lambda: {}
                )()
                state.timings_s.update(transport_metrics)
                self._log_activate_phase(
                    state,
                    "materialize",
                    materialize_started,
                    parameters=materialized_count,
                    joining=joining,
                )
                state.old_tensors = old
                # Release P2P staging memory and PyTorch cache after weight
                # replacement. Source ranks now hold smaller TP-sharded tensors
                # and the old (larger) data is only retained for rollback.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                phase_started = time.perf_counter()
                self._refresh_peer_topology(request, reset_data_plane=False)
                self.runner.afd_component_refresh_runtime(
                    state.target_tp, self.perspective
                )
                self._log_activate_phase(state, "refresh_runtime", phase_started)
                if self.topology_refresh_callback is not None and not self._is_ffn():
                    phase_started = time.perf_counter()
                    self.topology_refresh_callback(
                        state.target_tp, int(request.get("epoch", 0)) + 1
                    )
                    self._log_activate_phase(
                        state, "refresh_bootstrap_topology", phase_started
                    )
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
            logger.exception(
                "[AFD-reshard] op=%s stage=activate perspective=%s rank=%d "
                "local activation failed: %s",
                state.operation_id,
                self.perspective.value,
                operation_transport.rank,
                local_error,
            )
        consensus_started = time.perf_counter()
        logger.info(
            "[AFD-reshard] op=%s stage=activate phase=consensus_begin "
            "perspective=%s rank=%d local_error=%s",
            state.operation_id,
            self.perspective.value,
            operation_transport.rank,
            bool(local_error),
        )
        failure = operation_transport.consensus_error(local_error)
        self._log_activate_phase(state, "consensus", consensus_started)
        if failure:
            # Every rank takes the same rollback branch after consensus.  Active
            # ranks restore retained live tensors; joining ranks release their
            # just-created shell and return to the source topology/standby loop.
            if operation_transport.rank < state.target_tp:
                params = (
                    dict(self.runner.model.named_parameters())
                    if getattr(self.runner, "model", None) is not None
                    else {}
                )
                for name, old_tensor in state.old_tensors.items():
                    if name in params:
                        params[name].data = old_tensor
                if operation_transport.rank >= state.source_tp:
                    demote = getattr(self.runner, "afd_component_demote_rank", None)
                    if demote is not None:
                        demote(self.perspective)
            expected_attn_tp, expected_ffn_tp = self._topology(request, "expected")
            self.runner.afd_component_refresh_peer_topology(
                expected_attn_tp, expected_ffn_tp, reset_data_plane=False
            )
            self.runner.afd_component_rebuild_groups(state.source_tp)
            # destroy_model_parallel() invalidated the target references too;
            # rollback must publish the rebuilt source groups before activation
            # unwinds back to either the active or standby scheduler loop.
            if self.group_refresh_callback is not None:
                self.group_refresh_callback(state.source_tp)
            self.abort(failure)
            raise RuntimeError(f"component ACTIVATE aborted on all ranks: {failure}")
        state.state = TransactionState.COMMITTED
        # This is the commit-success boundary. Rollback tensors must remain live
        # until every max-world rank has observed successful activation. Once the
        # barrier returns, retirement is rank-local: no later control command may
        # require ranks that have already resumed serving collectives.
        final_barrier_started = time.perf_counter()
        logger.info(
            "[AFD-reshard] op=%s stage=activate phase=final_barrier_begin "
            "perspective=%s rank=%d",
            state.operation_id,
            self.perspective.value,
            operation_transport.rank,
        )
        operation_transport.barrier()
        self._log_activate_phase(state, "final_barrier", final_barrier_started)
        state.timings_s["activate_s"] = time.perf_counter() - activate_started
        state.rank_timings_s = tuple(
            operation_transport.gather_timings(state.timings_s)
        )
        # Defer local cleanup to background thread — the reshard protocol
        # is complete after the final barrier; serving can resume immediately.
        retire_started = time.perf_counter()
        _retire_state = state  # keep alive for the background thread
        threading.Thread(
            target=self._retire_committed_state,
            name=f"afd-retire-{_retire_state.operation_id[:8]}",
            daemon=True,
        ).start()
        state.timings_s["retire_s"] = time.perf_counter() - retire_started
        self.last_completed_status = {
            "operation_id": state.operation_id,
            "timings_s": dict(state.timings_s),
            "rank_timings_s": [dict(item) for item in state.rank_timings_s],
        }
        logger.info(
            "[AFD-reshard-breakdown] op=%s perspective=%s metrics=%s",
            state.operation_id, self.perspective.value, self.last_completed_status,
        )

    def _retire_committed_state(self) -> None:
        """Release one committed shadow without any max-world collective.

        This method deliberately accepts COMMITTED only. READY state still owns
        rollback data and must go through abort(), never committed retirement.
        Post-commit cleanup is best-effort because activation cannot roll back
        after the final success barrier.
        """
        state = self.prepared
        if state is None:
            return
        if state.state != TransactionState.COMMITTED:
            raise RuntimeError("component shadow is not COMMITTED")
        logger.info(
            "[AFD-reshard] retire: rank=%d target_tp=%d source_tp=%d fake=%s",
            self.transport.rank, state.target_tp, state.source_tp, state.fake,
        )
        try:
            if self.transport.rank >= state.target_tp and not state.fake:
                logger.info(
                    "[AFD-reshard] retire: demoting rank %d (target_tp=%d source_tp=%d)",
                    self.transport.rank, state.target_tp, state.source_tp,
                )
                demote = getattr(self.runner, "afd_component_demote_rank", None)
                if demote is None:
                    logger.error(
                        "Committed AFD component shrink rank %d lacks demotion hook",
                        self.transport.rank,
                    )
                else:
                    try:
                        demote(self.perspective)
                    except Exception:
                        logger.exception(
                            "Post-commit AFD component demotion failed on rank %d",
                            self.transport.rank,
                        )
        finally:
            # Parameter.data owns committed shadow tensors now; dropping these
            # dictionaries only releases staging/rollback references.
            state.shadow.clear()
            state.deferred.clear()
            state.old_tensors.clear()
            self.prepared = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def abort(self, reason=""):
        state = self.prepared
        if state is not None and state.state != TransactionState.COMMITTED:
            state.shadow.clear()
            state.deferred.clear()
            state.old_tensors.clear()
            state.state = TransactionState.ABORTED
        self.prepared = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def retire(self, request):
        # Successful activation retires committed state locally on every rank.
        # Keep the external rank-zero RETIRE idempotent; importantly, it must not
        # reinterpret a READY shadow as committed cleanup.
        state = self.prepared
        if state is None:
            return
        if state.state == TransactionState.COMMITTED:
            self._retire_committed_state()
