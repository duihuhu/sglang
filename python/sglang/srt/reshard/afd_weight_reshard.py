"""Pure CPU-capable AFD weight resharding and transactional staging."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch

from sglang.srt.layers.afd_mixin import AFDWeightFilter
from sglang.srt.layers.afd_type import AFDPerspective


class ReshardValidationError(ValueError):
    pass


class Ownership(str, Enum):
    ATTN = "attn"
    FFN = "ffn"
    SHARED = "shared"


class TransactionState(str, Enum):
    PREPARING = "PREPARING"
    READY = "READY"
    COMMITTED = "COMMITTED"
    ABORTED = "ABORTED"


@dataclass(frozen=True)
class SplitRule:
    kind: str
    dim: Optional[int] = None
    segments: Tuple[int, ...] = ()

    @classmethod
    def from_tp_rule(cls, rule: Optional[Tuple]) -> "SplitRule":
        if rule is None:
            return cls("replicated")
        if not isinstance(rule, tuple) or not rule:
            raise ReshardValidationError(f"invalid TP split rule: {rule!r}")
        if rule[0] in ("column", "row") and len(rule) == 2:
            return cls(rule[0], int(rule[1]))
        if rule[0] == "column_fused" and len(rule) == 3:
            return cls(rule[0], int(rule[1]), tuple(int(x) for x in rule[2]))
        raise ReshardValidationError(f"unknown TP split rule: {rule!r}")


@dataclass(frozen=True)
class ParameterManifest:
    name: str
    ownership: Ownership
    rule: SplitRule
    dtype: str
    source_shapes: Tuple[Tuple[int, ...], ...]
    target_shapes: Tuple[Tuple[int, ...], ...]
    source_bytes: int
    target_bytes: int
    source_checksums: Tuple[str, ...]


@dataclass(frozen=True)
class ReshardManifest:
    operation_id: str
    epoch: int
    model_family: str
    perspective: str
    source_tp: int
    target_tp: int
    parameters: Tuple[ParameterManifest, ...]
    # Reserved for a future implementation that understands KV-head groups.
    kv_replication_metadata: Optional[Mapping[str, object]] = None
    manifest_hash: str = field(default="")

    def canonical_payload(self):
        payload = asdict(self)
        payload.pop("manifest_hash", None)
        return payload

    def computed_hash(self):
        raw = json.dumps(
            self.canonical_payload(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(raw).hexdigest()

    def validate(self):
        if not self.operation_id or self.epoch < 0:
            raise ReshardValidationError("invalid manifest operation or epoch")
        if self.source_tp < 1 or self.target_tp < 1:
            raise ReshardValidationError("manifest TP degrees must be positive")
        if self.kv_replication_metadata:
            raise ReshardValidationError(
                "GQA KV replication is not supported by AFD resharding"
            )
        names = [entry.name for entry in self.parameters]
        if len(names) != len(set(names)):
            raise ReshardValidationError("manifest contains duplicate parameters")
        for entry in self.parameters:
            _validate_manifest_entry(entry, self.source_tp, self.target_tp)

    def validate_hash(self):
        actual = self.computed_hash()
        if not self.manifest_hash or self.manifest_hash != actual:
            raise ReshardValidationError(
                "manifest hash mismatch: "
                f"expected={self.manifest_hash!r}, actual={actual}"
            )
        self.validate()


@dataclass(frozen=True)
class StagingChunk:
    parameter: str
    target_rank: int
    offset_bytes: int
    length_bytes: int
    device: str = "cpu"


@dataclass(frozen=True)
class StagingPlan:
    chunks: Tuple[StagingChunk, ...]
    total_bytes: int
    max_chunk_bytes: int


def tensor_checksum(tensor: torch.Tensor, mode: Optional[str] = None) -> str:
    """Checksum without a production-sized full-tensor CPU/numpy copy by default."""
    mode = (mode or os.getenv("AFD_RESHARD_CHECKSUM_MODE", "off")).lower()
    metadata = f"{tensor.dtype}:{tuple(tensor.shape)}:{tensor.numel()}".encode()
    if mode in ("off", "metadata"):
        return "metadata:" + hashlib.sha256(metadata).hexdigest()
    if mode == "sample":
        flat = tensor.detach().reshape(-1)
        count = min(flat.numel(), int(os.getenv("AFD_RESHARD_CHECKSUM_SAMPLE_ELEMENTS", "1024")))
        if count:
            indices = torch.linspace(0, flat.numel() - 1, count, device=flat.device).long()
            sample = flat.index_select(0, indices).cpu().contiguous().view(torch.uint8).numpy().tobytes()
        else:
            sample = b""
        return "sample:" + hashlib.sha256(metadata + sample).hexdigest()
    if mode == "strict":
        raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
        return "strict:" + hashlib.sha256(raw).hexdigest()
    raise ValueError("AFD_RESHARD_CHECKSUM_MODE must be off, metadata, sample, or strict")


def _checksum_mode(checksum: str) -> str:
    prefix = checksum.partition(":")[0]
    return "off" if prefix == "metadata" else prefix


def _validate_dense(name, tensor):
    lower = name.lower()
    if any(x in lower for x in ("quant", "qweight", "qzeros", "scales", "g_idx")):
        raise ReshardValidationError(f"quantized parameter is unsupported: {name}")
    if tensor.layout != torch.strided or tensor.is_sparse:
        raise ReshardValidationError(f"only dense strided tensors are supported: {name}")


def _dtype_from_string(value: str) -> torch.dtype:
    name = value.removeprefix("torch.")
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ReshardValidationError(f"unsupported manifest dtype: {value!r}")
    return dtype


def _shape_numel(shape: Sequence[int]) -> int:
    result = 1
    for dimension in shape:
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 0:
            raise ReshardValidationError(f"invalid tensor shape: {tuple(shape)!r}")
        result *= dimension
    return result


def _target_shape(shape, rule, old_tp, new_tp):
    if rule.kind == "replicated":
        return tuple(shape)
    if rule.dim is None or not -len(shape) <= rule.dim < len(shape):
        raise ReshardValidationError(f"invalid split dimension {rule.dim} for {shape}")
    dimension = rule.dim % len(shape)
    full = list(shape)
    full[dimension] *= old_tp
    if full[dimension] % new_tp:
        raise ReshardValidationError(
            f"split dimension {full[dimension]} is not divisible by TP={new_tp}"
        )
    full[dimension] //= new_tp
    return tuple(full)


def _validate_manifest_entry(entry, source_tp, target_tp):
    classified = AFDWeightFilter.classify_strict(entry.name)
    if classified is None or Ownership(classified) != entry.ownership:
        raise ReshardValidationError(
            f"{entry.name}: ownership invariant mismatch"
        )
    if entry.rule.kind not in ("replicated", "column", "row", "column_fused"):
        raise ReshardValidationError(
            f"{entry.name}: unsupported manifest split rule {entry.rule.kind!r}"
        )
    if entry.rule.kind == "replicated" and (
        entry.rule.dim is not None or entry.rule.segments
    ):
        raise ReshardValidationError(f"{entry.name}: invalid replicated split rule")
    if entry.rule.kind == "column_fused":
        if (
            entry.rule.dim != 0
            or not entry.rule.segments
            or any(segment <= 0 for segment in entry.rule.segments)
        ):
            raise ReshardValidationError(f"{entry.name}: invalid fused segment rule")
        if any(
            segment % source_tp or segment % target_tp
            for segment in entry.rule.segments
        ):
            raise ReshardValidationError(
                f"{entry.name}: fused segment is not TP-divisible"
            )
    if len(entry.source_shapes) != source_tp:
        raise ReshardValidationError(f"{entry.name}: source shape count mismatch")
    if len(entry.target_shapes) != target_tp:
        raise ReshardValidationError(f"{entry.name}: target shape count mismatch")
    if len(entry.source_checksums) != source_tp:
        raise ReshardValidationError(f"{entry.name}: source checksum count mismatch")
    if not entry.source_shapes or any(
        shape != entry.source_shapes[0] for shape in entry.source_shapes
    ):
        raise ReshardValidationError(f"{entry.name}: source shape mismatch")
    if entry.rule.kind == "column_fused" and sum(entry.rule.segments) != (
        entry.source_shapes[0][0] * source_tp
    ):
        raise ReshardValidationError(
            f"{entry.name}: fused segment sizes do not match full shape"
        )
    expected_target = _target_shape(
        entry.source_shapes[0], entry.rule, source_tp, target_tp
    )
    if any(shape != expected_target for shape in entry.target_shapes):
        raise ReshardValidationError(f"{entry.name}: target shape invariant mismatch")
    element_size = torch.empty((), dtype=_dtype_from_string(entry.dtype)).element_size()
    source_bytes = sum(_shape_numel(shape) * element_size for shape in entry.source_shapes)
    target_bytes = sum(_shape_numel(shape) * element_size for shape in entry.target_shapes)
    if entry.source_bytes != source_bytes:
        raise ReshardValidationError(f"{entry.name}: source byte invariant mismatch")
    if entry.target_bytes != target_bytes:
        raise ReshardValidationError(f"{entry.name}: target byte invariant mismatch")


def build_manifest(
    operation_id,
    epoch,
    perspective,
    source_tp,
    target_tp,
    named_source_shards,
    tp_rules,
    *,
    model_family="llama",
    kv_replication_metadata=None,
    checksum_mode=None,
):
    family = model_family.lower()
    if family not in ("llama", "qwen", "qwen2", "qwen3"):
        raise ReshardValidationError(f"unsupported dense model family: {model_family}")
    if not operation_id or epoch < 0 or source_tp < 1 or target_tp < 1:
        raise ReshardValidationError("invalid operation, epoch, or TP degree")
    if kv_replication_metadata:
        raise ReshardValidationError(
            "GQA KV replication is not supported by AFD resharding"
        )

    entries = []
    for name in sorted(named_source_shards):
        if not AFDWeightFilter.should_load(name, perspective):
            continue
        classified = AFDWeightFilter.classify_strict(name)
        if classified is None:
            raise ReshardValidationError(
                f"{name}: unknown or ambiguous parameter ownership"
            )
        ownership = Ownership(classified)
        shards = tuple(named_source_shards[name])
        if len(shards) != source_tp:
            raise ReshardValidationError(
                f"{name}: expected {source_tp} source shards"
            )
        for tensor in shards:
            _validate_dense(name, tensor)
        if any(t.dtype != shards[0].dtype for t in shards):
            raise ReshardValidationError(f"{name}: source dtype mismatch")
        shapes = tuple(tuple(t.shape) for t in shards)
        if any(shape != shapes[0] for shape in shapes):
            raise ReshardValidationError(f"{name}: source shape mismatch")

        raw_rule = tp_rules.get(name)
        tp_markers = (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "qkv_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "gate_up_proj",
            "embed_tokens",
            "lm_head",
        )
        if raw_rule is None and any(marker in name.lower() for marker in tp_markers):
            raise ReshardValidationError(f"{name}: missing TP split rule")
        rule = SplitRule.from_tp_rule(raw_rule)
        if rule.kind == "column_fused":
            if rule.dim != 0 or not rule.segments or any(x <= 0 for x in rule.segments):
                raise ReshardValidationError(f"{name}: invalid fused segment rule")
            if sum(rule.segments) != shapes[0][0] * source_tp:
                raise ReshardValidationError(
                    f"{name}: fused segment sizes do not match full shape"
                )
            if any(x % source_tp or x % target_tp for x in rule.segments):
                raise ReshardValidationError(
                    f"{name}: fused segment is not TP-divisible"
                )
        target_shape = _target_shape(shapes[0], rule, source_tp, target_tp)
        source_bytes = sum(t.numel() * t.element_size() for t in shards)
        target_bytes = (
            _shape_numel(target_shape) * shards[0].element_size() * target_tp
        )
        entries.append(
            ParameterManifest(
                name=name,
                ownership=ownership,
                rule=rule,
                dtype=str(shards[0].dtype),
                source_shapes=shapes,
                target_shapes=tuple(target_shape for _ in range(target_tp)),
                source_bytes=source_bytes,
                target_bytes=target_bytes,
                source_checksums=tuple(tensor_checksum(t, checksum_mode) for t in shards),
            )
        )

    base = ReshardManifest(
        operation_id,
        epoch,
        family,
        perspective.value,
        source_tp,
        target_tp,
        tuple(entries),
        kv_replication_metadata,
    )
    manifest = ReshardManifest(
        operation_id,
        epoch,
        family,
        perspective.value,
        source_tp,
        target_tp,
        tuple(entries),
        kv_replication_metadata,
        base.computed_hash(),
    )
    manifest.validate_hash()
    return manifest


def reshard_tensor_shards(source_shards, rule, target_tp):
    if not source_shards or target_tp < 1:
        raise ReshardValidationError(
            "source shards and positive target TP are required"
        )
    old_tp = len(source_shards)
    if rule.kind == "replicated":
        if any(not torch.equal(source_shards[0], shard) for shard in source_shards[1:]):
            raise ReshardValidationError("replicated source shards differ")
        return tuple(source_shards[0].detach().cpu().clone() for _ in range(target_tp))
    if rule.kind not in ("column", "row", "column_fused") or rule.dim is None:
        raise ReshardValidationError(f"unsupported split rule: {rule}")
    if any(tuple(t.shape) != tuple(source_shards[0].shape) for t in source_shards):
        raise ReshardValidationError("source shard shape mismatch")
    dim = rule.dim
    if rule.kind == "column_fused":
        local_offset, full_segments = 0, []
        for full_size in rule.segments:
            if full_size % old_tp or full_size % target_tp:
                raise ReshardValidationError("fused segment is not TP-divisible")
            local_size = full_size // old_tp
            full_segments.append(
                torch.cat(
                    [
                        shard.narrow(dim, local_offset, local_size)
                        for shard in source_shards
                    ],
                    dim=dim,
                )
            )
            local_offset += local_size
        if local_offset != source_shards[0].shape[dim]:
            raise ReshardValidationError(
                "fused local layout does not match segment rule"
            )
        return tuple(
            torch.cat(
                [
                    segment.narrow(
                        dim, rank * (size // target_tp), size // target_tp
                    )
                    for size, segment in zip(rule.segments, full_segments)
                ],
                dim=dim,
            )
            .contiguous()
            .cpu()
            for rank in range(target_tp)
        )
    full = torch.cat(tuple(source_shards), dim=dim)
    if full.shape[dim] % target_tp:
        raise ReshardValidationError(
            "full split dimension is not target-TP-divisible"
        )
    size = full.shape[dim] // target_tp
    return tuple(
        full.narrow(dim, rank * size, size).contiguous().cpu()
        for rank in range(target_tp)
    )


def build_staging_plan(manifest, max_chunk_bytes=64 * 1024 * 1024):
    manifest.validate_hash()
    if max_chunk_bytes < 1:
        raise ReshardValidationError("max_chunk_bytes must be positive")
    chunks, total = [], 0
    for parameter in manifest.parameters:
        if parameter.target_bytes % manifest.target_tp:
            raise ReshardValidationError(
                f"{parameter.name}: target bytes are not divisible by target TP"
            )
        per_target = parameter.target_bytes // manifest.target_tp
        for rank in range(manifest.target_tp):
            for offset in range(0, per_target, max_chunk_bytes):
                length = min(max_chunk_bytes, per_target - offset)
                chunks.append(StagingChunk(parameter.name, rank, offset, length))
                total += length
    return StagingPlan(tuple(chunks), total, max_chunk_bytes)


class AFDWeightReshardTransaction:
    """Two-phase shadow staging; commit never mutates live Parameter.data."""

    def __init__(self, manifest):
        self.manifest = manifest
        self.state = TransactionState.PREPARING
        self.shadow_tensors: Dict[int, Dict[str, torch.Tensor]] = {}
        self.abort_reason: Optional[str] = None

    def prepare(self, named_source_shards, expected_manifest_hash=None):
        if self.state != TransactionState.PREPARING:
            raise ReshardValidationError(f"cannot prepare from {self.state.value}")
        self.manifest.validate_hash()
        if (
            expected_manifest_hash
            and expected_manifest_hash != self.manifest.manifest_hash
        ):
            raise ReshardValidationError("executor manifest hash mismatch")
        required = {entry.name for entry in self.manifest.parameters}
        missing = required - set(named_source_shards)
        if missing:
            raise ReshardValidationError(
                f"manifest source parameters missing: {sorted(missing)}"
            )

        staged = {rank: {} for rank in range(self.manifest.target_tp)}
        for entry in self.manifest.parameters:
            shards = tuple(named_source_shards[entry.name])
            if len(shards) != self.manifest.source_tp:
                raise ReshardValidationError(
                    f"{entry.name}: source shard count changed"
                )
            if any(str(tensor.dtype) != entry.dtype for tensor in shards):
                raise ReshardValidationError(f"{entry.name}: source dtype changed")
            if tuple(tuple(t.shape) for t in shards) != entry.source_shapes:
                raise ReshardValidationError(f"{entry.name}: source shape changed")
            if tuple(
                tensor_checksum(t, _checksum_mode(expected))
                for t, expected in zip(shards, entry.source_checksums)
            ) != entry.source_checksums:
                raise ReshardValidationError(f"{entry.name}: source checksum mismatch")
            outputs = reshard_tensor_shards(
                shards, entry.rule, self.manifest.target_tp
            )
            if tuple(tuple(t.shape) for t in outputs) != entry.target_shapes:
                raise ReshardValidationError(f"{entry.name}: staged shape mismatch")
            if sum(t.numel() * t.element_size() for t in outputs) != entry.target_bytes:
                raise ReshardValidationError(f"{entry.name}: staged byte mismatch")
            for rank, tensor in enumerate(outputs):
                staged[rank][entry.name] = tensor
        self.shadow_tensors = staged
        self.state = TransactionState.READY

    def commit(self):
        if self.state != TransactionState.READY:
            raise ReshardValidationError(f"cannot commit from {self.state.value}")
        self.state = TransactionState.COMMITTED
        return self.shadow_tensors

    def abort(self, reason=""):
        if self.state == TransactionState.COMMITTED:
            raise ReshardValidationError("cannot abort a committed transaction")
        self.shadow_tensors.clear()
        self.abort_reason = reason
        self.state = TransactionState.ABORTED
