"""Static RDMA fabric planning for Mooncake PD deployments.

Each Mooncake worker is bound to exactly one HCA.  A fabric-local route group
contains Prefill and Decode instances whose communicating TP ranks use matching
HCAs, preventing Mooncake from pairing endpoints across disconnected fabrics.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_HCAS = ("mlx5_2", "mlx5_3", "mlx5_6")


def enabled() -> bool:
    return os.environ.get("MN_ENABLE_FABRIC_AFFINITY", "0").lower() in (
        "1",
        "true",
        "yes",
    )


def hcas() -> tuple[str, ...]:
    value = os.environ.get("MN_RDMA_FABRICS", ",".join(DEFAULT_HCAS))
    devices = tuple(item.strip() for item in value.split(",") if item.strip())
    if not devices or len(set(devices)) != len(devices):
        raise ValueError(f"Invalid MN_RDMA_FABRICS={value!r}")
    return devices


def single_hca_map(tp: int, hca: str) -> dict[str, str]:
    return {str(rank): hca for rank in range(tp)}


def pd_tp_components(
    prefill_tp: int, decode_tp: int
) -> list[tuple[list[int], list[int]]]:
    """Return connected P/D TP-rank components matching CommonKVManager mapping."""
    if prefill_tp == decode_tp:
        return [([rank], [rank]) for rank in range(prefill_tp)]
    if prefill_tp > decode_tp:
        if prefill_tp % decode_tp:
            raise ValueError(f"P TP {prefill_tp} is not divisible by D TP {decode_tp}")
        width = prefill_tp // decode_tp
        return [
            (list(range(d_rank * width, (d_rank + 1) * width)), [d_rank])
            for d_rank in range(decode_tp)
        ]
    if decode_tp % prefill_tp:
        raise ValueError(f"D TP {decode_tp} is not divisible by P TP {prefill_tp}")
    width = decode_tp // prefill_tp
    return [
        ([p_rank], list(range(p_rank * width, (p_rank + 1) * width)))
        for p_rank in range(prefill_tp)
    ]


@dataclass(frozen=True)
class PDFabricGroup:
    fabric: str
    prefill_indices: tuple[int, ...]
    decode_indices: tuple[int, ...]


def plan_pd_groups(
    num_prefill: int, prefill_tp: int, num_decode: int, decode_tp: int
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[PDFabricGroup]]:
    """Plan independent P/D instance pools and matching per-rank HCA maps.

    Each Prefill instance owns a route group and a disjoint Decode subset.  Rank
    components rotate across fabrics, so TP-parallel KV slices use independent
    links while every source/destination edge remains on the same fabric.
    """
    devices = hcas()
    if num_decode < num_prefill or num_decode % num_prefill:
        raise ValueError(
            "fabric-local PD routing requires Decode replicas divisible by Prefill replicas"
        )
    decodes_per_prefill = num_decode // num_prefill
    components = pd_tp_components(prefill_tp, decode_tp)
    p_maps: list[dict[str, str]] = []
    d_maps: list[dict[str, str]] = [{} for _ in range(num_decode)]
    groups: list[PDFabricGroup] = []
    for p_index in range(num_prefill):
        p_map: dict[str, str] = {}
        decode_indices = tuple(
            range(p_index * decodes_per_prefill, (p_index + 1) * decodes_per_prefill)
        )
        for component_index, (p_ranks, d_ranks) in enumerate(components):
            fabric = devices[(p_index + component_index) % len(devices)]
            for rank in p_ranks:
                p_map[str(rank)] = fabric
            for d_index in decode_indices:
                for rank in d_ranks:
                    d_maps[d_index][str(rank)] = fabric
        p_maps.append(p_map)
        groups.append(
            PDFabricGroup(
                "+".join(sorted(set(p_map.values()))), (p_index,), decode_indices
            )
        )
    return p_maps, d_maps, groups


def validate_single_hca_mapping(mapping: dict[str, str]) -> None:
    for rank, device in mapping.items():
        devices = [item.strip() for item in device.split(",") if item.strip()]
        if len(devices) != 1:
            raise ValueError(f"Rank {rank} must bind exactly one HCA, got {device!r}")


def plan_instance_affinity(
    num_prefill: int, num_decode: int
) -> tuple[list[str], list[str], list[tuple[int, ...]]]:
    """Assign every P to a fabric-local Decode pool without dropping replicas."""
    devices = hcas()
    if num_decode < 1:
        raise ValueError("At least one Decode instance is required")
    if num_decode >= num_prefill and num_decode % num_prefill == 0:
        width = num_decode // num_prefill
        p_to_d = [tuple(range(i * width, (i + 1) * width)) for i in range(num_prefill)]
        p_fabrics = [devices[i % len(devices)] for i in range(num_prefill)]
        d_fabrics = ["" for _ in range(num_decode)]
        for p_index, d_indices in enumerate(p_to_d):
            for d_index in d_indices:
                d_fabrics[d_index] = p_fabrics[p_index]
    else:
        d_fabrics = [devices[index % len(devices)] for index in range(num_decode)]
        p_to_d = [(index % num_decode,) for index in range(num_prefill)]
        p_fabrics = [d_fabrics[indices[0]] for indices in p_to_d]
    return p_fabrics, d_fabrics, p_to_d
