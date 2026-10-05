"""Fixed-target workload construction independent of neighbor partition and labels."""

from dataclasses import dataclass
import hashlib
import json
import random


@dataclass(frozen=True)
class ServiceRequest:
    rid: str
    group: str
    input_ids: list[int]
    output_len: int
    arrival_offset_s: float
    cache_salt: str
    tenant_key: str
    common_prefix_len: int = 0


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def workload(run_id, target, neighbor, *, rep=1, waves=2, interval=1.5, first="A", swap=False):
    if target not in ("packed", "split") or neighbor not in ("alone", "packed", "split"):
        raise ValueError("Unknown shape")
    if waves < 1 or interval < 0 or first not in ("A", "B"):
        raise ValueError("Invalid arrival plan")
    result = []
    for role, shape in (("A", target), ("B", neighbor)):
        if shape == "alone":
            continue
        parts = 1 if shape == "packed" else 16
        for wave in range(waves):
            seed = int(digest(["m2-service-f1-v1", rep, role, wave])[:16], 16)
            rng = random.Random(seed)
            # Repeated blocks retain the previous synthetic workload family;
            # the canonical stream is identical across neighbor partitions.
            tokens = [t for _ in range(16) for t in [rng.randrange(1000, 151000)] * 1024]
            for index in range(parts):
                name = f"{role}-w{wave}-r{index}"
                length = 16384 // parts
                result.append(ServiceRequest(
                    rid=f"{run_id}-{name}", group=role,
                    input_ids=tokens[index * length:(index + 1) * length],
                    output_len=512 // parts,
                    arrival_offset_s=wave * interval + (0 if role == first else .02),
                    cache_salt=f"{run_id}-{name}",
                    tenant_key=("B" if role == "A" else "A") if swap else role))
    return sorted(result, key=lambda req: (req.arrival_offset_s, req.group))


def fingerprints(specs):
    result = {}
    for role in ("A", "B"):
        rows = [r for r in specs if r.group == role]
        if not rows:
            continue
        # Preserve canonical numeric wave/request ordering, not lexical r10/r2.
        rows.sort(key=lambda r: tuple(map(int, r.rid.rsplit(f"-{role}-w", 1)[1].split("-r"))))
        result[role] = {
            "request_shape_arrivals": digest([(r.input_ids, r.output_len, r.arrival_offset_s) for r in rows]),
            "canonical_input": digest([token for r in rows for token in r.input_ids]),
            "input_tokens": sum(len(r.input_ids) for r in rows),
            "output_tokens": sum(r.output_len for r in rows), "requests": len(rows)}
    return result
