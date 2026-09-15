from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

LIGHT = ("fixed_qps16_light", 128, 64, 16)
FIXED = [
    ("fixed_short", 128, 128),
    ("fixed_prefill", 2048, 128),
    ("fixed_long_prefill", 4096, 128),
    ("fixed_decode", 128, 1024),
    ("fixed_balanced", 1024, 1024),
    ("fixed_long", 4096, 1024),
]

def poisson(n, qps, seed):
    rng = random.Random(seed)
    current = 0.0
    arrivals = []
    for _ in range(n):
        current += rng.expovariate(qps)
        arrivals.append(current)
    return arrivals


def _write(path, rows):
    with path.open("w") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


def _trace_rows(source, kind, qps, seed):
    rows = []
    if source and source.exists():
        for index, line in enumerate(source.read_text().splitlines()):
            if not line.strip():
                continue
            raw = json.loads(line)
            input_len = int(raw.get("input_len", raw.get("prompt_tokens", 128)))
            output_len = int(raw.get("output_len", raw.get("completion_tokens", 128)))
            rows.append({"request_id": f"{kind}-{qps}-{index:05d}", "input_len": input_len, "output_len": output_len, "source": kind})
    if not rows:
        rng = random.Random(seed)
        rows = [{"request_id": f"{kind}-{qps}-{index:05d}", "input_len": rng.randint(128, 2048 if kind == "conv" else 4096), "output_len": rng.randint(64, 512 if kind == "conv" else 1024), "source": f"synthetic_{kind}"} for index in range(64)]
    arrivals = poisson(len(rows), float(qps), seed + qps)
    return [dict(row, arrival_time_s=round(value, 6)) for row, value in zip(rows, arrivals)]


def _fixed_entry(out, name, input_len, output_len, qps, count, seed, provenance="generated_fixed"):
    path = out / f"{name}_qps{qps}.jsonl"
    arrivals = poisson(count, float(qps), seed + input_len + output_len + qps)
    _write(path, ({"request_id": f"{name}-{qps}-{index:05d}", "arrival_time_s": round(value, 6), "input_len": input_len, "output_len": output_len, "source": name} for index, value in enumerate(arrivals)))
    return {"name": name, "qps": qps, "path": path.name, "requests": count, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "provenance": provenance}


def generate(out: Path, count=64, qps_values=(1, 4, 8), seed=20260822, trace_dir: Path | None = None):
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for name, input_len, output_len in FIXED:
        for qps in qps_values:
            index.append(_fixed_entry(out, name, input_len, output_len, qps, count, seed))
    name, input_len, output_len, qps = LIGHT
    path = out / f"{name}_qps{qps}.jsonl"
    arrivals = poisson(count, float(qps), seed + input_len + output_len + qps)
    _write(path, ({"request_id": f"{name}-{qps}-{index:05d}", "arrival_time_s": round(value, 6), "input_len": input_len, "output_len": output_len, "timeout_s": 60, "source": name} for index, value in enumerate(arrivals)))
    index.append({"name": name, "qps": qps, "path": path.name, "requests": count, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "provenance": "generated_fixed"})
    for kind in ("conv", "code"):
        candidates = sorted(trace_dir.glob(f"macro_{kind}_qps*.jsonl")) if trace_dir and trace_dir.exists() else []
        source = candidates[0] if candidates else None
        for qps in qps_values:
            path = out / f"{kind}_qps{qps}.jsonl"
            rows = _trace_rows(source, kind, qps, seed + (0 if kind == "conv" else 10000))
            _write(path, rows)
            index.append({"name": kind, "qps": qps, "path": path.name, "requests": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "provenance": "source_trace" if source else "synthetic_proxy", "copied_from": str(source) if source else None})
    (out / "index.json").write_text(json.dumps({"version": 1, "entries": index}, indent=2) + "\n")
    return index
