#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from rq1lib_v2 import atomic_json, canonical_qps, load, workload_filename


def trace_seed(base_seed, qps, repeat):
    key = f"rq1-v2-arrival|{int(base_seed)}|{canonical_qps(qps)}|{int(repeat)}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


def normalized_arrivals(count, qps, seed):
    if count < 2:
        raise ValueError("request_count must be at least 2")
    rng = random.Random(seed)
    current = 0.0
    values = [0.0]
    for _ in range(1, count):
        current += rng.expovariate(float(qps))
        values.append(current)
    target_last = (count - 1) / float(qps)
    scale = target_last / values[-1]
    result = [value * scale for value in values]
    result[0], result[-1] = 0.0, target_last
    return result


def generate(output_dir=None):
    config = load(ROOT / "configs/workloads_v2.json")
    out = Path(output_dir) if output_dir else ROOT / "data/workloads_v2"
    out.mkdir(parents=True, exist_ok=True)
    for stale_path in out.glob("*.jsonl"):
        stale_path.unlink()
    entries = []
    for qps in config["qps"]:
        for repeat in config["repeats"]:
            count = int(
                config.get("request_count_by_qps", {}).get(
                    canonical_qps(qps), config["request_count"]
                )
            )
            seed = trace_seed(config["seed"], qps, repeat)
            arrivals = normalized_arrivals(count, qps, seed)
            for name in config["classes"]:
                spec = config["specs"][name]
                path = out / workload_filename(name, qps, repeat)
                rows = [{"request_id": f"rq1-v2-{name}-qps{canonical_qps(qps)}-repeat{repeat}-{index:05d}", "arrival_time_s": arrival, "input_len": spec["input_len"], "output_len": spec["output_len"], "source": f"rq1_v2_fixed_{name}"} for index, arrival in enumerate(arrivals)]
                content = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
                path.write_text(content)
                entries.append({"name": name, "qps": float(qps), "repeat": repeat, "path": path.name, "requests": len(rows), "input_len": spec["input_len"], "output_len": spec["output_len"], "arrival_seed": seed, "sha256": hashlib.sha256(content.encode()).hexdigest()})
    atomic_json(out / "index.json", {"schema_version": 2, "rq": "RQ1-v2", "frozen": True, "entries": entries})
    return {
        "files": len(entries),
        "request_counts": {
            canonical_qps(qps): int(
                config.get("request_count_by_qps", {}).get(
                    canonical_qps(qps), config["request_count"]
                )
            )
            for qps in config["qps"]
        },
        "index": str(out / "index.json"),
    }


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2))
