#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from aflex_benchmark.config import load_bundle
from aflex_benchmark.runner import expand_queue
from run_measurement_rdma_qps4 import write
import run_measurement_rdma4n_only as r4


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "results/raw/measurement_rdma_qps4_20260825")
    args = parser.parse_args()
    lock = r4.acquire(args.results)
    gate_path = args.results / r4.GATE_NAME
    gate_path.unlink(missing_ok=True)
    bundle = load_bundle(ROOT / "configs", ROOT / "configs/measurement_rdma_qps4_qwen3_30b_a3b_4gpu.json")
    base = {(item["point"]["architecture"], item["workload"]): item for item in expand_queue(bundle) if item["point"]["metadata"]["topology_id"] == "rdma4n"}
    records = []
    for ordinal, (arch, lane) in enumerate(zip(r4.ARCHITECTURES, r4.LANES), 1):
        item, actual, summary, path, ok, reason = r4.execute_once(
            bundle, base[arch, r4.WORKLOADS[0]], args.results, 0, ordinal, lane,
            [f"canary:rdma4n:{arch}"], 1, canary=True,
        )
        record = {"architecture": arch, "lane_gpu": lane, "run_id": item["run_id"], "artifact": str(path.resolve()), "layout": actual, "status": "pass" if ok else "fail", "reason": reason, "finished_at": now()}
        if ok:
            record["artifact_hash"] = r4.artifact_hash(path)
        records.append(record)
        write(args.results / "rdma4n_canary" / "canary.json", records)
        if not ok:
            raise RuntimeError(record)
    gate = {
        "schema_version": 1,
        "status": "pass",
        "generated_at": now(),
        "generator_pid": os.getpid(),
        "topology": {"lanes": [0, 2, 6], "nodes": ["node1", "node2", "node3", "node4"], "hca_shared": False},
        "artifacts": records,
    }
    write(gate_path, gate)
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
