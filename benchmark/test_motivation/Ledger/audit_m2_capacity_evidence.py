"""Audit logical cache bounds and existing traces, without running a GPU model."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


STATE_BYTES = 21708800
KV_BYTES = 8064
RAW_BYTES = 19481072000


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def logical_bound(shapes, budget=RAW_BYTES, request_cap=607):
    if not shapes or any(len(s) != 3 or any(not isinstance(x, int) or x < 1 for x in s) for s in shapes):
        raise ValueError("Need positive integer count/input/output shapes")
    count = sum(n for n, _, _ in shapes)
    tokens = sum(n * (p + o) for n, p, o in shapes)
    state, kv = count * STATE_BYTES, tokens * KV_BYTES
    return {"shapes": shapes, "requests": count, "context_token_upper": tokens,
            "state_body_upper_bytes": state, "kv_body_upper_bytes": kv,
            "joint_body_upper_bytes": state + kv,
            "shared_raw_bytes": budget, "logical_upper_headroom_bytes": budget - state - kv,
            "all_requests_below_request_cap": count < request_cap,
            "kv_body_alone_fits": kv <= budget,
            "joint_body_upper_exceeds_budget": state + kv > budget,
            "scope": "All requests hypothetically retain their full input+output at once; not an observed peak or allocator feasibility test."}


def audit_item(item):
    requests = item["requests"]
    if not requests or any(not r["success"] or r["cached_tokens"] or r["num_retractions"] for r in requests):
        raise ValueError("Invalid/prefix/retracted request")
    resource, batch = item["scheduler_resource_trace"], item["scheduler_batch_trace"]
    if not resource["events"] or resource.get("dropped_events") or batch.get("dropped_admission_blocks"):
        raise ValueError("Incomplete trace")
    states = item["server_initial"]["internal_states"]
    if len(states) != 1:
        raise ValueError("Expected one scheduler")
    raw = states[0]["m2_admission"]["shared_pool_bytes"]
    if raw != RAW_BYTES or states[0]["page_size"] != 1:
        raise ValueError("Wrong frozen cache configuration")
    if any(e["active_requests"] != e["state_slots"] for e in resource["events"]):
        raise ValueError("State/request equivalence changed")
    counts = Counter((r["input_len"], r["output_len"]) for r in requests)
    bound = logical_bound([(n, p, o) for (p, o), n in sorted(counts.items())], raw)
    peak_event = max(resource["events"], key=lambda e: e["state_slots"] * STATE_BYTES + e["kv_tokens"] * KV_BYTES)
    peak_bytes = peak_event["state_slots"] * STATE_BYTES + peak_event["kv_tokens"] * KV_BYTES
    if peak_bytes > bound["joint_body_upper_bytes"]:
        raise ValueError("Ownership peak exceeds request-derived bound")
    blocks = Counter(b["reason"] for b in batch["admission_blocks"])
    return {"experiment": item["experiment"], "logical_bound": bound,
            "peak_joint_live_body_bytes": peak_bytes, "minimum_logical_headroom_bytes": raw - peak_bytes,
            "peak_joint_event_state_slots": peak_event["state_slots"], "peak_joint_event_kv_tokens": peak_event["kv_tokens"],
            "admission_blocks": dict(blocks), "no_token_events": sum(n for reason, n in blocks.items() if "no_token" in reason.lower()),
            "current_active_at_end": resource["system"]["current_active_requests"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analysis = json.loads(args.analysis.read_text())
    if not analysis["complete"]:
        parser.error("Need a complete historical analysis")
    rows = []
    for row in analysis["rows"]:
        path = Path(row["source"])
        if digest(path) != analysis["artifact_sha256"][str(path)]:
            raise ValueError(f"Historical artifact changed: {path}")
        item = json.loads(path.read_text())
        rows.append({**audit_item(item), "source": str(path), "sha256": digest(path)})
    scenarios = {
        "historical_512_plus_64": logical_bound([(512, 512, 128), (64, 8192, 64)]),
        "held_out_384_plus_64": logical_bound([(384, 1024, 128), (64, 8192, 64)]),
        "unmeasured_24_probe_feasibility": logical_bound([(512, 512, 128), (64, 8192, 64), (24, 512, 64)]),
    }
    result = {"complete": True, "audit_script_sha256": digest(Path(__file__)),
              "historical_analysis": str(args.analysis), "historical_analysis_sha256": digest(args.analysis),
              "state_entry_bytes": STATE_BYTES, "kv_entry_bytes": KV_BYTES,
              "rows": rows, "theoretical_scenarios": scenarios,
              "scope": "Read-only historical audit. Body-byte occupancy omits reserved sink, fragmentation, transient reservations and workspace. An over-budget upper bound does not prove the peak is reachable."}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"audited_runs": len(rows), "no_token_events": sum(r["no_token_events"] for r in rows),
                      "peak_joint_live_body_range": [min(r["peak_joint_live_body_bytes"] for r in rows), max(r["peak_joint_live_body_bytes"] for r in rows)],
                      "theoretical_scenarios": scenarios}, indent=2))


if __name__ == "__main__":
    main()
