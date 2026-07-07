#!/usr/bin/env python3
"""Structured diff of golden-diff scheduler state JSON dumps.

Compares per-rank state files produced by SGLANG_RESHARD_STATE_DUMP, e.g.:
  golden TP2 normal_entry  vs  in-place reshard post_reshard

Usage:
  python3 diff_reshard_state.py \\
      --golden-dir /tmp/golden_diff/golden_tp2 \\
      --reshard-dir /tmp/golden_diff/inplace_tp2 \\
      --golden-tag normal_entry --reshard-tag post_reshard \\
      --out /tmp/golden_diff/diff_report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else k
            out.update(_flatten(v, key))
    else:
        out[prefix] = obj
    return out


def _load_rank(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _find_file(d: Path, scenario: str, tag: str, rank: int) -> Path | None:
    candidates = [
        d / f"state_{scenario}_{tag}_rank{rank}.json",
        d / f"state_{tag}_rank{rank}.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    # fallback: any file matching tag+rank
    for p in sorted(d.glob(f"*{tag}*rank{rank}.json")):
        return p
    return None


def diff_pair(golden: dict, reshard: dict) -> list[dict]:
    g = _flatten(golden)
    r = _flatten(reshard)
    keys = sorted(set(g) | set(r))
    diffs = []
    for k in keys:
        gv, rv = g.get(k, "<missing>"), r.get(k, "<missing>")
        if gv != rv:
            diffs.append({"field": k, "golden": gv, "reshard": rv})
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden-dir", required=True)
    ap.add_argument("--reshard-dir", required=True)
    ap.add_argument("--golden-tag", default="event_loop_entry")
    ap.add_argument("--reshard-tag", default="post_reshard")
    ap.add_argument("--golden-scenario", default="golden_tp2")
    ap.add_argument("--reshard-scenario", default="inplace_tp2")
    ap.add_argument("--ranks", default="0,1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    golden_dir = Path(args.golden_dir)
    reshard_dir = Path(args.reshard_dir)
    ranks = [int(x) for x in args.ranks.split(",") if x.strip()]

    report: dict[str, Any] = {
        "golden_dir": str(golden_dir),
        "reshard_dir": str(reshard_dir),
        "golden_tag": args.golden_tag,
        "reshard_tag": args.reshard_tag,
        "ranks": {},
        "summary": {"total_diffs": 0, "critical_diffs": []},
    }

    critical_prefixes = (
        "max_total_num_tokens",
        "max_running_requests",
        "max_req_input_len",
        "max_prefill_tokens",
        "req_to_token_pool_size",
        "kv_allocator_available",
        "kv_allocator_size",
        "model_runner.",
        "enable_overlap",
        "page_size",
        "chunked_prefill_size",
    )

    for rank in ranks:
        gpath = _find_file(golden_dir, args.golden_scenario, args.golden_tag, rank)
        rpath = _find_file(reshard_dir, args.reshard_scenario, args.reshard_tag, rank)
        entry = {"golden_file": str(gpath) if gpath else None, "reshard_file": str(rpath) if rpath else None}
        if gpath is None or rpath is None:
            entry["error"] = "missing state file"
            report["ranks"][str(rank)] = entry
            continue
        golden = _load_rank(gpath)
        reshard = _load_rank(rpath)
        diffs = diff_pair(golden, reshard)
        entry["diff_count"] = len(diffs)
        entry["diffs"] = diffs
        report["ranks"][str(rank)] = entry
        report["summary"]["total_diffs"] += len(diffs)
        for d in diffs:
            if any(d["field"] == p or d["field"].startswith(p) for p in critical_prefixes):
                report["summary"]["critical_diffs"].append({"rank": rank, **d})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print("=== Golden vs In-place Reshard State Diff ===")
    for rank in ranks:
        e = report["ranks"].get(str(rank), {})
        if "error" in e:
            print(f"rank{rank}: ERROR {e['error']}")
            continue
        print(f"rank{rank}: {e['diff_count']} diffs")
        for d in e.get("diffs", [])[:30]:
            print(f"  {d['field']}: golden={d['golden']!r} reshard={d['reshard']!r}")
        if e["diff_count"] > 30:
            print(f"  ... and {e['diff_count'] - 30} more")
    print(f"\nWrote {out}")
    print(f"Critical diffs: {len(report['summary']['critical_diffs'])}")


if __name__ == "__main__":
    main()
