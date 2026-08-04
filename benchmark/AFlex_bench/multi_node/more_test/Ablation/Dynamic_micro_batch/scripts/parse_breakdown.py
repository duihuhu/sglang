#!/usr/bin/env python3
"""Parse AFD breakdown logs into machine-readable JSON/CSV and Markdown."""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


def payload(line, marker, key):
    pos = line.find(f"[{marker}]")
    if pos < 0:
        return None
    search_start = pos
    while True:
        k = line.find(key + "=", search_start)
        if k < 0:
            return None
        if k == 0 or line[k - 1] == " ":
            break
        search_start = k + 1
    text = line[k + len(key) + 1 :].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def last_json(lines, marker, key):
    vals = [payload(x, marker, key) for x in lines]
    vals = [x for x in vals if x is not None]
    return vals[-1] if vals else None


def parse_merge(lines):
    vals = []
    rx = re.compile(r"\[AFD_MERGE\].*?merge_ms=([\d.]+)")
    for line in lines:
        m = rx.search(line)
        if m:
            vals.append(float(m.group(1)))
    return vals[-1] if vals else None


def parse_overhead(lines):
    rows = []
    rx = re.compile(
        r"\[AFD_FWD_OVERHEAD\].*?total=([\d.]+)ms .*?split_inputs=([\d.]+)ms "
        r".*?pre_pipeline=([\d.]+)ms .*?pipeline=([\d.]+)ms .*?drain=([\d.]+)ms"
    )
    for line in lines:
        m = rx.search(line)
        if m:
            rows.append(
                dict(
                    zip(
                        (
                            "total_ms",
                            "split_ms",
                            "pre_pipeline_ms",
                            "pipeline_ms",
                            "drain_ms",
                        ),
                        map(float, m.groups()),
                    )
                )
            )
    return rows[-1] if rows else None


def intervals(steps):
    out = []
    for s in steps or []:
        starts = [v for k, v in s.items() if k.endswith("_wall_start_ms")]
        ends = [v for k, v in s.items() if k.endswith("_wall_end_ms")]
        if starts and ends:
            out.append((min(starts), max(ends), s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", type=Path, required=True)
    args = ap.parse_args()
    summary = {
        "case": args.case_dir.name,
        "roles": {},
        "definitions": {
            "cuda_compute": (
                "CUDA event elapsed time; DA attn includes prep_attn/attn/prep_mlp, "
                "DF ffn includes mlp/postprocess"
            ),
            "ipc_host": "host perf-counter duration of send/recv calls from AFD_HOST_EVENTS",
            "bubble": (
                "idle gap between consecutive aligned CUDA compute intervals on each role; "
                "includes dependency/IPC wait and launch gaps"
            ),
            "software_overhead": (
                "forward wall minus split, pipeline and drain; pipeline residual is pipeline "
                "wall minus CUDA compute union and is not additive across GPUs"
            ),
        },
    }
    csv_rows = []
    missing = []
    for role in ("da", "df"):
        path = args.case_dir / f"{role}.log"
        lines = path.read_text(errors="replace").splitlines() if path.exists() else []
        steps = last_json(lines, "AFD_PER_STEP", "steps")
        host = last_json(lines, "AFD_HOST_EVENTS", "events")
        timeline = last_json(lines, "AFD_TIMELINE", "timeline")
        overhead = parse_overhead(lines)
        merge_ms = parse_merge(lines)
        for label, val in (
            ("AFD_PER_STEP", steps),
            ("AFD_HOST_EVENTS", host),
            ("AFD_TIMELINE", timeline),
            ("AFD_FWD_OVERHEAD", overhead),
        ):
            if val is None:
                missing.append(f"{role}:{label}")
        if merge_ms is None and args.case_dir.name != "m1-low":
            missing.append(f"{role}:AFD_MERGE")
        iv = sorted(intervals(steps))
        gaps = [max(0, iv[i][0] - iv[i - 1][1]) for i in range(1, len(iv))]
        cuda = {}
        for s in steps or []:
            for k, v in s.items():
                if k.endswith("_ms") and "wall_" not in k:
                    cuda[k] = cuda.get(k, 0.0) + float(v)
            csv_rows.append({"role": role, **s})
        host_durations = {}
        pending = {}
        for e in host or []:
            event = e.get("event", "")
            kind = "send" if "send" in event else "recv" if "recv" in event else event
            if event.endswith("start"):
                pending[(e.get("layer"), e.get("mb"), kind)] = e.get("ts_ms")
            elif event.endswith("end"):
                start = pending.pop((e.get("layer"), e.get("mb"), kind), None)
                if start is not None:
                    host_durations.setdefault(kind, []).append(e["ts_ms"] - start)
        software = None
        if overhead:
            software = (
                overhead["total_ms"]
                - overhead["split_ms"]
                - overhead["pipeline_ms"]
                - overhead["drain_ms"]
            )
        summary["roles"][role] = {
            "cuda_totals_ms": cuda,
            "ipc_host_ms": {
                k: {"sum": sum(v), "mean": statistics.mean(v), "count": len(v)}
                for k, v in host_durations.items()
            },
            "bubble_ms": {
                "sum": sum(gaps),
                "mean": statistics.mean(gaps) if gaps else 0,
                "count": len(gaps),
            },
            "overhead": overhead,
            "software_boundary_ms": software,
            "merge_ms": merge_ms if merge_ms is not None else 0.0,
            "timeline_steps": len(timeline or []),
        }
    summary["complete"] = not missing
    summary["missing_records"] = missing
    (args.case_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fields = sorted({k for r in csv_rows for k in r})
    with (args.case_dir / "per_layer.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(csv_rows)
    md = [
        f"# {args.case_dir.name} breakdown",
        "",
        f"Instrumentation complete: **{summary['complete']}**",
        "",
    ]
    if missing:
        md += ["Missing: " + ", ".join(missing), ""]
    for role, r in summary["roles"].items():
        md += [
            f"## {role.upper()}",
            f"- CUDA totals (ms): `{json.dumps(r['cuda_totals_ms'], sort_keys=True)}`",
            f"- IPC host (ms): `{json.dumps(r['ipc_host_ms'], sort_keys=True)}`",
            f"- Pipeline bubble (ms): `{json.dumps(r['bubble_ms'], sort_keys=True)}`",
            f"- Forward overhead (ms): `{json.dumps(r['overhead'], sort_keys=True)}`",
            f"- Software boundary (ms): `{r['software_boundary_ms']}`",
            f"- Merge (ms): `{r['merge_ms']}` (M=1 is zero; M>1 unavailable in current core instrumentation)",
            "",
        ]
    (args.case_dir / "summary.md").write_text("\n".join(md) + "\n")
    if missing:
        import sys

        print("incomplete instrumentation: " + ", ".join(missing), file=sys.stderr)
    print(args.case_dir / "summary.json")


if __name__ == "__main__":
    main()
