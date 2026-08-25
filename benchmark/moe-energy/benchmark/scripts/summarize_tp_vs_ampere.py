#!/usr/bin/env python3
"""Summarize TP4 vs ampere_ep benchmark results."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def load_bench_json(path: Path) -> dict:
    lines = [x for x in path.read_text().splitlines() if x.strip()]
    return json.loads(lines[-1])


def parse_nvlink_tx_gib(path: Path) -> float | None:
    if not path.exists():
        return None
    text = path.read_text(errors="replace")
    total = 0.0
    for line in text.splitlines():
        # nvidia-smi nvlink -gt d: "GPU 0: ... Tx: 123456 MB"
        m = re.search(r"Tx:\s*([0-9,]+)\s*([KMGT]?B)", line, re.I)
        if not m:
            continue
        val = float(m.group(1).replace(",", ""))
        unit = m.group(2).upper()
        if unit in ("B",):
            total += val
        elif unit in ("KB",):
            total += val / 1024
        elif unit in ("MB",):
            total += val / 1024 / 1024
        elif unit in ("GB", "GIB"):
            total += val / 1024 / 1024 / 1024
        elif unit in ("TB",):
            total += val / 1024 / 1024 / 1024 / 1024
    return total if total > 0 else None


def main() -> None:
    result_dir = Path(sys.argv[1])
    rows: list[dict] = []

    for backend in ("tp4", "ampere_ep"):
        for path in sorted(result_dir.glob(f"{backend}_c*.json")):
            if "_nvlink" in path.name:
                tag = "nvlink"
            else:
                tag = "bench"
            m = re.search(r"_c(\d+)", path.name)
            if not m:
                continue
            c = int(m.group(1))
            d = load_bench_json(path)
            row = {
                "backend": backend,
                "concurrency": c,
                "tag": tag,
                "completed": d.get("completed"),
                "total_output_tokens": d.get("total_output_tokens"),
                "output_throughput": d.get("output_throughput"),
                "mean_ttft_ms": d.get("mean_ttft_ms"),
                "mean_tpot_ms": d.get("mean_tpot_ms"),
                "mean_e2e_latency_ms": d.get("mean_e2e_latency_ms"),
                "duration_s": d.get("duration"),
                "max_concurrent_requests": d.get("max_concurrent_requests"),
            }
            before = result_dir / f"{backend}_c{c}_nvlink_before.txt"
            after = result_dir / f"{backend}_c{c}_nvlink_after.txt"
            if before.exists() and after.exists():
                b, a = parse_nvlink_tx_gib(before), parse_nvlink_tx_gib(after)
                if b is not None and a is not None:
                    row["nvlink_tx_delta_gib"] = a - b
            rows.append(row)

    sched = {}
    for backend in ("tp4", "ampere_ep"):
        p = result_dir / f"{backend}_scheduler.json"
        if p.exists():
            sched[backend] = json.loads(p.read_text())

    # print tables
    bench_rows = [r for r in rows if r["tag"] == "bench" or r.get("nvlink_tx_delta_gib") is not None]
    # dedupe: prefer nvlink row for same C if exists
    by_key: dict[tuple, dict] = {}
    for r in bench_rows:
        key = (r["backend"], r["concurrency"])
        if key not in by_key or r.get("nvlink_tx_delta_gib") is not None:
            by_key[key] = r
    ordered = sorted(by_key.values(), key=lambda x: (x["concurrency"], x["backend"]))

    print("\n=== TP4 vs Ampere EP4 ===")
    print(f"{'C':>6} {'backend':<10} {'out/s':>9} {'TTFT':>8} {'TPOT':>8} {'E2E':>10} {'dur':>7} {'NVLink ΔTX':>12}")
    print("-" * 80)
    for r in ordered:
        nv = r.get("nvlink_tx_delta_gib")
        nv_s = f"{nv:.2f} GiB" if nv is not None else "-"
        print(
            f"{r['concurrency']:>6} {r['backend']:<10} "
            f"{r.get('output_throughput', 0):>9.1f} "
            f"{r.get('mean_ttft_ms', 0):>8.0f} "
            f"{r.get('mean_tpot_ms', 0):>8.1f} "
            f"{r.get('mean_e2e_latency_ms', 0):>10.0f} "
            f"{r.get('duration_s', 0):>7.1f} {nv_s:>12}"
        )

    print("\n=== Ampere EP vs TP4 (ratio) ===")
    print(f"{'C':>6} {'out_tps':>10} {'TTFT':>10} {'TPOT':>10} {'NVLink TX':>12}")
    print("-" * 54)
    by_c: dict[int, dict] = {}
    for r in ordered:
        by_c.setdefault(r["concurrency"], {})[r["backend"]] = r
    for c in sorted(by_c):
        tp, ep = by_c[c].get("tp4"), by_c[c].get("ampere_ep")
        if not tp or not ep:
            continue

        def ratio(a, b, higher_better=False):
            if a is None or b is None or b == 0:
                return "-"
            r = a / b if higher_better else b / a
            return f"{r:.2f}x"

        nv = "-"
        if tp.get("nvlink_tx_delta_gib") and ep.get("nvlink_tx_delta_gib"):
            nv = ratio(ep["nvlink_tx_delta_gib"], tp["nvlink_tx_delta_gib"], higher_better=False)
        print(
            f"{c:>6} "
            f"{ratio(ep.get('output_throughput'), tp.get('output_throughput'), True):>10} "
            f"{ratio(ep.get('mean_ttft_ms'), tp.get('mean_ttft_ms'), False):>10} "
            f"{ratio(ep.get('mean_tpot_ms'), tp.get('mean_tpot_ms'), False):>10} "
            f"{nv:>12}"
        )

    if sched:
        print("\n=== Scheduler peaks (from server log) ===")
        for backend, s in sched.items():
            print(
                f"{backend}: max_running={s.get('max_running_req')} "
                f"max_queue={s.get('max_queue_req')} "
                f"max_token_usage={s.get('max_token_usage')} "
                f"oom={s.get('oom_count')} traceback={s.get('traceback_count')}"
            )

    summary_path = result_dir / "summary.json"
    summary_path.write_text(
        json.dumps({"rows": ordered, "scheduler": sched}, indent=2)
    )
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
