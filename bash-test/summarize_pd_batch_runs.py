#!/usr/bin/env python3
"""
Summarize PD batch nvtx test runs from merged logs + work_dir artifacts.

Reads per-run logs (batch_pd_nvtx_test.py --log-dir) and checks:
  - pipeline markers (bench / rep / background submit)
  - processed/nvtx_PD_combined_stats.csv (ready for pd_latency_big_table merge)
  - common failure signatures in appended server/bench sections

Usage:
  python bash-test/summarize_pd_batch_runs.py
  python bash-test/summarize_pd_batch_runs.py --log-dir bash-test/pd_batch_logs --out-csv bash-test/pd_batch_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


_RUN_ID_RE = re.compile(r"^tp(\d+)_in(\d+)_clk(\d+)$")


@dataclass
class RunSummary:
    run_id: str
    tp: int
    input_len: int
    gpu_clock: int
    log_path: str
    # Parsed from [pd-batch] header lines
    bench_ok: bool = False
    rep_after_shutdown: Optional[bool] = None  # True/False from last "after shutdown wait" line
    submit_bg: bool = False
    # Artifacts
    run_dir: str = ""
    has_nsys_rep: bool = False
    nsys_rep_bytes: int = 0
    has_combined_csv: bool = False
    combined_csv_bytes: int = 0
    # Heuristics from full log text
    has_traceback: bool = False
    has_file_not_found_rep: bool = False
    has_timeout: bool = False
    has_watchdog: bool = False
    has_connection_error: bool = False
    has_stop_profile_failed: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        # Primary: artifacts + pipeline markers from [pd-batch] header only.
        if self.has_file_not_found_rep or (
            self.rep_after_shutdown is False and self.bench_ok
        ):
            return "fail_rep"
        if not self.bench_ok:
            return "fail_bench_or_early"
        if self.submit_bg and self.has_combined_csv:
            return "ok"
        if self.submit_bg and not self.has_combined_csv:
            return "ok_pipeline_missing_csv"
        if self.bench_ok and self.rep_after_shutdown and not self.submit_bg:
            return "partial"
        return "unknown"


def _parse_run_id(name: str) -> Optional[Tuple[str, int, int, int]]:
    base = os.path.splitext(os.path.basename(name))[0]
    m = _RUN_ID_RE.match(base)
    if not m:
        return None
    tp, inl, clk = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return base, tp, inl, clk


def _header_section(text: str) -> str:
    """Only the [pd-batch] header (before appended server/bench dumps)."""
    idx = text.find("\n\n===== server_stdout:")
    if idx == -1:
        idx = text.find("\n\n===== server_stderr:")
    if idx == -1:
        return text
    return text[:idx]


def _scan_log_text(text: str, s: RunSummary) -> None:
    head = _header_section(text)
    if "bench finished successfully" in head:
        s.bench_ok = True
    if "server stopped, submit background processing" in head:
        s.submit_bg = True
    # Last "after shutdown wait" wins (only in header; rep lines are there)
    for line in head.splitlines():
        if "after shutdown wait" in line and "rep_file exists=" in line:
            if "rep_file exists=True" in line:
                s.rep_after_shutdown = True
            elif "rep_file exists=False" in line:
                s.rep_after_shutdown = False

    low = text.lower()
    if "traceback (most recent call last)" in low:
        s.has_traceback = True
    if "nsys rep not found" in text or "filenotfounderror" in low and "nsys" in low:
        s.has_file_not_found_rep = True
    if "timeouterror" in low or "timeout:" in low and "bench" in low:
        s.has_timeout = True
    if "scheduler watchdog timeout" in text:
        s.has_watchdog = True
    if "connectionerror" in low or "remote disconnected" in low or "connection aborted" in low:
        s.has_connection_error = True
    if "stop_profile failed" in text:
        s.has_stop_profile_failed = True

    notes: List[str] = []
    if s.has_watchdog:
        notes.append("scheduler_watchdog")
    if s.has_connection_error:
        notes.append("connection/remote_closed")
    if s.has_stop_profile_failed:
        notes.append("stop_profile_failed")
    if s.has_traceback and "bench_stdout" in text:
        notes.append("traceback_in_appended_logs")
    s.notes = notes


def _artifact_checks(work_dir: str, run_id: str, s: RunSummary) -> None:
    s.run_dir = os.path.join(work_dir, run_id)
    if not os.path.isdir(s.run_dir):
        return
    reps = glob.glob(os.path.join(s.run_dir, "*.nsys-rep"))
    if reps:
        s.has_nsys_rep = True
        s.nsys_rep_bytes = max(os.path.getsize(p) for p in reps)
    combined = os.path.join(s.run_dir, "processed", "nvtx_PD_combined_stats.csv")
    if os.path.isfile(combined):
        s.has_combined_csv = True
        s.combined_csv_bytes = os.path.getsize(combined)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize PD batch nvtx runs from logs + work_dir.")
    parser.add_argument(
        "--log-dir",
        type=str,
        default="bash-test/pd_batch_logs",
        help="Merged per-run logs from batch_pd_nvtx_test.py",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="bash-test/pd_batch_work",
        help="Per-run directories (tp*_in*_clk*) with processed/ and .nsys-rep",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="bash-test/pd_batch_summary.csv",
        help="Summary table CSV for statistics / filtering before big-table merge",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default="",
        help="Optional repo root; log paths in files are resolved relative to cwd if unset.",
    )
    args = parser.parse_args()

    repo = os.path.abspath(args.repo_root) if args.repo_root else os.getcwd()
    log_dir = os.path.abspath(os.path.join(repo, args.log_dir))
    work_dir = os.path.abspath(os.path.join(repo, args.work_dir))
    out_csv = os.path.abspath(os.path.join(repo, args.out_csv))

    if not os.path.isdir(log_dir):
        print(f"[error] log-dir not found: {log_dir}", file=sys.stderr)
        sys.exit(1)

    pattern = os.path.join(log_dir, "tp*_in*_clk*.log")
    log_files = sorted(glob.glob(pattern))
    rows: List[RunSummary] = []

    for path in log_files:
        parsed = _parse_run_id(path)
        if not parsed:
            continue
        run_id, tp, input_len, gpu_clock = parsed
        s = RunSummary(
            run_id=run_id,
            tp=tp,
            input_len=input_len,
            gpu_clock=gpu_clock,
            log_path=path,
        )
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError as e:
            s.notes.append(f"read_error:{e}")
            rows.append(s)
            continue
        _scan_log_text(text, s)
        _artifact_checks(work_dir, run_id, s)
        rows.append(s)

    # Runs present in work_dir but missing merged log (optional)
    if os.path.isdir(work_dir):
        for rd in sorted(glob.glob(os.path.join(work_dir, "tp*_in*_clk*"))):
            if not os.path.isdir(rd):
                continue
            base = os.path.basename(rd)
            if _parse_run_id(base + ".log") is None:
                continue
            if any(r.run_id == base for r in rows):
                continue
            p = _parse_run_id(base + ".log")
            assert p
            _, tp, inl, clk = p
            s = RunSummary(
                run_id=base,
                tp=tp,
                input_len=inl,
                gpu_clock=clk,
                log_path="",
                notes=["no_merged_log_in_log_dir"],
            )
            _artifact_checks(work_dir, base, s)
            rows.append(s)

    rows.sort(key=lambda r: (r.tp, r.input_len, r.gpu_clock, r.run_id))

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    fieldnames = [
        "run_id",
        "tp",
        "input_len",
        "gpu_clock",
        "status",
        "bench_ok",
        "rep_after_shutdown",
        "submit_background",
        "has_nsys_rep",
        "nsys_rep_bytes",
        "has_combined_csv",
        "combined_csv_bytes",
        "has_traceback",
        "log_path",
        "run_dir",
        "notes",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in rows:
            w.writerow(
                {
                    "run_id": s.run_id,
                    "tp": s.tp,
                    "input_len": s.input_len,
                    "gpu_clock": s.gpu_clock,
                    "status": s.status,
                    "bench_ok": s.bench_ok,
                    "rep_after_shutdown": s.rep_after_shutdown
                    if s.rep_after_shutdown is not None
                    else "",
                    "submit_background": s.submit_bg,
                    "has_nsys_rep": s.has_nsys_rep,
                    "nsys_rep_bytes": s.nsys_rep_bytes,
                    "has_combined_csv": s.has_combined_csv,
                    "combined_csv_bytes": s.combined_csv_bytes,
                    "has_traceback": s.has_traceback,
                    "log_path": s.log_path,
                    "run_dir": s.run_dir,
                    "notes": ";".join(s.notes),
                }
            )

    # Console summary
    from collections import Counter

    c = Counter(r.status for r in rows)
    print(f"[saved] {out_csv}")
    print(f"[info] runs: {len(rows)}")
    for k, v in sorted(c.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {k}: {v}")
    ok_n = c.get("ok", 0)
    print(
        f"[info] ready for big-table merge (has combined csv): {ok_n} / {len(rows)}"
    )


if __name__ == "__main__":
    main()
