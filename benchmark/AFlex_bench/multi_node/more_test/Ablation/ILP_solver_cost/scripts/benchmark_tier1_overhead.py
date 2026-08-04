#!/usr/bin/env python3
"""Measure Tier-1 discrete solver and synthetic monitoring overhead.

This benchmark instruments production classes only from the experiment side.
It does not modify or monkey-patch the production implementation.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
import types
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

AFLEX_ROOT = Path(__file__).resolve().parents[5]
REPO = Path(__file__).resolve().parents[7]
ENERGY_DIR = REPO / "python/sglang/srt/energy"
DEFAULT_PROFILE_DIR = AFLEX_ROOT / "energy_model/Qwen3-32B/data/v1_layer_profile"
DEFAULT_MODEL_DIR = AFLEX_ROOT / "energy_model/Qwen3-32B/models_v1"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data"
SCHEMA = "aflex.tier1_overhead.v1"
STATS = ("mean", "std", "p50", "p90", "p95", "p99", "min", "max")


def _load_energy_modules() -> dict[str, types.ModuleType]:
    """Load only energy modules, bypassing sglang/__init__.py and GPU imports."""
    for name in ("sglang", "sglang.srt", "sglang.srt.energy"):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = []  # type: ignore[attr-defined]
            sys.modules[name] = mod
    loaded = {}
    for short in ("af_profile_predictor", "profile_table", "workload_monitor", "workload_collector", "tier1_solver"):
        name = f"sglang.srt.energy.{short}"
        spec = importlib.util.spec_from_file_location(name, ENERGY_DIR / f"{short}.py")
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {name}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        loaded[short] = mod
    return loaded


MOD = _load_energy_modules()
ProfileTable = MOD["profile_table"].ProfileTable
Tier1Solver = MOD["tier1_solver"].Tier1Solver
WorkloadProfile = MOD["tier1_solver"].WorkloadProfile
SLOConfig = MOD["tier1_solver"].SLOConfig
WorkloadMetricsCollector = MOD["workload_collector"].WorkloadMetricsCollector
WorkloadMonitor = MOD["workload_monitor"].WorkloadMonitor


def now_ns() -> int:
    return time.perf_counter_ns()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def summarize_ns(values: list[int]) -> dict[str, float]:
    us = [v / 1_000.0 for v in values] or [0.0]
    return {"mean": statistics.fmean(us),
            "std": statistics.stdev(us) if len(us) > 1 else 0.0,
            "p50": percentile(us, .50), "p90": percentile(us, .90),
            "p95": percentile(us, .95), "p99": percentile(us, .99),
            "min": min(us), "max": max(us)}


class InstrumentedProfileTable(ProfileTable):
    def reset_instrumentation(self) -> None:
        self.query_ns = 0
        self.query_calls = 0
        self.query_sources: Counter[str] = Counter()

    def query_metrics(self, *args: Any, **kwargs: Any):
        start = now_ns()
        result = super().query_metrics(*args, **kwargs)
        self.query_ns += now_ns() - start
        self.query_calls += 1
        source = str(getattr(result, "source", "unknown"))
        canonical = {"exact": "exact", "lut": "LUT", "gbdt": "GBDT",
                     "linearreg": "LinearReg"}.get(source.lower(), source)
        self.query_sources[canonical] += 1
        return result


class InstrumentedTier1Solver(Tier1Solver):
    """Per-solve, phase-aware instrumentation around production methods."""
    def reset_instrumentation(self) -> None:
        self.instrument = {"enumerate_ns": Counter(), "enumerate_calls": Counter(),
                           "pairs_before": Counter(), "pairs_after": Counter(),
                           "pareto_ns": Counter(), "pareto_calls": Counter(),
                           "pareto_input": Counter(), "pareto_output": Counter(),
                           "search_ns": 0, "search_calls": 0}
        self._pending_pareto_phases: list[str] = []
        self._op_counts: dict[tuple[str, str], int] = {}
        self.pt.reset_instrumentation()
        self._last_infeasibility = {}

    def _enumerate_op_candidates(self, phase, op, *args, **kwargs):
        result = super()._enumerate_op_candidates(phase, op, *args, **kwargs)
        self._op_counts[(phase, op)] = len(result)
        return result

    def _enumerate_pairs(self, phase, *args, **kwargs):
        start = now_ns()
        result = super()._enumerate_pairs(phase, *args, **kwargs)
        self.instrument["enumerate_ns"][phase] += now_ns() - start
        self.instrument["enumerate_calls"][phase] += 1
        self.instrument["pairs_before"][phase] += (self._op_counts.get((phase, "A"), 0) * self._op_counts.get((phase, "F"), 0))
        self.instrument["pairs_after"][phase] += len(result)
        self._pending_pareto_phases.append(phase)
        return result

    def _pareto_by_gpu_group(self, pairs):
        phase = self._pending_pareto_phases.pop(0) if self._pending_pareto_phases else "decode"
        start = now_ns(); result = super()._pareto_by_gpu_group(pairs)
        self.instrument["pareto_ns"][phase] += now_ns() - start
        self.instrument["pareto_calls"][phase] += 1
        self.instrument["pareto_input"][phase] += len(pairs)
        self.instrument["pareto_output"][phase] += len(result)
        return result

    def _search_energy_only(self, *args, **kwargs):
        self._pending_pareto_phases.insert(0, "decode")
        start = now_ns()
        try:
            return super()._search_energy_only(*args, **kwargs)
        finally:
            self.instrument["search_ns"] += now_ns() - start
            self.instrument["search_calls"] += 1

    def solve(self, *args, **kwargs):
        self.reset_instrumentation()
        return super().solve(*args, **kwargs)


LOADS = {
    "low": dict(lambda_prefill=2.0, bs_avg_p=4, bs_avg_d=8, ol_rep_d=128),
    "medium": dict(lambda_prefill=8.0, bs_avg_p=8, bs_avg_d=32, ol_rep_d=256),
    "high": dict(lambda_prefill=20.0, bs_avg_p=16, bs_avg_d=64, ol_rep_d=512),
}


def make_cases(quick: bool) -> list[dict[str, Any]]:
    predictor_fallback = (16, "medium", 3072, 32)
    if quick:
        specs = [
            (8, "low", 128, 8),
            (16, "medium", 1024, 32),
            (32, "high", 8192, 128),
            predictor_fallback,
        ]
    else:
        base = (16, "medium", 1024, 32)
        specs = ([(g, base[1], base[2], base[3]) for g in (8, 16, 32)] +
                 [(base[0], load, base[2], base[3]) for load in ("low", "medium", "high")] +
                 [(base[0], base[1], il, base[3]) for il in (128, 1024, 4096, 8192)] +
                 [(base[0], base[1], base[2], active) for active in (8, 32, 64, 128)] +
                 [predictor_fallback])
    unique = []
    for spec in specs:
        if spec not in unique:
            unique.append(spec)
    return [{"case_id": f"G{g}_{load}_il{il}_a{active}", "G": g,
             "load": load, "prefill_il": il, "active": active}
            for g, load, il, active in unique]


def workload_for(case: dict[str, Any]) -> Any:
    cfg = LOADS[case["load"]]
    return WorkloadProfile(lambda_prefill=cfg["lambda_prefill"], n_active_decode=case["active"],
                           il_rep_p=case["prefill_il"], bs_avg_p=cfg["bs_avg_p"], il_rep_d=512,
                           ol_rep_d=cfg["ol_rep_d"], bs_avg_d=cfg["bs_avg_d"], mean_ol=float(cfg["ol_rep_d"]))


def run_solver(pt: InstrumentedProfileTable, args: argparse.Namespace) -> list[dict[str, Any]]:
    records = []; slo = SLOConfig(ttft_ms=5000.0, tpot_ms=200.0)
    for case in make_cases(args.quick):
        solver = InstrumentedTier1Solver(pt)
        for iteration in range(args.warmups + args.repeats):
            workload = workload_for(case)
            start = now_ns(); solution = solver.solve(case["G"], workload, slo); solve_ns = now_ns() - start
            if iteration < args.warmups:
                continue
            diag = getattr(solver, "_last_infeasibility", {}) or solution.infeasibility_diagnostics or {}
            rejected = dict(diag.get("rejected_counts", {}))
            records.append({"kind": "solver", **case, "iteration": iteration - args.warmups,
                            "workload": asdict(workload), "feasible": bool(solution.feasible),
                            "solution": solution.to_dict(), "solve_ns": solve_ns,
                            "instrument": {"enumerate_ns": dict(solver.instrument["enumerate_ns"]),
                                "enumerate_calls": dict(solver.instrument["enumerate_calls"]),
                                "pairs_before": dict(solver.instrument["pairs_before"]),
                                "pairs_after": dict(solver.instrument["pairs_after"]),
                                "pareto_ns": dict(solver.instrument["pareto_ns"]),
                                "pareto_calls": dict(solver.instrument["pareto_calls"]),
                                "pareto_input": dict(solver.instrument["pareto_input"]),
                                "pareto_output": dict(solver.instrument["pareto_output"]),
                                "search_ns": solver.instrument["search_ns"], "search_calls": solver.instrument["search_calls"],
                                "query_ns": pt.query_ns, "query_calls": pt.query_calls,
                                "query_sources": dict(pt.query_sources),
                                "checked_candidates": int(diag.get("checked_candidates", 0) or 0),
                                "rejected_counts": rejected, "rejected_total": sum(rejected.values())}})
    return records


def fake_batch(n: int, trigger: bool) -> Any:
    now = time.time(); reqs = []
    for i in range(n):
        ttft_s = 0.8 if trigger else 0.1; tpot_s = 0.08 if trigger else 0.02
        ts = SimpleNamespace(prefill_run_batch_start_time=now - ttft_s, prefill_finished_time=now,
                             api_server_dispatch_time=now - ttft_s, completion_time=now + tpot_s * 8)
        reqs.append(SimpleNamespace(rid=f"r{i}", time_stats=ts, output_ids=list(range(8)), origin_input_ids=list(range(1024))))
    return SimpleNamespace(reqs=reqs, batch_size=lambda: n)


def prepare_clean_batch(n: int, trigger: bool) -> tuple[Any, Any]:
    """Return a collector whose batch has started but has not ended."""
    collector = WorkloadMetricsCollector(
        ttft_slo_ms=500,
        tpot_slo_us=50_000,
        window_s=0,
        enable_nvml_polling=False,
    )
    batch = fake_batch(n, trigger)
    collector.record_batch_start(is_prefill=False)
    return collector, batch


def prepare_completed_batch(n: int, trigger: bool) -> tuple[Any, Any]:
    """Return a collector containing exactly one completed synthetic batch."""
    collector, batch = prepare_clean_batch(n, trigger)
    collector.record_batch_end(
        batch,
        is_decode=True,
        t_iter_us=80_000 if trigger else 20_000,
    )
    return collector, batch


def timed_call(factory: Callable[[], tuple[Callable[[], Any], dict[str, Any]]], repeats: int) -> tuple[list[int], list[dict[str, Any]]]:
    timings, outcomes = [], []
    for _ in range(repeats):
        call, outcome = factory(); start = now_ns(); result = call(); timings.append(now_ns() - start)
        if callable(outcome.get("extract")):
            outcome.update(outcome.pop("extract")(result))
        outcomes.append(outcome)
    return timings, outcomes


def run_components(args: argparse.Namespace) -> list[dict[str, Any]]:
    records = []; counts = (1, 16, 64) if args.quick else (1, 16, 64, 256, 1024); reps = max(args.repeats, 3)
    for n in counts:
        for scenario in ("normal", "trigger"):
            trigger = scenario == "trigger"
            def record_batch_start_factory():
                collector = WorkloadMetricsCollector(
                    window_s=0, enable_nvml_polling=False
                )
                return (lambda: collector.record_batch_start(False)), {}

            def record_batch_end_factory():
                collector, batch = prepare_clean_batch(n, trigger)
                return (
                    lambda: collector.record_batch_end(
                        batch,
                        True,
                        80_000 if trigger else 20_000,
                    )
                ), {}

            def build_window_factory():
                collector, _batch = prepare_completed_batch(n, trigger)
                return collector.build_window, {}

            def reset_window_factory():
                collector, _batch = prepare_completed_batch(n, trigger)
                return collector.reset_window, {}

            factories = {
                "record_batch_start": record_batch_start_factory,
                "record_batch_end": record_batch_end_factory,
                "build_window": build_window_factory,
                "reset_window": reset_window_factory,
            }
            for component, factory in factories.items():
                values, outcomes = timed_call(factory, reps)
                records.append({"kind": "component", "component": component, "request_count": n,
                                "scenario": scenario, "timings_ns": values, "outcomes": outcomes})
            # Isolate monitor semantics from collector timing proxies: the normal
            # window is balanced, while trigger is a repeatable SLO/util anomaly.
            window = MOD["workload_monitor"].MonitoringWindow(
                slo_violation_rate=0.10 if trigger else 0.0,
                a_util=0.90 if trigger else 0.50,
                f_util=0.30 if trigger else 0.50,
                p_util=0.85 if trigger else 0.50,
                d_util=0.25 if trigger else 0.50,
                load_distribution={"il_8192p": 1.0} if trigger else {"il_1024": 1.0},
                active_requests=n,
            )
            def monitor_record_factory():
                mon = WorkloadMonitor(window_s=0)
                # Time the second equal window in both scenarios.  For trigger,
                # this timed record is exactly the hysteresis-confirming window.
                mon.record_window(window)
                return (lambda: mon.record_window(window)), {}
            values, outcomes = timed_call(monitor_record_factory, reps)
            records.append({"kind": "component", "component": "monitor.record_window", "request_count": n,
                            "scenario": scenario, "timings_ns": values, "outcomes": outcomes})
            def should_factory():
                mon = WorkloadMonitor(window_s=0); mon.record_window(window); mon.record_window(window)
                return (lambda: mon.should_replan(), {"extract": lambda result: {"triggered": result[0], "reasons": result[1]}})
            values, outcomes = timed_call(should_factory, reps)
            records.append({"kind": "component", "component": "monitor.should_replan", "request_count": n,
                            "scenario": scenario, "timings_ns": values, "outcomes": outcomes})
    return records


def solver_summary_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []; grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records: grouped[record["case_id"]].append(record)
    phases = {"solve_total": lambda r: r["solve_ns"],
              "enumerate_decode": lambda r: r["instrument"]["enumerate_ns"].get("decode", 0),
              "enumerate_prefill": lambda r: r["instrument"]["enumerate_ns"].get("prefill", 0),
              "pareto_decode": lambda r: r["instrument"]["pareto_ns"].get("decode", 0),
              "pareto_prefill": lambda r: r["instrument"]["pareto_ns"].get("prefill", 0),
              "search_energy_only": lambda r: r["instrument"]["search_ns"],
              "query_metrics": lambda r: r["instrument"]["query_ns"]}
    for case_id, group in grouped.items():
        base = group[0]
        for component, getter in phases.items():
            row = {"row_type": "solver", "case_id": case_id, "component": component, "scenario": "",
                   "G": base["G"], "load": base["load"], "prefill_il": base["prefill_il"], "active": base["active"],
                   "request_count": "", "unit": "us", "feasible_rate": statistics.fmean(r["feasible"] for r in group),
                   "query_calls_mean": statistics.fmean(r["instrument"]["query_calls"] for r in group),
                   "pairs_before_mean": statistics.fmean(sum(r["instrument"]["pairs_before"].values()) for r in group),
                   "pairs_after_mean": statistics.fmean(sum(r["instrument"]["pairs_after"].values()) for r in group),
                   "pareto_input_mean": statistics.fmean(sum(r["instrument"]["pareto_input"].values()) for r in group),
                   "pareto_output_mean": statistics.fmean(sum(r["instrument"]["pareto_output"].values()) for r in group),
                   "checked_mean": statistics.fmean(r["instrument"]["checked_candidates"] for r in group),
                   "rejected_mean": statistics.fmean(r["instrument"]["rejected_total"] for r in group)}
            for source in ("exact", "LUT", "GBDT", "LinearReg"):
                row[f"source_{source}_mean"] = statistics.fmean(r["instrument"]["query_sources"].get(source, 0) for r in group)
            row.update(summarize_ns([getter(r) for r in group])); rows.append(row)
    return rows


def component_summary_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for r in records:
        row = {"row_type": "component", "case_id": "", "component": r["component"], "scenario": r["scenario"],
               "G": "", "load": "", "prefill_il": "", "active": "", "request_count": r["request_count"], "unit": "us",
               "feasible_rate": "", "query_calls_mean": "", "pairs_before_mean": "", "pairs_after_mean": "",
               "pareto_input_mean": "", "pareto_output_mean": "", "checked_mean": "", "rejected_mean": "",
               "source_exact_mean": "", "source_LUT_mean": "", "source_GBDT_mean": "", "source_LinearReg_mean": ""}
        row.update(summarize_ns(r["timings_ns"])); rows.append(row)
    return rows


def finite_json(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value): return None
    if isinstance(value, dict): return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [finite_json(v) for v in value]
    return value


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        json.dump(finite_json(payload), f, indent=2, ensure_ascii=False, allow_nan=False); f.write("\n"); temp = Path(f.name)
    os.replace(temp, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["row_type", "case_id", "component", "scenario", "G", "load", "prefill_il", "active", "request_count", "unit",
              *STATS, "feasible_rate", "query_calls_mean", "pairs_before_mean", "pairs_after_mean", "pareto_input_mean",
              "pareto_output_mean", "checked_mean", "rejected_mean", "source_exact_mean", "source_LUT_mean",
              "source_GBDT_mean", "source_LinearReg_mean"]
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows); temp = Path(f.name)
    os.replace(temp, path)


def git_revision() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True, text=True, capture_output=True).stdout.strip()
    except (OSError, subprocess.SubprocessError): return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill-profile", type=Path, default=DEFAULT_PROFILE_DIR / "prefill_data_v1.txt")
    parser.add_argument("--decode-profile", type=Path, default=DEFAULT_PROFILE_DIR / "decode_data_v1.txt")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--warmups", type=int, default=1); parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026); parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick: args.warmups, args.repeats = 0, 1
    if args.warmups < 0 or args.repeats < 1: raise ValueError("warmups must be >= 0 and repeats must be >= 1")
    random.seed(args.seed)
    for path in (args.prefill_profile, args.decode_profile, args.model_dir):
        if not path.exists(): raise FileNotFoundError(path)
    start = now_ns(); pt = InstrumentedProfileTable(str(args.prefill_profile), str(args.decode_profile), str(args.model_dir)); profile_init_ns = now_ns() - start
    pt.reset_instrumentation(); solver_records = run_solver(pt, args); component_records = run_components(args)
    rows = solver_summary_rows(solver_records) + component_summary_rows(component_records)
    rows.append({"row_type": "component", "case_id": "", "component": "ProfileTable.cold_init", "scenario": "cold",
                 "G": "", "load": "", "prefill_il": "", "active": "", "request_count": "", "unit": "us", **summarize_ns([profile_init_ns])})
    model_files = [{"name": p.name, "bytes": p.stat().st_size} for p in sorted(args.model_dir.glob("*.pkl"))]
    payload = {"metadata": {"schema": SCHEMA, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "git_revision": git_revision(), "python": sys.version, "platform": platform.platform(), "repo": str(REPO),
               "paths": {"prefill_profile": str(args.prefill_profile.resolve()), "decode_profile": str(args.decode_profile.resolve()),
                         "model_dir": str(args.model_dir.resolve()), "output_dir": str(args.output_dir.resolve())},
               "cli": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "model_files": model_files,
               "sweep_design": "orthogonal one-factor-at-a-time", "loads": LOADS},
               "profile_table_cold_init_ns": profile_init_ns, "solver_runs": solver_records, "component_runs": component_records}
    atomic_json(args.output_dir / "tier1_overhead_raw.json", payload)
    atomic_csv(args.output_dir / "tier1_overhead_summary.csv", rows)
    print(f"wrote {args.output_dir / 'tier1_overhead_raw.json'}"); print(f"wrote {args.output_dir / 'tier1_overhead_summary.csv'}")


if __name__ == "__main__": main()
