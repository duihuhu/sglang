#!/usr/bin/env python3
"""Benchmark average non-overlapping Tier1Solver phases for Code and Conversation."""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, os, platform, random, statistics, sys, tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
AFLEX = ROOT.parents[3]
DERIVE_SOURCE = AFLEX / "multi_node/more_test/macro/scripts/solve_tier1_datasets.py"
DATASETS = (
    ("code", "Code", AFLEX / "multi_node/more_test/macro/data/workloads/macro_code_qps1.jsonl"),
    ("conversation", "Conversation", AFLEX / "multi_node/more_test/macro/data/workloads/macro_conv_qps1.jsonl"),
)
spec = importlib.util.spec_from_file_location("aflex_tier1_overhead_helper", HERE / "benchmark_tier1_overhead.py")
if spec is None or spec.loader is None:
    raise ImportError("cannot load benchmark helper")
helper = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helper
spec.loader.exec_module(helper)
WorkloadProfile, SLOConfig = helper.WorkloadProfile, helper.SLOConfig
FIELDS = ["panel", "dataset", "dataset_label", "source_trace", "G", "samples", "feasible_rate", "qps", "lambda_prefill", "n_active_decode", "il_rep_p", "ol_rep_d", "mean_ol", "bs_avg_d", "decode_token_rate_demand", "total_mean_ms", "decode_enumeration_mean_ms", "prefill_enumeration_mean_ms", "pareto_mean_ms", "global_search_mean_ms", "other_mean_ms"]

class BreakdownSolver(helper.InstrumentedTier1Solver):
    """Exclude search-internal Pareto calls from the outer Pareto timer."""
    def reset_instrumentation(self):
        super().reset_instrumentation()
        self.outer_pareto_ns, self._inside_search = 0, False
    def _pareto_by_gpu_group(self, pairs):
        start = helper.now_ns()
        result = super()._pareto_by_gpu_group(pairs)
        if not self._inside_search:
            self.outer_pareto_ns += helper.now_ns() - start
        return result
    def _search_energy_only(self, *args, **kwargs):
        self._inside_search = True
        try:
            return super()._search_energy_only(*args, **kwargs)
        finally:
            self._inside_search = False

def trace_rows(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty trace: {path}")
    if any("input_len" not in row or "output_len" not in row for row in rows):
        raise ValueError(f"invalid workload row: {path}")
    return rows

def trace_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _pct(values, p):
    ordered = sorted(values)
    index = int(len(ordered) * p / 100)
    return ordered[min(index, len(ordered) - 1)]

def derive_workload(rows, qps=1, calibration=None):
    """Exact steady-state recipe from solve_tier1_datasets.py::_derive_workload."""
    ils = [row["input_len"] for row in rows]
    ols = [row["output_len"] for row in rows]
    mean_il, mean_ol = int(statistics.mean(ils)), int(statistics.mean(ols))
    p50_ol = max(1, _pct(ols, 50))
    if calibration:
        prefill_s = calibration["conservative_prefill_latency_ms"] / 1000.0
        decode_tps = calibration["conservative_per_request_decode_tps"]
        calibration_source = calibration.get("source_records", [])
    else:
        prefill_s, decode_tps, calibration_source = mean_il / 2000.0, 30.0, []
    window_seconds = prefill_s + mean_ol / decode_tps
    n_active = max(qps, int(qps * window_seconds + 0.999999))
    demand_tps = float(qps * mean_ol)
    workload = WorkloadProfile(
        lambda_prefill=float(qps), n_active_decode=n_active,
        il_rep_p=mean_il, bs_avg_p=8, il_rep_d=mean_il,
        ol_rep_d=p50_ol, bs_avg_d=min(32, max(8, n_active // 2)),
        mean_ol=float(mean_ol), decode_token_rate_demand=demand_tps,
    )
    meta = {
        "lambda": qps, "effective_lambda_prefill": float(qps),
        "mode": "steady_state", "n_active": n_active,
        "il_rep_p": mean_il, "ol_rep_d": p50_ol, "mean_ol": mean_ol,
        "n_requests": len(rows), "calibration_applied": bool(calibration),
        "calibration_source_records": calibration_source,
        "conservative_prefill_latency_ms": round(prefill_s * 1000.0, 4),
        "conservative_per_request_decode_tps": round(decode_tps, 4),
        "decode_token_rate_demand": round(demand_tps, 4),
        "deadline_s": None, "trace_request_count": len(rows),
        "active_policy": "steady-state Little's Law with dataset calibration or profile-only fallback",
        "batch_policy": "bs_avg_d = min(32, max(8, n_active_decode // 2))",
    }
    return workload, meta

def build_datasets():
    derived = {}
    for key, label, path in DATASETS:
        if not path.exists():
            raise FileNotFoundError(path)
        rows = trace_rows(path)
        workload, meta = derive_workload(rows, qps=1, calibration=None)
        derived[key] = {
            "dataset": key, "dataset_label": label,
            "source_trace": str(path.resolve()),
            "source_trace_sha256": trace_sha256(path),
            "trace_rows_read": len(rows), "requested_qps": 1, "trace_qps": 1,
            "derive_policy": "steady-state Little's Law with profile-only fallback",
            "calibration_applied": False,
            "workload": workload, "derive_meta": meta,
        }
    return derived

def design(derived):
    cases = []
    for key in ("code", "conversation"):
        for g in (8, 16, 32):
            cases.append({"panel": key, "G": g, **derived[key]})
    return cases

def summarize(records, cases):
    output = []
    for case in cases:
        group = [r for r in records if r["panel"] == case["panel"] and r["dataset"] == case["dataset"] and r["G"] == case["G"]]
        workload = case["workload"]
        row = {"panel": case["panel"], "dataset": case["dataset"], "dataset_label": case["dataset_label"], "source_trace": case["source_trace"], "G": case["G"], "samples": len(group), "feasible_rate": statistics.fmean(r["feasible"] for r in group), "qps": case["derive_meta"]["lambda"], "lambda_prefill": workload.lambda_prefill, "n_active_decode": workload.n_active_decode, "il_rep_p": workload.il_rep_p, "ol_rep_d": workload.ol_rep_d, "mean_ol": workload.mean_ol, "bs_avg_d": workload.bs_avg_d, "decode_token_rate_demand": workload.decode_token_rate_demand}
        for raw, name in (("total_ns", "total_mean_ms"), ("decode_enumeration_ns", "decode_enumeration_mean_ms"), ("prefill_enumeration_ns", "prefill_enumeration_mean_ms"), ("pareto_ns", "pareto_mean_ms"), ("global_search_ns", "global_search_mean_ms"), ("other_ns", "other_mean_ms")):
            row[name] = statistics.fmean(r[raw] / 1e6 for r in group)
        output.append(row)
    return output

def atomic_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows); temporary = Path(stream.name)
    os.replace(temporary, path)

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmups", type=int, default=3); parser.add_argument("--repeats", type=int, default=30); parser.add_argument("--output-dir", type=Path, default=ROOT / "data"); parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--prefill-profile", type=Path, default=helper.DEFAULT_PROFILE_DIR / "prefill_data_v1.txt"); parser.add_argument("--decode-profile", type=Path, default=helper.DEFAULT_PROFILE_DIR / "decode_data_v1.txt"); parser.add_argument("--model-dir", type=Path, default=helper.DEFAULT_MODEL_DIR)
    return parser.parse_args()

def main():
    args = parse_args()
    if args.warmups < 0 or args.repeats < 1: raise ValueError("warmups must be >= 0 and repeats >= 1")
    random.seed(args.seed)
    for path in (args.prefill_profile, args.decode_profile, args.model_dir):
        if not path.exists(): raise FileNotFoundError(path)
    derived = build_datasets()
    start = helper.now_ns(); profile_table = helper.InstrumentedProfileTable(str(args.prefill_profile), str(args.decode_profile), str(args.model_dir)); init_ns = helper.now_ns() - start
    solver = BreakdownSolver(profile_table); slo = SLOConfig(ttft_ms=5000.0, tpot_ms=200.0); cases = design(derived); records = []
    for case_index, case in enumerate(cases, 1):
        for iteration in range(args.warmups + args.repeats):
            start = helper.now_ns(); solution = solver.solve(case["G"], case["workload"], slo); total = helper.now_ns() - start
            if iteration < args.warmups: continue
            instrument = solver.instrument
            enum_d = int(instrument["enumerate_ns"].get("decode", 0)); enum_p = int(instrument["enumerate_ns"].get("prefill", 0)); pareto = int(solver.outer_pareto_ns); search = int(instrument["search_ns"]); subtotal = enum_d + enum_p + pareto + search
            if subtotal > total: raise RuntimeError(f"stages exceed total: {subtotal} > {total}")
            records.append({"panel": case["panel"], "dataset": case["dataset"], "dataset_label": case["dataset_label"], "source_trace": case["source_trace"], "G": case["G"], "iteration": iteration - args.warmups, "workload": asdict(case["workload"]), "derive_meta": case["derive_meta"], "total_ns": total, "decode_enumeration_ns": enum_d, "prefill_enumeration_ns": enum_p, "pareto_ns": pareto, "global_search_ns": search, "other_ns": total - subtotal, "feasible": bool(solution.feasible)})
        print(f"[{case_index:02d}/{len(cases)}] {case['panel']} {case['dataset']} G={case['G']} done", flush=True)
    metadata = {
        "schema": "aflex.tier1_solver_average_breakdown.v6",
        "statistic": "arithmetic mean across measured runs",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": helper.git_revision(), "python": sys.version,
        "platform": platform.platform(), "repo": str(helper.REPO),
        "cli": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "paths": {"prefill_profile": str(args.prefill_profile.resolve()), "decode_profile": str(args.decode_profile.resolve()), "model_dir": str(args.model_dir.resolve()), "derive_source": str(DERIVE_SOURCE.resolve())},
        "slo": asdict(slo), "profile_table_cold_init_ns": init_ns,
        "calibration": {"applied": False, "reason": "Code and Conversation use profile-only fallback"},
        "active_policy": "steady-state Little's Law from solve_tier1_datasets.py with profile-only fallback",
        "batch_policy": "bs_avg_d = min(32, max(8, n_active_decode // 2))",
        "datasets": {key: {**{k: v for k, v in value.items() if k != "workload"}, "workload": asdict(value["workload"])} for key, value in derived.items()},
        "design": {"panels": "(a) Code and (b) Conversation, each at requested QPS=1 and G=8/16/32", "points": 6, "warmups_per_point": args.warmups, "samples_per_point": args.repeats},
        "stage_definition": {"decode_enumeration": "outer decode enumeration including query_metrics", "prefill_enumeration": "outer prefill enumeration including query_metrics", "pareto": "two outer Pareto calls only", "global_search": "entire search including its internal Pareto/query logic", "other": "max(total - listed stages, 0)"},
    }
    raw = args.output_dir / "solver_avg_breakdown_raw.json"; summary = args.output_dir / "solver_avg_breakdown_summary.csv"
    helper.atomic_json(raw, {"metadata": metadata, "solver_runs": records}); atomic_csv(summary, summarize(records, cases))
    print(f"wrote {raw} ({len(records)} runs)"); print(f"wrote {summary}")

if __name__ == "__main__": main()
