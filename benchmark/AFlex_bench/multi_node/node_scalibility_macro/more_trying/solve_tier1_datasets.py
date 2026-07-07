#!/usr/bin/env python3
"""Run Tier1Solver for macro/micro benchmark datasets.

Derives WorkloadProfile from workload jsonl traces (same recipe as code),
then solves min-energy 1P+kD layout under G=16 and SLO TTFT=5s / TPOT=300ms.

Usage:
  python3 solve_tier1_datasets.py
  python3 solve_tier1_datasets.py --datasets conv,qa_lpld --qps-list 2,8,16
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
ROOT = HERE.parents[2]  # benchmark/AFlex_bench
RESULTS_DIR = HERE / "results"
sys.path.insert(0, str(REPO / "python"))

from sglang.srt.energy.profile_table import ProfileTable
from sglang.srt.energy.tier1_solver import Tier1Solver, WorkloadProfile, SLOConfig

MACRO_WL = HERE.parent / "workloads"
MICRO_WL = HERE.parent.parent / "node_scalibility" / "exp-all" / "workloads"

DATASETS = {
    "code": ("macro", MACRO_WL),
    "conv": ("macro", MACRO_WL),
    "qa_lpld": ("micro", MICRO_WL),
    "chatbot_lphd": ("micro", MICRO_WL),
    "balanced_mpmd": ("micro", MICRO_WL),
    "summary_hphd": ("micro", MICRO_WL),
}

DEFAULT_QPS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16]


def _pct(vals: list[int], p: int) -> int:
    s = sorted(vals)
    i = int(len(s) * p / 100)
    return s[min(i, len(s) - 1)]


def _workload_path(kind: str, base: Path, dataset: str, qps: int) -> Path:
    prefix = "macro" if kind == "macro" else "micro"
    return base / f"{prefix}_{dataset}_qps{qps}.jsonl"


def _derive_workload(rows: list[dict], qps: int) -> tuple[WorkloadProfile, dict]:
    ils = [r["input_len"] for r in rows]
    ols = [r["output_len"] for r in rows]
    mean_il = int(statistics.mean(ils))
    mean_ol = int(statistics.mean(ols))
    p50_ol = max(1, _pct(ols, 50))
    n_active = max(qps, int(qps * (mean_il / 2000 + mean_ol * 0.15) * 1.15))
    wl = WorkloadProfile(
        lambda_prefill=float(qps),
        n_active_decode=n_active,
        il_rep_p=mean_il,
        bs_avg_p=8,
        il_rep_d=mean_il,
        ol_rep_d=p50_ol,
        bs_avg_d=min(32, max(8, n_active // 2)),
    )
    meta = {
        "lambda": qps,
        "n_active": n_active,
        "il_rep_p": mean_il,
        "ol_rep_d": p50_ol,
        "mean_ol": mean_ol,
        "n_requests": len(rows),
    }
    return wl, meta


def _solve_row(solver: Tier1Solver, wl: WorkloadProfile, slo: SLOConfig, qps: int, meta: dict) -> dict:
    sol = solver.solve(G=16, workload=wl, slo=slo)
    feasible = sol.feasible
    if not feasible:
        sol = solver.warm_start(G=16, workload=wl, slo=slo)
    gpu = sol.k_p * (sol.tp_pa + sol.tp_pf) + sol.k_d * (sol.tp_da + sol.tp_df)
    return {
        "qps": qps,
        "feasible_ilp": feasible,
        "workload": meta,
        "solution": {
            "k_p": sol.k_p,
            "k_d": sol.k_d,
            "pa": [sol.tp_pa, sol.f_pa],
            "pf": [sol.tp_pf, sol.f_pf],
            "da": [sol.tp_da, sol.f_da],
            "df": [sol.tp_df, sol.f_df],
            "gpus": gpu,
            "e_mj_layer": sol.total_energy_mj_per_layer,
        },
    }


def solve_dataset(
    solver: Tier1Solver,
    slo: SLOConfig,
    dataset: str,
    qps_list: list[int],
) -> list[dict]:
    kind, base = DATASETS[dataset]
    rows_cache: dict[int, list[dict]] = {}
    out: list[dict] = []
    for qps in qps_list:
        path = _workload_path(kind, base, dataset, qps)
        if not path.exists():
            # fall back to qps16 trace stats if per-qps file missing
            path = _workload_path(kind, base, dataset, 16)
        if not path.exists():
            raise FileNotFoundError(f"no workload for {dataset} qps{qps}: {path}")
        if qps not in rows_cache:
            rows_cache[qps] = [
                json.loads(l) for l in path.read_text().splitlines() if l.strip()
            ]
        wl, meta = _derive_workload(rows_cache[qps], qps)
        out.append(_solve_row(solver, wl, slo, qps, meta))
    return out


def _fmt_pair(tp: int, f: int) -> str:
    return f"TP{tp}@{f}"


def _summary_line(row: dict) -> str:
    s = row["solution"]
    tag = "OK" if row["feasible_ilp"] else "WS"
    return (
        f"QPS{row['qps']:2d} [{tag}] "
        f"{s['k_p']}P+{s['k_d']}D | "
        f"PA {_fmt_pair(*s['pa'])} PF {_fmt_pair(*s['pf'])} | "
        f"DA {_fmt_pair(*s['da'])} DF {_fmt_pair(*s['df'])} | "
        f"GPU={s['gpus']}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--qps-list", default=",".join(str(q) for q in DEFAULT_QPS))
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "tier1_all_datasets_solutions.json")
    args = parser.parse_args()

    qps_list = [int(x.strip()) for x in args.qps_list.split(",") if x.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    pt = ProfileTable(
        prefill_path=str(ROOT / "energy_model/Qwen3-32B/data/v1_layer_profile/prefill_data_v1.txt"),
        decode_path=str(ROOT / "energy_model/Qwen3-32B/data/v1_layer_profile/decode_data_v1.txt"),
        energy_model_dir=str(ROOT / "03_sensitivity/slo_sweep/retrain/models_v1"),
    )
    solver = Tier1Solver(
        pt, num_layers=64, num_kv_heads=8, head_dim=128,
        hidden_size=5120, gpu_mem_gb=80.0,
    )
    slo = SLOConfig(ttft_ms=5000.0, tpot_ms=300.0)

    all_results: dict[str, list[dict]] = {}
    for ds in datasets:
        print(f"\n=== {ds} ===")
        rows = solve_dataset(solver, slo, ds, qps_list)
        all_results[ds] = rows
        for row in rows:
            print(" ", _summary_line(row))
        per_ds = RESULTS_DIR / f"tier1_{ds}_solutions.json"
        per_ds.write_text(json.dumps(rows, indent=2))

    payload = {
        "meta": {
            "datasets": datasets,
            "qps": qps_list,
            "gpu_budget": 16,
            "slo": {"ttft_ms": 5000, "tpot_ms": 300},
            "recipe": "il_rep_p=mean(input_len), ol_rep_d=p50(output_len), n_active=max(qps, qps*(il/2000+ol*0.15)*1.15)",
        },
        "results": all_results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
