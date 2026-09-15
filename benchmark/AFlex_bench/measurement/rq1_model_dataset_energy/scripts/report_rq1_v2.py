#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from rq1lib_v2 import ARCH_ORDER, QPS_ORDER, aggregate, atomic_json, canary_gate, load


def main():
    parser = argparse.ArgumentParser(description="Generate isolated RQ1-v2 reports")
    parser.add_argument("--results", type=Path, default=ROOT / "results/v2")
    args = parser.parse_args()
    progress = load(args.results / "progress.json")
    models = load(ROOT / "configs/models.json")["models"]
    workloads = load(ROOT / "configs/workloads_v2.json")
    matrix = load(ROOT / "configs/matrix_v2.json")
    active_qps = {float(value) for value in QPS_ORDER}
    expected_runs = (
        len(models)
        * len(ARCH_ORDER)
        * len(workloads["classes"])
        * len(QPS_ORDER)
        * len(matrix["repeats"])
    )
    expected_groups = (
        len(models)
        * len(ARCH_ORDER)
        * len(workloads["classes"])
        * len(QPS_ORDER)
    )
    groups = aggregate(progress)
    rows = []
    for group in groups:
        row = {key: group[key] for key in ("model", "architecture", "workload", "qps", "repeats")}
        for metric, value in group["metrics"].items():
            row[metric + "_mean"] = value["mean"]
            row[metric + "_sample_std"] = value["sample_std"]
        rows.append(row)
    formal = [
        row
        for row in progress["runs"].values()
        if row["phase"] == "formal_v2" and float(row["qps"]) in active_qps
    ]
    statuses = dict(Counter(row["status"] for row in formal))
    terminal_statuses = {"valid", "loadgen_invalid", "model_failure"}
    gate = canary_gate(
        progress,
        matrix["canary_expected"],
        workload=matrix["canary_workload"],
        qps=matrix["canary_qps"],
    )
    report = {"schema_version": 2, "rq": "RQ1-v2", "canary_gate": gate, "complete": len(formal) == expected_runs and all(row["status"] in terminal_statuses for row in formal), "expected_runs": expected_runs, "observed_runs": len(formal), "status_counts": statuses, "expected_groups": expected_groups, "groups": groups}
    args.results.mkdir(parents=True, exist_ok=True)
    atomic_json(args.results / "rq1_report_v2.json", report)
    if rows:
        fields = sorted(set().union(*(row.keys() for row in rows)))
        with (args.results / "rq1_report_v2.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fields); writer.writeheader(); writer.writerows(rows)
    lines = ["# RQ1-v2 model × dataset × architecture energy report", "", f"Complete: {report['complete']}", f"Runs: {len(formal)}/{expected_runs}", f"Groups with valid repeats: {len(groups)}/{expected_groups}", f"Canary: {report['canary_gate']['valid']}/18", f"Statuses: {json.dumps(statuses, sort_keys=True)}"]
    (args.results / "rq1_report_v2.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"runs": len(formal), "groups": len(groups), "complete": report["complete"], "statuses": statuses, "output": str(args.results)}, indent=2))


if __name__ == "__main__":
    main()
