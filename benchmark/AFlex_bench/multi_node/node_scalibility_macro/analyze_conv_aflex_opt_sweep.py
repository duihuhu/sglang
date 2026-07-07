#!/usr/bin/env python3
"""Analyze AFlex conv optimization sweep results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "results" / "conv_aflex_opt_sweep_20260704_204328.json"
DEFAULT_OUT = HERE / "results" / "conv_aflex_opt_sweep_summary.json"
DEFAULT_CHART = HERE / "charts" / "conv_aflex_opt_sweep_dashboard.png"
QPS = [8, 12, 16]
SERIES = [
    ("distserve_ref", "DistServe"),
    ("biscale_ref", "BiScale"),
    ("aflex_m1_v2_ref", "AFlex M1 V2"),
    ("aflex_m1_v1_ref", "AFlex M1 V1"),
    ("aflex_dynamic_m2_v2", "AFlex dynamicM V2"),
    ("aflex_m2_v2", "AFlex M2 V2"),
    ("aflex_m2_v1", "AFlex M2 V1"),
    ("aflex_2decode_m1_v2", "AFlex 2Decode V2"),
]
METRICS = [
    ("throughput_tok_s", "Throughput (tok/s)", "higher"),
    ("tpot_avg_ms", "TPOT avg (ms)", "lower"),
    ("ttft_proc_avg_ms", "TTFT avg (ms)", "lower"),
    ("energy_per_token_mj", "Energy/token (mJ)", "lower"),
    ("slo_violation_rate", "SLO violation (%)", "lower"),
]


def pick(results: dict, series: str, qps: int, metric: str):
    return results.get(series, {}).get(f"conv_qps{qps}", {}).get(metric)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--chart", default=str(DEFAULT_CHART))
    args = parser.parse_args()

    input_path = Path(args.input)
    data = json.loads(input_path.read_text())
    results = data["results"]

    summary = {"input": str(input_path), "qps": QPS, "series": {}}
    for key, label in SERIES:
        if key not in results:
            continue
        summary["series"][key] = {"label": label, "metrics": {}}
        for qps in QPS:
            item = results[key].get(f"conv_qps{qps}")
            if not item:
                continue
            summary["series"][key]["metrics"][str(qps)] = {
                metric: item.get(metric) for metric, _, _ in METRICS
            }

    # Identify best AFlex variant under low SLO violation.
    aflex_candidates = [
        k for k in summary["series"]
        if k.startswith("aflex_") and "2decode" not in k
    ]
    best = {}
    for qps in QPS:
        viable = []
        for k in aflex_candidates:
            m = summary["series"][k]["metrics"].get(str(qps), {})
            if m.get("slo_violation_rate", 100.0) <= 1.0 and m.get("tpot_avg_ms") is not None:
                viable.append((m["tpot_avg_ms"], -m.get("throughput_tok_s", 0.0), k))
        if viable:
            _, _, k = sorted(viable)[0]
            best[str(qps)] = k
    summary["best_aflex_by_tpot_under_1pct_slo"] = best
    summary["notes"] = [
        "M2 variants regress TPOT/throughput versus M1; dynamicM only recovers near M1 at high QPS.",
        "AFlex M1 V1 remains the best single-decode AFlex option in qps8/12/16 by energy/token and similar TPOT.",
        "2Decode reduced decode TPOT but caused multi-second TTFT and high SLO violation, so it is not valid for dashboard replacement without deeper routing/prefill fixes.",
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    chart_path = Path(args.chart)
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    plot_keys = [k for k, _ in SERIES if k in summary["series"]]
    labels = {k: summary["series"][k]["label"] for k in plot_keys}
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    for ax, (metric, title, _) in zip(axes.flat, METRICS[:4]):
        for k in plot_keys:
            vals = [pick(results, k, qps, metric) for qps in QPS]
            if all(v is None for v in vals):
                continue
            ax.plot(QPS, vals, marker="o", linewidth=2, label=labels[k])
        ax.set_title(title)
        ax.set_xlabel("QPS")
        ax.grid(True, alpha=0.25)
    axes.flat[0].legend(fontsize=8, ncol=2)
    fig.suptitle("Conv AFlex Optimization Sweep (qps8/12/16)", fontsize=15)
    fig.savefig(chart_path, dpi=180)
    print(out_path)
    print(chart_path)


if __name__ == "__main__":
    main()
