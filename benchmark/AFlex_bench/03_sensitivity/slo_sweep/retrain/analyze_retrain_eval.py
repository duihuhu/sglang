#!/usr/bin/env python3
"""Analyze retrain evaluation results: compare V1 vs V2 model performance.

Reads benchmark results JSON + DVFS decision logs, produces comparison table
and detailed frequency decision analysis.

Usage:
    python analyze_retrain_eval.py --results-dir results --logs-dir logs
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_results(results_dir: str) -> dict:
    """Load all *_results.json from subdirs."""
    data = {}
    for scheme in ["baseline", "tier_v1", "tier_v2"]:
        json_dir = Path(results_dir) / scheme / "json"
        if not json_dir.exists():
            continue
        for f in sorted(json_dir.glob("*_results.json")):
            r = json.loads(f.read_text())
            data.setdefault(scheme, []).append(r)
    return data


def load_dvfs_logs(logs_dir: str, scheme: str) -> list:
    """Load DVFS decision log files for a scheme."""
    log_dir = Path(logs_dir) / scheme
    decisions = []
    for f in sorted(log_dir.glob("**/dvfs_decisions_*.jsonl")):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        decisions.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return decisions


def analyze_dvfs_decisions(decisions: list) -> dict:
    """Analyze frequency decisions."""
    if not decisions:
        return {"count": 0}

    freq_a_counter = Counter()
    freq_f_counter = Counter()
    switches = 0
    lats = []
    energies = []

    for d in decisions:
        fa = d.get("f_a", d.get("freq_a", 0))
        ff = d.get("f_f", d.get("freq_f", 0))
        freq_a_counter[fa] += 1
        freq_f_counter[ff] += 1
        if d.get("switched", False):
            switches += 1
        if "latency_us" in d:
            lats.append(d["latency_us"])
        if "energy_mj" in d:
            energies.append(d["energy_mj"])

    return {
        "count": len(decisions),
        "switches": switches,
        "switch_rate": switches / max(len(decisions), 1) * 100,
        "freq_a_top5": freq_a_counter.most_common(5),
        "freq_f_top5": freq_f_counter.most_common(5),
        "avg_lat_us": np.mean(lats) if lats else 0,
        "avg_energy_mj": np.mean(energies) if energies else 0,
    }


def print_comparison(data: dict, logs_dir: str):
    """Print comparison table."""
    print("\n" + "=" * 100)
    print("  RETRAIN MODEL EVALUATION: V1 vs V2 COMPARISON")
    print("=" * 100)

    labels = {
        "baseline": "PDAF DynM (MaxFreq, no DVFS)",
        "tier_v1": "PDAF DynM + Tier (V1 old model)",
        "tier_v2": "PDAF DynM + Tier (V2 coupled model)",
    }

    # Performance table
    print(f"\n{'Scheme':<40} {'Thpt':>8} {'TTFT':>8} {'TPOT':>8}"
          f" {'Energy':>10} {'E/tok':>8} {'SLO%':>6} {'TokSLO%':>8}")
    print("-" * 100)

    baseline_energy = None
    for scheme in ["baseline", "tier_v1", "tier_v2"]:
        results = data.get(scheme, [])
        if not results:
            print(f"  {labels.get(scheme, scheme):<38} {'N/A':>8}")
            continue
        for r in results:
            thpt = r.get("throughput_tok_s", 0)
            ttft = r.get("ttft_avg_ms", 0)
            tpot = r.get("tpot_avg_ms", 0)
            energy = r.get("total_energy_j", 0)
            e_tok = r.get("energy_per_token_mj", 0)
            slo = r.get("slo_violation_rate", 0)
            tok_slo = r.get("tpot_per_token_viol_rate", 0)
            wl = Path(r.get("workload", "")).stem
            label = f"{labels.get(scheme, scheme)}"

            if scheme == "baseline" and energy > 0:
                baseline_energy = energy

            saving = ""
            if baseline_energy and energy > 0 and scheme != "baseline":
                s = (1 - energy / baseline_energy) * 100
                saving = f" ({s:+.1f}%)"

            print(f"  {label:<38} {thpt:>8.1f} {ttft:>8.0f} {tpot:>8.0f}"
                  f" {energy:>9.0f}J {e_tok:>7.1f} {slo:>6.1f} {tok_slo:>7.2f}%{saving}")

    # DVFS decision analysis
    print("\n" + "=" * 100)
    print("  DVFS DECISION ANALYSIS")
    print("=" * 100)

    for scheme in ["tier_v1", "tier_v2"]:
        decisions = load_dvfs_logs(logs_dir, scheme)
        analysis = analyze_dvfs_decisions(decisions)
        print(f"\n  {labels.get(scheme, scheme)}:")
        print(f"    Total decisions: {analysis['count']}")
        if analysis["count"] == 0:
            print(f"    (No DVFS decision logs found)")
            continue
        print(f"    Frequency switches: {analysis.get('switches', 0)} "
              f"({analysis.get('switch_rate', 0):.1f}%)")
        if analysis.get("freq_a_top5"):
            print(f"    f_A distribution (top 5):")
            for freq, cnt in analysis["freq_a_top5"]:
                pct = cnt / max(analysis["count"], 1) * 100
                print(f"      {freq} MHz: {cnt} ({pct:.1f}%)")
        if analysis.get("freq_f_top5"):
            print(f"    f_F distribution (top 5):")
            for freq, cnt in analysis["freq_f_top5"]:
                pct = cnt / max(analysis["count"], 1) * 100
                print(f"      {freq} MHz: {cnt} ({pct:.1f}%)")
        if analysis.get("avg_lat_us"):
            print(f"    Avg predicted latency: {analysis['avg_lat_us']:.0f} us")
        if analysis.get("avg_energy_mj"):
            print(f"    Avg predicted energy: {analysis['avg_energy_mj']:.1f} mJ")

    print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--logs-dir", default="logs")
    args = parser.parse_args()

    data = load_results(args.results_dir)
    if not data:
        print("No results found. Run the benchmark first.")
        return

    print_comparison(data, args.logs_dir)


if __name__ == "__main__":
    main()
