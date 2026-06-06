"""Analyze multi-SLO sweep results and generate comparison plots."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results_slo_sweep"
OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

SLOS = [300, 250, 200, 150, 100, 90, 80, 70]
SCHEMES = [
    ("baseline", "Baseline (MaxFreq)", "#3498db"),
    ("tier_v1", "V1 (Old Model)", "#e74c3c"),
    ("tier_v2", "V2 (Coupled Model)", "#2ecc71"),
]


def load_result(slo, scheme):
    json_dir = RESULTS_DIR / f"slo_{slo}" / scheme / "json"
    if not json_dir.exists():
        return None
    files = list(json_dir.glob("*.json"))
    if not files:
        return None
    with open(files[0]) as f:
        data = json.load(f)
    # Navigate nested structure: {deploy: {workload_group: {qps_tag: {...}}}}
    for deploy_key, wl_groups in data.items():
        if isinstance(wl_groups, dict):
            for wl_key, qps_data in wl_groups.items():
                if isinstance(qps_data, dict):
                    for qps_key, result in qps_data.items():
                        if isinstance(result, dict) and "throughput_tok_s" in result:
                            return result
    return None


def main():
    data = {}
    for slo in SLOS:
        data[slo] = {}
        for scheme_id, label, color in SCHEMES:
            r = load_result(slo, scheme_id)
            data[slo][scheme_id] = r

    # Extract metrics
    metrics = {
        "throughput": ("throughput_tok_s", "Throughput (tok/s)"),
        "tpot": ("tpot_avg_ms", "Avg TPOT (ms)"),
        "energy": ("total_energy_j", "Total Energy (J)"),
        "token_slo_viol": ("tpot_per_token_viol_rate", "Per-token SLO Violation (%)"),
        "energy_saving": (None, "Energy Saving vs Baseline (%)"),
    }

    # Build arrays
    results = {}
    for scheme_id, label, color in SCHEMES:
        results[scheme_id] = {
            "slos": [], "thpt": [], "tpot": [], "energy": [],
            "token_viol": [], "saving": []
        }

    for slo in SLOS:
        baseline_energy = None
        if data[slo].get("baseline"):
            baseline_energy = data[slo]["baseline"].get("total_energy_j", 0)

        for scheme_id, label, color in SCHEMES:
            r = data[slo].get(scheme_id)
            if r is None:
                results[scheme_id]["slos"].append(slo)
                results[scheme_id]["thpt"].append(0)
                results[scheme_id]["tpot"].append(0)
                results[scheme_id]["energy"].append(0)
                results[scheme_id]["token_viol"].append(100)
                results[scheme_id]["saving"].append(0)
                continue

            results[scheme_id]["slos"].append(slo)
            results[scheme_id]["thpt"].append(r.get("throughput_tok_s", 0))
            results[scheme_id]["tpot"].append(r.get("tpot_avg_ms", 0))
            results[scheme_id]["energy"].append(r.get("total_energy_j", 0))
            results[scheme_id]["token_viol"].append(r.get("tpot_per_token_viol_rate", 0))

            if baseline_energy and baseline_energy > 0:
                e = r.get("total_energy_j", 0)
                saving = (1 - e / baseline_energy) * 100 if e > 0 else 0
            else:
                saving = 0
            results[scheme_id]["saving"].append(saving)

    # Print summary table
    print("\n" + "=" * 110)
    print(f"  {'SLO(ms)':<8}", end="")
    for _, label, _ in SCHEMES:
        print(f" | {label:^30}", end="")
    print("\n" + " " * 8, end="")
    for _ in SCHEMES:
        print(f" | {'Thpt':>6} {'TPOT':>5} {'TokViol%':>8} {'Save%':>6}", end="")
    print("\n" + "-" * 110)

    for i, slo in enumerate(SLOS):
        print(f"  {slo:<8}", end="")
        for scheme_id, _, _ in SCHEMES:
            thpt = results[scheme_id]["thpt"][i]
            tpot = results[scheme_id]["tpot"][i]
            viol = results[scheme_id]["token_viol"][i]
            save = results[scheme_id]["saving"][i]
            print(f" | {thpt:>6.0f} {tpot:>5.0f} {viol:>8.2f} {save:>6.1f}", end="")
        print()
    print("=" * 110)

    # --- Plot 1: Per-token SLO violation rate vs SLO ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Multi-SLO Comparison: Baseline vs V1 vs V2", fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    for scheme_id, label, color in SCHEMES:
        ax.plot(results[scheme_id]["slos"], results[scheme_id]["token_viol"],
                'o-', color=color, label=label, linewidth=2, markersize=6)
    ax.set_xlabel("TPOT SLO (ms)")
    ax.set_ylabel("Per-token SLO Violation (%)")
    ax.set_title("SLO Violation Rate")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    # --- Plot 2: Energy saving vs SLO ---
    ax = axes[0, 1]
    for scheme_id, label, color in SCHEMES:
        if scheme_id == "baseline":
            continue
        ax.plot(results[scheme_id]["slos"], results[scheme_id]["saving"],
                'o-', color=color, label=label, linewidth=2, markersize=6)
    ax.set_xlabel("TPOT SLO (ms)")
    ax.set_ylabel("Energy Saving vs Baseline (%)")
    ax.set_title("Energy Savings")
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    # --- Plot 3: Throughput vs SLO ---
    ax = axes[1, 0]
    for scheme_id, label, color in SCHEMES:
        ax.plot(results[scheme_id]["slos"], results[scheme_id]["thpt"],
                'o-', color=color, label=label, linewidth=2, markersize=6)
    ax.set_xlabel("TPOT SLO (ms)")
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title("Throughput")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    # --- Plot 4: Avg TPOT vs SLO ---
    ax = axes[1, 1]
    for scheme_id, label, color in SCHEMES:
        ax.plot(results[scheme_id]["slos"], results[scheme_id]["tpot"],
                'o-', color=color, label=label, linewidth=2, markersize=6)
    # Add SLO line
    ax.plot(SLOS, SLOS, 'k--', alpha=0.5, label='SLO boundary')
    ax.set_xlabel("TPOT SLO (ms)")
    ax.set_ylabel("Avg TPOT (ms)")
    ax.set_title("Average TPOT vs SLO Requirement")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()
    out_path = OUT_DIR / "slo_sweep_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
