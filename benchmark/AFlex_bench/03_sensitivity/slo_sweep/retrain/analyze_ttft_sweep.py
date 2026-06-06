"""Plot TTFT SLO sweep: Prefill DVFS behavior under tightening TTFT constraints."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LOG_BASE = Path(__file__).parent / "logs_ttft_sweep"
RESULTS_BASE = Path(__file__).parent / "results_ttft_sweep"
OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

TTFT_SLOS = [5000, 2000, 1000, 500, 300, 200]


def load_prefill_decisions(ttft_slo):
    log_dir = LOG_BASE / f"ttft_{ttft_slo}" / "pdaf_4g_dyn_tier_var_steady"
    f = log_dir / "pdaf_4g_dyn_tier_var_steady_dvfs_decisions_ffn_prefill_gpu4.jsonl"
    if not f.exists():
        return []
    return [json.loads(l) for l in open(f)]


def load_result(subdir):
    json_dir = RESULTS_BASE / subdir / "json"
    if not json_dir.exists():
        return None
    files = list(json_dir.glob("*.json"))
    if not files:
        return None
    with open(files[0]) as f:
        data = json.load(f)
    for dk, wl in data.items():
        if isinstance(wl, dict):
            for wk, qps in wl.items():
                if isinstance(qps, dict):
                    for qk, r in qps.items():
                        if isinstance(r, dict) and "throughput_tok_s" in r:
                            return r
    return None


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Prefill DVFS: Effect of Tightening TTFT SLO (V2 Model, TPOT=150ms)",
                 fontsize=13, fontweight='bold')

    # Collect metrics
    ttfts = []
    energies = []
    thpts = []
    prefill_energies = []

    baseline = load_result("baseline")
    baseline_energy = baseline["total_energy_j"] if baseline else 0

    for ttft_slo in TTFT_SLOS:
        r = load_result(f"ttft_{ttft_slo}")
        if r:
            ttfts.append(r.get("ttft_avg_ms", 0))
            energies.append(r.get("total_energy_j", 0))
            thpts.append(r.get("throughput_tok_s", 0))
            prefill_energies.append(r.get("prefill_energy_j", 0))
        else:
            ttfts.append(0)
            energies.append(0)
            thpts.append(0)
            prefill_energies.append(0)

    # Panel 1: Prefill frequency distribution per TTFT SLO
    ax = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(TTFT_SLOS)))
    for i, ttft_slo in enumerate(TTFT_SLOS):
        records = load_prefill_decisions(ttft_slo)
        if records:
            freqs = [r["f_f"] for r in records]
            ax.hist(freqs, bins=range(600, 1500, 60), alpha=0.5, color=colors[i],
                    label=f"TTFT={ttft_slo}ms", density=True)
    ax.set_xlabel("Prefill FFN Frequency (MHz)")
    ax.set_ylabel("Density")
    ax.set_title("Prefill Freq Distribution by TTFT SLO")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Energy vs TTFT SLO
    ax = axes[0, 1]
    ax.plot(TTFT_SLOS, energies, 'o-', color='#e74c3c', linewidth=2, markersize=8)
    if baseline_energy:
        ax.axhline(baseline_energy, color='gray', linestyle='--', alpha=0.7,
                   label=f'Baseline ({baseline_energy:.0f}J)')
    ax.set_xlabel("TTFT SLO (ms)")
    ax.set_ylabel("Total Energy (J)")
    ax.set_title("Energy vs TTFT SLO")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    # Panel 3: TTFT vs TTFT SLO
    ax = axes[1, 0]
    ax.plot(TTFT_SLOS, ttfts, 'o-', color='#3498db', linewidth=2, markersize=8,
            label='Actual TTFT')
    ax.plot(TTFT_SLOS, TTFT_SLOS, 'k--', alpha=0.5, label='SLO boundary')
    if baseline:
        ax.axhline(baseline["ttft_avg_ms"], color='green', linestyle=':',
                   alpha=0.7, label=f'Baseline ({baseline["ttft_avg_ms"]:.0f}ms)')
    ax.set_xlabel("TTFT SLO (ms)")
    ax.set_ylabel("Actual TTFT (ms)")
    ax.set_title("Actual TTFT vs SLO Target")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    # Panel 4: Summary text / Energy saving
    ax = axes[1, 1]
    savings = [(1 - e / baseline_energy) * 100 if baseline_energy else 0 for e in energies]
    ax.bar(range(len(TTFT_SLOS)), savings, color='#2ecc71', alpha=0.8)
    ax.set_xticks(range(len(TTFT_SLOS)))
    ax.set_xticklabels([str(s) for s in TTFT_SLOS])
    ax.set_xlabel("TTFT SLO (ms)")
    ax.set_ylabel("Energy Saving vs Baseline (%)")
    ax.set_title("Energy Savings vs Baseline")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUT_DIR / "ttft_slo_sweep.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

    # Print summary table
    print(f"\n{'TTFT SLO':<10} {'TTFT(ms)':<10} {'Thpt':<8} {'Energy(J)':<10} {'Save%':<8} {'PrefillFreq'}")
    print("-" * 65)
    for i, ttft_slo in enumerate(TTFT_SLOS):
        records = load_prefill_decisions(ttft_slo)
        if records:
            from collections import Counter
            fc = Counter(r["f_f"] for r in records).most_common(1)
            top_f = f"{fc[0][0]}MHz({fc[0][1]}/{len(records)})" if fc else "N/A"
        else:
            top_f = "N/A"
        print(f"{ttft_slo:<10} {ttfts[i]:<10.0f} {thpts[i]:<8.0f} {energies[i]:<10.0f} {savings[i]:<8.1f} {top_f}")
    if baseline:
        print(f"{'Baseline':<10} {baseline['ttft_avg_ms']:<10.0f} {baseline['throughput_tok_s']:<8.0f} {baseline_energy:<10.0f} {'0.0':<8} 1410 MHz")


if __name__ == "__main__":
    main()
