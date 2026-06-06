"""Plot Prefill DVFS decisions timeline (V2, SLO=150ms as representative)."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = Path(__file__).parent / "logs_slo_sweep" / "slo_150" / "tier_v2" / "pdaf_4g_dyn_tier_var_steady"
OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)


def load_prefill_decisions():
    fa_file = LOG_DIR / "pdaf_4g_dyn_tier_var_steady_dvfs_decisions_attn_prefill_gpu1.jsonl"
    ff_file = LOG_DIR / "pdaf_4g_dyn_tier_var_steady_dvfs_decisions_ffn_prefill_gpu0.jsonl"

    fa_records = [json.loads(l) for l in open(fa_file)]
    ff_records = [json.loads(l) for l in open(ff_file)]
    return fa_records, ff_records


def main():
    fa_records, ff_records = load_prefill_decisions()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Prefill Stage DVFS Decisions (V2, TTFT SLO=5000ms)",
                 fontsize=13, fontweight='bold')

    t0 = fa_records[0]["t"]

    # --- Panel 1: Frequency decisions ---
    ax = axes[0]
    times_a = [(r["t"] - t0) for r in fa_records]
    freqs_a = [r["f_a"] for r in fa_records]
    freqs_f_from_a = [r["f_f"] for r in fa_records]

    ax.step(times_a, freqs_a, where='post', color='#e74c3c', linewidth=1.5,
            label='f_A (Attn GPU)', alpha=0.8)
    ax.step(times_a, freqs_f_from_a, where='post', color='#2ecc71', linewidth=1.5,
            label='f_F (FFN GPU)', alpha=0.8)
    ax.axhline(1410, color='gray', linestyle=':', alpha=0.5, label='Max (1410 MHz)')
    ax.set_ylabel("Frequency (MHz)")
    ax.set_ylim(200, 1600)
    ax.set_title("Prefill Frequency: Always locked at 930 MHz (lowest)")
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Slack (remaining TTFT budget) ---
    ax = axes[1]
    slack_ms = [r["slack_us"] / 1000 for r in fa_records]
    ax.scatter(times_a, slack_ms, c='#3498db', s=15, alpha=0.7, label='TTFT slack')
    ax.axhline(5000, color='red', linestyle='--', alpha=0.5, label='TTFT SLO (5000ms)')
    ax.set_ylabel("Slack (ms)")
    ax.set_title("TTFT Slack: Always > 3000ms (huge margin)")
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Predicted latency and batch size ---
    ax = axes[2]
    pred_lat_ms = [r["pred_lat_us"] / 1000 for r in fa_records]
    bs_values = [r["bs"] for r in fa_records]

    color_lat = '#9b59b6'
    ax.scatter(times_a, pred_lat_ms, c=color_lat, s=15, alpha=0.7, label='Pred latency')
    ax.set_ylabel("Predicted Latency (ms)", color=color_lat)
    ax.set_xlabel("Time (s)")
    ax.set_title("Predicted Prefill Latency & Batch Size")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.scatter(times_a, bs_values, c='#f39c12', s=10, alpha=0.5, marker='x',
                label='Batch size')
    ax2.set_ylabel("Batch Size", color='#f39c12')
    ax2.set_ylim(0, max(bs_values) * 1.2)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

    plt.tight_layout()
    out_path = OUT_DIR / "prefill_dvfs_timeline_v2.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
