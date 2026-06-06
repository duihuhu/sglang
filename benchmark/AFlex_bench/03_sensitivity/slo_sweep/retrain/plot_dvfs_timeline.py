"""Plot DVFS frequency timeline for V1 vs V2 (Decode stage only)."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LOG_BASE = Path(__file__).parent / "logs"
OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

def load_decisions(log_dir: Path, component: str, gpu_label: str):
    """Load DVFS decision JSONL for a given component (attn/ffn) in decode."""
    pattern = f"*dvfs_decisions_{component}_decode_gpu*.jsonl"
    files = list(log_dir.glob(pattern))
    if not files:
        return []
    records = []
    for f in sorted(files):
        for line in open(f):
            r = json.loads(line.strip())
            records.append(r)
    records.sort(key=lambda x: x["t"])
    return records


def plot_timeline():
    v1_dir = LOG_BASE / "tier_v1" / "pdaf_4g_dyn_tier_var_steady"
    v2_dir = LOG_BASE / "tier_v2" / "pdaf_4g_dyn_tier_var_steady"

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    fig.suptitle("DVFS Frequency Timeline: V1 vs V2 (Decode Stage, SLO=90ms)",
                 fontsize=14, fontweight='bold')

    configs = [
        ("attn", "Attention (DA)"),
        ("ffn", "FFN (DF)"),
    ]

    for col, (comp, comp_label) in enumerate(configs):
        for row, (version, log_dir, color) in enumerate([
            ("V1 (Old Model)", v1_dir, "#e74c3c"),
            ("V2 (Coupled Model)", v2_dir, "#2ecc71"),
        ]):
            ax = axes[row, col]
            records = load_decisions(log_dir, comp, "")
            if not records:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha='center')
                continue

            t0 = records[0]["t"]
            times = [(r["t"] - t0) for r in records]
            freqs_a = [r["sel_f_a"] for r in records]
            freqs_f = [r["sel_f_f"] for r in records]

            if comp == "attn":
                freqs = freqs_a
                freq_label = "f_A (MHz)"
            else:
                freqs = freqs_f
                freq_label = "f_F (MHz)"

            ax.step(times, freqs, where='post', color=color, linewidth=1.2,
                    alpha=0.85, label=freq_label)

            # Also show the other freq as lighter
            other_freqs = freqs_f if comp == "attn" else freqs_a
            other_label = "f_F" if comp == "attn" else "f_A"
            ax.step(times, other_freqs, where='post', color=color,
                    linewidth=0.7, alpha=0.4, linestyle='--', label=f"{other_label} (MHz)")

            ax.set_ylabel("Frequency (MHz)")
            ax.set_ylim(200, 1600)
            ax.axhline(1410, color='gray', linestyle=':', alpha=0.5, label='Max (1410)')
            ax.legend(loc='upper right', fontsize=8)
            ax.set_title(f"{version} — {comp_label}")
            ax.grid(True, alpha=0.3)

    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 1].set_xlabel("Time (s)")
    plt.tight_layout()
    out_path = OUT_DIR / "dvfs_timeline_v1_v2.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

    # Also plot a combined overlay for easier comparison
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 5))
    fig2.suptitle("DVFS Decode Frequency: V1 vs V2 Overlay (SLO=90ms)",
                  fontsize=13, fontweight='bold')

    for col, (comp, comp_label) in enumerate(configs):
        ax = axes2[col]
        for version, log_dir, color, ls in [
            ("V1", v1_dir, "#e74c3c", "-"),
            ("V2", v2_dir, "#2ecc71", "-"),
        ]:
            records = load_decisions(log_dir, comp, "")
            if not records:
                continue
            t0 = records[0]["t"]
            times = [(r["t"] - t0) for r in records]
            if comp == "attn":
                freqs = [r["sel_f_a"] for r in records]
            else:
                freqs = [r["sel_f_f"] for r in records]
            ax.step(times, freqs, where='post', color=color, linewidth=1.2,
                    linestyle=ls, alpha=0.8, label=f"{version}")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (MHz)")
        ax.set_ylim(200, 1600)
        ax.axhline(1410, color='gray', linestyle=':', alpha=0.5, label='Max')
        ax.set_title(f"Decode {comp_label}")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path2 = OUT_DIR / "dvfs_timeline_overlay.png"
    plt.savefig(out_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path2}")


if __name__ == "__main__":
    plot_timeline()
