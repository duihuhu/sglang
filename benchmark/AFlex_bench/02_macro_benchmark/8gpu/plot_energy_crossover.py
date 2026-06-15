#!/usr/bin/env python3
"""Plot the energy crossover between heterogeneous prefill-heavy PDAF (2PA4PF,
P6+D2) and symmetric PDAF (P4+D4) as input length grows (fixed short output).

Finds: het wins (lower energy) once IL is large enough that prefill dominates;
crossover near IL~4-6k tokens on Qwen3-32B / 8xA800.
"""
import json
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
JSON_DIR = HERE / "results_xover" / "json"
OUT_DIR = HERE / "charts_8gpu_azure"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load(scheme, il, ol, q):
    pat = f"{scheme}_il{il}_ol{ol}_qpsworkload_xover_il{il}_ol{ol}_q{q}_results.json"
    f = JSON_DIR / pat
    if f.exists():
        try:
            return json.load(open(f))
        except Exception:
            return {}
    return {}


# IL sweep at fixed ol=8, q=2
ILS = [2048, 4096, 6144, 8192]
het_e, sym_e, het_pd, sym_pd = [], [], [], []
for il in ILS:
    h = load("pdaf_8g_2pa4pf_tier", il, 8, 2)
    s = load("pdaf_8g_dyn_tier", il, 8, 2)
    het_e.append(h.get("total_energy_j", 0) / 1000)
    sym_e.append(s.get("total_energy_j", 0) / 1000)
    het_pd.append((h.get("prefill_energy_j", 0) / 1000,
                   h.get("decode_energy_j", 0) / 1000))
    sym_pd.append((s.get("prefill_energy_j", 0) / 1000,
                   s.get("decode_energy_j", 0) / 1000))

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Energy Crossover: Heterogeneous (P6+D2) vs Symmetric (P4+D4) PDAF\n"
             "Qwen3-32B / 8xA800, Tier DVFS, fixed output ol=8, QPS=2",
             fontsize=12, fontweight="bold")

ax = axes[0]
ax.plot(ILS, het_e, "o-", color="#C0504D", lw=2, ms=8, label="2PA4PF (Hetero, P6+D2)")
ax.plot(ILS, sym_e, "s-", color="#70AD47", lw=2, ms=8, label="Sym (P4+D4)")
for il, h, s in zip(ILS, het_e, sym_e):
    if h and s:
        d = (h / s - 1) * 100
        ax.annotate(f"{d:+.1f}%", (il, min(h, s)), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=8,
                    color="green" if d < 0 else "red")
ax.set_xlabel("Input Length (tokens)", fontsize=10)
ax.set_ylabel("Total Energy (kJ)", fontsize=10)
ax.set_title("Total energy vs input length", fontsize=11, fontweight="bold")
ax.grid(alpha=0.3)
ax.legend(fontsize=9)

ax = axes[1]
x = np.arange(len(ILS))
w = 0.35
hp = [p for p, d in het_pd]
hd = [d for p, d in het_pd]
sp = [p for p, d in sym_pd]
sd = [d for p, d in sym_pd]
ax.bar(x - w/2, hp, w, color="#C0504D", label="Hetero prefill")
ax.bar(x - w/2, hd, w, bottom=hp, color="#E6A8A6", label="Hetero decode")
ax.bar(x + w/2, sp, w, color="#70AD47", label="Sym prefill")
ax.bar(x + w/2, sd, w, bottom=sp, color="#B5D6A0", label="Sym decode")
ax.set_xticks(x)
ax.set_xticklabels([f"il{il}" for il in ILS])
ax.set_ylabel("Energy (kJ)", fontsize=10)
ax.set_title("Prefill / Decode energy split", fontsize=11, fontweight="bold")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout(rect=[0, 0, 1, 0.92])
save_path = OUT_DIR / "8gpu_energy_crossover_il_sweep.png"
fig.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"Saved: {save_path}")

print("\n=== IL sweep (ol=8, q=2), Tier mode ===")
print(f"{'IL':>6} {'Hetero kJ':>11} {'Sym kJ':>9} {'het vs sym':>11}")
for il, h, s in zip(ILS, het_e, sym_e):
    cmp = f"{(h/s-1)*100:+.1f}%" if (h and s) else "-"
    print(f"{il:>6} {h:>11.1f} {s:>9.1f} {cmp:>11}")
