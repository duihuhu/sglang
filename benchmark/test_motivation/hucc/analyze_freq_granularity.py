#!/usr/bin/env python3
"""Analyze: does finer frequency granularity yield diminishing returns for AF scheduling?

Compare AF grid energy savings under different freq granularity levels:
- 2 freqs: {210, 1410} (coarsest)
- 3 freqs: {210, 690, 1410}
- 4 freqs: {210, 450, 930, 1410}
- 6 freqs: {210, 450, 690, 930, 1170, 1410} (finest, all available)
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "prefill_data.txt"

FREQ_SETS = {
    "2 freqs (210,1410)":       [210, 1410],
    "3 freqs (210,690,1410)":   [210, 690, 1410],
    "4 freqs (210,450,930,1410)": [210, 450, 930, 1410],
    "6 freqs (all)":            [210, 450, 690, 930, 1170, 1410],
}


def unified_best(df, tp, il, bs, freqs):
    """Best unified DVFS: min energy among same-freq configs."""
    best_e = float("inf")
    best_l = None
    for f in freqs:
        row = df[(df.tp == tp) & (df.input_len == il) & (df.gpu_clock == f) & (df.batch_size == bs)]
        if len(row) == 0:
            return None, None
        r = row.iloc[0]
        l = float(r.P_A_lat) + float(r.P_F_lat)
        e = float(r.P_A_energy) + float(r.P_F_energy)
        if e < best_e:
            best_e = e
            best_l = l
    return best_l, best_e


def af_best(df, tp, il, bs, freqs):
    """Best AF grid point: min energy among all (fa, ff) combos."""
    best_e = float("inf")
    best_l = None
    best_fa, best_ff = None, None
    base = (df.tp == tp) & (df.input_len == il) & (df.batch_size == bs)
    for fa in freqs:
        ra = df[base & (df.gpu_clock == fa)]
        if len(ra) == 0:
            continue
        for ff in freqs:
            rf = df[base & (df.gpu_clock == ff)]
            if len(rf) == 0:
                continue
            l = float(ra.iloc[0].P_A_lat) + float(rf.iloc[0].P_F_lat)
            e = float(ra.iloc[0].P_A_energy) + float(rf.iloc[0].P_F_energy)
            if e < best_e:
                best_e = e
                best_l = l
                best_fa, best_ff = fa, ff
    return best_l, best_e, best_fa, best_ff


def main():
    df = pd.read_csv(DATA_FILE, sep="\t")
    df.columns = df.columns.str.strip()

    tps = sorted(df.tp.unique())
    all_il = sorted(df.input_len.unique())
    bss = sorted(df.batch_size.unique())

    # Collect per-granularity savings
    print("=" * 140)
    print(f"{'':>30} | ", end="")
    for name in FREQ_SETS:
        print(f"{name:>28} | ", end="")
    print()
    print("=" * 140)

    # Aggregate stats
    all_savings = {name: [] for name in FREQ_SETS}

    for tp in tps:
        for il in all_il:
            for bs in bss:
                ngpus = int(tp)
                results = {}
                skip = False
                for name, freqs in FREQ_SETS.items():
                    ul, ue = unified_best(df, tp, il, bs, freqs)
                    al, ae, fa, ff = af_best(df, tp, il, bs, freqs)
                    if ul is None or al is None:
                        skip = True
                        break
                    ue_s = ue * ngpus
                    ae_s = ae * ngpus
                    saving = (ue_s - ae_s) / ue_s * 100
                    results[name] = (saving, fa, ff)
                    all_savings[name].append(saving)

                if skip:
                    continue

                # Only print rows where at least one granularity has >1% saving
                if max(r[0] for r in results.values()) > 1.0:
                    label = f"tp={tp} il={il:>5} bs={bs:>2}"
                    print(f"{label:>30} | ", end="")
                    for name in FREQ_SETS:
                        s, fa, ff = results[name]
                        print(f"{s:>5.1f}% (A={fa:>4},F={ff:>4}) | ", end="")
                    print()

    print("=" * 140)
    print()

    # Summary statistics
    print("=== Summary: Average energy saving (%) by frequency granularity ===")
    print(f"{'Granularity':>35} | {'Mean':>6} | {'Median':>6} | {'Max':>6} | {'P90':>6} | {'Count>1%':>8} | {'Count>5%':>8}")
    print("-" * 100)
    for name in FREQ_SETS:
        vals = np.array(all_savings[name])
        if len(vals) == 0:
            continue
        print(f"{name:>35} | {np.mean(vals):>5.2f}% | {np.median(vals):>5.2f}% | {np.max(vals):>5.2f}% | "
              f"{np.percentile(vals, 90):>5.2f}% | {np.sum(vals > 1):>8d} | {np.sum(vals > 5):>8d}")

    print()
    print("=== Marginal gain from adding more frequencies ===")
    names = list(FREQ_SETS.keys())
    for i in range(1, len(names)):
        prev = np.array(all_savings[names[i - 1]])
        curr = np.array(all_savings[names[i]])
        delta = curr - prev
        print(f"  {names[i-1]:>35} -> {names[i]:<28}: "
              f"mean delta = {np.mean(delta):>+.3f}%, max delta = {np.max(delta):>+.3f}%")


if __name__ == "__main__":
    main()
