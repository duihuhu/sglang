#!/usr/bin/env python3
"""Optimized decode v1 analysis — precompute per-config results, then aggregate."""

from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter

BASE_DIR = Path(__file__).parent
DECODE_V1_FILE = BASE_DIR / "decode_data_v1.txt"

FREQS = [210, 450, 690, 930, 1170, 1410]
MAX_FREQ = 1410
SLO_MULTS = [1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0]
SWEET_THRESH = 5.0
IL_WEIGHTS = {128: 0.30, 512: 0.30, 1024: 0.20, 2048: 0.10, 4096: 0.07, 8192: 0.03}


def load_decode_v1():
    df = pd.read_csv(DECODE_V1_FILE, sep='\t', skiprows=1)
    df = df.rename(columns={
        'A': 'D_A_lat', 'F': 'D_F_lat',
        'A_energy_mj': 'D_A_energy', 'F_energy_mj': 'D_F_energy',
    })
    return df


def compute_all_configs(df):
    """Precompute AFlex/Lex/BiScale results for every (tp,il,ol,bs) config at all SLO mults."""
    results = []
    grouped = df.groupby(['tp', 'input_len', 'output_len', 'batch_size'])
    total = len(grouped)
    for i, ((tp, il, ol, bs), cfg) in enumerate(grouped):
        if i % 200 == 0:
            print(f"  processing config {i}/{total}...", flush=True)
        # Build freq->data lookup
        freq_data = {}
        for _, row in cfg.iterrows():
            f = int(row.gpu_clock)
            freq_data[f] = (float(row.D_A_lat), float(row.D_F_lat),
                            float(row.D_A_energy), float(row.D_F_energy))
        if MAX_FREQ not in freq_data:
            continue
        ref_lat = freq_data[MAX_FREQ][0] + freq_data[MAX_FREQ][1]

        for slo_m in SLO_MULTS:
            slo = ref_lat * slo_m

            # Lex (OptB): min-energy unified freq
            lex_best = None
            for f in FREQS:
                if f not in freq_data:
                    continue
                la, lf, ea, ef = freq_data[f]
                if la + lf <= slo:
                    eng = ea + ef
                    if lex_best is None or eng < lex_best[0]:
                        lex_best = (eng, f)

            # BiScale: lowest-freq-first
            biscale_best = None
            for f in sorted(FREQS):
                if f not in freq_data:
                    continue
                la, lf, ea, ef = freq_data[f]
                if la + lf <= slo:
                    biscale_best = (ea + ef, f)
                    break

            # AFlex: min-energy (f_A, f_F)
            aflex_best = None
            for fa in FREQS:
                if fa not in freq_data:
                    continue
                la_a, _, ea_a, _ = freq_data[fa]
                for ff in FREQS:
                    if ff not in freq_data:
                        continue
                    _, lf_f, _, ef_f = freq_data[ff]
                    if la_a + lf_f <= slo:
                        eng = ea_a + ef_f
                        if aflex_best is None or eng < aflex_best[0]:
                            aflex_best = (eng, fa, ff)

            if lex_best is not None and aflex_best is not None:
                sav_vs_lex = (lex_best[0] - aflex_best[0]) / lex_best[0] * 100.0
            else:
                sav_vs_lex = None

            if biscale_best is not None and aflex_best is not None:
                sav_vs_bi = (biscale_best[0] - aflex_best[0]) / biscale_best[0] * 100.0
            else:
                sav_vs_bi = None

            results.append({
                'tp': tp, 'il': il, 'ol': ol, 'bs': bs, 'slo_m': slo_m,
                'sav_vs_lex': sav_vs_lex, 'sav_vs_bi': sav_vs_bi,
                'aflex_fa': aflex_best[1] if aflex_best else None,
                'aflex_ff': aflex_best[2] if aflex_best else None,
            })

    return pd.DataFrame(results)


def main():
    df = load_decode_v1()
    print(f"Loaded decode_v1: {len(df)} rows")
    print(f"  tp: {sorted(df.tp.unique())}")
    print(f"  input_len: {sorted(df.input_len.unique())}")
    print(f"  output_len: {sorted(df.output_len.unique())}")
    print(f"  batch_size: {sorted(df.batch_size.unique())}")

    print("\nPrecomputing all configs...")
    R = compute_all_configs(df)
    print(f"Total result rows: {len(R)}")

    tps = sorted(R.tp.unique())

    # ═══════════════════════════════════════════════════════════════
    # [1] AFlex vs BiScale (mixed workload weighted) — Section 2.5
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("[1] Decode: AFlex vs BiScale, mixed-workload weighted")
    print("=" * 80)
    slo_show = [1.0, 1.1, 1.5, 2.0, 5.0]
    print(f"{'TP':>4} | " + " | ".join(f"SLO×{s}" for s in slo_show))
    print("-" * 70)
    for tp in tps:
        row_vals = []
        for slo_m in slo_show:
            sub = R[(R.tp == tp) & (R.slo_m == slo_m) & R.sav_vs_bi.notna()].copy()
            sub['w'] = sub.il.map(IL_WEIGHTS).fillna(0)
            sub = sub[sub.w > 0]
            if len(sub) > 0:
                val = (sub.sav_vs_bi * sub.w).sum() / sub.w.sum()
                row_vals.append(f"{val:>5.1f}%")
            else:
                row_vals.append("  N/A ")
        print(f"tp={tp} | " + " | ".join(row_vals))

    # ═══════════════════════════════════════════════════════════════
    # [2] AFlex vs Lex (mixed workload weighted) — Section 2.6
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("[2] Decode: AFlex vs Lex, mixed-workload weighted")
    print("=" * 80)
    print(f"{'TP':>4} | " + " | ".join(f"SLO×{s}" for s in slo_show))
    print("-" * 70)
    for tp in tps:
        row_vals = []
        for slo_m in slo_show:
            sub = R[(R.tp == tp) & (R.slo_m == slo_m) & R.sav_vs_lex.notna()].copy()
            sub['w'] = sub.il.map(IL_WEIGHTS).fillna(0)
            sub = sub[sub.w > 0]
            if len(sub) > 0:
                val = (sub.sav_vs_lex * sub.w).sum() / sub.w.sum()
                row_vals.append(f"{val:>5.1f}%")
            else:
                row_vals.append("  N/A ")
        print(f"tp={tp} | " + " | ".join(row_vals))

    # ═══════════════════════════════════════════════════════════════
    # [3] (tp, bs) breakdown, AFlex vs Lex, SLO×1.0
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("[3] (tp, bs) breakdown, AFlex vs Lex, SLO×1.0")
    print("=" * 80)
    bs_list = sorted(R.bs.unique())
    print("TP  | " + " | ".join(f"bs={b:>3}" for b in bs_list))
    print("-" * (6 + 9 * len(bs_list)))
    for tp in tps:
        row_vals = []
        for bs in bs_list:
            sub = R[(R.tp == tp) & (R.slo_m == 1.0) & (R.bs == bs) & R.sav_vs_lex.notna()]
            if len(sub) > 0:
                m = sub.sav_vs_lex.mean()
                sym = "★" if m >= 10 else ("▲" if m >= 5 else " ")
                row_vals.append(f"{sym}{m:>4.1f}%")
            else:
                row_vals.append("  N/A ")
        print(f"tp={tp} | " + " | ".join(row_vals))

    # ═══════════════════════════════════════════════════════════════
    # [4] Sweet-spot stats
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("[4] Sweet-spot stats: saving > 5%, SLO×1.0")
    print("=" * 80)
    print(f"{'TP':>4} | {'sweet#':>6} | {'total#':>6} | {'ratio':>6} | {'mean':>6} | {'max':>6}")
    print("-" * 55)
    for tp in tps:
        sub = R[(R.tp == tp) & (R.slo_m == 1.0) & R.sav_vs_lex.notna()]
        n_total = len(sub)
        sweet = sub[sub.sav_vs_lex > SWEET_THRESH]
        n_sweet = len(sweet)
        ratio = n_sweet / n_total * 100 if n_total > 0 else 0
        mean_s = sweet.sav_vs_lex.mean() if n_sweet > 0 else 0
        max_s = sweet.sav_vs_lex.max() if n_sweet > 0 else 0
        print(f"tp={tp} | {n_sweet:>6} | {n_total:>6} | {ratio:>5.0f}% | "
              f"{mean_s:>5.1f}% | {max_s:>5.1f}%")
        if n_sweet > 0:
            pairs = list(zip(sweet.aflex_fa.astype(int), sweet.aflex_ff.astype(int)))
            ctr = Counter(pairs).most_common(5)
            for (fa, ff), cnt in ctr:
                print(f"       ({fa:>4}MHz, {ff:>4}MHz): {cnt:>4} configs")

    # ═══════════════════════════════════════════════════════════════
    # [5] SLO decay: AFlex vs Lex — Section 2.10 维度1
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("[5] SLO decay: AFlex vs Lex, mixed-workload weighted")
    print("=" * 80)
    print(f"{'TP':>4} | " + " | ".join(f"SLO×{s}" for s in SLO_MULTS))
    print("-" * 80)
    for tp in tps:
        row_vals = []
        for slo_m in SLO_MULTS:
            sub = R[(R.tp == tp) & (R.slo_m == slo_m) & R.sav_vs_lex.notna()].copy()
            sub['w'] = sub.il.map(IL_WEIGHTS).fillna(0)
            sub = sub[sub.w > 0]
            if len(sub) > 0:
                val = (sub.sav_vs_lex * sub.w).sum() / sub.w.sum()
                row_vals.append(f"{val:>5.1f}%")
            else:
                row_vals.append("  N/A ")
        print(f"tp={tp} | " + " | ".join(row_vals))

    # ═══════════════════════════════════════════════════════════════
    # [6] Workload sensitivity
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("[6] Workload sensitivity: SLO×1.0, AFlex vs Lex")
    print("=" * 80)
    workloads = {
        "论文默认":    {128: 0.30, 512: 0.30, 1024: 0.20, 2048: 0.10, 4096: 0.07},
        "短请求为主":  {128: 0.60, 512: 0.25, 1024: 0.10, 2048: 0.05},
        "中等请求为主": {128: 0.10, 512: 0.20, 1024: 0.40, 2048: 0.20, 4096: 0.10},
        "长请求为主":  {128: 0.05, 512: 0.05, 1024: 0.10, 2048: 0.20, 4096: 0.60},
        "均匀分布":    {128: 0.20, 512: 0.20, 1024: 0.20, 2048: 0.20, 4096: 0.20},
    }
    for wname, wdict in workloads.items():
        tp_vals = []
        for tp in tps:
            sub = R[(R.tp == tp) & (R.slo_m == 1.0) & R.sav_vs_lex.notna()].copy()
            sub['w'] = sub.il.map(wdict).fillna(0)
            sub = sub[sub.w > 0]
            if len(sub) > 0:
                val = (sub.sav_vs_lex * sub.w).sum() / sub.w.sum()
                tp_vals.append(val)
            else:
                tp_vals.append(0)
        mean_all = np.mean(tp_vals)
        print(f"{wname:>14} | " + " | ".join(f"{v:>4.1f}%" for v in tp_vals) + f" | {mean_all:.1f}%")

    # ═══════════════════════════════════════════════════════════════
    # [7] A_ratio table
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("[7] A_ratio: A_lat(1410)/A_lat(210), ol=64")
    print("=" * 80)
    test_ils = [128, 512, 1024, 2048, 4096]
    for bs_test in [1, 4, 16, 64, 256]:
        print(f"\n  bs={bs_test}:")
        print(f"  {'TP':>4} | " + " | ".join(f"il={il:>5}" for il in test_ils))
        print("  " + "-" * 60)
        for tp in tps:
            row_vals = []
            for il in test_ils:
                cfg = df[(df.tp==tp)&(df.input_len==il)&(df.output_len==64)&(df.batch_size==bs_test)]
                a210 = cfg[cfg.gpu_clock == 210]
                a1410 = cfg[cfg.gpu_clock == 1410]
                if len(a210) > 0 and len(a1410) > 0:
                    ratio = float(a1410.iloc[0].D_A_lat) / float(a210.iloc[0].D_A_lat)
                    row_vals.append(f"{ratio:>6.3f}")
                else:
                    row_vals.append("   N/A")
            print(f"  tp={tp} | " + " | ".join(row_vals))


if __name__ == "__main__":
    main()
