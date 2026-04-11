#!/usr/bin/env python3
"""Analyze decode: AF best energy saving vs highest freq (1410 MHz) unified.

Two freq sets: all 6 freqs, and no-210.
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "decode_data.txt"

FREQS_ALL = [210, 450, 690, 930, 1170, 1410]
FREQS_NO210 = [450, 690, 930, 1170, 1410]
HIGHEST_FREQ = 1410


def get_unified_highest(df, tp, il, ol, bs):
    row = df[(df.tp == tp) & (df.input_len == il) & (df.output_len == ol)
             & (df.gpu_clock == HIGHEST_FREQ) & (df.batch_size == bs)]
    if len(row) == 0:
        return None, None
    r = row.iloc[0]
    return float(r.D_A_lat) + float(r.D_F_lat), float(r.D_A_energy) + float(r.D_F_energy)


def af_best_energy(df, tp, il, ol, bs, freqs):
    best_e = float("inf")
    best_l, best_fa, best_ff = None, None, None
    base = (df.tp == tp) & (df.input_len == il) & (df.output_len == ol) & (df.batch_size == bs)
    for fa in freqs:
        ra = df[base & (df.gpu_clock == fa)]
        if len(ra) == 0:
            continue
        for ff in freqs:
            rf = df[base & (df.gpu_clock == ff)]
            if len(rf) == 0:
                continue
            l = float(ra.iloc[0].D_A_lat) + float(rf.iloc[0].D_F_lat)
            e = float(ra.iloc[0].D_A_energy) + float(rf.iloc[0].D_F_energy)
            if e < best_e:
                best_e = e
                best_l = l
                best_fa, best_ff = fa, ff
    if best_l is None:
        return None, None, None, None
    return best_l, best_e, best_fa, best_ff


def unified_best_energy(df, tp, il, ol, bs, freqs):
    """Best unified DVFS: min energy among same-freq configs."""
    best_e = float("inf")
    best_l = None
    for f in freqs:
        row = df[(df.tp == tp) & (df.input_len == il) & (df.output_len == ol)
                 & (df.gpu_clock == f) & (df.batch_size == bs)]
        if len(row) == 0:
            return None, None
        r = row.iloc[0]
        l = float(r.D_A_lat) + float(r.D_F_lat)
        e = float(r.D_A_energy) + float(r.D_F_energy)
        if e < best_e:
            best_e = e
            best_l = l
    return best_l, best_e


def analyze(df, freqs, label):
    tps = sorted(df.tp.unique())
    all_il = sorted(df.input_len.unique())
    all_ol = sorted(df.output_len.unique())
    bss = sorted(df.batch_size.unique())

    print(f"\n{'='*150}")
    print(f"  {label}")
    print(f"{'='*150}")

    # --- vs highest freq ---
    e_save_vs_high = []
    lat_chg_vs_high = []
    # --- vs best unified ---
    e_save_vs_best_uni = []
    lat_chg_vs_best_uni = []

    rows = []

    for tp in tps:
        for ol in all_ol:
            for il in all_il:
                for bs in bss:
                    ngpus = int(tp)
                    ul_h, ue_h = get_unified_highest(df, tp, il, ol, bs)
                    ul_b, ue_b = unified_best_energy(df, tp, il, ol, bs, freqs)
                    al, ae, fa, ff = af_best_energy(df, tp, il, ol, bs, freqs)
                    if ul_h is None or al is None or ul_b is None:
                        continue

                    ue_h_s = ue_h * ngpus
                    ue_b_s = ue_b * ngpus
                    ae_s = ae * ngpus

                    sv_h = (ue_h_s - ae_s) / ue_h_s * 100
                    lc_h = (al - ul_h) / ul_h * 100
                    sv_b = (ue_b_s - ae_s) / ue_b_s * 100
                    lc_b = (al - ul_b) / ul_b * 100

                    e_save_vs_high.append(sv_h)
                    lat_chg_vs_high.append(lc_h)
                    e_save_vs_best_uni.append(sv_b)
                    lat_chg_vs_best_uni.append(lc_b)

                    rows.append((tp, il, ol, bs, sv_h, lc_h, sv_b, lc_b, fa, ff))

    e_save_vs_high = np.array(e_save_vs_high)
    lat_chg_vs_high = np.array(lat_chg_vs_high)
    e_save_vs_best_uni = np.array(e_save_vs_best_uni)
    lat_chg_vs_best_uni = np.array(lat_chg_vs_best_uni)

    # --- Print detailed table ---
    print(f"\n{'tp':>3} {'il':>5} {'ol':>5} {'bs':>4} | {'E_save_vs_1410':>14} {'lat_chg_1410':>12} | "
          f"{'E_save_vs_bestU':>15} {'lat_chg_bestU':>13} | {'af_fa':>5} {'af_ff':>5}")
    print("-" * 120)
    for tp, il, ol, bs, sv_h, lc_h, sv_b, lc_b, fa, ff in rows:
        print(f"{tp:>3} {il:>5} {ol:>5} {bs:>4} | {sv_h:>13.1f}% {lc_h:>11.1f}% | "
              f"{sv_b:>14.1f}% {lc_b:>12.1f}% | {fa:>5} {ff:>5}")

    # --- Summary ---
    print(f"\n{'='*80}")
    print(f"  Summary: {label}")
    print(f"{'='*80}")

    print(f"\n  [1] AF best vs 1410 MHz unified:")
    print(f"      Energy saving:  mean={np.mean(e_save_vs_high):.2f}%, median={np.median(e_save_vs_high):.2f}%, "
          f"max={np.max(e_save_vs_high):.2f}%, P90={np.percentile(e_save_vs_high, 90):.2f}%")
    print(f"      Latency change: mean={np.mean(lat_chg_vs_high):+.2f}%, median={np.median(lat_chg_vs_high):+.2f}%")

    print(f"\n  [2] AF best vs best unified DVFS (any freq):")
    print(f"      Energy saving:  mean={np.mean(e_save_vs_best_uni):.2f}%, median={np.median(e_save_vs_best_uni):.2f}%, "
          f"max={np.max(e_save_vs_best_uni):.2f}%, P90={np.percentile(e_save_vs_best_uni, 90):.2f}%")
    print(f"      Latency change: mean={np.mean(lat_chg_vs_best_uni):+.2f}%, median={np.median(lat_chg_vs_best_uni):+.2f}%")

    # Breakdown by tp
    print(f"\n  Breakdown by tp (vs 1410 MHz):")
    for tp in tps:
        mask = [i for i, r in enumerate(rows) if r[0] == tp]
        if not mask:
            continue
        tp_e = e_save_vs_high[mask]
        tp_l = lat_chg_vs_high[mask]
        print(f"    tp={tp}: E_save mean={np.mean(tp_e):.2f}%, median={np.median(tp_e):.2f}%, "
              f"max={np.max(tp_e):.2f}%  |  lat_chg mean={np.mean(tp_l):+.2f}%")

    # Breakdown by tp (vs best unified)
    print(f"\n  Breakdown by tp (vs best unified):")
    for tp in tps:
        mask = [i for i, r in enumerate(rows) if r[0] == tp]
        if not mask:
            continue
        tp_e = e_save_vs_best_uni[mask]
        tp_l = lat_chg_vs_best_uni[mask]
        print(f"    tp={tp}: E_save mean={np.mean(tp_e):.2f}%, median={np.median(tp_e):.2f}%, "
              f"max={np.max(tp_e):.2f}%  |  lat_chg mean={np.mean(tp_l):+.2f}%")

    # Breakdown by output_len (vs 1410)
    print(f"\n  Breakdown by output_len (vs 1410 MHz):")
    for ol in all_ol:
        mask = [i for i, r in enumerate(rows) if r[2] == ol]
        if not mask:
            continue
        ol_e = e_save_vs_high[mask]
        ol_l = lat_chg_vs_high[mask]
        print(f"    ol={ol:>5}: E_save mean={np.mean(ol_e):.2f}%, median={np.median(ol_e):.2f}%, "
              f"max={np.max(ol_e):.2f}%  |  lat_chg mean={np.mean(ol_l):+.2f}%")

    return e_save_vs_high, e_save_vs_best_uni


def main():
    df = pd.read_csv(DATA_FILE, sep="\t")
    df.columns = df.columns.str.strip()

    e1_h, e1_b = analyze(df, FREQS_ALL, "All 6 freqs (210-1410)")
    e2_h, e2_b = analyze(df, FREQS_NO210, "No-210 (450-1410)")

    print(f"\n{'='*80}")
    print("  Comparison: All 6 freqs vs No-210")
    print(f"{'='*80}")
    print(f"  vs 1410 MHz:")
    print(f"    All 6:  mean={np.mean(e1_h):.2f}%, max={np.max(e1_h):.2f}%")
    print(f"    No-210: mean={np.mean(e2_h):.2f}%, max={np.max(e2_h):.2f}%")
    print(f"    Delta:  {np.mean(e1_h) - np.mean(e2_h):+.2f}%")
    print(f"  vs best unified:")
    print(f"    All 6:  mean={np.mean(e1_b):.2f}%, max={np.max(e1_b):.2f}%")
    print(f"    No-210: mean={np.mean(e2_b):.2f}%, max={np.max(e2_b):.2f}%")
    print(f"    Delta:  {np.mean(e1_b) - np.mean(e2_b):+.2f}%")


if __name__ == "__main__":
    main()
