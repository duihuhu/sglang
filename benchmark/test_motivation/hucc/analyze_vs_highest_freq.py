#!/usr/bin/env python3
"""Prefill: AF best energy saving vs 1410 MHz unified AND vs best unified DVFS."""

from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "prefill_data.txt"

FREQS_ALL = [210, 450, 690, 930, 1170, 1410]
FREQS_NO210 = [450, 690, 930, 1170, 1410]
HIGHEST_FREQ = 1410


def af_best_energy(df, tp, il, bs, freqs):
    best_e = float("inf")
    best_l, best_fa, best_ff = None, None, None
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
    if best_l is None:
        return None, None, None, None
    return best_l, best_e, best_fa, best_ff


def analyze(df, freqs, label):
    tps = sorted(df.tp.unique())
    all_il = sorted(df.input_len.unique())
    bss = sorted(df.batch_size.unique())

    e_save_vs_high = []
    lat_chg_vs_high = []
    e_save_vs_best = []
    lat_chg_vs_best = []
    rows = []

    for tp in tps:
        for il in all_il:
            for bs in bss:
                ngpus = int(tp)
                # highest freq unified
                rh = df[(df.tp == tp) & (df.input_len == il) & (df.gpu_clock == HIGHEST_FREQ) & (df.batch_size == bs)]
                if len(rh) == 0:
                    continue
                rh = rh.iloc[0]
                ul_h = float(rh.P_A_lat) + float(rh.P_F_lat)
                ue_h = (float(rh.P_A_energy) + float(rh.P_F_energy)) * ngpus

                # best unified
                best_ue, best_ul = float("inf"), None
                for f in freqs:
                    row = df[(df.tp == tp) & (df.input_len == il) & (df.gpu_clock == f) & (df.batch_size == bs)]
                    if len(row) == 0:
                        continue
                    r = row.iloc[0]
                    l_ = float(r.P_A_lat) + float(r.P_F_lat)
                    e_ = (float(r.P_A_energy) + float(r.P_F_energy)) * ngpus
                    if e_ < best_ue:
                        best_ue = e_
                        best_ul = l_

                # AF best
                al, ae, fa, ff = af_best_energy(df, tp, il, bs, freqs)
                if al is None or best_ul is None:
                    continue
                ae_s = ae * ngpus

                sv_h = (ue_h - ae_s) / ue_h * 100
                lc_h = (al - ul_h) / ul_h * 100
                sv_b = (best_ue - ae_s) / best_ue * 100
                lc_b = (al - best_ul) / best_ul * 100

                e_save_vs_high.append(sv_h)
                lat_chg_vs_high.append(lc_h)
                e_save_vs_best.append(sv_b)
                lat_chg_vs_best.append(lc_b)
                rows.append((tp, il, bs, sv_h, lc_h, sv_b, lc_b, fa, ff))

    e_save_vs_high = np.array(e_save_vs_high)
    lat_chg_vs_high = np.array(lat_chg_vs_high)
    e_save_vs_best = np.array(e_save_vs_best)
    lat_chg_vs_best = np.array(lat_chg_vs_best)

    print(f"\n{'='*80}")
    print(f"  Summary: {label}")
    print(f"{'='*80}")

    print(f"\n  [1] AF best vs {HIGHEST_FREQ} MHz unified:")
    print(f"      Energy saving:  mean={np.mean(e_save_vs_high):.2f}%, median={np.median(e_save_vs_high):.2f}%, "
          f"max={np.max(e_save_vs_high):.2f}%, P90={np.percentile(e_save_vs_high, 90):.2f}%")
    print(f"      Latency change: mean={np.mean(lat_chg_vs_high):+.2f}%, median={np.median(lat_chg_vs_high):+.2f}%")

    print(f"\n  [2] AF best vs best unified DVFS (any freq):")
    print(f"      Energy saving:  mean={np.mean(e_save_vs_best):.2f}%, median={np.median(e_save_vs_best):.2f}%, "
          f"max={np.max(e_save_vs_best):.2f}%, P90={np.percentile(e_save_vs_best, 90):.2f}%")
    print(f"      Latency change: mean={np.mean(lat_chg_vs_best):+.2f}%, median={np.median(lat_chg_vs_best):+.2f}%")

    print(f"\n  Breakdown by tp (vs {HIGHEST_FREQ} MHz):")
    for tp in sorted(df.tp.unique()):
        mask = [i for i, r in enumerate(rows) if r[0] == tp]
        if not mask:
            continue
        tp_e = e_save_vs_high[mask]
        tp_l = lat_chg_vs_high[mask]
        print(f"    tp={tp}: E_save mean={np.mean(tp_e):.2f}%, median={np.median(tp_e):.2f}%, "
              f"max={np.max(tp_e):.2f}%  |  lat_chg mean={np.mean(tp_l):+.2f}%")

    print(f"\n  Breakdown by tp (vs best unified):")
    for tp in sorted(df.tp.unique()):
        mask = [i for i, r in enumerate(rows) if r[0] == tp]
        if not mask:
            continue
        tp_e = e_save_vs_best[mask]
        tp_l = lat_chg_vs_best[mask]
        print(f"    tp={tp}: E_save mean={np.mean(tp_e):.2f}%, median={np.median(tp_e):.2f}%, "
              f"max={np.max(tp_e):.2f}%  |  lat_chg mean={np.mean(tp_l):+.2f}%")

    print(f"\n  Breakdown by input_len (vs {HIGHEST_FREQ} MHz):")
    for il in sorted(df.input_len.unique()):
        mask = [i for i, r in enumerate(rows) if r[1] == il]
        if not mask:
            continue
        il_e = e_save_vs_high[mask]
        il_l = lat_chg_vs_high[mask]
        print(f"    il={il:>5}: E_save mean={np.mean(il_e):.2f}%, median={np.median(il_e):.2f}%, "
              f"max={np.max(il_e):.2f}%  |  lat_chg mean={np.mean(il_l):+.2f}%")

    return e_save_vs_high, e_save_vs_best


def main():
    df = pd.read_csv(DATA_FILE, sep="\t")
    df.columns = df.columns.str.strip()

    e1_h, e1_b = analyze(df, FREQS_ALL, "Prefill - All 6 freqs (210-1410)")
    e2_h, e2_b = analyze(df, FREQS_NO210, "Prefill - No-210 (450-1410)")

    print(f"\n{'='*80}")
    print("  Comparison: All 6 freqs vs No-210")
    print(f"{'='*80}")
    print(f"  vs {HIGHEST_FREQ} MHz:")
    print(f"    All 6:  mean={np.mean(e1_h):.2f}%, max={np.max(e1_h):.2f}%")
    print(f"    No-210: mean={np.mean(e2_h):.2f}%, max={np.max(e2_h):.2f}%")
    print(f"    Delta:  {np.mean(e1_h) - np.mean(e2_h):+.2f}%")
    print(f"  vs best unified:")
    print(f"    All 6:  mean={np.mean(e1_b):.2f}%, max={np.max(e1_b):.2f}%")
    print(f"    No-210: mean={np.mean(e2_b):.2f}%, max={np.max(e2_b):.2f}%")
    print(f"    Delta:  {np.mean(e1_b) - np.mean(e2_b):+.2f}%")


if __name__ == "__main__":
    main()
