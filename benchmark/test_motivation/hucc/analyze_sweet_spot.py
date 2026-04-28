#!/usr/bin/env python3
"""
Sweet-spot analysis: AFlex vs OptB, pure saving > 5% definition (no A_ratio filter).

Outputs:
  Section D  — sweet-spot stats by TP (count, ratio, mean, max)
  Section B  — (tp, bs) breakdown at SLO×1.0
  Section C  — (tp, il) breakdown at SLO×1.0, bs=16
  Freq pairs — top-5 optimal (f_A, f_F) pairs inside sweet-spot
  Summary    — updated conclusion table
"""

from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter

BASE_DIR = Path(__file__).parent
DECODE_FILE  = BASE_DIR / "decode_data.txt"
PREFILL_FILE = BASE_DIR / "prefill_data.txt"

FREQS    = [210, 450, 690, 930, 1170, 1410]
MAX_FREQ = 1410
SLO_MULTS = [1.0, 1.1, 1.5, 2.0, 5.0]
SWEET_THRESH = 5.0   # % saving vs OptB to count as sweet-spot


# ─────────────────────────────────────────────────────────────────────────────
# Solvers (same as analyze_slo_motivation.py)
# ─────────────────────────────────────────────────────────────────────────────

def optb_energy(df_cfg, slo_budget, lat_col_a, lat_col_f, eng_col_a, eng_col_f):
    """OptB: min-energy unified freq under SLO."""
    best = None
    for f in FREQS:
        row = df_cfg[df_cfg.gpu_clock == f]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        lat = float(r[lat_col_a]) + float(r[lat_col_f])
        eng = float(r[eng_col_a]) + float(r[eng_col_f])
        if lat <= slo_budget:
            if best is None or eng < best[0]:
                best = (eng, f)
    return best  # (energy, freq) or None


def aflex_energy(df_cfg, slo_budget, lat_col_a, lat_col_f, eng_col_a, eng_col_f):
    """AFlex: min-energy (f_A, f_F) pair under SLO."""
    best = None
    for fa in FREQS:
        ra = df_cfg[df_cfg.gpu_clock == fa]
        if len(ra) == 0:
            continue
        ra = ra.iloc[0]
        for ff in FREQS:
            rf = df_cfg[df_cfg.gpu_clock == ff]
            if len(rf) == 0:
                continue
            rf = rf.iloc[0]
            lat = float(ra[lat_col_a]) + float(rf[lat_col_f])
            eng = float(ra[eng_col_a]) + float(rf[eng_col_f])
            if lat <= slo_budget:
                if best is None or eng < best[0]:
                    best = (eng, fa, ff)
    return best  # (energy, f_A, f_F) or None


def saving_pct(optb, aflex):
    if optb is None or aflex is None:
        return None
    return (optb[0] - aflex[0]) / optb[0] * 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Decode analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_decode(df):
    print("\n" + "="*80)
    print("DECODE: AFlex vs OptB  (sweet-spot = saving > 5%, NO A_ratio filter)")
    print("="*80)

    tps  = sorted(df.tp.unique())
    ils  = sorted(df.input_len.unique())
    ols  = sorted(df.output_len.unique())
    bss  = sorted(df.batch_size.unique())

    la, lf, ea, ef = 'D_A_lat', 'D_F_lat', 'D_A_energy', 'D_F_energy'

    # ── Section A: mixed-workload weighted table ──────────────────────────────
    # weight: il distribution {128:30%,512:30%,1024:20%,2048:10%,4096:7%,8192:3%}
    il_weights = {128: 0.30, 512: 0.30, 1024: 0.20, 2048: 0.10, 4096: 0.07, 8192: 0.03}

    print("\n[A] Mixed-workload weighted mean saving (AFlex vs OptB)")
    header = "TP  | " + " | ".join(f"SLO×{s:.1f}" for s in SLO_MULTS)
    print(header)
    print("-" * len(header))

    for tp in tps:
        row_vals = []
        for slo_m in SLO_MULTS:
            weighted_savings = []
            weights_used = []
            for il in ils:
                w = il_weights.get(il, 0.0)
                if w == 0:
                    continue
                for ol in ols:
                    for bs in bss:
                        cfg = df[(df.tp==tp)&(df.input_len==il)&
                                 (df.output_len==ol)&(df.batch_size==bs)]
                        if len(cfg) == 0:
                            continue
                        ref = cfg[cfg.gpu_clock==MAX_FREQ]
                        if len(ref) == 0:
                            continue
                        ref_lat = float(ref.iloc[0][la]) + float(ref.iloc[0][lf])
                        slo = ref_lat * slo_m
                        ob = optb_energy(cfg, slo, la, lf, ea, ef)
                        af = aflex_energy(cfg, slo, la, lf, ea, ef)
                        s = saving_pct(ob, af)
                        if s is not None:
                            weighted_savings.append(s * w)
                            weights_used.append(w)
            if weights_used:
                total_w = sum(weights_used)
                row_vals.append(f"{sum(weighted_savings)/total_w:>5.1f}%")
            else:
                row_vals.append("  N/A ")
        print(f"tp={tp} | " + " | ".join(row_vals))

    # ── Section B: (tp, bs) at SLO×1.0 ──────────────────────────────────────
    print("\n[B] (tp, bs) breakdown, AFlex vs OptB, SLO×1.0")
    slo_m = 1.0
    bs_list = sorted(bss)
    print("TP  | " + " | ".join(f"bs={b:>3}" for b in bs_list))
    print("-" * (6 + 9 * len(bs_list)))
    for tp in tps:
        row_vals = []
        for bs in bs_list:
            savings = []
            for il in ils:
                for ol in ols:
                    cfg = df[(df.tp==tp)&(df.input_len==il)&
                             (df.output_len==ol)&(df.batch_size==bs)]
                    if len(cfg) == 0:
                        continue
                    ref = cfg[cfg.gpu_clock==MAX_FREQ]
                    if len(ref) == 0:
                        continue
                    ref_lat = float(ref.iloc[0][la]) + float(ref.iloc[0][lf])
                    slo = ref_lat * slo_m
                    ob = optb_energy(cfg, slo, la, lf, ea, ef)
                    af = aflex_energy(cfg, slo, la, lf, ea, ef)
                    s = saving_pct(ob, af)
                    if s is not None:
                        savings.append(s)
            if savings:
                m = np.mean(savings)
                sym = "★" if m >= 10 else ("▲" if m >= 5 else " ")
                row_vals.append(f"{sym}{m:>4.1f}%")
            else:
                row_vals.append("  N/A ")
        print(f"tp={tp} | " + " | ".join(row_vals))
    print("★≥10%  ▲5~10%  space<5%")

    # ── Section C: (tp, il) at SLO×1.0, bs=16 ───────────────────────────────
    print("\n[C] (tp, il) breakdown, AFlex vs OptB, SLO×1.0, bs=16")
    bs_fixed = 16
    il_list = sorted(ils)
    print("TP  | " + " | ".join(f"il={i:>4}" for i in il_list))
    print("-" * (6 + 10 * len(il_list)))
    for tp in tps:
        row_vals = []
        for il in il_list:
            savings = []
            for ol in ols:
                cfg = df[(df.tp==tp)&(df.input_len==il)&
                         (df.output_len==ol)&(df.batch_size==bs_fixed)]
                if len(cfg) == 0:
                    continue
                ref = cfg[cfg.gpu_clock==MAX_FREQ]
                if len(ref) == 0:
                    continue
                ref_lat = float(ref.iloc[0][la]) + float(ref.iloc[0][lf])
                slo = ref_lat * slo_m
                ob = optb_energy(cfg, slo, la, lf, ea, ef)
                af = aflex_energy(cfg, slo, la, lf, ea, ef)
                s = saving_pct(ob, af)
                if s is not None:
                    savings.append(s)
            if savings:
                m = np.mean(savings)
                sym = "★" if m >= 10 else ("▲" if m >= 5 else " ")
                row_vals.append(f"{sym}{m:>4.1f}%")
            else:
                row_vals.append("  N/A ")
        print(f"tp={tp} | " + " | ".join(row_vals))

    # ── Section D: sweet-spot stats (pure saving > 5%, no A_ratio) ───────────
    print("\n[D] Sweet-spot stats: saving > 5% (NO A_ratio filter), SLO×1.0")
    print(f"{'TP':>4} | {'sweet#':>6} | {'total#':>6} | {'ratio':>6} | {'mean':>6} | {'max':>6}")
    print("-" * 50)
    slo_m = 1.0
    for tp in tps:
        sweet_savings = []
        all_savings   = []
        freq_pairs    = []
        for il in ils:
            for ol in ols:
                for bs in bss:
                    cfg = df[(df.tp==tp)&(df.input_len==il)&
                             (df.output_len==ol)&(df.batch_size==bs)]
                    if len(cfg) == 0:
                        continue
                    ref = cfg[cfg.gpu_clock==MAX_FREQ]
                    if len(ref) == 0:
                        continue
                    ref_lat = float(ref.iloc[0][la]) + float(ref.iloc[0][lf])
                    slo = ref_lat * slo_m
                    ob = optb_energy(cfg, slo, la, lf, ea, ef)
                    af = aflex_energy(cfg, slo, la, lf, ea, ef)
                    s = saving_pct(ob, af)
                    if s is None:
                        continue
                    all_savings.append(s)
                    if s > SWEET_THRESH:
                        sweet_savings.append(s)
                        freq_pairs.append((af[1], af[2]))  # (f_A, f_F)
        n_sweet = len(sweet_savings)
        n_total = len(all_savings)
        ratio   = n_sweet / n_total * 100 if n_total > 0 else 0
        mean_s  = np.mean(sweet_savings) if sweet_savings else 0
        max_s   = np.max(sweet_savings)  if sweet_savings else 0
        print(f"tp={tp} | {n_sweet:>6} | {n_total:>6} | {ratio:>5.0f}% | "
              f"{mean_s:>5.1f}% | {max_s:>5.1f}%")

        # top-5 freq pairs
        if freq_pairs:
            ctr = Counter(freq_pairs).most_common(5)
            total_pairs = len(freq_pairs)
            print(f"       top freq pairs (f_A, f_F):")
            for (fa, ff), cnt in ctr:
                print(f"         ({fa:>4}MHz, {ff:>4}MHz): {cnt:>4} configs "
                      f"({cnt/total_pairs*100:.0f}%)")

    # ── Freq pair distribution (all TP combined) ─────────────────────────────
    print("\n[D2] Top freq pairs inside sweet-spot (all TP, SLO×1.0)")
    all_pairs = []
    for tp in tps:
        for il in ils:
            for ol in ols:
                for bs in bss:
                    cfg = df[(df.tp==tp)&(df.input_len==il)&
                             (df.output_len==ol)&(df.batch_size==bs)]
                    if len(cfg) == 0:
                        continue
                    ref = cfg[cfg.gpu_clock==MAX_FREQ]
                    if len(ref) == 0:
                        continue
                    ref_lat = float(ref.iloc[0][la]) + float(ref.iloc[0][lf])
                    slo = ref_lat * 1.0
                    ob = optb_energy(cfg, slo, la, lf, ea, ef)
                    af = aflex_energy(cfg, slo, la, lf, ea, ef)
                    s = saving_pct(ob, af)
                    if s is not None and s > SWEET_THRESH:
                        all_pairs.append((af[1], af[2]))
    ctr = Counter(all_pairs).most_common(8)
    total = len(all_pairs)
    print(f"{'f_A':>6} | {'f_F':>6} | {'count':>6} | {'pct':>6} | meaning")
    print("-" * 55)
    for (fa, ff), cnt in ctr:
        meaning = "A降频，F保持高频" if ff == MAX_FREQ else ("均降频" if fa == ff else "A/F各自降频")
        print(f"{fa:>6} | {ff:>6} | {cnt:>6} | {cnt/total*100:>5.0f}% | {meaning}")


# ─────────────────────────────────────────────────────────────────────────────
# Prefill analysis (sweet-spot)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_prefill(df):
    print("\n" + "="*80)
    print("PREFILL: AFlex vs OptB  (sweet-spot = saving > 5%, NO A_ratio filter)")
    print("="*80)

    tps = sorted(df.tp.unique())
    ils = sorted(df.input_len.unique())
    bss = sorted(df.batch_size.unique())
    la, lf, ea, ef = 'P_A_lat', 'P_F_lat', 'P_A_energy', 'P_F_energy'

    slo_mults_p = [1.0, 1.05, 1.1, 1.5]
    print(f"\n{'TP':>4} | " + " | ".join(f"SLO×{s:.2f}" for s in slo_mults_p))
    print("-" * 60)
    for tp in tps:
        row_vals = []
        for slo_m in slo_mults_p:
            savings = []
            for il in ils:
                for bs in bss:
                    cfg = df[(df.tp==tp)&(df.input_len==il)&(df.batch_size==bs)]
                    if len(cfg) == 0:
                        continue
                    ref = cfg[cfg.gpu_clock==MAX_FREQ]
                    if len(ref) == 0:
                        continue
                    ref_lat = float(ref.iloc[0][la]) + float(ref.iloc[0][lf])
                    slo = ref_lat * slo_m
                    ob = optb_energy(cfg, slo, la, lf, ea, ef)
                    af = aflex_energy(cfg, slo, la, lf, ea, ef)
                    s = saving_pct(ob, af)
                    if s is not None:
                        savings.append(s)
            if savings:
                row_vals.append(f"{np.mean(savings):>5.1f}%")
            else:
                row_vals.append("  N/A ")
        print(f"tp={tp} | " + " | ".join(row_vals))

    # sweet-spot stats for prefill
    print("\n[D-Prefill] Sweet-spot stats: saving > 5%, SLO×1.0")
    print(f"{'TP':>4} | {'sweet#':>6} | {'total#':>6} | {'ratio':>6} | {'mean':>6} | {'max':>6}")
    print("-" * 50)
    for tp in tps:
        sweet_savings = []
        all_savings   = []
        for il in ils:
            for bs in bss:
                cfg = df[(df.tp==tp)&(df.input_len==il)&(df.batch_size==bs)]
                if len(cfg) == 0:
                    continue
                ref = cfg[cfg.gpu_clock==MAX_FREQ]
                if len(ref) == 0:
                    continue
                ref_lat = float(ref.iloc[0][la]) + float(ref.iloc[0][lf])
                slo = ref_lat * 1.0
                ob = optb_energy(cfg, slo, la, lf, ea, ef)
                af = aflex_energy(cfg, slo, la, lf, ea, ef)
                s = saving_pct(ob, af)
                if s is None:
                    continue
                all_savings.append(s)
                if s > SWEET_THRESH:
                    sweet_savings.append(s)
        n_sweet = len(sweet_savings)
        n_total = len(all_savings)
        ratio   = n_sweet / n_total * 100 if n_total > 0 else 0
        mean_s  = np.mean(sweet_savings) if sweet_savings else 0
        max_s   = np.max(sweet_savings)  if sweet_savings else 0
        print(f"tp={tp} | {n_sweet:>6} | {n_total:>6} | {ratio:>5.0f}% | "
              f"{mean_s:>5.1f}% | {max_s:>5.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    df_decode  = pd.read_csv(DECODE_FILE,  sep='\t')
    df_prefill = pd.read_csv(PREFILL_FILE, sep='\t')
    print(f"Loaded decode:  {len(df_decode)} rows")
    print(f"Loaded prefill: {len(df_prefill)} rows")

    analyze_decode(df_decode)
    analyze_prefill(df_prefill)

    print("\n" + "="*80)
    print("SWEET-SPOT DEFINITION (updated, no A_ratio):")
    print("  AFlex vs OptB saving > 5%  (pure energy saving threshold)")
    print("  No A_ratio condition — captures all configs where AF helps")
    print("="*80)


if __name__ == "__main__":
    main()
