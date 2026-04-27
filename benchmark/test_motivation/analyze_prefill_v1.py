#!/usr/bin/env python3
"""
Rerun all prefill analyses using prefill_data_v1.txt (new data with corrected F measurements).

Outputs:
  1. Prefill AFlex vs Lex saving table (SLO multipliers)
  2. F frequency sensitivity ratio table (F_lat(1410)/F_lat(210))
  3. A frequency sensitivity ratio table (A_ratio)
  4. Sweet-spot stats
  5. F/A latency ratio for ILP配比 analysis
  6. Freq pair distribution
"""

from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter

BASE_DIR = Path(__file__).parent
PREFILL_V1_FILE = BASE_DIR / "paper" / "prefill_data_v1.txt"

FREQS = [210, 450, 690, 930, 1170, 1410]
MAX_FREQ = 1410
SLO_MULTS_P = [1.0, 1.05, 1.1, 1.5, 2.0]
SWEET_THRESH = 5.0


def load_prefill_v1():
    """Load new prefill data and rename columns to match old format."""
    df = pd.read_csv(PREFILL_V1_FILE, sep='\t')
    # New columns: tp, input_len, gpu_clock, batch_size, A, F, TTFT_ms, (A+F)*64_ms, A_energy_mj, F_energy_mj
    # Map to old names used by analysis scripts
    df = df.rename(columns={
        'A': 'P_A_lat',
        'F': 'P_F_lat',
        'A_energy_mj': 'P_A_energy',
        'F_energy_mj': 'P_F_energy',
    })
    return df


def optb_energy(df_cfg, slo_budget):
    """OptB/Lex: min-energy unified freq under SLO."""
    best = None
    for f in FREQS:
        row = df_cfg[df_cfg.gpu_clock == f]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        lat = float(r.P_A_lat) + float(r.P_F_lat)
        eng = float(r.P_A_energy) + float(r.P_F_energy)
        if lat <= slo_budget:
            if best is None or eng < best[0]:
                best = (eng, f)
    return best


def aflex_energy(df_cfg, slo_budget):
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
            lat = float(ra.P_A_lat) + float(rf.P_F_lat)
            eng = float(ra.P_A_energy) + float(rf.P_F_energy)
            if lat <= slo_budget:
                if best is None or eng < best[0]:
                    best = (eng, fa, ff)
    return best


def saving_pct(optb, aflex):
    if optb is None or aflex is None:
        return None
    return (optb[0] - aflex[0]) / optb[0] * 100.0


def main():
    df = load_prefill_v1()
    print(f"Loaded prefill_v1: {len(df)} rows")
    print(f"  tp: {sorted(df.tp.unique())}")
    print(f"  input_len: {sorted(df.input_len.unique())}")
    print(f"  batch_size: {sorted(df.batch_size.unique())}")
    print(f"  gpu_clock: {sorted(df.gpu_clock.unique())}")

    tps = sorted(df.tp.unique())
    ils = sorted(df.input_len.unique())
    bss = sorted(df.batch_size.unique())

    # ═══════════════════════════════════════════════════════════════════════
    # 1. Prefill AFlex vs Lex saving table (Section 2.5 / Section E)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("[1] Prefill: AFlex vs Lex (=BiScale), mean saving by TP and SLO")
    print("=" * 80)
    print(f"{'TP':>4} | " + " | ".join(f"SLO×{s}" for s in SLO_MULTS_P))
    print("-" * 70)
    for tp in tps:
        row_vals = []
        for slo_m in SLO_MULTS_P:
            savings = []
            for il in ils:
                for bs in bss:
                    cfg = df[(df.tp == tp) & (df.input_len == il) & (df.batch_size == bs)]
                    if len(cfg) == 0:
                        continue
                    ref = cfg[cfg.gpu_clock == MAX_FREQ]
                    if len(ref) == 0:
                        continue
                    ref_lat = float(ref.iloc[0].P_A_lat) + float(ref.iloc[0].P_F_lat)
                    slo = ref_lat * slo_m
                    ob = optb_energy(cfg, slo)
                    af = aflex_energy(cfg, slo)
                    s = saving_pct(ob, af)
                    if s is not None:
                        savings.append(s)
            if savings:
                row_vals.append(f"{np.mean(savings):>5.1f}%")
            else:
                row_vals.append("  N/A ")
        print(f"tp={tp} | " + " | ".join(row_vals))

    # ═══════════════════════════════════════════════════════════════════════
    # 2. F frequency sensitivity: F_lat(1410)/F_lat(210) — key table
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("[2] F frequency sensitivity: F_lat(1410)/F_lat(210), bs=4")
    print("    (ratio≈1.0 means memory-bound, ratio≈0.17 means compute-bound)")
    print("=" * 80)
    test_ils = [128, 512, 1024, 2048, 4096]
    print(f"{'TP':>4} | " + " | ".join(f"il={il:>5}" for il in test_ils))
    print("-" * 60)
    for tp in tps:
        row_vals = []
        for il in test_ils:
            cfg = df[(df.tp == tp) & (df.input_len == il) & (df.batch_size == 4)]
            f210 = cfg[cfg.gpu_clock == 210]
            f1410 = cfg[cfg.gpu_clock == 1410]
            if len(f210) > 0 and len(f1410) > 0:
                ratio = float(f1410.iloc[0].P_F_lat) / float(f210.iloc[0].P_F_lat)
                row_vals.append(f"{ratio:>6.3f}")
            else:
                row_vals.append("   N/A")
        print(f"tp={tp} | " + " | ".join(row_vals))

    # ═══════════════════════════════════════════════════════════════════════
    # 3. A frequency sensitivity: A_lat(1410)/A_lat(210) — A_ratio table
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("[3] A frequency sensitivity (A_ratio): A_lat(1410)/A_lat(210)")
    print("=" * 80)
    for bs_test in [1, 4, 16]:
        print(f"\n  bs={bs_test}:")
        print(f"  {'TP':>4} | " + " | ".join(f"il={il:>5}" for il in test_ils))
        print("  " + "-" * 60)
        for tp in tps:
            row_vals = []
            for il in test_ils:
                cfg = df[(df.tp == tp) & (df.input_len == il) & (df.batch_size == bs_test)]
                a210 = cfg[cfg.gpu_clock == 210]
                a1410 = cfg[cfg.gpu_clock == 1410]
                if len(a210) > 0 and len(a1410) > 0:
                    ratio = float(a1410.iloc[0].P_A_lat) / float(a210.iloc[0].P_A_lat)
                    row_vals.append(f"{ratio:>6.3f}")
                else:
                    row_vals.append("   N/A")
            print(f"  tp={tp} | " + " | ".join(row_vals))

    # ═══════════════════════════════════════════════════════════════════════
    # 4. Sweet-spot stats
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("[4] Sweet-spot stats: saving > 5%, SLO×1.0")
    print("=" * 80)
    print(f"{'TP':>4} | {'sweet#':>6} | {'total#':>6} | {'ratio':>6} | {'mean':>6} | {'max':>6}")
    print("-" * 55)
    for tp in tps:
        sweet_savings = []
        all_savings = []
        freq_pairs = []
        for il in ils:
            for bs in bss:
                cfg = df[(df.tp == tp) & (df.input_len == il) & (df.batch_size == bs)]
                if len(cfg) == 0:
                    continue
                ref = cfg[cfg.gpu_clock == MAX_FREQ]
                if len(ref) == 0:
                    continue
                ref_lat = float(ref.iloc[0].P_A_lat) + float(ref.iloc[0].P_F_lat)
                slo = ref_lat * 1.0
                ob = optb_energy(cfg, slo)
                af = aflex_energy(cfg, slo)
                s = saving_pct(ob, af)
                if s is None:
                    continue
                all_savings.append(s)
                if s > SWEET_THRESH:
                    sweet_savings.append(s)
                    freq_pairs.append((af[1], af[2]))
        n_sweet = len(sweet_savings)
        n_total = len(all_savings)
        ratio = n_sweet / n_total * 100 if n_total > 0 else 0
        mean_s = np.mean(sweet_savings) if sweet_savings else 0
        max_s = np.max(sweet_savings) if sweet_savings else 0
        print(f"tp={tp} | {n_sweet:>6} | {n_total:>6} | {ratio:>5.0f}% | "
              f"{mean_s:>5.1f}% | {max_s:>5.1f}%")
        if freq_pairs:
            ctr = Counter(freq_pairs).most_common(5)
            for (fa, ff), cnt in ctr:
                print(f"       ({fa:>4}MHz, {ff:>4}MHz): {cnt:>4} configs")

    # ═══════════════════════════════════════════════════════════════════════
    # 5. F/A latency ratio for ILP配比 (Section 3.3)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("[5] F/A latency ratio at 1410MHz (for ILP配比 analysis)")
    print("=" * 80)
    configs = [
        (1, 128, 1), (1, 8192, 1), (1, 16384, 1),
        (4, 128, 1), (4, 1024, 4),
        (8, 128, 4), (8, 8192, 4),
    ]
    print(f"{'tp':>3} | {'il':>6} | {'bs':>3} | {'A_lat':>8} | {'F_lat':>8} | {'F/A':>6}")
    print("-" * 50)
    for tp, il, bs in configs:
        cfg = df[(df.tp == tp) & (df.input_len == il) & (df.batch_size == bs) & (df.gpu_clock == 1410)]
        if len(cfg) > 0:
            r = cfg.iloc[0]
            a_lat = float(r.P_A_lat)
            f_lat = float(r.P_F_lat)
            print(f"{tp:>3} | {il:>6} | {bs:>3} | {a_lat:>8.0f} | {f_lat:>8.0f} | {f_lat / a_lat:>6.2f}")
        else:
            print(f"{tp:>3} | {il:>6} | {bs:>3} | N/A")

    # ═══════════════════════════════════════════════════════════════════════
    # 6. Detailed per-SLO breakdown with examples
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("[6] Detailed per-TP per-SLO breakdown")
    print("=" * 80)
    for tp in tps:
        print(f"\n  TP={tp}:")
        for slo_m in SLO_MULTS_P:
            savings = []
            examples = []
            for il in ils:
                for bs in bss:
                    cfg = df[(df.tp == tp) & (df.input_len == il) & (df.batch_size == bs)]
                    if len(cfg) == 0:
                        continue
                    ref = cfg[cfg.gpu_clock == MAX_FREQ]
                    if len(ref) == 0:
                        continue
                    ref_lat = float(ref.iloc[0].P_A_lat) + float(ref.iloc[0].P_F_lat)
                    slo = ref_lat * slo_m
                    ob = optb_energy(cfg, slo)
                    af = aflex_energy(cfg, slo)
                    s = saving_pct(ob, af)
                    if s is not None:
                        savings.append(s)
                        if len(examples) < 3 and s > 3.0:
                            examples.append((il, bs, ob[1], af[1], af[2], s))
            if savings:
                n_pos = sum(1 for v in savings if v > 1.0)
                print(f"    SLO×{slo_m}: mean={np.mean(savings):+.2f}%  "
                      f"median={np.median(savings):+.2f}%  "
                      f"max={np.max(savings):+.2f}%  "
                      f"P75={np.percentile(savings, 75):+.2f}%  "
                      f"configs_>1%={n_pos}/{len(savings)}")
                for ex in examples:
                    il, bs, uf, fa, ff, sav = ex
                    print(f"      e.g. il={il} bs={bs}: Lex_f={uf}MHz → AF f_A={fa}/f_F={ff}MHz, saving={sav:.1f}%")


if __name__ == "__main__":
    main()
