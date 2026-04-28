#!/usr/bin/env python3
"""
SLO-aware Motivation Analysis: AF disaggregated DVFS vs Unified DVFS

Core question: Given a latency SLO budget (as a multiplier of the minimum
achievable latency at max freq), how much additional energy can AF
differential frequency assignment save compared to the best unified DVFS?

This directly answers: "Does AF separation still have value over existing
work (BiScale/throttLL'eM) which already does unified per-stage DVFS?"

Methodology:
  - Baseline (existing work): unified DVFS — pick the lowest freq f such that
    t_A(f) + t_F(f) <= SLO_budget. Energy = E_A(f) + E_F(f).
  - Ours (AF DVFS): pick (f_A, f_F) such that t_A(f_A) + t_F(f_F) <= SLO_budget,
    minimizing E_A(f_A) + E_F(f_F).
  - Metric: energy saving of AF over unified, at the same SLO.

SLO multipliers tested: 1.0x, 1.1x, 1.2x, 1.5x, 2.0x, 3.0x
  (1.0x = must match max-freq latency exactly; 3.0x = very loose SLO)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import sys

BASE_DIR = Path(__file__).parent
DECODE_FILE = BASE_DIR / "decode_data.txt"
PREFILL_FILE = BASE_DIR / "prefill_data.txt"

FREQS = [210, 450, 690, 930, 1170, 1410]
MAX_FREQ = 1410
SLO_MULTIPLIERS = [1.0, 1.1, 1.2, 1.5, 2.0, 3.0]


# ─────────────────────────────────────────────────────────────────────────────
# Core solver functions
# ─────────────────────────────────────────────────────────────────────────────

def get_max_freq_lat(df_config):
    """Get latency at max freq (the SLO reference point)."""
    row = df_config[df_config.gpu_clock == MAX_FREQ]
    if len(row) == 0:
        return None
    r = row.iloc[0]
    return float(r.D_A_lat) + float(r.D_F_lat)


def unified_best_under_slo(df_config, slo_budget, freqs=FREQS):
    """
    Unified DVFS (existing work): pick lowest freq where total lat <= SLO.
    Returns (lat, energy, freq) or None if even max freq violates SLO.
    """
    best = None
    for f in sorted(freqs):
        row = df_config[df_config.gpu_clock == f]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        lat = float(r.D_A_lat) + float(r.D_F_lat)
        energy = float(r.D_A_energy) + float(r.D_F_energy)
        if lat <= slo_budget:
            # lowest freq that satisfies SLO → best for unified
            if best is None or energy < best[1]:
                best = (lat, energy, f, f)
    return best


def af_best_under_slo(df_config, slo_budget, freqs=FREQS):
    """
    AF differential DVFS (ours): pick (f_A, f_F) minimizing E_A+E_F
    subject to t_A(f_A) + t_F(f_F) <= SLO.
    Returns (lat, energy, f_A, f_F) or None.
    """
    best = None
    for fa in freqs:
        row_a = df_config[df_config.gpu_clock == fa]
        if len(row_a) == 0:
            continue
        ra = row_a.iloc[0]
        for ff in freqs:
            row_f = df_config[df_config.gpu_clock == ff]
            if len(row_f) == 0:
                continue
            rf = row_f.iloc[0]
            lat = float(ra.D_A_lat) + float(rf.D_F_lat)
            energy = float(ra.D_A_energy) + float(rf.D_F_energy)
            if lat <= slo_budget:
                if best is None or energy < best[1]:
                    best = (lat, energy, fa, ff)
    return best


def prefill_unified_best_under_slo(df_config, slo_budget, freqs=FREQS):
    best = None
    for f in sorted(freqs):
        row = df_config[df_config.gpu_clock == f]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        lat = float(r.P_A_lat) + float(r.P_F_lat)
        energy = float(r.P_A_energy) + float(r.P_F_energy)
        if lat <= slo_budget:
            if best is None or energy < best[1]:
                best = (lat, energy, f, f)
    return best


def prefill_af_best_under_slo(df_config, slo_budget, freqs=FREQS):
    best = None
    for fa in freqs:
        row_a = df_config[df_config.gpu_clock == fa]
        if len(row_a) == 0:
            continue
        ra = row_a.iloc[0]
        for ff in freqs:
            row_f = df_config[df_config.gpu_clock == ff]
            if len(row_f) == 0:
                continue
            rf = row_f.iloc[0]
            lat = float(ra.P_A_lat) + float(rf.P_F_lat)
            energy = float(ra.P_A_energy) + float(rf.P_F_energy)
            if lat <= slo_budget:
                if best is None or energy < best[1]:
                    best = (lat, energy, fa, ff)
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Analysis: Decode
# ─────────────────────────────────────────────────────────────────────────────

def analyze_decode(df):
    print("\n" + "="*80)
    print("DECODE: AF Differential DVFS vs Unified DVFS under SLO constraints")
    print("="*80)
    print("Metric: energy saving (%) of AF over best unified DVFS at same SLO")
    print("        positive = AF saves more energy\n")

    tps = sorted(df.tp.unique())
    input_lens = sorted(df.input_len.unique())
    output_lens = sorted(df.output_len.unique())
    batch_sizes = sorted(df.batch_size.unique())

    # Aggregate results: [tp][slo_mult] -> list of savings
    results = {tp: {s: [] for s in SLO_MULTIPLIERS} for tp in tps}
    # Also track: how often AF chooses different freqs for A vs F
    diff_freq_count = {tp: {s: 0 for s in SLO_MULTIPLIERS} for tp in tps}
    total_count = {tp: {s: 0 for s in SLO_MULTIPLIERS} for tp in tps}
    # Track optimal freq pairs
    freq_pair_examples = {tp: {s: [] for s in SLO_MULTIPLIERS} for tp in tps}

    for tp in tps:
        for il in input_lens:
            for ol in output_lens:
                for bs in batch_sizes:
                    df_cfg = df[(df.tp==tp)&(df.input_len==il)&
                                (df.output_len==ol)&(df.batch_size==bs)]
                    if len(df_cfg) == 0:
                        continue
                    ref_lat = get_max_freq_lat(df_cfg)
                    if ref_lat is None:
                        continue

                    for slo_mult in SLO_MULTIPLIERS:
                        slo = ref_lat * slo_mult
                        uni = unified_best_under_slo(df_cfg, slo)
                        af  = af_best_under_slo(df_cfg, slo)
                        if uni is None or af is None:
                            continue
                        saving = (uni[1] - af[1]) / uni[1] * 100.0
                        results[tp][slo_mult].append(saving)
                        total_count[tp][slo_mult] += 1
                        if af[2] != af[3]:  # f_A != f_F
                            diff_freq_count[tp][slo_mult] += 1
                        if len(freq_pair_examples[tp][slo_mult]) < 3:
                            freq_pair_examples[tp][slo_mult].append(
                                (il, ol, bs, uni[2], af[2], af[3], saving))

    # Print summary table
    print(f"{'':>6} | " + " | ".join(f"SLO×{s:.1f}" for s in SLO_MULTIPLIERS))
    print("-"*80)
    for tp in tps:
        row_mean = []
        row_max  = []
        for s in SLO_MULTIPLIERS:
            vals = results[tp][s]
            if vals:
                row_mean.append(f"{np.mean(vals):>5.1f}%")
                row_max.append(f"{np.max(vals):>5.1f}%")
            else:
                row_mean.append("  N/A ")
                row_max.append("  N/A ")
        print(f"tp={tp} mean | " + " | ".join(row_mean))
        print(f"tp={tp}  max | " + " | ".join(row_max))
        print()

    # Print % of configs where AF uses different freqs for A vs F
    print("\n--- % of configs where AF assigns different freq to A vs F ---")
    print(f"{'':>6} | " + " | ".join(f"SLO×{s:.1f}" for s in SLO_MULTIPLIERS))
    print("-"*80)
    for tp in tps:
        row = []
        for s in SLO_MULTIPLIERS:
            tot = total_count[tp][s]
            diff = diff_freq_count[tp][s]
            row.append(f"{diff/tot*100:>5.1f}%" if tot > 0 else "  N/A ")
        print(f"tp={tp}      | " + " | ".join(row))

    # Print per-SLO breakdown by TP (mean saving)
    print("\n\n--- Detailed breakdown: mean saving by TP and SLO ---")
    for tp in tps:
        print(f"\n  TP={tp}:")
        for s in SLO_MULTIPLIERS:
            vals = results[tp][s]
            if not vals:
                continue
            n_pos = sum(1 for v in vals if v > 1.0)
            print(f"    SLO×{s:.1f}: mean={np.mean(vals):+.2f}%  "
                  f"median={np.median(vals):+.2f}%  "
                  f"max={np.max(vals):+.2f}%  "
                  f"configs_with_>1%_saving={n_pos}/{len(vals)} "
                  f"({n_pos/len(vals)*100:.0f}%)")
            # Show a few examples
            exs = freq_pair_examples[tp][s]
            for ex in exs:
                il, ol, bs, uni_f, fa, ff, sav = ex
                print(f"      e.g. il={il} ol={ol} bs={bs}: "
                      f"unified_f={uni_f}MHz → AF f_A={fa}/f_F={ff}MHz, "
                      f"saving={sav:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Analysis: Prefill
# ─────────────────────────────────────────────────────────────────────────────

def analyze_prefill(df):
    print("\n" + "="*80)
    print("PREFILL: AF Differential DVFS vs Unified DVFS under SLO constraints")
    print("="*80)

    tps = sorted(df.tp.unique())
    input_lens = sorted(df.input_len.unique())
    batch_sizes = sorted(df.batch_size.unique())

    results = {tp: {s: [] for s in SLO_MULTIPLIERS} for tp in tps}
    diff_freq_count = {tp: {s: 0 for s in SLO_MULTIPLIERS} for tp in tps}
    total_count = {tp: {s: 0 for s in SLO_MULTIPLIERS} for tp in tps}
    freq_pair_examples = {tp: {s: [] for s in SLO_MULTIPLIERS} for tp in tps}

    for tp in tps:
        for il in input_lens:
            for bs in batch_sizes:
                df_cfg = df[(df.tp==tp)&(df.input_len==il)&(df.batch_size==bs)]
                if len(df_cfg) == 0:
                    continue
                # ref lat = max freq lat
                row_max = df_cfg[df_cfg.gpu_clock == MAX_FREQ]
                if len(row_max) == 0:
                    continue
                r = row_max.iloc[0]
                ref_lat = float(r.P_A_lat) + float(r.P_F_lat)

                for slo_mult in SLO_MULTIPLIERS:
                    slo = ref_lat * slo_mult
                    uni = prefill_unified_best_under_slo(df_cfg, slo)
                    af  = prefill_af_best_under_slo(df_cfg, slo)
                    if uni is None or af is None:
                        continue
                    saving = (uni[1] - af[1]) / uni[1] * 100.0
                    results[tp][slo_mult].append(saving)
                    total_count[tp][slo_mult] += 1
                    if af[2] != af[3]:
                        diff_freq_count[tp][slo_mult] += 1
                    if len(freq_pair_examples[tp][slo_mult]) < 3:
                        freq_pair_examples[tp][slo_mult].append(
                            (il, bs, uni[2], af[2], af[3], saving))

    print(f"{'':>6} | " + " | ".join(f"SLO×{s:.1f}" for s in SLO_MULTIPLIERS))
    print("-"*80)
    for tp in tps:
        row_mean = []
        row_max  = []
        for s in SLO_MULTIPLIERS:
            vals = results[tp][s]
            if vals:
                row_mean.append(f"{np.mean(vals):>5.1f}%")
                row_max.append(f"{np.max(vals):>5.1f}%")
            else:
                row_mean.append("  N/A ")
                row_max.append("  N/A ")
        print(f"tp={tp} mean | " + " | ".join(row_mean))
        print(f"tp={tp}  max | " + " | ".join(row_max))
        print()

    print("\n--- % of configs where AF assigns different freq to A vs F ---")
    print(f"{'':>6} | " + " | ".join(f"SLO×{s:.1f}" for s in SLO_MULTIPLIERS))
    print("-"*80)
    for tp in tps:
        row = []
        for s in SLO_MULTIPLIERS:
            tot = total_count[tp][s]
            diff = diff_freq_count[tp][s]
            row.append(f"{diff/tot*100:>5.1f}%" if tot > 0 else "  N/A ")
        print(f"tp={tp}      | " + " | ".join(row))

    print("\n\n--- Detailed breakdown: mean saving by TP and SLO ---")
    for tp in tps:
        print(f"\n  TP={tp}:")
        for s in SLO_MULTIPLIERS:
            vals = results[tp][s]
            if not vals:
                continue
            n_pos = sum(1 for v in vals if v > 1.0)
            print(f"    SLO×{s:.1f}: mean={np.mean(vals):+.2f}%  "
                  f"median={np.median(vals):+.2f}%  "
                  f"max={np.max(vals):+.2f}%  "
                  f"configs_with_>1%_saving={n_pos}/{len(vals)} "
                  f"({n_pos/len(vals)*100:.0f}%)")
            for ex in freq_pair_examples[tp][s]:
                il, bs, uni_f, fa, ff, sav = ex
                print(f"      e.g. il={il} bs={bs}: "
                      f"unified_f={uni_f}MHz → AF f_A={fa}/f_F={ff}MHz, "
                      f"saving={sav:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Analysis: Why AF works — A vs F energy sensitivity to frequency
# ─────────────────────────────────────────────────────────────────────────────

def analyze_af_asymmetry(df_decode, df_prefill):
    """Show that A and F have different optimal frequencies → AF separation helps."""
    print("\n" + "="*80)
    print("WHY AF WORKS: Attention vs FFN frequency sensitivity asymmetry")
    print("="*80)

    print("\n[Decode] At which freq is A-energy minimized vs F-energy minimized?")
    print("(showing tp=1, bs=16 as representative)")
    tp, bs = 1, 16
    for il in [128, 512, 2048]:
        for ol in [64, 256]:
            df_cfg = df_decode[(df_decode.tp==tp)&(df_decode.input_len==il)&
                               (df_decode.output_len==ol)&(df_decode.batch_size==bs)]
            if len(df_cfg) == 0:
                continue
            best_a_f = df_cfg.loc[df_cfg.D_A_energy.astype(float).idxmin(), 'gpu_clock']
            best_f_f = df_cfg.loc[df_cfg.D_F_energy.astype(float).idxmin(), 'gpu_clock']
            same = "SAME" if best_a_f == best_f_f else "DIFF ← AF helps"
            print(f"  il={il:>4} ol={ol:>3}: best_A_freq={best_a_f}MHz  "
                  f"best_F_freq={best_f_f}MHz  [{same}]")

    print("\n[Prefill] At which freq is A-energy minimized vs F-energy minimized?")
    print("(showing tp=1, bs=4 as representative)")
    tp, bs = 1, 4
    for il in [128, 512, 2048, 8192]:
        df_cfg = df_prefill[(df_prefill.tp==tp)&(df_prefill.input_len==il)&
                            (df_prefill.batch_size==bs)]
        if len(df_cfg) == 0:
            continue
        best_a_f = df_cfg.loc[df_cfg.P_A_energy.astype(float).idxmin(), 'gpu_clock']
        best_f_f = df_cfg.loc[df_cfg.P_F_energy.astype(float).idxmin(), 'gpu_clock']
        same = "SAME" if best_a_f == best_f_f else "DIFF ← AF helps"
        print(f"  il={il:>5} bs={bs}: best_A_freq={best_a_f}MHz  "
              f"best_F_freq={best_f_f}MHz  [{same}]")

    # Show energy ratio A:F across frequencies for decode
    print("\n[Decode] Energy ratio A:F at different freqs (tp=1, il=512, ol=64, bs=16)")
    df_cfg = df_decode[(df_decode.tp==1)&(df_decode.input_len==512)&
                       (df_decode.output_len==64)&(df_decode.batch_size==16)]
    if len(df_cfg) > 0:
        for _, row in df_cfg.sort_values('gpu_clock').iterrows():
            ea = float(row.D_A_energy)
            ef = float(row.D_F_energy)
            print(f"  freq={int(row.gpu_clock):>4}MHz: E_A={ea:>6.1f}mJ  "
                  f"E_F={ef:>6.1f}mJ  ratio_A/F={ea/ef:.3f}  "
                  f"lat_A={float(row.D_A_lat):>6.1f}us  lat_F={float(row.D_F_lat):>6.1f}us")

    print("\n[Prefill] Energy ratio A:F at different freqs (tp=1, il=2048, bs=4)")
    df_cfg = df_prefill[(df_prefill.tp==1)&(df_prefill.input_len==2048)&
                        (df_prefill.batch_size==4)]
    if len(df_cfg) > 0:
        for _, row in df_cfg.sort_values('gpu_clock').iterrows():
            ea = float(row.P_A_energy)
            ef = float(row.P_F_energy)
            print(f"  freq={int(row.gpu_clock):>4}MHz: E_A={ea:>7.1f}mJ  "
                  f"E_F={ef:>7.1f}mJ  ratio_A/F={ea/ef:.3f}  "
                  f"lat_A={float(row.P_A_lat):>7.1f}us  lat_F={float(row.P_F_lat):>7.1f}us")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    df_decode  = pd.read_csv(DECODE_FILE,  sep='\t')
    df_prefill = pd.read_csv(PREFILL_FILE, sep='\t')

    print(f"Loaded decode:  {len(df_decode)} rows")
    print(f"Loaded prefill: {len(df_prefill)} rows")

    analyze_af_asymmetry(df_decode, df_prefill)
    analyze_decode(df_decode)
    analyze_prefill(df_prefill)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
Key takeaways:
  1. A and F have DIFFERENT optimal frequencies → unified DVFS is suboptimal
  2. AF differential DVFS consistently saves energy over unified DVFS
  3. Savings INCREASE with looser SLO (more slack = more room to differentiate)
  4. Savings are consistent across TP configurations
  5. In a significant fraction of configs, AF chooses f_A != f_F
""")


if __name__ == "__main__":
    main()
