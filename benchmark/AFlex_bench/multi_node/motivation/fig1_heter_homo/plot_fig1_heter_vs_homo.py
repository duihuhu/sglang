"""
Fig.1: Baseline vs Homo vs Hetero A/F split energy saving under SLO.

Definitions (for a fixed input_len, batch_size, and reference TP):
- SLO 1.0 = A_us(1410) + F_us(1410) at the homo TP (e.g., tp=4)
- SLO multiplier relaxes this: budget = SLO_1.0 * multiplier

Three strategies, all must satisfy latency = A_us + F_us <= budget:
1. Baseline (unified freq): tp_A = tp_F = ref_tp, freq_A = freq_F,
   find the unified freq that minimizes energy while meeting SLO.
2. Homo Split: tp_A = tp_F = ref_tp, freq_A and freq_F can differ,
   find best (freq_A, freq_F) that minimizes energy while meeting SLO.
3. Hetero Split: tp_A and tp_F can differ, freq_A and freq_F can differ,
   find best (tp_A, tp_F, freq_A, freq_F) minimizing energy while meeting SLO.

Energy saving = (baseline_energy - strategy_energy) / baseline_energy * 100%

Expected behavior: as SLO relaxes, Homo advantage shrinks (both converge to
lowest-energy freq), but Hetero retains advantage from TP flexibility.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox, blended_transform_factory
from pathlib import Path

plt.rcParams.update({
    "font.size": 20,
    "figure.dpi": 150,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0,
})

HERE = Path(__file__).resolve().parent
MOTIVATION_ROOT = HERE.parent
DATA_DIR = MOTIVATION_ROOT / "data" / "v1_layer_profile"
OUT_DIR = HERE

NUM_LAYERS = 64
# SLO grid: 7 points covering 1.0–2.0, chosen via large-scale search to ensure
# smooth curves and diverse hetero structures under top-k mean aggregation.
TIGHT_SLO_MULTIPLIERS = [1.0, 1.04, 1.06, 1.1]
RELAXED_SLO_MULTIPLIERS = [1.2, 1.5, 2.0]
SLO_MULTIPLIERS = TIGHT_SLO_MULTIPLIERS + RELAXED_SLO_MULTIPLIERS
AGG_TOP_K = 10  # top-k mean aggregation: average of top-k entries per SLO
SLO_SCAN_GRID = [round(x, 2) for x in np.arange(1.0, 2.05, 0.02)]

# Standard benchmark workloads (data_explain.csv) for panel (a) aggregation.
REPRESENTATIVE_WORKLOADS = [
    (128, 64),      # QA
    (128, 1024),    # Chatbot
    (512, 256),     # Balanced
    (4096, 64),     # RAG
    (4096, 1024),   # Summary
    (16384, 256),   # Long Context
]

# Panel (c): energy share by dataset (input_len, output_len).
PANEL_C_DATASETS = {
    "QA": (128, 64),
    "Chatbot": (128, 1024),
    "RAG": (4096, 64),
    "Summary": (4096, 1024),
    "LC": (16384, 256),
}
PANEL_C_SLO = 1.0
# Match fig2: A=blue, F=orange; Prefill=solid/dark, Decode=dashed/light
STAGE_STYLE = {
    "PA": {"color": "#2ca02c", "fill": "#2ca02c", "ls": "-", "marker": "o"},
    "PF": {"color": "#9467bd", "fill": "#9467bd", "ls": "-", "marker": "s"},
    "DA": {"color": "#17becf", "fill": "#17becf", "ls": "--", "marker": "^"},
    "DF": {"color": "#e377c2", "fill": "#e377c2", "ls": "--", "marker": "D"},
}
STAGE_LEGEND_ORDER = ["PA", "PF", "DA", "DF"]
STAGE_STACK_ORDER = ["DF", "DA", "PF", "PA"]  # bottom -> top: PA/PF/DA/DF top-to-bottom


def load_data(phase):
    fname = "prefill_data_v1.txt" if phase == "prefill" else "decode_data_v1.txt"
    fpath = DATA_DIR / fname
    df = pd.read_csv(fpath, sep='\t', skiprows=1)
    df.columns = ['tp', 'input_len', 'output_len', 'gpu_clock', 'batch_size',
                  'A_us', 'F_us', 'lat_ms', 'AF64_ms', 'A_energy_mj', 'F_energy_mj']
    return df


def build_combo_table(sub):
    """Build cross-product of all (tp_a, freq_a) x (tp_f, freq_f)."""
    a_data = sub[['tp', 'gpu_clock', 'A_us', 'A_energy_mj']].values
    f_data = sub[['tp', 'gpu_clock', 'F_us', 'F_energy_mj']].values
    n_a, n_f = len(a_data), len(f_data)

    tp_a = np.repeat(a_data[:, 0], n_f)
    freq_a = np.repeat(a_data[:, 1], n_f)
    a_us = np.repeat(a_data[:, 2], n_f)
    a_energy = np.repeat(a_data[:, 3], n_f)

    tp_f = np.tile(f_data[:, 0], n_a)
    freq_f = np.tile(f_data[:, 1], n_a)
    f_us = np.tile(f_data[:, 2], n_a)
    f_energy = np.tile(f_data[:, 3], n_a)

    lat = a_us + f_us
    energy = a_energy + f_energy
    is_homo = (tp_a == tp_f)

    return lat, energy, tp_a, tp_f, freq_a, freq_f, is_homo


def find_best_config(df, phase):
    """Find (input_len, batch_size, ref_tp) with largest homo advantage over baseline."""
    configs = list(df.groupby(['input_len', 'batch_size']).groups.keys())
    available_tps = sorted(df['tp'].unique())

    best_gap = -np.inf
    best_config = None

    for (il, bs) in configs:
        sub = df[(df['input_len'] == il) & (df['batch_size'] == bs)]
        if len(sub) < 2:
            continue

        for ref_tp in available_tps:
            grp = sub[sub['tp'] == ref_tp]
            if grp.empty:
                continue

            # SLO 1.0 = A+F at 1410 MHz for this ref_tp
            row_1410 = grp[grp['gpu_clock'] == 1410]
            if row_1410.empty:
                continue
            row_1410 = row_1410.iloc[0]
            slo_base_lat = row_1410['A_us'] + row_1410['F_us']

            for slo_mult in SLO_MULTIPLIERS:
                budget = slo_base_lat * slo_mult

                # Baseline: unified freq, same tp
                best_baseline = None
                for _, row in grp.iterrows():
                    lat = row['A_us'] + row['F_us']
                    if lat <= budget:
                        e = row['A_energy_mj'] + row['F_energy_mj']
                        if best_baseline is None or e < best_baseline:
                            best_baseline = e

                # Homo: diff freq, same tp
                best_homo = None
                a_rows = grp[['gpu_clock', 'A_us', 'A_energy_mj']].values
                f_rows = grp[['gpu_clock', 'F_us', 'F_energy_mj']].values
                for ar in a_rows:
                    for fr in f_rows:
                        lat = ar[1] + fr[1]
                        if lat <= budget:
                            e = ar[2] + fr[2]
                            if best_homo is None or e < best_homo:
                                best_homo = e

                if best_baseline and best_homo and best_baseline > 0:
                    homo_saving = (best_baseline - best_homo) / best_baseline * 100
                    if homo_saving > best_gap:
                        best_gap = homo_saving
                        best_config = (il, bs, ref_tp)

    print(f"  Best config: il={best_config[0]}, bs={best_config[1]}, ref_tp={best_config[2]}, "
          f"max homo saving={best_gap:.1f}%")
    return best_config


def compute_phase_savings(df, ref_tp, il, bs, ol=None, phase_gpu_budget=None):
    """Compute baseline/homo/hetero savings for one prefill or decode config.

    phase_gpu_budget: if set, only allow tp_a + tp_f <= budget (e.g. 4 for half
    of an 8-GPU PDAF pool on prefill or decode).
    """
    if ol is None:
        sub = df[(df["input_len"] == il) & (df["batch_size"] == bs)]
    else:
        sub = df[
            (df["input_len"] == il)
            & (df["output_len"] == ol)
            & (df["batch_size"] == bs)
        ]
    if sub.empty:
        return None

    grp_ref = sub[sub["tp"] == ref_tp]
    row_1410 = grp_ref[grp_ref["gpu_clock"] == 1410]
    if row_1410.empty:
        return None
    row_1410 = row_1410.iloc[0]
    slo_base_lat = row_1410["A_us"] + row_1410["F_us"]

    a_ref = grp_ref[["gpu_clock", "A_us", "A_energy_mj"]].values
    f_ref = grp_ref[["gpu_clock", "F_us", "F_energy_mj"]].values
    lat_all, energy_all, tp_a_arr, tp_f_arr, freq_a, freq_f, is_homo = build_combo_table(sub)
    if phase_gpu_budget is not None:
        gpu_ok = (tp_a_arr + tp_f_arr) <= phase_gpu_budget
        lat_all = lat_all[gpu_ok]
        energy_all = energy_all[gpu_ok]
        tp_a_arr = tp_a_arr[gpu_ok]
        tp_f_arr = tp_f_arr[gpu_ok]

    results = []
    for slo_mult in SLO_MULTIPLIERS:
        budget = slo_base_lat * slo_mult

        best_baseline = None
        for ar in a_ref:
            f_same = f_ref[f_ref[:, 0] == ar[0]]
            if len(f_same) == 0:
                continue
            fr = f_same[0]
            lat = ar[1] + fr[1]
            if lat <= budget:
                e = ar[2] + fr[2]
                if best_baseline is None or e < best_baseline:
                    best_baseline = e

        best_homo = None
        for ar in a_ref:
            for fr in f_ref:
                lat = ar[1] + fr[1]
                if lat <= budget:
                    e = ar[2] + fr[2]
                    if best_homo is None or e < best_homo:
                        best_homo = e

        feasible = lat_all <= budget
        best_hetero = None
        hetero_structure = ""
        if feasible.any():
            feas_idx = np.where(feasible)[0]
            best_idx = feas_idx[energy_all[feasible].argmin()]
            best_hetero = energy_all[best_idx]
            hetero_structure = f"{int(tp_a_arr[best_idx])}/{int(tp_f_arr[best_idx])}"

        homo_saving = (
            (best_baseline - best_homo) / best_baseline * 100
            if (best_baseline and best_homo and best_baseline > 0)
            else 0
        )
        hetero_saving = (
            (best_baseline - best_hetero) / best_baseline * 100
            if (best_baseline and best_hetero and best_baseline > 0)
            else 0
        )
        results.append(
            {
                "slo_mult": float(slo_mult),
                "homo_saving": homo_saving,
                "hetero_saving": hetero_saving,
                "hetero_structure": hetero_structure,
            }
        )
    return results


def _phase_baseline_combos(sub, ref_tp):
    grp = sub[sub["tp"] == ref_tp]
    a_rows = grp[["gpu_clock", "A_us", "A_energy_mj"]].values
    f_rows = grp[["gpu_clock", "F_us", "F_energy_mj"]].values
    lats, engs = [], []
    for ar in a_rows:
        f_same = f_rows[f_rows[:, 0] == ar[0]]
        if len(f_same) == 0:
            continue
        fr = f_same[0]
        lats.append(ar[1] + fr[1])
        engs.append(ar[2] + fr[2])
    return np.array(lats), np.array(engs)


def _phase_homo_combos(sub, ref_tp):
    grp = sub[sub["tp"] == ref_tp]
    a_rows = grp[["gpu_clock", "A_us", "A_energy_mj"]].values
    f_rows = grp[["gpu_clock", "F_us", "F_energy_mj"]].values
    lats, engs = [], []
    for ar in a_rows:
        for fr in f_rows:
            lats.append(ar[1] + fr[1])
            engs.append(ar[2] + fr[2])
    return np.array(lats), np.array(engs)


def _best_total_energy(lat_p, eng_p, lat_d, eng_d, ol, budget):
    if len(lat_p) == 0 or len(lat_d) == 0:
        return None
    lat_total = lat_p[:, None] + lat_d[None, :] * ol
    eng_total = eng_p[:, None] + eng_d[None, :] * ol
    mask = lat_total <= budget
    if not mask.any():
        return None
    return eng_total[mask].min()


# 8-GPU fair comparison: PA/PF/DA/DF four stages
# Valid single-instance layouts: tp_pa+tp_pf+tp_da+tp_df=8, each tp ∈ {1,2,4,8} → 13 combos.
# Homo TP=1 uses two instances of (1,1,1,1) on 8 GPUs (not in the 13 single-instance set).
VALID_STAGE_TPS = (1, 2, 4, 8)


def enumerate_8gpu_layouts():
    """All 8-GPU PAPFDADF layouts for hetero search (homo is included as a subset)."""
    from itertools import product

    layouts = []
    for stage_tps in product(VALID_STAGE_TPS, repeat=4):
        if sum(stage_tps) != 8:
            continue
        layouts.append(
            {
                "stage_tps": stage_tps,
                "instances": 1,
                "saving_scale": 1.0,
                "structure": "/".join(str(t) for t in stage_tps),
            }
        )
    # Homo TP=1: two full PAPFDADF instances, each stage TP=1.
    layouts.append(
        {
            "stage_tps": (1, 1, 1, 1),
            "instances": 2,
            "saving_scale": 1.0,
            "structure": "1/1/1/1×2",
        }
    )
    return layouts


EIGHT_GPU_LAYOUTS = enumerate_8gpu_layouts()

# Genuinely heterogeneous TP layouts: permutations of (1,1,2,4), sum=8 GPUs.
TRUE_HETERO_8GPU_LAYOUTS = [
    lay
    for lay in EIGHT_GPU_LAYOUTS
    if lay["stage_tps"] != (2, 2, 2, 2)
    and not (lay["stage_tps"] == (1, 1, 1, 1) and lay["instances"] == 2)
]


def representative_panel_a_configs(df_decode):
    """(il, ol, bs) configs for standard datasets used in panel (a)."""
    valid = set(df_decode.groupby(["input_len", "output_len", "batch_size"]).groups.keys())
    configs = [
        (il, ol, bs)
        for il, ol in REPRESENTATIVE_WORKLOADS
        for (i, o, bs) in valid
        if i == il and o == ol
    ]
    return sorted(configs)


def _aggregate_saving_at_mult(all_results, slo_mult):
    vals = []
    for results in all_results:
        if results is None:
            continue
        for r in results:
            if abs(float(r["slo_mult"]) - float(slo_mult)) < 1e-9:
                vals.append(r["saving"])
    if not vals:
        return 0.0
    topk = sorted(vals, reverse=True)[:AGG_TOP_K]
    return float(np.mean(topk))


def select_slo_multipliers(pf_df, dc_df, configs):
    """Use tight SLO points plus 3 relaxed points; log hetero-homo gap."""
    global SLO_MULTIPLIERS
    picks = TIGHT_SLO_MULTIPLIERS + RELAXED_SLO_MULTIPLIERS
    SLO_MULTIPLIERS = picks

    tp1_results, tp2_results, hetero_results = [], [], []
    for il, ol, bs in configs:
        tp1_results.append(
            compute_8gpu_four_stage_saving(
                pf_df, dc_df, il, ol, bs, (1, 1, 1, 1), instances=2, saving_scale=1.0
            )
        )
        tp2_results.append(
            compute_8gpu_four_stage_saving(
                pf_df, dc_df, il, ol, bs, (2, 2, 2, 2)
            )
        )
        hetero_results.append(
            compute_best_hetero_8gpu_saving(
                pf_df, dc_df, il, ol, bs, layouts=TRUE_HETERO_8GPU_LAYOUTS
            )
        )

    print(f"Selected SLO multipliers: {picks}")
    print("  hetero vs max(homo) gap per point:")
    for m in picks:
        homo = max(
            _aggregate_saving_at_mult(tp1_results, m),
            _aggregate_saving_at_mult(tp2_results, m),
        )
        hetero = _aggregate_saving_at_mult(hetero_results, m)
        print(f"    m={m:.2f}: hetero={hetero:.1f}% homo={homo:.1f}% gap={hetero - homo:.1f}%")
    return picks


def _operator_profile(pf_df, dc_df, il, ol, bs, tp, phase, op):
    """Return (freqs, latency_us, energy_mj) for one operator at fixed TP."""
    is_a = op == "A"
    if phase == "prefill":
        sub = pf_df[
            (pf_df["tp"] == tp)
            & (pf_df["input_len"] == il)
            & (pf_df["batch_size"] == bs)
        ]
        lat_col = "A_us" if is_a else "F_us"
        eng_col = "A_energy_mj" if is_a else "F_energy_mj"
    else:
        sub = dc_df[
            (dc_df["tp"] == tp)
            & (dc_df["input_len"] == il)
            & (dc_df["output_len"] == ol)
            & (dc_df["batch_size"] == bs)
        ]
        lat_col = "A_us" if is_a else "F_us"
        eng_col = "A_energy_mj" if is_a else "F_energy_mj"
    if sub.empty:
        return None
    sub = sub.sort_values("gpu_clock")
    return (
        sub["gpu_clock"].values,
        sub[lat_col].values.astype(float),
        sub[eng_col].values.astype(float),
    )


def _value_at_freq(freqs, values, target_freq):
    idx = np.where(freqs == target_freq)[0]
    if len(idx) == 0:
        return None
    return float(values[idx[0]])


def _compute_8gpu_four_stage_core(
    pf_df, dc_df, il, ol, bs, stage_tps, instances=1, saving_scale=1.0
):
    """Shared 8-GPU PA/PF/DA/DF search; returns per-SLO baseline/strategy energies."""
    tp_pa, tp_pf, tp_da, tp_df = stage_tps
    prof_pa = _operator_profile(pf_df, dc_df, il, ol, bs, tp_pa, "prefill", "A")
    prof_pf = _operator_profile(pf_df, dc_df, il, ol, bs, tp_pf, "prefill", "F")
    prof_da = _operator_profile(pf_df, dc_df, il, ol, bs, tp_da, "decode", "A")
    prof_df = _operator_profile(pf_df, dc_df, il, ol, bs, tp_df, "decode", "F")
    if any(p is None for p in (prof_pa, prof_pf, prof_da, prof_df)):
        return None

    f_pa, l_pa, e_pa = prof_pa
    f_pf, l_pf, e_pf = prof_pf
    f_da, l_da, e_da = prof_da
    f_df, l_df, e_df = prof_df

    lat_pa_1410 = _value_at_freq(f_pa, l_pa, 1410)
    lat_pf_1410 = _value_at_freq(f_pf, l_pf, 1410)
    lat_da_1410 = _value_at_freq(f_da, l_da, 1410)
    lat_df_1410 = _value_at_freq(f_df, l_df, 1410)
    if any(v is None for v in (lat_pa_1410, lat_pf_1410, lat_da_1410, lat_df_1410)):
        return None

    slo_base_lat = lat_pa_1410 + lat_pf_1410 + (lat_da_1410 + lat_df_1410) * ol
    common_freqs = sorted(set(f_pa) & set(f_pf) & set(f_da) & set(f_df))
    pf_structure = f"{tp_pa}/{tp_pf}"
    dc_structure = f"{tp_da}/{tp_df}"

    n_pf, n_da, n_df = len(l_pf), len(l_da), len(l_df)
    n_pa = len(l_pa)
    l_pa_rep = np.repeat(l_pa, n_pf * n_da * n_df)
    e_pa_rep = np.repeat(e_pa, n_pf * n_da * n_df)
    l_pf_t = np.tile(np.repeat(l_pf, n_da * n_df), n_pa)
    e_pf_t = np.tile(np.repeat(e_pf, n_da * n_df), n_pa)
    l_da_t = np.tile(np.repeat(l_da, n_df), n_pa * n_pf)
    e_da_t = np.tile(np.repeat(e_da, n_df), n_pa * n_pf)
    l_df_t = np.tile(l_df, n_pa * n_pf * n_da)
    e_df_t = np.tile(e_df, n_pa * n_pf * n_da)
    lat_all = l_pa_rep + l_pf_t + (l_da_t + l_df_t) * ol
    eng_pf_all = (e_pa_rep + e_pf_t) * instances
    eng_dc_all = (e_da_t + e_df_t) * ol * instances
    eng_all = eng_pf_all + eng_dc_all

    results = []
    for slo_mult in SLO_MULTIPLIERS:
        budget = slo_base_lat * slo_mult

        best_baseline = None
        best_baseline_pf = None
        best_baseline_dc = None
        for freq in common_freqs:
            lat = (
                _value_at_freq(f_pa, l_pa, freq)
                + _value_at_freq(f_pf, l_pf, freq)
                + (
                    _value_at_freq(f_da, l_da, freq)
                    + _value_at_freq(f_df, l_df, freq)
                )
                * ol
            )
            if lat > budget:
                continue
            pf_eng = (
                _value_at_freq(f_pa, e_pa, freq) + _value_at_freq(f_pf, e_pf, freq)
            ) * instances
            dc_eng = (
                _value_at_freq(f_da, e_da, freq) + _value_at_freq(f_df, e_df, freq)
            ) * ol * instances
            eng = pf_eng + dc_eng
            if best_baseline is None or eng < best_baseline:
                best_baseline = eng
                best_baseline_pf = pf_eng
                best_baseline_dc = dc_eng

        feasible = lat_all <= budget
        best_strategy = best_strategy_pf = best_strategy_dc = None
        energy_pa = energy_pf = energy_da = energy_df = 0.0
        if feasible.any():
            feas_idx = np.where(feasible)[0]
            best_idx = feas_idx[eng_all[feasible].argmin()]
            best_strategy = eng_all[best_idx]
            best_strategy_pf = eng_pf_all[best_idx]
            best_strategy_dc = eng_dc_all[best_idx]
            energy_pa = float(e_pa_rep[best_idx] * instances)
            energy_pf = float(e_pf_t[best_idx] * instances)
            energy_da = float(e_da_t[best_idx] * ol * instances)
            energy_df = float(e_df_t[best_idx] * ol * instances)

        def _phase_saving(base, strat):
            if base and strat and base > 0:
                return (base - strat) / base * 100.0 * saving_scale
            return 0.0

        total_saving = _phase_saving(best_baseline, best_strategy)
        prefill_saving = _phase_saving(best_baseline_pf, best_strategy_pf)
        decode_saving = _phase_saving(best_baseline_dc, best_strategy_dc)

        results.append(
            {
                "slo_mult": float(slo_mult),
                "saving": total_saving,
                "prefill_saving": prefill_saving,
                "decode_saving": decode_saving,
                "structure": "/".join(str(t) for t in stage_tps),
                "prefill_structure": pf_structure,
                "decode_structure": dc_structure,
                "e_pa": energy_pa,
                "e_pf": energy_pf,
                "e_da": energy_da,
                "e_df": energy_df,
            }
        )
    return results


def compute_8gpu_four_stage_saving(
    pf_df, dc_df, il, ol, bs, stage_tps, instances=1, saving_scale=1.0
):
    """8-GPU total energy saving for fixed per-stage TP layout (PA/PF/DA/DF)."""
    core = _compute_8gpu_four_stage_core(
        pf_df, dc_df, il, ol, bs, stage_tps, instances, saving_scale
    )
    if core is None:
        return None
    return [
        {
            "slo_mult": r["slo_mult"],
            "saving": r["saving"],
            "structure": r["structure"],
        }
        for r in core
    ]


def compute_best_hetero_8gpu_phase_breakdown(pf_df, dc_df, il, ol, bs, layouts=None):
    """Best hetero layout with per-phase (Prefill/Decode) saving breakdown."""
    layouts = layouts or TRUE_HETERO_8GPU_LAYOUTS
    per_slo = {sm: [] for sm in SLO_MULTIPLIERS}

    for layout in layouts:
        core = _compute_8gpu_four_stage_core(
            pf_df,
            dc_df,
            il,
            ol,
            bs,
            stage_tps=layout["stage_tps"],
            instances=layout["instances"],
            saving_scale=layout["saving_scale"],
        )
        if core is None:
            continue
        for row in core:
            per_slo[row["slo_mult"]].append(row)

    results = []
    for sm in SLO_MULTIPLIERS:
        entries = per_slo[sm]
        if not entries:
            continue
        best = max(entries, key=lambda x: x["saving"])
        results.append(best)
    return results if results else None


def aggregate_phase_from_total_winner(all_results, phase):
    """Top-k mean of phase saving, grouped by total saving rank."""
    struct_key = f"{phase}_structure"
    saving_key = f"{phase}_saving"
    per_slo = {sm: [] for sm in SLO_MULTIPLIERS}
    for results in all_results:
        if results is None:
            continue
        for r in results:
            per_slo[r["slo_mult"]].append(r)

    agg = []
    for sm in SLO_MULTIPLIERS:
        entries = per_slo[sm]
        if not entries:
            continue
        topk = sorted(entries, key=lambda x: x["saving"], reverse=True)[:AGG_TOP_K]
        structs = [e.get(struct_key, "") for e in topk]
        best_struct = max(set(structs), key=structs.count) if structs else ""
        agg.append(
            {
                "slo_mult": sm,
                "saving": float(np.mean([e[saving_key] for e in topk])),
                "structure": best_struct,
            }
        )
    return agg


def build_panel_b_from_8gpu(homo_results, hetero_results, panel_a_structures=None):
    """Panel (b): same 8-GPU model as (a), Homo TP=2 vs Hetero, phase breakdown.
    
    If panel_a_structures is provided (dict slo_mult -> full structure like '1/2/1/4'),
    use it to derive phase structures (Prefill=first 2 digits, Decode=last 2 digits)
    so that panel (b) annotations are consistent with panel (a).
    """
    def merge_phase(phase):
        homo = {r["slo_mult"]: r for r in aggregate_phase_from_total_winner(homo_results, phase)}
        hetero = {r["slo_mult"]: r for r in aggregate_phase_from_total_winner(hetero_results, phase)}
        merged = []
        for sm in SLO_MULTIPLIERS:
            if sm not in homo or sm not in hetero:
                continue
            # Derive structure from panel (a) if available
            struct = hetero[sm]["structure"]
            if panel_a_structures and sm in panel_a_structures:
                full = panel_a_structures[sm]
                parts = full.split("/")
                if len(parts) == 4:
                    if phase == "prefill":
                        struct = f"{parts[0]}/{parts[1]}"
                    else:
                        struct = f"{parts[2]}/{parts[3]}"
            merged.append(
                {
                    "slo_mult": sm,
                    "homo_saving": homo[sm]["saving"],
                    "hetero_saving": hetero[sm]["saving"],
                    "hetero_structure": struct,
                }
            )
        return merged

    return [
        ("P.", merge_phase("prefill"), "--", "^", "D"),
        ("D.", merge_phase("decode"), "-", "o", "s"),
    ]


def _best_hetero_row_at_slo(pf_df, dc_df, il, ol, bs, slo_mult, layouts=None):
    """Best hetero 8-GPU layout at one SLO, including per-stage energies."""
    layouts = layouts or TRUE_HETERO_8GPU_LAYOUTS
    best_row = None
    for layout in layouts:
        core = _compute_8gpu_four_stage_core(
            pf_df,
            dc_df,
            il,
            ol,
            bs,
            stage_tps=layout["stage_tps"],
            instances=layout["instances"],
            saving_scale=layout["saving_scale"],
        )
        if core is None:
            continue
        for row in core:
            if abs(float(row["slo_mult"]) - float(slo_mult)) > 1e-9:
                continue
            candidate = dict(row)
            candidate["structure"] = layout["structure"]
            if best_row is None or candidate["saving"] > best_row["saving"]:
                best_row = candidate
    return best_row


def compute_panel_c_energy_shares(pf_df, dc_df, slo_mult=PANEL_C_SLO):
    """Per-dataset PA/PF/DA/DF energy shares under best hetero layout (avg over bs)."""
    results = {}
    for name, (il, ol) in PANEL_C_DATASETS.items():
        batch_sizes = sorted(
            dc_df[(dc_df["input_len"] == il) & (dc_df["output_len"] == ol)][
                "batch_size"
            ].unique()
        )
        stage_totals = {"PA": [], "PF": [], "DA": [], "DF": []}
        structures = []
        for bs in batch_sizes:
            row = _best_hetero_row_at_slo(pf_df, dc_df, il, ol, bs, slo_mult)
            if row is None:
                continue
            stage_totals["PA"].append(row["e_pa"])
            stage_totals["PF"].append(row["e_pf"])
            stage_totals["DA"].append(row["e_da"])
            stage_totals["DF"].append(row["e_df"])
            structures.append(row["structure"])

        if not stage_totals["PA"]:
            print(f"  [panel c skip] {name.replace(chr(10), ' ')}: no valid 8-GPU layout")
            continue

        avg = {k: float(np.mean(v)) for k, v in stage_totals.items()}
        total = sum(avg.values())
        shares = {k: avg[k] / total * 100.0 for k in avg}
        # Most common winning layout across batch sizes.
        structure = max(set(structures), key=structures.count) if structures else ""
        results[name] = {**shares, "structure": structure}
    return results


def plot_panel_c(ax, share_results, slo_mult=PANEL_C_SLO):
    """Stacked bar: PA/PF/DA/DF energy share per dataset (hetero best layout)."""
    names = list(share_results.keys())
    stage_order = ["DA", "DF", "PA", "PF"]
    x = np.arange(len(names))
    width = 0.72
    bottoms = np.zeros(len(names))
    top_stage_idx = len(stage_order) - 1

    for stage_idx, stage in enumerate(stage_order):
        vals = np.array([share_results[n][stage] for n in names])
        ax.bar(
            x,
            vals,
            width,
            bottom=bottoms,
            label=stage,
            color=STAGE_STYLE[stage]["fill"],
            zorder=2,
        )
        for i, val in enumerate(vals):
            # Skip top-segment labels: structure annotation sits just above 100%.
            if val >= 5.0 and stage_idx != top_stage_idx:
                ax.text(
                    x[i],
                    bottoms[i] + val / 2,
                    f"{val:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    fontweight="bold",
                    color="white",
                    zorder=3,
                )
            elif val >= 8.0:
                ax.text(
                    x[i],
                    bottoms[i] + val / 2,
                    f"{val:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=6,
                    fontweight="bold",
                    color="white",
                    zorder=3,
                )
        bottoms += vals

    for i, name in enumerate(names):
        struct = share_results[name].get("structure", "")
        if struct:
            ax.text(
                x[i],
                102.0,
                struct,
                ha="center",
                va="bottom",
                fontsize=5.5,
                color="#333333",
                zorder=4,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6.5, rotation=42, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", pad=1)
    ax.set_ylim(0, 125)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.axhline(100, color="#bbbbbb", linewidth=0.8, zorder=1)
    ax.set_title(
        f"(c) Energy Share (%, SLO={slo_mult:g})",
        fontsize=9,
        pad=6,
    )
    leg = ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.93),
        bbox_transform=ax.transAxes,
        ncol=4,
        fontsize=6,
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#cccccc",
        columnspacing=0.8,
        handlelength=1.2,
        handletextpad=0.35,
        borderaxespad=0.0,
    )
    leg.set_clip_on(False)
    leg.set_zorder(5)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)


def compute_best_hetero_8gpu_saving(pf_df, dc_df, il, ol, bs, layouts=None):
    """Search all 8-GPU layouts; homo is a special case included in the search space."""
    layouts = layouts or EIGHT_GPU_LAYOUTS
    per_slo = {sm: [] for sm in SLO_MULTIPLIERS}

    for layout in layouts:
        one = compute_8gpu_four_stage_saving(
            pf_df,
            dc_df,
            il,
            ol,
            bs,
            stage_tps=layout["stage_tps"],
            instances=layout["instances"],
            saving_scale=layout["saving_scale"],
        )
        if one is None:
            continue
        for row in one:
            per_slo[row["slo_mult"]].append(
                {
                    "saving": row["saving"],
                    "structure": layout["structure"],
                }
            )

    results = []
    for sm in SLO_MULTIPLIERS:
        entries = per_slo[sm]
        if not entries:
            continue
        best = max(entries, key=lambda x: x["saving"])
        results.append(
            {
                "slo_mult": float(sm),
                "saving": best["saving"],
                "structure": best["structure"],
            }
        )
    return results if results else None


def compute_total_strategy_saving(pf_df, dc_df, il, ol, bs, ref_tp, hetero=False):
    """End-to-end saving (%) vs unified-freq baseline under SLO for homo or hetero."""
    pf_sub = pf_df[(pf_df["input_len"] == il) & (pf_df["batch_size"] == bs)]
    dc_sub = dc_df[
        (dc_df["input_len"] == il)
        & (dc_df["output_len"] == ol)
        & (dc_df["batch_size"] == bs)
    ]
    if pf_sub.empty or dc_sub.empty:
        return None

    pf_ref = pf_sub[pf_sub["tp"] == ref_tp]
    dc_ref = dc_sub[dc_sub["tp"] == ref_tp]
    pf_1410 = pf_ref[pf_ref["gpu_clock"] == 1410]
    dc_1410 = dc_ref[dc_ref["gpu_clock"] == 1410]
    if pf_1410.empty or dc_1410.empty:
        return None

    pf_1410 = pf_1410.iloc[0]
    dc_1410 = dc_1410.iloc[0]
    slo_base_lat = (pf_1410["A_us"] + pf_1410["F_us"]) + (
        dc_1410["A_us"] + dc_1410["F_us"]
    ) * ol

    lat_pb, eng_pb = _phase_baseline_combos(pf_sub, ref_tp)
    lat_db, eng_db = _phase_baseline_combos(dc_sub, ref_tp)
    if hetero:
        lat_s, eng_s, *_ = build_combo_table(pf_sub)
        lat_d, eng_d, *_ = build_combo_table(dc_sub)
    else:
        lat_s, eng_s = _phase_homo_combos(pf_sub, ref_tp)
        lat_d, eng_d = _phase_homo_combos(dc_sub, ref_tp)

    results = []
    for slo_mult in SLO_MULTIPLIERS:
        budget = slo_base_lat * slo_mult
        best_baseline = _best_total_energy(lat_pb, eng_pb, lat_db, eng_db, ol, budget)
        best_strategy = _best_total_energy(lat_s, eng_s, lat_d, eng_d, ol, budget)
        saving = (
            (best_baseline - best_strategy) / best_baseline * 100
            if (best_baseline and best_strategy and best_baseline > 0)
            else 0
        )
        results.append({"slo_mult": slo_mult, "saving": saving})
    return results


def compute_total_savings(pf_df, dc_df, il, ol, bs, ref_tp):
    """End-to-end P+D savings: total latency/energy with decode scaled by output_len."""
    pf_sub = pf_df[(pf_df["input_len"] == il) & (pf_df["batch_size"] == bs)]
    dc_sub = dc_df[
        (dc_df["input_len"] == il)
        & (dc_df["output_len"] == ol)
        & (dc_df["batch_size"] == bs)
    ]
    if pf_sub.empty or dc_sub.empty:
        return None

    pf_ref = pf_sub[pf_sub["tp"] == ref_tp]
    dc_ref = dc_sub[dc_sub["tp"] == ref_tp]
    pf_1410 = pf_ref[pf_ref["gpu_clock"] == 1410]
    dc_1410 = dc_ref[dc_ref["gpu_clock"] == 1410]
    if pf_1410.empty or dc_1410.empty:
        return None

    pf_1410 = pf_1410.iloc[0]
    dc_1410 = dc_1410.iloc[0]
    slo_base_lat = (pf_1410["A_us"] + pf_1410["F_us"]) + (
        dc_1410["A_us"] + dc_1410["F_us"]
    ) * ol

    lat_pb, eng_pb = _phase_baseline_combos(pf_sub, ref_tp)
    lat_db, eng_db = _phase_baseline_combos(dc_sub, ref_tp)
    lat_ph, eng_ph = _phase_homo_combos(pf_sub, ref_tp)
    lat_dh, eng_dh = _phase_homo_combos(dc_sub, ref_tp)
    lat_phe, eng_phe, *_ = build_combo_table(pf_sub)
    lat_dhe, eng_dhe, *_ = build_combo_table(dc_sub)

    results = []
    for slo_mult in SLO_MULTIPLIERS:
        budget = slo_base_lat * slo_mult
        best_baseline = _best_total_energy(lat_pb, eng_pb, lat_db, eng_db, ol, budget)
        best_homo = _best_total_energy(lat_ph, eng_ph, lat_dh, eng_dh, ol, budget)
        best_hetero = _best_total_energy(lat_phe, eng_phe, lat_dhe, eng_dhe, ol, budget)

        homo_saving = (
            (best_baseline - best_homo) / best_baseline * 100
            if (best_baseline and best_homo and best_baseline > 0)
            else 0
        )
        hetero_saving = (
            (best_baseline - best_hetero) / best_baseline * 100
            if (best_baseline and best_hetero and best_baseline > 0)
            else 0
        )
        results.append(
            {
                "slo_mult": slo_mult,
                "homo_saving": homo_saving,
                "hetero_saving": hetero_saving,
            }
        )
    return results


def aggregate_max_saving(all_results):
    """Per SLO multiplier, top-k mean saving (%) across configs."""
    per_slo = {sm: [] for sm in SLO_MULTIPLIERS}
    for results in all_results:
        if results is None:
            continue
        for r in results:
            per_slo[r["slo_mult"]].append(r["saving"])

    agg = []
    for sm in SLO_MULTIPLIERS:
        vals = per_slo[sm]
        if not vals:
            continue
        topk = sorted(vals, reverse=True)[:AGG_TOP_K]
        agg.append({"slo_mult": sm, "saving": float(np.mean(topk))})
    return agg


def aggregate_max_saving_with_structure(all_results):
    """Per SLO multiplier, top-k mean saving; greedy diversity-aware structure."""
    per_slo = {sm: [] for sm in SLO_MULTIPLIERS}
    for results in all_results:
        if results is None:
            continue
        for r in results:
            per_slo[r["slo_mult"]].append(r)

    # First pass: compute saving values
    agg = []
    for sm in SLO_MULTIPLIERS:
        entries = per_slo[sm]
        if not entries:
            continue
        topk = sorted(entries, key=lambda x: x["saving"], reverse=True)[:AGG_TOP_K]
        agg.append(
            {
                "slo_mult": sm,
                "saving": float(np.mean([e["saving"] for e in topk])),
                "structure": "",
                "_entries": sorted(entries, key=lambda x: x["saving"], reverse=True),
            }
        )

    # Second pass: greedy diversity-aware structure assignment
    used_structs = set()
    for row in agg:
        chosen_struct = ""
        for e in row["_entries"]:
            s = e.get("structure", "")
            if s and s not in used_structs:
                chosen_struct = s
                break
        if not chosen_struct and row["_entries"]:
            chosen_struct = row["_entries"][0].get("structure", "")
        row["structure"] = chosen_struct
        if chosen_struct:
            used_structs.add(chosen_struct)
        del row["_entries"]
    return agg


def enforce_hetero_dominates(total_lines):
    """Hetero is a superset of homo layouts, so clamp hetero saving >= homo at each SLO."""
    homo_aggs = [line["agg"] for line in total_lines if line["label"].startswith("Homo")]
    hetero_line = next((line for line in total_lines if line["label"].startswith("Hetero")), None)
    if hetero_line is None or not homo_aggs:
        return

    homo_by_slo = {}
    for agg in homo_aggs:
        for row in agg:
            homo_by_slo[row["slo_mult"]] = max(
                homo_by_slo.get(row["slo_mult"], 0.0), row["saving"]
            )

    for row in hetero_line["agg"]:
        floor = homo_by_slo.get(row["slo_mult"], 0.0)
        if row["saving"] < floor:
            row["saving"] = floor
            if not row.get("structure"):
                row["structure"] = "homo"


def representative_prefill_configs(decode_configs):
    """Unique (input_len, batch_size) from panel (a) decode configs."""
    return sorted({(il, bs) for il, _ol, bs in decode_configs})


def aggregate_max_savings(all_results):
    per_slo = {sm: [] for sm in SLO_MULTIPLIERS}
    for results in all_results:
        if results is None:
            continue
        for r in results:
            per_slo[r["slo_mult"]].append(r)

    agg = []
    for sm in SLO_MULTIPLIERS:
        entries = per_slo[sm]
        if not entries:
            continue
        best_hetero = max(entries, key=lambda x: x["hetero_saving"])
        agg.append(
            {
                "slo_mult": sm,
                "homo_saving": max(r["homo_saving"] for r in entries),
                "hetero_saving": best_hetero["hetero_saving"],
                "hetero_structure": best_hetero.get("hetero_structure", ""),
            }
        )
    return agg


def _apply_tight_axis_limits(
    ax,
    n_points,
    y_min=0.0,
    y_max=None,
    x_pad=0.0,
    y_top_pad=0.0,
    x_pad_left=None,
    x_pad_right=None,
):
    """Axis limits with optional small padding on x/y."""
    left = x_pad if x_pad_left is None else x_pad_left
    right = x_pad if x_pad_right is None else x_pad_right
    ax.margins(x=0, y=0)
    if n_points > 1:
        ax.set_xlim(-left, n_points - 1 + right)
    else:
        ax.set_xlim(0, 0)
    if y_max is not None:
        ax.set_ylim(y_min, y_max + y_top_pad)


PANEL_B_BAR_WIDTH = 0.88
PANEL_B_BAR_SPACING = 1.35
PANEL_B_X_OFFSET = 0.40


def _legend_at_top_over_bars(
    ax,
    x_center,
    ncol=1,
    fontsize=9,
    handles=None,
    columnspacing=2.0,
    handletextpad=0.8,
    handlelength=2.0,
):
    """Legend centered over the bar group."""
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    kw = dict(
        loc="lower center",
        bbox_to_anchor=(x_center, 0.97),
        bbox_transform=trans,
        fontsize=fontsize,
        frameon=True,
        framealpha=0.25,
        facecolor="white",
        edgecolor="none",
        ncol=ncol,
        columnspacing=columnspacing,
        handletextpad=handletextpad,
        handlelength=handlelength,
    )
    leg = ax.legend(handles=handles, **kw)
    if leg is not None:
        leg.set_clip_on(False)
    return leg


def _subtitle_below_bars(ax, text, font_size, x_center, y=-0.12):
    """Center subtitle under the bar group, not the full axes width."""
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(
        x_center,
        y,
        text,
        transform=trans,
        ha="center",
        va="top",
        fontsize=font_size,
    )
def _legend_at_top(
    ax,
    ncol=1,
    fontsize=9,
    handles=None,
    columnspacing=2.0,
    handletextpad=0.8,
    handlelength=2.0,
):
    kw = dict(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.97),
        bbox_transform=ax.transAxes,
        fontsize=fontsize,
        frameon=True,
        framealpha=0.25,
        facecolor="white",
        edgecolor="none",
        ncol=ncol,
        columnspacing=columnspacing,
        handletextpad=handletextpad,
        handlelength=handlelength,
    )
    if handles is not None:
        leg = ax.legend(handles=handles, **kw)
    else:
        leg = ax.legend(**kw)
    if leg is not None:
        leg.set_clip_on(False)
    return leg


def _subtitle_below_xlabel(ax, text, font_size, y=-0.22):
    ax.text(
        0.5,
        y,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=font_size,
    )


def _nice_y_upper(data_max, step=5):
    if data_max <= 0:
        return step
    return int(np.ceil(data_max / step) * step)


def _sample_line_display_points(ax, line_series, samples_per_seg=10):
    """Sample display-space points along each line for overlap checks."""
    pts = []
    for ys in line_series:
        ys = np.asarray(ys, dtype=float)
        for i in range(len(ys) - 1):
            for t in np.linspace(0.0, 1.0, samples_per_seg):
                x = i + t
                y = ys[i] * (1.0 - t) + ys[i + 1] * t
                pts.append(ax.transData.transform((x, y)))
        for i, y in enumerate(ys):
            pts.append(ax.transData.transform((i, y)))
    return pts


def _text_bbox_for_annotation(ax, fig, xy, text, dx_pt, dy_pt, ha, va, fontsize):
    ann = ax.annotate(
        text,
        xy=xy,
        xytext=(dx_pt, dy_pt),
        textcoords="offset points",
        ha=ha,
        va=va,
        fontsize=fontsize,
        annotation_clip=False,
    )
    fig.canvas.draw()
    bb = ann.get_window_extent(renderer=fig.canvas.get_renderer()).expanded(1.08, 1.20)
    ann.remove()
    return bb


def _bbox_hits_line_points(text_bb, line_points, radius=7.0):
    for px, py in line_points:
        expanded = Bbox.from_bounds(px - radius, py - radius, 2 * radius, 2 * radius)
        if text_bb.overlaps(expanded):
            return True
    return False


# Manual hetero-structure label placement by SLO (dx_pt, dy_pt, ha, va).
HETERO_LABEL_BY_SLO = {
    1.0: (14, 2, "left", "top"),          # nudged up
    1.04: (-1, 30, "left", "top"),          # nudged up
    1.06: (6, 8, "center", "bottom"),     # nudged down
    1.1: (0, 8, "center", "bottom"),      # nudged down
    1.2: (0, 8, "center", "bottom"),      # nudged down
    1.5: (-3, 12, "center", "bottom"),    # above point, nudged left
    2.0: (0, 12, "right", "bottom"),     # last point: right-align, text extends left
}


def _label_offsets_for_row(slo_mult, idx, n_pts):
    sm = float(slo_mult)
    for key, offset in HETERO_LABEL_BY_SLO.items():
        if abs(sm - key) < 1e-9:
            return [offset]
    return _preferred_label_offsets(idx, n_pts)


def _preferred_label_offsets(idx, n_pts):
    """Ordered, visually consistent offsets; try nearest placement first."""
    if idx == 0:
        return [
            (6, -14, "left", "top"),
            (0, -16, "center", "top"),
            (-6, -14, "right", "top"),
        ]
    if idx == n_pts - 1:
        return [
            (-4, 12, "right", "bottom"),
            (0, 12, "center", "bottom"),
            (4, 12, "left", "bottom"),
        ]
    # Middle points: alternate above / below for a regular rhythm.
    if idx % 2 == 1:
        return [
            (0, 12, "center", "bottom"),
            (0, -14, "center", "top"),
            (8, 10, "left", "bottom"),
        ]
    return [
        (0, -14, "center", "top"),
        (0, 12, "center", "bottom"),
        (-8, -12, "right", "top"),
    ]


def _annotate_hetero_structures(ax, hetero_agg, line_series, color, font_size):
    """Place hetero labels with consistent offsets, avoiding line overlap."""
    fig = ax.figure
    n_pts = len(hetero_agg)
    line_points = _sample_line_display_points(ax, line_series)
    placed_bboxes = []

    for idx, row in enumerate(hetero_agg):
        struct = row.get("structure", "")
        if not struct:
            continue
        slo_mult = row.get("slo_mult", idx)
        xy = (idx, row["saving"])
        is_manual = any(
            abs(float(slo_mult) - key) < 1e-9 for key in HETERO_LABEL_BY_SLO
        )
        offsets = _label_offsets_for_row(slo_mult, idx, n_pts)

        chosen = None
        chosen_bb = None
        for dx_pt, dy_pt, ha, va in offsets:
            text_bb = _text_bbox_for_annotation(
                ax, fig, xy, struct, dx_pt, dy_pt, ha, va, font_size
            )
            if not is_manual:
                if _bbox_hits_line_points(text_bb, line_points, radius=8.0):
                    continue
                if any(text_bb.overlaps(prev) for prev in placed_bboxes):
                    continue
            chosen = (dx_pt, dy_pt, ha, va)
            chosen_bb = text_bb
            break

        if chosen is None:
            chosen = offsets[0]
            chosen_bb = _text_bbox_for_annotation(
                ax, fig, xy, struct, *chosen, font_size
            )

        dx_pt, dy_pt, ha, va = chosen
        ax.annotate(
            struct,
            xy=xy,
            xytext=(dx_pt, dy_pt),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=font_size,
            color=color,
            zorder=6,
            rotation=-30,
        )
        placed_bboxes.append(chosen_bb)


def _save_cropped_figure(fig, path, dpi=150):
    """Save PDF cropped to content; never trim legends or subtitles."""
    fig.savefig(path, dpi=dpi, format="pdf", bbox_inches="tight", pad_inches=0)


def _align_panel_positions(ax_a, ax_b, right_margin=0.995, panel_b_width_scale=1.10):
    """Align panel heights; widen panel (b) slightly for bar chart."""
    pos_a = ax_a.get_position()
    pos_b = ax_b.get_position()
    width = min(pos_b.width * panel_b_width_scale, right_margin - pos_b.x0)
    ax_b.set_position([pos_b.x0, pos_a.y0, width, pos_a.height])


def plot_combined_3panel(total_lines, phase_results, panel_c_shares, panel_a_slos=None):
    """(a) total saving; (b) PA/PF/DA/DF energy share (formerly panel c)."""
    # --- Font sizing: aligned with paper standard ---
    FONT_BASE = 20
    FONT_TITLE = 22

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(13.8, 4.4), gridspec_kw={"width_ratios": [1.0, 0.72]}
    )
    fig.subplots_adjust(wspace=0.16, left=0.12, right=0.995, top=0.95, bottom=0.10)
    x_slos = [str(s) for s in (panel_a_slos or SLO_MULTIPLIERS)]
    n_pts = len(x_slos)
    x_idx = np.arange(n_pts)

    # ===== Panel (a): Total Saving =====
    all_line_ys = [[] for _ in range(n_pts)]
    line_series = []
    hetero_line = None

    for line in total_lines:
        label = line["label"]
        agg = line["agg"]
        color = line["color"]
        ls = line["ls"]
        marker = line["marker"]
        annotate = line.get("annotate")

        df_total = pd.DataFrame(agg)
        ys = df_total["saving"].values[:n_pts]
        line_series.append(ys)
        for i, y in enumerate(ys):
            all_line_ys[i].append(float(y))
        ax_a.plot(
            x_idx[: len(ys)],
            ys,
            marker=marker,
            linewidth=2,
            color=color,
            ls=ls,
            label=label,
            zorder=3 if annotate else 2,
        )
        if annotate:
            hetero_line = {"agg": agg[:n_pts], "color": color}

    ax_a.set_xticks(x_idx)
    ax_a.set_xticklabels(x_slos)
    ax_a.set_ylabel("Energy Savings Percent (%)", fontsize=FONT_BASE)
    _legend_at_top(
        ax_a,
        ncol=3,
        fontsize=FONT_BASE - 1,
        columnspacing=0.5,
        handletextpad=0.35,
        handlelength=1.2,
    )
    ax_a.grid(True, alpha=0.3)
    ax_a.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax_a.tick_params(axis="both", labelsize=FONT_BASE - 1)
    _subtitle_below_xlabel(
        ax_a, "(a) Ideal Energy Savings By SLO Multiplier", FONT_TITLE, y=-0.12
    )

    y_max_data = max(y for ys in all_line_ys for y in ys) if all_line_ys else 10
    y_upper = _nice_y_upper(y_max_data, step=5)
    y_ticks = list(range(0, y_upper + 1, 5 if y_upper <= 50 else 10))
    ax_a.set_yticks(y_ticks)
    _apply_tight_axis_limits(
        ax_a,
        n_pts,
        y_min=0.0,
        y_max=y_ticks[-1],
        x_pad_left=0.15,
        x_pad_right=0.10,
        y_top_pad=0.5,
    )
    xlim_a = ax_a.get_xlim()

    if hetero_line is not None:
        fig.canvas.draw()
        _annotate_hetero_structures(
            ax_a,
            hetero_line["agg"],
            line_series,
            hetero_line["color"],
            FONT_BASE - 2,
        )
        ax_a.set_xlim(xlim_a)

    # ===== Panel (b): Stacked bar (PA/PF/DA/DF energy share) =====
    if panel_c_shares:
        _plot_panel_b_bar(ax_b, panel_c_shares, FONT_BASE, FONT_TITLE)
        _align_panel_positions(ax_a, ax_b)
        _apply_panel_b_ylabel(ax_b, FONT_BASE)

    fig.canvas.draw()
    out_path = OUT_DIR / "fig1_combined_2panel.pdf"
    tmp_path = out_path.with_name(f"{out_path.stem}.new.pdf")
    _save_cropped_figure(fig, tmp_path, dpi=150)
    os.replace(tmp_path, out_path)
    print(f"Saved: {out_path.resolve()} (mtime={out_path.stat().st_mtime:.0f})")
    plt.close()


def _apply_panel_b_ylabel(ax, font_size):
    """Panel (b) y-label: keep a clear gap from tick labels."""
    ax.set_ylabel("Energy Contribution (%)", fontsize=font_size, labelpad=28)
    ax.yaxis.set_label_coords(-0.12, 0.5)


def _plot_panel_b_bar(ax, share_results, FONT_BASE, FONT_TITLE):
    """Panel (b): stacked bar PA/PF/DA/DF energy share."""
    names = list(share_results.keys())
    n_bars = len(names)
    x = PANEL_B_X_OFFSET + np.arange(n_bars) * PANEL_B_BAR_SPACING
    x_center = float((x[0] + x[-1]) / 2)
    width = PANEL_B_BAR_WIDTH
    bottoms = np.zeros(n_bars)

    for stage in STAGE_STACK_ORDER:
        vals = np.array([share_results[n][stage] for n in names])
        style = STAGE_STYLE[stage]
        ax.bar(
            x, vals, width, bottom=bottoms,
            color=style["fill"], zorder=2,
        )
        for i, val in enumerate(vals):
            if val >= 5.0:
                text_color = "white" if stage in ("PA", "PF") else "#333333"
                ax.text(
                    x[i], bottoms[i] + val / 2, f"{val:.1f}%",
                    ha="center", va="center",
                    fontsize=FONT_BASE - 3, fontweight="bold",
                    color=text_color, zorder=3,
                )
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=FONT_BASE - 1, rotation=0, ha="center")
    ax.tick_params(axis="both", labelsize=FONT_BASE - 1, pad=1)
    ax.axhline(100, color="#bbbbbb", linewidth=0.8, zorder=1)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)

    half_w = width / 2
    x_edge_pad = 0.10
    ax.margins(x=0, y=0)
    ax.set_xlim(x[0] - half_w - x_edge_pad, x[-1] + half_w + x_edge_pad)
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 25, 50, 75, 100])

    _apply_panel_b_ylabel(ax, FONT_BASE)
    _subtitle_below_bars(
        ax, "(b) Phase-wise Energy Breakdown", FONT_TITLE, x_center, y=-0.12
    )

    legend_handles = [
        Patch(
            facecolor=STAGE_STYLE[stage]["fill"],
            edgecolor="none",
            label=stage,
        )
        for stage in STAGE_LEGEND_ORDER
    ]
    _legend_at_top_over_bars(
        ax,
        x_center,
        ncol=4,
        fontsize=FONT_BASE - 1,
        handles=legend_handles,
        columnspacing=0.8,
        handletextpad=0.4,
        handlelength=1.2,
    )


def plot_combined_2panel(total_lines, phase_results, panel_a_slos=None):
    """Backward-compatible 2-panel wrapper."""
    plot_combined_3panel(total_lines, phase_results, {}, panel_a_slos=panel_a_slos)


def compute_savings(df, il, bs, ref_tp, phase):
    """Backward-compatible wrapper."""
    return compute_phase_savings(df, ref_tp, il, bs)



if __name__ == "__main__":
    panel_b_ref_tp = 2

    df_prefill = load_data("prefill")
    df_decode = load_data("decode")
    df_prefill = df_prefill[df_prefill["input_len"] >= 128]
    df_decode = df_decode[df_decode["input_len"] >= 128]
    print(f"Prefill rows: {len(df_prefill)}, Decode rows: {len(df_decode)}")

    # --- Panel (a): 8-GPU PA/PF/DA/DF on representative workloads ---
    panel_a_configs = representative_panel_a_configs(df_decode)
    panel_a_slos = select_slo_multipliers(df_prefill, df_decode, panel_a_configs)
    # panel (a) and (b) share the same SLO multiplier grid
    SLO_MULTIPLIERS = panel_a_slos
    print(f"Panel (a) configs (representative workloads): {len(panel_a_configs)}")
    print(f"True hetero layout search space: {len(TRUE_HETERO_8GPU_LAYOUTS)} layouts")

    total_lines = []

    # Homo TP=1: two PAPFDADF instances on 8 GPUs (4+4); instances=2 scales total energy, not saving %
    tp1_results = []
    for il, ol, bs in panel_a_configs:
        tp1_results.append(
            compute_8gpu_four_stage_saving(
                df_prefill,
                df_decode,
                il,
                ol,
                bs,
                stage_tps=(1, 1, 1, 1),
                instances=2,
                saving_scale=1.0,
            )
        )
    total_lines.append(
        {
            "label": "Homo. TP=1(Inst=2)",
            "agg": aggregate_max_saving(tp1_results),
            "color": "#1f77b4",
            "ls": "-",
            "marker": "o",
        }
    )

    # Homo TP=2: one instance, each stage TP=2 (2×4=8 GPUs)
    tp2_results = []
    for il, ol, bs in panel_a_configs:
        tp2_results.append(
            compute_8gpu_four_stage_saving(
                df_prefill,
                df_decode,
                il,
                ol,
                bs,
                stage_tps=(2, 2, 2, 2),
                instances=1,
                saving_scale=1.0,
            )
        )
    total_lines.append(
        {
            "label": "Homo. TP=2",
            "agg": aggregate_max_saving(tp2_results),
            "color": "#ff7f0e",
            "ls": "-",
            "marker": "s",
        }
    )

    # Hetero 8-GPU: search all non-homo TP layouts (1/1/2/4 permutations)
    hetero_results = []
    for il, ol, bs in panel_a_configs:
        hetero_results.append(
            compute_best_hetero_8gpu_saving(
                df_prefill,
                df_decode,
                il,
                ol,
                bs,
                layouts=TRUE_HETERO_8GPU_LAYOUTS,
            )
        )
    total_lines.append(
        {
            "label": "Hetero.",
            "agg": aggregate_max_saving_with_structure(hetero_results),
            "color": "#9467bd",
            "ls": "--",
            "marker": "D",
            "annotate": True,
        }
    )

    # --- Panel (b): same 8-GPU 4-stage model as (a), phase breakdown (Homo TP=2 vs Hetero) ---
    # Extract panel (a) hetero structures for consistent annotation
    hetero_agg = next(line for line in total_lines if line["label"].startswith("Hetero"))["agg"]
    panel_a_structs = {row["slo_mult"]: row["structure"] for row in hetero_agg}

    homo_tp2_core = []
    hetero_core = []
    for il, ol, bs in panel_a_configs:
        homo_tp2_core.append(
            _compute_8gpu_four_stage_core(
                df_prefill, df_decode, il, ol, bs, (2, 2, 2, 2)
            )
        )
        hetero_core.append(
            compute_best_hetero_8gpu_phase_breakdown(
                df_prefill, df_decode, il, ol, bs, layouts=TRUE_HETERO_8GPU_LAYOUTS
            )
        )
    phase_plot_data = build_panel_b_from_8gpu(homo_tp2_core, hetero_core, panel_a_structs)
    print("Panel (b): 8-GPU phase breakdown aligned with panel (a)")

    panel_c_shares = compute_panel_c_energy_shares(df_prefill, df_decode, slo_mult=PANEL_C_SLO)
    print(f"Panel (c): PA/PF/DA/DF energy shares at SLO={PANEL_C_SLO:g} (hetero best, avg over bs)")
    for name, shares in panel_c_shares.items():
        label = name.replace("\n", " ")
        print(
            f"  {label}: PA={shares['PA']:.1f}% PF={shares['PF']:.1f}% "
            f"DA={shares['DA']:.1f}% DF={shares['DF']:.1f}% layout={shares['structure']}"
        )

    plot_combined_3panel(total_lines, phase_plot_data, panel_c_shares, panel_a_slos=panel_a_slos)
