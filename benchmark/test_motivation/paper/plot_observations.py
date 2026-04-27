#!/usr/bin/env python3
"""Plot all 5 Observation figures for the AFlex paper."""

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import defaultdict
from pathlib import Path

mpl.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 9, 'figure.dpi': 150,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

BASE = Path(__file__).parent.parent   # benchmark/test_motivation
OUT  = Path(__file__).parent / "figs"
OUT.mkdir(exist_ok=True)

FREQS = [210, 450, 690, 930, 1170, 1410]

# ── stub: will be filled by StrReplace ──
# ── Load data ──────────────────────────────────────────────
def load_decode_v1():
    rows = []
    with open(BASE / "decode_data_v1.txt") as f:
        header = None
        for line in f:
            if line.startswith("[D]"):
                continue
            if line.startswith("tp"):
                header = line.strip().split("\t")
                continue
            parts = line.strip().split("\t")
            if header and len(parts) >= len(header):
                rows.append({h: parts[i] for i, h in enumerate(header)})
    return rows

def load_prefill_v1():
    rows = []
    with open(BASE / "paper" / "prefill_data_v1.txt") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(r)
    return rows
decode_rows = load_decode_v1()
prefill_rows = load_prefill_v1()

D = {}  # decode: (tp, il, ol, freq, bs) -> (A_lat, F_lat, A_e, F_e)
P = {}  # prefill: (tp, il, freq, bs) -> (A_lat, F_lat, A_e, F_e)
for r in decode_rows:
    try:
        key = (int(r['tp']), int(r['input_len']), int(r['output_len']),
               int(r['gpu_clock']), int(r['batch_size']))
        D[key] = (float(r['A']), float(r['F']),
                  float(r['A_energy_mj']), float(r['F_energy_mj']))
    except (ValueError, KeyError):
        pass
for r in prefill_rows:
    try:
        key = (int(r['tp']), int(r['input_len']),
               int(r['gpu_clock']), int(r['batch_size']))
        P[key] = (float(r['A']), float(r['F']),
                  float(r['A_energy_mj']), float(r['F_energy_mj']))
    except (ValueError, KeyError):
        pass
print(f"Loaded {len(D)} decode, {len(P)} prefill entries")
# ══ Obs 1 ══════════════════════════════════════════════════
def plot_obs1():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # ── (a) Decode: tp=1 vs tp=4, il=1024, ol=64, bs=16 ──
    ax = axes[0]
    ax.set_title("(a) Decode")
    for tp, clr in [(1, 'C0'), (4, 'C1')]:
        base_a = D.get((tp,1024,64,1410,16), (1,))[0]
        base_f = D.get((tp,1024,64,1410,16), (0,1))[1]
        a_n = [D.get((tp,1024,64,f,16),(np.nan,))[0] / base_a for f in FREQS]
        f_n = [D.get((tp,1024,64,f,16),(0,np.nan))[1] / base_f for f in FREQS]
        ax.plot(FREQS, a_n, marker='o', ls='-',  label=f"A (tp={tp})", color=clr)
        ax.plot(FREQS, f_n, marker='^', ls='--', label=f"F (tp={tp})", color=clr)
    ax.axhline(1.0, color='gray', ls=':', lw=0.8)
    ax.set_xlabel("GPU Frequency (MHz)")
    ax.set_ylabel("Normalized Latency (rel. 1410 MHz)")
    ax.legend(fontsize=8, loc='upper right'); ax.grid(True, alpha=0.3)

    # ── (b) Prefill: tp=1 vs tp=4, il=128, bs=1 ──
    ax = axes[1]
    ax.set_title("(b) Prefill")
    for tp, clr in [(1, 'C0'), (4, 'C1')]:
        base_a = P.get((tp, 128, 1410, 1), (1,))[0]
        base_f = P.get((tp, 128, 1410, 1), (0,1))[1]
        a_n = [P.get((tp,128,f,1),(np.nan,))[0] / base_a for f in FREQS]
        f_n = [P.get((tp,128,f,1),(0,np.nan))[1] / base_f for f in FREQS]
        ax.plot(FREQS, a_n, marker='o', ls='-',  label=f"A (tp={tp})", color=clr)
        ax.plot(FREQS, f_n, marker='^', ls='--', label=f"F (tp={tp})", color=clr)
    ax.axhline(1.0, color='gray', ls=':', lw=0.8)
    ax.set_xlabel("GPU Frequency (MHz)")
    ax.set_ylabel("Normalized Latency (rel. 1410 MHz)")
    ax.legend(fontsize=8, loc='upper right'); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT/"obs1_normalized_latency.pdf")
    fig.savefig(OUT/"obs1_normalized_latency.png")
    print("Obs1 saved."); plt.close(fig)
# ══ Obs 2 ══════════════════════════════════════════════════
def _ratio_d(tp, il, bs, ol=64):
    k0 = (tp, il, ol, 210, bs); k1 = (tp, il, ol, 1410, bs)
    if k0 not in D or k1 not in D: return np.nan, np.nan
    return D[k1][0]/D[k0][0], D[k1][1]/D[k0][1]

def _ratio_p(tp, il, bs):
    k0 = (tp, il, 210, bs); k1 = (tp, il, 1410, bs)
    if k0 not in P or k1 not in P: return np.nan, np.nan
    return P[k1][0]/P[k0][0], P[k1][1]/P[k0][1]

def _heatmap(ax, mat, xlabels, ylabels, xlabel, ylabel, title, vmin=0.15, vmax=1.05):
    # RdYlBu: red=low(compute-bound/sensitive), blue=high(memory-bound/insensitive)
    im = ax.imshow(mat, aspect='auto', cmap='RdYlBu', vmin=vmin, vmax=vmax, origin='lower')
    ax.set_xticks(range(len(xlabels))); ax.set_xticklabels(xlabels)
    ax.set_yticks(range(len(ylabels))); ax.set_yticklabels(ylabels)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i,j]):
                ax.text(j,i,f"{mat[i,j]:.2f}",ha='center',va='center',fontsize=8,
                        color='white' if mat[i,j]>0.7 else 'black')
    return im

def plot_obs2():
    TPs=[1,2,4,8]; BSs=[1,4,16,64,128]
    TOKs=[128,512,2048,8192]
    TOK_labels=["128","512","2K","8K"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # (a) Decode A ratio
    m = np.full((len(BSs),len(TPs)),np.nan)
    for i,bs in enumerate(BSs):
        for j,tp in enumerate(TPs):
            ar,_ = _ratio_d(tp,1024,bs,64); m[i,j]=ar
    im = _heatmap(axes[0,0],m,TPs,BSs,"TP","Batch Size","(a) Decode: A ratio")

    # (b) Decode F ratio
    m = np.full((len(BSs),len(TPs)),np.nan)
    for i,bs in enumerate(BSs):
        for j,tp in enumerate(TPs):
            _,fr = _ratio_d(tp,1024,bs,64); m[i,j]=fr
    _heatmap(axes[0,1],m,TPs,BSs,"TP","Batch Size","(b) Decode: F ratio")

    # (c) Prefill A ratio: TP x BS (il=128)
    m = np.full((len(BSs),len(TPs)),np.nan)
    for i,bs in enumerate(BSs):
        for j,tp in enumerate(TPs):
            ar,_ = _ratio_p(tp, 128, bs)
            if not np.isnan(ar): m[i,j]=ar
    _heatmap(axes[1,0],m,TPs,BSs,"TP","Batch Size","(c) Prefill: A ratio")

    # (d) Prefill F ratio: TP x BS (il=128)
    m = np.full((len(BSs),len(TPs)),np.nan)
    for i,bs in enumerate(BSs):
        for j,tp in enumerate(TPs):
            _,fr = _ratio_p(tp, 128, bs)
            if not np.isnan(fr): m[i,j]=fr
    _heatmap(axes[1,1],m,TPs,BSs,"TP","Batch Size","(d) Prefill: F ratio")

    # Shared colorbar
    fig.subplots_adjust(right=0.82, hspace=0.35, wspace=0.25)
    cbar_ax = fig.add_axes([0.84, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    # ratio definition: vertical, tight to colorbar
    fig.text(0.91, 0.50, "ratio = lat(1410) / lat(210)",
             ha='center', va='center', fontsize=10, rotation=90)
    # Top/bottom annotations
    fig.text(0.88, 0.87, "insensitive\n(memory-bound)",
             ha='left', va='bottom', fontsize=7, color='#2166ac')
    fig.text(0.88, 0.13, "sensitive\n(compute-bound)",
             ha='left', va='top', fontsize=7, color='#b2182b')

    fig.savefig(OUT/"obs2_heatmap.pdf")
    fig.savefig(OUT/"obs2_heatmap.png")
    print("Obs2 saved."); plt.close(fig)
# ══ Obs 3 ══════════════════════════════════════════════════
def _energy_subplot(ax, tp, il, ol, bs, title):
    """Plot A and F energy curves with min annotations and AFlex vs Unified."""
    a_e = [D.get((tp,il,ol,f,bs),(0,0,np.nan,0))[2] for f in FREQS]
    f_e = [D.get((tp,il,ol,f,bs),(0,0,0,np.nan))[3] for f in FREQS]

    ax.plot(FREQS, a_e, 'o-', label="A energy", color="C0", zorder=3)
    ax.plot(FREQS, f_e, 's-', label="F energy", color="C1", zorder=3)

    # Mark min-energy points
    a_mi = int(np.nanargmin(a_e)); f_mi = int(np.nanargmin(f_e))
    ax.annotate(f"A opt: {FREQS[a_mi]}MHz\n{a_e[a_mi]:.0f}mJ",
                xy=(FREQS[a_mi], a_e[a_mi]), fontsize=7,
                xytext=(40, 20), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='C0', lw=1.2),
                color='C0', fontweight='bold')
    ax.annotate(f"F opt: {FREQS[f_mi]}MHz\n{f_e[f_mi]:.0f}mJ",
                xy=(FREQS[f_mi], f_e[f_mi]), fontsize=7,
                xytext=(15, -30), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='C1', lw=1.2),
                color='C1', fontweight='bold')

    ax.set_xlabel("GPU Frequency (MHz)")
    ax.set_ylabel("Energy per step (mJ)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

def _annotate_opt(ax, freqs, energies, label, color, text_offset):
    mi = int(np.nanargmin(energies))
    ax.annotate(f"{label} opt: {freqs[mi]}MHz\n{energies[mi]:.0f}mJ",
                xy=(freqs[mi], energies[mi]), fontsize=7,
                xytext=text_offset, textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2),
                color=color, fontweight='bold')

def plot_obs3():
    # Unified config: tp=8,il=128,bs=1 and tp=4,il=128,bs=1
    # (a)(b) Decode, (c)(d) Prefill — same (tp,il,bs) for direct comparison
    configs = [
        (8, 128, 1, "(a) Decode (tp=8, il=128, bs=1)"),
        (4, 128, 1, "(b) Decode (tp=4, il=128, bs=1)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for col, (tp, il, bs, _) in enumerate(configs):
        ol = 64
        # Decode row
        ax = axes[0, col]
        a_e = [D.get((tp,il,ol,f,bs),(0,0,np.nan,0))[2] for f in FREQS]
        f_e = [D.get((tp,il,ol,f,bs),(0,0,0,np.nan))[3] for f in FREQS]
        ax.plot(FREQS, a_e, 'o-', label="A energy", color="C0", zorder=3)
        ax.plot(FREQS, f_e, 's-', label="F energy", color="C1", zorder=3)
        _annotate_opt(ax, FREQS, a_e, "A", "C0", (40, 15))
        _annotate_opt(ax, FREQS, f_e, "F", "C1", (15, -35))
        ax.set_title(f"({'ab'[col]}) Decode (tp={tp})")
        ax.set_xlabel("GPU Frequency (MHz)")
        ax.set_ylabel("Energy (mJ)")
        ax.legend(fontsize=8, loc='upper left'); ax.grid(True, alpha=0.3)

        # Prefill row
        ax = axes[1, col]
        a_e = [P.get((tp,il,f,bs),(0,0,np.nan,0))[2] for f in FREQS]
        f_e = [P.get((tp,il,f,bs),(0,0,0,np.nan))[3] for f in FREQS]
        ax.plot(FREQS, a_e, 'o-', label="A energy", color="C0", zorder=3)
        ax.plot(FREQS, f_e, 's-', label="F energy", color="C1", zorder=3)
        _annotate_opt(ax, FREQS, a_e, "A", "C0", (40, 15))
        _annotate_opt(ax, FREQS, f_e, "F", "C1", (15, -35))
        ax.set_title(f"({'cd'[col]}) Prefill (tp={tp})")
        ax.set_xlabel("GPU Frequency (MHz)")
        ax.set_ylabel("Energy (mJ)")
        ax.legend(fontsize=8, loc='upper left'); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT/"obs3_energy_vs_freq.pdf")
    fig.savefig(OUT/"obs3_energy_vs_freq.png")
    print("Obs3 saved."); plt.close(fig)

def plot_obs3_prefill():
    pass  # merged into plot_obs3
# ══ Obs 4 ══════════════════════════════════════════════════
def _best_aflex_d(tp, il, ol, bs):
    """Find best (f_A, f_F) that minimizes A_energy(f_A)+F_energy(f_F)."""
    best_e, best_lat = np.inf, np.inf
    for fa in FREQS:
        for ff in FREQS:
            ka = (tp, il, ol, fa, bs); kf = (tp, il, ol, ff, bs)
            if ka not in D or kf not in D: continue
            e = D[ka][2] + D[kf][3]
            lat = max(D[ka][0], D[kf][1])
            if e < best_e:
                best_e, best_lat = e, lat
                best_fa, best_ff = fa, ff
    return best_e, best_lat, best_fa, best_ff

def _saving_vs_slo_decode(tp, il, ol, bs, slo_mults):
    """For each SLO multiplier, compute AFlex vs Unified saving %."""
    k14 = (tp, il, ol, 1410, bs)
    if k14 not in D: return None
    base_lat = max(D[k14][0], D[k14][1])
    results = []
    for sm in slo_mults:
        slo = base_lat * sm
        best_ue = np.inf
        for f in FREQS:
            k = (tp, il, ol, f, bs)
            if k not in D: continue
            if max(D[k][0], D[k][1]) <= slo:
                e = D[k][2] + D[k][3]
                if e < best_ue: best_ue = e
        best_ae = np.inf
        for fa in FREQS:
            for ff in FREQS:
                ka = (tp,il,ol,fa,bs); kf = (tp,il,ol,ff,bs)
                if ka not in D or kf not in D: continue
                if max(D[ka][0], D[kf][1]) <= slo:
                    e = D[ka][2] + D[kf][3]
                    if e < best_ae: best_ae = e
        if best_ue < np.inf and best_ae < np.inf and best_ue > 0:
            results.append((best_ue - best_ae) / best_ue * 100)
        else:
            results.append(0)
    return results

def _saving_vs_slo_prefill(tp, il, bs, slo_mults):
    k14 = (tp, il, 1410, bs)
    if k14 not in P: return None
    base_lat = max(P[k14][0], P[k14][1])
    results = []
    for sm in slo_mults:
        slo = base_lat * sm
        best_ue = np.inf
        for f in FREQS:
            k = (tp, il, f, bs)
            if k not in P: continue
            if max(P[k][0], P[k][1]) <= slo:
                e = P[k][2] + P[k][3]
                if e < best_ue: best_ue = e
        best_ae = np.inf
        for fa in FREQS:
            for ff in FREQS:
                ka = (tp,il,fa,bs); kf = (tp,il,ff,bs)
                if ka not in P or kf not in P: continue
                if max(P[ka][0], P[kf][1]) <= slo:
                    e = P[ka][2] + P[kf][3]
                    if e < best_ae: best_ae = e
        if best_ue < np.inf and best_ae < np.inf and best_ue > 0:
            results.append((best_ue - best_ae) / best_ue * 100)
        else:
            results.append(0)
    return results

def plot_obs4():
    slo_mults = [1.0, 1.05, 1.1, 1.2, 1.5, 2.0]
    slo_labels = ["1.0", "1.05", "1.1", "1.2", "1.5", "2.0"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # (a) Decode saving vs SLO — sweet spot (max, p90, p75)
    ax = axes[0]; ax.set_title("(a) Decode")
    for tp, mk, clr in [(1,'o','C0'),(2,'s','C1'),(4,'^','C2'),(8,'D','C3')]:
        all_savings = [[] for _ in slo_mults]
        for il in [128, 256, 512, 1024, 2048, 4096]:
            for ol in [64, 256, 1024, 4096]:
                for bs in [1, 4, 16, 64, 128]:
                    r = _saving_vs_slo_decode(tp, il, ol, bs, slo_mults)
                    if r:
                        for i, s in enumerate(r):
                            if s > 0: all_savings[i].append(s)
        maxs = [np.max(v) if v else 0 for v in all_savings]
        ax.plot(slo_labels, maxs, marker=mk, ls='-', label=f"tp={tp}", color=clr)
    ax.set_xlabel("SLO Multiplier"); ax.set_ylabel("Max Energy Saving (%)")
    ax.legend(fontsize=8, loc='upper right'); ax.grid(True, alpha=0.3)

    # (b) Prefill saving vs SLO — sweet spot
    ax = axes[1]; ax.set_title("(b) Prefill")
    for tp, mk, clr in [(1,'o','C0'),(2,'s','C1'),(4,'^','C2'),(8,'D','C3')]:
        all_savings = [[] for _ in slo_mults]
        for il in [128, 256, 512, 1024, 2048]:
            for bs in [1, 2, 4, 8, 16]:
                r = _saving_vs_slo_prefill(tp, il, bs, slo_mults)
                if r:
                    for i, s in enumerate(r):
                        if s > 0: all_savings[i].append(s)
        maxs = [np.max(v) if v else 0 for v in all_savings]
        ax.plot(slo_labels, maxs, marker=mk, ls='-', label=f"tp={tp}", color=clr)
    ax.set_xlabel("SLO Multiplier"); ax.set_ylabel("Max Energy Saving (%)")
    ax.legend(fontsize=8, loc='upper right'); ax.grid(True, alpha=0.3)

    # (c) Energy share: Prefill vs Decode per scenario
    ax = axes[2]; ax.set_title("(c) Energy Breakdown")
    scenarios = {
        "Chatbot\nil=128\nol=1024": (128, 1024),
        "QA\nil=512\nol=256": (512, 256),
        "RAG\nil=2048\nol=64": (2048, 64),
        "Summary\nil=4096\nol=64": (4096, 64),
    }
    names, p_shares, d_shares = [], [], []
    for name, (il, ol) in scenarios.items():
        kp = (4, il, 1410, 1)
        kd = (4, il, ol, 1410, 1)
        if kp in P and kd in D:
            pe = P[kp][2] + P[kp][3]
            de = (D[kd][2] + D[kd][3]) * ol
            total = pe + de
            p_shares.append(pe / total * 100)
            d_shares.append(de / total * 100)
        else:
            p_shares.append(0); d_shares.append(0)
        names.append(name)
    x = np.arange(len(names))
    b1 = ax.bar(x, d_shares, label="Decode", color="#4ECDC4")
    b2 = ax.bar(x, p_shares, bottom=d_shares, label="Prefill", color="#FF6B6B")
    # Label percentages
    for i in range(len(names)):
        ax.text(i, d_shares[i]/2, f"{d_shares[i]:.1f}%", ha='center', va='center', fontsize=7, color='white')
        ax.text(i, d_shares[i]+p_shares[i]/2, f"{p_shares[i]:.1f}%", ha='center', va='center', fontsize=7, color='white')
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("Energy Share (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, loc='upper right'); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(OUT/"obs4_saving.pdf")
    fig.savefig(OUT/"obs4_saving.png")
    print("Obs4 saved."); plt.close(fig)

# ══ Obs 5 ══════════════════════════════════════════════════
def plot_obs5():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # (a) F/A ratio distribution: box plot, all configs @1410MHz
    ax = axes[0]; ax.set_title("(a) F/A Latency Ratio Distribution")
    TPs = [1, 2, 4, 8]
    positions_p, positions_d = [], []
    data_p, data_d = [], []
    for i, tp in enumerate(TPs):
        # Prefill
        ratios = []
        for il in [128,256,512,1024,2048,4096]:
            for bs in [1,2,4,8,16,32,64,128]:
                k = (tp, il, 1410, bs)
                if k in P and P[k][0] > 0:
                    ratios.append(P[k][1]/P[k][0])
        data_p.append(ratios)
        positions_p.append(i*3)
        # Decode
        ratios = []
        for il in [128,256,512,1024,2048,4096]:
            for ol in [64,256,1024,4096]:
                for bs in [1,4,16,64,128]:
                    k = (tp, il, ol, 1410, bs)
                    if k in D and D[k][0] > 0:
                        ratios.append(D[k][1]/D[k][0])
        data_d.append(ratios)
        positions_d.append(i*3 + 1)

    bp1 = ax.boxplot(data_p, positions=positions_p, widths=0.7,
                     patch_artist=True, boxprops=dict(facecolor='#FF6B6B', alpha=0.7),
                     medianprops=dict(color='black', lw=1.5))
    bp2 = ax.boxplot(data_d, positions=positions_d, widths=0.7,
                     patch_artist=True, boxprops=dict(facecolor='#4ECDC4', alpha=0.7),
                     medianprops=dict(color='black', lw=1.5))
    ax.axhline(1.0, color='gray', ls='--', lw=1.2, label='F = A')
    ax.set_xticks([i*3 + 0.5 for i in range(len(TPs))])
    ax.set_xticklabels([f"tp={tp}" for tp in TPs])
    ax.set_ylabel("F/A Latency Ratio")
    ax.legend([bp1['boxes'][0], bp2['boxes'][0], ax.lines[-1]],
              ['Prefill', 'Decode', 'F = A'], fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    # Annotate regions
    ax.text(0.98, 0.85, "F > A\n(F bottleneck)", transform=ax.transAxes,
            fontsize=7, ha='right', va='top', color='#FF6B6B', fontstyle='italic')
    ax.text(0.98, 0.15, "A > F\n(A bottleneck)", transform=ax.transAxes,
            fontsize=7, ha='right', va='bottom', color='#4ECDC4', fontstyle='italic')

    # (b) Required A:F GPU ratio shifts with load intensity
    axes[1].remove()
    ax_b = fig.add_axes([0.58, 0.15, 0.38, 0.75])
    ax_b.set_title("(b) Required F/A GPU Ratio vs Load")

    BSs_b = [1, 2, 4, 8, 16]

    for tp, clr, mk in [(2, 'C1', 's'), (4, 'C2', '^'), (8, 'C3', 'D')]:
        p_ratios, d_ratios, c_ratios = [], [], []
        for bs in BSs_b:
            kp = (tp, 1024, 1410, bs)
            kd = (tp, 1024, 64, 1410, bs)
            if kp in P and kd in D:
                p_ratios.append(P[kp][1] / P[kp][0])
                d_ratios.append(D[kd][1] / D[kd][0])
                total_a = P[kp][0] + 64 * D[kd][0]
                total_f = P[kp][1] + 64 * D[kd][1]
                c_ratios.append(total_f / total_a)
            else:
                p_ratios.append(np.nan)
                d_ratios.append(np.nan)
                c_ratios.append(np.nan)
        ax_b.plot(BSs_b, p_ratios, marker=mk, ls='-', color=clr,
                  label=f"tp={tp} Prefill")
        ax_b.plot(BSs_b, d_ratios, marker=mk, ls='--', color=clr, alpha=0.5,
                  label=f"tp={tp} Decode")

    ax_b.axhline(1.0, color='gray', ls=':', lw=1.2)
    ax_b.text(BSs_b[-1]*1.1, 1.05, "F = A", fontsize=8, color='gray', va='bottom')

    # Annotate regions
    ax_b.text(0.97, 0.92, "F bottleneck\n(need more F GPUs)", transform=ax_b.transAxes,
              fontsize=7, ha='right', va='top', color='#FF6B6B', fontstyle='italic')
    ax_b.text(0.97, 0.08, "A bottleneck\n(need more A GPUs)", transform=ax_b.transAxes,
              fontsize=7, ha='right', va='bottom', color='#4ECDC4', fontstyle='italic')

    ax_b.set_xlabel("Batch Size (load intensity →)")
    ax_b.set_ylabel("F/A Latency Ratio")
    ax_b.set_xscale('log', base=2)
    ax_b.set_xticks(BSs_b); ax_b.set_xticklabels(BSs_b)
    ax_b.legend(fontsize=6.5, loc='center left')
    ax_b.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT/"obs5_adaptive.pdf")
    fig.savefig(OUT/"obs5_adaptive.png")
    print("Obs5 saved."); plt.close(fig)

# ══ Main ═══════════════════════════════════════════════════
if __name__ == "__main__":
    plot_obs1()
    plot_obs2()
    plot_obs3()
    plot_obs3_prefill()
    plot_obs4()
    plot_obs5()
    print(f"All figures saved to {OUT}")
