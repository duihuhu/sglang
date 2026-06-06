#!/usr/bin/env python3
"""PD+AF M=3 QPS=2 — 3-row pipeline Gantt from iteration 500 data.

  DA  [A0][A1][A2][F0][A0][F1][A1][F2][A2]...
  DF       [F0].....[F1].....[F2]....
  Comm: DA→DF arrows, DF→DA arrows

  Correct per-micro-batch interleaving: F(l-1,m) | A(l,m)
"""

import re, os
import numpy as np
from collections import defaultdict

DA_LOG = "/workspace/sglang/benchmark/test_motivation/AzurePublicDataset/pipeline_analysis/throughput_logs/qps_sweep_m3_da.log"
DF_LOG = "/workspace/sglang/benchmark/test_motivation/AzurePublicDataset/pipeline_analysis/throughput_logs/qps_sweep_m3_df.log"
OUT = "/workspace/sglang/benchmark/test_motivation/AzurePublicDataset/pipeline_analysis/pipeline_dstage_gantt.png"

NL, M = 64, 3

# ═══════════════════════════════════════════════════════════════════════
# 1. Parse iteration 500 AFD_BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════
def parse_brk(path, perspective, idx):
    pat = re.compile(
        r"\[AFD_BREAKDOWN\]\s+perspective=(\w+)\s+layers=64\s+M=(\d+)\s+total=([\d.]+)ms\s+"
        r"\|\s+A_stage=([\d.]+)ms\s+\(prep_attn=([\d.]+)ms\s+attn=([\d.]+)ms\s+prep_mlp=([\d.]+)ms\)\s+"
        r"\|\s+F_stage=([\d.]+)ms\s+\(mlp=([\d.]+)ms\s+postprocess=([\d.]+)ms\)"
    )
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m and m.group(1) == perspective and int(m.group(2)) == M:
                if idx == 0:
                    g = m.groups()
                    return {
                        "total": float(g[2]), "A_total": float(g[3]),
                        "A_prep_attn": float(g[4]), "A_attn": float(g[5]),
                        "A_prep_mlp": float(g[6]), "F_total": float(g[7]),
                        "F_mlp": float(g[8]), "F_post": float(g[9]),
                    }
                idx -= 1
    raise ValueError(f"No entry at index {idx}")

da = parse_brk(DA_LOG, "attn", 499)  # 0-indexed: 499 = 500th entry
df = parse_brk(DF_LOG, "ffn", 499)

print(f"Iter 500 DA: total={da['total']}ms A={da['A_total']}ms F={da['F_total']}ms")
print(f"  A: prep_attn={da['A_prep_attn']} attn={da['A_attn']} prep_mlp={da['A_prep_mlp']}")
print(f"  F: proxy={da['F_mlp']} postprocess={da['F_post']}")
print(f"Iter 500 DF: total={df['total']}ms F_mlp={df['F_mlp']}ms")

# ═══════════════════════════════════════════════════════════════════════
# 2. Extract per-layer batch sizes (AFD_DBG around iteration 500)
# ═══════════════════════════════════════════════════════════════════════
dbg_pat = re.compile(r"\[AFD_DBG\]\s+L\s*(\d+)\s+batch=\s*(\d+)")

with open(DA_LOG) as f:
    lines = f.readlines()

brk_count = 0
brk_line = None
for i, line in enumerate(lines):
    if "AFD_BREAKDOWN" in line and f"M={M}" in line:
        if brk_count == 499:  # 0-indexed: 499 = 500th entry
            brk_line = i
            break
        brk_count += 1

prev_brk = None
for i in range(brk_line - 1, -1, -1):
    if "AFD_BREAKDOWN" in lines[i]:
        prev_brk = i
        break

mb_batches = {}
for i in range(prev_brk + 1, brk_line):
    m = dbg_pat.search(lines[i])
    if m:
        lyr = int(m.group(1))
        batch = int(m.group(2))
        key = (lyr, len([k for k in mb_batches if k[0] == lyr]))
        mb_batches[key] = batch

lyr_batches = defaultdict(list)
for (lyr, mb_idx), batch in sorted(mb_batches.items()):
    lyr_batches[lyr].append(batch)

batch_mb = lyr_batches[0]
print(f"Batch distribution: {batch_mb}")

# ═══════════════════════════════════════════════════════════════════════
# 3. Per-micro-batch timing
# ═══════════════════════════════════════════════════════════════════════
A_lyr = da["A_total"] / NL * 1000
PA_lyr = da["A_prep_attn"] / NL * 1000
AT_lyr = da["A_attn"] / NL * 1000
PM_lyr = da["A_prep_mlp"] / NL * 1000

F_lyr = da["F_total"] / NL * 1000
FP_lyr = da["F_mlp"] / NL * 1000
FW_lyr = da["F_post"] / NL * 1000

F_mlp_pmb = df["F_mlp"] / (NL * M) * 1000  # DF FFN per mb

total_b = sum(batch_mb)
weights = [b / total_b for b in batch_mb]

A_mb = []
for mb in range(M):
    w = weights[mb]
    A_mb.append({
        "batch": batch_mb[mb], "total": A_lyr * w,
        "prep_attn": PA_lyr * w, "attn": AT_lyr * w, "prep_mlp": PM_lyr * w,
    })

F_pmb = F_lyr / M      # F_stage per mb on DA
F_proxy = FP_lyr / M
F_wait = FW_lyr / M

print(f"Per-mb A: {A_mb[0]['total']:.0f}/{A_mb[1]['total']:.0f}/{A_mb[2]['total']:.0f}us")
print(f"Per-mb F on DA: {F_pmb:.0f}us (proxy={F_proxy:.0f}+recv={F_wait:.0f})")
print(f"DF FFN per mb: {F_mlp_pmb:.0f}us")

# ═══════════════════════════════════════════════════════════════════════
# 4. Build DA timeline (attn_stage schedule)
# ═══════════════════════════════════════════════════════════════════════
# attn_stage: for (layer_id 0..64, m 0..2):
#   if layer_id > 0: F(layer_id-1, m)
#   if layer_id < 64: A(layer_id, m)

da_events = []  # (start_us, dur_us, stage, layer, mb)
t = 0.0
for layer_id in range(NL + 1):
    for mb in range(M):
        if layer_id > 0:
            da_events.append((t, F_pmb, "F", layer_id - 1, mb))
            t += F_pmb
        if layer_id < NL:
            dur = A_mb[mb]["total"]
            da_events.append((t, dur, "A", layer_id, mb))
            t += dur

print(f"DA total: {t:.0f}us (expected {da['total']}ms)")

# ═══════════════════════════════════════════════════════════════════════
# 5. Build DF timeline (ffn_stage schedule)
# ═══════════════════════════════════════════════════════════════════════
# When does each DA A(l,m) send data to DF? End of A stage.
da_send = {}
for ev in da_events:
    st, dur, stage, layer, mb = ev
    if stage == "A":
        da_send[(layer, mb)] = st + dur

df_events = []  # (start_us, dur_us, "FFN"|"IDLE", layer, mb)
df_busy = 0.0
df_ready = 0.0

for layer_id in range(NL):
    for mb in range(M):
        send_t = da_send[(layer_id, mb)]
        # DF recv_wait until data arrives
        recv_done = max(df_ready, send_t)
        idle = recv_done - df_ready
        if idle > 1:
            df_events.append((df_ready, idle, "IDLE", layer_id, mb))
            df_busy += idle  # counts as DF time even if idle
        # FFN compute
        df_events.append((recv_done, F_mlp_pmb, "FFN", layer_id, mb))
        df_busy += F_mlp_pmb
        df_ready = recv_done + F_mlp_pmb

print(f"DF FFN: {NL*M*F_mlp_pmb/1000:.1f}ms, DF total: {df_busy/1000:.1f}ms")

# ═══════════════════════════════════════════════════════════════════════
# 6. Plot
# ═══════════════════════════════════════════════════════════════════════
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(15, 4.5))

# Colors
C_prep  = "#8DB0D9"
C_attn  = "#4A7FB5"
C_post  = "#5D8A7A"  # post-attn GPU work: allreduce + output_norm (NOT send!)
C_proxy = "#B0A0D0"
C_recv  = "#C44E52"
C_ffn   = "#E28A3F"
C_idle  = "#F0E0C0"

bar_h = 0.30
Y_DA = 2
Y_DF = 0

N_SHOW = 4

# X range
t_max = 0
for ev in da_events:
    st, dur, stage, layer, mb = ev
    if layer < N_SHOW:
        t_max = max(t_max, st + dur)

# Filter
da_show = [ev for ev in da_events if ev[3] < N_SHOW]
df_show = [ev for ev in df_events if ev[3] < N_SHOW]

# ── DA row ──
for st, dur, stage, layer, mb in da_show:
    if stage == "A":
        a = A_mb[mb]
        x = st / 1000
        # prep_attn
        ax.barh(Y_DA, a["prep_attn"]/1000, left=x, height=bar_h,
                color=C_prep, edgecolor="white", linewidth=0.3)
        # attn
        ax.barh(Y_DA, a["attn"]/1000, left=x + a["prep_attn"]/1000, height=bar_h,
                color=C_attn, edgecolor="white", linewidth=0.3)
        # prep_mlp: allreduce + output_norm (GPU compute, NOT blocking send!)
        ax.barh(Y_DA, a["prep_mlp"]/1000,
                left=x + (a["prep_attn"] + a["attn"])/1000, height=bar_h,
                color=C_post, edgecolor="white", linewidth=0.3)
        # send_async marker (non-blocking, at the very end of prep_mlp)
        send_x = x + (a["prep_attn"] + a["attn"] + a["prep_mlp"])/1000
        ax.plot(send_x, Y_DA, marker="v", color="#3DAA55", markersize=3, zorder=5)
        if layer == 0:
            cx = x + (a["prep_attn"] + a["attn"]/2)/1000
            ax.text(cx, Y_DA, f"A{mb}", ha="center", va="center",
                    fontsize=5, color="white", fontweight="bold")
    else:  # "F"
        x = st / 1000
        # proxy_mlp
        ax.barh(Y_DA, F_proxy/1000, left=x, height=bar_h,
                color=C_proxy, edgecolor="white", linewidth=0.3, alpha=0.8)
        # recv_wait+allreduce
        ax.barh(Y_DA, F_wait/1000, left=x + F_proxy/1000, height=bar_h,
                color=C_recv, edgecolor="white", linewidth=0.3)
        if layer == 0:
            cx = x + (F_proxy + F_wait/2)/1000
            ax.text(cx, Y_DA, "R", ha="center", va="center",
                    fontsize=4.5, color="white", fontweight="bold")

# ── DF row ──
for st, dur, kind, layer, mb in df_show:
    x = st / 1000
    if kind == "FFN":
        ax.barh(Y_DF, dur/1000, left=x, height=bar_h,
                color=C_ffn, edgecolor="white", linewidth=0.3)
        if layer == 0:
            ax.text(x + dur/2000, Y_DF, f"F{mb}",
                    ha="center", va="center", fontsize=5, color="white", fontweight="bold")
    else:  # IDLE
        ax.barh(Y_DF, dur/1000, left=x, height=bar_h,
                color=C_idle, edgecolor="white", linewidth=0.2, alpha=0.7)

# ── Communication arrows ──
# DA→DF sends: end of A(l,m) prep_mlp
for st, dur, stage, layer, mb in da_show:
    if stage == "A":
        send_us = st + A_mb[mb]["prep_attn"] + A_mb[mb]["attn"] + A_mb[mb]["prep_mlp"]
        ax.annotate("",
                    xy=(send_us/1000, Y_DF + bar_h),
                    xytext=(send_us/1000, Y_DA - bar_h),
                    arrowprops=dict(arrowstyle="->", color="#3DAA55", lw=0.8),
                    clip_on=False)

# DF→DA sends: end of DF FFN
for st, dur, kind, layer, mb in df_show:
    if kind == "FFN":
        end_us = st + dur
        ax.annotate("",
                    xy=(end_us/1000, Y_DA - bar_h),
                    xytext=(end_us/1000, Y_DF + bar_h),
                    arrowprops=dict(arrowstyle="->", color=C_recv, lw=0.8),
                    clip_on=False)

# ── Layer separators ──
# Mark the boundary between layers (where A(l,0) starts)
layer_starts = {}
for ev in da_events:
    st, dur, stage, layer, mb = ev
    if stage == "A" and mb == 0 and layer < N_SHOW:
        layer_starts[layer] = st

for lyr in range(N_SHOW):
    if lyr not in layer_starts:
        continue
    ls = layer_starts[lyr]
    # Compute layer end = start of next A(l+1, 0) or F(layer, 2) end
    # A layer spans from A(l,0) start to A(l+1,0) start (or F(l-1,0) start if l>0)
    le = ls + A_mb[0]["total"] + A_mb[1]["total"] + A_mb[2]["total"] + F_pmb * 3
    # Bracket at y=3.0
    ax.plot([ls/1000, le/1000], [3.0, 3.0], color="gray", lw=0.5)
    ax.plot([ls/1000, ls/1000], [2.95, 3.0], color="gray", lw=0.5)
    ax.plot([le/1000, le/1000], [2.95, 3.0], color="gray", lw=0.5)
    ax.text((ls+le)/2000, 3.12, f"Layer {lyr}",
            ha="center", va="bottom", fontsize=7, color="gray")

# ── Row labels ──
ax.text(-0.015, Y_DA, "DA\n(Attn)", ha="right", va="center", fontsize=10, fontweight="bold",
        transform=ax.transData)
ax.text(-0.015, Y_DF, "DF\n(FFN)", ha="right", va="center", fontsize=10, fontweight="bold",
        transform=ax.transData)

# ── Axes ──
ax.set_xlabel("Time (ms)", fontsize=10)
ax.set_xlim(0, t_max/1000 * 1.03)
ax.set_ylim(-0.4, 3.4)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", alpha=0.15, lw=0.3)

# ── Legend ──
legend = [
    mpatches.Patch(color=C_attn, label="Attention"),
    mpatches.Patch(color=C_prep, label="prep_attn\n(layernorm+TP scatter)"),
    mpatches.Patch(color=C_post, label="allreduce+norm\n(GPU compute, NOT send)"),
    mpatches.Patch(color=C_proxy, label="proxy MLP\n(no-op on DA)"),
    mpatches.Patch(color=C_recv, label="recv_wait\n+ allreduce"),
    mpatches.Patch(color=C_ffn, label="FFN (DF GPU)"),
    mpatches.Patch(color=C_idle, label="DF idle\n(wait data)"),
    plt.Line2D([0],[0], marker="v", color="#3DAA55", lw=0, label="send_async\n(non-blocking)"),
    plt.Line2D([0],[0], color=C_recv, lw=1.5, label="DF→DA"),
]
ax.legend(handles=legend, fontsize=5.5, loc="upper left", ncol=5,
          bbox_to_anchor=(0, 1.02), handleheight=1.5)

# ── Title ──
per_layer = (A_lyr + F_lyr) / 1000
ax.set_title(
    f"PD+AF M=3 Decode Pipeline — Iteration 500  (batch={batch_mb}, QPS=2)\n"
    f"DA A={da['A_total']:.1f}ms + F={da['F_total']:.1f}ms = {da['total']}ms  |  "
    f"Per-layer: {per_layer:.2f}ms  |  "
    f"prep_mlp = allreduce+norm (GPU), send_async is non-blocking (▼ marker)",
    fontsize=9, fontweight="bold", pad=10
)

fig.subplots_adjust(left=0.07, right=0.95, top=0.77, bottom=0.12)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"\nSaved {OUT}")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════
# 7. Overlap analysis
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  PIPELINE ANALYSIS")
print("="*65)
print(f"""
  DA per layer: A={A_lyr:.0f}us  F={F_lyr:.0f}us  total={A_lyr+F_lyr:.0f}us
    ├─ prep_attn: {PA_lyr:.0f}us (layernorm + TP scatter)
    ├─ attn: {AT_lyr:.0f}us (attention kernel)
    ├─ prep_mlp: {PM_lyr:.0f}us (allreduce + output_norm ← GPU compute!)
    │                    send_async within prep_mlp is NON-BLOCKING (~2us CPU,
    │                    starts background thread, does NOT block next A-stage)
    └─ F_stage: {F_lyr:.0f}us (proxy_mlp + recv_wait+allreduce)

  DF per mb: FFN={F_mlp_pmb:.0f}us
  DF idle (waiting for DA data): ~{df_busy - NL*M*F_mlp_pmb:.0f}us

  Why A(0,1) waits for A(0,0)'s prep_mlp:
  - prep_mlp is allreduce+output_norm GPU kernel on the default CUDA stream
  - All kernels on the same stream execute sequentially (CUDA guarantee)
  - The network send (send_async) is a tiny CPU-side call that returns
    immediately — it does NOT block the pipeline.
  - BUT the GPU allreduce + norm kernels MUST complete before A(0,1)'s
    prep_attn can launch on the same stream.

  Can we avoid this wait?
  - The data IS independent (mb0 vs mb1 have different tensors)
  - We'd need separate CUDA streams per micro-batch + multi-stream scheduler
  - That's a significant refactoring of the pipeline loop
  - For now, R4 overlap already hides network transfer behind compute
""")
