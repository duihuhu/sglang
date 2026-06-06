#!/usr/bin/env python3
"""Draw the full Attn L(N) -> Attn L(N+1) cycle breakdown with all sub-components."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

fig, ax = plt.subplots(1, 1, figsize=(24, 12))
fig.patch.set_facecolor('white')
ax.set_xlim(-50, 2200)
ax.set_ylim(-1, 10)
ax.set_xlabel('Time (us)', fontsize=11)
ax.set_title('Full Cycle Breakdown: DA Attn L(N) -> DA Attn L(N+1)\n'
             'Qwen3-32B, tp=1, M=1, decode, ~16 tokens', fontsize=13, fontweight='bold')

# Y positions for each row
Y_DA_COMPUTE = 8.5   # DA GPU compute stream
Y_DA_DAEMON = 6.5    # DA daemon thread
Y_WIRE_DA_DF = 5.0   # Wire DA->DF
Y_DF_COMPUTE = 3.5   # DF GPU compute
Y_WIRE_DF_DA = 2.0   # Wire DF->DA
Y_DA_RECV = 0.5      # DA recv

bar_h = 0.8

# Colors
C_PREP_ATTN = '#5dade2'
C_ATTN = '#2e86c1'
C_PREP_MLP = '#f4d03f'
C_SEND_ASYNC = '#e67e22'
C_PROXY_MLP = '#d5dbdb'
C_POSTPROCESS = '#e74c3c'
C_EVENT_SYNC = '#8e44ad'
C_DISPATCH = '#f39c12'
C_UCX_META = '#2ecc71'
C_UCX_DATA = '#e74c3c'
C_DF_RECV = '#f39c12'
C_DF_MLP = '#27ae60'
C_DF_AR = '#3498db'
C_WIRE = '#95a5a6'

# === DA COMPUTE STREAM ===
# [1] prep_attn: 53us
x = 0
ax.barh(Y_DA_COMPUTE, 53, left=x, height=bar_h, color=C_PREP_ATTN, edgecolor='white')
ax.text(x + 26, Y_DA_COMPUTE, '53', ha='center', va='center', fontsize=7, fontweight='bold')
x += 53

# [2] attn: 419us
ax.barh(Y_DA_COMPUTE, 419, left=x, height=bar_h, color=C_ATTN, edgecolor='white')
ax.text(x + 209, Y_DA_COMPUTE, 'Attn\n419us', ha='center', va='center', fontsize=8, fontweight='bold', color='white')
x += 419

# [3] prep_mlp: 152us (AllReduce + norm)
ax.barh(Y_DA_COMPUTE, 152, left=x, height=bar_h, color=C_PREP_MLP, edgecolor='white')
ax.text(x + 76, Y_DA_COMPUTE, 'AllReduce+Norm\n152us', ha='center', va='center', fontsize=7, fontweight='bold')
prep_mlp_end = x + 152
x += 152

# [4] send_async main thread cost: 177us (overlaps slightly)
ax.barh(Y_DA_COMPUTE, 177, left=x, height=bar_h*0.6, color=C_SEND_ASYNC, edgecolor='white')
ax.text(x + 88, Y_DA_COMPUTE - 0.05, 'send_async\n177us', ha='center', va='center', fontsize=6)
send_async_start = x
x += 177

# [5] proxy MLP: 128us
ax.barh(Y_DA_COMPUTE, 128, left=x, height=bar_h, color=C_PROXY_MLP, edgecolor='white')
ax.text(x + 64, Y_DA_COMPUTE, 'proxy\n128us', ha='center', va='center', fontsize=7)
x += 128

# [6] postprocess (recv_wait): 1253us - BOTTLENECK
ax.barh(Y_DA_COMPUTE, 1253, left=x, height=bar_h, color=C_POSTPROCESS, edgecolor='white', alpha=0.7)
ax.text(x + 626, Y_DA_COMPUTE, 'BLOCKED: recv_wait = 1253us', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
postprocess_start = x
x += 1253

# Mark cycle end
cycle_end = x
ax.axvline(cycle_end, color='black', linestyle='--', alpha=0.5, linewidth=1)
ax.text(cycle_end + 10, Y_DA_COMPUTE + 0.5, f'L(N+1) starts\ncycle={cycle_end:.0f}us', fontsize=8, va='bottom')

# === DA DAEMON THREAD ===
# Starts after send_async fires (at prep_mlp_end)
daemon_start = prep_mlp_end

# [a] event.synchronize: 151us
dx = daemon_start
ax.barh(Y_DA_DAEMON, 151, left=dx, height=bar_h, color=C_EVENT_SYNC, edgecolor='white')
ax.text(dx + 75, Y_DA_DAEMON, 'event.sync\n151us', ha='center', va='center', fontsize=7, color='white')
dx += 151

# [b] cross-thread dispatch: 39us
ax.barh(Y_DA_DAEMON, 39, left=dx, height=bar_h, color=C_DISPATCH, edgecolor='white')
ax.text(dx + 19, Y_DA_DAEMON, '39', ha='center', va='center', fontsize=6)
dx += 39

# [c] send(meta): 37us
ax.barh(Y_DA_DAEMON, 37, left=dx, height=bar_h, color=C_UCX_META, edgecolor='white')
ax.text(dx + 18, Y_DA_DAEMON, '37', ha='center', va='center', fontsize=6)
dx += 37

# [d] send(tensor): 108us
ax.barh(Y_DA_DAEMON, 108, left=dx, height=bar_h, color=C_UCX_DATA, edgecolor='white')
ax.text(dx + 54, Y_DA_DAEMON, 'NIC DMA+RDMA\n108us', ha='center', va='center', fontsize=7, color='white')
da_send_done = dx + 108
dx += 108

# Arrow from daemon send done to DF recv
ax.annotate('', xy=(da_send_done + 50, Y_DF_COMPUTE + bar_h/2),
            xytext=(da_send_done, Y_DA_DAEMON - bar_h/2),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, connectionstyle='arc3,rad=0.2'))

# === WIRE DA->DF (network flight) ===
# Essentially free (negative time), but show as thin bar
wire_da_df_start = da_send_done
wire_da_df_dur = 50  # ~0 but show something
ax.barh(Y_WIRE_DA_DF, wire_da_df_dur, left=wire_da_df_start, height=0.3, color=C_WIRE, edgecolor='white')
ax.text(wire_da_df_start + 25, Y_WIRE_DA_DF, 'wire ~0us\n(pipelined)', ha='center', va='center', fontsize=6, color='gray')

# === DF COMPUTE ===
df_start = da_send_done + 50  # DF gets data shortly after DA sends

# [f] recv + decode: 238us
ax.barh(Y_DF_COMPUTE, 238, left=df_start, height=bar_h, color=C_DF_RECV, edgecolor='white')
ax.text(df_start + 119, Y_DF_COMPUTE, 'recv+decode\n238us', ha='center', va='center', fontsize=7)
df_x = df_start + 238

# [g] MLP kernel: 555us
ax.barh(Y_DF_COMPUTE, 555, left=df_x, height=bar_h, color=C_DF_MLP, edgecolor='white')
ax.text(df_x + 277, Y_DF_COMPUTE, 'DF MLP Kernel\n555us', ha='center', va='center', fontsize=8, fontweight='bold', color='white')
df_x += 555

# [h] AllReduce + send: 150us (estimated)
ax.barh(Y_DF_COMPUTE, 150, left=df_x, height=bar_h, color=C_DF_AR, edgecolor='white')
ax.text(df_x + 75, Y_DF_COMPUTE, 'AR+send\n150us', ha='center', va='center', fontsize=7, color='white')
df_send_start = df_x + 150
df_x += 150

# === WIRE DF->DA ===
# DF daemon send: similar 344us path
df_daemon_start = df_x
ax.barh(Y_WIRE_DF_DA, 151, left=df_daemon_start, height=0.6, color=C_EVENT_SYNC, edgecolor='white', alpha=0.7)
ax.text(df_daemon_start + 75, Y_WIRE_DF_DA, 'event.sync 151', ha='center', va='center', fontsize=6)
df_daemon_start += 151
ax.barh(Y_WIRE_DF_DA, 39, left=df_daemon_start, height=0.6, color=C_DISPATCH, edgecolor='white', alpha=0.7)
df_daemon_start += 39
ax.barh(Y_WIRE_DF_DA, 37, left=df_daemon_start, height=0.6, color=C_UCX_META, edgecolor='white', alpha=0.7)
df_daemon_start += 37
ax.barh(Y_WIRE_DF_DA, 108, left=df_daemon_start, height=0.6, color=C_UCX_DATA, edgecolor='white', alpha=0.7)
ax.text(df_daemon_start + 54, Y_WIRE_DF_DA, 'DF send 108', ha='center', va='center', fontsize=6)
df_data_arrives_da = df_daemon_start + 108

# Arrow from DF send to DA recv
ax.annotate('', xy=(df_data_arrives_da, Y_DA_RECV + bar_h/2),
            xytext=(df_daemon_start + 108, Y_WIRE_DF_DA - 0.3),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

# === DA RECV ===
ax.barh(Y_DA_RECV, 50, left=df_data_arrives_da, height=0.5, color=C_PREP_ATTN, edgecolor='white')
ax.text(df_data_arrives_da + 25, Y_DA_RECV, 'DA recv\n50us', ha='center', va='center', fontsize=6)

# Connect DA recv to postprocess unblock
ax.annotate('', xy=(postprocess_start + 1253, Y_DA_COMPUTE - bar_h/2),
            xytext=(df_data_arrives_da + 50, Y_DA_RECV + 0.3),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5, linestyle='dashed'))

# === ROW LABELS ===
ax.text(-45, Y_DA_COMPUTE, 'DA GPU\ncompute', ha='right', va='center', fontsize=9, fontweight='bold')
ax.text(-45, Y_DA_DAEMON, 'DA daemon\nthread', ha='right', va='center', fontsize=9, fontweight='bold')
ax.text(-45, Y_WIRE_DA_DF, 'Wire\nDA->DF', ha='right', va='center', fontsize=8, color='gray')
ax.text(-45, Y_DF_COMPUTE, 'DF GPU\ncompute', ha='right', va='center', fontsize=9, fontweight='bold')
ax.text(-45, Y_WIRE_DF_DA, 'Wire\nDF->DA', ha='right', va='center', fontsize=8, color='gray')
ax.text(-45, Y_DA_RECV, 'DA recv', ha='right', va='center', fontsize=8, color='gray')

# === LEGEND ===
legend_patches = [
    mpatches.Patch(color=C_PREP_ATTN, label='prep_attn (recv + layernorm)'),
    mpatches.Patch(color=C_ATTN, label='Attention kernel'),
    mpatches.Patch(color=C_PREP_MLP, label='AllReduce + post_attn_norm'),
    mpatches.Patch(color=C_SEND_ASYNC, label='send_async (main thread)'),
    mpatches.Patch(color=C_PROXY_MLP, label='proxy MLP (pass-through)'),
    mpatches.Patch(color=C_POSTPROCESS, label='recv_wait (BLOCKED)'),
    mpatches.Patch(color=C_EVENT_SYNC, label='event.synchronize (GPU wait)'),
    mpatches.Patch(color=C_DISPATCH, label='cross-thread dispatch (39us)'),
    mpatches.Patch(color=C_UCX_META, label='UCX send(meta) [64B]'),
    mpatches.Patch(color=C_UCX_DATA, label='UCX send(tensor) [NIC DMA+RDMA]'),
    mpatches.Patch(color=C_DF_RECV, label='DF recv + decode'),
    mpatches.Patch(color=C_DF_MLP, label='DF MLP kernel'),
    mpatches.Patch(color=C_DF_AR, label='DF AllReduce + send'),
]
ax.legend(handles=legend_patches, loc='upper right', fontsize=7, ncol=2, framealpha=0.9)

ax.set_yticks([])
ax.grid(axis='x', alpha=0.3)
ax.axvline(0, color='black', linewidth=1)

plt.tight_layout()
out_path = os.path.join(_HERE, 'gantt_full_cycle_breakdown.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved to {out_path}')
