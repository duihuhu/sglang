#!/usr/bin/env python3
"""Draw precise timeline for one full layer cycle (Layer 4) using actual timestamps.

All bars are positioned according to real profiling timestamps, not estimates.
DA cycle start (t=0) = DA L3 recv_end = when DA finishes receiving L3 result from DF.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

# === ACTUAL TIMESTAMPS (relative to DA cycle start, in us) ===
# DA side:
DA_L4_SEND_START = 800.0       # DA starts send_async for L4
DA_SENDER_LAUNCHED = 832.0     # daemon thread starts
DA_L4_SEND_END = 964.1         # send_async returns on main thread
DA_L4_RECV_START_F = 1133.1    # DA starts recv_wait for DF result
DA_SENDER_CUDA_SYNC = 1297.1   # daemon: event.synchronize() done
DA_SENDER_UCX_DONE = 1318.1    # daemon: send_tensor_nonblocking() returned
DA_UCX_INNER_START = 1452.1    # bridge: _async_send starts
DA_UCX_INNER_ENCODE = 15.0     # bridge: encode_meta duration
DA_UCX_INNER_SEND_META = 71.5  # bridge: await send(meta) duration
DA_UCX_INNER_SEND_DATA = 320.2 # bridge: await send(tensor) duration
DA_UCX_INNER_END = DA_UCX_INNER_START + 406.7  # = 1858.8
DA_L4_RECV_END_F = 3441.2     # DA recv_end (DF result arrived)
DA_L5_SEND_START = 4208.0     # next layer starts

# DF side (same reference t=0):
DF_RECV_START = -819.8         # DF pre-posted recv (before DA cycle!)
DF_UCX_INNER_RECV_START = -725.8  # bridge: _async_recv starts
DF_UCX_INNER_RECV_META_DUR = 2347.9  # await recv(meta) duration
DF_UCX_INNER_RECV_DATA_DUR = 242.5   # await recv(tensor) duration
DF_UCX_INNER_RECV_END = 1864.5       # bridge: _async_recv ends
DF_RECV_END = 1966.1                 # DF recv_end event
DF_SEND_START = 2385.0               # DF starts send (after MLP compute)
DF_SENDER_LAUNCHED = 2409.2
DF_SEND_END = 2524.2
DF_SENDER_CUDA_SYNC = 2696.0
DF_SENDER_UCX_DONE = 3070.1
DF_UCX_INNER_SEND_START = 2856.0
DF_UCX_INNER_SEND_END = 3295.2

# GPU timing (per-layer averages from CUDA events):
# DA: prep_attn=53us, attn=419us, prep_mlp=152us
# These happen between L3_recv_end (t=0) and L4_send_start (t=800)
# So: prep_attn(0-53) + attn(53-472) + prep_mlp(472-624) + overhead(624-800)
DA_PREP_ATTN = (0, 53)
DA_ATTN = (53, 472)
DA_PREP_MLP = (472, 624)
# Gap 624-800: scheduler overhead / Python dispatch
DA_SEND_ASYNC_MAIN = (800, 964)  # send_async on main thread
DA_PROXY_MLP = (964, 1133)       # proxy MLP until recv_start
DA_RECV_WAIT = (1133, 3441)      # BLOCKED waiting for DF

# DF MLP compute: between recv_end and send_start
DF_MLP_COMPUTE = (1966, 2385)    # 419us of MLP kernel

# === DRAW ===
fig, ax = plt.subplots(1, 1, figsize=(26, 10))
fig.patch.set_facecolor('white')

Y_DA_GPU = 7.0      # DA GPU compute stream
Y_DA_DAEMON = 5.5   # DA daemon thread
Y_DA_BRIDGE = 4.0   # DA UCX bridge thread
Y_DF_BRIDGE = 2.5   # DF UCX bridge thread (recv)
Y_DF_GPU = 1.0      # DF GPU compute + send
Y_DF_DAEMON = -0.5  # DF daemon thread (send back)

bar_h = 0.7
thin_h = 0.5

# Colors
C = {
    'prep_attn': '#5dade2',
    'attn': '#2e86c1',
    'prep_mlp': '#f4d03f',
    'send_main': '#e67e22',
    'proxy': '#d5dbdb',
    'recv_wait': '#e74c3c',
    'event_sync': '#8e44ad',
    'dispatch': '#f39c12',
    'ucx_meta': '#2ecc71',
    'ucx_data': '#c0392b',
    'df_recv_wait': '#f5b041',
    'df_recv_data': '#9b59b6',
    'df_mlp': '#27ae60',
    'df_send_main': '#e67e22',
    'overhead': '#bdc3c7',
}

def bar(y, start, end, color, label='', fontsize=7, text_color='black'):
    dur = end - start
    ax.barh(y, dur, left=start, height=bar_h, color=color, edgecolor='white', linewidth=0.5)
    if label and dur > 30:
        cx = start + dur/2
        ax.text(cx, y, label, ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color=text_color)

# === DA GPU COMPUTE ===
bar(Y_DA_GPU, DA_PREP_ATTN[0], DA_PREP_ATTN[1], C['prep_attn'], 'prep_attn\n53us', 6)
bar(Y_DA_GPU, DA_ATTN[0], DA_ATTN[1], C['attn'], 'Attention\n419us', 8, 'white')
bar(Y_DA_GPU, DA_PREP_MLP[0], DA_PREP_MLP[1], C['prep_mlp'], 'AllReduce+Norm\n152us', 7)
bar(Y_DA_GPU, 624, 800, C['overhead'], 'sched\n176us', 6)
bar(Y_DA_GPU, DA_SEND_ASYNC_MAIN[0], DA_SEND_ASYNC_MAIN[1], C['send_main'], 'send_async\n164us', 6)
bar(Y_DA_GPU, DA_PROXY_MLP[0], DA_PROXY_MLP[1], C['proxy'], 'proxy\n169us', 6)
bar(Y_DA_GPU, DA_RECV_WAIT[0], DA_RECV_WAIT[1], C['recv_wait'], 'BLOCKED: recv_wait = 2308us', 9, 'white')

# === DA DAEMON THREAD ===
# launched at 832, cuda_sync at 1297, ucx_done at 1318
bar(Y_DA_DAEMON, DA_SENDER_LAUNCHED, DA_SENDER_CUDA_SYNC, C['event_sync'],
    f'event.sync\n{DA_SENDER_CUDA_SYNC - DA_SENDER_LAUNCHED:.0f}us', 7, 'white')
bar(Y_DA_DAEMON, DA_SENDER_CUDA_SYNC, DA_SENDER_UCX_DONE, C['dispatch'],
    f'{DA_SENDER_UCX_DONE - DA_SENDER_CUDA_SYNC:.0f}', 6)

# === DA UCX BRIDGE THREAD ===
# _async_send starts at 1452, ends at 1859
bridge_s = DA_UCX_INNER_START
bridge_e1 = bridge_s + DA_UCX_INNER_ENCODE
bridge_e2 = bridge_e1 + DA_UCX_INNER_SEND_META
bridge_e3 = bridge_e2 + DA_UCX_INNER_SEND_DATA
bar(Y_DA_BRIDGE, bridge_s, bridge_e1, C['overhead'], '', 5)
bar(Y_DA_BRIDGE, bridge_e1, bridge_e2, C['ucx_meta'],
    f'send(meta)\n{DA_UCX_INNER_SEND_META:.0f}us', 6)
bar(Y_DA_BRIDGE, bridge_e2, bridge_e3, C['ucx_data'],
    f'send(tensor) NIC DMA+RDMA\n{DA_UCX_INNER_SEND_DATA:.0f}us', 7, 'white')

# === DF UCX BRIDGE THREAD (recv) ===
# _async_recv starts at -726, recv_meta takes 2348us, recv_data takes 243us
df_recv_meta_start = DF_UCX_INNER_RECV_START
df_recv_meta_end = df_recv_meta_start + DF_UCX_INNER_RECV_META_DUR
df_recv_data_end = df_recv_meta_end + DF_UCX_INNER_RECV_DATA_DUR

bar(Y_DF_BRIDGE, df_recv_meta_start, df_recv_meta_end, C['df_recv_wait'],
    f'await recv(meta): {DF_UCX_INNER_RECV_META_DUR:.0f}us (waiting for DA)', 7)
bar(Y_DF_BRIDGE, df_recv_meta_end, df_recv_data_end, C['df_recv_data'],
    f'recv(tensor)\n{DF_UCX_INNER_RECV_DATA_DUR:.0f}us', 6, 'white')

# === DF GPU COMPUTE ===
# MLP compute: 1966 -> 2385 (419us)
bar(Y_DF_GPU, DF_RECV_END, DF_SEND_START, C['df_mlp'],
    f'DF MLP Kernel\n{DF_SEND_START - DF_RECV_END:.0f}us', 8, 'white')
# DF send_async main thread
bar(Y_DF_GPU, DF_SEND_START, DF_SEND_END, C['df_send_main'],
    f'send\n{DF_SEND_END - DF_SEND_START:.0f}us', 6)

# === DF DAEMON THREAD (send back) ===
bar(Y_DF_DAEMON, DF_SENDER_LAUNCHED, DF_SENDER_CUDA_SYNC, C['event_sync'],
    f'event.sync\n{DF_SENDER_CUDA_SYNC - DF_SENDER_LAUNCHED:.0f}us', 6, 'white')
bar(Y_DF_DAEMON, DF_SENDER_CUDA_SYNC, DF_SENDER_UCX_DONE, C['dispatch'],
    f'ucx_send\n{DF_SENDER_UCX_DONE - DF_SENDER_CUDA_SYNC:.0f}us', 6)

# DF bridge send
df_bridge_encode = 12.4
df_bridge_meta = 73.0
df_bridge_data = 353.8
df_bs = DF_UCX_INNER_SEND_START
df_be1 = df_bs + df_bridge_encode
df_be2 = df_be1 + df_bridge_meta
df_be3 = df_be2 + df_bridge_data
bar(Y_DF_DAEMON - 1.0, df_be1, df_be2, C['ucx_meta'], f'meta {df_bridge_meta:.0f}', 5)
bar(Y_DF_DAEMON - 1.0, df_be2, df_be3, C['ucx_data'],
    f'send(tensor) {df_bridge_data:.0f}us', 6, 'white')

# === ARROWS showing data flow ===
# DA bridge send_data end -> DF recv_data arrives
ax.annotate('', xy=(df_recv_meta_end, Y_DF_BRIDGE + bar_h/2 + 0.05),
            xytext=(bridge_e3, Y_DA_BRIDGE - bar_h/2 - 0.05),
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2,
                           connectionstyle='arc3,rad=0.15'))
ax.text((bridge_e3 + df_recv_meta_end)/2 + 50, (Y_DA_BRIDGE + Y_DF_BRIDGE)/2,
        'RDMA\ntransfer', fontsize=7, ha='center', color='#e74c3c', style='italic')

# DF send end -> DA recv unblocks
ax.annotate('', xy=(DA_L4_RECV_END_F, Y_DA_GPU - bar_h/2 - 0.05),
            xytext=(df_be3, Y_DF_DAEMON - 1.0 - bar_h/2),
            arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2,
                           connectionstyle='arc3,rad=-0.2'))
ax.text((df_be3 + DA_L4_RECV_END_F)/2, (Y_DF_DAEMON - 1.0 + Y_DA_GPU)/2 - 0.3,
        'DF result\narrives DA', fontsize=7, ha='center', color='#27ae60', style='italic')

# === MARKERS ===
ax.axvline(0, color='black', linewidth=1.5, linestyle='-')
ax.axvline(DA_L5_SEND_START, color='black', linewidth=1.5, linestyle='--')
ax.text(0, Y_DA_GPU + 0.6, 't=0\nL3 recv_end\n(cycle start)', fontsize=7, ha='center', va='bottom')
ax.text(DA_L5_SEND_START, Y_DA_GPU + 0.6, f't={DA_L5_SEND_START:.0f}us\nL5 send_start\n(next cycle)', fontsize=7, ha='center', va='bottom')

# Cycle time annotation
ax.annotate('', xy=(DA_L5_SEND_START, Y_DA_GPU + 1.3), xytext=(DA_L4_SEND_START, Y_DA_GPU + 1.3),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
ax.text((DA_L4_SEND_START + DA_L5_SEND_START)/2, Y_DA_GPU + 1.5,
        f'Cycle = {DA_L5_SEND_START - DA_L4_SEND_START:.0f}us', ha='center', fontsize=9, fontweight='bold')

# === ROW LABELS ===
ax.text(-900, Y_DA_GPU, 'DA GPU\ncompute', ha='right', va='center', fontsize=9, fontweight='bold')
ax.text(-900, Y_DA_DAEMON, 'DA daemon\nthread', ha='right', va='center', fontsize=9, fontweight='bold')
ax.text(-900, Y_DA_BRIDGE, 'DA bridge\n(UCX loop)', ha='right', va='center', fontsize=9, fontweight='bold')
ax.text(-900, Y_DF_BRIDGE, 'DF bridge\n(UCX loop)', ha='right', va='center', fontsize=9, fontweight='bold')
ax.text(-900, Y_DF_GPU, 'DF GPU\ncompute', ha='right', va='center', fontsize=9, fontweight='bold')
ax.text(-900, Y_DF_DAEMON, 'DF daemon\nthread', ha='right', va='center', fontsize=9, fontweight='bold')
ax.text(-900, Y_DF_DAEMON - 1.0, 'DF bridge\n(send back)', ha='right', va='center', fontsize=8, color='gray')

# === LEGEND ===
legend_patches = [
    mpatches.Patch(color=C['prep_attn'], label='prep_attn (layernorm)'),
    mpatches.Patch(color=C['attn'], label='Attention kernel'),
    mpatches.Patch(color=C['prep_mlp'], label='AllReduce + norm'),
    mpatches.Patch(color=C['send_main'], label='send_async (main thread)'),
    mpatches.Patch(color=C['proxy'], label='proxy MLP'),
    mpatches.Patch(color=C['recv_wait'], label='recv_wait (BLOCKED)'),
    mpatches.Patch(color=C['event_sync'], label='event.synchronize()'),
    mpatches.Patch(color=C['ucx_meta'], label='UCX send/recv(meta)'),
    mpatches.Patch(color=C['ucx_data'], label='UCX send/recv(tensor) [NIC DMA+RDMA]'),
    mpatches.Patch(color=C['df_recv_wait'], label='DF await recv(meta) [blocking wait]'),
    mpatches.Patch(color=C['df_recv_data'], label='DF await recv(tensor)'),
    mpatches.Patch(color=C['df_mlp'], label='DF MLP kernel'),
    mpatches.Patch(color=C['overhead'], label='scheduler/dispatch overhead'),
]
ax.legend(handles=legend_patches, loc='lower right', fontsize=7, ncol=2, framealpha=0.9)

ax.set_xlim(-900, 4500)
ax.set_ylim(-2.5, 9.5)
ax.set_yticks([])
ax.set_xlabel('Time (us) relative to DA cycle start', fontsize=10)
ax.set_title('Precise Layer 4 Cycle: DA Attn L(N) -> DA Attn L(N+1)\n'
             'All bars positioned by actual profiling timestamps (Qwen3-32B, tp=1, M=1, decode)',
             fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
out_path = os.path.join(_HERE, 'gantt_full_cycle_precise.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved to {out_path}')
