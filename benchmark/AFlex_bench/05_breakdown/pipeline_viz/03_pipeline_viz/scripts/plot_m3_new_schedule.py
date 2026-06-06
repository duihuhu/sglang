#!/usr/bin/env python3
"""Draw precise M=3 new batch-schedule gantt chart using actual timestamps.

Shows DA and DF side-by-side for Layer 2 (steady state) with all 3 micro-batches.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

# === ACTUAL TIMESTAMPS (Layer 2, M=3, relative to DA L2 mb0 send_start = t=0) ===
# DA side:
#   mb0 send_start: 0, send_end: 44
#   mb1 send_start: 1057, send_end: 1093
#   mb2 send_start: 2323, send_end: 2357
#   mb0 recv_start: 2634, recv_end: 3088 (recv_dur=454)
#   mb1 recv_start: 3362, recv_end: 3395 (recv_dur=33)
#   mb2 recv_start: 3502, recv_end: 4493 (recv_dur=992)

# DA SENDER events (relative to DA L2 mb0 send_start):
# From L1 data pattern, sender events are ~19us after send_start
#   mb0 enqueued: ~19, cuda_sync: ~156, send_done: ~171
#   mb1 enqueued: ~1076, cuda_sync: ~1200, send_done: ~1214
#   mb2 enqueued: ~2342, cuda_sync: ~2385, send_done: ~2399

# DF side (clock offset = -1028us, so DF_abs = DA_abs - 1028):
# DF L2 events (relative to DF L2 first event):
#   mb0 recv_start: 0, recv_end: 1716 (recv_dur=1716)
#   mb0 send_start: 2018, send_end: 2050
#   mb1 recv_start: 2735, recv_end: 2957 (recv_dur=222)
#   mb1 send_start: 3249, send_end: 3285
#   mb2 recv_start: 4133, recv_end: 4300 (recv_dur=167)
#   mb2 send_start: 4592, send_end: 4623

# To align: DF L2 mb0 recv_start absolute = DA_L2_mb0_send_start + offset
# DF pre-posts recv before DA sends, so DF recv_start is BEFORE DA send_start
# From L1 data: DF L1 recv_start is 1028us before DA L1 send_start
# So for L2: DF_t0 = DA_t0 - 1028 (approximately)
# DF events relative to DA t=0:
DF_OFFSET = -1028  # DF recv_start is ~1028us before DA send_start

# DF L2 timeline (relative to DA L2 mb0 send_start):
DF_MB0_RECV_START = DF_OFFSET + 0
DF_MB0_RECV_END = DF_OFFSET + 1716
DF_MB0_SEND_START = DF_OFFSET + 2018
DF_MB0_SEND_END = DF_OFFSET + 2050
DF_MB1_RECV_START = DF_OFFSET + 2735
DF_MB1_RECV_END = DF_OFFSET + 2957
DF_MB1_SEND_START = DF_OFFSET + 3249
DF_MB1_SEND_END = DF_OFFSET + 3285
DF_MB2_RECV_START = DF_OFFSET + 4133
DF_MB2_RECV_END = DF_OFFSET + 4300
DF_MB2_SEND_START = DF_OFFSET + 4592
DF_MB2_SEND_END = DF_OFFSET + 4623

# DF MLP compute: between recv_end and send_start
# mb0: 1716 -> 2018 = 302us of MLP + overhead
# mb1: 2957 -> 3249 = 292us
# mb2: 4300 -> 4592 = 292us

# DF SENDER (send back to DA):
# mb0 enqueued: ~2033, cuda_sync: ~2490, send_done: ~2504
# mb1 enqueued: ~3264, cuda_sync: ~3759, send_done: ~3773
# mb2 enqueued: ~4607, cuda_sync: ~5100, send_done: ~5114
# Relative to DA t=0:
DF_SENDER_MB0_ENQUEUE = DF_OFFSET + 2033
DF_SENDER_MB0_CUDA_SYNC = DF_OFFSET + 2490
DF_SENDER_MB0_DONE = DF_OFFSET + 2504
DF_SENDER_MB1_ENQUEUE = DF_OFFSET + 3264
DF_SENDER_MB1_CUDA_SYNC = DF_OFFSET + 3759
DF_SENDER_MB1_DONE = DF_OFFSET + 3773
DF_SENDER_MB2_ENQUEUE = DF_OFFSET + 4607
DF_SENDER_MB2_CUDA_SYNC = DF_OFFSET + 5100
DF_SENDER_MB2_DONE = DF_OFFSET + 5114

# GPU timing per micro-batch (from CUDA events):
# DA: attn ~470us per mb, prep_mlp ~77us per mb
# Between send_end and next send_start: ~1013us = attn(470) + prep_attn(~130) + prep_mlp(77) + overhead
# Between last send_end and first recv_start: 2634-2357 = 277us = proxy + overhead

# === DRAW ===
fig, ax = plt.subplots(1, 1, figsize=(26, 10))
fig.patch.set_facecolor('white')

# Y positions
Y_DA_GPU = 7.0       # DA main thread (compute + sync send/recv)
Y_DA_SENDER = 5.5    # DA persistent sender daemon
Y_DF_RECV = 4.0      # DF main thread (recv + compute + send)
Y_DF_SENDER = 2.5    # DF persistent sender daemon

bar_h = 0.7

# Colors per micro-batch
MB_COLORS = ['#3498db', '#e74c3c', '#27ae60']  # blue, red, green
C = {
    'attn': ['#2e86c1', '#c0392b', '#1e8449'],
    'send': ['#5dade2', '#e74c3c', '#2ecc71'],
    'recv_wait': ['#85c1e9', '#f1948a', '#82e0aa'],
    'recv_fast': ['#d4efdf', '#fadbd8', '#d5f5e3'],
    'df_recv_wait': ['#f9e79f', '#f5cba7', '#d5f5e3'],
    'df_mlp': ['#1a5276', '#922b21', '#145a32'],
    'df_send': ['#aed6f1', '#f5b7b1', '#a9dfbf'],
    'event_sync': '#8e44ad',
    'ucx_send': '#e67e22',
    'overhead': '#d5dbdb',
}

def bar(y, start, end, color, label='', fontsize=7, text_color='black', alpha=1.0):
    dur = end - start
    if dur < 1:
        return
    ax.barh(y, dur, left=start, height=bar_h, color=color, edgecolor='white',
            linewidth=0.5, alpha=alpha)
    if label and dur > 80:
        ax.text(start + dur/2, y, label, ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color=text_color)

# === DA MAIN THREAD ===
# A-stage batch: A(2,0) -> A(2,1) -> A(2,2)
# Each A-stage: send_start -> send_end (the send_async call)
# Between sends: attention compute

# mb0: t=0..44 (send), before that was attn compute from prev layer
# Attn compute for mb0: approx -470..0 (but we start at t=0)
# Let's show from t=0 (first send of this layer)

# A(2,0): send at 0-44
bar(Y_DA_GPU, 0, 44, C['send'][0], 'send\n44us', 5)
# Attn compute for mb1: 44 -> 1057
bar(Y_DA_GPU, 44, 1057, C['attn'][1], 'Attn mb1 + prep\n1013us', 8, 'white')
# A(2,1): send at 1057-1093
bar(Y_DA_GPU, 1057, 1093, C['send'][1], '', 5)
# Attn compute for mb2: 1093 -> 2323
bar(Y_DA_GPU, 1093, 2323, C['attn'][2], 'Attn mb2 + prep\n1230us', 8, 'white')
# A(2,2): send at 2323-2357
bar(Y_DA_GPU, 2323, 2357, C['send'][2], '', 5)
# Gap before F-stage batch: 2357 -> 2634
bar(Y_DA_GPU, 2357, 2634, C['overhead'], 'proxy\n277us', 6)

# F-stage batch: F(2,0) -> F(2,1) -> F(2,2)
# F(2,0): recv 2634-3088 (454us wait)
bar(Y_DA_GPU, 2634, 3088, C['recv_wait'][0], 'recv mb0\n454us', 7)
# Gap 3088-3362: prep_attn for next layer mb0
bar(Y_DA_GPU, 3088, 3362, C['overhead'], 'prep\n274us', 6)
# F(2,1): recv 3362-3395 (33us! almost instant)
bar(Y_DA_GPU, 3362, 3395, C['recv_fast'][1], '', 5)
# Gap 3395-3502
bar(Y_DA_GPU, 3395, 3502, C['overhead'], '', 5)
# F(2,2): recv 3502-4493 (992us wait)
bar(Y_DA_GPU, 3502, 4493, C['recv_wait'][2], 'recv mb2\n992us', 7)

# === DA PERSISTENT SENDER DAEMON ===
# mb0: enqueued ~19, cuda_sync ~156, send_done ~171
bar(Y_DA_SENDER, 19, 156, C['event_sync'], 'event.sync\n137us', 6, 'white')
bar(Y_DA_SENDER, 156, 171, C['ucx_send'], '', 5)
# mb1: enqueued ~1076, cuda_sync ~1200, send_done ~1214
bar(Y_DA_SENDER, 1076, 1200, C['event_sync'], 'sync\n124us', 6, 'white')
bar(Y_DA_SENDER, 1200, 1214, C['ucx_send'], '', 5)
# mb2: enqueued ~2342, cuda_sync ~2385, send_done ~2399
bar(Y_DA_SENDER, 2342, 2385, C['event_sync'], '43', 5, 'white')
bar(Y_DA_SENDER, 2385, 2399, C['ucx_send'], '', 5)

# === DF MAIN THREAD ===
# mb0: recv -1028..-1028+1716=688, MLP 688..990, send 990..1022
bar(Y_DF_RECV, DF_MB0_RECV_START, DF_MB0_RECV_END, C['df_recv_wait'][0],
    f'recv mb0 (wait DA)\n{DF_MB0_RECV_END-DF_MB0_RECV_START:.0f}us', 7)
bar(Y_DF_RECV, DF_MB0_RECV_END, DF_MB0_SEND_START, C['df_mlp'][0],
    f'MLP mb0\n{DF_MB0_SEND_START-DF_MB0_RECV_END:.0f}us', 7, 'white')
bar(Y_DF_RECV, DF_MB0_SEND_START, DF_MB0_SEND_END, C['df_send'][0], '', 5)

# mb1
bar(Y_DF_RECV, DF_MB1_RECV_START, DF_MB1_RECV_END, C['df_recv_wait'][1],
    f'recv mb1\n{DF_MB1_RECV_END-DF_MB1_RECV_START:.0f}us', 6)
bar(Y_DF_RECV, DF_MB1_RECV_END, DF_MB1_SEND_START, C['df_mlp'][1],
    f'MLP mb1\n{DF_MB1_SEND_START-DF_MB1_RECV_END:.0f}us', 7, 'white')
bar(Y_DF_RECV, DF_MB1_SEND_START, DF_MB1_SEND_END, C['df_send'][1], '', 5)

# mb2
bar(Y_DF_RECV, DF_MB2_RECV_START, DF_MB2_RECV_END, C['df_recv_wait'][2],
    f'recv mb2\n{DF_MB2_RECV_END-DF_MB2_RECV_START:.0f}us', 6)
bar(Y_DF_RECV, DF_MB2_RECV_END, DF_MB2_SEND_START, C['df_mlp'][2],
    f'MLP mb2\n{DF_MB2_SEND_START-DF_MB2_RECV_END:.0f}us', 7, 'white')
bar(Y_DF_RECV, DF_MB2_SEND_START, DF_MB2_SEND_END, C['df_send'][2], '', 5)

# === DF PERSISTENT SENDER DAEMON ===
bar(Y_DF_SENDER, DF_SENDER_MB0_ENQUEUE, DF_SENDER_MB0_CUDA_SYNC, C['event_sync'],
    f'event.sync mb0\n{DF_SENDER_MB0_CUDA_SYNC-DF_SENDER_MB0_ENQUEUE:.0f}us', 6, 'white')
bar(Y_DF_SENDER, DF_SENDER_MB0_CUDA_SYNC, DF_SENDER_MB0_DONE, C['ucx_send'], '', 5)

bar(Y_DF_SENDER, DF_SENDER_MB1_ENQUEUE, DF_SENDER_MB1_CUDA_SYNC, C['event_sync'],
    f'event.sync mb1\n{DF_SENDER_MB1_CUDA_SYNC-DF_SENDER_MB1_ENQUEUE:.0f}us', 6, 'white')
bar(Y_DF_SENDER, DF_SENDER_MB1_CUDA_SYNC, DF_SENDER_MB1_DONE, C['ucx_send'], '', 5)

bar(Y_DF_SENDER, DF_SENDER_MB2_ENQUEUE, DF_SENDER_MB2_CUDA_SYNC, C['event_sync'],
    f'event.sync mb2\n{DF_SENDER_MB2_CUDA_SYNC-DF_SENDER_MB2_ENQUEUE:.0f}us', 6, 'white')
bar(Y_DF_SENDER, DF_SENDER_MB2_CUDA_SYNC, DF_SENDER_MB2_DONE, C['ucx_send'], '', 5)

# === ARROWS: data flow ===
# DA send mb0 done -> DF recv mb0 unblocks
ax.annotate('', xy=(DF_MB0_RECV_END, Y_DF_RECV + bar_h/2 + 0.05),
            xytext=(171, Y_DA_SENDER - bar_h/2 - 0.05),
            arrowprops=dict(arrowstyle='->', color=MB_COLORS[0], lw=1.5,
                           connectionstyle='arc3,rad=0.1'))

# DA send mb1 done -> DF recv mb1 unblocks
ax.annotate('', xy=(DF_MB1_RECV_END, Y_DF_RECV + bar_h/2 + 0.05),
            xytext=(1214, Y_DA_SENDER - bar_h/2 - 0.05),
            arrowprops=dict(arrowstyle='->', color=MB_COLORS[1], lw=1.5,
                           connectionstyle='arc3,rad=0.1'))

# DA send mb2 done -> DF recv mb2 unblocks
ax.annotate('', xy=(DF_MB2_RECV_END, Y_DF_RECV + bar_h/2 + 0.05),
            xytext=(2399, Y_DA_SENDER - bar_h/2 - 0.05),
            arrowprops=dict(arrowstyle='->', color=MB_COLORS[2], lw=1.5,
                           connectionstyle='arc3,rad=0.1'))

# DF send mb0 done -> DA recv mb0 unblocks
ax.annotate('', xy=(3088, Y_DA_GPU - bar_h/2 - 0.05),
            xytext=(DF_SENDER_MB0_DONE, Y_DF_SENDER - bar_h/2 - 0.05),
            arrowprops=dict(arrowstyle='->', color=MB_COLORS[0], lw=1.5, linestyle='dashed',
                           connectionstyle='arc3,rad=-0.15'))

# DF send mb1 done -> DA recv mb1 (almost instant)
ax.annotate('', xy=(3395, Y_DA_GPU - bar_h/2 - 0.05),
            xytext=(DF_SENDER_MB1_DONE, Y_DF_SENDER - bar_h/2 - 0.05),
            arrowprops=dict(arrowstyle='->', color=MB_COLORS[1], lw=1.5, linestyle='dashed',
                           connectionstyle='arc3,rad=-0.15'))

# === ANNOTATIONS ===
# Highlight the key win: F(2,1) recv = 33us
ax.annotate('recv = 33us!\n(data already ready)', xy=(3378, Y_DA_GPU + bar_h/2 + 0.05),
            fontsize=8, ha='center', va='bottom', color='#27ae60', fontweight='bold')

# Cycle time
ax.axvline(0, color='black', linewidth=1.5, linestyle='-')
ax.axvline(4493, color='black', linewidth=1.5, linestyle='--')
ax.text(0, Y_DA_GPU + 0.8, 't=0\nL2 A-batch start', fontsize=7, ha='center', va='bottom')
ax.text(4493, Y_DA_GPU + 0.8, 't=4493us\nL2 F-batch end', fontsize=7, ha='center', va='bottom')

ax.annotate('', xy=(4493, Y_DA_GPU + 1.3), xytext=(0, Y_DA_GPU + 1.3),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
ax.text(2246, Y_DA_GPU + 1.5, 'Layer cycle = 4493us (3 mbs) = 1498us/mb',
        ha='center', fontsize=10, fontweight='bold')

# === ROW LABELS ===
ax.text(-200, Y_DA_GPU, 'DA main\nthread', ha='right', va='center', fontsize=9, fontweight='bold')
ax.text(-200, Y_DA_SENDER, 'DA sender\ndaemon', ha='right', va='center', fontsize=9, fontweight='bold')
ax.text(-200, Y_DF_RECV, 'DF main\nthread', ha='right', va='center', fontsize=9, fontweight='bold')
ax.text(-200, Y_DF_SENDER, 'DF sender\ndaemon', ha='right', va='center', fontsize=9, fontweight='bold')

# === LEGEND ===
legend_patches = [
    mpatches.Patch(color=C['attn'][0], label='DA Attention + prep (mb0)'),
    mpatches.Patch(color=C['attn'][1], label='DA Attention + prep (mb1)'),
    mpatches.Patch(color=C['attn'][2], label='DA Attention + prep (mb2)'),
    mpatches.Patch(color=C['send'][0], label='send_async (main thread, ~35us)'),
    mpatches.Patch(color=C['recv_wait'][0], label='recv_wait (blocking)'),
    mpatches.Patch(color=C['recv_fast'][1], label='recv_wait (instant, data ready)'),
    mpatches.Patch(color=C['df_recv_wait'][0], label='DF recv (wait for DA)'),
    mpatches.Patch(color=C['df_mlp'][0], label='DF MLP kernel'),
    mpatches.Patch(color=C['event_sync'], label='daemon: event.synchronize()'),
    mpatches.Patch(color=C['ucx_send'], label='daemon: UCX send'),
    mpatches.Patch(color=C['overhead'], label='overhead / proxy'),
]
ax.legend(handles=legend_patches, loc='lower right', fontsize=7, ncol=2, framealpha=0.9)

ax.set_xlim(-1200, 5500)
ax.set_ylim(1.0, 9.5)
ax.set_yticks([])
ax.set_xlabel('Time (us) relative to DA Layer 2 A-batch start', fontsize=10)
ax.set_title('M=3 New Batch Schedule: Layer 2 Precise Timeline\n'
             'A(2,0) A(2,1) A(2,2) | F(2,0) F(2,1) F(2,2)\n'
             'Qwen3-32B, tp=1, UCX, decode',
             fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
out_path = os.path.join(_HERE, 'gantt_m3_new_schedule_precise.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved to {out_path}')
