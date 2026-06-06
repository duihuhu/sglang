#!/usr/bin/env python3
"""Draw precise IPC cycle breakdown using actual profiling timestamps."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_HERE = os.path.dirname(os.path.abspath(__file__))

# === ACTUAL TIMESTAMPS (relative to DA cycle start = L3 recv_end, in us) ===
# DA side (from profiling):
#   L3 recv_end (t=0)
#   L4 send_start: +591.1us
#   L4 send_end:   +784.2us  (send_dur=193.1us)
#   L4 recv_start: +912.1us
#   L4 recv_end:   +1737.1us (recv_dur=825.6us)
#   L5 send_start: +2322.0us
#   Cycle time (L4->L5): 1731.0us

# DF side (same t=0 reference):
#   DF recv_start: -11.0us (pre-posted before DA cycle!)
#   DF recv_end:   +901.1us (recv_dur=912.7us)
#   DF send_start: +1150.1us
#   DF send_end:   +1597.2us (send_dur=447.3us)

# IPC_INNER DA send_breakdown (layer 4):
#   contiguous: 1.0us
#   slot_wait: 1.8us
#   event_sync: 118.5us (GPU copy to send_buf + cudaEventSync)
#   flag_write: 2.3us
#   total: 125.9us

# IPC_INNER DA recv_breakdown (layer 4):
#   flag_poll: 659.9us (waiting for DF to write flag)
#   memcpy_peer_sync: 74.8us (cudaMemcpyPeer + event.sync)
#   decode: 67.6us
#   total: 802.3us

# IPC_INNER DF recv_breakdown (layer 4):
#   flag_poll: 768.6us (waiting for DA to write flag)
#   memcpy_peer_sync: 66.1us (cudaMemcpyPeer + event.sync)
#   decode: 57.8us
#   total: 892.5us

# IPC_INNER DF send_breakdown (layer 4):
#   contiguous: 1.0us
#   slot_wait: 1.5us
#   event_sync: 382.6us (GPU copy + MLP result sync)
#   flag_write: 2.0us
#   total: 389.4us

# GPU timing (per layer, from CUDA events, last steady-state line):
# DA: prep_attn=56.3us, attn=423.4us, prep_mlp=228.1us, mlp=20.3us, postprocess=892.2us
# DF: prep_mlp=906.3us (recv from DA), mlp=553.1us, postprocess=110.9us

# Derived timeline:
DA_PREP_ATTN = (0, 56)           # 3.6ms/64 = 56us
DA_ATTN = (56, 480)              # 27.1ms/64 = 423us
DA_PREP_MLP = (480, 591)         # ends at send_start (AllReduce+norm, but IPC send is inside prep_mlp)
DA_SEND = (591, 784)             # send_start -> send_end = 193us (SYNCHRONOUS, blocks main thread!)
DA_RECV_START = (784, 912)       # gap between send_end and recv_start (proxy MLP)
DA_RECV_WAIT = (912, 1737)       # recv_start -> recv_end = 825us (flag_poll + memcpy + decode)
# Next layer prep
DA_NEXT_PREP = (1737, 2322)      # overhead before L5 send_start

# DF timeline:
DF_RECV_START = -11              # DF pre-posted recv
DF_FLAG_POLL_END = -11 + 768.6   # = 757.6 (flag_poll waiting for DA)
DF_MEMCPY_END = DF_FLAG_POLL_END + 66.1  # = 823.7
DF_DECODE_END = DF_MEMCPY_END + 57.8     # = 881.5
DF_RECV_END = 901.1
DF_MLP_START = DF_RECV_END       # DF starts MLP after recv
DF_MLP_END = 1150.1              # DF send_start (MLP done)
DF_SEND_START = 1150.1
DF_SEND_END = 1597.2

# IPC send breakdown inside DA send (591 -> 784):
DA_IPC_CONTIG = (591, 592)       # 1us
DA_IPC_SLOT_WAIT = (592, 594)    # 1.8us
DA_IPC_GPU_COPY_SYNC = (594, 712)  # 118.5us (copy to send_buf + event.sync)
DA_IPC_FLAG_WRITE = (712, 714)   # 2.3us
# Note: send_dur=193us but IPC total=126us, gap is Python overhead around _send_tensor_impl

# IPC recv breakdown inside DA recv_wait (912 -> 1737):
DA_IPC_FLAG_POLL = (912, 1572)   # 659.9us
DA_IPC_MEMCPY = (1572, 1647)     # 74.8us
DA_IPC_DECODE = (1647, 1714)     # 67.6us

# DF IPC send breakdown (1150 -> 1597):
DF_IPC_CONTIG = (1150, 1151)
DF_IPC_SLOT_WAIT = (1151, 1153)
DF_IPC_GPU_COPY_SYNC = (1153, 1536)  # 382.6us (includes MLP result sync!)
DF_IPC_FLAG_WRITE = (1536, 1538)

fig, ax = plt.subplots(1, 1, figsize=(24, 9))
fig.patch.set_facecolor('white')

Y_DA_GPU = 5.5
Y_DA_IPC = 4.0    # IPC send/recv detail (same thread, synchronous)
Y_DF_IPC = 2.5    # DF IPC recv detail
Y_DF_GPU = 1.0    # DF GPU compute + send

bar_h = 0.7

C = {
    'prep_attn': '#5dade2',
    'attn': '#2e86c1',
    'prep_mlp': '#f4d03f',
    'send_sync': '#e74c3c',
    'proxy': '#d5dbdb',
    'recv_wait': '#e74c3c',
    'flag_poll': '#f5b041',
    'memcpy_peer': '#9b59b6',
    'decode': '#1abc9c',
    'gpu_copy_sync': '#e67e22',
    'flag_write': '#2ecc71',
    'slot_wait': '#bdc3c7',
    'df_mlp': '#27ae60',
    'overhead': '#bdc3c7',
}

def bar(y, start, end, color, label='', fontsize=7, text_color='black'):
    dur = end - start
    if dur < 0.5:
        return
    ax.barh(y, dur, left=start, height=bar_h, color=color, edgecolor='white', linewidth=0.5)
    if label and dur > 40:
        ax.text(start + dur/2, y, label, ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color=text_color)

# === DA GPU COMPUTE ===
bar(Y_DA_GPU, *DA_PREP_ATTN, C['prep_attn'], 'prep_attn\n56us', 6)
bar(Y_DA_GPU, *DA_ATTN, C['attn'], 'Attention\n423us', 8, 'white')
bar(Y_DA_GPU, *DA_PREP_MLP, C['prep_mlp'], 'AR+Norm\n111us', 7)
bar(Y_DA_GPU, *DA_SEND, C['send_sync'], 'send_tensor (SYNC)\n193us', 7, 'white')
bar(Y_DA_GPU, *DA_RECV_START, C['proxy'], 'proxy\n128us', 6)
bar(Y_DA_GPU, *DA_RECV_WAIT, C['recv_wait'], 'recv_tensor (SYNC)\n825us', 8, 'white')
bar(Y_DA_GPU, *DA_NEXT_PREP, C['overhead'], 'sched\n585us', 7)

# === DA IPC SEND DETAIL ===
bar(Y_DA_IPC, *DA_IPC_GPU_COPY_SYNC, C['gpu_copy_sync'],
    'GPU copy + event.sync\n118us', 7, 'white')
bar(Y_DA_IPC, *DA_IPC_FLAG_WRITE, C['flag_write'], '', 5)

# === DA IPC RECV DETAIL ===
bar(Y_DA_IPC, *DA_IPC_FLAG_POLL, C['flag_poll'],
    'flag_poll (wait DF)\n660us', 7)
bar(Y_DA_IPC, *DA_IPC_MEMCPY, C['memcpy_peer'],
    'cudaMemcpyPeer\n75us', 6, 'white')
bar(Y_DA_IPC, *DA_IPC_DECODE, C['decode'], 'decode\n68us', 6)

# === DF IPC RECV DETAIL ===
bar(Y_DF_IPC, DF_RECV_START, DF_FLAG_POLL_END, C['flag_poll'],
    'flag_poll (wait DA)\n769us', 7)
bar(Y_DF_IPC, DF_FLAG_POLL_END, DF_MEMCPY_END, C['memcpy_peer'],
    'memcpy\n66us', 6, 'white')
bar(Y_DF_IPC, DF_MEMCPY_END, DF_DECODE_END, C['decode'], 'decode\n58us', 6)

# === DF GPU COMPUTE + SEND ===
bar(Y_DF_GPU, DF_MLP_START, DF_MLP_END, C['df_mlp'],
    f'DF MLP Kernel\n{DF_MLP_END - DF_MLP_START:.0f}us', 8, 'white')
bar(Y_DF_GPU, DF_IPC_GPU_COPY_SYNC[0], DF_IPC_GPU_COPY_SYNC[1], C['gpu_copy_sync'],
    f'GPU copy + event.sync\n{DF_IPC_GPU_COPY_SYNC[1]-DF_IPC_GPU_COPY_SYNC[0]:.0f}us', 7, 'white')
bar(Y_DF_GPU, *DF_IPC_FLAG_WRITE, C['flag_write'], '', 5)

# === ARROWS ===
# DA flag_write -> DF flag_poll ends
ax.annotate('', xy=(DF_FLAG_POLL_END, Y_DF_IPC + bar_h/2 + 0.05),
            xytext=(DA_IPC_FLAG_WRITE[1], Y_DA_IPC - bar_h/2 - 0.05),
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2,
                           connectionstyle='arc3,rad=0.15'))
ax.text((DA_IPC_FLAG_WRITE[1] + DF_FLAG_POLL_END)/2 + 30, (Y_DA_IPC + Y_DF_IPC)/2,
        'NVLink\nflag signal', fontsize=7, ha='center', color='#e74c3c', style='italic')

# DF flag_write -> DA flag_poll ends
ax.annotate('', xy=(DA_IPC_FLAG_POLL[1], Y_DA_IPC + bar_h/2 + 0.05),
            xytext=(DF_IPC_FLAG_WRITE[1], Y_DF_GPU - bar_h/2 - 0.05),
            arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2,
                           connectionstyle='arc3,rad=-0.15'))
ax.text((DF_IPC_FLAG_WRITE[1] + DA_IPC_FLAG_POLL[1])/2 - 30, (Y_DA_IPC + Y_DF_GPU)/2,
        'NVLink\nflag signal', fontsize=7, ha='center', color='#27ae60', style='italic')

# === MARKERS ===
ax.axvline(0, color='black', linewidth=1.5, linestyle='-')
ax.axvline(2322, color='black', linewidth=1.5, linestyle='--')
ax.text(0, Y_DA_GPU + 0.6, 't=0\nL3 recv_end', fontsize=7, ha='center', va='bottom')
ax.text(2322, Y_DA_GPU + 0.6, 't=2322us\nL5 start', fontsize=7, ha='center', va='bottom')

# Cycle annotation
ax.annotate('', xy=(2322, Y_DA_GPU + 1.1), xytext=(591, Y_DA_GPU + 1.1),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
ax.text((591 + 2322)/2, Y_DA_GPU + 1.3,
        'Cycle = 1731us', ha='center', fontsize=10, fontweight='bold')

# === ROW LABELS ===
ax.text(-80, Y_DA_GPU, 'DA GPU\n(main thread)', ha='right', va='center', fontsize=9, fontweight='bold')
ax.text(-80, Y_DA_IPC, 'DA IPC\ndetail', ha='right', va='center', fontsize=9, fontweight='bold')
ax.text(-80, Y_DF_IPC, 'DF IPC\nrecv detail', ha='right', va='center', fontsize=9, fontweight='bold')
ax.text(-80, Y_DF_GPU, 'DF GPU\n(main thread)', ha='right', va='center', fontsize=9, fontweight='bold')

# === LEGEND ===
legend_patches = [
    mpatches.Patch(color=C['prep_attn'], label='prep_attn (layernorm)'),
    mpatches.Patch(color=C['attn'], label='Attention kernel'),
    mpatches.Patch(color=C['prep_mlp'], label='AllReduce + norm'),
    mpatches.Patch(color=C['send_sync'], label='send_tensor (SYNCHRONOUS, blocks main thread)'),
    mpatches.Patch(color=C['recv_wait'], label='recv_tensor (SYNCHRONOUS, blocks main thread)'),
    mpatches.Patch(color=C['flag_poll'], label='flag_poll (busy-wait for peer)'),
    mpatches.Patch(color=C['gpu_copy_sync'], label='GPU copy + cudaEventSync'),
    mpatches.Patch(color=C['memcpy_peer'], label='cudaMemcpyPeer + sync'),
    mpatches.Patch(color=C['decode'], label='decode meta + reshape'),
    mpatches.Patch(color=C['flag_write'], label='flag_write (signal peer)'),
    mpatches.Patch(color=C['df_mlp'], label='DF MLP kernel'),
    mpatches.Patch(color=C['overhead'], label='scheduler overhead'),
]
ax.legend(handles=legend_patches, loc='lower right', fontsize=7, ncol=2, framealpha=0.9)

ax.set_xlim(-100, 2500)
ax.set_ylim(-0.5, 7.5)
ax.set_yticks([])
ax.set_xlabel('Time (us) relative to DA cycle start', fontsize=10)
ax.set_title('IPC Precise Layer 4 Cycle: DA Attn L(N) -> DA Attn L(N+1)\n'
             'All bars from actual profiling timestamps (Qwen3-32B, tp=1, M=1, decode, NVLink IPC)',
             fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
out_path = os.path.join(_HERE, 'gantt_ipc_cycle_precise.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved to {out_path}')
