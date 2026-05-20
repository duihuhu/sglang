#!/usr/bin/env python3
"""Generate Wire DA->DF breakdown chart from UCX_INNER profiling data."""
import json, re, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def parse_events(log_path):
    with open(log_path) as f:
        for line in f:
            m = re.search(r'events=(\[.*\])', line)
            if m:
                return json.loads(m.group(1))
    return []

LOG_DIR = '/workspace/sglang/benchmark/test_motivation/AzurePublicDataset/pipeline_analysis/wire_breakdown_logs'
da_events = parse_events(os.path.join(LOG_DIR, 'da.log'))
df_events = parse_events(os.path.join(LOG_DIR, 'df.log'))

send_brk = [e for e in da_events if e.get('event') == 'async_send_breakdown']
recv_brk = [e for e in df_events if e.get('event') == 'async_recv_breakdown']

layers_to_show = list(range(2, min(12, len(send_brk), len(recv_brk))))

fig, axes = plt.subplots(3, 1, figsize=(20, 14), gridspec_kw={'height_ratios': [3, 3, 2]})
fig.patch.set_facecolor('white')

# -- Top panel: Gantt timeline per layer --
ax = axes[0]
ax.set_title('Wire DA->DF: _async_send to _async_recv Timeline (per layer)',
             fontsize=13, fontweight='bold', pad=10)

colors_send = {'encode_meta': '#3498db', 'send_meta': '#2ecc71', 'send_data': '#e74c3c'}
colors_recv = {'recv_meta': '#f39c12', 'recv_data': '#9b59b6'}

bar_h = 0.35
y_positions = []

for i, layer_idx in enumerate(layers_to_show):
    sb = send_brk[layer_idx]
    rb = recv_brk[layer_idx]

    y_da = len(layers_to_show) - i - 0.2
    y_df = y_da - bar_h - 0.05
    y_positions.append((y_da, y_df, sb['layer']))

    # DA _async_send: stacked bar
    x = 0
    ax.barh(y_da, sb['encode_meta_us'], left=x, height=bar_h,
            color=colors_send['encode_meta'], edgecolor='white', linewidth=0.5)
    x += sb['encode_meta_us']
    ax.barh(y_da, sb['send_meta_us'], left=x, height=bar_h,
            color=colors_send['send_meta'], edgecolor='white', linewidth=0.5)
    x += sb['send_meta_us']
    ax.barh(y_da, sb['send_data_us'], left=x, height=bar_h,
            color=colors_send['send_data'], edgecolor='white', linewidth=0.5)

    # DF _async_recv: offset by absolute timestamp difference
    send_start_ms = sb['ts_ms']
    recv_start_ms = rb['ts_ms']
    offset_us = (recv_start_ms - send_start_ms) * 1000

    x = offset_us
    ax.barh(y_df, rb['recv_meta_us'], left=x, height=bar_h,
            color=colors_recv['recv_meta'], edgecolor='white', linewidth=0.5)
    x += rb['recv_meta_us']
    ax.barh(y_df, rb['recv_data_us'], left=x, height=bar_h,
            color=colors_recv['recv_data'], edgecolor='white', linewidth=0.5)

for y_da, y_df, layer in y_positions:
    ax.text(-80, (y_da + y_df) / 2, f'L{layer}', ha='right', va='center',
            fontsize=9, fontweight='bold')

ax.set_xlabel('Time (us)', fontsize=10)
ax.set_yticks([(y_da + y_df) / 2 for y_da, y_df, _ in y_positions])
ax.set_yticklabels([f'Layer {l}' for _, _, l in y_positions], fontsize=9)
ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

legend_patches = [
    mpatches.Patch(color=colors_send['encode_meta'], label='DA: encode_meta (~9us)'),
    mpatches.Patch(color=colors_send['send_meta'], label='DA: await send(meta) (~37us)'),
    mpatches.Patch(color=colors_send['send_data'], label='DA: await send(tensor) (~307us)'),
    mpatches.Patch(color=colors_recv['recv_meta'], label='DF: await recv(meta) (~1460us, includes wait)'),
    mpatches.Patch(color=colors_recv['recv_data'], label='DF: await recv(tensor) (~238us)'),
]
ax.legend(handles=legend_patches, loc='upper right', fontsize=8, framealpha=0.9)
ax.set_xlim(-100, 3000)
ax.grid(axis='x', alpha=0.3)

# -- Middle panel: p50 bar chart --
ax2 = axes[1]
ax2.set_title('Wire DA->DF Breakdown: Where Time is Spent (p50, layers 2-63)',
              fontsize=13, fontweight='bold', pad=10)

s_encode = sorted([e['encode_meta_us'] for e in send_brk[2:]])
s_meta = sorted([e['send_meta_us'] for e in send_brk[2:]])
s_data = sorted([e['send_data_us'] for e in send_brk[2:]])
r_meta = sorted([e['recv_meta_us'] for e in recv_brk[2:]])
r_data = sorted([e['recv_data_us'] for e in recv_brk[2:]])

p50 = lambda d: d[len(d)//2]

categories = [
    'DA: encode_meta',
    'DA: send(meta)',
    'DA: send(tensor)\n[RDMA submit]',
    'DF: recv(meta)\n[wait + receive]',
    'DF: recv(tensor)\n[data arrival]',
]
values = [p50(s_encode), p50(s_meta), p50(s_data), p50(r_meta), p50(r_data)]
colors_bar = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

bars = ax2.barh(categories, values, color=colors_bar, edgecolor='white', height=0.6)
for bar, val in zip(bars, values):
    ax2.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
             f'{val:.1f} us', va='center', fontsize=10, fontweight='bold')

ax2.set_xlabel('Time (us)', fontsize=10)
ax2.set_xlim(0, max(values) * 1.3)
ax2.grid(axis='x', alpha=0.3)
ax2.invert_yaxis()

# -- Bottom panel: Explanation --
ax3 = axes[2]
ax3.axis('off')

explanation = (
    "TIMELINE INTERPRETATION:\n"
    "\n"
    "DA bridge thread                                    DF bridge thread\n"
    "----------------                                    ----------------\n"
    "+--encode_meta--+                                   +--------------------------------------+\n"
    "|    8.6 us     |                                   |  await recv(meta): 1460 us           |\n"
    "+--send(meta)---+                                   |  (DF is BLOCKING here, waiting for   |\n"
    "|   36.7 us     |  ---- 64B meta over RDMA ---->   |   DA to send meta + network flight)  |\n"
    "+--send(tensor)-+                                   +--------------------------------------+\n"
    "|  306.8 us     |  ---- tensor over RDMA ------>   |  await recv(buf): 237.7 us           |\n"
    "+---------------+                                   |  (tensor arrives almost immediately  |\n"
    "  total: 352 us                                     |   after meta due to UCX pipelining)  |\n"
    "                                                    +--------------------------------------+\n"
    "                                                      total: 1672 us\n"
    "\n"
    "KEY INSIGHT: 'Pure network flight' = -37 us (NEGATIVE!)\n"
    "  - UCX pipelining means DF finishes recv BEFORE DA's await send() returns.\n"
    "  - The 1460 us 'recv(meta)' on DF is NOT network latency -- it is DF waiting\n"
    "    for DA to reach the send point (DA is still doing compute/AllReduce/etc).\n"
    "  - The REAL wire transfer is essentially FREE (overlapped with DA processing).\n"
    "  - On the gantt chart, 'Wire DA->DF' is dominated by DA-side processing delay,\n"
    "    not actual network cost."
)
ax3.text(0.02, 0.95, explanation, transform=ax3.transAxes, fontsize=9,
         fontfamily='monospace', va='top', ha='left',
         bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))

plt.tight_layout()
out_path = '/workspace/sglang/benchmark/test_motivation/AzurePublicDataset/pipeline_analysis/gantt_wire_breakdown_inner.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved to {out_path}')
