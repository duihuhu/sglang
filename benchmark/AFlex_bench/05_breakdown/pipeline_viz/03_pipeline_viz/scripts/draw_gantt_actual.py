#!/usr/bin/env python3
"""Draw publication-quality pipeline Gantt chart comparing M=1 and M=3.

Uses AFD_PER_STEP data with real measured sub-stage wall-clock timestamps
(prep_attn, attn, prep_mlp/send, recv_wait/mlp, postprocess).

No hardcoded sub-stage proportions — every bar is drawn from actual CUDA-event
wall-clock anchors.
"""
import json, re, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9

COLORS = {
    'attn_compute': '#1976D2',       # blue - self-attention kernel
    'ffn_compute': '#388E3C',        # green - FFN MLP kernel
    'da_send': '#D32F2F',            # red - send to DF (prep_mlp includes layernorm+send)
    'df_send': '#E64A19',            # deep orange - send to DA
    'da_recv_wait': '#FFA000',       # amber - recv_wait (network + fence)
    'df_recv_wait': '#FBC02D',       # yellow - recv_wait from DA
    'prep_attn': '#7B1FA2',          # purple - input_layernorm + residual_add
    'postprocess': '#00796B',        # teal - residual_add + input_layernorm (for next layer)
    'prep_mlp': '#C62828',           # dark red - post_attention_layernorm + send
    'df_proxy_attn': '#E0E0E0',      # light grey - proxy attention on DF (no-op)
}


def parse_per_step(log_path, m_target):
    """Parse AFD_PER_STEP entries from log, return list of per-iteration step arrays.

    Each step dict has real wall-clock timestamps:
      DA A-stage:  prep_attn_wall_{start,end}_ms, attn_wall_{start,end}_ms, prep_mlp_wall_{start,end}_ms
      DA F-stage:  mlp_wall_{start,end}_ms (recv_wait), postprocess_wall_{start,end}_ms
      DF A-stage:  prep_attn_wall_{start,end}_ms (recv_wait), attn_wall_{start,end}_ms, prep_mlp_wall_{start,end}_ms
      DF F-stage:  mlp_wall_{start,end}_ms (FFN compute), postprocess_wall_{start,end}_ms (send)
    """
    results = []
    pattern = re.compile(r'\[AFD_PER_STEP\].*?M=(\d+)\s+nsteps=(\d+)\s+steps=(\[.*\])')
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m and int(m.group(1)) == m_target:
                try:
                    steps = json.loads(m.group(3))
                    results.append(steps)
                except json.JSONDecodeError:
                    continue
    return results


def parse_host_events(log_path, m_target):
    """Parse AFD_HOST_EVENTS entries — CPU-side time.perf_counter() recv timing.

    Returns list of per-iteration event dicts keyed by (role, layer, mb, event).
    Each recv_end has 'recv_dur_us': actual CPU blocking time in microseconds.
    """
    results = []
    pattern = re.compile(r'\[AFD_HOST_EVENTS\].*?events=(\[.*\])(?:\s|$)')
    afd_pat = re.compile(r'\[AFD_PER_STEP\].*?M=(\d+)')
    with open(log_path) as f:
        lines = f.readlines()

    # Walk lines: match HOST_EVENTS to the preceding PER_STEP to get M value.
    # PER_STEP and HOST_EVENTS are always emitted as a pair per forward pass.
    prev_m = None
    for line in lines:
        m_ps = afd_pat.search(line)
        if m_ps:
            prev_m = int(m_ps.group(1))
        if 'AFD_HOST_EVENTS' in line:
            m = pattern.search(line)
            if m and prev_m == m_target:
                try:
                    events = json.loads(m.group(1))
                    results.append(events)
                except json.JSONDecodeError:
                    continue
    return results


def _wall_key(key, suffix):
    """e.g. _wall_key('prep_attn', 'start') -> 'prep_attn_wall_start_ms'"""
    return f"{key}_wall_{suffix}_ms"


def _get_range(step, key):
    """Get (start_ms, end_ms) wall-clock range for a sub-stage key."""
    sk = _wall_key(key, "start")
    ek = _wall_key(key, "end")
    return step.get(sk), step.get(ek)


def build_recv_lookup(host_sets, iter_idx):
    """Build (layer, mb) -> recv_dur_ms map from host events for a specific iteration."""
    if not host_sets or iter_idx >= len(host_sets):
        return {}
    events = host_sets[iter_idx]
    lookup = {}
    for evt in events:
        if evt.get('role') == 'DA' and evt.get('event') == 'recv_end':
            key = (evt.get('layer', -1), evt.get('mb', 0))
            lookup[key] = evt.get('recv_dur_us', 0) / 1000.0
    return lookup


def _pick_per_step_and_host(sets, host_sets):
    """Pick a representative iteration index and return (per_step_steps, host_recv_lookup)."""
    if not sets:
        return [], {}
    idx = min(5, len(sets) - 1)
    return sets[idx], build_recv_lookup(host_sets, idx)


def draw_combined_gantt(output_path):
    """Draw M=1 vs M=3 combined Gantt chart using real sub-stage timings."""
    da_m1_sets = parse_per_step('/tmp/gantt_DA_m1_detailed.log', 1)
    df_m1_sets = parse_per_step('/tmp/gantt_DF_m1_detailed.log', 1)
    da_m3_sets = parse_per_step('/tmp/gantt_DA_m3_detailed.log', 3)
    df_m3_sets = parse_per_step('/tmp/gantt_DF_m3_detailed.log', 3)
    da_m3_il = parse_per_step('/tmp/gantt_DA_m3_async.log', 3)
    df_m3_il = parse_per_step('/tmp/gantt_DF_m3_async.log', 3)
    da_m3_pf = parse_per_step('/tmp/gantt_DA_m3_prefetch.log', 3)
    df_m3_pf = parse_per_step('/tmp/gantt_DF_m3_prefetch.log', 3)
    da_m3_nl = parse_per_step('/tmp/gantt_DA_m3_nolock.log', 3)
    df_m3_nl = parse_per_step('/tmp/gantt_DF_m3_nolock.log', 3)

    da_m1_host = parse_host_events('/tmp/gantt_DA_m1_detailed.log', 1)
    da_m3_host = parse_host_events('/tmp/gantt_DA_m3_detailed.log', 3)
    da_m3_il_host = parse_host_events('/tmp/gantt_DA_m3_async.log', 3)
    da_m3_pf_host = parse_host_events('/tmp/gantt_DA_m3_prefetch.log', 3)
    da_m3_nl_host = parse_host_events('/tmp/gantt_DA_m3_nolock.log', 3)

    print(f"M=1: DA={len(da_m1_sets)} iters, DF={len(df_m1_sets)} iters, host={len(da_m1_host)}")
    print(f"M=3: DA={len(da_m3_sets)} iters, DF={len(df_m3_sets)} iters")
    print(f"M=3 interleaved: DA={len(da_m3_il)} iters, DF={len(df_m3_il)} iters")
    print(f"M=3 prefetch: DA={len(da_m3_pf)} iters, DF={len(df_m3_pf)} iters")
    print(f"M=3 nolock: DA={len(da_m3_nl)} iters, DF={len(df_m3_nl)} iters")

    da_m1_steps, da_m1_recv = _pick_per_step_and_host(da_m1_sets, da_m1_host)
    df_m1_steps, _ = _pick_per_step_and_host(df_m1_sets, [])
    da_m3_steps, da_m3_recv = _pick_per_step_and_host(da_m3_sets, da_m3_host)
    df_m3_steps, _ = _pick_per_step_and_host(df_m3_sets, [])
    da_m3_il_steps, da_m3_il_recv = _pick_per_step_and_host(da_m3_il, da_m3_il_host)
    df_m3_il_steps, _ = _pick_per_step_and_host(df_m3_il, [])
    da_m3_pf_steps, da_m3_pf_recv = _pick_per_step_and_host(da_m3_pf, da_m3_pf_host)
    df_m3_pf_steps, _ = _pick_per_step_and_host(df_m3_pf, [])
    da_m3_nl_steps, da_m3_nl_recv = _pick_per_step_and_host(da_m3_nl, da_m3_nl_host)
    df_m3_nl_steps, _ = _pick_per_step_and_host(df_m3_nl, [])

    LAYER_START = 0
    LAYER_END = 5
    NUM_LAYERS = LAYER_END - LAYER_START

    fig, axes = plt.subplots(5, 1, figsize=(18, 22))

    configs = [
        (axes[0], da_m1_steps, df_m1_steps, da_m1_recv, 1,
         f'M=1: Sequential Execution (Layers {LAYER_START}–{LAYER_END-1}, batch≈10, Qwen3-32B)'),
        (axes[1], da_m3_steps, df_m3_steps, da_m3_recv, 3,
         f'M=3: Batch Schedule [BEFORE] (Layers {LAYER_START}–{LAYER_END-1}, batch≈10, Qwen3-32B)'),
        (axes[2], da_m3_il_steps, df_m3_il_steps, da_m3_il_recv, 3,
         f'M=3: Interleaved Schedule (Layers {LAYER_START}–{LAYER_END-1}, batch≈10, Qwen3-32B)'),
        (axes[3], da_m3_pf_steps, df_m3_pf_steps, da_m3_pf_recv, 3,
         f'M=3: Interleaved + Prefetch (Layers {LAYER_START}–{LAYER_END-1}, batch≈10, Qwen3-32B)'),
        (axes[4], da_m3_nl_steps, df_m3_nl_steps, da_m3_nl_recv, 3,
         f'M=3: Intrlvd + Prefetch + Per-Slot Stream [AFTER] (Layers {LAYER_START}–{LAYER_END-1}, batch≈10, Qwen3-32B)'),
    ]

    for ax, da_steps, df_steps, da_recv_lookup, m_stage, title in configs:
        if not da_steps or not df_steps:
            ax.set_title(f'{title} - No Data')
            continue

        # Filter to target layers
        da_filt = [s for s in da_steps
                   if LAYER_START <= s.get('layer_id', -1) < LAYER_END]
        df_filt = [s for s in df_steps
                   if LAYER_START <= s.get('layer_id', -1) < LAYER_END]

        if not da_filt or not df_filt:
            ax.set_title(f'{title} - No Data')
            continue

        # Find global t0 for normalization
        all_wall_starts = []
        for s in da_filt + df_filt:
            for key in ('prep_attn', 'attn', 'prep_mlp', 'mlp', 'postprocess'):
                v = s.get(_wall_key(key, 'start'))
                if v is not None:
                    all_wall_starts.append(v)
        t0 = min(all_wall_starts) if all_wall_starts else 0

        # Y positions
        bar_h = 0.38 if m_stage == 1 else 0.20
        gap = 0.06
        if m_stage == 1:
            y_da = {0: 1.0}
            y_df = {0: 0.0}
            y_min, y_max = -0.4, 1.6
        else:
            y_da = {mb: 2.0 + (m_stage - 1 - mb) * (bar_h + gap) for mb in range(m_stage)}
            y_df = {mb: 0.0 + (m_stage - 1 - mb) * (bar_h + gap) for mb in range(m_stage)}
            y_min = -0.3
            y_max = max(y_da.values()) + bar_h + 0.3

        # Build lookup for cross-GPU inference
        da_idx2 = defaultdict(dict)
        for s in da_filt:
            da_idx2[(s['layer_id'], s['mb'])][s['stage']] = s
        df_idx2 = defaultdict(dict)
        for s in df_filt:
            df_idx2[(s['layer_id'], s['mb'])][s['stage']] = s

        # === Draw DA bars (real sub-stage timings) ===
        for step in da_filt:
            mb = step.get('mb', 0)
            layer = step.get('layer_id', -1)
            if mb not in y_da:
                continue
            y = y_da[mb]
            stage = step.get('stage', '')

            if stage == 'A':
                # Real sub-stages: prep_attn | attn | prep_mlp(+send)
                for key, color, edge, alpha, label, fontsize in [
                    ('prep_attn', COLORS['prep_attn'], '#4A148C', 0.88, 'prep', 5),
                    ('attn', COLORS['attn_compute'], '#0D47A1', 0.92, 'Attn L{layer}', 6),
                    ('prep_mlp', COLORS['prep_mlp'], '#B71C1C', 0.85, 'send', 5),
                ]:
                    ts, te = _get_range(step, key)
                    if ts is not None and te is not None:
                        dur = te - ts
                        ax.barh(y, dur, left=ts - t0, height=bar_h,
                                color=color, edgecolor=edge, linewidth=0.4, alpha=alpha)
                        lbl = label.format(layer=layer) if '{layer}' in label else label
                        if key == 'attn' and dur > 0.3:
                            ax.text(ts - t0 + dur / 2, y, lbl,
                                    ha='center', va='center', fontsize=fontsize,
                                    color='white', fontweight='bold')
            elif stage == 'F':
                # CPU recv_sync() blocks the DA thread. CUDA events see only
                # a tiny GPU-idle gap; the real wait is in recv_dur_us from
                # AFD_HOST_EVENTS (time.perf_counter() on the CPU side).
                da_a = da_idx2.get((layer, mb), {}).get('A')
                send_end = da_a.get(_wall_key('prep_mlp', 'end')) if da_a else None
                post_start = step.get(_wall_key('postprocess', 'start'))
                post_end = step.get(_wall_key('postprocess', 'end'))

                # Prefer CPU-measured recv duration; fall back to GPU gap
                cpu_recv_ms = da_recv_lookup.get((layer, mb))
                if cpu_recv_ms is None:
                    cpu_recv_ms = max(0, post_start - send_end) if (send_end is not None and post_start is not None) else 0

                if send_end is not None and cpu_recv_ms > 0:
                    ax.barh(y, cpu_recv_ms, left=send_end - t0, height=bar_h,
                            color=COLORS['da_recv_wait'], edgecolor='#E65100',
                            linewidth=0.4, alpha=0.82)
                    if cpu_recv_ms > 0.3:
                        marker = 'rcv(CPU)' if cpu_recv_ms > 0.5 else 'rcv'
                        ax.text(send_end - t0 + cpu_recv_ms / 2, y, marker,
                                ha='center', va='center', fontsize=5, color='#333')

                # Postprocess: res+LN — GPU-measured duration, positioned after recv_wait
                post_dur_ms = step.get('postprocess_ms', 0)
                if post_dur_ms > 0 and send_end is not None:
                    post_left = send_end - t0 + cpu_recv_ms
                    ax.barh(y, post_dur_ms, left=post_left, height=bar_h,
                            color=COLORS['postprocess'], edgecolor='#004D40',
                            linewidth=0.4, alpha=0.88)
                    if post_dur_ms > 0.15:
                        ax.text(post_left + post_dur_ms / 2, y, 'res+LN',
                                ha='center', va='center', fontsize=5.5, color='white')

        # === Draw DF bars (real sub-stage timings) ===
        for step in df_filt:
            mb = step.get('mb', 0)
            if mb not in y_df:
                continue
            y = y_df[mb]
            stage = step.get('stage', '')

            if stage == 'A':
                # DF A-stage is: prep_attn (recv_wait) | attn (proxy, nearly zero) | prep_mlp
                for key, color, edge, alpha, label, fontsize, fg in [
                    ('prep_attn', COLORS['df_recv_wait'], '#F9A825', 0.82, 'recv', 5.5, '#333'),
                    ('attn', COLORS['df_proxy_attn'], '#BDBDBD', 0.6, '', 0, ''),
                    ('prep_mlp', '#81C784', '#2E7D32', 0.7, '', 0, ''),
                ]:
                    ts, te = _get_range(step, key)
                    if ts is not None and te is not None:
                        dur = te - ts
                        if dur > 0:
                            ax.barh(y, dur, left=ts - t0, height=bar_h,
                                    color=color, edgecolor=edge, linewidth=0.3, alpha=alpha)
                        if key == 'prep_attn' and dur > 0.3:
                            ax.text(ts - t0 + dur / 2, y, label,
                                    ha='center', va='center', fontsize=fontsize, color=fg)

            elif stage == 'F':
                # DF F-stage is: mlp (FFN compute) | postprocess (send back to DA)
                for key, color, edge, alpha, label, fontsize in [
                    ('mlp', COLORS['ffn_compute'], '#1B5E20', 0.92, 'FFN L{layer}', 6),
                    ('postprocess', COLORS['df_send'], '#BF360C', 0.85, 'send', 5),
                ]:
                    ts, te = _get_range(step, key)
                    if ts is not None and te is not None:
                        dur = te - ts
                        layer = step.get('layer_id', -1)
                        ax.barh(y, dur, left=ts - t0, height=bar_h,
                                color=color, edgecolor=edge, linewidth=0.5, alpha=alpha)
                        lbl = label.format(layer=layer) if '{layer}' in label else label
                        if key == 'mlp' and dur > 0.5:
                            ax.text(ts - t0 + dur / 2, y, lbl,
                                    ha='center', va='center', fontsize=fontsize,
                                    color='white', fontweight='bold')
                        elif key == 'mlp' and dur > 0:
                            ax.text(te - t0 + 0.05, y, lbl,
                                    ha='left', va='center', fontsize=5.5,
                                    color='#1B5E20', fontweight='bold')

        # === Build lookup index for arrows ===
        da_idx = defaultdict(dict)  # (layer, mb) -> {stage: step}
        for s in da_filt:
            da_idx[(s['layer_id'], s['mb'])][s['stage']] = s
        df_idx = defaultdict(dict)
        for s in df_filt:
            df_idx[(s['layer_id'], s['mb'])][s['stage']] = s

        # === Draw arrows ===
        for (layer, mb), da_s in da_idx.items():
            if mb not in y_da:
                continue
            df_s = df_idx.get((layer, mb), {})

            # DA→DF arrow: end of DA's prep_mlp (send) → end of DF's prep_attn (recv completes)
            da_a = da_s.get('A')
            df_a = df_s.get('A')
            if da_a and df_a:
                da_send_end = da_a.get(_wall_key('prep_mlp', 'end'))
                df_recv_end = df_a.get(_wall_key('prep_attn', 'end'))
                if da_send_end is not None and df_recv_end is not None:
                    x1 = da_send_end - t0
                    x2 = df_recv_end - t0
                    y1 = y_da[mb] - bar_h * 0.5
                    y2 = y_df[mb] + bar_h * 0.5
                    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                               arrowprops=dict(arrowstyle='->', color='#7B1FA2',
                                              lw=0.8, alpha=0.5,
                                              connectionstyle='arc3,rad=0.2'))

            # DF→DA arrow: end of DF's postprocess (send) → end of DA recv_wait (postprocess starts)
            df_f = df_s.get('F')
            da_f = da_s.get('F')
            da_a = da_s.get('A')
            if df_f and da_f:
                df_send_end = df_f.get(_wall_key('postprocess', 'end'))
                # Align arrow target with the CPU recv position used in the bar
                send_end = da_a.get(_wall_key('prep_mlp', 'end')) if da_a else None
                cpu_recv = da_recv_lookup.get((layer, mb))
                if send_end is not None and cpu_recv is not None:
                    da_post_pos = send_end - t0 + cpu_recv
                else:
                    da_post_pos = da_f.get(_wall_key('postprocess', 'start'))
                    if da_post_pos is not None:
                        da_post_pos -= t0
                if df_send_end is not None and da_post_pos is not None:
                    x1 = df_send_end - t0
                    x2 = da_post_pos
                    y1 = y_df[mb] + bar_h * 0.5
                    y2 = y_da[mb] - bar_h * 0.5
                    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                               arrowprops=dict(arrowstyle='->', color='#E65100',
                                              lw=0.8, alpha=0.6,
                                              connectionstyle='arc3,rad=-0.2'))

        # === Formatting ===
        if m_stage > 1:
            yticks, ylabels = [], []
            for mb in range(m_stage):
                yticks.append(y_da[mb])
                ylabels.append(f'DA mb{mb}')
            for mb in range(m_stage):
                yticks.append(y_df[mb])
                ylabels.append(f'DF mb{mb}')
            ax.set_yticks(yticks)
            ax.set_yticklabels(ylabels, fontsize=9)
            sep_y = (min(y_da.values()) - bar_h + max(y_df.values()) + bar_h) / 2
            ax.axhline(sep_y, color='#78909C', linestyle='--', linewidth=0.8, alpha=0.4)
        else:
            ax.set_yticks([y_df[0], y_da[0]])
            ax.set_yticklabels(['DF (FFN GPU 4)', 'DA (Attn GPU 0)'], fontsize=11)

        ax.set_xlabel('Wall-clock time (ms)', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_ylim(y_min, y_max)
        ax.grid(axis='x', alpha=0.2, linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Timing annotation box
        da_a_sends = []
        recv_waits_cpu = []
        recv_waits_gpu = []
        da_posts = []
        df_ffns = []
        for s in da_filt:
            layer, mb = s.get('layer_id', -1), s.get('mb', 0)
            if s.get('stage') == 'A':
                da_a_sends.append(s.get('attn_ms', 0))
            elif s.get('stage') == 'F':
                da_a = da_idx2.get((layer, mb), {}).get('A')
                send_end = da_a.get(_wall_key('prep_mlp', 'end')) if da_a else None
                post_start = s.get(_wall_key('postprocess', 'start'))
                post_end = s.get(_wall_key('postprocess', 'end'))
                if send_end is not None and post_start is not None:
                    recv_waits_gpu.append(max(0, post_start - send_end))
                cr = da_recv_lookup.get((layer, mb))
                if cr is not None:
                    recv_waits_cpu.append(cr)
                if post_start is not None and post_end is not None:
                    da_posts.append(post_end - post_start)
        for s in df_filt:
            if s.get('stage') == 'F':
                df_ffns.append(s.get('mlp_ms', 0))

        rw_cpu = np.mean(recv_waits_cpu) if recv_waits_cpu else 0
        rw_gpu = np.mean(recv_waits_gpu) if recv_waits_gpu else 0
        dp = np.mean(da_posts) if da_posts else 0
        tf = np.mean(df_ffns) if df_ffns else 0
        total_f = (rw_cpu if rw_cpu > 0 else rw_gpu) + dp
        rw_pct = (rw_cpu if rw_cpu > 0 else rw_gpu) / total_f * 100 if total_f > 0 else 0

        if rw_cpu > 0:
            info = (f"DA attn: {np.mean(da_a_sends):.2f} ms avg\n"
                    f"DA recv(CPU): {rw_cpu:.2f} ms avg\n"
                    f"DA recv(GPU gap): {rw_gpu:.2f} ms\n"
                    f"DA post(res+LN): {dp:.2f} ms\n"
                    f"DF FFN: {tf:.2f} ms\n"
                    f"DA F-total: {total_f:.2f} ms\n"
                    f"Recv/CPU frac: {rw_pct:.0f}%")
        else:
            info = (f"DA attn: {np.mean(da_a_sends):.2f} ms avg\n"
                    f"DA recv(GPU gap): {rw_gpu:.2f} ms avg\n"
                    f"DA post(res+LN): {dp:.2f} ms\n"
                    f"DF FFN: {tf:.2f} ms\n"
                    f"DA F-total: {total_f:.2f} ms\n"
                    f"Recv fraction: {rw_pct:.0f}%")
        ax.text(0.98, 0.95, info, transform=ax.transAxes,
               fontsize=7.5, ha='right', va='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                        edgecolor='#90A4AE', alpha=0.92))

    # Shared legend at top
    legend_patches = [
        mpatches.Patch(color=COLORS['prep_attn'], label='InputLN + ResAdd'),
        mpatches.Patch(color=COLORS['attn_compute'], label='Self-Attention'),
        mpatches.Patch(color=COLORS['prep_mlp'], label='PostAttnLN + Send'),
        mpatches.Patch(color=COLORS['da_recv_wait'], label='Recv Wait (DA waits DF)'),
        mpatches.Patch(color=COLORS['postprocess'], label='ResAdd + InputLN (next)'),
        mpatches.Patch(color=COLORS['ffn_compute'], label='FFN Compute (DF)'),
        mpatches.Patch(color=COLORS['df_send'], label='Send (DF→DA)'),
        mpatches.Patch(color=COLORS['df_recv_wait'], label='DF Recv Wait'),
    ]
    fig.legend(handles=legend_patches, loc='upper center', ncol=4, fontsize=9,
              bbox_to_anchor=(0.5, 0.995), framealpha=0.95)

    plt.tight_layout(rect=[0, 0, 1, 0.955])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()


def main():
    out_dir = '/workspace/sglang/benchmark/af_bench/03_pipeline_viz/charts'
    import os
    os.makedirs(out_dir, exist_ok=True)

    # Check if detailed logs exist
    for f in ['/tmp/gantt_DA_m1_detailed.log', '/tmp/gantt_DF_m1_detailed.log',
              '/tmp/gantt_DA_m3_detailed.log', '/tmp/gantt_DF_m3_detailed.log']:
        if not os.path.exists(f):
            print(f"Warning: {f} not found")

    draw_combined_gantt(f'{out_dir}/gantt_m1_vs_m3_combined.png')


if __name__ == "__main__":
    main()
