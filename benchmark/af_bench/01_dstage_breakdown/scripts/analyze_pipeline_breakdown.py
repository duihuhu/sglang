#!/usr/bin/env python3
"""Parse AFD_HOST_EVENTS + AFD_SCHED_TS logs and compute pipeline breakdown.

Usage:
  python analyze_pipeline_breakdown.py <da_log> <df_log> [--first-only]
"""

import json, re, sys
from collections import defaultdict

def extract_afd_host_events(log_path):
    """Extract [AFD_HOST_EVENTS] lines from log file."""
    events = []
    sched_ts = []
    pattern_host = re.compile(r'\[AFD_HOST_EVENTS\]\s+role=(\S+).*?events=(\[.*\])')
    pattern_sched = re.compile(r'\[AFD_SCHED_TS\]\s+sched_timestamps=(\{.*?\})')

    with open(log_path) as f:
        for line in f:
            m = pattern_host.search(line)
            if m:
                role = m.group(1)
                events_json = m.group(2)
                try:
                    events.append({"role": role, "events": json.loads(events_json)})
                except json.JSONDecodeError:
                    pass
            m = pattern_sched.search(line)
            if m:
                try:
                    sched_ts.append(json.loads(m.group(1)))
                except json.JSONDecodeError:
                    pass
    return events, sched_ts


def analyze_first_iteration(da_events, df_events, da_sched, df_sched):
    """Break down the first iteration's DA zmq_send → DF recv gap."""
    print("=" * 80)
    print("  BREAKDOWN #1: First Iteration (L0.0) DA→DF Gap")
    print("=" * 80)

    # Scheduler-level timeline
    if da_sched:
        print("\n--- DA Scheduler Timestamps (ms) ---")
        for ts in da_sched[:3]:
            print(f"  {json.dumps(ts, indent=2)}")

    if df_sched:
        print("\n--- DF Scheduler Timestamps (ms) ---")
        for ts in df_sched[:3]:
            print(f"  {json.dumps(ts, indent=2)}")

    # Align first DA and DF iterations
    if da_sched and df_sched:
        da0 = da_sched[0]
        df0 = df_sched[0]

        da_zmq = da0.get("zmq_sent")
        da_fwd_start = da0.get("forward_start")
        da_fwd_end = da0.get("forward_end")
        df_zmq = df0.get("zmq_recv")
        df_fwd_start = df0.get("forward_start")
        df_fwd_end = df0.get("forward_end")

        print("\n--- Cross-GPU Timeline (all times in ms) ---")
        if da_zmq and df_zmq:
            gap_zmq = df_zmq - da_zmq
            print(f"  DA zmq_sent         = {da_zmq:.3f}")
            print(f"  DF zmq_recv         = {df_zmq:.3f}")
            print(f"  >> ZMQ round-trip   = {gap_zmq:.3f} ms  (DA scheduler → DF scheduler)")

        if df_zmq and df_fwd_start:
            gap_setup = df_fwd_start - df_zmq
            print(f"  DF forward_start    = {df_fwd_start:.3f}")
            print(f"  >> DF batch setup   = {gap_setup:.3f} ms  (zmq_recv → forward_start)")

        if da_fwd_start and df_fwd_start:
            gap_fwd = df_fwd_start - da_fwd_start
            print(f"  >> DA→DF forward    = {gap_fwd:.3f} ms  (DA forward_start → DF forward_start)")

        if da_zmq and df_fwd_start:
            total = df_fwd_start - da_zmq
            print(f"\n  ** TOTAL zmq_sent→DF forward = {total:.3f} ms **")

    # Host events breakdown
    print("\n--- DA Host Events (first iteration) ---")
    for ev in da_events[:1]:
        for e in ev["events"]:
            if e["role"] == "DA":
                print(f"  L{e['layer']}.{e['mb']} {e['event']}: {e['ts_ms']:.3f} ms | stage={e['stage']}")

    print("\n--- DF Host Events (first iteration) ---")
    for ev in df_events[:1]:
        for e in ev["events"]:
            if e["role"] == "DF":
                print(f"  L{e['layer']}.{e['mb']} {e['event']}: {e['ts_ms']:.3f} ms | stage={e['stage']}")

    # UCX-level events
    print("\n--- UCX Send Thread Events (first iteration) ---")
    for ev in da_events[:1]:
        for e in ev["events"]:
            if e["role"] == "SENDER":
                print(f"  {e['event']}: {e['ts_ms']:.3f} ms | dur={e.get('ucx_send_dur_us','?')}")

    print("\n--- UCX Recv Events (first iteration) ---")
    for ev in df_events[:1]:
        for e in ev["events"]:
            if e["role"] == "RECVER":
                print(f"  {e['event']}: {e['ts_ms']:.3f} ms | dur={e.get('ucx_recv_dur_us','?')}")


def analyze_per_mb_gap(da_events, df_events):
    """Analyze per-micro-batch DA send → DF recv gap."""
    print("\n" + "=" * 80)
    print("  BREAKDOWN #2: Per-MicroBatch DA send→DF recv Gap")
    print("=" * 80)

    # Build timeline: DA send events and DF recv events by (layer, mb)
    da_sends = []  # (layer, mb, send_start_ms, send_end_ms)
    df_recvs = []  # (layer, mb, recv_start_ms, recv_end_ms)

    for ev_set in da_events:
        for e in ev_set["events"]:
            if e["role"] == "DA" and e["event"] == "send_start":
                da_sends.append((e["layer"], e["mb"], e["ts_ms"], None))
            elif e["role"] == "DA" and e["event"] == "send_end":
                # Update the matching send_start
                for i, (l, m, ss, se) in enumerate(da_sends):
                    if l == e["layer"] and m == e["mb"] and se is None:
                        da_sends[i] = (l, m, ss, e["ts_ms"])
                        break

    for ev_set in df_events:
        for e in ev_set["events"]:
            if e["role"] == "DF" and e["event"] == "recv_start":
                df_recvs.append((e["layer"], e["mb"], e["ts_ms"], None))
            elif e["role"] == "DF" and e["event"] == "recv_end":
                for i, (l, m, rs, re) in enumerate(df_recvs):
                    if l == e["layer"] and m == e["mb"] and re is None:
                        df_recvs[i] = (l, m, rs, e["ts_ms"])
                        break

    # Match DA sends with DF recvs
    print(f"\n  {'Layer.MB':<10} {'DA send(ms)':<14} {'DF recv(ms)':<14} {'Gap(ms)':<10} {'Send dur(us)':<12} {'Recv dur(us)':<12}")
    print("  " + "-" * 75)

    gaps = []
    for (l, m, ss, se) in sorted(da_sends):
        # Find matching DF recv
        match = None
        for (l2, m2, rs, re) in df_recvs:
            if l == l2 and m == m2 and rs is not None:
                match = (rs, re)
                break

        if match and ss:
            rs, re = match
            gap = rs - ss
            send_dur = (se - ss) * 1e3 if se else 0
            recv_dur = (re - rs) * 1e3 if re else 0
            gaps.append((l, m, gap, send_dur, recv_dur))
            print(f"  L{l}.{m:<7} {ss:<14.3f} {rs:<14.3f} {gap:<10.3f} {send_dur:<12.1f} {recv_dur:<12.1f}")

    if gaps:
        avg_gap = sum(g[2] for g in gaps) / len(gaps)
        min_gap = min(g[2] for g in gaps)
        max_gap = max(g[2] for g in gaps)
        print(f"\n  Summary: avg_gap={avg_gap:.3f} ms, min={min_gap:.3f} ms, max={max_gap:.3f} ms")
        print(f"  Total pairs matched: {len(gaps)}")

    return gaps


def analyze_ucx_detail(da_events, df_events):
    """Analyze UCX-level timing details."""
    print("\n" + "=" * 80)
    print("  BREAKDOWN #3: UCX Send/Recv Wire Time")
    print("=" * 80)

    ucx_sends = []  # (send_launch_ms, cuda_sync_done_ms, ucx_send_done_ms)
    ucx_recvs = []  # (recv_done_ms, recv_dur_us)

    for ev_set in da_events:
        launch = sync = send = None
        for e in ev_set["events"]:
            if e["role"] == "SENDER":
                if e["event"] == "ucx_send_launched":
                    launch = e["ts_ms"]
                elif e["event"] == "ucx_cuda_sync_done":
                    sync = e["ts_ms"]
                elif e["event"] == "ucx_send_done":
                    send = e["ts_ms"]
        if launch and sync and send:
            ucx_sends.append((launch, sync, send))

    for ev_set in df_events:
        for e in ev_set["events"]:
            if e["role"] == "RECVER" and e["event"] == "ucx_recv_done":
                ucx_recvs.append((e["ts_ms"], e.get("ucx_recv_dur_us", 0)))

    print(f"\n  UCX Send Thread Breakdown (first 10):")
    print(f"  {'Event':<30} {'Time(ms)':<14} {'Delta(us)':<10}")
    for i, (launch, sync, send) in enumerate(ucx_sends[:10]):
        cuda_wait = (sync - launch) * 1e3  # us
        ucx_send = (send - sync) * 1e3     # us
        total = (send - launch) * 1e3       # us
        print(f"  {'  thread launched':<30} {launch:<14.3f}")
        print(f"  {'  cuda sync done':<30} {sync:<14.3f} {cuda_wait:<10.1f}")
        print(f"  {'  ucx send done':<30} {send:<14.3f} {ucx_send:<10.1f}")
        print(f"  {'  TOTAL daemon thread':<30} {'':14} {total:<10.1f}")
        if i < len(ucx_recvs):
            print(f"  DF ucx_recv_done: {ucx_recvs[i][0]:.3f} ms  (dur={ucx_recvs[i][1]:.1f} us)")
        print()

    if ucx_sends:
        avg_cuda = sum((s - l) * 1e3 for l, s, _ in ucx_sends) / len(ucx_sends)
        avg_ucx = sum((d - s) * 1e3 for _, s, d in ucx_sends) / len(ucx_sends)
        avg_total = sum((d - l) * 1e3 for l, _, d in ucx_sends) / len(ucx_sends)
        print(f"  Averages: CUDA fence={avg_cuda:.1f} us, UCX send={avg_ucx:.1f} us, Total thread={avg_total:.1f} us")


def main():
    if len(sys.argv) < 3:
        print("Usage: python analyze_pipeline_breakdown.py <da_log> <df_log> [--first-only]")
        sys.exit(1)

    da_log = sys.argv[1]
    df_log = sys.argv[2]
    first_only = "--first-only" in sys.argv

    print(f"Parsing DA log: {da_log}")
    da_events, da_sched = extract_afd_host_events(da_log)
    print(f"  Found {len(da_events)} AFD_HOST_EVENTS entries, {len(da_sched)} AFD_SCHED_TS entries")

    print(f"Parsing DF log: {df_log}")
    df_events, df_sched = extract_afd_host_events(df_log)
    print(f"  Found {len(df_events)} AFD_HOST_EVENTS entries, {len(df_sched)} AFD_SCHED_TS entries")

    if not da_events and not df_events:
        print("\nNo AFD_HOST_EVENTS found! Check that modified code is running.")
        sys.exit(1)

    analyze_first_iteration(da_events, df_events, da_sched, df_sched)

    if not first_only:
        analyze_per_mb_gap(da_events, df_events)
        analyze_ucx_detail(da_events, df_events)


if __name__ == "__main__":
    main()
