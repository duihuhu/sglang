#!/usr/bin/env python3
"""Parse AFD_HOST_EVENTS + AFD_SCHED_TS and produce detailed gap breakdown.

Output:
  1. Scheduler-level ZMQ latency per iteration
  2. Per-(layer, mb) DA→DF UCX transfer breakdown (matched send→recv pairs)
  3. UCX daemon thread timing (CUDA fence + send + wire)
  4. Timeline with aligned DA/DF send/recv events for Gantt-like text visualization
"""

import json, re, sys
from collections import defaultdict

def extract_afd_host_events(log_path):
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


def compute_zmq_latency(da_sched, df_sched):
    """Print per-iteration ZMQ latency breakdown."""
    print("=" * 85)
    print("  SCHEDULER ZMQ LATENCY BREAKDOWN")
    print("=" * 85)
    print(f"  {'Iter':<6} {'DA zmq_sent':<14} {'DF zmq_recv':<14} {'ZMQ(ms)':<10} "
          f"{'DA fwd_start→':<14} {'DF fwd_start→':<14}")
    print("  " + "-" * 82)

    latencies = []
    for i in range(min(len(da_sched), len(df_sched))):
        da = da_sched[i]
        df = df_sched[i]
        zsent = da.get("zmq_sent")
        zrecv = df.get("zmq_recv")
        if zsent and zrecv:
            lat = zrecv - zsent
            latencies.append(lat)
            da_setup = da.get("forward_start", 0) - zsent
            df_setup = df.get("forward_start", 0) - zrecv
            if i < 5 or i == min(len(da_sched), len(df_sched)) - 1:
                print(f"  {i:<6} {zsent:<14.3f} {zrecv:<14.3f} {lat:<10.3f} "
                      f"{da_setup:<14.3f} {df_setup:<14.3f}")
            elif i == 5:
                print(f"  {'...':<6}")

    if latencies:
        avg = sum(latencies) / len(latencies)
        mn = min(latencies)
        mx = max(latencies)
        print(f"\n  ZMQ latency: avg={avg:.3f} ms  min={mn:.3f} ms  max={mx:.3f} ms")
        print(f"  First iter ZMQ: {latencies[0]:.3f} ms")


def match_send_recv_pairs(da_events, df_events):
    """Match DA send events with DF recv events by (layer, mb) and temporal proximity.

    Returns list of (layer, mb, send_start_ms, send_end_ms, recv_end_ms, recv_dur_us, ucx_send_us) tuples.
    """
    # Build DA sends: list of (layer, mb, send_start, send_end, stage, event_idx)
    da_sends = []
    for ev_idx, ev_set in enumerate(da_events):
        pending = {}  # (layer, mb) -> send_start
        for e in ev_set["events"]:
            if e["role"] == "DA" and e["event"] == "send_start":
                key = (e["layer"], e["mb"])
                pending[key] = e["ts_ms"]
            elif e["role"] == "DA" and e["event"] == "send_end":
                key = (e["layer"], e["mb"])
                if key in pending:
                    da_sends.append({
                        "layer": e["layer"], "mb": e["mb"],
                        "send_start": pending[key],
                        "send_end": e["ts_ms"],
                        "stage": e["stage"],
                    })
                    del pending[key]

    # Build DF recvs
    df_recvs = []
    for ev_idx, ev_set in enumerate(df_events):
        pending = {}
        for e in ev_set["events"]:
            if e["role"] == "DF" and e["event"] == "recv_start":
                key = (e["layer"], e["mb"])
                pending[key] = e["ts_ms"]
            elif e["role"] == "DF" and e["event"] == "recv_end":
                key = (e["layer"], e["mb"])
                if key in pending:
                    df_recvs.append({
                        "layer": e["layer"], "mb": e["mb"],
                        "recv_start": pending[key],
                        "recv_end": e["ts_ms"],
                        "recv_dur_us": e.get("recv_dur_us", 0),
                        "stage": e["stage"],
                    })
                    del pending[key]

    # Build UCX send timing lookup
    ucx_sends = []
    for ev_set in da_events:
        launch = cuda_sync = send_done = None
        for e in ev_set["events"]:
            if e["role"] == "SENDER":
                if e["event"] == "ucx_send_launched":
                    launch = e["ts_ms"]
                elif e["event"] == "ucx_cuda_sync_done":
                    cuda_sync = e["ts_ms"]
                elif e["event"] == "ucx_send_done":
                    send_done = e["ts_ms"]
                    if launch and cuda_sync:
                        ucx_sends.append({
                            "launch": launch, "cuda_sync": cuda_sync,
                            "ucx_done": send_done,
                            "cuda_us": (cuda_sync - launch) * 1e3,
                            "send_us": (send_done - cuda_sync) * 1e3,
                        })
                    launch = cuda_sync = send_done = None

    # Build UCX recv timing lookup
    ucx_recvs = []
    for ev_set in df_events:
        for e in ev_set["events"]:
            if e["role"] == "RECVER":
                ucx_recvs.append({
                    "ts_ms": e["ts_ms"],
                    "dur_us": e.get("ucx_recv_dur_us", 0),
                })

    # Match DA sends with DF recvs: for each (layer, mb), pair by sequential order
    pairs = []
    # Group sends by (layer, mb)
    sends_by_key = defaultdict(list)
    for s in da_sends:
        sends_by_key[(s["layer"], s["mb"])].append(s)
    recvs_by_key = defaultdict(list)
    for r in df_recvs:
        recvs_by_key[(r["layer"], r["mb"])].append(r)

    for key in sorted(sends_by_key.keys()):
        ss = sends_by_key[key]
        rr = recvs_by_key.get(key, [])
        for i in range(min(len(ss), len(rr))):
            s = ss[i]
            r = rr[i]
            gap = r["recv_end"] - s["send_start"]  # total DA→DF gap
            pairs.append({
                "layer": key[0], "mb": key[1],
                "send_start": s["send_start"],
                "send_end": s["send_end"],
                "recv_end": r["recv_end"],
                "total_gap_ms": gap,
                "send_dur_us": (s["send_end"] - s["send_start"]) * 1e3,
                "recv_dur_us": r["recv_dur_us"],
            })

    return pairs, ucx_sends, ucx_recvs


def analyze_gap_breakdown(pairs, ucx_sends, ucx_recvs):
    """Detailed breakdown: per-MB-gap statistics and UCX timing."""
    print("\n" + "=" * 85)
    print("  PER (LAYER, MB) DA→DF UCX TRANSFER BREAKDOWN")
    print("=" * 85)

    if not pairs:
        print("  No matched pairs found!")
        return

    # Filter out obvious outliers (first few pairs with huge gaps from cold start)
    gaps_sorted = sorted(pairs, key=lambda p: p["total_gap_ms"])
    n = len(gaps_sorted)
    p50_idx = int(n * 0.5)
    p90_idx = int(n * 0.9)
    p99_idx = int(n * 0.99)

    print(f"\n  Statistics (n={n} matched pairs):")
    print(f"  {'Metric':<25} {'Value':<15}")
    print(f"  {'-' * 40}")
    print(f"  {'P50 total gap':<25} {gaps_sorted[p50_idx]['total_gap_ms']:<15.3f} ms")
    print(f"  {'P90 total gap':<25} {gaps_sorted[p90_idx]['total_gap_ms']:<15.3f} ms")
    print(f"  {'P99 total gap':<25} {gaps_sorted[p99_idx]['total_gap_ms']:<15.3f} ms")
    print(f"  {'Min total gap':<25} {gaps_sorted[0]['total_gap_ms']:<15.3f} ms")
    print(f"  {'Max total gap':<25} {gaps_sorted[-1]['total_gap_ms']:<15.3f} ms")

    # Show detailed per-layer breakdown for the first few iterations
    print(f"\n  First 30 matched pairs (chronological order):")
    print(f"  {'Layer.MB':<10} {'DA send_start':<15} {'DA send_end':<15} "
          f"{'DF recv_end':<15} {'Total gap(ms)':<13} {'Send dur(us)':<12} {'Recv dur(us)':<12}")
    print(f"  {'-' * 100}")
    for p in pairs[:30]:
        print(f"  L{p['layer']}.{p['mb']:<7} {p['send_start']:<15.3f} {p['send_end']:<15.3f} "
              f"{p['recv_end']:<15.3f} {p['total_gap_ms']:<13.3f} "
              f"{p['send_dur_us']:<12.1f} {p['recv_dur_us']:<12.1f}")

    # UCX daemon thread analysis
    if ucx_sends:
        print(f"\n  UCX Send Daemon Thread Timing (n={len(ucx_sends)}):")
        avg_cuda = sum(s["cuda_us"] for s in ucx_sends) / len(ucx_sends)
        avg_ucx = sum(s["send_us"] for s in ucx_sends) / len(ucx_sends)
        avg_total = avg_cuda + avg_ucx
        print(f"  {'CUDA fence (GPU→CPU sync)':<35} {avg_cuda:>8.1f} μs")
        print(f"  {'UCX send_tensor_nonblocking':<35} {avg_ucx:>8.1f} μs")
        print(f"  {'Total daemon thread (host)':<35} {avg_total:>8.1f} μs")

    if ucx_recvs:
        avg_recv = sum(r["dur_us"] for r in ucx_recvs) / len(ucx_recvs)
        print(f"\n  UCX Recv Timing (n={len(ucx_recvs)}):")
        print(f"  {'Avg recv_tensor() duration':<35} {avg_recv:>8.1f} μs")


def analyze_mb_pipeline_gaps(pairs):
    """Analyze per-MB gaps for M>1 to show whether MBs are pipelined efficiently."""
    print("\n" + "=" * 85)
    print("  PER-MICROBATCH PIPELINE EFFICIENCY")
    print("=" * 85)

    # Group by mb
    by_mb = defaultdict(list)
    for p in pairs:
        by_mb[p["mb"]].append(p["total_gap_ms"])

    for mb in sorted(by_mb.keys()):
        gaps = by_mb[mb]
        if gaps:
            avg = sum(gaps) / len(gaps)
            print(f"  MB={mb}: avg_gap={avg:.3f} ms  n={len(gaps)}  "
                  f"min={min(gaps):.3f}  max={max(gaps):.3f}")

    # Check if MB0 (first in pipeline) has larger gaps than subsequent MBs
    mb_keys = sorted(by_mb.keys())
    if len(mb_keys) >= 2:
        print(f"\n  MB0 vs MB1+ comparison:")
        mb0_gaps = by_mb[mb_keys[0]]
        mb1plus_gaps = [g for k in mb_keys[1:] for g in by_mb[k]]
        if mb0_gaps and mb1plus_gaps:
            print(f"    MB{mb_keys[0]}:  avg={sum(mb0_gaps)/len(mb0_gaps):.3f} ms")
            print(f"    MB{'+'.join(str(k) for k in mb_keys[1:])}: avg={sum(mb1plus_gaps)/len(mb1plus_gaps):.3f} ms")


def main():
    if len(sys.argv) < 3:
        print("Usage: python analyze_pipeline_breakdown_v2.py <da_log> <df_log>")
        sys.exit(1)

    da_log = sys.argv[1]
    df_log = sys.argv[2]

    print(f"Parsing DA: {da_log}")
    da_events, da_sched = extract_afd_host_events(da_log)
    print(f"  {len(da_events)} HOST_EVENTS, {len(da_sched)} SCHED_TS")

    print(f"Parsing DF: {df_log}")
    df_events, df_sched = extract_afd_host_events(df_log)
    print(f"  {len(df_events)} HOST_EVENTS, {len(df_sched)} SCHED_TS")

    # 1. ZMQ latency
    compute_zmq_latency(da_sched, df_sched)

    # 2. Match send/recv pairs
    pairs, ucx_sends, ucx_recvs = match_send_recv_pairs(da_events, df_events)
    print(f"\n  Matched {len(pairs)} DA-send→DF-recv pairs")

    # 3. Gap breakdown
    analyze_gap_breakdown(pairs, ucx_sends, ucx_recvs)

    # 4. Per-MB analysis
    analyze_mb_pipeline_gaps(pairs)

    print("\n" + "=" * 85)
    print("  SUMMARY")
    print("=" * 85)

    # Filter extreme outliers (>3x median) to get steady-state numbers
    if pairs:
        gaps = sorted([p["total_gap_ms"] for p in pairs])
        median = gaps[len(gaps) // 2]
        steady = [g for g in gaps if g < median * 3]

        if steady:
            print(f"  Steady-state DA→DF per-(layer,mb) transfer: "
                  f"median={sorted(steady)[len(steady)//2]:.3f} ms, "
                  f"mean={sum(steady)/len(steady):.3f} ms")
        if ucx_sends:
            avg_ucx_total = sum(s["cuda_us"]+s["send_us"] for s in ucx_sends) / len(ucx_sends)
            avg_ucx_wire = sum(s["send_us"] for s in ucx_sends) / len(ucx_sends)
            print(f"  UCX daemon thread: {avg_ucx_total:.0f} μs total")
            print(f"    └─ CUDA fence: {sum(s['cuda_us'] for s in ucx_sends)/len(ucx_sends):.0f} μs")
            print(f"    └─ UCX send:   {avg_ucx_wire:.0f} μs")


if __name__ == "__main__":
    main()
