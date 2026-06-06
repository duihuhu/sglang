#!/usr/bin/env python3
"""
Minimal PD+AF M=1 test: 10 requests, decode-only bubble breakdown.

Launches PD+AF (4 modules), sends 10 requests with IL=1024 OL=32,
then parses DA/DF logs to breakdown per-layer bubble into:
  - Python dispatch overhead (host-side scheduling between stages)
  - CUDA fence (event.synchronize before UCX send)
  - UCX send submission (actual UCX API call)
  - Wire transfer (network latency)
  - UCX recv (blocking recv_tensor or poll+copy)
  - GPU compute (attn / mlp kernel time)
"""

import json, logging, os, re, socket, subprocess, sys, time
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("bubble")

PYTHON = "/workspace/env/af-test/bin/python"
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent.parent
_LAUNCHER = _REPO / "python" / "sglang" / "srt" / "energy" / "af_launcher.py"
_CONFIG = _REPO / "python" / "sglang" / "srt" / "energy" / "af_launch_config.json"
_BENCHMARK = _HERE.parent / "benchmark_replay.py"
LOG_DIR = _HERE / "bubble_logs"


def kill_all():
    for t in ["sglang.launch_server", "sglang_router", "af_launcher"]:
        subprocess.run(["pkill", "-9", "-f", t], capture_output=True)
    time.sleep(3)


def wait_health(url, timeout=180):
    import urllib.request
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=5)
            return True
        except Exception:
            time.sleep(2)
    return False


def launch_pdaf():
    """Launch PD+AF with timing instrumentation enabled."""
    cfg = json.loads(_CONFIG.read_text())
    cfg["model"]["mem_fraction_static"] = 0.93

    for mod in cfg["modules"]:
        mod["extra_cli_args"] = [
            "--skip-server-warmup",
            "--disable-cuda-graph",
            "--disable-piecewise-cuda-graph",
        ]
    for mod in cfg["modules"]:
        if mod["name"] in ("DF", "DA"):
            mod["extra_cli_args"] += [
                "--afd-micro-batch", "1",
                "--max-running-requests", "16",
            ]
            mod["ucx_base_port"] = 25200
        elif mod["name"] in ("PF", "PA"):
            mod["ucx_base_port"] = 25100

    cfg["afd"]["comm_backend"] = "ucx"
    cfg["afd"]["dvfs_enabled"] = False
    cfg["tier1"]["enable_tier1_pa"] = False

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_subdir = LOG_DIR / "logs"
    log_subdir.mkdir(exist_ok=True)
    cfg["logs"]["dir"] = str(log_subdir)

    tmp_cfg = LOG_DIR / "config.json"
    tmp_cfg.write_text(json.dumps(cfg, indent=2))

    env = os.environ.copy()
    env["AFD_DETAILED_TIMING"] = "1"
    env["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"

    proc = subprocess.Popen(
        [PYTHON, str(_LAUNCHER), "--config", str(tmp_cfg)],
        stdout=open(LOG_DIR / "launcher_stdout.log", "w"),
        stderr=subprocess.STDOUT,
        env=env,
    )
    return proc


def run_benchmark():
    """Send 10 requests with IL=1024 OL=32 at qps=2."""
    cmd = [
        PYTHON, str(_BENCHMARK),
        "--url", "http://127.0.0.1:50000",
        "--dataset", "sample",
        "--sample-num", "10",
        "--max-requests", "10",
        "--sample-input-len", "1024",
        "--sample-output-len", "32",
        "--sample-qps", "2",
        "--concurrency", "10",
        "--timeout", "300",
        "--dump", str(LOG_DIR / "results.json"),
    ]
    log.info("Benchmark: %s", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    log.info("Benchmark stdout (last 2000 chars):\n%s", r.stdout[-2000:])
    if r.returncode != 0:
        log.error("Benchmark stderr:\n%s", r.stderr[-2000:])


def parse_host_events(log_path):
    """Extract AFD_HOST_EVENTS from a module log."""
    all_events = []
    pattern = re.compile(r'\[AFD_HOST_EVENTS\]\s+role=(\S+).*?events=(\[.*\])')
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                role = m.group(1)
                try:
                    evts = json.loads(m.group(2))
                    for e in evts:
                        e["_role_header"] = role
                    all_events.extend(evts)
                except json.JSONDecodeError:
                    pass
    return all_events


def analyze_bubble(da_events, df_events):
    """
    For each layer in each forward pass, compute the bubble breakdown.

    Key timeline for one layer (M=1):
      DA: send_start → send_end (includes thread launch)
      DF: recv_start → recv_end (blocking UCX recv or poll+copy)
      DF: [FFN compute]
      DF: send_start → send_end
      DA: recv_start → recv_end

    Bubble for DA = time between DA.send_end and DA.recv_end (waiting for DF)
    Bubble for DF = time between DF.send_end(prev layer) and DF.recv_end(this layer)
    """
    da_by_layer = defaultdict(list)
    df_by_layer = defaultdict(list)

    for e in da_events:
        layer = e.get("layer", -1)
        if layer >= 0:
            da_by_layer[layer].append(e)
    for e in df_events:
        layer = e.get("layer", -1)
        if layer >= 0:
            df_by_layer[layer].append(e)

    print("\n" + "=" * 100)
    print("  PER-LAYER BUBBLE BREAKDOWN (DA→DF→DA round-trip)")
    print("=" * 100)

    # Collect UCX profile events for detailed breakdown
    ucx_profiles_da = [e for e in da_events if e.get("role") == "UCX_PROFILE"]
    ucx_profiles_df = [e for e in df_events if e.get("role") == "UCX_PROFILE"]

    # Group DA events by (layer, stage, event)
    da_send_a = {}  # layer -> {send_start_ms, send_end_ms, send_dur_us}
    da_recv_f = {}  # layer -> {recv_start_ms, recv_end_ms, recv_dur_us}
    df_recv_a = {}  # layer -> {recv_start_ms, recv_end_ms, recv_dur_us}
    df_send_f = {}  # layer -> {send_start_ms, send_end_ms, send_dur_us}

    for e in da_events:
        layer = e.get("layer", -1)
        if layer < 0:
            continue
        event = e.get("event", "")
        stage = e.get("stage", "")
        if event == "send_start" and stage == "A":
            da_send_a.setdefault(layer, {})["start_ms"] = e["ts_ms"]
        elif event == "send_end" and stage == "A":
            da_send_a.setdefault(layer, {})["end_ms"] = e["ts_ms"]
            da_send_a[layer]["dur_us"] = e.get("send_dur_us", 0)
        elif event == "recv_start" and stage == "F":
            da_recv_f.setdefault(layer, {})["start_ms"] = e["ts_ms"]
        elif event == "recv_end" and stage == "F":
            da_recv_f.setdefault(layer, {})["end_ms"] = e["ts_ms"]
            da_recv_f[layer]["dur_us"] = e.get("recv_dur_us", 0)

    for e in df_events:
        layer = e.get("layer", -1)
        if layer < 0:
            continue
        event = e.get("event", "")
        stage = e.get("stage", "")
        if event == "recv_start" and stage == "A":
            df_recv_a.setdefault(layer, {})["start_ms"] = e["ts_ms"]
        elif event == "recv_end" and stage == "A":
            df_recv_a.setdefault(layer, {})["end_ms"] = e["ts_ms"]
            df_recv_a[layer]["dur_us"] = e.get("recv_dur_us", 0)
        elif event == "send_start" and stage == "F":
            df_send_f.setdefault(layer, {})["start_ms"] = e["ts_ms"]
        elif event == "send_end" and stage == "F":
            df_send_f.setdefault(layer, {})["end_ms"] = e["ts_ms"]
            df_send_f[layer]["dur_us"] = e.get("send_dur_us", 0)

    # UCX profile: send_async_detail per layer
    ucx_send_by_layer = defaultdict(list)
    for e in ucx_profiles_da:
        if e.get("event") == "send_async_detail":
            ucx_send_by_layer[e.get("layer", -1)].append(e)

    ucx_recv_by_layer = defaultdict(list)
    for e in ucx_profiles_df:
        if e.get("event") in ("recv_start_detail", "recv_sync_detail"):
            ucx_recv_by_layer[e.get("layer", -1)].append(e)

    # recv_wait detail on DA side (DF→DA)
    da_recv_wait_by_layer = defaultdict(list)
    for e in [ev for ev in da_events if ev.get("role") == "UCX_PROFILE"]:
        if e.get("event") == "recv_wait_detail":
            da_recv_wait_by_layer[e.get("layer", -1)].append(e)

    # Print header
    print(f"  {'Layer':<6} {'DA_send':<10} {'DF_recv':<10} {'DF_FFN':<10} "
          f"{'DF_send':<10} {'DA_recv':<10} {'Total':<10} "
          f"{'UCX_cuda_sync':<14} {'UCX_submit':<12} {'UCX_recv_inner':<15}")
    print("  " + "-" * 110)

    layers_data = []
    for layer in sorted(set(da_send_a.keys()) & set(df_recv_a.keys()) & set(df_send_f.keys()) & set(da_recv_f.keys())):
        da_s = da_send_a[layer]
        df_r = df_recv_a[layer]
        df_s = df_send_f[layer]
        da_r = da_recv_f[layer]

        da_send_ms = da_s.get("dur_us", 0) / 1000
        df_recv_ms = df_r.get("dur_us", 0) / 1000
        df_ffn_ms = (df_s.get("start_ms", 0) - df_r.get("end_ms", 0)) if df_r.get("end_ms") and df_s.get("start_ms") else 0
        df_send_ms = df_s.get("dur_us", 0) / 1000
        da_recv_ms = da_r.get("dur_us", 0) / 1000

        total_ms = da_send_ms + df_recv_ms + df_ffn_ms + df_send_ms + da_recv_ms

        # UCX detail
        ucx_cuda_sync = 0
        ucx_submit = 0
        if layer in ucx_send_by_layer and ucx_send_by_layer[layer]:
            e = ucx_send_by_layer[layer][0]
            ucx_cuda_sync = e.get("cuda_sync_us", 0) / 1000
            ucx_submit = e.get("ucx_submit_us", 0) / 1000

        ucx_recv_inner = 0
        if layer in ucx_recv_by_layer and ucx_recv_by_layer[layer]:
            e = ucx_recv_by_layer[layer][0]
            ucx_recv_inner = e.get("recv_inner_us", 0) / 1000
            if not ucx_recv_inner:
                ucx_recv_inner = e.get("ucx_recv_dur_us", 0) / 1000

        print(f"  {layer:<6} {da_send_ms:<10.3f} {df_recv_ms:<10.3f} {df_ffn_ms:<10.3f} "
              f"{df_send_ms:<10.3f} {da_recv_ms:<10.3f} {total_ms:<10.3f} "
              f"{ucx_cuda_sync:<14.3f} {ucx_submit:<12.3f} {ucx_recv_inner:<15.3f}")

        layers_data.append({
            "layer": layer,
            "da_send_ms": da_send_ms,
            "df_recv_ms": df_recv_ms,
            "df_ffn_ms": df_ffn_ms,
            "df_send_ms": df_send_ms,
            "da_recv_ms": da_recv_ms,
            "total_ms": total_ms,
            "ucx_cuda_sync_ms": ucx_cuda_sync,
            "ucx_submit_ms": ucx_submit,
            "ucx_recv_inner_ms": ucx_recv_inner,
        })

    if layers_data:
        n = len(layers_data)
        avg = lambda k: sum(d[k] for d in layers_data) / n
        print("  " + "-" * 110)
        print(f"  {'AVG':<6} {avg('da_send_ms'):<10.3f} {avg('df_recv_ms'):<10.3f} "
              f"{avg('df_ffn_ms'):<10.3f} {avg('df_send_ms'):<10.3f} "
              f"{avg('da_recv_ms'):<10.3f} {avg('total_ms'):<10.3f} "
              f"{avg('ucx_cuda_sync_ms'):<14.3f} {avg('ucx_submit_ms'):<12.3f} "
              f"{avg('ucx_recv_inner_ms'):<15.3f}")

        print("\n  SUMMARY (averages across all layers):")
        print(f"    DA send (thread launch + async):  {avg('da_send_ms'):.3f} ms")
        print(f"      └─ CUDA fence (event.sync):     {avg('ucx_cuda_sync_ms'):.3f} ms")
        print(f"      └─ UCX send API call:           {avg('ucx_submit_ms'):.3f} ms")
        print(f"    DF recv (blocking/poll):           {avg('df_recv_ms'):.3f} ms")
        print(f"      └─ UCX recv inner:              {avg('ucx_recv_inner_ms'):.3f} ms")
        print(f"    DF FFN compute:                    {avg('df_ffn_ms'):.3f} ms")
        print(f"    DF send (thread launch + async):   {avg('df_send_ms'):.3f} ms")
        print(f"    DA recv (blocking/poll):           {avg('da_recv_ms'):.3f} ms")
        print(f"    ─────────────────────────────────")
        print(f"    Total per-layer round-trip:        {avg('total_ms'):.3f} ms")
        comm_total = avg('da_send_ms') + avg('df_recv_ms') + avg('df_send_ms') + avg('da_recv_ms')
        print(f"    Communication overhead:            {comm_total:.3f} ms ({comm_total/avg('total_ms')*100:.1f}%)")
        print(f"    FFN compute:                       {avg('df_ffn_ms'):.3f} ms ({avg('df_ffn_ms')/avg('total_ms')*100:.1f}%)")

    # Also print recv_wait detail if available
    if da_recv_wait_by_layer:
        print("\n  DA recv_wait DETAIL (DF→DA path):")
        print(f"  {'Layer':<6} {'thread_join':<12} {'send_join':<12} {'fence':<10} {'ev_sync':<10}")
        print("  " + "-" * 55)
        for layer in sorted(da_recv_wait_by_layer.keys())[:10]:
            for e in da_recv_wait_by_layer[layer][:1]:
                tj = e.get("recv_thread_join_us", 0) / 1000
                sj = e.get("send_thread_join_us", 0) / 1000
                fence = e.get("fence_us", 0) / 1000
                ev = e.get("event_sync_us", 0) / 1000
                print(f"  {layer:<6} {tj:<12.3f} {sj:<12.3f} {fence:<10.3f} {ev:<10.3f}")

    return layers_data


def main():
    kill_all()
    time.sleep(2)

    log.info("Launching PD+AF with timing instrumentation...")
    launcher = launch_pdaf()

    log.info("Waiting for router health...")
    if not wait_health("http://127.0.0.1:50000/health", timeout=180):
        log.error("Router did not become healthy in 180s")
        launcher.terminate()
        return

    log.info("Router healthy. Running benchmark (10 reqs, IL=1024, OL=32, qps=2)...")
    run_benchmark()

    log.info("Benchmark done. Waiting 5s for logs to flush...")
    time.sleep(5)

    # Parse logs
    da_log = LOG_DIR / "logs" / "DA.log"
    df_log = LOG_DIR / "logs" / "DF.log"

    if not da_log.exists() or not df_log.exists():
        log.error("DA/DF logs not found!")
        kill_all()
        return

    log.info("Parsing DA events from %s", da_log)
    da_events = parse_host_events(str(da_log))
    log.info("Parsing DF events from %s", df_log)
    df_events = parse_host_events(str(df_log))

    log.info("DA events: %d, DF events: %d", len(da_events), len(df_events))

    if da_events and df_events:
        analyze_bubble(da_events, df_events)
    else:
        log.warning("No events found. Check if AFD_HOST_EVENTS logging is active.")
        log.info("Trying to grep DA log for host events...")
        r = subprocess.run(["grep", "-c", "AFD_HOST_EVENTS", str(da_log)], capture_output=True, text=True)
        log.info("  DA AFD_HOST_EVENTS lines: %s", r.stdout.strip())
        r = subprocess.run(["grep", "-c", "AFD_HOST_EVENTS", str(df_log)], capture_output=True, text=True)
        log.info("  DF AFD_HOST_EVENTS lines: %s", r.stdout.strip())

    kill_all()
    log.info("Done.")


if __name__ == "__main__":
    main()
