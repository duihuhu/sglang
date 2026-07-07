#!/usr/bin/env python3
"""Poll per-GPU SM clocks while a BiScale PD benchmark runs.

Maps each GPU to a PD instance (P0-P7 on prefill node, D0-D7 on decode node)
and records a time-series JSONL + per-workload summary.

Usage (on prefill node host, while benchmark is running):
    python3 monitor_biscale_freq_timeline.py \\
        --bench-log /mnt/workspace/lt/sglang/benchmark/AFlex_bench/multi_node/logs/fixed_6scheme_code_biscale_only.log \\
        --node1 10.252.129.34 --node2 10.252.129.33 \\
        --out-dir results/freq_timeline_biscale_code
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTAINER = "operator_test"
INTERVAL_S = 0.1
WORKLOAD_RE = re.compile(r"\b(\w+_qps\d+)\b")


def _local_ips() -> set[str]:
    try:
        out = subprocess.check_output(["ip", "-4", "addr"], timeout=3).decode()
        return set(re.findall(r"inet (\d+\.\d+\.\d+\.\d+)/", out))
    except Exception:
        return set()


LOCAL_IPS = _local_ips()


def _is_local_host(host: str) -> bool:
    return host in LOCAL_IPS or host in ("127.0.0.1", "localhost")


def _ssh(host: str, cmd: str) -> str:
    return subprocess.check_output(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", host, cmd],
        timeout=5,
    ).decode()


def poll_gpus(host: str, node_label: str, phase: str) -> list[dict]:
    """Return one sample per GPU on *host*."""
    inner = (
        f"docker exec {CONTAINER} nvidia-smi "
        "--query-gpu=index,clocks.current.sm,clocks.max.sm,power.draw "
        "--format=csv,noheader,nounits"
    )
    if _is_local_host(host):
        out = subprocess.check_output(inner, shell=True, timeout=3).decode()
    else:
        out = _ssh(host, inner)
    t = time.time()
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        gpu = int(parts[0])
        inst = f"{phase}{gpu}"
        rows.append({
            "t": round(t, 3),
            "node": node_label,
            "host": host,
            "phase": phase,
            "instance": inst,
            "gpu": gpu,
            "freq_mhz": int(float(parts[1])),
            "freq_max_mhz": int(float(parts[2])),
            "power_w": float(parts[3]) if len(parts) > 3 else None,
        })
    return rows


class BenchLogWatcher:
  def __init__(self, log_path: Path):
    self.log_path = log_path
    self.workload = "unknown"
    self._pos = 0

  def update(self):
    if not self.log_path.exists():
      return
    with open(self.log_path) as f:
      f.seek(self._pos)
      chunk = f.read()
      self._pos = f.tell()
    for line in chunk.splitlines():
      m = WORKLOAD_RE.search(line)
      if m and "code_qps" in m.group(1):
        self.workload = m.group(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-log", type=Path, required=True)
    parser.add_argument("--node1", default="10.252.129.34")
    parser.add_argument("--node2", default="10.252.129.33")
    parser.add_argument("--out-dir", type=Path,
                        default=HERE / "results" / "freq_timeline_biscale_code")
    parser.add_argument("--interval", type=float, default=INTERVAL_S)
    parser.add_argument("--watch-pid", type=int, default=0,
                        help="Stop when this PID exits (0=watch bench log process)")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "nvidia_smi_timeline.jsonl"
    summary_path = out_dir / "capture_meta.json"

    watcher = BenchLogWatcher(args.bench_log)
    stop = threading.Event()
    samples: list[dict] = []
    lock = threading.Lock()

    def _poll_loop(host, node_label, phase):
        while not stop.is_set():
            t0 = time.time()
            try:
                watcher.update()
                wl = watcher.workload
                for row in poll_gpus(host, node_label, phase):
                    row["workload"] = wl
                    with lock:
                        samples.append(row)
                        with open(jsonl_path, "a") as f:
                            f.write(json.dumps(row) + "\n")
            except Exception as e:
                print(f"poll {host} failed: {e}")
            elapsed = time.time() - t0
            stop.wait(max(0, args.interval - elapsed))

    threads = [
        threading.Thread(target=_poll_loop,
                         args=(args.node1, "prefill", "P"), daemon=True),
        threading.Thread(target=_poll_loop,
                         args=(args.node2, "decode", "D"), daemon=True),
    ]
    for th in threads:
        th.start()

    print(f"Monitoring freq timeline → {jsonl_path}")
    print(f"Tailing workload markers from {args.bench_log}")

    # Wait until benchmark driver exits
    while True:
        watcher.update()
        if args.watch_pid > 0:
            try:
                subprocess.check_call(["kill", "-0", str(args.watch_pid)],
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                break
        else:
            try:
                if _is_local_host(args.node1):
                    out = subprocess.check_output(
                        ["pgrep", "-f", "run_fixed_6scheme_7dataset.py"],
                        timeout=3).decode()
                else:
                    out = _ssh(args.node1,
                               "pgrep -f run_fixed_6scheme_7dataset.py || true")
                if not out.strip():
                    time.sleep(5)
                    if _is_local_host(args.node1):
                        out = subprocess.check_output(
                            ["pgrep", "-f", "run_fixed_6scheme_7dataset.py"],
                            timeout=3).decode()
                    else:
                        out = _ssh(args.node1,
                                   "pgrep -f run_fixed_6scheme_7dataset.py || true")
                    if not out.strip():
                        break
            except Exception:
                pass
        time.sleep(10)

    stop.set()
    for th in threads:
        th.join(timeout=5)

    # Per-instance / per-workload stats
    by_inst: dict[str, list[int]] = {}
    by_wl_inst: dict[str, dict[str, list[int]]] = {}
    with lock:
        for s in samples:
            inst = s["instance"]
            by_inst.setdefault(inst, []).append(s["freq_mhz"])
            by_wl_inst.setdefault(s["workload"], {}).setdefault(
                inst, []).append(s["freq_mhz"])

    def _stats(freqs):
        if not freqs:
            return {}
        import statistics
        return {
            "n": len(freqs),
            "min": min(freqs),
            "max": max(freqs),
            "mean": round(statistics.mean(freqs), 1),
            "unique": sorted(set(freqs)),
        }

    meta = {
        "t_start": samples[0]["t"] if samples else None,
        "t_end": samples[-1]["t"] if samples else None,
        "n_samples": len(samples),
        "per_instance": {k: _stats(v) for k, v in sorted(by_inst.items())},
        "per_workload_instance": {
            wl: {inst: _stats(fs) for inst, fs in sorted(d.items())}
            for wl, d in sorted(by_wl_inst.items())
        },
    }
    summary_path.write_text(json.dumps(meta, indent=2))
    print(f"Done: {len(samples)} samples, summary → {summary_path}")


if __name__ == "__main__":
    main()
