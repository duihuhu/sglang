#!/usr/bin/env python3
"""Diagnose client submission stalls; never contributes M2 timing samples."""

import argparse
import asyncio
import fcntl
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from run_m2_state_mediation import BASE, ROOT, command, dump, ensure_available, replay, stop_owned_server, wait_ready


async def observed_replay(args, label, rep, reference, lag):
    stop = asyncio.Event()
    async def observe_loop():
        previous = time.perf_counter()
        while not stop.is_set():
            await asyncio.sleep(.001)
            current = time.perf_counter()
            if current - previous > .005:
                lag.append({"start": previous, "end": current, "seconds": current - previous})
            previous = current
    monitor = asyncio.create_task(observe_loop())
    try:
        await replay(args, "natural", "mixed-4096", rep, 128, label, reference)
    finally:
        stop.set()
        await monitor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity-reference", type=Path, required=True)
    parser.add_argument("--port", type=int, default=30133)
    args = parser.parse_args()
    reference = json.loads(args.capacity_reference.read_text())["pools"]
    # ABBA in one owned server, with distinct run IDs. These are diagnostic only.
    order = (("gc-enabled", 21), ("gc-disabled", 22), ("gc-disabled", 23), ("gc-enabled", 24))
    with (BASE / ".m2_gpu67.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot = ensure_available(args.port)
        args.output.mkdir(parents=True, exist_ok=False)
        dump(args.output / "manifest.json", {"purpose": __doc__, "order": order, "gpu_snapshot": snapshot,
                                             "command": command("mixed-4096", "natural", args.port)})
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "6,7", "SGLANG_ENABLE_TENANT_RESOURCE_TRACKING": "1",
               "SGLANG_TENANT_EVENT_TRACE": "1", "SGLANG_TENANT_PROGRESS_TRACE": "1"}
        def interrupted(signum, frame):
            raise KeyboardInterrupt(signum)
        signal.signal(signal.SIGTERM, interrupted)
        with (args.output / "server.log").open("x") as log:
            process = subprocess.Popen(command("mixed-4096", "natural", args.port), cwd=ROOT, env=env,
                                       stdout=log, stderr=log, start_new_session=True)
            try:
                wait_ready(process, f"http://127.0.0.1:{args.port}")
                summaries = []
                for label, rep in order:
                    cycles, lag = [], []
                    def callback(phase, info):
                        cycles.append({"phase": phase, "time": time.perf_counter(), **info})
                    prior = gc.isenabled()
                    if label == "gc-disabled":
                        gc.collect()
                        gc.disable()
                    gc.callbacks.append(callback)
                    error = None
                    try:
                        asyncio.run(observed_replay(args, label, rep, reference, lag))
                    except RuntimeError as exc:
                        error = str(exc)
                    finally:
                        gc.callbacks.remove(callback)
                        if prior:
                            gc.enable()
                    path = args.output / f"m2mediate-natural-mixed-4096-cap128-rep{rep}-{label}.json"
                    if not path.exists():
                        raise RuntimeError(f"Diagnostic did not produce artifact: {error}")
                    item = json.loads(path.read_text())
                    origin = item["experiment_clock_origin_s"]
                    last_submission = max(r["submitted_s"] for r in item["requests"])
                    pauses, starts = [], {}
                    for event in cycles:
                        generation = event["generation"]
                        if event["phase"] == "start":
                            starts[generation] = event["time"]
                        elif generation in starts:
                            left, right = starts.pop(generation), event["time"]
                            if right >= origin and left <= origin + last_submission:
                                pauses.append({"generation": generation, "start_s": left-origin,
                                               "end_s": right-origin, "duration_ms": 1000*(right-left)})
                    summary = {"label": label, "rep": rep, "integrity_errors": item["integrity_errors"],
                               "max_lateness_ms": 1000*max(r["submitted_s"]-r["scheduled_arrival_s"] for r in item["requests"]),
                               "gc_pauses_during_submission": pauses,
                               "loop_stalls_during_submission": [{"start_s": e["start"]-origin, "duration_ms": 1000*e["seconds"]}
                                                                for e in lag if e["end"] >= origin and e["start"] <= origin+last_submission]}
                    dump(args.output / f"client-{label}-rep{rep}.json", {"summary": summary, "gc_events": cycles, "loop_stalls": lag})
                    summaries.append(summary)
                    print(json.dumps(summary), flush=True)
                    if any(e != "arrival lateness" for e in item["integrity_errors"]):
                        raise RuntimeError("Non-timing diagnostic integrity error")
                dump(args.output / "summary.json", {"diagnostic_only": True, "rows": summaries})
            finally:
                stop_owned_server(process)


if __name__ == "__main__":
    main()
