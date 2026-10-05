#!/usr/bin/env python3
"""Profile a cold client's arrival window; data is diagnostic, not M2 timing."""

import argparse
import asyncio
import cProfile
import fcntl
import io
import json
import os
from pathlib import Path
import pstats
import signal
import subprocess
import time

import run_m2_state_mediation as runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity-reference", type=Path, required=True)
    parser.add_argument("--port", type=int, default=30133)
    args = parser.parse_args()
    reference = json.loads(args.capacity_reference.read_text())["pools"]
    original = runner.run_specs
    profile = cProfile.Profile()
    async def profiled(*call_args):
        origin = call_args[3]
        async def stop_profile():
            await asyncio.sleep(max(0., origin + .8 - time.perf_counter()))
            profile.disable()
        profile.enable()
        stopper = asyncio.create_task(stop_profile())
        try:
            return await original(*call_args)
        finally:
            await stopper
            profile.disable()
    runner.run_specs = profiled
    with (runner.BASE / ".m2_gpu67.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot = runner.ensure_available(args.port)
        args.output.mkdir(parents=True, exist_ok=False)
        runner.dump(args.output / "manifest.json", {"diagnostic_only": True, "gpu_snapshot": snapshot,
                    "measurement_hashes": runner.measurement_hashes(), "window_end_after_origin_s": .8})
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "6,7", "SGLANG_ENABLE_TENANT_RESOURCE_TRACKING": "1",
               "SGLANG_TENANT_EVENT_TRACE": "1", "SGLANG_TENANT_PROGRESS_TRACE": "1"}
        def interrupted(signum, frame):
            raise KeyboardInterrupt(signum)
        signal.signal(signal.SIGTERM, interrupted)
        with (args.output / "server.log").open("x") as log:
            process = subprocess.Popen(runner.command("mixed-4096", "natural", args.port), cwd=runner.ROOT,
                                       env=env, stdout=log, stderr=log, start_new_session=True)
            try:
                runner.wait_ready(process, f"http://127.0.0.1:{args.port}")
                error = None
                try:
                    asyncio.run(runner.replay(args, "natural", "mixed-4096", 31, 128, "client-profile", reference))
                except RuntimeError as exc:
                    error = str(exc)
                stream = io.StringIO()
                pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("tottime").print_stats(60)
                stats = stream.getvalue()
                (args.output / "profile.txt").write_text(stats)
                profile.dump_stats(str(args.output / "client.prof"))
                runner.dump(args.output / "summary.json", {"diagnostic_only": True, "replay_error": error})
                print(stats, flush=True)
            finally:
                runner.stop_owned_server(process)


if __name__ == "__main__":
    main()
