"""Untimed concurrent/reallocation output parity before performance measurement."""

import argparse
import asyncio
import contextlib
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess

import aiohttp

from run_m2_batch_frontier import BASE, ROOT, dump, hash_file, wait_ready
from run_m2_pool_comparison import command, ensure_idle, source_hashes, stop_owned
from validate_dynamic_cost_motivation import flush_cache, post_json, unique_input


async def check(url, output):
    records = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as client:
        await flush_cache(client, url)
        for rep in (1, 2):
            # No flush between rounds: exercise release/reallocation of live pools.
            async def generate(index, length, output_len):
                response = await post_json(client, url + "/generate", {
                    "input_ids": unique_input(length, 9821 + index), "stream": False,
                    "return_logprob": True, "top_logprobs_num": 5,
                    "sampling_params": {"temperature": 0, "max_new_tokens": output_len, "ignore_eos": True}})
                meta = response["meta_info"]
                probs = meta.get("output_token_logprobs", [])
                if (len(probs) != output_len or meta["prompt_tokens"] != length
                        or meta["completion_tokens"] != output_len or meta.get("num_retractions", 0)):
                    raise ValueError("Invalid correctness response")
                return {"repeat": rep, "index": index, "input_length": length,
                        "output_ids": [p[1] for p in probs], "response": response}
            records.extend(await asyncio.gather(*(generate(i, p, o) for i, (p, o) in enumerate(
                ((512, 16), (1024, 16), (8192, 16), (512, 64))))))
    dump(output, {"records": records})
    ids = [r["output_ids"] for r in records]
    if ids[:4] != ids[4:]:
        raise ValueError("Concurrent self-repeat output mismatch")
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs=2, required=True)
    parser.add_argument("--port", type=int, default=30143)
    args = parser.parse_args()
    plan = json.loads(args.budget_plan.read_text())
    hashes = source_hashes()
    if plan["source_hashes"] != hashes:
        raise ValueError("Budget plan source mismatch")
    sessions = [s for s in plan["sessions"] if s["arm"] != "unified-old-cap"]
    args.output.mkdir(parents=True, exist_ok=False)
    dump(args.output / "manifest.json", {"source_hashes": hashes, "plan": plan,
         "gpus": args.gpus, "script_sha256": hash_file(Path(__file__))})
    with contextlib.ExitStack() as stack:
        for gpu in sorted(args.gpus):
            lock = stack.enter_context((BASE / f".m2_gpu{gpu}.lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"signal {signum}")
        signal.signal(signal.SIGTERM, interrupted)
        reference = None
        for s in sessions:
            ensure_idle(args.port, args.gpus)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, args.gpus))}
            env.pop("M2_POOL_AUDIT_DIR", None)
            cmd = command(s["mode"], s["arm"], args.port, kv_tokens=s["kv_tokens"])
            with (args.output / f"{s['arm']}.log").open("x") as log:
                process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=log, start_new_session=True)
                try:
                    print(f"START concurrent outputs: {s['arm']}", flush=True)
                    wait_ready(process, f"http://127.0.0.1:{args.port}")
                    ids = asyncio.run(check(f"http://127.0.0.1:{args.port}", args.output / f"{s['arm']}.json"))
                    if reference is None:
                        reference = ids
                    elif ids != reference:
                        raise ValueError("Concurrent cross-pool output mismatch")
                finally:
                    stop_owned(process)
            print(f"PASS concurrent outputs: {s['arm']}", flush=True)
        if source_hashes() != hashes:
            raise ValueError("Measurement sources changed")
        dump(args.output / "completed.json", {"completed": True, "source_hashes": hashes,
             "cases_per_arm": 8, "arms": [s["arm"] for s in sessions],
             "exact_output_ids_match": True, "script_sha256": hash_file(Path(__file__))})


if __name__ == "__main__":
    main()
