#!/usr/bin/env python3
"""Poll /v1/loads during steady QPS benchmark to record running batch size."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stop-file", required=True)
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/v1/loads?include=core"
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as out:
        while not os.path.exists(args.stop_file):
            sample = {"ts": time.time()}
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    data = json.loads(resp.read().decode())
                loads = data.get("loads", [])
                sample["sum_running_reqs"] = sum(
                    int(x.get("num_running_reqs", 0)) for x in loads
                )
                sample["sum_waiting_reqs"] = sum(
                    int(x.get("num_waiting_reqs", 0)) for x in loads
                )
                sample["sum_gen_throughput"] = sum(
                    float(x.get("gen_throughput", 0.0)) for x in loads
                )
                sample["per_rank"] = [
                    {
                        "dp_rank": x.get("dp_rank"),
                        "num_running_reqs": x.get("num_running_reqs"),
                        "num_waiting_reqs": x.get("num_waiting_reqs"),
                        "gen_throughput": x.get("gen_throughput"),
                    }
                    for x in loads
                ]
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                sample["error"] = str(e)
            out.write(json.dumps(sample) + "\n")
            out.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
