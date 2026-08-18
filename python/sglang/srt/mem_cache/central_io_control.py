#!/usr/bin/env python3
"""Set or inspect a LatticeKV model quota target through the central agent."""

from __future__ import annotations

import argparse
import json

from sglang.srt.mem_cache.central_io import CentralIOControlClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--target-gib", type=int)
    parser.add_argument(
        "--targets-gib-json",
        help='Atomic multi-model target map, for example {"cold-b":60,"hot-a":140}.',
    )
    args = parser.parse_args()

    if args.targets_gib_json is None and args.model_id is None:
        parser.error("--model-id is required unless --targets-gib-json is provided")
    if args.targets_gib_json is None and args.target_gib is None and args.model_id is None:
        parser.error("provide a model id or an atomic target map")
    if args.targets_gib_json is not None and (args.model_id is not None or args.target_gib is not None):
        parser.error("--targets-gib-json cannot be combined with --model-id or --target-gib")

    client = CentralIOControlClient(args.socket)
    try:
        if args.targets_gib_json is not None:
            try:
                requested = json.loads(args.targets_gib_json)
            except json.JSONDecodeError as error:
                parser.error(f"invalid --targets-gib-json: {error.msg}")
            if not isinstance(requested, dict) or not requested:
                parser.error("--targets-gib-json must be a non-empty object")

            # Resolve each GiB target before the final RPC.  The final target
            # publication itself is one all-or-nothing agent operation.
            capacities: dict[str, int] = {}
            effective: dict[str, dict[str, float | int]] = {}
            for model_id, gib in requested.items():
                if not isinstance(model_id, str) or isinstance(gib, bool) or not isinstance(gib, int) or gib <= 0:
                    parser.error("atomic targets must map model ids to positive integer GiB values")
                status = client.status(model_id)
                capacity = (gib * 1024**3) // int(status["token_bytes"])
                capacity -= capacity % int(status["page_size"])
                if capacity <= 0:
                    parser.error(f"target for {model_id!r} is smaller than one KV page")
                capacities[model_id] = capacity
                effective[model_id] = {
                    "requested_gib": gib,
                    "effective_gib": capacity * int(status["token_bytes"]) / 1024**3,
                    "effective_pages": capacity // int(status["page_size"]),
                }
            response = client.set_quota_targets(capacities)
            response["effective_targets"] = effective
            print(json.dumps(response, indent=2))
        else:
            status = client.status(args.model_id)
            if args.target_gib is not None:
                status = client.set_quota_target_gib(args.model_id, args.target_gib)
            print(json.dumps(status, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()
