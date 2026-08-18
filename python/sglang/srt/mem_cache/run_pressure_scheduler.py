"""Run LatticeKV's pressure-triggered scheduler beside Central I/O."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time

from sglang.srt.mem_cache.central_io import CentralIOControlClient
from sglang.srt.mem_cache.pressure_scheduler_runtime import PressureSchedulerRuntime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket-path", required=True)
    parser.add_argument("--max-transfer-gib", type=float, required=True)
    parser.add_argument(
        "--growth-quantum-gib",
        type=float,
        help="Bounded quota step used when real reclaim loss requests growth.",
    )
    parser.add_argument("--interval-s", type=float, default=0.25)
    parser.add_argument("--connect-retry-s", type=float, default=1.0)
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        help="Optional audit stream of pressure decisions and effective quota facts.",
    )
    args = parser.parse_args()
    if (
        args.max_transfer_gib <= 0
        or args.interval_s <= 0
        or args.connect_retry_s <= 0
        or (
            args.growth_quantum_gib is not None
            and (
                args.growth_quantum_gib <= 0
                or args.growth_quantum_gib > args.max_transfer_gib
            )
        )
    ):
        raise ValueError("scheduler transfer size and interval must be positive")

    logging.basicConfig(level=logging.INFO)
    control = None
    runtime = None
    output = None
    if args.output_jsonl is not None:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        output = args.output_jsonl.open("a", encoding="utf-8")
    try:
        while True:
            try:
                if control is None:
                    control = CentralIOControlClient(args.socket_path)
                    runtime = PressureSchedulerRuntime(
                        control,
                        max_transfer_bytes=int(args.max_transfer_gib * 1024**3),
                        growth_quantum_bytes=(
                            None
                            if args.growth_quantum_gib is None
                            else int(args.growth_quantum_gib * 1024**3)
                        ),
                    )
                    logging.info("Connected to LatticeKV Central I/O at %s", args.socket_path)
                assert runtime is not None
                result = runtime.tick()
            except (BrokenPipeError, ConnectionError, EOFError, OSError) as error:
                if control is not None:
                    control.close()
                control = None
                runtime = None
                logging.warning("Central I/O unavailable (%s); retrying", error)
                time.sleep(args.connect_retry_s)
                continue
            if output is not None:
                decision = result["decision"]
                output.write(
                    json.dumps(
                        {
                            "monotonic_ns": time.monotonic_ns(),
                            "published_targets": result["published_targets"],
                            "effective_targets": result["effective_targets"],
                            "quota_transitions": result["quota_transitions"],
                            "published_target_set_ns": result["published_target_set_ns"],
                            "missing_reports": result["missing_reports"],
                            "pending_models": result["pending_models"],
                            # This is the evidence behind a published intent:
                            # local watermarks, reclaim-ready supply, and any
                            # admission shortfall are captured in the same tick.
                            "local_health": result["local_health"],
                            "decision": None
                            if decision is None
                            else {
                                "target_bytes": decision.target_bytes,
                                "unmet_bytes": decision.unmet_bytes,
                                "transfers": [
                                    {
                                        "source_model_id": item.source_model_id,
                                        "destination_model_id": item.destination_model_id,
                                        "bytes": item.bytes,
                                        "readiness": item.readiness,
                                    }
                                    for item in decision.transfers
                                ],
                            },
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                output.flush()
            if result["published_targets"]:
                logging.info(
                    "LatticeKV pressure intent=%s effective=%s",
                    result["published_targets"],
                    result["effective_targets"],
                )
            time.sleep(args.interval_s)
    finally:
        if control is not None:
            control.close()
        if output is not None:
            output.close()


if __name__ == "__main__":
    main()
