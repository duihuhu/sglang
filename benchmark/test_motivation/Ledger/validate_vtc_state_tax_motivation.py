#!/usr/bin/env python3
"""Run no-prefix VTC state-tax observation experiments."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import aiohttp

from validate_dynamic_cost_motivation import (
    RequestResult,
    RequestSpec,
    extract_trace,
    flush_cache,
    get_json,
    interleave,
    make_group,
    post_json,
    reassign_offsets,
    run_specs,
    summarize,
)


VTC_WEIGHT_KEYS = (
    "request_weight",
    "input_weight",
    "input_quadratic_weight",
    "output_weight",
    "input_output_weight",
    "output_quadratic_weight",
)


def build_profile(args: argparse.Namespace) -> list[RequestSpec]:
    return make_group(
        args.run_id,
        "profile",
        1,
        args.profile_prompt,
        args.profile_output,
        tenant_key="profile",
    )


def build_partition(args: argparse.Namespace) -> list[RequestSpec]:
    if args.packed_prompt % args.split_factor or args.packed_output % args.split_factor:
        raise ValueError("packed prompt/output must be divisible by split factor")
    split = make_group(
        args.run_id,
        "split",
        args.waves * args.split_factor,
        args.packed_prompt // args.split_factor,
        args.packed_output // args.split_factor,
        tenant_key="split",
    )
    packed = make_group(
        args.run_id,
        "packed",
        args.waves,
        args.packed_prompt,
        args.packed_output,
        tenant_key="packed",
    )
    specs: list[RequestSpec] = []
    for wave in range(args.waves):
        split_wave = split[
            wave * args.split_factor : (wave + 1) * args.split_factor
        ]
        packed_wave = packed[wave : wave + 1]
        if args.composition == "split-only":
            ordered = split_wave
        elif args.composition == "packed-only":
            ordered = packed_wave
        elif args.wave_order == "split-first":
            ordered = interleave([split_wave, packed_wave])
        else:
            ordered = interleave([packed_wave, split_wave])
        specs.extend(
            reassign_offsets(
                ordered,
                base_offset=wave * args.wave_interval_s,
                gap_ms=args.arrival_gap_ms,
            )
        )
    return specs


async def apply_runtime_policy(
    session: aiohttp.ClientSession, base_url: str, args: argparse.Namespace
) -> None:
    code = {"fcfs": 0, "vtc": 1, "instant": 4}[args.runtime_policy]
    server_args: dict[str, float | int] = {"tenant_schedule_policy_code": code}
    for key in VTC_WEIGHT_KEYS:
        server_args[f"tenant_vtc_{key}"] = float(getattr(args, key))
    result = await post_json(
        session,
        f"{base_url}/set_internal_state",
        {"server_args": server_args},
    )
    if not result or not all(result):
        raise RuntimeError(f"scheduler policy update failed: {result}")


def extract_vtc_trace(
    server_info: dict[str, Any], run_id: str
) -> dict[str, Any] | None:
    prefix = f"hybrid-tenant:{run_id}:"
    for state in server_info.get("internal_states", []):
        trace = state.get("tenant_vtc_trace")
        if not trace:
            continue
        counters = {
            tenant.removeprefix(prefix): value
            for tenant, value in trace.get("counters", {}).items()
            if tenant.startswith(prefix)
        }
        queue_waits = {
            tenant.removeprefix(prefix): value
            for tenant, value in trace.get("queue_waits", {}).items()
            if tenant.startswith(prefix)
        }
        if counters or queue_waits:
            return {
                **trace,
                "counters": counters,
                "queue_waits": queue_waits,
                "run_id": run_id,
            }
    return None


async def async_main(args: argparse.Namespace) -> None:
    base_url = args.base_url.rstrip("/")
    timeout = aiohttp.ClientTimeout(total=args.timeout_seconds)
    async with aiohttp.ClientSession(
        timeout=timeout, connector=aiohttp.TCPConnector(limit=0)
    ) as session:
        initial_info = await get_json(session, f"{base_url}/get_server_info")
        await apply_runtime_policy(session, base_url, args)
        await flush_cache(session, base_url)
        specs = build_profile(args) if args.mode == "profile" else build_partition(args)
        experiment_start = time.perf_counter() + 0.25
        results = await run_specs(
            session, base_url, specs, experiment_start, args.run_id
        )
        final_info = await get_json(session, f"{base_url}/get_server_info")

    resource_trace = extract_trace(final_info, args.run_id)
    vtc_trace = extract_vtc_trace(final_info, args.run_id)
    summary = summarize(results, args.slo_seconds)
    if resource_trace:
        for group, usage in resource_trace["tenants"].items():
            if group in summary["groups"]:
                summary["groups"][group]["scheduler_active_resources"] = usage

    resolved = []
    for state in final_info.get("internal_states", []):
        item = {"schedule_policy": state.get("schedule_policy")}
        for key in VTC_WEIGHT_KEYS:
            item[f"tenant_vtc_{key}"] = state.get(f"tenant_vtc_{key}")
        resolved.append(item)
    artifact = {
        "schema_version": 3,
        "experiment": {
            key: value for key, value in vars(args).items() if key != "output"
        },
        "server": {
            "version": initial_info.get("version"),
            "model_path": initial_info.get("model_path"),
            "max_total_num_tokens": initial_info.get("max_total_num_tokens"),
            "internal_states": [
                {
                    "max_mamba_cache_size": state.get("max_mamba_cache_size"),
                    "effective_max_running_requests_per_dp": state.get(
                        "effective_max_running_requests_per_dp"
                    ),
                }
                for state in initial_info.get("internal_states", [])
            ],
            "resolved_experiment_policy": resolved,
        },
        "summary": summary,
        "scheduler_resource_trace": resource_trace,
        "scheduler_vtc_trace": vtc_trace,
        "requests": [
            {
                **asdict(result),
                "latency_s": result.latency_s,
                "ttft_s": result.ttft_s,
            }
            for result in results
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    failed: list[RequestResult] = [result for result in results if not result.success]
    if failed:
        raise SystemExit(f"{len(failed)} requests failed; inspect {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["profile", "partition"], required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    parser.add_argument("--slo-seconds", type=float, default=60)
    parser.add_argument("--arrival-gap-ms", type=float, default=1)
    parser.add_argument("--runtime-policy", choices=["fcfs", "vtc", "instant"], default="fcfs")
    parser.add_argument("--policy-label", default="fcfs")
    parser.add_argument("--request-weight", type=float, default=0)
    parser.add_argument("--input-weight", type=float, default=1)
    parser.add_argument("--input-quadratic-weight", type=float, default=0)
    parser.add_argument("--output-weight", type=float, default=2)
    parser.add_argument("--input-output-weight", type=float, default=0)
    parser.add_argument("--output-quadratic-weight", type=float, default=0)
    parser.add_argument("--experiment-stage", choices=["profile", "screen", "confirm"], required=True)
    parser.add_argument("--cell-id", default="profile")
    parser.add_argument("--physical-gpus", default="6,7")
    parser.add_argument("--profile-prompt", type=int, default=1024)
    parser.add_argument("--profile-output", type=int, default=32)
    parser.add_argument("--composition", choices=["split-only", "packed-only", "mixed"], default="mixed")
    parser.add_argument("--waves", type=int, default=8)
    parser.add_argument("--wave-interval-s", type=float, default=1.5)
    parser.add_argument("--packed-prompt", type=int, default=16384)
    parser.add_argument("--packed-output", type=int, default=512)
    parser.add_argument("--split-factor", type=int, default=16)
    parser.add_argument("--wave-order", choices=["split-first", "packed-first"], default="split-first")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if any(float(getattr(args, key)) < 0 for key in VTC_WEIGHT_KEYS):
        raise SystemExit("VTC weights must be non-negative")
    if args.mode == "partition" and args.composition is None:
        raise SystemExit("partition mode requires --composition")
    if args.split_factor <= 0:
        raise SystemExit("split factor must be positive")
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
