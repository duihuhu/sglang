#!/usr/bin/env python3
"""A800 motivation experiments for lifecycle-cost-aware hybrid scheduling."""

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
    warm_spec,
)


CLASS_ORDER = ("decode", "prefill", "reuse", "medium")


def build_state_pair(args: argparse.Namespace) -> list[RequestSpec]:
    decode = make_group(
        args.run_id,
        "decode-heavy",
        args.pair_count,
        1024,
        512,
        tenant_key="decode-heavy",
    )
    prefill = make_group(
        args.run_id,
        "prefill-heavy",
        args.pair_count,
        16384,
        32,
        tenant_key="prefill-heavy",
    )
    if args.order == "decode-first":
        ordered = decode + prefill
    elif args.order == "prefill-first":
        ordered = prefill + decode
    elif args.order == "interleave":
        ordered = interleave([decode, prefill])
    else:
        raise ValueError(args.order)
    return reassign_offsets(ordered, gap_ms=args.arrival_gap_ms)


def permutation_groups(
    args: argparse.Namespace,
) -> tuple[dict[str, list[RequestSpec]], int, str]:
    prefix_len = int(16384 * 0.9)
    shared_salt = f"lifecycle-gates:{args.run_id}:reuse"
    return (
        {
            "decode": make_group(
                args.run_id,
                "decode",
                12,
                1024,
                512,
                tenant_key="decode",
            ),
            "prefill": make_group(
                args.run_id,
                "prefill",
                8,
                16384,
                32,
                tenant_key="prefill",
            ),
            "reuse": make_group(
                args.run_id,
                "reuse",
                8,
                16384,
                32,
                common_prefix_len=prefix_len,
                shared_salt=shared_salt,
                tenant_key="reuse",
            ),
            "medium": make_group(
                args.run_id,
                "medium",
                12,
                4096,
                128,
                tenant_key="medium",
            ),
        },
        prefix_len,
        shared_salt,
    )


def build_permutation(
    args: argparse.Namespace,
) -> tuple[list[RequestSpec], int, str]:
    groups, prefix_len, shared_salt = permutation_groups(args)
    order = tuple(args.order.split(","))
    if len(order) != len(CLASS_ORDER) or set(order) != set(CLASS_ORDER):
        raise ValueError(
            "--order must be a permutation of " + ",".join(CLASS_ORDER)
        )
    specs = [spec for name in order for spec in groups[name]]
    return (
        reassign_offsets(specs, gap_ms=args.arrival_gap_ms),
        prefix_len,
        shared_salt,
    )


def build_joint(
    args: argparse.Namespace,
) -> tuple[list[RequestSpec], int, str]:
    prefix_len = int(16384 * 0.9)
    shared_salt = f"lifecycle-gates:{args.run_id}:joint-reuse"
    background = make_group(
        args.run_id,
        "background",
        args.background_count,
        1024,
        512,
        tenant_key="background",
    )
    pressure = make_group(
        args.run_id,
        "pressure",
        args.pressure_count,
        16384,
        32,
        tenant_key="pressure",
    )
    reuse = make_group(
        args.run_id,
        "future-reuse",
        args.reuse_count,
        16384,
        32,
        common_prefix_len=prefix_len,
        shared_salt=shared_salt,
        tenant_key="future-reuse",
    )
    if args.order == "pressure-first":
        candidates = pressure + reuse
    elif args.order == "reuse-first":
        candidates = reuse + pressure
    elif args.order == "interleave":
        candidates = interleave([pressure, reuse])
    else:
        raise ValueError(args.order)
    specs = reassign_offsets(background, gap_ms=args.arrival_gap_ms)
    specs += reassign_offsets(
        candidates,
        base_offset=args.candidate_delay_s,
        gap_ms=args.arrival_gap_ms,
    )
    return specs, prefix_len, shared_salt


def build_fairness(args: argparse.Namespace) -> list[RequestSpec]:
    # Both tenants request exactly 33,792 input+output tokens.
    split = make_group(
        args.run_id,
        "split",
        32,
        1024,
        32,
        tenant_key="split",
    )
    packed = make_group(
        args.run_id,
        "packed",
        2,
        16384,
        512,
        tenant_key="packed",
    )
    if args.composition == "split-only":
        ordered = split
    elif args.composition == "packed-only":
        ordered = packed
    elif args.composition == "mixed":
        ordered = interleave([split, packed])
    else:
        raise ValueError(args.composition)
    return reassign_offsets(ordered, gap_ms=args.arrival_gap_ms)


def build_fairness_sustained(args: argparse.Namespace) -> list[RequestSpec]:
    if args.packed_prompt % args.split_factor or args.packed_output % args.split_factor:
        raise ValueError(
            "packed prompt/output must both be divisible by --split-factor"
        )
    # Every wave offers exactly the same input and output tokens to both tenants.
    split_per_wave = args.split_factor
    packed_per_wave = 1
    split = make_group(
        args.run_id,
        "split",
        args.waves * split_per_wave,
        args.packed_prompt // args.split_factor,
        args.packed_output // args.split_factor,
        tenant_key="split",
    )
    packed = make_group(
        args.run_id,
        "packed",
        args.waves * packed_per_wave,
        args.packed_prompt,
        args.packed_output,
        tenant_key="packed",
    )
    specs: list[RequestSpec] = []
    for wave in range(args.waves):
        split_wave = split[
            wave * split_per_wave : (wave + 1) * split_per_wave
        ]
        packed_wave = packed[
            wave * packed_per_wave : (wave + 1) * packed_per_wave
        ]
        if args.composition == "split-only":
            ordered = split_wave
        elif args.composition == "packed-only":
            ordered = packed_wave
        elif args.composition == "mixed":
            ordered = interleave([split_wave, packed_wave])
        else:
            raise ValueError(args.composition)
        specs.extend(
            reassign_offsets(
                ordered,
                base_offset=wave * args.wave_interval_s,
                gap_ms=args.arrival_gap_ms,
            )
        )
    return specs


async def apply_runtime_policy(
    session: aiohttp.ClientSession,
    base_url: str,
    policy: str,
    *,
    request_weight: float = 0,
    input_weight: float = 1,
    output_weight: float = 1,
    lifecycle_state_weight: float = 1,
    lifecycle_kv_weight: float = 1,
) -> None:
    if policy == "server":
        return
    code = {
        "fcfs": 0,
        "vtc": 1,
        "vector": 2,
        "lifecycle": 3,
        "instant": 4,
    }[policy]
    server_args: dict[str, float | int] = {
        "tenant_schedule_policy_code": code,
        "tenant_vtc_request_weight": request_weight,
        "tenant_vtc_input_weight": input_weight,
        "tenant_vtc_output_weight": output_weight,
    }
    if policy == "lifecycle":
        server_args.update(
            {
                "tenant_lifecycle_state_weight": lifecycle_state_weight,
                "tenant_lifecycle_kv_weight": lifecycle_kv_weight,
            }
        )
    result = await post_json(
        session,
        f"{base_url}/set_internal_state",
        {"server_args": server_args},
    )
    if not result or not all(result):
        raise RuntimeError(f"scheduler policy update failed: {result}")


async def warm_reusable_prefix(
    session: aiohttp.ClientSession,
    base_url: str,
    args: argparse.Namespace,
    prefix_len: int,
    shared_salt: str,
) -> None:
    warm = await run_specs(
        session,
        base_url,
        [
            warm_spec(
                args.run_id,
                prefix_len,
                shared_salt,
                "warmup",
                tenant_key="warmup",
            )
        ],
        time.perf_counter() + 0.1,
        args.run_id,
    )
    if not warm[0].success:
        raise RuntimeError(f"prefix warmup failed: {warm[0].error}")


async def async_main(args: argparse.Namespace) -> None:
    base_url = args.base_url.rstrip("/")
    timeout = aiohttp.ClientTimeout(total=args.timeout_seconds)
    async with aiohttp.ClientSession(
        timeout=timeout, connector=aiohttp.TCPConnector(limit=0)
    ) as session:
        initial_info = await get_json(session, f"{base_url}/get_server_info")
        await apply_runtime_policy(
            session,
            base_url,
            args.runtime_policy,
            request_weight=args.request_weight,
            input_weight=args.input_weight,
            output_weight=args.output_weight,
            lifecycle_state_weight=args.lifecycle_state_weight,
            lifecycle_kv_weight=args.lifecycle_kv_weight,
        )
        await flush_cache(session, base_url)

        prefix_len = 0
        shared_salt: str | None = None
        if args.mode == "state-pair":
            specs = build_state_pair(args)
        elif args.mode == "permutation":
            specs, prefix_len, shared_salt = build_permutation(args)
        elif args.mode == "joint":
            specs, prefix_len, shared_salt = build_joint(args)
        elif args.mode == "fairness":
            specs = build_fairness(args)
        elif args.mode == "fairness-sustained":
            specs = build_fairness_sustained(args)
        else:
            raise ValueError(args.mode)

        if prefix_len and shared_salt:
            await warm_reusable_prefix(
                session, base_url, args, prefix_len, shared_salt
            )
            # Rebuild the experiment policy to reset its accounting after warmup.
            await apply_runtime_policy(
                session,
                base_url,
                args.runtime_policy,
                request_weight=args.request_weight,
                input_weight=args.input_weight,
                output_weight=args.output_weight,
                lifecycle_state_weight=args.lifecycle_state_weight,
                lifecycle_kv_weight=args.lifecycle_kv_weight,
            )

        experiment_start = time.perf_counter() + 0.25
        results = await run_specs(
            session, base_url, specs, experiment_start, args.run_id
        )
        final_info = await get_json(session, f"{base_url}/get_server_info")

    trace = extract_trace(final_info, args.run_id)
    summary = summarize(results, args.slo_seconds)
    if trace:
        for group, usage in trace["tenants"].items():
            if group in summary["groups"]:
                summary["groups"][group]["scheduler_active_resources"] = usage
    artifact: dict[str, Any] = {
        "schema_version": 2,
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
            "resolved_experiment_policy": [
                {
                    "schedule_policy": state.get("schedule_policy"),
                    "tenant_vtc_request_weight": state.get(
                        "tenant_vtc_request_weight"
                    ),
                    "tenant_vtc_input_weight": state.get(
                        "tenant_vtc_input_weight"
                    ),
                    "tenant_vtc_output_weight": state.get(
                        "tenant_vtc_output_weight"
                    ),
                    "tenant_lifecycle_state_weight": state.get(
                        "tenant_lifecycle_state_weight"
                    ),
                    "tenant_lifecycle_kv_weight": state.get(
                        "tenant_lifecycle_kv_weight"
                    ),
                }
                for state in final_info.get("internal_states", [])
            ],
        },
        "summary": summary,
        "scheduler_resource_trace": trace,
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
    failed: list[RequestResult] = [item for item in results if not item.success]
    if failed:
        raise SystemExit(f"{len(failed)} requests failed; inspect {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=[
            "state-pair",
            "permutation",
            "joint",
            "fairness",
            "fairness-sustained",
        ],
        required=True,
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    parser.add_argument("--slo-seconds", type=float, default=20)
    parser.add_argument("--arrival-gap-ms", type=float, default=1)
    parser.add_argument(
        "--runtime-policy",
        choices=["server", "fcfs", "vtc", "instant", "vector", "lifecycle"],
        default="fcfs",
    )
    parser.add_argument("--request-weight", type=float, default=0)
    parser.add_argument("--input-weight", type=float, default=1)
    parser.add_argument("--output-weight", type=float, default=1)
    parser.add_argument("--lifecycle-state-weight", type=float, default=1)
    parser.add_argument("--lifecycle-kv-weight", type=float, default=1)
    parser.add_argument(
        "--experiment-stage", choices=["calibration", "screen", "confirm", "kill"]
    )
    parser.add_argument("--cell-id", default="legacy")
    parser.add_argument("--physical-gpus", default="6,7")
    parser.add_argument("--regime", default="unspecified")
    parser.add_argument("--order")
    parser.add_argument("--pair-count", type=int, default=16)
    parser.add_argument("--composition", choices=["split-only", "packed-only", "mixed"])
    parser.add_argument("--background-count", type=int, default=32)
    parser.add_argument("--pressure-count", type=int, default=12)
    parser.add_argument("--reuse-count", type=int, default=8)
    parser.add_argument("--candidate-delay-s", type=float, default=0.5)
    parser.add_argument("--waves", type=int, default=8)
    parser.add_argument("--wave-interval-s", type=float, default=1.5)
    parser.add_argument("--packed-prompt", type=int, default=16384)
    parser.add_argument("--packed-output", type=int, default=512)
    parser.add_argument("--split-factor", type=int, default=16)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode in {"state-pair", "permutation", "joint"} and not args.order:
        raise SystemExit(f"--order is required for {args.mode}")
    if args.mode in {"fairness", "fairness-sustained"} and not args.composition:
        raise SystemExit(f"--composition is required for {args.mode}")
    if (
        args.runtime_policy == "lifecycle"
        and args.lifecycle_state_weight == 0
        and args.lifecycle_kv_weight == 0
    ):
        raise SystemExit("at least one lifecycle weight must be positive")
    if args.input_weight < 0 or args.output_weight < 0 or args.request_weight < 0:
        raise SystemExit("VTC weights must be non-negative")
    if args.split_factor <= 0:
        raise SystemExit("--split-factor must be positive")
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
