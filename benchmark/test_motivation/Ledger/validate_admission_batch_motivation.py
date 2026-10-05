#!/usr/bin/env python3
"""Run admission-batch coupling and runtime-cost motivation experiments."""

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
    flush_cache,
    get_json,
    make_group,
    post_json,
    run_specs,
    summarize,
)
from validate_vtc_state_tax_motivation import (
    VTC_WEIGHT_KEYS,
    apply_runtime_policy,
    build_partition,
    extract_vtc_trace,
)


def build_pulse(args: argparse.Namespace) -> list[RequestSpec]:
    incumbent = make_group(
        args.run_id,
        "incumbent",
        args.incumbent_count,
        args.incumbent_prompt,
        args.incumbent_output,
        gap_ms=args.arrival_gap_ms,
        tenant_key="incumbent",
    )
    pulser = make_group(
        args.run_id,
        "pulser",
        args.pulse_waves * args.pulse_width,
        args.pulser_prompt,
        args.pulser_output,
        tenant_key="pulser",
    )
    pulsed: list[RequestSpec] = []
    for wave in range(args.pulse_waves):
        for index, spec in enumerate(
            pulser[wave * args.pulse_width : (wave + 1) * args.pulse_width]
        ):
            pulsed.append(
                RequestSpec(
                    rid=spec.rid,
                    group=spec.group,
                    input_ids=spec.input_ids,
                    output_len=spec.output_len,
                    arrival_offset_s=(
                        args.pulse_start_s
                        + wave * args.pulse_interval_s
                        + index * args.arrival_gap_ms / 1000
                    ),
                    cache_salt=spec.cache_salt,
                    common_prefix_len=0,
                    tenant_key=spec.tenant_key,
                )
            )
    if args.composition == "incumbent-only":
        return incumbent
    if args.composition == "pulser-only":
        return pulsed
    if args.composition == "mixed":
        return incumbent + pulsed
    raise ValueError(f"invalid pulse composition: {args.composition}")


def extract_batch_trace(
    server_info: dict[str, Any], run_id: str
) -> dict[str, Any] | None:
    prefix = f"hybrid-tenant:{run_id}:"
    for state in server_info.get("internal_states", []):
        trace = state.get("tenant_batch_trace")
        if not trace:
            continue
        events = []
        for event in trace.get("events", []):
            tenants = {
                tenant.removeprefix(prefix): count
                for tenant, count in event.get("tenant_counts", {}).items()
                if tenant.startswith(prefix)
            }
            if tenants:
                events.append({**event, "tenant_counts": tenants})
        if events:
            return {
                **trace,
                "event_count": len(events),
                "events": events,
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
        specs = build_partition(args) if args.mode == "partition" else build_pulse(args)
        experiment_start = time.perf_counter() + 0.25
        results = await run_specs(
            session, base_url, specs, experiment_start, args.run_id
        )
        final_info = await get_json(session, f"{base_url}/get_server_info")

    resource_trace = None
    prefix = f"hybrid-tenant:{args.run_id}:"
    for state in final_info.get("internal_states", []):
        trace = state.get("tenant_resource_trace")
        if not trace:
            continue
        tenants = {
            tenant.removeprefix(prefix): value
            for tenant, value in trace.get("tenants", {}).items()
            if tenant.startswith(prefix)
        }
        if tenants:
            resource_trace = {**trace, "tenants": tenants, "run_id": args.run_id}
            break
    vtc_trace = extract_vtc_trace(final_info, args.run_id)
    batch_trace = extract_batch_trace(final_info, args.run_id)
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
        "schema_version": 4,
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
        "scheduler_batch_trace": batch_trace,
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
    parser.add_argument("--mode", choices=["partition", "pulse"], required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    parser.add_argument("--slo-seconds", type=float, default=60)
    parser.add_argument("--arrival-gap-ms", type=float, default=1)
    parser.add_argument(
        "--runtime-policy", choices=["fcfs", "vtc", "instant"], default="fcfs"
    )
    parser.add_argument("--policy-label", default="fcfs")
    parser.add_argument("--request-weight", type=float, default=0)
    parser.add_argument("--input-weight", type=float, default=1)
    parser.add_argument("--input-quadratic-weight", type=float, default=0)
    parser.add_argument("--output-weight", type=float, default=2)
    parser.add_argument("--input-output-weight", type=float, default=0)
    parser.add_argument("--output-quadratic-weight", type=float, default=0)
    parser.add_argument(
        "--experiment-stage", choices=["smoke", "confirm", "pulse"], required=True
    )
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--physical-gpus", default="6,7")
    parser.add_argument(
        "--composition",
        choices=[
            "split-only",
            "packed-only",
            "incumbent-only",
            "pulser-only",
            "mixed",
        ],
        default="mixed",
    )

    parser.add_argument("--waves", type=int, default=8)
    parser.add_argument("--wave-interval-s", type=float, default=1.5)
    parser.add_argument("--packed-prompt", type=int, default=16384)
    parser.add_argument("--packed-output", type=int, default=512)
    parser.add_argument("--split-factor", type=int, default=16)
    parser.add_argument(
        "--wave-order", choices=["split-first", "packed-first"], default="split-first"
    )

    parser.add_argument("--incumbent-count", type=int, default=8)
    parser.add_argument("--incumbent-prompt", type=int, default=16384)
    parser.add_argument("--incumbent-output", type=int, default=512)
    parser.add_argument("--pulse-waves", type=int, default=8)
    parser.add_argument("--pulse-width", type=int, default=16)
    parser.add_argument("--pulse-start-s", type=float, default=0.75)
    parser.add_argument("--pulse-interval-s", type=float, default=1.5)
    parser.add_argument("--pulser-prompt", type=int, default=1024)
    parser.add_argument("--pulser-output", type=int, default=32)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if any(float(getattr(args, key)) < 0 for key in VTC_WEIGHT_KEYS):
        raise SystemExit("VTC weights must be non-negative")
    if args.mode == "partition" and args.composition not in {
        "split-only",
        "packed-only",
        "mixed",
    }:
        raise SystemExit("partition mode requires a partition composition")
    if args.mode == "pulse" and args.composition not in {
        "incumbent-only",
        "pulser-only",
        "mixed",
    }:
        raise SystemExit("pulse mode requires a pulse composition")
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
