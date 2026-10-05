#!/usr/bin/env python3
"""Black-box motivation experiments for hybrid state/KV scheduling.

This harness changes request arrival order while keeping the request multiset
fixed, and records a streaming timeline for each request. When the server enables
``SGLANG_ENABLE_TENANT_RESOURCE_TRACKING=1``, it also collects scheduler-side
active state/KV resource-time accounting. The streaming-response integrals remain
as portable lower-bound proxies.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import aiohttp


@dataclass(frozen=True)
class RequestSpec:
    rid: str
    group: str
    input_len: int
    output_len: int
    arrival_rank: int
    arrival_offset_s: float


@dataclass
class RequestResult:
    rid: str
    group: str
    input_len: int
    requested_output_len: int
    arrival_rank: int
    scheduled_arrival_s: float
    submitted_s: float = 0.0
    first_token_s: float | None = None
    finished_s: float = 0.0
    completion_tokens: int = 0
    num_retractions: int = 0
    finish_reason: Any = None
    success: bool = False
    error: str | None = None
    token_events: list[tuple[float, int]] = field(default_factory=list)

    @property
    def latency_s(self) -> float:
        return self.finished_s - self.submitted_s

    @property
    def ttft_s(self) -> float | None:
        if self.first_token_s is None:
            return None
        return self.first_token_s - self.submitted_s


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def evenly_interleave(groups: list[list[RequestSpec]]) -> list[RequestSpec]:
    """Interleave groups by normalized position, independent of group size."""
    tagged: list[tuple[float, int, int, RequestSpec]] = []
    for group_index, group in enumerate(groups):
        for item_index, item in enumerate(group):
            tagged.append((item_index / len(group), group_index, item_index, item))
    tagged.sort(key=lambda item: item[:3])
    return [item[-1] for item in tagged]


def make_group(
    run_id: str, name: str, count: int, input_len: int, output_len: int
) -> list[RequestSpec]:
    return [
        RequestSpec(
            rid=f"{run_id}-{name}-{index:04d}",
            group=name,
            input_len=input_len,
            output_len=output_len,
            arrival_rank=-1,
            arrival_offset_s=0.0,
        )
        for index in range(count)
    ]


def assign_arrivals(
    ordered: list[RequestSpec], arrival_gap_ms: float, boundary_gap_ms: float
) -> list[RequestSpec]:
    result: list[RequestSpec] = []
    extra_offset_s = 0.0
    previous_group: str | None = None
    for rank, spec in enumerate(ordered):
        if previous_group is not None and spec.group != previous_group:
            extra_offset_s += boundary_gap_ms / 1000.0
        result.append(
            RequestSpec(
                rid=spec.rid,
                group=spec.group,
                input_len=spec.input_len,
                output_len=spec.output_len,
                arrival_rank=rank,
                arrival_offset_s=rank * arrival_gap_ms / 1000.0 + extra_offset_s,
            )
        )
        previous_group = spec.group
    return result


def build_specs(args: argparse.Namespace) -> list[RequestSpec]:
    if args.mode == "lifetime-order":
        short = make_group(
            args.run_id,
            "short",
            args.short_count,
            args.short_input,
            args.short_output,
        )
        long = make_group(
            args.run_id, "long", args.long_count, args.long_input, args.long_output
        )
        if args.order == "short-first":
            ordered = short + long
        elif args.order == "long-first":
            ordered = long + short
        elif args.order == "interleave":
            ordered = evenly_interleave([short, long])
        else:
            raise ValueError(f"Unsupported lifetime order: {args.order}")
        boundary_gap_ms = args.boundary_gap_ms if args.order != "interleave" else 0.0
        return assign_arrivals(ordered, args.arrival_gap_ms, boundary_gap_ms)
    elif args.mode == "fairness-pair":
        state = make_group(
            args.run_id,
            "state-heavy",
            args.state_count,
            args.state_input,
            args.state_output,
        )
        kv = make_group(
            args.run_id,
            "kv-heavy",
            args.kv_count,
            args.kv_input,
            args.kv_output,
        )
        if args.order == "state-first":
            ordered = state + kv
        elif args.order == "kv-first":
            ordered = kv + state
        elif args.order == "interleave":
            ordered = evenly_interleave([state, kv])
        else:
            raise ValueError(f"Unsupported fairness order: {args.order}")

        boundary_gap_ms = args.boundary_gap_ms if args.order != "interleave" else 0.0
        return assign_arrivals(ordered, args.arrival_gap_ms, boundary_gap_ms)

    state = make_group(
        args.run_id,
        "state-heavy",
        args.waves * args.state_per_wave,
        args.state_input,
        args.state_output,
    )
    kv = make_group(
        args.run_id,
        "kv-heavy",
        args.waves * args.kv_per_wave,
        args.kv_input,
        args.kv_output,
    )
    sustained: list[RequestSpec] = []
    arrival_rank = 0
    for wave in range(args.waves):
        state_wave = state[
            wave * args.state_per_wave : (wave + 1) * args.state_per_wave
        ]
        kv_wave = kv[wave * args.kv_per_wave : (wave + 1) * args.kv_per_wave]
        if args.order == "state-first":
            wave_order = state_wave + kv_wave
        elif args.order == "kv-first":
            wave_order = kv_wave + state_wave
        elif args.order == "interleave":
            wave_order = evenly_interleave([state_wave, kv_wave])
        else:
            raise ValueError(f"Unsupported sustained order: {args.order}")

        for index, spec in enumerate(wave_order):
            sustained.append(
                RequestSpec(
                    rid=spec.rid,
                    group=spec.group,
                    input_len=spec.input_len,
                    output_len=spec.output_len,
                    arrival_rank=arrival_rank,
                    arrival_offset_s=(
                        wave * args.wave_interval_ms / 1000.0
                        + index * args.arrival_gap_ms / 1000.0
                    ),
                )
            )
            arrival_rank += 1
    return sustained


def make_input_ids(spec: RequestSpec) -> list[int]:
    # cache_salt isolates radix namespaces. Distinct valid low token ids make
    # accidental unsalted prefix sharing easy to detect from cached_tokens.
    group_offset = sum(ord(char) for char in spec.group) % 1000
    request_offset = int(spec.rid.rsplit("-", 1)[-1])
    token = 1000 + group_offset + request_offset
    return [token] * spec.input_len


async def flush_cache(session: aiohttp.ClientSession, base_url: str) -> None:
    async with session.post(f"{base_url}/flush_cache") as response:
        body = await response.text()
        if response.status != 200:
            raise RuntimeError(f"flush_cache failed ({response.status}): {body}")


async def get_server_info(
    session: aiohttp.ClientSession, base_url: str
) -> dict[str, Any]:
    async with session.get(f"{base_url}/get_server_info") as response:
        response.raise_for_status()
        return await response.json()


async def run_request(
    session: aiohttp.ClientSession,
    base_url: str,
    spec: RequestSpec,
    experiment_start: float,
    run_id: str,
) -> RequestResult:
    target = experiment_start + spec.arrival_offset_s
    delay = target - time.perf_counter()
    if delay > 0:
        await asyncio.sleep(delay)

    result = RequestResult(
        rid=spec.rid,
        group=spec.group,
        input_len=spec.input_len,
        requested_output_len=spec.output_len,
        arrival_rank=spec.arrival_rank,
        scheduled_arrival_s=spec.arrival_offset_s,
    )
    result.submitted_s = time.perf_counter() - experiment_start
    payload = {
        "rid": spec.rid,
        "input_ids": make_input_ids(spec),
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": spec.output_len,
            "ignore_eos": True,
        },
        "stream": True,
        "cache_salt": f"hybrid-motivation-{spec.rid}",
        "extra_key": f"hybrid-tenant:{run_id}:{spec.group}",
    }

    try:
        async with session.post(f"{base_url}/generate", json=payload) as response:
            if response.status != 200:
                result.error = f"HTTP {response.status}: {await response.text()}"
                return result

            while True:
                raw_line = await response.content.readline()
                if not raw_line:
                    break
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith(b"data: "):
                    line = line[6:]
                if line == b"[DONE]":
                    continue

                data = json.loads(line)
                meta = data.get("meta_info") or {}
                completion_tokens = int(meta.get("completion_tokens") or 0)
                event_time = time.perf_counter() - experiment_start
                if completion_tokens > result.completion_tokens:
                    result.completion_tokens = completion_tokens
                    result.token_events.append((event_time, completion_tokens))
                    if result.first_token_s is None:
                        result.first_token_s = event_time
                result.num_retractions = max(
                    result.num_retractions, int(meta.get("num_retractions") or 0)
                )
                if meta.get("finish_reason") is not None:
                    result.finish_reason = meta["finish_reason"]

            result.success = result.completion_tokens == spec.output_len
            if not result.success:
                result.error = (
                    f"expected {spec.output_len} completion tokens, "
                    f"received {result.completion_tokens}"
                )
    except Exception as exc:  # Keep every failed request in the raw artifact.
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        result.finished_s = time.perf_counter() - experiment_start
    return result


def integrate_decode_resources(
    results: Iterable[RequestResult], state_slots_per_request: int
) -> dict[str, float | int]:
    events: list[tuple[float, int, int]] = []
    for result in results:
        if not result.success or not result.token_events or result.first_token_s is None:
            continue
        first_time, first_count = result.token_events[0]
        resident_tokens = result.input_len + first_count
        events.append((first_time, state_slots_per_request, resident_tokens))
        previous_count = first_count
        for event_time, count in result.token_events[1:]:
            events.append((event_time, 0, count - previous_count))
            previous_count = count
        events.append(
            (
                result.finished_s,
                -state_slots_per_request,
                -(result.input_len + previous_count),
            )
        )

    events.sort(key=lambda event: event[0])
    state_slots = 0
    kv_tokens = 0
    peak_state_slots = 0
    peak_kv_tokens = 0
    state_slot_seconds = 0.0
    kv_token_seconds = 0.0
    previous_time = events[0][0] if events else 0.0
    for event_time, state_delta, kv_delta in events:
        duration = event_time - previous_time
        state_slot_seconds += state_slots * duration
        kv_token_seconds += kv_tokens * duration
        state_slots += state_delta
        kv_tokens += kv_delta
        peak_state_slots = max(peak_state_slots, state_slots)
        peak_kv_tokens = max(peak_kv_tokens, kv_tokens)
        previous_time = event_time

    return {
        "decode_state_slot_seconds_lower_bound": state_slot_seconds,
        "decode_kv_token_seconds_lower_bound": kv_token_seconds,
        "peak_decode_state_slots_lower_bound": peak_state_slots,
        "peak_decode_kv_tokens_lower_bound": peak_kv_tokens,
    }


def summarize_group(
    results: list[RequestResult], state_slots_per_request: int, slo_seconds: float
) -> dict[str, Any]:
    successful = [result for result in results if result.success]
    latencies = [result.latency_s for result in successful]
    ttfts = [result.ttft_s for result in successful if result.ttft_s is not None]
    summary: dict[str, Any] = {
        "requests": len(results),
        "completed": len(successful),
        "completion_rate": len(successful) / len(results) if results else 0.0,
        "input_tokens_completed": sum(result.input_len for result in successful),
        "output_tokens_completed": sum(
            result.completion_tokens for result in successful
        ),
        "mean_latency_s": statistics.fmean(latencies) if latencies else None,
        "p50_latency_s": percentile(latencies, 0.50),
        "p95_latency_s": percentile(latencies, 0.95),
        "mean_ttft_s": statistics.fmean(ttfts) if ttfts else None,
        "p50_ttft_s": percentile(ttfts, 0.50),
        "p95_ttft_s": percentile(ttfts, 0.95),
        "completed_within_slo": sum(
            result.latency_s <= slo_seconds for result in successful
        ),
        "slo_seconds": slo_seconds,
        "total_retractions": sum(result.num_retractions for result in successful),
    }
    summary.update(integrate_decode_resources(successful, state_slots_per_request))
    return summary


def summarize(
    results: list[RequestResult], state_slots_per_request: int, slo_seconds: float
) -> dict[str, Any]:
    by_group: dict[str, list[RequestResult]] = {}
    for result in results:
        by_group.setdefault(result.group, []).append(result)
    return {
        "overall": summarize_group(results, state_slots_per_request, slo_seconds),
        "groups": {
            group: summarize_group(items, state_slots_per_request, slo_seconds)
            for group, items in sorted(by_group.items())
        },
    }


def extract_tenant_resource_trace(
    server_info: dict[str, Any], run_id: str
) -> dict[str, Any] | None:
    run_prefix = f"hybrid-tenant:{run_id}:"
    for state in server_info.get("internal_states", []):
        trace = state.get("tenant_resource_trace")
        if not trace:
            continue
        tenants = {
            tenant.removeprefix(run_prefix): usage
            for tenant, usage in trace.get("tenants", {}).items()
            if tenant.startswith(run_prefix)
        }
        if tenants:
            return {**trace, "tenants": tenants, "run_id": run_id}
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=["lifetime-order", "fairness-pair", "fairness-sustained"],
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order", required=True)
    parser.add_argument(
        "--run-id",
        help="Unique experiment-run identifier; generated automatically if omitted.",
    )
    parser.add_argument("--arrival-gap-ms", type=float, default=1.0)
    parser.add_argument("--boundary-gap-ms", type=float, default=50.0)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--state-slots-per-request", type=int, default=5)
    parser.add_argument("--slo-seconds", type=float, default=10.0)
    parser.add_argument("--skip-flush-cache", action="store_true")
    parser.add_argument("--capacity-label")
    parser.add_argument("--policy-label")
    parser.add_argument("--scalar-weight", type=float)

    parser.add_argument("--short-count", type=int, default=96)
    parser.add_argument("--short-input", type=int, default=1024)
    parser.add_argument("--short-output", type=int, default=32)
    parser.add_argument("--long-count", type=int, default=24)
    parser.add_argument("--long-input", type=int, default=1024)
    parser.add_argument("--long-output", type=int, default=512)

    parser.add_argument("--state-count", type=int, default=96)
    parser.add_argument("--state-input", type=int, default=1024)
    parser.add_argument("--state-output", type=int, default=32)
    parser.add_argument("--kv-count", type=int, default=6)
    parser.add_argument("--kv-input", type=int, default=16384)
    parser.add_argument("--kv-output", type=int, default=512)
    parser.add_argument("--waves", type=int, default=12)
    parser.add_argument("--state-per-wave", type=int, default=16)
    parser.add_argument("--kv-per-wave", type=int, default=1)
    parser.add_argument("--wave-interval-ms", type=float, default=750.0)
    return parser


async def async_main(args: argparse.Namespace) -> None:
    args.base_url = args.base_url.rstrip("/")
    if args.run_id is None:
        args.run_id = f"{args.mode}-{args.order}-{time.time_ns()}"
    specs = build_specs(args)
    timeout = aiohttp.ClientTimeout(total=args.timeout_seconds)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        server_info = await get_server_info(session, args.base_url)
        if not args.skip_flush_cache:
            await flush_cache(session, args.base_url)
        experiment_start = time.perf_counter() + 0.25
        results = await asyncio.gather(
            *[
                run_request(
                    session, args.base_url, spec, experiment_start, args.run_id
                )
                for spec in specs
            ]
        )
        final_server_info = await get_server_info(session, args.base_url)

    resource_trace = extract_tenant_resource_trace(final_server_info, args.run_id)
    summary = summarize(results, args.state_slots_per_request, args.slo_seconds)
    if resource_trace is not None:
        for group, usage in resource_trace["tenants"].items():
            if group in summary["groups"]:
                summary["groups"][group]["scheduler_active_resources"] = usage

    artifact = {
        "schema_version": 2,
        "experiment": {
            "mode": args.mode,
            "order": args.order,
            "run_id": args.run_id,
            "base_url": args.base_url,
            "arrival_gap_ms": args.arrival_gap_ms,
            "boundary_gap_ms": args.boundary_gap_ms,
            "state_slots_per_request": args.state_slots_per_request,
            "slo_seconds": args.slo_seconds,
            "request_count": len(specs),
            "capacity_label": args.capacity_label,
            "policy_label": args.policy_label,
            "scalar_weight": args.scalar_weight,
            "waves": args.waves if args.mode == "fairness-sustained" else None,
            "state_per_wave": (
                args.state_per_wave if args.mode == "fairness-sustained" else None
            ),
            "kv_per_wave": (
                args.kv_per_wave if args.mode == "fairness-sustained" else None
            ),
            "wave_interval_ms": (
                args.wave_interval_ms if args.mode == "fairness-sustained" else None
            ),
        },
        "server": {
            "version": server_info.get("version"),
            "model_path": server_info.get("model_path"),
            "schedule_policy": server_info.get("schedule_policy"),
            "retraction_policy": server_info.get("retraction_policy"),
            "enable_unified_memory": server_info.get("enable_unified_memory"),
            "max_total_num_tokens": server_info.get("max_total_num_tokens"),
            "internal_states": [
                {
                    "schedule_policy": state.get("schedule_policy"),
                    "tenant_vtc_request_weight": state.get(
                        "tenant_vtc_request_weight"
                    ),
                    "max_mamba_cache_size": state.get("max_mamba_cache_size"),
                    "effective_max_running_requests_per_dp": state.get(
                        "effective_max_running_requests_per_dp"
                    ),
                    "mamba_ssm_dtype": state.get("mamba_ssm_dtype"),
                    "mamba_radix_cache_strategy": state.get(
                        "mamba_radix_cache_strategy"
                    ),
                }
                for state in server_info.get("internal_states", [])
            ],
        },
        "summary": summary,
        "scheduler_resource_trace": resource_trace,
        "requests": [
            {
                **{
                    key: value
                    for key, value in asdict(result).items()
                    if key != "token_events"
                },
                "latency_s": result.latency_s,
                "ttft_s": result.ttft_s,
                "token_events": result.token_events,
            }
            for result in sorted(results, key=lambda item: item.arrival_rank)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact["summary"], indent=2))
    failed = [result for result in results if not result.success]
    if failed:
        raise SystemExit(f"{len(failed)} requests failed; inspect {args.output}")


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
