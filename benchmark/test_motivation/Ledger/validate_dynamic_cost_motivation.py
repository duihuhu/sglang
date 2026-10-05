#!/usr/bin/env python3
"""Controlled request-cost, reuse, and batch-externality motivation experiments."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import aiohttp


@dataclass(frozen=True)
class RequestSpec:
    rid: str
    group: str
    input_ids: list[int]
    output_len: int
    arrival_offset_s: float
    cache_salt: str
    common_prefix_len: int = 0
    tenant_key: str | None = None


@dataclass
class RequestResult:
    rid: str
    group: str
    input_len: int
    output_len: int
    common_prefix_len: int
    scheduled_arrival_s: float
    submitted_s: float = 0.0
    first_token_s: float | None = None
    finished_s: float = 0.0
    completion_tokens: int = 0
    cached_tokens: int = 0
    prompt_tokens: int = 0
    num_retractions: int = 0
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


def token(seed: int) -> int:
    return 1000 + seed % 150000


def unique_input(length: int, seed: int) -> list[int]:
    return [token(seed)] * length


def prefixed_input(total_len: int, prefix_len: int, seed: int) -> list[int]:
    if prefix_len <= 0:
        return unique_input(total_len, seed)
    return [token(777)] * prefix_len + [token(seed)] * (total_len - prefix_len)


async def post_json(
    session: aiohttp.ClientSession, url: str, payload: dict[str, Any] | None = None
) -> Any:
    async with session.post(url, json=payload) as response:
        body = await response.text()
        if response.status != 200:
            raise RuntimeError(f"POST {url} failed ({response.status}): {body}")
        if not body:
            return None
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return body


async def get_json(session: aiohttp.ClientSession, url: str) -> Any:
    async with session.get(url) as response:
        response.raise_for_status()
        return await response.json()


async def reset_tracker(session: aiohttp.ClientSession, base_url: str) -> None:
    await configure_policy(session, base_url, 0, 0)


async def configure_policy(
    session: aiohttp.ClientSession,
    base_url: str,
    policy_code: int | None,
    request_weight: float,
) -> None:
    server_args: dict[str, float | int] = {
        "tenant_vtc_request_weight": request_weight
    }
    if policy_code is not None:
        server_args["tenant_schedule_policy_code"] = policy_code
    result = await post_json(
        session,
        f"{base_url}/set_internal_state",
        {"server_args": server_args},
    )
    if not result or not all(result):
        raise RuntimeError(f"scheduler reset failed: {result}")


async def flush_cache(session: aiohttp.ClientSession, base_url: str) -> None:
    await post_json(session, f"{base_url}/flush_cache")


async def run_request(
    session: aiohttp.ClientSession,
    base_url: str,
    spec: RequestSpec,
    experiment_start: float,
    run_id: str,
) -> RequestResult:
    delay = experiment_start + spec.arrival_offset_s - time.perf_counter()
    if delay > 0:
        await asyncio.sleep(delay)
    result = RequestResult(
        rid=spec.rid,
        group=spec.group,
        input_len=len(spec.input_ids),
        output_len=spec.output_len,
        common_prefix_len=spec.common_prefix_len,
        scheduled_arrival_s=spec.arrival_offset_s,
    )
    result.submitted_s = time.perf_counter() - experiment_start
    payload = {
        "rid": spec.rid,
        "input_ids": spec.input_ids,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": spec.output_len,
            "ignore_eos": True,
        },
        "stream": True,
        "cache_salt": spec.cache_salt,
        "extra_key": f"hybrid-tenant:{run_id}:{spec.tenant_key or spec.group}",
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
                result.cached_tokens = max(
                    result.cached_tokens, int(meta.get("cached_tokens") or 0)
                )
                result.prompt_tokens = max(
                    result.prompt_tokens, int(meta.get("prompt_tokens") or 0)
                )
                result.num_retractions = max(
                    result.num_retractions, int(meta.get("num_retractions") or 0)
                )
                completion_tokens = int(meta.get("completion_tokens") or 0)
                event_time = time.perf_counter() - experiment_start
                if completion_tokens > result.completion_tokens:
                    result.completion_tokens = completion_tokens
                    result.token_events.append((event_time, completion_tokens))
                    if result.first_token_s is None:
                        result.first_token_s = event_time
            result.success = result.completion_tokens == spec.output_len
            if not result.success:
                result.error = (
                    f"expected {spec.output_len} completion tokens, "
                    f"received {result.completion_tokens}"
                )
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        result.finished_s = time.perf_counter() - experiment_start
    return result


async def run_specs(
    session: aiohttp.ClientSession,
    base_url: str,
    specs: list[RequestSpec],
    experiment_start: float,
    run_id: str,
) -> list[RequestResult]:
    return await asyncio.gather(
        *[
            run_request(session, base_url, spec, experiment_start, run_id)
            for spec in specs
        ]
    )


def warm_spec(
    run_id: str,
    prefix_len: int,
    salt: str,
    target_group: str,
    *,
    tenant_key: str | None = None,
) -> RequestSpec:
    return RequestSpec(
        rid=f"{run_id}-warm",
        group=target_group,
        input_ids=[token(777)] * prefix_len,
        output_len=1,
        arrival_offset_s=0,
        cache_salt=salt,
        common_prefix_len=prefix_len,
        tenant_key=tenant_key,
    )


def make_group(
    run_id: str,
    group: str,
    count: int,
    prompt_len: int,
    output_len: int,
    *,
    start_offset: float = 0,
    gap_ms: float = 1,
    common_prefix_len: int = 0,
    shared_salt: str | None = None,
    tenant_key: str | None = None,
) -> list[RequestSpec]:
    specs = []
    for index in range(count):
        salt = shared_salt or f"dynamic-cost:{run_id}:{group}:{index}"
        seed_material = f"{run_id}\0{group}\0{index}".encode()
        seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
        specs.append(
            RequestSpec(
                rid=f"{run_id}-{group}-{index:04d}",
                group=group,
                input_ids=prefixed_input(
                    prompt_len, common_prefix_len, seed
                ),
                output_len=output_len,
                arrival_offset_s=start_offset + index * gap_ms / 1000,
                cache_salt=salt,
                common_prefix_len=common_prefix_len,
                tenant_key=tenant_key,
            )
        )
    return specs


def interleave(groups: list[list[RequestSpec]]) -> list[RequestSpec]:
    tagged = []
    for group_index, group in enumerate(groups):
        for item_index, item in enumerate(group):
            tagged.append((item_index / max(len(group), 1), group_index, item))
    return [item[-1] for item in sorted(tagged, key=lambda item: item[:2])]


def reassign_offsets(
    specs: list[RequestSpec], *, base_offset: float = 0, gap_ms: float = 1
) -> list[RequestSpec]:
    return [
        RequestSpec(
            rid=spec.rid,
            group=spec.group,
            input_ids=spec.input_ids,
            output_len=spec.output_len,
            arrival_offset_s=base_offset + index * gap_ms / 1000,
            cache_salt=spec.cache_salt,
            common_prefix_len=spec.common_prefix_len,
            tenant_key=spec.tenant_key,
        )
        for index, spec in enumerate(specs)
    ]


def build_pair(args: argparse.Namespace) -> tuple[list[RequestSpec], int, str | None]:
    shared_salt = f"dynamic-cost:{args.run_id}:shared"
    if args.pair == "state-kv":
        left = make_group(args.run_id, "many-short", 96, 1024, 32)
        right = make_group(args.run_id, "few-long", 6, 16384, 512)
        prefix_len = 0
        salt = None
    elif args.pair == "lifetime":
        left = make_group(args.run_id, "short-total-long-output", 48, 1024, 512)
        right = make_group(args.run_id, "long-total-short-output", 48, 4096, 32)
        prefix_len = 0
        salt = None
    elif args.pair == "reuse":
        left = make_group(args.run_id, "short-unique", 48, 4096, 32)
        prefix_len = int(16384 * 0.9)
        right = make_group(
            args.run_id,
            "long-reused",
            48,
            16384,
            32,
            common_prefix_len=prefix_len,
            shared_salt=shared_salt,
        )
        salt = shared_salt
    else:
        raise ValueError(args.pair)

    if args.order == "left-first":
        ordered = left + right
    elif args.order == "right-first":
        ordered = right + left
    elif args.order == "interleave":
        ordered = interleave([left, right])
    else:
        raise ValueError(args.order)
    return reassign_offsets(ordered, gap_ms=args.arrival_gap_ms), prefix_len, salt


def build_cost(args: argparse.Namespace) -> tuple[list[RequestSpec], int, str | None]:
    prefix_len = int(args.prompt_len * args.reuse_fraction)
    shared_salt = f"dynamic-cost:{args.run_id}:shared" if prefix_len else None
    return (
        make_group(
            args.run_id,
            "measured",
            args.count,
            args.prompt_len,
            args.output_len,
            common_prefix_len=prefix_len,
            shared_salt=shared_salt,
        ),
        prefix_len,
        shared_salt,
    )


def build_batch(args: argparse.Namespace) -> tuple[list[RequestSpec], int, str | None]:
    background = make_group(
        args.run_id, "background", args.background_count, 1024, 512
    )
    if args.candidate == "none":
        return background, 0, None
    prefix_len = int(16384 * 0.9) if args.candidate == "long-reused" else 0
    prompt_len = 16384 if args.candidate.startswith("long-") else 1024
    shared_salt = f"dynamic-cost:{args.run_id}:shared" if prefix_len else None
    candidate = make_group(
        args.run_id,
        "candidate",
        args.candidate_count,
        prompt_len,
        32,
        start_offset=args.candidate_delay_s,
        common_prefix_len=prefix_len,
        shared_salt=shared_salt,
    )
    return background + candidate, prefix_len, shared_salt


def eviction_probe(
    args: argparse.Namespace, group: str, salt: str
) -> RequestSpec:
    prefix_len = args.eviction_prefix_len
    return RequestSpec(
        rid=f"{args.run_id}-{group}",
        group=group,
        input_ids=prefixed_input(prefix_len + 32, prefix_len, 9001),
        output_len=1,
        arrival_offset_s=0,
        cache_salt=salt,
        common_prefix_len=prefix_len,
        tenant_key="cache-probe",
    )


def build_eviction_pressure(args: argparse.Namespace) -> list[RequestSpec]:
    if args.pressure == "none":
        return []
    if args.pressure == "state":
        return make_group(
            args.run_id,
            "pressure",
            args.state_pressure_count,
            1024,
            512,
        )
    if args.pressure == "kv":
        return make_group(
            args.run_id,
            "pressure",
            args.kv_pressure_count,
            16384,
            32,
        )
    raise ValueError(args.pressure)


def make_phase(
    args: argparse.Namespace,
    phase_name: str,
    phase_kind: str,
    base_offset_s: float,
) -> list[RequestSpec]:
    if phase_kind == "state":
        a_per_wave, a_prompt, a_output = 16, 1024, 32
    elif phase_kind == "kv":
        a_per_wave, a_prompt, a_output = 4, 4096, 128
    else:
        raise ValueError(phase_kind)
    b_per_wave, b_prompt, b_output = 1, 16384, 512
    a = make_group(
        args.run_id,
        f"{phase_name}-a",
        args.phase_waves * a_per_wave,
        a_prompt,
        a_output,
        tenant_key="user-a",
    )
    b = make_group(
        args.run_id,
        f"{phase_name}-b",
        args.phase_waves * b_per_wave,
        b_prompt,
        b_output,
        tenant_key="user-b",
    )
    specs: list[RequestSpec] = []
    for wave in range(args.phase_waves):
        a_wave = a[wave * a_per_wave : (wave + 1) * a_per_wave]
        b_wave = b[wave * b_per_wave : (wave + 1) * b_per_wave]
        ordered = interleave([a_wave, b_wave])
        wave_base = base_offset_s + wave * args.phase_wave_interval_s
        specs.extend(
            reassign_offsets(
                ordered,
                base_offset=wave_base,
                gap_ms=args.arrival_gap_ms,
            )
        )
    return specs


def build_phase(args: argparse.Namespace) -> tuple[list[RequestSpec], list[tuple[float, float]]]:
    if args.phase_pattern == "state":
        kinds = ["state"]
    elif args.phase_pattern == "kv":
        kinds = ["kv"]
    elif args.phase_pattern == "switch":
        kinds = ["state", "kv", "state"]
    else:
        raise ValueError(args.phase_pattern)
    specs: list[RequestSpec] = []
    for index, kind in enumerate(kinds):
        specs.extend(
            make_phase(
                args,
                f"phase{index + 1}-{kind}",
                kind,
                index * args.phase_duration_s,
            )
        )

    updates: list[tuple[float, float]] = []
    if args.policy == "dynamic-scalar":
        weights = [float(value) for value in args.phase_weights.split(",")]
        if len(weights) != len(kinds):
            raise ValueError(
                f"--phase-weights needs {len(kinds)} values for {args.phase_pattern}"
            )
        updates = [
            (index * args.phase_duration_s, weight)
            for index, weight in enumerate(weights[1:], start=1)
        ]
        args.scalar_weight = weights[0]
    return specs, updates


async def apply_weight_updates(
    session: aiohttp.ClientSession,
    base_url: str,
    experiment_start: float,
    updates: list[tuple[float, float]],
) -> list[dict[str, Any]]:
    applied = []
    for offset_s, weight in updates:
        delay = experiment_start + offset_s - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        started = time.perf_counter()
        await configure_policy(session, base_url, None, weight)
        applied.append(
            {
                "scheduled_offset_s": offset_s,
                "applied_offset_s": time.perf_counter() - experiment_start,
                "request_weight": weight,
                "update_latency_s": time.perf_counter() - started,
            }
        )
    return applied


def itl_values(result: RequestResult) -> list[float]:
    return [
        current[0] - previous[0]
        for previous, current in zip(result.token_events, result.token_events[1:])
    ]


def summarize(results: list[RequestResult], slo_seconds: float) -> dict[str, Any]:
    groups: dict[str, list[RequestResult]] = {}
    for result in results:
        groups.setdefault(result.group, []).append(result)

    def one(items: list[RequestResult]) -> dict[str, Any]:
        successful = [item for item in items if item.success]
        latencies = [item.latency_s for item in successful]
        ttfts = [item.ttft_s for item in successful if item.ttft_s is not None]
        itls = [value for item in successful for value in itl_values(item)]
        return {
            "requests": len(items),
            "completed": len(successful),
            "mean_jct_s": statistics.fmean(latencies) if latencies else None,
            "p95_jct_s": percentile(latencies, 0.95),
            "mean_ttft_s": statistics.fmean(ttfts) if ttfts else None,
            "p95_ttft_s": percentile(ttfts, 0.95),
            "p95_itl_s": percentile(itls, 0.95),
            "max_itl_s": max(itls) if itls else None,
            "slo_goodput": sum(item.latency_s <= slo_seconds for item in successful),
            "mean_cached_tokens": (
                statistics.fmean(item.cached_tokens for item in successful)
                if successful
                else None
            ),
            "total_retractions": sum(item.num_retractions for item in items),
        }

    return {
        "overall": one(results),
        "groups": {name: one(items) for name, items in sorted(groups.items())},
    }


def extract_trace(server_info: dict[str, Any], run_id: str) -> dict[str, Any] | None:
    prefix = f"hybrid-tenant:{run_id}:"
    for state in server_info.get("internal_states", []):
        trace = state.get("tenant_resource_trace")
        if not trace:
            continue
        tenants = {
            tenant.removeprefix(prefix): usage
            for tenant, usage in trace.get("tenants", {}).items()
            if tenant.startswith(prefix)
        }
        if tenants:
            return {**trace, "tenants": tenants, "run_id": run_id}
    return None


async def async_main(args: argparse.Namespace) -> None:
    base_url = args.base_url.rstrip("/")
    timeout = aiohttp.ClientTimeout(total=args.timeout_seconds)
    async with aiohttp.ClientSession(
        timeout=timeout, connector=aiohttp.TCPConnector(limit=0)
    ) as session:
        initial_info = await get_json(session, f"{base_url}/get_server_info")
        await reset_tracker(session, base_url)
        await flush_cache(session, base_url)

        if args.mode == "cost":
            specs, prefix_len, shared_salt = build_cost(args)
        elif args.mode == "pair":
            specs, prefix_len, shared_salt = build_pair(args)
        elif args.mode == "batch":
            specs, prefix_len, shared_salt = build_batch(args)
        elif args.mode == "eviction":
            prefix_len = args.eviction_prefix_len
            shared_salt = f"dynamic-cost:{args.run_id}:eviction-shared"
            specs = []
        elif args.mode == "phase":
            specs, weight_updates = build_phase(args)
            prefix_len = 0
            shared_salt = None
        else:
            raise ValueError(args.mode)

        if args.mode == "eviction":
            warm_start = time.perf_counter() + 0.1
            warm = await run_specs(
                session,
                base_url,
                [
                    warm_spec(
                        args.run_id,
                        prefix_len,
                        shared_salt,
                        "probe-warmup",
                        tenant_key="cache-probe",
                    )
                ],
                warm_start,
                args.run_id,
            )
            if not warm[0].success:
                raise RuntimeError(f"prefix warmup failed: {warm[0].error}")
            await reset_tracker(session, base_url)

            before = await run_specs(
                session,
                base_url,
                [eviction_probe(args, "probe-before", shared_salt)],
                time.perf_counter() + 0.1,
                args.run_id,
            )
            pressure_specs = build_eviction_pressure(args)
            pressure = (
                await run_specs(
                    session,
                    base_url,
                    pressure_specs,
                    time.perf_counter() + 0.1,
                    args.run_id,
                )
                if pressure_specs
                else []
            )
            after = await run_specs(
                session,
                base_url,
                [eviction_probe(args, "probe-after", shared_salt)],
                time.perf_counter() + 0.1,
                args.run_id,
            )
            results = before + pressure + after
        elif args.mode == "phase":
            policy_code = {
                "fcfs": 0,
                "scalar": 1,
                "dynamic-scalar": 1,
                "vector": 2,
            }[args.policy]
            await configure_policy(
                session, base_url, policy_code, args.scalar_weight
            )
            experiment_start = time.perf_counter() + 0.25
            results, applied_updates = await asyncio.gather(
                run_specs(session, base_url, specs, experiment_start, args.run_id),
                apply_weight_updates(
                    session,
                    base_url,
                    experiment_start,
                    weight_updates,
                ),
            )
        elif prefix_len and shared_salt:
            warm_group = {
                "cost": "measured",
                "pair": "long-reused",
                "batch": "candidate",
            }[args.mode]
            warm_start = time.perf_counter() + 0.1
            warm = await run_specs(
                session,
                base_url,
                [warm_spec(args.run_id, prefix_len, shared_salt, warm_group)],
                warm_start,
                args.run_id,
            )
            if not warm[0].success:
                raise RuntimeError(f"prefix warmup failed: {warm[0].error}")
            await reset_tracker(session, base_url)
            experiment_start = time.perf_counter() + 0.25
            results = await run_specs(
                session, base_url, specs, experiment_start, args.run_id
            )
        else:
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
    artifact = {
        "schema_version": 1,
        "experiment": {
            key: value
            for key, value in vars(args).items()
            if key not in {"output"}
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
        },
        "summary": summary,
        "scheduler_resource_trace": trace,
        "policy_updates": applied_updates if args.mode == "phase" else [],
        "requests": [
            {**asdict(result), "latency_s": result.latency_s, "ttft_s": result.ttft_s}
            for result in results
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    failed = [item for item in results if not item.success]
    if failed:
        raise SystemExit(f"{len(failed)} requests failed; inspect {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["cost", "pair", "batch", "eviction", "phase"],
        required=True,
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    parser.add_argument("--slo-seconds", type=float, default=20)
    parser.add_argument("--arrival-gap-ms", type=float, default=1)

    parser.add_argument("--prompt-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--reuse-fraction", type=float, default=0)

    parser.add_argument("--pair", choices=["state-kv", "lifetime", "reuse"])
    parser.add_argument(
        "--order", choices=["left-first", "right-first", "interleave"]
    )

    parser.add_argument(
        "--candidate",
        choices=["none", "short-unique", "long-unique", "long-reused"],
    )
    parser.add_argument("--background-count", type=int, default=64)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--candidate-delay-s", type=float, default=0.75)
    parser.add_argument("--pressure", choices=["none", "state", "kv"])
    parser.add_argument("--eviction-prefix-len", type=int, default=16384)
    parser.add_argument("--state-pressure-count", type=int, default=80)
    parser.add_argument("--kv-pressure-count", type=int, default=16)
    parser.add_argument("--phase-pattern", choices=["state", "kv", "switch"])
    parser.add_argument(
        "--policy", choices=["fcfs", "scalar", "dynamic-scalar", "vector"]
    )
    parser.add_argument("--scalar-weight", type=float, default=0)
    parser.add_argument("--phase-weights", default="65536,0,65536")
    parser.add_argument("--phase-waves", type=int, default=4)
    parser.add_argument("--phase-wave-interval-s", type=float, default=0.75)
    parser.add_argument("--phase-duration-s", type=float, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "pair" and (args.pair is None or args.order is None):
        raise SystemExit("--pair and --order are required for pair mode")
    if args.mode == "batch" and args.candidate is None:
        raise SystemExit("--candidate is required for batch mode")
    if args.mode == "eviction" and args.pressure is None:
        raise SystemExit("--pressure is required for eviction mode")
    if args.mode == "phase" and (args.phase_pattern is None or args.policy is None):
        raise SystemExit("--phase-pattern and --policy are required for phase mode")
    if not 0 <= args.reuse_fraction < 1:
        raise SystemExit("--reuse-fraction must be in [0, 1)")
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
