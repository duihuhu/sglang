"""Indexed SSE replay and fail-closed NVML/SLO accounting."""

import asyncio
import json
import math
import time

TTFT_SLO_MS = 5000.0
TPOT_SLO_MS = 300.0


async def _send_one(
    session,
    url: str,
    req: dict,
    base_time: float,
    index: int,
    results: list[dict],
) -> None:
    delay = req["arrival_time_s"] - (time.monotonic() - base_time)
    if delay > 0:
        await asyncio.sleep(delay)

    prompt = (
        {"input_ids": [1000] * req["input_len"]}
        if req.get("use_input_ids")
        else {"text": "x" * req["input_len"]}
    )
    payload = {
        **prompt,
        "sampling_params": {
            "max_new_tokens": req["output_len"],
            "temperature": 0.0,
            "ignore_eos": True,
        },
        "stream": True,
    }

    t0 = time.monotonic()
    first_token_time = None
    token_count = 0
    last_meta: dict = {}
    http_status = None

    try:
        async with session.post(url, json=payload) as resp:
            http_status = resp.status
            if resp.status != 200:
                results.append(
                    {
                        "request_index": index,
                        "success": False,
                        "http_status": http_status,
                        "input_len": req["input_len"],
                        "output_len": req["output_len"],
                        "arrival_time_s": req["arrival_time_s"],
                    }
                )
                return
            async for line in resp.content:
                now = time.monotonic()
                text = line.decode().strip()
                if not text or text.startswith(":"):
                    continue
                if text.startswith("data:"):
                    text = text[5:].strip()
                if text == "[DONE]":
                    break
                try:
                    chunk = json.loads(text)
                    if first_token_time is None:
                        first_token_time = now
                    token_count += 1
                    if isinstance(chunk, dict) and "meta_info" in chunk:
                        last_meta = chunk["meta_info"]
                except json.JSONDecodeError:
                    pass
    except Exception as exc:
        results.append(
            {
                "request_index": index,
                "success": False,
                "error": repr(exc),
                "input_len": req["input_len"],
                "output_len": req["output_len"],
                "arrival_time_s": req["arrival_time_s"],
            }
        )
        return

    t_end = time.monotonic()
    streamed_chunks = token_count
    reported_completion_tokens = last_meta.get("completion_tokens")
    finish_reason = last_meta.get("finish_reason")
    finish_type = (
        finish_reason.get("type") if isinstance(finish_reason, dict) else finish_reason
    )
    complete = (
        first_token_time is not None
        and finish_type == "length"
        and reported_completion_tokens is not None
        and int(reported_completion_tokens) == int(req["output_len"])
    )
    if not complete:
        results.append(
            {
                "request_index": index,
                "success": False,
                "http_status": http_status,
                "error": "INCOMPLETE_STREAM",
                "input_len": req["input_len"],
                "output_len": req["output_len"],
                "arrival_time_s": req["arrival_time_s"],
                "streamed_chunks": streamed_chunks,
                "reported_completion_tokens": reported_completion_tokens,
                "finish_reason": finish_reason,
            }
        )
        return
    token_count = int(reported_completion_tokens)
    ttft_ms = (first_token_time - t0) * 1000 if first_token_time else 0.0
    ttft_proc_ms = 0.0
    if last_meta.get("ttft_pure_processing"):
        ttft_proc_ms = last_meta["ttft_pure_processing"] * 1000
    elif last_meta.get("time_to_first_token_processing"):
        ttft_proc_ms = last_meta["time_to_first_token_processing"] * 1000
    tpot_ms = 0.0
    if token_count > 1 and first_token_time:
        tpot_ms = (t_end - first_token_time) * 1000 / (token_count - 1)

    results.append(
        {
            "request_index": index,
            "success": True,
            "http_status": http_status,
            "input_len": req["input_len"],
            "output_len": req["output_len"],
            "arrival_time_s": req["arrival_time_s"],
            "completion_tokens": token_count,
            "streamed_chunks": streamed_chunks,
            "ttft_ms": round(ttft_ms, 3),
            "ttft_proc_ms": round(ttft_proc_ms, 3),
            "tpot_ms": round(tpot_ms, 3),
            "e2e_s": round(t_end - t0, 4),
            "ttft_violation": bool((ttft_proc_ms or ttft_ms) > TTFT_SLO_MS),
            "tpot_violation": bool(tpot_ms > TPOT_SLO_MS),
        }
    )


def summarize(reqs, results, duration, energy_j, timed_out=0):
    ok = [r for r in results if r.get("success")]
    failed = len(results) - len(ok)
    missing = max(0, len(reqs) - len(results) - timed_out)
    violated = sum(
        (r.get("ttft_proc_ms") or r["ttft_ms"]) > TTFT_SLO_MS
        or r["tpot_ms"] > TPOT_SLO_MS
        for r in ok
    )
    inp = sum(r["input_len"] for r in ok)
    out = sum(r["completion_tokens"] for r in ok)
    summary = {
        "status": (
            "PASS"
            if len(ok) == len(reqs)
            else (
                "PARTIAL_TIMEOUT"
                if timed_out and ok
                else "TIMEOUT" if timed_out else "FAIL"
            )
        ),
        "total_requests": len(reqs),
        "successful": len(ok),
        "failed": failed,
        "timed_out": timed_out,
        "missing": missing,
        "duration_s": duration,
        "total_tokens": out,
        "total_input_tokens": inp,
        "total_tokens_all": inp + out,
        "throughput_tok_s": out / duration if duration else 0,
        "total_energy_j": energy_j,
        "energy_per_token_mj": energy_j * 1000 / (inp + out) if inp + out else None,
        "energy_denominator": "successful_input_plus_output",
        "slo_violating_requests": violated,
        "slo_violation_rate": (
            100 * (violated + len(reqs) - len(ok)) / len(reqs) if reqs else 0
        ),
        "request_results": sorted(results, key=lambda r: r["request_index"]),
    }
    for metric, values in (
        ("ttft_proc", [(r.get("ttft_proc_ms") or r["ttft_ms"]) for r in ok]),
        ("tpot", [r["tpot_ms"] for r in ok]),
    ):
        values = sorted(values)
        summary[metric + "_avg_ms"] = sum(values) / len(values) if values else 0
        for percentile in (50, 90, 95, 99):
            pos = (len(values) - 1) * percentile / 100
            lo, hi = math.floor(pos), math.ceil(pos)
            summary[f"{metric}_p{percentile}_ms"] = (
                values[lo] + (values[hi] - values[lo]) * (pos - lo) if values else 0
            )
    return summary


async def run_requests(reqs, runtime, hosts, max_run_s):
    import aiohttp

    start = {
        host: runtime.get_energy_on_host(host, gpus)
        for host, gpus in hosts.items()
        if gpus
    }
    results = []
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=max_run_s + 60)
    ) as session:
        base = time.monotonic()
        tasks = [
            asyncio.create_task(
                _send_one(session, r["_target_url"], r, base, i, results)
            )
            for i, r in enumerate(reqs)
        ]
        done, pending = await asyncio.wait(tasks, timeout=max_run_s)
        for task in pending:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    duration = time.monotonic() - base
    end = {
        host: runtime.get_energy_on_host(host, gpus)
        for host, gpus in hosts.items()
        if gpus
    }
    energy = 0
    for host, gpus in hosts.items():
        for gpu in gpus:
            a, b = start[host].get(gpu), end[host].get(gpu)
            if a is None or b is None or a <= 0 or b < a:
                raise RuntimeError(f"invalid NVML counter: {host} GPU{gpu}: {a} -> {b}")
            energy += (b - a) / 1000
    return summarize(reqs, results, duration, energy, len(pending))
