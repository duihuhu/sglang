#!/usr/bin/env python3
"""Shared benchmark helpers with per-request recording and full percentile stats."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np

log = logging.getLogger("macro_bench")

MACRO_ROOT = Path(__file__).resolve().parent
MULTI_NODE = MACRO_ROOT.parents[1]
WORKLOAD_DIR = MACRO_ROOT / "data" / "workloads"
DATA_DIR = MACRO_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ALL_DATA_FILE = DATA_DIR / "macro_e2e_all.json"
RESULT_PREFIX = "macro_e2e"

DATASETS = ("code", "conv")
QPS_LIST = (2, 4, 8, 16)
TTFT_SLO_MS = 5000.0
TPOT_SLO_MS = 300.0
MAX_RUN_S = 400

SCHEME_DEPLOY: dict[str, dict] = {
    "native_tp1_baseline": {
        "label": "SGLang", "topology": "native_tp1", "tier": False,
        "gpus": 16, "nodes": 2, "tp": 1, "freq_policy": "max_locked",
        "deploy_policy": "restart_per_point",
    },
    "native_tp1_tier": {
        "label": "DynamoLLM", "topology": "native_tp1", "tier": True,
        "gpus": 16, "nodes": 2, "tp": 1, "freq_policy": "tier_dvfs",
        "deploy_policy": "restart_per_point",
    },
    "pd_hetero_baseline": {
        "label": "DistServe", "topology": "pd_disagg", "tier": False,
        "prefill": "2x TP4", "decode": "4x TP2", "gpus": 16, "nodes": 2,
        "deploy_policy": "restart_per_point",
    },
    "pd_hetero_tier_biscale": {
        "label": "BiScale", "topology": "pd_disagg", "tier": True,
        "prefill": "2x TP4", "decode": "4x TP2", "gpus": 16, "nodes": 2,
        "deploy_policy": "restart_per_point",
    },
    "aflex_tier1": {
        "label": "AFlex", "topology": "tier1_afd", "tier": True,
        "deploy_policy": "plan_dense_e2e_per_qps",
        "layout_source": "plan_dense_e2e.json",
    },
}

SCHEME_LABELS = {k: v["label"] for k, v in SCHEME_DEPLOY.items()}


def wl_key(dataset: str, qps: int) -> str:
    return f"{dataset}_qps{qps}"


def qps_key(qps: int) -> str:
    return f"qps_{qps}"


def workload_file(dataset: str, qps: int) -> Path | None:
    path = WORKLOAD_DIR / f"macro_{dataset}_qps{qps}.jsonl"
    return path if path.exists() else None


def load_workload(dataset: str, qps: int) -> list[dict]:
    path = workload_file(dataset, qps)
    if path is None:
        raise FileNotFoundError(f"missing workload: macro_{dataset}_qps{qps}.jsonl")
    with open(path) as f:
        return [json.loads(line) for line in f]


def parse_wl_key(key: str) -> tuple[str, int]:
    dataset, qps_s = key.rsplit("_qps", 1)
    return dataset, int(qps_s)


def workload_meta(dataset: str, qps: int) -> dict:
    reqs = load_workload(dataset, qps)
    inp = sum(r["input_len"] for r in reqs)
    out = sum(r["output_len"] for r in reqs)
    return {
        "workload_file": f"macro_{dataset}_qps{qps}.jsonl",
        "n_requests": len(reqs),
        "total_input_tokens": inp,
        "total_output_tokens": out,
        "total_tokens_all": inp + out,
    }


def workload_token_totals(dataset: str, qps: int) -> tuple[int, int, int]:
    m = workload_meta(dataset, qps)
    return m["total_input_tokens"], m["total_output_tokens"], m["total_tokens_all"]


def _token_totals_from_run(ok: list[dict], reqs: list[dict]) -> tuple[int, int, int]:
    req_by_idx = {i: req for i, req in enumerate(reqs)}
    total_output_tokens = sum(r.get("completion_tokens", 0) for r in ok)
    total_input_tokens = 0
    for r in ok:
        idx = r.get("request_index")
        inp = r.get("input_len")
        if inp is None and idx is not None:
            inp = req_by_idx.get(idx, {}).get("input_len", 0)
        total_input_tokens += inp or 0
    return total_input_tokens, total_output_tokens, total_input_tokens + total_output_tokens


def recompute_energy_per_total_token(entry: dict, wl: str) -> dict:
    if not isinstance(entry, dict) or entry.get("status") not in ("PASS", "PARTIAL_TIMEOUT"):
        return entry
    dataset, qps = parse_wl_key(wl)
    _, _, workload_total = workload_token_totals(dataset, qps)
    if workload_total <= 0:
        return entry
    out = dict(entry)
    total_energy_j = entry.get("total_energy_j")
    if total_energy_j is not None:
        out["energy_per_token_mj"] = round(total_energy_j * 1000 / workload_total, 2)
    elif entry.get("energy_per_token_mj") and entry.get("total_tokens"):
        out["energy_per_token_mj"] = round(
            entry["energy_per_token_mj"] * entry["total_tokens"] / workload_total, 2
        )
    inp, _, total_all = workload_token_totals(dataset, qps)
    out["total_input_tokens"] = inp
    out["total_tokens_all"] = total_all
    out["energy_denominator"] = "input_plus_output"
    return out


def recompute_percentiles_from_requests(entry: dict) -> dict:
    if not isinstance(entry, dict):
        return entry
    ok = [r for r in entry.get("request_results") or [] if r.get("success")]
    if not ok:
        return entry
    ttfts_proc = [r["ttft_proc_ms"] for r in ok if r.get("ttft_proc_ms", 0) > 0]
    ttfts = [r["ttft_ms"] for r in ok if r.get("ttft_ms", 0) > 0]
    tpots = [r["tpot_ms"] for r in ok if r.get("tpot_ms", 0) > 0]
    src = ttfts_proc if ttfts_proc else ttfts
    out = dict(entry)
    out.update(_latency_fields("ttft_proc", src))
    out.update(_latency_fields("tpot", tpots))
    return out


def _point_deploy(scheme: str, entry: dict) -> dict:
    deploy = dict(SCHEME_DEPLOY.get(scheme, {}))
    if scheme == "aflex_tier1":
        if entry.get("config"):
            deploy["config"] = entry["config"]
        if entry.get("tier1_layout"):
            deploy["tier1_layout"] = entry["tier1_layout"]
    return deploy


def pack_results(flat_results: dict, meta: dict) -> dict:
    workloads: dict = {}
    results: dict = {}
    for ds in DATASETS:
        workloads[ds] = {}
        results[ds] = {}
        for q in QPS_LIST:
            wk = qps_key(q)
            workloads[ds][wk] = workload_meta(ds, q)
            results[ds][wk] = {}
            key = wl_key(ds, q)
            for scheme, bucket in flat_results.items():
                entry = bucket.get(key)
                if not entry:
                    continue
                point = recompute_percentiles_from_requests(
                    recompute_energy_per_total_token(dict(entry), key)
                )
                point["deploy"] = _point_deploy(scheme, entry)
                results[ds][wk][scheme] = point

    packed_meta = {
        "benchmark": "macro_e2e",
        "version": 1,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "datasets": list(DATASETS),
        "qps": list(QPS_LIST),
        "schemes": list(flat_results.keys()),
        "scheme_labels": {k: SCHEME_LABELS[k] for k in flat_results if k in SCHEME_LABELS},
        "energy_denominator": "input_plus_output",
        "ttft_slo_ms": TTFT_SLO_MS,
        "tpot_slo_ms": TPOT_SLO_MS,
    }
    packed_meta.update({k: v for k, v in meta.items() if k not in ("results", "schemes", "scheme_labels")})

    return {
        "meta": packed_meta,
        "deploy": SCHEME_DEPLOY,
        "workloads": workloads,
        "results": results,
    }


def flatten_results(payload: dict) -> dict[str, dict]:
    if "results" not in payload:
        return {}
    results = payload["results"]
    if results and isinstance(next(iter(results.values())), dict):
        first = next(iter(results.values()))
        if "qps_2" in first or "qps_4" in first:
            flat: dict[str, dict] = {}
            for ds, by_qps in results.items():
                for wk, by_scheme in by_qps.items():
                    qps = int(wk.split("_", 1)[1])
                    key = wl_key(ds, qps)
                    for scheme, entry in by_scheme.items():
                        flat.setdefault(scheme, {})[key] = entry
            return flat
    return results


def save_all(flat_results: dict, meta: dict) -> Path:
    packed = pack_results(flat_results, meta)
    ALL_DATA_FILE.write_text(json.dumps(packed, indent=2))
    log.info("Saved %s", ALL_DATA_FILE)
    return ALL_DATA_FILE


def save_partial(flat_results: dict, meta: dict, tag: str = "partial") -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = DATA_DIR / f"{RESULT_PREFIX}_{tag}_{ts}.json"
    out.write_text(json.dumps({"meta": meta, "results": flat_results}, indent=2))
    log.info("Saved %s", out)
    return out


def load_resume(prefix: str = RESULT_PREFIX) -> dict:
    if ALL_DATA_FILE.exists():
        data = json.loads(ALL_DATA_FILE.read_text())
        flat = flatten_results(data)
        if flat:
            log.info("Resume from %s", ALL_DATA_FILE.name)
            return flat
    candidates = sorted(DATA_DIR.glob(f"{prefix}_partial_*.json"))
    if candidates:
        data = json.loads(candidates[-1].read_text())
        log.info("Resume from %s", candidates[-1].name)
        return data.get("results", {})
    return {}


def run_window_s(reqs: list[dict], max_run_s: int = MAX_RUN_S) -> int:
    last_arrival = max((r["arrival_time_s"] for r in reqs), default=0)
    return int(min(max(max_run_s, last_arrival + 150), 900))


def _percentiles(values: list[float], ps: tuple[int, ...] = (50, 90, 95, 99)) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=float)
    out = {"avg": round(float(arr.mean()), 1)}
    for p in ps:
        out[f"p{p}"] = round(float(np.percentile(arr, p)), 1)
    return out


def _latency_fields(prefix: str, values: list[float]) -> dict[str, float]:
    stats = _percentiles(values)
    return {
        f"{prefix}_avg_ms": stats.get("avg", 0.0),
        f"{prefix}_p50_ms": stats.get("p50", 0.0),
        f"{prefix}_p90_ms": stats.get("p90", 0.0),
        f"{prefix}_p95_ms": stats.get("p95", 0.0),
        f"{prefix}_p99_ms": stats.get("p99", 0.0),
    }


async def _send_one(
    session: aiohttp.ClientSession,
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
    last_meta: dict[str, Any] = {}
    http_status = None

    try:
        async with session.post(url, json=payload) as resp:
            http_status = resp.status
            if resp.status != 200:
                results.append({
                    "request_index": index,
                    "success": False,
                    "http_status": http_status,
                    "input_len": req["input_len"],
                    "output_len": req["output_len"],
                    "arrival_time_s": req["arrival_time_s"],
                })
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
        results.append({
            "request_index": index,
            "success": False,
            "error": repr(exc),
            "input_len": req["input_len"],
            "output_len": req["output_len"],
            "arrival_time_s": req["arrival_time_s"],
        })
        return

    t_end = time.monotonic()
    ttft_ms = (first_token_time - t0) * 1000 if first_token_time else 0.0
    ttft_proc_ms = 0.0
    if last_meta.get("ttft_pure_processing"):
        ttft_proc_ms = last_meta["ttft_pure_processing"] * 1000
    elif last_meta.get("time_to_first_token_processing"):
        ttft_proc_ms = last_meta["time_to_first_token_processing"] * 1000
    tpot_ms = 0.0
    if token_count > 1 and first_token_time:
        tpot_ms = (t_end - first_token_time) * 1000 / (token_count - 1)

    results.append({
        "request_index": index,
        "success": True,
        "http_status": http_status,
        "input_len": req["input_len"],
        "output_len": req["output_len"],
        "arrival_time_s": req["arrival_time_s"],
        "completion_tokens": token_count,
        "ttft_ms": round(ttft_ms, 3),
        "ttft_proc_ms": round(ttft_proc_ms, 3),
        "tpot_ms": round(tpot_ms, 3),
        "e2e_s": round(t_end - t0, 4),
        "ttft_violation": bool((ttft_proc_ms or ttft_ms) > TTFT_SLO_MS),
        "tpot_violation": bool(tpot_ms > TPOT_SLO_MS),
    })


async def run_workload_with_requests(
    reqs: list[dict],
    url: str,
    get_energy_local,
    get_energy_remote,
    n1_gpus: list[int],
    n2_gpus: list[int],
    max_run_s: int,
) -> dict:
    e1s = get_energy_local(n1_gpus)
    e2s = get_energy_remote(n2_gpus)
    results: list[dict] = []
    timeout = aiohttp.ClientTimeout(total=max_run_s + 60)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        base_time = time.monotonic()
        tasks = [
            asyncio.create_task(_send_one(session, url, req, base_time, i, results))
            for i, req in enumerate(reqs)
        ]
        timed_out = 0
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=max_run_s)
            timed_out = len(pending)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
                log.warning("Timed out after %ds; cancelled %d pending", max_run_s, timed_out)
            if done:
                await asyncio.gather(*done, return_exceptions=True)

    duration_s = time.monotonic() - base_time
    e1e = get_energy_local(n1_gpus)
    e2e = get_energy_remote(n2_gpus)
    energy_n1_j = sum((e1e.get(i, 0) - e1s.get(i, 0)) / 1000.0 for i in n1_gpus)
    energy_n2_j = sum((e2e.get(i, 0) - e2s.get(i, 0)) / 1000.0 for i in n2_gpus)
    total_energy_j = energy_n1_j + energy_n2_j

    ok = [r for r in results if r.get("success")]
    fail = [r for r in results if not r.get("success")]
    missing = max(0, len(reqs) - len(results) - timed_out)

    if not ok:
        return {
            "status": "FAIL",
            "duration_s": round(duration_s, 1),
            "total_requests": len(reqs),
            "successful": 0,
            "failed": len(fail),
            "timed_out": timed_out,
            "missing": missing,
            "request_results": sorted(results, key=lambda r: r.get("request_index", -1)),
        }

    ttfts_proc = [r["ttft_proc_ms"] for r in ok if r.get("ttft_proc_ms", 0) > 0]
    ttfts = [r["ttft_ms"] for r in ok if r.get("ttft_ms", 0) > 0]
    tpots = [r["tpot_ms"] for r in ok if r.get("tpot_ms", 0) > 0]
    src = ttfts_proc if ttfts_proc else ttfts
    total_input_tokens, total_output_tokens, total_tokens_all = _token_totals_from_run(ok, reqs)
    throughput = total_output_tokens / duration_s if duration_s > 0 else 0.0

    n_ttft_viol = sum(1 for v in src if v > TTFT_SLO_MS)
    n_tpot_viol = sum(1 for r in ok if r.get("tpot_ms", 0) > TPOT_SLO_MS)
    slo_rate = (n_ttft_viol + n_tpot_viol + len(fail) + timed_out + missing) / len(reqs) * 100

    status = "PASS"
    if timed_out or missing:
        status = "PARTIAL_TIMEOUT" if ok else "TIMEOUT"
    elif len(ok) < len(reqs):
        status = "FAIL"

    return {
        "status": status,
        "duration_s": round(duration_s, 1),
        "total_requests": len(reqs),
        "successful": len(ok),
        "failed": len(fail),
        "timed_out": timed_out,
        "missing": missing,
        "total_tokens": total_output_tokens,
        "total_input_tokens": total_input_tokens,
        "total_tokens_all": total_tokens_all,
        "throughput_tok_s": round(throughput, 1),
        **_latency_fields("ttft_proc", src),
        **_latency_fields("tpot", tpots),
        "energy_node1_j": round(energy_n1_j, 1),
        "energy_node2_j": round(energy_n2_j, 1),
        "total_energy_j": round(total_energy_j, 1),
        "energy_per_token_mj": (
            round(total_energy_j * 1000 / total_tokens_all, 2) if total_tokens_all else 0
        ),
        "energy_denominator": "input_plus_output",
        "slo_violation_rate": round(slo_rate, 1),
        "ttft_violations": n_ttft_viol,
        "tpot_violations": n_tpot_viol,
        "request_results": sorted(results, key=lambda r: r.get("request_index", -1)),
    }
