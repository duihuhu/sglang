#!/usr/bin/env python3
"""Shared helpers for micro 4-dataset benchmark (per-request recording)."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np

log = logging.getLogger("micro_bench")

MICRO_ROOT = Path(__file__).resolve().parent
MULTI_NODE = MICRO_ROOT.parents[1]
WORKLOAD_DIR = MICRO_ROOT / "data" / "workloads"
DATA_DIR = MICRO_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RESULT_PREFIX = "micro_e2e"


def dataset_data_file(dataset: str) -> Path:
    return DATA_DIR / f"micro_e2e_{dataset}.json"

DATASETS = ("qa_lpld", "chatbot_lphd", "rag_hpld", "summary_hphd")
QPS_LIST = (2, 4, 8, 16)
TTFT_SLO_MS = 5000.0
TPOT_SLO_MS = 300.0
MAX_RUN_S = 400

DATASET_META = {
    "qa_lpld": {"input_len": 128, "output_len": 64, "label": "QA LPLD"},
    "chatbot_lphd": {"input_len": 128, "output_len": 1024, "label": "Chatbot LPHD"},
    "rag_hpld": {"input_len": 4096, "output_len": 64, "label": "RAG HPLD"},
    "summary_hphd": {"input_len": 4096, "output_len": 1024, "label": "Summary HPHD"},
}

SCHEME_DEPLOY: dict[str, dict] = {
    "native_tp1_baseline": {
        "label": "SGLang", "topology": "native_tp1", "tier": False,
        "gpus": 16, "nodes": 2, "tp": 1, "freq_policy": "locked_1410mhz",
        "deploy_policy": "restart_per_point",
    },
    "native_tp1_tier": {
        "label": "DynamoLLM", "topology": "native_tp1", "tier": True,
        "gpus": 16, "nodes": 2, "tp": 1, "freq_policy": "unified_dvfs",
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
    "aflex": {
        "label": "AFlex", "topology": "tier1_afd", "tier": True,
        "deploy_policy": "solver_per_qps",
        "layout_source": "tier1_selected_solutions",
    },
}

SCHEME_LABELS = {k: v["label"] for k, v in SCHEME_DEPLOY.items()}


def wl_key(dataset: str, qps: int) -> str:
    return f"{dataset}_qps{qps}"


def qps_key(qps: int) -> str:
    return f"qps_{qps}"


def workload_file(dataset: str, qps: int) -> Path | None:
    path = WORKLOAD_DIR / f"micro_{dataset}_qps{qps}.jsonl"
    return path if path.exists() else None


def load_workload(dataset: str, qps: int) -> list[dict]:
    path = workload_file(dataset, qps)
    if path is None:
        raise FileNotFoundError(f"missing workload: micro_{dataset}_qps{qps}.jsonl")
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
        "workload_file": f"micro_{dataset}_qps{qps}.jsonl",
        "n_requests": len(reqs),
        "total_input_tokens": inp,
        "total_output_tokens": out,
        "total_tokens_all": inp + out,
        **DATASET_META.get(dataset, {}),
    }


def workload_token_totals(dataset: str, qps: int) -> tuple[int, int, int]:
    m = workload_meta(dataset, qps)
    return m["total_input_tokens"], m["total_output_tokens"], m["total_tokens_all"]


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
    for r in ok:
        if "input_len" not in r and "request_index" in r:
            pass
    return out


def _point_deploy(scheme: str, entry: dict) -> dict:
    deploy = dict(SCHEME_DEPLOY.get(scheme, {}))
    if scheme == "aflex":
        if entry.get("config"):
            deploy["config"] = entry["config"]
        if entry.get("provenance"):
            deploy["provenance"] = entry["provenance"]
        if entry.get("router_urls"):
            deploy["router_urls"] = entry["router_urls"]
    return deploy


def _packed_meta(flat_results: dict, meta: dict, dataset: str) -> dict:
    packed_meta = {
        "benchmark": "micro_4datasets_e2e",
        "version": 1,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": dataset,
        "datasets": [dataset],
        "qps": list(QPS_LIST),
        "schemes": list(flat_results.keys()),
        "scheme_labels": {k: SCHEME_LABELS[k] for k in flat_results if k in SCHEME_LABELS},
        "energy_denominator": "input_plus_output",
        "ttft_slo_ms": TTFT_SLO_MS,
        "tpot_slo_ms": TPOT_SLO_MS,
    }
    packed_meta.update({
        k: v for k, v in meta.items()
        if k not in ("results", "schemes", "scheme_labels", "dataset", "datasets")
    })
    return packed_meta


def pack_dataset_payload(flat_results: dict, dataset: str, meta: dict) -> dict:
    workloads: dict = {dataset: {}}
    results: dict = {dataset: {}}
    for q in QPS_LIST:
        wk = qps_key(q)
        workloads[dataset][wk] = workload_meta(dataset, q)
        results[dataset][wk] = {}
        key = wl_key(dataset, q)
        for scheme, bucket in flat_results.items():
            entry = bucket.get(key)
            if not entry:
                continue
            point = recompute_percentiles_from_requests(
                recompute_energy_per_total_token(dict(entry), key)
            )
            point["deploy"] = _point_deploy(scheme, entry)
            results[dataset][wk][scheme] = point

    return {
        "meta": _packed_meta(flat_results, meta, dataset),
        "deploy": SCHEME_DEPLOY,
        "workloads": workloads,
        "results": results,
    }


def pack_results(flat_results: dict, meta: dict) -> dict:
    workloads: dict = {}
    results: dict = {}
    for ds in DATASETS:
        payload = pack_dataset_payload(flat_results, ds, meta)
        workloads[ds] = payload["workloads"][ds]
        results[ds] = payload["results"][ds]
    return {
        "meta": _packed_meta(flat_results, meta, "all"),
        "deploy": SCHEME_DEPLOY,
        "workloads": workloads,
        "results": results,
    }


def flatten_results(payload: dict) -> dict[str, dict]:
    results = payload.get("results", {})
    if not results:
        return {}
    first = next(iter(results.values()))
    if isinstance(first, dict) and any(k.startswith("qps_") for k in first):
        flat: dict[str, dict] = {}
        for ds, by_qps in results.items():
            for wk, by_scheme in by_qps.items():
                qps = int(wk.split("_", 1)[1])
                key = wl_key(ds, qps)
                for scheme, entry in by_scheme.items():
                    flat.setdefault(scheme, {})[key] = entry
        return flat
    return results


def _load_flat_from_dataset_files() -> dict[str, dict]:
    flat: dict[str, dict] = {}
    for ds in DATASETS:
        path = dataset_data_file(ds)
        if not path.exists():
            continue
        part = flatten_results(json.loads(path.read_text()))
        for scheme, bucket in part.items():
            flat.setdefault(scheme, {}).update(bucket)
    return flat


def save_all(
    flat_results: dict,
    meta: dict,
    datasets: tuple[str, ...] | list[str] | None = None,
) -> list[Path]:
    existing = _load_flat_from_dataset_files()
    for scheme, bucket in flat_results.items():
        existing.setdefault(scheme, {}).update(bucket)

    targets = tuple(datasets or DATASETS)
    paths: list[Path] = []
    for ds in targets:
        path = dataset_data_file(ds)
        path.write_text(json.dumps(pack_dataset_payload(existing, ds, meta), indent=2))
        log.info("Saved %s", path)
        paths.append(path)
    return paths


def save_partial(
    flat_results: dict,
    meta: dict,
    tag: str = "partial",
    datasets: tuple[str, ...] | list[str] | None = None,
) -> list[Path]:
    return save_all(flat_results, meta, datasets=datasets)


def load_resume(prefix: str = RESULT_PREFIX) -> dict:
    flat = _load_flat_from_dataset_files()
    if flat:
        return flat

    legacy = DATA_DIR / "micro_e2e_all.json"
    if legacy.exists():
        flat = flatten_results(json.loads(legacy.read_text()))
        if flat:
            log.info("Resume from legacy %s", legacy.name)
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


def run_window_meta(reqs: list[dict], run_s: int) -> dict:
    last_arrival = max((r["arrival_time_s"] for r in reqs), default=0)
    max_output_len = max((r["output_len"] for r in reqs), default=0)
    return {
        "run_s": run_s,
        "last_arrival_s": round(last_arrival, 4),
        "max_output_len": max_output_len,
        "queue_margin_s": 150.0,
        "deadline_s": round(last_arrival + 150 + max_output_len * 0.3, 4),
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

    payload = {
        "text": "x" * req["input_len"],
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
                    "request_index": index, "success": False, "http_status": http_status,
                    "input_len": req["input_len"], "output_len": req["output_len"],
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
            "request_index": index, "success": False, "error": repr(exc),
            "input_len": req["input_len"], "output_len": req["output_len"],
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
        "request_index": index, "success": True, "http_status": http_status,
        "input_len": req["input_len"], "output_len": req["output_len"],
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
        "run_window": run_window_meta(reqs, max_run_s),
        "request_results": sorted(results, key=lambda r: r.get("request_index", -1)),
    }


async def run_workload_rr_with_requests(
    reqs: list[dict],
    urls: list[str],
    get_energy_local,
    get_energy_remote,
    n1_gpus: list[int],
    n2_gpus: list[int],
    max_run_s: int,
    prefill_weights: list[int] | None = None,
) -> dict:
    """Round-robin (or weighted RR) workload across multiple router/base URLs."""
    from collections import Counter

    if not urls:
        return {"status": "FAIL", "total_requests": len(reqs), "successful": 0}

    generate_urls = [u if u.endswith("/generate") else u + "/generate" for u in urls]
    if prefill_weights and len(prefill_weights) == len(generate_urls) and len(set(prefill_weights)) > 1:
        pool: list[str] = []
        for url, weight in zip(generate_urls, prefill_weights):
            pool.extend([url] * weight)
        assigned = [pool[i % len(pool)] for i in range(len(reqs))]
    else:
        assigned = [generate_urls[i % len(generate_urls)] for i in range(len(reqs))]

    e1s = get_energy_local(n1_gpus)
    e2s = get_energy_remote(n2_gpus)
    results: list[dict] = []
    timeout = aiohttp.ClientTimeout(total=max_run_s + 60)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        base_time = time.monotonic()
        tasks = [
            asyncio.create_task(_send_one(session, assigned[i], req, base_time, i, results))
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
            "run_window": run_window_meta(reqs, max_run_s),
            "router_urls": urls,
            "assigned_route_distribution": dict(Counter(assigned)),
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

    for r in ok:
        idx = r.get("request_index")
        if idx is not None and 0 <= idx < len(assigned):
            r["target_url"] = assigned[idx]

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
        "run_window": run_window_meta(reqs, max_run_s),
        "router_urls": urls,
        "route_distribution": dict(Counter(r.get("target_url", "unknown") for r in ok)),
        "assigned_route_distribution": dict(Counter(assigned)),
        "request_results": sorted(results, key=lambda r: r.get("request_index", -1)),
    }
