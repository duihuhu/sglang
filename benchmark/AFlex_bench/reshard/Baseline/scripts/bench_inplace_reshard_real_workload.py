#!/usr/bin/env python3
"""Real-workload timeline benchmark for in-place TP reshard (TP1 -> TP2 -> ...).

Unlike the synthetic transfer-timeline script, this drives a REAL sglang server
with REAL generate requests replayed from a macro workload trace, and triggers
REAL in-place reshard partway through. For every request it records arrival time,
TTFT, end-to-end latency, output length, and which TP size was active when it was
issued. The output timeline lets us quantify the reshard pause window and the
throughput/latency change across TP steps.

Usage (inside the container, server already reachable at --base):
  python3 bench_inplace_reshard_real_workload.py \
      --workload /path/macro_code_qps2.jsonl \
      --base http://127.0.0.1:31700 \
      --reshard-plan '[{"at_s":30,"new_tp":2},{"at_s":60,"new_tp":4}]' \
      --out results/inplace_reshard_real_qps2.json

Legacy single-step mode still works:
  --reshard-at-s 30 --new-tp 2
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inplace_reshard_server_ctl import stop_inplace_reshard_server


def load_workload(path: str):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r["arrival_time_s"])
    return rows


def parse_reshard_plan(args) -> list[dict]:
    if args.reshard_plan:
        plan = json.loads(args.reshard_plan)
        if not isinstance(plan, list) or not plan:
            raise ValueError("--reshard-plan must be a non-empty JSON array")
        return sorted(
            [
                {
                    "at_s": float(s["at_s"]),
                    "new_tp": int(s["new_tp"]),
                    **(
                        {"pre_drain_sec": float(s["pre_drain_sec"])}
                        if s.get("pre_drain_sec") is not None
                        else {}
                    ),
                }
                for s in plan
            ],
            key=lambda s: s["at_s"],
        )
    return [{"at_s": float(args.reshard_at_s), "new_tp": int(args.new_tp)}]


class State:
    """Shared reshard state so each request can tag the TP size it saw."""

    def __init__(self, initial_tp: int = 1):
        self.current_tp = initial_tp
        self.reshard_events: list[dict] = []
        self.lock = threading.Lock()


def decode_tpot_s(req: dict) -> float | None:
    """TPOT ≈ decode time / (#tokens after the first)."""
    ttft = req.get("ttft_s")
    e2e = req.get("e2e_s")
    out_len = int(req.get("output_len") or 0)
    if ttft is None or e2e is None:
        return None
    decode_s = max(float(e2e) - float(ttft), 0.0)
    denom = max(out_len - 1, 1)
    return decode_s / denom


def slo_pass(
    req: dict,
    ttft_limit_s: float = 2.0,
    tpot_limit_s: float = 0.1,
) -> bool:
    """SLO: TTFT <= 2000ms and TPOT <= 100ms, plus HTTP success."""
    if not req.get("ok"):
        return False
    ttft = req.get("ttft_s")
    if ttft is None:
        return False
    tpot = decode_tpot_s(req)
    if tpot is None:
        return False
    return float(ttft) <= ttft_limit_s and tpot <= tpot_limit_s


def compute_downtime_windows(
    timeline: list[dict],
    ttft_limit_s: float = 2.0,
    tpot_limit_s: float = 0.1,
) -> list[dict]:
    """Contiguous runs of SLO-failing requests = effective service outage."""
    if not timeline:
        return []
    ordered = sorted(timeline, key=lambda r: r["issue_s"])
    windows: list[dict] = []
    cur_start = None
    cur_end = None
    cur_fail = 0

    def flush():
        nonlocal cur_start, cur_end, cur_fail
        if cur_start is not None:
            windows.append(
                {
                    "start_s": round(cur_start, 3),
                    "end_s": round(cur_end, 3),
                    "duration_s": round(cur_end - cur_start, 3),
                    "n_slo_fail": cur_fail,
                }
            )
        cur_start = cur_end = None
        cur_fail = 0

    for req in ordered:
        issue = float(req["issue_s"])
        e2e = float(req.get("e2e_s") or 0.0)
        fail = not slo_pass(req, ttft_limit_s, tpot_limit_s)
        if fail:
            if cur_start is None:
                cur_start = issue
            cur_end = issue + e2e
            cur_fail += 1
        else:
            flush()
    flush()
    return windows


def merge_time_windows(windows: list[dict]) -> list[dict]:
    if not windows:
        return []
    spans = sorted((float(w["start_s"]), float(w["end_s"])) for w in windows)
    merged: list[tuple[float, float]] = []
    cur_s, cur_e = spans[0]
    for s, e in spans[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return [
        {
            "start_s": round(s, 3),
            "end_s": round(e, 3),
            "duration_s": round(e - s, 3),
        }
        for s, e in merged
    ]


def _reshard_slo_impact(events: list[dict], merged: list[dict]) -> list[dict]:
    """Per-step reshard pause vs overlapping merged SLO-downtime."""
    out = []
    for ev in events:
        rs = float(ev["trigger_s"])
        re = float(ev["done_s"]) if ev.get("done_s") is not None else rs
        overlap = 0.0
        for w in merged:
            ws, we = float(w["start_s"]), float(w["end_s"])
            ov = max(0.0, min(re, we) - max(rs, ws))
            overlap = max(overlap, ov)
        out.append(
            {
                "new_tp": ev["new_tp"],
                "trigger_s": ev["trigger_s"],
                "pause_s": ev.get("pause_s"),
                "slo_overlap_s": round(overlap, 3),
            }
        )
    return out


def annotate_slo(results: list[dict], ttft_limit_s: float, tpot_limit_s: float) -> None:
    for req in results:
        tpot = decode_tpot_s(req)
        req["tpot_s"] = round(tpot, 4) if tpot is not None else None
        req["slo_pass"] = slo_pass(req, ttft_limit_s, tpot_limit_s)


def make_prompt(input_len: int) -> str:
    if input_len <= 1:
        return "Hi"
    return " ".join(["code"] * max(1, input_len))


def issue_request(base, row, t0, state, results, timeout):
    target = t0 + row["arrival_time_s"]
    now = time.time()
    if target > now:
        time.sleep(target - now)

    with state.lock:
        tp_at_issue = state.current_tp
    issue_s = time.time() - t0
    prompt = make_prompt(row["input_len"])
    out_len = max(1, int(row["output_len"]))
    payload = {
        "text": prompt,
        "sampling_params": {"max_new_tokens": out_len, "temperature": 0},
        "stream": True,
    }
    ttft = None
    ok = False
    err = None
    start = time.time()
    try:
        with requests.post(base + "/generate", json=payload, stream=True, timeout=timeout) as r:
            for line in r.iter_lines():
                if not line:
                    continue
                if ttft is None:
                    ttft = time.time() - start
            ok = r.status_code == 200
    except Exception as e:
        err = repr(e)[:80]
    e2e = time.time() - start
    results.append(
        {
            "arrival_s": row["arrival_time_s"],
            "issue_s": round(issue_s, 3),
            "input_len": row["input_len"],
            "output_len": out_len,
            "tp_at_issue": tp_at_issue,
            "ttft_s": round(ttft, 3) if ttft is not None else None,
            "e2e_s": round(e2e, 3),
            "ok": ok,
            "err": err,
        }
    )


def poll_reshard_status(
    base: str,
    new_tp: int,
    t0: float,
    timeout_s: float,
    min_generation: int = 0,
):
    """Poll /inplace_reshard_status until rank0 reports phase=done at target TP."""
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        try:
            r = requests.get(base + "/inplace_reshard_status", timeout=5)
            if r.status_code == 200:
                last = r.json()
                phase = last.get("phase")
                active_tp = last.get("active_tp")
                target_tp = last.get("target_tp")
                generation = int(last.get("generation") or 0)
                if phase == "failed":
                    return None, last
                if (
                    phase == "done"
                    and active_tp == new_tp
                    and (target_tp is None or target_tp == new_tp)
                    and generation >= min_generation
                ):
                    done_at = last.get("done_at")
                    return (
                        (done_at - t0) if done_at is not None else time.time() - t0,
                        last,
                    )
        except Exception:
            pass
        time.sleep(0.25)
    return None, last


def reshard_once(base, state, t0, step: dict, status_timeout_s, step_idx: int):
    at_s = step["at_s"]
    new_tp = step["new_tp"]
    time.sleep(max(0.0, at_s - (time.time() - t0)))
    trigger_s = time.time() - t0
    min_generation = step_idx + 1
    try:
        payload = {"new_tp_size": new_tp}
        pre_drain = step.get("pre_drain_sec")
        if pre_drain is not None:
            payload["pre_drain_sec"] = float(pre_drain)
        r = requests.post(base + "/reshard_tp", json=payload, timeout=15)
        accepted = r.status_code
    except Exception as e:
        accepted = repr(e)[:60]

    done, status = poll_reshard_status(
        base, new_tp, t0, status_timeout_s, min_generation=min_generation
    )
    with state.lock:
        if done is not None:
            state.current_tp = new_tp
        event = {
            "step": step_idx,
            "at_s": at_s,
            "new_tp": new_tp,
            "trigger_s": round(trigger_s, 3),
            "done_s": round(done, 3) if done is not None else None,
            "pause_s": round(done - trigger_s, 2) if done is not None else None,
            "accepted": accepted,
            "phase": (status or {}).get("phase"),
            "generation": (status or {}).get("generation"),
            "server_elapsed_s": (status or {}).get("elapsed_s"),
            "timings": (status or {}).get("timings"),
        }
        state.reshard_events.append(event)
    print(
        "[reshard step %d] trigger@%.1fs target=TP%d accepted=%s done@%s pause=%s"
        % (
            step_idx,
            trigger_s,
            new_tp,
            accepted,
            done,
            event["pause_s"],
        )
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True)
    ap.add_argument("--base", default="http://127.0.0.1:31700")
    ap.add_argument("--reshard-at-s", type=float, default=30.0)
    ap.add_argument("--new-tp", type=int, default=2)
    ap.add_argument(
        "--reshard-plan",
        default=None,
        help='JSON array, e.g. \'[{"at_s":30,"new_tp":2},{"at_s":60,"new_tp":4}]\'',
    )
    ap.add_argument("--max-workers", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--reshard-status-timeout", type=float, default=300.0)
    ap.add_argument(
        "--reshard-sequential",
        action="store_true",
        help="Run reshard steps one after another (avoids overlapping triggers)",
    )
    ap.add_argument("--slo-ttft-ms", type=float, default=2000.0)
    ap.add_argument("--slo-tpot-ms", type=float, default=100.0)
    ap.add_argument(
        "--keep-server",
        action="store_true",
        help="Do not stop sglang after the benchmark (default: stop)",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    try:
        _run_benchmark(args)
    finally:
        if not args.keep_server:
            port = urlparse(args.base).port or 31700
            stop_inplace_reshard_server(port)


def _run_benchmark(args) -> None:
    plan = parse_reshard_plan(args)
    rows = load_workload(args.workload)
    print("loaded %d requests, span=%.1fs" % (len(rows), rows[-1]["arrival_time_s"]))
    print("reshard plan:", plan)

    state = State()
    results = []
    t0 = time.time()

    threads = []
    if args.reshard_sequential:
        rt = threading.Thread(
            target=lambda: [
                reshard_once(args.base, state, t0, step, args.reshard_status_timeout, i)
                for i, step in enumerate(plan)
            ],
            daemon=True,
        )
        rt.start()
        threads.append(rt)
    else:
        for i, step in enumerate(plan):
            rt = threading.Thread(
                target=reshard_once,
                args=(args.base, state, t0, step, args.reshard_status_timeout, i),
                daemon=True,
            )
            rt.start()
            threads.append(rt)

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [
            ex.submit(issue_request, args.base, row, t0, state, results, args.timeout)
            for row in rows
        ]
        for f in futs:
            f.result()
    for rt in threads:
        rt.join(timeout=max(args.reshard_status_timeout, rows[-1]["arrival_time_s"] + 120))

    results.sort(key=lambda x: x["issue_s"])
    ttft_lim = args.slo_ttft_ms / 1000.0
    tpot_lim = args.slo_tpot_ms / 1000.0
    annotate_slo(results, ttft_lim, tpot_lim)

    n_ok = sum(1 for r in results if r["ok"])
    n_slo = sum(1 for r in results if r.get("slo_pass"))
    downtime = compute_downtime_windows(results, ttft_lim, tpot_lim)
    downtime_merged = merge_time_windows(downtime)

    def med(xs):
        xs = sorted(xs)
        return round(xs[len(xs) // 2], 3) if xs else None

    tp_stats = {}
    for r in results:
        tp = int(r["tp_at_issue"])
        tp_stats.setdefault(tp, []).append(r)

    per_tp_summary = {}
    for tp, reqs in sorted(tp_stats.items()):
        slo_reqs = [r for r in reqs if r.get("slo_pass")]
        per_tp_summary[f"tp{tp}_n"] = len(reqs)
        per_tp_summary[f"tp{tp}_slo_n"] = len(slo_reqs)
        per_tp_summary[f"tp{tp}_ttft_med"] = med(
            [r["ttft_s"] for r in reqs if r.get("ttft_s") is not None]
        )
        per_tp_summary[f"tp{tp}_tpot_med"] = med(
            [r["tpot_s"] for r in reqs if r.get("tpot_s") is not None]
        )

    summary = {
        "workload": args.workload,
        "reshard_plan": plan,
        "slo": {"ttft_ms": args.slo_ttft_ms, "tpot_ms": args.slo_tpot_ms},
        "n_total": len(results),
        "n_ok": n_ok,
        "n_slo_pass": n_slo,
        "n_slo_fail": len(results) - n_slo,
        "slo_downtime_windows": downtime,
        "slo_downtime_windows_merged": downtime_merged,
        "slo_downtime_total_s": round(
            sum(w["duration_s"] for w in downtime_merged), 3
        ),
        "reshard_events": state.reshard_events,
        "reshard_slo_impact": _reshard_slo_impact(state.reshard_events, downtime_merged),
        **per_tp_summary,
    }
    out = {"summary": summary, "timeline": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
