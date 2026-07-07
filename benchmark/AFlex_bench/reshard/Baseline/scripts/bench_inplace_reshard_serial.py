#!/usr/bin/env python3
"""Serial (one-request-at-a-time) in-place TP reshard timeline.

This locks in the validated result: it drives a REAL sglang server with REAL
generate requests issued strictly serially (each request waits for the previous
to finish), triggers a REAL in-place reshard partway through, and records a clean
per-request timeline. Serial issue avoids the continuous-batching scheduling
divergence that concurrent load currently exposes after in-place reshard, so this
measures exactly what is stable today:

  * TP1 per-request latency (pre-reshard)
  * reshard trigger -> ready pause window
  * TP2 per-request latency (post-reshard)

Usage (inside the container, server already up at --base):
  python3 bench_inplace_reshard_serial.py \
      --base http://127.0.0.1:31700 \
      --n-pre 12 --n-post 12 --new-tp 2 --max-new 32 \
      --out results/inplace_reshard_serial_tp2.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

PROMPTS = [
    "The capital of France is",
    "Write a Python function to reverse a string:",
    "Explain what a binary search tree is:",
    "The three laws of thermodynamics are",
    "def quicksort(arr):",
    "In machine learning, gradient descent is",
    "Summarize the plot of Romeo and Juliet:",
    "The chemical formula for water is",
]


def gen(base, prompt, max_new, timeout):
    t = time.time()
    ttft = None
    ok = False
    text = ""
    err = None
    try:
        with requests.post(
            base + "/generate",
            json={"text": prompt, "sampling_params": {"max_new_tokens": max_new, "temperature": 0}, "stream": True},
            stream=True,
            timeout=timeout,
        ) as r:
            for line in r.iter_lines():
                if not line:
                    continue
                if ttft is None:
                    ttft = time.time() - t
            ok = r.status_code == 200
        # non-stream fetch of final text for sanity
    except Exception as e:
        err = repr(e)[:80]
    e2e = time.time() - t
    return {"ttft_s": round(ttft, 3) if ttft else None, "e2e_s": round(e2e, 3), "ok": ok, "err": err}


def gen_text(base, prompt, max_new, timeout):
    try:
        r = requests.post(
            base + "/generate",
            json={"text": prompt, "sampling_params": {"max_new_tokens": max_new, "temperature": 0}},
            timeout=timeout,
        )
        return r.status_code, r.json().get("text", "")
    except Exception as e:
        return "ERR", repr(e)[:60]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:31700")
    ap.add_argument("--n-pre", type=int, default=12)
    ap.add_argument("--n-post", type=int, default=12)
    ap.add_argument("--new-tp", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t_start = time.time()
    timeline = []

    # Phase A: pre-reshard TP1
    for k in range(args.n_pre):
        p = PROMPTS[k % len(PROMPTS)]
        r = gen(args.base, p, args.max_new, args.timeout)
        r.update({"phase": "pre_tp1", "idx": k, "t_rel": round(time.time() - t_start, 2)})
        timeline.append(r)
        print("pre  #%02d ok=%s ttft=%s e2e=%s" % (k, r["ok"], r["ttft_s"], r["e2e_s"]))

    # Phase B: reshard
    print("--- triggering reshard -> TP%d ---" % args.new_tp)
    t_trig = time.time()
    try:
        rr = requests.post(args.base + "/reshard_tp", json={"new_tp_size": args.new_tp}, timeout=15)
        accepted = rr.status_code
    except Exception as e:
        accepted = repr(e)[:60]
    # Probe until first post-reshard request succeeds -> ready pause window
    ready_after = None
    probe_sc, probe_txt = None, None
    for _ in range(120):
        sc, txt = gen_text(args.base, "The capital of France is", 8, 15)
        if sc == 200 and isinstance(txt, str) and txt.strip():
            ready_after = time.time() - t_trig
            probe_sc, probe_txt = sc, txt
            break
        time.sleep(1)
    reshard_info = {
        "accepted": accepted,
        "ready_pause_s": round(ready_after, 2) if ready_after else None,
        "first_post_text": probe_txt,
        "t_rel": round(t_trig - t_start, 2),
    }
    print("reshard accepted=%s ready_pause=%ss first=%r" % (accepted, reshard_info["ready_pause_s"], (probe_txt or "")[:50]))

    # Phase C: post-reshard TP2
    for k in range(args.n_post):
        p = PROMPTS[k % len(PROMPTS)]
        r = gen(args.base, p, args.max_new, args.timeout)
        r.update({"phase": "post_tp2", "idx": k, "t_rel": round(time.time() - t_start, 2)})
        timeline.append(r)
        print("post #%02d ok=%s ttft=%s e2e=%s" % (k, r["ok"], r["ttft_s"], r["e2e_s"]))

    def med(xs):
        xs = sorted([x for x in xs if x is not None])
        return round(xs[len(xs) // 2], 3) if xs else None

    pre = [x for x in timeline if x["phase"] == "pre_tp1" and x["ok"]]
    post = [x for x in timeline if x["phase"] == "post_tp2" and x["ok"]]
    summary = {
        "new_tp": args.new_tp,
        "max_new": args.max_new,
        "n_pre_ok": len(pre),
        "n_post_ok": len(post),
        "reshard": reshard_info,
        "pre_tp1_ttft_med": med([x["ttft_s"] for x in pre]),
        "post_tp2_ttft_med": med([x["ttft_s"] for x in post]),
        "pre_tp1_e2e_med": med([x["e2e_s"] for x in pre]),
        "post_tp2_e2e_med": med([x["e2e_s"] for x in post]),
    }
    out = {"summary": summary, "timeline": timeline}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
