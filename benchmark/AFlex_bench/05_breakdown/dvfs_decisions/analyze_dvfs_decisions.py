#!/usr/bin/env python3
"""Analyze DVFS decision logs: frequency choices + predictor accuracy.

For each tier1_freq run we have 4 JSONL decision logs (PA/PF/DA/DF). The most
informative accuracy signal is on the DECODE side: at each re-evaluation we
logged the observed iteration time (obs_iter_us) and the predicted iteration
time at the *currently running* frequency (pred_iter_cur_us). Their relative
error tells us how well the energy/latency predictor models reality.

We drop the first WARMUP_SKIP decode samples per file to avoid the startup
transient (queue backlog inflates obs_iter_us right after warmup).
"""

import glob
import json
import re
from collections import defaultdict

import numpy as np

WARMUP_SKIP = 5
LOG_ROOT = "logs"


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def parse_meta(path):
    name = path.split("/")[-1]
    m = re.search(r"(il\d+_ol\d+)_qps([0-9p]+)_tier1_freq_dvfs_decisions_(\w+)_(\w+)_gpu(\d+)",
                  name)
    if not m:
        return None
    return {
        "group": m.group(1), "qps": m.group(2).replace("p", "."),
        "persp": m.group(3), "disagg": m.group(4), "gpu": int(m.group(5)),
    }


def main():
    files = sorted(glob.glob(f"{LOG_ROOT}/**/*dvfs_decisions*.jsonl", recursive=True))
    files = [f for f in files if "_archive" not in f]

    # ── Decode predictor accuracy (obs vs pred iter latency) ──
    print("=" * 104)
    print("  DECODE PREDICTOR ACCURACY: predicted iteration latency vs observed")
    print("  (pred_iter_err_pct = (pred - obs)/obs*100; negative = predictor UNDER-estimates)")
    print("=" * 104)
    print(f"  {'group':>13} {'qps':>4} {'persp':>5} | {'n':>4} {'switch':>6} "
          f"{'obs_iter_ms':>11} {'pred_iter_ms':>12} {'err%_med':>9} {'err%_p90':>9} {'|err|%_med':>10}")
    print("  " + "-" * 100)

    accuracy_rows = []
    for f in files:
        meta = parse_meta(f)
        if not meta or meta["disagg"] != "decode":
            continue
        rows = load_jsonl(f)
        rows = [r for r in rows if r.get("phase") == "decode"][WARMUP_SKIP:]
        if not rows:
            continue
        errs = [r["pred_iter_err_pct"] for r in rows
                if r.get("pred_iter_err_pct") is not None]
        obs = [r["obs_iter_us"] / 1000.0 for r in rows if r.get("obs_iter_us", 0) > 0]
        pred = [r["pred_iter_cur_us"] / 1000.0 for r in rows if r.get("pred_iter_cur_us", 0) > 0]
        n_switch = sum(1 for r in rows if r.get("switched"))
        if not errs:
            continue
        abs_errs = [abs(e) for e in errs]
        row = {
            **meta, "n": len(rows), "n_switch": n_switch,
            "obs_med": float(np.median(obs)) if obs else 0,
            "pred_med": float(np.median(pred)) if pred else 0,
            "err_med": float(np.median(errs)),
            "err_p90": float(np.percentile(errs, 90)),
            "abserr_med": float(np.median(abs_errs)),
        }
        accuracy_rows.append(row)
        print(f"  {meta['group']:>13} {meta['qps']:>4} {meta['persp']:>5} | "
              f"{row['n']:>4} {n_switch:>6} {row['obs_med']:>11.1f} {row['pred_med']:>12.1f} "
              f"{row['err_med']:>9.1f} {row['err_p90']:>9.1f} {row['abserr_med']:>10.1f}")

    if accuracy_rows:
        all_abs = [r["abserr_med"] for r in accuracy_rows]
        all_err = [r["err_med"] for r in accuracy_rows]
        print("  " + "-" * 100)
        print(f"  OVERALL: median |err| across runs = {np.median(all_abs):.1f}%, "
              f"median signed err = {np.median(all_err):.1f}% "
              f"(negative ⇒ predictor systematically under-estimates decode iter time)")

    # ── Frequency-choice distribution (decode attn / ffn) ──
    print()
    print("=" * 104)
    print("  SELECTED FREQUENCY DISTRIBUTION (decode side, post-warmup)")
    print("=" * 104)
    print(f"  {'group':>13} {'qps':>4} {'persp':>5} | sel_freq histogram (MHz: count)")
    print("  " + "-" * 100)
    for f in files:
        meta = parse_meta(f)
        if not meta or meta["disagg"] != "decode":
            continue
        rows = load_jsonl(f)
        rows = [r for r in rows if r.get("phase") == "decode"][WARMUP_SKIP:]
        if not rows:
            continue
        key = "sel_f_a" if meta["persp"] == "attn" else "sel_f_f"
        hist = defaultdict(int)
        for r in rows:
            if key in r:
                hist[r[key]] += 1
        hist_s = " ".join(f"{k}:{v}" for k, v in sorted(hist.items()))
        print(f"  {meta['group']:>13} {meta['qps']:>4} {meta['persp']:>5} | {hist_s}")


if __name__ == "__main__":
    main()
