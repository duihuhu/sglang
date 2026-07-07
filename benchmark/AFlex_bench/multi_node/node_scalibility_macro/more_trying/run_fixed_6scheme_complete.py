#!/usr/bin/env python3
"""Complete 6-scheme benchmark with fixed architecture + min-energy DVFS.

Config:
  SGLang      = native_tp1_baseline
  DynamoLLM   = native_tp1_tier
  DistServe   = pd_xnode_tp1_baseline
  BiScale     = pd_xnode_tp1_tier
  MegaScale   = pdaf_tp1_baseline  (TP1 x4 instances)
  AFlex       = pdaf_tp1_tier

QPS: 2,4,6,8,12,16 | conv dataset | node3+node4

Reuses prior JSON results (skip already-tested QPS). Default merge sources:
  - fixed_6scheme_complete_partial_*.json (native 6 QPS)
  - quick_dvfs_ab_*_tp1_*.json (pd/native spot checks)
  - more_trying_sweep_*.json (pdaf; AFD DVFS unchanged)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fixed_6scheme")

os.environ.setdefault("MN_NODE1_IP", "10.252.129.34")
os.environ.setdefault("MN_NODE2_IP", "10.252.129.33")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_more_trying_sweep as MTS
import run_macro_benchmark as RMB

# AFlex: M=1 + V1 compositional DVFS (same as conv_6scheme)
_orig_afd_common = RMB._afd_common


def _patched_afd_common(tp, ib_dev, gpu_step, tier, ngpu=8):
    result = _orig_afd_common(tp, ib_dev, gpu_step, tier, ngpu)
    result = result.replace("--afd-micro-batch 2", "--afd-micro-batch 1")
    result = result.replace("--afd-dynamic-micro-batch", "")
    if tier:
        result = result.replace(RMB.ENERGY_MODEL_DIR_V2, RMB.ENERGY_MODEL_DIR_V1)
        result += " --afd-dvfs-decode-compositional"
    return result


RMB._afd_common = _patched_afd_common
RMB.TTFT_SLO_MS = 5000.0
RMB.TPOT_SLO_MS = 300.0

QPS_LIST = [2, 4, 6, 8, 12, 16]
DATASET = "conv"
NGPU = 16
MAX_RUN_S = 400
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SCHEMES = [
    ("native_tp1_baseline", lambda: MTS.start_native_tp_variant(1, False)),
    ("native_tp1_tier", lambda: MTS.start_native_tp_variant(1, True)),
    ("pd_xnode_tp1_baseline", lambda: MTS.start_pd_xnode(1, False)),
    ("pd_xnode_tp1_tier", lambda: MTS.start_pd_xnode(1, True)),
    ("pdaf_tp1_baseline", lambda: MTS.start_pdaf_multi(1, False)),
    ("pdaf_tp1_tier", lambda: MTS.start_pdaf_multi(1, True)),
]

LABELS = {
    "native_tp1_baseline": "SGLang",
    "native_tp1_tier": "DynamoLLM",
    "pd_xnode_tp1_baseline": "DistServe",
    "pd_xnode_tp1_tier": "BiScale",
    "pdaf_tp1_baseline": "MegaScale",
    "pdaf_tp1_tier": "AFlex",
}

# (glob, scheme filter). None = all schemes in file. Listed in low→high priority.
MERGE_SOURCES: list[tuple[str, list[str] | None]] = [
    ("more_trying_sweep_20260706_041646.json", ["pdaf_tp1_baseline", "pdaf_tp1_tier"]),
    ("quick_dvfs_ab_pd_xnode_tp1_*.json", ["pd_xnode_tp1_baseline", "pd_xnode_tp1_tier"]),
    ("quick_dvfs_ab_native_tp1_*.json", ["native_tp1_baseline", "native_tp1_tier"]),
    ("fixed_6scheme_complete_partial_*.json", None),
]


def _discover_merge_files(extra: list[str] | None) -> list[tuple[Path, list[str] | None]]:
    if extra:
        return [(Path(p), None) for p in extra]
    found: list[tuple[Path, list[str] | None]] = []
    for pattern, schemes in MERGE_SOURCES:
        for path in sorted(RESULTS_DIR.glob(pattern)):
            found.append((path, schemes))
    return found


def _load_merge_files(sources: list[tuple[Path, list[str] | None]]) -> dict:
    """Later files override earlier ones (highest priority last)."""
    merged: dict = {}
    for path, scheme_filter in sources:
        if not path.exists():
            log.warning("Merge source missing: %s", path)
            continue
        data = json.loads(path.read_text())
        results = data.get("results", data)
        added = 0
        for scheme, wl in results.items():
            if scheme_filter is not None and scheme not in scheme_filter:
                continue
            if not isinstance(wl, dict) or "__status__" in wl:
                continue
            bucket = merged.setdefault(scheme, {})
            for key, metric in wl.items():
                if not _is_pass(metric):
                    continue
                if bucket.get(key) != metric:
                    added += 1
                bucket[key] = metric
        log.info("Merged %s (%d workloads touched)", path.name, added)
    return merged


def _workload_key(qps: int) -> str:
    return f"conv_qps{qps}"


def _is_pass(entry: dict | None) -> bool:
    return isinstance(entry, dict) and entry.get("status") == "PASS"


def _missing_qps(scheme_data: dict) -> list[int]:
    return [q for q in QPS_LIST if not _is_pass(scheme_data.get(_workload_key(q)))]


def _print_gap_plan(all_results: dict) -> None:
    log.info("Gap-fill plan (skip PASS workloads already merged):")
    for name, _ in SCHEMES:
        missing = _missing_qps(all_results.get(name, {}))
        if missing:
            log.info("  %s (%s): need QPS %s", name, LABELS[name], missing)
        else:
            log.info("  %s (%s): complete", name, LABELS[name])


def _save(all_results, tag="partial"):
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"fixed_6scheme_complete_{tag}_{ts}.json"
    payload = {
        "meta": {
            "node1": RMB.NODE1_IP,
            "node2": RMB.NODE2_IP,
            "dataset": DATASET,
            "qps": QPS_LIST,
            "dvfs_policy": "min_energy_under_slo",
            "config": {
                "native": "TP1 x16",
                "pd": "xnode TP1 (8P+8D)",
                "pdaf": "TP1 x4 instances",
            },
            "scheme_labels": LABELS,
        },
        "results": all_results,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Saved %s", out)
    return out


def _print_summary(all_results: dict) -> None:
    # Summary table
    print("\n" + "=" * 104)
    print("  FIXED 6-SCHEME COMPLETE")
    print("=" * 104)
    print(f"{'Scheme':<22} {'Label':<12} " + " ".join(f"Q{q:>2}" for q in QPS_LIST))
    print("-" * 104)
    for name, _ in SCHEMES:
        wl = all_results.get(name, {})
        if "__status__" in wl:
            print(f"{name:<22} {LABELS[name]:<12} {wl['__status__']}")
            continue
        cols = []
        for q in QPS_LIST:
            m = wl.get(f"conv_qps{q}", {})
            if m.get("status") == "PASS":
                cols.append(f"{m['energy_per_token_mj']:>6.0f}")
            else:
                cols.append("   FAIL")
        print(f"{name:<22} {LABELS[name]:<12} " + " ".join(cols))

    # Check 3 requirements per QPS
    print("\n" + "-" * 104)
    print("Energy ordering check (Dynamo<SGLang, BiScale<DistServe, AFlex<all):")
    keys = list(LABELS.keys())
    for q in QPS_LIST:
        vals = {}
        ok = True
        for name in keys:
            wl = all_results.get(name, {})
            if "__status__" in wl:
                ok = False
                break
            m = wl.get(f"conv_qps{q}", {})
            vals[name] = m.get("energy_per_token_mj") if m.get("status") == "PASS" else None
            if vals[name] is None:
                ok = False
        if not ok:
            print(f"  QPS{q}: incomplete")
            continue
        r1 = vals["native_tp1_tier"] < vals["native_tp1_baseline"]
        r2 = vals["pd_xnode_tp1_tier"] < vals["pd_xnode_tp1_baseline"]
        r3 = vals["pdaf_tp1_tier"] < min(
            vals["native_tp1_baseline"], vals["native_tp1_tier"],
            vals["pd_xnode_tp1_baseline"], vals["pd_xnode_tp1_tier"],
            vals["pdaf_tp1_baseline"],
        )
        mark = "PASS" if (r1 and r2 and r3) else "FAIL"
        print(f"  QPS{q}: {mark}  (Dynamo:{r1} BiScale:{r2} AFlex:{r3})")
    print("=" * 104)


def main():
    import resource

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--merge-from",
        nargs="*",
        default=None,
        help="JSON files to merge (default: auto-discover in results/)",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Ignore prior results; run full matrix",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Only merge existing JSON into final output; no benchmark runs",
    )
    parser.add_argument(
        "--schemes",
        nargs="*",
        choices=[s[0] for s in SCHEMES],
        help="Limit to specific scheme keys",
    )
    args = parser.parse_args()

    resource.setrlimit(resource.RLIMIT_MEMLOCK,
                       (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

    schemes = SCHEMES
    if args.schemes:
        wanted = set(args.schemes)
        schemes = [(n, fn) for n, fn in SCHEMES if n in wanted]

    all_results: dict = {}
    if not args.no_merge:
        merge_paths = _discover_merge_files(args.merge_from)
        merged = _load_merge_files(merge_paths)
        for name, wl in merged.items():
            all_results[name] = dict(wl)

    _print_gap_plan(all_results)

    if args.merge_only:
        out = _save(all_results, tag="final")
        _print_summary(all_results)
        log.info("Merge-only done -> %s", out)
        return

    gpus = RMB.card_gpus(NGPU)

    log.info("=" * 72)
    log.info("FIXED 6-SCHEME GAP-FILL | up to %d schemes x %d QPS", len(schemes), len(QPS_LIST))
    log.info("=" * 72)

    for name, deploy_fn in schemes:
        existing = all_results.get(name, {})
        if "__status__" in existing:
            existing = {}
        missing = _missing_qps(existing)
        if not missing:
            log.info("SKIP %s (%s): all QPS already present", name, LABELS[name])
            all_results[name] = existing
            continue

        log.info("\n" + "=" * 72)
        log.info("DEPLOY: %s (%s) | missing QPS %s", name, LABELS[name], missing)
        log.info("=" * 72)

        RMB.cleanup_all()
        url = deploy_fn()
        if url is None:
            existing["__status__"] = "DEPLOY_FAILED"
            all_results[name] = existing
            _save(all_results)
            continue

        RMB.lock_freq_both(gpus, RMB.MAX_GPU_FREQ)
        if not RMB.test_generate(url):
            RMB.unlock_freq_both(gpus)
            RMB.cleanup_all()
            existing["__status__"] = "WARMUP_FAILED"
            all_results[name] = existing
            _save(all_results)
            continue
        time.sleep(3)

        deploy_results = dict(existing)
        for qps in missing:
            log.info("-" * 50)
            log.info("  conv_qps%d (%d reqs, run_window=%ds)", qps, 200, MAX_RUN_S)
            res = RMB.run_one_workload(url, DATASET, qps, gpus, gpus, MAX_RUN_S)
            if res is not None:
                deploy_results[res[0]] = res[1]
            time.sleep(5)

        RMB.unlock_freq_both(gpus)
        RMB.cleanup_all()
        deploy_results.pop("__status__", None)
        all_results[name] = deploy_results
        _save(all_results)

    out = _save(all_results, tag="final")
    _print_summary(all_results)
    log.info("Done -> %s", out)


if __name__ == "__main__":
    main()
