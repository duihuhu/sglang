#!/usr/bin/env python3
"""Measure AFlex predictor accuracy: per-request pred vs actual TTFT/TPOT + energy.

Modes:
  replay  - offline: predictor vs profile LUT ground truth on macro workload
  live    - deploy AFlex, run workload subset, compare client metrics vs predictor
  logs    - analyze existing AFD_DVFS_DECISION_LOG jsonl files

Usage:
  python3 measure_aflex_predictor_accuracy.py replay --dataset code --qps 2 --limit 50
  python3 measure_aflex_predictor_accuracy.py live --dataset code --qps 2 --limit 30
  python3 measure_aflex_predictor_accuracy.py logs --log-dir timeline/logs/aflex_tier1
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pred_accuracy")

HERE = Path(__file__).resolve().parent
MACRO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(MACRO_ROOT))
sys.path.insert(0, str(HERE / "other_tier1"))
sys.path.insert(0, "/mnt/workspace/lt/sglang/python")

import bench_common as BC
import run_macro_benchmark as RMB

ENERGY_MODEL_DIR = (
    "/mnt/workspace/lt/sglang/benchmark/AFlex_bench/energy_model/Qwen3-32B/models_v1"
)
NUM_LAYERS = 64
T_COMM_US = 2900.0
T_DRAIN_US = 0.0
M_MICROBATCH = 1

OUT_DIR = HERE / "results" / "predictor_accuracy"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class AflexLayout:
    dataset: str
    qps: int
    tp_pa: int
    tp_pf: int
    tp_da: int
    tp_df: int
    f_pa: int
    f_pf: int
    f_da: int
    f_df: int
    k_p: int = 2
    k_d: int = 1

    @classmethod
    def from_macro_results(cls, dataset: str, qps: int) -> "AflexLayout":
        data = json.loads((MACRO_ROOT / "data" / "macro_e2e_all.json").read_text())
        key = BC.qps_key(qps)
        entry = (
            data.get("results", {})
            .get(dataset, {})
            .get(key, {})
            .get("aflex_tier1", {})
        )
        layout = entry.get("tier1_layout") or entry.get("config") or {}
        if not layout:
            raise KeyError(f"no AFlex layout for {dataset}/{key} in macro_e2e_all.json")
        pa, pf = layout.get("pa", [2, 1170]), layout.get("pf", [2, 930])
        da, df = layout.get("da", [1, 930]), layout.get("df", [1, 930])
        return cls(
            dataset=dataset, qps=qps,
            k_p=layout.get("k_p", 2), k_d=layout.get("k_d", 1),
            tp_pa=layout.get("tp_pa", pa[0]), tp_pf=layout.get("tp_pf", pf[0]),
            tp_da=layout.get("tp_da", da[0]), tp_df=layout.get("tp_df", df[0]),
            f_pa=layout.get("f_pa", pa[1]), f_pf=layout.get("f_pf", pf[1]),
            f_da=layout.get("f_da", da[1]), f_df=layout.get("f_df", df[1]),
        )

    def to_tier1_config(self):
        from bench_tier1_v2 import Tier1TestConfig
        return Tier1TestConfig(
            name=f"pred_acc_{self.dataset}_q{self.qps}",
            k_p=self.k_p, k_d=self.k_d,
            tp_pa=self.tp_pa, tp_pf=self.tp_pf,
            tp_da=self.tp_da, tp_df=self.tp_df,
            f_pa=self.f_pa, f_pf=self.f_pf,
            f_da=self.f_da, f_df=self.f_df,
            tier=True,
        )


def mape(y_true: list[float], y_pred: list[float]) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = (yt > 0) & np.isfinite(yp)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100)


def mae(y_true: list[float], y_pred: list[float]) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(yt[mask] - yp[mask])))


def pct_err(true: float, pred: float) -> float | None:
    if true <= 0 or not np.isfinite(pred):
        return None
    return (pred - true) / true * 100


class AflexPredictorEstimator:
    """Mirror AFlex compositional DVFS predictor formulas (M=1, no calibration)."""

    def __init__(self, layout: AflexLayout):
        from sglang.srt.energy.af_profile_predictor import AFProfilePredictor

        self.layout = layout
        self.predictor = AFProfilePredictor(ENERGY_MODEL_DIR)
        self.num_layers = NUM_LAYERS

    def _layer_lat_us(self, phase: str, tp_a: int, tp_f: int,
                      f_a: int, f_f: int, bs: int, il: int, ol: int | None) -> float:
        lat_a = self.predictor.predict_latency(phase, "A", tp_a, f_a, bs, il, ol).value
        lat_f = self.predictor.predict_latency(phase, "F", tp_f, f_f, bs, il, ol).value
        return lat_a + lat_f + T_COMM_US

    def _layer_energy_mj(self, phase: str, tp_a: int, tp_f: int,
                         f_a: int, f_f: int, bs: int, il: int, ol: int | None) -> float:
        e_a = self.predictor.predict_energy(phase, "A", tp_a, f_a, bs, il, ol).value
        e_f = self.predictor.predict_energy(phase, "F", tp_f, f_f, bs, il, ol).value
        return e_a + e_f

    def predict_prefill(self, il: int, bs: int = 1) -> dict:
        ly = self.layout
        t_layer = self._layer_lat_us("prefill", ly.tp_pa, ly.tp_pf,
                                     ly.f_pa, ly.f_pf, bs, il, None)
        lat_us = t_layer * self.num_layers
        e_layer = self._layer_energy_mj("prefill", ly.tp_pa, ly.tp_pf,
                                        ly.f_pa, ly.f_pf, bs, il, None)
        e_mj = e_layer * self.num_layers
        return {
            "ttft_pred_ms": lat_us / 1000,
            "ttft_energy_pred_mj": e_mj,
        }

    def predict_decode_step(self, il: int, ol: int, bs: int = 1) -> dict:
        ly = self.layout
        t_layer = self._layer_lat_us("decode", ly.tp_da, ly.tp_df,
                                     ly.f_da, ly.f_df, bs, il, max(ol, 1))
        lat_us = t_layer * self.num_layers + T_DRAIN_US
        e_layer = self._layer_energy_mj("decode", ly.tp_da, ly.tp_df,
                                        ly.f_da, ly.f_df, bs, il, max(ol, 1))
        e_mj = e_layer * self.num_layers
        return {
            "tpot_pred_ms": lat_us / 1000,
            "tpot_energy_pred_mj": e_mj,
        }

    def predict_request(self, il: int, ol: int, bs: int = 1) -> dict:
        pre = self.predict_prefill(il, bs)
        # TPOT: use mid-decode ol for representative iteration
        mid_ol = max(ol // 2, 1)
        dec = self.predict_decode_step(il, mid_ol, bs)
        decode_energy_total = dec["tpot_energy_pred_mj"] * max(ol, 1)
        return {
            **pre,
            **dec,
            "total_energy_pred_mj": pre["ttft_energy_pred_mj"] + decode_energy_total,
        }


class ProfileGroundTruth:
    """LUT lookup from v1 layer profile data (same training set as predictor)."""

    def __init__(self):
        self._prefill: dict[tuple, dict] = {}
        self._decode: dict[tuple, dict] = {}
        data_dir = Path(ENERGY_MODEL_DIR).parent / "data" / "v1_layer_profile"
        self._load_prefill(data_dir / "prefill_data_v1.txt")
        self._load_decode(data_dir / "decode_data_v1.txt")

    def _load_prefill(self, path: Path):
        for line in path.read_text().splitlines():
            if line.startswith("tp\t") or line.startswith("[") or not line.strip():
                continue
            p = line.strip().split("\t")
            if len(p) < 11:
                continue
            key = (int(p[0]), int(p[1]), int(p[3]), int(p[4]))
            self._prefill[key] = {
                "a_lat": float(p[5]), "f_lat": float(p[6]),
                "a_e": float(p[9]), "f_e": float(p[10]),
            }

    def _load_decode(self, path: Path):
        for line in path.read_text().splitlines():
            if line.startswith("tp\t") or line.startswith("[") or not line.strip():
                continue
            p = line.strip().split("\t")
            if len(p) < 11:
                continue
            key = (int(p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[4]))
            self._decode[key] = {
                "a_lat": float(p[5]), "f_lat": float(p[6]),
                "a_e": float(p[9]), "f_e": float(p[10]),
            }

    @staticmethod
    def _nearest(table: dict, target_key: tuple, numeric_idx: tuple[int, ...]):
        if target_key in table:
            return table[target_key]
        keys = list(table.keys())
        if not keys:
            return None
        best, best_d = None, float("inf")
        for k in keys:
            d = sum(abs(k[i] - target_key[i]) for i in numeric_idx)
            if d < best_d:
                best_d, best = d, k
        return table.get(best) if best is not None else None

    def actual_prefill(self, tp: int, il: int, freq: int, bs: int = 1) -> dict | None:
        row = self._nearest(self._prefill, (tp, il, freq, bs), (1, 2, 3))
        if row is None:
            return None
        lat_us = (row["a_lat"] + row["f_lat"]) * NUM_LAYERS
        e_mj = (row["a_e"] + row["f_e"]) * NUM_LAYERS
        return {"ttft_actual_ms": lat_us / 1000, "ttft_energy_actual_mj": e_mj}

    def actual_decode(self, tp: int, il: int, ol: int, freq: int, bs: int = 1) -> dict | None:
        row = self._nearest(self._decode, (tp, il, ol, freq, bs), (1, 2, 3, 4))
        if row is None:
            return None
        lat_us = (row["a_lat"] + row["f_lat"]) * NUM_LAYERS
        e_mj = (row["a_e"] + row["f_e"]) * NUM_LAYERS
        return {"tpot_actual_ms": lat_us / 1000, "tpot_energy_actual_mj": e_mj}


def summarize(rows: list[dict], prefix: str) -> dict:
    lat_true = [r[f"{prefix}_actual_ms"] for r in rows if r.get(f"{prefix}_actual_ms")]
    lat_pred = [r[f"{prefix}_pred_ms"] for r in rows if r.get(f"{prefix}_actual_ms")]
    e_true = [r[f"{prefix}_energy_actual_mj"] for r in rows if r.get(f"{prefix}_energy_actual_mj")]
    e_pred = [r[f"{prefix}_energy_pred_mj"] for r in rows if r.get(f"{prefix}_energy_actual_mj")]
    err_lat = [r[f"{prefix}_lat_err_pct"] for r in rows if r.get(f"{prefix}_lat_err_pct") is not None]
    err_e = [r[f"{prefix}_energy_err_pct"] for r in rows if r.get(f"{prefix}_energy_err_pct") is not None]
    return {
        "n": len(lat_true),
        "lat_mape_pct": round(mape(lat_true, lat_pred), 2),
        "lat_mae_ms": round(mae(lat_true, lat_pred), 3),
        "lat_err_mean_pct": round(float(np.mean(err_lat)), 2) if err_lat else None,
        "lat_err_p50_pct": round(float(np.percentile(err_lat, 50)), 2) if err_lat else None,
        "energy_mape_pct": round(mape(e_true, e_pred), 2),
        "energy_mae_mj": round(mae(e_true, e_pred), 3),
        "energy_err_mean_pct": round(float(np.mean(err_e)), 2) if err_e else None,
    }


def run_replay(args: argparse.Namespace) -> dict:
    layout = AflexLayout.from_macro_results(args.dataset, args.qps)
    est = AflexPredictorEstimator(layout)
    gt = ProfileGroundTruth()
    reqs = BC.load_workload(args.dataset, args.qps)[: args.limit]

    rows = []
    for i, req in enumerate(reqs):
        il, ol = req["input_len"], req["output_len"]
        pred = est.predict_request(il, ol, bs=1)
        # Prefill ground truth: use PA side freqs (hetero: lookup per op separately in profile)
        pre_gt = gt.actual_prefill(layout.tp_pa, il, layout.f_pa, bs=1)
        dec_gt = gt.actual_decode(layout.tp_da, il, max(ol // 2, 1), layout.f_da, bs=1)

        row = {
            "request_index": i,
            "input_len": il,
            "output_len": ol,
            "ttft_pred_ms": round(pred["ttft_pred_ms"], 3),
            "tpot_pred_ms": round(pred["tpot_pred_ms"], 3),
            "ttft_energy_pred_mj": round(pred["ttft_energy_pred_mj"], 3),
            "tpot_energy_pred_mj": round(pred["tpot_energy_pred_mj"], 3),
            "total_energy_pred_mj": round(pred["total_energy_pred_mj"], 3),
        }
        if pre_gt:
            row["ttft_actual_ms"] = round(pre_gt["ttft_actual_ms"], 3)
            row["ttft_energy_actual_mj"] = round(pre_gt["ttft_energy_actual_mj"], 3)
            row["ttft_lat_err_pct"] = round(pct_err(pre_gt["ttft_actual_ms"], pred["ttft_pred_ms"]), 2)
            row["ttft_energy_err_pct"] = round(
                pct_err(pre_gt["ttft_energy_actual_mj"], pred["ttft_energy_pred_mj"]), 2)
        if dec_gt:
            row["tpot_actual_ms"] = round(dec_gt["tpot_actual_ms"], 3)
            row["tpot_energy_actual_mj"] = round(dec_gt["tpot_energy_actual_mj"], 3)
            row["tpot_lat_err_pct"] = round(pct_err(dec_gt["tpot_actual_ms"], pred["tpot_pred_ms"]), 2)
            row["tpot_energy_err_pct"] = round(
                pct_err(dec_gt["tpot_energy_actual_mj"], pred["tpot_energy_pred_mj"]), 2)
        rows.append(row)

    result = {
        "mode": "replay",
        "dataset": args.dataset,
        "qps": args.qps,
        "layout": layout.__dict__,
        "n_requests": len(rows),
        "summary": {
            "ttft": summarize(rows, "ttft"),
            "tpot": summarize(rows, "tpot"),
        },
        "per_request": rows,
        "note": (
            "replay mode: actual values from profile LUT nearest-neighbor lookup "
            "(tp/freq/bs/il/ol grid), not live server measurement"
        ),
    }
    return result


def analyze_dvfs_logs(log_dir: Path) -> dict:
    decode_obs, decode_pred = [], []
    prefill_obs, prefill_pred = [], []
    decode_e_pred = []
    prefill_e_pred = []

    for p in sorted(log_dir.glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            phase = rec.get("phase")
            if phase == "decode":
                obs = rec.get("obs_iter_us")
                pred = rec.get("pred_iter_cur_us")
                if obs and pred and obs > 0 and pred > 0:
                    decode_obs.append(obs / 1000)
                    decode_pred.append(pred / 1000)
                e = rec.get("pred_sel_energy_mj")
                if e:
                    decode_e_pred.append(e)
            elif phase == "prefill":
                obs = rec.get("obs_lat_us")
                pred = rec.get("pred_lat_us")
                if obs and pred and obs > 0 and pred > 0:
                    prefill_obs.append(obs / 1000)
                    prefill_pred.append(pred / 1000)
                e = rec.get("pred_energy_mj")
                if e:
                    prefill_e_pred.append(e)

    return {
        "decode_iteration": {
            "n": len(decode_obs),
            "tpot_mape_pct": round(mape(decode_obs, decode_pred), 2),
            "tpot_mae_ms": round(mae(decode_obs, decode_pred), 3),
            "mean_obs_ms": round(float(np.mean(decode_obs)), 3) if decode_obs else None,
            "mean_pred_ms": round(float(np.mean(decode_pred)), 3) if decode_pred else None,
        },
        "prefill_batch": {
            "n": len(prefill_obs),
            "ttft_mape_pct": round(mape(prefill_obs, prefill_pred), 2),
            "ttft_mae_ms": round(mae(prefill_obs, prefill_pred), 3),
            "mean_obs_ms": round(float(np.mean(prefill_obs)), 3) if prefill_obs else None,
            "mean_pred_ms": round(float(np.mean(prefill_pred)), 3) if prefill_pred else None,
        },
        "n_decode_log_entries": len(decode_e_pred),
        "n_prefill_log_entries": len(prefill_e_pred),
    }


async def run_live_benchmark(url: str, reqs: list[dict], n1_gpus: list[int], n2_gpus: list[int]):
    import run_macro_benchmark as RMB_mod
    return await BC.run_workload_with_requests(
        reqs, url + "/generate",
        RMB_mod.get_energy_local, RMB_mod.get_energy_remote,
        n1_gpus, n2_gpus, max_run_s=300,
    )


def run_live(args: argparse.Namespace) -> dict:
    os.environ.setdefault("MN_NODE1_IP", "10.252.129.36")
    os.environ.setdefault("MN_NODE2_IP", "10.252.129.35")
    # bench_tier1_v2 defaults to MN_NODE3/4 at import time
    os.environ["MN_NODE3_IP"] = os.environ["MN_NODE1_IP"]
    os.environ["MN_NODE4_IP"] = os.environ["MN_NODE2_IP"]
    RMB.NODE1_IP = os.environ["MN_NODE1_IP"]
    RMB.NODE2_IP = os.environ["MN_NODE2_IP"]

    import bench_tier1_v2 as BT2
    from freq_timeline_utils import ensure_container_log_dir
    import deploy_schemes as DS

    DS._sync_sweep_nodes()
    BT2.RMB.NODE1_IP = RMB.NODE1_IP
    BT2.RMB.NODE2_IP = RMB.NODE2_IP
    BT2.RMB.TTFT_SLO_MS = RMB.TTFT_SLO_MS
    BT2.RMB.TPOT_SLO_MS = RMB.TPOT_SLO_MS

    # run_macro_benchmark parents[4] resolves to AFlex_bench/, fix dvfs.py path
    dvfs_src = Path("/mnt/workspace/lt/sglang/python/sglang/srt/layers/dvfs.py")
    if dvfs_src.exists():
        RMB.DVFS_PY_SRC = dvfs_src

    layout = AflexLayout.from_macro_results(args.dataset, args.qps)
    cfg = layout.to_tier1_config()
    est = AflexPredictorEstimator(layout)

    log_dir = HERE / "timeline" / "logs" / "aflex_pred_acc"
    log_dir.mkdir(parents=True, exist_ok=True)
    container_log = (
        "/workspace/sglang/benchmark/AFlex_bench/multi_node/"
        "more_test/macro/scripts/timeline/logs/aflex_pred_acc"
    )
    os.environ["AFD_DVFS_DECISION_LOG"] = f"{container_log}/dvfs_{{persp}}_{{disagg}}_gpu{{gpu}}.jsonl"

    log.info("Deploying AFlex %s qps=%d (limit %d reqs)...", args.dataset, args.qps, args.limit)
    DS.cleanup()
    ensure_container_log_dir("aflex_pred_acc")

    BT2.RMB = RMB
    BT2.DATASET = args.dataset
    BT2.QPS = args.qps
    urls = BT2.deploy(cfg)
    if not urls:
        RMB.cleanup_all()
        raise RuntimeError("AFlex deploy failed")

    if not DS.warmup_url(urls[0], "aflex_tier1"):
        RMB.cleanup_all()
        raise RuntimeError("warmup failed")

    all_reqs = BC.load_workload(args.dataset, args.qps)[: args.limit]
    assigned = BT2.assign_prefill_urls(urls, cfg, len(all_reqs))
    for r, target in zip(all_reqs, assigned):
        r["_target_url"] = target if target.endswith("/generate") else target + "/generate"

    gpus = list(range(8))
    log.info("Running %d requests...", len(all_reqs))
    bench = asyncio.run(BT2._run_workload_rr(all_reqs, gpus, gpus, max_run_s=300))
    RMB.cleanup_all()

    ok = [r for r in bench.get("request_results", []) if r.get("success")]
    req_by_idx = {i: r for i, r in enumerate(all_reqs)}
    total_energy_j = bench.get("total_energy_j", 0)
    total_tokens = bench.get("total_tokens_all") or bench.get("total_tokens") or 1
    energy_per_token_mj = total_energy_j * 1000 / total_tokens

    rows = []
    for r in ok:
        src = req_by_idx.get(r.get("request_index", -1), r)
        il = src.get("input_len", r.get("input_len", 0))
        ol = src.get("output_len", r.get("output_len", 0))
        pred = est.predict_request(il, ol, bs=1)
        ttft_actual = r.get("ttft_proc_ms") or r.get("ttft_ms", 0)
        tpot_actual = r.get("tpot_ms", 0)
        row = {
            "request_index": r["request_index"],
            "input_len": il,
            "output_len": ol,
            "ttft_actual_ms": ttft_actual,
            "tpot_actual_ms": tpot_actual,
            "ttft_pred_ms": round(pred["ttft_pred_ms"], 3),
            "tpot_pred_ms": round(pred["tpot_pred_ms"], 3),
            "ttft_energy_pred_mj": round(pred["ttft_energy_pred_mj"], 3),
            "tpot_energy_pred_mj": round(pred["tpot_energy_pred_mj"], 3),
            "total_energy_pred_mj": round(pred["total_energy_pred_mj"], 3),
            "energy_actual_per_token_mj": round(energy_per_token_mj, 3),
            "energy_pred_per_token_mj": round(pred["total_energy_pred_mj"] / max(il + ol, 1), 3),
        }
        if ttft_actual > 0:
            row["ttft_lat_err_pct"] = round(pct_err(ttft_actual, pred["ttft_pred_ms"]), 2)
        if tpot_actual > 0:
            row["tpot_lat_err_pct"] = round(pct_err(tpot_actual, pred["tpot_pred_ms"]), 2)
        rows.append(row)

    dvfs_summary = {}
    host_log = Path("/mnt/workspace/lt/sglang/benchmark/AFlex_bench/multi_node/more_test/macro/scripts/timeline/logs/aflex_pred_acc")
    if host_log.exists():
        dvfs_summary = analyze_dvfs_logs(host_log)

    result = {
        "mode": "live",
        "dataset": args.dataset,
        "qps": args.qps,
        "layout": layout.__dict__,
        "n_requests": len(rows),
        "summary": {
            "ttft": summarize(rows, "ttft"),
            "tpot": summarize(rows, "tpot"),
            "energy_per_token": {
                "actual_mj": round(energy_per_token_mj, 3),
                "pred_mean_mj": round(float(np.mean([r["energy_pred_per_token_mj"] for r in rows])), 3)
                if rows else None,
            },
            "dvfs_logs": dvfs_summary,
        },
        "per_request": rows,
        "note": (
            "live mode: TTFT/TPOT actual from client stream metrics (includes queueing); "
            "energy actual is run-level average per token (no per-request NVML)"
        ),
    }
    return result


def print_summary(result: dict):
    s = result["summary"]
    print("\n" + "=" * 72)
    print(f"AFlex Predictor Accuracy  mode={result['mode']}  "
          f"{result['dataset']} qps={result['qps']}  n={result['n_requests']}")
    print("=" * 72)
    for metric in ("ttft", "tpot"):
        m = s.get(metric, {})
        if not m or not m.get("n"):
            print(f"\n[{metric.upper()}] no data")
            continue
        print(f"\n[{metric.upper()}] n={m['n']}")
        print(f"  Latency  MAPE={m['lat_mape_pct']}%  MAE={m['lat_mae_ms']}ms  "
              f"mean_err={m['lat_err_mean_pct']}%  p50_err={m['lat_err_p50_pct']}%")
        if m.get("energy_mape_pct") is not None and not np.isnan(m["energy_mape_pct"]):
            print(f"  Energy   MAPE={m['energy_mape_pct']}%  MAE={m['energy_mae_mj']}mJ  "
                  f"mean_err={m['energy_err_mean_pct']}%")
    if "dvfs_logs" in s and s["dvfs_logs"]:
        d = s["dvfs_logs"]
        print(f"\n[DVFS LOGS - batch level]")
        if d.get("decode_iteration", {}).get("n"):
            di = d["decode_iteration"]
            print(f"  Decode iter: n={di['n']} MAPE={di['tpot_mape_pct']}% "
                  f"obs={di['mean_obs_ms']}ms pred={di['mean_pred_ms']}ms")
        if d.get("prefill_batch", {}).get("n"):
            pi = d["prefill_batch"]
            print(f"  Prefill batch: n={pi['n']} MAPE={pi['ttft_mape_pct']}% "
                  f"obs={pi['mean_obs_ms']}ms pred={pi['mean_pred_ms']}ms")
    if s.get("energy_per_token"):
        e = s["energy_per_token"]
        print(f"\n[ENERGY per token] actual={e.get('actual_mj')}mJ  pred_mean={e.get('pred_mean_mj')}mJ")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    for mode in ("replay", "live"):
        p = sub.add_parser(mode)
        p.add_argument("--dataset", default="code", choices=["code", "conv"])
        p.add_argument("--qps", type=int, default=2)
        p.add_argument("--limit", type=int, default=50)
        p.add_argument("--output", type=Path, default=None)

    p_logs = sub.add_parser("logs")
    p_logs.add_argument("--log-dir", type=Path, required=True)
    p_logs.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()

    if args.mode == "replay":
        result = run_replay(args)
    elif args.mode == "live":
        result = run_live(args)
    else:
        summary = analyze_dvfs_logs(args.log_dir)
        result = {"mode": "logs", "log_dir": str(args.log_dir), "summary": summary}

    print_summary(result)
    tag = f"{getattr(args, 'dataset', 'logs')}_q{getattr(args, 'qps', '')}"
    out = args.output or OUT_DIR / f"pred_acc_{result['mode']}_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out = Path(out)
    out.write_text(json.dumps(result, indent=2))
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
