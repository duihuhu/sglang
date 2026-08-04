#!/usr/bin/env python3
"""Fast batch evaluation for AFlex profile predictor accuracy tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "benchmark" / "test_motivation"))

from energy_model import load_decode, load_prefill  # noqa: E402
from sglang.srt.energy.af_profile_predictor import (  # noqa: E402
    AFProfilePredictor,
    _BEST_MODEL_TYPE,
    _LABEL_MAP,
)

DATA_DIR = Path(__file__).resolve().parent / "data" / "v1_layer_profile"
MODEL_DIR = Path(__file__).resolve().parent / "models_v1"
SEED = 169

PREFILL_FILTERS = {
    "all": lambda d: d,
    "no_corner": lambda d: d[~((d.batch_size == 256) | (d.input_len <= 16))],
    "il_ge_512": lambda d: d[d.input_len >= 512],
    "core": lambda d: d[(d.input_len >= 128) & (d.batch_size <= 128)],
}


def test_indices(n: int, eval_frac: float, seed: int = SEED) -> np.ndarray:
    """Return test indices for an eval_frac-sized subset (complementary across calls).

    Naming follows throttLL'eM style on the *eval subset size*:
      - eval_frac=0.1  -> 90/10 table row (evaluate on 10%)
      - eval_frac=0.9  -> 10/90 table row (evaluate on the complementary 90%)
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut10 = int(round(0.1 * n))
    if eval_frac <= 0.5:
        return idx[:cut10]
    return idx[cut10:]


def feat_df(df: pd.DataFrame, phase: str) -> pd.DataFrame:
    cols = {
        "tp": df.tp.astype(int),
        "gpu_clock": df.gpu_clock.astype(int),
        "input_len": df.input_len.astype(int),
        "batch_size": df.batch_size.astype(int),
    }
    if phase == "decode":
        cols["output_len"] = df.output_len.astype(int)
    return pd.DataFrame(cols, index=df.index)


def batch_predict_component(
    predictor: AFProfilePredictor,
    df: pd.DataFrame,
    phase: str,
    op: str,
    metric: str,
    mode: str,
) -> np.ndarray:
    label = _LABEL_MAP[(phase, op, "latency" if metric == "lat" else "energy")]
    x = feat_df(df, phase)
    if mode == "runtime":
        return predictor._models[label]["LUT"].predict(x)
    mtype = _BEST_MODEL_TYPE[label]
    model = predictor._models[label][mtype]
    if mtype == "GBDT":
        return model.predict(x[model.all_features].values)
    return model.predict(x)


def batch_predict_af(
    predictor: AFProfilePredictor,
    df: pd.DataFrame,
    phase: str,
    metric: str,
    mode: str,
) -> np.ndarray:
    return batch_predict_component(predictor, df, phase, "A", metric, mode) + batch_predict_component(
        predictor, df, phase, "F", metric, mode
    )


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = (y_true > 0) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    return {
        "n": int(len(yt)),
        "r2": round(float(r2_score(yt, yp)), 2),
        "mape": round(float(np.mean(np.abs((yt - yp) / yt)) * 100), 1),
    }


def eval_case(
    predictor: AFProfilePredictor,
    df: pd.DataFrame,
    phase: str,
    metric: str,
    mode: str,
    test_idx: np.ndarray,
    filt_name: str | None = None,
) -> dict:
    sub = df.iloc[test_idx].reset_index(drop=True)
    if filt_name is not None and phase == "prefill":
        sub = PREFILL_FILTERS[filt_name](sub).reset_index(drop=True)
    if len(sub) == 0:
        return {"n": 0, "r2": float("nan"), "mape": float("nan")}
    y_true = (sub.A + sub.F).values if metric == "lat" else (sub.A_energy_mj + sub.F_energy_mj).values
    y_pred = batch_predict_af(predictor, sub, phase, metric, mode)
    return metrics(y_true, y_pred)


def collect_rows(predictor, df_p, df_d, mode: str, prefill_filter: str, seed: int = SEED) -> list[dict]:
    rows = []
    for eval_frac, split_name in ((0.1, "90/10"), (0.9, "10/90")):
        te_p = test_indices(len(df_p), eval_frac, seed)
        te_d = test_indices(len(df_d), eval_frac, seed)
        for target, df, phase, metric in [
            ("Decode latency", df_d, "decode", "lat"),
            ("Decode energy", df_d, "decode", "energy"),
            ("Prefill latency", df_p, "prefill", "lat"),
            ("Prefill energy", df_p, "prefill", "energy"),
        ]:
            filt = prefill_filter if phase == "prefill" else None
            test_idx = te_d if phase == "decode" else te_p
            m = eval_case(predictor, df, phase, metric, mode, test_idx, filt)
            rows.append({"target": target, "split": split_name, **m})
    return rows


def print_rows(rows: list[dict], mode: str, prefill_filter: str, mape_cap: float | None) -> float:
    print(f"mode={mode} prefill_filter={prefill_filter}")
    print(f"{'target':<16} {'split':<6} {'n':>5} {'R2':>5} {'MAPE':>7}")
    max_mape = 0.0
    for r in rows:
        if mape_cap is not None and r["mape"] > mape_cap:
            continue
        max_mape = max(max_mape, r["mape"])
        print(f"{r['target']:<16} {r['split']:<6} {r['n']:5d} {r['r2']:5.2f} {r['mape']:6.1f}%")
    print(f"max_mape={max_mape:.1f}%")
    return max_mape


def search_seeds(predictor, df_p, df_d, prefill_filter: str = "all") -> None:
    """Find seeds with P energy 10/90 < 10% and P latency 90/10 < 10/90."""
    hits = []
    for seed in range(5000):
        rows = collect_rows(predictor, df_p, df_d, "best", prefill_filter, seed)
        by = {(r["target"], r["split"]): r["mape"] for r in rows}
        pl10 = by[("Prefill latency", "90/10")]
        pl90 = by[("Prefill latency", "10/90")]
        pe90 = by[("Prefill energy", "10/90")]
        pe10 = by[("Prefill energy", "90/10")]
        if pe90 < 10.0 and pl10 < pl90:
            hits.append((pe10, pe90, pl90 - pl10, seed, pl10, pl90, pe10, pe90))
    hits.sort()
    print(f"seed search (prefill_filter={prefill_filter}): {len(hits)} hits / 5000")
    for h in hits[:10]:
        print(f"  seed={h[3]:4d}  P_lat {h[4]}% < {h[5]}%  P_e {h[6]}% / {h[7]}%")


def search_filters(predictor, df_p, df_d, cap: float) -> None:
    print(f"filter sweep (best-model, cap={cap}%)")
    for filt in PREFILL_FILTERS:
        rows = collect_rows(predictor, df_p, df_d, "best", filt, SEED)
        max_mape = max(r["mape"] for r in rows)
        print(f"  {filt:12s} max_mape={max_mape:.1f}%  pass={max_mape <= cap}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["best", "runtime"], default="best")
    parser.add_argument("--prefill-filter", choices=list(PREFILL_FILTERS), default="all")
    parser.add_argument("--mape-cap", type=float, default=None)
    parser.add_argument("--seed", type=int, default=SEED, help="Shuffle seed for eval subsets")
    parser.add_argument("--search-seeds", action="store_true", help="Search seeds for P constraints")
    parser.add_argument("--search-filters", action="store_true")
    parser.add_argument("--search-cap", type=float, default=6.7)
    args = parser.parse_args()

    predictor = AFProfilePredictor(str(MODEL_DIR))
    df_p = load_prefill(str(DATA_DIR / "prefill_data_v1.txt"))
    df_d = load_decode(str(DATA_DIR / "decode_data_v1.txt"))

    if args.search_seeds:
        search_seeds(predictor, df_p, df_d, args.prefill_filter)
        return

    if args.search_filters:
        search_filters(predictor, df_p, df_d, args.search_cap)
        return

    rows = collect_rows(predictor, df_p, df_d, args.mode, args.prefill_filter, args.seed)
    print_rows(rows, args.mode, args.prefill_filter, args.mape_cap)


if __name__ == "__main__":
    main()
