#!/usr/bin/env python3
"""Energy Model V3: Decode Pipeline with Heterogeneous TP Training.

Extends V2 to support heterogeneous TP (tp_a != tp_f) for AF-disaggregated
decode pipeline. Feature columns: (tp_a, tp_f, M, f_A, f_F, input_len, batch_size).

Trains 3 coupled models:
  - Decode_iter_lat:      Predict iteration latency (us)
  - Decode_iter_energy_A: Predict DA energy per iteration (mJ)
  - Decode_iter_energy_F: Predict DF energy per iteration (mJ)

Input:  decode_pipeline_merged.txt (tp_a, tp_f, M, f_A, f_F, ...)
Output: models_v3/ directory with GBDT + LUT pickles

Usage:
    python energy_model_v3.py
    python energy_model_v3.py --data hucc/paper/decode_pipeline_merged.txt
    python energy_model_v3.py --output-dir /path/to/models_v3
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent


def load_pipeline_data(path: str) -> pd.DataFrame:
    """Load pipeline profiling data with tp_a/tp_f columns."""
    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.strip()
    required = ["tp_a", "tp_f", "M", "f_A", "f_F", "input_len",
                "batch_size", "iter_lat_us", "DA_energy_mj", "DF_energy_mj"]
    for col in required:
        assert col in df.columns, f"Missing column: {col}. Have: {df.columns.tolist()}"
    df = df.dropna(subset=["iter_lat_us", "DA_energy_mj", "DF_energy_mj"])
    df = df[df["iter_lat_us"] > 0]
    return df


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true > 0
    if mask.sum() == 0:
        return float("nan")
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


class GBDTModel:
    def __init__(self, name: str, feature_cols: list):
        self.name = name
        self.feature_cols = feature_cols
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray, n_rounds: int = 300):
        import lightgbm as lgb
        dtrain = lgb.Dataset(X, label=y, feature_name=self.feature_cols)
        params = {
            "objective": "regression",
            "metric": "mape",
            "num_leaves": 63,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 5,
            "verbose": -1,
        }
        self.model = lgb.train(params, dtrain, num_boost_round=n_rounds)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class LookupModel:
    def __init__(self, name: str, feature_cols: list, k: int = 5):
        self.name = name
        self.feature_cols = feature_cols
        self.k = k
        self.tree = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        from scipy.spatial import cKDTree
        self.mins = X.min(axis=0)
        self.maxs = X.max(axis=0)
        scale = self.maxs - self.mins
        scale[scale == 0] = 1.0
        self.scale = scale
        normed = (X - self.mins) / self.scale
        self.tree = cKDTree(normed)
        self.values = y.copy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        normed = (X - self.mins) / self.scale
        k = min(self.k, len(self.values))
        dists, idxs = self.tree.query(normed, k=k)
        if k == 1:
            return self.values[idxs]
        weights = 1.0 / (dists + 1e-10)
        weights /= weights.sum(axis=1, keepdims=True)
        return (weights * self.values[idxs]).sum(axis=1)


def cross_validate(df: pd.DataFrame, feature_cols: list, target_col: str,
                   n_splits: int = 5) -> dict:
    from sklearn.model_selection import KFold
    results = {"GBDT": [], "LUT": []}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(df):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        X_train = train_df[feature_cols].values.astype(float)
        y_train = train_df[target_col].values.astype(float)
        X_test = test_df[feature_cols].values.astype(float)
        y_test = test_df[target_col].values.astype(float)

        try:
            gbdt = GBDTModel(f"cv_{target_col}", feature_cols)
            gbdt.fit(X_train, y_train)
            pred = gbdt.predict(X_test)
            results["GBDT"].append(mape(y_test, pred))
        except ImportError:
            results["GBDT"].append(float("nan"))

        lut = LookupModel(f"cv_{target_col}", feature_cols)
        lut.fit(X_train, y_train)
        pred = lut.predict(X_test)
        results["LUT"].append(mape(y_test, pred))

        sys.stdout.write(".")
        sys.stdout.flush()

    sys.stdout.write(" ")
    return {k: (np.mean(v), np.std(v)) for k, v in results.items() if v}


def train_and_save(df: pd.DataFrame, feature_cols: list, target_col: str,
                   label: str, output_dir: str) -> dict:
    X = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(float)
    models = {}

    try:
        gbdt = GBDTModel(f"{label}_GBDT", feature_cols)
        gbdt.fit(X, y)
        models["GBDT"] = gbdt
        pred = gbdt.predict(X)
        print(f"    GBDT train MAPE: {mape(y, pred):.2f}%")
    except ImportError:
        print("    [WARN] lightgbm not installed, skipping GBDT")

    lut = LookupModel(f"{label}_LUT", feature_cols)
    lut.fit(X, y)
    models["LUT"] = lut
    pred = lut.predict(X)
    print(f"    LUT  train MAPE: {mape(y, pred):.2f}%")

    for mname, model in models.items():
        path = os.path.join(output_dir, f"{label}_{mname}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"    Saved: {path}")

    return models


def main():
    parser = argparse.ArgumentParser(
        description="Energy Model V3: Prefill layer + Decode pipeline (heterogeneous TP)")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(SCRIPT_DIR / "data/v2_pipeline_profile"),
        help="Self-contained V2/V3 dataset directory",
    )
    parser.add_argument(
        "--decode-data",
        type=str,
        default=None,
        help="Decode pipeline file (relative to --data-dir or absolute path)",
    )
    parser.add_argument("--output-dir", type=str,
                        default=str(SCRIPT_DIR / "models_v3"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--skip-cv", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    prefill_path = str(data_dir / "prefill_data_v1.txt")
    if args.decode_data:
        decode_path = Path(args.decode_data)
        if not decode_path.is_absolute():
            decode_path = data_dir / decode_path
    else:
        decode_path = data_dir / "decode_pipeline_merged.txt"
    decode_path = str(decode_path)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print(" Energy Model V3: Prefill Layer + Decode Pipeline (Heterogeneous TP)")
    print("=" * 70)
    print(f" Data dir: {data_dir}")

    from energy_model_v2 import train_prefill_models
    train_prefill_models(prefill_path, args.output_dir,
                         folds=args.folds, skip_cv=args.skip_cv)

    print(f"\n[Decode] Loading data from {decode_path}")
    df = load_pipeline_data(decode_path)
    print(f"    Rows: {len(df)}")
    print(f"    TP combinations: {sorted(df[['tp_a','tp_f']].drop_duplicates().values.tolist())}")
    print(f"    M values: {sorted(df['M'].unique())}")
    print(f"    f_A range: {df['f_A'].min()}-{df['f_A'].max()}")
    print(f"    f_F range: {df['f_F'].min()}-{df['f_F'].max()}")
    print(f"    BS range: {df['batch_size'].min()}-{df['batch_size'].max()}")
    print(f"    IL range: {df['input_len'].min()}-{df['input_len'].max()}")

    # V3 feature columns: tp_a, tp_f instead of single tp
    feature_cols = ["tp_a", "tp_f", "M", "f_A", "f_F", "input_len", "batch_size"]

    targets = [
        ("Decode_iter_lat", "iter_lat_us"),
        ("Decode_iter_energy_A", "DA_energy_mj"),
        ("Decode_iter_energy_F", "DF_energy_mj"),
    ]

    if not args.skip_cv:
        print(f"\n[2] {args.folds}-fold cross-validation")
        print("-" * 70)
        for label, target_col in targets:
            print(f"  {label} ({target_col}):", end=" ")
            cv = cross_validate(df, feature_cols, target_col, args.folds)
            for mtype, (mean_mape, std_mape) in cv.items():
                print(f"{mtype}={mean_mape:.2f}±{std_mape:.2f}%", end="  ")
            print()

    print(f"\n[3] Training final models → {args.output_dir}")
    print("-" * 70)
    for label, target_col in targets:
        print(f"  {label}:")
        train_and_save(df, feature_cols, target_col, label, args.output_dir)

    # Print data statistics
    print(f"\n[4] Data Statistics")
    print(f"{'=' * 70}")
    for (tp_a, tp_f), grp in df.groupby(["tp_a", "tp_f"]):
        print(f"\n  tp_a={tp_a}, tp_f={tp_f} ({len(grp)} rows):")
        for M_val in sorted(grp["M"].unique()):
            sub = grp[grp["M"] == M_val]
            print(f"    M={M_val} ({len(sub)} rows):")
            print(f"      iter_lat_us: mean={sub['iter_lat_us'].mean():.0f}, "
                  f"min={sub['iter_lat_us'].min():.0f}, max={sub['iter_lat_us'].max():.0f}")
            print(f"      DA_energy:   mean={sub['DA_energy_mj'].mean():.1f} mJ")
            print(f"      DF_energy:   mean={sub['DF_energy_mj'].mean():.1f} mJ")

    print(f"\n\nDone. V3 models saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
