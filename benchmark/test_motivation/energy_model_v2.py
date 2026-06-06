#!/usr/bin/env python3
"""
Energy Model V2: Decode Pipeline Coupled Model Training.

Trains 3 coupled models for the decode pipeline where A and F run on
separate GPUs with independent frequencies:
  - Decode_iter_lat:      Predict iteration latency (us) from (f_A, f_F, M, bs, il)
  - Decode_iter_energy_A: Predict DA energy per iteration (mJ)
  - Decode_iter_energy_F: Predict DF energy per iteration (mJ)

Note: output_len is NOT a feature — decode iteration latency/energy is
independent of KV cache length under paged attention.

Uses GBDT (LightGBM) as primary model type since the max(A,F) structure
and pipeline overlap create non-linear interactions well-suited to tree models.

Input:  decode_pipeline_v1.txt (from bench_decode_pipeline.py)
Output: Fitted models (pickle) to energy_models/ directory

Usage:
    python energy_model_v2.py
    python energy_model_v2.py --data hucc/paper/decode_pipeline_v1.txt
    python energy_model_v2.py --output-dir /path/to/energy_models
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
    """Load pipeline profiling data."""
    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.strip()
    required = ["tp", "M", "f_A", "f_F", "input_len",
                "batch_size", "iter_lat_us", "DA_energy_mj", "DF_energy_mj"]
    for col in required:
        assert col in df.columns, f"Missing column: {col}. Have: {df.columns.tolist()}"
    # Drop any invalid rows
    df = df.dropna(subset=["iter_lat_us", "DA_energy_mj", "DF_energy_mj"])
    df = df[df["iter_lat_us"] > 0]
    return df


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%)."""
    mask = y_true > 0
    if mask.sum() == 0:
        return float("nan")
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


class GBDTModel:
    """LightGBM GBDT model for coupled pipeline prediction."""

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
    """KNN lookup with inverse-distance weighting (normalized features)."""

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
    """K-fold CV returning MAPE for GBDT and LUT models."""
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

        # GBDT
        try:
            gbdt = GBDTModel(f"cv_{target_col}", feature_cols)
            gbdt.fit(X_train, y_train)
            pred = gbdt.predict(X_test)
            results["GBDT"].append(mape(y_test, pred))
        except ImportError:
            results["GBDT"].append(float("nan"))

        # LUT
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
    """Train GBDT + LUT on full data, save to pickle."""
    X = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(float)

    models = {}

    # GBDT (primary model)
    try:
        gbdt = GBDTModel(f"{label}_GBDT", feature_cols)
        gbdt.fit(X, y)
        models["GBDT"] = gbdt
        pred = gbdt.predict(X)
        print(f"    GBDT train MAPE: {mape(y, pred):.2f}%")
    except ImportError:
        print("    [WARN] lightgbm not installed, skipping GBDT")

    # LUT (fallback)
    lut = LookupModel(f"{label}_LUT", feature_cols)
    lut.fit(X, y)
    models["LUT"] = lut
    pred = lut.predict(X)
    print(f"    LUT  train MAPE: {mape(y, pred):.2f}%")

    # Save
    for mname, model in models.items():
        path = os.path.join(output_dir, f"{label}_{mname}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"    Saved: {path}")

    return models


def main():
    parser = argparse.ArgumentParser(
        description="Energy Model V2: Decode Pipeline Coupled Training")
    parser.add_argument("--data", type=str,
                        default=str(SCRIPT_DIR / "hucc/paper/decode_pipeline_v1.txt"))
    parser.add_argument("--output-dir", type=str,
                        default=str(SCRIPT_DIR / "energy_models"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--skip-cv", action="store_true",
                        help="Skip cross-validation, just train final models")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print(" Energy Model V2: Decode Pipeline Coupled Training")
    print("=" * 70)

    # Load data
    print(f"\n[1] Loading data from {args.data}")
    df = load_pipeline_data(args.data)
    print(f"    Rows: {len(df)}")
    print(f"    M values: {sorted(df['M'].unique())}")
    print(f"    f_A range: {df['f_A'].min()}-{df['f_A'].max()}")
    print(f"    f_F range: {df['f_F'].min()}-{df['f_F'].max()}")
    print(f"    BS range: {df['batch_size'].min()}-{df['batch_size'].max()}")
    print(f"    IL range: {df['input_len'].min()}-{df['input_len'].max()}")

    # Feature columns for coupled model (no output_len — decode is OL-independent)
    feature_cols = ["M", "f_A", "f_F", "input_len", "batch_size"]

    # 3 target models
    targets = [
        ("Decode_iter_lat", "iter_lat_us"),
        ("Decode_iter_energy_A", "DA_energy_mj"),
        ("Decode_iter_energy_F", "DF_energy_mj"),
    ]

    # Cross-validation
    if not args.skip_cv:
        print(f"\n[2] {args.folds}-fold cross-validation")
        print(f"\n{'Target':<25} {'GBDT MAPE%':>14} {'LUT MAPE%':>14}")
        print("-" * 55)

        cv_results = {}
        for label, target_col in targets:
            sys.stdout.write(f"{label:<25} ")
            sys.stdout.flush()
            res = cross_validate(df, feature_cols, target_col, n_splits=args.folds)
            cv_results[label] = res
            gbdt_str = f"{res['GBDT'][0]:.2f}±{res['GBDT'][1]:.2f}" if "GBDT" in res else "N/A"
            lut_str = f"{res['LUT'][0]:.2f}±{res['LUT'][1]:.2f}" if "LUT" in res else "N/A"
            print(f"{gbdt_str:>14} {lut_str:>14}")

        # Save CV report
        report_path = os.path.join(args.output_dir, "cv_mape_report_v2.tsv")
        with open(report_path, "w") as f:
            f.write("label\tmodel\tmape_mean\tmape_std\n")
            for label, res in cv_results.items():
                for mname, (mean, std) in res.items():
                    f.write(f"{label}\t{mname}\t{mean:.4f}\t{std:.4f}\n")
        print(f"\n    CV report saved: {report_path}")
    else:
        print("\n[2] Cross-validation skipped")

    # Train final models
    print(f"\n[3] Training final models on full data ({len(df)} rows)")
    for label, target_col in targets:
        print(f"\n  {label} ({target_col}):")
        train_and_save(df, feature_cols, target_col, label, args.output_dir)

    # Summary statistics
    print(f"\n{'=' * 70}")
    print(" Data Statistics:")
    print(f"{'=' * 70}")
    for M_val in sorted(df["M"].unique()):
        sub = df[df["M"] == M_val]
        print(f"\n  M={M_val} ({len(sub)} rows):")
        print(f"    iter_lat_us: mean={sub['iter_lat_us'].mean():.0f}, "
              f"min={sub['iter_lat_us'].min():.0f}, max={sub['iter_lat_us'].max():.0f}")
        print(f"    DA_energy:   mean={sub['DA_energy_mj'].mean():.1f} mJ")
        print(f"    DF_energy:   mean={sub['DF_energy_mj'].mean():.1f} mJ")

    print(f"\n\nDone. Models saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
