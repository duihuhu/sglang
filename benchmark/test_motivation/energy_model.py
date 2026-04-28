#!/usr/bin/env python3
"""
T2-1: Energy model fitting for AF-disaggregated LLM inference.

Fits 4 sub-models (Prefill-A, Prefill-F, Decode-A, Decode-F) using three
approaches and compares their MAPE:
  1. Lookup Table (LUT) with linear interpolation
  2. Linear Regression (polynomial features)
  3. GBDT (LightGBM)

Input:  prefill_data_v1.txt, decode_data_v1.txt
Output: MAPE comparison table + fitted models (pickle)

Usage:
    python energy_model.py
    python energy_model.py --prefill paper/prefill_data_v1.txt --decode decode_data_v1.txt
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent


# ═══════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════

def load_prefill(path: str) -> pd.DataFrame:
    """Load prefill profiling data (skip first comment line)."""
    df = pd.read_csv(path, sep="\t", skiprows=1)
    df.columns = df.columns.str.strip()
    assert "A_energy_mj" in df.columns, f"Missing A_energy_mj in {df.columns.tolist()}"
    return df


def load_decode(path: str) -> pd.DataFrame:
    """Load decode profiling data (skip first comment line)."""
    df = pd.read_csv(path, sep="\t", skiprows=1)
    df.columns = df.columns.str.strip()
    assert "A_energy_mj" in df.columns, f"Missing A_energy_mj in {df.columns.tolist()}"
    return df


# ═══════════════════════════════════════════════════════════════════════
# Model 1: Lookup Table + Interpolation
# ═══════════════════════════════════════════════════════════════════════

class LookupTableModel:
    """K-nearest-neighbor lookup table (weighted by inverse distance).

    Conceptually equivalent to a lookup table with interpolation, but
    uses KD-tree for O(log N) queries instead of grid construction.
    Features are normalized to [0,1] before distance computation.
    """

    def __init__(self, name: str, k: int = 3):
        self.name = name
        self.k = k
        self.models = {}  # tp -> (KDTree, values, scaler)

    def fit(self, df: pd.DataFrame, feature_cols: list[str],
            target_col: str, group_col: str = "tp"):
        from scipy.spatial import cKDTree

        self.feature_cols = feature_cols
        self.target_col = target_col

        for tp_val, grp in df.groupby(group_col):
            points = grp[feature_cols].values.astype(float)
            values = grp[target_col].values.astype(float)
            # Normalize features to [0,1] for balanced distance
            mins = points.min(axis=0)
            maxs = points.max(axis=0)
            scale = maxs - mins
            scale[scale == 0] = 1.0
            normed = (points - mins) / scale
            tree = cKDTree(normed)
            self.models[tp_val] = (tree, values, mins, scale)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        preds = np.full(len(df), np.nan)
        for tp_val, grp in df.groupby("tp"):
            if tp_val not in self.models:
                continue
            tree, values, mins, scale = self.models[tp_val]
            idx = grp.index.values
            points = grp[self.feature_cols].values.astype(float)
            normed = (points - mins) / scale
            k = min(self.k, len(values))
            dists, idxs = tree.query(normed, k=k)
            if k == 1:
                preds[idx] = values[idxs]
            else:
                # Inverse distance weighting
                weights = 1.0 / (dists + 1e-10)
                weights /= weights.sum(axis=1, keepdims=True)
                preds[idx] = (weights * values[idxs]).sum(axis=1)
        return preds


# ═══════════════════════════════════════════════════════════════════════
# Model 2: Polynomial Linear Regression
# ═══════════════════════════════════════════════════════════════════════

class LinearRegressionModel:
    """Linear regression on log-transformed features and target.

    All features and target are log-transformed to handle the wide dynamic
    range (gpu_clock ~210-1410, input_len ~128-40000, batch_size ~1-256).
    Features: log(f), log(il), log(bs), log(f)*log(il), log(f)*log(bs), etc.
    Fitted per-tp with StandardScaler for numerical stability.
    """

    def __init__(self, name: str):
        self.name = name
        self.models = {}  # tp -> (pipeline, feature_names)

    def _make_features(self, df: pd.DataFrame,
                       base_cols: list[str]) -> np.ndarray:
        """Generate features in log space."""
        X_parts = []
        log_vals = {c: np.log(df[c].values.astype(float) + 1.0)
                    for c in base_cols}

        # Log features
        for c in base_cols:
            X_parts.append(log_vals[c])

        # Squared log features
        for c in base_cols:
            X_parts.append(log_vals[c] ** 2)

        # Pairwise interactions (in log space)
        for i in range(len(base_cols)):
            for j in range(i + 1, len(base_cols)):
                X_parts.append(log_vals[base_cols[i]] * log_vals[base_cols[j]])

        return np.column_stack(X_parts)

    def fit(self, df: pd.DataFrame, feature_cols: list[str],
            target_col: str, group_col: str = "tp"):
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        self.feature_cols = feature_cols
        self.target_col = target_col

        for tp_val, grp in df.groupby(group_col):
            X = self._make_features(grp, feature_cols)
            y = np.log(grp[target_col].values + 1.0)  # log target
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=10.0)),
            ])
            pipe.fit(X, y)
            self.models[tp_val] = pipe

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        preds = np.zeros(len(df))
        for tp_val, grp in df.groupby("tp"):
            if tp_val not in self.models:
                continue
            idx = grp.index
            X = self._make_features(grp, self.feature_cols)
            log_pred = self.models[tp_val].predict(X)
            preds[idx] = np.exp(log_pred) - 1.0  # inverse log transform
        return preds


# ═══════════════════════════════════════════════════════════════════════
# Model 3: GBDT (LightGBM)
# ═══════════════════════════════════════════════════════════════════════

class GBDTModel:
    """LightGBM gradient boosted decision tree model.

    Single model across all tp values (tp as a feature).
    """

    def __init__(self, name: str):
        self.name = name
        self.model = None

    def fit(self, df: pd.DataFrame, feature_cols: list[str],
            target_col: str, **kwargs):
        import lightgbm as lgb
        all_features = ["tp"] + feature_cols
        X = df[all_features].values
        y = df[target_col].values

        dtrain = lgb.Dataset(X, label=y, feature_name=all_features)
        params = {
            "objective": "regression",
            "metric": "mape",
            "num_leaves": 31,
            "learning_rate": 0.1,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }
        self.model = lgb.train(params, dtrain, num_boost_round=200)
        self.feature_cols = feature_cols
        self.all_features = all_features

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.all_features].values
        return self.model.predict(X)


# ═══════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%)."""
    mask = y_true > 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_models(df: pd.DataFrame, feature_cols: list[str],
                    target_col: str, label: str,
                    n_splits: int = 5) -> dict:
    """K-fold cross-validation for all three model types.

    Uses shuffled KFold (not grouped by tp) so that each fold has all tp
    values in both train and test. This tests interpolation accuracy for
    LUT and generalization for regression/GBDT.
    """
    from sklearn.model_selection import KFold
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*Ill-conditioned.*")

    results = {"LUT": [], "LinearReg": [], "GBDT": []}

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        y_test = test_df[target_col].values

        # LUT
        lut = LookupTableModel(f"{label}_LUT")
        lut.fit(train_df, feature_cols, target_col)
        pred_lut = lut.predict(test_df)
        valid = ~np.isnan(pred_lut)
        if valid.sum() > 0:
            results["LUT"].append(mape(y_test[valid], pred_lut[valid]))

        # Linear Regression
        lr = LinearRegressionModel(f"{label}_LR")
        lr.fit(train_df, feature_cols, target_col)
        pred_lr = lr.predict(test_df)
        results["LinearReg"].append(mape(y_test, pred_lr))

        # GBDT
        try:
            gbdt = GBDTModel(f"{label}_GBDT")
            gbdt.fit(train_df, feature_cols, target_col)
            pred_gbdt = gbdt.predict(test_df)
            results["GBDT"].append(mape(y_test, pred_gbdt))
        except ImportError:
            results["GBDT"].append(float("nan"))

        sys.stdout.write(".")
        sys.stdout.flush()

    sys.stdout.write(" ")
    sys.stdout.flush()
    return {k: (np.mean(v), np.std(v)) for k, v in results.items() if v}


def fit_final_models(df: pd.DataFrame, feature_cols: list[str],
                     target_col: str, label: str) -> dict:
    """Fit all models on full data and return them."""
    import warnings
    warnings.filterwarnings("ignore", message=".*Ill-conditioned.*")

    models = {}

    lut = LookupTableModel(f"{label}_LUT")
    lut.fit(df, feature_cols, target_col)
    models["LUT"] = lut

    lr = LinearRegressionModel(f"{label}_LR")
    lr.fit(df, feature_cols, target_col)
    models["LinearReg"] = lr

    try:
        gbdt = GBDTModel(f"{label}_GBDT")
        gbdt.fit(df, feature_cols, target_col)
        models["GBDT"] = gbdt
    except ImportError:
        print("  [WARN] lightgbm not installed, skipping GBDT")

    return models


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="T2-1: Energy model fitting")
    parser.add_argument("--prefill", type=str,
                        default=str(SCRIPT_DIR / "paper" / "prefill_data_v1.txt"))
    parser.add_argument("--decode", type=str,
                        default=str(SCRIPT_DIR / "paper" / "decode_data_v1.txt"))
    parser.add_argument("--output-dir", type=str,
                        default=str(SCRIPT_DIR / "energy_models"))
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print(" T2-1: Energy Model Fitting")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading data...")
    df_p = load_prefill(args.prefill)
    df_d = load_decode(args.decode)
    print(f"  Prefill: {len(df_p)} rows, cols: {df_p.columns.tolist()}")
    print(f"  Decode:  {len(df_d)} rows, cols: {df_d.columns.tolist()}")

    # Define 4 sub-models
    tasks = [
        ("Prefill_A", df_p, ["gpu_clock", "input_len", "batch_size"], "A_energy_mj"),
        ("Prefill_F", df_p, ["gpu_clock", "input_len", "batch_size"], "F_energy_mj"),
        ("Decode_A",  df_d, ["gpu_clock", "input_len", "output_len", "batch_size"], "A_energy_mj"),
        ("Decode_F",  df_d, ["gpu_clock", "input_len", "output_len", "batch_size"], "F_energy_mj"),
    ]

    # Also fit latency models (needed for Tier 2 SLO check)
    latency_tasks = [
        ("Prefill_A_lat", df_p, ["gpu_clock", "input_len", "batch_size"], "A"),
        ("Prefill_F_lat", df_p, ["gpu_clock", "input_len", "batch_size"], "F"),
        ("Decode_A_lat",  df_d, ["gpu_clock", "input_len", "output_len", "batch_size"], "A"),
        ("Decode_F_lat",  df_d, ["gpu_clock", "input_len", "output_len", "batch_size"], "F"),
    ]

    all_tasks = tasks + latency_tasks

    # Cross-validation
    print(f"\n[2/4] {args.folds}-fold cross-validation...")
    print(f"\n{'Label':<20} {'LUT MAPE%':>12} {'LinearReg%':>12} {'GBDT%':>12}")
    print("-" * 60)

    cv_results = {}
    for label, df, feat_cols, target in all_tasks:
        sys.stdout.write(f"{label:<20} ")
        sys.stdout.flush()
        res = evaluate_models(df, feat_cols, target, label, n_splits=args.folds)
        cv_results[label] = res
        lut_str = f"{res['LUT'][0]:.2f}±{res['LUT'][1]:.2f}" if "LUT" in res else "N/A"
        lr_str = f"{res['LinearReg'][0]:.2f}±{res['LinearReg'][1]:.2f}" if "LinearReg" in res else "N/A"
        gbdt_str = f"{res['GBDT'][0]:.2f}±{res['GBDT'][1]:.2f}" if "GBDT" in res else "N/A"
        print(f"{lut_str:>12} {lr_str:>12} {gbdt_str:>12}")

    # Fit final models on full data
    print(f"\n[3/4] Fitting final models on full data...")
    all_models = {}
    for label, df, feat_cols, target in all_tasks:
        models = fit_final_models(df, feat_cols, target, label)
        all_models[label] = models

        # Report train MAPE
        y = df[target].values
        for mname, model in models.items():
            pred = model.predict(df)
            valid = ~np.isnan(pred)
            if valid.sum() > 0:
                m = mape(y[valid], pred[valid])
                print(f"  {label} {mname}: train MAPE = {m:.2f}%")

    # Save models
    print(f"\n[4/4] Saving models to {args.output_dir}/")
    for label, models in all_models.items():
        for mname, model in models.items():
            path = os.path.join(args.output_dir, f"{label}_{mname}.pkl")
            with open(path, "wb") as f:
                pickle.dump(model, f)

    # Save CV results as TSV
    report_path = os.path.join(args.output_dir, "cv_mape_report.tsv")
    with open(report_path, "w") as f:
        f.write("label\tmodel\tmape_mean\tmape_std\n")
        for label, res in cv_results.items():
            for mname, (mean, std) in res.items():
                f.write(f"{label}\t{mname}\t{mean:.4f}\t{std:.4f}\n")
    print(f"  CV report: {report_path}")

    # Summary
    print(f"\n{'=' * 70}")
    print(" Summary: Best model per sub-task (by CV MAPE)")
    print(f"{'=' * 70}")
    for label, res in cv_results.items():
        best = min(res.items(), key=lambda x: x[1][0])
        print(f"  {label:<20} → {best[0]:<12} MAPE = {best[1][0]:.2f}%")

    print(f"\nDone. Models saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
