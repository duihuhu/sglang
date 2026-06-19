#!/usr/bin/env python3
"""Train V4 energy models with expert load features for MoE.

V4 extends V3 with expert distribution features:
  Features: (tp_a, tp_f, M, f_A, f_F, input_len, batch_size, max_expert_tokens, els)
  Targets:  iter_lat_us, DA_energy_mj, DF_energy_mj

Also retrains V2/V3 models with all available data (without expert features)
for backward compatibility.
"""
import sys
sys.path.insert(0, "/workspace/sglang-tier/benchmark/AFlex_bench/03_sensitivity/slo_sweep/retrain")

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from energy_model_v3 import LookupModel

DATA_DIR = Path("/workspace/sglang-tier/benchmark/AFlex_bench/06_others/more_model/energy_model/data")
V2_MODEL_DIR = Path("/workspace/sglang-tier/benchmark/AFlex_bench/06_others/more_model/energy_model/models")
V3_MODEL_DIR = Path("/workspace/sglang-tier/benchmark/AFlex_bench/06_others/more_model/energy_model/models_v3")
V4_MODEL_DIR = Path("/workspace/sglang-tier/benchmark/AFlex_bench/06_others/more_model/energy_model/models_v4")
V2_MODEL_DIR.mkdir(parents=True, exist_ok=True)
V3_MODEL_DIR.mkdir(parents=True, exist_ok=True)
V4_MODEL_DIR.mkdir(parents=True, exist_ok=True)

V2_FEATURES = ["tp", "M", "f_A", "f_F", "input_len", "batch_size"]
V3_FEATURES = ["tp_a", "tp_f", "M", "f_A", "f_F", "input_len", "batch_size"]
V4_FEATURES = ["tp_a", "tp_f", "M", "f_A", "f_F", "input_len", "batch_size",
               "max_expert_tokens", "els"]

TARGETS = [
    ("Decode_iter_lat", "iter_lat_us"),
    ("Decode_iter_energy_A", "DA_energy_mj"),
    ("Decode_iter_energy_F", "DF_energy_mj"),
]


def mape(y_true, y_pred):
    mask = y_true > 0
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def load_all_data():
    """Load all available decode pipeline data."""
    dfs = []

    # TP=1 data
    tp1_path = DATA_DIR / "decode_pipeline_moe_tp1.txt"
    if tp1_path.exists():
        df = pd.read_csv(tp1_path, sep="\t")
        df["tp_a"] = df.get("tp_a", df["tp"])
        df["tp_f"] = df.get("tp_f", df["tp"])
        dfs.append(df)
        print(f"  TP1: {len(df)} rows")

    # TP=2 data
    tp2_path = DATA_DIR / "decode_pipeline_moe_tp2.txt"
    if tp2_path.exists():
        df = pd.read_csv(tp2_path, sep="\t")
        df["tp_a"] = df.get("tp_a", df["tp"])
        df["tp_f"] = df.get("tp_f", df["tp"])
        dfs.append(df)
        print(f"  TP2: {len(df)} rows")

    # Hetero data (DA-TP2, DF-TP4)
    hetero_path = DATA_DIR / "decode_pipeline_moe_hetero_2a4f.txt"
    if hetero_path.exists():
        df = pd.read_csv(hetero_path, sep="\t")
        dfs.append(df)
        print(f"  Hetero 2A4F: {len(df)} rows")

    # Hetero data (DA-TP1, DF-TP4)
    hetero_1a4f_path = DATA_DIR / "decode_pipeline_moe_1a4f.txt"
    if hetero_1a4f_path.exists():
        df = pd.read_csv(hetero_1a4f_path, sep="\t")
        dfs.append(df)
        print(f"  Hetero 1A4F: {len(df)} rows")

    # V4 expert data
    v4_path = DATA_DIR / "decode_pipeline_moe_v4_expert.txt"
    if v4_path.exists():
        df = pd.read_csv(v4_path, sep="\t")
        df["tp_a"] = df.get("tp_a", df["tp"])
        df["tp_f"] = df.get("tp_f", df["tp"])
        dfs.append(df)
        print(f"  V4 expert: {len(df)} rows")

    # Merged data
    merged_path = DATA_DIR / "decode_pipeline_merged.txt"
    if merged_path.exists():
        df = pd.read_csv(merged_path, sep="\t")
        if "tp_a" not in df.columns:
            df["tp_a"] = df["tp"]
            df["tp_f"] = df["tp"]
        dfs.append(df)
        print(f"  Merged: {len(df)} rows")

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.dropna(subset=["iter_lat_us"]).query("iter_lat_us > 0")
    print(f"  Combined total: {len(combined)} rows")
    return combined


def train_v2_models(df):
    """Train V2 models (backward compatible, no expert features)."""
    print("\n" + "=" * 60)
    print("  V2 Models (features: tp, M, f_A, f_F, input_len, batch_size)")
    print("=" * 60)

    v2_df = df.dropna(subset=V2_FEATURES + ["iter_lat_us"])
    if "tp" not in v2_df.columns:
        v2_df = v2_df.copy()
        v2_df["tp"] = v2_df["tp_a"]

    for label, target_col in TARGETS:
        valid = v2_df.dropna(subset=[target_col])
        X = valid[V2_FEATURES].values.astype(float)
        y = valid[target_col].values.astype(float)

        lut = LookupModel(label, V2_FEATURES, k=5)
        lut.fit(X, y)
        pred = lut.predict(X)
        m = mape(y, pred)
        print(f"  {label}_LUT: n={len(y)}, MAPE={m:.2f}%")

        for suffix in ("LUT", "GBDT"):
            with open(V2_MODEL_DIR / f"{label}_{suffix}.pkl", "wb") as f:
                pickle.dump(lut, f)


def train_v3_models(df):
    """Train V3 models (hetero TP, no expert features)."""
    print("\n" + "=" * 60)
    print("  V3 Models (features: tp_a, tp_f, M, f_A, f_F, input_len, batch_size)")
    print("=" * 60)

    v3_df = df.dropna(subset=V3_FEATURES + ["iter_lat_us"])

    for label, target_col in TARGETS:
        valid = v3_df.dropna(subset=[target_col])
        X = valid[V3_FEATURES].values.astype(float)
        y = valid[target_col].values.astype(float)

        lut = LookupModel(label, V3_FEATURES, k=5)
        lut.fit(X, y)
        pred = lut.predict(X)
        m = mape(y, pred)
        print(f"  {label}_LUT: n={len(y)}, MAPE={m:.2f}%")

        for suffix in ("LUT", "GBDT"):
            with open(V3_MODEL_DIR / f"{label}_{suffix}.pkl", "wb") as f:
                pickle.dump(lut, f)


def train_v4_models(df):
    """Train V4 models with expert load features."""
    print("\n" + "=" * 60)
    print("  V4 Models (features: tp_a, tp_f, M, f_A, f_F, input_len, batch_size, "
          "max_expert_tokens, els)")
    print("=" * 60)

    has_expert = df["max_expert_tokens"].notna() & (df["max_expert_tokens"] > 0)
    v4_df = df[has_expert].dropna(subset=V4_FEATURES + ["iter_lat_us"])
    print(f"  V4 data with expert features: {len(v4_df)} rows")

    if len(v4_df) < 10:
        print("  Not enough V4 data, skipping V4 training")
        return

    for label, target_col in TARGETS:
        valid = v4_df.dropna(subset=[target_col])
        X = valid[V4_FEATURES].values.astype(float)
        y = valid[target_col].values.astype(float)

        lut = LookupModel(label, V4_FEATURES, k=5)
        lut.fit(X, y)
        pred = lut.predict(X)
        m = mape(y, pred)
        print(f"  {label}_LUT: n={len(y)}, MAPE={m:.2f}%")

        for suffix in ("LUT", "GBDT"):
            with open(V4_MODEL_DIR / f"{label}_{suffix}.pkl", "wb") as f:
                pickle.dump(lut, f)

    # Also train a combined model: V3 data (with default expert features) + V4 data
    print("\n  Training V4 combined (V3 data with default expert values + V4 data):")
    df_combined = df.copy()
    df_combined["max_expert_tokens"] = df_combined["max_expert_tokens"].fillna(0)
    df_combined["els"] = df_combined["els"].fillna(1.0)
    v4_combined = df_combined.dropna(subset=V4_FEATURES[:7] + ["iter_lat_us"])

    for label, target_col in TARGETS:
        valid = v4_combined.dropna(subset=[target_col])
        X = valid[V4_FEATURES].values.astype(float)
        y = valid[target_col].values.astype(float)

        lut = LookupModel(label, V4_FEATURES, k=5)
        lut.fit(X, y)
        pred = lut.predict(X)
        m = mape(y, pred)
        print(f"  {label}_LUT (combined): n={len(y)}, MAPE={m:.2f}%")

        with open(V4_MODEL_DIR / f"{label}_combined_LUT.pkl", "wb") as f:
            pickle.dump(lut, f)


def main():
    print("Training MoE Energy Models (V2 + V3 + V4)")
    print("=" * 60)
    print("\nLoading data...")
    df = load_all_data()

    train_v2_models(df)
    train_v3_models(df)
    train_v4_models(df)

    print("\n" + "=" * 60)
    print("  All models saved!")
    print(f"  V2: {V2_MODEL_DIR}")
    print(f"  V3: {V3_MODEL_DIR}")
    print(f"  V4: {V4_MODEL_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
