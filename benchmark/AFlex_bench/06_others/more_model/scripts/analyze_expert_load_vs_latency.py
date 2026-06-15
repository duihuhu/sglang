#!/usr/bin/env python3
"""Analyze MoE expert load features vs latency: regression + visualization.

Loads the profiling data (both single-request and batch modes), extracts
expert distribution features, performs regression analysis to determine
which features best predict latency, and generates visualization plots.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("/workspace/sglang-tier/benchmark/AFlex_bench/06_others/more_model/data")
OUT_DIR = DATA_DIR.parent / "analysis_expert_load"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SINGLE_FILE = DATA_DIR / "expert_load_vs_latency.tsv"
BATCH_FILE = DATA_DIR / "expert_load_vs_latency_batch.tsv"


def load_data():
    """Load both datasets and combine."""
    dfs = []

    if SINGLE_FILE.exists():
        df_single = pd.read_csv(SINGLE_FILE, sep="\t")
        df_single["batch_size"] = 1
        df_single.rename(columns={"tpot_ms": "tpot_ms_avg"}, inplace=True)
        dfs.append(df_single)
        print(f"Single-request data: {len(df_single)} rows")

    if BATCH_FILE.exists():
        df_batch = pd.read_csv(BATCH_FILE, sep="\t")
        dfs.append(df_batch)
        print(f"Batch data: {len(df_batch)} rows")

    if not dfs:
        raise FileNotFoundError("No data files found")

    df = pd.concat(dfs, ignore_index=True)
    print(f"Combined: {len(df)} rows")
    return df


def prepare_features(df):
    """Extract feature matrix and target variable."""
    feature_cols = ["els", "std_tokens_per_expert", "num_active_experts",
                    "top1_share", "glr", "gini", "max_tokens_per_expert"]
    baseline_cols = ["batch_size"]
    all_cols = baseline_cols + feature_cols

    available_cols = [c for c in all_cols if c in df.columns]
    missing = [c for c in all_cols if c not in df.columns]
    if missing:
        print(f"  Warning: missing columns: {missing}")

    df_clean = df.dropna(subset=["tpot_ms_avg"] + available_cols)
    X = df_clean[available_cols].values.astype(float)
    y = df_clean["tpot_ms_avg"].values.astype(float)

    return X, y, available_cols, df_clean


def run_regression_analysis(X, y, feature_names):
    """Run multiple regression models and compare with/without expert features."""
    print("\n" + "=" * 70)
    print("  REGRESSION ANALYSIS: Expert Features vs Latency")
    print("=" * 70)

    bs_idx = feature_names.index("batch_size") if "batch_size" in feature_names else None
    expert_feature_indices = [i for i, n in enumerate(feature_names) if n != "batch_size"]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    results = {}

    # Model 1: Baseline (only batch_size)
    if bs_idx is not None:
        X_baseline = X[:, [bs_idx]]
        lr_base = LinearRegression()
        scores_base = cross_val_score(lr_base, X_baseline, y, cv=kf, scoring="r2")
        lr_base.fit(X_baseline, y)
        y_pred_base = lr_base.predict(X_baseline)
        mape_base = mean_absolute_percentage_error(y, y_pred_base)
        results["LinearReg (bs only)"] = {
            "r2_cv": scores_base.mean(), "r2_cv_std": scores_base.std(),
            "r2_train": r2_score(y, y_pred_base), "mape": mape_base,
        }
        print(f"\n  LinearReg (bs only): R2_cv={scores_base.mean():.4f}±{scores_base.std():.4f}, "
              f"MAPE={mape_base:.4f}")

    # Model 2: Linear Regression (all features)
    lr_full = LinearRegression()
    scores_full = cross_val_score(lr_full, X, y, cv=kf, scoring="r2")
    lr_full.fit(X, y)
    y_pred_full = lr_full.predict(X)
    mape_full = mean_absolute_percentage_error(y, y_pred_full)
    results["LinearReg (all)"] = {
        "r2_cv": scores_full.mean(), "r2_cv_std": scores_full.std(),
        "r2_train": r2_score(y, y_pred_full), "mape": mape_full,
    }
    print(f"  LinearReg (all):    R2_cv={scores_full.mean():.4f}±{scores_full.std():.4f}, "
          f"MAPE={mape_full:.4f}")
    print(f"    Coefficients:")
    for name, coef in zip(feature_names, lr_full.coef_):
        print(f"      {name:25s}: {coef:+.6f}")

    # Model 3: Random Forest (all features)
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    scores_rf = cross_val_score(rf, X, y, cv=kf, scoring="r2")
    rf.fit(X, y)
    y_pred_rf = rf.predict(X)
    mape_rf = mean_absolute_percentage_error(y, y_pred_rf)
    results["RandomForest (all)"] = {
        "r2_cv": scores_rf.mean(), "r2_cv_std": scores_rf.std(),
        "r2_train": r2_score(y, y_pred_rf), "mape": mape_rf,
    }
    print(f"  RandomForest (all): R2_cv={scores_rf.mean():.4f}±{scores_rf.std():.4f}, "
          f"MAPE={mape_rf:.4f}")

    # Model 4: GBDT (all features)
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                                    random_state=42)
    scores_gb = cross_val_score(gb, X, y, cv=kf, scoring="r2")
    gb.fit(X, y)
    y_pred_gb = gb.predict(X)
    mape_gb = mean_absolute_percentage_error(y, y_pred_gb)
    results["GBDT (all)"] = {
        "r2_cv": scores_gb.mean(), "r2_cv_std": scores_gb.std(),
        "r2_train": r2_score(y, y_pred_gb), "mape": mape_gb,
    }
    print(f"  GBDT (all):         R2_cv={scores_gb.mean():.4f}±{scores_gb.std():.4f}, "
          f"MAPE={mape_gb:.4f}")

    # Model 5: GBDT (only expert features, no batch_size)
    if expert_feature_indices:
        X_expert_only = X[:, expert_feature_indices]
        expert_names = [feature_names[i] for i in expert_feature_indices]
        gb_expert = GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                              learning_rate=0.1, random_state=42)
        scores_gbe = cross_val_score(gb_expert, X_expert_only, y, cv=kf, scoring="r2")
        gb_expert.fit(X_expert_only, y)
        y_pred_gbe = gb_expert.predict(X_expert_only)
        mape_gbe = mean_absolute_percentage_error(y, y_pred_gbe)
        results["GBDT (expert only)"] = {
            "r2_cv": scores_gbe.mean(), "r2_cv_std": scores_gbe.std(),
            "r2_train": r2_score(y, y_pred_gbe), "mape": mape_gbe,
        }
        print(f"  GBDT (expert only): R2_cv={scores_gbe.mean():.4f}±{scores_gbe.std():.4f}, "
              f"MAPE={mape_gbe:.4f}")

    # Feature importance
    print(f"\n  Feature Importance (GBDT):")
    importances = gb.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for idx in sorted_idx:
        print(f"    {feature_names[idx]:25s}: {importances[idx]:.4f}")

    return results, gb, rf, lr_full, importances


def plot_analysis(df_clean, X, y, feature_names, importances, gb):
    """Generate analysis plots."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("MoE Expert Load Features vs Per-Token Latency\n(Qwen3-30B-A3B, TP=1)",
                 fontsize=13, fontweight="bold")

    # 1. Scatter: batch_size vs tpot
    ax = axes[0, 0]
    if "batch_size" in df_clean.columns:
        bs_vals = df_clean["batch_size"].values
        colors = plt.cm.viridis(np.log2(bs_vals + 1) / 7)
        ax.scatter(bs_vals, y, c=df_clean["els"].values if "els" in df_clean.columns else "blue",
                   cmap="RdYlGn_r", alpha=0.7, s=40, edgecolors="black", linewidth=0.3)
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("TPOT (ms)")
        ax.set_title("Batch Size vs TPOT\n(color = ELS)")
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label("ELS")
    ax.grid(alpha=0.3)

    # 2. Scatter: ELS vs tpot (fixed bs subsets)
    ax = axes[0, 1]
    if "els" in df_clean.columns:
        for bs_val in sorted(df_clean["batch_size"].unique()):
            mask = df_clean["batch_size"] == bs_val
            ax.scatter(df_clean.loc[mask, "els"], y[mask],
                       label=f"bs={bs_val}", alpha=0.7, s=30)
        ax.set_xlabel("ELS (Expert Load Skew)")
        ax.set_ylabel("TPOT (ms)")
        ax.set_title("ELS vs TPOT (by batch size)")
        ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    # 3. Scatter: gini vs tpot
    ax = axes[0, 2]
    if "gini" in df_clean.columns:
        for bs_val in sorted(df_clean["batch_size"].unique()):
            mask = df_clean["batch_size"] == bs_val
            ax.scatter(df_clean.loc[mask, "gini"], y[mask],
                       label=f"bs={bs_val}", alpha=0.7, s=30)
        ax.set_xlabel("Gini Coefficient")
        ax.set_ylabel("TPOT (ms)")
        ax.set_title("Gini vs TPOT (by batch size)")
        ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    # 4. Feature importance bar chart
    ax = axes[1, 0]
    sorted_idx = np.argsort(importances)[::-1]
    bars = ax.barh([feature_names[i] for i in sorted_idx], importances[sorted_idx],
                   color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Feature Importance (GBDT)")
    ax.set_title("Feature Importance")
    ax.grid(axis="x", alpha=0.3)

    # 5. Correlation heatmap
    ax = axes[1, 1]
    available_feats = [c for c in feature_names if c in df_clean.columns]
    corr_cols = available_feats + ["tpot_ms_avg"]
    corr_data = df_clean[[c for c in corr_cols if c in df_clean.columns]].astype(float)
    corr_matrix = corr_data.corr()
    im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(corr_matrix.columns, fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Feature Correlation Matrix")

    # 6. Predicted vs Actual
    ax = axes[1, 2]
    y_pred = gb.predict(X)
    ax.scatter(y, y_pred, alpha=0.6, s=30, edgecolors="black", linewidth=0.3)
    lims = [min(y.min(), y_pred.min()) - 2, max(y.max(), y_pred.max()) + 2]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect")
    ax.set_xlabel("Actual TPOT (ms)")
    ax.set_ylabel("Predicted TPOT (ms)")
    ax.set_title(f"GBDT: Predicted vs Actual\nR²={r2_score(y, y_pred):.4f}")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_path = OUT_DIR / "expert_load_vs_latency_analysis.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved plot: {save_path}")


def plot_per_bs_analysis(df_clean):
    """Additional plot: within-batch-size variation."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Within-Batch-Size Latency Variation vs Expert Features",
                 fontsize=13, fontweight="bold")

    batch_sizes = sorted(df_clean["batch_size"].unique())
    expert_features = ["els", "gini", "max_tokens_per_expert",
                       "std_tokens_per_expert", "num_active_experts", "top1_share"]

    for i, feat in enumerate(expert_features):
        ax = axes[i // 3, i % 3]
        if feat not in df_clean.columns:
            ax.set_visible(False)
            continue

        for bs_val in batch_sizes:
            mask = df_clean["batch_size"] == bs_val
            subset = df_clean[mask]
            if len(subset) < 3:
                continue
            x_vals = subset[feat].values.astype(float)
            y_vals = subset["tpot_ms_avg"].values.astype(float)
            corr = np.corrcoef(x_vals, y_vals)[0, 1] if len(x_vals) > 2 else 0
            ax.scatter(x_vals, y_vals, label=f"bs={bs_val} (r={corr:.2f})",
                       alpha=0.7, s=25)

        ax.set_xlabel(feat)
        ax.set_ylabel("TPOT (ms)")
        ax.set_title(f"{feat} vs TPOT")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_path = OUT_DIR / "expert_load_within_bs_variation.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved plot: {save_path}")


def compute_improvement_metrics(results):
    """Compute the improvement from adding expert features."""
    print("\n" + "=" * 70)
    print("  IMPROVEMENT SUMMARY")
    print("=" * 70)

    base_r2 = results.get("LinearReg (bs only)", {}).get("r2_cv", 0)
    base_mape = results.get("LinearReg (bs only)", {}).get("mape", 999)

    for name, res in results.items():
        if name == "LinearReg (bs only)":
            continue
        r2_improvement = res["r2_cv"] - base_r2
        mape_improvement = base_mape - res["mape"]
        print(f"  {name:25s}: R2_cv improvement = {r2_improvement:+.4f}, "
              f"MAPE improvement = {mape_improvement:+.4f}")

    expert_only_r2 = results.get("GBDT (expert only)", {}).get("r2_cv", 0)
    print(f"\n  Expert features alone explain R²={expert_only_r2:.4f} of latency variance")
    print(f"  Adding expert features to bs improves R² by "
          f"{results.get('GBDT (all)', {}).get('r2_cv', 0) - base_r2:+.4f}")


def main():
    print("MoE Expert Load vs Latency — Regression Analysis")
    print("=" * 70)

    df = load_data()
    X, y, feature_names, df_clean = prepare_features(df)
    print(f"\n  Features: {feature_names}")
    print(f"  Samples: {len(y)}")
    print(f"  Target (tpot_ms_avg): mean={y.mean():.2f}, std={y.std():.2f}, "
          f"min={y.min():.2f}, max={y.max():.2f}")

    results, gb, rf, lr, importances = run_regression_analysis(X, y, feature_names)
    plot_analysis(df_clean, X, y, feature_names, importances, gb)
    plot_per_bs_analysis(df_clean)
    compute_improvement_metrics(results)

    # Save results summary
    summary_path = OUT_DIR / "regression_summary.txt"
    with open(summary_path, "w") as f:
        f.write("MoE Expert Load vs Latency — Regression Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Data: {len(y)} samples\n")
        f.write(f"Target: tpot_ms_avg (mean={y.mean():.2f}, std={y.std():.2f})\n\n")
        f.write("Models:\n")
        for name, res in results.items():
            f.write(f"  {name:25s}: R2_cv={res['r2_cv']:.4f}±{res['r2_cv_std']:.4f}, "
                    f"MAPE={res['mape']:.4f}\n")
        f.write(f"\nFeature Importance (GBDT):\n")
        sorted_idx = np.argsort(importances)[::-1]
        for idx in sorted_idx:
            f.write(f"  {feature_names[idx]:25s}: {importances[idx]:.4f}\n")
    print(f"\n  Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
