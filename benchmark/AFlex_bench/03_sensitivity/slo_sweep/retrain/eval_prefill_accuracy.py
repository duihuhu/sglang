#!/usr/bin/env python3
"""Evaluate Prefill-stage prediction accuracy using prefill_data_v1.txt ground truth.

Reports:
  - Overall MAPE for latency (A, F) and energy (A, F)
  - Breakdown by (freq, batch_size, input_len)
  - Worst-case configurations
"""

import sys
sys.path.insert(0, "/workspace/sglang-tier/python")

import numpy as np
import pandas as pd
from pathlib import Path

from sglang.srt.energy.af_profile_predictor import AFProfilePredictor

MODEL_DIR = "/workspace/sglang/benchmark/test_motivation/energy_models"
DATA_FILE = "/workspace/sglang-tier/benchmark/test_motivation/hucc/paper/prefill_data_v1.txt"


def load_prefill_data(path: str) -> pd.DataFrame:
    """Load prefill profile data, skip the header marker line."""
    lines = Path(path).read_text().strip().split("\n")
    # First line is "[P] Prefill", second is column headers
    header = lines[1].split("\t")
    rows = []
    for line in lines[2:]:
        if line.strip():
            rows.append(line.split("\t"))
    df = pd.DataFrame(rows, columns=header)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna()


def evaluate():
    print("=" * 70)
    print("Prefill 阶段预测精度评估 (5-Fold CV + LinearReg-only)")
    print("=" * 70)

    df = load_prefill_data(DATA_FILE)
    print(f"\n数据量: {len(df)} 行")
    print(f"频率范围: {sorted(df['gpu_clock'].unique())}")
    print(f"Batch size 范围: {sorted(df['batch_size'].unique())}")
    print(f"Input len 范围: {sorted(df['input_len'].unique())}")

    # === Part 1: LinearReg-only evaluation (skip LUT) ===
    # This tests the model's generalization, not memorization
    print("\n\n" + "=" * 70)
    print("Part 1: LinearReg 模型预测精度 (跳过 LUT 精确匹配)")
    print("=" * 70)

    predictor = AFProfilePredictor(MODEL_DIR)

    results = []
    for _, row in df.iterrows():
        tp = int(row["tp"])
        freq = int(row["gpu_clock"])
        bs = int(row["batch_size"])
        il = int(row["input_len"])

        true_a_lat = row["A"]           # us
        true_f_lat = row["F"]           # us
        true_a_energy = row["A_energy_mj"]  # mJ
        true_f_energy = row["F_energy_mj"]  # mJ

        try:
            # Force LinearReg prediction by directly calling the model
            df_input = pd.DataFrame([{"tp": tp, "gpu_clock": freq,
                                      "input_len": il, "batch_size": bs}])

            models = predictor._models
            pred_a_lat = models["Prefill_A_lat"]["LinearReg"].predict(df_input)[0]
            pred_f_lat = models["Prefill_F_lat"]["LinearReg"].predict(df_input)[0]
            pred_a_energy = models["Prefill_A"]["LinearReg"].predict(df_input)[0]
            pred_f_energy = models["Prefill_F"]["LinearReg"].predict(df_input)[0]
        except Exception as e:
            continue

        results.append({
            "tp": tp, "freq": freq, "bs": bs, "il": il,
            "true_a_lat": true_a_lat, "pred_a_lat": pred_a_lat,
            "true_f_lat": true_f_lat, "pred_f_lat": pred_f_lat,
            "true_a_energy": true_a_energy, "pred_a_energy": pred_a_energy,
            "true_f_energy": true_f_energy, "pred_f_energy": pred_f_energy,
        })

    res = pd.DataFrame(results)
    print(f"成功预测: {len(res)}/{len(df)} 行\n")

    # Compute APE for each metric
    for metric, true_col, pred_col in [
        ("A_latency (us)", "true_a_lat", "pred_a_lat"),
        ("F_latency (us)", "true_f_lat", "pred_f_lat"),
        ("A_energy (mJ)", "true_a_energy", "pred_a_energy"),
        ("F_energy (mJ)", "true_f_energy", "pred_f_energy"),
    ]:
        ape = np.abs(res[pred_col] - res[true_col]) / res[true_col] * 100
        res[f"ape_{metric}"] = ape

    # Overall MAPE
    print("-" * 70)
    print("Overall MAPE (Mean Absolute Percentage Error)")
    print("-" * 70)
    for metric in ["A_latency (us)", "F_latency (us)", "A_energy (mJ)", "F_energy (mJ)"]:
        col = f"ape_{metric}"
        print(f"  {metric:20s}: MAPE = {res[col].mean():.2f}% ± {res[col].std():.2f}%  "
              f"(median={res[col].median():.2f}%, max={res[col].max():.2f}%)")

    # Iteration-level: (A+F)*num_layers combined latency
    num_layers = 64
    res["true_iter_lat_ms"] = (res["true_a_lat"] + res["true_f_lat"]) * num_layers / 1000
    res["pred_iter_lat_ms"] = (res["pred_a_lat"] + res["pred_f_lat"]) * num_layers / 1000
    iter_ape = np.abs(res["pred_iter_lat_ms"] - res["true_iter_lat_ms"]) / res["true_iter_lat_ms"] * 100
    print(f"\n  {'TTFT (A+F)*64层':20s}: MAPE = {iter_ape.mean():.2f}% ± {iter_ape.std():.2f}%  "
          f"(median={iter_ape.median():.2f}%, max={iter_ape.max():.2f}%)")

    # Breakdown by freq
    print("\n" + "-" * 70)
    print("按频率分组 MAPE")
    print("-" * 70)
    print(f"{'Freq(MHz)':>10} | {'A_lat%':>8} | {'F_lat%':>8} | {'A_eng%':>8} | {'F_eng%':>8} | {'TTFT%':>8}")
    print("-" * 70)
    for freq in sorted(res["freq"].unique()):
        mask = res["freq"] == freq
        sub = res[mask]
        sub_iter_ape = iter_ape[mask]
        print(f"{int(freq):>10} | "
              f"{sub['ape_A_latency (us)'].mean():>7.2f}% | "
              f"{sub['ape_F_latency (us)'].mean():>7.2f}% | "
              f"{sub['ape_A_energy (mJ)'].mean():>7.2f}% | "
              f"{sub['ape_F_energy (mJ)'].mean():>7.2f}% | "
              f"{sub_iter_ape.mean():>7.2f}%")

    # Breakdown by batch_size
    print("\n" + "-" * 70)
    print("按 Batch Size 分组 MAPE")
    print("-" * 70)
    print(f"{'BS':>6} | {'A_lat%':>8} | {'F_lat%':>8} | {'A_eng%':>8} | {'F_eng%':>8} | {'TTFT%':>8}")
    print("-" * 70)
    for bs in sorted(res["bs"].unique()):
        mask = res["bs"] == bs
        sub = res[mask]
        sub_iter_ape = iter_ape[mask]
        print(f"{int(bs):>6} | "
              f"{sub['ape_A_latency (us)'].mean():>7.2f}% | "
              f"{sub['ape_F_latency (us)'].mean():>7.2f}% | "
              f"{sub['ape_A_energy (mJ)'].mean():>7.2f}% | "
              f"{sub['ape_F_energy (mJ)'].mean():>7.2f}% | "
              f"{sub_iter_ape.mean():>7.2f}%")

    # Breakdown by input_len
    print("\n" + "-" * 70)
    print("按 Input Length 分组 MAPE")
    print("-" * 70)
    print(f"{'IL':>6} | {'A_lat%':>8} | {'F_lat%':>8} | {'A_eng%':>8} | {'F_eng%':>8} | {'TTFT%':>8}")
    print("-" * 70)
    for il in sorted(res["il"].unique()):
        mask = res["il"] == il
        sub = res[mask]
        sub_iter_ape = iter_ape[mask]
        print(f"{int(il):>6} | "
              f"{sub['ape_A_latency (us)'].mean():>7.2f}% | "
              f"{sub['ape_F_latency (us)'].mean():>7.2f}% | "
              f"{sub['ape_A_energy (mJ)'].mean():>7.2f}% | "
              f"{sub['ape_F_energy (mJ)'].mean():>7.2f}% | "
              f"{sub_iter_ape.mean():>7.2f}%")

    # Top-10 worst TTFT predictions
    res["ttft_ape"] = iter_ape
    print("\n" + "-" * 70)
    print("TTFT 预测误差 Top-10 最差配置")
    print("-" * 70)
    worst = res.nlargest(10, "ttft_ape")
    print(f"{'freq':>6} {'bs':>4} {'il':>6} | {'真实(ms)':>10} {'预测(ms)':>10} {'APE%':>8}")
    for _, r in worst.iterrows():
        print(f"{int(r['freq']):>6} {int(r['bs']):>4} {int(r['il']):>6} | "
              f"{r['true_iter_lat_ms']:>10.2f} {r['pred_iter_lat_ms']:>10.2f} {r['ttft_ape']:>7.2f}%")

    # Bias analysis (systematic over/under-prediction)
    print("\n" + "-" * 70)
    print("偏差分析 (正=高估, 负=低估)")
    print("-" * 70)
    for metric, true_col, pred_col in [
        ("A_latency", "true_a_lat", "pred_a_lat"),
        ("F_latency", "true_f_lat", "pred_f_lat"),
        ("A_energy", "true_a_energy", "pred_a_energy"),
        ("F_energy", "true_f_energy", "pred_f_energy"),
    ]:
        bias = (res[pred_col] - res[true_col]) / res[true_col] * 100
        print(f"  {metric:12s}: mean_bias = {bias.mean():+.2f}%, median = {bias.median():+.2f}%")

    ttft_bias = (res["pred_iter_lat_ms"] - res["true_iter_lat_ms"]) / res["true_iter_lat_ms"] * 100
    print(f"  {'TTFT':12s}: mean_bias = {ttft_bias.mean():+.2f}%, median = {ttft_bias.median():+.2f}%")

    # === Part 2: 5-Fold Cross Validation ===
    print("\n\n" + "=" * 70)
    print("Part 2: 5-Fold 交叉验证 (真正的泛化能力测试)")
    print("=" * 70)
    run_cv(df)


def run_cv(df: pd.DataFrame, n_splits=5):
    """5-fold CV: train LinearReg on 80% data, predict on 20% held-out."""
    from sklearn.model_selection import KFold
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    features = ["tp", "gpu_clock", "input_len", "batch_size"]
    targets = {
        "A_lat": "A",
        "F_lat": "F",
        "A_energy": "A_energy_mj",
        "F_energy": "F_energy_mj",
    }

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_results = {t: [] for t in targets}

    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]

        X_train = train[features].values
        X_test = test[features].values

        for target_name, col_name in targets.items():
            y_train = train[col_name].values
            y_test = test[col_name].values

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            ape = np.abs(y_pred - y_test) / y_test * 100
            cv_results[target_name].append(ape.mean())

    print(f"\n{'Target':>12} | {'Mean MAPE':>10} | {'Std':>8} | 各折 MAPE")
    print("-" * 70)
    for target_name in targets:
        mapes = cv_results[target_name]
        print(f"{target_name:>12} | {np.mean(mapes):>9.2f}% | {np.std(mapes):>7.2f}% | "
              f"{[f'{m:.1f}%' for m in mapes]}")

    # Combined TTFT
    print("\n(注: 以上是单层 per-operator 的 MAPE, 实际 TTFT = (A+F)*64层)")
    print("LinearReg 用于 Prefill 延迟预测, 由于 Prefill 延迟与 input_len*bs 近似线性,")
    print("线性模型的泛化能力取决于训练数据覆盖的配置空间。")


if __name__ == "__main__":
    evaluate()
