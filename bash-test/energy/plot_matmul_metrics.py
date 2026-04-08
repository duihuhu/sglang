#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from PIL import Image


def _read_rows(csv_path: str):
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"gpu_clock_mhz", "n", "latency_us", "power_w", "energy_uj"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"CSV 缺少列: {sorted(missing)}")
        rows = []
        for r in reader:
            clk_raw = (r.get("gpu_clock_mhz") or "").strip()
            if not clk_raw or not clk_raw.isdigit():
                continue
            rows.append(
                {
                    "gpu_clock_mhz": int(clk_raw),
                    "n": int((r.get("n") or "0").strip()),
                    "latency_us": float((r.get("latency_us") or "0").strip()),
                    "power_w": float((r.get("power_w") or "0").strip()),
                    "energy_uj": float((r.get("energy_uj") or "0").strip()),
                }
            )
    return rows


def _group_by_n(rows: List[dict], metric: str) -> Dict[int, List[Tuple[int, float]]]:
    by_n: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for r in rows:
        by_n[r["n"]].append((r["gpu_clock_mhz"], r[metric]))
    for n in by_n:
        by_n[n].sort(key=lambda x: x[0])
    return dict(sorted(by_n.items(), key=lambda kv: kv[0]))


def _plot_metric(rows: List[dict], metric: str, ylabel: str, out_path: str, yscale: str) -> None:
    by_n = _group_by_n(rows, metric)
    plt.figure(figsize=(8, 5))
    for n, series in by_n.items():
        xs = [x for x, _ in series]
        ys = [y for _, y in series]
        plt.plot(xs, ys, marker="o", label=f"n={n}")
    plt.xlabel("gpu_clock_mhz")
    plt.ylabel(ylabel)
    plt.title(f"{metric} vs gpu_clock_mhz")
    plt.yscale(yscale)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _stack_vertical(image_paths: List[str], out_path: str) -> None:
    imgs = [Image.open(p).convert("RGB") for p in image_paths]
    max_w = max(im.width for im in imgs)
    resized = []
    for im in imgs:
        if im.width != max_w:
            h = int(im.height * (max_w / im.width))
            im = im.resize((max_w, h), Image.Resampling.LANCZOS)
        resized.append(im)
    total_h = sum(im.height for im in resized)
    out = Image.new("RGB", (max_w, total_h), "white")
    y = 0
    for im in resized:
        out.paste(im, (0, y))
        y += im.height
    out.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="bash-test/energy/matmul_power_results.csv")
    parser.add_argument("--out-dir", type=str, default="bash-test/energy/matmul_plots")
    parser.add_argument("--log-y-metrics", type=str, default="latency_us,energy_uj")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    rows = _read_rows(os.path.abspath(args.csv))
    log_set = {x.strip() for x in args.log_y_metrics.split(",") if x.strip()}

    _plot_metric(
        rows,
        "latency_us",
        "latency_us",
        os.path.join(out_dir, "latency_vs_clock.png"),
        "log" if "latency_us" in log_set else "linear",
    )
    _plot_metric(
        rows,
        "power_w",
        "power_w",
        os.path.join(out_dir, "power_vs_clock.png"),
        "log" if "power_w" in log_set else "linear",
    )
    _plot_metric(
        rows,
        "energy_uj",
        "energy_uj",
        os.path.join(out_dir, "energy_vs_clock.png"),
        "log" if "energy_uj" in log_set else "linear",
    )
    _stack_vertical(
        [
            os.path.join(out_dir, "latency_vs_clock.png"),
            os.path.join(out_dir, "power_vs_clock.png"),
            os.path.join(out_dir, "energy_vs_clock.png"),
        ],
        os.path.join(out_dir, "matmul_metrics_stacked_vertical.png"),
    )

    print("[saved]")
    print(os.path.join(out_dir, "latency_vs_clock.png"))
    print(os.path.join(out_dir, "power_vs_clock.png"))
    print(os.path.join(out_dir, "energy_vs_clock.png"))
    print(os.path.join(out_dir, "matmul_metrics_stacked_vertical.png"))


if __name__ == "__main__":
    main()
