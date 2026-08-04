#!/usr/bin/env python3
"""Search profile-predicted M=2 token partitions for a homogeneous A/F pipeline."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

AFLEX_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PROFILE = (
    AFLEX_ROOT / "energy_model/Qwen3-32B/data/v1_layer_profile/decode_data_v1.txt"
)


def load_curve(path: Path, tp: int, input_len: int, output_len: int, clock: int):
    curve = {}
    with path.open() as f:
        next(f)
        for row in csv.DictReader(f, delimiter="\t"):
            row = {k.strip(): v for k, v in row.items()}
            if (
                int(row["tp"]),
                int(row["input_len"]),
                int(row["output_len"]),
                int(row["gpu_clock"]),
            ) != (tp, input_len, output_len, clock):
                continue
            curve[int(row["batch_size"])] = (
                float(row["A"]) / 1000,
                float(row["F"]) / 1000,
            )
    if not curve:
        raise ValueError("No matching profile rows")
    return curve


def predict(curve, batch: int, stage: int) -> float:
    xs = sorted(curve)
    if batch in curve:
        return curve[batch][stage]
    if batch < xs[0]:
        x0, x1 = xs[:2]
    elif batch > xs[-1]:
        x0, x1 = xs[-2:]
    else:
        x0, x1 = next((a, b) for a, b in zip(xs, xs[1:]) if a < batch < b)
    y0, y1 = curve[x0][stage], curve[x1][stage]
    return y0 + (batch - x0) * (y1 - y0) / (x1 - x0)


def evaluate(curve, parts, layers):
    a = [predict(curve, b, 0) for b in parts]
    f = [predict(curve, b, 1) for b in parts]
    fill = sum(a)
    cycle = max(sum(a), sum(f))
    drain = sum(f)
    forward = fill + (layers - 1) * cycle + drain
    return {
        "parts": parts,
        "a": a,
        "f": f,
        "fill": fill,
        "cycle": cycle,
        "drain": drain,
        "forward": forward,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    ap.add_argument("--total", type=int, default=512)
    ap.add_argument("--layers", type=int, default=64)
    ap.add_argument("--min-part", type=int, default=128)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--input-len", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=64)
    ap.add_argument("--clock", type=int, default=210)
    ap.add_argument(
        "--anchor-256-a",
        type=float,
        default=2.366,
        help="Measured A(256) ms from M=2 breakdown",
    )
    ap.add_argument(
        "--anchor-256-f",
        type=float,
        default=5.349,
        help="Measured F(256) ms from M=2 breakdown",
    )
    args = ap.parse_args()
    curve = load_curve(args.profile, args.tp, args.input_len, args.output_len, args.clock)
    curve[256] = (args.anchor_256_a, args.anchor_256_f)
    rows = [
        evaluate(curve, (b, args.total - b), args.layers)
        for b in range(args.min_part, args.total - args.min_part + 1)
    ]
    rows.sort(key=lambda x: (x["forward"], x["parts"]))
    best_value = rows[0]["forward"]
    tied = [r for r in rows if abs(r["forward"] - best_value) <= 1e-6]
    best = next((r for r in tied if r["parts"][0] == args.total // 2), tied[0])
    equal = evaluate(curve, (args.total // 2, args.total - args.total // 2), args.layers)
    print(f"measured batch range: {min(curve)}..{max(curve)}")
    for name, r in (("best", best), ("equal", equal)):
        print(
            f"{name}: split={r['parts'][0]}+{r['parts'][1]} "
            f"A={r['a'][0]:.4f}+{r['a'][1]:.4f}ms "
            f"F={r['f'][0]:.4f}+{r['f'][1]:.4f}ms "
            f"fill={r['fill']:.4f}ms cycle={r['cycle']:.4f}ms "
            f"drain={r['drain']:.4f}ms forward={r['forward']:.4f}ms"
        )
    print(f"predicted gain vs equal: {(equal['forward'] / best['forward'] - 1) * 100:.4f}%")
    print(f"number of tied optima (1e-6 ms tolerance): {len(tied)} / {len(rows)}")
    if len(tied) == len(rows):
        print(
            "result: no unique optimum; the affine latency model makes all fixed-total partitions equivalent"
        )
    print("top-10:")
    for r in rows[:10]:
        print(f"  {r['parts'][0]}+{r['parts'][1]}: {r['forward']:.4f} ms")


if __name__ == "__main__":
    main()
