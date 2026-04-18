#!/usr/bin/env python3
"""
D_data 的 SLO 能耗收益分析（按你的“中间那版”选择策略恢复）。

核心点：
1) 不固定 SLO 倍数；仍用候选 m 集合自动搜索。
2) 对每个 tp 先找 m_peak(tp)：使该 tp 的“均值能耗降幅(%)”在 m 维度上达到最大（或第 N 大，见 --tp-peak-rank）。
3) 只做碰撞消解：如果不同 tp 的 peak 在 round(m,2) 后落到同一个倍数（例如都变成 1.01），则把冲突的 tp 往后挪到它的下一个候选 peak，直到冲突消失或无法再挪。
4) 最终 six 档来自：{m_min=1.0, m_pct_stable(稳定平台起点), 各 tp 的 peak_m}，不足 6 档用最大间隙中点补齐。

输出：
  figures/slo_energy/
    - slo_energy_mean_savings_pct.csv
    - slo_energy_mean_savings_heatmap.png
    - slo_energy_multipliers.txt
"""

from __future__ import annotations

import argparse
import bisect
import csv
import math
import os
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


CSV_PATH = os.path.join(os.path.dirname(__file__), "D_data.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "figures", "slo_energy")

INPUT_LEN_WEIGHTS: dict[int, float] = {
    128: 0.30,
    512: 0.30,
    1024: 0.20,
    2048: 0.10,
    4096: 0.07,
    8192: 0.03,
}


def load_rows(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="") as f:
        r = csv.reader(f)
        next(r)  # [D] Decode
        header = next(r)
        idx = {name: i for i, name in enumerate(header)}
        for line in r:
            if len(line) < len(header):
                continue
            rows.append(
                {
                    "tp": int(line[idx["tp"]]),
                    "input_len": int(line[idx["input_len"]]),
                    "output_len": int(line[idx["output_len"]]),
                    "gpu_clock": int(line[idx["gpu_clock"]]),
                    "batch_size": int(line[idx["batch_size"]]),
                    "A": float(line[idx["A"]]),
                    "F": float(line[idx["F"]]),
                    "A_energy_mj": float(line[idx["A_energy_mj"]]),
                    "F_energy_mj": float(line[idx["F_energy_mj"]]),
                }
            )
    return rows


def group_by_key(rows: list[dict]) -> dict:
    """(tp, input_len, output_len, batch_size) -> gpu_clock -> record"""
    g: dict = defaultdict(dict)
    for rec in rows:
        k = (rec["tp"], rec["input_len"], rec["output_len"], rec["batch_size"])
        g[k][rec["gpu_clock"]] = rec
    return g


def _build_frontier(pairs: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    """
    给定若干 (lat, energy) 对，构建：
      - lats：按 lat 升序
      - prefix_best_energy：在 lats[i] 作为阈值时的最小能耗
    """
    pairs_sorted = sorted((float(lat), float(en)) for lat, en in pairs)
    lats: list[float] = []
    prefix_best: list[float] = []
    best = math.inf
    for lat, en in pairs_sorted:
        lats.append(lat)
        best = min(best, en)
        prefix_best.append(best)
    return np.asarray(lats, dtype=float), np.asarray(prefix_best, dtype=float)


def preprocess_grouped(
    grouped: dict,
    *,
    min_clocks: int = 6,
    log_every: int = 200,
) -> tuple[dict, dict]:
    """
    预计算每个 key 的：
      - ref = min_aflex_lat
      - max_lat = max_aflex_lat
      - optb frontier（同频：每个 clock 一条 A+F）
      - aflex frontier（独立选频：所有 ca/cf 组合）
      - critical_multipliers：来自 aflex 所有 lat/ref（用于构建候选 m）

    返回：
      prepared_by_key: key -> prepared
      keys_by_tp: tp -> [key, ...]
    """
    prepared_by_key: dict = {}
    keys_by_tp: dict = defaultdict(list)

    total = len(grouped)
    t0 = time.perf_counter()
    scanned = 0

    for i, (k, by_c) in enumerate(grouped.items(), start=1):
        scanned += 1
        if len(by_c) < min_clocks:
            if i % max(1, log_every) == 0 or i == total:
                print(f"[preprocess] scanned={i}/{total}, prepared={len(prepared_by_key)}", flush=True)
            continue

        clocks = sorted(by_c.keys())

        optb_pairs: list[tuple[float, float]] = []
        aflex_pairs: list[tuple[float, float]] = []

        ref = math.inf
        hi = -math.inf
        for c in clocks:
            rec = by_c[c]
            lat = rec["A"] + rec["F"]
            en = rec["A_energy_mj"] + rec["F_energy_mj"]
            optb_pairs.append((lat, en))

        # aflex：所有 A_clock + F_clock 组合
        for ca in clocks:
            ra = by_c[ca]
            for cf in clocks:
                rb = by_c[cf]
                lat = ra["A"] + rb["F"]
                en = ra["A_energy_mj"] + rb["F_energy_mj"]
                aflex_pairs.append((lat, en))
                ref = min(ref, lat)
                hi = max(hi, lat)

        if not math.isfinite(ref) or ref <= 0:
            if i % max(1, log_every) == 0 or i == total:
                print(f"[preprocess] scanned={i}/{total}, prepared={len(prepared_by_key)}", flush=True)
            continue

        optb_lats, optb_energy_prefix = _build_frontier(optb_pairs)
        aflex_lats, aflex_energy_prefix = _build_frontier(aflex_pairs)

        ratios = {1.0}
        # critical candidates 来自 aflex 可行边界：m = L / SLO_ref
        for lat, _ in aflex_pairs:
            m = lat / ref
            if math.isfinite(m) and m >= 1.0 - 1e-15:
                ratios.add(float(m))

        prepared_by_key[k] = {
            "ref": float(ref),
            "max_lat": float(hi),
            "critical_multipliers": sorted(ratios),
            "optb_lats": optb_lats,
            "optb_best_energy_prefix": optb_energy_prefix,
            "aflex_lats": aflex_lats,
            "aflex_best_energy_prefix": aflex_energy_prefix,
        }
        keys_by_tp[k[0]].append(k)

        if i % max(1, log_every) == 0 or i == total:
            print(f"[preprocess] scanned={i}/{total}, prepared={len(prepared_by_key)}", flush=True)

    dt = time.perf_counter() - t0
    print(f"[preprocess] done: prepared={len(prepared_by_key)}/{total} in {dt:.2f}s", flush=True)
    return prepared_by_key, keys_by_tp


def _best_energy_under_slo(lats: np.ndarray, prefix_best: np.ndarray, slo: float) -> float | None:
    # 找最后一个 lats[idx] <= slo（略放宽误差，防止边界误差）
    idx = bisect.bisect_right(lats, float(slo) + 1e-12) - 1
    if idx < 0:
        return None
    return float(prefix_best[idx])


def pct_savings(optb_e: float, aflex_e: float) -> float:
    if optb_e <= 0 or not math.isfinite(optb_e):
        return float("nan")
    return (optb_e - aflex_e) / optb_e * 100.0


def weighted_mean_over_marked_input_lens(pct_by_ilen: dict[int, float]) -> float | None:
    avail = {il for il in INPUT_LEN_WEIGHTS if il in pct_by_ilen and math.isfinite(pct_by_ilen[il])}
    if not avail:
        return None
    wsum = sum(INPUT_LEN_WEIGHTS[il] for il in avail)
    return float(sum(pct_by_ilen[il] * (INPUT_LEN_WEIGHTS[il] / wsum) for il in avail))


def union_critical_multipliers(prepared_by_key: dict, *, m_round_digits: int = 5) -> list[float]:
    bag: set[float] = set()
    q = max(2, min(8, int(m_round_digits)))
    for prepared in prepared_by_key.values():
        for m in prepared["critical_multipliers"]:
            bag.add(round(float(m), q))
    if not bag:
        return [1.0]
    arr = sorted(bag)
    merged: list[float] = []
    for x in arr:
        if not merged or x > merged[-1] + 10 ** (-q) * max(1.0, merged[-1]):
            merged.append(x)
    return merged


def aggregate_savings_pct_for_tp_at_m(
    prepared_by_key: dict,
    keys_by_tp: dict,
    *,
    tp: int,
    mult: float,
    min_clocks: int = 6,
) -> tuple[float, int, int]:
    """
    返回 (mean_pct, valid_count_of_batchsize, skips_optb_infeasible)
    """
    by_ob: dict[tuple[int, int], dict[int, float]] = defaultdict(dict)  # (olen,bs) -> {ilen: pct}
    skips = 0

    for k in keys_by_tp.get(tp, []):
        ilen, olen, bs = k[1], k[2], k[3]
        if ilen not in INPUT_LEN_WEIGHTS:
            continue
        prepared = prepared_by_key.get(k)
        if prepared is None:
            continue

        slo = prepared["ref"] * mult
        optb = _best_energy_under_slo(prepared["optb_lats"], prepared["optb_best_energy_prefix"], slo)
        if optb is None:
            skips += 1
            continue
        aflex = _best_energy_under_slo(prepared["aflex_lats"], prepared["aflex_best_energy_prefix"], slo)
        if aflex is None:
            continue

        p = pct_savings(optb, aflex)
        if not math.isfinite(p):
            continue
        by_ob[(olen, bs)][ilen] = p

    per_ob: list[float] = []
    for ilmap in by_ob.values():
        wm = weighted_mean_over_marked_input_lens(ilmap)
        if wm is not None and math.isfinite(wm):
            per_ob.append(wm)

    if not per_ob:
        return float("nan"), 0, skips
    return float(np.mean(per_ob)), len(per_ob), skips


def _savings_pct_vector_at_m(prepared_by_key: dict, keys_by_tp: dict, *, tps: list[int], m: float) -> tuple:
    out = []
    for tp in tps:
        v, _, _ = aggregate_savings_pct_for_tp_at_m(prepared_by_key, keys_by_tp, tp=tp, mult=m)
        out.append(float(v) if math.isfinite(v) else float("nan"))
    return tuple(out)


def _vec_sig_round(vec: tuple[float, ...], ndigits: int) -> tuple:
    t = []
    for x in vec:
        if not math.isfinite(x):
            t.append(None)
        else:
            t.append(round(float(x), ndigits))
    return tuple(t)


def ranked_savings_m_per_tp(
    prepared_by_key: dict,
    keys_by_tp: dict,
    *,
    tp: int,
    candidates: list[float],
    log_every: int = 100,
) -> list[tuple[float, float]]:
    """
    返回该 tp 上按 (savings desc, m asc) 排序后的列表：(savings_pct, m)。
    """
    rows: list[tuple[float, float]] = []
    seen_m: set[float] = set()
    total = len(candidates)
    t0 = time.perf_counter()

    for i, m in enumerate(candidates, start=1):
        if m in seen_m:
            continue
        seen_m.add(m)

        v, _, _ = aggregate_savings_pct_for_tp_at_m(prepared_by_key, keys_by_tp, tp=tp, mult=m)
        if math.isfinite(v):
            rows.append((float(v), float(m)))

        if i % max(1, log_every) == 0 or i == total:
            print(f"[discover][tp={tp}] candidate_progress={i}/{total}, valid={len(rows)}", flush=True)

    rows.sort(key=lambda t: (-t[0], t[1]))
    print(f"[discover][tp={tp}] done: ranked={len(rows)} in {time.perf_counter() - t0:.2f}s", flush=True)
    return rows


def m_four_tp_pct_stable_plateau_start(
    prepared_by_key: dict,
    keys_by_tp: dict,
    *,
    tps: list[int],
    candidates: list[float],
    sig_ndigits: int = 4,
) -> float:
    cand = sorted(candidates)
    if not cand:
        return 1.0
    if len(cand) == 1:
        return float(cand[0])

    sigs = [
        _vec_sig_round(
            _savings_pct_vector_at_m(prepared_by_key, keys_by_tp, tps=tps, m=m),
            sig_ndigits,
        )
        for m in cand
    ]

    sig_last = sigs[-1]
    j = len(cand) - 2
    while j >= 0 and sigs[j] == sig_last:
        j -= 1
    if j < 0:
        return float(cand[0])
    return float(cand[j + 1])


def global_m_max_multiplier(prepared_by_key: dict) -> float:
    mx = 1.0
    for prepared in prepared_by_key.values():
        mx = max(mx, prepared["max_lat"] / prepared["ref"])
    return float(mx)


def _fill_sorted_to_length(sorted_vals: list[float], target: int) -> list[float]:
    v = sorted({round(float(x), 10) for x in sorted_vals})
    if not v:
        return [1.0] * target
    while len(v) < target:
        best_gap = -1.0
        best_mid = None
        for i in range(len(v) - 1):
            g = v[i + 1] - v[i]
            if g > best_gap:
                best_gap = g
                best_mid = 0.5 * (v[i] + v[i + 1])
        if best_mid is None or best_gap <= 1e-15:
            best_mid = v[-1] * 1.0001
        v.append(float(best_mid))
        v = sorted({round(float(x), 10) for x in v})
    return v[:target]


def slo_column_label(m: float, plateau_plus: bool = False) -> str:
    s = f"SLO×{float(m):.3f}"
    return s + ("+" if plateau_plus else "")


def slo_column_labels_unique(mults: list[float], m_plateau_plus: float | None = None) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for m in mults:
        plus = m_plateau_plus is not None and round(float(m), 8) == round(float(m_plateau_plus), 8)
        base = slo_column_label(m, plateau_plus=plus)
        seen[base] = seen.get(base, 0) + 1
        out.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
    return out


def discover_six_multipliers(
    prepared_by_key: dict,
    keys_by_tp: dict,
    *,
    tps: list[int],
    m_scan_round_digits: int = 5,
    stable_sig_ndigits: int = 4,
    log_every: int = 100,
    tp_peak_rank: int = 1,
    tp_peak_collision_round_digits: int = 2,
) -> tuple[list[float], dict]:
    m_min = 1.0
    m_max = global_m_max_multiplier(prepared_by_key)
    raw_cand = union_critical_multipliers(prepared_by_key, m_round_digits=m_scan_round_digits)
    candidates = sorted(
        {round(x, 8) for x in raw_cand if m_min - 1e-12 <= x <= m_max + 1e-9} | {round(m_min, 8), round(m_max, 8)}
    )
    print(f"[discover] candidates={len(candidates)}, m_min={m_min:.2f}, m_max={m_max:.2f}", flush=True)

    ranked_by_tp: dict[int, list[tuple[float, float]]] = {}
    for tp in tps:
        print(f"[discover] start tp={tp}", flush=True)
        ranked_by_tp[tp] = ranked_savings_m_per_tp(
            prepared_by_key,
            keys_by_tp,
            tp=tp,
            candidates=candidates,
            log_every=log_every,
        )

    # 选初始 peak：第 N 大均值降幅
    rank_ix: dict[int, int] = {}
    peak_m: dict[int, float] = {}
    peak_v: dict[int, float] = {}
    for tp in tps:
        ranked = ranked_by_tp.get(tp) or []
        if not ranked:
            rank_ix[tp] = 0
            peak_m[tp] = float("nan")
            peak_v[tp] = float("nan")
            continue
        ix = max(0, min(len(ranked) - 1, int(tp_peak_rank) - 1))
        rank_ix[tp] = ix
        peak_v[tp], peak_m[tp] = ranked[ix]

    # 碰撞消解：只看 tp 的 peak_m 是否在 round(m,2) 后变成同值
    max_pass = 1000
    for _ in range(max_pass):
        peak_rounds = [round(float(peak_m[tp]), tp_peak_collision_round_digits) for tp in tps]
        dup_values = {x for x in peak_rounds if peak_rounds.count(x) > 1}
        if not dup_values:
            break
        moved = False
        for tp in reversed(tps):
            if round(float(peak_m[tp]), tp_peak_collision_round_digits) not in dup_values:
                continue
            ranked = ranked_by_tp.get(tp) or []
            if not ranked:
                continue
            if rank_ix[tp] + 1 >= len(ranked):
                continue
            rank_ix[tp] += 1
            peak_v[tp], peak_m[tp] = ranked[rank_ix[tp]]
            moved = True
            break
        if not moved:
            break

    m_pct_stable = m_four_tp_pct_stable_plateau_start(
        prepared_by_key,
        keys_by_tp,
        tps=tps,
        candidates=candidates,
        sig_ndigits=stable_sig_ndigits,
    )
    vec_stable = _savings_pct_vector_at_m(prepared_by_key, keys_by_tp, tps=tps, m=m_pct_stable)
    savings_at_stable = {tp: float(vec_stable[i]) for i, tp in enumerate(tps)}

    seeds = [m_min, float(m_pct_stable)] + [float(peak_m[tp]) for tp in tps if math.isfinite(peak_m.get(tp, float("nan")))]
    mults = _fill_sorted_to_length(seeds, 6)

    info = {
        "m_min": m_min,
        "m_max": m_max,
        "m_pct_stable": float(m_pct_stable),
        "stable_vec_sig_ndigits": stable_sig_ndigits,
        "savings_pct_at_m_pct_stable": savings_at_stable,
        "peak_m": peak_m,
        "peak_mean_savings_pct": peak_v,
        "candidates_n": len(candidates),
        "tp_peak_rank": int(tp_peak_rank),
        "tp_peak_collision_round_digits": int(tp_peak_collision_round_digits),
        "tp_peak_rank_ix_after_resolve": dict(rank_ix),
    }
    print(
        f"[discover] done: m_pct_stable={info['m_pct_stable']:.2f}, multipliers={len(mults)}",
        flush=True,
    )
    return mults, info


def run_analysis(
    prepared_by_key: dict,
    keys_by_tp: dict,
    *,
    slo_multipliers: list[float],
    slo_column_m_plus: float | None = None,
    log_every: int = 1,
) -> tuple[np.ndarray, list[int], list[str], dict]:
    n_slo = len(slo_multipliers)
    tps = sorted(keys_by_tp)
    matrix = np.full((len(tps), n_slo), np.nan, dtype=float)
    valid_counts = np.zeros((len(tps), n_slo), dtype=int)
    excluded_optb = np.zeros((len(tps), n_slo), dtype=int)

    slo_labels = slo_column_labels_unique(slo_multipliers, m_plateau_plus=slo_column_m_plus)

    total = len(tps) * n_slo
    done = 0
    t0 = time.perf_counter()
    for ti, tp in enumerate(tps):
        for si, mult in enumerate(slo_multipliers):
            v, c, sk = aggregate_savings_pct_for_tp_at_m(
                prepared_by_key,
                keys_by_tp,
                tp=tp,
                mult=mult,
            )
            excluded_optb[ti, si] = sk
            if math.isfinite(v):
                matrix[ti, si] = v
                valid_counts[ti, si] = c
            done += 1
        if (ti + 1) % max(1, log_every) == 0 or (ti + 1) == len(tps):
            print(f"[run_analysis] tp_progress={ti+1}/{len(tps)}, cell_progress={done}/{total}", flush=True)

    print(f"[run_analysis] done in {time.perf_counter() - t0:.2f}s", flush=True)

    meta = {
        "valid_counts": valid_counts,
        "excluded_optb_infeasible": excluded_optb,
        "slo_multipliers": np.asarray(slo_multipliers, dtype=float),
        "slo_labels": slo_labels,
        "column_mean_vectors": [matrix[:, si].copy() for si in range(n_slo)],
    }
    return matrix, tps, slo_labels, meta


def write_csv(path: str, tps: list[int], slo_labels: list[str], matrix: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tp"] + slo_labels)
        for ti, tp in enumerate(tps):
            row = [tp]
            for si in range(matrix.shape[1]):
                v = matrix[ti, si]
                row.append("" if math.isnan(v) else f"{v:.2f}")
            w.writerow(row)


def plot_heatmap(tps: list[int], slo_labels: list[str], matrix: np.ndarray, out_path: str) -> None:
    w = min(28.0, max(6.0, len(slo_labels) * 0.35))
    fig, ax = plt.subplots(figsize=(w, 3.2))
    data = np.nan_to_num(matrix, nan=0.0)
    vmax = float(np.nanmax(matrix)) if np.isfinite(np.nanmax(matrix)) else 1.0
    im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0, vmax=max(vmax, 1e-6))
    ax.set_yticks(range(len(tps)))
    ax.set_yticklabels([f"tp={tp}" for tp in tps])
    ax.set_xticks(range(len(slo_labels)))
    ax.set_xticklabels(slo_labels, rotation=30, ha="right")
    ax.set_xlabel(r"SLO = $m \cdot L_{\min}^{(36)}$ (column: SLO$\times m$, $+$ = plateau tail)")
    ax.set_ylabel("tp")
    ax.set_title("D-stage mean energy savings AFlex vs OptB (%)")
    fig.colorbar(im, ax=ax, label="mean savings %")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def parse_slo_multipliers(s: str) -> list[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("empty --fixed-slo-mults")
    vals = [float(p) for p in parts]
    if any(v <= 0 for v in vals):
        raise ValueError("SLO multipliers must be positive")
    return vals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV_PATH, help="D_data.csv 路径")
    ap.add_argument("--out-dir", default=OUT_DIR, help="输出目录")
    ap.add_argument("--fixed-slo-mults", default=None, help="逗号分隔倍数列表，例如 1,1.2,1.5,2")
    ap.add_argument("--six-scan-m-round-digits", type=int, default=5, help="six 模式：候选 m 舍入位数")
    ap.add_argument("--six-stable-sig-ndigits", type=int, default=4, help="六档尾部稳定判定签名 round 位数")
    ap.add_argument("--tp-peak-rank", type=int, default=1, help="每个 tp 的 m_peak 取第 N 大（默认 1=最大）")
    ap.add_argument("--tp-peak-collision-round-digits", type=int, default=2, help="碰撞消解 round 位数（默认 2，即 round 到 0.01）")
    ap.add_argument("--log-every", type=int, default=100, help="进度日志步长（默认 100）")
    args = ap.parse_args()

    t_start = time.perf_counter()
    print(f"[main] loading csv: {args.csv}", flush=True)
    rows = load_rows(args.csv)
    print(f"[main] rows={len(rows)}", flush=True)

    grouped = group_by_key(rows)
    print(f"[main] grouped_keys={len(grouped)}", flush=True)

    prepared_by_key, keys_by_tp = preprocess_grouped(grouped, log_every=max(1, args.log_every))
    tps = sorted(keys_by_tp)
    print(f"[main] valid_tps={tps}", flush=True)

    if args.fixed_slo_mults:
        mults = parse_slo_multipliers(args.fixed_slo_mults)
        mode = "fixed"
        six_info = None
    else:
        mults, six_info = discover_six_multipliers(
            prepared_by_key,
            keys_by_tp,
            tps=tps,
            m_scan_round_digits=args.six_scan_m_round_digits,
            stable_sig_ndigits=args.six_stable_sig_ndigits,
            log_every=max(1, args.log_every),
            tp_peak_rank=args.tp_peak_rank,
            tp_peak_collision_round_digits=args.tp_peak_collision_round_digits,
        )
        mode = "six"

    plateau_for_label = float(six_info["m_pct_stable"]) if six_info is not None else None
    matrix, tps, slo_labels, meta = run_analysis(
        prepared_by_key,
        keys_by_tp,
        slo_multipliers=mults,
        slo_column_m_plus=plateau_for_label,
        log_every=1,
    )
    if six_info is not None:
        meta["six_info"] = six_info

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "slo_energy_mean_savings_pct.csv")
    png_path = os.path.join(args.out_dir, "slo_energy_mean_savings_heatmap.png")
    mults_path = os.path.join(args.out_dir, "slo_energy_multipliers.txt")

    write_csv(csv_path, tps, slo_labels, matrix)
    plot_heatmap(tps, slo_labels, matrix, png_path)

    with open(mults_path, "w") as mf:
        mf.write("# D_data SLO multipliers: m = SLO / SLO_ref, SLO_ref=min_36(A+F)\n")
        mf.write(f"# mode={mode}\n")
        if six_info is not None:
            mf.write(f"# m_scan_upper_bound={six_info['m_max']:.2f}\n")
            mf.write(
                f"# m_pct_stable={six_info['m_pct_stable']:.2f} "
                f"(4-tp vector stable; vec round {six_info['stable_vec_sig_ndigits']} dp)\n"
            )
            for tp in tps:
                if tp in six_info["savings_pct_at_m_pct_stable"] and math.isfinite(
                    float(six_info["savings_pct_at_m_pct_stable"][tp])
                ):
                    mf.write(f"#   tp{tp} savings_pct_at_m_pct_stable={float(six_info['savings_pct_at_m_pct_stable'][tp]):.2f}\n")
            for tp in tps:
                if tp in six_info["peak_m"] and math.isfinite(float(six_info["peak_m"][tp])):
                    mf.write(
                        f"# peak_tp{tp}: m={float(six_info['peak_m'][tp]):.2f} "
                        f"mean_savings_pct={float(six_info['peak_mean_savings_pct'][tp]):.2f}\n"
                    )
        for i, m in enumerate(mults):
            mf.write(f"{i}\t{float(m):.3f}\n")

    print(" wrote:", csv_path)
    print(" wrote:", png_path)
    print(" wrote:", mults_path)
    print(f"[main] total_elapsed={time.perf_counter() - t_start:.2f}s", flush=True)
    if six_info is not None:
        print(f"\n模式: {mode}\n  档位数={len(mults)}  倍数序列: {[round(float(m),2) for m in mults]}")
    else:
        print(f"\n模式: {mode}\n  档位数={len(mults)}  倍数序列: {[round(float(m),2) for m in mults]}")


if __name__ == "__main__":
    main()

