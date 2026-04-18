#!/usr/bin/env python3
"""
SLO 能耗收益分析：在 A+F 时延之和不超过 SLO 的前提下，对比 OptB（A/F 同频）
与 AFlex（A/F 独立选频）的总能耗，并聚合为 (tp × SLO 档位) 的均值百分比表。

百分比定义：(OptB_total_energy - AFlex_total_energy) / OptB_total_energy * 100
（AFlex 相对 OptB 的能耗降幅；OptB 不可行时该样本不参与该 SLO 档位的均值。）

SLO 档位（默认 **six**，固定 6 列）：
- 每 key：SLO_ref = 36 组合下 min(A+F)；SLO = SLO_ref × m（列名 ``SLO×m``；末档稳定列带 ``+``）。
- **m_min=1.0**；**m_max** 仅用于界定候选扫描上界；最后一档不用 m_max，而用 **四 tp 聚合降幅 %% 随 m 增大已不再变化**（与最大候选处向量一致）时的 **最松一侧的平台起点** ``m_pct_stable``。
- 对每个 tp，在「各 key 的 L/SLO_ref」并集（舍入扫描）上取使该 tp **均值能耗降幅 %% 最大**的 **m_peak(tp)**。
- 六档 = 排序去重 ``{1.0, m_pct_stable, 各 tp 的 m_peak}``；不足 6 个则在最大空隙插中点补足。
- ``--adaptive``：旧版按向量变化加档；``--fixed-slo-mults``：手动倍数列表。

聚合（表格中每个 tp×SLO 格）：
- 仅使用 input_len ∈ {128,512,1024,2048,4096,8192}；权重 30%/30%/20%/10%/7%/3%。
- 对每个 batch_size：在「本 bs 下、有数据的上述 input_len」子集上，将权重归一化后做加权平均。
- 再对所有 batch_size 的上述结果做算术平均。其它 input_len 不参与。

输出：CSV 与 ``slo_energy_multipliers.txt`` 中数值与 SLO 倍数均保留两位小数；列名为 ``SLO×X.XX``（乘号），末档稳定 SLO 为 ``SLO×X.XX+`` 表示更大 m 下同值；若碰撞则加 ``_2`` 后缀。
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = os.path.join(os.path.dirname(__file__), "P_data.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "figures", "slo_energy")

# 仅下列 input_len 参与聚合；权重和为 1.0。
INPUT_LEN_WEIGHTS: dict[int, float] = {
    128: 0.30,
    512: 0.30,
    1024: 0.20,
    2048: 0.10,
    4096: 0.07,
    8192: 0.03,
}


def load_rows(path: str):
    rows = []
    with open(path, newline="") as f:
        r = csv.reader(f)
        next(r)
        header = next(r)
        idx = {name: i for i, name in enumerate(header)}
        for line in r:
            if len(line) < len(header):
                continue
            rows.append(
                {
                    "tp": int(line[idx["tp"]]),
                    "input_len": int(line[idx["input_len"]]),
                    "gpu_clock": int(line[idx["gpu_clock"]]),
                    "batch_size": int(line[idx["batch_size"]]),
                    "A": float(line[idx["A"]]),
                    "F": float(line[idx["F"]]),
                    "A_energy_mj": float(line[idx["A_energy_mj"]]),
                    "F_energy_mj": float(line[idx["F_energy_mj"]]),
                }
            )
    return rows


def group_by_key(rows):
    """(tp, input_len, batch_size) -> gpu_clock -> record"""
    g = defaultdict(dict)
    for rec in rows:
        k = (rec["tp"], rec["input_len"], rec["batch_size"])
        g[k][rec["gpu_clock"]] = rec
    return g


def best_optb_energy(by_clock: dict, slo: float) -> float | None:
    """统一频率：在 A+F<=SLO 的时钟里最小化 A_energy+F_energy。"""
    best = math.inf
    for rec in by_clock.values():
        lat = rec["A"] + rec["F"]
        if lat <= slo + 1e-12:
            e = rec["A_energy_mj"] + rec["F_energy_mj"]
            if e < best:
                best = e
    return None if best == math.inf else best


def best_aflex_energy(by_clock: dict, slo: float) -> float | None:
    """独立频率：36 种组合里满足 A+F<=SLO 的最小能耗。"""
    clocks = list(by_clock.keys())
    best = math.inf
    for ca in clocks:
        for cf in clocks:
            ra, rb = by_clock[ca], by_clock[cf]
            lat = ra["A"] + rb["F"]
            if lat <= slo + 1e-12:
                e = ra["A_energy_mj"] + rb["F_energy_mj"]
                if e < best:
                    best = e
    return None if best == math.inf else best


def lat_bounds_aflex(by_clock: dict) -> tuple[float, float]:
    """所有 A_ca+F_cf（独立选频）的时延下界与上界。"""
    clocks = list(by_clock.keys())
    vals = []
    for ca in clocks:
        for cf in clocks:
            vals.append(by_clock[ca]["A"] + by_clock[cf]["F"])
    return min(vals), max(vals)


def slo_ref_min_aflex(by_clock: dict) -> float:
    """SLO_ref = 36 组合里 A+F 的最小值（最紧可达时延）。"""
    return lat_bounds_aflex(by_clock)[0]


def aflex_latencies(by_clock: dict) -> list[float]:
    """36 种 (f_a,f_f) 的 A+F 列表。"""
    clocks = list(by_clock.keys())
    out = []
    for ca in clocks:
        for cf in clocks:
            out.append(by_clock[ca]["A"] + by_clock[cf]["F"])
    return out


def critical_slo_multipliers_for_key(by_clock: dict) -> list[float]:
    """该 key 下 SLO=SLO_ref*m 时，可行域发生变化的倍数 m = L/SLO_ref（去重排序）。"""
    if len(by_clock) < 6:
        return []
    ref = slo_ref_min_aflex(by_clock)
    if ref <= 0:
        return []
    ratios = {L / ref for L in aflex_latencies(by_clock)}
    ratios.add(1.0)
    return sorted(ratios)


def union_critical_multipliers(
    grouped: dict, min_clocks: int, m_round_digits: int = 4
) -> list[float]:
    """跨所有 key 合并候选倍数；对 m 做 round 以合并相近边界，减少无意义扫描步数。"""
    bag: set[float] = set()
    q = max(2, min(8, int(m_round_digits)))
    for by_c in grouped.values():
        if len(by_c) < min_clocks:
            continue
        for m in critical_slo_multipliers_for_key(by_c):
            if math.isfinite(m) and m >= 1.0 - 1e-15:
                bag.add(round(float(m), q))
    if not bag:
        return [1.0]
    arr = sorted(bag)
    merged: list[float] = []
    for x in arr:
        if not merged or x > merged[-1] + 10 ** (-q) * max(1.0, merged[-1]):
            merged.append(x)
    return merged


def pct_mean_signature(v: np.ndarray, ndigits: int = 4) -> tuple:
    """用于判定「均值 %% 是否变化」的量化签名，抑制浮点噪声。"""
    out = []
    for i in range(len(v)):
        x = float(v[i])
        if not math.isfinite(x):
            out.append(None)
        else:
            out.append(round(x, ndigits))
    return tuple(out)


def pct_savings(optb_e: float, aflex_e: float) -> float:
    if optb_e <= 0 or not math.isfinite(optb_e):
        return float("nan")
    return (optb_e - aflex_e) / optb_e * 100.0


def weighted_mean_over_marked_input_lens(
    pct_by_ilen: dict[int, float],
) -> float | None:
    """在 INPUT_LEN_WEIGHTS 中有权重的 input_len 上，对「有 pct 的项」重归一化权重后加权平均。"""
    avail = {
        il
        for il in INPUT_LEN_WEIGHTS
        if il in pct_by_ilen and math.isfinite(pct_by_ilen[il])
    }
    if not avail:
        return None
    wsum = sum(INPUT_LEN_WEIGHTS[il] for il in avail)
    return float(
        sum(pct_by_ilen[il] * (INPUT_LEN_WEIGHTS[il] / wsum) for il in avail)
    )


def pct_savings_for_key(
    grouped: dict, k: tuple, mult: float, min_clocks: int
) -> tuple[float | None, bool]:
    """
    返回 (pct, optb_infeasible)。
    optb_infeasible=True 表示该 key 上 OptB 无可行点（计入 skip）。
    """
    by_c = grouped[k]
    if len(by_c) < min_clocks:
        return None, False
    ref = slo_ref_min_aflex(by_c)
    if ref <= 0:
        return None, False
    slo = ref * mult
    optb = best_optb_energy(by_c, slo)
    if optb is None:
        return None, True
    aflex = best_aflex_energy(by_c, slo)
    assert aflex is not None
    p = pct_savings(optb, aflex)
    if not math.isfinite(p):
        return None, False
    return p, False


def aggregate_savings_pct_for_tp_at_m(
    grouped: dict, tp: int, mult: float, min_clocks: int
) -> tuple[float, int, int]:
    """
    对单个 tp、倍数 mult：先按 (input_len 加权) × batch_size 等权平均。
    返回 (聚合均值 %%, 参与最终均值的 batch_size 个数, OptB 不可行 key 次数)。
    """
    by_bs: dict[int, dict[int, float]] = defaultdict(dict)
    skips = 0
    for k in grouped:
        if k[0] != tp:
            continue
        ilen, bs = k[1], k[2]
        if ilen not in INPUT_LEN_WEIGHTS:
            continue
        p, infeas = pct_savings_for_key(grouped, k, mult, min_clocks)
        if infeas:
            skips += 1
        if p is not None:
            by_bs[bs][ilen] = p
    per_bs: list[float] = []
    for _bs, ilmap in by_bs.items():
        wm = weighted_mean_over_marked_input_lens(ilmap)
        if wm is not None and math.isfinite(wm):
            per_bs.append(wm)
    if not per_bs:
        return float("nan"), 0, skips
    return float(np.mean(per_bs)), len(per_bs), skips


def mean_pct_row_at_m(
    grouped: dict, tps: list[int], mult: float, min_clocks: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对每个 tp 返回 (聚合 mean %%, 参与均值的 batch_size 个数, OptB 不可行次数)。"""
    n = len(tps)
    means = np.full(n, np.nan, dtype=float)
    counts = np.zeros(n, dtype=int)
    skips = np.zeros(n, dtype=int)
    for ti, tp in enumerate(tps):
        v, c, sk = aggregate_savings_pct_for_tp_at_m(grouped, tp, mult, min_clocks)
        if math.isfinite(v):
            means[ti] = v
        counts[ti] = c
        skips[ti] = sk
    return means, counts, skips


def discover_adaptive_multipliers(
    grouped: dict,
    tps: list[int],
    min_clocks: int = 6,
    pct_round_digits: int = 2,
    m_round_digits: int = 4,
) -> tuple[list[float], list[np.ndarray]]:
    """
    第一档恒为 1.0；之后按候选 m 递增，当 4×tp 均值 % 向量相对上一档变化时加入该 m。
    返回 (multipliers, history_vectors) 其中 history_vectors[k] 为第 k 档的均值向量。
    """
    raw = sorted(
        set(union_critical_multipliers(grouped, min_clocks, m_round_digits)) | {1.0}
    )
    snapped = sorted({round(x, 8) for x in raw})
    candidates = [1.0] + [x for x in snapped if x > 1.0 + 1e-8]

    mults: list[float] = []
    hist: list[np.ndarray] = []
    last_sig: tuple | None = None

    for m in candidates:
        vec, _, _ = mean_pct_row_at_m(grouped, tps, m, min_clocks)
        sig = pct_mean_signature(vec, ndigits=pct_round_digits)
        if last_sig is None:
            mults.append(m)
            hist.append(vec.copy())
            last_sig = sig
            continue
        if sig != last_sig:
            mults.append(m)
            hist.append(vec.copy())
            last_sig = sig

    if not mults:
        mults = [1.0]
        v, _, _ = mean_pct_row_at_m(grouped, tps, 1.0, min_clocks)
        hist = [v]
    return mults, hist


def global_m_max_multiplier(grouped: dict, min_clocks: int = 6) -> float:
    """最松倍数：所有 key 上 max_36(A+F)/SLO_ref 的全局最大值（再与 1.0 取 max）。"""
    mx = 1.0
    for by_c in grouped.values():
        if len(by_c) < min_clocks:
            continue
        ref = slo_ref_min_aflex(by_c)
        if ref <= 0:
            continue
        _lo, hi = lat_bounds_aflex(by_c)
        mx = max(mx, hi / ref)
    return mx


def mean_pct_single_tp(
    grouped: dict, tp: int, mult: float, min_clocks: int
) -> float:
    """单个 tp 在倍数 mult 下的聚合能耗降幅 %%（与表格同一套加权规则）。"""
    v, _c, _sk = aggregate_savings_pct_for_tp_at_m(grouped, tp, mult, min_clocks)
    return v


def ranked_savings_m_per_tp(
    grouped: dict,
    tp: int,
    candidates: list[float],
    min_clocks: int,
) -> list[tuple[float, float]]:
    """
    该 tp 上每个候选 m 的聚合能耗降幅 %%，按降幅从高到低、m 从小到大排序；同一 m 只保留一条。
    """
    rows: list[tuple[float, float]] = []
    seen_m: set[float] = set()
    for m in candidates:
        if m in seen_m:
            continue
        s = mean_pct_single_tp(grouped, tp, m, min_clocks)
        if not math.isfinite(s):
            continue
        seen_m.add(m)
        rows.append((s, m))
    rows.sort(key=lambda t: (-t[0], t[1]))
    return rows


def _has_round2_collision(vals: list[float]) -> bool:
    r = [round(float(x), 2) for x in vals]
    return len(r) != len(set(r))


def _savings_pct_vector_at_m(
    grouped: dict,
    tps: list[int],
    m: float,
    min_clocks: int,
) -> tuple[float, ...]:
    out = []
    for tp in tps:
        v, _, _ = aggregate_savings_pct_for_tp_at_m(
            grouped, tp, m, min_clocks
        )
        out.append(float(v) if math.isfinite(v) else float("nan"))
    return tuple(out)


def _vec_sig_round(vec: tuple[float, ...], ndigits: int) -> tuple:
    t = []
    for x in vec:
        if not math.isfinite(x):
            t.append(None)
        else:
            t.append(round(x, ndigits))
    return tuple(t)


def m_four_tp_pct_stable_plateau_start(
    grouped: dict,
    tps: list[int],
    candidates: list[float],
    min_clocks: int,
    sig_ndigits: int = 4,
) -> float:
    """
    在升序候选 m 上，从最右端向左找：四 tp 聚合降幅向量（经 round）首次与最右端不同的位置；
    返回其右侧相邻的 m，即「百分比已不再随 SLO 放宽而变化」的平台 **起点**。
    若全程不变则返回最小候选 m。
    """
    cand = sorted(candidates)
    if not cand:
        return 1.0
    if len(cand) == 1:
        return float(cand[0])
    sigs = [
        _vec_sig_round(
            _savings_pct_vector_at_m(grouped, tps, m, min_clocks),
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


def _resolve_peak_m_round2_collisions(
    tps: list[int],
    ranked: dict[int, list[tuple[float, float]]],
    m_min: float,
    m_max: float,
) -> tuple[dict[int, float], dict[int, float], dict[int, int]]:
    """
    初始取各 tp 排名第一的 m；若与 1.0、m_max 或其它 tp 的 m 在保留两位小数时冲突，
    则对参与冲突的 tp 依次改用「第二大、第三大…」候选 m，直到无冲突或无法继续。
    返回 (peak_m, peak_v, rank_ix)。
    """
    rank_ix: dict[int, int] = {tp: 0 for tp in tps}
    peak_m: dict[int, float] = {}
    peak_v: dict[int, float] = {}
    for tp in tps:
        if not ranked[tp]:
            peak_m[tp] = m_min
            peak_v[tp] = float("nan")
            continue
        peak_m[tp] = ranked[tp][0][1]
        peak_v[tp] = ranked[tp][0][0]

    def combined() -> list[float]:
        return [m_min, m_max] + [peak_m[tp] for tp in tps]

    max_pass = 500
    for _ in range(max_pass):
        if not _has_round2_collision(combined()):
            break
        moved = False
        for tp in reversed(tps):
            arr = combined()
            my = peak_m[tp]
            if sum(1 for x in arr if round(x, 2) == round(my, 2)) <= 1:
                continue
            if rank_ix[tp] + 1 >= len(ranked[tp]):
                continue
            rank_ix[tp] += 1
            peak_m[tp] = ranked[tp][rank_ix[tp]][1]
            peak_v[tp] = ranked[tp][rank_ix[tp]][0]
            moved = True
            if not _has_round2_collision(combined()):
                break
        if not moved:
            break

    return peak_m, peak_v, rank_ix


def _fill_sorted_to_length(sorted_vals: list[float], target: int) -> list[float]:
    """在已排序唯一列表上通过向最大区间插入中点扩充到 target 长度。"""
    v = sorted({round(x, 10) for x in sorted_vals})
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
        v.append(best_mid)
        v = sorted({round(x, 10) for x in v})
    return v[:target]


def discover_six_multipliers(
    grouped: dict,
    tps: list[int],
    min_clocks: int = 6,
    m_scan_round_digits: int = 5,
    stable_sig_ndigits: int = 4,
) -> tuple[list[float], dict]:
    """
    六档：m_min=1.0、``m_pct_stable``（四 tp 聚合降幅 %% 已进入最终不变平台时的最小 m）、各 tp 的 m_peak。
    若两个倍数在「保留两位小数」下同号（列名会撞），则对相关 tp 依次改用候选里降幅第二大、第三大…的 m，直到区分。
    返回 (mults, info)。
    """
    m_min = 1.0
    m_max = global_m_max_multiplier(grouped, min_clocks)
    raw_cand = union_critical_multipliers(
        grouped, min_clocks, m_round_digits=m_scan_round_digits
    )
    candidates = sorted(
        {round(x, 8) for x in raw_cand if m_min - 1e-12 <= x <= m_max + 1e-9}
        | {round(m_min, 8), round(m_max, 8)}
    )

    ranked: dict[int, list[tuple[float, float]]] = {
        tp: ranked_savings_m_per_tp(grouped, tp, candidates, min_clocks)
        for tp in tps
    }
    peak_m, peak_v, rank_ix = _resolve_peak_m_round2_collisions(
        tps, ranked, m_min, m_max
    )

    m_pct_stable = m_four_tp_pct_stable_plateau_start(
        grouped,
        tps,
        candidates,
        min_clocks,
        sig_ndigits=stable_sig_ndigits,
    )
    vec_stable = _savings_pct_vector_at_m(
        grouped, tps, m_pct_stable, min_clocks
    )
    savings_at_stable: dict[int, float] = {
        tp: vec_stable[i] for i, tp in enumerate(tps)
    }

    seeds = [m_min, m_pct_stable] + [peak_m[tp] for tp in tps]
    mults = _fill_sorted_to_length(seeds, 6)
    info = {
        "m_min": m_min,
        "m_max": m_max,
        "m_pct_stable": m_pct_stable,
        "stable_vec_sig_ndigits": stable_sig_ndigits,
        "savings_pct_at_m_pct_stable": savings_at_stable,
        "peak_m": peak_m,
        "peak_mean_savings_pct": peak_v,
        "peak_rank_ix": rank_ix,
        "candidates_n": len(candidates),
    }
    return mults, info


def parse_slo_multipliers(s: str) -> list[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("empty --fixed-slo-mults")
    mults = [float(p) for p in parts]
    if any(m <= 0 for m in mults):
        raise ValueError("SLO multipliers must be positive")
    return mults


def slo_column_label(m: float, plateau_plus: bool = False) -> str:
    """列名：``SLO×`` + 倍数两位小数；``plateau_plus`` 时在末尾加 ``+``（表示更大 m 下同）。"""
    s = f"SLO×{m:.2f}"
    return s + ("+" if plateau_plus else "")


def slo_column_labels_unique(
    mults: list[float],
    m_plateau_plus: float | None = None,
) -> list[str]:
    """
    为每个倍数生成列名；与 ``m_plateau_plus`` 数值匹配（8 位舍入一致）的列加 ``+``。
    四舍五入碰撞时加 ``_2``、``_3``… 后缀保证唯一。
    """
    seen: dict[str, int] = {}
    out: list[str] = []
    for m in mults:
        plus = False
        if m_plateau_plus is not None:
            plus = round(m, 8) == round(float(m_plateau_plus), 8)
        base = slo_column_label(m, plateau_plus=plus)
        seen[base] = seen.get(base, 0) + 1
        if seen[base] == 1:
            out.append(base)
        else:
            out.append(f"{base}_{seen[base]}")
    return out


def run_analysis(
    grouped: dict,
    slo_multipliers: list[float],
    min_clocks: int = 6,
    slo_column_m_plus: float | None = None,
) -> tuple[np.ndarray, list[int], list[str], dict]:
    """
    返回:
      matrix: shape (len(tps), n_slo) 每个格子为聚合后的均值能耗降幅 %%
      tps: 排序后的 tp 列表
      slo_labels: 列名，如 SLO×1.00；与 slo_column_m_plus 匹配的列为 SLO×m.XX+
      meta: 含 valid_counts（参与均值的 batch_size 个数）、slo_multipliers 等
    """
    n_slo = len(slo_multipliers)
    tps = sorted({k[0] for k in grouped})
    matrix = np.full((len(tps), n_slo), np.nan, dtype=float)
    valid_counts = np.zeros((len(tps), n_slo), dtype=int)
    excluded_optb = np.zeros((len(tps), n_slo), dtype=int)
    slo_labels = slo_column_labels_unique(
        slo_multipliers, m_plateau_plus=slo_column_m_plus
    )

    for ti, tp in enumerate(tps):
        for si, mult in enumerate(slo_multipliers):
            v, c, sk = aggregate_savings_pct_for_tp_at_m(
                grouped, tp, mult, min_clocks
            )
            excluded_optb[ti, si] = sk
            if math.isfinite(v):
                matrix[ti, si] = v
                valid_counts[ti, si] = c

    meta = {
        "valid_counts": valid_counts,
        "excluded_optb_infeasible": excluded_optb,
        "slo_multipliers": np.asarray(slo_multipliers, dtype=float),
        "slo_labels": slo_labels,
        "column_mean_vectors": [matrix[:, si].copy() for si in range(n_slo)],
    }
    return matrix, tps, slo_labels, meta


def write_csv(path: str, tps: list[int], slo_labels: list[str], matrix: np.ndarray):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tp"] + slo_labels)
        for ti, tp in enumerate(tps):
            row = [tp]
            for si in range(matrix.shape[1]):
                v = matrix[ti, si]
                row.append("" if math.isnan(v) else f"{v:.2f}")
            w.writerow(row)


def plot_heatmap(tps: list[int], slo_labels: list[str], matrix: np.ndarray, out_path: str):
    w = min(28.0, max(6.0, len(slo_labels) * 0.35))
    fig, ax = plt.subplots(figsize=(w, 3.2))
    data = np.nan_to_num(matrix, nan=0.0)
    vmax = float(np.nanmax(matrix)) if np.isfinite(np.nanmax(matrix)) else 1.0
    im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0, vmax=max(vmax, 1e-6))
    ax.set_yticks(range(len(tps)))
    ax.set_yticklabels([f"tp={tp}" for tp in tps])
    ax.set_xticks(range(len(slo_labels)))
    ax.set_xticklabels(slo_labels, rotation=30, ha="right")
    ax.set_xlabel(
        r"SLO = $m \cdot L_{\min}^{(36)}$ (column: SLO$\times m$; $+$ = plateau tail)"
    )
    ax.set_ylabel("tp")
    ax.set_title("Mean energy savings AFlex vs OptB (%) — avg over (input_len, batch_size)")
    fig.colorbar(im, ax=ax, label="mean savings %")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV_PATH, help="P_data.csv 路径")
    ap.add_argument(
        "--fixed-slo-mults",
        default=None,
        help="固定倍数列表（逗号分隔），例如 1,1.5,2,3；指定后覆盖默认 six 档",
    )
    ap.add_argument(
        "--adaptive",
        action="store_true",
        help="使用旧版 adaptive 选档（按 4×tp 均值向量变化加列），而非默认 six",
    )
    ap.add_argument(
        "--adaptive-pct-round-digits",
        type=int,
        default=2,
        help="仅 --adaptive：均值 %% round 位数（默认 2）",
    )
    ap.add_argument(
        "--adaptive-m-round-digits",
        type=int,
        default=2,
        help="仅 --adaptive：候选 m 舍入位数（默认 2）",
    )
    ap.add_argument(
        "--six-scan-m-round-digits",
        type=int,
        default=5,
        help="默认 six 模式：扫描候选 L/SLO_ref 时的舍入小数位（默认 5）",
    )
    ap.add_argument(
        "--six-stable-sig-ndigits",
        type=int,
        default=4,
        help="six 末档：判定四 tp 向量「是否变化」时对 %% 做 round 的小数位（默认 4）",
    )
    ap.add_argument("--out-dir", default=OUT_DIR, help="输出目录")
    args = ap.parse_args()

    rows = load_rows(args.csv)
    grouped = group_by_key(rows)
    tps = sorted({k[0] for k in grouped})

    six_info: dict | None = None
    if args.fixed_slo_mults:
        mults = parse_slo_multipliers(args.fixed_slo_mults)
        mode = "fixed"
    elif args.adaptive:
        mults, _ = discover_adaptive_multipliers(
            grouped,
            tps,
            pct_round_digits=args.adaptive_pct_round_digits,
            m_round_digits=args.adaptive_m_round_digits,
        )
        mode = "adaptive"
    else:
        mults, six_info = discover_six_multipliers(
            grouped,
            tps,
            m_scan_round_digits=args.six_scan_m_round_digits,
            stable_sig_ndigits=args.six_stable_sig_ndigits,
        )
        mode = "six"

    plateau_for_label = (
        float(six_info["m_pct_stable"]) if six_info is not None else None
    )
    matrix, tps, slo_labels, meta = run_analysis(
        grouped,
        slo_multipliers=mults,
        slo_column_m_plus=plateau_for_label,
    )
    if six_info is not None:
        meta["six_info"] = six_info
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "slo_energy_mean_savings_pct.csv")
    write_csv(csv_path, tps, slo_labels, matrix)
    png_path = os.path.join(args.out_dir, "slo_energy_mean_savings_heatmap.png")
    plot_heatmap(tps, slo_labels, matrix, png_path)

    mults_path = os.path.join(args.out_dir, "slo_energy_multipliers.txt")
    with open(mults_path, "w") as mf:
        mf.write("# m = SLO / SLO_ref per key; SLO_ref = min_36(A+F)\n")
        mf.write(f"# mode={mode}\n")
        if six_info is not None:
            mf.write(f"# m_scan_upper_bound={six_info['m_max']:.2f}\n")
            nd = int(six_info.get("stable_vec_sig_ndigits", 4))
            mf.write(
                f"# m_pct_stable={six_info['m_pct_stable']:.2f} "
                f"(4-tp mean savings %% stable vs max m; vec round {nd} dp)\n"
            )
            sv = six_info.get("savings_pct_at_m_pct_stable", {})
            for tp in tps:
                if tp in sv and math.isfinite(float(sv[tp])):
                    mf.write(
                        f"#   tp{tp} savings_pct_at_m_pct_stable={float(sv[tp]):.2f}\n"
                    )
            rxi = six_info.get("peak_rank_ix", {})
            for tp in tps:
                rk = rxi.get(tp, 0)
                mf.write(
                    f"# peak_tp{tp}: m={six_info['peak_m'][tp]:.2f} "
                    f"mean_savings_pct={six_info['peak_mean_savings_pct'][tp]:.2f} "
                    f"rank_idx={rk}\n"
                )
        for i, m in enumerate(mults):
            mf.write(f"{i}\t{m:.2f}\n")

    print(" wrote:", csv_path)
    print(" wrote:", png_path)
    print(" wrote:", mults_path)
    print(f"\n模式: {mode}")
    print(
        "列含义：SLO×k 表示各 key 上 SLO = SLO_ref×k；+ 表示更大 k 下四 tp 均值 %% 不变；"
        "SLO_ref 为该 key 下 36 组合的 min(A+F)。单元格为能耗降幅均值(%)（AFlex vs OptB）。"
    )
    print(f"  档位数={len(mults)}  倍数序列: {[round(m, 2) for m in mults]}")
    if mode == "six" and six_info is not None:
        nd = int(six_info.get("stable_vec_sig_ndigits", 4))
        print(
            f"  m_scan_upper={six_info['m_max']:.2f}  m_pct_stable="
            f"{six_info['m_pct_stable']:.2f}  vec_round_ndigits={nd}  "
            f"candidates_n={six_info['candidates_n']}"
        )
        sv = six_info.get("savings_pct_at_m_pct_stable", {})
        for tp in tps:
            if tp in sv and math.isfinite(float(sv[tp])):
                print(f"    at m_pct_stable: tp{tp} mean_savings%={float(sv[tp]):.2f}")
        rxi = six_info.get("peak_rank_ix", {})
        for tp in tps:
            rk = rxi.get(tp, 0)
            print(
                f"  tp={tp} peak m={six_info['peak_m'][tp]:.2f} "
                f"mean_savings%={six_info['peak_mean_savings_pct'][tp]:.2f} "
                f"rank_idx={rk}"
            )
    if mode == "adaptive" and meta.get("column_mean_vectors") and len(mults) <= 30:
        print("  各档 4×tp 均值向量 (%):")
        for i, m in enumerate(mults):
            v = meta["column_mean_vectors"][i]
            parts = ", ".join(
                f"tp{tps[j]}={v[j]:.2f}" for j in range(len(tps)) if np.isfinite(v[j])
            )
            print(f"    [{i}] m={m:.2f} -> {parts}")


if __name__ == "__main__":
    main()
