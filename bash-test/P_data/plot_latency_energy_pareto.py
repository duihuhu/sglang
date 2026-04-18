#!/usr/bin/env python3
"""
从 P_data.csv 绘制阶段时延之和–能耗散点图。

表头中的列 A、列 F 分别为 A 阶段、F 阶段的时延；横轴为 t_A+t_F = A+F（与 CSV 数值单位一致）。
纵轴为 A_energy_mj + F_energy_mj（mJ）。

- 整图标题仅一行：`tp=…, bs=…`；无底部图例。子图左上角标 input_len；彩点旁标 gpu_clock（MHz），文字颜色与点一致。
- 灰点 (i≠j)：x = A[i]+F[j]，y = A_energy[i]+F_energy[j]。
- 按 (tp, batch_size) 各出一张图：共 len(tp)×len(batch_size) 张（由 CSV 中实际组合决定）。
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib import font_manager

try:
    _tab10 = matplotlib.colormaps["tab10"]
except AttributeError:
    _tab10 = plt.cm.get_cmap("tab10")

# 与 CSV 表头一致
COL_TP = "tp"
COL_BATCH = "batch_size"
COL_INPUT = "input_len"
COL_CLOCK = "gpu_clock"
COL_A = "A"  # A 阶段时延（与 CSV 单位一致）
COL_F = "F"  # F 阶段时延
COL_AE = "A_energy_mj"
COL_FE = "F_energy_mj"

# 避免默认字体缺中文 glyph（终端里 U+81f4 等替换警告）
_CJK_FONT_CANDIDATES = (
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Noto Serif CJK SC",
    "WenQuanYi Zen Hei",
    "WenQuanYi Micro Hei",
    "Source Han Sans SC",
    "Source Han Sans CN",
    "Droid Sans Fallback",
    "SimHei",
    "Microsoft YaHei",
    "PingFang SC",
)


def setup_matplotlib_cjk_font() -> bool:
    """若系统有可用的 CJK 字体则配置；返回是否已配置。"""
    ttflist = font_manager.fontManager.ttflist
    installed = {f.name for f in ttflist}
    for name in _CJK_FONT_CANDIDATES:
        if name in installed:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans", "Bitstream Vera Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    # 注册名可能与候选表不完全一致（例如 Noto 子集）
    for f in ttflist:
        n = f.name
        path = (f.fname or "").lower()
        if "cjk" in path or "notosanscjk" in path.replace(" ", ""):
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [n, "DejaVu Sans", "Bitstream Vera Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


def load_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="") as f:
        r = csv.reader(f)
        next(r)  # [P] Prefill
        header = next(r)
        for parts in r:
            if len(parts) < len(header):
                continue
            row = dict(zip(header, parts))
            if not row.get(COL_TP, "").strip().isdigit():
                continue
            rows.append(row)
    return rows


def _f(x: str) -> float:
    return float(x.strip())


def _batch_size(r: dict[str, str]) -> int:
    v = r.get(COL_BATCH, "1")
    if not str(v).strip():
        return 1
    return int(str(v).strip())


def subplot_grid(n: int) -> tuple[int, int]:
    if n <= 0:
        return 1, 1
    ncols = int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / ncols))
    return nrows, ncols


def dedupe_sci_offset_texts(axes_flat, n: int, ncols: int) -> None:
    """仅对 x 轴去重：同列最底行子图保留 ×10^n，减轻与下一行子图标题的叠压。
    y 轴不在此隐藏——各子图能耗量级可能不同（如 10³ vs 10⁴），每个子图须各自显示 y 的 offset。"""
    for idx in range(n):
        ax = axes_flat[idx]
        col = idx % ncols
        in_col = [i for i in range(n) if i % ncols == col]
        bottom_idx = max(in_col, key=lambda i: i // ncols)
        ax.xaxis.get_offset_text().set_visible(idx == bottom_idx)


def phase_delays(chunk: list[dict[str, str]]) -> tuple[list[float], list[float]]:
    """每行 t_A、t_F 即表列 A、F。"""
    t_a = [_f(r[COL_A]) for r in chunk]
    t_f = [_f(r[COL_F]) for r in chunk]
    return t_a, t_f


def gray_cross_points(
    chunk: list[dict[str, str]], t_a: list[float], t_f: list[float]
) -> tuple[list[float], list[float]]:
    """i≠j：x = t_A[i]+t_F[j]，y = A_energy_i+F_energy_j。"""
    m = len(chunk)
    gx: list[float] = []
    gy: list[float] = []
    ae = [_f(r[COL_AE]) for r in chunk]
    fe = [_f(r[COL_FE]) for r in chunk]
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            gx.append(t_a[i] + t_f[j])
            gy.append(ae[i] + fe[j])
    return gx, gy


def plot_for_tp_bs(
    rows: list[dict[str, str]],
    tp: int,
    batch_size: int,
    out_path: Path,
) -> None:
    sub = [r for r in rows if int(r[COL_TP]) == tp and _batch_size(r) == batch_size]
    input_lens = sorted({int(r[COL_INPUT]) for r in sub})
    n = len(input_lens)
    nrows, ncols = subplot_grid(n)

    # 全 tp 共用色映射：6 档 gpu_clock
    all_clocks = sorted({int(r[COL_CLOCK]) for r in rows})
    clock_to_color = {c: _tab10(i % 10) for i, c in enumerate(all_clocks)}

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 3.65 * nrows),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    for idx, ilen in enumerate(input_lens):
        ax = axes_flat[idx]
        chunk = [r for r in sub if int(r[COL_INPUT]) == ilen]
        chunk.sort(key=lambda r: int(r[COL_CLOCK]))

        t_a, t_f = phase_delays(chunk)
        xs = [a + b for a, b in zip(t_a, t_f)]
        ys = [_f(r[COL_AE]) + _f(r[COL_FE]) for r in chunk]
        colors = [clock_to_color[int(r[COL_CLOCK])] for r in chunk]

        gx, gy = gray_cross_points(chunk, t_a, t_f)
        if gx:
            ax.scatter(
                gx,
                gy,
                c="#9e9e9e",
                s=32,
                edgecolors="none",
                alpha=0.85,
                zorder=2,
            )

        ax.scatter(xs, ys, c=colors, s=55, edgecolors="k", linewidths=0.35, zorder=3)

        # 频率（gpu_clock）+ 与散点一致的颜色；白描边便于压在灰点/深色区上可读
        for x, y, r, fc in zip(xs, ys, chunk, colors):
            gc = int(r[COL_CLOCK])
            t = ax.annotate(
                f"{gc} MHz",
                (x, y),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7.5,
                color=fc,
                alpha=0.95,
                zorder=4,
            )
            t.set_path_effects(
                [pe.withStroke(linewidth=2.25, foreground="white", capstyle="round")]
            )

        ax.text(
            0.02,
            0.98,
            f"input_len={ilen}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="#212121",
            zorder=5,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#bdbdbd", alpha=0.88),
        )
        if idx % ncols == 0:
            ax.set_ylabel("Energy (mJ)\n$A_{energy}+F_{energy}$", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.55)
        # 横纵轴科学计数法，减少长串 0
        ax.ticklabel_format(
            axis="both",
            style="sci",
            scilimits=(0, 0),
            useMathText=True,
            useOffset=False,
        )
        ax.tick_params(axis="both", labelsize=8, pad=2)

    h_pad = 0.55 + (0.42 if nrows > 1 else 0.0)
    reserve_bottom = 0.12 + 0.015 * max(0, len(all_clocks) - 4)
    fig.tight_layout(
        rect=[0.02, reserve_bottom, 0.98, 0.90],
        h_pad=h_pad,
        w_pad=0.45,
    )
    fig.subplots_adjust(bottom=reserve_bottom, top=0.90)

    dedupe_sci_offset_texts(axes_flat, n, ncols)

    fig.suptitle(f"tp={tp}, bs={batch_size}", fontsize=12, y=0.997)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


def main() -> None:
    here = Path(__file__).resolve().parent
    csv_path = here / "P_data.csv"
    rows = load_rows(csv_path)
    if not rows:
        raise SystemExit(f"无数据: {csv_path}")

    use_cjk = setup_matplotlib_cjk_font()
    if not use_cjk:
        print(
            "未检测到常见中文字体；若图内将来使用中文标注，可安装例如 fonts-noto-cjk。",
            flush=True,
        )

    pairs = sorted({(int(r[COL_TP]), _batch_size(r)) for r in rows})
    out_dir = here / "figures"
    for tp, bs in pairs:
        out = out_dir / f"latency_energy_tp{tp}_bs{bs}_phase_delay.png"
        plot_for_tp_bs(rows, tp, bs, out)
        print(f"写入 {out}")


if __name__ == "__main__":
    main()
