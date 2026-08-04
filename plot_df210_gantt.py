#!/usr/bin/env python3
"""Plot M=1 vs M=2 Gantt comparison for DA=210MHz, DF=210MHz."""
import json, re, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

STEP_RE = re.compile(r"steps=(\[.*\])")

def forward_steps(path, total_bs, m, pick_index=None):
    found = []
    for line in path.open(errors="replace"):
        if "[AFD_PER_STEP]" not in line:
            continue
        mt = STEP_RE.search(line)
        if not mt:
            continue
        steps = json.loads(mt.group(1))
        l0 = [s for s in steps if s["stage"] == "A" and s["layer_id"] == 0]
        if len(l0) == m and sum(s["batch_size"] for s in l0) >= total_bs - 10:
            found.append(steps)
    if not found:
        raise ValueError(f"No matching forward in {path} (bs={total_bs}, M={m})")
    if pick_index is not None and pick_index < len(found):
        idx = pick_index
    else:
        idx = len(found) // 2  # use middle forward for steady state
    return found[idx], len(found)

def interval(s):
    starts = [v for k, v in s.items() if k.endswith("_wall_start_ms")]
    ends = [v for k, v in s.items() if k.endswith("_wall_end_ms")]
    return min(starts), max(ends)

def by(xs, stage, l, mb):
    return next((s for s in xs if s["stage"] == stage and s["layer_id"] == l and s["mb"] == mb), None)

LAYERS = 6
h = 0.55

def draw_panel(ax, da_steps, df_steps, m, layers, title):
    da = [s for s in da_steps if s["layer_id"] < layers]
    df = [s for s in df_steps if s["layer_id"] < layers]
    if not da or not df:
        ax.set_title(title + " [NO DATA]")
        return 100
    t0 = min(interval(s)[0] for s in da + df)
    if m == 1:
        colors = {"A0": "#4472C4", "F0": "#ED7D31", "wait": "#A9D18E"}
    else:
        colors = {"A0": "#4472C4", "A1": "#8FAADC", "F0": "#ED7D31", "F1": "#F4B183", "wait": "#A9D18E"}
    yda, yda_wait, ydf_wait, ydf = 3.15, 2.35, 1.15, 0.35

    for s in da:
        st, en = interval(s)
        x, w = st - t0, en - st
        if s["stage"] == "A":
            c = colors.get(f"A{s['mb']}", "#4472C4")
            ax.barh(yda, w, left=x, height=h, color=c, edgecolor="white", lw=0.3, zorder=3)
        else:
            ax.barh(yda_wait, w, left=x, height=h, color=colors["wait"], edgecolor="white", lw=0.3, zorder=3)
    for s in df:
        st, en = interval(s)
        x, w = st - t0, en - st
        if s["stage"] == "F":
            c = colors.get(f"F{s['mb']}", "#ED7D31")
            ax.barh(ydf, w, left=x, height=h, color=c, edgecolor="white", lw=0.3, zorder=3)
        else:
            ax.barh(ydf_wait, w, left=x, height=h, color=colors["wait"], edgecolor="white", lw=0.3, zorder=3)

    for s in da + df:
        st, en = interval(s)
        x, w = st - t0, en - st
        label = f"L{s['layer_id']}"
        if m > 1:
            label += f"m{s['mb']}"
        if w > 0.5:
            if s in da:
                y_pos = yda if s["stage"] == "A" else yda_wait
            else:
                y_pos = ydf if s["stage"] == "F" else ydf_wait
            fc = "white" if s["stage"] == "F" else "#263238"
            fw = "bold" if s["stage"] == "F" else "normal"
            ax.text(x + w / 2, y_pos, label, ha="center", va="center",
                   fontsize=5.5, color=fc, fontweight=fw)

    for l in range(layers):
        for mb in range(m):
            aa = by(da, "A", l, mb)
            ar = by(df, "A", l, mb)
            ff = by(df, "F", l, mb)
            fw2 = by(da, "F", l, mb)
            if aa and ar:
                _, ae = interval(aa)
                rs, _ = interval(ar)
                ax.annotate("", xy=(rs - t0, ydf_wait + h / 2),
                           xytext=(ae - t0, yda - h / 2),
                           arrowprops=dict(arrowstyle="->", color="#548235",
                                          lw=0.6, alpha=0.7,
                                          connectionstyle="arc3,rad=.06"))
            if ff and fw2:
                _, fe = interval(ff)
                ws, _ = interval(fw2)
                ax.annotate("", xy=(ws - t0, yda_wait - h / 2),
                           xytext=(fe - t0, ydf + h / 2),
                           arrowprops=dict(arrowstyle="->", color="#C00000",
                                          lw=0.6, alpha=0.7,
                                          connectionstyle="arc3,rad=-.06"))

    ax.set_yticks([ydf, ydf_wait, yda_wait, yda])
    ax.set_yticklabels(["DF: FFN compute", "DF: recv/prepare",
                       "DA: recv/postprocess", "DA: Attention compute"])
    ax.set_ylim(-0.15, 4.25)
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    all_ends = [interval(s)[1] - t0 for s in da + df]
    return max(all_ends) if all_ends else 100


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--target-bs", type=int, default=512)
    ap.add_argument("--m1-case", default="m1-high")
    ap.add_argument("--m2-case", default="m2-high")
    ap.add_argument("--title-prefix", default="")
    args = ap.parse_args()

    result_dir = args.result_dir
    out_path = args.output
    target_bs = args.target_bs

    m2_dir = result_dir / args.m2_case
    m1_dir = result_dir / args.m1_case

    m2_da, n2da = forward_steps(m2_dir / "da.log", target_bs, 2)
    m2_df, n2df = forward_steps(m2_dir / "df.log", target_bs, 2)
    print(f"M=2: {n2da} DA, {n2df} DF matching forwards")

    m1_da, n1da = forward_steps(m1_dir / "da.log", target_bs, 1)
    m1_df, n1df = forward_steps(m1_dir / "df.log", target_bs, 1)
    print(f"M=1: {n1da} DA, {n1df} DF matching forwards")

    prefix = args.title_prefix + " " if args.title_prefix else ""
    m2_bs_str = f"{target_bs} ({target_bs//2}+{target_bs//2})" if target_bs > 1 else "2 (1+1)"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 7), sharex=False)

    da1 = [s for s in m1_da if s["perspective"] in ("DA", "attn")]
    df1 = [s for s in m1_df if s["perspective"] in ("DF", "ffn")]
    xmax1 = draw_panel(ax1, da1, df1, 1, LAYERS,
                       f"{prefix}M=1, bs={target_bs}, DA=210MHz, DF=210MHz — Layers 0-{LAYERS-1}")

    da2 = [s for s in m2_da if s["perspective"] in ("DA", "attn")]
    df2 = [s for s in m2_df if s["perspective"] in ("DF", "ffn")]
    xmax2 = draw_panel(ax2, da2, df2, 2, LAYERS,
                       f"{prefix}M=2, bs={m2_bs_str}, DA=210MHz, DF=210MHz — Layers 0-{LAYERS-1}")

    xmax = max(xmax1, xmax2) * 1.02
    ax1.set_xlim(-1, xmax)
    ax2.set_xlim(-1, xmax)
    ax2.set_xlabel("Time from forward start (ms)")

    legend_patches = [
        mpatches.Patch(color="#4472C4", label="Attention mb0"),
        mpatches.Patch(color="#8FAADC", label="Attention mb1"),
        mpatches.Patch(color="#ED7D31", label="FFN mb0"),
        mpatches.Patch(color="#F4B183", label="FFN mb1"),
        mpatches.Patch(color="#A9D18E", label="Recv / wait"),
        mpatches.Patch(color="#548235", label="DA→DF dep"),
        mpatches.Patch(color="#C00000", label="DF→DA dep"),
    ]
    fig.legend(handles=legend_patches, ncol=7, loc="lower center",
              fontsize=9, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
