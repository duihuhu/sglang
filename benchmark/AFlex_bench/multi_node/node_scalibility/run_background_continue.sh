#!/usr/bin/env bash
# 后台续跑脚本：等 pdaf_baseline rag 完成 → 更新图 → 续4卡
set -u
cd /mnt/workspace/lt/sglang/benchmark/AFlex_bench/multi_node/node_scalibility
PY=python3
RL=run_logs

stamp() { date "+%F %T"; }

echo "[$(stamp)] 等待 pdaf_baseline rag 重跑完成..."
while pgrep -f "run_node_scalability.py --ngpu 16" > /dev/null 2>&1; do sleep 30; done
echo "[$(stamp)] pdaf_baseline rag 重跑完成"

# 更新合并数据
echo "[$(stamp)] 重新合并 16 卡数据..."
$PY - <<'PY'
import json, glob
merged = {}; meta = None
for f in sorted(glob.glob("results/scal_16card_2026*.json")):
    d = json.load(open(f))
    meta = d.get("meta", meta)
    for sk, wl in d.get("results", {}).items():
        if not isinstance(wl, dict) or "__status__" in wl: continue
        dst = merged.setdefault(sk, {})
        for w, m in wl.items():
            if isinstance(m, dict) and m.get("status") == "PASS": dst[w] = m
meta = meta or {}
meta["ngpu_total"] = 16
meta["scenarios"] = ["chatbot", "qa", "rag", "summary"]
meta["qps"] = [1, 2, 4, 6, 8]
json.dump({"meta": meta, "results": merged}, open("results/scal_16card_merged_ALL.json", "w"), indent=2)
print("merged OK:", {k: len(v) for k, v in merged.items()})
PY

# 重画 Figure 4-7 + energy_per_token 总览
echo "[$(stamp)] 重画图表..."
$PY - <<'PY'
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, re
from pathlib import Path

data = json.load(open("results/scal_16card_merged_ALL.json"))
results = data["results"]
CHARTS = Path("charts/16card")
CHARTS.mkdir(exist_ok=True)

SCENARIOS_FIG = [
    ("qa",      "Figure 4", "HPHD", "in=512, out=256"),
    ("rag",     "Figure 5", "HPLD", "in=2048, out=64"),
    ("summary", "Figure 6", "HPLD (extreme)", "in=4096, out=64"),
    ("chatbot", "Figure 7", "LPHD", "in=128, out=1024"),
]
SCHEME_MODES = [
    ("native_dp","baseline","Native DP","#B0BEC5","o-"),
    ("native_dp","tier","Native DP+DVFS","#546E7A","o--"),
    ("pd_dp","baseline","PD","#64B5F6","s-"),
    ("pd_dp","tier","PD+DVFS","#1565C0","s--"),
    ("pdaf","baseline","PDAF","#81C784","^-"),
    ("pdaf","tier","PDAF+DVFS","#2E7D32","^--"),
]

def get_series(scenario, scheme, mode, metric):
    sk = f"{scheme}_baseline" if mode=="baseline" else f"{scheme}_tier"
    wl = results.get(sk, {})
    pts = {}
    for k, m in wl.items():
        mm = re.match(rf"{scenario}_qps(\d+)$", k)
        if mm and m.get("status")=="PASS" and m.get(metric) is not None:
            pts[int(mm.group(1))] = m[metric]
    xs = sorted(pts); return xs, [pts[q] for q in xs]

for scenario, fig_name, fig_type, fig_desc in SCENARIOS_FIG:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(f"{fig_name}: {fig_type} ({fig_desc}) — 16-card cross-node, Qwen3-32B", fontsize=13, fontweight="bold")
    ax = axes[0]
    for scheme, mode, label, color, mk in SCHEME_MODES:
        xs, ys = get_series(scenario, scheme, mode, "energy_per_token_mj")
        if xs: ax.plot(xs, ys, mk, color=color, linewidth=2, markersize=6, label=label)
    ax.set_xlabel("QPS"); ax.set_ylabel("Energy/token (mJ)"); ax.set_title("(a) Energy per Token", fontsize=11)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7, loc="upper right")
    ax = axes[1]
    for scheme, mode, label, color, mk in SCHEME_MODES:
        xs50, ys50 = get_series(scenario, scheme, mode, "ttft_proc_p50_ms")
        xs99, ys99 = get_series(scenario, scheme, mode, "ttft_proc_p99_ms")
        if xs50: ax.plot(xs50, ys50, mk, color=color, linewidth=2, markersize=5, label=f"{label} P50")
        if xs99:
            mk99 = mk.replace("-",":") if "--" not in mk else mk.replace("--",":")
            ax.plot(xs99, ys99, mk99, color=color, linewidth=1.2, markersize=4, alpha=0.7, label=f"{label} P99")
    ax.set_xlabel("QPS"); ax.set_ylabel("TTFT (ms)"); ax.set_title("(b) TTFT (P50 solid, P99 dotted)", fontsize=11)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=5.5, ncol=2, loc="upper left")
    ax = axes[2]
    for scheme, mode, label, color, mk in SCHEME_MODES:
        xs50, ys50 = get_series(scenario, scheme, mode, "tpot_p50_ms")
        xs99, ys99 = get_series(scenario, scheme, mode, "tpot_p99_ms")
        if xs50: ax.plot(xs50, ys50, mk, color=color, linewidth=2, markersize=5, label=f"{label} P50")
        if xs99:
            mk99 = mk.replace("-",":") if "--" not in mk else mk.replace("--",":")
            ax.plot(xs99, ys99, mk99, color=color, linewidth=1.2, markersize=4, alpha=0.7, label=f"{label} P99")
    ax.set_xlabel("QPS"); ax.set_ylabel("TPOT (ms)"); ax.set_title("(c) TPOT (P50 solid, P99 dotted)", fontsize=11)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=5.5, ncol=2, loc="upper left")
    plt.tight_layout()
    fname = f"{fig_name.replace(' ','_').lower()}_{fig_type.split()[0].lower()}_{scenario}.png"
    fig.savefig(CHARTS/fname, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {fname}")

# energy_per_token 4 datasets overview
SCENARIOS2 = [("qa","HPHD (in=512, out=256)"),("rag","HPLD (in=2048, out=64)"),
              ("summary","HPLD-extreme (in=4096, out=64)"),("chatbot","LPHD (in=128, out=1024)")]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("16-card cross-node: Energy per Token (mJ) vs QPS\nQwen3-32B | 6 schemes x 4 datasets", fontsize=14, fontweight="bold")
for idx, (scenario, title) in enumerate(SCENARIOS2):
    ax = axes[idx//2][idx%2]
    for scheme, mode, label, color, mk in SCHEME_MODES:
        xs, ys = get_series(scenario, scheme, mode, "energy_per_token_mj")
        if xs: ax.plot(xs, ys, mk, color=color, linewidth=2, markersize=6, label=label)
    ax.set_title(title, fontsize=11); ax.set_xlabel("QPS"); ax.set_ylabel("Energy/token (mJ)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7, loc="upper right")
plt.tight_layout()
fig.savefig(CHARTS/"energy_per_token_4datasets.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("  saved energy_per_token_4datasets.png")
PY

echo "[$(stamp)] 图表更新完成"

# 续跑 4 卡剩余
echo "[$(stamp)] 清理并启动 4 卡续跑..."
docker exec operator_test bash -lc "pkill -9 -f 'sglang'; sleep 3" 2>/dev/null
ssh -o StrictHostKeyChecking=no -o BatchMode=yes 10.252.129.35 "docker exec operator_test bash -lc 'pkill -9 -f sglang; sleep 3'" 2>/dev/null
sleep 5
$PY -u run_node_scalability.py --ngpu 4 --deploy pd_dp,pdaf --mode all --scenario all --qps 1,2,3 > "$RL/resume_04_remaining.log" 2>&1
echo "[$(stamp)] 4 卡续跑完成 (exit=$?)"

echo "ALL_BG_DONE $(stamp)" > "$RL/background_DONE.flag"
