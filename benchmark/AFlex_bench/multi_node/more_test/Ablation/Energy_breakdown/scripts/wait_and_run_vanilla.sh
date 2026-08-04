#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
while pgrep -f "run_aflex_e2e_no_dvfs.py" >/dev/null; do sleep 30; done
python3 force_cleanup_cluster.py
python3 run_vanilla_1p1d_tp4_no_dvfs.py --resume >> "${ROOT}/logs/vanilla_1p1d_tp4_no_dvfs.log" 2>&1
python3 merge_tier_perf_data.py >> "${ROOT}/logs/merge_tier_perf_data.log" 2>&1
python3 "${ROOT}/charts/plot_tier_perf.py" >> "${ROOT}/logs/plot_tier_perf.log" 2>&1
