#!/usr/bin/env bash
# Run in-place reshard timeline benchmarks at fixed QPS levels.
set -euo pipefail
ROOT=/workspace/sglang/benchmark/AFlex_bench/reshard/Baseline
SCRIPTS="$ROOT/scripts"
RESULTS="$ROOT/results/timeline_$(date +%Y%m%d_%H%M%S)"
SEED="$ROOT/results/macro_code_qps240.jsonl"
mkdir -p "$RESULTS"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0

restart_server() {
  python3 "$SCRIPTS/inplace_reshard_server_ctl.py" || true
  sleep 3
  cd /workspace/sglang
  nohup python3 -m sglang.launch_server \
    --model-path /models/Qwen3-32B --tp 1 --inplace-reshard-max-tp 8 \
    --host 127.0.0.1 --port 31700 --nccl-port 32800 \
    --mem-fraction-static 0.85 --disable-cuda-graph --disable-piecewise-cuda-graph \
    --skip-server-warmup --base-gpu-id 0 --attention-backend triton \
    > /tmp/sglang_timeline_bench.log 2>&1 &
  for i in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:31700/health || echo 000)
    if [ "$code" = "200" ]; then echo "server ready"; return 0; fi
    sleep 5
  done
  echo "server failed to start" >&2
  exit 1
}

run_case() {
  local name=$1 qps=$2 duration=$3 plan_style=$4
  local wl="$RESULTS/workload_${name}.jsonl"
  local plan_file="$RESULTS/workload_${name}.reshard_plan.json"
  local out="$RESULTS/result_${name}.json"

  echo "========== $name (qps=$qps) =========="
  python3 "$SCRIPTS/generate_fixed_qps_workload.py" \
    --seed-workload "$SEED" --qps "$qps" --duration-s "$duration" \
    --label "$name" --plan-style "$plan_style" \
    --output "$wl" --plan-out "$plan_file"

  restart_server

  PLAN=$(cat "$plan_file")
  python3 "$SCRIPTS/bench_inplace_reshard_real_workload.py" \
    --workload "$wl" \
    --base http://127.0.0.1:31700 \
    --reshard-plan "$PLAN" \
    --reshard-sequential \
    --max-workers 64 \
    --timeout 180 \
    --reshard-status-timeout 600 \
    --keep-server \
    --out "$out"

  python3 - <<PY
import json
from pathlib import Path
p = Path("$out")
d = json.loads(p.read_text())
tl = sorted(d.get("timeline", []), key=lambda r: r["issue_s"])
ev = d.get("reshard_events", [])
print(f"--- summary {p.name} ---")
print(f"  requests: {len(tl)} ok={sum(1 for r in tl if r.get('ok'))} slo_pass={sum(1 for r in tl if r.get('slo_pass'))}")
for e in ev:
    print(f"  reshard TP{e['new_tp']}: trigger={e['trigger_s']}s pause={e.get('pause_s')}s phase={e.get('phase')}")
# requests during each reshard window
for e in ev:
    if e.get('done_s') is None:
        continue
    rs, re = float(e['trigger_s']), float(e['done_s'])
    win = [r for r in tl if rs <= float(r['issue_s']) <= re]
    ok = sum(1 for r in win if r.get('ok'))
    slo = sum(1 for r in win if r.get('slo_pass'))
    print(f"  window TP{e['new_tp']} [{rs:.1f}-{re:.1f}s]: {len(win)} reqs issued, ok={ok}, slo={slo}")
    for r in win[:8]:
        print(f"    issue={r['issue_s']:.1f}s tp={r['tp_at_issue']} ttft={r.get('ttft_s')}s ok={r.get('ok')} slo={r.get('slo_pass')}")
    if len(win) > 8:
        print(f"    ... +{len(win)-8} more")
PY
}

# QPS=1 (B-round style)
run_case qps1 1.0 85 qps1

# QPS=2/3 ≈ 0.667
run_case qps2_3 0.6667 90 qps1

# QPS=3 (C-round style, plan compressed)
run_case qps3 3.0 55 qps3

echo "Results in $RESULTS"
ls -la "$RESULTS"
