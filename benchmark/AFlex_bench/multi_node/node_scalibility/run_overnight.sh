#!/usr/bin/env bash
# Overnight large-scale scalability sweep: 16 -> 8 -> 4 card.
# Each stage is independent (a failing stage does NOT abort the rest).
# summary (in=4096) is rate-limited to qps<=6 per plan.md §4 to avoid crashes.
set -u
cd /mnt/workspace/lt/sglang/benchmark/AFlex_bench/multi_node/node_scalibility
RL=run_logs
mkdir -p "$RL"
PY=python3

stamp() { date "+%F %T"; }
banner() { echo; echo "############################################################"; \
           echo "## [$(stamp)] $*"; echo "############################################################"; }

run() {  # $1=tag  rest=args
  local tag="$1"; shift
  banner "START $tag :: $*"
  $PY -u run_node_scalability.py "$@" > "$RL/stage_${tag}.log" 2>&1
  banner "END   $tag (exit=$?)"
}

banner "OVERNIGHT SWEEP BEGIN"

# ===== 16-card =====
# chatbot/qa/rag: qps up to 8 (qps10 has no workload file, would be skipped)
run 16_main --ngpu 16 --deploy all --mode all --scenario chatbot,qa,rag --qps 1,2,4,6,8
# summary: rate-limited to <=6
run 16_summary --ngpu 16 --deploy all --mode all --scenario summary --qps 1,2,4,6

# ===== 8-card =====
run 08_all --ngpu 8 --deploy all --mode all --scenario all --qps 1,2,3,4,5,6

# ===== 4-card =====
run 04_all --ngpu 4 --deploy all --mode all --scenario all --qps 1,2,3

banner "OVERNIGHT SWEEP COMPLETE"
echo "DONE_MARKER $(stamp)" > "$RL/overnight_DONE.flag"
