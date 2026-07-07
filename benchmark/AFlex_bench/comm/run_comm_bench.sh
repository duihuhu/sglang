#!/usr/bin/env bash
# Orchestrated cross-node communication benchmark suite.
# Runs from node1 host, coordinates both nodes via ssh + docker exec.
#
# Tests:
#   1. NVLink baseline (same-node GPU P2P, on node1)
#   2. Mooncake RDMA Write - single NIC (mlx5_bond_0, cross-node)
#   3. Mooncake RDMA Write - 4 NIC aggregate (mlx5_0/1/4/5, cross-node)
#   4. Mooncake RDMA Write - batch mode (simulates multi-layer KV transfer)
#
# Usage (on node1 host):
#   cd /mnt/workspace/lt/sglang/benchmark/AFlex_bench/comm
#   bash run_comm_bench.sh [--quick]
#
# Requires: mooncake installed in both containers, RDMA working.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
BENCH_SCRIPT="benchmark/AFlex_bench/comm/bench_mooncake_rdma_write.py"

NODE1_IP="${MN_NODE1_IP:-10.252.129.36}"
NODE2_IP="${MN_NODE2_IP:-10.252.129.35}"
CONTAINER="${MN_CONTAINER:-operator_test}"
IB_DEV="${MN_IB_DEV:-mlx5_bond_0}"
IB_MULTI="${MN_IB_MULTI:-mlx5_0,mlx5_1,mlx5_4,mlx5_5}"

ITERS=200
WARMUP=50
QUICK=0

log() { echo "[$(date '+%H:%M:%S')] $*"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick) QUICK=1; ITERS=50; WARMUP=20 ;;
        --iters) ITERS=$2; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

n1_exec() {
    docker exec "$CONTAINER" bash -c "cd /mnt/workspace/lt/sglang && $1"
}

n2_exec() {
    ssh "$NODE2_IP" "docker exec $CONTAINER bash -c 'cd /mnt/workspace/lt/sglang && $1'"
}

n2_exec_bg() {
    ssh "$NODE2_IP" "docker exec -d $CONTAINER bash -c 'cd /mnt/workspace/lt/sglang && $1'" &
}

kill_bench() {
    n1_exec "pkill -f bench_mooncake_rdma_write || true" 2>/dev/null || true
    n2_exec "pkill -f bench_mooncake_rdma_write || true" 2>/dev/null || true
    sleep 2
}

# ──────────────────────────────────────────────────────────────────────────────
# Test 1: NVLink baseline (same node, GPU0 -> GPU4)
# ──────────────────────────────────────────────────────────────────────────────
run_nvlink() {
    log "=== [1/5] NVLink P2P: GPU0 -> GPU4 (node1) ==="
    local out="${RESULTS_DIR}/nvlink_${TIMESTAMP}.json"
    n1_exec "python3 ${BENCH_SCRIPT} nvlink \
        --src-gpu 0 --dst-gpu 4 \
        --sizes default --iters ${ITERS} --warmup ${WARMUP} \
        --outfile ${BENCH_SCRIPT%/*}/results/nvlink_${TIMESTAMP}.json"
    log "NVLink results: $out"
}

# ──────────────────────────────────────────────────────────────────────────────
# Test 2: Mooncake RDMA Write - single NIC (mlx5_bond_0)
# ──────────────────────────────────────────────────────────────────────────────
run_rdma_single() {
    log "=== [2/5] Mooncake RDMA Write: single NIC (${IB_DEV}) ==="
    kill_bench

    # Start receiver on node2
    n2_exec_bg "python3 ${BENCH_SCRIPT} receiver \
        --gpu 0 --ib-device ${IB_DEV} \
        --sizes default --iters ${ITERS} --warmup ${WARMUP}"
    sleep 5

    # Run sender on node1
    n1_exec "python3 ${BENCH_SCRIPT} sender \
        --gpu 0 --ib-device ${IB_DEV} \
        --remote-host ${NODE2_IP} \
        --sizes default --iters ${ITERS} --warmup ${WARMUP} \
        --outfile ${BENCH_SCRIPT%/*}/results/rdma_single_${TIMESTAMP}.json"

    sleep 2
    kill_bench
    log "Single NIC results saved."
}

# ──────────────────────────────────────────────────────────────────────────────
# Test 3: Mooncake RDMA Write - 4 NIC aggregate
# ──────────────────────────────────────────────────────────────────────────────
run_rdma_multi() {
    log "=== [3/5] Mooncake RDMA Write: 4 NIC aggregate ==="
    kill_bench

    # Start multi-NIC receiver on node2
    n2_exec_bg "python3 ${BENCH_SCRIPT} receiver \
        --multi-nic 4 --ib-devices ${IB_MULTI} \
        --sizes default --iters ${ITERS} --warmup ${WARMUP}"
    sleep 8

    # Run multi-NIC sender on node1
    n1_exec "python3 ${BENCH_SCRIPT} sender \
        --multi-nic 4 --ib-devices ${IB_MULTI} \
        --remote-host ${NODE2_IP} \
        --sizes default --iters ${ITERS} --warmup ${WARMUP} \
        --outfile ${BENCH_SCRIPT%/*}/results/rdma_4nic_${TIMESTAMP}.json"

    sleep 2
    kill_bench
    log "4-NIC aggregate results saved."
}

# ──────────────────────────────────────────────────────────────────────────────
# Test 4: Mooncake RDMA Write - batch mode (multi-layer simulation)
# batch_count=64 simulates sending 64 layers' KV data in one batch call
# ──────────────────────────────────────────────────────────────────────────────
run_rdma_batch() {
    log "=== [4/5] Mooncake RDMA Write: batch mode (64 layers) ==="
    kill_bench

    n2_exec_bg "python3 ${BENCH_SCRIPT} receiver \
        --gpu 0 --ib-device ${IB_DEV} \
        --sizes default --iters ${ITERS} --warmup ${WARMUP} \
        --batch-count 64"
    sleep 5

    n1_exec "python3 ${BENCH_SCRIPT} sender \
        --gpu 0 --ib-device ${IB_DEV} \
        --remote-host ${NODE2_IP} \
        --sizes default --iters ${ITERS} --warmup ${WARMUP} \
        --use-batch --batch-count 64 \
        --outfile ${BENCH_SCRIPT%/*}/results/rdma_batch64_${TIMESTAMP}.json"

    sleep 2
    kill_bench
    log "Batch mode results saved."
}

# ──────────────────────────────────────────────────────────────────────────────
# Test 5: AFD-specific sizes (activation tensor sizes)
# ──────────────────────────────────────────────────────────────────────────────
run_rdma_afd() {
    log "=== [5/5] Mooncake RDMA Write: AFD activation sizes ==="
    kill_bench

    n2_exec_bg "python3 ${BENCH_SCRIPT} receiver \
        --gpu 0 --ib-device ${IB_DEV} \
        --sizes afd --iters ${ITERS} --warmup ${WARMUP}"
    sleep 5

    n1_exec "python3 ${BENCH_SCRIPT} sender \
        --gpu 0 --ib-device ${IB_DEV} \
        --remote-host ${NODE2_IP} \
        --sizes afd --iters ${ITERS} --warmup ${WARMUP} \
        --outfile ${BENCH_SCRIPT%/*}/results/rdma_afd_${TIMESTAMP}.json"

    sleep 2
    kill_bench
    log "AFD sizes results saved."
}

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
log "Starting communication benchmark suite (timestamp=${TIMESTAMP})"
log "Node1=${NODE1_IP}, Node2=${NODE2_IP}, Container=${CONTAINER}"
log "IB single=${IB_DEV}, IB multi=${IB_MULTI}"
log "Iters=${ITERS}, Warmup=${WARMUP}"
echo ""

run_nvlink
run_rdma_single
run_rdma_multi
run_rdma_batch
run_rdma_afd

log "=== All tests complete ==="
log "Results in: ${RESULTS_DIR}/"
log "Run 'python3 ${SCRIPT_DIR}/plot_comm_comparison.py' to generate comparison charts."
