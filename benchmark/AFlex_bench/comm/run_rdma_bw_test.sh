#!/usr/bin/env bash
# RDMA bandwidth/latency test between two nodes using perftest (ib_write_bw / ib_write_lat).
#
# Usage:
#   Server (run on remote, e.g. 10.252.129.35):
#     ./run_rdma_bw_test.sh server
#     ./run_rdma_bw_test.sh server --multi          # all 4 IB ports
#
#   Client (run on local, e.g. 10.252.129.36):
#     ./run_rdma_bw_test.sh client 10.252.129.35
#     ./run_rdma_bw_test.sh client 10.252.129.35 --multi
#     ./run_rdma_bw_test.sh client 10.252.129.35 --wait 120   # retry until server up
#
#   Latency (small message):
#     ./run_rdma_bw_test.sh server --lat
#     ./run_rdma_bw_test.sh client 10.252.129.35 --lat

set -euo pipefail

# RDMA 需要 pin 内存；Docker 容器默认 memlock 往往只有 64KB，会导致 syndrom 0x88
ulimit -l unlimited 2>/dev/null || true

# 4x ConnectX-6 HDR IB HCA (skip mlx5_bond_0 which is RoCE/Ethernet)
DEVICES=(mlx5_0 mlx5_1 mlx5_4 mlx5_5)
BASE_PORT=18515
MSG_SIZE=1048576   # 1MB, near saturation for HDR
ITERS=5000
MTU=4096
WAIT_SEC=0
MODE="bw"          # bw | lat
MULTI=0

log() { echo "[$(date '+%H:%M:%S')] $*"; }

usage() {
    sed -n '3,16p' "$0" | sed 's/^# \?//'
    exit 1
}

[[ $# -lt 1 ]] && usage

ROLE=$1; shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --multi) MULTI=1 ;;
        --wait)  WAIT_SEC=${2:?}; shift ;;
        --lat)   MODE="lat" ;;
        -h|--help) usage ;;
        *) REMOTE=$1 ;;
    esac
    shift
done

check_tools() {
    for cmd in ib_write_bw ib_write_lat ibstat; do
        command -v "$cmd" >/dev/null || { echo "Missing $cmd. Install: apt-get install -y perftest infiniband-diags"; exit 1; }
    done
}

show_devices() {
    log "Local IB devices:"
    ibstat | grep -E "CA '|Rate:|Base lid:|Link layer:" | paste - - - - || true
}

wait_for_port() {
    local host=$1 port=$2 timeout=$3
    local elapsed=0
    while (( elapsed < timeout )); do
        if timeout 1 bash -c "echo >/dev/tcp/${host}/${port}" 2>/dev/null; then
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
        log "Waiting for ${host}:${port} ... (${elapsed}s/${timeout}s)"
    done
    return 1
}

run_server_one() {
    local dev=$1 port=$2
    if [[ "$MODE" == "lat" ]]; then
        ib_write_lat -d "$dev" -p "$port" -m "$MTU" -F
    else
        ib_write_bw -d "$dev" -p "$port" -m "$MTU" -F
    fi
}

run_client_one() {
    local dev=$1 port=$2 remote=$3
    local out="/tmp/rdma_${dev}_${port}.log"
    if [[ "$MODE" == "lat" ]]; then
        ib_write_lat -d "$dev" -p "$port" -m "$MTU" -F "$remote" 2>&1 | tee "$out"
    else
        ib_write_bw -d "$dev" -p "$port" -m "$MTU" -s "$MSG_SIZE" -n "$ITERS" -F "$remote" 2>&1 | tee "$out"
    fi
}

parse_bw_mbps() {
    # Extract average BW[MB/sec] from ib_write_bw output (last numeric column before MsgRate)
    awk '/^[[:space:]]+[0-9]+[[:space:]]+[0-9]+/ {
        for (i=NF; i>=1; i--) if ($i ~ /^[0-9]+\.?[0-9]*$/) { print $(i-1); exit }
    }' "$1" 2>/dev/null || true
}

summarize_multi() {
    local total=0 count=0
    log "=== Per-port results ==="
    for i in "${!DEVICES[@]}"; do
        local dev=${DEVICES[$i]}
        local port=$((BASE_PORT + i))
        local f="/tmp/rdma_${dev}_${port}.log"
        if [[ -f "$f" ]]; then
            local mbps
            mbps=$(parse_bw_mbps "$f")
            if [[ -n "$mbps" ]]; then
                local gbs
                gbs=$(awk -v m="$mbps" 'BEGIN{printf "%.2f", m/1024}')
                log "  $dev (port $port): ${mbps} MB/s = ${gbs} GB/s"
                total=$(awk -v a="$total" -v b="$mbps" 'BEGIN{print a+b}')
                count=$((count + 1))
            else
                log "  $dev (port $port): FAILED (see $f)"
            fi
        fi
    done
    if (( count > 0 )); then
        local total_gbs
        total_gbs=$(awk -v t="$total" 'BEGIN{printf "%.2f", t/1024}')
        local total_gbps
        total_gbps=$(awk -v g="$total_gbs" 'BEGIN{printf "%.1f", g*8}')
        log "=== Aggregate ($count ports) ==="
        log "  Total: ${total} MB/s = ${total_gbs} GB/s (~${total_gbps} Gb/s)"
    fi
}

cleanup() {
    jobs -p 2>/dev/null | xargs -r kill 2>/dev/null || true
}
trap cleanup EXIT

check_tools
show_devices

if [[ "$ROLE" == "server" ]]; then
    if (( MULTI )); then
        log "Starting ${#DEVICES[@]}-port RDMA ${MODE} server ..."
        for i in "${!DEVICES[@]}"; do
            run_server_one "${DEVICES[$i]}" "$((BASE_PORT + i))" &
            log "  ${DEVICES[$i]} listening on TCP port $((BASE_PORT + i))"
        done
        wait
    else
        log "Starting single-port server on ${DEVICES[0]} (port ${BASE_PORT}) ..."
        run_server_one "${DEVICES[0]}" "$BASE_PORT"
    fi

elif [[ "$ROLE" == "client" ]]; then
    REMOTE=${REMOTE:?Usage: $0 client <remote_ip> [--multi] [--wait N]}
    if (( WAIT_SEC > 0 )); then
        wait_for_port "$REMOTE" "$BASE_PORT" "$WAIT_SEC" || {
            log "ERROR: server not reachable at ${REMOTE}:${BASE_PORT} within ${WAIT_SEC}s"
            log "On ${REMOTE}, run: $0 server [--multi]"
            exit 1
        }
    fi

    if (( MULTI )); then
        log "Running ${#DEVICES[@]}-port RDMA ${MODE} client -> ${REMOTE} ..."
        for i in "${!DEVICES[@]}"; do
            run_client_one "${DEVICES[$i]}" "$((BASE_PORT + i))" "$REMOTE" &
        done
        wait
        if [[ "$MODE" == "bw" ]]; then
            summarize_multi
        fi
    else
        log "Running single-port RDMA ${MODE} client -> ${REMOTE} (${DEVICES[0]}) ..."
        run_client_one "${DEVICES[0]}" "$BASE_PORT" "$REMOTE"
    fi
else
    usage
fi
