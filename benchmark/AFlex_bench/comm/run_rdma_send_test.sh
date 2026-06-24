#!/usr/bin/env bash
# RDMA Send/Recv benchmark suite (matches sglang UCX send/recv semantics).
#
# sglang AFD uses UCX endpoint.send()/recv() over IB RC — NOT one-sided RDMA Write.
# Use ib_send_bw / ib_send_lat (not ib_write_*).
#
# Usage:
#   Step 1 — on server node (.35):
#     ./run_rdma_send_test.sh server
#
#   Step 2 — on client node (.36):
#     ./run_rdma_send_test.sh client 10.252.129.35
#     ./run_rdma_send_test.sh client 10.252.129.35 --outdir /tmp/rdma_results
#
# Tests run in order (server & client must use same script):
#   1. ib_send_lat  -a       single port mlx5_0  (latency sweep)
#   2. ib_send_bw   -a       single port         (bandwidth sweep, unidirectional)
#   3. ib_send_bw   -a -b    single port         (bandwidth sweep, bidirectional)
#   4. ib_send_bw   -a       4 ports parallel    (aggregate bandwidth sweep)

set -euo pipefail

DEVICES=(mlx5_0 mlx5_1 mlx5_4 mlx5_5)
DEV0=mlx5_0
BASE_PORT=18515
MTU=4096
WAIT_SEC=180
OUTDIR=""
REMOTE=""

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
die()  { log "ERROR: $*"; exit 1; }

usage() {
    cat <<'EOF'
Usage:
  ./run_rdma_send_test.sh server
  ./run_rdma_send_test.sh client <remote_ip> [--outdir DIR] [--wait SEC]

Run server on .35 first, then client on .36.
Requires: perftest (ib_send_bw, ib_send_lat), ulimit -l unlimited.
EOF
    exit 1
}

setup() {
    ulimit -l unlimited 2>/dev/null || true
    if [[ "$(ulimit -l 2>/dev/null || echo 0)" != "unlimited" && "$(ulimit -l 2>/dev/null || echo 0)" -lt 65536 ]]; then
        log "WARN: memlock=$(ulimit -l); run 'ulimit -l unlimited' or use --ulimit memlock=-1:-1 in Docker"
    fi
    for cmd in ib_send_bw ib_send_lat ibstat; do
        command -v "$cmd" >/dev/null || die "Missing $cmd. Install: apt-get install -y perftest"
    done
}

kill_stale() {
    pkill -9 -f 'ib_send_bw|ib_send_lat' 2>/dev/null || true
    sleep 1
}

wait_for_port() {
    local host=$1 port=$2 timeout=${3:-$WAIT_SEC}
    local elapsed=0
    while (( elapsed < timeout )); do
        if timeout 1 bash -c "echo >/dev/tcp/${host}/${port}" 2>/dev/null; then
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
        if (( elapsed % 10 == 0 )); then
            log "  waiting ${host}:${port} ... (${elapsed}s/${timeout}s)"
        fi
    done
    return 1
}

wait_for_ports() {
    local host=$1 timeout=${2:-$WAIT_SEC}
    shift 2
    local ports=("$@")
    local elapsed=0
    while (( elapsed < timeout )); do
        local all=1
        for p in "${ports[@]}"; do
            if ! timeout 1 bash -c "echo >/dev/tcp/${host}/${p}" 2>/dev/null; then
                all=0
                break
            fi
        done
        if (( all )); then
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
        if (( elapsed % 10 == 0 )); then
            log "  waiting ports ${ports[*]} on ${host} ... (${elapsed}s/${timeout}s)"
        fi
    done
    return 1
}

show_devices() {
    log "Local IB devices:"
    ibstat 2>/dev/null | grep -E "CA '|Rate:|Base lid:" | paste - - - || true
}

# ---- server ----

run_server_lat() {
    log ">>> [1/4] SERVER: ib_send_lat -a (${DEV0}, port ${BASE_PORT})"
    ib_send_lat -d "$DEV0" -p "$BASE_PORT" -m "$MTU" -a -F
    log "<<< [1/4] done"
}

run_server_bw_uni() {
    log ">>> [2/4] SERVER: ib_send_bw -a unidirectional (${DEV0}, port ${BASE_PORT})"
    ib_send_bw -d "$DEV0" -p "$BASE_PORT" -m "$MTU" -a -F
    log "<<< [2/4] done"
}

run_server_bw_bi() {
    log ">>> [3/4] SERVER: ib_send_bw -a -b bidirectional (${DEV0}, port ${BASE_PORT})"
    ib_send_bw -d "$DEV0" -p "$BASE_PORT" -m "$MTU" -a -b -F
    log "<<< [3/4] done"
}

run_server_multi() {
    log ">>> [4/4] SERVER: ib_send_bw -a 4-port parallel (ports ${BASE_PORT}-$((BASE_PORT+3)))"
    local pids=()
    for i in "${!DEVICES[@]}"; do
        local dev=${DEVICES[$i]}
        local port=$((BASE_PORT + i))
        ib_send_bw -d "$dev" -p "$port" -m "$MTU" -a -F &
        pids+=($!)
        log "  listening ${dev} port ${port} pid=$!"
    done
    local fail=0
    for pid in "${pids[@]}"; do
        wait "$pid" || fail=1
    done
    log "<<< [4/4] done (exit=${fail})"
    return "$fail"
}

run_server_all() {
    setup
    kill_stale
    show_devices
    log "=== RDMA Send/Recv server suite — waiting for client ==="
    log "Run on client: $0 client <this_host_ip>"
    run_server_lat
    sleep 2
    run_server_bw_uni
    sleep 2
    run_server_bw_bi
    sleep 2
    run_server_multi
    log "=== All server tests complete ==="
}

# ---- client ----

run_client_lat() {
    local remote=$1 out=$2
    log ">>> [1/4] CLIENT: ib_send_lat -a -> ${remote}"
    wait_for_port "$remote" "$BASE_PORT" || die "server not ready for send_lat on ${remote}:${BASE_PORT}"
    ib_send_lat -d "$DEV0" -p "$BASE_PORT" -m "$MTU" -a -F "$remote" 2>&1 | tee "$out/send_lat.log"
    log "<<< [1/4] done"
}

run_client_bw_uni() {
    local remote=$1 out=$2
    log ">>> [2/4] CLIENT: ib_send_bw -a unidirectional -> ${remote}"
    wait_for_port "$remote" "$BASE_PORT" || die "server not ready for send_bw_uni"
    ib_send_bw -d "$DEV0" -p "$BASE_PORT" -m "$MTU" -a -F "$remote" 2>&1 | tee "$out/send_bw_uni.log"
    log "<<< [2/4] done"
}

run_client_bw_bi() {
    local remote=$1 out=$2
    log ">>> [3/4] CLIENT: ib_send_bw -a -b bidirectional -> ${remote}"
    wait_for_port "$remote" "$BASE_PORT" || die "server not ready for send_bw_bi"
    ib_send_bw -d "$DEV0" -p "$BASE_PORT" -m "$MTU" -a -b -F "$remote" 2>&1 | tee "$out/send_bw_bi.log"
    log "<<< [3/4] done"
}

run_client_multi() {
    local remote=$1 out=$2
    log ">>> [4/4] CLIENT: ib_send_bw -a 4-port -> ${remote}"
    local ports=()
    for i in "${!DEVICES[@]}"; do
        ports+=($((BASE_PORT + i)))
    done
    wait_for_ports "$remote" "$WAIT_SEC" "${ports[@]}" || die "server not ready for 4-port test"

    local pids=()
    for i in "${!DEVICES[@]}"; do
        local dev=${DEVICES[$i]}
        local port=$((BASE_PORT + i))
        ib_send_bw -d "$dev" -p "$port" -m "$MTU" -a -F "$remote" \
            > "$out/send_4port_${dev}.log" 2>&1 &
        pids+=($!)
        log "  client ${dev} port ${port} pid=$!"
    done
    local fail=0
    for pid in "${pids[@]}"; do
        wait "$pid" || fail=1
    done
    log "<<< [4/4] done (exit=${fail})"
    return "$fail"
}

summarize_results() {
    local out=$1
    local summary="$out/summary.txt"
    {
        echo "RDMA Send/Recv benchmark summary"
        echo "Generated: $(date -Iseconds)"
        echo ""

        echo "=== [1] Latency (ib_send_lat, single port) ==="
        if [[ -f "$out/send_lat.log" ]]; then
            grep -E '^ [0-9]' "$out/send_lat.log" | awk '
                NR<=5 || NR==6 { print }
                END {
                    # print last line (largest size)
                }
            ' 2>/dev/null | head -3
            echo "  ..."
            grep -E '^ [0-9]' "$out/send_lat.log" | tail -3
            local lat2
            lat2=$(grep -E '^[[:space:]]+2[[:space:]]' "$out/send_lat.log" | awk '{print $6}')
            [[ -n "$lat2" ]] && echo "  2B latency: ${lat2} us"
        else
            echo "  MISSING send_lat.log"
        fi
        echo ""

        echo "=== [2] Bandwidth unidirectional (ib_send_bw -a) ==="
        if [[ -f "$out/send_bw_uni.log" ]]; then
            local peak
            peak=$(grep -E '^ [0-9]' "$out/send_bw_uni.log" | awk '{if($4>max) max=$4} END{printf "%.2f GB/s", max/1024}')
            echo "  peak: ${peak}"
            grep -E '^ [0-9]' "$out/send_bw_uni.log" | tail -3
        else
            echo "  MISSING send_bw_uni.log"
        fi
        echo ""

        echo "=== [3] Bandwidth bidirectional (ib_send_bw -a -b) ==="
        if [[ -f "$out/send_bw_bi.log" ]]; then
            local peak
            peak=$(grep -E '^ [0-9]' "$out/send_bw_bi.log" | awk '{if($4>max) max=$4} END{printf "%.2f GB/s", max/1024}')
            echo "  peak: ${peak}"
            grep -E '^ [0-9]' "$out/send_bw_bi.log" | tail -3
        else
            echo "  MISSING send_bw_bi.log"
        fi
        echo ""

        echo "=== [4] 4-port aggregate (ib_send_bw -a) ==="
        if ls "$out"/send_4port_*.log &>/dev/null; then
            echo "  Per-port peak @ saturation:"
            for f in "$out"/send_4port_*.log; do
                local dev peak_mb
                dev=$(basename "$f" .log | sed 's/send_4port_//')
                peak_mb=$(grep -E '^ [0-9]' "$f" | awk '{if($4>max) max=$4} END{print max+0}')
                printf "    %-8s %8.0f MB/s  (%6.2f GB/s)\n" "$dev" "$peak_mb" "$(awk -v m="$peak_mb" 'BEGIN{printf "%.2f", m/1024}')"
            done
            echo "  Aggregate by message size (sum of 4 ports):"
            grep -h '^ [0-9]' "$out"/send_4port_*.log | \
                awk '{size=$1; sum[size]+=$4} END{for(s in sum) printf "    %10d  %8.2f GB/s\n", s, sum[s]/1024}' | sort -n | tail -5
        else
            echo "  MISSING send_4port_*.log"
        fi
    } | tee "$summary"
    log "Summary written to $summary"
}

run_client_all() {
    local remote=$1
    setup
    kill_stale
    show_devices

    local out="${OUTDIR:-$(dirname "$0")/results/send_$(date +%Y%m%d_%H%M%S)}"
    mkdir -p "$out"
    log "Results -> $out"

    log "=== RDMA Send/Recv client suite -> ${remote} ==="
    run_client_lat   "$remote" "$out"
    sleep 2
    run_client_bw_uni "$remote" "$out"
    sleep 2
    run_client_bw_bi  "$remote" "$out"
    sleep 2
    run_client_multi  "$remote" "$out" || true

    summarize_results "$out"
    log "=== All client tests complete ==="
    log "Logs in: $out"
}

# ---- main ----

[[ $# -lt 1 ]] && usage

ROLE=$1
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --outdir) OUTDIR=${2:?}; shift ;;
        --wait)   WAIT_SEC=${2:?}; shift ;;
        -h|--help) usage ;;
        *) REMOTE=$1 ;;
    esac
    shift
done

trap kill_stale EXIT

case "$ROLE" in
    server) run_server_all ;;
    client)
        REMOTE=${REMOTE:?Usage: $0 client <remote_ip>}
        run_client_all "$REMOTE"
        ;;
    *) usage ;;
esac
