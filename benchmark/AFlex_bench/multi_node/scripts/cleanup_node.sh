#!/bin/bash
# Kill all sglang/router processes for the multi-node PDAF benchmark.
# Runs INSIDE the operator_test container. Kills by process pattern AND by port
# (the rust router renames itself to "sglang::router", so pkill -f misses it).
pkill -9 -f sglang.launch_server 2>/dev/null
pkill -9 -f launch_router 2>/dev/null
pkill -9 -x sglang::router 2>/dev/null
pkill -9 -f sglang::schedul 2>/dev/null
pkill -9 -f sglang_router 2>/dev/null

# Managed port ranges across all architectures:
#   42000-42029  router + PDAF servers
#   53100-53179  PD DP prefill/decode
#   53200-53279  Native DP instances
#   49100-49179  PD DP bootstrap
#   49999        PDAF bootstrap
#   33300-33379  Native DP nccl
#   34000-34079  PD DP nccl
#   39411-39441  PDAF nccl
PORTS=""
for r in $(seq 42000 42029) $(seq 53100 53179) $(seq 53200 53279) \
         $(seq 49100 49179) 49999 $(seq 33300 33379) $(seq 34000 34079) \
         39411 39421 39431 39441; do
  PORTS="$PORTS $r"
done
for p in $PORTS; do
  for pid in $(ss -tlnp "sport = :$p" 2>/dev/null | grep -oP 'pid=\K[0-9]+'); do
    kill -9 "$pid" 2>/dev/null
  done
done
exit 0
