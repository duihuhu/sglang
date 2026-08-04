#!/bin/bash
# Kill all sglang/router processes for the multi-node PDAF benchmark.
# Runs INSIDE the operator_test container. Kills by process pattern AND by port
# (the rust router renames itself to "sglang::router", so pkill -f misses it).
pkill -9 -f sglang.launch_server 2>/dev/null
pkill -9 -f launch_router 2>/dev/null
pkill -9 -x sglang::router 2>/dev/null
pkill -9 -f sglang::schedul 2>/dev/null
pkill -9 -f sglang::detokenizer 2>/dev/null
pkill -9 -f sglang_router 2>/dev/null

# Managed port ranges across all architectures:
#   42000-42029  router + PDAF servers
#   43000-43399  Tier1 router + AFD prefill/decode (deploy_tier1_layout)
#   44000-44029  Tier1 router
#   45000-45069  Tier1 sub-routers
#   45300-45369  shared-PA MVP servers
#   53100-53179  PD DP prefill/decode
#   53200-53279  Native DP instances
#   49100-49179  PD DP bootstrap
#   49999        PDAF bootstrap
#   33300-33379  Native DP nccl
#   34000-34089  PD DP nccl
#   37300-37509  Tier1 AFD nccl (4P+4D TP1 needs up to 37450)
#   39411-39441  PDAF nccl
PORTS=""
for r in $(seq 42000 42029) $(seq 43000 43399) $(seq 44000 44029) \
         $(seq 45000 45069) $(seq 45300 45369) \
         $(seq 53100 53189) $(seq 53200 53279) \
         $(seq 49100 49179) 49999 $(seq 33300 33379) $(seq 34000 34089) \
         $(seq 37300 37509) \
         39411 39421 39431 39441; do
  PORTS="$PORTS $r"
done
for p in $PORTS; do
  for pid in $(ss -tlnp "sport = :$p" 2>/dev/null | grep -oP 'pid=\K[0-9]+'); do
    kill -9 "$pid" 2>/dev/null
  done
done
# Fallback: kill any listener in benchmark port bands even if ss omits pid=
for p in $(ss -tlnp 2>/dev/null | grep -oP ':\K(43[0-9]{3}|45[0-9]{3})' | sort -u); do
  fuser -k "${p}/tcp" 2>/dev/null
done
exit 0
