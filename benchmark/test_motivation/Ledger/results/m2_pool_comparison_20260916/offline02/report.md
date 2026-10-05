# M2 Pool Comparison: Offline Preflight

Status: `offline_preflight_passed_live_validation_pending`.

Per rank: state=21708800 B, KV=8064 B/token, context=1048576.

CPU/meta audit only. No model startup, kernel parity, full-HBM match, or performance result.

| Arm | Bootstrap state / KV | Resolved requests (offline) | Main allocation B | Request table B |
|---|---:|---:|---:|---:|
| static-natural | 485 / 1458643 | 485 | 22312982016 | 2038439520 |
| static-rebalanced | 607 / 1130211 | 607 | 22312979968 | 2550146560 |
| unified-raised-cap | 607 / 1130211 | 607 | 22291271168 | 2550146560 |
| unified-old-cap-control | 485 / 1458643 | 485 | 22291273216 | 2038439520 |

The main allocation column includes static KV padding or unified MLA tail padding.
Other auxiliary allocations still require a live storage census; this table is not full HBM.

Unified virtual state/KV maxima cannot be filled simultaneously. The unchanged-cap control
still resolves to 485 requests even though the byte pool is shared.

Next gate: live per-rank storage/memory audit and output correctness, then freeze budget-matched runs.
