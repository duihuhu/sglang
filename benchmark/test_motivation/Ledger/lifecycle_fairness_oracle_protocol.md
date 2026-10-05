# Lifecycle-Cost Fairness Oracle Protocol

This experiment compares token VTC, tuned scalar VTC, and a realized
lifecycle-cost fairness oracle on A800 GPUs 6 and 7. Parameter selection and
confirmation use disjoint runs.

## Workload

The server uses `state=48` and `KV=196608`. Eight waves arrive 1.5 seconds
apart. In every wave:

- tenant `split`: 16 requests with `P=1024,O=32`;
- tenant `packed`: one request with `P=16384,O=512`.

Both tenants offer exactly 16,896 input-plus-output tokens per wave. Isolation
runs use the same arrival timestamps and request shapes.

## Policies

- Token VTC: input weight 1, output weight 1, request weight 0.
- Tuned scalar VTC: the same token weights plus request weight in
  `{0,1024,4096,8192,12288,16384,32768,65536}`. The token VTC run is also the
  scalar sweep's zero-weight sample.
- Realized lifecycle cost: order tenants by
  `w_s * state_slot_seconds / state_capacity +
  w_k * KV_token_seconds / KV_capacity`, using exact scheduler-side accounting.
  Fix `w_k=1` and scan `w_s={0,0.25,0.5,1,2,4,8,16}`.

The best throughput-bounded scalar and lifecycle candidates minimize the
absolute gap between each tenant's p95 JCT slowdown relative to isolation.
Calibration candidates whose makespan exceeds token VTC by more than 5% are
ineligible. The selected weights are frozen before confirmation.

The lifecycle candidate is called an oracle because it uses realized
resource-time and selects the best price from observed calibration outcomes. It
does not predict future request cost and must not be presented as an online
production mechanism.

## Confirmation gate

Run fresh isolation, token VTC, selected scalar VTC and selected lifecycle
policy conditions three times. The realized lifecycle-cost gate passes only if:

1. its mean normalized-slowdown gap improves by at least 15% relative to both
   token VTC and tuned scalar VTC;
2. it has a smaller gap than both baselines in all three paired repetitions;
3. its mean makespan is no more than 5% above token VTC.

Report both tenant slowdowns, the gap and bootstrap confidence interval,
makespan, SLO goodput, request failures, retractions, and exact state/KV
resource-time.
