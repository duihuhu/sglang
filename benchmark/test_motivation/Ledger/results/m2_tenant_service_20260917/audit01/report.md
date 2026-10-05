# Historical M2 Tenant-Service Audit

Counts: {'verified_artifacts': 60, 'requests': 8000, 'mixed_runs': 20, 'isolation_runs': 40}; new GPU runs: 0.

All selected raw SHA256 hashes verified. Historical gates unchanged.

## Trace Coverage (20 Mixed Runs)

| Field | Runs |
|---|---:|
| client_output_stream | 20 |
| vtc_counter_summary | 20 |
| vtc_counter_events | 0 |
| resource_summary | 20 |
| resource_events | 0 |
| batch_shape_events | 20 |
| per_request_batch_work | 0 |
| per_tenant_waiting_snapshots | 0 |
| admission_block_records | 0 |
| server_event_clock_origin | 0 |
| client_absolute_origin | 0 |

## Full-Run Tenant Results

Five-repeat means; post-hoc descriptive audit, no new CI or PASS.

| Shape | Cap | Tenant | P95 JCT s | Slowdown | Output 50% s | State slot-s | KV token-s |
|---|---:|---|---:|---:|---:|---:|---:|
| f16 | 48 | packed | 20.116 | 1.381 | 17.775 | 349.344 | 1877940 |
| f16 | 48 | split | 7.825 | 8.479 | 10.729 | 975.065 | 334208 |
| f16 | 96 | packed | 19.994 | 1.373 | 17.399 | 349.636 | 1881706 |
| f16 | 96 | split | 10.635 | 11.599 | 10.390 | 1685.025 | 576884 |
| f32 | 48 | packed | 20.197 | 1.385 | 17.921 | 350.363 | 1884736 |
| f32 | 48 | split | 7.071 | 8.889 | 8.987 | 1075.282 | 184241 |
| f32 | 96 | packed | 20.140 | 1.381 | 17.605 | 352.656 | 1894592 |
| f32 | 96 | split | 7.097 | 8.931 | 11.357 | 1988.181 | 340264 |

## Common Client Drain Windows

Output-only incremental gaps, NOT token-VTC fairness. Same start/end for each paired cap run.

| Shape | Cap | Mean window s | Mean max output fraction gap | State-time split/packed | KV-time split/packed |
|---|---:|---:|---:|---:|---:|
| f16 | 48 | 4.365 | 32.78% | 2.791 | 0.178 |
| f16 | 96 | 4.365 | 33.54% | 4.819 | 0.307 |
| f32 | 48 | 4.341 | 26.13% | 3.069 | 0.098 |
| f32 | 96 | 4.341 | 55.59% | 5.634 | 0.179 |

## Limits

- Client output milestones/gaps exclude input and do not reconstruct VTC counters.
- Client outstanding is not scheduler waiting backlog; common windows are outcome-dependent descriptive drain windows.
- Resource integrals cover the full run, not the client window; physical state slots can include extra buffers.
- Batch tenant counts are participation counts, not per-tenant work or GPU time.
- Paired payload verification applies within each factor/rep, not across factors or server sessions.
- F16/f32 use different full payload fingerprints and sessions; they are not a fixed-victim causal comparison.
