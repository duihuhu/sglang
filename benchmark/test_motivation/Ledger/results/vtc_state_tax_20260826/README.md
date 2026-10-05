# VTC state-tax compact results

This directory contains only the compact, replayable analysis products. The 387
formal raw artifacts remain on `182.43.57.115` at:

`/mnt/nvme0/mllm/sglang/benchmark/test_motivation/Ledger/results/vtc_state_tax_20260826`

Key files:

- `vtc_state_tax_observation_report.md`: human-readable gate results;
- `analysis.json`: complete integrity audit, per-cell confirmation metrics, and
  O1/O2/O3 decisions;
- `profiled_vtc.json`: frozen nonnegative profiled-VTC coefficients and fit;
- `selection.json`: screening-only global and per-cell request-tax selection;
- `vtc_state_tax_observation.png`: post-hoc visualization of the preregistered
  metrics; it does not alter selection or gates.

The overall decision is FAIL because O2 failed, despite O1 and O3 passing. The
supported observation is that equal-token request partitioning changes fixed
state demand, KV residency, admission concurrency, and slowdown. The data do
not support attributing the fairness gap to the state-capacity wall alone.
