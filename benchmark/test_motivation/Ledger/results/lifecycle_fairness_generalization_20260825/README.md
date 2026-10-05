# Lifecycle fairness generalization results

The 626 raw A800 artifacts remain on the remote host at:

`/mnt/nvme0/mllm/sglang/benchmark/test_motivation/Ledger/results/lifecycle_fairness_generalization_20260825`

Only compact analysis products are synchronized here:

- `lifecycle_fairness_generalization_report.md`: final screening, confirmation,
  and independent kill-confirm decision;
- `screening_report.md`: screening-only selections;
- `selection.json`: frozen per-cell screening choices;
- `audit.md` and `audit.json`: reproducible artifact-integrity audit;
- `hardware-gpu6-7.txt`: physical A800 GPU metadata.

The final mechanism decision is **FAIL**. Frozen lifecycle cost improves the
slowdown gap over instantaneous share by 4.0% on average, with a cell-bootstrap
95% CI of [-24.7%, 27.3%], wins in 7/10 cells, and changes makespan by +0.4%
relative to token VTC.
