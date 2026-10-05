# Lifecycle-Cost Fairness Oracle Artifacts

These artifacts compare token VTC, tuned scalar VTC and a realized state/KV
lifecycle-cost fairness policy on physical A800 GPUs 6 and 7 with
`/models/Kimi-Linear-48B-A3B`.

## Reproduction

From `/workspace/sglang` inside the `sglang` container:

```bash
bash benchmark/test_motivation/Ledger/run_lifecycle_fairness_oracle_experiments.sh calibrate
bash benchmark/test_motivation/Ledger/run_lifecycle_fairness_oracle_experiments.sh confirm
```

Calibration writes and freezes `selection.json`; confirmation only consumes
that selection. The analyzer command is:

```bash
.venv/bin/python benchmark/test_motivation/Ledger/analyze_lifecycle_fairness_oracle.py \
  benchmark/test_motivation/Ledger/results/lifecycle_fairness_oracle_20260825 \
  --output benchmark/test_motivation/Ledger/results/lifecycle_fairness_oracle_20260825/lifecycle_fairness_oracle_report.md \
  --selection-output benchmark/test_motivation/Ledger/results/lifecycle_fairness_oracle_20260825/selection.json
```

## Integrity and decision

- 33 formal JSON artifacts and 3,944 requests
- zero failed requests and zero retractions
- exact scheduler-side state/KV resource trace in 33/33 artifacts
- resolved policy metadata in 15/15 confirmation artifacts
- frozen scalar request weight: 0
- frozen realized lifecycle weights: state 0.25, KV 1
- confirmation decision: PASS

The realized-cost policy reduced the normalized-slowdown gap by 82.6% versus
token VTC and 82.3% versus tuned scalar VTC, with a 3.1% lower makespan. This is
a workload-calibrated price oracle, not evidence that the price generalizes.
