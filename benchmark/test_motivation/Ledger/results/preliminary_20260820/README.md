# Preliminary Result Artifacts

- `canonical/` contains the 24 valid request-level JSON artifacts: 8 conditions x 3 replicates.
- `preliminary_report.md` is generated from those artifacts by `../../analyze_motivation_results.py`.
- Runs affected by reused request IDs or overlapping benchmark processes were excluded before this directory was assembled.

Regenerate the report from the repository root:

```bash
python3 benchmark/test_motivation/Ledger/analyze_motivation_results.py \
  --input-dir benchmark/test_motivation/Ledger/results/preliminary_20260820/canonical \
  --output benchmark/test_motivation/Ledger/results/preliminary_20260820/preliminary_report.md
```
