# Admission-batch motivation results

Experiment date: 2026-08-28. Reports synchronized locally on 2026-09-08.

- Model: `/models/Kimi-Linear-48B-A3B`, TP=2, physical A800 GPUs 6 and 7.
- Scope: 57 artifacts, 6,120 requests, no prefix reuse.
- Remote host: `182.43.57.115`.
- Raw results: `/mnt/nvme0/mllm/sglang/benchmark/test_motivation/Ledger/results/admission_batch_20260828`.
- Container mapping: `/workspace/sglang/benchmark/test_motivation/Ledger/results/admission_batch_20260828`.

The report and `analysis.json` are copies of the remote analyzer outputs.
Raw request artifacts and server logs remain on the remote host.
See the [protocol](../../admission_batch_motivation_protocol.md) for frozen gates
and the [current motivation](../../hybrid_state_aware_admission_motivation.md)
for interpretation and limitations.

The original Python JSON export contains six `NaN` values for unavailable
VTC queue-wait metrics under instant-share. It is not strict RFC JSON;
strict consumers must explicitly map these missing values to null.
The token-VTC queue measurements quoted in M2 are present. The source export
is preserved without substituting or imputing measurements.

PASS denotes the preregistered motivation gates in this experiment. It does
not establish a new scheduler's effectiveness, a VTC theorem violation, or
generalization beyond the tested workloads. The report averages three
repetitions per configuration and does not supply effect confidence intervals.
