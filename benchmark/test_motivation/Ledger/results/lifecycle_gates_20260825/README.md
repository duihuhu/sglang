# Lifecycle-Cost Gate Artifacts

These artifacts were produced on 2026-08-25 in the `sglang` container on
`182.43.57.115`, using physical NVIDIA A800 80GB GPUs 6 and 7 with tensor
parallel size 2. The model was `/models/Kimi-Linear-48B-A3B`.

## Reproduction

From `/workspace/sglang` inside the container:

```bash
bash benchmark/test_motivation/Ledger/run_lifecycle_gate_experiments.sh all
.venv/bin/python benchmark/test_motivation/Ledger/analyze_lifecycle_gate_results.py \
  benchmark/test_motivation/Ledger/results/lifecycle_gates_20260825 \
  --output benchmark/test_motivation/Ledger/results/lifecycle_gates_20260825/lifecycle_gate_report.md
```

The runner owns SGLang processes only inside this container, sets
`CUDA_VISIBLE_DEVICES=6,7`, and restores a full-capacity FCFS server on exit.

## Integrity

- M1: 54 JSON artifacts
- M2: 18 JSON artifacts
- M3: 12 JSON artifacts
- M4: 12 JSON artifacts
- M5: 24 JSON artifacts
- Total: 120 formal artifacts and 2,898 successful requests; zero failed requests
- Final decisions: M1 PASS, M2 FAIL, M3 FAIL, M4 PASS, M5 FAIL

Each JSON artifact contains request-level streaming timestamps, cache metadata,
the resolved model/capacity metadata, summary latency/SLO metrics, and the exact
active-request state-slot-time and KV-token-time trace. The corresponding
`.stdout` file is the human-readable output captured during that run. Server
startup and shutdown logs are under `server_logs/`.

The M5 result is a one-repetition kill-test screen because its 2.5% oracle
headroom was far below the preregistered 15% confirmation threshold. It should
not be interpreted as an online scheduler evaluation.
