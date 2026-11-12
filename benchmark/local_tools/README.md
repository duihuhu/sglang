# Local benchmark and diagnostic tools

These scripts were moved out of the repository root without changing their
experiment settings:

- `bench_afd_async_smoke.py`: asynchronous AFD smoke test.
- `bench_afd_async_e2e.py`: asynchronous AFD end-to-end benchmark.
- `bench_afd_full_compare.py`: microbatch/communication/optimization comparison.
- `bench_pdaf_compare_v2.py`: PD+AF IPC versus UCX comparison.
- `check_bs.py`: inspect batch sizes in AFD per-step logs.
- `plot_df210_gantt.py`: plot M=1/M=2 pipeline timing at DA/DF 210 MHz.
- `check_loogle_dataset_lengths.py`: inspect Loogle dataset lengths using
  `benchmark/hicache/data_processing.py`.

Run these tools from the repository root using their full relative paths.
Several scripts retain historical `/workspace` environment paths and GPU/model
settings. Inspect them before use; benchmark launchers may terminate existing
experiment processes. Moving a script does not certify its runtime environment.

`build_afd_ipc.py` remains at the repository root: it is an extension build tool,
not an experiment result or test script, and resolves CUDA sources relative to
its own location.
