# AFlex Prediction Models and Profiling Tools

Only source code is migrated here; assets such as model weights, profiling tables, pickle files, and shared libraries are not included. All input and output paths are passed explicitly; historical benchmark directories are never read implicitly. Training retains the original scripts' LUT (normalized KD-tree inverse-distance weighting), log-feature Ridge, and LightGBM algorithms, along with their original hyperparameters and cross-validation procedures. V2/V3 retain the `GBDTModel`, `LookupModel`, `cross_validate`, and `train_and_save` interfaces.

## Installation and CPU Help

In a complete SGLang environment, install from the repository root using `pip install -e python/.`; CPU training also requires `numpy pandas scipy scikit-learn`, with optional `lightgbm` (GBDT is skipped if unavailable). Profiling tools import torch, NVML, and the serving runtime only when profiling actually starts; `--help` does not import them. However, ordinary `python -m sglang...` still executes SGLang's top-level initialization and requires its basic dependencies (including pybase64); this migration does not change top-level initialization. CPU tests use an independent namespace bootstrap and therefore do not require these serving dependencies.

```bash
python -m sglang.srt.energy.prediction.train --help
python -m sglang.srt.energy.prediction.energy_model_v2 --help
python -m sglang.srt.energy.prediction.energy_model_v3 --help
python -m sglang.srt.energy.profiling.bench_prefill_af --help
python -m sglang.srt.energy.profiling.bench_decode_af --help
python -m sglang.srt.energy.profiling.normalize --help
```

## Real GPU Profiling (Users Must Run It Themselves)

An external model, working CUDA/NVML, and frequency-control privileges are required. The original profiling algorithms use operators from the real model's first layer, retaining TP synchronization, warmup, probing, and repeated measurements. The DVFS library uses the runtime's build/discovery mechanism or an explicit `DVFS_CTRL_LIB`; it is not loaded from the old directory. Output paths are relative to the current working directory, not the installation directory. `--help` shows profiling parameters and common server parameters; actual runs still support the full `ServerArgs` parameter set.

```bash
python -m sglang.srt.energy.profiling.bench_prefill_af --model-path /external/model --tp-size 1 --output /external/profiles/prefill_raw.tsv --quick
python -m sglang.srt.energy.profiling.bench_decode_af --model-path /external/model --tp-size 1 --output /external/profiles/decode_raw.tsv --quick
```

## Normalization and V1 Training

Raw `P_A_lat/P_F_lat/P_A_energy/P_F_energy` or `D_*` headers are converted to `A/F/A_energy_mj/F_energy_mj`. Latency is already in microseconds, and energy is already in mJ per layer for the single GPU on rank0: **do not multiply by TP, divide by TP, or multiply by 1000**. Except for the additional `output_len` in decode, V1 headers are ordered as `tp,input_len,gpu_clock,batch_size,A,F,A_energy_mj,F_energy_mj` (tab-separated); the first line of a normalized file is a description line. Training can also read raw tables or standard tables without a description line directly.

```bash
python -m sglang.srt.energy.profiling.normalize --phase prefill --input /external/profiles/prefill_raw.tsv --output /external/profiles/prefill_v1.tsv
python -m sglang.srt.energy.profiling.normalize --phase decode --input /external/profiles/decode_raw.tsv --output /external/profiles/decode_v1.tsv
python -m sglang.srt.energy.prediction.train --prefill /external/profiles/prefill_v1.tsv --decode /external/profiles/decode_v1.tsv --output-dir /external/models/v1 --folds 5
```

V1 outputs LUT, LinearReg, and, when available, GBDT pickle files for eight tasks, along with `cv_mape_report.tsv`. `prediction.energy_model` can also serve as the V1 CLI. The CLI calls training functions defined in the package, so new pickle class paths are stable as `sglang.srt.energy.prediction.energy_model*` rather than `__main__`.

## V2/V3 Training Compatibility

V2 requires the columns `tp,M,f_A,f_F,input_len,batch_size,iter_lat_us,DA_energy_mj,DF_energy_mj`; the actual model features are `M,f_A,f_F,input_len,batch_size`. V3 replaces `tp` in the table with `tp_a,tp_f`, with model features `tp_a,tp_f,M,f_A,f_F,input_len,batch_size`. Both inputs are externally available pipeline profiling data; this package's independent A/F profilers do not generate pipeline data.

```bash
python -m sglang.srt.energy.prediction.energy_model_v2 --data /external/profiles/decode_pipeline_v2.tsv --output-dir /external/models/v2 --folds 5
python -m sglang.srt.energy.prediction.energy_model_v3 --data /external/profiles/decode_pipeline_v3.tsv --output-dir /external/models/v3 --skip-cv
```

Each outputs GBDT/LUT files for `Decode_iter_lat`, `Decode_iter_energy_A`, and `Decode_iter_energy_F`. The runtime predictor's `model_dir` may contain V1 and V2 files; V3 uses an explicit `v3_model_dir`. Existing compatibility discovery of adjacent `models_v3/models_v4` by predictors is retained; the training CLIs themselves have no historical default asset paths.

The predictor supports legacy `energy_model`, `energy_model_v1/v2/v3`, and `__main__` class references without reading scripts in ignored directories. Historical `__main__.GBDTModel` is interpreted as V1 when it contains `all_features`; otherwise, V2/V3 are distinguished by whether `feature_cols` contains `tp_a`. Pickle can execute arbitrary code: **load only trusted files**; the compatibility unpickler is not a security sandbox.

## CPU Tests

```bash
python -m pytest test/srt/test_aflex_prediction.py -q
```

Coverage includes class paths and pickle round trips, distinguishing historical GBDT states, raw-table normalization and unit preservation, small-data fitting, V2 runtime feature layouts, and GPU-free CLI help entry points for six modules. This migration does not run GPU profiling.
