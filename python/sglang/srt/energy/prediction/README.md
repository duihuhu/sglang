# AFlex 预测模型与采样工具

这里只迁移源代码，不包含模型权重、采样表、pickle、共享库等资产。所有输入与输出路径均显式传入，不会隐式读取历史 benchmark 目录。训练沿用原脚本的 LUT（归一化 KD-tree 逆距离加权）、对数特征 Ridge 与 LightGBM 算法，以及原有超参数和交叉验证方式。V2/V3 保留 `GBDTModel`、`LookupModel`、`cross_validate`、`train_and_save` 接口。

## 安装与 CPU 帮助

在完整 SGLang 环境中，从仓库根目录安装 `pip install -e python/.`；CPU 训练还需 `numpy pandas scipy scikit-learn`，可选 `lightgbm`（缺失时跳过 GBDT）。采样工具仅在真正开始采样时导入 torch、NVML 与服务运行时，`--help` 不导入它们；不过普通 `python -m sglang...` 仍会执行 SGLang 顶层初始化，须满足其基础依赖（包括 pybase64），本迁移不更改顶层初始化。CPU 测试采用独立 namespace bootstrap，因此无需这些服务依赖。

```bash
python -m sglang.srt.energy.prediction.train --help
python -m sglang.srt.energy.prediction.energy_model_v2 --help
python -m sglang.srt.energy.prediction.energy_model_v3 --help
python -m sglang.srt.energy.profiling.bench_prefill_af --help
python -m sglang.srt.energy.profiling.bench_decode_af --help
python -m sglang.srt.energy.profiling.normalize --help
```

## 真实 GPU 采样（需要用户自行运行）

须提供外部模型、可用 CUDA/NVML、频率控制权限；原采样算法使用真实模型首层算子，保留 TP 同步、预热、探测与重复测量。DVFS 库使用运行时的构建/发现机制，或显式设置 `DVFS_CTRL_LIB`，不从旧目录加载库。输出路径相对于当前工作目录，而非安装目录。`--help` 展示采样参数与常用服务参数，真正运行仍支持 `ServerArgs` 的完整参数集。

```bash
python -m sglang.srt.energy.profiling.bench_prefill_af --model-path /external/model --tp-size 1 --output /external/profiles/prefill_raw.tsv --quick
python -m sglang.srt.energy.profiling.bench_decode_af --model-path /external/model --tp-size 1 --output /external/profiles/decode_raw.tsv --quick
```

## 规范化与 V1 训练

原始 `P_A_lat/P_F_lat/P_A_energy/P_F_energy` 或 `D_*` 表头转换为 `A/F/A_energy_mj/F_energy_mj`。延迟已经是微秒，能量已经是 rank0 单 GPU 每层 mJ：**不乘 TP、不除 TP、不乘 1000**。除 decode 额外的 `output_len` 外，V1 表头依次为 `tp,input_len,gpu_clock,batch_size,A,F,A_energy_mj,F_energy_mj`（制表符分隔）；规范化文件首行是说明行。训练也能直接读取原始表或无说明行的标准表。

```bash
python -m sglang.srt.energy.profiling.normalize --phase prefill --input /external/profiles/prefill_raw.tsv --output /external/profiles/prefill_v1.tsv
python -m sglang.srt.energy.profiling.normalize --phase decode --input /external/profiles/decode_raw.tsv --output /external/profiles/decode_v1.tsv
python -m sglang.srt.energy.prediction.train --prefill /external/profiles/prefill_v1.tsv --decode /external/profiles/decode_v1.tsv --output-dir /external/models/v1 --folds 5
```

V1 输出 8 个任务的 LUT、LinearReg，以及可用时的 GBDT pickle 和 `cv_mape_report.tsv`。`prediction.energy_model` 同样可作为 V1 CLI。CLI 调用包内定义的训练函数，因此新 pickle 类路径稳定为 `sglang.srt.energy.prediction.energy_model*`，而非 `__main__`。

## V2/V3 训练兼容

V2 必需列：`tp,M,f_A,f_F,input_len,batch_size,iter_lat_us,DA_energy_mj,DF_energy_mj`，实际模型特征是 `M,f_A,f_F,input_len,batch_size`。V3 将表中的 `tp` 换为 `tp_a,tp_f`，模型特征为 `tp_a,tp_f,M,f_A,f_F,input_len,batch_size`。这两种输入是外部已有 pipeline 采样数据，本包的独立 A/F 采样器不生成 pipeline 数据。

```bash
python -m sglang.srt.energy.prediction.energy_model_v2 --data /external/profiles/decode_pipeline_v2.tsv --output-dir /external/models/v2 --folds 5
python -m sglang.srt.energy.prediction.energy_model_v3 --data /external/profiles/decode_pipeline_v3.tsv --output-dir /external/models/v3 --skip-cv
```

各自输出 `Decode_iter_lat`、`Decode_iter_energy_A`、`Decode_iter_energy_F` 的 GBDT/LUT 文件。运行预测器的 `model_dir` 可放 V1 和 V2 文件，V3 使用显式 `v3_model_dir`。保留既有预测器对相邻 `models_v3/models_v4` 的兼容发现行为；训练 CLI 本身没有历史默认资产路径。

预测器兼容旧 `energy_model`、`energy_model_v1/v2/v3` 与 `__main__` 类引用，不读取忽略目录中的脚本。历史 `__main__.GBDTModel` 含 `all_features` 时解释为 V1，否则根据 `feature_cols` 中是否包含 `tp_a` 区分 V2/V3。pickle 可执行任意代码，**仅加载可信文件**；兼容 unpickler 不是安全沙箱。

## CPU 测试

```bash
python -m pytest test/srt/test_aflex_prediction.py -q
```

覆盖类路径与 pickle 往返、历史 GBDT 状态区分、原始表规范化与单位不变、小数据拟合、V2 运行特征布局，以及六个模块 CLI 的无 GPU 帮助入口。本次迁移不运行 GPU 采样。
