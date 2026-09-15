# RQ2 运行手册

所有命令从 `rq2_network_energy/` 执行。

```bash
python3 scripts/generate_measurement_workloads.py
python3 -m pytest -q tests
python3 scripts/run_measurement_rdma2n_qps2.py --offline-preflight
python3 scripts/run_measurement_rdma2n_4gpu_qps4.py --offline-preflight
```

在线 preflight 和正式执行具有副作用，必须显式传递 runner 要求的 `--execute` 和 `--resume`。正式结果默认位于 `results/raw/`。

主实验口径是 Qwen3-30B-A3B、总 4 GPU、QPS 4、六负载、三次重复；2 GPU QPS2 以及本地 PCIe/NVLink 2 GPU 结果作为补充实验保留，不与主实验直接混算。
