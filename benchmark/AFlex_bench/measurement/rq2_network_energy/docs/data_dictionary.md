# RQ2 数据字典

六种冻结负载：QA 128/64、Chatbot 128/1024、Balanced 512/256、RAG 4096/64、Summary 4096/1024、LongContext 16384/256。`offered_qps` 是 Poisson 开环到达率；`achieved_qps` 是成功请求除以测量窗口。能耗以计划实际占用 GPU 的 NVML 累计能量差计。

原始 artifact 至少包括 `deployment_manifest.json`、`requests.jsonl`、`summary.json`、`system.json`、`energy.json`，网络实验另含 `link_telemetry.json` 和 `link_validation.json`。历史日志中的旧绝对路径不修改；移动映射及完整性见 `migration_manifest.json`。
