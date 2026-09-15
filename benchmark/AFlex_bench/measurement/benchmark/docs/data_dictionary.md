# 数据字典

## workload JSONL
`request_id` 唯一标识；`timeout_s` 可为单请求超时；`arrival_time_s` 相对计划到达；`input_len`/`output_len` 为 token 数；`source` 为固定类、trace 类或 synthetic 标记。

## requests.jsonl
`scheduled_arrival_s` 是原计划到达偏移；`sent_offset_s` 是实际发送偏移；`arrival_lag_ms=max(0, sent_offset_s-scheduled_arrival_s)`，用于量化 admission 排队造成的发送偏差。`token_timestamps_s` 是相对 run 起点的逐流式 chunk 单调时间；`itl_ms` 为相邻时间差；`ttft_client_ms` 为发送至首 chunk；`ttft_server_ms` 来自服务端 meta（可空）；`tpot_ms` 为 ITL 均值；`e2e_ms` 为请求全程。另含成功、错误、token 数和服务端元数据。

## energy.json
`before_mj`/`after_mj` 保存 NVML 原始累计毫焦；`normalized.per_node_gpu_j` 为逐 GPU 焦耳差；`per_node_j` 为节点和；`total_j` 为集群和。计数器回绕或重置会被钳制为零并应结合 system 快照审阅。

## summary.json
`metrics` 对 TTFT/TPOT/E2E/ITL 给出 count、mean、p50/p90/p95/p99；另含吞吐、成功率、J/token、J/request、point/workload/QPS 与状态。AF QPS sweep 还记录 `offered_qps`（仅表示 open-loop 到达率）、`max_inflight`（统一 admission 上限）、`arrival_span`（首个计划时刻 0 到最后计划到达的秒数）和 `wall_limit`（该档动态 point wall 上限，秒）。`arrival_lag_ms` 可由 admission 排队产生，不代表失败；只有 wall limit 到达时仍未完成的任务才记 point wall timeout。


## semantic_results.json / semantic summary.json
`semantic_results.json` 保存统一模型、确定性 `sampling_params`，以及 baseline/AF 每题的 prompt、目标 endpoint、完整 `response_json`、提取后的 `text`/`normalized_text`、`meta_info`、可用时的 `output_ids`/`output_token_count`、错误和耗时。语义 summary 明确区分三层判据：`api_success`（API/JSON 可解析）、`semantic_expected`（alias/regex 命中）与 `deterministic_match`（Native/AF 规范化文本严格相同；双方都有 output IDs 时另报 token 严格相同）。语义通过不表示文本或 token 完全一致。
