# Micro Benchmark 测试进度 (2026-07-27 00:03 更新)

## 1. AFlex 全数据集能耗修正测试

**脚本**: `run_aflex_all_energy_corrected.py`
**节点**: node1(10.252.129.36) + node3(10.252.129.34)
**能耗统计**: 只统计实际启动的 GPU（corrected energy scope）
**tmux session**: `aflexbench`（正在后台运行）
**日志**: `/tmp/aflex_summary_n3.log`
**结果文件**: `data/aflex_all_energy_corrected.json`

### 已完成（12/16 点 PASS）

| 数据集 | QPS | GPU数 | 吞吐 | E/tok | 状态 |
|---|---|---|---|---|---|
| qa_lpld | 2 | 6 | 128.8 tok/s | 1733.56 mJ | ✅ PASS |
| qa_lpld | 4 | 8 | 235.5 tok/s | 1308.94 mJ | ✅ PASS |
| qa_lpld | 8 | 8 | 468.4 tok/s | 679.65 mJ | ✅ PASS |
| qa_lpld | 16 | 10 | 647.7 tok/s | 608.59 mJ | ✅ PASS |
| chatbot_lphd | 2 | 6 | 943.9 tok/s | 639.82 mJ | ✅ PASS |
| chatbot_lphd | 4 | 6 | 1029.7 tok/s | 593.29 mJ | ✅ PASS |
| chatbot_lphd | 8 | 6 | 1101.3 tok/s | 555.71 mJ | ✅ PASS |
| chatbot_lphd | 16 | 6 | 1140.8 tok/s | 533.65 mJ | ✅ PASS |
| rag_hpld | 2 | 12 | 128.3 tok/s | 128.59 mJ | ✅ PASS |
| rag_hpld | 4 | 14 | 241.8 tok/s | 92.79 mJ | ✅ PASS |
| rag_hpld | 8 | 16 | 447.6 tok/s | 58.11 mJ | ✅ PASS |
| rag_hpld | 16 | 16 | 689.6 tok/s | 37.71 mJ | ✅ PASS |
| summary_hphd | 2 | 12 | 960.3 tok/s | 254.42 mJ | ✅ PASS |
| summary_hphd | 4 | 12 | — | — | 🔄 运行中 |
| summary_hphd | 8 | 16 | — | — | ⏳ 待测 |
| summary_hphd | 16 | 12 | — | — | ⏳ 待测 |

### 备注
- qa_lpld/chatbot_lphd/rag_hpld 使用 node1+node2(35) 测试完成
- summary_hphd 因 node2 被占，改用 node1+node3(34) 测试
- 预计 summary_hphd 剩余 3 个点约 15 分钟完成

---

## 2. TP2 Baseline 测试（SGLang-TP2 + DynamoLLM-TP2）

**脚本**: `run_tp2_benchmark.py`
**节点**: node1(10.252.129.36) + node2(10.252.129.35)
**部署**: 8×TP2 (4实例/节点), round-robin router
**结果文件**: `data/micro_tp2_node1_node2.json`

### 已完成: SGLang-TP2 部分（6/16 点）

| 数据集 | QPS | 吞吐 | TTFT p90 | TPOT p90 | E/tok | 状态 |
|---|---|---|---|---|---|---|
| qa_lpld | 2 | 132.1 | 84.3 ms | 38.5 ms | 7280.5 mJ | ✅ |
| qa_lpld | 4 | 246.3 | 77.7 ms | 39.4 ms | 4979.1 mJ | ✅ |
| qa_lpld | 8 | 510.8 | 79.2 ms | 41.4 ms | 2501.0 mJ | ✅ |
| qa_lpld | 16 | 745.7 | 79.0 ms | 42.5 ms | 1703.4 mJ | ✅ |
| chatbot_lphd | 2 | 1473.2 | 79.5 ms | 40.1 ms | 2349.5 mJ | ✅ |
| chatbot_lphd | 4 | 1908.5 | 78.4 ms | 39.9 ms | 1868.4 mJ | ✅ |

### 未完成
- SGLang-TP2: chatbot_lphd QPS 8/16, rag_hpld 全部, summary_hphd 全部
- DynamoLLM-TP2: 全部 16 点

**状态**: 被中断（node2 被占），需等 node2 空闲后用 `python3 run_tp2_benchmark.py` 恢复（不加 --force 会自动跳过已 PASS 的点）

---

## 3. 待办

1. 等 AFlex summary_hphd 4点跑完 → 检查结果
2. 等 node2 空闲 → 继续 TP2 benchmark（或改用 node1+node3）
3. 汇总 AFlex + TP2 + 原有 baseline 数据，更新图表
