1）Macro-benchmark（变长数据集），2节点*8=16卡

模型：Qwen3-32B
数据集：AwareZure两个：code/conv
数据图表：
横坐标：QPS
纵坐标：（a） 能耗，（b）TTFT， （c）TPOT。
6个方案：（a）Native DP；（b）PD DP；（c）PDAF TP。以及有无（Tier）

