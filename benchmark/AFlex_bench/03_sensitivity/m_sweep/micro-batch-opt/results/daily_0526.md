# 5/26 进展总结

完成了 PD+AF 系统在多并发场景下的全面性能评估。测试覆盖四种架构（PD TP=2、PD+AF M=2 async、PD+AF M=1、PD TP=1），并发从 1 扫到 2048。结果表明：PD TP=2 在所有场景下性能最优，峰值吞吐 3704 tok/s；AF M=2 async pipeline 在高并发（≥256）下表现突出，conc=2048 时吞吐达 3281 tok/s，仅比 TP=2 低 6%，验证了 interleaved pipeline 在大 batch 下的 overlap 收益；AF M=1 饱和在 ~1950 tok/s，M=2 相比 M=1 在高并发下有 60-70% 的吞吐提升。小并发（≤32）下 AF 方案无优势，M=2 因 GEMM 效率损失反而更差，收益拐点在 conc≈64。数据和折线图已整理至 `results/` 目录。
