# 先用小配置验证流程（两段独立；同一批 GPU 时请只开一段或等上一段结束再开下一段）
# cd /workspace/benchmark/sglang-main

# ---------- P 阶段（prefill-only）----------
nohup bash -lc '
cd /workspace/benchmark/sglang-main
/workspace/env/sglang-main/bin/python bash-test/batch_pd_nvtx_test.py \
  --keep-processed \
  --config bash-test/pd_batch_config_p_small.json \
  > bash-test/test_p_small.log 2>&1
' > bash-test/test_pd_driver_p.log 2>&1 &
P_PID=$!
wait "$P_PID"   # P 结束后再启动 D（nohup 仍可防 SSH 断开）

# ---------- D 阶段（decode，依赖上面 wait，仅在 P 完成后执行）----------
# nohup bash -lc '
# cd /workspace/benchmark/sglang-main
# /workspace/env/sglang-main/bin/python bash-test/batch_pd_nvtx_test.py \
#   --keep-processed \
#   --config bash-test/pd_batch_config_d_small.json \
#   > bash-test/test_d_small.log 2>&1
# ' > bash-test/test_pd_driver_d.log 2>&1 &
