# 先用小配置验证流程（两段独立；同一批 GPU 时请只开一段或等上一段结束再开下一段）
# cd /workspace/benchmark/sglang-main

# ---------- P 阶段（prefill-only）----------
# nohup bash -lc '
# cd /workspace/benchmark/sglang-main
# /workspace/env/sglang-main/bin/python bash-test/batch_pd_nvtx_test.py \
#   --keep-processed \
#   --config bash-test/pd_batch_config_p.json \
#   > bash-test/test_p.log 2>&1
# ' > bash-test/test_pd_driver_p.log 2>&1 &

# nohup bash -lc '
# cd /workspace/benchmark/sglang-main
# /workspace/env/sglang-main/bin/python bash-test/batch_pd_nvtx_test.py \
#   --keep-processed \
#   --config bash-test/pd_batch_config_p_small.json \
#   > bash-test/test_p_small.log 2>&1
# ' > bash-test/test_pd_driver_p.log 2>&1 &

# ---------- D 阶段（decode）----------
nohup bash -lc '
cd /workspace/benchmark/sglang-main
/workspace/env/sglang-main/bin/python bash-test/batch_pd_nvtx_test.py \
  --keep-processed \
  --config bash-test/pd_batch_config_d_small.json \
  > bash-test/test_d_small.log 2>&1
' > bash-test/test_pd_driver_d.log 2>&1 &

# nohup bash -lc '
# cd /workspace/benchmark/sglang-main
# /workspace/env/sglang-main/bin/python bash-test/batch_pd_nvtx_test.py \
#   --keep-processed \
#   --config bash-test/pd_batch_config_d.json \
#   > bash-test/test_d.log 2>&1
# ' > bash-test/test_pd_driver_d.log 2>&1 &

# 验证通过后再换成完整配置，例如：
# --config bash-test/pd_batch_config_p.json
# --config bash-test/pd_batch_config_d.json
