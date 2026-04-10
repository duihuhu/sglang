nohup bash -lc '
echo "[pd-test] $(date "+%F %T") stage=P start"
/workspace/env/sglang-main/bin/python bash-test/batch_pd_nvtx_test.py \
  --keep-processed \
  --config bash-test/pd_batch_config_p.json \
  > bash-test/test_p.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  echo "[pd-test] $(date "+%F %T") stage=P failed, exit_code=$rc"
  exit $rc
fi
echo "[pd-test] $(date "+%F %T") stage=P done"

# echo "[pd-test] $(date "+%F %T") stage=D start"
# /workspace/env/sglang-main/bin/python bash-test/batch_pd_nvtx_test.py \
#   --keep-processed \
#   --config bash-test/pd_batch_config_d.json \
#   > bash-test/test_d.log 2>&1
# rc=$?
# if [ $rc -ne 0 ]; then
#   echo "[pd-test] $(date "+%F %T") stage=D failed, exit_code=$rc"
#   exit $rc
# fi
# echo "[pd-test] $(date "+%F %T") stage=D done"
# ' > bash-test/test_pd_driver.log 2>&1 &
