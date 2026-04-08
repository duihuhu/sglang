
# nohup /workspace/env/sglang-main/bin/python bash-test/batch_pd_nvtx_test.py \
#   --gpus 0,1,2,3,4,5,6,7 \
#   --model-path /mnt/nvme1/models/Qwen/Qwen3-32B/ \
#   --tp-list 8 \
#   --input-lens 4096 \
#   --output-lens 512 \
#   --batch-size 512 \
#   --gpu-clocks 1200 \
#   --work-dir bash-test/pd_batch_work_ut_both \
#   --log-dir bash-test/pd_batch_logs_ut_both \
#   --final-csv bash-test/pd_latency_big_table_ut_both.csv > bash-test/test_both.log 2>&1 &



nohup /workspace/env/sglang-main/bin/python bash-test/batch_pd_nvtx_test.py \
  --config bash-test/pd_batch_config.example.json > bash-test/test_both.log 2>&1 &
