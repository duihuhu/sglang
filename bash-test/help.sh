CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
    --model-path /mnt/nvme1/models/llama3.1-8/ \
    --disable-overlap-schedule \
    --disable-cuda-graph \
    --afd-perspective attn \
    --afd-mirco-batch 3

CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
    --model-path /mnt/nvme1/models/llama3.1-8/ \
    --disable-overlap-schedule \
    --disable-cuda-graph \
    --port 30001 \
    --skip-server-warmup \
    --watchdog-timeout 3600 \
    --afd-perspective ffn \
    --afd-mirco-batch 3

export HF_ENDPOINT=https://hf-mirror.com

nohup hf download Qwen/Qwen3-32B --local-dir /mnt/nvme1/models/Qwen/Qwen3-32B > download.log 2>&1 &

curl -s "http://127.0.0.1:30000/generate" -H "Content-Type: application/json" -d '{"text":"你好，介绍一下你自己","sampling_params":{"max_new_tokens":64,"temperature":0.7}}'

CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path /mnt/nvme1/models/llama3.1-8/ \
  --disable-overlap-schedule \
  --disable-cuda-graph \
  --port 30000


CUDA_VISIBLE_DEVICES=4,5,6,7  \
nohup nsys profile \
  --capture-range=cudaProfilerApi \
  --trace=cuda,nvtx \
  --cuda-memory-usage=false \
  --sample=none \
  --backtrace=none \
  --cpuctxsw=none \
  --trace-fork-before-exec=true \
  --force-overwrite=true \
  -o sglang.out \
  python -m sglang.launch_server \
    --model-path /mnt/nvme1/models/Qwen/Qwen3-32B/ \
    --port 30000 \
    --max-prefill-tokens 1000000 \
    --tensor-parallel-size 4 \
    --disable-cuda-graph \
    --disable-overlap-schedule \
    --chunked-prefill-size -1 \
    --mem-fraction-static 0.9 > model_input.log 2>&1 &

  

CUDA_VISIBLE_DEVICES=4,5,6,7 python bash-test/bench_sglang.py \
  --server-url http://127.0.0.1:30000 \
  --batch_size 32 \
  --input_len 1024 \
  --output_len 512 \
  --ignore_eos \
  --use-server-profile-range \
  --profile-activities CUDA_PROFILER


python bash-test/nvtx_stats_from_rep.py sglang.out.nsys-rep \
  --out-dir /workspace/benchmark/sglang-main/bash-test \
  --d-bucket-count 1 \
  --d-pick-positions 1

----------------------------------------------


CUDA_VISIBLE_DEVICES=4,5,6,7  \
nohup python -m sglang.launch_server \
    --model-path /mnt/nvme1/models/Qwen/Qwen3-32B/ \
    --port 30000 \
    --max-prefill-tokens 1000000 \
    --tensor-parallel-size 4 \
    --disable-cuda-graph \
    --disable-overlap-schedule \
    --chunked-prefill-size -1 \
    --mem-fraction-static 0.9 > model_input.log 2>&1 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python bash-test/bench_sglang.py \
  --server-url http://127.0.0.1:30000 \
  --batch_size 1024 \
  --input_len 128 \
  --output_len 512 \
  --ignore_eos


rm -rf /var/lib/apport/coredump/*





# 汇总各 run 日志与 processed 产物，生成 pd_batch_summary.csv，便于筛失败/再跑/合并大表
python bash-test/summarize_pd_batch_runs.py \
  --log-dir bash-test/pd_batch_logs \
  --work-dir bash-test/pd_batch_work \
  --out-csv bash-test/pd_batch_summary.csv

# 将各 run 的 processed/nvtx_PD_combined_stats.csv 合并为长表（与 batch 跑完生成的 pd_latency_big_table 同格式），默认不删 work 目录
python bash-test/merge_pd_big_table.py \
  --work-dir bash-test/pd_batch_work \
  --output-lens 64,256,512 \
  --final-csv bash-test/pd_latency_big_table.csv

----------------------------------------------

# batch_pd_nvtx_test.py 单测（快速自检）:
# 运行 TTFT + per-op（自动两种都测，并合并进一个大表）:
nohup /workspace/env/sglang-main/bin/python bash-test/batch_pd_nvtx_test.py \
  --gpus 7 \
  --model-path /mnt/nvme1/models/Qwen/Qwen3-32B/ \
  --tp-list 1 \
  --input-lens 16384 \
  --output-lens 512 \
  --batch-size 1 \
  --gpu-clocks 1200 \
  --work-dir bash-test/pd_batch_work_ut_both \
  --log-dir bash-test/pd_batch_logs_ut_both \
  --final-csv bash-test/pd_latency_big_table_ut_both.csv > bash-test/test_both.log 2>&1 &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m sglang.bench_one_batch  \
 --model-path /mnt/nvme1/models/Qwen/Qwen3-32B/  \
 --tp-size 8   \
 --batch-size 1     \
 --input-len 128     \
 --output-len 1     \
 --disable-cuda-graph \
 --disable-overlap-schedule \
 --chunked-prefill-size -1 \
 --mem-fraction-static 0.9 \
 --profile-activities CUDA_PROFILER

 du -x -B1 --max-depth=1 /var/lib/apport 2>/dev/null | awk '$1>=1073741824 {printf "%.2f GB\t%s\n",$1/1073741824,$2}' | sort -nr

CUDA_VISIBLE_DEVICES=0,1 SGLANG_DEBUG_A_INPUT=1 BENCH_SM_LOCK_SETTLE_S=3 \
nohup python bench_prefill_af.py \
  --tp-size 2 \
  --model-path /mnt/nvme1/models/Qwen/Qwen3-32B/ \
  --no-resume > bench_prefill_af.log 2>&1 &

CUDA_VISIBLE_DEVICES=4,5,6,7 \
python bench_prefill_af.py \
  --tp-size 1 \
  --model-path /mnt/nvme1/models/Qwen/Qwen3-32B/

./bash-test/set_gpu_frequency.sh --gpus all --gpu-clock 210
./set_gpu_frequency.sh --gpus 6,7 --gpu-clock 1410
./set_gpu_frequency.sh --gpus 4,5 --gpu-clock 210


python bench_prefill_af.py \
  --model-path /mnt/nvme1/models/Qwen/Qwen3-32B/ \
  --tp-size 1 \
  --output prefill_data_v1_tp2.txt