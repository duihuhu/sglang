
CUDA_VISIBLE_DEVICES=4,5,6,7 python bash-test/bench_sglang.py \
  --server-url http://127.0.0.1:30000 \
  --batch_size 1024 \
  --input_len 128 \
  --output_len 512 \
  --ignore_eos


rm -rf /var/lib/apport/coredump/*


python bash-test/pivot_pd_ops_wide.py \
--csv bash-test/pd_latency_big_table_ut_both.csv \
--out-dir bash-test/pivot_pd_ops_out \
--metric energy_uj

CUDA_VISIBLE_DEVICES=6 python bash-test/energy/get_energy.py --mat-n 128

nohup python matmul_power_bench.py \
  --gpu 5,6 \
  --sizes 128,512,2048,8192,32768 \
  --gpu-clocks 210,510,810,1110,1410 \
  --dtype float16 \
  --measure-seconds 3 \
  --out-csv matmul_power_results.csv > matmul_power_results.log 2>&1 &

python plot_matmul_metrics.py \
  --csv matmul_power_results.csv \
  --out-dir matmul_plots