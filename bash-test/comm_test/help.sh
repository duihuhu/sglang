torchrun --nproc_per_node=2 benchmark_comm_energy.py --seq-lens 8192 --batch-sizes 1 --hidden-sizes 5120

python3 benchmark_comm_energy_stability_sweep.py \
  --python /workspace/env/sglang-main/bin/python3 \
  --comm-iters-sweep 200,500,1000,2000,4000 \
  --metric latency_us_mean \
  -- \
  --seq-lens 128 --batch-sizes 1 --hidden-sizes 4096 --quiet --no-energy


/workspace/env/sglang-main/bin/python3 -m torch.distributed.run   --nproc_per_node=2   /workspace/benchmark/sglang-main/bash-test/comm_test/benchmark_comm_energy.py   \
--fixed-comm-iters 10   \
--fixed-idle-iters 1  \
--seq-lens 8192   \
--batch-sizes 256  \
--hidden-sizes 5120  \
--quiet \
--no-energy \
--out-csv /workspace/benchmark/sglang-main/bash-test/comm_test/comm_benchmark_results.csv