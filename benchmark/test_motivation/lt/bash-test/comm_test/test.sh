nohup /workspace/env/sglang-main/bin/python3 -m torch.distributed.run   --nproc_per_node=2   /workspace/benchmark/sglang-main/bash-test/comm_test/benchmark_comm_energy.py   \
--default-total-seq-batch-product \
--seq-lens 1,128,256,512,1024,2048,4096,8192   \
--batch-sizes 1,2,4,8,16,32,64,128,256  \
--hidden-sizes 5120  \
--no-energy \
--quiet \
--out-csv /workspace/benchmark/sglang-main/bash-test/comm_test/comm_benchmark_results.csv > /workspace/benchmark/sglang-main/bash-test/comm_test/test.log 2>&1 &
