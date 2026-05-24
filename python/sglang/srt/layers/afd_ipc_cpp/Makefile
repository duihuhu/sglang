.PHONY: build clean test

# Build the afd_ipc_cpp extension (requires PyTorch + CUDA)
build:
	cd /workspace/sglang-tier && python3 -c "\
	from torch.utils.cpp_extension import load; \
	import os; \
	csrc = 'sgl-kernel/csrc/afd_ipc'; \
	mod = load( \
	    name='afd_ipc_cpp', \
	    sources=[ \
	        os.path.join(csrc, 'afd_ipc.cpp'), \
	        os.path.join(csrc, 'afd_ipc_kernels.cu'), \
	        os.path.join(csrc, 'afd_ipc_pybind.cpp'), \
	    ], \
	    extra_include_paths=[csrc], \
	    extra_cflags=['-O3', '-std=c++17'], \
	    extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr', \
	        '-gencode=arch=compute_80,code=sm_80', \
	        '-gencode=arch=compute_90,code=sm_90'], \
	    extra_ldflags=['-lpthread', '-lrt'], \
	    verbose=True, \
	); \
	print('Build successful:', mod)"

# Run unit tests
test:
	cd /workspace/sglang-tier && python3 -m pytest test/srt/test_afd_ipc_cpp.py -xvs

# Run benchmark (needs 2 GPUs)
bench:
	cd /workspace/sglang-tier && python3 benchmark/af_bench/bench_cpp_ipc.py \
		--launch --backend cpp --sync-mode ipc_event --iters 1000

bench-compare:
	@echo "=== C++ IPC (IPC_EVENT) ==="
	cd /workspace/sglang-tier && python3 benchmark/af_bench/bench_cpp_ipc.py \
		--launch --backend cpp --sync-mode ipc_event --iters 500
	@echo ""
	@echo "=== C++ IPC (CPU_FLAG) ==="
	cd /workspace/sglang-tier && python3 benchmark/af_bench/bench_cpp_ipc.py \
		--launch --backend cpp --sync-mode cpu_flag --iters 500
	@echo ""
	@echo "=== Python IPC (baseline) ==="
	cd /workspace/sglang-tier && python3 benchmark/af_bench/bench_cpp_ipc.py \
		--launch --backend python --sync-mode cpu_flag --iters 500

clean:
	rm -rf /root/.cache/torch_extensions/py*/afd_ipc_cpp*
	rm -f python/sglang/srt/layers/afd_ipc_cpp/*.so
