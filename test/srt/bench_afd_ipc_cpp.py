"""Latency benchmark: C++ IPC vs Python IPC round-trip."""
import torch
import os
import sys
import time
import subprocess

PYTHON = '/workspace/env/sglang-tier/bin/python'
REPO_ROOT = '/workspace/sglang-tier'


def run_bench(backend, sync_mode, bs=1, hidden=5120, iters=200, warmup=50):
    """Run benchmark with given config, return results dict."""

    FFN_SCRIPT = f"""
import torch, os, sys, time
os.chdir('{REPO_ROOT}')
sys.path.insert(0, '{REPO_ROOT}/python')
torch.cuda.set_device(1)

from torch.utils.cpp_extension import load
csrc = 'sgl-kernel/csrc/afd_ipc'
mod = load(name='afd_ipc_cpp',
    sources=[os.path.join(csrc, f) for f in ['afd_ipc.cpp', 'afd_ipc_kernels.cu', 'afd_ipc_pybind.cpp']],
    extra_include_paths=[csrc],
    extra_cflags=['-O3', '-std=c++17'],
    extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr', '-gencode=arch=compute_80,code=sm_80'],
    extra_ldflags=['-lpthread', '-lrt'], verbose=False)

comm = mod.AfdIpcComm(True, 1, 0, 802, -1, '{sync_mode}')
comm.handshake()

ack = torch.ones(1, dtype=torch.bfloat16, device='cuda:1')
for i in range({warmup + iters}):
    data = comm.recv_tensor()
    comm.send_tensor(ack)
"""

    ATTN_SCRIPT = f"""
import torch, os, sys, time
import numpy as np
os.chdir('{REPO_ROOT}')
sys.path.insert(0, '{REPO_ROOT}/python')
torch.cuda.set_device(0)

from torch.utils.cpp_extension import load
csrc = 'sgl-kernel/csrc/afd_ipc'
mod = load(name='afd_ipc_cpp',
    sources=[os.path.join(csrc, f) for f in ['afd_ipc.cpp', 'afd_ipc_kernels.cu', 'afd_ipc_pybind.cpp']],
    extra_include_paths=[csrc],
    extra_cflags=['-O3', '-std=c++17'],
    extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr', '-gencode=arch=compute_80,code=sm_80'],
    extra_ldflags=['-lpthread', '-lrt'], verbose=False)

time.sleep(2)
comm = mod.AfdIpcComm(False, 0, 1, 802, -1, '{sync_mode}')
comm.handshake()

x = torch.randn({bs}, {hidden}, dtype=torch.bfloat16, device='cuda:0')

# Warmup
for _ in range({warmup}):
    comm.send_tensor(x)
    _ = comm.recv_tensor()

# Benchmark
torch.cuda.synchronize()
latencies = []
for i in range({iters}):
    t0 = time.perf_counter()
    comm.send_tensor(x)
    _ = comm.recv_tensor()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    latencies.append((t1 - t0) * 1e6)

lat = np.array(latencies)
print(f'RESULT|{{lat.mean():.1f}}|{{np.median(lat):.1f}}|{{np.percentile(lat, 95):.1f}}|{{lat.min():.1f}}|{{lat.max():.1f}}')
"""

    os.system('rm -f /tmp/afd_ipc_cpp_802* /dev/shm/afd_ipc_cpp_*802*')

    ffn_proc = subprocess.Popen(
        [PYTHON, '-c', FFN_SCRIPT],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    time.sleep(1)

    attn_proc = subprocess.Popen(
        [PYTHON, '-c', ATTN_SCRIPT],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    try:
        attn_out, _ = attn_proc.communicate(timeout=60)
        ffn_proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        attn_proc.kill()
        ffn_proc.kill()
        return None

    for line in attn_out.split('\n'):
        if line.startswith('RESULT|'):
            parts = line.split('|')[1:]
            return {
                'mean': float(parts[0]),
                'median': float(parts[1]),
                'p95': float(parts[2]),
                'min': float(parts[3]),
                'max': float(parts[4]),
            }
    print(f'  ERROR output: {attn_out[-500:]}')
    return None


def main():
    print('=== afd_ipc_cpp Latency Benchmark ===')
    print(f'GPU 0: {torch.cuda.get_device_name(0)}')
    print(f'GPU 1: {torch.cuda.get_device_name(1)}')
    print()

    configs = [
        ('C++ IPC (cpu_flag)', 'cpp', 'cpu_flag'),
        ('C++ IPC (ipc_event)', 'cpp', 'ipc_event'),
    ]

    # Test with typical decode tensor: [1, 5120] bf16 = 10KB
    bs, hidden = 1, 5120
    data_kb = bs * hidden * 2 / 1024
    print(f'Tensor: [{bs}, {hidden}] bf16 = {data_kb:.1f} KB')
    print(f'Iters: 200 (warmup: 50)')
    print()
    print(f'{"Config":<30} {"Mean":>8} {"Median":>8} {"P95":>8} {"Min":>8}')
    print('-' * 70)

    for name, backend, sync_mode in configs:
        result = run_bench(backend, sync_mode, bs=bs, hidden=hidden)
        if result:
            print(f'{name:<30} {result["mean"]:>7.1f}us {result["median"]:>7.1f}us {result["p95"]:>7.1f}us {result["min"]:>7.1f}us')
        else:
            print(f'{name:<30} FAILED')

    # Also test with larger tensor: [257, 5120] bf16 = 2.5MB (typical prefill)
    print()
    bs, hidden = 257, 5120
    data_kb = bs * hidden * 2 / 1024
    print(f'Tensor: [{bs}, {hidden}] bf16 = {data_kb:.1f} KB')
    print()
    print(f'{"Config":<30} {"Mean":>8} {"Median":>8} {"P95":>8} {"Min":>8}')
    print('-' * 70)

    for name, backend, sync_mode in configs:
        result = run_bench(backend, sync_mode, bs=bs, hidden=hidden)
        if result:
            print(f'{name:<30} {result["mean"]:>7.1f}us {result["median"]:>7.1f}us {result["p95"]:>7.1f}us {result["min"]:>7.1f}us')
        else:
            print(f'{name:<30} FAILED')


if __name__ == '__main__':
    main()
