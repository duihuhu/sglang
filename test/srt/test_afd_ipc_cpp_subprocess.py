"""Simple cross-process IPC handshake + send/recv test."""
import torch
import os
import sys
import time
import subprocess
import signal

PYTHON = '/workspace/env/sglang-tier/bin/python'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(SCRIPT_DIR, '..', '..')

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

print('[FFN] Module loaded, creating comm...', flush=True)
comm = mod.AfdIpcComm(True, 1, 0, 801, -1, 'cpu_flag')
print('[FFN] Calling handshake...', flush=True)
comm.handshake()
print(f'[FFN] Handshake done! is_ready={{comm.is_ready()}}', flush=True)

# Recv tensor
data = comm.recv_tensor()
print(f'[FFN] Received: shape={{list(data.shape)}}, dtype={{data.dtype}}, sum={{data.sum().item():.1f}}', flush=True)

# Verify
expected = torch.arange(2560, dtype=torch.bfloat16, device='cuda:1').reshape(10, 256)
match = torch.allclose(data, expected, atol=1.0)
print(f'[FFN] Data match: {{match}}', flush=True)

# Send ack
ack = torch.tensor([1.0 if match else 0.0], dtype=torch.bfloat16, device='cuda:1')
comm.send_tensor(ack)
print('[FFN] Sent ack, done.', flush=True)
"""

ATTN_SCRIPT = f"""
import torch, os, sys, time
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

print('[ATTN] Module loaded, creating comm...', flush=True)
time.sleep(2)  # Wait for FFN to start listening
comm = mod.AfdIpcComm(False, 0, 1, 801, -1, 'cpu_flag')
print('[ATTN] Calling handshake...', flush=True)
comm.handshake()
print(f'[ATTN] Handshake done! is_ready={{comm.is_ready()}}', flush=True)

# Send tensor
x = torch.arange(2560, dtype=torch.bfloat16, device='cuda:0').reshape(10, 256)
comm.send_tensor(x)
print(f'[ATTN] Sent tensor: shape={{list(x.shape)}}, sum={{x.sum().item():.1f}}', flush=True)

# Recv ack
ack = comm.recv_tensor()
print(f'[ATTN] Received ack: {{ack.item():.1f}}', flush=True)

if ack.item() > 0.5:
    print('[ATTN] === TEST PASSED ===', flush=True)
else:
    print('[ATTN] === TEST FAILED ===', flush=True)
    sys.exit(1)
"""


def main():
    # Clean up stale files
    os.system('rm -f /tmp/afd_ipc_cpp_801* /dev/shm/afd_ipc_cpp_*801*')

    print('=== Cross-Process IPC Test ===')
    print(f'GPU 0: {torch.cuda.get_device_name(0)}')
    print(f'GPU 1: {torch.cuda.get_device_name(1)}')
    print()

    # Start FFN (server) first
    ffn_proc = subprocess.Popen(
        [PYTHON, '-c', FFN_SCRIPT],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    time.sleep(1)

    # Start ATTN (client)
    attn_proc = subprocess.Popen(
        [PYTHON, '-c', ATTN_SCRIPT],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    # Wait and collect output
    try:
        attn_out, _ = attn_proc.communicate(timeout=45)
        ffn_out, _ = ffn_proc.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        print('TIMEOUT!')
        attn_proc.kill()
        ffn_proc.kill()
        attn_out, _ = attn_proc.communicate()
        ffn_out, _ = ffn_proc.communicate()

    print('--- FFN output ---')
    print(ffn_out)
    print('--- ATTN output ---')
    print(attn_out)

    if 'TEST PASSED' in attn_out:
        print('=== OVERALL: PASSED ===')
    else:
        print('=== OVERALL: FAILED ===')
        sys.exit(1)


if __name__ == '__main__':
    main()
