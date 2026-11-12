"""Cross-process IPC test for afd_ipc_cpp."""
import torch
import torch.multiprocessing as mp
import os
import sys
import time


def sender_fn(result_queue):
    """ATTN side: device 0, sends tensor to FFN on device 1."""
    try:
        torch.cuda.set_device(0)

        from torch.utils.cpp_extension import load
        csrc = os.path.join(os.path.dirname(__file__), '..', '..', 'sgl-kernel', 'csrc', 'afd_ipc')
        csrc = os.path.abspath(csrc)
        mod = load(name='afd_ipc_cpp',
            sources=[os.path.join(csrc, f) for f in ['afd_ipc.cpp', 'afd_ipc_kernels.cu', 'afd_ipc_pybind.cpp']],
            extra_include_paths=[csrc],
            extra_cflags=['-O3', '-std=c++17'],
            extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr', '-gencode=arch=compute_80,code=sm_80'],
            extra_ldflags=['-lpthread', '-lrt'], verbose=False)

        comm = mod.AfdIpcComm(False, 0, 1, 700, -1, 'cpu_flag')
        comm.handshake()

        # Create test tensor: [10, 256] bf16
        x = torch.arange(2560, dtype=torch.bfloat16, device='cuda:0').reshape(10, 256)

        # Warmup
        for _ in range(3):
            comm.send_tensor(x)
            _ = comm.recv_tensor()

        # Benchmark
        torch.cuda.synchronize()
        latencies = []
        for i in range(100):
            t0 = time.perf_counter()
            comm.send_tensor(x)
            _ = comm.recv_tensor()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1e6)

        import numpy as np
        lat = np.array(latencies)
        result_queue.put(('sender_ok', {
            'mean_us': float(lat.mean()),
            'median_us': float(np.median(lat)),
            'p95_us': float(np.percentile(lat, 95)),
            'min_us': float(lat.min()),
        }))
    except Exception as e:
        import traceback
        result_queue.put(('sender_error', f'{e}\n{traceback.format_exc()}'))


def receiver_fn(result_queue):
    """FFN side: device 1, receives tensor from ATTN on device 0."""
    try:
        torch.cuda.set_device(1)

        from torch.utils.cpp_extension import load
        csrc = os.path.join(os.path.dirname(__file__), '..', '..', 'sgl-kernel', 'csrc', 'afd_ipc')
        csrc = os.path.abspath(csrc)
        mod = load(name='afd_ipc_cpp',
            sources=[os.path.join(csrc, f) for f in ['afd_ipc.cpp', 'afd_ipc_kernels.cu', 'afd_ipc_pybind.cpp']],
            extra_include_paths=[csrc],
            extra_cflags=['-O3', '-std=c++17'],
            extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr', '-gencode=arch=compute_80,code=sm_80'],
            extra_ldflags=['-lpthread', '-lrt'], verbose=False)

        comm = mod.AfdIpcComm(True, 1, 0, 700, -1, 'cpu_flag')
        comm.handshake()

        # Warmup + benchmark: recv then send ack
        expected = torch.arange(2560, dtype=torch.bfloat16, device='cuda:1').reshape(10, 256)
        ack = torch.tensor([1.0], dtype=torch.bfloat16, device='cuda:1')

        all_match = True
        for i in range(3 + 100):
            data = comm.recv_tensor()
            if i == 0:
                # Verify first recv
                match = torch.allclose(data, expected, atol=1e-2)
                if not match:
                    all_match = False
            comm.send_tensor(ack)

        result_queue.put(('receiver_ok', {
            'data_match': all_match,
            'shape': list(data.shape),
            'dtype': str(data.dtype),
        }))
    except Exception as e:
        import traceback
        result_queue.put(('receiver_error', f'{e}\n{traceback.format_exc()}'))


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue()

    print('=== afd_ipc_cpp Cross-Process Test ===')
    print(f'PyTorch: {torch.__version__}')
    print(f'GPUs: {torch.cuda.device_count()}')
    print(f'GPU 0: {torch.cuda.get_device_name(0)}')
    print(f'GPU 1: {torch.cuda.get_device_name(1)}')
    print()

    p_recv = mp.Process(target=receiver_fn, args=(result_queue,))
    p_send = mp.Process(target=sender_fn, args=(result_queue,))

    p_recv.start()
    time.sleep(2)
    p_send.start()

    p_send.join(timeout=60)
    p_recv.join(timeout=60)

    results = {}
    while not result_queue.empty():
        item = result_queue.get()
        results[item[0]] = item[1]

    print('--- Results ---')
    if 'sender_error' in results:
        print(f'SENDER ERROR: {results["sender_error"]}')
    if 'receiver_error' in results:
        print(f'RECEIVER ERROR: {results["receiver_error"]}')

    if 'receiver_ok' in results:
        r = results['receiver_ok']
        print(f'Data integrity: {"PASS" if r["data_match"] else "FAIL"}')
        print(f'Received shape: {r["shape"]}')
        print(f'Received dtype: {r["dtype"]}')

    if 'sender_ok' in results:
        s = results['sender_ok']
        print()
        print(f'Round-trip latency (100 iters):')
        print(f'  Mean:   {s["mean_us"]:.1f} us')
        print(f'  Median: {s["median_us"]:.1f} us')
        print(f'  P95:    {s["p95_us"]:.1f} us')
        print(f'  Min:    {s["min_us"]:.1f} us')

    if 'receiver_ok' in results and results['receiver_ok']['data_match'] and 'sender_ok' in results:
        print()
        print('=== ALL TESTS PASSED ===')
    else:
        print()
        print('=== TESTS FAILED ===')
        sys.exit(1)
