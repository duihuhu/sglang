#!/usr/bin/env python3
"""Synthetic decode benchmark with pre-loaded batch.

Strategy: send N requests with max_new_tokens=1024, but first "pre-warm" by
sending them in a burst and waiting for all to enter running batch before
measuring steady-state TPOT.

Key trick: send all requests at once, then wait until running_batch stabilizes
at the target batch size. Measure TPOT only during the steady-state period.
"""
import asyncio, json, logging, os, re, socket, subprocess, sys, time, urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("preload")

PYTHON = "/workspace/env/sglang-tier/bin/python"
MODEL = "/models/Qwen/Qwen3-32B/"
HERE = os.path.dirname(os.path.abspath(__file__))
BOOTSTRAP = 18999
ALL_PORTS = [50020, 50021]

# Use long output so requests stay in decode long enough for batch to build up
OUTPUT_TOKENS = 2048
BATCH_TARGETS = [256]  # Target batch sizes to test


def kill_all():
    for t in ["sglang.launch_server", "sglang_router"]:
        subprocess.run(["pkill", "-f", t], capture_output=True)
    time.sleep(3)
    for t in ["sglang.launch_server", "sglang_router"]:
        subprocess.run(["pkill", "-9", "-f", t], capture_output=True)
    time.sleep(3)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        out = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True).stdout
        if not any(f":{p}" in out for p in ALL_PORTS): break
        time.sleep(3)

def cleanup_procs(procs):
    for _, p, f in procs:
        try: os.killpg(os.getpgid(p.pid), 9)
        except:
            try: p.kill()
            except: pass
        try: f.close()
        except: pass
    time.sleep(3)

def wait_port(host, port, timeout=300):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            s = socket.create_connection((host, port), timeout=2); s.close(); return True
        except: time.sleep(2)
    return False

def wait_health(url, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try: urllib.request.urlopen(url, timeout=5); return True
        except: time.sleep(3)
    return False

def warmup(url, n=8):
    import requests, concurrent.futures
    payloads = [{"text": f"Hello {i}", "sampling_params": {"max_new_tokens": 8, "temperature": 0.0}} for i in range(n)]
    def _send(p):
        try: requests.post(url + "/generate", json=p, timeout=120)
        except: pass
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        list(ex.map(_send, payloads))
    time.sleep(2)


def start_decode_only(micro_batch: int, async_pipeline: bool = False):
    kill_all(); time.sleep(3)
    procs = []
    env_base = os.environ.copy()
    env_base['SGLANG_DISABLE_REQUEST_LOGGING'] = 'true'
    env_base['AFD_UCX_TLS'] = 'rc,tcp,cuda_copy,cuda_ipc'
    env_base['UCX_LOG_LEVEL'] = 'fatal'
    if async_pipeline:
        env_base['AFD_ASYNC_PIPELINE'] = '1'

    extra = [
        '--skip-server-warmup', '--disable-cuda-graph', '--disable-piecewise-cuda-graph',
        '--afd-micro-batch', str(micro_batch),
        '--afd-disagg-interleave-poll', '--disable-radix-cache',
        '--num-reserved-decode-tokens', '64',
        '--tokenizer-worker-num', '4',  # Faster tokenization to fill batch quicker
    ]
    if async_pipeline:
        extra.append('--afd-async-pipeline')

    def _start(name, perspective, port, ucx_base, sched_port,
               visible_gpus, base_gpu_id, peer_device, ffn_host=None):
        env = env_base.copy()
        env['CUDA_VISIBLE_DEVICES'] = visible_gpus
        env['AFD_UCX_BASE_PORT'] = str(ucx_base)
        env['AFD_SCHED_PORT'] = str(sched_port)
        env['AFD_IPC_SYNC_MODE'] = 'ipc_event'
        env['AFD_IPC_PEER_DEVICE'] = str(peer_device)
        env['SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT'] = '600'
        env['SGLANG_DISAGGREGATION_WAITING_TIMEOUT'] = '600'
        if ffn_host: env['AFD_UCX_FFN_HOST'] = ffn_host
        cmd = [PYTHON, '-m', 'sglang.launch_server',
            '--model-path', MODEL, '--tp', '1',
            '--host', '127.0.0.1', '--port', str(port),
            '--afd-perspective', perspective,
            '--afd-comm-backend', 'ipc_cpp',
            '--mem-fraction-static', '0.85',
            '--disaggregation-mode', 'decode',
            '--disaggregation-transfer-backend', 'fake',
            '--disaggregation-bootstrap-port', str(BOOTSTRAP),
            '--disaggregation-ib-device', 'mlx5_4',
            '--base-gpu-id', str(base_gpu_id),
        ] + extra
        tag = f"m{micro_batch}_{'async' if async_pipeline else 'sync'}"
        fh = open(os.path.join(HERE, f'preload_{tag}_{name}.log'), 'w')
        p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        procs.append((name, p, fh))

    d_vis = '4,5'
    _start('df', 'ffn', 50021, 25300, 65500, d_vis, 0, 1)
    time.sleep(2)
    _start('da', 'attn', 50020, 25300, 65500, d_vis, 1, 0, '127.0.0.1')
    if not wait_port('127.0.0.1', 50020, 300) or not wait_port('127.0.0.1', 50021, 300):
        cleanup_procs(procs); return None
    time.sleep(5)
    if not wait_health('http://127.0.0.1:50020/health', 180):
        cleanup_procs(procs); return None
    warmup('http://127.0.0.1:50020')
    return procs, 'http://127.0.0.1:50020'


import aiohttp
import requests as sync_requests


def send_batch_sync(url, n, max_tokens):
    """Send n requests synchronously (fire-and-forget via threads)."""
    import concurrent.futures
    payloads = [{"text": "Hello " * 8, "sampling_params": {"max_new_tokens": max_tokens, "temperature": 0.0}} for _ in range(n)]
    def _send(p):
        try:
            sync_requests.post(url + "/generate", json=p, timeout=600)
        except:
            pass
    # Fire all in threads (non-blocking from main thread's perspective)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=n)
    futures = [executor.submit(_send, p) for p in payloads]
    return executor, futures


def poll_running_batch(da_log, target, timeout=60):
    """Wait until running_batch reaches target size."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(2)
        if not os.path.exists(da_log):
            continue
        with open(da_log) as f:
            lines = f.readlines()
        for line in reversed(lines):
            m = re.search(r'#running-req: (\d+)', line)
            if m:
                current = int(m.group(1))
                if current >= target:
                    return current
                break
    return 0


def measure_steady_state_tpot(da_log, wait_seconds=10):
    """Measure gen throughput from DA log during steady state."""
    time.sleep(wait_seconds)  # Let it run for a while at steady state
    
    # Read throughput from last few log entries
    throughputs = []
    batch_sizes = []
    if os.path.exists(da_log):
        with open(da_log) as f:
            lines = f.readlines()
        for line in lines[-20:]:
            m_thru = re.search(r'gen throughput \(token/s\): ([\d.]+)', line)
            m_batch = re.search(r'#running-req: (\d+)', line)
            if m_thru:
                throughputs.append(float(m_thru.group(1)))
            if m_batch:
                batch_sizes.append(int(m_batch.group(1)))
    
    if throughputs and batch_sizes:
        avg_thru = sum(throughputs) / len(throughputs)
        avg_batch = sum(batch_sizes) / len(batch_sizes)
        # TPOT = batch_size / throughput * 1000 (ms per token per request)
        tpot_ms = avg_batch / avg_thru * 1000 if avg_thru > 0 else 0
        return {
            'throughput_tok_s': round(avg_thru, 1),
            'avg_batch': round(avg_batch, 1),
            'tpot_ms': round(tpot_ms, 1),
        }
    return None


def main():
    configs = [
        ("M=1", 1, False),
        ("M=2 async", 2, True),
    ]

    all_results = {}

    for target_batch in BATCH_TARGETS:
        log.info('\n' + '=' * 70)
        log.info('  Target batch = %d', target_batch)
        log.info('=' * 70)

        for label, m, async_pipe in configs:
            log.info('\n>>> %s @ target_batch=%d <<<', label, target_batch)
            ret = start_decode_only(m, async_pipeline=async_pipe)
            if not ret:
                log.error('%s: Failed to start', label)
                kill_all(); time.sleep(10); continue
            procs, url = ret

            tag = f"m{m}_{'async' if async_pipe else 'sync'}"
            da_log = os.path.join(HERE, f'preload_{tag}_da.log')

            # Step 1: Send target_batch requests with long output (fire-and-forget)
            log.info('%s: Sending %d requests (out=%d)...', label, target_batch, OUTPUT_TOKENS)
            executor, futures = send_batch_sync(url, target_batch, OUTPUT_TOKENS)

            # Step 2: Wait for running batch to reach target
            log.info('%s: Waiting for batch to reach %d...', label, target_batch)
            actual_batch = poll_running_batch(da_log, target_batch * 0.8, timeout=120)
            log.info('%s: Batch reached %d', label, actual_batch)

            if actual_batch < target_batch * 0.5:
                log.warning('%s: Batch only reached %d (target %d), measuring anyway',
                           label, actual_batch, target_batch)

            # Step 3: Measure steady-state TPOT for 15 seconds
            log.info('%s: Measuring steady-state TPOT (15s)...', label)
            result = measure_steady_state_tpot(da_log, wait_seconds=15)

            if result:
                key = f"{label}@bs={target_batch}"
                all_results[key] = result
                log.info('%s: TPOT=%.1fms Thru=%.1f tok/s AvgBatch=%.0f',
                         label, result['tpot_ms'], result['throughput_tok_s'], result['avg_batch'])
            else:
                log.warning('%s: Failed to measure', label)

            # Cleanup
            executor.shutdown(wait=False)
            cleanup_procs(procs); kill_all(); time.sleep(10)

    # Summary
    print('\n' + '=' * 90)
    print(f'  Pre-loaded Batch Results (out={OUTPUT_TOKENS})')
    print('=' * 90)
    for target_batch in BATCH_TARGETS:
        print(f'\n  Target batch = {target_batch}:')
        for label, _, _ in configs:
            key = f"{label}@bs={target_batch}"
            r = all_results.get(key)
            if r:
                print(f'    {label:<15} TPOT={r["tpot_ms"]:.1f}ms  Thru={r["throughput_tok_s"]:.1f} tok/s  AvgBatch={r["avg_batch"]:.0f}')
            else:
                print(f'    {label:<15} FAILED')

        m1 = all_results.get(f"M=1@bs={target_batch}")
        m2 = all_results.get(f"M=2 async@bs={target_batch}")
        if m1 and m2:
            tpot_d = (m2['tpot_ms'] / m1['tpot_ms'] - 1) * 100
            thru_d = (m2['throughput_tok_s'] / m1['throughput_tok_s'] - 1) * 100
            print(f'    → M=2 vs M=1: TPOT {tpot_d:+.1f}%  Throughput {thru_d:+.1f}%')
    print('=' * 90)

    with open(os.path.join(HERE, 'preload_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)


if __name__ == '__main__':
    main()
