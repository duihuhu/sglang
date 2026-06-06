#!/usr/bin/env python3
"""Real PD+AF system test with high transfer concurrency to reach batch=256.

Key changes to reach batch=256:
1. SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=128 (from default 8)
2. SGLANG_DISAGGREGATION_QUEUE_SIZE=32 (from default 4)
3. output=2048 (long decode to let batch accumulate)
4. Send 300 requests simultaneously
5. --num-reserved-decode-tokens 64 (reduce KV reservation)
"""
import asyncio, json, logging, os, re, socket, subprocess, sys, time, urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("real_pd_256")

PYTHON = "/workspace/env/sglang-tier/bin/python"
MODEL = "/models/Qwen/Qwen3-32B/"
HERE = os.path.dirname(os.path.abspath(__file__))
BOOTSTRAP = 18999
ALL_PORTS = [50000, 50010, 50011, 50020, 50021]

OUTPUT_TOKENS = 200
N_REQUESTS = 256
CONCURRENCY = 256


def kill_all():
    for t in ["sglang.launch_server", "sglang_router", "sglang::router"]:
        subprocess.run(["pkill", "-f", t], capture_output=True)
    time.sleep(5)
    for t in ["sglang.launch_server", "sglang_router", "sglang::router"]:
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
    time.sleep(3)


def start_pdaf(micro_batch: int, async_pipeline: bool = False):
    kill_all(); time.sleep(3)
    procs = []
    env_base = os.environ.copy()
    env_base['SGLANG_DISABLE_REQUEST_LOGGING'] = 'true'
    env_base['AFD_UCX_TLS'] = 'rc,tcp,cuda_copy,cuda_ipc'
    env_base['UCX_LOG_LEVEL'] = 'fatal'
    # Key: increase transfer concurrency
    env_base['SGLANG_DISAGGREGATION_THREAD_POOL_SIZE'] = '128'
    env_base['SGLANG_DISAGGREGATION_QUEUE_SIZE'] = '32'
    if async_pipeline:
        env_base['AFD_ASYNC_PIPELINE'] = '1'

    extra = [
        '--skip-server-warmup', '--disable-cuda-graph', '--disable-piecewise-cuda-graph',
        '--afd-micro-batch', str(micro_batch),
        '--afd-disagg-interleave-poll', '--disable-radix-cache',
        '--num-reserved-decode-tokens', '64',
    ]
    if async_pipeline:
        extra.append('--afd-async-pipeline')

    def _start(name, perspective, disagg_mode, port, ucx_base, sched_port,
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
            '--disaggregation-mode', disagg_mode,
            '--disaggregation-transfer-backend', 'mooncake',
            '--disaggregation-bootstrap-port', str(BOOTSTRAP),
            '--disaggregation-ib-device', 'mlx5_4',
            '--base-gpu-id', str(base_gpu_id),
        ] + extra
        tag = f"m{micro_batch}_{'async' if async_pipeline else 'sync'}"
        fh = open(os.path.join(HERE, f'real256_{tag}_{name}.log'), 'w')
        p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        procs.append((name, p, fh))

    p_vis, d_vis = '6,7', '4,5'
    _start('pf', 'ffn', 'prefill', 50011, 25200, 65400, p_vis, 0, 1)
    time.sleep(2)
    _start('pa', 'attn', 'prefill', 50010, 25200, 65400, p_vis, 1, 0, '127.0.0.1')
    if not wait_port('127.0.0.1', 50010, 300) or not wait_port('127.0.0.1', 50011, 300):
        cleanup_procs(procs); return None
    _start('df', 'ffn', 'decode', 50021, 25300, 65500, d_vis, 0, 1)
    time.sleep(2)
    _start('da', 'attn', 'decode', 50020, 25300, 65500, d_vis, 1, 0, '127.0.0.1')
    if not wait_port('127.0.0.1', 50020, 300) or not wait_port('127.0.0.1', 50021, 300):
        cleanup_procs(procs); return None
    time.sleep(5)
    router_cmd = [PYTHON, '-m', 'sglang_router.launch_router',
        '--pd-disaggregation', '--mini-lb',
        '--prefill', 'http://127.0.0.1:50010', '--decode', 'http://127.0.0.1:50020',
        '--host', '127.0.0.1', '--port', '50000']
    rf = open(os.path.join(HERE, f'real256_m{micro_batch}_router.log'), 'w')
    rp = subprocess.Popen(router_cmd, env=os.environ.copy(), stdout=rf, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(('router', rp, rf))
    if not wait_health('http://127.0.0.1:50000/health', 180):
        cleanup_procs(procs); return None
    warmup('http://127.0.0.1:50000')
    return procs, 'http://127.0.0.1:50000'


import aiohttp

async def send_requests_and_monitor(url, da_log_path, target_batch=256, monitor_duration=20):
    """Send all requests, wait for batch to reach target, then measure throughput."""
    prompt = 'Hello ' * 8
    payload = {'text': prompt, 'sampling_params': {'max_new_tokens': OUTPUT_TOKENS, 'temperature': 0.0}}
    timeout = aiohttp.ClientTimeout(total=600)
    sem = asyncio.Semaphore(CONCURRENCY)

    results = []
    
    async def single(session, idx):
        async with sem:
            t0 = time.perf_counter()
            try:
                async with session.post(f'{url}/generate', json=payload) as resp:
                    result = await resp.json()
                    elapsed = time.perf_counter() - t0
                    return elapsed
            except:
                return None

    # Send all requests in background
    connector = aiohttp.TCPConnector(limit=0)  # No connection limit!
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [asyncio.create_task(single(session, i)) for i in range(N_REQUESTS)]
        
        # Monitor batch size from DA log
        max_batch = 0
        batch_reached_target = False
        start_time = time.monotonic()
        
        while time.monotonic() - start_time < 120:  # max 2 min wait
            await asyncio.sleep(3)
            if os.path.exists(da_log_path):
                with open(da_log_path) as f:
                    for line in f:
                        m = re.search(r'#running-req: (\d+)', line)
                        if m:
                            batch = int(m.group(1))
                            max_batch = max(max_batch, batch)
                if max_batch >= target_batch * 0.8:
                    batch_reached_target = True
                    break
        
        if not batch_reached_target:
            log.warning('Batch only reached %d (target %d)', max_batch, target_batch)
        else:
            log.info('Batch reached %d', max_batch)
        
        # Measure throughput from DA log for 15 seconds
        await asyncio.sleep(15)
        
        # Get throughput from log
        throughputs = []
        if os.path.exists(da_log_path):
            with open(da_log_path) as f:
                for line in f:
                    m = re.search(r'gen throughput \(token/s\): ([\d.]+)', line)
                    if m:
                        throughputs.append(float(m.group(1)))
        
        # Wait for all requests to complete (or timeout)
        done, pending = await asyncio.wait(tasks, timeout=300)
        for t in pending:
            t.cancel()
        
        results = [t.result() for t in done if t.result() is not None]
    
    # Calculate metrics
    avg_thru = sum(throughputs[-10:]) / len(throughputs[-10:]) if len(throughputs) >= 10 else (sum(throughputs) / len(throughputs) if throughputs else 0)
    
    return {
        'max_batch': max_batch,
        'avg_throughput': round(avg_thru, 1),
        'n_ok': len(results),
        'n_total': N_REQUESTS,
    }


def main():
    configs = [
        ("M=1", 1, False),
        ("M=2 async", 2, True),
    ]

    all_results = {}

    for label, m, async_pipe in configs:
        log.info('\n>>> Starting %s (out=%d, conc=%d) <<<', label, OUTPUT_TOKENS, CONCURRENCY)
        ret = start_pdaf(m, async_pipeline=async_pipe)
        if not ret:
            log.error('%s: Failed to start', label); kill_all(); time.sleep(10); continue
        procs, url = ret

        tag = f"m{m}_{'async' if async_pipe else 'sync'}"
        da_log = os.path.join(HERE, f'real256_{tag}_da.log')

        log.info('%s: Sending %d requests and monitoring batch...', label, N_REQUESTS)
        r = asyncio.run(send_requests_and_monitor(url, da_log, target_batch=256))

        if r:
            all_results[label] = r
            log.info('%s: MaxBatch=%d AvgThru=%.1f tok/s (%d/%d ok)',
                     label, r['max_batch'], r['avg_throughput'], r['n_ok'], r['n_total'])
        else:
            all_results[label] = None
            log.warning('%s: FAILED', label)

        cleanup_procs(procs); kill_all(); time.sleep(10)

    # Summary
    print('\n' + '=' * 90)
    print(f'  Real PD+AF System (out={OUTPUT_TOKENS}, conc={CONCURRENCY}, transfer_threads=128)')
    print('=' * 90)
    for label, _, _ in configs:
        r = all_results.get(label)
        if r:
            print(f'  {label:<15} MaxBatch={r["max_batch"]}  AvgThru={r["avg_throughput"]:.1f} tok/s  [{r["n_ok"]}/{r["n_total"]} ok]')
        else:
            print(f'  {label:<15} FAILED')

    m1 = all_results.get("M=1")
    m2 = all_results.get("M=2 async")
    if m1 and m2 and m1['avg_throughput'] > 0:
        thru_d = (m2['avg_throughput'] / m1['avg_throughput'] - 1) * 100
        print(f'\n  M=2 vs M=1: Throughput {thru_d:+.1f}%')
    print('=' * 90)

    with open(os.path.join(HERE, 'real256_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)


if __name__ == '__main__':
    main()
