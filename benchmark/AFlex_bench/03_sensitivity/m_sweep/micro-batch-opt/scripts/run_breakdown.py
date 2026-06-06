#!/usr/bin/env python3
"""Per-layer timing breakdown: M=1 vs M=2 interleaved.

Measures exact per-layer Attn/FFN/send/recv times using AFD_TIMING=1.
Runs a single request at batch=100 (simulated by 100 concurrent short requests).
"""
import asyncio, json, logging, os, socket, subprocess, sys, time, urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("breakdown")

PYTHON = "/workspace/env/sglang-tier/bin/python"
MODEL = "/models/Qwen/Qwen3-32B/"
HERE = os.path.dirname(os.path.abspath(__file__))
BOOTSTRAP = 18999
UCX_TLS = "rc,tcp,cuda_copy,cuda_ipc"
ALL_PORTS = [50000, 50010, 50011, 50020, 50021]


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
        if not any(f":{p}" in out for p in ALL_PORTS):
            break
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

def warmup(url, n=12):
    import requests, concurrent.futures
    payloads = [{"text": f"Hello {i}", "sampling_params": {"max_new_tokens": 16, "temperature": 0.0}} for i in range(n)]
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
    env_base['AFD_UCX_TLS'] = UCX_TLS
    env_base['UCX_LOG_LEVEL'] = 'fatal'
    env_base['AFD_TIMING'] = '1'  # Enable per-layer timing
    if async_pipeline:
        env_base['AFD_ASYNC_PIPELINE'] = '1'

    extra = [
        '--skip-server-warmup', '--disable-cuda-graph', '--disable-piecewise-cuda-graph',
        '--afd-micro-batch', str(micro_batch),
        '--afd-disagg-interleave-poll', '--disable-radix-cache',
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
        fh = open(os.path.join(HERE, f'bkdn_{tag}_{name}.log'), 'w')
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
    rf = open(os.path.join(HERE, f'bkdn_m{micro_batch}_router.log'), 'w')
    rp = subprocess.Popen(router_cmd, env=os.environ.copy(), stdout=rf, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(('router', rp, rf))
    if not wait_health('http://127.0.0.1:50000/health', 180):
        cleanup_procs(procs); return None
    warmup('http://127.0.0.1:50000')
    return procs, 'http://127.0.0.1:50000'


import aiohttp

async def run_concurrent(url, concurrency, n_total, output_tokens=32):
    """Run concurrent requests to build up batch, measure TPOT."""
    prompt = 'Hello world, this is a test prompt for timing.'
    payload = {'text': prompt, 'sampling_params': {'max_new_tokens': output_tokens, 'temperature': 0.0}}
    sem = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=120)

    async def single(session):
        async with sem:
            t0 = time.perf_counter()
            try:
                async with session.post(f'{url}/generate', json=payload) as resp:
                    result = await resp.json()
                    return time.perf_counter() - t0
            except:
                return None

    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = await asyncio.gather(*[single(session) for _ in range(n_total)])
    return [r for r in results if r is not None]


def main():
    log.info('=' * 70)
    log.info('  Per-layer Breakdown: M=1 vs M=2 interleaved @ batch~100')
    log.info('=' * 70)

    configs = [
        ("M=1", 1, False),
        ("M=2 interleaved", 2, True),
    ]

    for label, m, async_pipe in configs:
        log.info('\n>>> %s <<<', label)
        ret = start_pdaf(m, async_pipeline=async_pipe)
        if not ret:
            log.error('%s: Failed to start', label)
            kill_all(); time.sleep(10); continue
        procs, url = ret

        # Run 100 concurrent requests with 32 output tokens
        # This builds up batch~100 in decode phase
        log.info('%s: Running 100 concurrent requests (out=32)', label)
        results = asyncio.run(run_concurrent(url, 100, 100, output_tokens=32))
        log.info('%s: %d/%d completed, mean=%.1fms', label, len(results), 100,
                 sum(results)/len(results)*1000 if results else 0)

        # Wait a bit for timing logs to flush
        time.sleep(3)

        cleanup_procs(procs)
        kill_all()
        time.sleep(10)

    # Parse timing from DA logs
    print('\n' + '=' * 70)
    print('  Per-layer Breakdown Analysis')
    print('=' * 70)

    for label, m, async_pipe in configs:
        tag = f"m{m}_{'async' if async_pipe else 'sync'}"
        da_log = os.path.join(HERE, f'bkdn_{tag}_da.log')
        df_log = os.path.join(HERE, f'bkdn_{tag}_df.log')

        print(f'\n--- {label} ---')

        # Look for AFD_FWD_OVERHEAD or AFD_TIMING lines
        for logfile, role in [(da_log, 'DA'), (df_log, 'DF')]:
            if not os.path.exists(logfile):
                print(f'  {role}: log not found')
                continue
            with open(logfile) as f:
                lines = f.readlines()
            # Find timing lines
            timing_lines = [l for l in lines if 'AFD_FWD_OVERHEAD' in l or 'AFD_TIMING' in l or 'TPOT_BREAKDOWN' in l]
            if timing_lines:
                print(f'  {role} timing ({len(timing_lines)} entries):')
                # Show last 5
                for l in timing_lines[-5:]:
                    print(f'    {l.strip()}')
            else:
                # Try to find gen throughput lines
                gen_lines = [l for l in lines if 'gen throughput' in l]
                if gen_lines:
                    print(f'  {role} throughput:')
                    for l in gen_lines[-3:]:
                        print(f'    {l.strip()}')

    print('\n' + '=' * 70)


if __name__ == '__main__':
    main()
