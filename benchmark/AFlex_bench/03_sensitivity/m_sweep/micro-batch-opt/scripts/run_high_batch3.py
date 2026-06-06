#!/usr/bin/env python3
"""High batch test: M=1 vs M=2 with output=1024 to build up large decode batch.

With out=1024, decode takes ~70s per request. Send 500 requests at once,
they'll pile up in decode giving batch=300-500.
"""
import asyncio, json, logging, os, socket, subprocess, sys, time, urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("highbatch2")

PYTHON = "/workspace/env/sglang-tier/bin/python"
MODEL = "/models/Qwen/Qwen3-32B/"
HERE = os.path.dirname(os.path.abspath(__file__))
BOOTSTRAP = 18999
ALL_PORTS = [50000, 50010, 50011, 50020, 50021]

OUTPUT_TOKENS = 256
N_REQUESTS = 600
CONCURRENCY = 600


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

def warmup(url, n=16):
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
    env_base['AFD_UCX_TLS'] = 'rc,tcp,cuda_copy,cuda_ipc'
    env_base['UCX_LOG_LEVEL'] = 'fatal'
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
        fh = open(os.path.join(HERE, f'hb3_{tag}_{name}.log'), 'w')
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
    rf = open(os.path.join(HERE, f'hb3_m{micro_batch}_router.log'), 'w')
    rp = subprocess.Popen(router_cmd, env=os.environ.copy(), stdout=rf, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(('router', rp, rf))
    if not wait_health('http://127.0.0.1:50000/health', 180):
        cleanup_procs(procs); return None
    warmup('http://127.0.0.1:50000')
    return procs, 'http://127.0.0.1:50000'


import aiohttp

async def measure(url):
    prompt = 'Hello ' * 8
    payload = {'text': prompt, 'sampling_params': {'max_new_tokens': OUTPUT_TOKENS, 'temperature': 0.0}, 'stream': True}
    sem = asyncio.Semaphore(CONCURRENCY)
    timeout = aiohttp.ClientTimeout(total=600)

    async def single(session):
        async with sem:
            t0 = time.perf_counter(); first_token_time = None; token_count = 0
            try:
                async with session.post(f'{url}/generate', json=payload) as resp:
                    async for line in resp.content:
                        now = time.perf_counter()
                        text = line.decode().strip()
                        if not text or text.startswith(':'): continue
                        if text.startswith('data:'): text = text[5:].strip()
                        if text == '[DONE]': break
                        try:
                            json.loads(text)
                            if first_token_time is None: first_token_time = now - t0
                            token_count += 1
                        except: pass
            except: return None
            if first_token_time and token_count > 1:
                total = time.perf_counter() - t0
                return {'tpot_ms': (total - first_token_time) / (token_count - 1) * 1000,
                        'tokens': token_count}
            return None

    wall_start = time.perf_counter()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        raw = await asyncio.gather(*[single(session) for _ in range(N_REQUESTS)])
    wall_s = time.perf_counter() - wall_start
    results = [r for r in raw if r]
    if not results: return None
    tpots = sorted(r['tpot_ms'] for r in results)
    return {
        'n_ok': len(results), 'n_total': N_REQUESTS,
        'mean_tpot_ms': round(sum(r['tpot_ms'] for r in results) / len(results), 1),
        'p50_tpot_ms': round(tpots[len(tpots)//2], 1),
        'throughput_tok_s': round(sum(r['tokens'] for r in results) / wall_s, 1),
    }


def main():
    configs = [
        ("M=1", 1, False),
        ("M=2 async", 2, True),
    ]

    all_results = {}

    for label, m, async_pipe in configs:
        log.info('\n>>> Starting %s (out=%d, conc=%d, n=%d) <<<', label, OUTPUT_TOKENS, CONCURRENCY, N_REQUESTS)
        ret = start_pdaf(m, async_pipeline=async_pipe)
        if not ret:
            log.error('%s: Failed', label); kill_all(); time.sleep(10); continue
        procs, url = ret

        log.info('%s: Running benchmark...', label)
        r = asyncio.run(measure(url))
        if r:
            all_results[label] = r
            log.info('%s: TPOT=%.1fms Thru=%.1f tok/s (%d/%d ok)',
                     label, r['mean_tpot_ms'], r['throughput_tok_s'], r['n_ok'], r['n_total'])
        else:
            all_results[label] = None
            log.warning('%s: FAILED', label)

        # Check actual batch size from DA log
        tag = f"m{m}_{'async' if async_pipe else 'sync'}"
        da_log = os.path.join(HERE, f'hb3_{tag}_da.log')
        if os.path.exists(da_log):
            with open(da_log) as f:
                lines = [l for l in f if '#running-req' in l]
            if lines:
                # Extract max running-req
                import re
                max_batch = 0
                for l in lines:
                    m_match = re.search(r'#running-req: (\d+)', l)
                    if m_match:
                        max_batch = max(max_batch, int(m_match.group(1)))
                log.info('%s: Max decode batch observed = %d', label, max_batch)

        cleanup_procs(procs); kill_all(); time.sleep(10)

    # Summary
    print('\n' + '=' * 80)
    print(f'  High-Batch Results (out={OUTPUT_TOKENS}, conc={CONCURRENCY}, n={N_REQUESTS})')
    print('=' * 80)
    for label, _, _ in configs:
        r = all_results.get(label)
        if r:
            print(f'  {label:<15} TPOT={r["mean_tpot_ms"]:.1f}ms  p50={r["p50_tpot_ms"]:.1f}ms  Thru={r["throughput_tok_s"]:.1f} tok/s  [{r["n_ok"]}/{r["n_total"]} ok]')
        else:
            print(f'  {label:<15} FAILED')

    m1 = all_results.get("M=1")
    m2 = all_results.get("M=2 async")
    if m1 and m2:
        tpot_d = (m2['mean_tpot_ms'] / m1['mean_tpot_ms'] - 1) * 100
        thru_d = (m2['throughput_tok_s'] / m1['throughput_tok_s'] - 1) * 100
        print(f'\n  M=2 vs M=1: TPOT {tpot_d:+.1f}%  Throughput {thru_d:+.1f}%')
    print('=' * 80)

    with open(os.path.join(HERE, 'high_batch3_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)


if __name__ == '__main__':
    main()
