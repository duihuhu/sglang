#!/usr/bin/env python3
"""Synthetic decode-only benchmark: bypass prefill/transfer, test pure decode throughput.

Starts only DA+DF (decode attn + decode ffn) with fake transfer backend.
Sends requests directly to DA, which skips KV transfer and immediately starts decode.
Tests M=1 vs M=2 at batch=128, 256, 384.
"""
import asyncio, json, logging, os, re, socket, subprocess, sys, time, urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("synth_decode")

PYTHON = "/workspace/env/sglang-tier/bin/python"
MODEL = "/models/Qwen/Qwen3-32B/"
HERE = os.path.dirname(os.path.abspath(__file__))
BOOTSTRAP = 18999
ALL_PORTS = [50020, 50021]

OUTPUT_TOKENS = 128
BATCH_SIZES = [128, 256, 384]


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
    """Start only DA + DF with fake transfer (no prefill needed)."""
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
        '--num-reserved-decode-tokens', '64',  # Minimal reservation for max batch
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
            '--disaggregation-transfer-backend', 'fake',  # Skip KV transfer!
            '--disaggregation-bootstrap-port', str(BOOTSTRAP),
            '--disaggregation-ib-device', 'mlx5_4',
            '--base-gpu-id', str(base_gpu_id),
        ] + extra
        tag = f"m{micro_batch}_{'async' if async_pipeline else 'sync'}"
        fh = open(os.path.join(HERE, f'synth_{tag}_{name}.log'), 'w')
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

async def measure(url, batch_size):
    """Send batch_size requests simultaneously, all with output_tokens."""
    prompt = 'Hello ' * 8
    payload = {'text': prompt, 'sampling_params': {'max_new_tokens': OUTPUT_TOKENS, 'temperature': 0.0}, 'stream': True}
    sem = asyncio.Semaphore(batch_size)
    timeout = aiohttp.ClientTimeout(total=300)

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
        raw = await asyncio.gather(*[single(session) for _ in range(batch_size)])
    wall_s = time.perf_counter() - wall_start
    results = [r for r in raw if r]
    if not results: return None
    tpots = sorted(r['tpot_ms'] for r in results)
    return {
        'n_ok': len(results), 'n_total': batch_size,
        'mean_tpot_ms': round(sum(r['tpot_ms'] for r in results) / len(results), 1),
        'p50_tpot_ms': round(tpots[len(tpots)//2], 1),
        'throughput_tok_s': round(sum(r['tokens'] for r in results) / wall_s, 1),
    }


def get_max_batch(da_log):
    max_batch = 0
    if os.path.exists(da_log):
        with open(da_log) as f:
            for line in f:
                m = re.search(r'#running-req: (\d+)', line)
                if m:
                    max_batch = max(max_batch, int(m.group(1)))
    return max_batch


def main():
    configs = [
        ("M=1", 1, False),
        ("M=2 async", 2, True),
    ]

    all_results = {}

    for label, m, async_pipe in configs:
        log.info('\n>>> Starting %s <<<', label)
        ret = start_decode_only(m, async_pipeline=async_pipe)
        if not ret:
            log.error('%s: Failed to start', label); kill_all(); time.sleep(10); continue
        procs, url = ret

        for batch_size in BATCH_SIZES:
            key = f"{label}@bs={batch_size}"
            log.info('%s: batch=%d, out=%d', key, batch_size, OUTPUT_TOKENS)
            r = asyncio.run(measure(url, batch_size))

            tag = f"m{m}_{'async' if async_pipe else 'sync'}"
            da_log = os.path.join(HERE, f'synth_{tag}_da.log')
            max_batch = get_max_batch(da_log)

            if r:
                all_results[key] = {**r, 'max_batch': max_batch}
                log.info('%s: TPOT=%.1fms Thru=%.1f tok/s (%d/%d ok) MaxBatch=%d',
                         key, r['mean_tpot_ms'], r['throughput_tok_s'], r['n_ok'], r['n_total'], max_batch)
            else:
                all_results[key] = None
                log.warning('%s: FAILED', key)
            time.sleep(5)

        cleanup_procs(procs); kill_all(); time.sleep(10)

    # Summary
    print('\n' + '=' * 100)
    print(f'  Synthetic Decode-Only Results (out={OUTPUT_TOKENS}, fake transfer)')
    print('=' * 100)
    print(f'{"Config":<25} {"TPOT":<12} {"Throughput":<15} {"MaxBatch":<10} {"OK":<10}')
    print('-' * 75)
    for batch_size in BATCH_SIZES:
        for label, _, _ in configs:
            key = f"{label}@bs={batch_size}"
            r = all_results.get(key)
            if r:
                print(f'{key:<25} {r["mean_tpot_ms"]:<10.1f}ms {r["throughput_tok_s"]:<13.1f}tok/s {r["max_batch"]:<10} {r["n_ok"]}/{r["n_total"]}')
            else:
                print(f'{key:<25} FAILED')
        print()

    # Delta at each batch size
    print('  Comparison:')
    for batch_size in BATCH_SIZES:
        m1 = all_results.get(f"M=1@bs={batch_size}")
        m2 = all_results.get(f"M=2 async@bs={batch_size}")
        if m1 and m2:
            tpot_d = (m2['mean_tpot_ms'] / m1['mean_tpot_ms'] - 1) * 100
            thru_d = (m2['throughput_tok_s'] / m1['throughput_tok_s'] - 1) * 100
            print(f'    bs={batch_size}: M=2 vs M=1 → TPOT {tpot_d:+.1f}%  Throughput {thru_d:+.1f}%')
    print('=' * 100)

    with open(os.path.join(HERE, 'synth_decode_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)


if __name__ == '__main__':
    main()
