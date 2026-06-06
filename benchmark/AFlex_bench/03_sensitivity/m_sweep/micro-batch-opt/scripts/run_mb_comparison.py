#!/usr/bin/env python3
"""Micro-batch optimization benchmark: M=1 vs M=2 vs M=3 at high concurrency.

Tests PD+AF with different --afd-micro-batch values to measure the overlap
benefit at concurrency=128.

GPU allocation (A800-SXM4-80GB):
  PF=GPU6, PA=GPU7 (Prefill side, TP=1)
  DF=GPU4, DA=GPU5 (Decode side, TP=1)

For each M value, we:
  1. Start fresh PD+AF servers with --afd-micro-batch M
  2. Warmup
  3. Run concurrency=128 benchmark (in=16, out=128)
  4. Collect TPOT, TTFT, throughput
"""
import asyncio, json, logging, os, socket, subprocess, sys, time, urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("mb_opt")

PYTHON = "/workspace/env/sglang-tier/bin/python"
MODEL = "/models/Qwen/Qwen3-32B/"
HERE = os.path.dirname(os.path.abspath(__file__))
BOOTSTRAP = 18999
UCX_TLS = "rc,tcp,cuda_copy,cuda_ipc"
ALL_PORTS = [50000, 50010, 50011, 50020, 50021]

INPUT_LEN = 16
OUTPUT_TOKENS = 128
CONCURRENCY = 128
N_REQUESTS = 256
M_VALUES = [1, 2, 3]


def _ports_free():
    out = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True).stdout
    return [p for p in ALL_PORTS if f":{p}" in out]


def kill_all():
    for t in ["sglang.launch_server", "sglang_router", "sglang::router", "sglang::scheduler"]:
        subprocess.run(["pkill", "-f", t], capture_output=True)
    time.sleep(5)
    for t in ["sglang.launch_server", "sglang_router", "sglang::router", "sglang::scheduler"]:
        subprocess.run(["pkill", "-9", "-f", t], capture_output=True)
    time.sleep(3)
    subprocess.run(["pkill", "-9", "-f", "ucx"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "ucp"], capture_output=True)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        busy = _ports_free()
        if not busy:
            break
        log.info("Waiting for ports: %s", busy)
        time.sleep(5)


def cleanup_procs(procs):
    for _, p, f in procs:
        try:
            os.killpg(os.getpgid(p.pid), 9)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
        try:
            f.close()
        except Exception:
            pass
    time.sleep(3)


def wait_port(host, port, timeout=300):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            s = socket.create_connection((host, port), timeout=2)
            s.close()
            return True
        except Exception:
            time.sleep(2)
    return False


def wait_health(url, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=5)
            return True
        except Exception:
            time.sleep(3)
    return False


def warmup(url, n=12):
    import requests, concurrent.futures
    payloads = [
        {"text": f"Hello world {i}, warmup:", "sampling_params": {"max_new_tokens": 16, "temperature": 0.0}}
        for i in range(n)
    ]
    def _send(p):
        try:
            requests.post(url + "/generate", json=p, timeout=120)
        except Exception:
            pass
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        list(ex.map(_send, payloads))
    time.sleep(3)


def start_pdaf(micro_batch: int):
    """Start PD+AF with given micro-batch count."""
    kill_all()
    time.sleep(3)
    procs = []
    env_base = os.environ.copy()
    env_base['SGLANG_DISABLE_REQUEST_LOGGING'] = 'true'
    env_base['AFD_UCX_TLS'] = UCX_TLS
    env_base['UCX_LOG_LEVEL'] = 'fatal'

    da_port, df_port = 50020, 50021
    pa_port, pf_port = 50010, 50011
    router_port = 50000
    ucx_p, ucx_d = 25200, 25300
    sched_p, sched_d = 65400, 65500

    extra = [
        '--skip-server-warmup', '--disable-cuda-graph', '--disable-piecewise-cuda-graph',
        '--afd-micro-batch', str(micro_batch),
        '--afd-disagg-interleave-poll',
        '--disable-radix-cache',
    ]

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
        if ffn_host:
            env['AFD_UCX_FFN_HOST'] = ffn_host
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
        logfile = os.path.join(HERE, f'mb{micro_batch}_{name}.log')
        fh = open(logfile, 'w')
        p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        procs.append((name, p, fh))

    # Prefill: PF(GPU6) + PA(GPU7)
    p_vis = '6,7'
    _start('pf', 'ffn', 'prefill', pf_port, ucx_p, sched_p, p_vis, 0, 1)
    time.sleep(2)
    _start('pa', 'attn', 'prefill', pa_port, ucx_p, sched_p, p_vis, 1, 0, ffn_host='127.0.0.1')

    if not wait_port('127.0.0.1', pa_port, timeout=300) or not wait_port('127.0.0.1', pf_port, timeout=300):
        log.error('M=%d: Prefill servers failed to start', micro_batch)
        cleanup_procs(procs)
        return None

    # Decode: DF(GPU4) + DA(GPU5)
    d_vis = '4,5'
    _start('df', 'ffn', 'decode', df_port, ucx_d, sched_d, d_vis, 0, 1)
    time.sleep(2)
    _start('da', 'attn', 'decode', da_port, ucx_d, sched_d, d_vis, 1, 0, ffn_host='127.0.0.1')

    if not wait_port('127.0.0.1', da_port, timeout=300) or not wait_port('127.0.0.1', df_port, timeout=300):
        log.error('M=%d: Decode servers failed to start', micro_batch)
        cleanup_procs(procs)
        return None

    time.sleep(5)

    # Router
    router_cmd = [PYTHON, '-m', 'sglang_router.launch_router',
        '--pd-disaggregation', '--mini-lb',
        '--prefill', f'http://127.0.0.1:{pa_port}',
        '--decode', f'http://127.0.0.1:{da_port}',
        '--host', '127.0.0.1', '--port', str(router_port),
    ]
    rf = open(os.path.join(HERE, f'mb{micro_batch}_router.log'), 'w')
    rp = subprocess.Popen(router_cmd, env=os.environ.copy(), stdout=rf, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(('router', rp, rf))
    if not wait_health(f'http://127.0.0.1:{router_port}/health', timeout=180):
        log.error('M=%d: Router failed', micro_batch)
        cleanup_procs(procs)
        return None

    url = f'http://127.0.0.1:{router_port}'
    warmup(url)
    return procs, url


import aiohttp

async def measure_concurrent(url, input_len, output_len, concurrency, n_total):
    """Measure TPOT/TTFT/throughput at given concurrency."""
    prompt = 'Hello ' * (input_len // 2)
    payload = {'text': prompt, 'sampling_params': {'max_new_tokens': output_len, 'temperature': 0.0}, 'stream': True}
    sem = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=300)

    async def single(session):
        async with sem:
            t0 = time.perf_counter()
            first_token_time = None
            token_count = 0
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
            except Exception:
                return None
            if first_token_time and token_count > 1:
                total = time.perf_counter() - t0
                tpot = (total - first_token_time) / (token_count - 1)
                return {'tpot_ms': tpot * 1000, 'ttft_ms': first_token_time * 1000, 'tokens': token_count}
            return None

    wall_start = time.perf_counter()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [single(session) for _ in range(n_total)]
        raw = await asyncio.gather(*tasks)
    wall_s = time.perf_counter() - wall_start

    results = [r for r in raw if r is not None]
    if not results:
        return None
    tpots = sorted(r['tpot_ms'] for r in results)
    total_output_tokens = sum(r['tokens'] for r in results)
    throughput = total_output_tokens / wall_s
    return {
        'n_ok': len(results),
        'n_total': n_total,
        'mean_tpot_ms': round(sum(r['tpot_ms'] for r in results) / len(results), 1),
        'p50_tpot_ms': round(tpots[len(tpots)//2], 1),
        'p99_tpot_ms': round(tpots[int(len(tpots)*0.99)], 1),
        'mean_ttft_ms': round(sum(r['ttft_ms'] for r in results) / len(results), 1),
        'throughput_tok_s': round(throughput, 1),
    }


def main():
    log.info('=' * 80)
    log.info('  Micro-batch Optimization: M=1 vs M=2 vs M=3 @ concurrency=%d', CONCURRENCY)
    log.info('  Model: Qwen3-32B | Input=%d Output=%d | GPU 4-7', INPUT_LEN, OUTPUT_TOKENS)
    log.info('=' * 80)

    all_results = {}

    for m in M_VALUES:
        log.info('\n>>> Starting PD+AF M=%d <<<', m)
        ret = start_pdaf(m)
        if ret is None:
            log.error('M=%d: Failed to start, skipping', m)
            all_results[m] = None
            kill_all()
            time.sleep(10)
            continue
        procs, url = ret

        log.info('M=%d: Running benchmark (conc=%d, n=%d, in=%d, out=%d)',
                 m, CONCURRENCY, N_REQUESTS, INPUT_LEN, OUTPUT_TOKENS)
        r = asyncio.run(measure_concurrent(url, INPUT_LEN, OUTPUT_TOKENS, CONCURRENCY, N_REQUESTS))
        if r:
            all_results[m] = r
            log.info('M=%d: TPOT=%.1fms (p50=%.1f p99=%.1f) TTFT=%.1fms Thru=%.1f tok/s (%d/%d ok)',
                     m, r['mean_tpot_ms'], r['p50_tpot_ms'], r['p99_tpot_ms'],
                     r['mean_ttft_ms'], r['throughput_tok_s'], r['n_ok'], r['n_total'])
        else:
            all_results[m] = None
            log.warning('M=%d: Benchmark FAILED', m)

        cleanup_procs(procs)
        kill_all()
        time.sleep(10)

    # Summary
    print()
    print('=' * 90)
    print(f'  Micro-batch Optimization Results (conc={CONCURRENCY}, in={INPUT_LEN}, out={OUTPUT_TOKENS})')
    print('=' * 90)
    print(f'{"M":<4} {"TPOT(mean)":<12} {"TPOT(p50)":<12} {"TPOT(p99)":<12} {"TTFT":<12} {"Throughput":<14} {"Success":<10}')
    print('-' * 80)
    for m in M_VALUES:
        r = all_results.get(m)
        if r:
            print(f'{m:<4} {r["mean_tpot_ms"]:<10.1f}ms {r["p50_tpot_ms"]:<10.1f}ms '
                  f'{r["p99_tpot_ms"]:<10.1f}ms {r["mean_ttft_ms"]:<10.1f}ms '
                  f'{r["throughput_tok_s"]:<12.1f}tok/s {r["n_ok"]}/{r["n_total"]}')
        else:
            print(f'{m:<4} {"FAILED":<12} {"FAILED":<12} {"FAILED":<12} {"FAILED":<12} {"FAILED":<14} {"N/A":<10}')
    print('=' * 90)

    # Comparison
    if all_results.get(1) and any(all_results.get(m) for m in [2, 3]):
        base = all_results[1]
        print('\n--- vs M=1 baseline ---')
        for m in [2, 3]:
            r = all_results.get(m)
            if r:
                tpot_delta = (r['mean_tpot_ms'] / base['mean_tpot_ms'] - 1) * 100
                thru_delta = (r['throughput_tok_s'] / base['throughput_tok_s'] - 1) * 100
                print(f'  M={m}: TPOT {tpot_delta:+.1f}%  Throughput {thru_delta:+.1f}%')

    out_path = os.path.join(HERE, 'results.json')
    with open(out_path, 'w') as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2, default=str)
    log.info(f'\nResults saved to {out_path}')


if __name__ == '__main__':
    main()
