#!/usr/bin/env python3
"""Async pipeline benchmark: M=2 async-pipeline vs M=1 baseline at conc=128.

Tests the cross-layer async pipeline optimization that overlaps DA's Attention
compute with DF's FFN compute across layers using multi-stream + CUDA events.

Comparison:
  - M=1 (baseline): serial Attn→send→recv→FFN per layer
  - M=2 + async pipeline: overlapped compute/communication via per-mb streams
"""
import asyncio, json, logging, os, socket, subprocess, sys, time, urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("async_pipe")

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


def start_pdaf(micro_batch: int, async_pipeline: bool = False):
    """Start PD+AF with given config."""
    kill_all()
    time.sleep(3)
    procs = []
    env_base = os.environ.copy()
    env_base['SGLANG_DISABLE_REQUEST_LOGGING'] = 'true'
    env_base['AFD_UCX_TLS'] = UCX_TLS
    env_base['UCX_LOG_LEVEL'] = 'fatal'
    if async_pipeline:
        env_base['AFD_ASYNC_PIPELINE'] = '1'

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
    if async_pipeline:
        extra.append('--afd-async-pipeline')

    def _start(name, perspective, disagg_mode, port, ucx_base, sched_port,
               visible_gpus, base_gpu_id, peer_device, ffn_host=None):
        env = env_base.copy()
        env['CUDA_VISIBLE_DEVICES'] = visible_gpus
        env['AFD_UCX_BASE_PORT'] = str(ucx_base)
        env['AFD_SCHED_PORT'] = str(sched_port)
        env['AFD_IPC_SYNC_MODE'] = 'gpu_signal'
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
        tag = "async" if async_pipeline else "sync"
        logfile = os.path.join(HERE, f'{tag}_m{micro_batch}_{name}.log')
        fh = open(logfile, 'w')
        p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        procs.append((name, p, fh))

    # Prefill: PF(GPU6) + PA(GPU7)
    p_vis = '6,7'
    _start('pf', 'ffn', 'prefill', pf_port, ucx_p, sched_p, p_vis, 0, 1)
    time.sleep(2)
    _start('pa', 'attn', 'prefill', pa_port, ucx_p, sched_p, p_vis, 1, 0, ffn_host='127.0.0.1')

    if not wait_port('127.0.0.1', pa_port, timeout=300) or not wait_port('127.0.0.1', pf_port, timeout=300):
        log.error('Prefill servers failed to start')
        cleanup_procs(procs)
        return None

    # Decode: DF(GPU4) + DA(GPU5)
    d_vis = '4,5'
    _start('df', 'ffn', 'decode', df_port, ucx_d, sched_d, d_vis, 0, 1)
    time.sleep(2)
    _start('da', 'attn', 'decode', da_port, ucx_d, sched_d, d_vis, 1, 0, ffn_host='127.0.0.1')

    if not wait_port('127.0.0.1', da_port, timeout=300) or not wait_port('127.0.0.1', df_port, timeout=300):
        log.error('Decode servers failed to start')
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
    rf = open(os.path.join(HERE, f'{"async" if async_pipeline else "sync"}_m{micro_batch}_router.log'), 'w')
    rp = subprocess.Popen(router_cmd, env=os.environ.copy(), stdout=rf, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(('router', rp, rf))
    if not wait_health(f'http://127.0.0.1:{router_port}/health', timeout=180):
        log.error('Router failed')
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
    log.info('  Async Pipeline Benchmark: M=2 async vs M=1 baseline @ conc=%d', CONCURRENCY)
    log.info('  Model: Qwen3-32B | Input=%d Output=%d | GPU 4-7', INPUT_LEN, OUTPUT_TOKENS)
    log.info('=' * 80)

    configs = [
        ("M=1 baseline", 1, False),
        ("M=2 async-pipeline", 2, True),
    ]

    # Skip M=1 if we already have cached results
    cached_m1 = None
    cached_path = os.path.join(HERE, 'results.json')
    if os.path.exists(cached_path):
        try:
            with open(cached_path) as f:
                cached = json.load(f)
            if "1" in cached and cached["1"]:
                cached_m1 = cached["1"]
                log.info("Using cached M=1 result: TPOT=%.1fms Thru=%.1f",
                         cached_m1["mean_tpot_ms"], cached_m1["throughput_tok_s"])
                configs = [("M=2 async-pipeline", 2, True)]
        except Exception:
            pass

    all_results = {}

    for label, m, async_pipe in configs:
        log.info('\n>>> Starting %s <<<', label)
        ret = start_pdaf(m, async_pipeline=async_pipe)
        if ret is None:
            log.error('%s: Failed to start, skipping', label)
            all_results[label] = None
            kill_all()
            time.sleep(10)
            continue
        procs, url = ret

        log.info('%s: Running benchmark (conc=%d, n=%d, in=%d, out=%d)',
                 label, CONCURRENCY, N_REQUESTS, INPUT_LEN, OUTPUT_TOKENS)
        r = asyncio.run(measure_concurrent(url, INPUT_LEN, OUTPUT_TOKENS, CONCURRENCY, N_REQUESTS))
        if r:
            all_results[label] = r
            log.info('%s: TPOT=%.1fms (p50=%.1f p99=%.1f) TTFT=%.1fms Thru=%.1f tok/s (%d/%d ok)',
                     label, r['mean_tpot_ms'], r['p50_tpot_ms'], r['p99_tpot_ms'],
                     r['mean_ttft_ms'], r['throughput_tok_s'], r['n_ok'], r['n_total'])
        else:
            all_results[label] = None
            log.warning('%s: Benchmark FAILED', label)

        cleanup_procs(procs)
        kill_all()
        time.sleep(10)

    # Summary
    print()
    print('=' * 90)
    print(f'  Async Pipeline Results (conc={CONCURRENCY}, in={INPUT_LEN}, out={OUTPUT_TOKENS})')
    print('=' * 90)
    print(f'{"Config":<25} {"TPOT(mean)":<12} {"TPOT(p50)":<12} {"TPOT(p99)":<12} {"TTFT":<12} {"Throughput":<14} {"OK":<8}')
    print('-' * 95)
    for label, _, _ in configs:
        r = all_results.get(label)
        if r:
            print(f'{label:<25} {r["mean_tpot_ms"]:<10.1f}ms {r["p50_tpot_ms"]:<10.1f}ms '
                  f'{r["p99_tpot_ms"]:<10.1f}ms {r["mean_ttft_ms"]:<10.1f}ms '
                  f'{r["throughput_tok_s"]:<12.1f}tok/s {r["n_ok"]}/{r["n_total"]}')
        else:
            print(f'{label:<25} {"FAILED":<12} {"FAILED":<12} {"FAILED":<12} {"FAILED":<12} {"FAILED":<14} {"N/A":<8}')
    print('=' * 90)

    # Use cached M=1 if available
    if cached_m1 and "M=1 baseline" not in all_results:
        all_results["M=1 baseline"] = cached_m1

    # Delta
    base = all_results.get("M=1 baseline")
    exp = all_results.get("M=2 async-pipeline")
    if base and exp:
        tpot_delta = (exp['mean_tpot_ms'] / base['mean_tpot_ms'] - 1) * 100
        thru_delta = (exp['throughput_tok_s'] / base['throughput_tok_s'] - 1) * 100
        print(f'\n  M=2 async vs M=1: TPOT {tpot_delta:+.1f}%  Throughput {thru_delta:+.1f}%')

    out_path = os.path.join(HERE, 'async_pipeline_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f'\nResults saved to {out_path}')


if __name__ == '__main__':
    main()
