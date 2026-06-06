#!/usr/bin/env python3
"""PD+AF M=1 (C++ IPC backend) vs Pure PD TP=1 comparison.

GPU allocation (A800-SXM4-80GB, NV8):
  PD TP=1:   P=GPU4, D=GPU5
  PD+AF M=1: PA=GPU7, PF=GPU6, DA=GPU5, DF=GPU4

Usage:
  python run_pdaf_vs_pd.py                    # Full comparison (single + conc sweep)
  python run_pdaf_vs_pd.py --phase 1          # Single-request sweep only
  python run_pdaf_vs_pd.py --phase 2          # Concurrency sweep only
  python run_pdaf_vs_pd.py --skip-pd          # Skip PD baseline (use cached results)
"""
import asyncio, aiohttp, json, logging, os, socket, subprocess, sys, time, urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pdaf_vs_pd")

PYTHON = sys.executable
MODEL = "/models/Qwen/Qwen3-32B/"
HERE = os.path.dirname(os.path.abspath(__file__))
BOOTSTRAP = 18999
UCX_TLS = "rc,tcp,cuda_copy,cuda_ipc"
ALL_PORTS = [30000, 30001, 50000, 50010, 50011, 50020, 50021]

OUTPUT_TOKENS = 1024
INPUT_LENS = [128, 256, 512, 1024, 2048]
CONCURRENCIES = [256]
N_REQUESTS_SINGLE = 10
N_REQUESTS_CONC = 0  # will be set per-concurrency level below


# ── process utils ──────────────────────────────────────────────────────────

def _ports_free():
    out = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True).stdout
    return [p for p in ALL_PORTS if f":{p}" in out]


def kill_all():
    for t in ["sglang.launch_server", "sglang_router", "sglang::router", "sglang::scheduler"]:
        subprocess.run(["pkill", "-f", t], capture_output=True)
    time.sleep(8)
    for t in ["sglang.launch_server", "sglang_router", "sglang::router", "sglang::scheduler"]:
        subprocess.run(["pkill", "-9", "-f", t], capture_output=True)
    time.sleep(3)
    subprocess.run(["pkill", "-9", "-f", "ucx"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "ucp"], capture_output=True)
    subprocess.run(["sync"], capture_output=True)
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


# ── server launchers ────────────────────────────────────────────────────────

def start_pd_tp1(dev_p, dev_d, name=""):
    """PD TP=1: Prefill + Decode on separate GPUs."""
    kill_all()
    time.sleep(3)
    procs = []
    env_base = os.environ.copy()
    env_base['SGLANG_DISABLE_REQUEST_LOGGING'] = 'true'
    prefix = f'{name}_' if name else ''

    env_p = env_base.copy()
    env_p['CUDA_VISIBLE_DEVICES'] = str(dev_p)
    p_cmd = [PYTHON, '-m', 'sglang.launch_server',
        '--model-path', MODEL, '--tp', '1',
        '--host', '127.0.0.1', '--port', '50010',
        '--mem-fraction-static', '0.85',
        '--disaggregation-mode', 'prefill',
        '--disaggregation-transfer-backend', 'mooncake',
        '--disaggregation-bootstrap-port', str(BOOTSTRAP),
        '--disaggregation-ib-device', 'mlx5_4',
        '--skip-server-warmup',
        '--disable-cuda-graph', '--disable-piecewise-cuda-graph',
        '--disable-radix-cache',
    ]
    fh = open(os.path.join(HERE, f'{prefix}pd_tp1_p.log'), 'w')
    p = subprocess.Popen(p_cmd, env=env_p, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(('prefill', p, fh))

    env_d = env_base.copy()
    env_d['CUDA_VISIBLE_DEVICES'] = str(dev_d)
    d_cmd = [PYTHON, '-m', 'sglang.launch_server',
        '--model-path', MODEL, '--tp', '1',
        '--host', '127.0.0.1', '--port', '50020',
        '--mem-fraction-static', '0.85',
        '--disaggregation-mode', 'decode',
        '--disaggregation-transfer-backend', 'mooncake',
        '--disaggregation-bootstrap-port', str(BOOTSTRAP),
        '--disaggregation-ib-device', 'mlx5_4',
        '--skip-server-warmup',
        '--disable-cuda-graph', '--disable-piecewise-cuda-graph',
        '--disable-radix-cache',
    ]
    fh2 = open(os.path.join(HERE, f'{prefix}pd_tp1_d.log'), 'w')
    p2 = subprocess.Popen(d_cmd, env=env_d, stdout=fh2, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(('decode', p2, fh2))

    if not wait_port('127.0.0.1', 50010) or not wait_port('127.0.0.1', 50020):
        log.error('PD TP=1: servers failed to start')
        cleanup_procs(procs)
        return None

    router_cmd = [PYTHON, '-m', 'sglang_router.launch_router',
        '--pd-disaggregation', '--mini-lb',
        '--prefill', 'http://127.0.0.1:50010',
        '--decode', 'http://127.0.0.1:50020',
        '--host', '127.0.0.1', '--port', '50000',
    ]
    rf = open(os.path.join(HERE, f'{prefix}pd_tp1_router.log'), 'w')
    rp = subprocess.Popen(router_cmd, env=os.environ.copy(), stdout=rf, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(('router', rp, rf))
    if not wait_health('http://127.0.0.1:50000/health', timeout=180):
        log.error('PD TP=1: router failed')
        cleanup_procs(procs)
        return None

    url = 'http://127.0.0.1:50000'
    warmup(url)
    return procs, url


def start_pdaf_m1_cpp(pa_dev, pf_dev, da_dev, df_dev, name=""):
    """PD+AF M=1 with C++ IPC backend."""
    kill_all()
    time.sleep(3)
    procs = []
    env_base = os.environ.copy()
    env_base['SGLANG_DISABLE_REQUEST_LOGGING'] = 'true'
    env_base['AFD_UCX_TLS'] = UCX_TLS
    env_base['UCX_LOG_LEVEL'] = 'fatal'
    prefix = f'{name}_' if name else ''

    da_port, df_port = 50020, 50021
    pa_port, pf_port = 50010, 50011
    router_port = 50000
    ucx_p, ucx_d = 25200, 25300
    sched_p, sched_d = 65400, 65500

    extra = [
        '--skip-server-warmup', '--disable-cuda-graph', '--disable-piecewise-cuda-graph',
        '--afd-micro-batch', '1',
        '--afd-disagg-interleave-poll',
        '--disable-radix-cache',
    ]

    def _start(name, gpu, perspective, disagg_mode, port, ucx_base, sched_port, ffn_host=None, visible_gpus=None, base_gpu_id=0, peer_device=None):
        env = env_base.copy()
        env['CUDA_VISIBLE_DEVICES'] = visible_gpus if visible_gpus else str(gpu)
        env['AFD_UCX_BASE_PORT'] = str(ucx_base)
        env['AFD_SCHED_PORT'] = str(sched_port)
        env['SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT'] = '600'
        env['SGLANG_DISAGGREGATION_WAITING_TIMEOUT'] = '600'
        env['AFD_IPC_SYNC_MODE'] = 'ipc_event'
        if peer_device is not None:
            env['AFD_IPC_PEER_DEVICE'] = str(peer_device)
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
        fh = open(os.path.join(HERE, f'{prefix}pdaf_{name}.log'), 'w')
        p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        procs.append((name, p, fh))

    # Start order matters: FFN first, then Attn
    s_vis = f'{pf_dev},{pa_dev}'
    d_vis = f'{df_dev},{da_dev}'

    _start('pf', pf_dev, 'ffn', 'prefill', pf_port, ucx_p, sched_p, visible_gpus=s_vis, base_gpu_id=0, peer_device=1)
    time.sleep(2)
    _start('pa', pa_dev, 'attn', 'prefill', pa_port, ucx_p, sched_p, ffn_host='127.0.0.1', visible_gpus=s_vis, base_gpu_id=1, peer_device=0)

    if not wait_port('127.0.0.1', pa_port, timeout=300) or not wait_port('127.0.0.1', pf_port, timeout=300):
        log.error('PD+AF C++ IPC: prefill servers failed')
        cleanup_procs(procs)
        return None

    _start('df', df_dev, 'ffn', 'decode', df_port, ucx_d, sched_d, visible_gpus=d_vis, base_gpu_id=0, peer_device=1)
    time.sleep(2)
    _start('da', da_dev, 'attn', 'decode', da_port, ucx_d, sched_d, ffn_host='127.0.0.1', visible_gpus=d_vis, base_gpu_id=1, peer_device=0)

    if not wait_port('127.0.0.1', da_port) or not wait_port('127.0.0.1', df_port):
        log.error('PD+AF C++ IPC: decode servers failed')
        cleanup_procs(procs)
        return None

    time.sleep(5)

    router_cmd = [PYTHON, '-m', 'sglang_router.launch_router',
        '--pd-disaggregation', '--mini-lb',
        '--prefill', f'http://127.0.0.1:{pa_port}',
        '--decode', f'http://127.0.0.1:{da_port}',
        '--host', '127.0.0.1', '--port', str(router_port),
    ]
    rf = open(os.path.join(HERE, f'{prefix}pdaf_router.log'), 'w')
    rp = subprocess.Popen(router_cmd, env=os.environ.copy(), stdout=rf, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(('router', rp, rf))
    if not wait_health(f'http://127.0.0.1:{router_port}/health', timeout=120):
        log.error('PD+AF C++ IPC: router failed')
        cleanup_procs(procs)
        return None

    url = f'http://127.0.0.1:{router_port}'
    warmup(url)
    return procs, url


# ── measurement ─────────────────────────────────────────────────────────────

async def measure_single(url, input_len, output_len, n=10):
    """Measure single-request TTFT and TPOT."""
    prompt = 'Hello ' * (input_len // 2)
    payload = {'text': prompt, 'sampling_params': {'max_new_tokens': output_len, 'temperature': 0.0}, 'stream': True}
    results = []
    timeout = aiohttp.ClientTimeout(total=180)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for _ in range(n):
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
            except Exception as e:
                log.warning(f'  Request failed: {e}')
                continue
            if first_token_time and token_count > 1:
                total = time.perf_counter() - t0
                tpot = (total - first_token_time) / (token_count - 1)
                throughput = token_count / total
                results.append({'ttft_ms': first_token_time * 1000, 'tpot_ms': tpot * 1000, 'tokens': token_count, 'throughput_tok_s': throughput})
    if not results:
        return None
    return {
        'n_ok': len(results),
        'mean_ttft_ms': round(sum(r['ttft_ms'] for r in results) / len(results), 1),
        'mean_tpot_ms': round(sum(r['tpot_ms'] for r in results) / len(results), 1),
        'p50_tpot_ms': round(sorted(r['tpot_ms'] for r in results)[len(results)//2], 1),
        'mean_throughput_tok_s': round(sum(r['throughput_tok_s'] for r in results) / len(results), 1),
    }


async def measure_concurrent(url, input_len, output_len, concurrency, n_total):
    """Measure TPOT at given concurrency level."""
    prompt = 'Hello ' * (input_len // 2)
    payload = {'text': prompt, 'sampling_params': {'max_new_tokens': output_len, 'temperature': 0.0}, 'stream': True}
    sem = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=300)
    running_samples = []

    async def poll_running(session, stop_event):
        """Poll running requests count during benchmark."""
        while not stop_event.is_set():
            try:
                async with session.get(f'{url}/v1/models', timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    pass
            except Exception:
                pass
            running_samples.append(concurrency - sem._value)
            await asyncio.sleep(0.5)

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
    stop_event = asyncio.Event()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        poll_task = asyncio.create_task(poll_running(session, stop_event))
        tasks = [single(session) for _ in range(n_total)]
        raw = await asyncio.gather(*tasks)
        stop_event.set()
        await poll_task
    wall_s = time.perf_counter() - wall_start

    results = [r for r in raw if r is not None]
    if not results:
        return None
    tpots = sorted(r['tpot_ms'] for r in results)
    total_output_tokens = sum(r['tokens'] for r in results)
    throughput = total_output_tokens / wall_s
    max_running = max(running_samples) if running_samples else 0
    avg_running = round(sum(running_samples) / len(running_samples), 1) if running_samples else 0
    return {
        'n_ok': len(results),
        'mean_tpot_ms': round(sum(r['tpot_ms'] for r in results) / len(results), 1),
        'p50_tpot_ms': round(tpots[len(tpots)//2], 1),
        'mean_ttft_ms': round(sum(r['ttft_ms'] for r in results) / len(results), 1),
        'throughput_tok_s': round(throughput, 1),
        'max_running': max_running,
        'avg_running': avg_running,
    }


def run_single_sweep(url, label=""):
    """Single-request sweep across input lengths."""
    tag = f'{label}: ' if label else ''
    log.info(f'--- {tag}Single-request sweep (out={OUTPUT_TOKENS}) ---')
    data = {}
    for il in INPUT_LENS:
        r = asyncio.run(measure_single(url, il, OUTPUT_TOKENS, n=N_REQUESTS_SINGLE))
        if r:
            data[il] = r
            log.info(f'  in={il}: TTFT={r["mean_ttft_ms"]}ms TPOT={r["mean_tpot_ms"]}ms Thru={r["mean_throughput_tok_s"]}tok/s (n={r["n_ok"]})')
        else:
            log.warning(f'  in={il}: FAILED')
    return data


def run_conc_sweep(url, label=""):
    """Concurrency sweep with fixed input=512."""
    tag = f'{label}: ' if label else ''
    log.info(f'--- {tag}Concurrency sweep (in=512, out={OUTPUT_TOKENS}) ---')
    data = {}
    for conc in CONCURRENCIES:
        n_req = max(conc, 40)
        r = asyncio.run(measure_concurrent(url, 16, OUTPUT_TOKENS, conc, n_req))
        if r:
            data[conc] = r
            log.info(f'  conc={conc}: TPOT={r["mean_tpot_ms"]}ms TTFT={r["mean_ttft_ms"]}ms Thru={r["throughput_tok_s"]}tok/s running_max={r["max_running"]} running_avg={r["avg_running"]} (n={r["n_ok"]}/{n_req})')
        else:
            log.warning(f'  conc={conc}: FAILED')
    return data


# ── main ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description='PD+AF M=1 (C++ IPC) vs PD TP=1 comparison')
    parser.add_argument('--phase', type=int, default=0, help='0=both, 1=single-req, 2=conc-sweep')
    parser.add_argument('--skip-pd', action='store_true', help='Skip PD TP=1 baseline')
    parser.add_argument('--gpu-pd-p', type=int, default=4, help='PD prefill GPU')
    parser.add_argument('--gpu-pd-d', type=int, default=5, help='PD decode GPU')
    parser.add_argument('--gpu-af-pa', type=int, default=7, help='AF Attn-P GPU')
    parser.add_argument('--gpu-af-pf', type=int, default=6, help='AF FFN-P GPU')
    parser.add_argument('--gpu-af-da', type=int, default=5, help='AF Attn-D GPU')
    parser.add_argument('--gpu-af-df', type=int, default=4, help='AF FFN-D GPU')
    args = parser.parse_args()

    log.info('=' * 80)
    log.info('  PD+AF M=1 (C++ IPC) vs PD TP=1  |  GPU 4-7, Qwen3-32B')
    log.info('=' * 80)

    results = {'pd_tp1': {}, 'pdaf_cpp_ipc': {}}

    # ── PD TP=1 Baseline ──
    if not args.skip_pd:
        log.info('\n>>> Starting PD TP=1 (P=GPU%d, D=GPU%d) <<<', args.gpu_pd_p, args.gpu_pd_d)
        ret = start_pd_tp1(args.gpu_pd_p, args.gpu_pd_d)
        if ret is None:
            log.error('PD TP=1 failed to start, aborting')
            sys.exit(1)
        procs, url = ret

        if args.phase in (0, 1):
            results['pd_tp1']['single'] = run_single_sweep(url, 'PD_TP1')
        if args.phase in (0, 2):
            results['pd_tp1']['conc'] = run_conc_sweep(url, 'PD_TP1')

        cleanup_procs(procs)
        kill_all()
        time.sleep(10)

    # ── PD+AF M=1 C++ IPC ──
    log.info('\n>>> Starting PD+AF M=1 C++ IPC (PA=GPU%d, PF=GPU%d, DA=GPU%d, DF=GPU%d) <<<',
             args.gpu_af_pa, args.gpu_af_pf, args.gpu_af_da, args.gpu_af_df)
    ret = start_pdaf_m1_cpp(args.gpu_af_pa, args.gpu_af_pf, args.gpu_af_da, args.gpu_af_df)
    if ret is None:
        log.error('PD+AF C++ IPC failed to start, aborting')
        sys.exit(1)
    procs, url = ret

    if args.phase in (0, 1):
        results['pdaf_cpp_ipc']['single'] = run_single_sweep(url, 'PDAF_CPP_IPC')
    if args.phase in (0, 2):
        results['pdaf_cpp_ipc']['conc'] = run_conc_sweep(url, 'PDAF_CPP_IPC')

    cleanup_procs(procs)
    kill_all()

    # ── Summary ──
    print('\n' + '=' * 100)
    print('  RESULTS: PD+AF M=1 (C++ IPC) vs PD TP=1')
    print('=' * 100)

    if args.phase in (0, 1):
        print('\n--- Phase 1: Single Request (concurrency=1, out=%d) ---' % OUTPUT_TOKENS)
        print(f'{"Input":<8} {"PD TTFT":<10} {"PD TPOT":<10} {"PD Thru":<12} {"AF TTFT":<10} {"AF TPOT":<10} {"AF Thru":<12} {"TPOT Δ":<14}')
        print('-' * 90)
        pd_single = results.get('pd_tp1', {}).get('single', {})
        af_single = results.get('pdaf_cpp_ipc', {}).get('single', {})
        for il in INPUT_LENS:
            pd = pd_single.get(il)
            af = af_single.get(il)
            pd_ttft = f'{pd["mean_ttft_ms"]:.1f}ms' if pd else 'N/A'
            pd_tpot = f'{pd["mean_tpot_ms"]:.1f}ms' if pd else 'N/A'
            pd_thru = f'{pd["mean_throughput_tok_s"]:.1f}tok/s' if pd and 'mean_throughput_tok_s' in pd else 'N/A'
            af_ttft = f'{af["mean_ttft_ms"]:.1f}ms' if af else 'N/A'
            af_tpot = f'{af["mean_tpot_ms"]:.1f}ms' if af else 'N/A'
            af_thru = f'{af["mean_throughput_tok_s"]:.1f}tok/s' if af and 'mean_throughput_tok_s' in af else 'N/A'
            if pd and af:
                delta = af['mean_tpot_ms'] - pd['mean_tpot_ms']
                pct = (delta / pd['mean_tpot_ms']) * 100
                delta_str = f'{delta:+.1f}ms ({pct:+.1f}%)'
            else:
                delta_str = 'N/A'
            print(f'{il:<8} {pd_ttft:<10} {pd_tpot:<10} {pd_thru:<12} {af_ttft:<10} {af_tpot:<10} {af_thru:<12} {delta_str:<14}')

    if args.phase in (0, 2):
        print('\n--- Phase 2: Concurrency Sweep (in=512, out=%d) ---' % OUTPUT_TOKENS)
        print(f'{"Conc":<6} {"PD TPOT":<10} {"AF TPOT":<10} {"Overhead":<10} {"PD Thru":<12} {"AF Thru":<12} {"PD TTFT":<10} {"AF TTFT":<10}')
        print('-' * 85)
        pd_conc = results.get('pd_tp1', {}).get('conc', {})
        af_conc = results.get('pdaf_cpp_ipc', {}).get('conc', {})
        for conc in CONCURRENCIES:
            pd = pd_conc.get(conc)
            af = af_conc.get(conc)
            pd_tpot = f'{pd["mean_tpot_ms"]:.1f}ms' if pd else 'N/A'
            af_tpot = f'{af["mean_tpot_ms"]:.1f}ms' if af else 'N/A'
            pd_thru = f'{pd["throughput_tok_s"]:.1f}' if pd and 'throughput_tok_s' in pd else 'N/A'
            af_thru = f'{af["throughput_tok_s"]:.1f}' if af and 'throughput_tok_s' in af else 'N/A'
            pd_ttft = f'{pd["mean_ttft_ms"]:.1f}ms' if pd else 'N/A'
            af_ttft = f'{af["mean_ttft_ms"]:.1f}ms' if af else 'N/A'
            if pd and af:
                overhead = (af['mean_tpot_ms'] / pd['mean_tpot_ms'] - 1) * 100
                overhead_str = f'+{overhead:.1f}%'
            else:
                overhead_str = 'N/A'
            print(f'{conc:<6} {pd_tpot:<10} {af_tpot:<10} {overhead_str:<10} {pd_thru:<12} {af_thru:<12} {pd_ttft:<10} {af_ttft:<10}')

    # Save results
    out_path = os.path.join(HERE, 'results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f'\nResults saved to {out_path}')


if __name__ == '__main__':
    main()
