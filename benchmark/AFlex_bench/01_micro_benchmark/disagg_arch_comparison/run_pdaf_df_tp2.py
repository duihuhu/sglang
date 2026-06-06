#!/usr/bin/env python3
"""PD+AF with DF TP=2 (heterogeneous: PA=1, PF=1, DA=1, DF=2).

GPU allocation (A800-SXM4-80GB):
  PA=GPU7 (Prefill Attn, TP=1)
  PF=GPU6 (Prefill FFN, TP=1)
  DA=GPU5 (Decode Attn, TP=1)
  DF=GPU3,4 (Decode FFN, TP=2)  <-- key change

Total: 5 GPUs.

This tests whether DF TP=2 resolves the FFN compute-bound bottleneck
at high concurrency while keeping DA at TP=1 (minimal overhead).
"""
import asyncio, json, logging, os, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from run_pdaf_vs_pd import (kill_all, wait_port, wait_health, warmup,
                            cleanup_procs, measure_concurrent, measure_single,
                            OUTPUT_TOKENS, INPUT_LENS)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pdaf_df_tp2")

PYTHON = "/workspace/env/sglang-tier/bin/python"
MODEL = "/models/Qwen/Qwen3-32B/"
HERE = os.path.dirname(os.path.abspath(__file__))
BOOTSTRAP = 18999
UCX_TLS = "rc,tcp,cuda_copy,cuda_ipc"

CONCURRENCIES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def start_pdaf_df_tp2():
    """PD+AF with DF TP=2: DA(TP=1) on GPU5, DF(TP=2) on GPU3,4."""
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

    extra_common = [
        '--skip-server-warmup', '--disable-cuda-graph', '--disable-piecewise-cuda-graph',
        '--afd-micro-batch', '1',
        '--afd-disagg-interleave-poll',
        '--disable-radix-cache',
    ]

    # ── Prefill side: PF(TP=1, GPU6) + PA(TP=1, GPU7) ──
    # Same as original: both TP=1, IPC between GPU6↔GPU7
    pf_vis = '6,7'
    env_pf = env_base.copy()
    env_pf['CUDA_VISIBLE_DEVICES'] = pf_vis
    env_pf['AFD_UCX_BASE_PORT'] = str(ucx_p)
    env_pf['AFD_SCHED_PORT'] = str(sched_p)
    env_pf['AFD_IPC_SYNC_MODE'] = 'ipc_event'
    env_pf['AFD_IPC_PEER_DEVICE'] = '1'
    env_pf['SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT'] = '600'
    env_pf['SGLANG_DISAGGREGATION_WAITING_TIMEOUT'] = '600'
    pf_cmd = [PYTHON, '-m', 'sglang.launch_server',
        '--model-path', MODEL, '--tp', '1',
        '--host', '127.0.0.1', '--port', str(pf_port),
        '--afd-perspective', 'ffn',
        '--afd-comm-backend', 'ipc_cpp',
        '--mem-fraction-static', '0.85',
        '--disaggregation-mode', 'prefill',
        '--disaggregation-transfer-backend', 'mooncake',
        '--disaggregation-bootstrap-port', str(BOOTSTRAP),
        '--disaggregation-ib-device', 'mlx5_4',
        '--base-gpu-id', '0',
    ] + extra_common
    fh = open(os.path.join(HERE, 'dftp2_pf.log'), 'w')
    p = subprocess.Popen(pf_cmd, env=env_pf, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(('pf', p, fh))
    time.sleep(2)

    env_pa = env_base.copy()
    env_pa['CUDA_VISIBLE_DEVICES'] = pf_vis
    env_pa['AFD_UCX_BASE_PORT'] = str(ucx_p)
    env_pa['AFD_SCHED_PORT'] = str(sched_p)
    env_pa['AFD_IPC_SYNC_MODE'] = 'ipc_event'
    env_pa['AFD_IPC_PEER_DEVICE'] = '0'
    env_pa['AFD_UCX_FFN_HOST'] = '127.0.0.1'
    env_pa['SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT'] = '600'
    env_pa['SGLANG_DISAGGREGATION_WAITING_TIMEOUT'] = '600'
    pa_cmd = [PYTHON, '-m', 'sglang.launch_server',
        '--model-path', MODEL, '--tp', '1',
        '--host', '127.0.0.1', '--port', str(pa_port),
        '--afd-perspective', 'attn',
        '--afd-comm-backend', 'ipc_cpp',
        '--mem-fraction-static', '0.85',
        '--disaggregation-mode', 'prefill',
        '--disaggregation-transfer-backend', 'mooncake',
        '--disaggregation-bootstrap-port', str(BOOTSTRAP),
        '--disaggregation-ib-device', 'mlx5_4',
        '--base-gpu-id', '1',
    ] + extra_common
    fh2 = open(os.path.join(HERE, 'dftp2_pa.log'), 'w')
    p2 = subprocess.Popen(pa_cmd, env=env_pa, stdout=fh2, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(('pa', p2, fh2))

    if not wait_port('127.0.0.1', pa_port, timeout=300) or not wait_port('127.0.0.1', pf_port, timeout=300):
        log.error('Prefill servers failed to start')
        cleanup_procs(procs)
        return None

    # ── Decode side: DF(TP=2, GPU3,4) + DA(TP=1, GPU5) ──
    # DF uses TP=2 on GPU3,4. DA uses TP=1 on GPU5.
    # Communication: DA(GPU5) ↔ DF(GPU3,4 rank0)
    # We use AFD_IPC_PEER_OFFSET so DA rank0 talks to DF rank0.
    # CUDA_VISIBLE_DEVICES for DF: "3,4" (TP=2)
    # CUDA_VISIBLE_DEVICES for DA: "3,4,5" (sees all 3, but --tp 1 --base-gpu-id 2 uses GPU5)
    # Actually simpler: DA sees "5,3,4" with base_gpu_id=0, peer_offset=+1 → talks to device 1 (=GPU3)
    # But IPC needs both sides to see each other's GPU.
    #
    # Cleanest approach: both DA and DF see "3,4,5".
    # DF: tp=2, base_gpu_id=0 → uses cuda:0,1 (=GPU3,4)
    # DA: tp=1, base_gpu_id=2 → uses cuda:2 (=GPU5)
    # IPC: DA peer_device=0 (DF rank0 = cuda:0 = GPU3)
    #       DF peer_device=2 (DA = cuda:2 = GPU5)

    d_vis = '3,4,5'

    env_df = env_base.copy()
    env_df['CUDA_VISIBLE_DEVICES'] = d_vis
    env_df['AFD_UCX_BASE_PORT'] = str(ucx_d)
    env_df['AFD_SCHED_PORT'] = str(sched_d)
    env_df['AFD_IPC_SYNC_MODE'] = 'ipc_event'
    env_df['AFD_IPC_PEER_DEVICE'] = '2'  # DF rank0 (cuda:0) talks to DA (cuda:2)
    env_df['SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT'] = '600'
    env_df['SGLANG_DISAGGREGATION_WAITING_TIMEOUT'] = '600'
    df_cmd = [PYTHON, '-m', 'sglang.launch_server',
        '--model-path', MODEL, '--tp', '2',
        '--host', '127.0.0.1', '--port', str(df_port),
        '--afd-perspective', 'ffn',
        '--afd-attn-tp', '1',
        '--afd-comm-backend', 'ipc_cpp',
        '--mem-fraction-static', '0.88',
        '--disaggregation-mode', 'decode',
        '--disaggregation-transfer-backend', 'mooncake',
        '--disaggregation-bootstrap-port', str(BOOTSTRAP),
        '--disaggregation-ib-device', 'mlx5_4',
        '--base-gpu-id', '0',
    ] + extra_common
    fh3 = open(os.path.join(HERE, 'dftp2_df.log'), 'w')
    p3 = subprocess.Popen(df_cmd, env=env_df, stdout=fh3, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(('df', p3, fh3))
    time.sleep(5)

    env_da = env_base.copy()
    env_da['CUDA_VISIBLE_DEVICES'] = d_vis
    env_da['AFD_UCX_BASE_PORT'] = str(ucx_d)
    env_da['AFD_SCHED_PORT'] = str(sched_d)
    env_da['AFD_IPC_SYNC_MODE'] = 'ipc_event'
    env_da['AFD_IPC_PEER_DEVICE'] = '0'  # DA (cuda:2) talks to DF rank0 (cuda:0)
    env_da['AFD_UCX_FFN_HOST'] = '127.0.0.1'
    env_da['SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT'] = '600'
    env_da['SGLANG_DISAGGREGATION_WAITING_TIMEOUT'] = '600'
    da_cmd = [PYTHON, '-m', 'sglang.launch_server',
        '--model-path', MODEL, '--tp', '1',
        '--host', '127.0.0.1', '--port', str(da_port),
        '--afd-perspective', 'attn',
        '--afd-ffn-tp', '2',
        '--afd-comm-backend', 'ipc_cpp',
        '--mem-fraction-static', '0.88',
        '--disaggregation-mode', 'decode',
        '--disaggregation-transfer-backend', 'mooncake',
        '--disaggregation-bootstrap-port', str(BOOTSTRAP),
        '--disaggregation-ib-device', 'mlx5_4',
        '--base-gpu-id', '2',
    ] + extra_common
    fh4 = open(os.path.join(HERE, 'dftp2_da.log'), 'w')
    p4 = subprocess.Popen(da_cmd, env=env_da, stdout=fh4, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(('da', p4, fh4))

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
    rf = open(os.path.join(HERE, 'dftp2_router.log'), 'w')
    rp = subprocess.Popen(router_cmd, env=os.environ.copy(), stdout=rf, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(('router', rp, rf))
    if not wait_health(f'http://127.0.0.1:{router_port}/health', timeout=180):
        log.error('Router failed')
        cleanup_procs(procs)
        return None

    url = f'http://127.0.0.1:{router_port}'
    warmup(url)
    return procs, url


def main():
    log.info('=' * 80)
    log.info('  PD+AF DF_TP=2 (PA=1,PF=1,DA=1,DF=2) | 5 GPU | Qwen3-32B')
    log.info('=' * 80)

    ret = start_pdaf_df_tp2()
    if ret is None:
        log.error('Failed to start PD+AF DF_TP=2, aborting')
        sys.exit(1)
    procs, url = ret

    results = {}

    # Single-request sweep
    log.info('--- Single-request sweep (out=%d) ---', OUTPUT_TOKENS)
    single_data = {}
    for il in INPUT_LENS:
        r = asyncio.run(measure_single(url, il, OUTPUT_TOKENS, n=10))
        if r:
            single_data[il] = r
            log.info(f'  in={il}: TTFT={r["mean_ttft_ms"]}ms TPOT={r["mean_tpot_ms"]}ms '
                     f'Thru={r["mean_throughput_tok_s"]}tok/s (n={r["n_ok"]})')
        else:
            log.warning(f'  in={il}: FAILED')
    results['single'] = single_data

    # Concurrency sweep
    log.info('--- Concurrency sweep (in=16, out=%d) ---', OUTPUT_TOKENS)
    conc_data = {}
    for conc in CONCURRENCIES:
        n_req = max(conc, 40)
        r = asyncio.run(measure_concurrent(url, 16, OUTPUT_TOKENS, conc, n_req))
        if r:
            conc_data[conc] = r
            log.info(f'  conc={conc}: TPOT={r["mean_tpot_ms"]}ms TTFT={r["mean_ttft_ms"]}ms '
                     f'Thru={r["throughput_tok_s"]}tok/s running_max={r["max_running"]} '
                     f'running_avg={r["avg_running"]} (n={r["n_ok"]}/{n_req})')
        else:
            log.warning(f'  conc={conc}: FAILED')
    results['conc'] = conc_data

    cleanup_procs(procs)
    kill_all()

    # Print summary
    print()
    print('=' * 100)
    print('  PD+AF DF_TP=2 (PA=1, PF=1, DA=1, DF=2) — 5 GPU')
    print('=' * 100)

    print('\n--- Single Request (out=%d) ---' % OUTPUT_TOKENS)
    print(f'{"Input":<8} {"TTFT":<12} {"TPOT":<12} {"Throughput":<14}')
    print('-' * 50)
    for il in INPUT_LENS:
        r = single_data.get(il)
        if r:
            print(f'{il:<8} {r["mean_ttft_ms"]:<10.1f}ms {r["mean_tpot_ms"]:<10.1f}ms '
                  f'{r["mean_throughput_tok_s"]:<12.1f}tok/s')
        else:
            print(f'{il:<8} {"FAILED":<12} {"FAILED":<12} {"FAILED":<14}')

    print('\n--- Concurrency Sweep (in=16, out=%d) ---' % OUTPUT_TOKENS)
    print(f'{"Conc":<6} {"TPOT":<12} {"TTFT":<12} {"Throughput":<14} {"MaxRun":<8} {"AvgRun":<8}')
    print('-' * 70)
    for conc in CONCURRENCIES:
        r = conc_data.get(conc)
        if r:
            print(f'{conc:<6} {r["mean_tpot_ms"]:<10.1f}ms {r["mean_ttft_ms"]:<10.1f}ms '
                  f'{r["throughput_tok_s"]:<12.1f}tok/s {r["max_running"]:<8} {r["avg_running"]:<8}')
        else:
            print(f'{conc:<6} {"FAILED":<12} {"FAILED":<12} {"FAILED":<14} {"N/A":<8} {"N/A":<8}')
    print('=' * 100)

    out_path = os.path.join(HERE, 'pdaf_df_tp2_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f'Results saved to {out_path}')


if __name__ == '__main__':
    main()
