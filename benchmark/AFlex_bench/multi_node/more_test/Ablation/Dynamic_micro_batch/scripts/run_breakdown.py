#!/usr/bin/env python3
"""Launch the node1 4-GPU full PD+A/F micro-batch breakdown experiment.

The script deliberately does not run on import. It validates the container paths,
launches PF/PA/DF/DA on physical GPUs 4-7 with ipc_cpp plus a PD router,
drives low-load M=1 and high-load scheduled M=3, and invokes parse_breakdown.py after each case.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "breakdown"
BOOTSTRAP_PORT = 18999
ROUTER_PORT = 50000
ERROR_RE = re.compile(r"IndexKernel|index out of bounds|CUDA error|Traceback \(most recent call last\)|RuntimeError:")

ROLES = {
    # name: perspective, PD mode, port, physical GPU, local peer, UCX base, sched port
    "pf": ("ffn", "prefill", 50011, "6", 0, 25200, 65400),
    "pa": ("attn", "prefill", 50010, "7", 0, 25200, 65400),
    "df": ("ffn", "decode", 50021, "4", 0, 25300, 65500),
    "da": ("attn", "decode", 50020, "5", 0, 25300, 65500),
}


def _failure(procs: list[tuple[str, subprocess.Popen, object]]) -> str | None:
    for name, proc, fh in procs:
        rc = proc.poll()
        if rc is not None:
            return f"{name} exited early with code {rc}; see {fh.name}"
        try:
            text = Path(fh.name).read_text(errors="replace")
        except OSError:
            continue
        matches = ERROR_RE.findall(text)
        if matches:
            return f"{name} log contains fatal error {matches[-1]!r}; see {fh.name}"
    return None


def _check(procs: list[tuple[str, subprocess.Popen, object]]) -> None:
    failure = _failure(procs)
    if failure:
        raise RuntimeError(failure)


def wait_url(url: str, timeout: int, procs: list[tuple[str, subprocess.Popen, object]]) -> None:
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        _check(procs)
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status < 500:
                    return
        except Exception as exc:
            last_error = exc
        time.sleep(2)
    raise TimeoutError(f"service not ready: {url}; last error: {last_error!r}")


def wait_port(port: int, timeout: int, procs: list[tuple[str, subprocess.Popen, object]]) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _check(procs)
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"service port not ready: {port}")


def validate(args: argparse.Namespace) -> None:
    problems = []
    if not Path(args.repo).is_dir():
        problems.append(f"repo missing: {args.repo}")
    if not Path(args.model).is_dir():
        problems.append(f"model missing: {args.model}")
    if shutil.which(args.python) is None:
        problems.append(f"python missing: {args.python}")
    if problems:
        raise SystemExit("container path validation failed:\n  " + "\n  ".join(problems))


def stop(procs: list[tuple[str, subprocess.Popen, object]]) -> None:
    for _, proc, _ in reversed(procs):
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and any(p.poll() is None for _, p, _ in procs):
        time.sleep(0.2)
    for _, proc, _ in procs:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    for _, _, fh in procs:
        fh.close()


def launch(case: dict, args: argparse.Namespace, case_dir: Path):
    env0 = os.environ.copy()
    env0.update(
        {
            "AFD_TIMING": "1",
            "AFD_DETAILED_TIMING": "1",
            "SGLANG_DISABLE_REQUEST_LOGGING": "true",
            "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
            "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "600",
            "SGLANG_DISAGGREGATION_THREAD_POOL_SIZE": "128",
            "SGLANG_DISAGGREGATION_QUEUE_SIZE": "32",
            "AFD_IPC_SYNC_MODE": "ipc_event",
        }
    )
    common = [
        args.python,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model,
        "--tp",
        "1",
        "--host",
        "127.0.0.1",
        "--afd-comm-backend",
        "ipc_cpp",
        "--afd-micro-batch",
        str(case["m"]),
        "--afd-disagg-interleave-poll",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--disable-radix-cache",
        "--skip-server-warmup",
        "--mem-fraction-static",
        str(args.mem_fraction),
        "--num-reserved-decode-tokens",
        "64",
        "--disaggregation-transfer-backend",
        args.transfer_backend,
        "--disaggregation-bootstrap-port",
        str(args.bootstrap_port),
    ]
    if args.transfer_backend == "mooncake":
        common += ["--disaggregation-ib-device", args.ib_device]
    if case["m"] > 1:
        common += ["--afd-async-schedule"]
    procs = []
    for name in ("pf", "pa", "df", "da"):
        perspective, mode, port, gpu, peer, ucx_base, sched_port = ROLES[name]
        env = env0.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": gpu,
                "AFD_IPC_PEER_DEVICE": str(peer),
                "AFD_UCX_BASE_PORT": str(ucx_base),
                "AFD_SCHED_PORT": str(sched_port),
            }
        )
        if perspective == "attn":
            env["AFD_UCX_FFN_HOST"] = "127.0.0.1"
        cmd = common + [
            "--port",
            str(port),
            "--afd-perspective",
            perspective,
            "--disaggregation-mode",
            mode,
            "--base-gpu-id",
            "0",
        ]
        fh = (case_dir / f"{name}.log").open("w")
        proc = subprocess.Popen(
            cmd,
            cwd=args.repo,
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        procs.append((name, proc, fh))
        if name in ("pf", "df"):
            time.sleep(2)
        if name == "pa":
            wait_port(50010, args.startup_timeout, procs)
            wait_port(50011, args.startup_timeout, procs)
        if name == "da":
            wait_port(50020, args.startup_timeout, procs)
            wait_port(50021, args.startup_timeout, procs)
    router_cmd = [
        args.python,
        "-m",
        "sglang_router.launch_router",
        "--pd-disaggregation",
        "--mini-lb",
        "--prefill",
        "http://127.0.0.1:50010",
        "--decode",
        "http://127.0.0.1:50020",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.router_port),
    ]
    fh = (case_dir / "router.log").open("w")
    proc = subprocess.Popen(
        router_cmd,
        cwd=args.repo,
        env=os.environ.copy(),
        stdout=fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    procs.append(("router", proc, fh))
    wait_url(f"http://127.0.0.1:{args.router_port}/health", args.startup_timeout, procs)
    return procs


async def drive(case: dict, args: argparse.Namespace, procs) -> dict:
    import aiohttp

    connector = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    payload = {
        "text": args.prompt,
        "sampling_params": {
            "max_new_tokens": case["output_tokens"],
            "temperature": 0.0,
        },
    }
    sem = asyncio.Semaphore(case["concurrency"])
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        async def one(i):
            _check(procs)
            async with sem:
                t = time.perf_counter()
                try:
                    async with session.post(
                        f"http://127.0.0.1:{args.router_port}/generate", json=payload
                    ) as r:
                        body = await r.text()
                        return {
                            "ok": r.status == 200,
                            "status": r.status,
                            "latency_ms": (time.perf_counter() - t) * 1e3,
                            "error": None if r.status == 200 else body[:200],
                        }
                except Exception as exc:
                    return {
                        "ok": False,
                        "status": None,
                        "latency_ms": (time.perf_counter() - t) * 1e3,
                        "error": repr(exc),
                    }

        tasks = [asyncio.create_task(one(i)) for i in range(case["requests"])]
        pending = set(tasks)
        while pending:
            done, pending = await asyncio.wait(pending, timeout=1)
            _check(procs)
        return {"requests": [task.result() for task in tasks]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/workspace/sglang")
    ap.add_argument("--python", default="python3")
    ap.add_argument("--model", default="/models/Qwen3-32B")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--cases",
        nargs="+",
        choices=("m1-low", "m2-low", "m2-high", "m3-high", "m1-high", "m3-low"),
        default=("m1-low", "m1-high", "m3-low", "m3-high"),
    )
    ap.add_argument("--low-concurrency", type=int, default=1)
    ap.add_argument("--high-concurrency", type=int, default=192)
    ap.add_argument("--low-requests", type=int, default=8)
    ap.add_argument("--high-requests", type=int, default=192)
    ap.add_argument("--output-tokens", type=int, default=1024)
    ap.add_argument("--warmup-requests", type=int, default=8)
    ap.add_argument(
        "--prompt",
        default="Hello world, this is a test for microbatch pipeline performance.",
    )
    ap.add_argument("--mem-fraction", type=float, default=0.85)
    ap.add_argument("--startup-timeout", type=int, default=600)
    ap.add_argument("--request-timeout", type=int, default=300)
    ap.add_argument("--router-port", type=int, default=ROUTER_PORT)
    ap.add_argument("--bootstrap-port", type=int, default=BOOTSTRAP_PORT)
    ap.add_argument("--transfer-backend", choices=("mooncake", "fake"), default="mooncake")
    ap.add_argument("--ib-device", default="mlx5_4")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    validate(args)
    specs = {
        "m1-low": {
            "name": "m1-low",
            "m": 1,
            "concurrency": args.low_concurrency,
            "requests": args.low_requests,
            "output_tokens": args.output_tokens,
        },
        "m1-high": {
            "name": "m1-high",
            "m": 1,
            "concurrency": args.high_concurrency,
            "requests": args.high_requests,
            "output_tokens": args.output_tokens,
        },
        "m2-high": {
            "name": "m2-high",
            "m": 2,
            "concurrency": args.high_concurrency,
            "requests": args.high_requests,
            "output_tokens": args.output_tokens,
        },
        "m2-low": {
            "name": "m2-low",
            "m": 2,
            "concurrency": args.low_concurrency,
            "requests": args.low_requests,
            "output_tokens": args.output_tokens,
        },
        "m3-low": {
            "name": "m3-low",
            "m": 3,
            "concurrency": args.low_concurrency,
            "requests": args.low_requests,
            "output_tokens": args.output_tokens,
        },
        "m3-high": {
            "name": "m3-high",
            "m": 3,
            "concurrency": args.high_concurrency,
            "requests": args.high_requests,
            "output_tokens": args.output_tokens,
        },
    }
    run_dir = args.output_dir / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "repo": args.repo,
                "python": args.python,
                "model": args.model,
                "gpus": [4, 5, 6, 7],
                "backend": "ipc_cpp",
                "transfer_backend": args.transfer_backend,
                "router_port": args.router_port,
                "bootstrap_port": args.bootstrap_port,
                "cases": [specs[x] for x in args.cases],
            },
            indent=2,
        )
        + "\n"
    )
    if args.dry_run:
        print(run_dir)
        return
    for key in args.cases:
        case = specs[key]
        case_dir = run_dir / key
        case_dir.mkdir()
        procs = []
        try:
            procs = launch(case, args, case_dir)
            if args.warmup_requests:
                warm = {
                    **case,
                    "concurrency": min(args.warmup_requests, case["concurrency"]),
                    "requests": args.warmup_requests,
                    "output_tokens": 8,
                }
                asyncio.run(drive(warm, args, procs))
            result = asyncio.run(drive(case, args, procs))
            (case_dir / "client.json").write_text(json.dumps(result, indent=2) + "\n")
            time.sleep(3)
        finally:
            stop(procs)
        subprocess.run(
            [
                args.python,
                str(ROOT / "scripts" / "parse_breakdown.py"),
                "--case-dir",
                str(case_dir),
            ],
            check=True,
        )
    print(run_dir)


if __name__ == "__main__":
    main()
