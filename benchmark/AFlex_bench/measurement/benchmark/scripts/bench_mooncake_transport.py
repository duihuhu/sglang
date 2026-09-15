#!/usr/bin/env python3
"""Fail-closed two-process Mooncake GPU transport microbenchmark.

The launcher starts a receiver on GPU 1 and a sender on GPU 0.  Both processes
register their CUDA buffers directly with Mooncake; no SGLang server or CUDA IPC
fallback is involved.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

TRANSPORTS = ("nvlink", "rdma", "tcp")
COUNTER_NOISE_BYTES = 1024**2
CONTAINER_SCRIPT = (
    "/workspace/moe-tier/benchmark/AFlex_bench/measurement/benchmark/scripts/"
    "bench_mooncake_transport.py"
)


def _recv_exact(sock, size):
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise RuntimeError("control socket closed early")
        data.extend(chunk)
    return bytes(data)


def send_json(sock, value):
    payload = json.dumps(value).encode()
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def recv_json(sock):
    return json.loads(_recv_exact(sock, struct.unpack("!I", _recv_exact(sock, 4))[0]))


def transport_environment(transport):
    env = {"SGLANG_MOONCAKE_TRANSPORT": transport, "MOONCAKE_USE_CUDA_IPC": "0"}
    if transport == "nvlink":
        env.update({"MC_FORCE_MNNVL": "true", "MC_FORCE_TCP": ""})
    elif transport == "tcp":
        env.update({"MC_FORCE_TCP": "1", "MC_FORCE_MNNVL": ""})
    else:
        env.update({"MC_FORCE_TCP": "", "MC_FORCE_MNNVL": ""})
    return env


def apply_transport_environment(transport):
    for key, value in transport_environment(transport).items():
        if value:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)


def make_pattern(size, seed):
    import torch

    return ((torch.arange(size, dtype=torch.int64) + seed) % 251).to(torch.uint8)


def make_engine(host, transport, device):
    from mooncake.engine import TransferEngine

    apply_transport_environment(transport)
    engine = TransferEngine()
    protocol = "tcp" if transport == "tcp" else "rdma"
    ret = engine.initialize(host, "P2PHANDSHAKE", protocol, device or "")
    if ret != 0:
        raise RuntimeError(f"Mooncake initialize failed: ret={ret}")
    return engine, f"{host}:{engine.get_rpc_port()}", protocol


def worker(args):
    import torch

    torch.cuda.set_device(0)
    engine, session, protocol = make_engine(args.host, args.transport, args.device)
    buffer = torch.empty(args.bytes, dtype=torch.uint8, device="cuda:0")
    buffer.copy_(
        make_pattern(args.bytes, args.seed).to("cuda:0")
    ) if args.role == "sender" else buffer.zero_()
    torch.cuda.synchronize()
    ptr = buffer.data_ptr()
    ret = engine.register_memory(ptr, args.bytes)
    if ret != 0:
        raise RuntimeError(f"register_memory failed: ret={ret}")
    result = {
        "role": args.role,
        "session_id": session,
        "ptr": ptr,
        "protocol": protocol,
    }
    try:
        if args.role == "receiver":
            with socket.socket() as listener:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind((args.control_host, args.port))
                listener.listen(1)
                listener.settimeout(args.timeout)
                conn, _ = listener.accept()
                with conn:
                    send_json(
                        conn, {"session_id": session, "ptr": ptr, "bytes": args.bytes}
                    )
                    command = recv_json(conn)
                    if command != {"command": "verify"}:
                        raise RuntimeError(f"unexpected control command: {command}")
                    torch.cuda.synchronize()
                    expected = make_pattern(args.bytes, args.seed)
                    actual = buffer.cpu()
                    mismatches = int(torch.count_nonzero(actual != expected).item())
                    verification = {"ok": mismatches == 0, "mismatches": mismatches}
                    send_json(conn, verification)
                    result["verification"] = verification
        else:
            deadline = time.monotonic() + args.timeout
            while True:
                try:
                    conn = socket.create_connection(
                        (args.control_host, args.port), timeout=2
                    )
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.1)
            with conn:
                peer = recv_json(conn)
                latencies = []
                for iteration in range(args.warmup + args.iters):
                    start = time.perf_counter_ns()
                    ret = engine.transfer_sync_write(
                        peer["session_id"], ptr, peer["ptr"], args.bytes
                    )
                    elapsed = time.perf_counter_ns() - start
                    if ret != 0:
                        raise RuntimeError(f"transfer_sync_write failed: ret={ret}")
                    if iteration >= args.warmup:
                        latencies.append(elapsed / 1000)
                send_json(conn, {"command": "verify"})
                verification = recv_json(conn)
                ordered = sorted(latencies)
                p50 = ordered[len(ordered) // 2]
                result.update(
                    {
                        "verification": verification,
                        "latency_us": {
                            "p50": p50,
                            "p95": ordered[int(len(ordered) * 0.95)],
                        },
                        "bandwidth_GBps": args.bytes / (p50 / 1e6) / 1e9,
                    }
                )
    finally:
        with contextlib.suppress(Exception):
            engine.unregister_memory(ptr)
    print(json.dumps(result), flush=True)
    return 0 if result.get("verification", {}).get("ok") else 1


def run_command(command):
    proc = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "available": proc.returncode == 0 and bool(proc.stdout.strip()),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr.strip(),
    }


def parse_nvlink(text):
    values = {"rx_bytes": 0, "tx_bytes": 0}
    hit = False
    for line in text.splitlines():
        direction = (
            "rx_bytes"
            if re.search(r"\b(rx|receive)\b", line, re.IGNORECASE)
            else "tx_bytes"
            if re.search(r"\b(tx|transmit)\b", line, re.IGNORECASE)
            else None
        )
        marker = re.search(r"\b(rx|receive|tx|transmit)\b", line, re.IGNORECASE)
        match = (
            re.search(
                r"(\d+(?:\.\d+)?)\s*(KiB|MiB|GiB|KB|MB|GB|bytes?)?",
                line[marker.end() :],
                re.IGNORECASE,
            )
            if marker
            else None
        )
        if direction and match:
            unit = (match.group(2) or "bytes").lower()
            values[direction] += float(match.group(1)) * {
                "kib": 1024,
                "mib": 1024**2,
                "gib": 1024**3,
                "kb": 1e3,
                "mb": 1e6,
                "gb": 1e9,
            }.get(unit, 1)
            hit = True
    return values if hit else {}


def read_rdma():
    values = {"rx_bytes": 0, "tx_bytes": 0}
    paths = list(Path("/sys/class/infiniband").glob("*/ports/*/counters/port_*_data"))
    read = 0
    for path in paths:
        try:
            value = int(path.read_text().strip()) * 4
        except (OSError, ValueError):
            continue
        values["rx_bytes" if path.name == "port_rcv_data" else "tx_bytes"] += value
        read += 1
    return {"available": read > 0, "values": values, "paths_read": read}


def snapshot(gpus):
    nv = run_command(
        ["nvidia-smi", "nvlink", "-i", ",".join(map(str, gpus)), "--getthroughput", "d"]
    )
    nv["values"] = parse_nvlink(nv.pop("stdout"))
    nv["available"] = nv["available"] and bool(nv["values"])
    return {"nvlink": nv, "rdma": read_rdma()}


def counter_delta(before, after):
    result = {}
    for kind in ("nvlink", "rdma"):
        available = before[kind]["available"] and after[kind]["available"]
        keys = set(before[kind].get("values", {})) | set(after[kind].get("values", {}))
        result[kind] = {
            "available": available,
            "delta": {
                key: max(
                    0,
                    after[kind].get("values", {}).get(key, 0)
                    - before[kind].get("values", {}).get(key, 0),
                )
                for key in keys
            },
        }
    return result


def validate_counters(transport, counters):
    nv = sum(counters["nvlink"]["delta"].values())
    rdma = sum(counters["rdma"]["delta"].values())
    if transport == "nvlink":
        ok = (
            counters["nvlink"]["available"]
            and nv > COUNTER_NOISE_BYTES
            and rdma <= COUNTER_NOISE_BYTES
        )
    elif transport == "rdma":
        ok = (
            counters["rdma"]["available"]
            and rdma > COUNTER_NOISE_BYTES
            and nv <= COUNTER_NOISE_BYTES
        )
    else:
        ok = (
            counters["nvlink"]["available"]
            and counters["rdma"]["available"]
            and nv <= COUNTER_NOISE_BYTES
            and rdma <= COUNTER_NOISE_BYTES
        )
    return {
        "ok": ok,
        "nvlink_bytes": nv,
        "rdma_bytes": rdma,
        "noise_threshold": COUNTER_NOISE_BYTES,
    }


def parse_worker(proc, role):
    lines = [line for line in proc.stdout.splitlines() if line.strip().startswith("{")]
    if proc.returncode != 0 or not lines:
        details = f"{proc.stderr.strip()} {proc.stdout.strip()}"
        raise RuntimeError(f"{role} failed rc={proc.returncode}: {details}")
    return json.loads(lines[-1])


def build_worker_command(args, role):
    gpu = args.dst_gpu if role == "receiver" else args.src_gpu
    common = [
        "--worker",
        "--transport",
        args.transport,
        "--bytes",
        str(args.bytes),
        "--iters",
        str(args.iters),
        "--warmup",
        str(args.warmup),
        "--seed",
        str(args.seed),
        "--host",
        args.host,
        "--control-host",
        args.control_host,
        "--port",
        str(args.port),
        "--timeout",
        str(args.timeout),
        "--device",
        args.device or "",
    ]
    env = os.environ.copy()
    if args.container:
        command = [
            "docker",
            "exec",
            "-e",
            f"CUDA_VISIBLE_DEVICES={gpu}",
            args.container,
            "python3",
            CONTAINER_SCRIPT,
            role,
            *common,
        ]
    else:
        command = [sys.executable, str(Path(__file__).resolve()), role, *common]
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return command, env


def launch(args):
    before = snapshot((args.src_gpu, args.dst_gpu))
    receiver_command, receiver_env = build_worker_command(args, "receiver")
    sender_command, sender_env = build_worker_command(args, "sender")
    receiver = subprocess.Popen(
        receiver_command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=receiver_env,
    )
    time.sleep(0.5)
    sender = subprocess.run(
        sender_command,
        text=True,
        capture_output=True,
        env=sender_env,
        timeout=args.timeout + 30,
        check=False,
    )
    recv_out, recv_err = receiver.communicate(timeout=args.timeout + 30)
    receiver.returncode = receiver.wait()
    receiver.stdout, receiver.stderr = recv_out, recv_err
    after = snapshot((args.src_gpu, args.dst_gpu))
    counters = counter_delta(before, after)
    send_result, recv_result = (
        parse_worker(sender, "sender"),
        parse_worker(receiver, "receiver"),
    )
    counter_check = validate_counters(args.transport, counters)
    ok = (
        send_result["verification"]["ok"]
        and recv_result["verification"]["ok"]
        and counter_check["ok"]
    )
    return {
        "schema_version": 1,
        "status": "pass" if ok else "fail",
        "transport": args.transport,
        "config": {
            "src_gpu": args.src_gpu,
            "dst_gpu": args.dst_gpu,
            "bytes": args.bytes,
            "iters": args.iters,
            "warmup": args.warmup,
            "device": args.device,
        },
        "sender": send_result,
        "receiver": recv_result,
        "counters": counters,
        "counter_validation": counter_check,
    }


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("role", nargs="?", choices=("sender", "receiver"))
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--transport", required=True, choices=TRANSPORTS)
    p.add_argument("--src-gpu", type=int, default=0)
    p.add_argument("--dst-gpu", type=int, default=1)
    p.add_argument("--bytes", type=int, default=64 * 1024**2)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument(
        "--container",
        help="run workers with docker exec in this host-networked container",
    )
    p.add_argument("--control-host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=29611)
    p.add_argument("--timeout", type=float, default=60)
    p.add_argument("--device", default="")
    p.add_argument("--output", type=Path)
    return p


def main():
    args = parser().parse_args()
    try:
        if args.worker:
            if args.role is None:
                raise ValueError("worker requires sender or receiver role")
            if args.container:
                raise ValueError("--container is launcher-only and cannot be recursive")
            return worker(args)
        result = launch(args)
    except Exception as exc:  # noqa: BLE001 - JSON must capture every failure
        result = {
            "schema_version": 1,
            "status": "fail",
            "transport": args.transport,
            "error": f"{type(exc).__name__}: {exc}",
        }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temp = args.output.with_suffix(args.output.suffix + ".tmp")
        temp.write_text(encoded + "\n")
        temp.replace(args.output)
    print(encoded)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
