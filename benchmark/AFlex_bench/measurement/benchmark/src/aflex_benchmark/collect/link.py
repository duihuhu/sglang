from __future__ import annotations

import json
import re
import shlex
import time
import uuid
from pathlib import Path

LINK_COUNTER_NOISE_BYTES = 1024**2


def parse_nvlink_counters(text):
    out = {"rx_bytes": 0.0, "tx_bytes": 0.0}
    hit = False
    for line in text.splitlines():
        low = line.lower()
        key = (
            "rx_bytes"
            if re.search(r"\b(rx|receive)\b", low)
            else ("tx_bytes" if re.search(r"\b(tx|transmit)\b", low) else None)
        )
        marker = re.search(r"\b(rx|receive|tx|transmit)\b", low)
        match = (
            re.search(
                r"(-?\d+(?:\.\d+)?)\s*(kb|kib|mb|mib|gb|gib|bytes?)?",
                low[marker.end() :],
            )
            if marker
            else None
        )
        if key and match:
            out[key] += float(match.group(1)) * {
                "kb": 1e3,
                "kib": 1024,
                "mb": 1e6,
                "mib": 1024**2,
                "gb": 1e9,
                "gib": 1024**3,
            }.get(match.group(2), 1)
            hit = True
    return out if hit else {}


def parse_pcie_rates(text):
    rx = tx = 0.0
    hit = False
    for line in text.splitlines():
        fields = [x.strip() for x in line.split(",")]
        try:
            rx += float(fields[-2]) * 1024
            tx += float(fields[-1]) * 1024
            hit = len(fields) >= 3
        except (ValueError, IndexError):
            pass
    return {"rx_bytes_s": rx, "tx_bytes_s": tx} if hit else {}


def parse_dmon_pcie_rates(text):
    """Parse ``nvidia-smi dmon -s t`` and return aggregate PCIe peaks."""
    header = None
    peak_rx = peak_tx = 0.0
    hit = False
    for line in text.splitlines():
        fields = line.strip().lstrip("#").strip().lower().split()
        if not fields:
            continue
        if "gpu" in fields and ("rxpci" in fields or "txpci" in fields):
            header = fields
            continue
        if header is None or not fields[0].isdigit() or len(fields) < len(header):
            continue
        row = dict(zip(header, fields))
        try:
            rx = float(row.get("rxpci", 0)) * 1024.0
            tx = float(row.get("txpci", 0)) * 1024.0
        except ValueError:
            continue
        peak_rx = max(peak_rx, rx)
        peak_tx = max(peak_tx, tx)
        hit = True
    return {"peak_rx_bytes_s": peak_rx, "peak_tx_bytes_s": peak_tx} if hit else {}


def parse_rdma_counters(text):
    out = {"rx_bytes": 0.0, "tx_bytes": 0.0}
    hit = False
    for line in text.splitlines():
        match = re.match(r"\s*(port_rcv_data|port_xmit_data)\s+(-?\d+)\s*$", line)
        if match:
            out["rx_bytes" if match.group(1) == "port_rcv_data" else "tx_bytes"] += (
                int(match.group(2)) * 4
            )
            hit = True
    return out if hit else {}


def collect_link_snapshot(executor, plan):
    rows = []
    inventory = {n["name"]: n for n in plan.cluster.get("nodes", [])}
    for name in sorted({p.node for p in plan.processes if p.gpus}):
        host = inventory.get(name, {}).get(
            "host", next(p.host for p in plan.processes if p.node == name)
        )
        gpus = ",".join(
            map(
                str,
                sorted({g for p in plan.processes if p.node == name for g in p.gpus}),
            )
        )
        calls = [
            (
                "pcie",
                executor.run(
                    host,
                    f"nvidia-smi -i {gpus} --query-gpu=index,pcie.rx_util,pcie.tx_util --format=csv,noheader,nounits",
                    check=False,
                    quiet=True,
                ),
                parse_pcie_rates,
            ),
            (
                "nvlink",
                executor.run(
                    host,
                    f"nvidia-smi nvlink -i {gpus} --getthroughput d",
                    check=False,
                    quiet=True,
                ),
                parse_nvlink_counters,
            ),
            (
                "rdma",
                executor.host_run(
                    host,
                    'for f in /sys/class/infiniband/*/ports/*/counters/port_rcv_data /sys/class/infiniband/*/ports/*/counters/port_xmit_data; do test -r "$f" && printf \'%s %s\\n\' "${f##*/}" "$(cat "$f")"; done',
                    check=False,
                    quiet=True,
                ),
                parse_rdma_counters,
            ),
        ]
        row = {"node": name, "host": host, "gpus": gpus}
        for kind, result, parser in calls:
            values = parser(result.stdout or "")
            row[kind] = {
                "available": bool(values),
                "values": values,
                "returncode": result.returncode,
                "error": (result.stderr or "").strip(),
            }
        rows.append(row)
    return {"timestamp_unix_s": time.time(), "nodes": rows}


def start_link_telemetry(executor, plan, before):
    """Start dmon for nodes whose PCIe query fields are unavailable."""
    snapshots = {row["node"]: row for row in before.get("nodes", [])}
    handles = []
    for node, row in snapshots.items():
        if row.get("pcie", {}).get("available"):
            continue
        token = uuid.uuid4().hex
        output = f"/tmp/aflex-pcie-{token}.dmon"
        pidfile = f"/tmp/aflex-pcie-{token}.pid"
        command = (
            f"rm -f -- {shlex.quote(output)} {shlex.quote(pidfile)}; "
            f"nvidia-smi dmon -i {shlex.quote(row['gpus'])} -s t -d 1 "
            f">{shlex.quote(output)} 2>&1 & echo $! >{shlex.quote(pidfile)}"
        )
        result = executor.run(row["host"], command, check=False, quiet=True)
        if result.returncode == 0:
            handles.append(
                {
                    "node": node,
                    "host": row["host"],
                    "output": output,
                    "pidfile": pidfile,
                }
            )
    return handles


def finish_link_telemetry(executor, handles):
    """Stop every dmon process, parse its full-window samples, and clean up."""
    rows = {}
    for handle in handles:
        output, pidfile = handle["output"], handle["pidfile"]
        command = (
            f"if test -s {shlex.quote(pidfile)}; then pid=$(cat {shlex.quote(pidfile)}); "
            f"kill -TERM $pid 2>/dev/null || true; wait $pid 2>/dev/null || true; fi; "
            f"test ! -f {shlex.quote(output)} || cat {shlex.quote(output)}; "
            f"rm -f -- {shlex.quote(output)} {shlex.quote(pidfile)}"
        )
        result = executor.run(
            handle["host"], command, check=False, quiet=True, timeout=30
        )
        values = parse_dmon_pcie_rates(result.stdout or "")
        rows[handle["node"]] = {
            "available": bool(values),
            "values": values,
            "returncode": result.returncode,
            "error": (result.stderr or "").strip(),
        }
    return rows


def build_link_telemetry(before, after, pcie_window=None):
    first = {r["node"]: r for r in before.get("nodes", [])}
    last = {r["node"]: r for r in after.get("nodes", [])}
    rows = []
    for node in sorted(set(first) | set(last)):
        a, b = first.get(node, {}), last.get(node, {})
        row = {"node": node}
        for kind in ("nvlink", "rdma"):
            x, y = a.get(kind, {}), b.get(kind, {})
            keys = set(x.get("values", {})) | set(y.get("values", {}))
            row[kind] = {
                "available": bool(x.get("available") and y.get("available")),
                "delta": {
                    k: max(
                        0.0,
                        float(y.get("values", {}).get(k, 0))
                        - float(x.get("values", {}).get(k, 0)),
                    )
                    for k in keys
                },
            }
        samples = [
            a.get("pcie", {}).get("values", {}),
            b.get("pcie", {}).get("values", {}),
        ]
        window = (pcie_window or {}).get(node, {})
        window_values = window.get("values", {})
        query_available = all(bool(x) for x in samples)
        row["pcie"] = {
            "available": query_available or bool(window.get("available")),
            "source": "query"
            if query_available
            else ("dmon_window" if window.get("available") else "unavailable"),
            "peak_rx_bytes_s": max(
                [x.get("rx_bytes_s", 0) for x in samples]
                + [window_values.get("peak_rx_bytes_s", 0)]
            ),
            "peak_tx_bytes_s": max(
                [x.get("tx_bytes_s", 0) for x in samples]
                + [window_values.get("peak_tx_bytes_s", 0)]
            ),
        }
        rows.append(row)
    return {"schema_version": 1, "before": before, "after": after, "nodes": rows}


def read_backend_logs(log_dir: Path):
    return (
        ""
        if not log_dir.exists()
        else "\n".join(
            p.read_text(errors="replace") for p in sorted(log_dir.glob("*.log"))
        )
    )


_COMM_LEDGER_RE = re.compile(r"AFLEX_COMM_LEDGER\s+(\{[^\r\n]+\})")
_COMM_LEDGER_FIELDS = (
    "logical_tx_bytes",
    "logical_rx_bytes",
    "expected_d2h_bytes",
    "expected_h2d_bytes",
    "calls",
    "tx_calls",
    "rx_calls",
)


def parse_comm_ledgers(logs):
    """Aggregate the latest cumulative marker for every process/backend pair."""
    latest = {}
    for match in _COMM_LEDGER_RE.finditer(logs):
        try:
            row = json.loads(match.group(1))
            key = (str(row.get("process_id", row["pid"])), str(row["backend"]))
            previous = latest.get(key)
            if previous is None or int(row.get("calls", 0)) >= int(
                previous.get("calls", 0)
            ):
                latest[key] = row
        except (KeyError, TypeError, ValueError):
            continue
    totals = {field: 0 for field in _COMM_LEDGER_FIELDS}
    processes = []
    for key in sorted(latest):
        row = latest[key]
        normalized = {
            "process_id": key[0],
            "pid": int(row["pid"]),
            "backend": key[1],
            **{
                field: max(0, int(row.get(field, 0) or 0))
                for field in _COMM_LEDGER_FIELDS
            },
        }
        processes.append(normalized)
        for field in _COMM_LEDGER_FIELDS:
            totals[field] += normalized[field]
    return {"marker": "AFLEX_COMM_LEDGER", **totals, "processes": processes}


def _paths(expected, arch):
    if isinstance(expected, dict):
        expected = expected.get(arch, expected.get("paths", expected.get("link")))
    if isinstance(expected, str):
        return [x.strip().lower() for x in expected.split(",") if x.strip()]
    return [str(x).lower() for x in expected] if isinstance(expected, list) else []


def _pattern(arch, path):
    if arch == "native":
        return {
            "shm": r"NCCL.*\bvia\s+SHM(?:/|\b)",
            "pcie": r"NCCL.*\bvia\s+SHM(?:/|\b)",
            "pcie_host_staged": r"NCCL.*\bvia\s+SHM(?:/|\b)",
            "p2p": r"NCCL.*\bP2P\b",
            "nvlink": r"NCCL.*\bP2P\b",
            "rdma": r"NCCL.*NET/IB",
            "ib": r"NCCL.*NET/IB",
        }.get(path, r"NCCL")
    if arch in {"pd", "pdaf"}:
        return {
            "tcp": r"(?i)(?:MC_FORCE_TCP is set, using TCP transport only|TcpTransport:\s*listen on port)",
            "pcie_host_staged": r"(?i)(?:MC_FORCE_TCP is set, using TCP transport only|TcpTransport:\s*listen on port)",
            "rdma": r"(?i)(mooncake|transfer).*(rdma|ib|verbs)",
            "ib": r"(?i)(mooncake|transfer).*(rdma|ib|verbs)",
            "nvlink": r"(?i)(?:Initialized custom memory pool:\s*NVLINK\b|CUDA IPC KV backend ready\b)",
        }.get(path, r"(?i)mooncake")
    afd_ipc = r"(?:\bCppIPC\b|afd_ipc_cpp|\bipc_cpp\b|\bAFD IPC\b|cuda.?ipc)"
    return {
        "ipc_cpp": rf"(?i){afd_ipc}",
        "pcie": rf"(?i)({afd_ipc}|AFD ZMQ handshake ready|\bZMQ\b|UCX communicator ready|AF communicator ready|\bUCX\b)",
        "pcie_host_staged": r"(?i)AFD ZMQ handshake ready",
        "zmq": r"(?i)(AFD ZMQ handshake ready|\bZMQ\b)",
        "tcp": r"(?i)(AFD ZMQ handshake ready|\bZMQ\b|UCX.*tcp)",
        "ucx": r"(?i)(UCX communicator ready|AF communicator ready|\bUCX\b)",
        "rdma": r"(?i)(?:(UCX|AF communicator).*(rc|rdma|ib|ready)|(?:Mooncake|TransferEngine).*(rdma|ib|verbs))",
        "nvlink": rf"(?i)({afd_ipc}|(?:UCX|AF communicator).*nvlink)",
    }.get(path, r"(?i)(ipc_cpp|ZMQ|UCX)")


def _counter_value(nodes, kind):
    """Return the largest per-node link-counter delta used for validation."""
    return max(
        (sum(row.get(kind, {}).get("delta", {}).values()) for row in nodes),
        default=0.0,
    )


def validate_link(
    point, telemetry, logs, semantic_comm_ledger=None, request_succeeded=False
):
    arch = str(point.get("architecture", "")).lower()
    paths = _paths(point.get("expected_link"), arch)
    nodes = telemetry.get("nodes", [])
    nvlink_counter = _counter_value(nodes, "nvlink")
    rdma_counter = _counter_value(nodes, "rdma")
    nvlink_active = nvlink_counter > LINK_COUNTER_NOISE_BYTES
    rdma_active = rdma_counter > LINK_COUNTER_NOISE_BYTES
    ledger = (
        semantic_comm_ledger
        if semantic_comm_ledger is not None
        else parse_comm_ledgers(logs)
    )
    ledger_backends = {row.get("backend") for row in ledger.get("processes", [])}
    expected_d2h = int(ledger.get("expected_d2h_bytes", 0) or 0)
    expected_h2d = int(ledger.get("expected_h2d_bytes", 0) or 0)
    backend_matches_arch = (arch == "af" and "af_zmq" in ledger_backends) or (
        arch in {"pd", "pdaf"} and "mooncake_tcp" in ledger_backends
    )
    point_metadata = point.get("metadata", {})
    expected_internal_link = point.get(
        "expected_internal_link", point_metadata.get("expected_internal_link")
    )
    component_tp = point.get("component_tp", point_metadata.get("component_tp"))
    internal_nvlink_expected = expected_internal_link == "nvlink" or component_tp == 2
    host_staged = any(
        path in {"pcie", "pcie_host_staged", "zmq", "tcp"} for path in paths
    )
    allow_internal_nvlink = (
        arch in {"af", "pd", "pdaf"}
        and host_staged
        and internal_nvlink_expected
        and backend_matches_arch
        and expected_d2h > 0
        and expected_h2d > 0
    )
    internal_link_evidence = {
        "expected_internal_link": expected_internal_link,
        "component_tp": component_tp,
        "nvlink_counter_value": nvlink_counter,
        "nvlink_active": nvlink_active,
        "accepted_as_internal": bool(
            nvlink_active and allow_internal_nvlink and not rdma_active
        ),
    }
    results = []
    for path in paths:
        ledger_backend = (
            arch == "af"
            and "af_zmq" in ledger_backends
            and path in {"pcie", "pcie_host_staged", "zmq", "tcp"}
        ) or (
            arch in {"pd", "pdaf"}
            and "mooncake_tcp" in ledger_backends
            and path in {"pcie", "pcie_host_staged", "tcp"}
        )
        backend = ledger_backend or bool(re.search(_pattern(arch, path), logs))
        if path == "nvlink":
            supported = any(r["nvlink"]["available"] for r in nodes)
            counter_value = nvlink_counter
            noise_threshold = LINK_COUNTER_NOISE_BYTES
            positive = nvlink_active
            negative = rdma_active
            if arch not in {"pd", "pdaf"}:
                negative = negative or bool(
                    re.search(r"(?i)\bvia\s+NET/Socket\b|UCX.*tcp", logs)
                )
            if arch == "af":
                negative = negative or bool(
                    re.search(r"(?i)\bZMQ\b|UCX.*(?:tcp|rdma|\brc\b|\bib\b)", logs)
                )
        elif path in {"rdma", "ib"}:
            supported = any(r["rdma"]["available"] for r in nodes)
            counter_value = rdma_counter
            noise_threshold = LINK_COUNTER_NOISE_BYTES
            positive = rdma_active
            negative = bool(re.search(r"(?i)\bvia\s+NET/Socket\b|UCX.*tcp", logs))
        else:
            supported = any(r["pcie"]["available"] for r in nodes)
            counter_value = max(
                (
                    max(
                        r["pcie"]["peak_rx_bytes_s"],
                        r["pcie"]["peak_tx_bytes_s"],
                    )
                    for r in nodes
                ),
                default=0.0,
            )
            # PCIe dmon is a sampled rate, so any positive short-smoke sample counts.
            noise_threshold = 0
            positive = counter_value > noise_threshold
            if arch == "native" and path in {"pcie", "pcie_host_staged", "shm"}:
                counter_conflict = rdma_active or nvlink_active
                channel_conflict = bool(
                    re.search(
                        r"(?i)\bvia\s+(?:NET/(?:IB|Socket)|P2P(?:/|\b)|NVLS(?:/|\b))",
                        logs,
                    )
                )
                negative = counter_conflict or channel_conflict
            elif path in {"tcp", "zmq", "pcie_host_staged"}:
                # Component-local TP collectives may legitimately use NVLink while
                # the AF/PD boundary itself remains host-staged. RDMA is always a
                # boundary contradiction for these paths.
                negative = rdma_active or (nvlink_active and not allow_internal_nvlink)
            else:
                negative = bool(re.search(r"(?i)NET/(IB|Socket)|UCX.*(rc|tcp)", logs))
                if arch == "af" and path == "ipc_cpp":
                    negative = negative or bool(
                        re.search(
                            r"(?i)\bZMQ\b|UCX.*(?:tcp|rdma|\brc\b|\bib\b)",
                            logs,
                        )
                    )
        background_noise = 0 < counter_value <= noise_threshold
        valid = backend and supported and positive and not negative
        reason = (
            None
            if valid
            else (
                "missing_backend_log_evidence"
                if not backend
                else "hardware_counter_unsupported"
                if not supported
                else "missing_positive_counter_evidence"
                if not positive
                else "contradictory_negative_evidence"
            )
        )
        results.append(
            {
                "path": path,
                "valid": valid,
                "backend_log": backend,
                "counter_supported": supported,
                "counter_value": counter_value,
                "noise_threshold": noise_threshold,
                "background_noise": background_noise,
                "positive_counter": positive,
                "contradictory_evidence": negative,
                "reason": reason,
            }
        )
    valid = bool(paths) and all(x["valid"] for x in results)
    cuda_ipc_transfer_complete = bool(
        re.search(r"(?i)CUDA IPC KV transfer complete bytes=[1-9]\d*\b", logs)
    )
    cuda_ipc_semantic_candidate = (
        arch == "pd"
        and point.get("recipe") == "pd_cuda_ipc_same_node"
        and bool(request_succeeded)
        and cuda_ipc_transfer_complete
        and bool(results)
        and all(
            row["path"] == "nvlink"
            and row["backend_log"]
            and not row["counter_supported"]
            and not row["contradictory_evidence"]
            for row in results
        )
    )
    semantic_candidate = cuda_ipc_semantic_candidate or (
        arch in {"af", "pd", "pdaf"}
        and any(path in {"pcie", "pcie_host_staged", "zmq"} for path in paths)
        and expected_d2h > 0
        and expected_h2d > 0
        and backend_matches_arch
        and bool(results)
        and all(
            row["backend_log"] and not row["contradictory_evidence"] for row in results
        )
    )
    status = (
        "pass"
        if valid
        else ("verified_semantic" if semantic_candidate else "invalid_link")
    )
    if semantic_candidate and not valid:
        for row in results:
            row["semantic_valid"] = bool(
                row["backend_log"] and not row["contradictory_evidence"]
            )
            row["validation_mode"] = "semantic"
            row["reason"] = None if row["semantic_valid"] else row["reason"]
    else:
        for row in results:
            row["semantic_valid"] = False
            row["validation_mode"] = "physical"
    return {
        "schema_version": 2,
        "status": status,
        "validation_mode": "semantic" if status == "verified_semantic" else "physical",
        "physical_verified": status == "pass",
        "semantic_comm_ledger": ledger,
        "cuda_ipc_semantic_evidence": {
            "eligible_recipe": point.get("recipe") == "pd_cuda_ipc_same_node",
            "request_succeeded": bool(request_succeeded),
            "transfer_complete_log": cuda_ipc_transfer_complete,
            "accepted": cuda_ipc_semantic_candidate,
        },
        "internal_link_evidence": internal_link_evidence,
        "architecture": arch,
        "expected_link": point.get("expected_link"),
        "paths": results,
        "reason": None
        if status != "invalid_link"
        else ("missing_expected_link" if not paths else "path_validation_failed"),
    }
