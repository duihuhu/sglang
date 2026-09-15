from __future__ import annotations

import json
import shlex

from .planner_common import PlanContext
from .recipes import AF_SERVER_FLAGS, COMMON_FLAGS, _nic, _router, _server


def _nodes(cluster, point):
    count = int(point["nodes"])
    placement_fields = ("native_placements", "prefill_placements", "decode_placements", "ffn_placements", "attention_placements")
    requested = []
    for field in placement_fields:
        for placement in point.get(field, []):
            if placement.get("node") not in requested:
                requested.append(placement.get("node"))
    if requested:
        by_name = {node["name"]: node for node in cluster["nodes"]}
        try:
            nodes = [by_name[name] for name in requested]
        except KeyError as exc:
            raise ValueError(f"placement references unknown node {exc.args[0]}") from exc
    else:
        nodes = list(cluster["nodes"][:count])
    if len(nodes) != count:
        raise ValueError(f"matrix workload requires exactly {count} placed cluster nodes")
    return nodes


def _placements(cluster, point, nodes, architecture):
    explicit_fields = {"native": {"NATIVE": "native_placements"}, "pd": {"P": "prefill_placements", "D": "decode_placements"}, "af": {"F": "ffn_placements", "A": "attention_placements"}}
    fields = explicit_fields.get(architecture, {})
    if fields and all(point.get(field) for field in fields.values()):
        return {role: _named_placements(cluster, point, field) for role, field in fields.items()}
    count = len(nodes)
    if architecture == "native":
        if count == 1:
            return {"NATIVE": [(nodes[0], [0, 1, 2, 3])]}
        if count == 2:
            return {"NATIVE": [(nodes[0], [0, 1]), (nodes[1], [0, 1])]}
        if count == 4:
            return {"NATIVE": [(node, [0]) for node in nodes]}
    elif architecture in {"pd", "af"}:
        left, right = ("P", "D") if architecture == "pd" else ("F", "A")
        if count == 1:
            return {left: [(nodes[0], [0, 1])], right: [(nodes[0], [2, 3])]}
        if count == 2:
            return {left: [(nodes[0], [0, 1])], right: [(nodes[1], [0, 1])]}
        if count == 4:
            return {
                left: [(nodes[0], [0]), (nodes[1], [0])],
                right: [(nodes[2], [0]), (nodes[3], [0])],
            }
    raise ValueError(f"unsupported {architecture} matrix topology with {count} nodes")


def _global_network_env(node, extra=""):
    """Pin global-component collectives and service binding to node IPv4/RoCE."""
    interface = node["roce"]
    values = (
        f"GLOO_SOCKET_IFNAME={interface} NCCL_SOCKET_IFNAME={interface} "
        f"SGLANG_HOST_IP={node['host']}"
    )
    return f"{values} {extra}".strip()


def _global_component(
    context,
    point,
    model,
    role,
    placements,
    *,
    port,
    tp,
    extra="",
    env_for_rank=None,
    stage=0,
    health_path="/health",
    metadata=None,
    bootstrap=None,
    internal_for_rank=None,
    visible_for_rank=None,
    common_flags=COMMON_FLAGS,
):
    """Launch one global TP component and expose readiness only on rank zero."""
    world_nodes = len(placements)
    dist = f"{placements[0][0]['host']}:{port + 2}"
    specs = []
    for rank, (node, gpus) in enumerate(placements):
        distributed = (
            (f"--nnodes {world_nodes} --node-rank {rank} --dist-init-addr {dist}")
            if world_nodes > 1
            else ""
        )
        internal = list(
            internal_for_rank(rank, node, gpus) if internal_for_rank else ()
        )
        if world_nodes > 1 and rank == 0:
            internal.insert(0, port + 2)
        spec = _server(
            context,
            role,
            node,
            gpus,
            port,
            model,
            tp,
            extra=f"{extra} {distributed}",
            env=_global_network_env(
                node, env_for_rank(rank, node, gpus) if env_for_rank else ""
            ),
            nccl=port + 1,
            bootstrap=bootstrap if rank == 0 else None,
            internal=internal,
            visible=(visible_for_rank(rank, node, gpus) if visible_for_rank else None),
            health=health_path if rank == 0 else None,
            metadata={
                **(metadata or {}),
                "component_role": role,
                "global_tp": tp,
                "component_nodes": world_nodes,
                "node_rank": rank,
                "readiness_rank": 0,
            },
            common_flags=common_flags,
            startup_stage=stage,
        )
        specs.append(spec)
    return specs


def _named_placements(cluster, point, field):
    """Resolve an explicit component placement into per-node TP launch groups."""
    by_name = {node["name"]: node for node in cluster["nodes"]}
    raw = point.get(field)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{field} must be a non-empty placement list")
    placements = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"{field}[{index}] must be an object")
        node = by_name.get(item.get("node"))
        gpus = item.get("gpus")
        if node is None:
            raise ValueError(f"{field}[{index}] references an unknown node")
        if (
            not isinstance(gpus, list)
            or not gpus
            or any(
                isinstance(gpu, bool)
                or not isinstance(gpu, int)
                or gpu not in node["gpus"]
                for gpu in gpus
            )
        ):
            raise ValueError(f"{field}[{index}] has invalid GPUs")
        placements.append((node, list(gpus)))
    return placements


def _afd_mooncake_timeout_ms(point):
    """Bound Mooncake waits to the request deadline plus a cleanup buffer."""
    value = point.get("afd_mooncake_timeout_ms")
    if value is None:
        value = max(300_000, int(point.get("request_timeout_s", 240)) * 1000 + 60_000)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("afd_mooncake_timeout_ms must be a positive integer")
    return value


def matrix_workloads_af_mooncake_tp(cluster, model, point):
    """Homogeneous AF Mooncake TP, either local or rank-mapped multi-node."""
    if point.get("architecture") != "af" or point.get("expected_link") != "rdma":
        raise ValueError("matrix_workloads_af_mooncake_tp requires AF over RDMA")
    if point.get("afd_backend") != "mooncake":
        raise ValueError("matrix_workloads_af_mooncake_tp requires Mooncake")

    context = PlanContext(cluster)
    tp = int(point["af_tp"])
    if tp < 1:
        raise ValueError("af_tp must be positive")
    aplacements = _named_placements(cluster, point, "attention_placements")
    fplacements = _named_placements(cluster, point, "ffn_placements")
    if (
        sum(len(gpus) for _, gpus in aplacements) != tp
        or sum(len(gpus) for _, gpus in fplacements) != tp
    ):
        raise ValueError("each Mooncake component must place exactly af_tp GPUs")
    if len(aplacements) != len(fplacements):
        raise ValueError("Mooncake A/F components require the same node-rank count")
    if len(aplacements) > 1 and any(
        len(gpus) != 1 for _, gpus in aplacements + fplacements
    ):
        raise ValueError("global Mooncake TP currently requires one GPU per node rank")

    base = int(point["port_base"])
    mooncake_timeout_ms = _afd_mooncake_timeout_ms(point)
    sched, ffn_control, attn_control = base + 5, base + 30, base + 40
    common_flags = (
        f"--afd-comm-backend mooncake --afd-micro-batch 1 "
        f"--afd-attn-tp {tp} --afd-ffn-tp {tp}"
    )
    metadata = {
        "recipe": "matrix_workloads_af_mooncake_tp",
        "matrix": "af-mooncake-rdma-smoke",
        "topology_id": point["metadata"]["topology_id"],
        "expected_link": "rdma",
        "transport": "mooncake",
        "tp": tp,
    }

    def component(role, placements, peer_placements, offset, perspective):
        def env(rank, node, gpus):
            peer = peer_placements[rank][0]
            nics = [_nic(node, gpu, cluster) for gpu in gpus]
            nic_list = ",".join(dict.fromkeys(nics))
            ucx_list = ",".join(f"{nic}:1" for nic in dict.fromkeys(nics))
            mooncake_map = json.dumps({str(local): nic for local, nic in enumerate(nics)}, separators=(",", ":"))
            return (
                "SGLANG_MOONCAKE_TRANSPORT=rdma MOONCAKE_USE_CUDA_IPC=0 "
                f"MOONCAKE_IB_DEVICE={shlex.quote(mooncake_map)} "
                f"UCX_NET_DEVICES={ucx_list} "
                f"NCCL_IB_HCA={nic_list} "
                f"AFD_MOONCAKE_TIMEOUT_MS={mooncake_timeout_ms} "
                f"AFD_MOONCAKE_ATTN_CONTROL_PORT={attn_control} "
                f"AFD_MOONCAKE_FFN_CONTROL_PORT={ffn_control} "
                f"AFD_SCHED_HOST={fplacements[0][0]['host']} "
                f"AFD_SCHED_PORT={sched} AFD_MOONCAKE_PEER_HOST={peer['host']}"
            )

        def internal(rank, node, gpus):
            ports = []
            if role == "F" and rank == 0:
                ports.append(sched)
            local_control = attn_control if role == "F" else ffn_control
            if len(placements) == 1:
                ports.extend(range(local_control, local_control + tp))
            else:
                ports.append(local_control + rank)
            return ports

        return _global_component(
            context,
            point,
            model,
            role,
            placements,
            port=base + offset,
            tp=tp,
            extra=f"--afd-perspective {perspective} --base-gpu-id 0 {common_flags}",
            env_for_rank=env,
            stage=0,
            health_path="/get_model_info",
            metadata={**metadata, "perspective": perspective},
            internal_for_rank=internal,
            common_flags=COMMON_FLAGS,
        )

    # Both sides must start in one stage: communicator construction performs a
    # symmetric rank-paired HELLO exchange before either HTTP server is ready.
    fs = component("F", fplacements, aplacements, 0, "ffn")
    ass = component("A", aplacements, fplacements, 10, "attn")
    return context.finish(
        "af", fs + ass, f"http://{aplacements[0][0]['host']}:{base + 10}"
    )


def _meta(point):
    metadata = {
        "recipe": "matrix_workloads_4gpu",
        "matrix": "matrix-workloads",
        "topology_id": point["metadata"]["topology_id"],
        "expected_link": point["expected_link"],
    }
    for key in ("expected_internal_link", "component_tp"):
        if key in point:
            metadata[key] = point[key]
    if point["expected_link"] == "pcie_host_staged":
        metadata["comm_ledger_backend"] = {
            "af": "af_zmq",
            "pd": "mooncake_tcp",
        }.get(point["architecture"])
    return {key: value for key, value in metadata.items() if value is not None}


def matrix_workloads_4gpu(cluster, model, point):
    architecture = point["architecture"]
    context = PlanContext(cluster)
    nodes = _nodes(cluster, point)
    layout = _placements(cluster, point, nodes, architecture)
    base = int(point["port_base"])
    metadata = _meta(point)
    if architecture == "native":
        placements = layout["NATIVE"]

        def env(rank, node, gpus):
            rdma = len(nodes) > 1
            nics = ",".join(dict.fromkeys(_nic(node, gpu, cluster) for gpu in gpus))
            return (
                "NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P,SHM,NET "
                f"NCCL_IB_DISABLE={0 if rdma else 1} NCCL_NVLS_ENABLE=0 "
                f"NCCL_IB_HCA={nics}"
            )

        processes = _global_component(
            context,
            point,
            model,
            "NATIVE",
            placements,
            port=base,
            tp=int(point.get("component_tp", 4)),
            extra="--disable-custom-all-reduce",
            env_for_rank=env,
            metadata={
                **metadata,
                "transport": "nccl_rdma" if len(nodes) > 1 else "nccl",
            },
        )
        return context.finish(
            "native", processes, f"http://{placements[0][0]['host']}:{base}"
        )
    if architecture == "pd":
        bootstrap = base + 5
        cuda_ipc = point["expected_link"] == "nvlink"
        protocol = "rdma" if len(nodes) > 1 else "tcp"
        shared_visible = list(range(4)) if cuda_ipc else None

        def component(role, stage, offset):
            placements = layout[role]
            mode = "prefill" if role == "P" else "decode"

            def env(rank, node, gpus):
                if cuda_ipc:
                    return ""
                if point["expected_link"] == "pcie_host_staged":
                    return (
                        "unset MC_FORCE_HCA MC_FORCE_MNNVL MC_INTRANODE_NVLINK "
                        "MC_INTRA_NVLINK SGLANG_MOONCAKE_CUSTOM_MEM_POOL; "
                        "export MC_FORCE_TCP=1 MC_LOG_LEVEL=INFO "
                        "MOONCAKE_USE_CUDA_IPC=0 MOONCAKE_PROTOCOL=tcp "
                        "SGLANG_MOONCAKE_TRANSPORT=tcp"
                    )
                nics = list(dict.fromkeys(_nic(node, gpu, cluster) for gpu in gpus))
                nic_list = ",".join(nics)
                ucx_list = ",".join(f"{nic}:1" for nic in nics)
                return (f"MOONCAKE_PROTOCOL={protocol} MOONCAKE_USE_CUDA_IPC=0 "
                        f"UCX_NET_DEVICES={ucx_list} NCCL_IB_HCA={nic_list}")

            backend = "cuda_ipc" if cuda_ipc else "mooncake"
            backend_flags = (
                "--disaggregation-transfer-backend cuda_ipc"
                if cuda_ipc
                else "--disaggregation-transfer-backend mooncake "
                f"--disaggregation-ib-device {','.join(dict.fromkeys(_nic(placements[0][0], gpu, cluster) for gpu in placements[0][1]))}"
            )
            return _global_component(
                context,
                point,
                model,
                role,
                placements,
                port=base + offset,
                tp=int(point.get("component_tp", 2)),
                extra=(
                    f"--disaggregation-mode {mode} "
                    f"--base-gpu-id {2 if cuda_ipc and role == 'D' else 0} "
                    f"{backend_flags} --disaggregation-bootstrap-port {bootstrap}"
                ),
                env_for_rank=env,
                stage=stage,
                metadata={
                    **metadata,
                    "transport": backend if cuda_ipc else f"mooncake_{protocol}",
                    "phase": mode,
                },
                bootstrap=bootstrap if role == "P" else None,
                visible_for_rank=(lambda rank, node, gpus: shared_visible),
            )

        prefills, decodes = component("P", 0, 0), component("D", 1, 20)
        p0, d0 = prefills[0], decodes[0]
        args = (
            f"--pd-disaggregation --prefill http://{p0.host}:{p0.port} {bootstrap} "
            f"--decode http://{d0.host}:{d0.port}"
        )
        router = _router(
            context,
            "PD_ROUTER",
            nodes[0],
            base + 40,
            args,
            prom=base + 41,
            metadata={**metadata, "level": "router"},
            startup_stage=2,
        )
        return context.finish(
            "pd",
            prefills + decodes + [router],
            f"http://{nodes[0]['host']}:{base + 40}",
        )
    if architecture == "af":
        fplacements, aplacements = layout["F"], layout["A"]
        backend = {"nvlink": "ipc_cpp", "pcie_host_staged": "zmq"}.get(
            point["expected_link"], "ucx"
        )
        visible = list(range(4)) if backend == "ipc_cpp" else None
        sched = base + 5

        def component(role, placements, stage, offset, perspective):
            peer = aplacements if role == "F" else fplacements

            def env(rank, node, gpus):
                peer_node = peer[min(rank, len(peer) - 1)][0]
                common = (
                    f"AFD_SCHED_PORT={sched} AFD_UCX_BASE_PORT={base + 60} "
                    f"AFD_NVML_DEVICE_INDICES={','.join(map(str, gpus))}"
                )
                if backend == "ipc_cpp":
                    return common + (
                        " AFD_IPC_PEER_OFFSET=2"
                        if role == "F"
                        else " AFD_IPC_PEER_OFFSET=-2 AFD_UCX_FFN_HOST=127.0.0.1"
                    )
                if backend == "zmq":
                    return common
                return common + (
                    f" UCX_NET_DEVICES={_nic(node, gpus[0], cluster)}:1 "
                    f"AFD_UCX_FFN_HOST={peer_node['host']}"
                    if role == "A"
                    else f" UCX_NET_DEVICES={_nic(node, gpus[0], cluster)}:1"
                )

            return _global_component(
                context,
                point,
                model,
                role,
                placements,
                port=base + offset,
                tp=2,
                extra=(
                    f"--afd-perspective {perspective} --base-gpu-id {2 if perspective == 'attn' and backend == 'ipc_cpp' else 0} "
                    f"--afd-comm-backend {backend} --afd-micro-batch 1 "
                    "--afd-attn-tp 2 --afd-ffn-tp 2"
                ),
                env_for_rank=env,
                stage=stage,
                health_path="/get_model_info",
                metadata={**metadata, "transport": backend, "perspective": perspective},
                internal_for_rank=(
                    lambda rank, node, gpus: (
                        (sched, base + 60) if role == "F" and rank == 0 else ()
                    )
                ),
                visible_for_rank=(lambda rank, node, gpus: visible),
                common_flags=AF_SERVER_FLAGS,
            )

        fs = component("F", fplacements, 0, 0, "ffn")
        ass = component("A", aplacements, 1, 20, "attn")
        return context.finish(
            "af", fs + ass, f"http://{aplacements[0][0]['host']}:{base + 20}"
        )
    raise ValueError(f"unsupported matrix architecture: {architecture}")
