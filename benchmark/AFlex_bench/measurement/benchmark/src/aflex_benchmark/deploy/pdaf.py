from __future__ import annotations
from .planner_common import PlanContext, node_map
from ..topology import PortAllocator, partition, select_gpus


def _build_chain(context, node, refs, model, point, ports):
    groups = partition(
        refs, [int(point[k]) for k in ("pf_gpus", "pa_gpus", "df_gpus", "da_gpus")]
    )
    pf, pa, df, da = groups
    if not (len(pa) == len(pf) and len(da) == len(df)):
        raise ValueError("validated PDAF IPC smoke requires equal peers per phase")
    bootstrap = ports.allocate(node["name"])
    mb = int(point.get("micro_batch", 2))
    common_flags = (
        f"--afd-comm-backend ipc_cpp --afd-micro-batch {mb} "
        "--disaggregation-transfer-backend mooncake "
        f"--disaggregation-bootstrap-port {bootstrap}"
    )
    envbase = (
        "AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc "
        "SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=128 AFD_ASYNC_PIPELINE=1 "
        "AFD_IPC_SYNC_MODE=ipc_event"
    )
    processes = []
    for role, group, peer, mode, perspective, base in [
        ("PF", pf, len(pa), "prefill", "ffn", 28200),
        ("PA", pa, -len(pf), "prefill", "attn", 28200),
        ("DF", df, len(da), "decode", "ffn", 28300),
        ("DA", da, -len(df), "decode", "attn", 28300),
    ]:
        env = (
            f"{envbase} AFD_UCX_BASE_PORT={base} AFD_SCHED_PORT={base + 40200} "
            f"AFD_IPC_PEER_OFFSET={peer}"
        )
        if perspective == "attn":
            env += " AFD_UCX_FFN_HOST=127.0.0.1"
        extra = (
            f"--afd-perspective {perspective} --disaggregation-mode {mode} "
            f"--base-gpu-id {group[0].gpu} {common_flags}"
        )
        processes.append(context.server(
            role, node, group, ports.allocate(node["name"]), model, len(group),
            extra, env, bootstrap_port=bootstrap if role == "PA" else None,
            visible_gpus=node["gpus"], health_path="/get_model_info",
            startup_stage={"PF": 0, "PA": 1, "DF": 1, "DA": 2}[role],
        ))
    router_port = ports.allocate(node["name"])
    pa_process = next(p for p in processes if p.role == "PA")
    da_process = next(p for p in processes if p.role == "DA")
    args = (
        f"--pd-disaggregation --mini-lb --prefill http://{pa_process.host}:"
        f"{pa_process.port} {bootstrap} --decode http://{da_process.host}:{da_process.port}"
    )
    router = context.router("PDAF_NODE_ROUTER", node, router_port, args,
                         startup_stage=3)
    return processes + [router], router


def build(cluster, model, point, ports: PortAllocator):
    context = PlanContext(cluster)
    count = int(point["nodes"])
    topology = point.get("topology")
    if count != 1 and topology != "per_node_replicas":
        raise ValueError("multi-node PDAF pool topology is blocked pending preflight")
    nodes = node_map(cluster)
    refs = select_gpus(cluster, count)
    per_chain = sum(int(point[k]) for k in ("pf_gpus", "pa_gpus", "df_gpus", "da_gpus"))
    if topology == "per_node_replicas":
        if per_chain != len(cluster["nodes"][0]["gpus"]):
            raise ValueError("per_node_replicas PDAF must consume one complete node per chain")
        groups = partition(refs, [per_chain] * count)
    else:
        groups = [refs]
    server_processes = []
    node_routers = []
    for group in groups:
        if len({ref.node for ref in group}) != 1:
            raise ValueError("PDAF chain crosses node")
        chain, router = _build_chain(context, nodes[group[0].node], group, model, point, ports)
        server_processes.extend(chain[:-1])
        node_routers.append(router)
    processes = server_processes + node_routers
    if len(node_routers) == 1:
        node_routers[0].role = "PDAF_ROUTER"
        endpoint = f"http://{node_routers[0].host}:{node_routers[0].port}"
    else:
        owner = cluster["nodes"][0]
        top_port = ports.allocate(owner["name"])
        urls = " ".join(f"http://{p.host}:{p.port}" for p in node_routers)
        processes.append(context.router(
            "PDAF_TOP_ROUTER", owner, top_port,
            f"--policy round_robin --worker-urls {urls}",
            startup_stage=4,
        ))
        endpoint = f"http://{owner['host']}:{top_port}"
    return context.finish("pdaf", processes, endpoint)
