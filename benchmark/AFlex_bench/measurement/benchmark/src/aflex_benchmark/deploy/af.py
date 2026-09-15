from __future__ import annotations
from .planner_common import PlanContext, node_map, normalize_extra_env
from ..topology import PortAllocator, partition, select_gpus


def _build_replica(context, node, refs, model, point, ports, replica_id):
    fg, ag = partition(
        refs, [int(point["ffn_gpus"]), int(point["attention_gpus"])]
    )
    if len(ag) != len(fg):
        raise ValueError("validated AF IPC smoke requires equal A/F TP")
    tp = len(ag)
    sched_port = 68400 + replica_id * 10
    channel_base = 400 + replica_id * 10
    f_ucx_port = 28100 + replica_id * 200
    a_ucx_port = 28200 + replica_id * 200
    point_extra_env = normalize_extra_env(point.get("extra_env"))
    common = (
        "AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc "
        "SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=128 AFD_ASYNC_PIPELINE=1 "
        f"AFD_IPC_SYNC_MODE=ipc_event AFD_IPC_CHANNEL_BASE={channel_base} "
        f"AFD_SCHED_PORT={sched_port}"
    )
    fenv = (
        f"{common} AFD_UCX_BASE_PORT={f_ucx_port} AFD_IPC_PEER_OFFSET={tp} "
        f"{point_extra_env}"
    ).strip()
    aenv = (
        f"{common} AFD_UCX_BASE_PORT={a_ucx_port} "
        f"AFD_IPC_PEER_OFFSET=-{tp} AFD_UCX_FFN_HOST=127.0.0.1 "
        f"{point_extra_env}"
    )
    flags = f"--afd-comm-backend ipc_cpp --afd-micro-batch {int(point.get('micro_batch', 1))}"
    ffn = context.server(
        "F", node, fg, ports.allocate(node["name"]), model, tp,
        f"--afd-perspective ffn --base-gpu-id {fg[0].gpu} {flags}", fenv,
        visible_gpus=node["gpus"], health_path="/get_model_info",
    )
    ffn.internal_ports = [f_ucx_port, sched_port]
    ffn.metadata = {"replica": replica_id, "channel_strategy": "shared_tp0",
                    "channel_base": channel_base, "channel_id": None,
                    "channels": []}
    attn = context.server(
        "A", node, ag, ports.allocate(node["name"]), model, tp,
        f"--afd-perspective attn --base-gpu-id {ag[0].gpu} {flags}", aenv,
        visible_gpus=node["gpus"], health_path="/get_model_info",
        startup_stage=1,
    )
    attn.internal_ports = [a_ucx_port]
    attn.metadata = {"replica": replica_id, "channel_strategy": "shared_tp0",
                     "channel_base": channel_base, "channel_id": None,
                     "channels": []}
    return [ffn, attn]


def build(cluster, model, point, ports: PortAllocator):
    context = PlanContext(cluster)
    count = int(point["nodes"])
    topology = point.get("topology")
    if count != 1 and topology != "per_node_replicas":
        raise ValueError("multi-node AF is blocked pending a validated UCX topology")
    nodes = node_map(cluster)
    refs = select_gpus(cluster, count)
    per_replica = int(point["ffn_gpus"]) + int(point["attention_gpus"])
    replicas = int(point.get("replicas", count if topology == "per_node_replicas" else 1))
    if replicas < 1:
        raise ValueError("AF replicas must be positive")
    required = per_replica * replicas
    if required > len(refs):
        raise ValueError("AF point requests more GPUs than selected nodes provide")
    refs = refs[:required]
    if topology == "per_node_replicas":
        if replicas != count or per_replica != len(cluster["nodes"][0]["gpus"]):
            raise ValueError("per_node_replicas AF must consume one complete node per replica")
    groups = partition(refs, [per_replica] * replicas)
    processes = []
    attn_workers = []
    for replica_id, group in enumerate(groups):
        if len({ref.node for ref in group}) != 1:
            raise ValueError("AF replica crosses node")
        replica = _build_replica(
            context, nodes[group[0].node], group, model, point, ports, replica_id
        )
        processes.extend(replica)
        attn_workers.append(replica[-1])
    if len(attn_workers) == 1:
        endpoint = f"http://{attn_workers[0].host}:{attn_workers[0].port}"
        return context.finish("af", processes, endpoint)
    if point.get("client_round_robin", False):
        endpoints = [f"http://{process.host}:{process.port}" for process in attn_workers]
        return context.finish(
            "af", processes, endpoints, routing_policy="client_round_robin"
        )

    owner = cluster["nodes"][0]
    router_port = ports.allocate(owner["name"])
    urls = " ".join(f"http://{p.host}:{p.port}" for p in attn_workers)
    processes.append(context.router(
        "AF_ROUTER", owner, router_port,
        f"--policy round_robin --worker-urls {urls}",
        startup_stage=2,
    ))
    endpoint = f"http://{owner['host']}:{router_port}"
    return context.finish("af", processes, endpoint)
