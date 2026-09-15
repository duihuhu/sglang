from __future__ import annotations
from .planner_common import PlanContext, node_map
from ..topology import PortAllocator, partition, select_gpus

def build(cluster, model, point, ports: PortAllocator):
    context = PlanContext(cluster)
    refs = select_gpus(cluster, point["nodes"])
    pr, dr = int(point["prefill_replicas"]), int(point["decode_replicas"])
    pt, dt = int(point["prefill_tp"]), int(point["decode_tp"])
    groups = partition(refs, [pt] * pr + [dt] * dr)
    nodes = node_map(cluster)
    p_groups, d_groups = groups[:pr], groups[pr:]
    processes = []
    prefills = []
    for group in p_groups:
        if len({x.node for x in group}) != 1:
            raise ValueError("PD TP group crosses node")
        node = nodes[group[0].node]
        bootstrap = ports.allocate(node["name"])
        extra = ("--disaggregation-mode prefill --disaggregation-transfer-backend mooncake "
                 f"--disaggregation-bootstrap-port {bootstrap}")
        process = context.server("P", node, group, ports.allocate(node["name"]),
                              model, len(group), extra, bootstrap_port=bootstrap)
        processes.append(process)
        prefills.append(process)
    for index, group in enumerate(d_groups):
        if len({x.node for x in group}) != 1:
            raise ValueError("PD TP group crosses node")
        node = nodes[group[0].node]
        bootstrap = prefills[index % len(prefills)].bootstrap_port
        extra = ("--disaggregation-mode decode --disaggregation-transfer-backend mooncake "
                 f"--disaggregation-bootstrap-port {bootstrap}")
        processes.append(context.server("D", node, group, ports.allocate(node["name"]),
                                     model, len(group), extra, startup_stage=1))
    owner = cluster["nodes"][0]
    router = ports.allocate(owner["name"])
    args = ["--pd-disaggregation", "--mini-lb"]
    args += [f"--prefill http://{p.host}:{p.port} {p.bootstrap_port}" for p in prefills]
    args += [f"--decode http://{p.host}:{p.port}" for p in processes if p.role == "D"]
    processes.append(context.router("PD_ROUTER", owner, router, " ".join(args),
                                 startup_stage=2))
    return context.finish("pd", processes, f"http://{owner['host']}:{router}")
