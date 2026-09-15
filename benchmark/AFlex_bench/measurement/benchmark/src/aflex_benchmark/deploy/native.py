from __future__ import annotations
from .planner_common import PlanContext, node_map
from ..topology import PortAllocator, partition, select_gpus

def build(cluster, model, point, ports: PortAllocator):
    context = PlanContext(cluster)
    refs = select_gpus(cluster, point["nodes"])
    tp = int(point["tp"])
    reps = int(point.get("replicas", len(refs) // tp))
    groups = partition(refs, [tp] * reps)
    nodes = node_map(cluster)
    processes = []
    for group in groups:
        if len({x.node for x in group}) != 1:
            raise ValueError("TP group crosses node")
        node = nodes[group[0].node]
        processes.append(context.server("native", node, group,
                                     ports.allocate(node["name"]), model, tp))
    owner = cluster["nodes"][0]
    router_port = ports.allocate(owner["name"])
    urls = " ".join(f"http://{p.host}:{p.port}" for p in processes)
    processes.append(context.router("NATIVE_ROUTER", owner, router_port,
                                 f"--policy round_robin --worker-urls {urls}",
                                 startup_stage=1))
    return context.finish("native", processes, f"http://{owner['host']}:{router_port}")
