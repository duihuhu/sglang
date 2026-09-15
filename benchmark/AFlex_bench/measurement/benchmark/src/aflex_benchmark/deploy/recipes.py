from __future__ import annotations

import json
import shlex

"""Adapters for deployment shapes proven by the legacy AFlex scripts."""

from .planner_common import PlanContext, process_paths
from .base import ProcessSpec

COMMON_FLAGS = (
    "--mem-fraction-static 0.85 --max-running-requests 512 "
    "--disable-cuda-graph --disable-piecewise-cuda-graph --skip-server-warmup "
    "--disable-radix-cache --watchdog-timeout 900"
)
AF_SERVER_FLAGS = (
    "--afd-comm-backend ipc_cpp --afd-micro-batch 1 "
    "--mem-fraction-static 0.85 --max-running-requests 512 "
    "--watchdog-timeout 600"
)
PYTHON = "/usr/bin/python3"
PROM_BASE = 29100
ROUTER_PORT = 48000
NCCL_PORT_BASE = 37300


def _refs(node, gpus):
    from ..topology import GPURef
    return [GPURef(node["name"], gpu) for gpu in gpus]


def _nic(node, gpu, cluster):
    return str(node.get("gpu_nic", {}).get(str(gpu), cluster.get("ib_device", "mlx5_bond_0")))


def _server(context, role, node, gpus, port, model, tp, *, extra="", env="", nccl=None,
            bootstrap=None, internal=(), health="/health", visible=None, metadata=None,
            common_flags=COMMON_FLAGS, startup_stage=0):
    spec = context.server(role, node, _refs(node, gpus), port, model, tp,
                       f"--nccl-port {nccl} {common_flags} {extra}" if nccl else f"{common_flags} {extra}",
                       env, bootstrap_port=bootstrap, visible_gpus=visible,
                       health_path=health, startup_stage=startup_stage)
    spec.nccl_port = nccl
    spec.internal_ports = list(internal)
    spec.metadata = metadata or {}
    return spec


def _router(context, role, owner, port, args, *, prom, metadata=None, startup_stage=0):
    spec = context.router(role, owner, port, f"{args} --prometheus-port {prom}",
                       startup_stage=startup_stage)
    spec.internal_ports = [prom]
    spec.metadata = metadata or {}
    return spec


def legacy_native_tp(cluster, model, point):
    context = PlanContext(cluster)
    nodes = cluster["nodes"][:4]
    if len(nodes) != 4 or any(len(n["gpus"]) != 8 for n in nodes):
        raise ValueError("legacy_native_tp requires four 8-GPU nodes")
    processes = []
    for i, node in enumerate(nodes):
        processes.append(_server(context,
            "NATIVE", node, list(node["gpus"]), 53400 + i * 10, model, 8,
            nccl=33500 + i * 10,
            env=(f"UCX_LOG_LEVEL=fatal AFD_NVML_DEVICE_INDEX=0 "
                 f"AFD_NVML_DEVICE_INDICES={','.join(map(str, node['gpus']))} "
                 f"SGLANG_HOST_IP={node['host']}"),
            metadata={"recipe": "legacy_native_tp", "replica": i, "tp": 8},
        ))
    workers = " ".join(f"http://{p.host}:{p.port}" for p in processes)
    processes.append(_router(context, "NATIVE_ROUTER", nodes[0], ROUTER_PORT,
                             f"--policy round_robin --worker-urls {workers}",
                             prom=PROM_BASE + 30,
                             metadata={"level": "top", "policy": "round_robin"},
                             startup_stage=1))
    return context.finish("native", processes, f"http://{nodes[0]['host']}:{ROUTER_PORT}")


def legacy_pd_dual(cluster, model, point):
    context = PlanContext(cluster)
    nodes = cluster["nodes"][:4]
    if len(nodes) != 4:
        raise ValueError("legacy_pd_dual requires four nodes")
    processes, subrouters = [], []
    for inst, (pnode, dnode, subport) in enumerate(((nodes[0], nodes[1], 48001), (nodes[2], nodes[3], 48002))):
        prefills, decodes = [], []
        for i, gpus in enumerate(([0,1,2,3], [4,5,6,7])):
            port, bootstrap, nccl = 53100 + inst*100 + i*10, 49100 + inst*10 + i, 34000 + inst*100 + i*10
            extra = ("--disaggregation-mode prefill --disaggregation-transfer-backend mooncake "
                     f"--disaggregation-bootstrap-port {bootstrap} --disaggregation-ib-device {_nic(pnode, gpus[0], cluster)}")
            proc = _server(context, "P", pnode, gpus, port, model, 4, extra=extra, nccl=nccl,
                           bootstrap=bootstrap,
                           env=f"UCX_LOG_LEVEL=fatal SGLANG_HOST_IP={pnode['host']}",
                           metadata={"recipe":"legacy_pd_dual","cluster":inst,"index":i,"tp":4})
            processes.append(proc); prefills.append(proc)
        for i, gpus in enumerate(([0,1], [2,3], [4,5], [6,7])):
            port, nccl = 53150 + inst*100 + i*10, 34050 + inst*100 + i*10
            bootstrap = prefills[0].bootstrap_port
            extra = ("--disaggregation-mode decode --disaggregation-transfer-backend mooncake "
                     f"--disaggregation-bootstrap-port {bootstrap} --disaggregation-ib-device {_nic(dnode, gpus[0], cluster)}")
            proc = _server(context, "D", dnode, gpus, port, model, 2, extra=extra, nccl=nccl,
                           env=f"UCX_LOG_LEVEL=fatal SGLANG_HOST_IP={dnode['host']}",
                           metadata={"recipe":"legacy_pd_dual","cluster":inst,"index":i,"tp":2},
                           startup_stage=1)
            processes.append(proc); decodes.append(proc)
        args = ["--pd-disaggregation"]
        args += [f"--prefill http://{p.host}:{p.port} {p.bootstrap_port}" for p in prefills]
        args += [f"--decode http://{d.host}:{d.port}" for d in decodes]
        router = _router(context, "PD_SUBROUTER", pnode, subport, " ".join(args),
                         prom=PROM_BASE + inst,
                         metadata={"level":"sub","cluster":inst,"activation_warmup":True},
                         startup_stage=2)
        processes.append(router); subrouters.append(router)
    workers = " ".join(f"http://{r.host}:{r.port}" for r in subrouters)
    processes.append(_router(context, "PD_TOP_ROUTER", nodes[0], ROUTER_PORT,
                             f"--policy round_robin --worker-urls {workers}",
                             prom=PROM_BASE + 10, metadata={"level":"top"},
                             startup_stage=3))
    return context.finish("pd", processes, f"http://{nodes[0]['host']}:{ROUTER_PORT}")


def legacy_af_profile_replicas(cluster, model, point):
    context = PlanContext(cluster)
    nodes = cluster["nodes"][:4]
    if len(nodes) != 4 or any(len(node["gpus"]) != 8 for node in nodes):
        raise ValueError("legacy_af_profile_replicas requires four 8-GPU nodes")
    processes, attns = [], []
    pair_tp = int(point.get("pair_tp", 1))
    if pair_tp not in {1, 2}:
        raise ValueError("legacy_af_profile_replicas pair_tp must be 1 or 2")
    if pair_tp == 2 and not point.get("experimental"):
        raise ValueError("legacy_af_profile_replicas pair_tp=2 must be experimental")
    replicas_per_node = 8 // (pair_tp * 2)
    if int(point.get("replicas_per_node", replicas_per_node)) != replicas_per_node:
        raise ValueError("legacy_af_profile_replicas replicas_per_node does not match pair_tp")
    if any(int(point.get(key, pair_tp)) != pair_tp
           for key in ("attention_gpus", "ffn_gpus")):
        raise ValueError("legacy_af_profile_replicas attention_gpus/ffn_gpus must equal pair_tp")
    channel_strategy = point.get("ipc_channel_strategy", "shared_tp0")
    if channel_strategy not in {"shared_tp0", "per_rank"}:
        raise ValueError(
            "legacy_af_profile_replicas ipc_channel_strategy must be "
            "'shared_tp0' or 'per_rank'"
        )
    per_rank = channel_strategy == "per_rank"
    af_flags = AF_SERVER_FLAGS + ("--afd-ipc-per-rank " if per_rank else "")
    pair_width = pair_tp * 2
    for node_i, node in enumerate(nodes):
        for replica, base in enumerate(range(0, 8, pair_width)):
            rid = node_i * replicas_per_node + replica
            fg = list(range(base, base + pair_tp))
            ag = list(range(base + pair_tp, base + pair_width))
            f_ucx, a_ucx = 28100 + rid*200, 28200 + rid*200
            f_sched = 42000 + rid*20
            a_sched = f_sched + 1 if per_rank else f_sched
            visible = fg + ag
            channel_base = 400 + rid * 10
            common = (f"AFD_IPC_SYNC_MODE=ipc_event AFD_ASYNC_PIPELINE=1 "
                      f"AFD_IPC_CHANNEL_BASE={channel_base}")
            fenv = (f"{common} AFD_UCX_BASE_PORT={f_ucx} AFD_SCHED_PORT={f_sched} "
                    f"AFD_IPC_PEER_OFFSET={pair_tp} AFD_NVML_DEVICE_INDICES={','.join(map(str, fg))} AFD_NVML_DEVICE_INDEX={fg[0]}")
            aenv = (f"{common} AFD_UCX_BASE_PORT={a_ucx} AFD_SCHED_PORT={a_sched} "
                    f"AFD_IPC_PEER_OFFSET=-{pair_tp} AFD_NVML_DEVICE_INDICES={','.join(map(str, ag))} AFD_NVML_DEVICE_INDEX={ag[0]} "
                    "AFD_UCX_FFN_HOST=127.0.0.1")
            fport, aport = 40100 + rid*10, 40101 + rid*10
            channel_id = channel_base if per_rank else None
            metadata = {"recipe":"legacy_af_profile_replicas","replica":rid,"tp":pair_tp,
                        "channel_strategy":channel_strategy,"channel_id":channel_id,
                        "channel_base":channel_base,
                        "channels":[channel_base, channel_base + 1] if per_rank else []}
            f = _server(context, "F", node, fg, fport, model, pair_tp,
                        extra="--afd-perspective ffn --base-gpu-id 0",
                        env=fenv, common_flags=af_flags, nccl=36100+rid*10,
                        internal=(f_ucx,f_sched),
                        health="/get_model_info", visible=visible,
                        metadata=dict(metadata))
            a = _server(context, "A", node, ag, aport, model, pair_tp,
                        extra=f"--afd-perspective attn --base-gpu-id {pair_tp}",
                        env=aenv, common_flags=af_flags, nccl=36101+rid*10,
                        internal=(a_ucx,a_sched) if per_rank else (a_ucx,),
                        health="/get_model_info", visible=visible,
                        metadata=dict(metadata), startup_stage=1)
            processes.extend((f,a)); attns.append(a)
    endpoints = [f"http://{a.host}:{a.port}" for a in attns]
    return context.finish("af", processes, endpoints,
                          routing_policy="client_round_robin")


def legacy_pdaf_3p1d(cluster, model, point):
    context = PlanContext(cluster)
    nodes = cluster["nodes"][:4]
    processes, node_routers = [], []
    for ni, node in enumerate(nodes):
        server_base = 48100 + ni*200
        endpoints = {}
        pairs = [("prefill", i, i*2, i*2+1) for i in range(3)] + [("decode", 0, 6, 7)]
        for slot, (phase, idx, fg, ag) in enumerate(pairs):
            aport, bootstrap, fport = server_base+slot*3, server_base+slot*3+1, server_base+slot*3+2
            ucx, sched = 49200+ni*1000+slot*20, 49300+ni*1000+slot*20
            base_extra = (f"--afd-comm-backend ipc_cpp --afd-micro-batch 1 --afd-attn-tp 1 --afd-ffn-tp 1 "
                          f"--disaggregation-mode {phase} --disaggregation-transfer-backend mooncake "
                          f"--disaggregation-bootstrap-port {bootstrap} --disaggregation-ib-device {cluster.get('ib_device','mlx5_bond_0')} "
                          "--afd-disagg-interleave-poll --num-reserved-decode-tokens 512")
            channel_base = 600 + (ni * len(pairs) + slot) * 10
            envbase = (f"AFD_IPC_SYNC_MODE=ipc_event AFD_ASYNC_PIPELINE=1 AFD_UCX_FFN_HOST=127.0.0.1 "
                       f"AFD_IPC_CHANNEL_BASE={channel_base} AFD_UCX_BASE_PORT={ucx} "
                       f"AFD_SCHED_PORT={sched} SGLANG_HOST_IP={node['host']}")
            meta={"recipe":"legacy_pdaf_3p1d","phase":phase,"pair":idx,"node_index":ni,"tp":1,
                  "channel_base":channel_base}
            f = _server(context, "PF" if phase=="prefill" else "DF", node, [fg], fport, model, 1,
                        extra=f"--afd-perspective ffn --base-gpu-id 0 {base_extra}",
                        env=f"{envbase} AFD_IPC_PEER_DEVICE=1 AFD_NVML_DEVICE_INDEX={fg} AFD_NVML_DEVICE_INDICES={fg}",
                        nccl=49400+ni*1000+slot*20, internal=(ucx,sched),
                        health="/get_model_info", visible=[fg,ag], metadata=meta,
                        startup_stage=slot * 2)
            a = _server(context, "PA" if phase=="prefill" else "DA", node, [ag], aport, model, 1,
                        extra=f"--afd-perspective attn --base-gpu-id 1 {base_extra}",
                        env=f"{envbase} AFD_IPC_PEER_DEVICE=0 AFD_NVML_DEVICE_INDEX={ag} AFD_NVML_DEVICE_INDICES={ag}",
                        nccl=49401+ni*1000+slot*20, internal=(), bootstrap=bootstrap if phase=="prefill" else None,
                        health="/get_model_info", visible=[fg,ag], metadata=meta,
                        startup_stage=slot * 2 + 1)
            processes.extend((f,a)); endpoints[(phase,idx)] = (a,bootstrap)
        subrouters=[]
        decode=endpoints[("decode",0)][0]
        for i in range(3):
            prefill, bootstrap=endpoints[("prefill",i)]
            sp=ROUTER_PORT+1+i
            router=_router(context, "PDAF_SUBROUTER", node, sp,
                           f"--pd-disaggregation --mini-lb --prefill http://{node['host']}:{prefill.port} {bootstrap} --decode http://{node['host']}:{decode.port}",
                           prom=PROM_BASE+ni*10+i,
                           metadata={"level":"sub","node_index":ni,"prefill":i,"activation_warmup":True},
                           startup_stage=8)
            processes.append(router); subrouters.append(router)
        workers=" ".join(f"http://{node['host']}:{r.port}" for r in subrouters)
        nr=_router(context, "PDAF_NODE_ROUTER", node, ROUTER_PORT,
                   f"--policy round_robin --worker-urls {workers}",
                   prom=PROM_BASE+ni*10+5, metadata={"level":"node","node_index":ni,"activation_warmup":True},
                   startup_stage=9)
        processes.append(nr); node_routers.append(nr)
    workers=" ".join(f"http://{r.host}:{r.port}" for r in node_routers)
    processes.append(_router(context, "PDAF_TOP_ROUTER", nodes[0], 48050,
                             f"--policy round_robin --worker-urls {workers}",
                             prom=PROM_BASE+50, metadata={"level":"top"},
                             startup_stage=10))
    return context.finish("pdaf", processes, f"http://{nodes[0]['host']}:48050")



def _two_8gpu_nodes(cluster, recipe):
    nodes = cluster["nodes"]
    if len(nodes) != 2 or any(len(node["gpus"]) != 8 for node in nodes):
        raise ValueError(f"{recipe} requires exactly two 8-GPU nodes")
    return nodes


def legacy_native_pair_tp8(cluster, model, point):
    context = PlanContext(cluster)
    nodes = _two_8gpu_nodes(cluster, "legacy_native_pair_tp8")
    processes = []
    for index, node in enumerate(nodes):
        gpus = list(node["gpus"])
        processes.append(_server(
            context, "NATIVE", node, gpus, 53400 + index * 10, model, 8,
            nccl=33500 + index * 10,
            env=(f"UCX_LOG_LEVEL=fatal AFD_NVML_DEVICE_INDEX={gpus[0]} "
                 f"AFD_NVML_DEVICE_INDICES={','.join(map(str, gpus))} "
                 f"SGLANG_HOST_IP={node['host']}"),
            metadata={"recipe": "legacy_native_pair_tp8", "replica": index, "tp": 8},
        ))
    workers = " ".join(f"http://{process.host}:{process.port}" for process in processes)
    processes.append(_router(
        context, "NATIVE_ROUTER", nodes[0], ROUTER_PORT,
        f"--policy round_robin --worker-urls {workers}", prom=PROM_BASE + 30,
        metadata={"level": "top", "policy": "round_robin"}, startup_stage=1,
    ))
    return context.finish("native", processes, f"http://{nodes[0]['host']}:{ROUTER_PORT}")


def legacy_pd_xnode_16(cluster, model, point):
    context = PlanContext(cluster)
    pnode, dnode = _two_8gpu_nodes(cluster, "legacy_pd_xnode_16")
    p_groups = [list(pnode["gpus"][0:4]), list(pnode["gpus"][4:8])]
    d_groups = [list(dnode["gpus"][i:i + 2]) for i in range(0, 8, 2)]
    processes, prefills, decodes = [], [], []
    for index, gpus in enumerate(p_groups):
        bootstrap = 49100 + index
        extra = (
            "--disaggregation-mode prefill --disaggregation-transfer-backend mooncake "
            f"--disaggregation-bootstrap-port {bootstrap} "
            f"--disaggregation-ib-device {_nic(pnode, gpus[0], cluster)}"
        )
        process = _server(
            context, "P", pnode, gpus, 53100 + index * 10, model, 4,
            extra=extra, nccl=34000 + index * 10, bootstrap=bootstrap,
            env=f"UCX_LOG_LEVEL=fatal SGLANG_HOST_IP={pnode['host']}",
            metadata={"recipe": "legacy_pd_xnode_16", "index": index, "tp": 4},
        )
        processes.append(process); prefills.append(process)
    for index, gpus in enumerate(d_groups):
        bootstrap = prefills[0].bootstrap_port
        extra = (
            "--disaggregation-mode decode --disaggregation-transfer-backend mooncake "
            f"--disaggregation-bootstrap-port {bootstrap} "
            f"--disaggregation-ib-device {_nic(dnode, gpus[0], cluster)}"
        )
        process = _server(
            context, "D", dnode, gpus, 53150 + index * 10, model, 2,
            extra=extra, nccl=34050 + index * 10,
            env=f"UCX_LOG_LEVEL=fatal SGLANG_HOST_IP={dnode['host']}",
            metadata={"recipe": "legacy_pd_xnode_16", "index": index, "tp": 2},
            startup_stage=1,
        )
        processes.append(process); decodes.append(process)
    args = ["--pd-disaggregation"]
    args += [f"--prefill http://{p.host}:{p.port} {p.bootstrap_port}" for p in prefills]
    args += [f"--decode http://{d.host}:{d.port}" for d in decodes]
    processes.append(_router(
        context, "PD_ROUTER", pnode, ROUTER_PORT, " ".join(args), prom=PROM_BASE,
        metadata={"level": "top", "activation_warmup": True}, startup_stage=2,
    ))
    return context.finish("pd", processes, f"http://{pnode['host']}:{ROUTER_PORT}")


def af_node34_a1f1_pool(cluster, model, point):
    context = PlanContext(cluster)
    nodes = _two_8gpu_nodes(cluster, "af_node34_a1f1_pool")
    processes, attns = [], []
    for node_index, node in enumerate(nodes):
        gpus = list(node["gpus"])
        for local_replica in range(4):
            replica = node_index * 4 + local_replica
            fg, ag = [gpus[local_replica * 2]], [gpus[local_replica * 2 + 1]]
            visible = fg + ag
            channel_base = 400 + replica * 10
            sched = 42000 + replica * 20
            f_ucx, a_ucx = 28100 + replica * 200, 28200 + replica * 200
            common = (f"AFD_IPC_SYNC_MODE=ipc_event AFD_ASYNC_PIPELINE=1 "
                      f"AFD_IPC_CHANNEL_BASE={channel_base} AFD_SCHED_PORT={sched}")
            metadata = {"recipe": "af_node34_a1f1_pool", "replica": replica,
                        "tp": 1, "channel_base": channel_base}
            f = _server(
                context, "F", node, fg, 40100 + replica * 10, model, 1,
                extra="--afd-perspective ffn --base-gpu-id 0",
                env=(f"{common} AFD_UCX_BASE_PORT={f_ucx} AFD_IPC_PEER_OFFSET=1 "
                     f"AFD_NVML_DEVICE_INDICES={fg[0]} AFD_NVML_DEVICE_INDEX={fg[0]}"),
                nccl=36100 + replica * 10, internal=(f_ucx, sched),
                health="/get_model_info", visible=visible, metadata=dict(metadata),
                common_flags=AF_SERVER_FLAGS,
            )
            a = _server(
                context, "A", node, ag, 40101 + replica * 10, model, 1,
                extra="--afd-perspective attn --base-gpu-id 1",
                env=(f"{common} AFD_UCX_BASE_PORT={a_ucx} AFD_IPC_PEER_OFFSET=-1 "
                     f"AFD_NVML_DEVICE_INDICES={ag[0]} AFD_NVML_DEVICE_INDEX={ag[0]} "
                     "AFD_UCX_FFN_HOST=127.0.0.1"),
                nccl=36101 + replica * 10, internal=(a_ucx,),
                health="/get_model_info", visible=visible, metadata=dict(metadata),
                common_flags=AF_SERVER_FLAGS, startup_stage=1,
            )
            processes.extend((f, a)); attns.append(a)
    endpoints = [f"http://{process.host}:{process.port}" for process in attns]
    return context.finish("af", processes, endpoints, routing_policy="client_round_robin")


def legacy_pdaf_xnode_tp4(cluster, model, point):
    context = PlanContext(cluster)
    pnode, dnode = _two_8gpu_nodes(cluster, "legacy_pdaf_xnode_tp4")
    expected = list(range(8))
    if list(pnode["gpus"]) != expected or list(dnode["gpus"]) != expected:
        raise ValueError("legacy_pdaf_xnode_tp4 requires GPU IDs 0..7 on both nodes")
    attn_gpus, ffn_gpus = [0, 2, 4, 6], [1, 3, 5, 7]
    bootstrap = 49990
    ib_device = cluster.get("ib_device", "mlx5_bond_0")
    common_flags = (
        "--afd-comm-backend ipc_cpp --afd-micro-batch 2 --gpu-id-step 2 "
        "--mem-fraction-static 0.85 --max-running-requests 512 "
        "--watchdog-timeout 600 --afd-disagg-interleave-poll "
        "--num-reserved-decode-tokens 512 --disaggregation-transfer-backend mooncake "
        f"--disaggregation-bootstrap-port {bootstrap} --disaggregation-ib-device {ib_device} "
        "--enable-metrics"
    )
    processes = []
    roles = [
        ("PF", pnode, ffn_gpus, "ffn", "prefill", 1, -1, 48101, 34000, 28200, 68400, 0),
        ("PA", pnode, attn_gpus, "attn", "prefill", 0, 1, 48100, 34001, 28200, 68400, 1),
        ("DF", dnode, ffn_gpus, "ffn", "decode", 1, -1, 48103, 34050, 28300, 68500, 1),
        ("DA", dnode, attn_gpus, "attn", "decode", 0, 1, 48102, 34051, 28300, 68500, 2),
    ]
    for role, node, gpus, perspective, mode, base_gpu, offset, port, nccl, ucx, sched, stage in roles:
        nvml = ",".join(map(str, gpus))
        env = (
            "UCX_LOG_LEVEL=fatal AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc "
            "SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=128 AFD_ASYNC_PIPELINE=1 "
            "AFD_IPC_SYNC_MODE=ipc_event SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 "
            "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600 "
            "SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600 "
            f"AFD_UCX_BASE_PORT={ucx} AFD_SCHED_PORT={sched} "
            f"AFD_IPC_PEER_OFFSET={offset} AFD_NVML_DEVICE_INDICES={nvml} "
            f"AFD_NVML_DEVICE_INDEX={gpus[0]} SGLANG_HOST_IP={node['host']}"
        )
        if perspective == "attn":
            env += " AFD_UCX_FFN_HOST=127.0.0.1"
        process = _server(
            context, role, node, gpus, port, model, 4,
            extra=(f"--afd-perspective {perspective} --disaggregation-mode {mode} "
                   f"--base-gpu-id {base_gpu}"),
            env=env, nccl=nccl, bootstrap=bootstrap if role == "PA" else None,
            internal=(ucx, sched) if role in {"PF", "DF"} else (),
            health="/get_model_info", visible=list(node["gpus"]),
            metadata={"recipe": "legacy_pdaf_xnode_tp4", "phase": mode,
                      "perspective": perspective, "tp": 4},
            common_flags=common_flags, startup_stage=stage,
        )
        processes.append(process)
    pa = next(process for process in processes if process.role == "PA")
    da = next(process for process in processes if process.role == "DA")
    router_args = (f"--pd-disaggregation --mini-lb --prefill http://{pa.host}:{pa.port} "
                   f"{bootstrap} --decode http://{da.host}:{da.port}")
    processes.append(_router(
        context, "PDAF_ROUTER", pnode, ROUTER_PORT, router_args, prom=PROM_BASE,
        metadata={"level": "top", "activation_warmup": True}, startup_stage=3,
    ))
    return context.finish("pdaf", processes, f"http://{pnode['host']}:{ROUTER_PORT}")


def legacy_tier1_layout(cluster, model, point):
    """Historical Tier1 PDAF layout from bench_tier1_v2.py.

    The dataset/QPS lookup is intentionally local and currently exact only for
    Code QPS16. Explicit k/tp/frequency fields may be supplied for future points.
    """
    context = PlanContext(cluster)
    nodes = _two_8gpu_nodes(cluster, "legacy_tier1_layout")
    if any(list(node["gpus"]) != list(range(8)) for node in nodes):
        raise ValueError("legacy_tier1_layout requires GPU IDs 0..7 on both nodes")

    historical = {
        ("code", 16): dict(k_p=7, k_d=1, tp_pa=1, tp_pf=1,
                           tp_da=1, tp_df=1, f_pa=1410, f_pf=1410,
                           f_da=930, f_df=930, tier=True),
    }
    dataset = str(point.get("layout_dataset", "")).lower()
    qps = point.get("layout_qps")
    resolved = dict(historical.get((dataset, int(qps)) if qps is not None else None, {}))
    for key in ("k_p", "k_d", "tp_pa", "tp_pf", "tp_da", "tp_df",
                "f_pa", "f_pf", "f_da", "f_df", "tier"):
        if key in point:
            resolved[key] = point[key]
    required = ("k_p", "k_d", "tp_pa", "tp_pf", "tp_da", "tp_df",
                "f_pa", "f_pf", "f_da", "f_df")
    missing = [key for key in required if key not in resolved]
    if missing:
        raise ValueError(f"legacy_tier1_layout missing fields: {', '.join(missing)}")
    c = {key: int(resolved[key]) for key in required}
    tier = bool(resolved.get("tier", True))
    total = c["k_p"] * (c["tp_pa"] + c["tp_pf"]) + c["k_d"] * (c["tp_da"] + c["tp_df"])
    if total != 16:
        raise ValueError(f"legacy_tier1_layout currently requires exactly 16 GPUs, got {total}")

    # Exact plan_allocation order: decode first, then prefill; FFN precedes ATTN.
    free = [list(node["gpus"]) for node in nodes]
    decodes = []
    for index in range(c["k_d"]):
        need = c["tp_df"] + c["tp_da"]
        for ni, available in enumerate(free):
            if len(available) >= need:
                chosen = available[:need]
                decodes.append((nodes[ni], chosen[c["tp_df"]:], chosen[:c["tp_df"]]))
                del available[:need]
                break
        else:
            raise ValueError(f"cannot place decode instance {index}")
    prefills = []
    for index in range(c["k_p"]):
        need = c["tp_pf"] + c["tp_pa"]
        for ni, available in enumerate(free):
            if len(available) >= need:
                chosen = available[:need]
                prefills.append((nodes[ni], chosen[c["tp_pf"]:], chosen[:c["tp_pf"]]))
                del available[:need]
                break
        else:
            raise ValueError(f"cannot place prefill instance {index}")

    ib_path = "/tmp/ib_scal_map.json"
    ib_maps = {}
    freq_maps = {node["host"]: {} for node in nodes}
    for node in nodes:
        mapping = {str(gpu): _nic(node, gpu, cluster) for gpu in node["gpus"]}
        ib_maps[node["host"]] = mapping
    for node, attn, ffn in prefills:
        freq_maps[node["host"]].update({gpu: c["f_pa"] for gpu in attn})
        freq_maps[node["host"]].update({gpu: c["f_pf"] for gpu in ffn})
    for node, attn, ffn in decodes:
        freq_maps[node["host"]].update({gpu: c["f_da"] for gpu in attn})
        freq_maps[node["host"]].update({gpu: c["f_df"] for gpu in ffn})

    processes, pa_specs, da_specs = [], [], []
    nccl_index = 0
    source_root = str(cluster.get("python_source", "/workspace/moe-tier/python")).rsplit("/python", 1)[0]
    energy_dir = (f"{source_root}/benchmark/AFlex_bench/energy_model/"
                  "Qwen3-32B/models_v1")
    base_flags = (
        "--afd-comm-backend ipc_cpp --afd-micro-batch 1 "
        "--max-running-requests 64 --watchdog-timeout 600 "
        "--afd-disagg-interleave-poll --num-reserved-decode-tokens 512 "
        "--disaggregation-transfer-backend mooncake "
        f"--disaggregation-ib-device {ib_path} --enable-metrics"
    )
    if tier:
        base_flags += (f" --afd-dvfs-enabled --afd-energy-model-dir {energy_dir} "
                       "--afd-ttft-slo-ms 2000 --afd-tpot-slo-us 100000 "
                       "--afd-dvfs-decode-compositional --afd-dvfs-idle-lock")

    def add_pair(node, attn, ffn, phase, index, attn_port, ffn_port, stage_f, stage_a):
        nonlocal nccl_index
        ucx = 28200 + index * 100
        sched = 68400 + index * 100
        visible = ffn + attn
        common_env = (
            "UCX_LOG_LEVEL=fatal AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc "
            "SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=128 AFD_ASYNC_PIPELINE=1 "
            "AFD_IPC_SYNC_MODE=ipc_event SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 "
            "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600 "
            "SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600 "
            f"AFD_UCX_BASE_PORT={ucx} AFD_SCHED_PORT={sched} SGLANG_HOST_IP={node['host']}"
        )
        is_decode = phase == "decode"
        tp_a, tp_f = ((c["tp_da"], c["tp_df"]) if is_decode
                      else (c["tp_pa"], c["tp_pf"]))
        mem = "0.88" if is_decode else "0.75"
        common = (f"{base_flags} --mem-fraction-static {mem} "
                  f"--afd-attn-tp {tp_a} --afd-ffn-tp {tp_f}")
        meta = {"recipe": "legacy_tier1_layout", "phase": phase,
                "pair": index, "tier": tier, "ib_json_file": ib_path,
                "ib_mapping": ib_maps[node["host"]]}
        f_spec = _server(
            context, "DF" if is_decode else "PF", node, ffn, ffn_port,
            model, tp_f,
            extra=f"--afd-perspective ffn --disaggregation-mode {phase} --base-gpu-id 0",
            env=(f"{common_env} AFD_IPC_PEER_DEVICE={len(ffn)} "
                 f"AFD_NVML_DEVICE_INDICES={','.join(map(str, ffn))} "
                 f"AFD_NVML_DEVICE_INDEX={ffn[0]}"),
            nccl=NCCL_PORT_BASE + nccl_index * 10, internal=(ucx, sched),
            health="/get_model_info", visible=visible, metadata=dict(meta),
            common_flags=common, startup_stage=stage_f)
        nccl_index += 1
        a_spec = _server(
            context, "DA" if is_decode else "PA", node, attn, attn_port,
            model, tp_a,
            extra=(f"--afd-perspective attn --disaggregation-mode {phase} "
                   f"--base-gpu-id {len(ffn)}"),
            env=(f"{common_env} AFD_IPC_PEER_DEVICE=0 "
                 f"AFD_NVML_DEVICE_INDICES={','.join(map(str, attn))} "
                 f"AFD_NVML_DEVICE_INDEX={attn[0]} AFD_UCX_FFN_HOST=127.0.0.1"),
            nccl=NCCL_PORT_BASE + nccl_index * 10,
            bootstrap=attn_port + 1 if not is_decode else None,
            health="/get_model_info", visible=visible, metadata=dict(meta),
            common_flags=common, startup_stage=stage_a)
        nccl_index += 1
        processes.extend((f_spec, a_spec))
        return a_spec

    for index, (node, attn, ffn) in enumerate(prefills):
        pa_specs.append(add_pair(node, attn, ffn, "prefill", index,
                                 43200 + index * 10, 43202 + index * 10, 0, 1))
    for index, (node, attn, ffn) in enumerate(decodes):
        da_specs.append(add_pair(node, attn, ffn, "decode", c["k_p"] + index,
                                 43020 + index * 20, 43022 + index * 20, 1, 2))

    subrouters = []
    for index, pa in enumerate(pa_specs):
        port = 45000 + index
        decode_args = " ".join(f"--decode http://{d.host}:{d.port}" for d in da_specs)
        args = (f"--pd-disaggregation --mini-lb --prefill http://{pa.host}:{pa.port} "
                f"{pa.port + 1} {decode_args}")
        router = _router(context, "PDAF_SUBROUTER", nodes[0], port, args,
                         prom=29200 + index,
                         metadata={"recipe": "legacy_tier1_layout", "level": "sub",
                                   "prefill": index, "decode_pool_size": len(da_specs),
                                   "activation_warmup": True}, startup_stage=3)
        processes.append(router); subrouters.append(router)

    endpoints = [f"http://{router.host}:{router.port}" for router in subrouters]
    plan = context.finish("pdaf", processes, endpoints,
                          routing_policy="client_round_robin")
    pre_actions = []
    for node in nodes:
        mapping = __import__("json").dumps(ib_maps[node["host"]], separators=(",", ":"))
        pre_actions.append((node["host"],
                            f"printf '%s\\n' '{mapping}' > {ib_path}"))
        if tier:
            for gpu, freq in sorted(freq_maps[node["host"]].items()):
                pre_actions.append((node["host"],
                                    f"nvidia-smi -i {gpu} --lock-gpu-clocks={freq},{freq}"))
    plan.pre_actions = pre_actions
    plan.cluster["legacy_tier1"] = {
        "layout_dataset": dataset, "layout_qps": qps, **c, "tier": tier,
        "ib_json_file": ib_path, "ib_maps": ib_maps, "freq_maps": freq_maps,
        "allocation_order": "decode_first_then_prefill",
    }
    plan.validate()
    return plan



def pd_exact_2p2d_tp8(cluster, model, point):
    """Four-node PD baseline: node[0:2] prefill, node[2:4] decode."""
    context = PlanContext(cluster)
    nodes = cluster["nodes"][:4]
    if len(nodes) != 4 or any(len(node["gpus"]) != 8 for node in nodes):
        raise ValueError("pd_exact_2p2d_tp8 requires four 8-GPU nodes")
    processes, prefills, decodes = [], [], []
    for index, node in enumerate(nodes[:2]):
        gpus = list(node["gpus"])
        bootstrap = 49100 + index
        process = _server(
            context, "P", node, gpus, 53100 + index * 10, model, 8,
            extra=("--disaggregation-mode prefill --disaggregation-transfer-backend mooncake "
                   f"--disaggregation-bootstrap-port {bootstrap} "
                   f"--disaggregation-ib-device {_nic(node, gpus[0], cluster)}"),
            env=f"UCX_LOG_LEVEL=fatal SGLANG_HOST_IP={node['host']}",
            nccl=34000, bootstrap=bootstrap,
            metadata={"recipe": "pd_exact_2p2d_tp8", "index": index, "tp": 8,
                      "pool": "prefill"}, startup_stage=0,
        )
        processes.append(process); prefills.append(process)
    for index, node in enumerate(nodes[2:]):
        gpus = list(node["gpus"])
        # The decode pool is repeated on every prefill route by the single router.
        bootstrap = prefills[index % len(prefills)].bootstrap_port
        process = _server(
            context, "D", node, gpus, 53200 + index * 10, model, 8,
            extra=("--disaggregation-mode decode --disaggregation-transfer-backend mooncake "
                   f"--disaggregation-bootstrap-port {bootstrap} "
                   f"--disaggregation-ib-device {_nic(node, gpus[0], cluster)}"),
            env=f"UCX_LOG_LEVEL=fatal SGLANG_HOST_IP={node['host']}",
            nccl=34100,
            metadata={"recipe": "pd_exact_2p2d_tp8", "index": index, "tp": 8,
                      "pool": "decode", "bootstrap_prefill": index % len(prefills)},
            startup_stage=1,
        )
        processes.append(process); decodes.append(process)
    args = ["--pd-disaggregation", "--mini-lb"]
    args += [f"--prefill http://{p.host}:{p.port} {p.bootstrap_port}" for p in prefills]
    args += [f"--decode http://{d.host}:{d.port}" for d in decodes]
    processes.append(_router(
        context, "PD_ROUTER", nodes[0], ROUTER_PORT, " ".join(args),
        prom=PROM_BASE, metadata={"recipe": "pd_exact_2p2d_tp8", "level": "top",
                                  "decode_pool_repeated": True,
                                  "activation_warmup": True}, startup_stage=2,
    ))
    return context.finish("pd", processes, f"http://{nodes[0]['host']}:{ROUTER_PORT}")


def _cross_zmq_pair_specs(context, model, *, attn_node, ffn_node, phase,
                          attn_role, ffn_role, attn_port, ffn_port,
                          ffn_base, attn_base, sched_port, nccl_port,
                          bootstrap, recipe, require_marker=True):
    if len(attn_node["gpus"]) != 8 or len(ffn_node["gpus"]) != 8:
        raise ValueError(f"{recipe} requires 8 GPUs for every A/F role")
    tp = 8
    common_flags = (
        "--afd-comm-backend zmq --afd-micro-batch 1 --afd-attn-tp 8 --afd-ffn-tp 8 "
        "--mem-fraction-static 0.85 --max-running-requests 512 --watchdog-timeout 900 "
        "--afd-disagg-interleave-poll --num-reserved-decode-tokens 512 "
        "--disaggregation-transfer-backend mooncake "
        f"--disaggregation-bootstrap-port {bootstrap} "
        "--disable-cuda-graph --disable-piecewise-cuda-graph --disable-radix-cache "
        "--enable-metrics"
    )
    handshake_ffn = 60000 if phase == "prefill" else 62000
    handshake_attn = 61000 if phase == "prefill" else 63000
    def make(role, node, peer, perspective, port, stage):
        gpus = list(node["gpus"])
        sched_host = ffn_node["host"] if perspective == "attn" else "0.0.0.0"
        env = (
            "UCX_LOG_LEVEL=fatal AFD_LOCAL_TP=8 AFD_CROSS_NODE_EXPERIMENTAL=1 "
            "AFD_ZMQ_SHARDING=0 AFD_ZMQ_DOUBLE_BUFFER=0 AFD_ZMQ_TIMEOUT_MS=300000 "
            "AFD_ZMQ_HANDSHAKE_TIMEOUT_MS=300000 "
            f"AFD_ZMQ_PEER_HOST={peer['host']} AFD_FFN_BASE_PORT={ffn_base} "
            f"AFD_ATTN_BASE_PORT={attn_base} "
            f"AFD_ZMQ_FFN_HANDSHAKE_BASE_PORT={handshake_ffn} "
            f"AFD_ZMQ_ATTN_HANDSHAKE_BASE_PORT={handshake_attn} "
            f"AFD_SCHED_HOST={sched_host} AFD_SCHED_PORT={sched_port} "
            f"AFD_NVML_DEVICE_INDICES={','.join(map(str, gpus))} "
            f"AFD_NVML_DEVICE_INDEX={gpus[0]} SGLANG_HOST_IP={node['host']} "
            "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 "
            "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600 "
            "SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600"
        )
        listener_data_base = ffn_base if perspective == "attn" else attn_base
        listener_handshake_base = handshake_ffn if perspective == "attn" else handshake_attn
        internal = list(range(listener_data_base + 1, listener_data_base + tp + 1))
        internal += list(range(listener_handshake_base + 1, listener_handshake_base + tp + 1))
        if perspective == "ffn":
            internal.append(sched_port)
        spec = _server(
            context, role, node, gpus, port, model, tp,
            extra=(f"--afd-perspective {perspective} --disaggregation-mode {phase} "
                   f"--base-gpu-id 0 {common_flags} "
                   f"--disaggregation-ib-device {_nic(node, gpus[0], context.cluster)}"),
            env=env, nccl=nccl_port, internal=internal,
            bootstrap=bootstrap if role == "PA" else None,
            health="/get_model_info", visible=gpus,
            metadata={"recipe": recipe, "phase": phase, "perspective": perspective,
                      "tp": tp, "afd_backend": "zmq", "zmq_sharding": False,
                      "peer_role": ffn_role if role == attn_role else attn_role,
                      "peer_host": peer["host"], "scheduler_host": sched_host,
                      "communicator_marker": "AFD ZMQ handshake ready"},
            common_flags="", startup_stage=stage,
        )
        if require_marker:
            spec.ready_log_patterns = ["AFD ZMQ handshake ready"]
            spec.ready_timeout_s = 600
        return spec
    ffn = make(ffn_role, ffn_node, attn_node, "ffn", ffn_port, 0)
    attn = make(attn_role, attn_node, ffn_node, "attn", attn_port, 1)
    return ffn, attn


def pdaf_cross_zmq_tp8(cluster, model, point):
    """Strict bench_common_cross-compatible four-node ZMQ PDAF baseline."""
    context = PlanContext(cluster)
    nodes = cluster["nodes"][:4]
    if len(nodes) != 4:
        raise ValueError("pdaf_cross_zmq_tp8 requires four nodes")
    pa_node, pf_node, da_node, df_node = nodes
    bootstrap = 46999
    pf, pa = _cross_zmq_pair_specs(
        context, model, attn_node=pa_node, ffn_node=pf_node, phase="prefill",
        attn_role="PA", ffn_role="PF", attn_port=46010, ffn_port=46020,
        ffn_base=47000, attn_base=47100, sched_port=46400,
        nccl_port=46600, bootstrap=bootstrap, recipe="pdaf_cross_zmq_tp8")
    df, da = _cross_zmq_pair_specs(
        context, model, attn_node=da_node, ffn_node=df_node, phase="decode",
        attn_role="DA", ffn_role="DF", attn_port=46030, ffn_port=46040,
        ffn_base=47200, attn_base=47300, sched_port=46500,
        nccl_port=46700, bootstrap=bootstrap, recipe="pdaf_cross_zmq_tp8")
    processes = [pf, df, pa, da]
    args = (f"--pd-disaggregation --mini-lb --prefill http://{pa.host}:{pa.port} "
            f"{bootstrap} --decode http://{da.host}:{da.port}")
    processes.append(_router(
        context, "PDAF_ROUTER", pa_node, 46000, args, prom=29100,
        metadata={"recipe": "pdaf_cross_zmq_tp8", "level": "top",
                  "activation_warmup": True}, startup_stage=2))
    plan = context.finish("pdaf", processes, f"http://{pa_node['host']}:46000")
    for node in nodes:
        for gpu in node["gpus"]:
            plan.pre_actions.append((node["host"],
                                     f"nvidia-smi -i {gpu} --lock-gpu-clocks=1410,1410"))
    plan.cluster["fixed_gpu_clock_mhz"] = 1410
    plan.cluster["afd_backend"] = "zmq"
    plan.cluster["zmq_sharding"] = False
    plan.validate()
    return plan


def shared_bipartite_pool(cluster, model, point):
    """Build a fixed, fully connected cross-node Attention/FFN pool."""
    context = PlanContext(cluster)
    nodes = cluster["nodes"]
    node_by_name = {node["name"]: node for node in nodes}
    default_tp = int(point.get("tp", 8))
    if len(nodes) < 4 and ("attention_instances" not in point or "ffn_instances" not in point):
        raise ValueError("shared_bipartite_pool default placement requires four nodes")
    raw_attn = point.get("attention_instances")
    raw_ffn = point.get("ffn_instances")
    if raw_attn is None:
        raw_attn = [
            {"node": nodes[0]["name"], "gpus": nodes[0]["gpus"][:default_tp], "id": "A0"},
            {"node": nodes[2]["name"], "gpus": nodes[2]["gpus"][:default_tp], "id": "A1"},
        ]
    if raw_ffn is None:
        raw_ffn = [
            {"node": nodes[1]["name"], "gpus": nodes[1]["gpus"][:default_tp], "id": "F0"},
            {"node": nodes[3]["name"], "gpus": nodes[3]["gpus"][:default_tp], "id": "F1"},
        ]
    def normalize(raw, prefix):
        result = []
        for index, item in enumerate(raw):
            node = node_by_name.get(item.get("node"))
            gpus = item.get("gpus")
            if node is None or not isinstance(gpus, list) or not gpus or any(g not in node["gpus"] for g in gpus):
                raise ValueError(f"invalid {prefix} instance placement at index {index}")
            tp = int(item.get("tp", point.get("tp", len(gpus))))
            if tp != len(gpus):
                raise ValueError("instance tp must equal GPU count")
            result.append((str(item.get("id", f"{prefix}{index}")), node, list(gpus), tp))
        if not result or len({x[0] for x in result}) != len(result):
            raise ValueError(f"{prefix} instances require unique IDs")
        return result
    attn_instances = normalize(raw_attn, "A")
    ffn_instances = normalize(raw_ffn, "F")

    def require_capacity(value, field):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{field} must be a positive integer")
        return value

    default_pf_capacity = require_capacity(point.get("pf_capacity", 1), "pf_capacity")
    pf_capacities = [
        require_capacity(item.get("capacity", default_pf_capacity),
                         f"ffn_instances[{index}].capacity")
        for index, item in enumerate(raw_ffn)
    ]
    static_pfs = [
        {"instance_id": instance_id, "capacity": capacity}
        for instance_id, capacity in zip((x[0] for x in ffn_instances), pf_capacities)
    ]
    if len({x[3] for x in attn_instances + ffn_instances}) != 1:
        raise ValueError("shared_bipartite_pool requires homogeneous TP")
    tp = attn_instances[0][3]
    allocated = [(node["name"], gpu) for _, node, gpus, _ in attn_instances + ffn_instances for gpu in gpus]
    if len(allocated) != len(set(allocated)):
        raise ValueError("shared_bipartite_pool instance GPUs overlap")
    attn_nodes = tuple(x[1] for x in attn_instances)
    ffn_nodes = tuple(x[1] for x in ffn_instances)
    attn_ids = [x[0] for x in attn_instances]
    ffn_ids = [x[0] for x in ffn_instances]
    attn_gpus = [x[2] for x in attn_instances]
    ffn_gpus = [x[2] for x in ffn_instances]
    coordinator_node = node_by_name.get(point.get("coordinator_node", attn_nodes[0]["name"]))
    if coordinator_node is None:
        raise ValueError("unknown coordinator_node")
    coordinator_port = int(point.get("afd_coordinator_port", 19090))
    port_base = int(point.get("afd_port_base", 50000))
    coordinator_endpoint = f"tcp://{coordinator_node['host']}:{coordinator_port}"
    sched_ports = {instance_id: port_base - 800 + i for i, instance_id in enumerate(ffn_ids)}

    edges = []
    for ai, attn_node in enumerate(attn_nodes):
        for fi, ffn_node in enumerate(ffn_nodes):
            edge_index = ai * len(ffn_nodes) + fi
            base = port_base + edge_index * 400
            edges.append({
                "edge_id": f"{attn_ids[ai]}-{ffn_ids[fi]}",
                "pa_id": attn_ids[ai],
                "pf_id": ffn_ids[fi],
                "pa_host": attn_node["host"],
                "pf_host": ffn_node["host"],
                "ffn_base_port": base,
                "attn_base_port": base + 100,
                "ffn_handshake_base_port": base + 200,
                "attn_handshake_base_port": base + 300,
                "channel": 0,
                "control_endpoint": f"tcp://{ffn_node['host']}:{sched_ports[ffn_ids[fi]]}",
            })

    log_path, pid_path = process_paths(
        context.run_tag, coordinator_node["name"], "AF_COORDINATOR", coordinator_port
    )
    pythonpath = ""
    if context.python_source:
        source = shlex.quote(context.python_source)
        pythonpath = f"export PYTHONPATH={source}:${{PYTHONPATH:-}}; "
    coordinator_args = " ".join(
        f"--pf {pf['instance_id']}:{pf['capacity']}" for pf in static_pfs
    )
    coordinator_command = (
        f"{pythonpath}setsid {PYTHON} -m sglang.srt.managers.afd_pool_coordinator "
        f"--host {coordinator_node['host']} --port {coordinator_port} {coordinator_args} "
        f">> {shlex.quote(log_path)} 2>&1 < /dev/null & echo $! > {shlex.quote(pid_path)}"
    )
    status_script = (
        "from sglang.srt.managers.afd_pool_coordinator import "
        "AFDPoolCoordinatorClient; "
        f"s=AFDPoolCoordinatorClient({coordinator_endpoint!r}).call('STATUS'); "
        f"assert len(s['pfs']) == {len(ffn_ids)}"
    )
    coordinator = ProcessSpec(
        "AF_COORDINATOR", coordinator_node["name"], coordinator_node["host"], [],
        coordinator_port, coordinator_command, health_path=None,
        ready_command=f"{pythonpath}{PYTHON} -c {shlex.quote(status_script)}",
        startup_stage=0, log_path=log_path, pid_path=pid_path,
        metadata={"recipe": "shared_bipartite_pool", "service": "afd_pool_coordinator",
                  "endpoint": coordinator_endpoint, "static_pfs": static_pfs},
    )

    common_flags = (
        "--afd-shared-pool --afd-comm-backend zmq --afd-micro-batch 1 "
        f"--afd-attn-tp {tp} --afd-ffn-tp {tp}"
    )
    lifecycle_timeout_ms = int(
        max(
            float(point.get("health_timeout_s", 0)),
            float(point.get("warmup_timeout_s", 0)),
            float(point.get("warmup_curl_timeout_s", 0)),
        )
        * 1000
    )
    zmq_timeout_ms = int(
        point.get("afd_zmq_timeout_ms", max(lifecycle_timeout_ms, 300000))
    )
    handshake_timeout_ms = int(
        point.get("afd_zmq_handshake_timeout_ms", zmq_timeout_ms)
    )
    if zmq_timeout_ms <= 0 or handshake_timeout_ms <= 0:
        raise ValueError("AFD ZMQ timeouts must be positive")
    common_env = (
        "AFD_CROSS_NODE_EXPERIMENTAL=1 AFD_LOCAL_TP=8 "
        f"AFD_ZMQ_TIMEOUT_MS={zmq_timeout_ms} "
        f"AFD_ZMQ_HANDSHAKE_TIMEOUT_MS={handshake_timeout_ms}"
    )
    processes = [coordinator]

    def peer_json(local_id, perspective):
        selected = []
        for edge in edges:
            if perspective == "attn" and edge["pa_id"] == local_id:
                selected.append({
                    "peer_id": edge["pf_id"], "peer_host": edge["pf_host"],
                    "ffn_base_port": edge["ffn_base_port"],
                    "attn_base_port": edge["attn_base_port"],
                    "ffn_handshake_base_port": edge["ffn_handshake_base_port"],
                    "attn_handshake_base_port": edge["attn_handshake_base_port"],
                    "channel": edge["channel"],
                    "control_endpoint": edge["control_endpoint"],
                })
            elif perspective == "ffn" and edge["pf_id"] == local_id:
                selected.append({
                    "peer_id": edge["pa_id"], "peer_host": edge["pa_host"],
                    "ffn_base_port": edge["ffn_base_port"],
                    "attn_base_port": edge["attn_base_port"],
                    "ffn_handshake_base_port": edge["ffn_handshake_base_port"],
                    "attn_handshake_base_port": edge["attn_handshake_base_port"],
                    "channel": edge["channel"],
                })
        return json.dumps(selected, separators=(",", ":"))

    for fi, node in enumerate(ffn_nodes):
        instance_id = ffn_ids[fi]
        specs = peer_json(instance_id, "ffn")
        local_edges = [edge for edge in edges if edge["pf_id"] == instance_id]
        internal = [sched_ports[instance_id]]
        for edge in local_edges:
            internal.extend(range(edge["attn_base_port"] + 1,
                                  edge["attn_base_port"] + tp + 1))
            internal.extend(range(edge["attn_handshake_base_port"] + 1,
                                  edge["attn_handshake_base_port"] + tp + 1))
        ffn = _server(
            context, "F", node, ffn_gpus[fi], port_base - 600 + fi * 10,
            model, tp,
            extra=(f"--afd-perspective ffn --base-gpu-id 0 {common_flags} "
                   f"--afd-instance-id {instance_id} "
                   f"--afd-coordinator-endpoint {coordinator_endpoint} "
                   f"--afd-shared-peer-specs {shlex.quote(specs)}"),
            env=(f"{common_env} AFD_SCHED_HOST={node['host']} "
                 f"AFD_SCHED_PORT={sched_ports[instance_id]}"),
            nccl=port_base - 700 + fi, internal=internal, health=None,
            metadata={"recipe": "shared_bipartite_pool", "instance_id": instance_id,
                      "perspective": "ffn", "tp": tp,
                      "peer_ids": [edge["pa_id"] for edge in local_edges],
                      "control_endpoint": f"tcp://{node['host']}:{sched_ports[instance_id]}"},
            common_flags="", startup_stage=1,
        )
        processes.append(ffn)

    endpoints = []
    for ai, node in enumerate(attn_nodes):
        instance_id = attn_ids[ai]
        specs = peer_json(instance_id, "attn")
        local_edges = [edge for edge in edges if edge["pa_id"] == instance_id]
        internal = []
        for edge in local_edges:
            internal.extend(range(edge["ffn_base_port"] + 1,
                                  edge["ffn_base_port"] + tp + 1))
            internal.extend(range(edge["ffn_handshake_base_port"] + 1,
                                  edge["ffn_handshake_base_port"] + tp + 1))
        port = port_base - 500 + ai * 10
        attn = _server(
            context, "A", node, attn_gpus[ai], port, model, tp,
            extra=(f"--afd-perspective attn --base-gpu-id 0 {common_flags} "
                   f"--afd-instance-id {instance_id} "
                   f"--afd-coordinator-endpoint {coordinator_endpoint} "
                   f"--afd-shared-peer-specs {shlex.quote(specs)}"),
            env=common_env, nccl=port_base - 750 + ai, internal=internal,
            health="/get_model_info",
            metadata={"recipe": "shared_bipartite_pool", "instance_id": instance_id,
                      "perspective": "attn", "tp": tp,
                      "peer_ids": [edge["pf_id"] for edge in local_edges]},
            common_flags="", startup_stage=2,
        )
        processes.append(attn)
        endpoints.append(f"http://{node['host']}:{port}")

    plan = context.finish(
        "af", processes, endpoints, routing_policy="client_round_robin"
    )
    plan.runtime_options.update({
        "warmup_parallel": True,
        "afd_pool_coordinator": {
            "instance_id": "coordinator", "node": coordinator_node["name"],
            "endpoint": coordinator_endpoint, "process_role": "AF_COORDINATOR",
            "static_pfs": static_pfs,
        },
        "afd_pool_edges": edges,
    })
    plan.validate()
    return plan


def af_shared_pool_preflight(cluster, model, point):
    """Compile-only A16/F16 placement pending shared-pool communication preflight."""
    context = PlanContext(cluster)
    nodes = cluster["nodes"][:4]
    if len(nodes) != 4 or any(len(node["gpus"]) != 8 for node in nodes):
        raise ValueError("af_shared_pool_preflight requires four 8-GPU nodes")
    processes, endpoints = [], []
    for pair, (ffn_node, attn_node) in enumerate(((nodes[0], nodes[2]), (nodes[1], nodes[3]))):
        ffn, attn = _cross_zmq_pair_specs(
            context, model, attn_node=attn_node, ffn_node=ffn_node, phase="prefill",
            attn_role="A", ffn_role="F", attn_port=46110 + pair * 10,
            ffn_port=46120 + pair * 10, ffn_base=47400 + pair * 400,
            attn_base=47500 + pair * 400, sched_port=46800 + pair,
            nccl_port=46810, bootstrap=46990 + pair,
            recipe="af_shared_pool_preflight", require_marker=True)
        # AF-only must not carry PD/Mooncake semantics; strip inherited flags.
        for spec in (ffn, attn):
            for fragment in (
                "--disaggregation-mode prefill ",
                "--afd-disagg-interleave-poll ", "--num-reserved-decode-tokens 512 ",
                "--disaggregation-transfer-backend mooncake ",
                f"--disaggregation-bootstrap-port {46990 + pair} ",
                f"--disaggregation-ib-device {_nic(spec and (ffn_node if spec is ffn else attn_node), spec.gpus[0], cluster)} ",
            ):
                spec.command = spec.command.replace(fragment, "")
            spec.bootstrap_port = None
            spec.metadata["topology"] = "shared_pool_preflight"
        processes.extend((ffn, attn)); endpoints.append(f"http://{attn.host}:{attn.port}")
    return context.finish("af", processes, endpoints, routing_policy="client_round_robin")



def _comm_ablation_nodes(cluster):
    by_name = {node["name"]: node for node in cluster["nodes"]}
    try:
        return by_name["node3"], by_name["node4"]
    except KeyError as exc:
        raise ValueError("communication ablation recipes require node3 and node4") from exc


def _comm_meta(point, recipe, transport, **extra):
    return {"recipe": recipe, "matrix": "matrix-recipes", "expected_link": point["expected_link"],
            "transport": transport, **extra}


def comm_native_2gpu(cluster, model, point):
    context = PlanContext(cluster)
    node3, node4 = _comm_ablation_nodes(cluster)
    link, base = point["expected_link"], int(point["port_base"])
    common = "--disable-custom-all-reduce"
    nccl_debug = "NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P,SHM,NET"
    processes = []
    if link == "nvlink":
        env = (f"{nccl_debug} NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=0 "
               "NCCL_SHM_DISABLE=0 NCCL_NVLS_ENABLE=0")
        processes.append(_server(context, "NATIVE", node3, [0, 1], base, model, 2,
            env=env, nccl=base + 1, extra=common,
            metadata=_comm_meta(point, "comm_native_2gpu", "nccl", tp=2)))
    elif link == "pcie_host_staged":
        env = (f"{nccl_debug} NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 "
               "NCCL_SHM_DISABLE=0 NCCL_NVLS_ENABLE=0")
        processes.append(_server(context, "NATIVE", node3, [0, 1], base, model, 2,
            env=env, nccl=base + 1, extra=common,
            metadata=_comm_meta(point, "comm_native_2gpu", "nccl_shm", tp=2,
                                host_staged=True)))
    elif link == "rdma":
        dist = f"{node3['host']}:{base + 2}"
        for rank, node in enumerate((node3, node4)):
            env = (f"{nccl_debug} NCCL_IB_DISABLE=0 NCCL_P2P_DISABLE=0 "
                   f"NCCL_SHM_DISABLE=0 NCCL_NVLS_ENABLE=0 NCCL_SOCKET_IFNAME={node['roce']} "
                   f"NCCL_IB_HCA={_nic(node, 0, cluster)}")
            extra = (f"{common} --nnodes 2 --node-rank {rank} "
                     f"--dist-init-addr {dist}")
            processes.append(_server(
                context, "NATIVE", node, [0], base, model, 2, env=env,
                nccl=base + 1, extra=extra, health="/health" if rank == 0 else None,
                metadata=_comm_meta(
                    point, "comm_native_2gpu", "nccl_rdma", tp=2, node_rank=rank
                ),
            ))
    else:
        raise ValueError(f"unsupported native expected_link: {link}")
    return context.finish("native", processes,
                          f"http://{node3['host']}:{processes[0].port}")


def comm_pd_1p1d(cluster, model, point):
    context = PlanContext(cluster)
    node3, node4 = _comm_ablation_nodes(cluster)
    link, base = point["expected_link"], int(point["port_base"])
    pnode, dnode = node3, (node4 if link == "rdma" else node3)
    protocol = "rdma" if link == "rdma" else "tcp"
    device_p, device_d = _nic(pnode, 0, cluster), _nic(dnode, 0 if dnode is node4 else 1, cluster)
    if link == "nvlink":
        common_env = (
            "SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK MC_FORCE_MNNVL=true; "
            "unset MC_FORCE_TCP MOONCAKE_USE_CUDA_IPC; "
            "export MOONCAKE_PROTOCOL=tcp SGLANG_HOST_IP="
        )
    elif link == "pcie_host_staged":
        common_env = (
            "unset MC_FORCE_HCA MC_FORCE_MNNVL MC_INTRANODE_NVLINK MC_INTRA_NVLINK "
            "SGLANG_MOONCAKE_CUSTOM_MEM_POOL; "
            "export MC_FORCE_TCP=1 MC_LOG_LEVEL=INFO MOONCAKE_USE_CUDA_IPC=0 "
            "MOONCAKE_PROTOCOL=tcp SGLANG_HOST_IP="
        )
    else:
        common_env = (
            "MOONCAKE_USE_CUDA_IPC=0; "
            "unset SGLANG_MOONCAKE_CUSTOM_MEM_POOL MC_FORCE_MNNVL MC_FORCE_TCP; "
            f"export MOONCAKE_PROTOCOL={protocol} SGLANG_HOST_IP="
        )
    bootstrap = base + 2
    log_flags = " --log-level debug" if link == "nvlink" else ""
    pextra = ("--disaggregation-mode prefill --disaggregation-transfer-backend mooncake "
              f"--disaggregation-bootstrap-port {bootstrap} --disaggregation-ib-device {device_p}"
              f"{log_flags}")
    dextra = ("--disaggregation-mode decode --disaggregation-transfer-backend mooncake "
              f"--disaggregation-bootstrap-port {bootstrap} --disaggregation-ib-device {device_d}"
              f"{log_flags}")
    transport = "mooncake_nvlink" if link == "nvlink" else f"mooncake_{protocol}"
    p = _server(context, "P", pnode, [0], base, model, 1, extra=pextra,
        env=common_env + pnode["host"], bootstrap=bootstrap,
        metadata=_comm_meta(point, "comm_pd_1p1d", transport, phase="prefill"))
    dgpu = 0 if dnode is node4 else 1
    d = _server(context, "D", dnode, [dgpu], base + 10, model, 1, extra=dextra,
        env=common_env + dnode["host"], startup_stage=1,
        metadata=_comm_meta(point, "comm_pd_1p1d", transport, phase="decode",
                            host_staged=link == "pcie_host_staged"))
    args = (f"--pd-disaggregation --prefill http://{p.host}:{p.port} {bootstrap} "
            f"--decode http://{d.host}:{d.port}")
    router = _router(context, "PD_ROUTER", node3, base + 20, args, prom=base + 21,
        metadata=_comm_meta(point, "comm_pd_1p1d", transport, level="router"),
        startup_stage=2)
    return context.finish("pd", [p, d, router], f"http://{node3['host']}:{router.port}")


def pd_cuda_ipc_same_node(cluster, model, point):
    """Single-host PD smoke topology backed by direct CUDA IPC KV transfer."""
    context = PlanContext(cluster)
    node = cluster["nodes"][0]
    base = int(point["port_base"])
    tp = int(point["pd_tp"])
    if int(point["nodes"]) != 1 or point["expected_link"] != "nvlink":
        raise ValueError("pd_cuda_ipc_same_node requires one NVLink-connected node")
    if tp not in (1, 2, 4):
        raise ValueError("pd_cuda_ipc_same_node supports pd_tp 1, 2, or 4")
    if int(point["gpus_total"]) != 2 * tp:
        raise ValueError("pd_cuda_ipc_same_node requires equal P/D TP GPU counts")
    if len(node["gpus"]) < 2 * tp:
        raise ValueError(f"pd_cuda_ipc_same_node requires {2 * tp} GPUs on one node")

    pgpus = list(node["gpus"][:tp])
    dgpus = list(node["gpus"][tp : 2 * tp])
    # CUDA IPC device IDs are process-local CUDA ordinals.  Keep the complete
    # P/D union visible to both servers so P rank i can address D rank i as
    # tp + i (P base 0, D base tp).
    visible_gpus = pgpus + dgpus
    bootstrap = base + 2
    metadata = {
        "recipe": "pd_cuda_ipc_same_node",
        "matrix": "pd-cuda-ipc-smoke",
        "expected_link": point["expected_link"],
        "transport": "cuda_ipc",
        "hostname": node["host"],
        "tp": tp,
    }
    common = (
        "--disaggregation-transfer-backend cuda_ipc "
        f"--disaggregation-bootstrap-port {bootstrap}"
    )
    p = _server(
        context, "P", node, pgpus, base, model, tp,
        nccl=base + 1,
        extra=f"--disaggregation-mode prefill --base-gpu-id 0 {common}",
        bootstrap=bootstrap,
        visible=visible_gpus,
        metadata={**metadata, "phase": "prefill"},
    )
    d = _server(
        context, "D", node, dgpus, base + 10, model, tp,
        nccl=base + 11,
        extra=f"--disaggregation-mode decode --base-gpu-id {tp} {common}",
        visible=visible_gpus,
        startup_stage=1,
        metadata={**metadata, "phase": "decode"},
    )
    args = (
        f"--pd-disaggregation --prefill http://{p.host}:{p.port} {bootstrap} "
        f"--decode http://{d.host}:{d.port}"
    )
    router = _router(
        context, "PD_ROUTER", node, base + 20, args, prom=base + 21,
        startup_stage=2, metadata={**metadata, "level": "router"},
    )
    return context.finish("pd", [p, d, router], f"http://{node['host']}:{router.port}")


def comm_af_a1f1(cluster, model, point):
    context = PlanContext(cluster)
    node3, node4 = _comm_ablation_nodes(cluster)
    link, base = point["expected_link"], int(point["port_base"])
    anode, fnode = node3, (node4 if link == "rdma" else node3)
    same_node = anode is fnode
    # Preserve the validated legacy local-device layout: F is CUDA 0 and A is
    # CUDA 1 when both processes share CUDA_VISIBLE_DEVICES=0,1.  The IPC
    # peer offsets are therefore +1 from F and -1 from A.  Cross-node UCX
    # processes each expose one GPU as local CUDA 0 and do not use offsets.
    fgpu, agpu = 0, (1 if same_node else 0)
    default_backend = {"nvlink": "ipc_cpp", "pcie_host_staged": "zmq", "rdma": "ucx"}[link]
    backend = point.get("afd_backend", default_backend)
    if backend == "mooncake" and (link != "rdma" or same_node):
        raise ValueError("comm_af_a1f1 Mooncake requires cross-node expected_link=rdma")
    if backend not in {default_backend, "mooncake"}:
        raise ValueError(f"unsupported comm_af_a1f1 backend {backend!r} for {link}")
    common_flags = f"--afd-comm-backend {backend} --afd-micro-batch 1 --afd-attn-tp 1 --afd-ffn-tp 1"
    sched = base + 2
    if backend == "ipc_cpp":
        common = f"AFD_IPC_SYNC_MODE=ipc_event AFD_IPC_CHANNEL_BASE={base} AFD_SCHED_PORT={sched}"
        fenv = f"{common} AFD_IPC_PEER_OFFSET=1 AFD_UCX_BASE_PORT={base + 30}"
        aenv = f"{common} AFD_IPC_PEER_OFFSET=-1 AFD_UCX_BASE_PORT={base + 40} AFD_UCX_FFN_HOST=127.0.0.1"
        finternal, ainternal = [sched, base + 30], [base + 40]
    elif backend == "zmq":
        common = (f"AFD_CROSS_NODE_EXPERIMENTAL=1 AFD_LOCAL_TP=1 "
                  f"AFD_ZMQ_TIMEOUT_MS=300000 AFD_ZMQ_HANDSHAKE_TIMEOUT_MS=300000 "
                  f"AFD_ZMQ_PEER_HOST=127.0.0.1 AFD_FFN_BASE_PORT={base + 30} "
                  f"AFD_ATTN_BASE_PORT={base + 40} AFD_ZMQ_FFN_HANDSHAKE_BASE_PORT={base + 50} "
                  f"AFD_ZMQ_ATTN_HANDSHAKE_BASE_PORT={base + 60} AFD_ZMQ_HOST_STAGING=1 "
                  f"AFD_SCHED_PORT={sched}")
        fenv = aenv = common
        finternal, ainternal = [sched, base + 41, base + 61], [base + 31, base + 51]
    elif backend == "mooncake":
        common = (f"SGLANG_MOONCAKE_TRANSPORT=rdma MOONCAKE_IB_DEVICE=mlx5_0 "
                  f"MOONCAKE_USE_CUDA_IPC=0 AFD_MOONCAKE_TIMEOUT_MS=300000 "
                  f"AFD_MOONCAKE_ATTN_CONTROL_PORT={base + 40} "
                  f"AFD_MOONCAKE_FFN_CONTROL_PORT={base + 30} AFD_SCHED_PORT={sched}")
        fenv = (f"GLOO_SOCKET_IFNAME={fnode['roce']} NCCL_SOCKET_IFNAME={fnode['roce']} "
                f"SGLANG_HOST_IP={fnode['host']} {common} AFD_SCHED_HOST={fnode['host']} "
                f"AFD_MOONCAKE_PEER_HOST={anode['host']}")
        aenv = (f"GLOO_SOCKET_IFNAME={anode['roce']} NCCL_SOCKET_IFNAME={anode['roce']} "
                f"SGLANG_HOST_IP={anode['host']} {common} AFD_SCHED_HOST={fnode['host']} "
                f"AFD_MOONCAKE_PEER_HOST={fnode['host']}")
        finternal, ainternal = [sched, base + 40], [base + 30]
    else:
        # UCX's Python extension and its libucm/libuct/libucs dependencies are
        # installed in the Python libucx package on both RDMA nodes. Prepend that
        # exact directory (while retaining /usr/local/lib as a fallback). Keep the
        # shell expansion literal: planner_common.command emits these assignments
        # as one ``export ...`` statement at execution time.
        loader_env = (
            "LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/libucx/lib:"
            "/usr/local/lib:${LD_LIBRARY_PATH:-}"
        )
        common = (f"{loader_env} UCX_TLS=rc,tcp,cuda_copy "
                  f"AFD_UCX_TLS=rc,tcp,cuda_copy "
                  f"UCX_SOCKADDR_TLS_PRIORITY=rdmacm,tcp UCX_LOG_LEVEL=info "
                  f"UCX_PROTO_INFO=y AFD_UCX_NUM_NICS=1 "
                  f"AFD_UCX_BASE_PORT={base + 30} AFD_UCX_HOST_STAGING=0 "
                  f"AFD_UCX_GPU_DIRECT=1 AFD_SCHED_PORT={sched}")
        fenv = f"{common} UCX_NET_DEVICES={_nic(fnode, fgpu, cluster)}:1"
        aenv = (f"{common} UCX_NET_DEVICES={_nic(anode, agpu, cluster)}:1 "
                f"AFD_UCX_FFN_HOST={fnode['host']}")
        finternal, ainternal = [sched, base + 30], []
    visible_f = [0, 1] if same_node else [fgpu]
    visible_a = [0, 1] if same_node else [agpu]
    f = _server(context, "F", fnode, [fgpu], base, model, 1,
        extra=f"--afd-perspective ffn --base-gpu-id {fgpu} {common_flags}", env=fenv,
        internal=finternal, health="/get_model_info", visible=visible_f,
        metadata=_comm_meta(point, "comm_af_a1f1", backend, perspective="ffn",
                            host_staged=link == "pcie_host_staged"))
    a = _server(context, "A", anode, [agpu], base + 10, model, 1,
        extra=f"--afd-perspective attn --base-gpu-id {agpu} {common_flags}", env=aenv,
        internal=ainternal, health="/get_model_info", visible=visible_a,
        startup_stage=0 if backend == "mooncake" else 1,
        metadata=_comm_meta(point, "comm_af_a1f1", backend, perspective="attn",
                            host_staged=link == "pcie_host_staged"))
    plan = context.finish("af", [f, a], f"http://{anode['host']}:{a.port}")
    if backend == "ucx":
        import_ucp = (
            f"export {loader_env}; {PYTHON} -c {shlex.quote('import ucp')}"
        )
        plan.pre_actions = [(node["host"], import_ucp) for node in (fnode, anode)]
    return plan


RECIPES = {
    "comm_native_2gpu": comm_native_2gpu,
    "comm_pd_1p1d": comm_pd_1p1d,
    "pd_cuda_ipc_same_node": pd_cuda_ipc_same_node,
    "comm_af_a1f1": comm_af_a1f1,
    "legacy_native_tp": legacy_native_tp,
    "legacy_pd_dual": legacy_pd_dual,
    "legacy_af_profile_replicas": legacy_af_profile_replicas,
    "legacy_pdaf_3p1d": legacy_pdaf_3p1d,
    "legacy_native_pair_tp8": legacy_native_pair_tp8,
    "legacy_pd_xnode_16": legacy_pd_xnode_16,
    "af_node34_a1f1_pool": af_node34_a1f1_pool,
    "legacy_pdaf_xnode_tp4": legacy_pdaf_xnode_tp4,
    "legacy_tier1_layout": legacy_tier1_layout,
    "pd_exact_2p2d_tp8": pd_exact_2p2d_tp8,
    "pdaf_cross_zmq_tp8": pdaf_cross_zmq_tp8,
    "af_shared_pool_preflight": af_shared_pool_preflight,
    "shared_bipartite_pool": shared_bipartite_pool,
}
