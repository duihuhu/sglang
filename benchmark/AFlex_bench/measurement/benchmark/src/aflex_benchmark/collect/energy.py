from __future__ import annotations
import json, shlex

def energy_delta(before: dict, after: dict) -> dict:
    per = {}
    for node, gpus in before.items():
        per[node] = {str(gpu): max(0, float(after.get(node, {}).get(str(gpu), after.get(node, {}).get(gpu, 0))) - float(value)) / 1000.0 for gpu, value in gpus.items()}
    return {"unit": "joule", "per_node_gpu_j": per, "per_node_j": {node: sum(values.values()) for node, values in per.items()}, "total_j": sum(sum(values.values()) for values in per.values())}

def read_cluster_energy(executor, cluster, nodes=None, energy_scope=None):
    out = {}
    selected = cluster["nodes"][:nodes] if nodes else cluster["nodes"]
    if energy_scope is not None:
        selected = [node for node in cluster["nodes"] if node["name"] in energy_scope]
    for node in selected:
        scoped_gpus = energy_scope[node["name"]] if energy_scope is not None else node["gpus"]
        indices = ",".join(map(str, scoped_gpus))
        code = "import json,pynvml;pynvml.nvmlInit();idx=[" + indices + "];print(json.dumps({str(i):pynvml.nvmlDeviceGetTotalEnergyConsumption(pynvml.nvmlDeviceGetHandleByIndex(i)) for i in idx}));pynvml.nvmlShutdown()"
        result = executor.run(node["host"], f"/usr/bin/python3 -c {shlex.quote(code)}", check=False)
        lines = [line for line in (result.stdout or "").splitlines() if line.startswith("{")]
        out[node["name"]] = json.loads(lines[-1]) if lines else {str(gpu): 0 for gpu in scoped_gpus}
    return out

def read_gpu_uuids(executor, cluster, energy_scope):
    out = {}
    by_name = {node["name"]: node for node in cluster["nodes"]}
    for name, gpus in energy_scope.items():
        node = by_name[name]
        indices = ",".join(map(str, gpus))
        command = f"nvidia-smi -i {indices} --query-gpu=index,uuid --format=csv,noheader,nounits"
        result = executor.run(node["host"], command, check=False)
        rows = {}
        for line in (result.stdout or "").splitlines():
            parts = [part.strip() for part in line.split(",", 1)]
            if len(parts) == 2:
                rows[parts[0]] = parts[1]
        out[name] = rows
    return out
