from __future__ import annotations
import json
from pathlib import Path
from typing import Any

class ConfigError(ValueError): pass

MAX_QPS = 16

def load_json(path: str | Path) -> dict[str, Any]:
    p=Path(path)
    try: data=json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as exc: raise ConfigError(f"cannot load {p}: {exc}") from exc
    if not isinstance(data, dict): raise ConfigError(f"{p}: root must be object")
    return data

def validate_cluster(c: dict[str, Any]) -> None:
    nodes=c.get("nodes", [])
    if not nodes: raise ConfigError("cluster.nodes is empty")
    names=[n.get("name") for n in nodes]; hosts=[n.get("host") for n in nodes]
    if len(names)!=len(set(names)) or len(hosts)!=len(set(hosts)): raise ConfigError("node names/hosts must be unique")
    for n in nodes:
        g=n.get("gpus")
        if not isinstance(g,list) or not g or len(g)!=len(set(g)): raise ConfigError(f"invalid GPUs for {n.get('name')}")
    if not isinstance(c.get("container"), str) or not c["container"].strip():
        raise ConfigError("cluster.container must be a non-empty string")
    if not isinstance(c.get("container_root"), str) or not c["container_root"].startswith("/"):
        raise ConfigError("cluster.container_root must be an absolute path")
    python_source = c.get("python_source")
    if python_source is not None and (not isinstance(python_source, str) or not python_source.startswith("/")):
        raise ConfigError("cluster.python_source must be an absolute path")
    host_cleanup = c.get("host_deep_cleanup_cmd")
    if host_cleanup is not None and (not isinstance(host_cleanup, str) or not host_cleanup.strip()):
        raise ConfigError("cluster.host_deep_cleanup_cmd must be a non-empty shell command")
    if not isinstance(c.get("host_deep_cleanup_strict", True), bool):
        raise ConfigError("cluster.host_deep_cleanup_strict must be a boolean")
    if not isinstance(c.get("host_network", False), bool):
        raise ConfigError("cluster.host_network must be a boolean")

def _validate_qps(values: Any, field: str) -> None:
    if not isinstance(values, list) or not values:
        raise ConfigError(f"{field} must be a non-empty list")
    for qps in values:
        if isinstance(qps, bool) or not isinstance(qps, (int, float)) or qps <= 0:
            raise ConfigError(f"{field} values must be positive numbers")
        if qps > MAX_QPS:
            raise ConfigError(f"{field} values must not exceed {MAX_QPS}")

def validate_workloads(w: dict[str, Any]) -> None:
    max_qps = w.get("max_qps")
    if max_qps != MAX_QPS:
        raise ConfigError(f"workloads.max_qps must be {MAX_QPS}")
    _validate_qps(w.get("qps"), "workloads.qps")

def validate_matrix(m: dict[str, Any]) -> None:
    if m.get("rq") not in {"RQ2","RQ5","RQ6","SMOKE"}: raise ConfigError("rq must be RQ2/RQ5/RQ6/SMOKE")
    for p in m.get("points",[]):
        _validate_qps(p.get("qps"), f"point {p.get('id', '<unknown>')} qps")
        if p.get("architecture") not in {"native","pd","af","pdaf"}: raise ConfigError("unknown architecture")
        if int(p.get("nodes",0))<1: raise ConfigError("nodes must be positive")
        if p.get("architecture")=="af" and ("disaggregation_mode" in p or p.get("pd_router")): raise ConfigError("AF-only cannot use PD semantics")
        if p.get("parallelism")=="ep" and not p.get("experimental"): raise ConfigError("EP must be experimental")
        for field in ("max_inflight", "sweep_max_inflight", "pf_capacity"):
            value = p.get(field)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ConfigError(f"{field} must be a positive integer")
        for index, instance in enumerate(p.get("ffn_instances", [])):
            capacity = instance.get("capacity")
            if capacity is not None and (isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1):
                raise ConfigError(f"ffn_instances[{index}].capacity must be a positive integer")
        wall_cap = p.get("point_wall_cap_s")
        if wall_cap is not None and (isinstance(wall_cap, bool) or not isinstance(wall_cap, (int, float)) or wall_cap <= 0):
            raise ConfigError("point_wall_cap_s must be a positive number")

def load_bundle(config_dir: str | Path, matrix: str | Path,
                cluster: str | Path | None = None) -> dict[str, Any]:
    d = Path(config_dir)
    matrix_config = load_json(matrix)
    cluster_path = Path(cluster) if cluster is not None else d / matrix_config.get("cluster", "cluster.json")
    if cluster is not None and not cluster_path.is_absolute() and not cluster_path.exists():
        cluster_path = d / cluster_path
    out = {
        "cluster": load_json(cluster_path),
        "models": load_json(d / "models.json"),
        "workloads": load_json(d / "workloads.json"),
        "matrix": matrix_config,
    }
    out["cluster_source"] = str(cluster_path)
    validate_cluster(out["cluster"])
    validate_workloads(out["workloads"])
    validate_matrix(out["matrix"])
    return out
