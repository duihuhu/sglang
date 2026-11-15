"""Offline configuration, workload and asset validation."""

import json
import math
import re
from pathlib import Path

SCHEMES = (
    "aflex_tier1",
    "native_tp1_baseline",
    "native_tp1_tier",
    "pd_hetero_baseline",
    "pd_hetero_tier_biscale",
)


def local_path(cfg, value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(cfg["_base"]) / path


def requests(path):
    rows = [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]
    if not rows:
        raise ValueError("empty workload")
    previous = 0
    for row in rows:
        for key in ("input_len", "output_len"):
            if type(row[key]) is not int or row[key] <= 0:
                raise ValueError(f"{key} must be a positive integer")
        t = row["arrival_time_s"]
        if type(t) not in (int, float) or not math.isfinite(t) or t < previous:
            raise ValueError("arrival times must be finite, nonnegative, nondecreasing")
        previous = t
    return rows


def load(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    cfg["_base"] = str(path.parent)
    if len(cfg["nodes"]) != 2 or len({n["host"] for n in cfg["nodes"]}) != 2:
        raise ValueError("two distinct nodes required")
    for node in cfg["nodes"]:
        for key in ("host", "container", "ssh_user"):
            if not re.fullmatch(r"[A-Za-z0-9_.:-]+", node[key]) or node[key].startswith(
                "-"
            ):
                raise ValueError(f"unsafe {key}")
        if node["gpus"] != list(range(8)):
            raise ValueError(
                "current baseline parity requires physical GPU pool 0..7 per node"
            )
        if set(node["gpu_nic"]) != set(map(str, node["gpus"])):
            raise ValueError("gpu_nic must cover all GPU IDs")
    allowed_env = {
        "MN_ENABLE_FABRIC_AFFINITY",
        "MN_RDMA_FABRICS",
        "AFLEX_DISABLE_RUNTIME_DVFS",
        "AFLEX_FREEZE_ACTIVE_FREQ",
        "AFLEX_SPLIT_PD_NODES",
        "AFLEX_PREFER_NODE2",
    }
    if set(cfg.get("environment", {})) - allowed_env:
        raise ValueError("unsupported environment override")
    for key in (
        "repo",
        "python",
        "model",
        "energy_model",
        "remote_output",
        "dvfs_library",
    ):
        value = cfg["paths"].get(key)
        if (
            not isinstance(value, str)
            or not value.startswith("/")
            or not re.fullmatch(r"[/A-Za-z0-9_.-]+", value)
        ):
            raise ValueError(f"absolute shell-safe container path required: {key}")
    for key in ("ttft_ms", "tpot_ms", "eval_ttft_ms", "eval_tpot_ms"):
        v = cfg["slo"][key]
        if type(v) not in (int, float) or not math.isfinite(v) or v <= 0:
            raise ValueError(f"invalid SLO: {key}")
    for value in cfg["ports"].values():
        if type(value) is not int or not 1024 <= value <= 60000:
            raise ValueError("invalid port base")
    required_ports = {
        "router",
        "prefill",
        "decode",
        "bootstrap",
        "native",
        "nccl",
        "sched",
        "ucx",
    }
    if set(cfg["ports"]) != required_ports:
        raise ValueError("all port bases must be supplied")
    bases = sorted(cfg["ports"].values())
    if any(b - a < 200 for a, b in zip(bases, bases[1:])):
        raise ValueError("port ranges must be separated by at least 200")
    if type(cfg["baseline_freq"]) is not int or cfg["baseline_freq"] <= 0:
        raise ValueError("baseline_freq must be positive")
    if not cfg["points"]:
        raise ValueError("points cannot be empty")
    for point in cfg["points"]:
        if point["scheme"] not in SCHEMES:
            raise ValueError("unsupported scheme")
        if type(point["qps"]) is not int or point["qps"] <= 0:
            raise ValueError("qps must be positive integer")
        requests(local_path(cfg, point["workload"]))
        if point["scheme"] == "aflex_tier1":
            parse_plan(local_path(cfg, point["plan"]))
    manifest = cfg["preflight"]
    if not manifest["native_libraries"] or not manifest["prediction_files"]:
        raise ValueError("non-empty native library and prediction manifests required")
    if cfg["paths"]["dvfs_library"] not in manifest["native_libraries"]:
        raise ValueError(
            "paths.dvfs_library must be included in preflight.native_libraries"
        )
    for path in manifest["native_libraries"]:
        if not Path(path).is_absolute():
            raise ValueError("native library paths must be absolute")
    for name in manifest["prediction_files"]:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("prediction manifest paths must stay within energy_model")
    options = cfg.get("cleanup", {})
    if type(options.get("force_kill", False)) is not bool:
        raise ValueError("force_kill must be boolean")
    for key in ("term_timeout_s", "kill_timeout_s"):
        value = options.get(key, 15)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("cleanup timeouts must be finite and nonnegative")
    return cfg


def parse_plan(path):
    from .aflex import PrefillSpec, Tier1TestConfig

    raw = json.loads(Path(path).read_text())
    for key in (
        "k_p",
        "k_d",
        "tp_pa",
        "tp_pf",
        "tp_da",
        "tp_df",
        "f_pa",
        "f_pf",
        "f_da",
        "f_df",
    ):
        if type(raw[key]) is not int or raw[key] <= 0:
            raise ValueError(f"invalid plan {key}")
    if type(raw.get("tier", True)) is not bool:
        raise ValueError("tier must be boolean")
    specs = raw.get("prefill_specs")
    if specs is not None:
        if not isinstance(specs, list) or len(specs) != raw["k_p"] or not specs:
            raise ValueError("prefill_specs must contain exactly k_p entries")
        parsed = []
        for spec in specs:
            if not isinstance(spec, dict) or set(spec) != {
                "tp_pa",
                "tp_pf",
                "f_pa",
                "f_pf",
            }:
                raise ValueError("invalid PrefillSpec fields")
            for key, value in spec.items():
                if type(value) is not int or value <= 0:
                    raise ValueError(f"invalid PrefillSpec {key}")
            parsed.append(PrefillSpec(**spec))
        raw["prefill_specs"] = tuple(parsed)
    config = Tier1TestConfig(**raw)
    tp_values = [config.tp_pa, config.tp_pf, config.tp_da, config.tp_df]
    for spec in config.resolved_prefill_specs():
        tp_values.extend((spec.tp_pa, spec.tp_pf))
    if any(value not in (1, 2, 4, 8) for value in tp_values):
        raise ValueError("unsupported TP")
    return config
