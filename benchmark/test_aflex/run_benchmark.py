#!/usr/bin/env python3
"""Portable, explicit-config AFlex macro benchmark CLI."""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "test_aflex"

from .config import load, local_path, parse_plan, requests


def configure(cfg, dry=False):
    # Import heavyweight dependencies only after the offline CLI has parsed.
    from . import aflex as af
    from . import deployments as base
    from . import runtime as rt

    rt.CONFIG, rt.DRY = cfg, dry
    rt.COMMANDS.clear()
    rt.NODE1_IP, rt.NODE2_IP = [n["host"] for n in cfg["nodes"]]
    for key, name in (
        ("repo", "REPO"),
        ("python", "PYTHON"),
        ("model", "MODEL"),
        ("energy_model", "ENERGY_MODEL_DIR_V1"),
        ("remote_output", "LOG_C"),
    ):
        setattr(rt, name, cfg["paths"][key])
    rt.IB_JSON_FILE = cfg["paths"]["remote_output"] + "/ib_" + rt.RUN_ID + ".json"
    rt.MAX_GPU_FREQ = cfg["baseline_freq"]
    rt.TTFT_SLO_MS, rt.TPOT_SLO_MS = cfg["slo"]["ttft_ms"], cfg["slo"]["tpot_ms"]
    base.NODE1, base.NODE2 = rt.NODE1_IP, rt.NODE2_IP
    for key in ("MODEL", "PYTHON", "LOG_C", "ENERGY_MODEL_DIR_V1"):
        setattr(base, key, getattr(rt, key))
    base.DVFS_LOG_DIR = rt.LOG_C
    base.GPU_NIC = {int(k): v for k, v in cfg["nodes"][0]["gpu_nic"].items()}
    base.MOONCAKE_IP_BASE = {n["host"]: n["mooncake_ip_base"] for n in cfg["nodes"]}
    for key in (
        "AFLEX_PORT_OFFSET",
        "AFD_TIMING",
        "AFD_DETAILED_TIMING",
        "AFD_DVFS_SWITCH_LOG",
        "AFD_DVFS_DECISION_LOG",
        "AFD_DECODE_BATCH_SAMPLE_DIR",
        "AFD_NVLINK_EQUIVALENT_LINKS",
        "SGLANG_MOONCAKE_BANDWIDTH_GBPS",
        "MN_USE_HIGH_SPEED_IB",
        "MN_ENABLE_FABRIC_AFFINITY",
        "AFLEX_DISABLE_RUNTIME_DVFS",
        "AFLEX_FREEZE_ACTIVE_FREQ",
        "AFLEX_SPLIT_PD_NODES",
        "AFLEX_PREFER_NODE2",
        "AFLEX_GPU_POOL",
        "AFLEX_SINGLE_HOST",
    ):
        os.environ.pop(key, None)
    for env, value in cfg.get("environment", {}).items():
        os.environ[env] = str(value)
    for i, n in enumerate(cfg["nodes"], 1):
        os.environ[f"MN_NODE{i}_MOONCAKE_IP_BASE"] = n["mooncake_ip_base"]
        os.environ[f"AFLEX_NODE{i}_GPU_POOL"] = ",".join(map(str, n["gpus"]))
    for module in (base, af):
        module.ROUTER_PORT = cfg["ports"]["router"]
        module.NCCL_PORT_BASE = cfg["ports"]["nccl"]
    base.NATIVE_PORT_BASE = cfg["ports"]["native"]
    base.PD_PREFILL_PORT_BASE = cfg["ports"]["prefill"]
    base.PD_DECODE_PORT_BASE = cfg["ports"]["decode"]
    base.PD_BOOTSTRAP_PORT_BASE = cfg["ports"]["bootstrap"]
    af.BOOTSTRAP_PORT_BASE = cfg["ports"]["bootstrap"]
    af.PREFILL_PORT_BASE = cfg["ports"]["prefill"]
    af.DECODE_PORT_BASE = cfg["ports"]["decode"]
    af.SUB_ROUTER_BASE = cfg["ports"]["router"]
    af.SCHED_PORT_BASE = cfg["ports"]["sched"]
    af.UCX_PORT_BASE = cfg["ports"]["ucx"]
    from . import requests as replay

    replay.TTFT_SLO_MS = cfg["slo"]["eval_ttft_ms"]
    replay.TPOT_SLO_MS = cfg["slo"]["eval_tpot_ms"]
    return rt, af, base


def deploy(cfg, point, modules):
    rt, af, base = modules
    if point["scheme"] == "aflex_tier1":
        config = parse_plan(local_path(cfg, point["plan"]))
        alloc = af.plan_allocation(config)
        urls = af.deploy(config)
        return urls, af.derive_active_energy_hosts(alloc), config
    scheme = {
        "native_tp1_baseline": "sglang",
        "native_tp1_tier": "dynamollm",
        "pd_hetero_baseline": "distserve",
        "pd_hetero_tier_biscale": "biscale",
    }[point["scheme"]]
    url = (
        base.deploy_native_tp1(scheme)
        if scheme in ("sglang", "dynamollm")
        else base.deploy_pd(scheme)
    )
    return [url] if url else None, {n["host"]: n["gpus"] for n in cfg["nodes"]}, None


def render(cfg):
    import time
    from unittest.mock import patch

    modules = configure(cfg, True)
    # Exercise the same real launch paths with transport and sleeping disabled.
    with patch.object(time, "sleep", lambda _: None):
        for point in cfg["points"]:
            deploy(cfg, point, modules)
    return modules[0].COMMANDS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "dry-run", "run"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        if name == "run":
            p.add_argument(
                "--execute",
                action="store_true",
                help="Explicitly allow SSH/docker/GPU changes",
            )
            p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    cfg = load(args.config)
    if args.command in ("preflight", "dry-run"):
        commands = render(cfg)
        print(
            json.dumps(
                (
                    commands
                    if args.command == "dry-run"
                    else {
                        "status": "OFFLINE_OK",
                        "points": len(cfg["points"]),
                        "commands": len(commands),
                        "remote_assets": "NOT_CHECKED",
                    }
                ),
                indent=2,
            )
        )
        return
    if not args.execute:
        parser.error("run requires --execute")
    from .requests import run_requests

    modules = configure(cfg)
    rt, af, _ = modules
    output = Path(args.output)
    if output.exists():
        parser.error("output already exists; choose a new result path")
    rt.remote_preflight()
    all_results = []
    try:
        for point in cfg["points"]:
            if point["scheme"] == "aflex_tier1":
                scope = af.derive_active_energy_hosts(
                    af.plan_allocation(parse_plan(local_path(cfg, point["plan"])))
                )
            else:
                scope = {n["host"]: n["gpus"] for n in cfg["nodes"]}
            try:
                urls, scope, plan = deploy(cfg, point, modules)
                if not urls or not rt.test_generate(urls[0]):
                    raise RuntimeError("deployment or warmup failed")
                rows = requests(local_path(cfg, point["workload"]))
                targets = (
                    af.assign_prefill_urls(urls, plan, len(rows))
                    if plan
                    else [urls[0] + "/generate"] * len(rows)
                )
                for row, target in zip(rows, targets):
                    row["_target_url"] = target
                window = int(min(max(400, rows[-1]["arrival_time_s"] + 150), 900))
                summary = asyncio.run(run_requests(rows, rt, scope, window))
                all_results.append(
                    {"point": point, "energy_hosts": scope, "summary": summary}
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps({"config": cfg, "results": all_results}, indent=2)
                )
            finally:
                rt.finalize_scope(scope)
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
