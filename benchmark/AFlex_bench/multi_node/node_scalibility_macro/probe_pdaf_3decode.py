#!/usr/bin/env python3
"""Deploy PDAF 3-Decode and probe each sub-router + top-level router once."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

os.environ.setdefault("MN_NODE1_IP", "10.252.129.34")
os.environ.setdefault("MN_NODE2_IP", "10.252.129.33")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_conv_pdaf_3decode as P3D  # noqa: E402
import run_macro_benchmark as RMB  # noqa: E402


def post_once(name, url):
    payload = {"text": "Hello", "sampling_params": {"max_new_tokens": 8, "temperature": 0}}
    t0 = time.time()
    try:
        r = requests.post(f"{url}/generate", json=payload, timeout=45)
        return {"target": name, "status": r.status_code,
                "latency_s": round(time.time() - t0, 3), "body_prefix": r.text[:160]}
    except Exception as e:
        return {"target": name, "error": repr(e), "latency_s": round(time.time() - t0, 3)}


def main():
    import resource
    resource.setrlimit(resource.RLIMIT_MEMLOCK,
                       (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    RMB.cleanup_all()
    url = P3D.start_pdaf_3decode(True)
    print("DEPLOY_URL", url, flush=True)
    results = []
    if url is not None:
        targets = [
            ("d0", f"http://{RMB.NODE1_IP}:{P3D.SUB_ROUTER_PORTS[0]}"),
            ("d1", f"http://{RMB.NODE1_IP}:{P3D.SUB_ROUTER_PORTS[1]}"),
            ("d2", f"http://{RMB.NODE1_IP}:{P3D.SUB_ROUTER_PORTS[2]}"),
            ("top", url), ("top", url), ("top", url),
        ]
        for name, u in targets:
            item = post_once(name, u)
            results.append(item)
            print("RESULT", json.dumps(item), flush=True)
            time.sleep(1)
    (HERE / "results" / "probe_pdaf_3decode.json").write_text(
        json.dumps({"deployed_url": url, "results": results}, indent=2))
    print("SAVED results/probe_pdaf_3decode.json", flush=True)
    RMB.cleanup_all()


if __name__ == "__main__":
    main()
