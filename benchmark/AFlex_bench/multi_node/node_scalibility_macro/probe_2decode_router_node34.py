#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

os.environ["MN_NODE1_IP"] = "10.252.129.34"
os.environ["MN_NODE2_IP"] = "10.252.129.33"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_conv_pdaf_2decode as P2D  # noqa: E402
import run_macro_benchmark as RMB  # noqa: E402


def post_once(name: str, url: str) -> dict:
    payload = {"text": "Hello", "sampling_params": {"max_new_tokens": 8, "temperature": 0}}
    t0 = time.time()
    try:
        r = requests.post(f"{url}/generate", json=payload, timeout=45)
        return {
            "target": name,
            "status": r.status_code,
            "latency_s": round(time.time() - t0, 3),
            "body_prefix": r.text[:200],
        }
    except Exception as exc:
        return {"target": name, "error": repr(exc), "latency_s": round(time.time() - t0, 3)}


def main():
    RMB.cleanup_all()
    deployed_url = P2D.start_pdaf_2decode(True)
    print("DEPLOY_URL", deployed_url, flush=True)
    results = []
    if deployed_url is not None:
        for name, url in [
            ("d0", "http://10.252.129.34:42040"),
            ("d1", "http://10.252.129.34:42041"),
            ("top", "http://10.252.129.34:42000"),
        ]:
            item = post_once(name, url)
            results.append(item)
            print("RESULT", json.dumps(item), flush=True)
            time.sleep(2)
    out = HERE / "results" / "conv_aflex_2decode_direct_subrouter_probe.json"
    out.write_text(json.dumps({"deployed_url": deployed_url, "results": results}, indent=2))
    print("SAVED", out, flush=True)
    RMB.cleanup_all()


if __name__ == "__main__":
    main()
