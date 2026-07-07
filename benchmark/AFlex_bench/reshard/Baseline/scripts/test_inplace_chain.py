#!/usr/bin/env python3
"""Quick in-place reshard chain: TP1 -> TP2 -> TP4 -> TP8 with /generate smoke."""
import argparse
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inplace_reshard_server_ctl import stop_inplace_reshard_server

BASE = "http://127.0.0.1:31700"


def wait_reshard(target_tp: int, timeout: float = 600.0) -> bool:
    t0 = time.time()
    st = {}
    while time.time() - t0 < timeout:
        st = requests.get(BASE + "/inplace_reshard_status", timeout=5).json()
        ph = st.get("phase")
        if ph == "done" and st.get("active_tp") == target_tp:
            msg = (st.get("message") or "")[:100]
            print(f"  reshard done TP{target_tp} in {time.time() - t0:.1f}s: {msg}")
            return True
        if ph == "failed":
            print(f"  FAILED: {st}")
            return False
        if ph == "executing" and int(time.time() - t0) % 10 == 0:
            print(
                f"  ... executing {time.time() - t0:.0f}s "
                f"active={st.get('active_tp')} target={target_tp}",
                flush=True,
            )
        time.sleep(2)
    print(f"  TIMEOUT waiting for TP{target_tp}, last={st}")
    return False


def gen(tag: str, n: int = 2) -> int:
    ok = 0
    for k in range(n):
        t = time.time()
        try:
            r = requests.post(
                BASE + "/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {"max_new_tokens": 8, "temperature": 0},
                },
                timeout=120,
            )
            txt = r.json().get("text", "")
            good = r.status_code == 200 and bool(txt.strip())
            status = "OK" if good else "FAIL"
            print(f"  {tag} #{k}: {r.status_code} {time.time() - t:.2f}s {txt[:60]!r} {status}")
            ok += int(good)
        except Exception as e:
            print(f"  {tag} #{k} ERR: {e}")
    return ok


def reshard(new_tp: int) -> bool:
    print(f"\n=== Reshard -> TP{new_tp} ===")
    r = requests.post(BASE + "/reshard_tp", json={"new_tp_size": new_tp}, timeout=10)
    print(f"  trigger: {r.status_code} {r.text[:120]}")
    return wait_reshard(new_tp)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--keep-server",
        action="store_true",
        help="Do not stop sglang after the test (default: stop)",
    )
    ap.add_argument("--port", type=int, default=31700)
    args = ap.parse_args()
    global BASE
    BASE = f"http://127.0.0.1:{args.port}"

    rc = 0
    try:
        print("=== TP1 generate ===")
        if gen("TP1") < 1:
            rc = 1
        else:
            for tp in [2, 4, 8]:
                if not reshard(tp):
                    rc = 2
                    break
                if gen(f"TP{tp}") < 1:
                    rc = 3
                    break
            else:
                print("\n=== ALL PASSED ===")
    finally:
        if not args.keep_server:
            stop_inplace_reshard_server(args.port)
    return rc


if __name__ == "__main__":
    sys.exit(main())
