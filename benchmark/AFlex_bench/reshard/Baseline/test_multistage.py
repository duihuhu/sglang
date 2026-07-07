import time, requests, sys
from pathlib import Path

LOG = "/workspace/sglang/benchmark/AFlex_bench/reshard/Baseline/logs/inplace_multistage.log"
BASE = "http://127.0.0.1:31700"


def wait_ready(timeout=200):
    p = Path(LOG)
    for _ in range(timeout // 2):
        if p.exists() and "fired up" in p.read_text(errors="replace"):
            return True
        time.sleep(2)
    return False


def gen(tag, n=3):
    ok = 0
    for k in range(n):
        t = time.time()
        try:
            r = requests.post(
                BASE + "/generate",
                json={"text": "The capital of France is", "sampling_params": {"max_new_tokens": 16, "temperature": 0}},
                timeout=90,
            )
            txt = r.json().get("text", "")
            print("  %s #%d: %s %.2fs | %r" % (tag, k, r.status_code, time.time() - t, txt[:70]))
            if r.status_code == 200 and txt.strip():
                ok += 1
        except Exception as e:
            print("  %s #%d ERR %.2fs %r" % (tag, k, time.time() - t, repr(e)[:50]))
    return ok


def reshard(new_tp):
    r = requests.post(BASE + "/reshard_tp", json={"new_tp_size": new_tp}, timeout=10)
    print("reshard -> TP%d: %s" % (new_tp, r.status_code))
    time.sleep(20)


if not wait_ready():
    print("NOT READY")
    sys.exit(1)
print("=== ready ===")
print("TP1 baseline OK:", gen("TP1"))
reshard(2)
print("TP2 OK:", gen("TP2"))
reshard(4)
print("TP4 OK:", gen("TP4"))
