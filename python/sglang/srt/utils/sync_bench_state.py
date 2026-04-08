import threading

_SYNC_BENCH_ACTIVE = False
_SYNC_BENCH_LOCK = threading.Lock()


def set_sync_bench_active(active: bool) -> None:
    global _SYNC_BENCH_ACTIVE
    with _SYNC_BENCH_LOCK:
        _SYNC_BENCH_ACTIVE = bool(active)


def is_sync_bench_active() -> bool:
    with _SYNC_BENCH_LOCK:
        return _SYNC_BENCH_ACTIVE
