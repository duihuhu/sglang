"""Low-overhead, time-binned pressure records for Central I/O host KV.

This deliberately records aggregate physical page activity rather than a log
entry for every token or range.  It is an observability aid only; it does not
participate in allocation, eviction, or scheduling.
"""

from __future__ import annotations

import json
from pathlib import Path
import time


class CentralIOPressureLogger:
    """Append one compact physical-host-KV record per time bin."""

    def __init__(
        self,
        path: str | Path | None,
        *,
        model_id: str,
        page_size: int,
        interval_s: float = 1.0,
        clock=time.monotonic,
        wall_clock=time.time,
    ) -> None:
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self.path = None if not path else Path(path)
        self.model_id = model_id
        self.page_size = page_size
        self.interval_s = interval_s
        self.clock = clock
        self.wall_clock = wall_clock
        self._bucket_start_s: float | None = None
        self._deltas = self._empty_deltas()

    @staticmethod
    def _empty_deltas() -> dict[str, int]:
        return {
            "alloc_pages": 0,
            "release_pages": 0,
            "host_evict_slots": 0,
            # ``host_evict_slots`` is the physical total.  Keep the three
            # semantic sources separate so a value-aware reclaim wave cannot
            # be mistaken for SGLang's fallback admission eviction.
            "local_value_reclaim_slots": 0,
            "fallback_admission_evict_slots": 0,
            "donor_drain_slots": 0,
            "backup_slots": 0,
            "restore_slots": 0,
            "quota_change_pages": 0,
            # A prefix that was locally reclaimed and then rebuilt is the
            # direct evidence that the current quota cannot retain its useful
            # working set. Keep it alongside physical page pressure.
            "retention_loss_tokens": 0,
            # A local value-aware reclaim wave could not produce enough
            # writable host pages for an imminent HBM backup. This is the
            # early pressure signal used before a later revisit proves loss.
            "admission_shortfall_pages": 0,
        }

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def record(
        self,
        kind: str,
        amount: int,
        *,
        live_pages: int,
        active_pages: int,
        clean_free_pages: int,
        draining_pages: int,
        now_s: float | None = None,
    ) -> None:
        if not self.enabled:
            return
        if kind not in self._deltas:
            raise ValueError(f"unknown pressure event: {kind}")
        if min(amount, live_pages, active_pages, clean_free_pages, draining_pages) < 0:
            raise ValueError("pressure values must be non-negative")
        now_s = self.clock() if now_s is None else now_s
        if self._bucket_start_s is None:
            self._bucket_start_s = now_s
        elif now_s - self._bucket_start_s >= self.interval_s:
            self._flush(
                now_s,
                live_pages=live_pages,
                active_pages=active_pages,
                clean_free_pages=clean_free_pages,
                draining_pages=draining_pages,
            )
            self._bucket_start_s = now_s
        self._deltas[kind] += amount

    def flush(
        self,
        *,
        live_pages: int,
        active_pages: int,
        clean_free_pages: int,
        draining_pages: int,
        now_s: float | None = None,
    ) -> None:
        if not self.enabled or self._bucket_start_s is None:
            return
        self._flush(
            self.clock() if now_s is None else now_s,
            live_pages=live_pages,
            active_pages=active_pages,
            clean_free_pages=clean_free_pages,
            draining_pages=draining_pages,
        )
        self._bucket_start_s = None

    def _flush(
        self,
        now_s: float,
        *,
        live_pages: int,
        active_pages: int,
        clean_free_pages: int,
        draining_pages: int,
    ) -> None:
        assert self.path is not None
        record = {
            "schema": "central_io_pressure_v2",
            "monotonic_time_s": now_s,
            "wall_time_s": self.wall_clock(),
            "model_id": self.model_id,
            "page_size_tokens": self.page_size,
            "live_pages": live_pages,
            "active_quota_pages": active_pages,
            "clean_free_pages": clean_free_pages,
            "draining_pages": draining_pages,
            **self._deltas,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._deltas = self._empty_deltas()
