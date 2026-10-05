from __future__ import annotations

import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable


TENANT_KEY_PREFIX = "hybrid-tenant:"


class TenantBatchTracker:
    """Record scheduler batch shape without synchronizing the GPU."""

    def __init__(
        self,
        *,
        tenant_key_prefix: str = TENANT_KEY_PREFIX,
        max_events: int = 10000,
    ) -> None:
        self.tenant_key_prefix = tenant_key_prefix
        self.max_events = max_events
        self.started_at = time.perf_counter()
        self.events: list[dict] = []
        self.dropped_events = 0
        self._last_launch_ts: float | None = None
        self._last_mode: str | None = None
        self.capture_events = os.environ.get("SGLANG_TENANT_EVENT_TRACE") == "1"
        self.capture_progress = self.capture_events and (
            os.environ.get("SGLANG_TENANT_PROGRESS_TRACE") == "1"
        )
        self.admission_blocks: list[dict] = []
        self.dropped_admission_blocks = 0

    def record_admission_block(self, details: dict) -> None:
        if not self.capture_events:
            return
        if len(self.admission_blocks) >= self.max_events:
            self.dropped_admission_blocks += 1
            return
        self.admission_blocks.append(
            {"observed_s": time.perf_counter() - self.started_at, **details}
        )

    def record(self, batch, result, *, waiting_requests: int) -> None:
        raw_launch_ts = getattr(batch, "launch_ts", None)
        launch_ts = (
            time.perf_counter() if raw_launch_ts is None else float(raw_launch_ts)
        )
        mode_value = getattr(batch, "forward_mode", None)
        mode = getattr(mode_value, "name", str(mode_value or "unknown")).lower()
        tenant_counts = Counter()
        for req in getattr(batch, "reqs", ()):
            tenant = getattr(req, "extra_key", None)
            if isinstance(tenant, str) and tenant.startswith(self.tenant_key_prefix):
                tenant_counts[tenant] += 1

        if not tenant_counts:
            return
        if len(self.events) >= self.max_events:
            self.dropped_events += 1
            return

        gap_us = None
        if self._last_launch_ts is not None and launch_ts >= self._last_launch_ts:
            gap_us = int((launch_ts - self._last_launch_ts) * 1e6)
        decoding_reqs = getattr(batch, "decoding_reqs", None)
        self.events.append(
            {
                "launch_s": launch_ts - self.started_at,
                "mode": mode,
                "batch_size": len(getattr(batch, "reqs", ())),
                "extend_tokens": int(
                    getattr(batch, "extend_num_tokens", 0) or 0
                ),
                "decoding_requests": len(decoding_reqs) if decoding_reqs else 0,
                "generated_tokens": int(
                    len(getattr(batch, "reqs", ()))
                    + int(getattr(result, "num_correct_drafts", 0) or 0)
                ),
                "waiting_requests": int(waiting_requests),
                "tenant_counts": dict(sorted(tenant_counts.items())),
                "previous_mode": self._last_mode,
                "since_previous_launch_us": gap_us,
            }
        )
        if self.capture_progress:
            # The result path may see mutated Req counters under overlap. Use
            # the batch's captured Python lists for the work actually submitted.
            decoding_ids = {req.rid for req in decoding_reqs or ()}
            extend_lens = getattr(batch, "extend_lens", None)
            prefix_lens = getattr(batch, "prefix_lens", None)
            members = []
            for index, req in enumerate(getattr(batch, "reqs", ())):
                tenant = getattr(req, "extra_key", None)
                if not isinstance(tenant, str) or not tenant.startswith(
                    self.tenant_key_prefix
                ):
                    continue
                decode = mode == "decode" or req.rid in decoding_ids
                work = 1 if decode else (
                    int(extend_lens[index]) if extend_lens is not None else None
                )
                prefix = None if decode or prefix_lens is None else int(prefix_lens[index])
                members.append({
                    "rid": req.rid, "tenant": tenant,
                    "phase": "decode" if decode else "prefill",
                    "work_tokens": work, "prefix_tokens": prefix,
                })
            self.events[-1]["members"] = members
            self.events[-1]["recorded_s"] = time.perf_counter() - self.started_at
        self._last_launch_ts = launch_ts
        self._last_mode = mode

    def snapshot(self) -> dict:
        return {
            "schema_version": 1,
            "measurement_scope": (
                "tenant-labeled scheduler batches; launch gaps are CPU-observed "
                "and do not synchronize the GPU"
            ),
            "elapsed_seconds": time.perf_counter() - self.started_at,
            "event_count": len(self.events),
            "dropped_events": self.dropped_events,
            "events": self.events,
            "event_clock_origin_s": self.started_at,
            "progress_trace_enabled": self.capture_progress,
            "progress_scope": (
                "Per-request submitted work, captured from batch lists after "
                "result processing; launch times are CPU observations, not GPU durations."
            ),
            "admission_blocks": self.admission_blocks,
            "dropped_admission_blocks": self.dropped_admission_blocks,
        }


@dataclass
class TenantResourceUsage:
    active_request_seconds: float = 0.0
    state_slot_seconds: float = 0.0
    kv_token_seconds: float = 0.0
    peak_active_requests: int = 0
    peak_state_slots: int = 0
    peak_kv_tokens: int = 0
    current_active_requests: int = 0
    current_state_slots: int = 0
    current_kv_tokens: int = 0


class TenantResourceTracker:
    """Integrate active request resources between scheduler lifecycle events.

    The tracker deliberately excludes waiting requests and inactive radix-cache
    entries. ``Req.kv.kv_allocated_len`` is allocator state, so this avoids the
    streaming-response timing approximation used by the black-box benchmark.
    """

    def __init__(
        self,
        *,
        tenant_key_prefix: str = TENANT_KEY_PREFIX,
        lazy_extra_buffer: bool = False,
        capture_events: bool | None = None,
        max_events: int = 10000,
    ) -> None:
        self.tenant_key_prefix = tenant_key_prefix
        self.lazy_extra_buffer = lazy_extra_buffer
        self.started_at = time.perf_counter()
        self.last_observed_at = self.started_at
        self.observation_count = 0
        self._current: dict[str, tuple[int, int, int]] = {}
        self._usage: dict[str, TenantResourceUsage] = {}
        self._system_usage = TenantResourceUsage()
        self.capture_events = (
            os.environ.get("SGLANG_TENANT_EVENT_TRACE") == "1"
            if capture_events is None else capture_events
        )
        self.max_events = max_events
        self.events: list[dict] = []
        self.dropped_events = 0
        self._members: set = set()
        self._event_state = 0
        self._event_kv = 0

    def _state_slots(self, req) -> int:
        slots = int(getattr(req, "mamba_pool_idx", None) is not None)
        ping_pong = getattr(req, "mamba_ping_pong_track_buffer", None)
        if ping_pong is None:
            return slots

        # Normal extra-buffer mode allocates every element. Lazy mode leaves
        # -1 sentinels, for which tracing explicitly accepts the device sync.
        if self.lazy_extra_buffer:
            return slots + sum(int(index) >= 0 for index in ping_pong.tolist())
        return slots + int(ping_pong.numel())

    @staticmethod
    def _kv_tokens(req) -> int:
        kv = getattr(req, "kv", None)
        if kv is None:
            return 0
        return max(
            0,
            int(kv.kv_allocated_len) - int(getattr(kv, "swa_evicted_seqlen", 0)),
        )

    def observe(self, reqs: Iterable, *, now: float | None = None) -> None:
        now = time.perf_counter() if now is None else now
        if now < self.last_observed_at:
            raise ValueError("tenant resource observations must be monotonic")

        duration = now - self.last_observed_at
        for tenant, (active_reqs, state_slots, kv_tokens) in self._current.items():
            usage = self._usage[tenant]
            usage.active_request_seconds += active_reqs * duration
            usage.state_slot_seconds += state_slots * duration
            usage.kv_token_seconds += kv_tokens * duration
        old_active = sum(values[0] for values in self._current.values())
        old_state = sum(values[1] for values in self._current.values())
        old_kv = sum(values[2] for values in self._current.values())
        self._system_usage.active_request_seconds += old_active * duration
        self._system_usage.state_slot_seconds += old_state * duration
        self._system_usage.kv_token_seconds += old_kv * duration

        current = defaultdict(lambda: [0, 0, 0])
        members = {}
        seen_rids = set()
        for req in reqs:
            rid = getattr(req, "rid", id(req))
            if rid in seen_rids:
                continue
            seen_rids.add(rid)

            tenant = getattr(req, "extra_key", None)
            if not isinstance(tenant, str) or not tenant.startswith(
                self.tenant_key_prefix
            ):
                continue

            state_slots = self._state_slots(req)
            kv_tokens = self._kv_tokens(req)
            if state_slots == 0 and kv_tokens == 0:
                continue
            current[tenant][0] += 1
            current[tenant][1] += state_slots
            current[tenant][2] += kv_tokens
            if self.capture_events:
                members[rid] = {"rid": rid, "tenant": tenant,
                                "state_slots": state_slots, "kv_tokens": kv_tokens}

        self._current = {tenant: tuple(values) for tenant, values in current.items()}
        system_active = sum(values[0] for values in self._current.values())
        system_state = sum(values[1] for values in self._current.values())
        system_kv = sum(values[2] for values in self._current.values())
        if self.capture_events:
            added = set(members) - self._members
            removed = self._members - set(members)
            if (added or removed or system_state != self._event_state
                    or abs(system_kv - self._event_kv) >= 2048):
                if len(self.events) < self.max_events:
                    self.events.append({
                        "observed_s": now - self.started_at,
                        "active_requests": system_active,
                        "state_slots": system_state, "kv_tokens": system_kv,
                        "added": [members[rid] for rid in sorted(added)],
                        "removed": sorted(removed),
                        "tenants": {k: list(v) for k, v in self._current.items()},
                    })
                else:
                    self.dropped_events += 1
                self._event_state, self._event_kv = system_state, system_kv
            self._members = set(members)
        self._system_usage.current_active_requests = system_active
        self._system_usage.current_state_slots = system_state
        self._system_usage.current_kv_tokens = system_kv
        self._system_usage.peak_active_requests = max(
            self._system_usage.peak_active_requests, system_active
        )
        self._system_usage.peak_state_slots = max(
            self._system_usage.peak_state_slots, system_state
        )
        self._system_usage.peak_kv_tokens = max(
            self._system_usage.peak_kv_tokens, system_kv
        )
        for tenant, (active_reqs, state_slots, kv_tokens) in self._current.items():
            usage = self._usage.setdefault(tenant, TenantResourceUsage())
            usage.current_active_requests = active_reqs
            usage.current_state_slots = state_slots
            usage.current_kv_tokens = kv_tokens
            usage.peak_active_requests = max(usage.peak_active_requests, active_reqs)
            usage.peak_state_slots = max(usage.peak_state_slots, state_slots)
            usage.peak_kv_tokens = max(usage.peak_kv_tokens, kv_tokens)

        for tenant, usage in self._usage.items():
            if tenant not in self._current:
                usage.current_active_requests = 0
                usage.current_state_slots = 0
                usage.current_kv_tokens = 0

        self.last_observed_at = now
        self.observation_count += 1

    def snapshot(self) -> dict:
        return {
            "schema_version": 2,
            "measurement_scope": "active requests only; inactive radix cache excluded",
            "state_unit": "physically assigned recurrent-state slots",
            "kv_unit": "allocated active KV token slots",
            "elapsed_seconds": self.last_observed_at - self.started_at,
            "observation_count": self.observation_count,
            "system": asdict(self._system_usage),
            "tenants": {
                tenant: asdict(usage)
                for tenant, usage in sorted(self._usage.items())
            },
            "event_clock_origin_s": self.started_at,
            "event_trace_enabled": self.capture_events,
            "events": self.events,
            "dropped_events": self.dropped_events,
        }

    def normalized_dominant_debts(
        self, *, state_capacity: int, kv_capacity: int
    ) -> dict[str, float]:
        if state_capacity <= 0 or kv_capacity <= 0:
            raise ValueError("tenant resource capacities must be positive")
        return {
            tenant: max(
                usage.state_slot_seconds / state_capacity,
                usage.kv_token_seconds / kv_capacity,
            )
            for tenant, usage in self._usage.items()
        }

    def current_normalized_dominant_shares(
        self, *, state_capacity: int, kv_capacity: int
    ) -> dict[str, float]:
        """Return each tenant's instantaneous dominant active-resource share."""
        if state_capacity <= 0 or kv_capacity <= 0:
            raise ValueError("tenant resource capacities must be positive")
        return {
            tenant: max(
                usage.current_state_slots / state_capacity,
                usage.current_kv_tokens / kv_capacity,
            )
            for tenant, usage in self._usage.items()
        }

    def normalized_lifecycle_costs(
        self,
        *,
        state_capacity: int,
        kv_capacity: int,
        state_weight: float = 1.0,
        kv_weight: float = 1.0,
    ) -> dict[str, float]:
        """Return scalar realized resource-time charged at capacity-normalized prices."""
        if state_capacity <= 0 or kv_capacity <= 0:
            raise ValueError("tenant resource capacities must be positive")
        if state_weight < 0 or kv_weight < 0:
            raise ValueError("tenant lifecycle weights must be non-negative")
        if state_weight == 0 and kv_weight == 0:
            raise ValueError("at least one tenant lifecycle weight must be positive")
        return {
            tenant: (
                state_weight * usage.state_slot_seconds / state_capacity
                + kv_weight * usage.kv_token_seconds / kv_capacity
            )
            for tenant, usage in self._usage.items()
        }
