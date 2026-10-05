from types import SimpleNamespace

import pytest

from sglang.srt.managers.scheduler_components.tenant_resource_tracker import (
    TenantBatchTracker,
    TenantResourceTracker,
)


class _Buffer:
    def __init__(self, values):
        self.values = values

    def numel(self):
        return len(self.values)

    def tolist(self):
        return self.values


def _req(rid, tenant, state=True, ping_pong=(1, 2), kv_tokens=0, evicted=0):
    return SimpleNamespace(
        rid=rid,
        extra_key=tenant,
        mamba_pool_idx=object() if state else None,
        mamba_ping_pong_track_buffer=(
            _Buffer(ping_pong) if ping_pong is not None else None
        ),
        mamba_lazy_extra_buffer=False,
        kv=(
            SimpleNamespace(
                kv_allocated_len=kv_tokens, swa_evicted_seqlen=evicted
            )
            if kv_tokens
            else None
        ),
    )


def test_integrates_exact_resource_time_and_deduplicates_requests():
    tracker = TenantResourceTracker()
    tracker.started_at = tracker.last_observed_at = 10.0
    tenant = "hybrid-tenant:run-1:state-heavy"
    first = _req("r1", tenant, kv_tokens=100)
    second = _req("r2", tenant, kv_tokens=60, evicted=10)

    tracker.observe([first, first], now=11.0)
    tracker.observe([first, second], now=13.0)
    tracker.observe([], now=14.0)

    usage = tracker.snapshot()["tenants"][tenant]
    assert usage["active_request_seconds"] == pytest.approx(4.0)
    assert usage["state_slot_seconds"] == pytest.approx(12.0)
    assert usage["kv_token_seconds"] == pytest.approx(350.0)
    assert usage["peak_active_requests"] == 2
    assert usage["peak_state_slots"] == 6
    assert usage["peak_kv_tokens"] == 150
    assert usage["current_active_requests"] == 0
    system = tracker.snapshot()["system"]
    assert system["active_request_seconds"] == pytest.approx(4.0)
    assert system["state_slot_seconds"] == pytest.approx(12.0)
    assert system["kv_token_seconds"] == pytest.approx(350.0)
    assert system["peak_active_requests"] == 2
    assert system["peak_state_slots"] == 6
    assert system["peak_kv_tokens"] == 150


def test_ignores_unlabeled_and_inactive_requests():
    tracker = TenantResourceTracker()
    tracker.started_at = tracker.last_observed_at = 1.0
    tracker.observe(
        [
            _req("plain", None, kv_tokens=10),
            _req(
                "inactive",
                "hybrid-tenant:run-1:kv-heavy",
                state=False,
                ping_pong=None,
                kv_tokens=0,
            ),
        ],
        now=2.0,
    )

    assert tracker.snapshot()["tenants"] == {}


def test_normalized_lifecycle_costs_apply_capacity_and_prices():
    tracker = TenantResourceTracker()
    tracker.started_at = tracker.last_observed_at = 10.0
    tenant = "hybrid-tenant:run-1:mixed"
    request = _req("r1", tenant, kv_tokens=100)
    tracker.observe([request], now=11.0)
    tracker.observe([], now=13.0)

    costs = tracker.normalized_lifecycle_costs(
        state_capacity=6,
        kv_capacity=200,
        state_weight=2.0,
        kv_weight=0.5,
    )

    assert costs[tenant] == pytest.approx(2.5)


def test_current_normalized_dominant_shares_use_active_snapshot():
    tracker = TenantResourceTracker()
    tracker.started_at = tracker.last_observed_at = 10.0
    tenant = "hybrid-tenant:run-1:state-heavy"
    tracker.observe([_req("r1", tenant, kv_tokens=20)], now=11.0)

    shares = tracker.current_normalized_dominant_shares(
        state_capacity=6, kv_capacity=200
    )
    assert shares[tenant] == pytest.approx(0.5)

    tracker.observe([], now=12.0)
    shares = tracker.current_normalized_dominant_shares(
        state_capacity=6, kv_capacity=200
    )
    assert shares[tenant] == 0.0


def test_batch_tracker_records_tenant_shape_and_launch_gap():
    tracker = TenantBatchTracker()
    tracker.started_at = 10.0
    tenant = "hybrid-tenant:run-1:split"
    batch = SimpleNamespace(
        launch_ts=11.0,
        forward_mode=SimpleNamespace(name="DECODE"),
        reqs=[_req("r1", tenant), _req("plain", None)],
        extend_num_tokens=0,
        decoding_reqs=None,
    )
    result = SimpleNamespace(num_correct_drafts=0)

    tracker.record(batch, result, waiting_requests=3)
    batch.launch_ts = 11.25
    tracker.record(batch, result, waiting_requests=2)

    trace = tracker.snapshot()
    assert trace["event_count"] == 2
    assert trace["events"][0]["tenant_counts"] == {tenant: 1}
    assert trace["events"][0]["waiting_requests"] == 3
    assert trace["events"][1]["previous_mode"] == "decode"
    assert trace["events"][1]["since_previous_launch_us"] == 250000
