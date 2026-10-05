from types import SimpleNamespace

from sglang.srt.managers.schedule_policy import SchedulePolicy


class _Req:
    def __init__(self, rid, tenant, input_len):
        self.rid = rid
        self.extra_key = tenant
        self.origin_input_ids = [1] * input_len
        self.output_ids = []
        self._finished = False

    def finished(self):
        return self._finished


class _Tracker:
    def current_normalized_dominant_shares(self, *, state_capacity, kv_capacity):
        assert state_capacity == 10
        assert kv_capacity == 100
        return {"tenant-a": 0.8, "tenant-b": 0.2}

    def normalized_dominant_debts(self, *, state_capacity, kv_capacity):
        assert state_capacity == 10
        assert kv_capacity == 100
        return {"tenant-a": 2.0, "tenant-b": 1.0}

    def normalized_lifecycle_costs(
        self, *, state_capacity, kv_capacity, state_weight, kv_weight
    ):
        assert state_capacity == 10
        assert kv_capacity == 100
        assert state_weight == 2.0
        assert kv_weight == 0.5
        return {"tenant-a": 3.0, "tenant-b": 1.5}


def _policy(name, tracker=None):
    return SchedulePolicy(
        name,
        tree_cache=SimpleNamespace(),
        enable_hierarchical_cache=False,
        enable_priority_scheduling=False,
        schedule_low_priority_values_first=False,
        tenant_resource_tracker=tracker,
        state_capacity=10,
        kv_capacity=100,
    )


def test_vtc_constructs_batch_by_iterative_minimum_counter():
    policy = _policy("tenant-vtc")
    a1 = _Req("a1", "tenant-a", 4)
    a2 = _Req("a2", "tenant-a", 4)
    b1 = _Req("b1", "tenant-b", 8)
    waiting = []
    for req in (a1, a2, b1):
        policy.on_request_queued(req, waiting)
        waiting.append(req)

    policy.calc_priority(waiting)
    assert [req.rid for req in waiting] == ["a1", "b1", "a2"]

    admitted = waiting[:2]
    waiting = waiting[2:]
    policy.on_requests_admitted(admitted, waiting)
    assert policy.tenant_counters == {"tenant-a": 4.0, "tenant-b": 8.0}

    a1.output_ids.append(10)
    b1.output_ids.extend([10, 11])
    policy.on_batch_processed(SimpleNamespace(reqs=[a1, b1]))
    assert policy.tenant_counters == {"tenant-a": 5.0, "tenant-b": 10.0}


def test_vtc_profiled_polynomial_cost_and_trace():
    policy = _policy("tenant-vtc")
    policy.vtc_request_weight = 3.0
    policy.vtc_input_weight = 2.0
    policy.vtc_input_quadratic_weight = 0.5
    policy.vtc_output_weight = 5.0
    policy.vtc_input_output_weight = 0.25
    policy.vtc_output_quadratic_weight = 0.75
    request = _Req("a1", "tenant-a", 4)
    waiting = []
    policy.on_request_queued(request, waiting)
    waiting.append(request)
    policy.on_requests_admitted([request], [])

    assert policy.tenant_counters["tenant-a"] == 19.0

    request.output_ids.extend([10, 11])
    policy.on_batch_processed(SimpleNamespace(reqs=[request]))
    # 2 * 5 + 4 * 2 * 0.25 + 2^2 * 0.75 = 15.
    assert policy.tenant_counters["tenant-a"] == 34.0
    trace = policy.vtc_snapshot()
    assert trace["observation_count"] == 3
    assert trace["counters"] == {"tenant-a": 34.0}
    assert trace["weights"]["input_output"] == 0.25
    assert trace["queue_waits"]["tenant-a"]["admitted_requests"] == 1
    assert trace["queue_waits"]["tenant-a"]["peak_waiting_requests"] == 1


def test_vector_debt_prioritizes_lower_normalized_dominant_debt():
    policy = _policy("tenant-vector-debt", tracker=_Tracker())
    a = _Req("a", "tenant-a", 4)
    b = _Req("b", "tenant-b", 4)
    waiting = [a, b]

    policy.calc_priority(waiting)

    assert [req.rid for req in waiting] == ["b", "a"]


def test_instant_share_prioritizes_lower_current_dominant_share():
    policy = _policy("tenant-instant-share", tracker=_Tracker())
    a = _Req("a", "tenant-a", 4)
    b = _Req("b", "tenant-b", 4)
    waiting = [a, b]

    policy.calc_priority(waiting)

    assert [req.rid for req in waiting] == ["b", "a"]


def test_lifecycle_cost_prioritizes_lower_realized_scalar_debt():
    policy = _policy("tenant-lifecycle-cost", tracker=_Tracker())
    policy.lifecycle_state_weight = 2.0
    policy.lifecycle_kv_weight = 0.5
    a = _Req("a", "tenant-a", 4)
    b = _Req("b", "tenant-b", 4)
    waiting = [a, b]

    policy.calc_priority(waiting)

    assert [req.rid for req in waiting] == ["b", "a"]
