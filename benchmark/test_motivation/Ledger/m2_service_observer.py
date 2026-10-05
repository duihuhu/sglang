"""Opt-in CPU-only observation hooks; no scheduling decisions are replaced."""

from collections import Counter
from contextvars import ContextVar
from functools import wraps
import math
import os
from pathlib import Path
import time


PREFIX = "hybrid-tenant:"
MAX_MEMBERS = 2048


def tenant(req):
    return getattr(req, "extra_key", None) or "__default_tenant__"


def member(req):
    return {"rid": req.rid, "tenant": tenant(req),
            "input_tokens": len(req.origin_input_ids), "output_tokens": len(req.output_ids),
            "has_request_slot": getattr(req, "req_pool_idx", None) is not None,
            "has_state_slot": getattr(req, "mamba_pool_idx", None) is not None}


def counts(reqs):
    return dict(Counter(tenant(req) for req in reqs))


def cpu_number(value):
    # Never convert device tensors (including scalar tensors) for observation.
    return value if type(value) in (int, float, bool) and math.isfinite(value) else None


class ServiceRecorder:
    def __init__(self, scheduler, max_events=20000):
        if max_events < 1:
            raise ValueError("max_events must be positive")
        self.scheduler = scheduler
        self.policy = scheduler.policy
        self.started_at = time.perf_counter()
        self.max_events = max_events
        self.events = []
        self.sequence = self.dropped = self.truncated = self.decision = 0
        self.batch_id = None
        self.run_ids = set()
        self.examined = set()
        self.resource_cursor = 0

    def rows(self, reqs):
        if len(reqs) > MAX_MEMBERS:
            self.truncated += 1
        rows = [member(req) for req in reqs[:MAX_MEMBERS]]
        for row in rows:
            if row["tenant"].startswith(PREFIX):
                self.run_ids.add(row["tenant"][len(PREFIX):].rsplit(":", 1)[0])
        return rows

    def emit(self, kind, **data):
        self.sequence += 1
        if len(self.events) >= self.max_events:
            self.dropped += 1
            return
        self.events.append({"seq": self.sequence, "observed_s": time.perf_counter() - self.started_at,
                            "kind": kind, "decision_id": self.decision,
                            "batch_id": self.batch_id, **data})

    def pools(self):
        scheduler = self.scheduler
        pool = scheduler.req_to_token_pool
        mamba = getattr(pool, "mamba_allocator", None)
        return {"request_slots_free": cpu_number(pool.available_size()),
                "state_slots_free": cpu_number(mamba.available_size()) if mamba else None,
                "kv_tokens_free": cpu_number(scheduler.token_to_kv_pool_allocator.available_size())}

    def queue(self):
        return counts(self.scheduler.waiting_queue)

    def snapshot(self):
        ps = self.scheduler.ps
        shared = getattr(self.scheduler.req_to_token_pool, "_unified_buffer", None)
        boot_file = Path("/proc/sys/kernel/random/boot_id")
        return {"schema_version": 1, "enabled": True,
                "scope": "CPU hook observations, not kernel time or exact allocator release calls",
                "rank": {"pp": ps.pp_rank, "attn_tp": ps.attn_tp_rank, "attn_cp": ps.attn_cp_rank},
                "pid": os.getpid(), "boot_id": boot_file.read_text().strip() if boot_file.exists() else None,
                "clock": time.get_clock_info("perf_counter").implementation,
                "event_clock_origin_s": self.started_at, "run_ids": sorted(self.run_ids),
                "events": self.events, "dropped_events": self.dropped,
                "truncated_fields": self.truncated, "max_events": self.max_events,
                "max_members_per_event": MAX_MEMBERS,
                "shared_pool_bytes": getattr(shared, "total_bytes", None)}


def install_observer(Scheduler, SchedulePolicy, PrefillAdder, *, enabled=False, max_events=20000):
    """Install only in the dedicated benchmark launcher; return an undo callback."""
    if not enabled:
        return lambda: None
    active = ContextVar("m2_service_scheduler", default=None)
    originals = []

    def patch(cls, name, wrapper):
        original = getattr(cls, name)
        originals.append((cls, name, original))
        setattr(cls, name, wraps(original)(wrapper(original)))

    def ensure(scheduler):
        ps = scheduler.ps
        if (ps.pp_rank, ps.attn_tp_rank, ps.attn_cp_rank) != (0, 0, 0):
            return None
        if scheduler.schedule_policy != "tenant-vtc":
            return None
        trace = getattr(scheduler, "_m2_service_recorder", None)
        if trace is None or trace.policy is not scheduler.policy:
            trace = ServiceRecorder(scheduler, max_events)
            scheduler._m2_service_recorder = trace
            scheduler.policy._m2_service_recorder = trace
        return trace

    def add_queue(original):
        def wrapped(self, req, *args, **kwargs):
            trace = ensure(self)
            result = original(self, req, *args, **kwargs)
            if trace is not None:
                trace.emit("queue_inserted", members=trace.rows([req]), waiting=trace.queue(),
                           inserted=any(r is req for r in self.waiting_queue))
            return result
        return wrapped

    def queued(original):
        def wrapped(self, req, waiting_queue, *, is_retracted=False):
            trace = getattr(self, "_m2_service_recorder", None)
            if trace is None:
                return original(self, req, waiting_queue, is_retracted=is_retracted)
            key = tenant(req)
            before = dict(self.tenant_counters)
            waiting = counts(waiting_queue)
            reason = ("retracted_no_lift" if is_retracted else "already_waiting_no_lift"
                      if key in waiting else "active_waiting_floor" if waiting
                      else "last_tenant_floor" if self.last_client_left_queue is not None
                      else "zero_floor")
            result = original(self, req, waiting_queue, is_retracted=is_retracted)
            trace.emit("queue_counter", members=trace.rows([req]), waiting_before=waiting,
                       counters_before=before, counters_after=dict(self.tenant_counters),
                       lift_reason=reason, lift_delta=self.tenant_counters.get(key, 0) - before.get(key, 0))
            return result
        return wrapped

    def sort(original):
        def wrapped(self, waiting_queue):
            trace = getattr(self, "_m2_service_recorder", None)
            if trace is None:
                return original(self, waiting_queue)
            before = dict(self.tenant_counters)
            result = original(self, waiting_queue)
            rows = trace.rows(waiting_queue)
            shadow = dict(before)
            for row, req in zip(rows, waiting_queue):
                key = row["tenant"]
                charge = 0 if req.rid in self._vtc_input_charged_rids else self._vtc_admission_cost(req)
                row.update(shadow_before=shadow.get(key, 0), shadow_charge=charge)
                shadow[key] = shadow.get(key, 0) + charge
            trace.emit("sort", counters=before, waiting=counts(waiting_queue), candidates=rows)
            return result
        return wrapped

    def admitted(original):
        def wrapped(self, admitted, remaining_waiting_queue):
            trace = getattr(self, "_m2_service_recorder", None)
            if trace is None:
                return original(self, admitted, remaining_waiting_queue)
            before = dict(self.tenant_counters)
            rows = trace.rows(admitted)
            for row, req in zip(rows, admitted):
                row["input_charge"] = (0 if req.rid in self._vtc_input_charged_rids
                                       else self._vtc_admission_cost(req))
                row["first_input_charge"] = req.rid not in self._vtc_input_charged_rids
            result = original(self, admitted, remaining_waiting_queue)
            trace.emit("admission", members=rows, counters_before=before,
                       counters_after=dict(self.tenant_counters), waiting=counts(remaining_waiting_queue),
                       scope="selection before prepare_for_extend allocation; not allocation completion")
            return result
        return wrapped

    def processed(original):
        def wrapped(self, batch):
            trace = getattr(self, "_m2_service_recorder", None)
            if trace is None:
                return original(self, batch)
            before = dict(self.tenant_counters)
            rows = trace.rows(batch.reqs)
            for row, req in zip(rows, batch.reqs):
                previous = self._vtc_output_lengths.get(req.rid, len(req.output_ids))
                current = len(req.output_ids)
                row.update(previous_output=previous, current_output=current,
                           output_charge=self._vtc_output_cost(req, previous, current) if current > previous else 0,
                           finished=req.finished())
            result = original(self, batch)
            trace.emit("output_charge", batch_id=batch.forward_iter, members=rows,
                       counters_before=before, counters_after=dict(self.tenant_counters))
            return result
        return wrapped

    def raw(original):
        def wrapped(self, prefill_delayer_single_pass, running_batch):
            trace = ensure(self)
            if trace is None or not (self.waiting_queue or running_batch.reqs or self.chunked_req is not None):
                return original(self, prefill_delayer_single_pass, running_batch)
            trace.decision += 1
            trace.examined = set()
            trace.emit("decision_begin", waiting=trace.queue(), running=counts(running_batch.reqs),
                       batch_is_full=running_batch.batch_is_full,
                       chunked_rid=getattr(self.chunked_req, "rid", None), pools=trace.pools())
            token = active.set(trace)
            try:
                result = original(self, prefill_delayer_single_pass, running_batch)
            finally:
                active.reset(token)
            trace.emit("decision_end", waiting=trace.queue(), pools=trace.pools(),
                       unexamined_waiting=[r.rid for r in self.waiting_queue[:MAX_MEMBERS]
                                           if r.rid not in trace.examined],
                       produced_batch=result[0] is not None)
            return result
        return wrapped

    def candidate(original):
        def wrapped(self, req, *args, **kwargs):
            trace = active.get()
            if trace is None:
                return original(self, req, *args, **kwargs)
            trace.examined.add(req.rid)
            row = trace.rows([req])[0]
            fields = ("rem_total_tokens", "cur_rem_tokens", "rem_input_tokens", "rem_chunk_tokens",
                      "rem_mamba_slots", "rem_dllm_tokens")
            if getattr(self, "is_hybrid_swa", False):
                fields += ("rem_swa_tokens",)
            budgets = {name: cpu_number(getattr(self, name, None)) for name in fields}
            result = original(self, req, *args, **kwargs)
            selected = any(r is req for r in self.can_run_list)
            trace.emit("candidate", member=row, method=original.__name__, budgets_before=budgets,
                       budgets_after={name: cpu_number(getattr(self, name, None)) for name in fields},
                       selected=selected, result=getattr(result, "name", None),
                       continued_chunk=original.__name__ == "add_chunked_req",
                       scope="actual method return; OTHER/NO_TOKEN do not identify an inner branch")
            return result
        return wrapped

    def blocked(original):
        def wrapped(self, reason, running_bs, req=None, adder=None):
            result = original(self, reason, running_bs, req, adder)
            trace = active.get()
            if trace is not None:
                trace.emit("scan_stop", reason=reason, candidate_rid=getattr(req, "rid", None),
                           waiting=trace.queue(), pools=trace.pools())
            return result
        return wrapped

    def run_batch(original):
        def wrapped(self, batch, *args, **kwargs):
            trace = ensure(self)
            if trace is not None:
                mode = batch.forward_mode.name.lower()
                decoding = {r.rid for r in getattr(batch, "decoding_reqs", None) or ()}
                rows = trace.rows(batch.reqs)
                extend = getattr(batch, "extend_lens", None)
                prefix = getattr(batch, "prefix_lens", None)
                for index, row in enumerate(rows):
                    decode = mode == "decode" or row["rid"] in decoding
                    row.update(phase="decode" if decode else "prefill",
                               work_tokens=1 if decode else cpu_number(extend[index]) if extend is not None else None,
                               prefix_tokens=None if decode or prefix is None else cpu_number(prefix[index]))
                trace.emit("batch_launch", batch_id=self.forward_ct + 1, members=rows,
                           mode=mode, waiting=trace.queue(), pools=trace.pools())
            return original(self, batch, *args, **kwargs)
        return wrapped

    def batch_result(original):
        def wrapped(self, batch, result):
            trace = ensure(self)
            value = original(self, batch, result)
            if trace is not None:
                trace.emit("batch_result", batch_id=batch.forward_iter, members=trace.rows(batch.reqs),
                           finished_rids=[r.rid for r in batch.reqs[:MAX_MEMBERS] if r.finished()],
                           waiting=trace.queue(), pools=trace.pools())
            return value
        return wrapped

    def resources(original):
        def wrapped(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            trace = ensure(self)
            tracker = self.tenant_resource_tracker
            if trace is not None and tracker is not None:
                for event in tracker.events[trace.resource_cursor:]:
                    trace.emit("resource_observation", source_origin_s=tracker.started_at, source_event=event,
                               pools=trace.pools(), release_reason="not observed at allocator call")
                trace.resource_cursor = len(tracker.events)
            return result
        return wrapped

    def info(original):
        def wrapped(self, req):
            trace = ensure(self)
            result = original(self, req)
            result.internal_state["m2_service_trace"] = trace.snapshot() if trace else {"enabled": False}
            return result
        return wrapped

    for name, wrapper in (("_add_request_to_queue", add_queue), ("_get_new_batch_prefill_raw", raw),
                          ("_record_tenant_admission_block", blocked), ("run_batch", run_batch),
                          ("process_batch_result", batch_result), ("observe_tenant_resources", resources),
                          ("get_internal_state", info)):
        patch(Scheduler, name, wrapper)
    for name, wrapper in (("on_request_queued", queued), ("_sort_by_tenant_vtc", sort),
                          ("on_requests_admitted", admitted), ("on_batch_processed", processed)):
        patch(SchedulePolicy, name, wrapper)
    for name in ("add_one_req", "add_chunked_req"):
        patch(PrefillAdder, name, candidate)

    def undo():
        for cls, name, original in reversed(originals):
            setattr(cls, name, original)
    return undo
