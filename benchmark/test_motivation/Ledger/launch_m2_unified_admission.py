"""Common diagnostic wrapper; never modify the model or allocation code."""

from collections import Counter
import os

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.min_free_slots_delayer import MinFreeSlotsDelayer
from sglang.srt.managers.prefill_delayer import PrefillDelayer


LIMIT = int(os.environ.get("M2_ADMISSION_DECODE_PASSES", "0"))
if LIMIT not in (0, 32):
    raise ValueError("Unregistered diagnostic limit")


def run_key(rid):
    for marker in ("-holder-", "-candidate-"):
        if marker in rid:
            return rid.rsplit(marker, 1)[0]
    return None


def decision(state, *, has_candidate, holders, chunked, limit):
    if not has_candidate or chunked or not holders or state["finished"]:
        if state["started"] and not holders:
            state["finished"] = True
        return False
    state["started"] = True
    if state["delayed"] >= limit:
        state["finished"] = True
        return False
    state["delayed"] += 1
    return True


_raw = Scheduler._get_new_batch_prefill_raw
_info = Scheduler.get_internal_state
_slot = MinFreeSlotsDelayer.should_delay
_prefill = PrefillDelayer._negotiate_should_allow_prefill
_active = None


def raw(self, prefill_delayer_single_pass, running_batch):
    global _active
    if not hasattr(self, "_m2_admission_runs"):
        self._m2_admission_runs = {}
    reqs = list(self.waiting_queue)
    current = next((run_key(r.rid) for r in reqs if run_key(r.rid)), None)
    state = None
    if current is not None:
        state = self._m2_admission_runs.setdefault(current, {
            "started": False, "finished": False, "delayed": 0,
            "native_slot_checks": 0, "native_slot_delays": 0,
            "native_prefill_reasons": Counter(), "diagnostic_limit": LIMIT,
        })
        holders = sum("-holder-" in r.rid for r in running_batch.reqs)
        candidate = any("-candidate-" in r.rid for r in reqs)
        if decision(state, has_candidate=candidate, holders=holders,
                    chunked=self.chunked_req is not None, limit=LIMIT):
            return None, running_batch
    _active = state
    try:
        return _raw(self, prefill_delayer_single_pass, running_batch)
    finally:
        _active = None


def slot(self, **kwargs):
    result = _slot(self, **kwargs)
    if _active is not None:
        _active["native_slot_checks"] += 1
        _active["native_slot_delays"] += int(result)
    return result


def prefill(self, *args, **kwargs):
    result = _prefill(self, *args, **kwargs)
    if _active is not None:
        _active["native_prefill_reasons"][result.output_reason] += 1
    return result


def info(self, req):
    result = _info(self, req)
    result.internal_state["m2_admission"] = {
        "runs": getattr(self, "_m2_admission_runs", {}),
        "diagnostic_limit": LIMIT,
        "shared_pool_bytes": self.req_to_token_pool._unified_buffer.total_bytes,
    }
    return result


Scheduler._get_new_batch_prefill_raw = raw
Scheduler.get_internal_state = info
MinFreeSlotsDelayer.should_delay = slot
PrefillDelayer._negotiate_should_allow_prefill = prefill


if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree
    load_plugins()
    try:
        run_server(prepare_server_args(__import__("sys").argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
