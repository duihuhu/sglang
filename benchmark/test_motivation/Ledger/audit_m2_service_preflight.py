"""Check observation conservation, not a performance or fairness PASS gate."""

from collections import Counter, defaultdict


def inspect(item):
    errors = []
    trace = item["service_trace"]
    requests = {r["rid"]: r for r in item["requests"]}
    events = trace.get("events", [])
    if not trace.get("enabled") or not events:
        return {"errors": ["missing service trace"]}
    if trace.get("dropped_events") or trace.get("truncated_fields"):
        errors.append("trace overflow")
    if trace.get("boot_id") != item["client_clock"]["boot_id"] or trace.get("clock") != item["client_clock"]["implementation"]:
        errors.append("unverified common monotonic clock")
    if trace.get("run_ids") != [item["experiment"]["run_id"]]:
        errors.append("unexpected run namespace")
    if [e["seq"] for e in events] != list(range(1, len(events) + 1)):
        errors.append("non-contiguous sequence")
    if any(b["observed_s"] < a["observed_s"] for a, b in zip(events, events[1:])):
        errors.append("non-monotonic event clock")

    first_inputs, input_charges, outputs, prefills = Counter(), Counter(), Counter(), Counter()
    activated, released, queued, completed = Counter(), Counter(), Counter(), set()
    launches, results = {}, set()
    last_counters = {}
    chunk_reuses = admitted_slots = stops_after_selection = 0
    kinds = Counter(e["kind"] for e in events)
    for event in events:
        kind = event["kind"]
        rows = event.get("members", [])
        if any(row["rid"] not in requests for row in rows):
            errors.append("unknown request in event")
        if kind in ("queue_counter", "admission", "output_charge"):
            before, after = event["counters_before"], event["counters_after"]
            if any(before.get(k, 0) != last_counters.get(k, 0) for k in set(before) | set(last_counters)):
                errors.append("broken counter trajectory")
            expected = defaultdict(float, before)
            if kind == "queue_counter":
                expected[rows[0]["tenant"]] += event["lift_delta"]
            else:
                field = "input_charge" if kind == "admission" else "output_charge"
                for row in rows:
                    expected[row["tenant"]] += row[field]
            if any(after.get(k, 0) != expected.get(k, 0) for k in set(after) | set(expected)):
                errors.append("counter change not explained by charges/lift")
            last_counters = after
        if kind == "queue_inserted" and event["inserted"]:
            queued.update(row["rid"] for row in rows)
        elif kind == "admission":
            for row in rows:
                rid = row["rid"]
                first_inputs[rid] += int(row["first_input_charge"])
                input_charges[rid] += row["input_charge"]
                admitted_slots += int(not row["has_request_slot"])
        elif kind == "candidate":
            chunk_reuses += int(event["continued_chunk"] and event["member"]["has_request_slot"])
            stops_after_selection += int(event["selected"] and event["result"] in ("OTHER", "NO_TOKEN"))
        elif kind == "output_charge":
            for row in rows:
                rid = row["rid"]
                if row["previous_output"] != outputs[rid]:
                    errors.append("output progress gap or duplicate")
                if row["output_charge"] != 2 * (row["current_output"] - row["previous_output"]):
                    errors.append("wrong token-VTC output charge")
                outputs[rid] = row["current_output"]
        elif kind == "batch_launch":
            bid = event["batch_id"]
            if bid in launches:
                errors.append("duplicate batch launch ID")
            launches[bid] = event
            for row in rows:
                if row["phase"] == "prefill":
                    if row["prefix_tokens"] != prefills[row["rid"]] or not isinstance(row["work_tokens"], int):
                        errors.append("prefill gap or invalid work")
                    else:
                        prefills[row["rid"]] += row["work_tokens"]
        elif kind == "batch_result":
            bid = event["batch_id"]
            if bid not in launches or bid in results:
                errors.append("unmatched or duplicate batch result")
            results.add(bid)
            completed.update(event["finished_rids"])
        elif kind == "resource_observation":
            source = event["source_event"]
            activated.update(row["rid"] for row in source["added"])
            released.update(source["removed"])
    if set(launches) != results:
        errors.append("unprocessed batch launch")
    if completed != set(requests):
        errors.append("missing completion")
    for rid, request in requests.items():
        if (first_inputs[rid] != 1 or input_charges[rid] != request["input_len"]
                or outputs[rid] != request["completion_tokens"] or prefills[rid] != request["input_len"]):
            errors.append("request charge/work conservation failed")
        if (activated[rid], released[rid], queued[rid]) != (1, 1, 1):
            errors.append("request queue/holder lifecycle not exactly once")

    # This is a measured waiting-backlog interval, not client outstanding work.
    prefix = f"hybrid-tenant:{item['experiment']['run_id']}:"
    waiting, intervals, last_time = {}, [], events[0]["observed_s"]
    for event in events:
        now = event["observed_s"]
        if all(waiting.get(prefix + role, 0) > 0 for role in ("A", "B")) and now > last_time:
            if intervals and abs(intervals[-1][1] - last_time) < 1e-9:
                intervals[-1][1] = now
            else:
                intervals.append([last_time, now])
        if "waiting" in event:
            waiting = event["waiting"]
        last_time = now
    return {"errors": sorted(set(errors)), "event_kinds": dict(kinds),
            "requests": len(requests), "batches": len(launches), "chunk_reuses": chunk_reuses,
            "first_slot_selections": admitted_slots,
            "scan_stop_after_successful_selection": stops_after_selection,
            "both_waiting_intervals_server_relative_s": intervals,
            "both_waiting_seconds": sum(end - start for start, end in intervals),
            "max_both_waiting_interval_s": max((b - a for a, b in intervals), default=0),
            "final_counters": last_counters,
            "scope": "functional preflight only; no causal hybrid or fairness verdict"}
