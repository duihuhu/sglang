from __future__ import annotations

import json
import math
import time
import urllib.request
from pathlib import Path


def _nonnegative_int(value):
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _server_duration(meta: dict, *keys: str):
    for key in keys:
        value = meta.get(key)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
        ):
            return float(value)
    return None


def _append_token_stamps(
    stamps: list[float], token_count: int, observed_at: float, sent: float, meta: dict
):
    """Extend stamps to a cumulative server token count without double counting."""
    missing = token_count - len(stamps)
    if missing <= 0:
        # Duplicate/control events and count regressions caused by retractions must not
        # create additional logical output tokens.
        return
    if missing == 1:
        # Keep the actual arrival time for the normal one-token-per-event path.
        stamps.append(observed_at)
        return

    server_e2e = _server_duration(meta, "e2e_latency")
    batch_end = sent + server_e2e if server_e2e is not None else observed_at
    batch_end = min(observed_at, max(sent, batch_end))

    if stamps:
        start = stamps[-1]
        batch_end = max(start, batch_end)
        step = (batch_end - start) / missing
        stamps.extend(start + step * index for index in range(1, missing + 1))
        return

    server_ttft = _server_duration(
        meta, "ttft_pure_processing", "time_to_first_token_processing", "time_to_first_token"
    )
    if server_ttft is not None:
        first = min(observed_at, max(sent, sent + server_ttft))
    else:
        # With only a batched arrival available, spread tokens over the observed
        # request interval so both TTFT and TPOT remain defined.
        first = sent + (batch_end - sent) / token_count
    batch_end = max(first, batch_end)
    if token_count == 1:
        stamps.append(first)
    else:
        step = (batch_end - first) / (token_count - 1)
        stamps.extend(first + step * index for index in range(token_count))


def stream_request(endpoint: str, req: dict, run_start: float, timeout_s=None) -> dict:
    target = run_start + float(req.get("arrival_time_s", 0))
    delay = target - time.monotonic()
    if delay > 0:
        time.sleep(delay)
    sent = time.monotonic()
    stamps = []
    meta = {}
    error = None
    saw_reported_count = False
    expected_tokens = int(req["output_len"])
    payload = {"input_ids": [1000] * int(req["input_len"]), "sampling_params": {"max_new_tokens": expected_tokens, "temperature": 0.0, "ignore_eos": True}, "stream": True}
    try:
        request = urllib.request.Request(endpoint.rstrip("/") + "/generate", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        timeout = float(timeout_s if timeout_s is not None else req.get("timeout_s", 30))
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                text = raw.decode(errors="replace").strip()
                if not text or text.startswith(":"):
                    continue
                if text.startswith("data:"):
                    text = text[5:].strip()
                if text == "[DONE]":
                    break
                try:
                    chunk = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(chunk, dict):
                    chunk_meta = chunk.get("meta_info")
                    if not isinstance(chunk_meta, dict):
                        chunk_meta = {}
                    meta.update(chunk_meta)
                    reported_tokens = _nonnegative_int(chunk_meta.get("completion_tokens"))
                    if reported_tokens is not None:
                        saw_reported_count = True
                        _append_token_stamps(
                            stamps,
                            reported_tokens,
                            time.monotonic(),
                            sent,
                            meta,
                        )
                    elif not saw_reported_count and chunk.get("text") not in (None, ""):
                        # Compatibility for servers that do not expose cumulative
                        # token counts. This remains exact for ordinary token chunks.
                        stamps.append(time.monotonic())
    except Exception as exc:
        error = repr(exc)
    if error is None and len(stamps) != expected_tokens:
        error = f"completion_token_mismatch: expected={expected_tokens}, actual={len(stamps)}"
    end = time.monotonic()
    ttft = (stamps[0] - sent) * 1000 if stamps else None
    itl = [(later - earlier) * 1000 for earlier, later in zip(stamps, stamps[1:])]
    server = meta.get("ttft_pure_processing", meta.get("time_to_first_token_processing"))
    server = server * 1000 if server is not None else None
    scheduled = float(req.get("arrival_time_s", 0))
    return {"request_id": req["request_id"], "success": error is None and len(stamps) == expected_tokens, "error": error, "input_tokens": req["input_len"], "expected_completion_tokens": expected_tokens, "completion_tokens": len(stamps), "scheduled_arrival_s": scheduled, "sent_offset_s": sent - run_start, "arrival_lag_ms": max(0.0, (sent - run_start - scheduled) * 1000), "completed_offset_s": end - run_start, "token_timestamps_s": [value - run_start for value in stamps], "itl_ms": itl, "ttft_client_ms": ttft, "ttft_server_ms": server, "tpot_ms": sum(itl) / len(itl) if itl else 0.0, "e2e_ms": (end - sent) * 1000, "server_meta": meta}


def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        file.write(json.dumps(row, sort_keys=True) + "\n")
