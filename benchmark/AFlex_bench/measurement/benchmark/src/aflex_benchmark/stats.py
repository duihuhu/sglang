from __future__ import annotations
import math, statistics
PCTS = (50, 90, 95, 99)

def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    k = (len(ys) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return ys[lo] if lo == hi else ys[lo] * (hi-k) + ys[hi] * (k-lo)

def describe(xs: list[float]) -> dict:
    return {"count": len(xs), "mean": statistics.fmean(xs) if xs else 0.0,
            **{f"p{p}": percentile(xs, p) for p in PCTS}}

def summarize_requests(rows: list[dict], energy: dict | None=None) -> dict:
    ok = [r for r in rows if r.get("success")]
    metrics = {k: describe([float(r[k]) for r in ok if r.get(k) is not None])
               for k in ("ttft_client_ms", "ttft_server_ms", "tpot_ms", "e2e_ms")}
    metrics["itl_ms"] = describe([float(x) for r in ok for x in r.get("itl_ms", [])])
    input_tokens = sum(int(r.get("input_tokens", 0)) for r in ok)
    output_tokens = sum(int(r.get("completion_tokens", 0)) for r in ok)
    all_tokens = input_tokens + output_tokens
    if rows:
        duration = max(r.get("completed_offset_s", 0) for r in rows) - min(r.get("sent_offset_s", 0) for r in rows)
    else:
        duration = 0.0
    duration = max(float(duration), 0.0)
    throughput = lambda count: count / duration if duration > 0 else 0.0
    failed = len(rows) - len(ok)
    status = "failed" if not ok else "partial" if failed else "complete"
    out = {"status": status, "requests_total": len(rows),
           "requests_success": len(ok),
           "requests_failed": failed, "input_tokens": input_tokens,
           "completion_tokens": output_tokens, "all_tokens": all_tokens,
           "duration_s": duration, "achieved_qps": throughput(len(ok)),
           "input_throughput_tokens_s": throughput(input_tokens),
           "output_throughput_tokens_s": throughput(output_tokens),
           "all_throughput_tokens_s": throughput(all_tokens), "metrics": metrics}
    if energy:
        out["energy"] = energy
        joules = float(energy.get("total_j", 0))
        avg_watts = joules / duration if duration > 0 else 0.0
        out.update({"average_cluster_power_w": avg_watts,
                    "energy_per_input_token_j": joules/input_tokens if input_tokens else None,
                    "energy_per_output_token_j": joules/output_tokens if output_tokens else None,
                    "energy_per_all_token_j": joules/all_tokens if all_tokens else None,
                    "energy_per_successful_request_j": joules/len(ok) if ok else None,
                    "qps_per_w": throughput(len(ok))/avg_watts if avg_watts > 0 else None,
                    "input_tokens_per_j": input_tokens/joules if joules > 0 else None,
                    "output_tokens_per_j": output_tokens/joules if joules > 0 else None,
                    "all_tokens_per_j": all_tokens/joules if joules > 0 else None})
    return out
