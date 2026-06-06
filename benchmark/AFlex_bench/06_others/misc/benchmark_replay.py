#!/usr/bin/env python3
"""
Trace-driven benchmark: replay Azure LLM inference trace or synthetic sample
against a running SGLang server.

Datasets:
  --dataset azure   Reads CSV (TIMESTAMP, ContextTokens, GeneratedTokens) and
                    replays following the real-world arrival pattern.
  --dataset sample  Generates N requests with fixed input/output lengths and
                    Poisson-like arrival at a given QPS (--sample-qps).

Supports GPU energy monitoring via DVFSController.
"""

import argparse
import asyncio
from asyncio import FIRST_COMPLETED
import csv
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import aiohttp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")

# ---------------------------------------------------------------------------
# Optional energy monitoring
# ---------------------------------------------------------------------------

def _init_energy_monitoring(gpu_indices_str: str) -> Optional[list]:
    """Initialise DVFSController for each GPU index in the comma-separated string.

    Returns list of (gpu_idx, controller) tuples, or None if DVFS lib unavailable.
    """
    if not gpu_indices_str:
        return None
    indices = [int(x.strip()) for x in gpu_indices_str.split(",") if x.strip()]
    if not indices:
        return None

    # Set env so dvfs.py can find the .so
    if "DVFS_CTRL_LIB" not in os.environ:
        _candidates = [
            os.path.join(os.path.dirname(__file__), "..", "hucc", "dvfs", "libdvfs_ctrl.so"),
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "benchmark",
                         "test_motivation", "hucc", "dvfs", "libdvfs_ctrl.so"),
        ]
        for p in _candidates:
            if os.path.exists(p):
                os.environ["DVFS_CTRL_LIB"] = os.path.realpath(p)
                break

    try:
        from sglang.srt.layers.dvfs import DVFSController
        controllers = []
        for idx in indices:
            ctrl = DVFSController(device_index=idx)
            controllers.append((idx, ctrl))
        logger.info("Energy monitoring enabled for GPUs: %s", indices)
        return controllers
    except Exception as e:
        logger.warning("Energy monitoring unavailable: %s", e)
        return None


def _read_energy_mj(controllers: list) -> dict:
    """Read cumulative energy (mJ) from each GPU controller.

    Returns dict of {gpu_idx: energy_mj}.
    """
    return {idx: ctrl.get_energy_mj() for idx, ctrl in controllers}


def _report_energy(energy_before: dict, energy_after: dict, label: str = ""):
    """Print energy delta for each GPU and total."""
    print(f"\n  --- Energy consumption{label} ---")
    total_mj = 0
    for idx in sorted(energy_before.keys()):
        delta = energy_after.get(idx, 0) - energy_before.get(idx, 0)
        total_mj += delta
        print(f"  GPU {idx}:                    {delta / 1000:.2f} J")
    print(f"  Total:                       {total_mj / 1000:.2f} J")
    print(f"  Total:                       {total_mj / 3600000:.4f} kWh")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Request:
    timestamp: datetime  # original arrival time
    input_len: int       # ContextTokens
    output_len: int      # GeneratedTokens


@dataclass
class Result:
    request: Request
    latency_s: float       # end-to-end response time (client-side)
    success: bool
    error: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    ttft_s: float = 0.0    # time to first token (processing only, excludes queueing)
    ttft_total_s: float = 0.0  # total TTFT including queue (server-side)
    tpot_s: float = 0.0    # time per output token (excluding prefill)
    meta: dict = None

    def __post_init__(self):
        if self.meta is None:
            self.meta = {}


# ---------------------------------------------------------------------------
# Dummy prompt generator (word-level)
# ---------------------------------------------------------------------------

_WORDS = [
    "the ", "quick ", "brown ", "fox ", "jumps ", "over ", "lazy ", "dog ",
    "cat ", "run ", "fast ", "slow ", "big ", "small ", "red ", "blue ",
    "sky ", "ocean ", "tree ", "mountain ", "river ", "cloud ", "sun ", "moon ",
]


def make_prompt(n_tokens: int, seed: int = 0) -> str:
    """Create a prompt of roughly `n_tokens` words, unique per seed."""
    rng = random.Random(seed)
    return "".join(rng.choices(_WORDS, k=max(1, n_tokens)))


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------

class TraceReplayBenchmark:
    def __init__(
        self,
        csv_path: str,
        url: str,
        speedup: float = 1.0,
        max_requests: int = 0,
        concurrency: int = 100,
        timeout_s: int = 600,
        dump_file: str = "",
        energy_controllers: Optional[list] = None,
        scenario_label: str = "",
        *,
        dataset: str = "azure",
        sample_input_len: int = 1024,
        sample_output_len: int = 128,
        sample_qps: float = 1.0,
        seed: int = 42,
    ):
        self.csv_path = csv_path
        self.url = url.rstrip("/") + "/generate"
        self.speedup = speedup
        self.max_requests = max_requests
        self.concurrency = concurrency
        self.timeout_s = timeout_s
        self.dump_file = dump_file
        self.energy_controllers = energy_controllers
        self.scenario_label = scenario_label
        self.dataset = dataset
        self.sample_input_len = sample_input_len
        self.sample_output_len = sample_output_len
        self.sample_qps = sample_qps
        self.seed = seed
        self.warmup_requests = 0

        self.requests: list[Request] = []
        self.results: list[Result] = []
        self._sem: asyncio.Semaphore | None = None
        self._session: aiohttp.ClientSession | None = None
        self._energy_before: dict = {}
        self._energy_after: dict = {}

    # ---- Load trace (Azure CSV) -----------------------------------------

    def load_trace(self):
        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts_raw = row.get("TIMESTAMP")
                il_raw = row.get("ContextTokens")
                ol_raw = row.get("GeneratedTokens")
                if ts_raw is None or il_raw is None or ol_raw is None:
                    continue
                ts = datetime.fromisoformat(ts_raw.strip())
                il = int(il_raw.strip())
                ol = int(ol_raw.strip())
                self.requests.append(Request(ts, il, ol))

        # Commented-out manual overrides kept for debugging
        # self.requests = self.requests[:100]
        # self.requests[0].input_len = 30000
        # self.requests[0].output_len = 256

        self.requests.sort(key=lambda r: r.timestamp)
        logger.info(
            "Loaded %d requests from %s – %s (%.1f h)",
            len(self.requests),
            self.requests[0].timestamp,
            self.requests[-1].timestamp,
            (self.requests[-1].timestamp - self.requests[0].timestamp).total_seconds() / 3600,
        )

        if self.max_requests:
            self.requests = self.requests[: self.max_requests]
            logger.info("Trimmed to %d requests", len(self.requests))

    # ---- Generate synthetic sample --------------------------------------

    def generate_sample(self):
        """Generate a synthetic dataset with fixed-length requests at a given QPS."""
        rng = random.Random(self.seed)

        num = self.max_requests
        il = self.sample_input_len
        ol = self.sample_output_len
        qps = self.sample_qps
        interval_s = 1.0 / qps if qps > 0 else 0.001

        # Use a fixed reference timestamp so results are reproducible
        ref_ts = datetime(2025, 1, 1, tzinfo=timezone.utc)

        for i in range(num):
            # Add small jitter (±10% of interval) for realistic arrival pattern
            jitter = rng.uniform(-interval_s * 0.1, interval_s * 0.1)
            ts = ref_ts + timedelta(seconds=i * interval_s + jitter)
            self.requests.append(Request(ts, il, ol))

        self.requests.sort(key=lambda r: r.timestamp)
        trace_dur = (self.requests[-1].timestamp - self.requests[0].timestamp).total_seconds()
        logger.info(
            "Generated %d synthetic requests (il=%d, ol=%d, qps=%.1f, seed=%d) "
            "over %.1f s",
            num, il, ol, qps, self.seed, trace_dur,
        )

    # ---- Send a single request ----------------------------------------

    async def _send(self, req: Request, req_id: int) -> Result:
        prompt = make_prompt(req.input_len, seed=req_id)
        payload = {
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": req.output_len,
                "temperature": 0.0,
            },
        }

        t0 = time.monotonic()
        try:
            async with self._session.post(
                self.url, json=payload, timeout=aiohttp.ClientTimeout(total=self.timeout_s)
            ) as resp:
                body = await resp.json()
                elapsed = time.monotonic() - t0

            if resp.status != 200:
                return Result(req, elapsed, False, error=f"HTTP {resp.status}: {body}")

            # SGLANG /generate returns: {"text": [...], "meta_info": {"prompt_tokens": ..., "completion_tokens": ..., ...}}
            meta = body.get("meta_info", {})
            in_tok = meta.get("prompt_tokens", req.input_len)
            out_tok = meta.get("completion_tokens", 0)

            # Compute TTFT from server-side timing fields
            # Priority:
            #   1) time_to_first_token_processing — scheduler dispatch→prefill_finish
            #      (matches PA.log Tier1 monitor, excludes queueing)
            #   2) time_to_first_token — client-perceived TTFT (includes queueing)
            #   3) prefill_finished_time - request_received_ts (unix ts diff)
            ttft_s = 0.0
            ttft_total_s = 0.0
            has_ttft = False

            if "time_to_first_token_processing" in meta:
                raw = meta["time_to_first_token_processing"]
                if 0 < raw < elapsed:
                    ttft_s = raw
                    has_ttft = True
            elif "time_to_first_token" in meta:
                raw = meta["time_to_first_token"]
                if 0 < raw < elapsed:
                    ttft_s = raw
                    has_ttft = True

            # Always capture total TTFT (client-perceived) for reference
            if "time_to_first_token" in meta:
                raw = meta["time_to_first_token"]
                if 0 < raw < elapsed:
                    ttft_total_s = raw

            # Fallback: derive TTFT from server-side e2e_latency when timing
            # fields are missing (e.g. native server without disaggregation)
            if not has_ttft and "e2e_latency" in meta and out_tok > 0:
                server_e2e = meta["e2e_latency"]
                if 0 < server_e2e < elapsed:
                    # e2e_latency = TTFT + (out_tok - 1) * TPOT
                    # Estimate TPOT as a fraction of e2e based on token count
                    # For single token: e2e = TTFT exactly
                    if out_tok == 1:
                        ttft_s = server_e2e
                        has_ttft = True
                    elif out_tok > 1:
                        # Estimate: assume TTFT ≈ prefill_time ≈ 3-5x decode step
                        # With e2e = TTFT + (out_tok-1)*TPOT:
                        #   If we approximate TTFT ≈ 3 * TPOT (typical for short inputs):
                        #   e2e ≈ 3*TPOT + (out_tok-1)*TPOT = (out_tok+2)*TPOT
                        #   → TPOT = e2e / (out_tok + 2)
                        #   → TTFT = e2e - (out_tok-1) * TPOT = 3 * e2e / (out_tok+2)
                        approx_tpot = server_e2e / (out_tok + 2)
                        derived = server_e2e - (out_tok - 1) * approx_tpot
                        if 0 < derived < elapsed:
                            ttft_s = derived
                            has_ttft = True

            # TPOT: prefer server-side per-token decode step average.
            # decode_tpot_avg_s = mean(DA+DF step time) recorded by scheduler.
            server_tpot = meta.get("decode_tpot_avg_s", 0.0)
            if server_tpot > 0.0:
                tpot_s = server_tpot
            elif out_tok > 0 and has_ttft:
                tpot_s = (elapsed - ttft_s) / out_tok
            elif out_tok > 0:
                tpot_s = elapsed / out_tok
            else:
                tpot_s = 0.0

            # Debug: log available metric fields for first request
            if req_id == 0:
                logger.info("[META_ALL_KEYS] %s", sorted(meta.keys()))
                logger.info("[META_FULL] %s", {k: v for k, v in meta.items() if not isinstance(v, (list, dict))})
                # Show timing breakdown
                ttft_proc = meta.get("time_to_first_token_processing", 0)
                fwd_ms = meta.get("scheduler_model_forward_ms", 0)
                logger.info("[BREAKDOWN] ttft_processing=%.1fms  model_forward=%.1fms  overhead=%.1fms",
                           ttft_proc * 1000, fwd_ms, ttft_proc * 1000 - fwd_ms)

            return Result(
                req, elapsed, True,
                input_tokens=in_tok, output_tokens=out_tok,
                ttft_s=ttft_s, ttft_total_s=ttft_total_s, tpot_s=tpot_s,
                meta=meta,
            )

        except asyncio.TimeoutError:
            elapsed = time.monotonic() - t0
            return Result(req, elapsed, False, error="timeout")
        except aiohttp.ClientError as e:
            elapsed = time.monotonic() - t0
            return Result(req, elapsed, False, error=str(e))

    async def _worker(self, req_id: int, req: Request):
        async with self._sem:
            result = await self._send(req, req_id)
            self.results.append(result)
            status = "OK" if result.success else "ERR"
            if result.success and result.ttft_s > 0:
                if result.ttft_total_s > result.ttft_s:
                    logger.info(
                        "[%05d] %s | il=%d ol=%d  latency=%.2fs  "
                        "ttft_proc=%.1fms ttft_total=%.1fms  tpot=%.1fms",
                        req_id, status, req.input_len, req.output_len,
                        result.latency_s, result.ttft_s * 1000,
                        result.ttft_total_s * 1000, result.tpot_s * 1000,
                    )
                else:
                    logger.info(
                        "[%05d] %s | il=%d ol=%d  latency=%.2fs  ttft=%.1fms  tpot=%.1fms",
                        req_id, status, req.input_len, req.output_len,
                        result.latency_s, result.ttft_s * 1000, result.tpot_s * 1000,
                    )
                # Print per-token decode step times
                per_token = result.meta.get("decode_tpot_per_token_ms", [])
                if per_token:
                    # Show first 10 and last 3
                    if len(per_token) <= 20:
                        detail = "  ".join(f"t{i}={v:.1f}ms" for i, v in enumerate(per_token))
                    else:
                        head = "  ".join(f"t{i}={v:.1f}ms" for i, v in enumerate(per_token[:10]))
                        tail = "  ".join(f"t{i}={v:.1f}ms" for i, v in enumerate(per_token[-3:], len(per_token)-3))
                        detail = f"{head}  ...  {tail}"
                    logger.info("[%05d] TPOT_PER_TOKEN (%d tokens): %s", req_id, len(per_token), detail)
            else:
                logger.info(
                    "[%05d] %s | il=%d ol=%d  latency=%.2fs  tpo=%.1fms",
                    req_id, status, req.input_len, req.output_len,
                    result.latency_s, result.tpot_s * 1000,
                )

    # ---- Warmup --------------------------------------------------------

    async def _warmup(self, session):
        """Send warmup requests; results are discarded."""
        if self.warmup_requests <= 0:
            return
        logger.info("Sending %d warmup requests...", self.warmup_requests)
        sem = asyncio.Semaphore(min(self.concurrency, self.warmup_requests))
        il = self.sample_input_len if self.dataset == "sample" else 1024
        ol = self.sample_output_len if self.dataset == "sample" else 128
        async def _warmup_one(req_id):
            async with sem:
                prompt = make_prompt(il, seed=req_id)
                payload = {
                    "text": prompt,
                    "sampling_params": {"max_new_tokens": ol, "temperature": 0.0},
                }
                try:
                    async with session.post(
                        self.url, json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout_s),
                    ) as resp:
                        await resp.json()
                except Exception:
                    pass
        tasks = [asyncio.create_task(_warmup_one(i)) for i in range(self.warmup_requests)]
        await asyncio.gather(*tasks)
        logger.info("Warmup complete (%d requests)", self.warmup_requests)

    # ---- Main loop ----------------------------------------------------

    async def run(self):
        if self.dataset == "sample":
            self.generate_sample()
        else:
            self.load_trace()
        self._sem = asyncio.Semaphore(self.concurrency)

        connector = aiohttp.TCPConnector(limit=max(self.concurrency, 10), force_close=True)
        async with aiohttp.ClientSession(connector=connector) as session:
            self._session = session

            # Warmup first
            await self._warmup(session)

            # Read energy before benchmark
            if self.energy_controllers:
                self._energy_before = _read_energy_mj(self.energy_controllers)
                logger.info("Energy readings taken BEFORE benchmark")

            t_start = time.monotonic()
            trace_start = self.requests[0].timestamp

            pending = []
            completed = 0

            for i, req in enumerate(self.requests):
                # Compute delay relative to real-time arrival, scaled by speedup
                wall_elapsed = time.monotonic() - t_start
                trace_elapsed = (req.timestamp - trace_start).total_seconds() / self.speedup

                if trace_elapsed > wall_elapsed:
                    await asyncio.sleep(trace_elapsed - wall_elapsed)

                task = asyncio.create_task(self._worker(i, req))
                pending.append(task)

                # Periodically reap finished tasks to keep pending list manageable
                if len(pending) >= max(self.concurrency * 4, 4):
                    done, pending = await asyncio.wait(pending, return_when=FIRST_COMPLETED)
                    pending = list(pending)
                    completed += len(done)

            # Wait for all remaining
            if pending:
                await asyncio.wait(pending)

            wall_duration = time.monotonic() - t_start

        # Read energy after benchmark
        if self.energy_controllers:
            self._energy_after = _read_energy_mj(self.energy_controllers)
            logger.info("Energy readings taken AFTER benchmark")

        self._report(wall_duration, trace_start)

    # ---- Reporting ----------------------------------------------------

    def _report(self, wall_duration: float, trace_start: datetime):
        total = len(self.results)
        ok = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]

        print("\n" + "=" * 70)
        print(" BENCHMARK RESULTS")
        print("=" * 70)

        trace_end = self.requests[-1].timestamp
        trace_dur = (trace_end - trace_start).total_seconds()

        print(f"  Trace duration:              {trace_dur:.1f}s")
        print(f"  Wall-clock duration:         {wall_duration:.1f}s")
        print(f"  Speedup factor:              {self.speedup:.1f}x")
        print(f"  Total requests:              {total}")
        print(f"  Succeeded:                   {len(ok)}")
        print(f"  Failed:                      {len(failed)}")
        if failed:
            err_counts: dict[str, int] = {}
            for r in failed:
                err_counts[r.error] = err_counts.get(r.error, 0) + 1
            for err, cnt in sorted(err_counts.items(), key=lambda x: -x[1]):
                print(f"    - [{cnt}] {err}")

        if ok:
            latencies = [r.latency_s for r in ok]
            ttfts = [r.ttft_s for r in ok if r.ttft_s > 0]
            tpots = [r.tpot_s for r in ok if r.tpot_s > 0]
            in_toks = [r.input_tokens for r in ok]
            out_toks = [r.output_tokens for r in ok]

            wall_rps = total / wall_duration if wall_duration > 0 else 0
            trace_rps = total / trace_dur if trace_dur > 0 else 0

            lat_sorted = sorted(latencies)
            n = len(lat_sorted)

            print(f"\n  --- Request rate ---")
            print(f"  Avg RPS (wall):              {wall_rps:.2f}")
            print(f"  Avg RPS (trace):             {trace_rps:.2f}")

            print(f"\n  --- Latency (seconds) ---")
            print(f"  Mean:                        {sum(latencies) / n:.3f}")
            print(f"  P50:                         {lat_sorted[int(n * 0.50)]:.3f}")
            print(f"  P90:                         {lat_sorted[int(n * 0.90)]:.3f}")
            print(f"  P95:                         {lat_sorted[int(n * 0.95)]:.3f}")
            print(f"  P99:                         {lat_sorted[int(n * 0.99)]:.3f}")
            print(f"  Max:                         {max(latencies):.3f}")

            if ttfts:
                ttft_sorted = sorted(ttfts)
                m = len(ttft_sorted)
                print(f"\n  --- TTFT (processing, ms) ---")
                print(f"  Mean:                        {sum(ttfts) / m * 1000:.2f}")
                print(f"  P50:                         {ttft_sorted[int(m * 0.50)] * 1000:.2f}")
                print(f"  P90:                         {ttft_sorted[int(m * 0.90)] * 1000:.2f}")
                print(f"  P95:                         {ttft_sorted[int(m * 0.95)] * 1000:.2f}")
                print(f"  P99:                         {ttft_sorted[int(m * 0.99)] * 1000:.2f}")
                print(f"  Max:                         {max(ttfts) * 1000:.2f}")

            # Total TTFT (including queueing) — captures user-perceived first-token latency
            ttft_totals = [r.ttft_total_s for r in ok if r.ttft_total_s > 0]
            if ttft_totals:
                ttft_total_sorted = sorted(ttft_totals)
                mt = len(ttft_total_sorted)
                print(f"\n  --- TTFT (total incl. queue, ms) ---")
                print(f"  Mean:                        {sum(ttft_totals) / mt * 1000:.2f}")
                print(f"  P50:                         {ttft_total_sorted[int(mt * 0.50)] * 1000:.2f}")
                print(f"  P90:                         {ttft_total_sorted[int(mt * 0.90)] * 1000:.2f}")
                print(f"  P95:                         {ttft_total_sorted[int(mt * 0.95)] * 1000:.2f}")
                print(f"  P99:                         {ttft_total_sorted[int(mt * 0.99)] * 1000:.2f}")
                print(f"  Max:                         {max(ttft_totals) * 1000:.2f}")

            if tpots:
                tpot_sorted = sorted(tpots)
                k = len(tpot_sorted)
                print(f"\n  --- TPOT (ms, excluding prefill) ---")
                print(f"  Mean:                        {sum(tpots) / k * 1000:.2f}")
                print(f"  P50:                         {tpot_sorted[int(k * 0.50)] * 1000:.2f}")
                print(f"  P90:                         {tpot_sorted[int(k * 0.90)] * 1000:.2f}")
                print(f"  P95:                         {tpot_sorted[int(k * 0.95)] * 1000:.2f}")
                print(f"  P99:                         {tpot_sorted[int(k * 0.99)] * 1000:.2f}")

            print(f"\n  --- Token counts & throughput ---")
            print(f"  Total input tokens:          {sum(in_toks)}")
            print(f"  Total output tokens:         {sum(out_toks)}")
            print(f"  Total tokens (in+out):       {sum(in_toks) + sum(out_toks)}")
            print(f"  Input throughput (tok/s):    {sum(in_toks) / wall_duration:.1f}")
            print(f"  Output throughput (tok/s):   {sum(out_toks) / wall_duration:.1f}")
            print(f"  Total throughput (tok/s):    {(sum(in_toks) + sum(out_toks)) / wall_duration:.1f}")

        # Energy
        if self._energy_before and self._energy_after:
            label = f" [{self.scenario_label}]" if self.scenario_label else ""
            _report_energy(self._energy_before, self._energy_after, label)

        print("=" * 70)

        # Dump raw results
        if self.dump_file:
            dump = []
            for r in self.results:
                dump.append({
                    "input_len": r.request.input_len,
                    "output_len": r.request.output_len,
                    "success": r.success,
                    "latency_s": round(r.latency_s, 4),
                    "ttft_ms": round(r.ttft_s * 1000, 2),
                    "ttft_total_ms": round(r.ttft_total_s * 1000, 2),
                    "tpot_ms": round(r.tpot_s * 1000, 2),
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "error": r.error,
                })
            out = {
                "scenario": self.scenario_label,
                "wall_duration_s": round(wall_duration, 2),
                "results": dump,
            }
            if self._energy_before and self._energy_after:
                energy_deltas = {}
                for idx in sorted(self._energy_before.keys()):
                    delta = self._energy_after.get(idx, 0) - self._energy_before.get(idx, 0)
                    energy_deltas[str(idx)] = round(delta, 2)
                out["energy_mj_delta"] = energy_deltas
            with open(self.dump_file, "w") as f:
                json.dump(out, f, indent=2)
            logger.info("Raw results dumped to %s", self.dump_file)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Replay Azure LLM inference trace or synthetic dataset against a SGLang server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="azure",
        choices=["azure", "sample"],
        help="Dataset source: 'azure' for CSV trace replay, 'sample' for synthetic requests",
    )
    parser.add_argument(
        "--trace",
        default="AzureLLMInferenceTrace_conv_1week.csv",
        help="Path to trace CSV (TIMESTAMP, ContextTokens, GeneratedTokens). "
             "The full 1-week trace is in AzureLLMInferenceTrace_conv_1week.csv.1 (~27M requests)."
             " The default file is a 3-min snippet (~4K requests).",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:50000",
        help="SGLang server URL",
    )
    parser.add_argument(
        "--speedup",
        type=float,
        default=1.0,
        help="Replay speedup factor (1.0 = real-time, 2.0 = 2x faster)",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=0,
        help="Stop after N requests (0 = all for azure; required for sample)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=200,
        help="Max concurrent in-flight requests",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-request timeout in seconds",
    )
    parser.add_argument(
        "--dump",
        type=str,
        default="",
        help="Dump raw per-request results to this JSON file",
    )
    parser.add_argument(
        "--monitor-energy",
        action="store_true",
        help="Monitor GPU energy consumption via DVFSController",
    )
    parser.add_argument(
        "--gpu-indices",
        type=str,
        default="0,1,2,7",
        help="Comma-separated GPU indices for energy monitoring",
    )
    parser.add_argument(
        "--scenario-label",
        type=str,
        default="",
        help="Label for this scenario (included in energy report and dump)",
    )

    # ── Sample dataset options ──────────────────────────────────────────
    sample_group = parser.add_argument_group("sample dataset options")
    sample_group.add_argument(
        "--sample-num",
        type=int,
        default=10,
        help="Number of synthetic requests (required when --dataset=sample)",
    )
    sample_group.add_argument(
        "--sample-input-len",
        type=int,
        default=1024,
        help="Fixed input/prompt length in tokens for all synthetic requests",
    )
    sample_group.add_argument(
        "--sample-output-len",
        type=int,
        default=128,
        help="Fixed output/completion length in tokens for all synthetic requests",
    )
    sample_group.add_argument(
        "--sample-qps",
        type=float,
        default=1.0,
        help="Target queries per second (determines arrival interval)",
    )
    sample_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible request generation",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Send N warmup requests before benchmark (results discarded)",
    )
    return parser.parse_args(argv)


async def main():
    args = parse_args()

    if args.dataset == "sample":
        if not args.max_requests:
            print("error: --max-requests is required when --dataset=sample")
            sys.exit(1)
        logger.info(
            "Dataset: sample | num=%d il=%d ol=%d qps=%.1f seed=%d | Server: %s | Concurrency: %d",
            args.max_requests, args.sample_input_len, args.sample_output_len,
            args.sample_qps, args.seed, args.url, args.concurrency,
        )
    else:
        logger.info(
            "Dataset: azure | Trace: %s | Server: %s | Speedup: %.1fx | Concurrency: %d",
            args.trace, args.url, args.speedup, args.concurrency,
        )

    # Initialise energy monitoring if requested
    energy_controllers = None
    if args.monitor_energy:
        energy_controllers = _init_energy_monitoring(args.gpu_indices)

    bench = TraceReplayBenchmark(
        csv_path=args.trace,
        url=args.url,
        speedup=args.speedup,
        max_requests=args.max_requests,
        concurrency=args.concurrency,
        timeout_s=args.timeout,
        dump_file=args.dump,
        energy_controllers=energy_controllers,
        scenario_label=args.scenario_label,
        dataset=args.dataset,
        sample_input_len=args.sample_input_len,
        sample_output_len=args.sample_output_len,
        sample_qps=args.sample_qps,
        seed=args.seed,
    )
    bench.warmup_requests = args.warmup
    await bench.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
