"""B-3a: Workload Metrics Collector — per-request SLO and GPU utilization tracking.

Runs on the PA (Prefill-Attention) scheduler.  Collects per-request TTFT/TPOT,
GPU utilization via NVML polling, and request length distributions.  Feeds
MonitoringWindow objects to WorkloadMonitor for Tier-1 re-planning detection.

Collection strategy (two modes, configured via ``cross_process`` flag):

  1. **Local-only** (default):  PA tracks its own metrics directly.  Other-pool
     utilisation is estimated from scheduler timing ratios and AFDReqInput
     messages from DA.  No extra processes or threads needed.

  2. **Cross-process** (future):  Each GPU process collects its own NVML stats
     and pushes them to PA via a lightweight ZMQ PUB/SUB channel.  PA
     subscribes and merges.

Usage (scheduler integration)::

    collector = WorkloadMetricsCollector(
        ttft_slo_ms=5000.0,
        tpot_slo_us=50000.0,
        window_s=30.0,
    )
    # Per-batch: after process_batch_result
    collector.record_batch(batch, is_decode=False)
    # Periodic: in tier1_monitor_check
    window = collector.build_window()
    if window:
        monitor.record_window(window)
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from sglang.srt.energy.workload_monitor import MonitoringWindow

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

_NVML_POLL_INTERVAL_S = 1.0       # GPU util sampling rate
_MAX_WINDOW_REQUESTS = 10_000     # cap per-window request records


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class _ReqRecord:
    """Per-request timing snapshot for one window."""
    rid: str = ""
    ttft_us: float = 0.0
    tpot_us: float = 0.0
    il: int = 0       # input length
    ol: int = 0       # output token count this window
    finished: bool = False


@dataclass
class _DecodeStep:
    """One decode iteration snapshot (received from DA via AFDReqInput)."""
    bs: int = 0
    t_iter_us: float = 0.0
    timestamp: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Collector
# ═══════════════════════════════════════════════════════════════════════════

class WorkloadMetricsCollector:
    """Collects per-request SLO metrics and GPU utilisation on PA.

    Monitors ALL GPUs in the AF-disaggregated system via NVML, regardless of
    CUDA_VISIBLE_DEVICES.  The ``attn_gpu_indices`` / ``ffn_gpu_indices``
    mapping is derived from the launch config's module→GPU assignments and
    passed through environment variables.

    Args:
        ttft_slo_ms:          TTFT SLO threshold (milliseconds).
        tpot_slo_us:          TPOT SLO threshold (microseconds).
        window_s:             Monitoring window duration.
        attn_gpu_indices:     NVML device indices for attention GPUs (PA, DA).
        ffn_gpu_indices:      NVML device indices for FFN GPUs (PF, DF).
        enable_nvml_polling:  Whether to start a background NVML polling thread.
    """

    def __init__(
        self,
        ttft_slo_ms: float = 5000.0,
        tpot_slo_us: float = 50000.0,
        window_s: float = 30.0,
        attn_gpu_indices: Optional[list[int]] = None,
        ffn_gpu_indices: Optional[list[int]] = None,
        enable_nvml_polling: bool = True,
        stats_path: Optional[str] = None,
    ):
        self._ttft_slo_ms = ttft_slo_ms
        self._tpot_slo_us = tpot_slo_us
        self._window_s = window_s
        self._stats_path = stats_path

        # GPU→pool mapping (NVML device indices)
        self._attn_gpu_indices: list[int] = attn_gpu_indices or []
        self._ffn_gpu_indices: list[int] = ffn_gpu_indices or []
        self._all_gpu_indices: list[int] = sorted(set(self._attn_gpu_indices + self._ffn_gpu_indices))

        # Per-window accumulators (reset each window)
        self._reqs: list[_ReqRecord] = []
        self._decode_steps: list[_DecodeStep] = []
        self._window_start = time.time()

        # Scheduler busy/idle tracking
        self._batch_start_ts: Optional[float] = None
        self._total_busy_us: float = 0.0
        self._total_idle_us: float = 0.0
        self._last_idle_start: Optional[float] = None
        self._batch_count: int = 0
        self._prefill_batch_count: int = 0
        self._decode_batch_count: int = 0

        # GPU utilisation from NVML: {nvml_index: [sm_util_samples]}
        self._gpu_util_samples: dict[int, list[float]] = {}
        self._nvml_thread: Optional[threading.Thread] = None
        self._nvml_stop = threading.Event()

        if enable_nvml_polling and self._all_gpu_indices:
            self._start_nvml_polling()

    # ── NVML polling (background thread) ─────────────────────────────────

    def _start_nvml_polling(self):
        """Start a daemon thread that samples GPU SM utilisation for all pools."""
        try:
            from pynvml import (
                nvmlDeviceGetCount,
                nvmlDeviceGetHandleByIndex,
                nvmlDeviceGetUtilizationRates,
                nvmlInit,
                nvmlShutdown,
            )
            nvmlInit()
            total_gpus = nvmlDeviceGetCount()
            gpu_handles: dict[int, object] = {}
            for idx in self._all_gpu_indices:
                if idx < total_gpus:
                    gpu_handles[idx] = nvmlDeviceGetHandleByIndex(idx)
                    self._gpu_util_samples[idx] = []
                else:
                    logger.warning(
                        "NVML GPU index %d out of range (total %d GPUs), skipping",
                        idx, total_gpus,
                    )

            if not gpu_handles:
                logger.warning("WorkloadMetricsCollector: no valid NVML GPUs found")
                try:
                    nvmlShutdown()
                except Exception:
                    pass
                return

            def _poll():
                while not self._nvml_stop.wait(_NVML_POLL_INTERVAL_S):
                    try:
                        for idx, handle in gpu_handles.items():
                            util = nvmlDeviceGetUtilizationRates(handle)
                            self._gpu_util_samples[idx].append(float(util.gpu))
                            # Cap samples per GPU
                            if len(self._gpu_util_samples[idx]) > 300:
                                self._gpu_util_samples[idx] = \
                                    self._gpu_util_samples[idx][-300:]
                    except Exception as e:
                        logger.warning(
                            "NVML poll thread died (GPU util tracking disabled): %s", e)
                        break
                try:
                    nvmlShutdown()
                except Exception:
                    pass

            self._nvml_thread = threading.Thread(target=_poll, daemon=True)
            self._nvml_thread.start()
            logger.info(
                "WorkloadMetricsCollector: NVML polling %d GPUs %s "
                "(attn=%s, ffn=%s)",
                len(gpu_handles), list(gpu_handles.keys()),
                self._attn_gpu_indices, self._ffn_gpu_indices,
            )
        except ImportError:
            logger.info("WorkloadMetricsCollector: pynvml not available, "
                        "GPU util tracking disabled")
        except Exception as e:
            logger.warning("WorkloadMetricsCollector: NVML init failed: %s", e)

    def stop(self):
        """Stop the NVML polling thread."""
        self._nvml_stop.set()
        if self._nvml_thread is not None:
            self._nvml_thread.join(timeout=2.0)

    # ── Batch-level recording ────────────────────────────────────────────

    def record_batch_start(self, is_prefill: bool):
        """Called when a batch begins GPU execution."""
        now = time.perf_counter()
        self._batch_start_ts = now
        # End idle period
        if self._last_idle_start is not None:
            self._total_idle_us += (now - self._last_idle_start) * 1_000_000
            self._last_idle_start = None

    def record_batch_end(
        self,
        batch,           # ScheduleBatch
        is_decode: bool,
        t_iter_us: float = 0.0,
    ):
        """Called after batch result is processed.

        Extracts per-request TTFT/TPOT from request time_stats and accumulates
        them for the current monitoring window.
        """
        now = time.perf_counter()

        # Poll cross-process decode timing from DA
        self._poll_decode_stats_file()

        # Update busy time
        if self._batch_start_ts is not None:
            self._total_busy_us += (now - self._batch_start_ts) * 1_000_000
            self._batch_start_ts = None

        self._batch_count += 1

        # Record per-request metrics
        for req in batch.reqs:
            ts = req.time_stats if hasattr(req, "time_stats") else None
            if ts is None:
                continue

            rec = _ReqRecord(rid=req.rid if hasattr(req, "rid") else "")

            # TTFT: time from API dispatch to first token (prefill finish)
            dispatch_t = getattr(ts, "api_server_dispatch_time", 0.0)
            prefill_finish_t = getattr(ts, "prefill_finished_time", 0.0)
            if dispatch_t > 0 and prefill_finish_t > 0:
                rec.ttft_us = max(0, (prefill_finish_t - dispatch_t) * 1_000_000)

            # TPOT: for decode iterations, time per output token
            # = decode iteration latency (from the last decode step)
            if is_decode and t_iter_us > 0:
                rec.tpot_us = t_iter_us

            # Completion: request-level TPOT = total decode time / num output tokens
            completion_t = getattr(ts, "completion_time", 0.0)
            if completion_t > 0 and prefill_finish_t > 0:
                rec.finished = True
                decode_dur_us = max(0, (completion_t - prefill_finish_t) * 1_000_000)
                # Count output tokens since prefill finished
                output_ids = getattr(req, "output_ids", [])
                ol = len(output_ids) if output_ids else 0
                if ol > 0 and decode_dur_us > 0:
                    rec.tpot_us = decode_dur_us / ol  # average TPOT
                rec.ol = ol

            # Input length
            origin_ids = getattr(req, "origin_input_ids", None)
            rec.il = len(origin_ids) if origin_ids else 0

            self._reqs.append(rec)

        # Cap to prevent memory growth
        if len(self._reqs) > _MAX_WINDOW_REQUESTS:
            self._reqs = self._reqs[-_MAX_WINDOW_REQUESTS:]

        # Track batch type
        if is_decode:
            self._decode_batch_count += 1
            if t_iter_us > 0:
                bs = batch.batch_size() if hasattr(batch, "batch_size") else 0
                self._decode_steps.append(_DecodeStep(
                    bs=bs, t_iter_us=t_iter_us, timestamp=now,
                ))
        else:
            self._prefill_batch_count += 1

    def record_idle_start(self):
        """Called when the scheduler enters idle (no batch to run)."""
        if self._last_idle_start is None:
            self._last_idle_start = time.perf_counter()

    def record_decode_step(self, bs: int, t_iter_us: float):
        """Record a decode step from DA (via AFDReqInput cross-process message).

        This allows PA to track decode TPOT even though it doesn't run decode
        batches directly.
        """
        self._decode_steps.append(_DecodeStep(
            bs=bs, t_iter_us=t_iter_us, timestamp=time.perf_counter(),
        ))

    def _poll_decode_stats_file(self):
        """Read the latest decode iteration timing from the DA→PA stats file."""
        if not self._stats_path:
            return
        try:
            import json
            with open(self._stats_path) as f:
                stats = json.load(f)
            ts = stats.get("timestamp", 0)
            # Only use data within the current window
            if ts >= self._window_start and stats.get("t_iter_us", 0) > 0:
                self._decode_steps.append(_DecodeStep(
                    bs=stats.get("bs", 0),
                    t_iter_us=stats["t_iter_us"],
                    timestamp=ts,
                ))
        except FileNotFoundError:
            pass  # expected before first write from DA
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning("Error reading decode stats from %s: %s", self._stats_path, e)

    # ── Window building ──────────────────────────────────────────────────

    def build_window(self) -> Optional[MonitoringWindow]:
        """Build a MonitoringWindow from all data accumulated since last reset.

        Returns None if the window has not yet elapsed or there is no data.
        """
        elapsed = time.time() - self._window_start
        if elapsed < self._window_s * 0.5 and len(self._reqs) < 10:
            return None  # not enough data yet

        # ── Cross-process decode timing (DA → PA via shared stats file) ──
        self._poll_decode_stats_file()

        # ── SLO violation rate ────────────────────────────────────────
        ttft_violations = 0
        tpot_violations = 0
        total_requests = max(len(self._reqs), 1)
        n_with_ttft_data = sum(1 for r in self._reqs if r.ttft_us > 0)
        n_with_tpot_data = sum(1 for r in self._reqs if r.tpot_us > 0)
        if total_requests > 0 and n_with_ttft_data == 0 and n_with_tpot_data == 0:
            logger.warning(
                "build_window: %d reqs in window but ZERO have TTFT data "
                "(api_server_dispatch_time or prefill_finished_time missing) "
                "and ZERO have TPOT data (no decode steps from DA). "
                "stats_path=%s decode_steps=%d",
                total_requests, self._stats_path, len(self._decode_steps),
            )

        il_buckets: dict[str, int] = {}  # for load distribution

        for rec in self._reqs:
            # TTFT violation
            if rec.ttft_us > 0 and rec.ttft_us > self._ttft_slo_ms * 1000:
                ttft_violations += 1
            # TPOT violation (only for finished requests with TPOT data)
            if rec.tpot_us > 0 and rec.tpot_us > self._tpot_slo_us:
                tpot_violations += 1
            # IL distribution bucketing
            bucket = _il_bucket(rec.il)
            il_buckets[bucket] = il_buckets.get(bucket, 0) + 1

        # Also count violations from decode step iteration times
        decode_tpot_violations = sum(
            1 for s in self._decode_steps
            if s.t_iter_us > self._tpot_slo_us
        )

        # SLO violation: a request violates if EITHER TTFT or TPOT exceeds SLO
        # Count unique requests with violations (TTFT or TPOT)
        # Use the higher of the two for the violation rate
        n_with_ttft = sum(1 for r in self._reqs if r.ttft_us > 0)
        n_with_tpot = sum(1 for r in self._reqs if r.tpot_us > 0 and r.finished)
        n_ttft_viol = sum(1 for r in self._reqs if r.ttft_us > self._ttft_slo_ms * 1000)
        n_tpot_viol = sum(1 for r in self._reqs if r.finished and r.tpot_us > self._tpot_slo_us)

        slo_violation_rate = 0.0
        if n_with_ttft > 0 or n_with_tpot > 0 or self._decode_steps:
            ttft_rate = n_ttft_viol / max(n_with_ttft, 1)
            tpot_rate = n_tpot_viol / max(n_with_tpot, 1)
            # Include decode step violation rate
            if self._decode_steps:
                decode_viol_rate = decode_tpot_violations / len(self._decode_steps)
                tpot_rate = max(tpot_rate, decode_viol_rate)
            slo_violation_rate = max(ttft_rate, tpot_rate)

        # ── GPU utilisation ───────────────────────────────────────────
        # a_util: average of attention GPUs (PA + DA)
        # f_util: average of FFN GPUs (PF + DF)
        # p_util: average of prefill GPUs = PA (attn) + PF (ffn) — proxied by all-GPU average
        # d_util: average of decode GPUs = DA (attn) + DF (ffn) — proxied by decode step timing

        a_util = _avg_gpu_util(self._gpu_util_samples, self._attn_gpu_indices)
        f_util = _avg_gpu_util(self._gpu_util_samples, self._ffn_gpu_indices)

        # p_util: from prefill busy ratio (PA + PF)
        total_tracked_us = self._total_busy_us + self._total_idle_us
        if total_tracked_us > 0:
            p_util = self._total_busy_us / total_tracked_us
        elif self._prefill_batch_count > 0:
            p_util = a_util  # fallback to attention util
        else:
            p_util = 0.0

        # d_util: estimate from decode step timing vs wall clock
        d_util = 0.0
        if self._decode_steps:
            total_decode_us = sum(s.t_iter_us for s in self._decode_steps)
            window_us = elapsed * 1_000_000
            d_util = min(total_decode_us / max(window_us, 1), 1.0)
        elif self._decode_batch_count > 0:
            d_util = max(a_util, f_util)

        # ── Load distribution ─────────────────────────────────────────
        total_buckets = sum(il_buckets.values())
        load_dist = {}
        if total_buckets > 0:
            load_dist = {k: v / total_buckets for k, v in il_buckets.items()}

        # ── P99 latency ───────────────────────────────────────────────
        ttft_values = sorted(
            [r.ttft_us for r in self._reqs if r.ttft_us > 0]
        )
        # TPOT from finished requests + decode step iteration times
        tpot_values = sorted(
            [r.tpot_us for r in self._reqs if r.finished and r.tpot_us > 0]
            + [s.t_iter_us for s in self._decode_steps if s.t_iter_us > 0]
        )
        ttft_p99 = _p99(ttft_values) if ttft_values else 0.0
        tpot_p99 = _p99(tpot_values) if tpot_values else 0.0

        # ── Active requests ───────────────────────────────────────────
        # Count requests that were processed in this window
        n_active = sum(1 for r in self._reqs if not r.finished)

        window = MonitoringWindow(
            slo_violation_rate=slo_violation_rate,
            a_util=a_util,
            f_util=f_util,
            p_util=p_util,
            d_util=d_util,
            load_distribution=load_dist,
            active_requests=n_active,
            tpot_p99_us=tpot_p99,
            ttft_p99_ms=ttft_p99 / 1000.0,
        )

        return window

    def reset_window(self):
        """Reset all per-window accumulators (call after build_window)."""
        self._reqs.clear()
        self._decode_steps.clear()
        self._window_start = time.time()
        self._total_busy_us = 0.0
        self._total_idle_us = 0.0
        self._batch_count = 0
        self._prefill_batch_count = 0
        self._decode_batch_count = 0
        for idx in self._gpu_util_samples:
            self._gpu_util_samples[idx].clear()
        self._last_idle_start = None
        self._batch_start_ts = None

    @property
    def window_elapsed_s(self) -> float:
        return time.time() - self._window_start


# ── Helpers ────────────────────────────────────────────────────────────────


def _il_bucket(il: int) -> str:
    """Bucket input length into coarse bins for distribution tracking."""
    if il <= 0:
        return "il_0"
    if il <= 128:
        return "il_128"
    if il <= 512:
        return "il_512"
    if il <= 1024:
        return "il_1024"
    if il <= 2048:
        return "il_2048"
    if il <= 4096:
        return "il_4096"
    return "il_8192p"


def _p99(sorted_values: list[float]) -> float:
    """Return the 99th percentile from a sorted list."""
    if not sorted_values:
        return 0.0
    idx = int(len(sorted_values) * 0.99)
    idx = min(idx, len(sorted_values) - 1)
    return sorted_values[idx]


def _avg_gpu_util(
    gpu_samples: dict[int, list[float]], indices: list[int],
) -> float:
    """Average SM utilisation (0–1) across the given NVML GPU indices."""
    if not indices:
        return 0.0
    all_samples: list[float] = []
    for idx in indices:
        all_samples.extend(gpu_samples.get(idx, []))
    if not all_samples:
        return 0.0
    return sum(all_samples) / len(all_samples) / 100.0
