"""Local host-KV admission and quota-transfer state for LatticeKV.

This module is deliberately free of CUDA, tensors, and radix-tree operations.
It only decides whether one model instance has enough immediately writable or
prepared host pages to keep accepting HBM backup, and whether it can safely
offer surplus capacity to another instance.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum
import math


class LocalAction(StrEnum):
    """The local action required to preserve host-KV admission capacity."""

    IDLE = "idle"
    PREPARE = "prepare"
    EMERGENCY = "emergency"
    QUOTA_DEFICIT = "quota_deficit"


@dataclass(frozen=True)
class Watermarks:
    min_pages: int
    low_pages: int
    high_pages: int


@dataclass(frozen=True)
class DonationOffer:
    """Capacity an instance can supply without compromising local recovery."""

    immediate_pages: int
    ready_pages: int
    low_loss_pages: int

    @property
    def total_pages(self) -> int:
        return self.immediate_pages + self.ready_pages + self.low_loss_pages


@dataclass
class QuotaTransfer:
    transfer_id: int
    target_pages: int
    assigned_pages: int = 0
    acknowledged_pages: int = 0


@dataclass(frozen=True)
class BackupIngressSample:
    observed_at_s: float
    pages: int
    pages_per_s: float


@dataclass(frozen=True)
class ReclaimThroughputSample:
    observed_at_s: float
    pages: int
    elapsed_s: float


class LocalResidencyState:
    """Maintain one model's clean/ready host-page safety envelope.

    ``ready_pages`` means pages whose low-value contents have already passed
    policy checks but still need final release.  They remain physically live
    until the host allocator releases them, so ``live_pages`` may include them.
    """

    def __init__(
        self,
        *,
        effective_pages: int,
        floor_pages: int,
        large_backup_batch_pages: int,
        backup_ingress_pages_per_s: float,
        ready_latency_p95_s: float,
        maintenance_batch_pages: int,
        maintenance_cycle_s: float,
        retention_debt_cooldown_s: float,
        clean_shortfall_cooldown_s: float | None = None,
        ingress_window_s: float = 5.0,
        backup_batch_window_s: float = 60.0,
        reclaim_throughput_window_s: float = 60.0,
        min_reclaim_throughput_samples: int = 4,
        feedback_horizon_min_s: float = 5.0,
        feedback_horizon_max_s: float = 300.0,
        loss_aware_maintenance: bool = True,
    ) -> None:
        if effective_pages <= 0:
            raise ValueError("effective_pages must be positive")
        if not 0 <= floor_pages <= effective_pages:
            raise ValueError("floor_pages must be within effective capacity")
        if min(large_backup_batch_pages, maintenance_batch_pages) < 0:
            raise ValueError("page counts must be non-negative")
        if min(
            backup_ingress_pages_per_s,
            ready_latency_p95_s,
            maintenance_cycle_s,
            retention_debt_cooldown_s,
        ) < 0:
            raise ValueError("timing inputs must be non-negative")
        if (
            ingress_window_s <= 0
            or backup_batch_window_s <= 0
            or reclaim_throughput_window_s <= 0
        ):
            raise ValueError("sample windows must be positive")
        if min_reclaim_throughput_samples <= 0:
            raise ValueError("min reclaim throughput samples must be positive")
        if (
            feedback_horizon_min_s <= 0
            or feedback_horizon_max_s < feedback_horizon_min_s
        ):
            raise ValueError("feedback horizon bounds must be positive and ordered")

        self.effective_pages = effective_pages
        self.floor_pages = floor_pages
        self.large_backup_batch_pages = large_backup_batch_pages
        self._baseline_backup_ingress_pages_per_s = backup_ingress_pages_per_s
        self.backup_ingress_pages_per_s = backup_ingress_pages_per_s
        self.ingress_window_s = ingress_window_s
        # Rate and size are independent admission facts.  A short ingress
        # window is useful for quickly forgetting an old burst rate, but it
        # must not also forget that one normal HBM eviction can be a large
        # node-sized backup.  Keep a longer bounded sample window for the
        # p95 batch size used by the hard admission minimum.
        self.backup_batch_window_s = backup_batch_window_s
        self.reclaim_throughput_window_s = reclaim_throughput_window_s
        self.min_reclaim_throughput_samples = min_reclaim_throughput_samples
        self.feedback_horizon_min_s = feedback_horizon_min_s
        self.feedback_horizon_max_s = feedback_horizon_max_s
        self.loss_aware_maintenance = loss_aware_maintenance
        self._backup_ingress_samples: deque[BackupIngressSample] = deque()
        self._backup_batch_samples: deque[BackupIngressSample] = deque()
        self._last_backup_observation_s: float | None = None
        self._baseline_ready_latency_p95_s = ready_latency_p95_s
        self._reclaim_latency_samples: deque[float] = deque(maxlen=64)
        self._reclaim_throughput_samples: deque[ReclaimThroughputSample] = deque()
        self.ready_latency_p95_s = ready_latency_p95_s
        self.maintenance_batch_pages = maintenance_batch_pages
        self.maintenance_cycle_s = maintenance_cycle_s
        self.retention_debt_cooldown_s = retention_debt_cooldown_s
        self.clean_shortfall_cooldown_s = (
            retention_debt_cooldown_s
            if clean_shortfall_cooldown_s is None
            else clean_shortfall_cooldown_s
        )
        if self.clean_shortfall_cooldown_s <= 0:
            raise ValueError("clean_shortfall_cooldown_s must be positive")

        self.clean_pages = effective_pages
        self.ready_pages = 0
        self.live_pages = 0
        self._preparing = False
        self._last_retention_loss_s: float | None = None
        self._retention_debt_tokens = 0
        self._retention_debt_epoch = 0
        # A shortfall means the local value-aware queue could not produce the
        # clean pages required for either a current HBM backup or the hard
        # admission minimum.  Unlike retention debt, it does not wait for a
        # later revisit to prove that the quota is already insufficient.
        self.unresolved_clean_pages = 0
        self._last_clean_shortfall_s: float | None = None
        self._next_transfer_id = 1
        self._outstanding_transfer: QuotaTransfer | None = None

    def _prune_ingress_samples(self, now_s: float) -> None:
        if now_s < 0:
            raise ValueError("now_s must be non-negative")
        while (
            self._backup_ingress_samples
            and now_s - self._backup_ingress_samples[0].observed_at_s
            >= self.ingress_window_s
        ):
            self._backup_ingress_samples.popleft()
        # ``pages_per_s`` attached to an individual copy describes PCIe/DMA
        # throughput, not how fast independent HBM eviction batches arrive.
        # A single fast backup must raise ``min_pages`` through its batch size,
        # but must not turn into a fictitious sustained host-ingress rate that
        # drives low/high to the whole quota.  Only elapsed wall-clock time
        # between multiple backup arrivals can establish that rate.
        if len(self._backup_ingress_samples) < 2:
            self.backup_ingress_pages_per_s = self._baseline_backup_ingress_pages_per_s
            return
        first = self._backup_ingress_samples[0]
        last = self._backup_ingress_samples[-1]
        interval_s = last.observed_at_s - first.observed_at_s
        if interval_s <= 0:
            self.backup_ingress_pages_per_s = self._baseline_backup_ingress_pages_per_s
            return
        pages_after_first = sum(
            sample.pages for sample in list(self._backup_ingress_samples)[1:]
        )
        self.backup_ingress_pages_per_s = max(
            self._baseline_backup_ingress_pages_per_s,
            pages_after_first / interval_s,
        )

    def _prune_backup_batch_samples(self, now_s: float) -> None:
        while (
            self._backup_batch_samples
            and now_s - self._backup_batch_samples[0].observed_at_s
            >= self.backup_batch_window_s
        ):
            self._backup_batch_samples.popleft()

    def _recent_backup_batch_pages(self) -> int:
        if not self._backup_batch_samples:
            return self.large_backup_batch_pages
        # Host admission must cover a normal large HBM eviction.  p95 avoids
        # pinning a permanent reserve because of one pathological outlier,
        # while preserving the ordinary node-sized batches that a short rate
        # window would otherwise forget before the next burst arrives.
        batches = sorted(sample.pages for sample in self._backup_batch_samples)
        index = max(0, math.ceil(len(batches) * 0.95) - 1)
        return max(self.large_backup_batch_pages, batches[index])

    def _prune_reclaim_throughput_samples(self, now_s: float) -> None:
        while (
            self._reclaim_throughput_samples
            and now_s - self._reclaim_throughput_samples[0].observed_at_s
            >= self.reclaim_throughput_window_s
        ):
            self._reclaim_throughput_samples.popleft()

    def safe_reclaim_pages_per_s(self, *, now_s: float) -> float | None:
        """Return a conservative observed local-reclaim throughput.

        This is deliberately measured from completed, value-aware reclaim
        waves.  It is not a DMA bandwidth number and it does not assume that
        the next leaf will be equally cheap.  A lower percentile prevents one
        unusually fast leaf from convincing the scheduler that local cache
        circulation can sustain a burst it cannot actually keep up with.
        """
        self._prune_reclaim_throughput_samples(now_s)
        if len(self._reclaim_throughput_samples) < self.min_reclaim_throughput_samples:
            return None
        rates = sorted(
            sample.pages / sample.elapsed_s
            for sample in self._reclaim_throughput_samples
            if sample.elapsed_s > 0
        )
        if not rates:
            return None
        percentile_index = max(0, math.ceil(len(rates) * 0.20) - 1)
        return rates[percentile_index]

    def _watermarks(self) -> Watermarks:
        ingress_during_ready = math.ceil(
            self.backup_ingress_pages_per_s * self.ready_latency_p95_s
        )
        largest_recent_batch = self._recent_backup_batch_pages()
        # A normal HBM backup can arrive while a selected ready page is still
        # becoming clean.  Both claims are concurrent admission demand: the
        # next discrete backup needs its full batch and new ingress continues
        # during the ready-to-clean p95.  Taking their maximum under-reserves
        # by up to one batch and can force a transient emergency despite ready
        # candidates already being available.
        min_pages = largest_recent_batch + ingress_during_ready
        # ``low`` is the point where the background maintainer must wake,
        # not merely an alias for the emergency minimum.  It must leave one
        # full maintenance cycle of observed HBM->host ingress, otherwise a
        # burst can cross ``min`` before a cooperative page-reclaim wave has
        # any chance to run.  ``high`` keeps one more cycle after recovery so
        # the worker does not immediately oscillate at its wake-up boundary.
        # A quiet instance still retains the configured small maintenance
        # quantum rather than a permanent percentage of its quota.
        cycle_ingress = math.ceil(
            self.backup_ingress_pages_per_s * self.maintenance_cycle_s
        )
        maintenance_slack = max(self.maintenance_batch_pages, cycle_ingress)
        low_pages = min_pages + maintenance_slack
        high_pages = low_pages + maintenance_slack
        high_pages = min(high_pages, self.effective_pages)
        low_pages = min(low_pages, high_pages)
        min_pages = min(min_pages, low_pages)
        return Watermarks(min_pages=min_pages, low_pages=low_pages, high_pages=high_pages)

    @property
    def watermarks(self) -> Watermarks:
        return self._watermarks()

    def watermarks_at(self, *, now_s: float) -> Watermarks:
        self._prune_ingress_samples(now_s)
        self._prune_backup_batch_samples(now_s)
        self._prune_reclaim_throughput_samples(now_s)
        return self._watermarks()

    @property
    def available_pages(self) -> int:
        return self.clean_pages + self.ready_pages

    @property
    def retention_debt_tokens(self) -> int:
        return self._retention_debt_tokens

    def active_retention_debt_tokens(self, *, now_s: float) -> int:
        """Return debt that is still relevant to a current quota decision.

        A reclaim loss is deliberately short-lived feedback, not a permanent
        label on an instance.  Status publication must advance the cooldown
        too; otherwise a model that is never asked to donate keeps reporting
        stale debt forever and the global scheduler sees a false shortage.
        """
        return self._retention_debt_tokens if self._debt_active(now_s) else 0

    def feedback_horizon_s(self, *, now_s: float) -> float:
        """Bound a reclaim-loss ghost to this instance's current turnover time.

        This is not a continuation-probability prediction. It asks a narrower
        question: did the reclaimed prefix return before host capacity could
        naturally turn over at the observed HBM-to-host ingress rate? If so,
        the reclaim is evidence of a current retention shortage. Otherwise it
        is historical traffic and must not permanently block donation/growth.
        """
        self._prune_ingress_samples(now_s)
        if self.backup_ingress_pages_per_s <= 0:
            horizon = self.retention_debt_cooldown_s
        else:
            horizon = self.effective_pages / self.backup_ingress_pages_per_s
        return min(
            self.feedback_horizon_max_s,
            max(self.feedback_horizon_min_s, horizon),
        )

    def observe_pages(
        self,
        *,
        clean_pages: int,
        ready_pages: int,
        live_pages: int,
        effective_pages: int | None = None,
    ) -> None:
        if effective_pages is None:
            effective_pages = self.effective_pages
        if effective_pages < self.floor_pages:
            raise ValueError("effective capacity cannot drop below the local floor")
        if min(clean_pages, ready_pages, live_pages) < 0:
            raise ValueError("page counts must be non-negative")
        if clean_pages + live_pages > effective_pages:
            raise ValueError("clean plus live pages exceeds effective capacity")
        if ready_pages > live_pages:
            raise ValueError("ready pages must still be live pages")
        self.effective_pages = effective_pages
        self.clean_pages = clean_pages
        self.ready_pages = ready_pages
        self.live_pages = live_pages

    def observe_backup(
        self, *, pages: int, elapsed_s: float, now_s: float | None = None
    ) -> None:
        """Update host-admission demand from an HBM backup arrival.

        ``elapsed_s`` is preserved for transport telemetry and API
        compatibility, but deliberately does not define host ingress: it is
        the time to copy this one batch, not the time until the next batch.
        """
        if pages <= 0 or elapsed_s <= 0:
            raise ValueError("backup observation must be positive")
        if now_s is None:
            now_s = (self._last_backup_observation_s or 0.0) + elapsed_s
        if now_s < 0:
            raise ValueError("now_s must be non-negative")
        if (
            self._last_backup_observation_s is not None
            and now_s < self._last_backup_observation_s
        ):
            raise ValueError("backup observation time cannot move backwards")
        self._prune_ingress_samples(now_s)
        self._backup_ingress_samples.append(
            BackupIngressSample(
                observed_at_s=now_s,
                pages=pages,
                pages_per_s=pages / elapsed_s,
            )
        )
        self._backup_batch_samples.append(
            BackupIngressSample(
                observed_at_s=now_s,
                pages=pages,
                pages_per_s=pages / elapsed_s,
            )
        )
        self._last_backup_observation_s = now_s
        self._prune_ingress_samples(now_s)

    def observe_reclaim_latency(self, *, elapsed_s: float) -> None:
        """Learn how long this instance needs to turn a safe page into clean.

        Watermarks protect the HBM backup path from that observed delay.  A
        bounded p95 avoids both a hard-coded latency and one transient sample
        permanently inflating the instance's local slack requirement.
        """
        if elapsed_s <= 0:
            raise ValueError("reclaim latency must be positive")
        self._reclaim_latency_samples.append(elapsed_s)
        samples = sorted(self._reclaim_latency_samples)
        p95_index = max(0, math.ceil(len(samples) * 0.95) - 1)
        self.ready_latency_p95_s = max(
            self._baseline_ready_latency_p95_s, samples[p95_index]
        )

    def observe_reclaim_result(
        self, *, pages: int, elapsed_s: float, now_s: float
    ) -> None:
        """Record one completed value-aware reclaim wave for local capacity.

        The caller only reports pages that really became allocator-free.  A
        candidate which was merely marked ready has not created HBM admission
        capacity and must not inflate the reclaim rate.
        """
        if pages <= 0 or elapsed_s <= 0 or now_s < 0:
            raise ValueError("reclaim result inputs must be positive")
        self.observe_reclaim_latency(elapsed_s=elapsed_s)
        self._prune_reclaim_throughput_samples(now_s)
        self._reclaim_throughput_samples.append(
            ReclaimThroughputSample(
                observed_at_s=now_s, pages=pages, elapsed_s=elapsed_s
            )
        )

    def turnover_deficit_pages(self, *, now_s: float) -> int:
        """Return an early grow signal when local circulation cannot keep up.

        A large ingress rate alone is not a grow request: clean capacity above
        ``low`` is exactly what the local maintainer is meant to absorb.  The
        deficit appears only after clean falls below ``low`` *and* repeated
        measurements show that value-aware reclaim is slower than backup
        ingress.  The returned pages cover one maintenance cycle of the
        measured rate gap, not an attempt to predict all future traffic.
        """
        watermarks = self.watermarks_at(now_s=now_s)
        safe_rate = self.safe_reclaim_pages_per_s(now_s=now_s)
        if (
            safe_rate is None
            or self.clean_pages >= watermarks.low_pages
            or self.backup_ingress_pages_per_s <= safe_rate
        ):
            return 0
        return math.ceil(
            (self.backup_ingress_pages_per_s - safe_rate)
            * self.maintenance_cycle_s
        )

    def action(self, *, now_s: float) -> LocalAction:
        if now_s < 0:
            raise ValueError("now_s must be non-negative")
        watermarks = self.watermarks_at(now_s=now_s)
        # ``ready`` pages have already passed policy checks but still contain
        # KV. HBM backup can consume only clean pages. Therefore ready pages
        # may describe how quickly a maintainer can recover, but must not
        # suppress the low-watermark work that turns candidates into actual
        # writable capacity.
        if self.clean_pages < watermarks.min_pages:
            self._preparing = True
            return LocalAction.EMERGENCY
        # Once real revisit loss and a measured reclaim-throughput deficit
        # agree, another ordinary recovery wave would only keep deleting
        # history that the instance has evidence it cannot retain at this
        # quota.  Report the deficit to the global layer and reserve further
        # deletion for an actual admission emergency.
        if (
            self.value_constrained(now_s=now_s)
            and self.clean_pages < watermarks.low_pages
        ):
            self._preparing = False
            return LocalAction.QUOTA_DEFICIT
        recovery_target = self.maintenance_target_pages(now_s=now_s)
        if self.clean_pages < recovery_target:
            self._preparing = True
        elif self._preparing and self.clean_pages >= recovery_target:
            self._preparing = False
        return LocalAction.PREPARE if self._preparing else LocalAction.IDLE

    def maintenance_target_pages(self, *, now_s: float) -> int:
        """Return the clean-page target the local maintainer may pursue.

        Normally the maintainer restores ``high`` so an HBM backup burst does
        not force synchronous reclaim.  A recent reclaim loss is different:
        it is evidence that this instance's local value ordering is already
        trimming useful state.  In that feedback window, continuing to delete
        merely to refill the speculative high watermark compounds the loss.
        Keep enough clean capacity to reach ``low`` and admit the next normal
        batch, then let a continued low-watermark failure ask the global layer
        for more quota.  This is a local safety valve, not a future-reuse
        prediction.
        """
        watermarks = self.watermarks_at(now_s=now_s)
        return (
            watermarks.low_pages
            if (
                self.turnover_deficit_pages(now_s=now_s)
                or (self.loss_aware_maintenance and self._debt_active(now_s))
            )
            else watermarks.high_pages
        )

    def value_constrained(self, *, now_s: float) -> bool:
        """Whether recent observed reclaim loss limits proactive reclaim."""
        return self.loss_aware_maintenance and self._debt_active(now_s)

    def record_retention_loss(self, *, reprefill_tokens: int, now_s: float) -> None:
        if reprefill_tokens <= 0:
            raise ValueError("reprefill_tokens must be positive")
        if now_s < 0:
            raise ValueError("now_s must be non-negative")
        self._retention_debt_tokens += reprefill_tokens
        self._retention_debt_epoch += 1
        self._last_retention_loss_s = now_s

    @property
    def retention_debt_epoch(self) -> int:
        return self._retention_debt_epoch

    def record_clean_shortfall(self, *, pages: int, now_s: float) -> None:
        """Publish a bounded admission failure for global help.

        The record deliberately outlives the immediate generic eviction that
        let one backup proceed.  Otherwise the global scheduler never sees
        that this instance had to abandon its value-aware local policy.
        """
        if pages < 0:
            raise ValueError("clean shortfall pages must be non-negative")
        if now_s < 0:
            raise ValueError("now_s must be non-negative")
        if pages:
            self.unresolved_clean_pages = max(self.unresolved_clean_pages, pages)
            self._last_clean_shortfall_s = now_s

    def active_clean_shortfall_pages(self, *, now_s: float) -> int:
        if self._last_clean_shortfall_s is None:
            return 0
        if now_s < self._last_clean_shortfall_s:
            raise ValueError("now_s cannot move backwards")
        if now_s - self._last_clean_shortfall_s < self.clean_shortfall_cooldown_s:
            return self.unresolved_clean_pages
        self.unresolved_clean_pages = 0
        self._last_clean_shortfall_s = None
        return 0

    def _debt_active(self, now_s: float) -> bool:
        if self._last_retention_loss_s is None:
            return False
        if now_s < self._last_retention_loss_s:
            raise ValueError("now_s cannot move backwards")
        if now_s - self._last_retention_loss_s < self.feedback_horizon_s(now_s=now_s):
            return True
        self._last_retention_loss_s = None
        self._retention_debt_tokens = 0
        return False

    def offer_donation(self, *, requested_pages: int, now_s: float) -> DonationOffer:
        if requested_pages < 0:
            raise ValueError("requested_pages must be non-negative")
        if self._debt_active(now_s) or self.active_clean_shortfall_pages(now_s=now_s):
            return DonationOffer(0, 0, 0)
        watermarks = self.watermarks_at(now_s=now_s)
        local_surplus = max(0, self.available_pages - watermarks.high_pages)
        floor_limited = max(0, self.effective_pages - self.floor_pages)
        immediate = min(
            requested_pages,
            floor_limited,
            max(0, self.clean_pages - watermarks.high_pages),
        )
        ready = min(
            requested_pages - immediate,
            floor_limited - immediate,
            local_surplus - immediate,
        )
        return DonationOffer(immediate_pages=immediate, ready_pages=ready, low_loss_pages=0)

    def begin_intent(self, *, target_pages: int) -> QuotaTransfer:
        if target_pages < self.effective_pages:
            raise ValueError("this state machine handles recipient growth only")
        if self._outstanding_transfer is not None:
            raise RuntimeError("another quota transfer is still outstanding")
        transfer = QuotaTransfer(
            transfer_id=self._next_transfer_id,
            target_pages=target_pages,
        )
        self._next_transfer_id += 1
        self._outstanding_transfer = transfer
        return transfer

    def _transfer(self, transfer_id: int) -> QuotaTransfer:
        transfer = self._outstanding_transfer
        if transfer is None or transfer.transfer_id != transfer_id:
            raise ValueError("unknown quota transfer")
        return transfer

    def mark_pages_assigned(self, transfer_id: int, *, pages: int) -> None:
        if pages <= 0:
            raise ValueError("assigned pages must be positive")
        transfer = self._transfer(transfer_id)
        remaining = transfer.target_pages - self.effective_pages - transfer.assigned_pages
        if pages > remaining:
            raise ValueError("assigned pages exceed quota intent")
        transfer.assigned_pages += pages

    def acknowledge_recipient(self, transfer_id: int, *, pages: int) -> None:
        if pages <= 0:
            raise ValueError("acknowledged pages must be positive")
        transfer = self._transfer(transfer_id)
        if transfer.acknowledged_pages + pages > transfer.assigned_pages:
            raise ValueError("cannot acknowledge pages that were not assigned")
        transfer.acknowledged_pages += pages
        self.effective_pages += pages
        if self.effective_pages == transfer.target_pages:
            self._outstanding_transfer = None
