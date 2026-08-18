"""Replaceable D-signal scheduler layers for LatticeKV quota trials.

The module is CPU-only.  It does not observe traffic and it does not claim
effective quota.  It separates facts, signal interpretation, trial policy,
resource planning, and outcome recording so D mappings can be iterated later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import time
from typing import Protocol


@dataclass(frozen=True)
class WindowObservation:
    """Immutable facts for one instance in one observation window."""

    model_id: str
    window_start_s: float
    window_end_s: float
    effective_pages: int
    floor_pages: int
    raw_retention_loss_tokens: int
    confirmed_reuse_loss_tokens: int
    revisit_opportunity_tokens: int
    safe_clean_offer_pages: int
    safe_ready_offer_pages: int
    admission_shortfall_pages: int
    host_evicted_tokens: int
    reprefill_tokens: int
    revisit_cache_match_tokens: int
    revisit_cache_miss_tokens: int = 0
    revisit_incremental_tokens: int = 0
    revisit_seen_cache_miss_tokens: int = 0
    d_coverage_of_seen_miss: float | None = None
    conditional_ttft_ms_p50: float | None = None
    legacy_retention_debt_tokens: int = 0

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        values = (
            self.effective_pages,
            self.floor_pages,
            self.raw_retention_loss_tokens,
            self.confirmed_reuse_loss_tokens,
            self.revisit_opportunity_tokens,
            self.safe_clean_offer_pages,
            self.safe_ready_offer_pages,
            self.admission_shortfall_pages,
            self.host_evicted_tokens,
            self.reprefill_tokens,
            self.revisit_cache_match_tokens,
            self.revisit_cache_miss_tokens,
            self.revisit_incremental_tokens,
            self.revisit_seen_cache_miss_tokens,
            self.legacy_retention_debt_tokens,
        )
        if min(values) < 0:
            raise ValueError("observation values must be non-negative")
        if (
            self.d_coverage_of_seen_miss is not None
            and self.d_coverage_of_seen_miss < 0
        ):
            raise ValueError("D coverage must be non-negative")
        if self.floor_pages > self.effective_pages:
            raise ValueError("floor_pages cannot exceed effective_pages")
        if self.window_end_s <= self.window_start_s:
            raise ValueError("window_end_s must be greater than window_start_s")
        if (
            self.conditional_ttft_ms_p50 is not None
            and self.conditional_ttft_ms_p50 < 0
        ):
            raise ValueError("conditional TTFT must be non-negative")

    @property
    def duration_s(self) -> float:
        return self.window_end_s - self.window_start_s

    @property
    def floor_limited_offer_pages(self) -> int:
        return max(0, self.effective_pages - self.floor_pages)

    def offer_breakdown(self) -> tuple[int, int]:
        """Return clean and ready pages that can be planned for donation."""
        if self.admission_shortfall_pages:
            return (0, 0)
        clean = min(self.floor_limited_offer_pages, self.safe_clean_offer_pages)
        ready = min(
            max(0, self.floor_limited_offer_pages - clean),
            self.safe_ready_offer_pages,
        )
        return (clean, ready)


@dataclass(frozen=True)
class DSignal:
    model_id: str
    d_raw_tokens: int
    revisit_opportunity_tokens: int
    severity: float
    volume_loss_tokens_per_s: float
    reprefill_tokens_per_s: float
    triggered: bool
    reason: str
    donor_allowed: bool


class SignalPolicy(Protocol):
    def evaluate(self, observation: WindowObservation) -> DSignal: ...


class InitialDSignalPolicy:
    """Initial, replaceable D mapping: raw + severity + volume."""

    def __init__(
        self,
        *,
        min_revisit_opportunity_tokens: int,
        min_confirmed_loss_tokens: int,
        min_severity: float,
        min_volume_loss_tokens_per_s: float,
        donor_max_severity: float = 0.0,
    ) -> None:
        if min_revisit_opportunity_tokens < 0 or min_confirmed_loss_tokens < 0:
            raise ValueError("D thresholds must be non-negative")
        if min_severity < 0 or min_volume_loss_tokens_per_s < 0:
            raise ValueError("D thresholds must be non-negative")
        if donor_max_severity < 0:
            raise ValueError("donor_max_severity must be non-negative")
        self.min_revisit_opportunity_tokens = min_revisit_opportunity_tokens
        self.min_confirmed_loss_tokens = min_confirmed_loss_tokens
        self.min_severity = min_severity
        self.min_volume_loss_tokens_per_s = min_volume_loss_tokens_per_s
        self.donor_max_severity = donor_max_severity

    def evaluate(self, observation: WindowObservation) -> DSignal:
        opportunity = observation.revisit_opportunity_tokens
        severity = (
            observation.confirmed_reuse_loss_tokens / opportunity
            if opportunity
            else 0.0
        )
        volume = observation.confirmed_reuse_loss_tokens / observation.duration_s
        reprefill_rate = observation.reprefill_tokens / observation.duration_s
        triggered = True
        reason = "triggered"
        if opportunity < self.min_revisit_opportunity_tokens:
            triggered = False
            reason = "insufficient_revisit_opportunity"
        elif observation.confirmed_reuse_loss_tokens < self.min_confirmed_loss_tokens:
            triggered = False
            reason = "insufficient_confirmed_loss"
        elif (
            severity < self.min_severity
            and volume < self.min_volume_loss_tokens_per_s
        ):
            triggered = False
            reason = "severity_and_volume_below_threshold"
        elif severity < self.min_severity:
            reason = "triggered_by_volume"
        elif volume < self.min_volume_loss_tokens_per_s:
            reason = "triggered_by_severity"
        return DSignal(
            model_id=observation.model_id,
            d_raw_tokens=observation.confirmed_reuse_loss_tokens,
            revisit_opportunity_tokens=opportunity,
            severity=severity,
            volume_loss_tokens_per_s=volume,
            reprefill_tokens_per_s=reprefill_rate,
            triggered=triggered,
            reason=reason,
            donor_allowed=(
                not triggered
                and opportunity >= self.min_revisit_opportunity_tokens
                and severity <= self.donor_max_severity
            ),
        )


@dataclass(frozen=True)
class CandidateDecision:
    model_id: str
    action: str
    pages: int
    reason: str
    rank: tuple

    def __post_init__(self) -> None:
        if self.action not in {"grow", "hold", "shrink"}:
            raise ValueError("candidate action must be grow, hold, or shrink")
        if self.pages < 0:
            raise ValueError("candidate pages must be non-negative")


class TrialPolicy(Protocol):
    def candidates(self, signals: list[DSignal]) -> list[CandidateDecision]: ...


class ReallocationPolicy(Protocol):
    def candidates(
        self,
        observations: list[WindowObservation],
        signals: list[DSignal],
    ) -> list[CandidateDecision]: ...


class BoundedDTrialPolicy:
    """Initial trial policy: severity gates, volume ranks, one bounded quantum."""

    def __init__(self, *, trial_quantum_pages: int, cooldown_windows: int = 1) -> None:
        if trial_quantum_pages <= 0:
            raise ValueError("trial_quantum_pages must be positive")
        if cooldown_windows < 0:
            raise ValueError("cooldown_windows must be non-negative")
        self.trial_quantum_pages = trial_quantum_pages
        self.cooldown_windows = cooldown_windows

    def candidates(self, signals: list[DSignal]) -> list[CandidateDecision]:
        decisions = [
            CandidateDecision(
                model_id=signal.model_id,
                action="grow",
                pages=self.trial_quantum_pages,
                reason="initial_d_severity_volume_trial",
                rank=(
                    -signal.volume_loss_tokens_per_s,
                    -signal.d_raw_tokens,
                    -signal.severity,
                    signal.model_id,
                ),
            )
            for signal in signals
            if signal.triggered
        ]
        return sorted(decisions, key=lambda item: item.rank)


class InitialDReallocationPolicy:
    """Initial replaceable policy for D-driven grow pressure only."""

    def __init__(
        self,
        *,
        grow_quantum_pages: int,
        shrink_quantum_pages: int,
        cold_revisit_opportunity_tokens: int,
        cold_max_confirmed_loss_tokens: int,
    ) -> None:
        if grow_quantum_pages <= 0 or shrink_quantum_pages <= 0:
            raise ValueError("grow and shrink quantum pages must be positive")
        if cold_revisit_opportunity_tokens < 0 or cold_max_confirmed_loss_tokens < 0:
            raise ValueError("cold D thresholds must be non-negative")
        self.grow_quantum_pages = grow_quantum_pages
        self.shrink_quantum_pages = shrink_quantum_pages
        self.cold_revisit_opportunity_tokens = cold_revisit_opportunity_tokens
        self.cold_max_confirmed_loss_tokens = cold_max_confirmed_loss_tokens

    def candidates(
        self,
        observations: list[WindowObservation],
        signals: list[DSignal],
    ) -> list[CandidateDecision]:
        obs_by_id = {item.model_id: item for item in observations}
        decisions = []
        for signal in signals:
            observation = obs_by_id[signal.model_id]
            if signal.triggered:
                decisions.append(
                    CandidateDecision(
                        model_id=signal.model_id,
                        action="grow",
                        pages=self.grow_quantum_pages,
                        reason="initial_d_reallocation_grow_trial",
                        rank=(
                            0,
                            -signal.volume_loss_tokens_per_s,
                            -signal.d_raw_tokens,
                            -signal.severity,
                            signal.model_id,
                        ),
                    )
                )
                continue

        return sorted(decisions, key=lambda item: item.rank)


class UrgencyScaledDReallocationPolicy(InitialDReallocationPolicy):
    """Initial D trial policy with urgency-scaled grow demand."""

    def __init__(
        self,
        *,
        grow_quantum_pages: int,
        shrink_quantum_pages: int,
        cold_revisit_opportunity_tokens: int,
        cold_max_confirmed_loss_tokens: int,
        high_urgency_multiplier: int = 4,
        medium_urgency_multiplier: int = 2,
        high_urgency_score: float = 8.0,
        medium_urgency_score: float = 4.0,
    ) -> None:
        super().__init__(
            grow_quantum_pages=grow_quantum_pages,
            shrink_quantum_pages=shrink_quantum_pages,
            cold_revisit_opportunity_tokens=cold_revisit_opportunity_tokens,
            cold_max_confirmed_loss_tokens=cold_max_confirmed_loss_tokens,
        )
        if high_urgency_multiplier < 1 or medium_urgency_multiplier < 1:
            raise ValueError("urgency multipliers must be positive")
        if high_urgency_score < medium_urgency_score or medium_urgency_score < 1:
            raise ValueError("urgency thresholds must be ordered and at least one")
        self.high_urgency_multiplier = high_urgency_multiplier
        self.medium_urgency_multiplier = medium_urgency_multiplier
        self.high_urgency_score = high_urgency_score
        self.medium_urgency_score = medium_urgency_score

    def candidates(
        self,
        observations: list[WindowObservation],
        signals: list[DSignal],
    ) -> list[CandidateDecision]:
        grow_multipliers = {
            signal.model_id: self._multiplier(signal)
            for signal in signals
            if signal.triggered
        }
        decisions = []
        for signal in signals:
            if not signal.triggered:
                continue
            multiplier = grow_multipliers[signal.model_id]
            decisions.append(
                CandidateDecision(
                    model_id=signal.model_id,
                    action="grow",
                    pages=self.grow_quantum_pages * multiplier,
                    reason=f"urgency_scaled_grow_x{multiplier}",
                    rank=(
                        0,
                        -signal.volume_loss_tokens_per_s,
                        -signal.d_raw_tokens,
                        -signal.severity,
                        signal.model_id,
                    ),
                )
            )
        return sorted(decisions, key=lambda item: item.rank)

    def _multiplier(self, signal: DSignal) -> int:
        ratios = [
            signal.severity / 0.05 if signal.severity else 0.0,
            signal.volume_loss_tokens_per_s / 10.0
            if signal.volume_loss_tokens_per_s
            else 0.0,
            signal.d_raw_tokens / 100.0 if signal.d_raw_tokens else 0.0,
        ]
        urgency = max(ratios)
        if urgency >= self.high_urgency_score:
            return self.high_urgency_multiplier
        if urgency >= self.medium_urgency_score:
            return self.medium_urgency_multiplier
        return 1


class ImpactSizedDReallocationPolicy(UrgencyScaledDReallocationPolicy):
    """Urgency policy whose transfer unit is large enough to affect service."""

    def __init__(
        self,
        *,
        grow_quantum_pages: int,
        shrink_quantum_pages: int,
        cold_revisit_opportunity_tokens: int,
        cold_max_confirmed_loss_tokens: int,
        min_grow_pages: int = 0,
        min_shrink_pages: int = 0,
        grow_effective_ratio: float = 0.0,
        shrink_effective_ratio: float = 0.0,
        high_urgency_multiplier: int = 4,
        medium_urgency_multiplier: int = 2,
        high_urgency_score: float = 8.0,
        medium_urgency_score: float = 4.0,
    ) -> None:
        super().__init__(
            grow_quantum_pages=grow_quantum_pages,
            shrink_quantum_pages=shrink_quantum_pages,
            cold_revisit_opportunity_tokens=cold_revisit_opportunity_tokens,
            cold_max_confirmed_loss_tokens=cold_max_confirmed_loss_tokens,
            high_urgency_multiplier=high_urgency_multiplier,
            medium_urgency_multiplier=medium_urgency_multiplier,
            high_urgency_score=high_urgency_score,
            medium_urgency_score=medium_urgency_score,
        )
        if min_grow_pages < 0 or min_shrink_pages < 0:
            raise ValueError("minimum transfer pages must be non-negative")
        if grow_effective_ratio < 0 or shrink_effective_ratio < 0:
            raise ValueError("effective quota ratios must be non-negative")
        self.min_grow_pages = min_grow_pages
        self.min_shrink_pages = min_shrink_pages
        self.grow_effective_ratio = grow_effective_ratio
        self.shrink_effective_ratio = shrink_effective_ratio

    def candidates(
        self,
        observations: list[WindowObservation],
        signals: list[DSignal],
    ) -> list[CandidateDecision]:
        obs_by_id = {item.model_id: item for item in observations}
        active_grow_multipliers = {
            signal.model_id: self._multiplier(signal)
            for signal in signals
            if signal.triggered
        }
        decisions = []
        for signal in signals:
            if not signal.triggered:
                continue
            observation = obs_by_id[signal.model_id]
            multiplier = active_grow_multipliers[signal.model_id]
            unit = self._impact_unit(
                base_pages=self.grow_quantum_pages,
                min_pages=self.min_grow_pages,
                effective_pages=observation.effective_pages,
                ratio=self.grow_effective_ratio,
            )
            decisions.append(
                CandidateDecision(
                    model_id=signal.model_id,
                    action="grow",
                    pages=unit * multiplier,
                    reason=f"impact_sized_grow_x{multiplier}_unit{unit}",
                    rank=(
                        0,
                        -signal.volume_loss_tokens_per_s,
                        -signal.d_raw_tokens,
                        -signal.severity,
                        signal.model_id,
                    ),
                )
            )
        return sorted(decisions, key=lambda item: item.rank)

    @staticmethod
    def _impact_unit(
        *,
        base_pages: int,
        min_pages: int,
        effective_pages: int,
        ratio: float,
    ) -> int:
        ratio_pages = int(effective_pages * ratio)
        return max(base_pages, min_pages, ratio_pages)


@dataclass(frozen=True)
class PlannedTransaction:
    source_model_id: str | None
    destination_model_id: str | None
    pages: int
    readiness: str

    def __post_init__(self) -> None:
        if self.pages <= 0:
            raise ValueError("planned transaction pages must be positive")
        if self.readiness not in {"central_free", "clean", "ready"}:
            raise ValueError("invalid transaction readiness")
        if self.source_model_id is None and self.destination_model_id is None:
            raise ValueError("planned transaction must have a source or destination")


@dataclass(frozen=True)
class ResourcePlan:
    targets: dict[str, int]
    transactions: list[PlannedTransaction]
    unmet_pages: dict[str, int]

    def as_dict(self) -> dict:
        return {
            "targets": dict(self.targets),
            "transactions": [asdict(item) for item in self.transactions],
            "unmet_pages": dict(self.unmet_pages),
        }


def manual_intents_from_resource_plan(
    plan: ResourcePlan,
    *,
    intent_id_prefix: str,
    async_overlap: bool = False,
) -> list[dict]:
    """Convert a shadow ResourcePlan into explicit policy-free manager intents."""
    if not intent_id_prefix:
        raise ValueError("intent_id_prefix must be non-empty")
    intents = []
    for index, transaction in enumerate(plan.transactions, start=1):
        intent_id = f"{intent_id_prefix}-{index}"
        base = {
            "intent_id": intent_id,
            "pages": transaction.pages,
            "tranche_pages": transaction.pages,
        }
        if async_overlap:
            base["async_overlap"] = True
            base["initial_tranche_pages"] = transaction.pages
        if transaction.source_model_id is None:
            if transaction.destination_model_id is None:
                raise ValueError("reserve grow transaction requires a destination")
            intents.append(
                {
                    **base,
                    "op": "grow",
                    "recipient": transaction.destination_model_id,
                }
            )
        elif transaction.destination_model_id is None:
            offer_id = (
                f"{intent_id_prefix}:offer:{transaction.source_model_id}:"
                f"reserve:{transaction.pages}:{transaction.readiness}"
            )
            intents.append(
                {
                    **base,
                    "op": "shrink",
                    "donor": transaction.source_model_id,
                    "offer_id": offer_id,
                    "require_donor_offer_pages": transaction.pages,
                }
            )
        else:
            offer_id = (
                f"{intent_id_prefix}:offer:{transaction.source_model_id}:"
                f"{transaction.destination_model_id}:{transaction.pages}:"
                f"{transaction.readiness}"
            )
            intents.append(
                {
                    **base,
                    "op": "transfer",
                    "donor": transaction.source_model_id,
                    "recipient": transaction.destination_model_id,
                    "offer_id": offer_id,
                    "require_donor_offer_pages": transaction.pages,
                }
            )
    return intents


def select_manual_intents_for_trial(
    intents: list[dict],
    *,
    op: str | None = None,
    donor: str | None = None,
    recipient: str | None = None,
    limit: int = 1,
) -> list[dict]:
    """Select a bounded runtime trial from already-generated shadow intents."""
    if limit <= 0:
        raise ValueError("trial intent limit must be positive")
    selected = []
    for intent in intents:
        if op is not None and intent.get("op") != op:
            continue
        if donor is not None and intent.get("donor") != donor:
            continue
        if recipient is not None and intent.get("recipient") != recipient:
            continue
        selected.append(dict(intent))
        if len(selected) >= limit:
            break
    return selected


class ResourcePlanner:
    """Turn grow candidates into reserve/donor transaction plans."""

    def __init__(
        self,
        *,
        offer_safety_margin_pages: int = 0,
        donor_contribution_cap_pages: int = 0,
    ) -> None:
        if offer_safety_margin_pages < 0:
            raise ValueError("offer_safety_margin_pages must be non-negative")
        if donor_contribution_cap_pages < 0:
            raise ValueError("donor_contribution_cap_pages must be non-negative")
        self.offer_safety_margin_pages = offer_safety_margin_pages
        self.donor_contribution_cap_pages = donor_contribution_cap_pages

    def plan(
        self,
        candidates: list[CandidateDecision],
        observations: list[WindowObservation],
        signals: list[DSignal],
        *,
        global_free_pages: int,
    ) -> ResourcePlan:
        if global_free_pages < 0:
            raise ValueError("global_free_pages must be non-negative")
        if len({item.model_id for item in observations}) != len(observations):
            raise ValueError("observation model ids must be unique")
        if len({item.model_id for item in signals}) != len(signals):
            raise ValueError("signal model ids must be unique")
        obs_by_id = {item.model_id: item for item in observations}
        signal_by_id = {item.model_id: item for item in signals}
        targets = {item.model_id: item.effective_pages for item in observations}
        offer_breakdown = {
            item.model_id: item.offer_breakdown()
            if signal_by_id[item.model_id].donor_allowed
            else (0, 0)
            for item in observations
        }
        remaining_free = global_free_pages
        transactions: list[PlannedTransaction] = []
        unmet: dict[str, int] = {}

        for candidate in sorted(candidates, key=lambda item: item.rank):
            if candidate.pages == 0:
                continue
            if candidate.model_id not in obs_by_id:
                raise ValueError(f"unknown candidate model: {candidate.model_id}")
            if candidate.action == "shrink":
                continue
            if candidate.action != "grow":
                continue
            needed = candidate.pages
            granted = min(needed, remaining_free)
            if granted:
                remaining_free -= granted
                targets[candidate.model_id] += granted
                transactions.append(
                    PlannedTransaction(
                        source_model_id=None,
                        destination_model_id=candidate.model_id,
                        pages=granted,
                        readiness="central_free",
                    )
                )

            donor_used_for_candidate: dict[str, int] = {}
            for readiness, index in (("clean", 0), ("ready", 1)):
                if granted >= needed:
                    break
                donors = sorted(
                    (
                        obs_by_id[model_id]
                        for model_id, offer in offer_breakdown.items()
                        if model_id != candidate.model_id and offer[index] > 0
                    ),
                    key=lambda item: (
                        signal_by_id[item.model_id].severity,
                        -offer_breakdown[item.model_id][index],
                        item.model_id,
                    ),
                )
                for donor in donors:
                    if granted >= needed:
                        break
                    clean_offer, ready_offer = offer_breakdown[donor.model_id]
                    available = self._usable_offer_pages(
                        clean_offer if readiness == "clean" else ready_offer
                    )
                    if self.donor_contribution_cap_pages:
                        already_used = donor_used_for_candidate.get(donor.model_id, 0)
                        cap_remaining = max(
                            0,
                            self.donor_contribution_cap_pages - already_used,
                        )
                        available = min(available, cap_remaining)
                    moved = min(needed - granted, available)
                    if moved <= 0:
                        continue
                    if readiness == "clean":
                        offer_breakdown[donor.model_id] = (
                            clean_offer - moved,
                            ready_offer,
                        )
                    else:
                        offer_breakdown[donor.model_id] = (
                            clean_offer,
                            ready_offer - moved,
                        )
                    targets[donor.model_id] -= moved
                    targets[candidate.model_id] += moved
                    granted += moved
                    donor_used_for_candidate[donor.model_id] = (
                        donor_used_for_candidate.get(donor.model_id, 0) + moved
                    )
                    transactions.append(
                        PlannedTransaction(
                            source_model_id=donor.model_id,
                            destination_model_id=candidate.model_id,
                            pages=moved,
                            readiness=readiness,
                        )
                    )
            if granted < needed:
                unmet[candidate.model_id] = needed - granted

        return ResourcePlan(targets=targets, transactions=transactions, unmet_pages=unmet)

    def _usable_offer_pages(self, offer_pages: int) -> int:
        return max(0, offer_pages - self.offer_safety_margin_pages)


class OutcomeLedger:
    """Profile-ready records for shadow and applied quota trials."""

    def __init__(self, *, clock_ns=time.monotonic_ns) -> None:
        self.clock_ns = clock_ns
        self.records: dict[str, dict] = {}

    def start_trial(
        self,
        *,
        trial_id: str,
        mode: str,
        observations: dict[str, WindowObservation],
        signals: dict[str, DSignal],
        transactions: list[PlannedTransaction],
        unmet_pages: dict[str, int] | None = None,
    ) -> dict:
        if mode not in {"shadow", "apply"}:
            raise ValueError("mode must be shadow or apply")
        if not trial_id:
            raise ValueError("trial_id must be non-empty")
        record = {
            "trial_id": trial_id,
            "mode": mode,
            "created_ns": self.clock_ns(),
            "pre_observations": {
                model_id: asdict(observation)
                for model_id, observation in observations.items()
            },
            "signals": {
                model_id: asdict(signal) for model_id, signal in signals.items()
            },
            "transactions": [asdict(item) for item in transactions],
            "quota_outcome": self._quota_outcome(
                transactions,
                unmet_pages={} if unmet_pages is None else unmet_pages,
            ),
            "intent_id": None,
            "intent_submitted_ns": None,
            "allocator_ack_ns": None,
            "effective_after_ack": None,
            "t_usable_ns": None,
            "post_window_recorded_ns": None,
            "post_window_observations": None,
        }
        self.records[trial_id] = record
        return self._snapshot(record)

    def record_intent_submitted(
        self, trial_id: str, *, intent_id: str, timestamp_ns: int | None = None
    ) -> dict:
        record = self._record(trial_id)
        record["intent_id"] = intent_id
        record["intent_submitted_ns"] = self.clock_ns() if timestamp_ns is None else timestamp_ns
        return self._snapshot(record)

    def record_allocator_ack(
        self,
        trial_id: str,
        *,
        effective_pages: dict[str, int],
        timestamp_ns: int | None = None,
    ) -> dict:
        record = self._record(trial_id)
        record["allocator_ack_ns"] = self.clock_ns() if timestamp_ns is None else timestamp_ns
        record["effective_after_ack"] = dict(effective_pages)
        return self._snapshot(record)

    def record_post_window(
        self,
        trial_id: str,
        *,
        observations: dict[str, WindowObservation],
        timestamp_ns: int | None = None,
        t_usable_ns: int | None = None,
    ) -> dict:
        record = self._record(trial_id)
        record["post_window_recorded_ns"] = self.clock_ns() if timestamp_ns is None else timestamp_ns
        record["post_window_observations"] = {
            model_id: asdict(observation)
            for model_id, observation in observations.items()
        }
        record["t_usable_ns"] = t_usable_ns
        return self._snapshot(record)

    def _record(self, trial_id: str) -> dict:
        try:
            return self.records[trial_id]
        except KeyError as error:
            raise ValueError(f"unknown trial_id: {trial_id}") from error

    @staticmethod
    def _snapshot(record: dict) -> dict:
        return copy.deepcopy(record)

    @staticmethod
    def _quota_outcome(
        transactions: list[PlannedTransaction],
        *,
        unmet_pages: dict[str, int],
    ) -> dict:
        supplied: dict[str, int] = {}
        donor_contributions: dict[str, dict[str, int]] = {}
        for transaction in transactions:
            if transaction.destination_model_id is None:
                continue
            recipient = transaction.destination_model_id
            supplied[recipient] = supplied.get(recipient, 0) + transaction.pages
            donor = transaction.source_model_id or "central_free"
            donor_entry = donor_contributions.setdefault(donor, {})
            donor_entry[recipient] = donor_entry.get(recipient, 0) + transaction.pages
        requested = dict(supplied)
        for recipient, pages in unmet_pages.items():
            requested[recipient] = requested.get(recipient, 0) + pages
        return {
            "requested_pages": requested,
            "supplied_pages": supplied,
            "unmet_pages": dict(unmet_pages),
            "donor_contributions": donor_contributions,
        }


class DMarginalTrialScheduler:
    """Compatibility wrapper around the initial replaceable policies."""

    def __init__(
        self,
        *,
        trial_quantum_pages: int,
        min_revisit_opportunity_tokens: int,
        min_confirmed_loss_tokens: int,
        min_loss_rate: float,
        max_trial_pages: int | None = None,
        donor_max_loss_rate: float = 0.0,
    ) -> None:
        self.signal_policy = InitialDSignalPolicy(
            min_revisit_opportunity_tokens=min_revisit_opportunity_tokens,
            min_confirmed_loss_tokens=min_confirmed_loss_tokens,
            min_severity=min_loss_rate,
            min_volume_loss_tokens_per_s=0.0,
            donor_max_severity=donor_max_loss_rate,
        )
        self.trial_policy = BoundedDTrialPolicy(
            trial_quantum_pages=trial_quantum_pages
            if max_trial_pages is None
            else min(trial_quantum_pages, max_trial_pages)
        )
        self.resource_planner = ResourcePlanner()

    def decide(self, observations: list[WindowObservation], *, global_free_pages: int) -> ResourcePlan:
        signals = [self.signal_policy.evaluate(item) for item in observations]
        candidates = self.trial_policy.candidates(signals)
        return self.resource_planner.plan(
            candidates,
            observations,
            signals,
            global_free_pages=global_free_pages,
        )
