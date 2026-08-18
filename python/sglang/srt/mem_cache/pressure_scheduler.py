"""Pressure-triggered, byte-based quota decisions for LatticeKV."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InstancePressure:
    """One instance's page-aligned host-cache health, expressed in bytes."""

    model_id: str
    effective_bytes: int
    floor_bytes: int
    clean_bytes: int
    ready_bytes: int
    min_bytes: int
    low_bytes: int
    high_bytes: int
    retention_debt_bytes: int = 0
    unresolved_clean_bytes: int = 0
    ready_reclaim_score: float = 0.0

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        values = (
            self.effective_bytes,
            self.floor_bytes,
            self.clean_bytes,
            self.ready_bytes,
            self.min_bytes,
            self.low_bytes,
            self.high_bytes,
            self.retention_debt_bytes,
            self.unresolved_clean_bytes,
        )
        if min(values) < 0:
            raise ValueError("pressure values must be non-negative")
        if self.ready_reclaim_score < 0:
            raise ValueError("ready reclaim score must be non-negative")
        if self.floor_bytes > self.effective_bytes:
            raise ValueError("floor cannot exceed effective quota")
        if self.clean_bytes + self.ready_bytes > self.effective_bytes:
            raise ValueError("clean plus ready cannot exceed effective quota")
        if not self.min_bytes <= self.low_bytes <= self.high_bytes:
            raise ValueError("watermarks must satisfy min <= low <= high")

    @property
    def available_bytes(self) -> int:
        return self.clean_bytes + self.ready_bytes

    @property
    def emergency(self) -> bool:
        return self.clean_bytes < self.min_bytes

    @property
    def local_gap_bytes(self) -> int:
        # Ready pages still contain KV and cannot accept HBM backup.
        return max(0, self.high_bytes - self.clean_bytes)

    @property
    def growth_need_bytes(self) -> int:
        if self.unresolved_clean_bytes > 0:
            return max(self.local_gap_bytes, self.unresolved_clean_bytes)
        # Re-prefill after a reclaim is useful feedback for retention and
        # donor safety, but it is not by itself proof that the instance needs
        # more quota right now.  If clean host space has recovered above low,
        # the local maintainer has room to absorb the next HBM backup and must
        # first use that quota to stabilize its value-aware cache.  Treating
        # every fresh ghost loss as an immediate grow causes a healthy cache
        # to ratchet through the entire central pool.
        if self.retention_debt_bytes > 0 and self.clean_bytes < self.low_bytes:
            return max(self.local_gap_bytes, self.retention_debt_bytes)
        return 0

    def offer_bytes(self) -> tuple[int, int]:
        """Return immediate-clean and ready-soon supply safe for donation."""
        if self.retention_debt_bytes or self.unresolved_clean_bytes:
            return (0, 0)
        floor_limited = max(0, self.effective_bytes - self.floor_bytes)
        immediate = min(
            floor_limited,
            max(0, self.clean_bytes - self.high_bytes),
        )
        ready = min(
            floor_limited - immediate,
            max(0, self.available_bytes - self.high_bytes - immediate),
        )
        return immediate, ready

    @property
    def ready_score_per_byte(self) -> float:
        return (
            self.ready_reclaim_score / self.ready_bytes
            if self.ready_bytes
            else 0.0
        )


@dataclass(frozen=True)
class QuotaTransfer:
    source_model_id: str | None
    destination_model_id: str
    bytes: int
    readiness: str


@dataclass(frozen=True)
class PressureDecision:
    target_bytes: dict[str, int]
    transfers: list[QuotaTransfer]
    unmet_bytes: dict[str, int]


class PressureScheduler:
    """Make conservative grow/shrink intents from proven local pressure."""

    def __init__(
        self,
        *,
        max_transfer_bytes: int,
        growth_quantum_bytes: int | None = None,
    ) -> None:
        if max_transfer_bytes <= 0:
            raise ValueError("max_transfer_bytes must be positive")
        if growth_quantum_bytes is None:
            growth_quantum_bytes = max_transfer_bytes
        if not 0 < growth_quantum_bytes <= max_transfer_bytes:
            raise ValueError(
                "growth_quantum_bytes must be positive and no larger than max_transfer_bytes"
            )
        self.max_transfer_bytes = max_transfer_bytes
        self.growth_quantum_bytes = growth_quantum_bytes

    def _needed_growth_bytes(self, instance: InstancePressure) -> int:
        """Turn local evidence into a bounded quota request.

        A clean shortfall has a concrete byte deficit. A retention debt does
        not: it proves that the current quota discarded useful KV, but its
        token count is not a byte forecast for the next quota. Debt therefore
        asks for one bounded allocation quantum, while clean-watermark facts
        can increase that request when they are larger.
        """
        if not instance.growth_need_bytes:
            return 0
        required = max(
            instance.local_gap_bytes,
            instance.unresolved_clean_bytes,
        )
        if instance.retention_debt_bytes:
            required = max(required, self.growth_quantum_bytes)
        return min(self.max_transfer_bytes, required)

    def decide(
        self,
        instances: list[InstancePressure],
        *,
        global_free_bytes: int,
    ) -> PressureDecision:
        if global_free_bytes < 0:
            raise ValueError("global_free_bytes must be non-negative")
        if len({item.model_id for item in instances}) != len(instances):
            raise ValueError("instance ids must be unique")

        targets = {item.model_id: item.effective_bytes for item in instances}
        unmet: dict[str, int] = {}
        transfers: list[QuotaTransfer] = []
        remaining_free = global_free_bytes
        donor_supply = {
            item.model_id: list(item.offer_bytes()) for item in instances
        }
        donors = {item.model_id: item for item in instances}

        recipients = sorted(
            (item for item in instances if item.growth_need_bytes),
            key=lambda item: (
                not item.emergency,
                -item.unresolved_clean_bytes,
                -item.retention_debt_bytes,
                -item.local_gap_bytes,
                item.model_id,
            ),
        )
        for recipient in recipients:
            needed = self._needed_growth_bytes(recipient)
            granted = min(needed, remaining_free)
            if granted:
                targets[recipient.model_id] += granted
                remaining_free -= granted
                transfers.append(
                    QuotaTransfer(None, recipient.model_id, granted, "central_free")
                )

            for readiness, supply_index in (("clean", 0), ("ready", 1)):
                if granted >= needed:
                    break
                ranked_donors = sorted(
                    (
                        donor
                        for donor in donors.values()
                        if donor.model_id != recipient.model_id
                        and donor_supply[donor.model_id][supply_index] > 0
                    ),
                    key=lambda donor: (
                        donor.ready_score_per_byte if readiness == "ready" else 0.0,
                        -donor_supply[donor.model_id][supply_index],
                        donor.model_id,
                    ),
                )
                for donor in ranked_donors:
                    if granted >= needed:
                        break
                    moved = min(
                        needed - granted, donor_supply[donor.model_id][supply_index]
                    )
                    donor_supply[donor.model_id][supply_index] -= moved
                    targets[donor.model_id] -= moved
                    targets[recipient.model_id] += moved
                    granted += moved
                    transfers.append(
                        QuotaTransfer(
                            donor.model_id, recipient.model_id, moved, readiness
                        )
                    )
            if granted < needed:
                unmet[recipient.model_id] = needed - granted

        return PressureDecision(
            target_bytes=targets,
            transfers=transfers,
            unmet_bytes=unmet,
        )
