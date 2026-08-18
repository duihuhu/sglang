"""Policy-free global quota ledger for LatticeKV Host pages.

The manager is a control-plane bookkeeper.  It accepts explicit human/scripted
quota intents and records the Central-I/O transaction stages that make pages
usable.  It deliberately does not inspect workload metrics or decide whether
an instance deserves more or less quota.
"""

from __future__ import annotations

import time
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


_POLICY_FREE_MESSAGE = (
    "GlobalQuotaManager is policy-free; pass only explicit quota intents, not "
    "D, TTFT, RPS, cache ratio, profile, scheduler, or threshold context"
)


@dataclass(frozen=True)
class QuotaIntent:
    intent_id: int
    action: str
    pages: int
    donor: str | None
    recipient: str | None
    offer_id: str | None
    created_ns: int


@dataclass(frozen=True)
class QuotaAuditEvent:
    intent_id: int
    kind: str
    time_ns: int
    model_id: str | None = None
    pages: int = 0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class _InstanceLedger:
    model_id: str
    floor_pages: int
    effective_pages: int
    draining_pages: int = 0
    scrub_pending_pages: int = 0
    transferring_pages: int = 0
    pending_ack_pages: int = 0

    def total_pages(self) -> int:
        return (
            self.effective_pages
            + self.draining_pages
            + self.scrub_pending_pages
            + self.transferring_pages
            + self.pending_ack_pages
        )


@dataclass
class _Transaction:
    intent: QuotaIntent
    stage: str
    reserve_pages: int = 0
    donor_detached: bool = False
    scrub_complete: bool = False
    allocator_ack: bool = False

    @property
    def involved_models(self) -> set[str]:
        models = set()
        if self.intent.donor is not None:
            models.add(self.intent.donor)
        if self.intent.recipient is not None:
            models.add(self.intent.recipient)
        return models


@dataclass
class _UsableAllocation:
    intent_id: int
    model_id: str
    pages: int
    intent_ns: int
    ack_ns: int
    observed: bool = False


class GlobalQuotaManager:
    """Conservative page-level quota manager with append-only audit records."""

    def __init__(
        self,
        *,
        global_pool_pages: int,
        clock_ns: Callable[[], int] | None = None,
        audit_path: str | Path | None = None,
    ) -> None:
        if global_pool_pages <= 0:
            raise ValueError("global_pool_pages must be positive")
        self.global_pool_pages = int(global_pool_pages)
        self.reserve_pages = int(global_pool_pages)
        self._clock_ns = clock_ns or time.monotonic_ns
        self.audit_path = Path(audit_path) if audit_path is not None else None
        self._instances: dict[str, _InstanceLedger] = {}
        self._transactions: dict[int, _Transaction] = {}
        self._audit: list[QuotaAuditEvent] = []
        self._next_intent_id = 1
        self._usable_allocations: list[_UsableAllocation] = []

    def register_instance(
        self,
        model_id: str,
        *,
        floor_pages: int,
        effective_pages: int,
    ) -> None:
        if not model_id:
            raise ValueError("model_id must be non-empty")
        if model_id in self._instances:
            raise ValueError(f"instance already registered: {model_id}")
        if floor_pages < 0 or effective_pages < 0:
            raise ValueError("floor and effective pages must be non-negative")
        if floor_pages > effective_pages:
            raise ValueError("floor cannot exceed effective quota")
        if effective_pages > self.reserve_pages:
            raise ValueError("registered instances exceed the global pool")
        self._instances[model_id] = _InstanceLedger(
            model_id=model_id,
            floor_pages=int(floor_pages),
            effective_pages=int(effective_pages),
        )
        self.reserve_pages -= int(effective_pages)
        self.assert_conserved()

    def submit_intent(
        self,
        action: str,
        *,
        pages: int,
        donor: str | None = None,
        recipient: str | None = None,
        offer_id: str | None = None,
        policy_context: dict[str, Any] | None = None,
    ) -> QuotaIntent:
        if policy_context:
            raise ValueError(_POLICY_FREE_MESSAGE)
        if pages <= 0:
            raise ValueError("intent pages must be positive")
        if action not in {"grow", "shrink", "transfer"}:
            raise ValueError("action must be grow, shrink, or transfer")
        self._validate_intent_shape(action, donor=donor, recipient=recipient)
        self._require_registered(donor)
        self._require_registered(recipient)
        self._require_no_conflict(donor, recipient)
        if action == "grow":
            if self.reserve_pages < pages:
                raise ValueError("reserve pages are insufficient for grow; use an explicit transfer intent")
        elif action == "transfer":
            if not offer_id:
                raise ValueError("transfer requires an explicit donor offer id")
            assert donor is not None
            self._check_donor_floor(donor, pages)
        else:
            assert donor is not None
            self._check_donor_floor(donor, pages)

        now = self._clock_ns()
        intent = QuotaIntent(
            intent_id=self._next_intent_id,
            action=action,
            pages=int(pages),
            donor=donor,
            recipient=recipient,
            offer_id=offer_id,
            created_ns=now,
        )
        self._next_intent_id += 1
        transaction = _Transaction(intent=intent, stage="received")
        self._transactions[intent.intent_id] = transaction
        self._record(intent, "intent_received", pages=pages, details={"action": action})

        if action == "grow":
            assert recipient is not None
            self.reserve_pages -= pages
            self._instances[recipient].pending_ack_pages += pages
            transaction.reserve_pages = pages
            transaction.stage = "pending_allocator_ack"
            self._record(intent, "reserve_allocated", model_id=recipient, pages=pages)
        elif action == "transfer":
            assert donor is not None and recipient is not None
            transaction.stage = "awaiting_donor_detach"
            self._record(
                intent,
                "offer_accepted",
                model_id=donor,
                pages=pages,
                details={"offer_id": offer_id},
            )
        else:
            assert donor is not None
            transaction.stage = "awaiting_donor_detach"

        self.assert_conserved()
        return intent

    def mark_donor_detached(self, intent_id: int) -> None:
        transaction = self._transaction(intent_id)
        intent = transaction.intent
        if intent.action not in {"shrink", "transfer"}:
            raise ValueError("only shrink or transfer intents detach donor pages")
        if transaction.donor_detached:
            raise RuntimeError("donor pages already detached")
        assert intent.donor is not None
        donor = self._instances[intent.donor]
        self._check_donor_floor(intent.donor, intent.pages)
        donor.effective_pages -= intent.pages
        donor.draining_pages += intent.pages
        transaction.donor_detached = True
        transaction.stage = "draining"
        self._record(intent, "donor_detached", model_id=intent.donor, pages=intent.pages)
        donor.draining_pages -= intent.pages
        donor.scrub_pending_pages += intent.pages
        self.assert_conserved()

    def mark_scrub_complete(self, intent_id: int) -> None:
        transaction = self._transaction(intent_id)
        intent = transaction.intent
        if intent.action not in {"shrink", "transfer"}:
            raise ValueError("only shrink or transfer intents scrub donor pages")
        if not transaction.donor_detached:
            raise RuntimeError("donor pages must detach before scrub")
        if transaction.scrub_complete:
            raise RuntimeError("scrub already completed")
        assert intent.donor is not None
        donor = self._instances[intent.donor]
        if donor.scrub_pending_pages < intent.pages:
            raise RuntimeError("donor scrub-pending ledger is inconsistent")
        donor.scrub_pending_pages -= intent.pages
        transaction.scrub_complete = True
        self._record(intent, "scrub_completed", model_id=intent.donor, pages=intent.pages)
        if intent.action == "shrink":
            self.reserve_pages += intent.pages
            transaction.stage = "completed"
            self._record(intent, "reserve_recredited", pages=intent.pages)
        else:
            assert intent.recipient is not None
            recipient = self._instances[intent.recipient]
            recipient.transferring_pages += intent.pages
            recipient.transferring_pages -= intent.pages
            recipient.pending_ack_pages += intent.pages
            transaction.stage = "pending_allocator_ack"
            self._record(
                intent,
                "recipient_allocator_install_ready",
                model_id=intent.recipient,
                pages=intent.pages,
            )
        self.assert_conserved()

    def ack_recipient_allocator(self, intent_id: int) -> dict[str, int]:
        transaction = self._transaction(intent_id)
        intent = transaction.intent
        if intent.action not in {"grow", "transfer"}:
            raise ValueError("only grow or transfer intents have recipient allocator ack")
        if transaction.allocator_ack:
            raise RuntimeError("allocator already acknowledged this intent")
        if transaction.stage != "pending_allocator_ack":
            raise RuntimeError("pages are not ready for allocator ack")
        assert intent.recipient is not None
        recipient = self._instances[intent.recipient]
        if recipient.pending_ack_pages < intent.pages:
            raise RuntimeError("recipient pending-ack ledger is inconsistent")
        recipient.pending_ack_pages -= intent.pages
        recipient.effective_pages += intent.pages
        transaction.allocator_ack = True
        transaction.stage = "completed"
        ack_ns = self._clock_ns()
        self._record(intent, "allocator_ack", model_id=intent.recipient, pages=intent.pages)
        self._record(
            intent,
            "effective_quota_changed",
            model_id=intent.recipient,
            pages=recipient.effective_pages,
        )
        self._usable_allocations.append(
            _UsableAllocation(
                intent_id=intent.intent_id,
                model_id=intent.recipient,
                pages=intent.pages,
                intent_ns=intent.created_ns,
                ack_ns=ack_ns,
            )
        )
        self.assert_conserved()
        return {"effective_pages": recipient.effective_pages}

    def record_actual_backup(self, model_id: str, *, pages: int) -> dict[str, int] | None:
        self._require_registered(model_id)
        if pages <= 0:
            raise ValueError("backup pages must be positive")
        candidate = next(
            (
                item
                for item in self._usable_allocations
                if item.model_id == model_id and not item.observed
            ),
            None,
        )
        if candidate is None:
            return None
        candidate.observed = True
        event = self._record(
            self._transactions[candidate.intent_id].intent,
            "first_actual_backup_on_new_page",
            model_id=model_id,
            pages=pages,
            details={
                "ack_ns": candidate.ack_ns,
                "t_usable_ns": self._clock_ns() - candidate.intent_ns,
            },
        )
        return {
            "intent_id": candidate.intent_id,
            "t_usable_ns": int(event.details["t_usable_ns"]),
        }

    def mark_timeout(self, intent_id: int, *, reason: str) -> None:
        if not reason:
            raise ValueError("timeout reason must be non-empty")
        transaction = self._transaction(intent_id)
        self._record(
            transaction.intent,
            "intent_timeout",
            pages=transaction.intent.pages,
            details={"reason": reason, "stage": transaction.stage},
        )
        self.assert_conserved()

    def effective_pages(self, model_id: str) -> int:
        self._require_registered(model_id)
        return self._instances[model_id].effective_pages

    def pending_ack_pages(self, model_id: str) -> int:
        self._require_registered(model_id)
        return self._instances[model_id].pending_ack_pages

    def scrub_pending_pages(self, model_id: str) -> int:
        self._require_registered(model_id)
        return self._instances[model_id].scrub_pending_pages

    def audit_events(self, intent_id: int | None = None) -> list[QuotaAuditEvent]:
        if intent_id is None:
            return list(self._audit)
        return [event for event in self._audit if event.intent_id == intent_id]

    def audit_kinds(self, intent_id: int) -> list[str]:
        return [event.kind for event in self.audit_events(intent_id)]

    def assert_conserved(self) -> None:
        total = self.reserve_pages + sum(
            instance.total_pages() for instance in self._instances.values()
        )
        if total != self.global_pool_pages:
            raise AssertionError(
                f"global pages are not conserved: {total} != {self.global_pool_pages}"
            )

    def _validate_intent_shape(
        self,
        action: str,
        *,
        donor: str | None,
        recipient: str | None,
    ) -> None:
        if action == "grow" and (recipient is None or donor is not None):
            raise ValueError("grow requires exactly one recipient")
        if action == "shrink" and (donor is None or recipient is not None):
            raise ValueError("shrink requires exactly one donor")
        if action == "transfer" and (donor is None or recipient is None or donor == recipient):
            raise ValueError("transfer requires distinct donor and recipient")

    def _require_registered(self, model_id: str | None) -> None:
        if model_id is not None and model_id not in self._instances:
            raise ValueError(f"unknown instance: {model_id}")

    def _require_no_conflict(self, donor: str | None, recipient: str | None) -> None:
        requested = {item for item in (donor, recipient) if item is not None}
        for transaction in self._transactions.values():
            if transaction.stage == "completed":
                continue
            if requested & transaction.involved_models:
                raise RuntimeError("conflicting in-flight quota intent")

    def _check_donor_floor(self, donor: str, pages: int) -> None:
        state = self._instances[donor]
        if state.effective_pages - pages < state.floor_pages:
            raise ValueError("donor shrink would violate floor")

    def _transaction(self, intent_id: int) -> _Transaction:
        try:
            return self._transactions[intent_id]
        except KeyError as exc:
            raise ValueError(f"unknown intent id: {intent_id}") from exc

    def _record(
        self,
        intent: QuotaIntent,
        kind: str,
        *,
        model_id: str | None = None,
        pages: int = 0,
        details: dict[str, Any] | None = None,
    ) -> QuotaAuditEvent:
        event = QuotaAuditEvent(
            intent_id=intent.intent_id,
            kind=kind,
            time_ns=self._clock_ns(),
            model_id=model_id,
            pages=int(pages),
            details=dict(details or {}),
        )
        self._audit.append(event)
        if self.audit_path is not None:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "intent_id": event.intent_id,
                            "kind": event.kind,
                            "time_ns": event.time_ns,
                            "model_id": event.model_id,
                            "pages": event.pages,
                            "details": event.details,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        return event
