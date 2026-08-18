"""Central I/O target backend for explicit manual quota intents."""

from __future__ import annotations

import time
from typing import Any


class CentralIOTargetQuotaBackend:
    """Publish explicit Central I/O quota targets and wait for acked capacity.

    This backend does not inspect local health fields except for the registered
    floor used by the policy-free manager ledger.  Quota decisions arrive from
    the caller as grow/shrink/transfer commands.
    """

    def __init__(
        self,
        control: Any,
        *,
        floor_pages: dict[str, int] | None = None,
        wait_timeout_s: float = 300.0,
        wait_interval_s: float = 0.5,
        clock=time.monotonic,
        sleep=time.sleep,
    ) -> None:
        if wait_timeout_s <= 0:
            raise ValueError("wait_timeout_s must be positive")
        if wait_interval_s < 0:
            raise ValueError("wait_interval_s must be non-negative")
        self.control = control
        self.floor_pages = dict(floor_pages or {})
        self.wait_timeout_s = wait_timeout_s
        self.wait_interval_s = wait_interval_s
        self.clock = clock
        self.sleep = sleep

    def snapshot(self) -> dict[str, Any]:
        status = self.control.global_status()
        models = status["models"]
        page_sizes = {int(state["page_size"]) for state in models.values()}
        token_bytes = {int(state["token_bytes"]) for state in models.values()}
        if len(page_sizes) != 1 or len(token_bytes) != 1:
            raise ValueError("manual quota backend requires uniform page geometry")
        page_size = page_sizes.pop()
        bytes_per_page = page_size * token_bytes.pop()
        if bytes_per_page <= 0:
            raise ValueError("invalid Central I/O page geometry")
        global_free_pages = int(status["global_free_bytes"]) // bytes_per_page
        result_models = {}
        effective_total = 0
        for model_id, state in models.items():
            capacity = int(state["capacity"])
            if capacity % page_size:
                raise ValueError("Central I/O capacity is not page aligned")
            effective_pages = capacity // page_size
            effective_total += effective_pages
            result_models[model_id] = {
                "effective_pages": effective_pages,
                "floor_pages": self._floor_for(model_id, state),
            }
        return {
            "global_pool_pages": global_free_pages + effective_total,
            "models": result_models,
        }

    def grow_from_reserve(self, model_id: str, pages: int) -> dict[str, Any]:
        plan = self.begin_grow_from_reserve(model_id, pages)
        target = int(plan["target_effective_pages"])
        return {"effective_pages": self._wait_effective_pages(model_id, target)}

    def shrink_to_reserve(self, model_id: str, pages: int) -> dict[str, Any]:
        current = self._effective_pages(model_id)
        target = current - pages
        if target < self._floor_for(model_id, self._model_state(model_id)):
            raise ValueError("manual shrink would violate donor floor")
        self._publish_page_targets({model_id: target})
        return {"effective_pages": self._wait_effective_pages(model_id, target)}

    def transfer_pages(
        self, donor: str, recipient: str, pages: int, offer_id: str
    ) -> dict[str, Any]:
        plan = self.begin_transfer_pages(donor, recipient, pages, offer_id)
        donor_target = int(plan["donor_target_effective_pages"])
        recipient_target = int(plan["recipient_target_effective_pages"])
        donor_effective = self._wait_effective_pages(donor, donor_target)
        recipient_effective = self._wait_effective_pages(recipient, recipient_target)
        return {
            "donor_effective_pages": donor_effective,
            "recipient_effective_pages": recipient_effective,
        }

    def begin_grow_from_reserve(self, model_id: str, pages: int) -> dict[str, Any]:
        current = self._effective_pages(model_id)
        target = current + pages
        self._publish_page_targets({model_id: target})
        return {
            "model_id": model_id,
            "start_effective_pages": current,
            "target_effective_pages": target,
        }

    def begin_transfer_pages(
        self, donor: str, recipient: str, pages: int, offer_id: str
    ) -> dict[str, Any]:
        if not offer_id:
            raise ValueError("manual transfer requires an explicit offer id")
        status = self.control.global_status()
        donor_state = status["models"][donor]
        recipient_state = status["models"][recipient]
        donor_current = self._effective_pages_from_state(donor_state)
        recipient_current = self._effective_pages_from_state(recipient_state)
        donor_target = donor_current - pages
        if donor_target < self._floor_for(donor, donor_state):
            raise ValueError("manual transfer would violate donor floor")
        recipient_target = recipient_current + pages
        self._publish_page_targets({donor: donor_target, recipient: recipient_target})
        return {
            "donor_start_effective_pages": donor_current,
            "recipient_start_effective_pages": recipient_current,
            "donor_target_effective_pages": donor_target,
            "recipient_target_effective_pages": recipient_target,
        }

    def donation_offer(self, model_id: str) -> dict[str, Any]:
        state = self._model_state(model_id)
        residency = dict(state.get("residency") or {})
        effective_pages = self._effective_pages_from_state(state)
        floor_limited = max(0, effective_pages - self._floor_for(model_id, state))
        immediate = min(
            floor_limited,
            max(0, int(residency.get("clean_pages", 0)) - int(residency.get("high_pages", 0))),
        )
        ready = min(
            max(0, floor_limited - immediate),
            int(residency.get("ready_pages", 0)),
        )
        total = immediate + ready
        if int(residency.get("unresolved_clean_pages", 0)):
            immediate = 0
            ready = 0
            total = 0
        return {
            "model_id": model_id,
            "immediate_pages": immediate,
            "ready_pages": ready,
            "total_pages": total,
            "residency": residency,
        }

    def wait_grow_boundary(
        self, model_id: str, target_effective_pages: int
    ) -> dict[str, Any]:
        effective_pages = self._wait_effective_pages_at_least(
            model_id, target_effective_pages
        )
        return {
            "effective_pages": effective_pages,
            "boundary_effective_pages": target_effective_pages,
        }

    def wait_transfer_boundary(
        self,
        donor: str,
        donor_effective_pages: int,
        recipient: str,
        recipient_effective_pages: int,
    ) -> dict[str, Any]:
        deadline = self.clock() + self.wait_timeout_s
        while True:
            status = self.control.global_status()
            donor_current = self._effective_pages_from_state(status["models"][donor])
            recipient_current = self._effective_pages_from_state(
                status["models"][recipient]
            )
            if (
                donor_current <= donor_effective_pages
                and recipient_current >= recipient_effective_pages
            ):
                return {
                    "donor_effective_pages": donor_current,
                    "donor_boundary_effective_pages": donor_effective_pages,
                    "recipient_effective_pages": recipient_current,
                    "recipient_boundary_effective_pages": recipient_effective_pages,
                }
            if self.clock() >= deadline:
                raise TimeoutError(
                    "timed out waiting for transfer boundary "
                    f"{donor}<={donor_effective_pages}, "
                    f"{recipient}>={recipient_effective_pages}; "
                    f"current {donor}={donor_current}, {recipient}={recipient_current}"
                )
            if self.wait_interval_s:
                self.sleep(self.wait_interval_s)

    def _publish_page_targets(self, page_targets: dict[str, int]) -> None:
        status = self.control.global_status()
        slot_targets = {}
        for model_id, target_pages in page_targets.items():
            state = status["models"][model_id]
            slot_targets[model_id] = int(target_pages) * int(state["page_size"])
        self.control.set_quota_targets(slot_targets)

    def _wait_effective_pages(self, model_id: str, target_pages: int) -> int:
        deadline = self.clock() + self.wait_timeout_s
        while True:
            current = self._effective_pages(model_id)
            if current == target_pages:
                return current
            if self.clock() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for {model_id} effective pages {target_pages}; current={current}"
                )
            if self.wait_interval_s:
                self.sleep(self.wait_interval_s)

    def _wait_effective_pages_at_least(self, model_id: str, target_pages: int) -> int:
        deadline = self.clock() + self.wait_timeout_s
        while True:
            current = self._effective_pages(model_id)
            if current >= target_pages:
                return current
            if self.clock() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for {model_id} effective pages >= {target_pages}; current={current}"
                )
            if self.wait_interval_s:
                self.sleep(self.wait_interval_s)

    def _effective_pages(self, model_id: str) -> int:
        state = self._model_state(model_id)
        return self._effective_pages_from_state(state)

    @staticmethod
    def _effective_pages_from_state(state: dict[str, Any]) -> int:
        page_size = int(state["page_size"])
        capacity = int(state["capacity"])
        if capacity % page_size:
            raise ValueError("Central I/O capacity is not page aligned")
        return capacity // page_size

    def _model_state(self, model_id: str) -> dict[str, Any]:
        status = self.control.global_status()
        try:
            return status["models"][model_id]
        except KeyError as error:
            raise ValueError(f"unknown Central I/O model: {model_id}") from error

    def _floor_for(self, model_id: str, state: dict[str, Any]) -> int:
        if model_id in self.floor_pages:
            return int(self.floor_pages[model_id])
        residency = state.get("residency") or {}
        if "floor_pages" not in residency:
            raise ValueError(f"manual quota backend needs floor_pages for {model_id}")
        return int(residency["floor_pages"])
