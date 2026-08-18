"""Bridge agent health reports to pressure-triggered quota intents."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from sglang.srt.mem_cache.pressure_scheduler import (
    InstancePressure,
    PressureScheduler,
)


class PressureSchedulerRuntime:
    """Run one conservative global decision without claiming effective quota.

    The central agent remains the authority on physical ownership.  This
    adapter publishes only target capacities; donor drain, scrub, recipient
    prepare and allocator acknowledgement advance later through their normal
    runtime paths.
    """

    def __init__(
        self,
        control: Any,
        *,
        max_transfer_bytes: int,
        growth_quantum_bytes: int | None = None,
    ) -> None:
        self.control = control
        self.scheduler = PressureScheduler(
            max_transfer_bytes=max_transfer_bytes,
            growth_quantum_bytes=growth_quantum_bytes,
        )
        # A retention debt is evidence from one concrete reclaim mistake.  It
        # may remain visible during its feedback horizon, but must not buy a
        # new growth quantum every scheduler tick after the first intent.
        self._handled_retention_debt_epochs: dict[str, int] = {}

    @staticmethod
    def _instance_pressure(model_id: str, state: dict[str, Any]) -> InstancePressure:
        report = state["residency"]
        page_size = int(state["page_size"])
        token_bytes = int(state["token_bytes"])
        if page_size <= 0 or token_bytes <= 0:
            raise ValueError("agent status has an invalid page geometry")
        page_bytes = page_size * token_bytes
        effective_capacity = int(state["capacity"])
        if effective_capacity % page_size:
            raise ValueError("agent effective capacity is not page aligned")
        if int(report["effective_pages"]) * page_size != effective_capacity:
            raise ValueError("local report does not match agent effective capacity")

        def bytes_for(name: str) -> int:
            return int(report[name]) * page_bytes

        return InstancePressure(
            model_id=model_id,
            effective_bytes=effective_capacity * token_bytes,
            floor_bytes=bytes_for("floor_pages"),
            clean_bytes=bytes_for("clean_pages"),
            ready_bytes=bytes_for("ready_pages"),
            min_bytes=bytes_for("min_pages"),
            low_bytes=bytes_for("low_pages"),
            high_bytes=bytes_for("high_pages"),
            retention_debt_bytes=int(report["retention_debt_tokens"]) * token_bytes,
            unresolved_clean_bytes=bytes_for("unresolved_clean_pages"),
            ready_reclaim_score=float(report.get("ready_reclaim_score", 0.0)),
        )

    @staticmethod
    def _slots_for_target(state: dict[str, Any], target_bytes: int) -> int:
        token_bytes = int(state["token_bytes"])
        page_size = int(state["page_size"])
        slots = target_bytes // token_bytes
        return slots - slots % page_size

    def tick(self) -> dict[str, Any]:
        snapshot = self.control.global_status()
        models = snapshot["models"]
        missing_reports = sorted(
            model_id
            for model_id, state in models.items()
            if state.get("residency") is None
        )
        def has_unfinished_intent(state: dict[str, Any]) -> bool:
            target = state.get("quota_target")
            if target is None or int(target) == int(state["capacity"]):
                return False

            # A recipient has not gained capacity until its allocator acks;
            # scheduling it again would repeatedly drain the donor.  A donor
            # is different: fresh retention debt revokes an unfinished shrink
            # so that its own cache health wins over the old intent.
            shrinking = int(target) < int(state["capacity"])
            residency = state.get("residency") or {}
            revoke_shrink = shrinking and int(
                residency.get("retention_debt_tokens", 0)
            ) > 0
            return not revoke_shrink

        pending_models = sorted(
            model_id
            for model_id, state in models.items()
            if (
                int(state.get("pending_capacity", 0)) > 0
                or int(state.get("pending_release_capacity", 0)) > 0
                or has_unfinished_intent(state)
            )
        )
        eligible = {
            model_id: state
            for model_id, state in models.items()
            if model_id not in missing_reports and model_id not in pending_models
        }
        if not eligible:
            return {
                "published_targets": {},
                "effective_targets": {},
                "missing_reports": missing_reports,
                "pending_models": pending_models,
                "local_health": {},
                "quota_transitions": {},
                "published_target_set_ns": None,
                "decision": None,
            }

        pressures = []
        debt_epochs: dict[str, int] = {}
        for model_id, state in eligible.items():
            pressure = self._instance_pressure(model_id, state)
            epoch = int((state.get("residency") or {}).get("retention_debt_epoch", 0))
            debt_epochs[model_id] = epoch
            if epoch <= self._handled_retention_debt_epochs.get(model_id, 0):
                pressure = replace(pressure, retention_debt_bytes=0)
            pressures.append(pressure)

        decision = self.scheduler.decide(
            pressures,
            global_free_bytes=int(snapshot["global_free_bytes"]),
        )
        published_targets: dict[str, int] = {}
        effective_targets = {
            model_id: int(state["capacity"]) for model_id, state in eligible.items()
        }
        for model_id, target_bytes in decision.target_bytes.items():
            target_slots = self._slots_for_target(eligible[model_id], target_bytes)
            current_intent = eligible[model_id].get("quota_target")
            if current_intent is None:
                current_intent = effective_targets[model_id]
            if target_slots != int(current_intent):
                published_targets[model_id] = target_slots
        published_target_set_ns = None
        if published_targets:
            published = self.control.set_quota_targets(published_targets)
            raw_timestamp = published.get("target_set_ns")
            if raw_timestamp is not None:
                published_target_set_ns = int(raw_timestamp)
            for model_id, target_slots in published_targets.items():
                if target_slots <= effective_targets[model_id]:
                    continue
                if self._instance_pressure(model_id, eligible[model_id]).retention_debt_bytes:
                    self._handled_retention_debt_epochs[model_id] = debt_epochs[model_id]
        return {
            "published_targets": published_targets,
            "effective_targets": effective_targets,
            "missing_reports": missing_reports,
            "pending_models": pending_models,
            "local_health": {
                model_id: state["residency"] for model_id, state in eligible.items()
            },
            "quota_transitions": {
                model_id: {
                    "target_slots": state.get("quota_target"),
                    "target_set_ns": state.get("quota_target_set_ns"),
                    "effective_slots": int(state["capacity"]),
                    "effective_change_ns": state.get("last_effective_capacity_change_ns"),
                    "effective_change_reason": state.get("last_effective_capacity_change_reason"),
                }
                for model_id, state in eligible.items()
            },
            "published_target_set_ns": published_target_set_ns,
            "decision": decision,
        }
