"""Runtime adapter for replaceable D-driven quota trial planning."""

from __future__ import annotations

from dataclasses import asdict
import inspect
from typing import Any

try:
    from sglang.srt.mem_cache.d_marginal_trial_scheduler import (
        BoundedDTrialPolicy,
        InitialDSignalPolicy,
        OutcomeLedger,
        ResourcePlanner,
        WindowObservation,
    )
except ModuleNotFoundError:
    from d_marginal_trial_scheduler import (
        BoundedDTrialPolicy,
        InitialDSignalPolicy,
        OutcomeLedger,
        ResourcePlanner,
        WindowObservation,
    )


class DMarginalTrialRuntime:
    """Build shadow/apply quota plans from Central I/O facts and D windows."""

    def __init__(
        self,
        control: Any,
        *,
        trial_quantum_pages: int,
        min_revisit_opportunity_tokens: int,
        min_confirmed_loss_tokens: int,
        min_severity: float,
        min_volume_loss_tokens_per_s: float,
        donor_max_severity: float = 0.0,
        signal_policy: Any | None = None,
        trial_policy: Any | None = None,
        planner: ResourcePlanner | None = None,
        ledger: OutcomeLedger | None = None,
        clock_ns=None,
    ) -> None:
        self.control = control
        self.signal_policy = signal_policy or InitialDSignalPolicy(
            min_revisit_opportunity_tokens=min_revisit_opportunity_tokens,
            min_confirmed_loss_tokens=min_confirmed_loss_tokens,
            min_severity=min_severity,
            min_volume_loss_tokens_per_s=min_volume_loss_tokens_per_s,
            donor_max_severity=donor_max_severity,
        )
        self.trial_policy = trial_policy or BoundedDTrialPolicy(
            trial_quantum_pages=trial_quantum_pages
        )
        self.planner = planner or ResourcePlanner()
        if ledger is not None:
            self.ledger = ledger
        elif clock_ns is not None:
            self.ledger = OutcomeLedger(clock_ns=clock_ns)
        else:
            self.ledger = OutcomeLedger()

    def tick(
        self,
        d_reports: dict[str, dict[str, Any]],
        *,
        mode: str = "shadow",
        trial_id: str,
    ) -> dict[str, Any]:
        if mode not in {"shadow", "apply"}:
            raise ValueError("mode must be shadow or apply")
        snapshot = self.control.global_status()
        models = snapshot["models"]
        missing_residency = sorted(
            model_id
            for model_id, state in models.items()
            if state.get("residency") is None
        )
        missing_observations = sorted(
            model_id
            for model_id in models
            if model_id not in d_reports and model_id not in missing_residency
        )
        pending_models = sorted(
            model_id
            for model_id, state in models.items()
            if (
                int(state.get("pending_capacity", 0)) > 0
                or int(state.get("pending_release_capacity", 0)) > 0
                or self._has_unfinished_target(state)
            )
        )
        eligible = {
            model_id: state
            for model_id, state in models.items()
            if model_id not in missing_residency
            and model_id not in missing_observations
            and model_id not in pending_models
        }
        observations = {
            model_id: self._window_observation(model_id, state, d_reports[model_id])
            for model_id, state in eligible.items()
        }
        signals = {
            model_id: self.signal_policy.evaluate(observation)
            for model_id, observation in observations.items()
        }
        candidates = self._candidate_decisions(
            list(observations.values()),
            list(signals.values()),
        )
        plan = self.planner.plan(
            candidates,
            list(observations.values()),
            list(signals.values()),
            global_free_pages=self._global_free_pages(snapshot, models),
        )
        outcome_record = self.ledger.start_trial(
            trial_id=trial_id,
            mode=mode,
            observations=observations,
            signals=signals,
            transactions=plan.transactions,
            unmet_pages=plan.unmet_pages,
        )
        published_targets: dict[str, int] = {}
        published_target_set_ns = None
        if mode == "apply":
            for model_id, target_pages in plan.targets.items():
                state = eligible[model_id]
                target_slots = int(target_pages) * int(state["page_size"])
                if target_slots != int(state["capacity"]):
                    published_targets[model_id] = target_slots
            if published_targets:
                published = self.control.set_quota_targets(published_targets)
                raw_timestamp = published.get("target_set_ns")
                if raw_timestamp is not None:
                    published_target_set_ns = int(raw_timestamp)
                outcome_record = self.ledger.record_intent_submitted(
                    trial_id,
                    intent_id=f"target_set:{published_target_set_ns}",
                    timestamp_ns=published_target_set_ns,
                )

        return {
            "mode": mode,
            "published_targets": published_targets,
            "published_target_set_ns": published_target_set_ns,
            "plan": plan.as_dict(),
            "candidates": [asdict(candidate) for candidate in candidates],
            "signals": {
                model_id: asdict(signal) for model_id, signal in signals.items()
            },
            "missing_residency": missing_residency,
            "missing_observations": missing_observations,
            "pending_models": pending_models,
            "outcome_record": outcome_record,
        }

    def record_allocator_ack(
        self,
        trial_id: str,
        *,
        effective_pages: dict[str, int],
        timestamp_ns: int | None = None,
    ) -> dict:
        return self.ledger.record_allocator_ack(
            trial_id,
            effective_pages=effective_pages,
            timestamp_ns=timestamp_ns,
        )

    def _candidate_decisions(self, observations, signals):
        parameters = inspect.signature(self.trial_policy.candidates).parameters
        if len(parameters) == 2:
            return self.trial_policy.candidates(observations, signals)
        return self.trial_policy.candidates(signals)

    @staticmethod
    def _has_unfinished_target(state: dict[str, Any]) -> bool:
        target = state.get("quota_target")
        return target is not None and int(target) != int(state["capacity"])

    @staticmethod
    def _effective_pages(state: dict[str, Any]) -> int:
        page_size = int(state["page_size"])
        capacity = int(state["capacity"])
        if capacity % page_size:
            raise ValueError("Central I/O capacity is not page aligned")
        return capacity // page_size

    def _window_observation(
        self,
        model_id: str,
        state: dict[str, Any],
        d_report: dict[str, Any],
    ) -> WindowObservation:
        residency = state["residency"]
        effective_pages = self._effective_pages(state)
        if int(residency["effective_pages"]) != effective_pages:
            raise ValueError("local residency effective pages disagree with Central I/O")
        clean_surplus = max(
            0,
            int(residency.get("clean_pages", 0)) - int(residency.get("high_pages", 0)),
        )
        return WindowObservation(
            model_id=model_id,
            window_start_s=float(d_report["window_start_s"]),
            window_end_s=float(d_report["window_end_s"]),
            effective_pages=effective_pages,
            floor_pages=int(residency["floor_pages"]),
            raw_retention_loss_tokens=int(
                d_report.get(
                    "raw_retention_loss_tokens",
                    d_report.get("confirmed_reuse_loss_tokens", 0),
                )
            ),
            confirmed_reuse_loss_tokens=int(
                d_report.get("confirmed_reuse_loss_tokens", 0)
            ),
            revisit_opportunity_tokens=int(
                d_report.get("revisit_opportunity_tokens", 0)
            ),
            safe_clean_offer_pages=clean_surplus,
            safe_ready_offer_pages=int(residency.get("ready_pages", 0)),
            admission_shortfall_pages=int(residency.get("unresolved_clean_pages", 0)),
            host_evicted_tokens=int(d_report.get("host_evicted_tokens", 0)),
            reprefill_tokens=int(d_report.get("reprefill_tokens", 0)),
            revisit_cache_match_tokens=int(
                d_report.get("revisit_cache_match_tokens", 0)
            ),
            revisit_cache_miss_tokens=int(
                d_report.get("revisit_cache_miss_tokens", 0)
            ),
            revisit_incremental_tokens=int(
                d_report.get("revisit_incremental_tokens", 0)
            ),
            revisit_seen_cache_miss_tokens=int(
                d_report.get("revisit_seen_cache_miss_tokens", 0)
            ),
            d_coverage_of_seen_miss=(
                None
                if d_report.get("d_coverage_of_seen_miss") is None
                else float(d_report["d_coverage_of_seen_miss"])
            ),
            conditional_ttft_ms_p50=(
                None
                if d_report.get("conditional_ttft_ms_p50") is None
                else float(d_report["conditional_ttft_ms_p50"])
            ),
            legacy_retention_debt_tokens=int(
                residency.get("retention_debt_tokens", 0)
            ),
        )

    @staticmethod
    def _global_free_pages(
        snapshot: dict[str, Any],
        models: dict[str, dict[str, Any]],
    ) -> int:
        geometries = {
            (int(state["page_size"]), int(state["token_bytes"]))
            for state in models.values()
        }
        if len(geometries) != 1:
            raise ValueError("D scheduler runtime requires uniform page geometry")
        page_size, token_bytes = geometries.pop()
        page_bytes = page_size * token_bytes
        if page_bytes <= 0:
            raise ValueError("invalid page geometry")
        return int(snapshot.get("global_free_bytes", 0)) // page_bytes
