"""Build live quota intents for fixed D decision windows.

This utility is policy-side only. It does not talk to Central I/O and it does
not apply quota changes.  A runner feeds it an audited Central I/O snapshot plus
explicit per-window D observations, then submits the emitted JSONL intents to
the policy-free manual quota runner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from sglang.srt.mem_cache.d_marginal_trial_runtime import DMarginalTrialRuntime
    from sglang.srt.mem_cache.d_marginal_trial_scheduler import (
        ImpactSizedDReallocationPolicy,
        InitialDSignalPolicy,
        ResourcePlan,
        ResourcePlanner,
        PlannedTransaction,
        manual_intents_from_resource_plan,
    )
except ModuleNotFoundError:
    from d_marginal_trial_runtime import DMarginalTrialRuntime
    from d_marginal_trial_scheduler import (
        ImpactSizedDReallocationPolicy,
        InitialDSignalPolicy,
        ResourcePlan,
        ResourcePlanner,
        PlannedTransaction,
        manual_intents_from_resource_plan,
    )


class SnapshotControl:
    def __init__(self, snapshot: dict[str, Any]) -> None:
        self.snapshot = snapshot

    def global_status(self) -> dict[str, Any]:
        return self.snapshot

    def set_quota_targets(self, targets: dict[str, int]) -> dict[str, Any]:
        raise RuntimeError("live window planner is shadow-only")


def plan_windows(
    *,
    snapshot: dict[str, Any],
    windows: list[dict[str, Any]],
    base_grow_pages: int,
    base_shrink_pages: int,
    min_revisit_opportunity_tokens: int,
    min_confirmed_loss_tokens: int,
    min_severity: float,
    min_volume_loss_tokens_per_s: float,
    donor_max_severity: float,
    cold_revisit_opportunity_tokens: int,
    cold_max_confirmed_loss_tokens: int,
    high_urgency_multiplier: int,
    medium_urgency_multiplier: int,
    min_grow_pages: int,
    min_shrink_pages: int,
    grow_effective_ratio: float,
    shrink_effective_ratio: float,
    donor_offer_safety_margin_pages: int,
    donor_contribution_cap_pages: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    proofs: list[dict[str, Any]] = []
    intents: list[dict[str, Any]] = []
    signal_policy = InitialDSignalPolicy(
        min_revisit_opportunity_tokens=min_revisit_opportunity_tokens,
        min_confirmed_loss_tokens=min_confirmed_loss_tokens,
        min_severity=min_severity,
        min_volume_loss_tokens_per_s=min_volume_loss_tokens_per_s,
        donor_max_severity=donor_max_severity,
    )
    trial_policy = ImpactSizedDReallocationPolicy(
        grow_quantum_pages=base_grow_pages,
        shrink_quantum_pages=base_shrink_pages,
        cold_revisit_opportunity_tokens=cold_revisit_opportunity_tokens,
        cold_max_confirmed_loss_tokens=cold_max_confirmed_loss_tokens,
        high_urgency_multiplier=high_urgency_multiplier,
        medium_urgency_multiplier=medium_urgency_multiplier,
        min_grow_pages=min_grow_pages,
        min_shrink_pages=min_shrink_pages,
        grow_effective_ratio=grow_effective_ratio,
        shrink_effective_ratio=shrink_effective_ratio,
    )
    for index, window in enumerate(windows, start=1):
        window_id = str(window.get("decision_window_id") or f"window-{index}")
        runtime = DMarginalTrialRuntime(
            SnapshotControl(snapshot),
            trial_quantum_pages=base_grow_pages,
            min_revisit_opportunity_tokens=min_revisit_opportunity_tokens,
            min_confirmed_loss_tokens=min_confirmed_loss_tokens,
            min_severity=min_severity,
            min_volume_loss_tokens_per_s=min_volume_loss_tokens_per_s,
            donor_max_severity=donor_max_severity,
            signal_policy=signal_policy,
            trial_policy=trial_policy,
            planner=ResourcePlanner(
                offer_safety_margin_pages=donor_offer_safety_margin_pages,
                donor_contribution_cap_pages=donor_contribution_cap_pages,
            ),
        )
        shadow = runtime.tick(
            window["reports"],
            mode="shadow",
            trial_id=f"shadow-{window_id}",
        )
        plan = ResourcePlan(
            targets={key: int(value) for key, value in shadow["plan"]["targets"].items()},
            transactions=[
                PlannedTransaction(**item)
                for item in shadow["plan"]["transactions"]
            ],
            unmet_pages={
                key: int(value) for key, value in shadow["plan"]["unmet_pages"].items()
            },
        )
        window_intents = manual_intents_from_resource_plan(
            plan,
            intent_id_prefix=window_id,
            async_overlap=bool(window.get("async_overlap", False)),
        )
        for intent in window_intents:
            intent["phase"] = "live_transaction_window"
            intent["decision_window_id"] = window_id
        proofs.append(
            {
                "decision_window_id": window_id,
                "shadow": shadow,
                "generated_intents": window_intents,
            }
        )
        intents.extend(window_intents)
    return proofs, intents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-json", type=Path, required=True)
    parser.add_argument("--windows-json", type=Path, required=True)
    parser.add_argument("--proof-json", type=Path, required=True)
    parser.add_argument("--intent-jsonl", type=Path, required=True)
    parser.add_argument("--base-grow-pages", type=int, required=True)
    parser.add_argument("--base-shrink-pages", type=int, required=True)
    parser.add_argument("--min-revisit-opportunity-tokens", type=int, default=1000)
    parser.add_argument("--min-confirmed-loss-tokens", type=int, default=100)
    parser.add_argument("--min-severity", type=float, default=0.05)
    parser.add_argument("--min-volume-loss-tokens-per-s", type=float, default=100.0)
    parser.add_argument("--donor-max-severity", type=float, default=0.0)
    parser.add_argument("--cold-revisit-opportunity-tokens", type=int, default=5000)
    parser.add_argument("--cold-max-confirmed-loss-tokens", type=int, default=0)
    parser.add_argument("--high-urgency-multiplier", type=int, default=4)
    parser.add_argument("--medium-urgency-multiplier", type=int, default=2)
    parser.add_argument("--min-grow-pages", type=int, default=0)
    parser.add_argument("--min-shrink-pages", type=int, default=0)
    parser.add_argument("--grow-effective-ratio", type=float, default=0.0)
    parser.add_argument("--shrink-effective-ratio", type=float, default=0.0)
    parser.add_argument("--donor-offer-safety-margin-pages", type=int, default=64)
    parser.add_argument("--donor-contribution-cap-pages", type=int, default=0)
    args = parser.parse_args()

    snapshot = json.loads(args.snapshot_json.read_text())
    windows = json.loads(args.windows_json.read_text())
    proofs, intents = plan_windows(
        snapshot=snapshot,
        windows=windows,
        base_grow_pages=args.base_grow_pages,
        base_shrink_pages=args.base_shrink_pages,
        min_revisit_opportunity_tokens=args.min_revisit_opportunity_tokens,
        min_confirmed_loss_tokens=args.min_confirmed_loss_tokens,
        min_severity=args.min_severity,
        min_volume_loss_tokens_per_s=args.min_volume_loss_tokens_per_s,
        donor_max_severity=args.donor_max_severity,
        cold_revisit_opportunity_tokens=args.cold_revisit_opportunity_tokens,
        cold_max_confirmed_loss_tokens=args.cold_max_confirmed_loss_tokens,
        high_urgency_multiplier=args.high_urgency_multiplier,
        medium_urgency_multiplier=args.medium_urgency_multiplier,
        min_grow_pages=args.min_grow_pages,
        min_shrink_pages=args.min_shrink_pages,
        grow_effective_ratio=args.grow_effective_ratio,
        shrink_effective_ratio=args.shrink_effective_ratio,
        donor_offer_safety_margin_pages=args.donor_offer_safety_margin_pages,
        donor_contribution_cap_pages=args.donor_contribution_cap_pages,
    )
    args.proof_json.parent.mkdir(parents=True, exist_ok=True)
    args.proof_json.write_text(json.dumps(proofs, indent=2, sort_keys=True) + "\n")
    args.intent_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.intent_jsonl.open("w", encoding="utf-8") as handle:
        for intent in intents:
            handle.write(json.dumps(intent, sort_keys=True, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
