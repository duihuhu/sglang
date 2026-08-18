"""CPU checks for the replaceable D-driven quota scheduler architecture."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from d_marginal_trial_scheduler import (
    CandidateDecision,
    InitialDReallocationPolicy,
    InitialDSignalPolicy,
    ImpactSizedDReallocationPolicy,
    OutcomeLedger,
    PlannedTransaction,
    ResourcePlanner,
    ResourcePlan,
    TrialPolicy,
    UrgencyScaledDReallocationPolicy,
    WindowObservation,
    manual_intents_from_resource_plan,
    select_manual_intents_for_trial,
)


def observation(
    model_id: str,
    *,
    effective: int = 1_000,
    floor: int = 500,
    d_raw: int = 0,
    opportunity: int = 10_000,
    duration: float = 10.0,
    clean_offer: int = 0,
    ready_offer: int = 0,
    admission_shortfall: int = 0,
    retention_debt: int = 0,
    reprefill: int = 0,
    revisit_cache_miss: int = 0,
    revisit_incremental: int = 0,
    revisit_seen_cache_miss: int = 0,
    d_coverage: float | None = None,
    raw_retention_loss: int | None = None,
) -> WindowObservation:
    return WindowObservation(
        model_id=model_id,
        window_start_s=0.0,
        window_end_s=duration,
        effective_pages=effective,
        floor_pages=floor,
        raw_retention_loss_tokens=(
            d_raw if raw_retention_loss is None else raw_retention_loss
        ),
        confirmed_reuse_loss_tokens=d_raw,
        revisit_opportunity_tokens=opportunity,
        safe_clean_offer_pages=clean_offer,
        safe_ready_offer_pages=ready_offer,
        admission_shortfall_pages=admission_shortfall,
        host_evicted_tokens=0,
        reprefill_tokens=reprefill,
        revisit_cache_match_tokens=opportunity - d_raw,
        revisit_cache_miss_tokens=revisit_cache_miss,
        revisit_incremental_tokens=revisit_incremental,
        revisit_seen_cache_miss_tokens=revisit_seen_cache_miss,
        d_coverage_of_seen_miss=d_coverage,
        conditional_ttft_ms_p50=None,
        legacy_retention_debt_tokens=retention_debt,
    )


class _VolumeFirstTrialPolicy(TrialPolicy):
    def candidates(self, signals):
        return sorted([
            CandidateDecision(
                model_id=signal.model_id,
                action="grow",
                pages=100,
                reason="volume",
                rank=(-signal.volume_loss_tokens_per_s, signal.model_id),
            )
            for signal in signals
            if signal.triggered
        ], key=lambda item: item.rank)


class SchedulerArchitectureTest(unittest.TestCase):
    def test_initial_signal_policy_keeps_d_raw_severity_and_volume_separate(self):
        policy = InitialDSignalPolicy(
            min_revisit_opportunity_tokens=1_000,
            min_confirmed_loss_tokens=100,
            min_severity=0.05,
            min_volume_loss_tokens_per_s=10.0,
        )

        signal = policy.evaluate(observation("hot", d_raw=500, opportunity=2_000, duration=5.0))

        self.assertEqual(signal.d_raw_tokens, 500)
        self.assertEqual(signal.revisit_opportunity_tokens, 2_000)
        self.assertEqual(signal.severity, 0.25)
        self.assertEqual(signal.volume_loss_tokens_per_s, 100.0)
        self.assertTrue(signal.triggered)

    def test_volume_first_policy_can_replace_severity_first_sorting(self):
        signal_policy = InitialDSignalPolicy(
            min_revisit_opportunity_tokens=1_000,
            min_confirmed_loss_tokens=100,
            min_severity=0.01,
            min_volume_loss_tokens_per_s=1.0,
        )
        observations = [
            observation("high-rate-small", d_raw=1_000, opportunity=2_000, duration=100.0),
            observation("low-rate-large", d_raw=50_000, opportunity=500_000, duration=10.0),
            observation("donor", clean_offer=10_000),
        ]
        signals = [signal_policy.evaluate(item) for item in observations]

        candidates = _VolumeFirstTrialPolicy().candidates(signals)

        self.assertEqual(candidates[0].model_id, "low-rate-large")
        self.assertEqual(candidates[0].reason, "volume")
        self.assertLess(candidates[0].rank, candidates[1].rank)

    def test_signal_policy_triggers_on_high_volume_even_when_severity_is_low(self):
        signal_policy = InitialDSignalPolicy(
            min_revisit_opportunity_tokens=1_000,
            min_confirmed_loss_tokens=100,
            min_severity=0.05,
            min_volume_loss_tokens_per_s=100.0,
        )

        signal = signal_policy.evaluate(
            observation("wide-opportunity-hot", d_raw=47_127, opportunity=1_334_191, duration=300.0)
        )

        self.assertLess(signal.severity, 0.05)
        self.assertGreater(signal.volume_loss_tokens_per_s, 100.0)
        self.assertTrue(signal.triggered)
        self.assertEqual(signal.reason, "triggered_by_volume")

    def test_signal_policy_keeps_small_low_severity_volume_as_hold(self):
        signal_policy = InitialDSignalPolicy(
            min_revisit_opportunity_tokens=1_000,
            min_confirmed_loss_tokens=100,
            min_severity=0.05,
            min_volume_loss_tokens_per_s=100.0,
        )

        signal = signal_policy.evaluate(
            observation("tiny-d", d_raw=4_961, opportunity=3_003_739, duration=300.0)
        )

        self.assertFalse(signal.triggered)
        self.assertEqual(signal.reason, "severity_and_volume_below_threshold")

    def test_signal_policy_does_not_treat_unobserved_instance_as_safe_donor(self):
        signal_policy = InitialDSignalPolicy(
            min_revisit_opportunity_tokens=1_000,
            min_confirmed_loss_tokens=100,
            min_severity=0.05,
            min_volume_loss_tokens_per_s=100.0,
            donor_max_severity=0.0,
        )

        signal = signal_policy.evaluate(observation("not-yet-observed", opportunity=0))

        self.assertFalse(signal.triggered)
        self.assertEqual(signal.reason, "insufficient_revisit_opportunity")
        self.assertFalse(signal.donor_allowed)

    def test_signal_policy_allows_observed_zero_d_instance_as_safe_donor(self):
        signal_policy = InitialDSignalPolicy(
            min_revisit_opportunity_tokens=1_000,
            min_confirmed_loss_tokens=100,
            min_severity=0.05,
            min_volume_loss_tokens_per_s=100.0,
            donor_max_severity=0.0,
        )

        signal = signal_policy.evaluate(observation("observed-cold", d_raw=0, opportunity=50_000))

        self.assertFalse(signal.triggered)
        self.assertEqual(signal.reason, "insufficient_confirmed_loss")
        self.assertTrue(signal.donor_allowed)

    def test_legacy_retention_debt_is_observed_but_not_a_donor_gate(self):
        signal_policy = InitialDSignalPolicy(
            min_revisit_opportunity_tokens=1_000,
            min_confirmed_loss_tokens=100,
            min_severity=0.10,
            min_volume_loss_tokens_per_s=1.0,
            donor_max_severity=0.0,
        )
        planner = ResourcePlanner()
        hot = observation("hot", d_raw=500, opportunity=2_000)
        donor = observation("donor", clean_offer=200, retention_debt=999)
        signals = [signal_policy.evaluate(hot), signal_policy.evaluate(donor)]
        candidate = CandidateDecision(
            model_id="hot",
            action="grow",
            pages=100,
            reason="trial",
            rank=(0, "hot"),
        )

        plan = planner.plan([candidate], [hot, donor], signals, global_free_pages=0)

        self.assertEqual(plan.transactions[0].source_model_id, "donor")
        self.assertEqual(plan.transactions[0].readiness, "clean")
        self.assertEqual(plan.targets["donor"], 900)

    def test_admission_shortfall_blocks_donor_offer_without_becoming_d_signal(self):
        signal_policy = InitialDSignalPolicy(
            min_revisit_opportunity_tokens=1_000,
            min_confirmed_loss_tokens=100,
            min_severity=0.10,
            min_volume_loss_tokens_per_s=1.0,
        )
        planner = ResourcePlanner()
        hot = observation("hot", d_raw=500, opportunity=2_000)
        blocked = observation("blocked", clean_offer=200, admission_shortfall=1)
        safe = observation("safe", clean_offer=200)
        signals = [signal_policy.evaluate(item) for item in [hot, blocked, safe]]
        candidate = CandidateDecision(
            model_id="hot",
            action="grow",
            pages=100,
            reason="trial",
            rank=(0, "hot"),
        )

        plan = planner.plan([candidate], [hot, blocked, safe], signals, global_free_pages=0)

        self.assertEqual(plan.transactions[0].source_model_id, "safe")
        self.assertEqual(plan.targets["blocked"], 1_000)
        self.assertFalse(signals[1].triggered)

    def test_resource_planner_keeps_donor_offer_safety_margin_for_live_drift(self):
        signal_policy = InitialDSignalPolicy(
            min_revisit_opportunity_tokens=1_000,
            min_confirmed_loss_tokens=100,
            min_severity=0.10,
            min_volume_loss_tokens_per_s=1.0,
        )
        planner = ResourcePlanner(offer_safety_margin_pages=64)
        hot = observation("hot", d_raw=500, opportunity=2_000)
        donor = observation("donor", effective=2_000, floor=500, ready_offer=1_000)
        signals = [signal_policy.evaluate(item) for item in [hot, donor]]
        candidate = CandidateDecision(
            model_id="hot",
            action="grow",
            pages=1_000,
            reason="trial",
            rank=(0, "hot"),
        )

        plan = planner.plan([candidate], [hot, donor], signals, global_free_pages=0)

        self.assertEqual(plan.transactions[0].pages, 936)
        self.assertEqual(plan.transactions[0].readiness, "ready")
        self.assertEqual(plan.unmet_pages, {"hot": 64})
        self.assertEqual(plan.targets["donor"], 1_064)

    def test_resource_planner_caps_each_donor_contribution_per_grow_candidate(self):
        signal_policy = InitialDSignalPolicy(
            min_revisit_opportunity_tokens=1_000,
            min_confirmed_loss_tokens=100,
            min_severity=0.10,
            min_volume_loss_tokens_per_s=1.0,
        )
        planner = ResourcePlanner(donor_contribution_cap_pages=5_000)
        hot = observation("hot", d_raw=10_000, opportunity=20_000)
        donor_a = observation("donor-a", effective=20_000, floor=1_000, clean_offer=20_000)
        donor_b = observation("donor-b", effective=20_000, floor=1_000, clean_offer=20_000)
        observations = [hot, donor_a, donor_b]
        signals = [signal_policy.evaluate(item) for item in observations]
        candidate = CandidateDecision(
            model_id="hot",
            action="grow",
            pages=20_000,
            reason="trial",
            rank=(0, "hot"),
        )

        plan = planner.plan([candidate], observations, signals, global_free_pages=0)

        self.assertEqual(
            [(item.source_model_id, item.pages) for item in plan.transactions],
            [("donor-a", 5_000), ("donor-b", 5_000)],
        )
        self.assertEqual(plan.unmet_pages, {"hot": 10_000})
        self.assertEqual(plan.targets["donor-a"], 15_000)
        self.assertEqual(plan.targets["donor-b"], 15_000)

    def test_resource_planner_uses_reserve_first_then_clean_then_ready_and_preserves_sum(self):
        planner = ResourcePlanner()
        hot = observation("hot", d_raw=500, opportunity=2_000)
        clean = observation("clean", clean_offer=30)
        ready = observation("ready", ready_offer=100)
        signal_policy = InitialDSignalPolicy(
            min_revisit_opportunity_tokens=1_000,
            min_confirmed_loss_tokens=100,
            min_severity=0.10,
            min_volume_loss_tokens_per_s=1.0,
        )
        signals = [signal_policy.evaluate(item) for item in [hot, clean, ready]]
        candidate = CandidateDecision(
            model_id="hot",
            action="grow",
            pages=100,
            reason="trial",
            rank=(0, "hot"),
        )

        plan = planner.plan([candidate], [hot, clean, ready], signals, global_free_pages=40)

        self.assertEqual(
            [(txn.source_model_id, txn.pages, txn.readiness) for txn in plan.transactions],
            [(None, 40, "central_free"), ("clean", 30, "clean"), ("ready", 30, "ready")],
        )
        self.assertEqual(sum(plan.targets.values()), 3_000 + 40)

    def test_shadow_plan_is_the_only_source_for_manual_manager_intents(self):
        planner = ResourcePlanner()
        hot = observation("hot", d_raw=500, opportunity=2_000)
        donor = observation("donor", clean_offer=100)
        signal_policy = InitialDSignalPolicy(
            min_revisit_opportunity_tokens=1_000,
            min_confirmed_loss_tokens=100,
            min_severity=0.10,
            min_volume_loss_tokens_per_s=1.0,
        )
        signals = [signal_policy.evaluate(item) for item in [hot, donor]]
        candidate = CandidateDecision(
            model_id="hot",
            action="grow",
            pages=100,
            reason="trial",
            rank=(0, "hot"),
        )
        plan = planner.plan([candidate], [hot, donor], signals, global_free_pages=40)

        intents = manual_intents_from_resource_plan(
            plan,
            intent_id_prefix="shadow-qualified",
        )

        self.assertEqual(
            intents,
            [
                {
                    "intent_id": "shadow-qualified-1",
                    "op": "grow",
                    "recipient": "hot",
                    "pages": 40,
                    "tranche_pages": 40,
                },
                {
                    "intent_id": "shadow-qualified-2",
                    "op": "transfer",
                    "donor": "donor",
                    "recipient": "hot",
                    "pages": 60,
                    "tranche_pages": 60,
                    "offer_id": "shadow-qualified:offer:donor:hot:60:clean",
                    "require_donor_offer_pages": 60,
                },
            ],
        )

    def test_initial_reallocation_policy_emits_grow_only_not_cold_shrink(self):
        signal_policy = InitialDSignalPolicy(
            min_revisit_opportunity_tokens=1_000,
            min_confirmed_loss_tokens=100,
            min_severity=0.10,
            min_volume_loss_tokens_per_s=10.0,
        )
        policy = InitialDReallocationPolicy(
            grow_quantum_pages=100,
            shrink_quantum_pages=40,
            cold_revisit_opportunity_tokens=5_000,
            cold_max_confirmed_loss_tokens=0,
        )
        hot = observation("hot", d_raw=2_000, opportunity=10_000, duration=10.0)
        cold = observation("cold", d_raw=0, opportunity=20_000, clean_offer=80)
        blocked = observation(
            "blocked",
            d_raw=0,
            opportunity=20_000,
            clean_offer=80,
            admission_shortfall=1,
        )
        signals = [signal_policy.evaluate(item) for item in [hot, cold, blocked]]

        candidates = policy.candidates([hot, cold, blocked], signals)

        self.assertEqual(
            [(item.model_id, item.action, item.pages, item.reason) for item in candidates],
            [("hot", "grow", 100, "initial_d_reallocation_grow_trial")],
        )

    def test_no_grow_demand_holds_even_with_safe_cold_donor_and_free_inventory(self):
        planner = ResourcePlanner()
        cold = observation("cold", d_raw=0, opportunity=20_000, clean_offer=80)
        signal_policy = InitialDSignalPolicy(
            min_revisit_opportunity_tokens=1_000,
            min_confirmed_loss_tokens=100,
            min_severity=0.10,
            min_volume_loss_tokens_per_s=10.0,
        )
        signals = [signal_policy.evaluate(cold)]
        candidate = CandidateDecision(
            model_id="cold",
            action="shrink",
            pages=40,
            reason="cold",
            rank=(0, "cold"),
        )

        plan = planner.plan([candidate], [cold], signals, global_free_pages=0)

        self.assertEqual(plan.targets["cold"], 1_000)
        self.assertEqual(plan.transactions, [])
        self.assertEqual(manual_intents_from_resource_plan(plan, intent_id_prefix="cold"), [])

    def test_resource_planner_splits_urgent_grow_across_multiple_safe_donors(self):
        planner = ResourcePlanner()
        hot = observation("hot", d_raw=5_000, opportunity=20_000, duration=10.0)
        donor_a = observation("donor-a", d_raw=0, opportunity=20_000, clean_offer=30)
        donor_b = observation("donor-b", d_raw=0, opportunity=20_000, clean_offer=50)
        signal_policy = InitialDSignalPolicy(
            min_revisit_opportunity_tokens=1_000,
            min_confirmed_loss_tokens=100,
            min_severity=0.10,
            min_volume_loss_tokens_per_s=10.0,
        )
        signals = [signal_policy.evaluate(item) for item in [hot, donor_a, donor_b]]
        candidate = CandidateDecision(
            model_id="hot",
            action="grow",
            pages=100,
            reason="urgent",
            rank=(0, "hot"),
        )

        plan = planner.plan([candidate], [hot, donor_a, donor_b], signals, global_free_pages=20)

        self.assertEqual(plan.targets, {"hot": 1_100, "donor-a": 970, "donor-b": 950})
        self.assertEqual(
            [(item.source_model_id, item.destination_model_id, item.pages, item.readiness)
             for item in plan.transactions],
            [
                (None, "hot", 20, "central_free"),
                ("donor-b", "hot", 50, "clean"),
                ("donor-a", "hot", 30, "clean"),
            ],
        )

    def test_kxq_grow_plan_aggregates_free_and_multiple_safe_donors(self):
        planner = ResourcePlanner()
        hot = observation("hot", d_raw=8_000, opportunity=20_000, duration=10.0)
        donor_a = observation("donor-a", clean_offer=120)
        donor_b = observation("donor-b", clean_offer=90)
        donor_c = observation("donor-c", ready_offer=100)
        signal_policy = InitialDSignalPolicy(
            min_revisit_opportunity_tokens=1_000,
            min_confirmed_loss_tokens=100,
            min_severity=0.05,
            min_volume_loss_tokens_per_s=10.0,
        )
        signals = [
            signal_policy.evaluate(item)
            for item in [hot, donor_a, donor_b, donor_c]
        ]
        candidate = CandidateDecision("hot", "grow", 320, "urgent_kxq", (0, "hot"))

        plan = planner.plan(
            [candidate],
            [hot, donor_a, donor_b, donor_c],
            signals,
            global_free_pages=50,
        )

        self.assertEqual(
            [(item.source_model_id, item.destination_model_id, item.pages, item.readiness)
             for item in plan.transactions],
            [
                (None, "hot", 50, "central_free"),
                ("donor-a", "hot", 120, "clean"),
                ("donor-b", "hot", 90, "clean"),
                ("donor-c", "hot", 60, "ready"),
            ],
        )
        self.assertEqual(plan.targets["hot"], 1_320)
        self.assertEqual(plan.unmet_pages, {})

    def test_safe_offer_floor_limits_report_unmet_when_grow_needs_more_pages(self):
        planner = ResourcePlanner()
        hot = observation("hot", d_raw=5_000, opportunity=20_000, duration=10.0)
        donor = observation("donor", d_raw=0, opportunity=20_000, clean_offer=80)
        signal_policy = InitialDSignalPolicy(
            min_revisit_opportunity_tokens=1_000,
            min_confirmed_loss_tokens=100,
            min_severity=0.10,
            min_volume_loss_tokens_per_s=10.0,
        )
        signals = [signal_policy.evaluate(item) for item in [hot, donor]]
        candidates = [CandidateDecision("hot", "grow", 120, "urgent", (0, "hot"))]

        plan = planner.plan(candidates, [hot, donor], signals, global_free_pages=0)

        self.assertEqual(plan.targets, {"hot": 1_080, "donor": 920})
        self.assertEqual(plan.unmet_pages, {"hot": 40})
        self.assertEqual(len(plan.transactions), 1)

    def test_grow_triggered_matching_uses_safe_donor_offer_without_policy_shrink_candidate(self):
        signal_policy = InitialDSignalPolicy(
            min_revisit_opportunity_tokens=1_000,
            min_confirmed_loss_tokens=100,
            min_severity=0.10,
            min_volume_loss_tokens_per_s=10.0,
        )
        hot = observation("hot", d_raw=5_000, opportunity=20_000, duration=10.0)
        donor_a = observation("donor-a", d_raw=0, opportunity=20_000, clean_offer=100)
        donor_b = observation("donor-b", d_raw=0, opportunity=20_000, clean_offer=100)
        signals = [signal_policy.evaluate(item) for item in [hot, donor_a, donor_b]]
        candidates = [CandidateDecision("hot", "grow", 100, "urgent", (0, "hot"))]

        plan = ResourcePlanner().plan(
            candidates,
            [hot, donor_a, donor_b],
            signals,
            global_free_pages=0,
        )

        self.assertEqual(
            [(item.source_model_id, item.destination_model_id, item.pages)
             for item in plan.transactions],
            [("donor-a", "hot", 100)],
        )
        self.assertEqual(plan.targets, {"hot": 1_100, "donor-a": 900, "donor-b": 1_000})
        self.assertEqual(plan.unmet_pages, {})

    def test_no_reservation_rotating_hot_windows_match_only_live_grow_demand(self):
        signal_policy = InitialDSignalPolicy(
            min_revisit_opportunity_tokens=1_000,
            min_confirmed_loss_tokens=100,
            min_severity=0.05,
            min_volume_loss_tokens_per_s=10.0,
        )
        policy = InitialDReallocationPolicy(
            grow_quantum_pages=100,
            shrink_quantum_pages=40,
            cold_revisit_opportunity_tokens=5_000,
            cold_max_confirmed_loss_tokens=0,
        )

        def plan_for(items):
            signals = [signal_policy.evaluate(item) for item in items]
            candidates = policy.candidates(items, signals)
            return ResourcePlanner().plan(
                candidates,
                items,
                signals,
                global_free_pages=0,
            )

        a_hot = plan_for([
            observation("A", d_raw=6_000, opportunity=30_000, clean_offer=100),
            observation("B", d_raw=0, opportunity=50_000, clean_offer=100),
            observation("C", d_raw=0, opportunity=50_000, clean_offer=100),
        ])
        b_hot = plan_for([
            observation("A", d_raw=0, opportunity=50_000, clean_offer=100),
            observation("B", d_raw=6_000, opportunity=30_000, clean_offer=100),
            observation("C", d_raw=0, opportunity=50_000, clean_offer=100),
        ])
        no_hot = plan_for([
            observation("A", d_raw=0, opportunity=50_000, clean_offer=100),
            observation("B", d_raw=0, opportunity=50_000, clean_offer=100),
            observation("C", d_raw=0, opportunity=50_000, clean_offer=100),
        ])

        self.assertEqual(
            [(item.source_model_id, item.destination_model_id, item.pages)
             for item in a_hot.transactions],
            [("B", "A", 100)],
        )
        self.assertEqual(a_hot.unmet_pages, {})
        self.assertEqual(
            [(item.source_model_id, item.destination_model_id, item.pages)
             for item in b_hot.transactions],
            [("A", "B", 100)],
        )
        self.assertEqual(b_hot.unmet_pages, {})
        self.assertEqual(no_hot.transactions, [])
        self.assertEqual(no_hot.targets, {"A": 1_000, "B": 1_000, "C": 1_000})

    def test_urgency_scaled_policy_increases_grow_without_emitting_donor_shrink(self):
        signal_policy = InitialDSignalPolicy(
            min_revisit_opportunity_tokens=1_000,
            min_confirmed_loss_tokens=100,
            min_severity=0.05,
            min_volume_loss_tokens_per_s=10.0,
        )
        policy = UrgencyScaledDReallocationPolicy(
            grow_quantum_pages=100,
            shrink_quantum_pages=40,
            cold_revisit_opportunity_tokens=5_000,
            cold_max_confirmed_loss_tokens=0,
            high_urgency_multiplier=4,
            medium_urgency_multiplier=2,
        )
        observations = [
            observation("hot", d_raw=8_000, opportunity=20_000, duration=10.0),
            observation("donor-a", d_raw=0, opportunity=50_000, clean_offer=200),
            observation("donor-b", d_raw=0, opportunity=50_000, clean_offer=200),
        ]
        signals = [signal_policy.evaluate(item) for item in observations]

        candidates = policy.candidates(observations, signals)
        plan = ResourcePlanner().plan(
            candidates,
            observations,
            signals,
            global_free_pages=0,
        )

        self.assertEqual(
            [(item.model_id, item.action, item.pages) for item in candidates],
            [("hot", "grow", 400)],
        )
        self.assertEqual(
            [(item.source_model_id, item.destination_model_id, item.pages)
             for item in plan.transactions],
            [("donor-a", "hot", 200), ("donor-b", "hot", 200)],
        )
        self.assertEqual(plan.targets, {"hot": 1_400, "donor-a": 800, "donor-b": 800})
        self.assertEqual(plan.unmet_pages, {})

    def test_urgency_scaled_policy_keeps_low_pressure_at_base_quantum(self):
        signal_policy = InitialDSignalPolicy(
            min_revisit_opportunity_tokens=1_000,
            min_confirmed_loss_tokens=100,
            min_severity=0.05,
            min_volume_loss_tokens_per_s=10.0,
        )
        policy = UrgencyScaledDReallocationPolicy(
            grow_quantum_pages=100,
            shrink_quantum_pages=40,
            cold_revisit_opportunity_tokens=5_000,
            cold_max_confirmed_loss_tokens=0,
            high_urgency_multiplier=4,
            medium_urgency_multiplier=2,
        )
        observations = [
            observation("hot", d_raw=120, opportunity=2_000, duration=10.0),
            observation("donor", d_raw=0, opportunity=50_000, clean_offer=200),
        ]
        signals = [signal_policy.evaluate(item) for item in observations]

        candidates = policy.candidates(observations, signals)

        self.assertEqual(
            [(item.model_id, item.action, item.pages) for item in candidates],
            [("hot", "grow", 100)],
        )

    def test_impact_sized_policy_uses_quota_ratio_floor_for_large_instances(self):
        signal_policy = InitialDSignalPolicy(
            min_revisit_opportunity_tokens=1_000,
            min_confirmed_loss_tokens=100,
            min_severity=0.05,
            min_volume_loss_tokens_per_s=10.0,
        )
        policy = ImpactSizedDReallocationPolicy(
            grow_quantum_pages=100,
            shrink_quantum_pages=40,
            min_grow_pages=5_000,
            min_shrink_pages=5_000,
            grow_effective_ratio=0.025,
            shrink_effective_ratio=0.025,
            cold_revisit_opportunity_tokens=5_000,
            cold_max_confirmed_loss_tokens=0,
            high_urgency_multiplier=4,
            medium_urgency_multiplier=2,
        )
        observations = [
            observation(
                "hot",
                effective=200_000,
                floor=100_000,
                d_raw=8_000,
                opportunity=20_000,
                duration=10.0,
                reprefill=4_000,
            ),
            observation("donor-a", effective=200_000, floor=100_000, opportunity=50_000, clean_offer=80_000),
            observation("donor-b", effective=200_000, floor=100_000, opportunity=50_000, clean_offer=80_000),
        ]
        signals = [signal_policy.evaluate(item) for item in observations]

        candidates = policy.candidates(observations, signals)
        plan = ResourcePlanner().plan(
            candidates,
            observations,
            signals,
            global_free_pages=0,
        )

        self.assertEqual(
            [(item.model_id, item.action, item.pages, item.reason) for item in candidates],
            [("hot", "grow", 20_000, "impact_sized_grow_x4_unit5000")],
        )
        self.assertEqual(
            [(item.source_model_id, item.destination_model_id, item.pages)
             for item in plan.transactions],
            [("donor-a", "hot", 20_000)],
        )
        self.assertEqual(plan.targets["hot"], 220_000)
        self.assertEqual(plan.targets["donor-a"], 180_000)
        self.assertEqual(plan.targets["donor-b"], 200_000)
        self.assertEqual(plan.unmet_pages, {})

    def test_impact_sized_policy_keeps_d_as_primary_trigger_without_ttft_or_reprefill_gate(self):
        signal_policy = InitialDSignalPolicy(
            min_revisit_opportunity_tokens=1_000,
            min_confirmed_loss_tokens=100,
            min_severity=0.05,
            min_volume_loss_tokens_per_s=10.0,
        )
        policy = ImpactSizedDReallocationPolicy(
            grow_quantum_pages=100,
            shrink_quantum_pages=40,
            min_grow_pages=5_000,
            min_shrink_pages=5_000,
            grow_effective_ratio=0.025,
            shrink_effective_ratio=0.025,
            cold_revisit_opportunity_tokens=5_000,
            cold_max_confirmed_loss_tokens=0,
        )
        observations = [
            observation(
                "hot",
                effective=200_000,
                floor=100_000,
                d_raw=8_000,
                opportunity=20_000,
                duration=10.0,
                reprefill=0,
            ),
            observation("donor", effective=200_000, floor=100_000, opportunity=50_000, clean_offer=80_000),
        ]
        signals = [signal_policy.evaluate(item) for item in observations]

        candidates = policy.candidates(observations, signals)

        self.assertEqual(
            [(item.model_id, item.action, item.pages, item.reason) for item in candidates],
            [("hot", "grow", 20_000, "impact_sized_grow_x4_unit5000")],
        )

    def test_manual_intent_selector_applies_only_one_shadow_chosen_trial(self):
        plan = ResourcePlan(
            targets={"hot": 1_100, "cold": 920, "idle": 960},
            transactions=[
                PlannedTransaction(None, "hot", 20, "central_free"),
                PlannedTransaction("cold", "hot", 80, "clean"),
                PlannedTransaction("idle", None, 40, "clean"),
            ],
            unmet_pages={},
        )
        intents = manual_intents_from_resource_plan(plan, intent_id_prefix="trial")

        self.assertEqual(
            select_manual_intents_for_trial(intents, op="transfer", limit=1),
            [intents[1]],
        )
        self.assertEqual(
            select_manual_intents_for_trial(intents, op="shrink", limit=1),
            [intents[2]],
        )
        self.assertEqual(
            select_manual_intents_for_trial(
                intents,
                op="transfer",
                donor="idle",
                recipient="hot",
                limit=1,
            ),
            [],
        )

    def test_outcome_ledger_records_shadow_and_apply_without_effective_before_ack(self):
        before = observation("hot", d_raw=500, opportunity=2_000)
        signal_policy = InitialDSignalPolicy(
            min_revisit_opportunity_tokens=1_000,
            min_confirmed_loss_tokens=100,
            min_severity=0.10,
            min_volume_loss_tokens_per_s=1.0,
        )
        signal = signal_policy.evaluate(before)
        txn = PlannedTransaction(
            source_model_id=None,
            destination_model_id="hot",
            pages=100,
            readiness="central_free",
        )
        ledger = OutcomeLedger(clock_ns=lambda: 10)

        shadow = ledger.start_trial(
            trial_id="shadow-1",
            mode="shadow",
            observations={"hot": before},
            signals={"hot": signal},
            transactions=[txn],
        )
        applied = ledger.start_trial(
            trial_id="apply-1",
            mode="apply",
            observations={"hot": before},
            signals={"hot": signal},
            transactions=[txn],
        )
        before_ack = ledger.record_intent_submitted(
            "apply-1", intent_id="intent-1", timestamp_ns=20
        )
        ledger.record_allocator_ack(
            "apply-1",
            effective_pages={"hot": 1_100},
            timestamp_ns=30,
        )
        ledger.record_post_window(
            "apply-1",
            observations={"hot": observation("hot", d_raw=200, opportunity=2_000)},
            timestamp_ns=40,
            t_usable_ns=35,
        )

        self.assertEqual(shadow["mode"], "shadow")
        self.assertIsNone(shadow["intent_id"])
        self.assertIsNone(applied["effective_after_ack"])
        self.assertIsNone(before_ack["effective_after_ack"])
        self.assertEqual(ledger.records["apply-1"]["effective_after_ack"], {"hot": 1_100})
        self.assertEqual(ledger.records["apply-1"]["t_usable_ns"], 35)
        self.assertIn("post_window_observations", ledger.records["apply-1"])

    def test_outcome_ledger_keeps_revisit_regret_explanation_fields(self):
        before = observation(
            "hot",
            d_raw=2_000,
            raw_retention_loss=3_000,
            opportunity=5_000,
            revisit_cache_miss=3_000,
            revisit_incremental=1_000,
            revisit_seen_cache_miss=2_000,
            d_coverage=1.0,
        )
        after = observation(
            "hot",
            d_raw=500,
            opportunity=5_000,
            revisit_cache_miss=1_200,
            revisit_incremental=800,
            revisit_seen_cache_miss=400,
            d_coverage=1.25,
        )
        signal_policy = InitialDSignalPolicy(
            min_revisit_opportunity_tokens=1_000,
            min_confirmed_loss_tokens=100,
            min_severity=0.05,
            min_volume_loss_tokens_per_s=1.0,
        )
        ledger = OutcomeLedger(clock_ns=lambda: 10)

        record = ledger.start_trial(
            trial_id="shadow-regret-fields",
            mode="shadow",
            observations={"hot": before},
            signals={"hot": signal_policy.evaluate(before)},
            transactions=[],
        )
        updated = ledger.record_post_window(
            "shadow-regret-fields",
            observations={"hot": after},
            timestamp_ns=20,
        )

        self.assertEqual(
            record["pre_observations"]["hot"]["raw_retention_loss_tokens"],
            3_000,
        )
        self.assertEqual(
            record["pre_observations"]["hot"]["revisit_seen_cache_miss_tokens"],
            2_000,
        )
        self.assertEqual(
            record["pre_observations"]["hot"]["d_coverage_of_seen_miss"],
            1.0,
        )
        self.assertEqual(
            updated["post_window_observations"]["hot"]["revisit_cache_miss_tokens"],
            1_200,
        )
        self.assertEqual(
            updated["post_window_observations"]["hot"]["d_coverage_of_seen_miss"],
            1.25,
        )

    def test_outcome_ledger_records_kxq_supply_unmet_and_donor_contributions(self):
        hot = observation("hot", d_raw=8_000, opportunity=20_000)
        donor_a = observation("donor-a", clean_offer=120)
        donor_b = observation("donor-b", ready_offer=90)
        signal_policy = InitialDSignalPolicy(
            min_revisit_opportunity_tokens=1_000,
            min_confirmed_loss_tokens=100,
            min_severity=0.05,
            min_volume_loss_tokens_per_s=10.0,
        )
        observations = [hot, donor_a, donor_b]
        signals = [signal_policy.evaluate(item) for item in observations]
        transactions = [
            PlannedTransaction(None, "hot", 50, "central_free"),
            PlannedTransaction("donor-a", "hot", 120, "clean"),
            PlannedTransaction("donor-b", "hot", 70, "ready"),
        ]
        ledger = OutcomeLedger(clock_ns=lambda: 10)

        record = ledger.start_trial(
            trial_id="shadow-kxq",
            mode="shadow",
            observations={item.model_id: item for item in observations},
            signals={item.model_id: signal for item, signal in zip(observations, signals)},
            transactions=transactions,
            unmet_pages={"hot": 80},
        )

        self.assertEqual(record["quota_outcome"]["requested_pages"], {"hot": 320})
        self.assertEqual(record["quota_outcome"]["supplied_pages"], {"hot": 240})
        self.assertEqual(record["quota_outcome"]["unmet_pages"], {"hot": 80})
        self.assertEqual(
            record["quota_outcome"]["donor_contributions"],
            {
                "central_free": {"hot": 50},
                "donor-a": {"hot": 120},
                "donor-b": {"hot": 70},
            },
        )


if __name__ == "__main__":
    unittest.main()
