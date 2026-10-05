"""Independent formula checks and negative controls for the finite models."""

import unittest
from fractions import Fraction as F
from pathlib import Path

from verify_mechanism_counterexamples import (
    Request,
    enumerate_schedules,
    relative_static_error_floor,
    source_policy_probe,
)


class MechanismTests(unittest.TestCase):
    def test_equal_token_snapshot_and_negative_controls(self):
        packed, split = [4], [2, 2]
        self.assertEqual(sum(packed), sum(split))
        self.assertLessEqual(len(packed), 1)
        self.assertGreater(len(split), 1)
        self.assertLessEqual(len(split), 2)
        for state_bytes in (0, 3, 54):
            self.assertEqual(
                state_bytes * len(split) + sum(split)
                - (state_bytes * len(packed) + sum(packed)),
                state_bytes,
            )

    def test_peak_state_does_not_imply_state_time(self):
        for factor in (2, 4, 16):
            for q in (1, 3):
                split = tuple(Request(str(i), 1, q) for i in range(factor))
                packed = (Request("packed", factor, factor * q),)
                constant = lambda batch: F(1)
                s = enumerate_schedules(split, state_cap=factor, kv_cap=4096, cost=constant)[0]
                p = enumerate_schedules(packed, state_cap=factor, kv_cap=4096, cost=constant)[0]
                self.assertEqual(sum(s.completion), sum(p.completion))
                self.assertEqual(s.makespan * factor, p.makespan)

    def test_equal_token_state_wall_delays_neighbor(self):
        for q in (1, 2, 4):
            split = (Request("A1", 1, q), Request("A2", 1, q), Request("B", 1, 1))
            packed = (Request("A", 2, 2 * q), Request("B", 1, 1))
            common = {"kv_cap": 64, "cost": lambda batch: F(1)}
            s = enumerate_schedules(split, state_cap=2, initial=("A1", "A2"), **common)
            p = enumerate_schedules(packed, state_cap=2, initial=("A",), **common)
            self.assertEqual(len(s), 1)
            self.assertEqual(s[0].completion[-1], q + 1)
            self.assertEqual(p[0].completion[-1], 1)
            relaxed = enumerate_schedules(split, state_cap=3, initial=("A1", "A2"), **common)
            self.assertEqual(relaxed[0].completion[-1], 1)
            # Even deliberate deferral cannot admit B before a running A finishes.
            all_s = enumerate_schedules(
                split, state_cap=2, initial=("A1", "A2"),
                work_conserving=False, **common,
            )
            self.assertEqual(min(o.completion[-1] for o in all_s), q + 1)

    def test_two_request_all_schedules_and_closed_form(self):
        jobs = (Request("A", 0, 1), Request("B", 0, 1))
        for ratio in (F(1, 2), F(1), F(3, 2), F(2), F(5, 2), F(4)):
            schedules = enumerate_schedules(
                jobs, state_cap=2, kv_cap=16,
                cost=lambda b, r=ratio: F(1) if len(b) == 1 else r,
                work_conserving=False,
            )
            self.assertEqual(len(schedules), 3)
            self.assertEqual(sorted(o.makespan for o in schedules), sorted((ratio, F(2), F(2))))
            self.assertEqual(min(o.makespan for o in schedules), min(ratio, F(2)))
            parallel = next(o for o in schedules if len(o.admissions[0]) == 2)
            serial = next(o for o in schedules if o.admissions[0] == ("A",))
            self.assertEqual(parallel.service_gap("A", "B"), 0)
            self.assertEqual(serial.service_gap("A", "B"), 1)
            zero_gap = [o for o in schedules if o.service_gap("A", "B") == 0]
            self.assertEqual(zero_gap, [parallel])
            strict_work = enumerate_schedules(
                jobs, state_cap=2, kv_cap=16,
                cost=lambda b, r=ratio: F(1) if len(b) == 1 else r,
            )
            self.assertEqual(len(strict_work), 1)
            self.assertEqual(strict_work[0].makespan, ratio)

    def test_composition_flip_formulas_over_context_sweep(self):
        for h in (1, 2, 8, 9, 10, 20, 100):
            jobs = (Request("X", 8, 1), Request("Y", 1, 2), Request("Z", h, 2))
            schedules = enumerate_schedules(jobs, state_cap=2, kv_cap=512, initial=("Z",))
            self.assertEqual(len(schedules), 2)
            x = next(o for o in schedules if o.admissions[0] == ("X",))
            y = next(o for o in schedules if o.admissions[0] == ("Y",))
            self.assertEqual(x.completion, (h + 9, 2 * h + 15, 2 * h + 12))
            self.assertEqual(y.completion, (2 * h + 15, 2 * h + 6, 2 * h + 6))
            self.assertEqual(x.mean_completion - y.mean_completion, F(9 - h, 3))
            self.assertEqual(x.makespan, y.makespan)
            self.assertEqual(x.service_gap("X", "Y"), 9)
            self.assertEqual(y.service_gap("X", "Y"), 6)
            for o in schedules:
                self.assertEqual(o.events[-1].served, (9, 3, h + 2))
                for e in o.events:
                    self.assertLessEqual(len(e.running), 2)

    def test_gains_are_not_free_fairness_or_worst_latency(self):
        jobs = (Request("X", 8, 1), Request("Y", 1, 2), Request("Z", 100, 2))
        schedules = enumerate_schedules(jobs, state_cap=2, kv_cap=512, initial=("Z",))
        x = next(o for o in schedules if o.admissions[0] == ("X",))
        y = next(o for o in schedules if o.admissions[0] == ("Y",))
        self.assertEqual(1 - x.mean_completion / y.mean_completion, F(91, 627))
        self.assertEqual(x.makespan, y.makespan)
        tight = [o for o in schedules if o.service_gap("X", "Y") <= 6]
        self.assertEqual(tight, [y])
        isolated = (F(9), F(5), F(203))
        x_worst = max(t / solo for t, solo in zip(x.completion, isolated))
        y_worst = max(t / solo for t, solo in zip(y.completion, isolated))
        self.assertGreater(x_worst, y_worst)

    def test_no_choice_when_all_candidates_fit(self):
        for h in (1, 100):
            jobs = (Request("X", 8, 1), Request("Y", 1, 2), Request("Z", h, 2))
            schedules = enumerate_schedules(jobs, state_cap=3, kv_cap=512, initial=("Z",))
            self.assertEqual(len(schedules), 1)
            self.assertEqual(schedules[0].admissions[0], ("X", "Y"))

    def test_nonzero_prefill_changes_crossover_not_always_gain(self):
        for h in (1, 9, 100):
            for coefficient in (F(0), F(1), F(7)):
                jobs = (Request("X", 8, 1), Request("Y", 1, 2), Request("Z", h, 2))
                schedules = enumerate_schedules(
                    jobs, state_cap=2, kv_cap=512, initial=("Z",),
                    prefill_per_token=coefficient,
                )
                x = next(o for o in schedules if o.admissions[0] == ("X",))
                y = next(o for o in schedules if o.admissions[0] == ("Y",))
                self.assertEqual(x.mean_completion, (5 * h + 36 + 26 * coefficient) / 3)
                self.assertEqual(y.mean_completion, (6 * h + 27 + 11 * coefficient) / 3)
                self.assertEqual(x.makespan, 2 * h + 15 + 9 * coefficient)
                self.assertEqual(x.makespan, y.makespan)

    def test_clock_scaling_changes_latency_not_service_trace(self):
        jobs = (Request("X", 8, 1), Request("Y", 1, 2), Request("Z", 10, 2))
        original = enumerate_schedules(jobs, state_cap=2, kv_cap=512, initial=("Z",))
        for multiplier in (2, 7):
            scaled = enumerate_schedules(
                jobs, state_cap=2, kv_cap=512, initial=("Z",),
                cost=lambda b, m=multiplier: F(m * (1 + sum(r.prompt + g for r, g in b))),
            )
            for base, changed in zip(original, scaled):
                self.assertEqual(base.admissions, changed.admissions)
                self.assertEqual(base.service_gap("X", "Y"), changed.service_gap("X", "Y"))
                self.assertEqual(changed.mean_completion, multiplier * base.mean_completion)
                self.assertEqual(changed.makespan, multiplier * base.makespan)

    def test_profile_error_lower_bound(self):
        for low, high in ((9, 109), (1, 1), (2, 8)):
            optimum = F(2 * low * high, low + high)
            floor = relative_static_error_floor(low, high)
            self.assertEqual(abs(optimum - low) / low, floor)
            self.assertEqual(abs(optimum - high) / high, floor)
            for estimate in range(high + 1):
                error = max(F(abs(estimate - low), low), F(abs(estimate - high), high))
                self.assertGreaterEqual(error, floor)

    def test_unmodified_source_queue_method(self):
        probe = source_policy_probe(Path(__file__).resolve().parents[3])
        for row in probe["equal_counter_queue_Y_X"]:
            self.assertEqual(row["order"], ["Y", "X"])

    def test_capacity_and_invalid_cost_checks(self):
        with self.assertRaises(ValueError):
            Request("bad", 1, 0)
        with self.assertRaises(ValueError):
            enumerate_schedules((Request("A", 8, 1),), state_cap=1, kv_cap=8, initial=("A",))
        with self.assertRaises(ValueError):
            enumerate_schedules((Request("A", 0, 1),), state_cap=1, kv_cap=8, cost=lambda _: 0)
        self.assertEqual(enumerate_schedules((Request("A", 8, 1),), state_cap=1, kv_cap=8), [])


if __name__ == "__main__":
    unittest.main()
