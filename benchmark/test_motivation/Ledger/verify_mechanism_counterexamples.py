#!/usr/bin/env python3
"""Exact finite counterexamples, not a GPU simulator or a scheduler benchmark."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from typing import Callable


@dataclass(frozen=True)
class Request:
    name: str
    prompt: int
    output: int

    def __post_init__(self):
        if self.prompt < 0 or self.output < 1:
            raise ValueError("Require nonnegative prompt and positive output")


@dataclass(frozen=True)
class Event:
    time: F
    kind: str
    running: tuple[str, ...]
    served: tuple[int, ...]


@dataclass(frozen=True)
class Outcome:
    names: tuple[str, ...]
    completion: tuple[F, ...]
    admissions: tuple[tuple[str, ...], ...]
    events: tuple[Event, ...]

    @property
    def makespan(self):
        return max(self.completion)

    @property
    def mean_completion(self):
        return sum(self.completion, F(0)) / len(self.completion)

    def service_gap(self, left: str, right: str):
        a, b = self.names.index(left), self.names.index(right)
        return max(abs(e.served[a] - e.served[b]) for e in self.events)


StepCost = Callable[[tuple[tuple[Request, int], ...]], F]


def context_cost(batch):
    return F(1 + sum(req.prompt + generated for req, generated in batch))


def enumerate_schedules(
    requests: tuple[Request, ...],
    *,
    state_cap: int,
    kv_cap: int,
    cost: StepCost = context_cost,
    initial: tuple[str, ...] = (),
    work_conserving: bool = True,
    prefill_per_token: F = F(0),
):
    """Enumerate admissions at decode boundaries; admitted jobs cannot pause.

    Every request arrives at zero. Prefill input is charged at admission;
    optional linear prefill time stalls decoding. Initial prefill is sunk.
    KV reserves P+O, so growth always fits.
    Work conservation here means inclusion-maximal feasible admission, not
    SGLang's break-on-first-failure scan. Examples avoid that distinction.
    """
    names = tuple(r.name for r in requests)
    if len(set(names)) != len(names) or not set(initial) <= set(names):
        raise ValueError("Unique request names and a valid initial subset required")
    if state_cap < 1 or kv_cap < 1:
        raise ValueError("Capacities must be positive")
    if prefill_per_token < 0:
        raise ValueError("Prefill duration cannot be negative")
    initial_ids = tuple(names.index(n) for n in initial)

    def fits(ids):
        return len(ids) <= state_cap and sum(
            requests[i].prompt + requests[i].output for i in ids
        ) <= kv_cap

    if not fits(initial_ids):
        raise ValueError("Initial running set exceeds reserved capacity")
    served = tuple(r.prompt if i in initial_ids else 0 for i, r in enumerate(requests))
    first = Event(F(0), "initial", initial, served)
    outcomes = []

    def visit(now, waiting, active, generated, completion, served, admissions, events):
        if not waiting and not active:
            outcomes.append(Outcome(names, completion, admissions, events))
            return
        for size in range(min(len(waiting), state_cap - len(active)) + 1):
            for selected in combinations(waiting, size):
                running = tuple(sorted(active + selected))
                if not running or not fits(running):
                    continue
                pending = tuple(i for i in waiting if i not in selected)
                if work_conserving and any(fits(running + (i,)) for i in pending):
                    continue
                current_service = list(served)
                for i in selected:
                    current_service[i] += requests[i].prompt
                running_names = tuple(names[i] for i in running)
                admitted_names = tuple(names[i] for i in selected)
                admission_event = Event(now, "admit", running_names, tuple(current_service))
                duration = F(cost(tuple((requests[i], generated[i]) for i in running)))
                if duration <= 0:
                    raise ValueError("Decode iteration duration must be positive")
                prefill_time = prefill_per_token * sum(requests[i].prompt for i in selected)
                next_time = now + prefill_time + duration
                next_generated, finished = list(generated), list(completion)
                remaining = []
                for i in running:
                    next_generated[i] += 1
                    current_service[i] += 1
                    if next_generated[i] == requests[i].output:
                        finished[i] = next_time
                    else:
                        remaining.append(i)
                decode_event = Event(
                    next_time, "decode", running_names, tuple(current_service)
                )
                visit(
                    next_time, pending, tuple(remaining), tuple(next_generated),
                    tuple(finished), tuple(current_service), admissions + (admitted_names,),
                    events + (admission_event, decode_event),
                )

    visit(
        F(0), tuple(i for i in range(len(names)) if i not in initial_ids),
        initial_ids, (0,) * len(names), (F(0),) * len(names), served, (), (first,),
    )
    return outcomes


def source_policy_probe(repo_root: Path):
    """Execute selected unmodified methods via AST, without importing CUDA."""
    path = repo_root / "python/sglang/srt/managers/schedule_policy.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SchedulePolicy")
    method_names = {
        "_tenant", "_sort_by_tenant_vtc", "_vtc_admission_cost", "_vtc_output_cost"
    }
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in method_names]
    if len(methods) != len(method_names):
        raise ValueError("Source methods changed; inspect before updating this probe")
    module = ast.Module(
        body=ast.parse("from __future__ import annotations").body + methods,
        type_ignores=[],
    )
    namespace = {"defaultdict": defaultdict, "deque": deque}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    policy_type = type("ExtractedPolicy", (), {n: namespace[n] for n in method_names})
    rows = []
    for tax, linear, quadratic in ((0, 1, 0), (54, 1, 0), (1, 2, 3)):
        policy = policy_type()
        policy.tenant_counters = {"X": 0.0, "Y": 0.0}
        policy._vtc_input_charged_rids = set()
        policy.vtc_request_weight = tax
        policy.vtc_input_weight = linear
        policy.vtc_input_quadratic_weight = quadratic
        queue = [
            SimpleNamespace(rid=n, extra_key=n, origin_input_ids=[0] * p)
            for n, p in (("Y", 1), ("X", 8))
        ]
        policy._sort_by_tenant_vtc(queue)
        rows.append({"weights": [tax, linear, quadratic], "order": [r.rid for r in queue]})
    return {
        "path": str(path.relative_to(repo_root)),
        "sha256": hashlib.sha256(source.encode()).hexdigest(),
        "method_lines": {n.name: n.lineno for n in methods},
        "scope": "Actual queue/cost methods only; not the full SGLang runtime or VTC proof",
        "equal_counter_queue_Y_X": rows,
    }


def summarize(outcome):
    return {
        "completion": dict(zip(outcome.names, outcome.completion)),
        "mean_completion": outcome.mean_completion,
        "makespan": outcome.makespan,
        "admissions": outcome.admissions,
    }


def relative_static_error_floor(low, high):
    low, high = F(low), F(high)
    if not 0 < low <= high:
        raise ValueError("Require 0 < low <= high")
    return (high - low) / (high + low)


def build_report(repo_root):
    blocked_jobs = (Request("A1", 1, 4), Request("A2", 1, 4), Request("B", 1, 1))
    packed_jobs = (Request("A", 2, 8), Request("B", 1, 1))
    flat = lambda batch: F(1)
    blocked = enumerate_schedules(
        blocked_jobs, state_cap=2, kv_cap=64, initial=("A1", "A2"), cost=flat
    )[0]
    packed = enumerate_schedules(
        packed_jobs, state_cap=2, kv_cap=64, initial=("A",), cost=flat
    )[0]
    relaxed = enumerate_schedules(
        blocked_jobs, state_cap=3, kv_cap=64, initial=("A1", "A2"), cost=flat
    )[0]
    m2 = []
    pair = (Request("A", 0, 1), Request("B", 0, 1))
    for ratio in (F(1), F(3, 2), F(2), F(5, 2)):
        outcomes = enumerate_schedules(
            pair, state_cap=2, kv_cap=16,
            cost=lambda batch, r=ratio: F(1) if len(batch) == 1 else r,
            work_conserving=False,
        )
        parallel = next(o for o in outcomes if len(o.admissions[0]) == 2)
        serial = next(o for o in outcomes if o.admissions[0] == ("A",))
        for envelope in (0, 1):
            feasible = [o for o in outcomes if o.service_gap("A", "B") <= envelope]
            best = min(o.makespan for o in feasible)
            m2.append({
                "tau2_over_tau1": ratio, "service_gap_budget": envelope,
                "all_schedules": len(outcomes), "admissible_schedules": len(feasible),
                "parallel": summarize(parallel), "serial": summarize(serial),
                "best_worst_completion": best,
                "parallel_worst_completion_headroom": 1 - best / parallel.makespan,
            })
    m3 = []
    for h in (1, 9, 100):
        jobs = (Request("X", 8, 1), Request("Y", 1, 2), Request("Z", h, 2))
        outcomes = enumerate_schedules(jobs, state_cap=2, kv_cap=512, initial=("Z",))
        x_first = next(o for o in outcomes if o.admissions[0] == ("X",))
        y_first = next(o for o in outcomes if o.admissions[0] == ("Y",))
        selected = min(outcomes, key=lambda o: o.mean_completion)
        solo = (F(9), F(5), F(2 * h + 3))
        m3.append({
            "background_context": h,
            "scope": "Exactly two work-conserving nonpreemptive schedules in this model",
            "schedules": len(outcomes),
            "X_first": summarize(x_first), "Y_first": summarize(y_first),
            "X_first_XY_service_gap": x_first.service_gap("X", "Y"),
            "Y_first_XY_service_gap": y_first.service_gap("X", "Y"),
            "mean_completion_optimum_first": selected.admissions[0],
            "Y_first_mean_headroom": 1 - selected.mean_completion / y_first.mean_completion,
            "X_first_mean_headroom": 1 - selected.mean_completion / x_first.mean_completion,
            "worst_completion_headroom": F(0),
            "X_first_worst_slowdown_ratio": max(t / s for t, s in zip(x_first.completion, solo)),
            "Y_first_worst_slowdown_ratio": max(t / s for t, s in zip(y_first.completion, solo)),
            "mean_optimum_with_XY_gap_budget_6": min(
                o.mean_completion for o in outcomes if o.service_gap("X", "Y") <= 6
            ),
        })
    prefill_rows = []
    for coefficient in (F(0), F(1), F(7)):
        jobs = (Request("X", 8, 1), Request("Y", 1, 2), Request("Z", 100, 2))
        outcomes = enumerate_schedules(
            jobs, state_cap=2, kv_cap=512, initial=("Z",),
            prefill_per_token=coefficient,
        )
        x_first = next(o for o in outcomes if o.admissions[0] == ("X",))
        y_first = next(o for o in outcomes if o.admissions[0] == ("Y",))
        prefill_rows.append({
            "prefill_time_per_input_token": coefficient,
            "X_first": summarize(x_first), "Y_first": summarize(y_first),
            "Y_first_mean_headroom": 1 - min(
                o.mean_completion for o in outcomes
            ) / y_first.mean_completion,
        })
    return {
        "date": "2026-09-08",
        "status": "Exact model checks; synthetic time units; no GPU or empirical PASS",
        "source_probe": source_policy_probe(repo_root),
        "M1": {
            "resident_token_total": 4, "state_slot_capacity": 1,
            "packed_lengths": [4], "split_lengths": [2, 2],
            "packed_feasible": True, "split_feasible": False,
            "split_state_time_over_packed_with_constant_step_time": F(1),
            "interpretation": "Snapshot insufficiency does not imply a state-time multiplier",
            "admission_example": {
                "scope": "Same A aggregate P=2,O=8 but different requests; not equivalent application semantics",
                "split": summarize(blocked), "packed": summarize(packed),
                "split_with_state_cap_3": summarize(relaxed),
                "B_split_completion": blocked.completion[-1],
                "B_packed_completion": packed.completion[-1],
                "queue_only_headroom_after_full_state_commitment": F(0),
            },
        },
        "M2": m2,
        "M3": m3,
        "M3_prefill_sensitivity": prefill_rows,
        "static_execution_cost_floor": {
            "X_alone": F(9), "X_first_with_context_100_Z": F(109),
            "minimax_relative_error": relative_static_error_floor(9, 109),
            "scope": "A single constant predicting these two execution times, not token service",
        },
        "gain_limit": "M3 gains are mean residual completion, not p95 or worst slowdown; finite X/Y gap <=9 is not a steady-state fairness guarantee.",
    }


def exact_json(value):
    if isinstance(value, F):
        return {"exact": str(value), "decimal": float(value)}
    raise TypeError(type(value).__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    print(json.dumps(build_report(args.repo_root), default=exact_json, indent=2, allow_nan=False))
