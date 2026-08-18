"""Run explicit quota intents through the policy-free GlobalQuotaManager."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Protocol

try:
    from sglang.srt.mem_cache.global_quota_manager import GlobalQuotaManager
except ModuleNotFoundError:
    from global_quota_manager import GlobalQuotaManager


_POLICY_FIELDS = {
    "D",
    "TTFT",
    "RPS",
    "cache_ratio",
    "profile",
    "scheduler",
    "threshold",
}


class ManualQuotaBackend(Protocol):
    def snapshot(self) -> dict[str, Any]: ...

    def grow_from_reserve(self, model_id: str, pages: int) -> dict[str, Any]: ...

    def shrink_to_reserve(self, model_id: str, pages: int) -> dict[str, Any]: ...

    def transfer_pages(
        self, donor: str, recipient: str, pages: int, offer_id: str
    ) -> dict[str, Any]: ...

    def begin_grow_from_reserve(self, model_id: str, pages: int) -> dict[str, Any]: ...

    def begin_transfer_pages(
        self, donor: str, recipient: str, pages: int, offer_id: str
    ) -> dict[str, Any]: ...

    def wait_grow_boundary(
        self, model_id: str, target_effective_pages: int
    ) -> dict[str, Any]: ...

    def wait_transfer_boundary(
        self,
        donor: str,
        donor_effective_pages: int,
        recipient: str,
        recipient_effective_pages: int,
    ) -> dict[str, Any]: ...

    def donation_offer(self, model_id: str) -> dict[str, Any]: ...


class BackupObserver(Protocol):
    def first_actual_backup(self, model_id: str, pages: int) -> dict[str, Any] | None: ...


class PressureLogBackupObserver:
    """Wait for the first new pressure-log backup record after construction."""

    def __init__(
        self,
        pressure_logs: dict[str, str | Path],
        *,
        wait_timeout_s: float = 300.0,
        wait_interval_s: float = 0.5,
        clock=time.monotonic,
        sleep=time.sleep,
    ) -> None:
        self.pressure_logs = {
            model_id: Path(path) for model_id, path in pressure_logs.items()
        }
        self.wait_timeout_s = wait_timeout_s
        self.wait_interval_s = wait_interval_s
        self.clock = clock
        self.sleep = sleep
        self._offsets = {
            model_id: path.stat().st_size if path.exists() else 0
            for model_id, path in self.pressure_logs.items()
        }

    def first_actual_backup(self, model_id: str, pages: int) -> dict[str, Any] | None:
        if model_id not in self.pressure_logs:
            return None
        deadline = self.clock() + self.wait_timeout_s
        path = self.pressure_logs[model_id]
        while True:
            result = self._read_new_backup(model_id, path)
            if result is not None:
                return result
            if self.clock() >= deadline:
                return None
            if self.wait_interval_s:
                self.sleep(self.wait_interval_s)

    def _read_new_backup(self, model_id: str, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        offset = self._offsets.get(model_id, 0)
        with path.open() as handle:
            handle.seek(offset)
            while True:
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                record = json.loads(line)
                backup_slots = int(record.get("backup_slots", 0))
                if backup_slots <= 0:
                    continue
                page_size = int(record.get("page_size_tokens", 1))
                self._offsets[model_id] = handle.tell()
                return {
                    "backup_slots": backup_slots,
                    "backup_pages": (backup_slots + page_size - 1) // page_size,
                }
            self._offsets[model_id] = handle.tell()
        return None


class ManualQuotaIntentRunner:
    """Apply manual/scripted quota intents without making quota policy decisions."""

    def __init__(
        self,
        backend: ManualQuotaBackend,
        *,
        backup_observer: BackupObserver | None = None,
        audit_path: str | Path | None = None,
        wait_for_backup_observation: bool = True,
        clock_ns=None,
    ) -> None:
        self.backend = backend
        self.backup_observer = backup_observer
        self.wait_for_backup_observation = wait_for_backup_observation
        snapshot = backend.snapshot()
        self._clock_ns = clock_ns
        manager = GlobalQuotaManager(
            global_pool_pages=int(snapshot["global_pool_pages"]),
            audit_path=audit_path,
            clock_ns=clock_ns,
        )
        for model_id, state in snapshot["models"].items():
            manager.register_instance(
                model_id,
                floor_pages=int(state["floor_pages"]),
                effective_pages=int(state["effective_pages"]),
            )
        self.manager = manager

    def run_jsonl(
        self,
        intent_path: str | Path,
        *,
        result_path: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        path = Path(intent_path)
        for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            intent = json.loads(line)
            records.extend(self.apply_external_intent(intent, line_number=line_number))
        if result_path is not None:
            output = Path(result_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w") as handle:
                for record in records:
                    handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        return records

    def apply_external_intent(
        self, intent: dict[str, Any], *, line_number: int | None = None
    ) -> list[dict[str, Any]]:
        self._reject_policy_context(intent)
        external_id = str(intent.get("intent_id") or intent.get("id") or "")
        if not external_id:
            raise ValueError("manual quota intent requires intent_id")
        action = str(intent.get("op") or intent.get("action") or "")
        pages = int(intent["pages"])
        tranche_pages = int(intent.get("tranche_pages", pages))
        if pages <= 0 or tranche_pages <= 0:
            raise ValueError("manual quota intent pages and tranche_pages must be positive")
        metadata = self._live_transaction_metadata(intent, line_number=line_number)
        if bool(intent.get("async_overlap")):
            self._validate_required_donor_offer(intent)
            return self._apply_async_overlap_intent(
                action=action,
                external_id=external_id,
                pages=pages,
                tranche_pages=tranche_pages,
                initial_tranche_pages=int(intent.get("initial_tranche_pages", tranche_pages)),
                donor=intent.get("donor"),
                recipient=intent.get("recipient"),
                offer_id=intent.get("offer_id"),
                line_number=line_number,
                metadata=metadata,
            )

        records = []
        self._validate_required_donor_offer(intent)
        remaining = pages
        tranche_index = 0
        while remaining:
            tranche_index += 1
            current_pages = min(tranche_pages, remaining)
            records.append(
                self._apply_tranche(
                    action=action,
                    external_id=external_id,
                    tranche_index=tranche_index,
                    pages=current_pages,
                    donor=intent.get("donor"),
                    recipient=intent.get("recipient"),
                    offer_id=intent.get("offer_id"),
                    line_number=line_number,
                    metadata=metadata,
                )
            )
            remaining -= current_pages
        return records

    def _apply_async_overlap_intent(
        self,
        *,
        action: str,
        external_id: str,
        pages: int,
        tranche_pages: int,
        initial_tranche_pages: int,
        donor: str | None,
        recipient: str | None,
        offer_id: str | None,
        line_number: int | None,
        metadata: dict[str, str],
    ) -> list[dict[str, Any]]:
        if initial_tranche_pages <= 0:
            raise ValueError("initial_tranche_pages must be positive")
        intent_start_ns = self._now_ns()
        if action == "grow":
            if recipient is None:
                raise ValueError("grow intent requires recipient")
            async_plan = self.backend.begin_grow_from_reserve(recipient, pages)
            return self._ack_async_grow_boundaries(
                external_id=external_id,
                line_number=line_number,
                recipient=recipient,
                pages=pages,
                tranche_pages=tranche_pages,
                initial_tranche_pages=initial_tranche_pages,
                async_plan=async_plan,
                intent_start_ns=intent_start_ns,
                metadata=metadata,
            )
        if action == "transfer":
            if donor is None or recipient is None:
                raise ValueError("transfer intent requires donor and recipient")
            if not offer_id:
                raise ValueError("transfer intent requires explicit offer_id")
            async_plan = self.backend.begin_transfer_pages(
                donor, recipient, pages, offer_id
            )
            return self._ack_async_transfer_boundaries(
                external_id=external_id,
                line_number=line_number,
                donor=donor,
                recipient=recipient,
                pages=pages,
                tranche_pages=tranche_pages,
                initial_tranche_pages=initial_tranche_pages,
                offer_id=offer_id,
                async_plan=async_plan,
                intent_start_ns=intent_start_ns,
                metadata=metadata,
            )
        if action == "shrink":
            raise ValueError("async_overlap currently supports grow and transfer")
        raise ValueError("manual quota intent op must be grow, shrink, or transfer")

    def _ack_async_grow_boundaries(
        self,
        *,
        external_id: str,
        line_number: int | None,
        recipient: str,
        pages: int,
        tranche_pages: int,
        initial_tranche_pages: int,
        async_plan: dict[str, Any],
        intent_start_ns: int,
        metadata: dict[str, str],
    ) -> list[dict[str, Any]]:
        start_effective = int(async_plan["start_effective_pages"])
        records: list[dict[str, Any]] = []
        cumulative = 0
        for tranche_index, current_pages in enumerate(
            self._tranche_sizes(pages, tranche_pages, initial_tranche_pages),
            start=1,
        ):
            cumulative += current_pages
            boundary = start_effective + cumulative
            backend_result = self.backend.wait_grow_boundary(recipient, boundary)
            quota_intent = self.manager.submit_intent(
                "grow", recipient=recipient, pages=current_pages
            )
            ack = self.manager.ack_recipient_allocator(quota_intent.intent_id)
            ack_done_ns = self._now_ns()
            backup = self._record_first_backup(recipient, current_pages)
            self.manager.assert_conserved()
            records.append(
                {
                    "external_intent_id": external_id,
                    "line_number": line_number,
                    "tranche_index": tranche_index,
                    "manager_intent_id": quota_intent.intent_id,
                    "manager_action": "grow",
                    "pages": current_pages,
                    "initial_tranche": tranche_index == 1,
                    "async_overlap": True,
                    "async_intent_done": cumulative == pages,
                    "async_plan": async_plan,
                    "backend": backend_result,
                    "ack": ack,
                    "tranche_target_effective_pages": boundary,
                    "t_ack_ns": ack_done_ns - intent_start_ns,
                    "first_actual_backup": backup,
                    **metadata,
                }
            )
        return records

    def _ack_async_transfer_boundaries(
        self,
        *,
        external_id: str,
        line_number: int | None,
        donor: str,
        recipient: str,
        pages: int,
        tranche_pages: int,
        initial_tranche_pages: int,
        offer_id: str,
        async_plan: dict[str, Any],
        intent_start_ns: int,
        metadata: dict[str, str],
    ) -> list[dict[str, Any]]:
        donor_start = int(async_plan["donor_start_effective_pages"])
        recipient_start = int(async_plan["recipient_start_effective_pages"])
        records: list[dict[str, Any]] = []
        cumulative = 0
        for tranche_index, current_pages in enumerate(
            self._tranche_sizes(pages, tranche_pages, initial_tranche_pages),
            start=1,
        ):
            cumulative += current_pages
            donor_boundary = donor_start - cumulative
            recipient_boundary = recipient_start + cumulative
            backend_result = self.backend.wait_transfer_boundary(
                donor, donor_boundary, recipient, recipient_boundary
            )
            quota_intent = self.manager.submit_intent(
                "transfer",
                donor=donor,
                recipient=recipient,
                pages=current_pages,
                offer_id=offer_id,
            )
            self.manager.mark_donor_detached(quota_intent.intent_id)
            self.manager.mark_scrub_complete(quota_intent.intent_id)
            ack = self.manager.ack_recipient_allocator(quota_intent.intent_id)
            ack_done_ns = self._now_ns()
            backup = self._record_first_backup(recipient, current_pages)
            self.manager.assert_conserved()
            records.append(
                {
                    "external_intent_id": external_id,
                    "line_number": line_number,
                    "tranche_index": tranche_index,
                    "manager_intent_id": quota_intent.intent_id,
                    "manager_action": "transfer",
                    "pages": current_pages,
                    "initial_tranche": tranche_index == 1,
                    "async_overlap": True,
                    "async_intent_done": cumulative == pages,
                    "async_plan": async_plan,
                    "backend": backend_result,
                    "ack": ack,
                    "donor_target_effective_pages": donor_boundary,
                    "tranche_target_effective_pages": recipient_boundary,
                    "t_ack_ns": ack_done_ns - intent_start_ns,
                    "first_actual_backup": backup,
                    **metadata,
                }
            )
        return records

    @staticmethod
    def _tranche_sizes(
        pages: int, tranche_pages: int, initial_tranche_pages: int
    ) -> list[int]:
        sizes = []
        remaining = pages
        first = min(initial_tranche_pages, remaining)
        sizes.append(first)
        remaining -= first
        while remaining:
            current = min(tranche_pages, remaining)
            sizes.append(current)
            remaining -= current
        return sizes

    def _apply_tranche(
        self,
        *,
        action: str,
        external_id: str,
        tranche_index: int,
        pages: int,
        donor: str | None,
        recipient: str | None,
        offer_id: str | None,
        line_number: int | None,
        metadata: dict[str, str],
    ) -> dict[str, Any]:
        tranche_start_ns = self._now_ns()
        if action == "grow":
            if recipient is None:
                raise ValueError("grow intent requires recipient")
            quota_intent = self.manager.submit_intent(
                "grow", recipient=recipient, pages=pages
            )
            backend_result = self.backend.grow_from_reserve(recipient, pages)
            ack = self.manager.ack_recipient_allocator(quota_intent.intent_id)
            ack_done_ns = self._now_ns()
            backup = self._record_first_backup(recipient, pages)
        elif action == "shrink":
            if donor is None:
                raise ValueError("shrink intent requires donor")
            quota_intent = self.manager.submit_intent(
                "shrink", donor=donor, pages=pages
            )
            backend_result = self.backend.shrink_to_reserve(donor, pages)
            self.manager.mark_donor_detached(quota_intent.intent_id)
            self.manager.mark_scrub_complete(quota_intent.intent_id)
            ack = {"effective_pages": self.manager.effective_pages(donor)}
            ack_done_ns = self._now_ns()
            backup = None
        elif action == "transfer":
            if donor is None or recipient is None:
                raise ValueError("transfer intent requires donor and recipient")
            if not offer_id:
                raise ValueError("transfer intent requires explicit offer_id")
            quota_intent = self.manager.submit_intent(
                "transfer",
                donor=donor,
                recipient=recipient,
                pages=pages,
                offer_id=offer_id,
            )
            backend_result = self.backend.transfer_pages(donor, recipient, pages, offer_id)
            self.manager.mark_donor_detached(quota_intent.intent_id)
            self.manager.mark_scrub_complete(quota_intent.intent_id)
            ack = self.manager.ack_recipient_allocator(quota_intent.intent_id)
            ack_done_ns = self._now_ns()
            backup = self._record_first_backup(recipient, pages)
        else:
            raise ValueError("manual quota intent op must be grow, shrink, or transfer")
        self.manager.assert_conserved()
        return {
            "external_intent_id": external_id,
            "line_number": line_number,
            "tranche_index": tranche_index,
            "manager_intent_id": quota_intent.intent_id,
            "manager_action": action,
            "pages": pages,
            "backend": backend_result,
            "ack": ack,
            "t_ack_ns": ack_done_ns - tranche_start_ns,
            "first_actual_backup": backup,
            **metadata,
        }

    def _record_first_backup(self, model_id: str, pages: int) -> dict[str, Any] | None:
        if not self.wait_for_backup_observation:
            return None
        if self.backup_observer is None:
            return None
        observed = self.backup_observer.first_actual_backup(model_id, pages)
        if observed is None:
            return None
        usable = self.manager.record_actual_backup(model_id, pages=pages)
        return {"observer": observed, "manager": usable}

    def _validate_required_donor_offer(self, intent: dict[str, Any]) -> None:
        required_raw = intent.get("require_donor_offer_pages")
        if required_raw is None:
            return
        required = int(required_raw)
        if required < 0:
            raise ValueError("require_donor_offer_pages must be non-negative")
        donor = intent.get("donor")
        if not donor:
            raise ValueError("require_donor_offer_pages requires donor")
        offer = self.backend.donation_offer(str(donor))
        total = int(offer.get("total_pages", 0))
        if total < required:
            raise ValueError(
                "manual transfer donor offer below explicit requirement: "
                f"donor={donor} required={required} offered={total} offer={offer}"
            )

    def _now_ns(self) -> int:
        if self._clock_ns is not None:
            return int(self._clock_ns())
        return time.monotonic_ns()

    @staticmethod
    def _reject_policy_context(intent: dict[str, Any]) -> None:
        present = sorted(_POLICY_FIELDS.intersection(intent))
        if present:
            raise ValueError(
                "manual quota runner is policy-free; forbidden fields: "
                + ", ".join(present)
            )

    @staticmethod
    def _live_transaction_metadata(
        intent: dict[str, Any], *, line_number: int | None
    ) -> dict[str, str]:
        phase = str(intent.get("phase") or "live_transaction_window")
        if phase == "startup_correction":
            raise ValueError(
                "startup correction is a launch-time configuration change, "
                "not a live quota intent"
            )
        if phase != "live_transaction_window":
            raise ValueError("manual quota intent phase must be live_transaction_window")
        decision_window_id = str(
            intent.get("decision_window_id")
            or (f"line-{line_number}" if line_number is not None else "manual")
        )
        if not decision_window_id:
            raise ValueError("decision_window_id must be non-empty")
        return {
            "phase": phase,
            "decision_window_id": decision_window_id,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply explicit policy-free Global Quota Manager intents."
    )
    parser.add_argument("--intent-jsonl", required=True)
    parser.add_argument("--result-jsonl", required=True)
    parser.add_argument("--audit-jsonl", required=True)
    parser.add_argument("--central-socket", required=True)
    parser.add_argument(
        "--floor-pages-json",
        default="{}",
        help="JSON object mapping model_id to floor pages when residency is unavailable.",
    )
    parser.add_argument(
        "--pressure-log",
        action="append",
        default=[],
        metavar="MODEL_ID=PATH",
        help="Optional pressure log used to record first actual backup after ack.",
    )
    parser.add_argument("--wait-timeout-s", type=float, default=300.0)
    parser.add_argument("--wait-interval-s", type=float, default=0.5)
    parser.add_argument("--backup-wait-timeout-s", type=float, default=300.0)
    parser.add_argument(
        "--defer-backup-observation",
        action="store_true",
        help="Do not wait for first-backup evidence before dispatching the next tranche.",
    )
    args = parser.parse_args()
    from sglang.srt.mem_cache.central_io import CentralIOControlClient
    from sglang.srt.mem_cache.central_io_manual_quota_backend import (
        CentralIOTargetQuotaBackend,
    )

    floor_pages = json.loads(args.floor_pages_json)
    pressure_logs = {}
    for item in args.pressure_log:
        model_id, separator, path = item.partition("=")
        if not separator or not model_id or not path:
            raise SystemExit("--pressure-log must be MODEL_ID=PATH")
        pressure_logs[model_id] = path
    control = CentralIOControlClient(args.central_socket)
    try:
        backend = CentralIOTargetQuotaBackend(
            control,
            floor_pages={key: int(value) for key, value in floor_pages.items()},
            wait_timeout_s=args.wait_timeout_s,
            wait_interval_s=args.wait_interval_s,
        )
        observer = (
            PressureLogBackupObserver(
                pressure_logs,
                wait_timeout_s=args.backup_wait_timeout_s,
                wait_interval_s=args.wait_interval_s,
            )
            if pressure_logs
            else None
        )
        runner = ManualQuotaIntentRunner(
            backend,
            backup_observer=observer,
            audit_path=args.audit_jsonl,
            wait_for_backup_observation=not args.defer_backup_observation,
        )
        runner.run_jsonl(args.intent_jsonl, result_path=args.result_jsonl)
    finally:
        control.close()


if __name__ == "__main__":
    main()
