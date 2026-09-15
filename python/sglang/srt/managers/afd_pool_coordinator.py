"""Shared Attention/FFN pool coordination protocol.

The in-memory coordinator is thread-safe and deliberately independent of CUDA.
A small ZeroMQ RPC facade makes the same protocol usable by multiple scheduler
processes. A PF failure fences every lease and dispatch for that PF epoch; a
later registration makes the PF eligible again at the incremented epoch.
"""

from __future__ import annotations

import argparse
import dataclasses
import threading
import time
from dataclasses import dataclass, field
from typing import Any

try:
    import zmq
except ImportError:  # The in-memory coordinator has no RPC dependency.
    zmq = None


class AFDCoordinatorError(RuntimeError):
    """Base class for structured coordinator protocol failures."""

    code = "COORDINATOR_ERROR"


class PFCapacityExhausted(AFDCoordinatorError):
    """The selected PF is healthy, but all of its lease slots are occupied."""

    code = "PF_CAPACITY_EXHAUSTED"


class PFCapacityWaitTimeout(TimeoutError, AFDCoordinatorError):
    """Waiting for a PF lease slot exceeded the configured bound."""

    code = "PF_CAPACITY_WAIT_TIMEOUT"


class AFDCoordinatorRemoteError(AFDCoordinatorError):
    """A non-retriable error returned by the coordinator."""

    def __init__(self, message: str, *, code: str, remote_type: str):
        super().__init__(message)
        self.code = code
        self.remote_type = remote_type


@dataclass(frozen=True)
class PFRegistration:
    pf_instance_id: str
    endpoint: str = ""
    capacity: int = 1
    epoch: int = 0
    registered_at: float = 0.0
    heartbeat_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Lease:
    lease_id: str
    rid: str
    pa_instance_id: str
    pf_instance_id: str
    pair_epoch: int
    cost: float = 1.0
    created_at: float = 0.0


@dataclass(frozen=True)
class Load:
    pf_instance_id: str
    pair_epoch: int
    inflight: int
    capacity: int
    inflight_cost: float


@dataclass(frozen=True)
class EdgeInfo:
    dispatch_id: str
    lease_id: str
    rid: str
    pa_instance_id: str
    pf_instance_id: str
    pair_epoch: int
    cost: float = 1.0
    started_at: float = 0.0
    # All request leases admitted by this authoritative forward dispatch.
    lease_ids: tuple[str, ...] = ()


class AFDPoolCoordinator:
    """Thread-safe PF registry, RID affinity table, and dispatch ledger."""

    def __init__(self):
        self._lock = threading.RLock()
        self._registrations: dict[str, PFRegistration] = {}
        self._epochs: dict[str, int] = {}
        self._leases_by_rid: dict[str, Lease] = {}
        self._leases_by_id: dict[str, Lease] = {}
        self._dispatches: dict[str, EdgeInfo] = {}
        self._completed_dispatches: dict[str, EdgeInfo] = {}
        # Capacity is owned by an admitted lease, not by each batch dispatch.
        # A lease may issue sequential or overlapping continuation dispatches.
        self._admitted_leases: set[str] = set()
        self._lease_seq = 0

    def register(
        self,
        pf_instance_id: str,
        endpoint: str = "",
        capacity: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> PFRegistration:
        if not pf_instance_id:
            raise ValueError("pf_instance_id must be non-empty")
        if isinstance(capacity, bool) or int(capacity) < 1:
            raise ValueError("capacity must be >= 1")
        now = time.time()
        with self._lock:
            epoch = self._epochs.setdefault(pf_instance_id, 0)
            old = self._registrations.get(pf_instance_id)
            registration = PFRegistration(
                pf_instance_id=pf_instance_id,
                endpoint=endpoint,
                capacity=int(capacity),
                epoch=epoch,
                registered_at=old.registered_at if old else now,
                heartbeat_at=now,
                metadata=dict(metadata or {}),
            )
            self._registrations[pf_instance_id] = registration
            return registration

    def unregister(self, pf_instance_id: str) -> int:
        return self.mark_failed(pf_instance_id)

    def heartbeat(self, pf_instance_id: str) -> PFRegistration:
        with self._lock:
            registration = self._registrations.get(pf_instance_id)
            if registration is None:
                raise KeyError(f"PF is not registered: {pf_instance_id}")
            registration = dataclasses.replace(registration, heartbeat_at=time.time())
            self._registrations[pf_instance_id] = registration
            return registration

    def _load_locked(self, pf_instance_id: str) -> Load:
        registration = self._registrations[pf_instance_id]
        active = [
            self._leases_by_id[lease_id]
            for lease_id in self._admitted_leases
            if lease_id in self._leases_by_id
            and self._leases_by_id[lease_id].pf_instance_id == pf_instance_id
            and self._leases_by_id[lease_id].pair_epoch == registration.epoch
        ]
        return Load(
            pf_instance_id=pf_instance_id,
            pair_epoch=registration.epoch,
            inflight=len(active),
            capacity=registration.capacity,
            inflight_cost=sum(lease.cost for lease in active),
        )

    def acquire(
        self,
        rid: str,
        pa_id: str,
        cost: float = 1.0,
        preferred_pf_instance_id: str | None = None,
    ) -> Lease:
        if not rid or not pa_id:
            raise ValueError("rid and pa_id must be non-empty")
        if float(cost) <= 0:
            raise ValueError("cost must be positive")
        with self._lock:
            existing = self._leases_by_rid.get(rid)
            if existing is not None and self._lease_valid_locked(existing):
                if existing.pa_instance_id != pa_id:
                    raise ValueError("RID lease belongs to another PA")
                return existing
            if existing is not None:
                self._drop_lease_locked(existing)
            if not self._registrations:
                raise RuntimeError("no healthy PF is registered")
            if preferred_pf_instance_id is not None:
                if preferred_pf_instance_id not in self._registrations:
                    raise KeyError(
                        f"preferred PF is not registered: {preferred_pf_instance_id}"
                    )
                loads = [self._load_locked(preferred_pf_instance_id)]
            else:
                loads = [self._load_locked(pf_id) for pf_id in self._registrations]
            chosen = min(
                loads,
                key=lambda load: (
                    load.inflight_cost,
                    load.inflight,
                    load.pf_instance_id,
                ),
            )
            self._lease_seq += 1
            lease = Lease(
                lease_id=f"{chosen.pair_epoch}:{self._lease_seq}",
                rid=rid,
                pa_instance_id=pa_id,
                pf_instance_id=chosen.pf_instance_id,
                pair_epoch=chosen.pair_epoch,
                cost=float(cost),
                created_at=time.time(),
            )
            self._leases_by_rid[rid] = lease
            self._leases_by_id[lease.lease_id] = lease
            return lease

    def lookup(self, rid: str) -> Lease | None:
        with self._lock:
            lease = self._leases_by_rid.get(rid)
            return (
                lease if lease is not None and self._lease_valid_locked(lease) else None
            )

    def begin_dispatch(
        self,
        dispatch_id: str,
        lease_id: str,
        pa_id: str | None = None,
        cost: float | None = None,
        lease_ids: list[str] | None = None,
    ) -> EdgeInfo:
        if not dispatch_id:
            raise ValueError("dispatch_id must be non-empty")
        if cost is not None and float(cost) <= 0:
            raise ValueError("cost must be positive")
        with self._lock:
            duplicate = self._dispatches.get(dispatch_id)
            if duplicate is not None:
                if duplicate.lease_id != lease_id:
                    raise ValueError(
                        "dispatch identity already belongs to another lease"
                    )
                return duplicate
            completed = self._completed_dispatches.get(dispatch_id)
            if completed is not None:
                if completed.lease_id != lease_id:
                    raise ValueError(
                        "dispatch identity already belongs to another lease"
                    )
                # A delayed/retried BEGIN remains idempotent even after COMPLETE.
                return completed
            requested_lease_ids = tuple(dict.fromkeys(lease_ids or [lease_id]))
            if lease_id not in requested_lease_ids:
                raise ValueError("primary lease must be present in lease_ids")
            leases = []
            for requested_id in requested_lease_ids:
                requested = self._leases_by_id.get(requested_id)
                if requested is None or not self._lease_valid_locked(requested):
                    raise ValueError("unknown or stale lease")
                if pa_id is not None and pa_id != requested.pa_instance_id:
                    raise ValueError("lease belongs to another PA")
                leases.append(requested)
            lease = self._leases_by_id[lease_id]
            if any(
                item.pf_instance_id != lease.pf_instance_id
                or item.pair_epoch != lease.pair_epoch
                or item.pa_instance_id != lease.pa_instance_id
                for item in leases
            ):
                raise ValueError("dispatch leases must share one PA/PF affinity")

            # A PF scheduler/data channel consumes exactly one authoritative
            # dispatch at a time. Capacity counts admitted request leases; it
            # does not permit concurrent dispatches from different PAs.
            if any(
                active.pf_instance_id == lease.pf_instance_id
                for active in self._dispatches.values()
            ):
                raise PFCapacityExhausted(
                    f"PF dispatch lane busy: {lease.pf_instance_id}"
                )
            new_lease_ids = [
                item.lease_id
                for item in leases
                if item.lease_id not in self._admitted_leases
            ]
            load = self._load_locked(lease.pf_instance_id)
            if load.inflight + len(new_lease_ids) > load.capacity:
                raise PFCapacityExhausted(
                    f"PF capacity exhausted: {lease.pf_instance_id}"
                )
            self._admitted_leases.update(new_lease_ids)
            edge = EdgeInfo(
                dispatch_id=dispatch_id,
                lease_id=lease.lease_id,
                rid=lease.rid,
                pa_instance_id=lease.pa_instance_id,
                pf_instance_id=lease.pf_instance_id,
                pair_epoch=lease.pair_epoch,
                cost=float(lease.cost if cost is None else cost),
                started_at=time.time(),
                lease_ids=requested_lease_ids,
            )
            self._dispatches[dispatch_id] = edge
            return edge

    def complete_dispatch(
        self, dispatch_id: str, lease_id: str | None = None
    ) -> bool:
        with self._lock:
            edge = self._dispatches.get(dispatch_id)
            if edge is None:
                return False
            if lease_id is not None and edge.lease_id != lease_id:
                return False
            lease = self._leases_by_id.get(edge.lease_id)
            if lease is None or not self._lease_valid_locked(lease):
                self._dispatches.pop(dispatch_id, None)
                return False
            self._dispatches.pop(dispatch_id)
            self._completed_dispatches[dispatch_id] = edge
            return True

    def release(self, rid: str, pa_id: str | None = None) -> bool:
        with self._lock:
            lease = self._leases_by_rid.get(rid)
            if lease is None or (pa_id is not None and lease.pa_instance_id != pa_id):
                return False
            if any(
                lease.lease_id in (edge.lease_ids or (edge.lease_id,))
                for edge in self._dispatches.values()
            ):
                raise RuntimeError("cannot release a RID with inflight dispatches")
            self._admitted_leases.discard(lease.lease_id)
            self._drop_lease_locked(lease)
            return True

    def mark_failed(self, pf_instance_id: str) -> int:
        """Fence a PF epoch; stale completions become harmless no-ops."""
        with self._lock:
            if pf_instance_id not in self._epochs:
                raise KeyError(f"unknown PF: {pf_instance_id}")
            self._registrations.pop(pf_instance_id, None)
            failed_leases = {
                lease_id
                for lease_id, lease in self._leases_by_id.items()
                if lease.pf_instance_id == pf_instance_id
            }
            self._admitted_leases.difference_update(failed_leases)
            for lease_id in failed_leases:
                lease = self._leases_by_id.get(lease_id)
                if lease is not None:
                    self._drop_lease_locked(lease)
            for dispatch_id, edge in list(self._dispatches.items()):
                if edge.pf_instance_id == pf_instance_id:
                    self._dispatches.pop(dispatch_id)
                    self._completed_dispatches[dispatch_id] = edge
            self._epochs[pf_instance_id] += 1
            return self._epochs[pf_instance_id]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "pfs": [
                    dataclasses.asdict(self._registrations[pf_id])
                    for pf_id in sorted(self._registrations)
                ],
                "loads": [
                    dataclasses.asdict(self._load_locked(pf_id))
                    for pf_id in sorted(self._registrations)
                ],
                "leases": [
                    dataclasses.asdict(lease)
                    for lease in sorted(
                        self._leases_by_id.values(), key=lambda item: item.lease_id
                    )
                ],
                "admitted_leases": sorted(self._admitted_leases),
                "dispatches": [
                    dataclasses.asdict(edge)
                    for edge in sorted(
                        self._dispatches.values(), key=lambda item: item.dispatch_id
                    )
                ],
                "epochs": dict(sorted(self._epochs.items())),
                "lease_seq": self._lease_seq,
            }

    status = snapshot

    def _lease_valid_locked(self, lease: Lease) -> bool:
        registration = self._registrations.get(lease.pf_instance_id)
        return registration is not None and registration.epoch == lease.pair_epoch

    def _drop_lease_locked(self, lease: Lease) -> None:
        self._leases_by_id.pop(lease.lease_id, None)
        if self._leases_by_rid.get(lease.rid) == lease:
            self._leases_by_rid.pop(lease.rid, None)


class AFDPoolCoordinatorRPCServer:
    """Single-socket ZeroMQ REP server for :class:`AFDPoolCoordinator`."""

    def __init__(self, endpoint: str, coordinator: AFDPoolCoordinator | None = None):
        if zmq is None:
            raise RuntimeError("pyzmq is required for AFD coordinator RPC")
        self.endpoint = endpoint
        self.coordinator = coordinator or AFDPoolCoordinator()
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(endpoint)
        self._stop = threading.Event()

    def serve_forever(self) -> None:
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        while not self._stop.is_set():
            if not dict(poller.poll(100)).get(self._socket):
                continue
            request = self._socket.recv_json()
            try:
                result = self._dispatch(request)
                self._socket.send_json({"ok": True, "result": result})
            except Exception as exc:  # noqa: BLE001 - RPC boundary serializes errors
                self._socket.send_json(
                    {
                        "ok": False,
                        "error": type(exc).__name__,
                        "error_code": getattr(exc, "code", "COORDINATOR_ERROR"),
                        "message": str(exc),
                    }
                )

    def close(self) -> None:
        self._stop.set()
        self._socket.close(0)
        self._context.term()

    def _dispatch(self, request: dict[str, Any]) -> Any:
        command = str(request.get("command", "")).upper()
        args = dict(request.get("args") or {})
        methods = {
            "REGISTER": self.coordinator.register,
            "UNREGISTER": self.coordinator.unregister,
            "HEARTBEAT": self.coordinator.heartbeat,
            "ACQUIRE": self.coordinator.acquire,
            "LOOKUP": self.coordinator.lookup,
            "BEGIN": self.coordinator.begin_dispatch,
            "COMPLETE": self.coordinator.complete_dispatch,
            "RELEASE": self.coordinator.release,
            "FAILED": self.coordinator.mark_failed,
            "SNAPSHOT": self.coordinator.snapshot,
            "STATUS": self.coordinator.status,
        }
        if command not in methods:
            raise ValueError(f"unknown command: {command}")
        result = methods[command](**args)
        return (
            dataclasses.asdict(result) if dataclasses.is_dataclass(result) else result
        )


class AFDPoolCoordinatorClient:
    """Process-safe API client; each call uses a short-lived REQ socket."""

    def __init__(self, endpoint: str, timeout_ms: int = 5000):
        self.endpoint = endpoint
        self.timeout_ms = int(timeout_ms)

    def call(self, command: str, **kwargs) -> Any:
        if zmq is None:
            raise RuntimeError("pyzmq is required for AFD coordinator RPC")
        context = zmq.Context.instance()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        socket.connect(self.endpoint)
        try:
            socket.send_json({"command": command.upper(), "args": kwargs})
            response = socket.recv_json()
        except zmq.Again as exc:
            raise TimeoutError(f"coordinator RPC timed out: {command}") from exc
        finally:
            socket.close(0)
        if not response.get("ok"):
            remote_type = response.get("error", "error")
            code = response.get("error_code", "COORDINATOR_ERROR")
            message = f"coordinator {remote_type}: {response.get('message')}"
            if code == PFCapacityExhausted.code:
                raise PFCapacityExhausted(message)
            raise AFDCoordinatorRemoteError(message, code=code, remote_type=remote_type)
        return response.get("result")

    def register(self, **kwargs):
        return self.call("REGISTER", **kwargs)

    def heartbeat(self, pf_instance_id: str):
        return self.call("HEARTBEAT", pf_instance_id=pf_instance_id)

    def acquire(
        self,
        rid: str,
        pa_id: str,
        cost: float = 1.0,
        preferred_pf_instance_id: str | None = None,
    ):
        return self.call(
            "ACQUIRE",
            rid=rid,
            pa_id=pa_id,
            cost=cost,
            preferred_pf_instance_id=preferred_pf_instance_id,
        )

    def lookup(self, rid: str):
        return self.call("LOOKUP", rid=rid)

    def begin_dispatch(
        self,
        dispatch_id: str,
        lease_id: str,
        *,
        wait_timeout_s: float = 0.0,
        retry_backoff_s: float = 0.05,
        **kwargs,
    ):
        """Begin a dispatch, briefly waiting only for transient PF backpressure.

        Every retry replays the exact dispatch and lease identity, preserving
        BEGIN idempotency. All non-capacity protocol and transport failures are
        deliberately propagated after the first attempt.
        """
        wait_timeout_s = float(wait_timeout_s)
        retry_backoff_s = float(retry_backoff_s)
        if wait_timeout_s < 0:
            raise ValueError("wait_timeout_s must be non-negative")
        if retry_backoff_s <= 0:
            raise ValueError("retry_backoff_s must be positive")
        deadline = time.monotonic() + wait_timeout_s
        while True:
            try:
                return self.call(
                    "BEGIN", dispatch_id=dispatch_id, lease_id=lease_id, **kwargs
                )
            except PFCapacityExhausted as exc:
                if wait_timeout_s == 0:
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise PFCapacityWaitTimeout(
                        "Timed out waiting for PF capacity for "
                        f"dispatch={dispatch_id}, lease={lease_id} after "
                        f"{wait_timeout_s:.3f}s"
                    ) from exc
                time.sleep(min(retry_backoff_s, remaining))

    def complete_dispatch(self, dispatch_id: str, lease_id: str | None = None):
        return self.call("COMPLETE", dispatch_id=dispatch_id, lease_id=lease_id)

    def release(self, rid: str, pa_id: str | None = None):
        return self.call("RELEASE", rid=rid, pa_id=pa_id)

    def mark_failed(self, pf_instance_id: str):
        return self.call("FAILED", pf_instance_id=pf_instance_id)

    def snapshot(self):
        return self.call("SNAPSHOT")


def make_dispatch_identity(pa_instance_id: str, pair_epoch: int, sequence: int) -> str:
    """Return a globally namespaced dispatch identity for scheduler messages."""
    if not pa_instance_id or pair_epoch < 0 or sequence < 0:
        raise ValueError("invalid dispatch namespace components")
    return f"{pa_instance_id}:{pair_epoch}:{sequence}"


def _parse_pf_spec(spec: str) -> tuple[str, int, str]:
    parts = spec.split(":", 2)
    pf_id = parts[0]
    capacity = int(parts[1]) if len(parts) > 1 and parts[1] else 1
    endpoint = parts[2] if len(parts) > 2 else ""
    return pf_id, capacity, endpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the shared AFD PF coordinator")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19090)
    parser.add_argument(
        "--pf", action="append", default=[], metavar="ID[:CAPACITY[:ENDPOINT]]"
    )
    args = parser.parse_args()
    coordinator = AFDPoolCoordinator()
    for spec in args.pf:
        pf_id, capacity, endpoint = _parse_pf_spec(spec)
        coordinator.register(pf_id, endpoint=endpoint, capacity=capacity)
    server = AFDPoolCoordinatorRPCServer(f"tcp://{args.host}:{args.port}", coordinator)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()


if __name__ == "__main__":
    main()
