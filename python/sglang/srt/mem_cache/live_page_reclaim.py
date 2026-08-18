"""Page-level eligibility for safe host-KV reclamation.

The radix tree remains the authority on prefix value and lifetime.  This
module only converts node facts into the smallest host-cache handoff unit:
one complete SGLang KV page.  A page is eligible only when *every* node slice
touching it is safe to remove in the selected policy mode.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import heapq
import math
import time
from typing import Iterable


def prefix_fingerprint(token_ids: Iterable[int], extra_key: object | None = None) -> str:
    """Return a stable, namespace-aware identity for ghost reclaim feedback."""
    digest = hashlib.blake2b(digest_size=16)
    digest.update(repr(extra_key).encode("utf-8"))
    digest.update(b"\\0")
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(8, byteorder="little", signed=True))
    return digest.hexdigest()


@dataclass(frozen=True)
class HostNodeSnapshot:
    """The reclaim-relevant facts of one radix node with host KV."""

    node_id: int
    host_slots: tuple[int, ...]
    hit_count: int
    child_count: int
    host_ref_counter: int
    evicted: bool
    # ``lock_ref`` fences a node while SGLang still owns an asynchronous
    # write-through/load operation.  It is distinct from host_ref_counter:
    # the latter protects host-I/O operations, while the former is the radix
    # request/backup lifetime.  Both make a host copy ineligible for local
    # reclamation.
    lock_ref: int = 0
    pinned: bool = False
    parent_id: int | None = None
    last_access_time: float = 0.0
    is_root: bool = False
    reuse_key: str | None = None
    storage_state: str = "none"
    # ``None`` preserves the legacy storage-backend contract, whose durable
    # acknowledgement is node-wide.  Central I/O supplies an explicit set so
    # that a partially persisted node can never make a shared host page
    # reclaimable.
    durable_page_ids: frozenset[int] | None = None


@dataclass
class ReuseRecord:
    """Short-lived, observed reuse evidence for one complete prefix path.

    This deliberately records events which actually happened in the local
    cache.  It is not a continuation predictor: a recent host hit says that
    the prefix was useful now, while a reclaim loss says that removing it was
    costly now.  Both observations expire with the owner's host-cache
    turnover horizon.
    """

    reprefill_tokens: int = 0
    last_loss_s: float | None = None
    host_hit_count: int = 0
    last_host_hit_s: float | None = None

    @property
    def last_evidence_s(self) -> float | None:
        timestamps = (
            timestamp
            for timestamp in (self.last_loss_s, self.last_host_hit_s)
            if timestamp is not None
        )
        return max(timestamps, default=None)


def _reclaim_value(node: HostNodeSnapshot) -> float:
    """Return the weak history prior used before real reclaim feedback exists.

    Historical hits order otherwise equally safe leaves, but do not impose a
    fixed protection cutoff in the local maintainer. ``ReuseRecord`` adds the
    real observed cost of a recent reclaim miss on top of this prior.
    """
    return float(node.hit_count)


@dataclass(frozen=True)
class PageReclaimPlan:
    """A read-only page census plus the fully reclaimable page ranges."""

    reclaimable_page_ranges: list[tuple[int, int]]
    page_reasons: dict[int, dict[str, int]]
    reclaimable_pages: int
    blocked_pages: int
    occupied_pages: int


@dataclass(frozen=True)
class PageDrainSelection:
    """Pages and whole radix leaves that may enter a targeted drain."""

    page_ranges: list[tuple[int, int]]
    node_ids: list[int]


@dataclass(frozen=True)
class PageReclaimActionSelection:
    """A page-complete reclaim plan with non-destructive and destructive actions.

    ``drop_host_copy_node_ids`` retain the radix node and its GPU KV.  They
    only relinquish the duplicate host backup.  ``delete_host_leaf_node_ids``
    remove host-only leaves after their children have been removed first.
    """

    page_ranges: list[tuple[int, int]]
    drop_host_copy_node_ids: list[int]
    delete_host_leaf_node_ids: list[int]
    blocker_counts: dict[str, int]


@dataclass(frozen=True)
class IncrementalLeafAction:
    """One safe low-value leaf action for an iterative reclaim worker.

    This deliberately does *not* require the action to empty a whole page.
    Releasing a child can expose its parent as the next low-value leaf.  The
    HostKV allocator reports a page back to Central I/O only after the final
    live slot in that page has disappeared.
    """

    node_id: int | None
    action_kind: str | None
    blocker_counts: dict[str, int]
    immediate_page_yield: int = 0


class IncrementalLeafReclaimQueue:
    """Maintain leaf eligibility across a quota-shrink operation.

    The queue is built from one radix census.  Deleting a child updates only
    its parent, so a newly exposed parent is discovered without materializing
    the full host-slot set again.  Callers must discard and rebuild the queue
    when unrelated cache activity invalidates the census.
    """

    def __init__(
        self,
        *,
        page_size: int = 16,
        protected_hit_count: int | None,
        protected_fraction: float | None = None,
        nodes: Iterable[HostNodeSnapshot],
    ) -> None:
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        if protected_hit_count is not None and protected_hit_count < 0:
            raise ValueError("protected_hit_count must be non-negative")
        if protected_fraction is not None and not 0.0 <= protected_fraction < 1.0:
            raise ValueError("protected_fraction must be in [0, 1)")
        if protected_fraction is not None and protected_hit_count is not None:
            raise ValueError("choose relative protection or an absolute hit cutoff, not both")
        self.page_size = page_size
        self.protected_hit_count = protected_hit_count
        self.protected_fraction = protected_fraction
        snapshots = [node for node in nodes if node.host_slots]
        self.nodes = {node.node_id: node for node in snapshots}
        if len(self.nodes) != len(snapshots):
            raise ValueError("radix snapshot contains duplicate node ids")
        self.remaining_children = {
            node_id: node.child_count for node_id, node in self.nodes.items()
        }
        self._ready: list[tuple[int, int, float, int, float, int, int, str, int]] = []
        self._finished: set[int] = set()
        self._queued: set[int] = set()
        self._entry_versions: dict[int, int] = defaultdict(int)
        self._node_pages = {
            node.node_id: {slot // self.page_size for slot in node.host_slots}
            for node in snapshots
        }
        self._page_remaining_nodes: dict[int, set[int]] = defaultdict(set)
        for node_id, page_ids in self._node_pages.items():
            for page_id in page_ids:
                self._page_remaining_nodes[page_id].add(node_id)
        self.snapshot_builds = 1
        self.blocker_counts: dict[str, int] = defaultdict(int)
        self.protected_node_ids = self._relative_protected_node_ids()
        for node_id in self.nodes:
            self._enqueue_if_ready(node_id)

    def _relative_protected_node_ids(self) -> set[int]:
        if not self.protected_fraction:
            return set()
        candidates = [
            node
            for node in self.nodes.values()
            if (
                not node.is_root
                and node.host_ref_counter == 0
                and node.lock_ref == 0
                and not node.pinned
            )
        ]
        count = math.ceil(len(candidates) * self.protected_fraction)
        return {
            node.node_id
            for node in sorted(
                candidates,
                key=lambda item: (_reclaim_value(item), item.last_access_time, item.node_id),
                reverse=True,
            )[:count]
        }

    def _immediate_page_yield(self, node_id: int) -> int:
        """Count pages that become allocator-free if this node is removed."""
        return sum(
            self._page_remaining_nodes[page_id] == {node_id}
            for page_id in self._node_pages.get(node_id, set())
        )

    def _enqueue_if_ready(self, node_id: int, *, force: bool = False) -> None:
        if node_id in self._finished or (node_id in self._queued and not force):
            return
        node = self.nodes.get(node_id)
        if node is None or node.is_root:
            return
        if node_id in self.protected_node_ids:
            self.blocker_counts["relative_high_reuse"] += 1
            return
        reason = _action_block_reason(node, self.protected_hit_count)
        if reason is not None:
            self.blocker_counts[reason] += 1
            return
        if self.remaining_children[node_id] != 0:
            self.blocker_counts["shared_child"] += 1
            return
        action_kind = "delete_host_leaf" if node.evicted else "drop_host_copy"
        immediate_page_yield = self._immediate_page_yield(node_id)
        # Prefer a duplicate host copy on ties: it frees host capacity while
        # keeping the GPU-resident radix node intact.
        kind_priority = 0 if action_kind == "drop_host_copy" else 1
        self._entry_versions[node_id] += 1
        entry_version = self._entry_versions[node_id]
        heapq.heappush(
            self._ready,
            (
                0 if immediate_page_yield else 1,
                _reclaim_value(node),
                node.last_access_time,
                kind_priority,
                node.node_id,
                action_kind,
                entry_version,
            ),
        )
        self._queued.add(node_id)

    def pop(self) -> IncrementalLeafAction:
        while self._ready:
            (
                _yield_priority,
                _reclaim_value_score,
                _access_time,
                _kind_priority,
                node_id,
                action_kind,
                entry_version,
            ) = heapq.heappop(self._ready)
            if self._entry_versions[node_id] != entry_version:
                continue
            self._queued.discard(node_id)
            if node_id in self._finished:
                continue
            return IncrementalLeafAction(
                node_id=node_id,
                action_kind=action_kind,
                blocker_counts=dict(self.blocker_counts),
                immediate_page_yield=self._immediate_page_yield(node_id),
            )
        return IncrementalLeafAction(None, None, dict(self.blocker_counts))

    def discard(self, node_id: int) -> None:
        """Forget a candidate whose runtime state changed after the census."""
        self._finished.add(node_id)

    def _release_node_page_ownership(self, node_id: int) -> None:
        """Update nearby candidates after a host node's slots become free."""
        for page_id in self._node_pages.get(node_id, set()):
            owners = self._page_remaining_nodes[page_id]
            owners.discard(node_id)
            for neighbor_id in owners:
                self._enqueue_if_ready(neighbor_id, force=True)

    def complete_drop_host_copy(self, node_id: int) -> None:
        self._finished.add(node_id)
        self._release_node_page_ownership(node_id)

    def complete_delete(self, node_id: int) -> None:
        """Record a deletion and promote only the direct parent if eligible."""
        if node_id in self._finished:
            return
        self._finished.add(node_id)
        self._release_node_page_ownership(node_id)
        node = self.nodes.get(node_id)
        if node is None or node.parent_id is None:
            return
        parent_id = node.parent_id
        if parent_id not in self.remaining_children:
            return
        self.remaining_children[parent_id] = max(
            0, self.remaining_children[parent_id] - 1
        )
        self._enqueue_if_ready(parent_id)


class PersistentLeafReclaimIndex:
    """Maintain reclaim candidates between quota changes.

    The one-shot queue above is useful for a test or a single drain, but it
    begins with a full radix snapshot. The runtime index receives only the
    nodes whose cache lifecycle changed, records page membership rather than
    every slot id, and keeps a lazy priority heap ready for the next shrink.
    The caller still rechecks a selected node before deleting it because a
    cache hit or an in-flight request can make a heap entry stale.
    """

    def __init__(
        self,
        *,
        page_size: int = 16,
        ghost_ttl_s: float = 60.0,
        requires_durable_storage: bool = False,
    ) -> None:
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        if ghost_ttl_s <= 0:
            raise ValueError("ghost_ttl_s must be positive")
        self.page_size = page_size
        self.ghost_ttl_s = ghost_ttl_s
        self.requires_durable_storage = requires_durable_storage
        self.nodes: dict[int, HostNodeSnapshot] = {}
        self._node_pages: dict[int, set[int]] = {}
        self._page_remaining_nodes: dict[int, set[int]] = defaultdict(set)
        # A host page is the atomic handoff unit, but a radix split can leave
        # more than one leaf inside that page.  Keep the complete safe owner
        # set instead of treating "one leaf owns this page" as a prerequisite
        # for local preparation.  The page remains only *ready*, never clean,
        # until every member has passed the final lifecycle recheck and been
        # removed together.
        self._ready_page_candidates: dict[int, frozenset[int]] = {}
        self._ready_page_scores: dict[int, float] = {}
        self._ready_page_reclaim_score = 0.0
        self._ready_node_pages: dict[int, set[int]] = defaultdict(set)
        self._ready_node_versions: dict[int, int] = defaultdict(int)
        self._ready_nodes: list[tuple[tuple[int, float, int, float, int, int, float], int, int]] = []
        self.remaining_children: dict[int, int] = {}
        self._ready: list[
            tuple[int, float, int, float, int, int, int, float, int, int, str, int]
        ] = []
        self._entry_versions: dict[int, int] = defaultdict(int)
        self._queued: set[int] = set()
        self._finished: set[int] = set()
        self.blocker_counts: dict[str, int] = defaultdict(int)
        self.node_updates = 0
        self._reuse_records: dict[str, ReuseRecord] = {}
        self._reclaim_ghosts: dict[str, float] = {}
        self._reclaim_ghost_prefixes: dict[str, tuple[tuple[int, ...], str]] = {}
        # Keep boundary evidence explicitly observable in runtime audits.  A
        # retention-loss total alone cannot tell whether a missing protection
        # was caused by an expired/mismatched ghost or by a later candidate
        # ordering decision.
        self._feedback_ghost_marks = 0
        self._feedback_revisit_matches = 0
        self._feedback_revisit_misses = 0

    @staticmethod
    def _reuse_key(node: HostNodeSnapshot) -> str:
        return node.reuse_key or f"node:{node.node_id}"

    def _purge_expired_reuse_records(self, now_s: float) -> None:
        expired_records = [
            key
            for key, record in self._reuse_records.items()
            if record.last_evidence_s is None
            or now_s - record.last_evidence_s >= self.ghost_ttl_s
        ]
        expired_ghosts = [
            key
            for key, reclaimed_at in self._reclaim_ghosts.items()
            if now_s - reclaimed_at >= self.ghost_ttl_s
        ]
        for key in expired_records:
            del self._reuse_records[key]
        for key in expired_ghosts:
            del self._reclaim_ghosts[key]
            self._reclaim_ghost_prefixes.pop(key, None)

        if not expired_records:
            return
        expired_keys = set(expired_records)
        for node_id, node in self.nodes.items():
            if self._reuse_key(node) in expired_keys:
                self._enqueue_if_ready(node_id, force=True)
                self._refresh_ready_node_priority(node_id)

    def set_ghost_ttl_s(self, ghost_ttl_s: float, *, now_s: float) -> None:
        """Synchronize feedback retention with the owner's current turnover."""
        if ghost_ttl_s <= 0 or now_s < 0:
            raise ValueError("ghost TTL update must be positive and monotonic")
        self.ghost_ttl_s = ghost_ttl_s
        self._purge_expired_reuse_records(now_s)

    def record_reclaim_loss(
        self, *, reuse_key: str, reprefill_tokens: int, now_s: float
    ) -> None:
        """Remember a real short-term miss without predicting future traffic."""
        if not reuse_key:
            raise ValueError("reuse_key must be non-empty")
        if reprefill_tokens <= 0 or now_s < 0:
            raise ValueError("loss inputs must be positive")
        previous = self._reuse_records.get(reuse_key)
        self._reuse_records[reuse_key] = ReuseRecord(
            reprefill_tokens=reprefill_tokens
            + (previous.reprefill_tokens if previous else 0),
            last_loss_s=now_s,
            host_hit_count=previous.host_hit_count if previous else 0,
            last_host_hit_s=previous.last_host_hit_s if previous else None,
        )
        for node_id, node in self.nodes.items():
            if self._reuse_key(node) == reuse_key:
                self._enqueue_if_ready(node_id, force=True, now_s=now_s)
                self._refresh_ready_node_priority(node_id)

    def record_host_hit(
        self, *, reuse_key: str, hit_tokens: int, now_s: float
    ) -> None:
        """Remember a real host-cache reuse without treating it as a forecast.

        The event is intentionally keyed by the complete prefix path, not a
        session label.  A future split can change radix nodes while preserving
        this identity.  The count is only a tie-breaker; freshness expires
        after local host turnover so old traffic does not permanently protect
        a prefix.
        """
        if not reuse_key:
            raise ValueError("reuse_key must be non-empty")
        if hit_tokens <= 0 or now_s < 0:
            raise ValueError("host hit inputs must be positive")
        self._purge_expired_reuse_records(now_s)
        previous = self._reuse_records.get(reuse_key)
        self._reuse_records[reuse_key] = ReuseRecord(
            reprefill_tokens=previous.reprefill_tokens if previous else 0,
            last_loss_s=previous.last_loss_s if previous else None,
            host_hit_count=(previous.host_hit_count if previous else 0) + 1,
            last_host_hit_s=now_s,
        )
        for node_id, node in self.nodes.items():
            if self._reuse_key(node) == reuse_key:
                self._enqueue_if_ready(node_id, force=True, now_s=now_s)
                self._refresh_ready_node_priority(node_id)

    def mark_reclaimed(
        self,
        *,
        reuse_key: str,
        now_s: float,
        token_ids: Iterable[int] | None = None,
        extra_key: object | None = None,
    ) -> None:
        """Leave a bounded ghost; deletion alone is not treated as a loss."""
        if not reuse_key or now_s < 0:
            raise ValueError("ghost inputs are invalid")
        self._purge_expired_reuse_records(now_s)
        self._reclaim_ghosts[reuse_key] = now_s
        if token_ids is not None:
            self._reclaim_ghost_prefixes[reuse_key] = (
                tuple(int(token_id) for token_id in token_ids),
                repr(extra_key),
            )
        self._feedback_ghost_marks += 1

    def record_revisit(
        self,
        *,
        reuse_key: str,
        reprefill_tokens: int,
        now_s: float,
        token_ids: Iterable[int] | None = None,
        extra_key: object | None = None,
    ) -> int:
        """Convert an outstanding ghost to feedback when re-prefill really occurs."""
        self._purge_expired_reuse_records(now_s)
        matched_key = reuse_key if reuse_key in self._reclaim_ghosts else None
        prefix_entry = (
            self._reclaim_ghost_prefixes.get(matched_key)
            if matched_key is not None
            else None
        )
        matched_tokens = len(prefix_entry[0]) if prefix_entry is not None else reprefill_tokens
        if matched_key is None and token_ids is not None:
            current_tokens = tuple(int(token_id) for token_id in token_ids)
            current_extra = repr(extra_key)
            candidates = [
                (len(prefix_tokens), ghost_key)
                for ghost_key, (prefix_tokens, ghost_extra) in self._reclaim_ghost_prefixes.items()
                if ghost_extra == current_extra
                and len(prefix_tokens) <= len(current_tokens)
                and current_tokens[: len(prefix_tokens)] == prefix_tokens
            ]
            if candidates:
                matched_tokens, matched_key = max(candidates)
        if matched_key is None:
            self._feedback_revisit_misses += 1
            return 0
        matched_prefix_entry = self._reclaim_ghost_prefixes.get(matched_key)
        covered_keys = {matched_key}
        if matched_prefix_entry is not None:
            matched_prefix, matched_extra = matched_prefix_entry
            covered_keys.update(
                ghost_key
                for ghost_key, (prefix_tokens, ghost_extra) in self._reclaim_ghost_prefixes.items()
                if ghost_extra == matched_extra
                and len(prefix_tokens) <= len(matched_prefix)
                and matched_prefix[: len(prefix_tokens)] == prefix_tokens
            )
        for ghost_key in covered_keys:
            self._reclaim_ghosts.pop(ghost_key, None)
            self._reclaim_ghost_prefixes.pop(ghost_key, None)
        self._feedback_revisit_matches += 1
        self.record_reclaim_loss(
            reuse_key=matched_key, reprefill_tokens=matched_tokens, now_s=now_s
        )
        return matched_tokens

    def feedback_observability(self, *, now_s: float | None = None) -> dict[str, int]:
        """Expose bounded reclaim-feedback state without changing policy.

        These counters make the reclaim -> revisit -> candidate path auditable:
        a ghost miss is distinct from a matched revisit whose later host copy
        is not presently represented among reclaim candidates.
        """
        if now_s is not None:
            self._purge_expired_reuse_records(now_s)
        feedback_candidate_nodes = sum(
            1
            for node in self.nodes.values()
            if (
                (record := self._reuse_records.get(self._reuse_key(node))) is not None
                and record.reprefill_tokens > 0
            )
        )
        return {
            "ghost_marks": self._feedback_ghost_marks,
            "revisit_matches": self._feedback_revisit_matches,
            "revisit_misses": self._feedback_revisit_misses,
            "active_ghosts": len(self._reclaim_ghosts),
            "active_loss_records": sum(
                record.reprefill_tokens > 0
                for record in self._reuse_records.values()
            ),
            "feedback_candidate_nodes": feedback_candidate_nodes,
        }

    def _feedback_loss_per_page(self, node: HostNodeSnapshot) -> float:
        """Return observed regret normalized by the capacity this node frees.

        ``hit_count`` and ``reprefill_tokens`` are not commensurate units, so
        they must not be added into one synthetic score.  A reclaim feedback
        record is stronger evidence than historical hits: it says this exact
        prefix was removed and soon had to be rebuilt.  Dividing by resident
        pages keeps a large leaf from looking expensive merely because it
        carries more bytes while it also returns proportionally more quota.
        """
        record = self._reuse_records.get(self._reuse_key(node))
        if record is None:
            return 0.0
        return record.reprefill_tokens / max(1, len(self._node_pages[node.node_id]))

    def _recent_host_hit(self, node: HostNodeSnapshot, now_s: float) -> tuple[int, float, int]:
        """Return bounded host-hit protection for local value ordering.

        A lower tuple is reclaimed first.  Recent host hits therefore protect
        a leaf ahead of an equally safe, unobserved leaf.  The timestamp is
        retained as a final ordering key so an older hit is reclaimed before
        a newer hit once both are otherwise comparable.
        """
        record = self._reuse_records.get(self._reuse_key(node))
        if record is None or record.last_host_hit_s is None:
            return (0, 0.0, 0)
        if now_s - record.last_host_hit_s >= self.ghost_ttl_s:
            return (0, 0.0, 0)
        return (1, record.last_host_hit_s, record.host_hit_count)

    def _reclaim_priority(
        self, node: HostNodeSnapshot, *, now_s: float
    ) -> tuple[int, float, int, float, int, int, float]:
        """Order low-loss leaves without pretending to predict future reuse."""
        loss_per_page = self._feedback_loss_per_page(node)
        # A recent real reclaim loss is an explicit protection signal.  Only
        # among equally unproven/proven candidates do hit and recency act as
        # weak local tie-breakers.
        recent_host_hit, last_host_hit, host_hit_count = self._recent_host_hit(
            node, now_s
        )
        return (
            1 if loss_per_page else 0,
            loss_per_page,
            recent_host_hit,
            last_host_hit,
            host_hit_count,
            node.hit_count,
            node.last_access_time,
        )

    def _remove_page_membership(self, node_id: int) -> None:
        for page_id in self._node_pages.pop(node_id, set()):
            owners = self._page_remaining_nodes[page_id]
            owners.discard(node_id)
            if not owners:
                del self._page_remaining_nodes[page_id]
            else:
                for neighbor_id in owners:
                    self._enqueue_if_ready(neighbor_id, force=True)
            self._refresh_ready_page(page_id)

    def _refresh_ready_page(self, page_id: int, *, force: bool = False) -> None:
        """Track pages one final safe action away from becoming clean.

        A ready page is deliberately conservative: every live fragment in it
        must belong to a currently safe radix leaf.  A radix split may leave
        two or more independently safe leaves in one physical page; that is
        still a real low-loss candidate and should not force an unnecessary
        quota grow.  A page with a shared parent, reader, pin, child, or an
        unfinished SSD write remains live.
        """
        owners = self._page_remaining_nodes.get(page_id)
        if not owners:
            self._replace_ready_page(page_id, None)
            return
        for node_id in owners:
            node = self.nodes.get(node_id)
            if (
                node is None
                or node.is_root
                or node.host_ref_counter > 0
                or node.lock_ref > 0
                or node.pinned
                or self.remaining_children.get(node_id, node.child_count) != 0
                or not self._node_storage_is_durable(node)
            ):
                self._replace_ready_page(page_id, None)
                return
        ready_owners = frozenset(owners)
        if self._ready_page_candidates.get(page_id) != ready_owners:
            self._replace_ready_page(page_id, ready_owners)
        elif force:
            for node_id in ready_owners:
                self._refresh_ready_node_priority(node_id)

    def _replace_ready_page(
        self, page_id: int, owners: frozenset[int] | None
    ) -> None:
        """Update one page's durable ready ordering after a lifecycle change."""
        previous_score = self._ready_page_scores.pop(page_id, 0.0)
        self._ready_page_reclaim_score -= previous_score
        previous_owners = self._ready_page_candidates.pop(page_id, frozenset())
        for node_id in previous_owners:
            node_pages = self._ready_node_pages.get(node_id)
            if node_pages is None:
                continue
            node_pages.discard(page_id)
            if not node_pages:
                del self._ready_node_pages[node_id]
                self._ready_node_versions[node_id] += 1
        if not owners:
            return
        score = sum(
            self._feedback_loss_per_page(self.nodes[node_id])
            for node_id in owners
            if node_id in self.nodes and node_id not in self._finished
        )
        self._ready_page_candidates[page_id] = owners
        self._ready_page_scores[page_id] = score
        self._ready_page_reclaim_score += score
        for node_id in owners:
            node_pages = self._ready_node_pages[node_id]
            was_absent = not node_pages
            node_pages.add(page_id)
            if was_absent:
                self._enqueue_ready_node(node_id)

    def _enqueue_ready_node(self, node_id: int, *, force: bool = False) -> None:
        if not self._ready_node_pages.get(node_id):
            return
        node = self.nodes.get(node_id)
        if node is None or node_id in self._finished:
            return
        if not force and self._ready_node_versions[node_id]:
            return
        self._ready_node_versions[node_id] += 1
        heapq.heappush(
            self._ready_nodes,
            (
                self._reclaim_priority(node, now_s=time.monotonic()),
                node_id,
                self._ready_node_versions[node_id],
            ),
        )

    def _refresh_ready_node_priority(self, node_id: int) -> None:
        """Refresh one changed node without walking every ready page at pop time."""
        pages = self._ready_node_pages.get(node_id, set())
        if not pages:
            return
        for page_id in pages:
            owners = self._ready_page_candidates.get(page_id, frozenset())
            previous_score = self._ready_page_scores.get(page_id, 0.0)
            score = sum(
                self._feedback_loss_per_page(self.nodes[owner_id])
                for owner_id in owners
                if owner_id in self.nodes and owner_id not in self._finished
            )
            self._ready_page_scores[page_id] = score
            self._ready_page_reclaim_score += score - previous_score
        self._enqueue_ready_node(node_id, force=True)

    def _node_storage_is_durable(self, node: HostNodeSnapshot) -> bool:
        """Keep page-atomic Central durability separate from legacy node acks.

        An upstream backend reports one completion state for a whole radix
        node.  Central I/O owns physical host pages, so its acknowledgements
        are deliberately finer: a node is a valid lower-tier exit only after
        every page it occupies has a durable acknowledgement.  ``None`` is a
        compatibility marker for the former protocol; an explicit empty set
        is meaningful and therefore rejects reclamation.
        """
        if not self.requires_durable_storage:
            return True
        if node.durable_page_ids is None:
            return node.storage_state == "durable"
        return self._node_pages.get(node.node_id, set()).issubset(
            node.durable_page_ids
        )

    @property
    def ready_page_count(self) -> int:
        """Number of page-complete, currently safe local reclaim candidates."""
        return len(self._ready_page_candidates)

    @property
    def ready_page_reclaim_score(self) -> float:
        """Aggregate observed loss score of pages that could be reclaimed now.

        Page ownership remains atomic, but one leaf may cover several pages.
        Divide that leaf's score across its resident pages so the local
        maintainer can compare ready capacity from different instances without
        treating a large low-value leaf as an arbitrarily expensive donor.
        This is a ranking signal, never permission to bypass the final radix
        safety check.
        """
        self._purge_expired_reuse_records(time.monotonic())
        if self._ready_page_reclaim_score < 0.0:
            if self._ready_page_reclaim_score > -1e-6:
                self._ready_page_reclaim_score = 0.0
            else:
                self._ready_page_reclaim_score = max(
                    0.0, sum(self._ready_page_scores.values())
                )
        return self._ready_page_reclaim_score

    def pop_ready_page_action(self, *, now_s: float) -> IncrementalLeafAction:
        """Choose an owner of an already page-complete safe candidate.

        Normal heap ordering remains the authority when no complete page is
        ready.  This fast path only prevents unrelated leaves from delaying
        the final owners of a page that the persistent index has already
        proved safe.  The caller must still recheck the live radix node before
        deleting it.
        """
        self._purge_expired_reuse_records(now_s)
        while self._ready_nodes:
            _priority, node_id, version = heapq.heappop(self._ready_nodes)
            if version != self._ready_node_versions[node_id]:
                continue
            if not self._ready_node_pages.get(node_id):
                continue
            node = self.nodes.get(node_id)
            if (
                node is None
                or node_id in self._finished
                or node.host_ref_counter > 0
                or node.lock_ref > 0
                or node.pinned
                or self.remaining_children.get(node_id, node.child_count) != 0
            ):
                self._ready_node_versions[node_id] += 1
                continue
            self._queued.discard(node.node_id)
            return IncrementalLeafAction(
                node_id=node.node_id,
                action_kind="delete_host_leaf" if node.evicted else "drop_host_copy",
                blocker_counts=dict(self.blocker_counts),
                immediate_page_yield=self._immediate_page_yield(node.node_id),
            )
        return IncrementalLeafAction(None, None, dict(self.blocker_counts))

    def remove(self, node_id: int) -> None:
        """Forget a node whose host copy no longer exists."""
        self._entry_versions[node_id] += 1
        self._queued.discard(node_id)
        self._finished.add(node_id)
        self.nodes.pop(node_id, None)
        self.remaining_children.pop(node_id, None)
        self._remove_page_membership(node_id)
        self.node_updates += 1

    def upsert(self, node: HostNodeSnapshot, page_ids: Iterable[int]) -> None:
        """Apply one ordinary radix/cache lifecycle update to the index."""
        page_set = set(page_ids)
        if not page_set:
            self.remove(node.node_id)
            return
        # Ordinary host hits and lock transitions often update a wide leaf
        # without changing its physical page membership.  Keep that page
        # relationship intact: tearing down and rebuilding every member page
        # would turn one lifecycle update into O(number of leaf pages) heap
        # churn, which is precisely what the persistent index is meant to
        # avoid.  Recheck safety in place, then refresh this node's one heap
        # entry for any changed access/value facts.
        if self._node_pages.get(node.node_id) == page_set:
            self._entry_versions[node.node_id] += 1
            self._queued.discard(node.node_id)
            self._finished.discard(node.node_id)
            self.nodes[node.node_id] = node
            self.remaining_children[node.node_id] = node.child_count
            for page_id in page_set:
                self._refresh_ready_page(page_id)
            self.node_updates += 1
            self._enqueue_if_ready(node.node_id, force=True)
            self._enqueue_ready_node(node.node_id, force=True)
            return
        self._entry_versions[node.node_id] += 1
        self._queued.discard(node.node_id)
        self._finished.discard(node.node_id)
        self._remove_page_membership(node.node_id)
        self.nodes[node.node_id] = node
        self.remaining_children[node.node_id] = node.child_count
        self._node_pages[node.node_id] = page_set
        for page_id in page_set:
            self._page_remaining_nodes[page_id].add(node.node_id)
            self._refresh_ready_page(page_id)
        self.node_updates += 1
        self._enqueue_if_ready(node.node_id, force=True)

    def _immediate_page_yield(self, node_id: int) -> int:
        return sum(
            self._page_remaining_nodes[page_id] == {node_id}
            for page_id in self._node_pages.get(node_id, set())
        )

    def _enqueue_if_ready(
        self, node_id: int, *, force: bool = False, now_s: float | None = None
    ) -> None:
        if node_id in self._finished or (node_id in self._queued and not force):
            return
        node = self.nodes.get(node_id)
        if node is None or node.is_root:
            return
        if node.host_ref_counter > 0 or node.lock_ref > 0 or node.pinned:
            return
        if self.remaining_children.get(node_id, node.child_count) != 0:
            return
        action_kind = "delete_host_leaf" if node.evicted else "drop_host_copy"
        self._entry_versions[node_id] += 1
        entry_version = self._entry_versions[node_id]
        (
            feedback_flag,
            feedback_loss_per_page,
            recent_host_hit,
            last_host_hit,
            host_hit_count,
            hit_count,
            last_access,
        ) = self._reclaim_priority(
            node, now_s=time.monotonic() if now_s is None else now_s
        )
        heapq.heappush(
            self._ready,
            (
                feedback_flag,
                feedback_loss_per_page,
                recent_host_hit,
                last_host_hit,
                host_hit_count,
                0 if self._immediate_page_yield(node_id) else 1,
                hit_count,
                last_access,
                0 if action_kind == "drop_host_copy" else 1,
                node.node_id,
                action_kind,
                entry_version,
            ),
        )
        self._queued.add(node_id)

    def _relative_protected_node_ids(self, protected_fraction: float | None) -> set[int]:
        if not protected_fraction:
            return set()
        candidates = [node for node_id, node in self.nodes.items() if not node.is_root and node.host_ref_counter == 0 and node.lock_ref == 0 and not node.pinned and self.remaining_children.get(node_id, node.child_count) == 0]
        count = math.ceil(len(candidates) * protected_fraction)
        return {node.node_id for node in sorted(candidates, key=lambda item: (_reclaim_value(item), item.last_access_time, item.node_id), reverse=True)[:count]}

    def pop(self, protected_hit_count: int | None = None, protected_fraction: float | None = None, now_s: float | None = None) -> IncrementalLeafAction:
        """Return the cheapest eligible leaf without rescanning host slots."""
        if now_s is not None:
            self._purge_expired_reuse_records(now_s)
        protected_ids = self._relative_protected_node_ids(protected_fraction)
        deferred = []
        try:
            while self._ready:
                entry = heapq.heappop(self._ready)
                (
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    node_id,
                    action_kind,
                    entry_version,
                ) = entry
                if self._entry_versions[node_id] != entry_version:
                    continue
                self._queued.discard(node_id)
                if node_id in self._finished:
                    continue
                node = self.nodes.get(node_id)
                if node is None:
                    continue
                reason = _action_block_reason(node, protected_hit_count)
                if reason is not None or node_id in protected_ids or self.remaining_children.get(node_id, node.child_count) != 0:
                    deferred.append(entry)
                    continue
                return IncrementalLeafAction(node_id=node_id, action_kind=action_kind, blocker_counts=dict(self.blocker_counts), immediate_page_yield=self._immediate_page_yield(node_id))
        finally:
            for entry in deferred:
                heapq.heappush(self._ready, entry)
                self._queued.add(entry[9])
        return IncrementalLeafAction(None, None, dict(self.blocker_counts))

    def complete_drop_host_copy(self, node_id: int) -> None:
        self.remove(node_id)

    def complete_delete(self, node_id: int) -> None:
        node = self.nodes.get(node_id)
        parent_id = node.parent_id if node is not None else None
        self.remove(node_id)
        if parent_id is None or parent_id not in self.remaining_children:
            return
        self.remaining_children[parent_id] = max(0, self.remaining_children[parent_id] - 1)
        self._enqueue_if_ready(parent_id, force=True)
        for page_id in self._node_pages.get(parent_id, set()):
            self._refresh_ready_page(page_id)


def _node_block_reason(
    node: HostNodeSnapshot, protected_hit_count: int
) -> str | None:
    """Return the first hard/value reason preventing this node's removal."""
    if node.host_ref_counter > 0:
        return "in_flight"
    if node.lock_ref > 0:
        return "in_flight"
    if not node.evicted:
        return "device_resident"
    if node.pinned:
        return "pinned"
    if node.child_count > 0:
        return "shared_prefix"
    if node.hit_count >= protected_hit_count:
        return "shared_prefix"
    return None


def _action_block_reason(
    node: HostNodeSnapshot, protected_hit_count: int | None
) -> str | None:
    """Return a hard or value blocker for a host-copy reclaim action."""
    if node.host_ref_counter > 0:
        return "in_flight"
    if node.lock_ref > 0:
        return "in_flight"
    if node.pinned:
        return "pinned"
    if protected_hit_count is not None and node.hit_count >= protected_hit_count:
        return "high_reuse"
    return None


def _coalesce_pages(page_ids: Iterable[int]) -> list[tuple[int, int]]:
    ordered = sorted(set(page_ids))
    if not ordered:
        return []
    result: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for page_id in ordered[1:]:
        if page_id == previous + 1:
            previous = page_id
            continue
        result.append((start, previous - start + 1))
        start = previous = page_id
    result.append((start, previous - start + 1))
    return result


def select_next_incremental_leaf_action(
    *,
    protected_hit_count: int,
    nodes: Iterable[HostNodeSnapshot],
) -> IncrementalLeafAction:
    """Pick exactly one least-valuable leaf without page-completeness gating.

    HiRadix invokes this repeatedly against fresh radix state.  That is what
    lets a parent become eligible immediately after its last child is removed,
    instead of waiting for an unrelated cache event.  A complete physical page
    is still required later by the allocator before Central I/O can scrub and
    transfer it.
    """
    if protected_hit_count < 0:
        raise ValueError("protected_hit_count must be non-negative")

    blockers: dict[str, int] = defaultdict(int)
    candidates: list[HostNodeSnapshot] = []
    for node in nodes:
        if not node.host_slots:
            continue
        if node.is_root:
            blockers["radix_root"] += 1
            continue
        reason = _action_block_reason(node, protected_hit_count)
        if reason is not None:
            blockers[reason] += 1
            continue
        if node.child_count:
            blockers["shared_child"] += 1
            continue
        candidates.append(node)

    if not candidates:
        return IncrementalLeafAction(None, None, dict(blockers))

    # Lower reuse and older access are less valuable.  Prefer dropping a
    # duplicate host copy over destructive deletion when value is otherwise
    # tied: it frees the same host slots while retaining GPU KV.
    node = min(
        candidates,
        key=lambda item: (
            _reclaim_value(item),
            item.last_access_time,
            0 if not item.evicted else 1,
            item.node_id,
        ),
    )
    return IncrementalLeafAction(
        node_id=node.node_id,
        action_kind="delete_host_leaf" if node.evicted else "drop_host_copy",
        blocker_counts=dict(blockers),
    )


def classify_reclaimable_pages(
    *,
    page_size: int,
    protected_hit_count: int,
    nodes: Iterable[HostNodeSnapshot],
) -> PageReclaimPlan:
    """Classify occupied pages without treating a contiguous range as atomic.

    ``host_slots`` need not be page-aligned: radix splitting can leave several
    node fragments in one page.  The page is reclaimable only if none of those
    fragments has a blocking reason.  Adjacent successful pages are coalesced
    only after this per-page decision.
    """
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if protected_hit_count < 0:
        raise ValueError("protected_hit_count must be non-negative")

    page_reasons: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    occupied_pages: set[int] = set()

    for node in nodes:
        if not node.host_slots:
            continue
        reason = _node_block_reason(node, protected_hit_count)
        for page_id in {slot // page_size for slot in node.host_slots}:
            if page_id < 0:
                raise ValueError("host slot must be non-negative")
            occupied_pages.add(page_id)
            if reason is not None:
                page_reasons[page_id][reason] += 1

    reclaimable = [page_id for page_id in occupied_pages if page_id not in page_reasons]
    normalized_reasons = {
        page_id: dict(reasons) for page_id, reasons in page_reasons.items()
    }
    return PageReclaimPlan(
        reclaimable_page_ranges=_coalesce_pages(reclaimable),
        page_reasons=normalized_reasons,
        reclaimable_pages=len(reclaimable),
        blocked_pages=len(page_reasons),
        occupied_pages=len(occupied_pages),
    )


def select_page_drain_candidates(
    *,
    page_size: int,
    protected_hit_count: int,
    nodes: Iterable[HostNodeSnapshot],
    max_pages: int | None = None,
) -> PageDrainSelection:
    """Select only pages that whole eligible leaves can actually empty.

    A page-level census alone is not enough: a low-value node can occupy a
    page partly while also extending into another, protected page.  Evicting
    that node would discard extra reusable KV without completing the desired
    handoff.  This fixed-point filter retains a page only when selected leaves
    cover all of its slots and every selected leaf stays inside selected pages.
    """
    snapshots = list(nodes)
    plan = classify_reclaimable_pages(
        page_size=page_size,
        protected_hit_count=protected_hit_count,
        nodes=snapshots,
    )
    candidate_pages = {
        page_id
        for start, count in plan.reclaimable_page_ranges
        for page_id in range(start, start + count)
    }
    if max_pages is not None:
        if max_pages <= 0:
            return PageDrainSelection(page_ranges=[], node_ids=[])
        candidate_pages = set(sorted(candidate_pages)[:max_pages])
    node_pages: dict[int, set[int]] = {}
    slots_by_page: dict[int, set[int]] = defaultdict(set)
    snapshots_by_id = {node.node_id: node for node in snapshots}
    for node in snapshots:
        pages = {slot // page_size for slot in node.host_slots}
        node_pages[node.node_id] = pages
        for slot in node.host_slots:
            slots_by_page[slot // page_size].add(slot)

    while candidate_pages:
        selected_nodes = {
            node_id
            for node_id, pages in node_pages.items()
            if pages and pages.issubset(candidate_pages)
        }
        covered: dict[int, set[int]] = defaultdict(set)
        for node_id in selected_nodes:
            for slot in snapshots_by_id[node_id].host_slots:
                covered[slot // page_size].add(slot)
        incomplete = {
            page_id
            for page_id in candidate_pages
            if covered[page_id]
            != set(range(page_id * page_size, (page_id + 1) * page_size))
        }
        if not incomplete:
            return PageDrainSelection(
                page_ranges=_coalesce_pages(candidate_pages),
                node_ids=sorted(selected_nodes),
            )
        candidate_pages.difference_update(incomplete)

    return PageDrainSelection(page_ranges=[], node_ids=[])


def _node_depth(node_id: int, snapshots_by_id: dict[int, HostNodeSnapshot]) -> int:
    depth = 0
    node = snapshots_by_id[node_id]
    seen = {node_id}
    while node.parent_id is not None and node.parent_id in snapshots_by_id:
        if node.parent_id in seen:
            raise ValueError("radix snapshot contains a parent cycle")
        seen.add(node.parent_id)
        depth += 1
        node = snapshots_by_id[node.parent_id]
    return depth


def _candidate_components(
    *,
    page_size: int,
    snapshots_by_id: dict[int, HostNodeSnapshot],
    action_kind: dict[int, str],
) -> tuple[set[int], set[int]]:
    """Return page-complete action/page sets after fixed-point pruning."""
    active_actions = set(action_kind)
    node_pages = {
        node_id: {slot // page_size for slot in node.host_slots}
        for node_id, node in snapshots_by_id.items()
    }
    page_nodes: dict[int, set[int]] = defaultdict(set)
    page_slots: dict[int, set[int]] = defaultdict(set)
    for node_id, node in snapshots_by_id.items():
        for slot in node.host_slots:
            page_id = slot // page_size
            page_nodes[page_id].add(node_id)
            page_slots[page_id].add(slot)

    while active_actions:
        candidate_pages = {
            page_id
            for page_id, node_ids in page_nodes.items()
            if node_ids and node_ids.issubset(active_actions)
        }
        complete_pages = {
            page_id
            for page_id in candidate_pages
            if page_slots[page_id]
            == set(range(page_id * page_size, (page_id + 1) * page_size))
        }
        next_actions = {
            node_id
            for node_id in active_actions
            if node_pages[node_id]
            and node_pages[node_id].issubset(complete_pages)
        }
        if next_actions == active_actions:
            return active_actions, complete_pages
        active_actions = next_actions
    return set(), set()


def _choose_component_pages(
    *,
    page_size: int,
    snapshots_by_id: dict[int, HostNodeSnapshot],
    action_ids: set[int],
    page_ids: set[int],
    max_pages: int | None,
) -> set[int]:
    """Choose complete page components by low reuse value before address order."""
    if max_pages is None:
        return set(page_ids)
    if max_pages <= 0:
        return set()

    node_pages = {
        node_id: {slot // page_size for slot in snapshots_by_id[node_id].host_slots}
        for node_id in action_ids
    }
    remaining_nodes = set(action_ids)
    components: list[tuple[tuple[float, float, float], set[int]]] = []
    while remaining_nodes:
        seed = remaining_nodes.pop()
        component_nodes = {seed}
        component_pages = set(node_pages[seed])
        changed = True
        while changed:
            changed = False
            for node_id in list(remaining_nodes):
                if node_pages[node_id] & component_pages:
                    remaining_nodes.remove(node_id)
                    component_nodes.add(node_id)
                    component_pages.update(node_pages[node_id])
                    changed = True
        component_pages.intersection_update(page_ids)
        values = [snapshots_by_id[node_id] for node_id in component_nodes]
        score = (
            float(max(node.hit_count for node in values)),
            float(sum(node.hit_count for node in values)),
            float(max(node.last_access_time for node in values)),
        )
        components.append((score, component_pages))

    chosen: set[int] = set()
    for _score, component_pages in sorted(components, key=lambda item: item[0]):
        if len(chosen) + len(component_pages) > max_pages:
            continue
        chosen.update(component_pages)
    return chosen


def select_page_reclaim_actions(
    *,
    page_size: int,
    protected_hit_count: int,
    nodes: Iterable[HostNodeSnapshot],
    max_pages: int | None = None,
) -> PageReclaimActionSelection:
    """Plan safe host-page handoff actions in post-order.

    A device-resident leaf contributes a ``drop_host_copy`` action: its GPU
    KV and radix entry stay valid.  An evicted leaf contributes a destructive
    ``delete_host_leaf`` action.  Once every child selected for deletion has
    been removed, an evicted low-value parent becomes a new leaf in this same
    planning pass rather than waiting for a later timer retry.
    """
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if protected_hit_count < 0:
        raise ValueError("protected_hit_count must be non-negative")

    snapshots = [node for node in nodes if node.host_slots]
    snapshots_by_id = {node.node_id: node for node in snapshots}
    if len(snapshots_by_id) != len(snapshots):
        raise ValueError("radix snapshot contains duplicate node ids")

    blocker_counts: dict[str, int] = defaultdict(int)
    children_by_parent: dict[int, set[int]] = defaultdict(set)
    for node in snapshots:
        if node.parent_id is not None and node.parent_id in snapshots_by_id:
            children_by_parent[node.parent_id].add(node.node_id)

    action_kind: dict[int, str] = {}
    for node in snapshots:
        if node.is_root:
            blocker_counts["radix_root"] += 1
            continue
        reason = _action_block_reason(node, protected_hit_count)
        if reason is not None:
            blocker_counts[reason] += 1
            continue
        if node.child_count != 0:
            blocker_counts["shared_child"] += 1
            continue
        action_kind[node.node_id] = (
            "delete_host_leaf" if node.evicted else "drop_host_copy"
        )

    # Delete-only post-order cascade. A dual-copy parent is deliberately not
    # cascaded while it can still have device-resident descendants.
    changed = True
    while changed:
        changed = False
        for node in snapshots:
            if node.node_id in action_kind or not node.evicted:
                continue
            reason = _action_block_reason(node, protected_hit_count)
            if reason is not None:
                continue
            known_children = children_by_parent.get(node.node_id, set())
            if node.child_count != len(known_children) or not known_children:
                continue
            if all(action_kind.get(child_id) == "delete_host_leaf" for child_id in known_children):
                action_kind[node.node_id] = "delete_host_leaf"
                changed = True

    action_ids, page_ids = _candidate_components(
        page_size=page_size,
        snapshots_by_id=snapshots_by_id,
        action_kind=action_kind,
    )
    page_ids = _choose_component_pages(
        page_size=page_size,
        snapshots_by_id=snapshots_by_id,
        action_ids=action_ids,
        page_ids=page_ids,
        max_pages=max_pages,
    )
    if not page_ids:
        return PageReclaimActionSelection([], [], [], dict(blocker_counts))

    node_pages = {
        node_id: {slot // page_size for slot in snapshots_by_id[node_id].host_slots}
        for node_id in action_ids
    }
    selected_ids = {
        node_id for node_id in action_ids if node_pages[node_id].issubset(page_ids)
    }
    drop_ids = [
        node_id
        for node_id in selected_ids
        if action_kind[node_id] == "drop_host_copy"
    ]
    delete_ids = [
        node_id
        for node_id in selected_ids
        if action_kind[node_id] == "delete_host_leaf"
    ]
    priority = lambda node_id: (
        snapshots_by_id[node_id].hit_count,
        snapshots_by_id[node_id].last_access_time,
        node_id,
    )
    drop_ids.sort(key=priority)
    delete_ids.sort(
        key=lambda node_id: (-_node_depth(node_id, snapshots_by_id), *priority(node_id))
    )
    return PageReclaimActionSelection(
        page_ranges=_coalesce_pages(page_ids),
        drop_host_copy_node_ids=drop_ids,
        delete_host_leaf_node_ids=delete_ids,
        blocker_counts=dict(blocker_counts),
    )
