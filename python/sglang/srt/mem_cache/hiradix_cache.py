from __future__ import annotations

import atexit
import heapq
import json
import logging
import math
import os
import threading
import time
from queue import Empty
from types import SimpleNamespace
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.cache_controller import HiCacheController, PrefetchOperation
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InitLoadBackParams,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    NSATokenToKVPool,
)
from sglang.srt.mem_cache.memory_pool_host import (
    CentralIOMHATokenToKVPoolHost,
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
    NSATokenToKVPoolHost,
)
from sglang.srt.mem_cache.live_page_reclaim import (
    HostNodeSnapshot,
    IncrementalLeafReclaimQueue,
    PersistentLeafReclaimIndex,
    PageReclaimPlan,
    classify_reclaimable_pages,
    prefix_fingerprint,
    select_page_reclaim_actions,
    select_page_drain_candidates,
)
from sglang.srt.mem_cache.radix_cache import (
    RadixCache,
    RadixKey,
    TreeNode,
    compute_node_hash_values,
    split_node_hash_value,
)
from sglang.srt.mem_cache.utils import convert_to_bigram_key
from sglang.srt.observability.metrics_collector import StorageMetricsCollector
from sglang.srt.utils import bind_to_closest_numa_node_cuda

if TYPE_CHECKING:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def _build_mha_host_pool(kv_cache, server_args, page_size: int):
    """Create the normal host cache unless an explicit Central I/O socket exists."""
    socket_path = os.getenv("SGLANG_CENTRAL_IO_SOCKET")
    if socket_path:
        return CentralIOMHATokenToKVPoolHost(
            kv_cache,
            server_args.hicache_ratio,
            server_args.hicache_size,
            page_size,
            server_args.hicache_mem_layout,
            socket_path=socket_path,
            model_id=os.getenv("SGLANG_CENTRAL_IO_MODEL_ID"),
        )
    return MHATokenToKVPoolHost(
        kv_cache,
        server_args.hicache_ratio,
        server_args.hicache_size,
        page_size,
        server_args.hicache_mem_layout,
        allocator_type=server_args.hicache_storage_backend,
    )


class HiRadixCache(RadixCache):

    def __init__(self, params: CacheInitParams, server_args: ServerArgs):
        self._enable_metrics_flag = params.enable_metrics
        if server_args.hicache_io_backend == "direct":
            # FIXME: move this logic into server_args parsing
            if server_args.hicache_mem_layout == "page_first":
                server_args.hicache_mem_layout = "page_first_direct"
                logger.warning(
                    "Page first layout is not supported with direct IO backend, switching to page first direct layout"
                )

        if not server_args.disable_hicache_numa_detect:
            bind_to_closest_numa_node_cuda()

        self.page_size = params.page_size
        self.kv_cache = params.token_to_kv_pool_allocator.get_kvcache()

        if isinstance(self.kv_cache, MHATokenToKVPool):
            self.token_to_kv_pool_host = _build_mha_host_pool(
                self.kv_cache, server_args, self.page_size
            )
        elif isinstance(self.kv_cache, NSATokenToKVPool):
            self.token_to_kv_pool_host = NSATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            )
        elif isinstance(self.kv_cache, MLATokenToKVPool):
            self.token_to_kv_pool_host = MLATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            )
        else:
            raise ValueError(f"HiRadixCache only supports MHA and MLA yet")

        self.tp_group = params.tp_cache_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)
        self.pp_rank = params.pp_rank
        self.pp_size = params.pp_size
        self.enable_storage = server_args.hicache_storage_backend is not None
        self.enable_storage_metrics = self.enable_storage and params.enable_metrics
        self.extra_metric_labels = server_args.extra_metric_labels

        (
            extra_config,
            prefetch_threshold,
            prefetch_timeout_base,
            prefetch_timeout_per_ki_token,
            hicache_storage_pass_prefix_keys,
        ) = self._parse_storage_backend_extra_config(
            server_args.hicache_storage_backend_extra_config
        )
        # TODO: support more timeout check functions
        self.is_prefetch_timeout = self._prefetch_timeout_check_linear_func
        self.prefetch_stop_policy = server_args.hicache_storage_prefetch_policy

        self.load_cache_event = threading.Event()
        self.cache_controller = HiCacheController(
            params.token_to_kv_pool_allocator,
            self.token_to_kv_pool_host,
            self.page_size,
            self.tp_group,
            load_cache_event=self.load_cache_event,
            write_policy=server_args.hicache_write_policy,
            io_backend=server_args.hicache_io_backend,
            storage_backend=server_args.hicache_storage_backend,
            prefetch_threshold=prefetch_threshold,
            model_name=server_args.served_model_name,
            storage_backend_extra_config=extra_config,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            enable_storage_metrics=self.enable_storage_metrics,
        )
        self._apply_storage_runtime_config(
            storage_backend=server_args.hicache_storage_backend,
            prefetch_threshold=prefetch_threshold,
            prefetch_timeout_base=prefetch_timeout_base,
            prefetch_timeout_per_ki_token=prefetch_timeout_per_ki_token,
            hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
            enable_storage=self.enable_storage,
            enable_storage_metrics=self.enable_storage_metrics,
            extra_metric_labels=self.extra_metric_labels,
        )

        # record the nodes with ongoing write through
        self.ongoing_write_through = {}
        # record the node segments with ongoing load back
        self.ongoing_load_back = {}
        # record the ongoing prefetch requests
        self.ongoing_prefetch = {}
        self.ongoing_backup = {}
        # Central I/O storage writes do not use the upstream controller's
        # tensor-backed StorageOperation.  They are page-atomic agent writes
        # and are polled here until the agent reports durable completion.
        self._central_storage_backups = {}
        # Durable lower-tier residency belongs to a physical central-pool
        # page, not to whichever radix node currently covers that page.  A
        # radix split can change node boundaries without moving its host KV.
        self._central_durable_host_pages: set[int] = set()
        # track per-request tokens loaded from storage (L3 hits)
        # key: request_id, value: number of tokens actually loaded from storage
        self.prefetch_loaded_tokens_by_reqid: dict[str, int] = {}
        # todo: dynamically adjust the threshold
        self.write_through_threshold = (
            1 if server_args.hicache_write_policy == "write_through" else 2
        )
        self.load_back_threshold = 10

        # Detach storage backend automatically on process shutdown
        atexit.register(self.shutdown)

        self.evictable_host_leaves = set()
        # A live quota shrink may need to wait until ordinary cache activity
        # makes a full host page reclaimable.  Keep a small event generation
        # rather than rescanning the radix tree on every HiCache poll.
        self._central_host_leaf_epoch = 0
        self._central_quota_wait_signature = None
        self._central_quota_next_retry_at = 0.0
        self._central_grow_wait_target = None
        self._central_grow_next_retry_at = 0.0
        # Keep candidate state warm during ordinary cache lifecycle events.
        # A shrink therefore consumes a queue that already exists instead of
        # materializing every host slot from the whole radix tree.
        self._central_live_reclaim_queue = PersistentLeafReclaimIndex(
            page_size=self.page_size,
            requires_durable_storage=self.enable_storage,
        )
        self._central_live_reclaim_nodes = {}
        self._central_live_reclaim_queue_epoch = None

        # Pin budget: max tokens that can be pinned = ratio * host pool capacity.
        pin_ratio = envs.SGLANG_HICACHE_MAX_PINNED_RATIO.get()
        if pin_ratio < 0 or pin_ratio >= 1:
            raise ValueError(
                f"SGLANG_HICACHE_MAX_PINNED_RATIO must be in [0, 1), got {pin_ratio}"
            )
        self._max_pinned_tokens = int(self.token_to_kv_pool_host.size * pin_ratio)
        self.pinned_size_ = 0
        logger.info(
            "Pin budget: %d tokens (ratio=%.3f)", self._max_pinned_tokens, pin_ratio
        )

        super().__init__(params=params)

    def shutdown(self):
        """Best-effort auto-detach of storage backend on process shutdown.

        This keeps startup and runtime behavior consistent: if a backend was attached
        (either via CLI args or via admin API), we attempt to detach it on exit.
        """
        try:
            if self.enable_storage:
                self.detach_storage_backend()
        except Exception:
            logger.exception("Failed to detach storage backend on process shutdown.")

    def _apply_storage_runtime_config(
        self,
        *,
        storage_backend: Optional[str],
        prefetch_threshold: int,
        prefetch_timeout_base: float,
        prefetch_timeout_per_ki_token: float,
        hicache_storage_pass_prefix_keys: bool,
        enable_storage: bool,
        enable_storage_metrics: bool,
        extra_metric_labels: Optional[Dict[str, str]],
    ) -> None:
        prefetch_timeout_per_page = (
            self.page_size / 1024 * prefetch_timeout_per_ki_token
        )

        self.enable_storage = enable_storage
        self.prefetch_threshold = prefetch_threshold
        self.prefetch_timeout_base = prefetch_timeout_base
        self.prefetch_timeout_per_page = prefetch_timeout_per_page
        self.hicache_storage_pass_prefix_keys = hicache_storage_pass_prefix_keys
        self.enable_storage_metrics = enable_storage_metrics

        if self.enable_storage_metrics:
            labels = {
                "storage_backend": storage_backend,
                "tp_rank": self.cache_controller.tp_rank,
                "dp_rank": self.cache_controller.dp_rank,
                "pp_rank": self.cache_controller.pp_rank,
                "pp_size": self.cache_controller.pp_size,
            }
            if extra_metric_labels:
                labels.update(extra_metric_labels)
            existing_collector = getattr(self, "storage_metrics_collector", None)
            if existing_collector is None:
                self.storage_metrics_collector = StorageMetricsCollector(labels=labels)
            elif set(existing_collector.labels.keys()) == set(labels.keys()):
                existing_collector.labels = labels
            else:
                logger.warning(
                    "Storage metrics labels changed (%s -> %s). Keep existing labels to "
                    "avoid duplicate metric registration.",
                    sorted(existing_collector.labels.keys()),
                    sorted(labels.keys()),
                )

    def attach_storage_backend(
        self,
        storage_backend: str,
        storage_backend_extra_config_json: Optional[str] = None,
        served_model_name: Optional[str] = None,
        hicache_storage_prefetch_policy: Optional[str] = None,
        hicache_write_policy: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Attach (enable) storage backend at runtime.

        This will start storage threads inside `HiCacheController` and enable
        prefetch/backup paths. Caller must ensure there are no running/queued
        requests to avoid races.
        """
        # Validate inputs first (no side effects).
        if hicache_storage_prefetch_policy is not None:
            allowed = ["best_effort", "wait_complete", "timeout"]
            if hicache_storage_prefetch_policy not in allowed:
                return (
                    False,
                    f"Invalid hicache_storage_prefetch_policy: {hicache_storage_prefetch_policy!r}. "
                    f"Expected one of {allowed}.",
                )

        if hicache_write_policy is not None:
            allowed = ["write_back", "write_through", "write_through_selective"]
            if hicache_write_policy not in allowed:
                return (
                    False,
                    f"Invalid hicache_write_policy: {hicache_write_policy!r}. "
                    f"Expected one of {allowed}.",
                )

        # If already enabled:
        # - backend unchanged: treat as success, update policies only.
        # - backend changed: treat as failure, do NOT update policies.
        if self.enable_storage:
            current_backend = self.cache_controller.storage_backend_type

            if current_backend == storage_backend:
                if hicache_storage_prefetch_policy is not None:
                    self.prefetch_stop_policy = hicache_storage_prefetch_policy
                    logger.info(
                        f"Set hicache_storage_prefetch_policy to {hicache_storage_prefetch_policy}"
                    )
                if hicache_write_policy is not None:
                    self.cache_controller.write_policy = hicache_write_policy
                    self.write_through_threshold = (
                        1 if hicache_write_policy == "write_through" else 2
                    )
                    logger.info(f"Set hicache_write_policy to {hicache_write_policy}")
                return (
                    True,
                    "HiCache storage backend already enabled with same backend; policies updated.",
                )

            return (
                False,
                f"HiCache storage backend is already enabled with backend '{current_backend}'. "
                f"Cannot attach different backend '{storage_backend}'. Detach first.",
            )

        # Not enabled: update policies before controller attach so storage threads observe new values.
        if hicache_storage_prefetch_policy is not None:
            self.prefetch_stop_policy = hicache_storage_prefetch_policy
            logger.info(
                f"Set hicache_storage_prefetch_policy to {hicache_storage_prefetch_policy}"
            )

        if hicache_write_policy is not None:
            self.cache_controller.write_policy = hicache_write_policy
            self.write_through_threshold = (
                1 if hicache_write_policy == "write_through" else 2
            )
            logger.info(f"Set hicache_write_policy to {hicache_write_policy}")

        logger.info(f"Attaching HiCache storage backend: {storage_backend}")
        try:
            (
                extra_config,
                prefetch_threshold,
                prefetch_timeout_base,
                prefetch_timeout_per_ki_token,
                hicache_storage_pass_prefix_keys,
            ) = self._parse_storage_backend_extra_config(
                storage_backend_extra_config_json
            )
        except Exception as e:
            logger.exception(f"Failed to parse storage_backend_extra_config_json: {e}")
            return (
                False,
                f"Failed to parse storage_backend_extra_config_json '{storage_backend_extra_config_json}': {e}",
            )

        try:
            self.cache_controller.attach_storage_backend(
                storage_backend=storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=served_model_name,
                storage_backend_extra_config=extra_config,
            )
        except Exception as e:
            logger.exception(
                f"Failed to attach storage backend '{storage_backend}': {e}"
            )
            return False, f"Failed to attach storage backend '{storage_backend}': {e}"

        self._apply_storage_runtime_config(
            storage_backend=storage_backend,
            prefetch_threshold=prefetch_threshold,
            prefetch_timeout_base=prefetch_timeout_base,
            prefetch_timeout_per_ki_token=prefetch_timeout_per_ki_token,
            hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
            enable_storage=True,
            enable_storage_metrics=self._enable_metrics_flag,
            extra_metric_labels=self.extra_metric_labels,
        )
        return True, "Attached HiCache storage backend successfully."

    def detach_storage_backend(self) -> tuple[bool, str]:
        """Detach (disable) storage backend at runtime.

        Caller must ensure there are no running/queued requests to avoid races.
        """
        try:
            # Drain any pending control queues before tearing down storage threads/backend.
            # IMPORTANT: this must happen before we clear `ongoing_*`, otherwise acks/releases
            # cannot be matched to nodes and may leak host pages / locks.
            self._drain_storage_control_queues_local()
            # Idempotent detach: always ask controller to best-effort cleanup, even if
            # `self.enable_storage` is already False (may be leftover state from a
            # previous partial detach).
            self.cache_controller.detach_storage_backend()
        except Exception as e:
            logger.exception("Failed to detach storage backend.")
            # Do NOT crash the server for admin operations. Return failure with detail.
            return False, f"Failed to detach HiCache storage backend: {e}"

        # Best-effort cleanup of any leftover bookkeeping.
        self._drain_storage_control_queues_local()
        # After controller threads are fully stopped, it's safe to force-release any
        # leftover pending ops (e.g., async prefetch/backup that didn't get a revoke/ack).
        self._force_release_pending_storage_ops()

        self.enable_storage = False
        self.enable_storage_metrics = False
        return True, "Detached HiCache storage backend successfully."

    def _force_release_pending_storage_ops(self):
        """Force release any leftover pending prefetch/backup bookkeeping.

        This is a safety net for detach/shutdown paths. It assumes storage threads
        have been stopped already (via controller.detach), so no concurrent access
        to these structures should happen.
        """
        cc = self.cache_controller

        # Force release leftover prefetch ops: free pre-allocated host pages and
        # drop the host protection on the matched prefix node.
        try:
            for req_id, info in list(self.ongoing_prefetch.items()):
                try:
                    last_host_node, token_ids, host_indices, _operation = info
                except Exception:
                    # Unexpected shape; just drop it.
                    self.ongoing_prefetch.pop(req_id, None)
                    continue

                try:
                    if host_indices is not None:
                        cc.mem_pool_host.free(host_indices)
                except Exception:
                    logger.exception(
                        "Failed to free host indices for prefetch %s", req_id
                    )

                try:
                    last_host_node.release_host()
                except Exception:
                    logger.exception(
                        "Failed to release host protection for prefetch %s", req_id
                    )

                try:
                    cc.prefetch_tokens_occupied -= len(token_ids)
                    if cc.prefetch_tokens_occupied < 0:
                        cc.prefetch_tokens_occupied = 0
                except Exception:
                    pass

                self.ongoing_prefetch.pop(req_id, None)
        except Exception:
            logger.exception("Force release pending prefetch ops failed.")

        # Force release leftover backup ops: drop host protection on nodes.
        # A detached controller cannot have delivered a durable-storage ack.
        # Keep the host copy retryable; never mistake an abandoned write for a
        # lower-tier copy that permits host-KV deletion.
        try:
            for operation_id, entry in list(
                getattr(self, "_central_storage_backups", {}).items()
            ):
                try:
                    entry["node"].release_host()
                except Exception:
                    logger.exception(
                        "Failed to release Central I/O storage backup %s", operation_id
                    )
                if getattr(entry["node"], "_latticekv_storage_state", "none") == "writing":
                    entry["node"]._latticekv_storage_state = "retry"
                self._observe_central_reclaim_node(entry["node"])
                self._central_storage_backups.pop(operation_id, None)
            for ack_id, node in list(self.ongoing_backup.items()):
                try:
                    node.release_host()
                except Exception:
                    logger.exception(
                        "Failed to release host protection for backup op %s", ack_id
                    )
                if getattr(node, "_latticekv_storage_state", "none") == "writing":
                    node._latticekv_storage_state = "retry"
                self._observe_central_reclaim_node(node)
                self.ongoing_backup.pop(ack_id, None)
        except Exception:
            logger.exception("Force release pending backup ops failed.")

    def _drain_storage_control_queues_local(self):
        """Drain storage control queues without TP synchronization.

        This is intended for shutdown/detach paths where we want to make best-effort
        cleanup even if queue sizes temporarily differ across ranks.
        """
        self._drain_storage_control_queues_impl(
            n_revoke=None,
            n_backup=None,
            n_release=None,
            log_metrics=False,
        )

    def _drain_storage_control_queues_impl(
        self,
        n_revoke: Optional[int],
        n_backup: Optional[int],
        n_release: Optional[int],
        log_metrics: bool,
    ):
        cc = self.cache_controller

        def _drain_queue(q, limit: Optional[int]):
            drained = 0
            while limit is None or drained < limit:
                try:
                    item = q.get_nowait()
                except Empty:
                    break
                drained += 1
                yield item

        def _drain_revoke():
            for req_id in _drain_queue(cc.prefetch_revoke_queue, n_revoke):
                info = self.ongoing_prefetch.pop(req_id, None)
                if info is not None:
                    last_host_node, token_ids, _, _ = info
                    last_host_node.release_host()
                    cc.prefetch_tokens_occupied -= len(token_ids)
                    if cc.prefetch_tokens_occupied < 0:
                        cc.prefetch_tokens_occupied = 0

        def _drain_backup():
            for operation in _drain_queue(cc.ack_backup_queue, n_backup):
                ack_id = operation.id
                entry = self.ongoing_backup.pop(ack_id, None)
                if entry is not None:
                    expected_tokens = len(entry.host_value) if entry.host_value is not None else 0
                    if operation.completed_tokens >= expected_tokens > 0:
                        entry._latticekv_storage_state = "durable"
                    else:
                        # A partial/failed SSD write must never be mistaken
                        # for a durable lower-tier copy. Keep host KV eligible
                        # only for a later retry, not for deletion.
                        entry._latticekv_storage_state = "retry"
                    entry.release_host()
                    self._observe_central_reclaim_node(entry)
                if log_metrics and self.enable_storage_metrics:
                    self.storage_metrics_collector.log_backuped_tokens(
                        operation.completed_tokens
                    )

        def _drain_release():
            host_indices_list = []
            for host_indices in _drain_queue(cc.host_mem_release_queue, n_release):
                host_indices_list.append(host_indices)
            if host_indices_list:
                host_indices = torch.cat(host_indices_list, dim=0)
                cc.mem_pool_host.free(host_indices)

        _drain_revoke()
        _drain_backup()
        _drain_release()

    def _central_node_page_ids(self, node: TreeNode) -> set[int]:
        """Return node pages without turning each host slot into Python state."""
        if node.host_value is None:
            return set()
        values = (
            node.host_value
            if isinstance(node.host_value, (list, tuple))
            else (node.host_value,)
        )
        page_ids: set[int] = set()
        for value in values:
            flat = value.detach().reshape(-1)
            page_ids.update(
                int(page_id)
                for page_id in torch.div(
                    flat, self.page_size, rounding_mode="floor"
                )
                .unique()
                .cpu()
                .tolist()
            )
        return page_ids

    def _central_durable_pages_for(
        self, node: TreeNode, page_ids: set[int]
    ) -> frozenset[int] | None:
        """Return page durability in the correct ownership domain.

        Upstream storage still acknowledges whole radix nodes, so it retains
        the legacy ``None`` marker.  Central I/O owns page bytes and keeps one
        per-instance page ledger instead.  That ledger survives a radix split
        and is explicitly cleared before the physical page is returned to the
        allocator, preventing a later allocation at the same page id from
        inheriting an old SSD copy.
        """
        host_pool = getattr(self, "token_to_kv_pool_host", None)
        if not getattr(host_pool, "_central_io_owns_storage", False):
            return getattr(node, "_latticekv_durable_page_ids", None)
        ledger = getattr(self, "_central_durable_host_pages", None)
        if ledger is None:
            ledger = set()
            self._central_durable_host_pages = ledger
        return frozenset(page_ids.intersection(ledger))

    def _forget_central_durable_pages(self, host_value) -> None:
        """Invalidate durable facts before an old host page may be reused."""
        host_pool = getattr(self, "token_to_kv_pool_host", None)
        if not getattr(host_pool, "_central_io_owns_storage", False):
            return
        if host_value is None:
            return
        values = host_value if isinstance(host_value, (list, tuple)) else (host_value,)
        page_ids: set[int] = set()
        for value in values:
            flat = value.detach().reshape(-1)
            page_ids.update(
                int(page_id)
                for page_id in torch.div(
                    flat, self.page_size, rounding_mode="floor"
                )
                .unique()
                .cpu()
                .tolist()
            )
        getattr(self, "_central_durable_host_pages", set()).difference_update(page_ids)

    def _drain_central_storage_backups(self) -> None:
        """Turn agent page acknowledgements into page-accurate node state.

        The old controller only exposes a node-wide ``completed_tokens``
        acknowledgement.  Central I/O instead returns a durable write for a
        concrete list of physical pages.  A node becomes durable only if that
        list covers every page the node currently occupies; otherwise it
        remains a host-only partial node and cannot be reclaimed to storage.
        """
        host_pool = getattr(self, "token_to_kv_pool_host", None)
        status_fn = getattr(host_pool, "agent_storage_status", None)
        if status_fn is None:
            return
        pending = getattr(self, "_central_storage_backups", {})
        for operation_id, entry in list(pending.items()):
            node = entry["node"]
            try:
                status = status_fn(operation_id)
            except Exception:
                logger.exception(
                    "Failed to poll Central I/O storage backup %s", operation_id
                )
                continue
            state = status.get("state")
            if state in ("writing", "reading"):
                continue
            pending.pop(operation_id, None)
            cancelled = bool(
                getattr(node, "_latticekv_storage_reclaim_cancelled", False)
            )
            if (
                not cancelled
                and state == "durable"
                and status.get("completed_pages") == len(entry["page_ids"])
            ):
                ledger = getattr(self, "_central_durable_host_pages", None)
                if ledger is None:
                    ledger = set()
                    self._central_durable_host_pages = ledger
                ledger.update(entry["page_ids"])
                node_page_ids = self._central_node_page_ids(node)
                durable_pages = frozenset(node_page_ids.intersection(ledger))
                node._latticekv_durable_page_ids = durable_pages
                node._latticekv_storage_state = (
                    "durable"
                    if node_page_ids.issubset(durable_pages)
                    else "partial"
                )
            else:
                node._latticekv_storage_state = "retry"
            self._write_central_storage_audit(
                host_pool,
                event=(
                    "write_durable"
                    if not cancelled and state == "durable"
                    else "write_cancelled" if cancelled else "write_failed"
                ),
                operation_id=operation_id,
                node_id=node.id,
                page_ids=list(entry["page_ids"]),
                agent_state=state,
                completed_pages=int(status.get("completed_pages", 0)),
                error=status.get("error"),
            )
            node._latticekv_storage_reclaim_cancelled = False
            node.release_host()
            self._observe_central_reclaim_node(node)

    def _parse_storage_backend_extra_config(
        self, storage_backend_extra_config: Optional[str]
    ):
        """
        Parse storage backend extra config JSON and extract specific parameters.

        Args:
            storage_backend_extra_config: JSON string containing extra configuration

        Returns:
            tuple: (extra_config_dict, prefetch_threshold, prefetch_timeout_base, prefetch_timeout_per_ki_token, hicache_storage_pass_prefix_keys)
        """
        # Parse extra config if provided. Extra config can be a JSON string or a json/toml/yaml file path prefixed with "@".
        extra_config = {}
        if storage_backend_extra_config:
            try:
                if storage_backend_extra_config.startswith("@"):
                    # Read config from a json/toml/yaml file
                    path = storage_backend_extra_config[1:]
                    ext = os.path.splitext(path)[1].lower()
                    with open(path, "rb" if ext == ".toml" else "r") as f:
                        if ext == ".json":
                            extra_config = json.load(f)
                        elif ext == ".toml":
                            import tomllib

                            extra_config = tomllib.load(f)
                        elif ext in (".yaml", ".yml"):
                            import yaml

                            extra_config = yaml.safe_load(f)
                        else:
                            raise ValueError(
                                f"Unsupported config file {path} (config format: {ext})"
                            )
                else:
                    # read config from JSON string
                    extra_config = json.loads(storage_backend_extra_config)
            except Exception as e:
                logger.error(f"Invalid backend extra config JSON: {e}")
                raise e

        prefetch_threshold = extra_config.pop("prefetch_threshold", 256)  # tokens
        prefetch_timeout_base = extra_config.pop("prefetch_timeout_base", 1)  # seconds
        prefetch_timeout_per_ki_token = extra_config.pop(
            "prefetch_timeout_per_ki_token", 0.25
        )  # seconds per 1024 tokens
        hicache_storage_pass_prefix_keys = extra_config.pop(
            "hicache_storage_pass_prefix_keys", False
        )

        if not isinstance(prefetch_threshold, int):
            raise ValueError(
                f"prefetch_threshold must be int, got {type(prefetch_threshold).__name__}"
            )
        if not isinstance(prefetch_timeout_base, (int, float)):
            raise ValueError(
                f"prefetch_timeout_base must be number, got {type(prefetch_timeout_base).__name__}"
            )
        if not isinstance(prefetch_timeout_per_ki_token, (int, float)):
            raise ValueError(
                f"prefetch_timeout_per_ki_token must be number, got {type(prefetch_timeout_per_ki_token).__name__}"
            )
        if not isinstance(hicache_storage_pass_prefix_keys, bool):
            raise ValueError(
                "hicache_storage_pass_prefix_keys must be bool, got "
                f"{type(hicache_storage_pass_prefix_keys).__name__}"
            )

        return (
            extra_config,
            prefetch_threshold,
            float(prefetch_timeout_base),
            float(prefetch_timeout_per_ki_token),
            hicache_storage_pass_prefix_keys,
        )

    def reset(self):
        TreeNode.counter = 0
        self.cache_controller.reset()
        self.token_to_kv_pool_host.clear()
        # Clear per-request tracking dicts
        self.prefetch_loaded_tokens_by_reqid.clear()
        self.evictable_host_leaves.clear()
        self._central_host_leaf_epoch = 0
        self._central_quota_wait_signature = None
        self._central_quota_next_retry_at = 0.0
        self._central_grow_wait_target = None
        self._central_grow_next_retry_at = 0.0
        self._central_live_reclaim_queue = PersistentLeafReclaimIndex(
            page_size=self.page_size,
            requires_durable_storage=self.enable_storage,
        )
        self._central_live_reclaim_nodes = {}
        self._central_live_reclaim_queue_epoch = None
        self.pinned_size_ = 0
        super().reset()

    def get_height(self, node: TreeNode):
        height = 0
        while node != self.root_node:
            node = node.parent
            height += 1
        return height

    def clear_storage_backend(self) -> bool:
        if self.enable_storage:
            try:
                # Check if the storage backend has a clear method (for nixl backends)
                if hasattr(self.cache_controller.storage_backend, "clear"):
                    self.cache_controller.storage_backend.clear()
                    logger.info(
                        "Hierarchical cache storage backend cleared successfully!"
                    )
                    return True
                else:
                    logger.warning(
                        f"Storage backend {type(self.cache_controller.storage_backend).__name__} does not support clear operation."
                    )
                    return False
            except Exception as e:
                logger.error(f"Failed to clear hierarchical cache storage backend: {e}")
                return False
        else:
            logger.warning("Hierarchical cache storage backend is not enabled.")
            return False

    def write_backup(self, node: TreeNode, write_back=False):
        host_pool = self.token_to_kv_pool_host
        required_clean_pages = (len(node.value) + self.page_size - 1) // self.page_size
        # Admission must be proactive.  Calling the maintainer only after
        # ``cache_controller.write`` fails turns low/high watermarks into an
        # after-the-fact eviction path.  Before this HBM batch asks for host
        # pages, give the persistent value-aware queue one bounded chance to
        # recover enough clean capacity.  The call is a no-op while the local
        # state is healthy, so ordinary backups retain SGLang's fast path.
        status_reader = getattr(host_pool, "local_residency_status", None)
        if status_reader is not None:
            status = status_reader()
            if (
                status["action"] in ("prepare", "emergency", "quota_deficit")
                or status["clean_pages"] < required_clean_pages
            ):
                self._maintain_local_residency(
                    required_clean_pages=required_clean_pages
                )
        host_indices = self.cache_controller.write(
            device_indices=node.value,
            node_id=node.id,
        )
        if host_indices is None:
            # HBM needs a host landing page now.  Prefer the instance's
            # lifecycle-maintained low-loss queue before falling back to
            # SGLang's generic host eviction; this preserves the quota's
            # normal reuse policy even on an admission burst.
            self._maintain_local_residency(
                required_clean_pages=(len(node.value) + self.page_size - 1)
                // self.page_size
            )
            host_indices = self.cache_controller.write(
                device_indices=node.value,
                node_id=node.id,
            )
        if host_indices is None:
            # The value-aware local maintainer could not make enough clean
            # pages before this HBM backup.  Keep that fact visible to the
            # global layer even though SGLang's generic eviction below may
            # let this one write succeed.  Otherwise a real admission
            # failure disappears before the scheduler can distinguish it
            # from an ordinary allocator miss.
            record_fallback = getattr(
                host_pool, "record_fallback_admission_eviction", None
            )
            if record_fallback is not None:
                record_fallback(len(node.value))
            record_eviction = getattr(host_pool, "record_host_eviction", None)
            if record_eviction is not None:
                record_eviction(len(node.value))
            publish_residency = getattr(host_pool, "publish_local_residency", None)
            if publish_residency is not None:
                publish_residency(force=True)
            self.evict_host(len(node.value))
            host_indices = self.cache_controller.write(
                device_indices=node.value,
                node_id=node.id,
            )
        if host_indices is not None:
            node.host_value = host_indices
            assert len(node.host_value) > 0
            self._observe_central_reclaim_node(node)
            self.ongoing_write_through[node.id] = node
            if not write_back:
                # no need to lock nodes if write back
                self.inc_lock_ref(node)
        else:
            return 0

        # Allocation refreshes the local residency state.  A batch that was
        # safely admitted can still consume the clean envelope past ``low``;
        # start one cooperative recovery wave now so the *next* backup does
        # not become the first opportunity to notice the deficit.  This is
        # not an admission retry: the current batch is already resident, so
        # the worker may recover toward its normal loss-aware target under
        # the existing bounded reclaim budget.
        if status_reader is not None and status_reader()["action"] in (
            "prepare",
            "emergency",
        ):
            self._maintain_local_residency(required_clean_pages=0)

        return len(host_indices)

    def write_backup_storage(self, node: TreeNode):
        storage_state = getattr(node, "_latticekv_storage_state", "none")
        if storage_state in ("writing", "durable", "partial"):
            return
        host_pool = self.token_to_kv_pool_host
        central_write = getattr(host_pool, "begin_agent_storage_write", None)
        if central_write is not None:
            # A prior host hit may have invalidated an earlier asynchronous
            # write. This is a new, cold reclamation attempt.
            node._latticekv_storage_reclaim_cancelled = False
            node._latticekv_storage_state = "writing"
            try:
                ticket = central_write(node.hash_value, node.host_value)
            except Exception as error:
                node._latticekv_storage_state = "retry"
                self._write_central_storage_audit(
                    host_pool,
                    event="write_start_failed",
                    node_id=node.id,
                    error=repr(error),
                )
                raise
            if ticket is None:
                # The node ends inside a page.  It has no page-complete
                # lower-tier representation and must remain a host candidate.
                node._latticekv_durable_page_ids = frozenset()
                node._latticekv_storage_state = "partial"
                self._write_central_storage_audit(
                    host_pool,
                    event="write_skipped_partial",
                    node_id=node.id,
                )
                self._observe_central_reclaim_node(node)
                return
            node.protect_host()
            self._central_storage_backups[ticket["operation_id"]] = {
                "node": node,
                "page_ids": ticket["page_ids"],
            }
            self._write_central_storage_audit(
                host_pool,
                event="write_started",
                operation_id=ticket["operation_id"],
                node_id=node.id,
                page_ids=list(ticket["page_ids"]),
            )
            self._observe_central_reclaim_node(node)
            return
        prefix_keys = (
            node.get_prefix_hash_values(node.parent)
            if self.hicache_storage_pass_prefix_keys
            else None
        )

        node._latticekv_storage_state = "writing"
        try:
            operation_id = self.cache_controller.write_storage(
                node.host_value, node.key, node.hash_value, prefix_keys
            )
        except Exception:
            node._latticekv_storage_state = "retry"
            raise
        self.ongoing_backup[operation_id] = node
        node.protect_host()
        self._observe_central_reclaim_node(node)

    @staticmethod
    def _storage_writes_on_host_backup() -> bool:
        """Return whether storage mirrors every completed host backup.

        The upstream write-through behavior remains the default.  LatticeKV's
        ``value_aware`` mode instead writes only a leaf selected by local
        reclaim, so SSD is a controlled lower-tier exit rather than an
        unconditional shadow copy of the entire host cache.
        """
        policy = os.getenv(
            "SGLANG_LATTICEKV_STORAGE_ADMISSION", "write_through"
        ).strip().lower()
        if policy in ("write_through", "mirror"):
            return True
        if policy == "value_aware":
            return False
        raise ValueError(
            "SGLANG_LATTICEKV_STORAGE_ADMISSION must be write_through or value_aware"
        )

    def _inc_hit_count(self, node: TreeNode, chunked=False):
        # skip the hit count update for chunked requests
        if self.cache_controller.write_policy == "write_back" or chunked:
            return
        node.hit_count += 1
        self._observe_central_reclaim_node(node)

        if not node.backuped:
            if node.hit_count >= self.write_through_threshold:
                # write to host if the node is not backuped
                self.write_backup(node)

    def writing_check(self, write_back=False):
        if write_back:
            # blocking till all write back complete
            while len(self.ongoing_write_through) > 0:
                for _, finish_event, ack_list in self.cache_controller.ack_write_queue:
                    finish_event.synchronize()
                    for ack_id in ack_list:
                        backuped_node = self.ongoing_write_through.pop(ack_id)
                        if self.enable_storage and self._storage_writes_on_host_backup():
                            self.write_backup_storage(backuped_node)
                self.cache_controller.ack_write_queue.clear()
                assert len(self.ongoing_write_through) == 0
            return

        # NOTE: all ranks has the same ongoing_write_through, can skip sync if empty
        if len(self.ongoing_write_through) == 0:
            return

        finish_count = 0
        for _, finish_event, ack_list in self.cache_controller.ack_write_queue:
            if not finish_event.query():
                break
            finish_count += 1
        queue_size = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        if self.tp_world_size > 1:
            # synchronize TP workers to make the same update to radix cache
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )

        finish_count = int(queue_size.item())
        while finish_count > 0:
            _, finish_event, ack_list = self.cache_controller.ack_write_queue.pop(0)
            finish_event.synchronize()
            for ack_id in ack_list:
                backuped_node = self.ongoing_write_through.pop(ack_id)
                self.dec_lock_ref(backuped_node)
                if self.enable_storage and self._storage_writes_on_host_backup():
                    self.write_backup_storage(backuped_node)
            finish_count -= 1

    def loading_check(self):
        finish_count = 0
        for _, finish_event, ack_list in self.cache_controller.ack_load_queue:
            if not finish_event.query():
                # the KV cache loading is still ongoing
                break
            finish_count += 1
            # no need to sync across TP workers as batch forwarding is synced
            for ack_id in ack_list:
                end_node = self.ongoing_load_back.pop(ack_id)
                self.dec_lock_ref(end_node)

        # ACK until all events are processed
        del self.cache_controller.ack_load_queue[:finish_count]

    def evictable_size(self):
        return self.evictable_size_

    def _is_pinned(self, node: TreeNode) -> bool:
        """Check if a node has an active (non-expired) pin."""
        return node.pin_expiry > 0 and time.monotonic() <= node.pin_expiry

    def _clear_pin(self, node: TreeNode):
        """Clear expired pin state and release host_ref_counter hold."""
        if node.pin_expiry > 0:
            self.pinned_size_ = max(0, self.pinned_size_ - len(node.key))
            node.host_ref_counter = max(0, node.host_ref_counter - 1)
        node.pin_expiry = 0.0
        node.pin_ttl = 0

    def pin_prefix(
        self, token_ids: List[int], ttl_seconds: int = 300
    ) -> Tuple[int, Optional[str]]:
        """Pin nodes along a prefix path. Returns (nodes_pinned, reject_reason)."""
        if self.disable or not token_ids:
            return (0, None)

        key, _ = self.maybe_bigram_convert(self._to_radix_key(token_ids))
        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]
        if len(key) == 0:
            return (0, None)

        expiry = time.monotonic() + ttl_seconds
        nodes_pinned = 0
        budget_exceeded = False
        node = self.root_node
        child_key = self.get_child_key_fn(key)

        while len(key) > 0 and child_key in node.children:
            child = node.children[child_key]
            prefix_len = self.key_match_fn(child.key, key)

            # First pin on this node: check budget, then acquire hold
            if child.pin_expiry == 0:
                if self.pinned_size_ + len(child.key) > self._max_pinned_tokens:
                    budget_exceeded = True
                    break
                child.host_ref_counter += 1
                self.pinned_size_ += len(child.key)

                # Eagerly back up to host so eviction finds pinned nodes
                # already backuped and never enters the write_back drain
                # path, which would leak lock_ref on in-flight
                # write-through entries. No-op under write_back policy.
                self._inc_hit_count(child)

            # Extend expiry and store TTL for refresh-on-hit
            child.pin_expiry = max(child.pin_expiry, expiry)
            child.pin_ttl = max(child.pin_ttl, ttl_seconds)
            nodes_pinned += 1

            if prefix_len < len(child.key):
                break

            node = child
            key = key[prefix_len:]
            if len(key):
                child_key = self.get_child_key_fn(key)

        logger.info(
            "[PIN] pin_prefix: nodes_pinned=%d, ttl=%ds", nodes_pinned, ttl_seconds
        )
        if budget_exceeded:
            msg = f"Pin budget exhausted ({self.pinned_size_}/{self._max_pinned_tokens} tokens pinned)"
            if nodes_pinned == 0:
                return (0, msg)
            return (nodes_pinned, f"prefix partially pinned; {msg}")
        return (nodes_pinned, None)

    def _to_radix_key(self, token_ids: List[int]) -> RadixKey:
        """Convert raw token_ids to a RadixKey for tree walking.

        Must use list (not tuple) to match scheduler's RadixKey format,
        since _key_match_paged compares slices directly and list != tuple.
        """
        return RadixKey(token_ids=list(token_ids))

    def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
        if self.disable:
            return IncLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.key)
                self.protected_size_ += len(node.key)
                delta -= len(node.key)
            node.lock_ref += 1
            self._update_leaf_status(node)
            self._update_host_leaf_status(node)
            node = node.parent
        return IncLockRefResult(delta=delta)

    def dec_lock_ref(
        self, node: TreeNode, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        if self.disable:
            return DecLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.key)
                self.protected_size_ -= len(node.key)
                delta += len(node.key)
            node.lock_ref -= 1
            self._update_leaf_status(node)
            self._update_host_leaf_status(node)
            if node.parent is None:
                assert (
                    node is self.root_node
                ), f"This request holds the node from another tree"
            node = node.parent
        return DecLockRefResult(delta=delta)

    def _update_host_leaf_status(self, node: TreeNode):
        was_evictable = node in self.evictable_host_leaves
        should_be_evictable = node.evicted and node.lock_ref <= 0 and not any(
            child.evicted for child in node.children.values()
        )
        if should_be_evictable and not was_evictable:
            self.evictable_host_leaves.add(node)
            self._central_host_leaf_epoch = (
                getattr(self, "_central_host_leaf_epoch", 0) + 1
            )
        elif was_evictable and not should_be_evictable:
            self.evictable_host_leaves.remove(node)
            self._central_host_leaf_epoch = (
                getattr(self, "_central_host_leaf_epoch", 0) + 1
            )
        self._observe_central_reclaim_node(node)

    def _central_reclaim_index(self):
        """Return the long-lived page candidate index.

        Normal servers create it in ``__init__`` and keep it current through
        cache lifecycle hooks. The small fallback is only for lightweight
        ``__new__`` unit tests and legacy cache objects; it deliberately pays
        one old-style snapshot rather than silently returning an incomplete
        queue.
        """
        index = getattr(self, "_central_live_reclaim_queue", None)
        if isinstance(index, PersistentLeafReclaimIndex):
            return index
        index = PersistentLeafReclaimIndex(
            page_size=self.page_size,
            requires_durable_storage=getattr(self, "enable_storage", False),
        )
        self._central_live_reclaim_queue = index
        self._central_live_reclaim_nodes = {}
        snapshots, nodes = self._live_host_node_snapshots()
        for snapshot in snapshots:
            node = nodes[snapshot.node_id]
            page_ids = {slot // self.page_size for slot in snapshot.host_slots}
            index.upsert(snapshot, page_ids)
            self._central_live_reclaim_nodes[snapshot.node_id] = node
        return index

    def _observe_central_reclaim_node(self, node: TreeNode) -> None:
        """Incrementally refresh one node after an ordinary cache event."""
        index = getattr(self, "_central_live_reclaim_queue", None)
        if not isinstance(index, PersistentLeafReclaimIndex):
            return
        if node.host_value is None:
            index.remove(node.id)
            self._central_live_reclaim_nodes.pop(node.id, None)
            return
        values = (
            node.host_value
            if isinstance(node.host_value, (list, tuple))
            else (node.host_value,)
        )
        page_ids = self._central_node_page_ids(node)
        now = time.monotonic()
        snapshot = HostNodeSnapshot(
            node_id=node.id,
            host_slots=(),
            hit_count=node.hit_count,
            child_count=len(node.children),
            host_ref_counter=node.host_ref_counter,
            evicted=node.evicted,
            lock_ref=node.lock_ref,
            pinned=node.pin_expiry > now,
            parent_id=node.parent.id if node.parent is not None else None,
            last_access_time=float(node.last_access_time),
            is_root=node is self.root_node,
            reuse_key=self._central_reclaim_key(node),
            storage_state=getattr(node, "_latticekv_storage_state", "none"),
            # ``None`` retains the upstream node-wide storage contract.  The
            # Central agent explicitly records a set here, including an empty
            # set for a partial tail, so the reclaim index cannot confuse a
            # partial durable write with a complete lower-tier copy.
            durable_page_ids=self._central_durable_pages_for(node, page_ids),
        )
        index.upsert(snapshot, page_ids)
        self._central_live_reclaim_nodes[node.id] = node

    def _central_reclaim_identity(
        self, node: TreeNode
    ) -> tuple[tuple[int, ...], object | None]:
        """Return the complete radix path tokens and namespace for feedback."""
        cached = getattr(node, "_latticekv_reuse_token_ids", None)
        cached_extra = getattr(node, "_latticekv_reuse_extra_key", None)
        if cached is not None:
            return cached, cached_extra
        path = []
        extra_key = None
        current = node
        while current is not None and current is not self.root_node:
            path.append(current.key.token_ids)
            extra_key = current.key.extra_key
            current = current.parent
        token_ids = tuple(token for segment in reversed(path) for token in segment)
        node._latticekv_reuse_token_ids = token_ids
        node._latticekv_reuse_extra_key = extra_key
        return token_ids, extra_key

    def _central_reclaim_key(self, node: TreeNode) -> str:
        """Fingerprint the complete radix path, independent of node splits."""
        cached = getattr(node, "_latticekv_reuse_key", None)
        if cached is not None:
            return cached
        token_ids, extra_key = self._central_reclaim_identity(node)
        reuse_key = prefix_fingerprint(token_ids, extra_key)
        # A radix split changes the parent's/child's local keys but preserves
        # the child's complete prefix. The identity is therefore stable for
        # this node and avoids rehashing a long document at every lifecycle
        # update.
        node._latticekv_reuse_key = reuse_key
        return reuse_key

    def _record_central_reclaim_revisit(self, node: TreeNode) -> None:
        """Turn a recent reclaimed-prefix re-prefill into local feedback."""
        queue = getattr(self, "_central_live_reclaim_queue", None)
        if not isinstance(queue, PersistentLeafReclaimIndex):
            return
        token_ids, extra_key = self._central_reclaim_identity(node)
        reprefill_tokens = queue.record_revisit(
            reuse_key=self._central_reclaim_key(node),
            token_ids=token_ids,
            extra_key=extra_key,
            reprefill_tokens=len(token_ids),
            now_s=time.monotonic(),
        )
        if not reprefill_tokens:
            return
        host_pool = getattr(self, "token_to_kv_pool_host", None)
        record_loss = getattr(host_pool, "record_retention_loss", None)
        if record_loss is not None:
            record_loss(reprefill_tokens)

    def _record_central_host_hit(self, node: TreeNode, hit_tokens: int) -> None:
        """Feed an actual host restore into the local reclaim ordering.

        A host hit is stronger and more relevant than a process-wide request
        rate: it says this complete prefix path consumed pinned host KV during
        the current turnover window.  It still does not make the node
        permanently protected; the persistent index expires it using the
        instance's dynamically measured host-turnover horizon.
        """
        if hit_tokens <= 0:
            return
        self._cancel_central_storage_reclaim_on_reuse(node)
        queue = getattr(self, "_central_live_reclaim_queue", None)
        if not isinstance(queue, PersistentLeafReclaimIndex):
            return
        queue.record_host_hit(
            reuse_key=self._central_reclaim_key(node),
            hit_tokens=hit_tokens,
            now_s=time.monotonic(),
        )

    def _cancel_central_storage_reclaim_on_reuse(self, node: TreeNode) -> None:
        """Keep a reused host prefix out of an in-flight SSD reclamation.

        Agent writes are deliberately asynchronous.  A host hit that races a
        write must therefore invalidate the pending durable fact before its
        acknowledgement is consumed; otherwise the later ack could make a
        newly valuable prefix immediately eligible for host deletion.  The
        bytes may still finish writing, but the next reclamation requires a
        fresh write after the updated prefix has become cold again.
        """
        if node.host_value is None:
            return
        if not getattr(
            getattr(self, "token_to_kv_pool_host", None),
            "_central_io_owns_storage",
            False,
        ):
            return
        storage_state = getattr(node, "_latticekv_storage_state", "none")
        if storage_state not in {"writing", "durable", "partial"}:
            return
        node._latticekv_storage_reclaim_cancelled = True
        self._write_central_storage_audit(
            self.token_to_kv_pool_host,
            event="write_cancel_requested_by_reuse",
            node_id=node.id,
            prior_state=storage_state,
            page_ids=sorted(self._central_node_page_ids(node)),
        )
        node._latticekv_storage_state = "retry"
        node._latticekv_durable_page_ids = frozenset()
        self._forget_central_durable_pages(node.host_value)
        self._observe_central_reclaim_node(node)

    def _central_quota_wait_key(self, host_pool, target: int) -> tuple[int, int, int]:
        """Summarize cache facts that can make a deferred drain progress."""
        return (
            target,
            host_pool.available_size(),
            getattr(self, "_central_host_leaf_epoch", 0),
        )

    def _clear_central_quota_wait(self) -> None:
        self._central_quota_wait_signature = None
        self._central_quota_next_retry_at = 0.0

    def _central_clean_epoch(self, host_pool) -> int:
        """Read the agent epoch that changes after every completed scrub batch."""
        try:
            return int(host_pool.quota_status().get("clean_epoch", 0))
        except AttributeError:
            # Small unit-test pools intentionally have no Central client.
            return 0

    def _defer_central_grow(self, host_pool, target: int) -> bool:
        """Throttle unchanged grow retries while waking on a newly clean batch."""
        now = time.monotonic()
        wait_key = (target, self._central_clean_epoch(host_pool))
        if (
            getattr(self, "_central_grow_wait_target", None) == wait_key
            and now < getattr(self, "_central_grow_next_retry_at", 0.0)
        ):
            return True
        self._central_grow_wait_target = wait_key
        # The agent increments clean_epoch at asynchronous scrub completion.
        # Keep only a short fallback throttle for an idle server; a one-second
        # delay here dominated the old end-to-end handoff time.
        self._central_grow_next_retry_at = now + float(
            os.getenv("SGLANG_CENTRAL_IO_QUOTA_RETRY_S", "0.05")
        )
        return False

    def _clear_central_grow_wait(self) -> None:
        self._central_grow_wait_target = None
        self._central_grow_next_retry_at = 0.0

    def _central_quota_batch_slots(self, host_pool, remaining_slots: int) -> int:
        """Return one page-aligned incremental handoff quantum.

        Scheduler targets may be tens of GiB, but neither donor drain nor hot
        grow should wait for the full target.  This is a host-allocator unit:
        HiRadix and HostKVCache can use different page coordinates.
        """
        host_page_size = max(1, int(getattr(host_pool, "page_size", self.page_size)))
        batch_gib = os.getenv("SGLANG_CENTRAL_IO_QUOTA_BATCH_GIB")
        if batch_gib is not None:
            raw_slots = int(float(batch_gib) * (1024**3) // host_pool.size_per_token)
            slots = max(host_page_size, raw_slots - raw_slots % host_page_size)
        else:
            pages = max(
                1, int(os.getenv("SGLANG_CENTRAL_IO_QUOTA_BATCH_PAGES", "128"))
            )
            slots = pages * host_page_size
        return min(remaining_slots, slots)

    def _defer_live_page_drain(self, host_pool, target: int) -> bool:
        """Avoid rescanning unchanged host state while a drain is pending.

        A normal host free or a leaf becoming evictable changes the key and
        retries immediately.  The short fallback timer is only a liveness
        guard for state changes not yet represented by those two signals.
        """
        now = time.monotonic()
        key = self._central_quota_wait_key(host_pool, target)
        if (
            getattr(self, "_central_quota_wait_signature", None) == key
            and now < getattr(self, "_central_quota_next_retry_at", 0.0)
        ):
            return True
        self._central_quota_wait_signature = key
        self._central_quota_next_retry_at = now + float(
            os.getenv("SGLANG_CENTRAL_IO_QUOTA_RETRY_S", "0.05")
        )
        return False

    def evict(self, params: EvictParams) -> EvictResult:
        start_time = time.perf_counter()
        num_tokens = params.num_tokens
        leaves = list(self.evictable_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        write_back_nodes = []
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)

            if x.lock_ref > 0:
                continue

            if self._is_pinned(x):
                # Still active: demote to host if possible
                if x.backuped:
                    num_evicted += self._evict_backuped(x)
                    continue
                written = self.write_backup(x, write_back=True)
                if written > 0:
                    num_evicted += written
                    write_back_nodes.append(x)
                    continue  # backup succeeded, pin holds on host
                # Host full -- drop pin so GPU can be freed
                self._clear_pin(x)
                logger.warning(
                    "[PIN] evict: can't backup node %d to host, releasing pin",
                    x.id,
                )
            elif x.pin_expiry > 0:
                # Expired pin: clear and fall through to normal eviction
                self._clear_pin(x)

            if not x.backuped:
                if self.cache_controller.write_policy == "write_back":
                    # write to host if the node is not backuped
                    num_evicted += self.write_backup(x, write_back=True)
                    write_back_nodes.append(x)
                else:
                    num_evicted += self._evict_regular(x)
            else:
                num_evicted += self._evict_backuped(x)

            for child in x.parent.children.values():
                if child in write_back_nodes:
                    continue
                if not child.evicted:
                    break
            else:
                # all children are evicted or no children
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

        if self.cache_controller.write_policy == "write_back":
            self.writing_check(write_back=True)
            for node in write_back_nodes:
                assert node.backuped
                self._evict_backuped(node)

        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)

    def _evict_backuped(self, node: TreeNode):
        # GPU -> CPU demotion: no BlockRemoved since block is still reachable via load_back
        num_evicted = self.cache_controller.evict_device(node.value)
        assert num_evicted > 0
        self.evictable_size_ -= num_evicted
        node.value = None
        self._update_leaf_status(node)
        self._update_host_leaf_status(node)
        # update leaf status for the parent because the node is evicted
        self._update_leaf_status(node.parent)
        return num_evicted

    def _evict_regular(self, node: TreeNode):
        # evict a node not initiated write to host -- emit BlockRemoved
        self._record_remove_event(node)
        self.cache_controller.mem_pool_device_allocator.free(node.value)
        num_evicted = len(node.value)
        self._delete_leaf(node)
        return num_evicted

    def evict_host(self, num_tokens: int):
        leaves = list(self.evictable_host_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)
            if x == self.root_node:
                break
            # only evict the host value of evicted nodes
            if not x.evicted:
                continue

            # Expire stale pins before checking host_ref_counter
            if x.pin_expiry > 0 and time.monotonic() > x.pin_expiry:
                self._clear_pin(x)

            # node is protected from eviction as it has ongoing prefetch, backup, or pin
            if x.host_ref_counter > 0:
                continue

            # Block deleted entirely (GPU already evicted, now CPU freed) --
            # emit BlockRemoved so the router removes this block from its index.
            queue = getattr(self, "_central_live_reclaim_queue", None)
            if isinstance(queue, PersistentLeafReclaimIndex):
                token_ids, extra_key = self._central_reclaim_identity(x)
                queue.mark_reclaimed(
                    reuse_key=self._central_reclaim_key(x),
                    token_ids=token_ids,
                    extra_key=extra_key,
                    now_s=time.monotonic(),
                )
            self._record_remove_event(x)
            self._forget_central_durable_pages(x.host_value)
            num_evicted += self.cache_controller.evict_host(x.host_value)

            key = self.get_child_key_fn(x.key)
            v = x.parent.children.pop(key, None)
            assert v == x, f"parent does not have child key, {key}"
            # Normal host eviction also mutates the persistent live-reclaim
            # index. Without this, a later quota shrink can select a node
            # that has already been detached from the radix tree.
            x.host_value = None
            if isinstance(queue, PersistentLeafReclaimIndex):
                queue.complete_delete(x.id)
                self._central_live_reclaim_nodes.pop(x.id, None)
            if x in self.evictable_host_leaves:
                self.evictable_host_leaves.remove(x)
            self._update_host_leaf_status(x.parent)

            if len(x.parent.children) == 0 and x.parent.evicted:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

    def live_page_reclaim_census(
        self, protected_hit_count: int = 3
    ) -> PageReclaimPlan:
        """Return a read-only page-level host-KV reclaim census.

        Radix nodes are allowed to begin/end inside a SGLang KV page after
        prefix splitting.  We therefore collect node facts first and let the
        page-level policy decide whether every fragment in a page is safe to
        relinquish.  This deliberately does not evict or fence anything; the
        controller uses the census to choose its later drain batch.
        """
        snapshots, _ = self._live_host_node_snapshots()
        return classify_reclaimable_pages(
            page_size=self.page_size,
            protected_hit_count=protected_hit_count,
            nodes=snapshots,
        )

    def _live_host_node_snapshots(
        self,
    ) -> tuple[list[HostNodeSnapshot], dict[int, TreeNode]]:
        """Read host-resident radix facts without changing cache state."""

        def host_slot_ids(host_value) -> tuple[int, ...]:
            """Normalize SGLang's tensor or page-chunked host index form."""
            values = host_value if isinstance(host_value, (list, tuple)) else (host_value,)
            return tuple(
                int(slot)
                for value in values
                for slot in value.detach().cpu().reshape(-1).tolist()
            )

        snapshots: list[HostNodeSnapshot] = []
        nodes: dict[int, TreeNode] = {}
        now = time.monotonic()
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            stack.extend(node.children.values())
            if node.host_value is None:
                continue
            nodes[node.id] = node
            snapshots.append(
                HostNodeSnapshot(
                    node_id=node.id,
                    host_slots=host_slot_ids(node.host_value),
                    hit_count=node.hit_count,
                    child_count=len(node.children),
                    host_ref_counter=node.host_ref_counter,
                    evicted=node.evicted,
                    lock_ref=node.lock_ref,
                    pinned=node.pin_expiry > now,
                    parent_id=node.parent.id if node.parent is not None else None,
                    last_access_time=float(node.last_access_time),
                    is_root=node is self.root_node,
                    reuse_key=self._central_reclaim_key(node),
                    storage_state=getattr(node, "_latticekv_storage_state", "none"),
                    durable_page_ids=self._central_durable_pages_for(
                        node,
                        {
                            slot // self.page_size
                            for slot in host_slot_ids(node.host_value)
                        },
                    ),
                )
            )
        return snapshots, nodes

    def select_live_page_drain(
        self, protected_hit_count: int = 3, max_pages: int | None = None
    ):
        """Return complete host pages and whole low-value leaves to drain.

        The result does not fence or evict anything.  It is safe to call from
        a controller before deciding whether enough capacity can be reclaimed.
        """
        snapshots, _ = self._live_host_node_snapshots()
        return select_page_drain_candidates(
            page_size=self.page_size,
            protected_hit_count=protected_hit_count,
            nodes=snapshots,
            max_pages=max_pages,
        )

    def select_live_page_reclaim_actions(
        self, protected_hit_count: int = 3, max_pages: int | None = None
    ):
        """Plan page-complete host-copy drops and host-only leaf deletions."""
        snapshots, _ = self._live_host_node_snapshots()
        return select_page_reclaim_actions(
            page_size=self.page_size,
            protected_hit_count=protected_hit_count,
            nodes=snapshots,
            max_pages=max_pages,
        )

    def select_next_incremental_live_reclaim_action(
        self,
        protected_hit_count: int | None = None,
        protected_fraction: float | None = None,
        prefer_ready_pages: bool = False,
    ):
        """Return one leaf from the lifecycle-maintained candidate index."""
        queue = self._central_reclaim_index()
        now_s = time.monotonic()
        if prefer_ready_pages:
            ready = queue.pop_ready_page_action(now_s=now_s)
            if ready.node_id is not None:
                return ready
        return queue.pop(
            protected_hit_count, protected_fraction, now_s=now_s
        )

    def reclaim_live_page_selection(
        self,
        protected_hit_count: int | None = None,
        protected_fraction: float | None = None,
        max_pages: int | None = None,
        selection_budget_ms: float | None = None,
        initial_min_pages: int = 1,
        on_allocator_pages_ready: Callable[[int], int] | None = None,
        prefer_ready_pages: bool = False,
        reclaim_origin: str = "local",
    ) -> dict:
        """Progressively release low-value host KV until pages become free.

        A prior version selected only actions that already formed a complete
        page.  That stranded a low-value child behind its parent: deleting the
        child could have exposed a new leaf, but no action was applied until a
        future cache event happened to change the page.  Here every safe leaf
        is applied immediately.  ``HostKVCache.free`` returns a page to the
        allocator only after its final live slot disappears; the normal
        allocator-free shrink then starts Central I/O's asynchronous scrub.
        """
        if reclaim_origin not in {"local", "donor"}:
            raise ValueError("reclaim_origin must be 'local' or 'donor'")
        host_pool = self.token_to_kv_pool_host
        if not getattr(host_pool, "dynamic_page_leases", False):
            raise RuntimeError("live page reclaim requires dynamic Central I/O leases")
        started_at = time.perf_counter()
        before_free_slots = host_pool.available_size()
        observed_free_slots = before_free_slots
        streamed_slots = 0
        stream_batches = 0
        first_stream_ms: float | None = None
        requested_slots = (
            max_pages * self.page_size
            if max_pages is not None
            else host_pool.active_size
        )
        action_budget = int(
            os.getenv(
                "SGLANG_CENTRAL_IO_LIVE_RECLAIM_MAX_ACTIONS",
                str(max(64, requested_slots * 2)),
            )
        )
        action_budget = max(1, action_budget)
        if selection_budget_ms is not None and selection_budget_ms <= 0:
            raise ValueError("selection_budget_ms must be positive")
        if initial_min_pages <= 0:
            raise ValueError("initial_min_pages must be positive")
        initial_min_slots = initial_min_pages * self.page_size
        evicted_tokens = 0
        drop_ids: list[int] = []
        delete_ids: list[int] = []
        blockers: dict[str, int] = {}
        reason = "action_budget_exhausted"
        census_ms = 0.0
        selection_ms = 0.0
        action_ms = 0.0
        evict_host_ms = 0.0
        record_remove_ms = 0.0
        queue_mark_ms = 0.0
        radix_tree_ms = 0.0
        queue_complete_ms = 0.0
        stream_ms = 0.0
        reclaimed_hits: list[int] = []
        reclaimed_slots = 0
        ssd_write_started = 0
        exposed_parent_ids: set[int] = set()
        newly_exposed_parent_deletions = 0

        def stream_newly_free_pages() -> None:
            """Hand complete pages to Central I/O as soon as a leaf frees them.

            The callback runs inside HiRadix's serialized lifecycle path, so
            it cannot race a node's deletion.  The agent's scrub and the hot
            model's later grow remain asynchronous.
            """
            nonlocal observed_free_slots, streamed_slots, stream_batches, first_stream_ms
            if on_allocator_pages_ready is None:
                return
            newly_free_slots = host_pool.available_size() - observed_free_slots
            newly_free_slots -= newly_free_slots % self.page_size
            if newly_free_slots <= 0:
                return
            # A 2.25MiB one-page handoff does not relieve a large hot-model
            # deficit. Accumulate the first useful wave, then stream every
            # later complete page immediately.
            if streamed_slots == 0 and newly_free_slots < initial_min_slots:
                return
            accepted_slots = int(on_allocator_pages_ready(newly_free_slots))
            if accepted_slots < 0 or accepted_slots > newly_free_slots:
                raise ValueError("page handoff callback returned an invalid slot count")
            if accepted_slots:
                streamed_slots += accepted_slots
                stream_batches += 1
                if first_stream_ms is None:
                    first_stream_ms = (time.perf_counter() - started_at) * 1000
            # In the runtime callback resize_quota consumes the accepted
            # pages. Tests may use a recording callback, so retain any
            # unaccepted/free tail as observable capacity.
            observed_free_slots = host_pool.available_size() - max(
                0, newly_free_slots - accepted_slots
            )

        def reuse_summary(queue) -> dict[str, int]:
            if queue is None:
                return {
                    "host_nodes": 0,
                    "host_slots": 0,
                    "high_reuse_nodes": 0,
                    "high_reuse_slots": 0,
                    "shared_nodes": 0,
                    "feedback": {},
                }
            remaining = [
                node
                for node_id, node in queue.nodes.items()
                if node_id not in queue._finished
            ]
            runtime_nodes = getattr(self, "_central_live_reclaim_nodes", {})

            def host_slot_count(snapshot) -> int:
                runtime_node = runtime_nodes.get(snapshot.node_id)
                if runtime_node is None or runtime_node.host_value is None:
                    return 0
                values = (
                    runtime_node.host_value
                    if isinstance(runtime_node.host_value, (list, tuple))
                    else (runtime_node.host_value,)
                )
                return sum(int(value.numel()) for value in values)

            return {
                "host_nodes": len(remaining),
                "host_slots": sum(host_slot_count(node) for node in remaining),
                "high_reuse_nodes": sum(node.hit_count >= 3 for node in remaining),
                "high_reuse_slots": sum(
                    host_slot_count(node)
                    for node in remaining
                    if node.hit_count >= 3
                ),
                "shared_nodes": sum(
                    queue.remaining_children.get(node.node_id, node.child_count) > 0
                    for node in remaining
                ),
                "feedback": queue.feedback_observability(now_s=time.monotonic()),
            }

        before_summary = None

        for _ in range(action_budget):
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            pending_slots = max(0, host_pool.available_size() - observed_free_slots)
            has_enough_pages = streamed_slots + pending_slots >= requested_slots
            if has_enough_pages and selection_budget_ms is None:
                reason = "allocator_pages_ready"
                break
            if (
                selection_budget_ms is not None
                and elapsed_ms >= selection_budget_ms
                and (streamed_slots > 0 or pending_slots >= initial_min_slots)
            ):
                reason = "initial_wave_budget_reached"
                break
            selection_started = time.perf_counter()
            selection = self.select_next_incremental_live_reclaim_action(
                protected_hit_count,
                protected_fraction,
                prefer_ready_pages=prefer_ready_pages,
            )
            selected_ms = (time.perf_counter() - selection_started) * 1000
            selection_ms += selected_ms
            queue = self._central_live_reclaim_queue
            if before_summary is None:
                before_summary = reuse_summary(queue)
            blockers = selection.blocker_counts
            if selection.node_id is None:
                reason = "no_safe_leaf"
                break

            nodes = self._central_live_reclaim_nodes
            node = nodes.get(selection.node_id)
            if (
                node is None
                or node.host_value is None
                or node.host_ref_counter > 0
                or node.lock_ref > 0
            ):
                reason = "stale_or_protected_node"
                queue.remove(selection.node_id)
                self._central_live_reclaim_nodes.pop(selection.node_id, None)
                break
            action_started = time.perf_counter()
            if node.pin_expiry > 0 and time.monotonic() > node.pin_expiry:
                self._clear_pin(node)
            node_slots = sum(
                int(value.numel())
                for value in (
                    node.host_value
                    if isinstance(node.host_value, (list, tuple))
                    else (node.host_value,)
                )
            )

            if selection.action_kind == "delete_host_leaf" and getattr(
                self, "enable_storage", False
            ):
                storage_state = getattr(node, "_latticekv_storage_state", "none")
                if storage_state != "durable":
                    if storage_state != "writing":
                        self.write_backup_storage(node)
                        ssd_write_started += 1
                    # The SSD operation holds host_ref_counter until its ack.
                    # A future maintenance event will revisit this same leaf
                    # only after the durable state has been confirmed.
                    reason = "waiting_for_ssd"
                    action_ms += (time.perf_counter() - action_started) * 1000
                    continue

            if selection.action_kind == "drop_host_copy":
                if node.evicted or node.children:
                    reason = "stale_or_protected_node"
                    queue.remove(node.id)
                    self._central_live_reclaim_nodes.pop(node.id, None)
                    continue
                evict_started = time.perf_counter()
                if (
                    getattr(self, "enable_storage", False)
                    and getattr(node, "_latticekv_storage_state", None) == "durable"
                ):
                    self._write_central_storage_audit(
                        host_pool,
                        event="host_released_after_durable",
                        node_id=node.id,
                        page_ids=sorted(self._central_node_page_ids(node)),
                        release_kind="drop_host_copy",
                    )
                self._forget_central_durable_pages(node.host_value)
                evicted_tokens += self.cache_controller.evict_host(node.host_value)
                recorder = getattr(
                    host_pool,
                    "record_local_value_reclaim"
                    if reclaim_origin == "local"
                    else "record_donor_drain",
                    None,
                )
                if recorder is not None:
                    recorder(node_slots)
                evict_host_ms += (time.perf_counter() - evict_started) * 1000
                queue_started = time.perf_counter()
                token_ids, extra_key = self._central_reclaim_identity(node)
                queue.mark_reclaimed(
                    reuse_key=self._central_reclaim_key(node),
                    token_ids=token_ids,
                    extra_key=extra_key,
                    now_s=time.monotonic(),
                )
                queue_mark_ms += (time.perf_counter() - queue_started) * 1000
                queue_started = time.perf_counter()
                node.host_value = None
                queue.complete_drop_host_copy(node.id)
                self._central_live_reclaim_nodes.pop(node.id, None)
                queue_complete_ms += (time.perf_counter() - queue_started) * 1000
                drop_ids.append(node.id)
                reclaimed_hits.append(node.hit_count)
                reclaimed_slots += node_slots
                action_ms += (time.perf_counter() - action_started) * 1000
                stream_started = time.perf_counter()
                stream_newly_free_pages()
                stream_ms += (time.perf_counter() - stream_started) * 1000
                continue

            if selection.action_kind != "delete_host_leaf":
                raise RuntimeError("unknown incremental live reclaim action")
            key = self.get_child_key_fn(node.key)
            parent = node.parent
            if (
                not node.evicted
                or node.children
                or parent is None
                or parent.children.get(key) is not node
            ):
                reason = "stale_or_protected_node"
                queue.remove(node.id)
                self._central_live_reclaim_nodes.pop(node.id, None)
                continue
            record_started = time.perf_counter()
            self._record_remove_event(node)
            record_remove_ms += (time.perf_counter() - record_started) * 1000
            queue_started = time.perf_counter()
            token_ids, extra_key = self._central_reclaim_identity(node)
            queue.mark_reclaimed(
                reuse_key=self._central_reclaim_key(node),
                token_ids=token_ids,
                extra_key=extra_key,
                now_s=time.monotonic(),
            )
            queue_mark_ms += (time.perf_counter() - queue_started) * 1000
            evict_started = time.perf_counter()
            if (
                getattr(self, "enable_storage", False)
                and getattr(node, "_latticekv_storage_state", None) == "durable"
            ):
                self._write_central_storage_audit(
                    host_pool,
                    event="host_released_after_durable",
                    node_id=node.id,
                    page_ids=sorted(self._central_node_page_ids(node)),
                    release_kind="delete_host_leaf",
                )
            self._forget_central_durable_pages(node.host_value)
            evicted_tokens += self.cache_controller.evict_host(node.host_value)
            recorder = getattr(
                host_pool,
                "record_local_value_reclaim"
                if reclaim_origin == "local"
                else "record_donor_drain",
                None,
            )
            if recorder is not None:
                recorder(node_slots)
            evict_host_ms += (time.perf_counter() - evict_started) * 1000
            radix_started = time.perf_counter()
            parent_id = parent.id
            value = parent.children.pop(key, None)
            assert value is node, "selected host leaf changed during deletion"
            self.evictable_host_leaves.discard(node)
            self._update_host_leaf_status(parent)
            radix_tree_ms += (time.perf_counter() - radix_started) * 1000
            queue_started = time.perf_counter()
            queue.complete_delete(node.id)
            self._central_live_reclaim_nodes.pop(node.id, None)
            # The deletion can advance the normal host-leaf epoch.  It is a
            # queue-owned change, so preserve the census instead of forcing a
            # new full-tree materialization on the next batch.
            delete_ids.append(node.id)
            reclaimed_hits.append(node.hit_count)
            reclaimed_slots += node_slots
            if node.id in exposed_parent_ids:
                newly_exposed_parent_deletions += 1
            exposed_parent_ids.add(parent_id)
            queue_complete_ms += (time.perf_counter() - queue_started) * 1000
            action_ms += (time.perf_counter() - action_started) * 1000
            stream_started = time.perf_counter()
            stream_newly_free_pages()
            stream_ms += (time.perf_counter() - stream_started) * 1000

        allocator_free_slots = max(0, host_pool.available_size() - observed_free_slots)
        queue = getattr(self, "_central_live_reclaim_queue", None)
        if before_summary is None:
            before_summary = reuse_summary(queue)

        reclaim_summary = {
            "actions": len(drop_ids) + len(delete_ids),
            "drop_host_copy_actions": len(drop_ids),
            "delete_host_leaf_actions": len(delete_ids),
            "newly_exposed_parent_deletions": newly_exposed_parent_deletions,
            "reclaimed_slots": reclaimed_slots,
            "ssd_write_started": ssd_write_started,
            "reclaimed_hit_0": sum(hit == 0 for hit in reclaimed_hits),
            "reclaimed_hit_1_2": sum(0 < hit < 3 for hit in reclaimed_hits),
            "reclaimed_high_reuse": sum(hit >= 3 for hit in reclaimed_hits),
            "before": before_summary,
            "after": reuse_summary(queue),
        }
        result = {
            "page_ranges": [],
            "drop_host_copy_node_ids": drop_ids,
            "delete_host_leaf_node_ids": delete_ids,
            "blocker_counts": blockers,
            "evicted_tokens": evicted_tokens,
            "allocator_free_slots": allocator_free_slots,
            "streamed_slots": streamed_slots,
            "reason": reason,
            "timing_ms": {
                "total": (time.perf_counter() - started_at) * 1000,
                "census": census_ms,
                "selection": selection_ms,
                "actions": action_ms,
                "evict_host": evict_host_ms,
                "radix_queue": (
                    record_remove_ms
                    + queue_mark_ms
                    + radix_tree_ms
                    + queue_complete_ms
                ),
                "record_remove": record_remove_ms,
                "queue_mark": queue_mark_ms,
                "radix_tree": radix_tree_ms,
                "queue_complete": queue_complete_ms,
                "stream": stream_ms,
                "first_stream": first_stream_ms,
            },
            "stream_batches": stream_batches,
            "reuse_summary": reclaim_summary,
        }
        audit_dir = os.getenv("SGLANG_CENTRAL_IO_RECLAIM_AUDIT_DIR")
        if audit_dir:
            os.makedirs(audit_dir, exist_ok=True)
            audit_record = {
                "timestamp_s": time.time(),
                "model_id": getattr(host_pool, "model_id", "unknown"),
                "reason": reason,
                "drop_host_copy_node_ids": drop_ids,
                "delete_host_leaf_node_ids": delete_ids,
                "blocker_counts": blockers,
                "allocator_free_slots": allocator_free_slots,
                "streamed_slots": streamed_slots,
                "stream_batches": stream_batches,
                "timing_ms": result["timing_ms"],
                "reuse_summary": reclaim_summary,
            }
            audit_path = os.path.join(
                audit_dir,
                f"{audit_record['model_id']}-live-reclaim.jsonl",
            )
            with open(audit_path, "a", encoding="utf-8") as audit_file:
                audit_file.write(json.dumps(audit_record, sort_keys=True) + "\n")
        return result

    def _write_central_storage_audit(self, host_pool, *, event: str, **fields) -> None:
        """Append a page-lifecycle event when Central-I/O storage is enabled.

        The agent's operation status is intentionally short-lived control-plane
        state.  Persisting terminal and cancellation transitions alongside the
        existing reclaim audits makes a serving run independently auditable:
        durable acknowledgement must precede host release, and a reuse race
        must be visible as a cancelled reclamation rather than an unexplained
        disappearance of a host page.
        """
        audit_dir = os.getenv("SGLANG_CENTRAL_IO_RECLAIM_AUDIT_DIR")
        if not audit_dir:
            return
        model_id = getattr(host_pool, "model_id", "unknown")
        record = {
            "timestamp_s": time.time(),
            "model_id": model_id,
            "event": event,
            **fields,
        }
        os.makedirs(audit_dir, exist_ok=True)
        audit_path = os.path.join(audit_dir, f"{model_id}-central-storage.jsonl")
        with open(audit_path, "a", encoding="utf-8") as audit_file:
            audit_file.write(json.dumps(record, sort_keys=True) + "\n")

    def _record_central_quota_control(
        self,
        host_pool,
        *,
        event: str,
        target: int,
        **fields,
    ) -> None:
        """Record the safe-point portion of an intent-to-effective handoff.

        The agent owns the target timestamp, while this scheduler thread owns
        the first point at which its allocator may safely change.  Keep these
        events separate from reclaim audits so a serving run can distinguish
        GPU batch residency from donor, scrub, and allocator work.
        """
        audit_dir = os.getenv("SGLANG_CENTRAL_IO_RECLAIM_AUDIT_DIR")
        if not audit_dir:
            return
        record = {
            "schema": "latticekv_quota_control_v1",
            "event": event,
            "monotonic_ns": time.monotonic_ns(),
            "model_id": getattr(host_pool, "model_id", "unknown"),
            "active_slots": int(host_pool.active_size),
            "target_slots": int(target),
            "target_set_ns": getattr(host_pool, "quota_target_set_ns", None),
            **fields,
        }
        os.makedirs(audit_dir, exist_ok=True)
        audit_path = os.path.join(
            audit_dir,
            f"{record['model_id']}-quota-control.jsonl",
        )
        with open(audit_path, "a", encoding="utf-8") as audit_file:
            audit_file.write(json.dumps(record, sort_keys=True) + "\n")

    def _apply_central_quota_target(self) -> dict | None:
        """Advance one model toward the agent-owned target quota.

        This runs in the normal HiCache event path, not in the external
        controller process.  Therefore node fencing and radix eviction remain
        serialized with ordinary cache lifecycle work.
        """
        host_pool = self.token_to_kv_pool_host
        if not getattr(host_pool, "dynamic_page_leases", False):
            return None
        target = host_pool.quota_target()
        if target is None or target == host_pool.active_size:
            self._clear_central_quota_wait()
            self._clear_central_grow_wait()
            return None
        observed_key = (target, getattr(host_pool, "quota_target_set_ns", None))
        if getattr(self, "_central_last_observed_quota_target", None) != observed_key:
            self._central_last_observed_quota_target = observed_key
            self._record_central_quota_control(
                host_pool,
                event="target_observed",
                target=target,
            )
        if target > host_pool.active_size:
            if self._defer_central_grow(host_pool, target):
                return {"direction": "grow", "state": "waiting_for_donor"}
            # A batch exists to make the *first* usable capacity available
            # promptly.  It must not make the remaining clean donor pages
            # wait for unrelated future requests: resize_quota is a small
            # control-plane operation (not a KV copy), so consume all clean
            # batches that fit in one short serialized-event budget.
            apply_budget_ms = float(
                os.getenv("SGLANG_CENTRAL_IO_QUOTA_APPLY_BUDGET_MS", "10.0")
            )
            if apply_budget_ms <= 0:
                raise ValueError(
                    "SGLANG_CENTRAL_IO_QUOTA_APPLY_BUDGET_MS must be positive"
                )
            self._record_central_quota_control(
                host_pool,
                event="recipient_apply_started",
                target=target,
            )
            apply_started = time.perf_counter()
            applied_slots = 0
            batches = 0
            blocked_on_scrub = False
            while host_pool.active_size < target:
                remaining_slots = target - host_pool.active_size
                batch_slots = self._central_quota_batch_slots(
                    host_pool, remaining_slots
                )
                try:
                    host_pool.resize_quota(host_pool.active_size + batch_slots)
                except (MemoryError, RuntimeError) as exc:
                    if isinstance(exc, RuntimeError) and (
                        "insufficient free page-aligned bytes" not in str(exc)
                    ):
                        raise
                    # Donor pages may still be in scrub. Keep the intent and
                    # wake on the next clean epoch rather than pretending the
                    # partial recipient capacity is already the target.
                    blocked_on_scrub = True
                    break
                applied_slots += batch_slots
                batches += 1
                if (time.perf_counter() - apply_started) * 1000 >= apply_budget_ms:
                    break
            if not applied_slots:
                return {"direction": "grow", "state": "waiting_for_scrub"}
            self._clear_central_grow_wait()
            state = "ready" if host_pool.active_size == target else "partial"
            if blocked_on_scrub:
                state = "waiting_for_scrub"
            apply_ms = (time.perf_counter() - apply_started) * 1000
            logger.info(
                "LatticeKV quota grow %s: active=%d target=%d applied_slots=%d "
                "batches=%d apply_ms=%.3f",
                state,
                host_pool.active_size,
                target,
                applied_slots,
                batches,
                apply_ms,
            )
            self._record_central_quota_control(
                host_pool,
                event="recipient_apply_finished",
                target=target,
                state=state,
                applied_slots=applied_slots,
                batches=batches,
                apply_ms=apply_ms,
            )
            if state == "ready":
                self._clear_central_quota_wait()
            return {
                "direction": "grow",
                "state": state,
                "capacity": host_pool.active_size,
                "target": target,
                "applied_slots": applied_slots,
                "batches": batches,
                "apply_ms": apply_ms,
            }

        # A global target is only an intent.  The donor must recheck its own
        # current HBM-admission safety before it detaches even allocator-free
        # pages: a backup or a recent reclaim loss may have arrived after the
        # scheduler sampled its health.  This prevents an old shrink intent
        # from pulling a donor below its recovery watermark.
        requested_pages = (host_pool.active_size - target) // self.page_size
        donation_reader = getattr(host_pool, "local_donation_offer", None)
        donation_offer = None
        if donation_reader is not None:
            donation_offer = donation_reader(requested_pages)
            allowed_pages = int(donation_offer.get("total_pages", 0))
            if allowed_pages <= 0:
                return {
                    "direction": "shrink",
                    "state": "donor_guarded",
                    "capacity": host_pool.active_size,
                    "target": target,
                    "donation_offer": donation_offer,
                }
            # A partial offer is intentional. Keep the original target in
            # the agent so the scheduler can later revoke or continue it,
            # but never detach more pages than this donor can safely spare
            # during the current lifecycle event.
            target = max(target, host_pool.active_size - allowed_pages * self.page_size)

        # Reclaim allocator-free pages first. They contain no radix-resident
        # KV, so returning them is both faster and strictly less destructive
        # than starting a live-node drain. Only the remaining shortfall needs
        # the fence -> evict -> scrub lifecycle below.
        free_slots = min(host_pool.available_size(), host_pool.active_size - target)
        free_slots -= free_slots % self.page_size
        if free_slots:
            host_pool.resize_quota(host_pool.active_size - free_slots)
            if host_pool.active_size == target:
                self._clear_central_quota_wait()
                complete = {
                    "direction": "shrink",
                    "state": "free_pages_ready",
                    "capacity": target,
                }
                if donation_offer is not None:
                    complete["donation_offer"] = donation_offer
                return complete

        if self._defer_live_page_drain(host_pool, target):
            return {"direction": "shrink", "state": "waiting_for_page"}

        required_pages = (host_pool.active_size - target) // self.page_size
        batch_pages = self._central_quota_batch_slots(
            host_pool, required_pages * self.page_size
        ) // self.page_size
        target_changed = getattr(self, "_central_reclaim_target", None) != target
        if target_changed:
            self._central_reclaim_target = target
        # Normal local maintenance and quota shrink must rank the same
        # candidate queue.  A fixed fraction here would protect an arbitrary
        # set of leaves only during shrink, even when recent reclaim feedback
        # has identified different prefixes as expensive.  Keep the fraction
        # only as an explicit diagnostic override.
        protected_fraction_raw = os.getenv(
            "SGLANG_CENTRAL_IO_RECLAIM_PROTECTED_FRACTION"
        )
        protected_fraction = (
            float(protected_fraction_raw)
            if protected_fraction_raw is not None
            else None
        )
        if protected_fraction is not None and not 0.0 <= protected_fraction < 1.0:
            raise ValueError(
                "SGLANG_CENTRAL_IO_RECLAIM_PROTECTED_FRACTION must be in [0, 1)"
            )
        selection_budget_ms = (
            float(os.getenv("SGLANG_CENTRAL_IO_INITIAL_WAVE_BUDGET_MS", "100"))
            if target_changed
            else None
        )

        def handoff_newly_free_pages(newly_free_slots: int) -> int:
            """Begin scrub immediately; do not wait for the reclaim wave."""
            handoff_slots = min(
                newly_free_slots, host_pool.active_size - target
            )
            handoff_slots -= handoff_slots % self.page_size
            if handoff_slots:
                host_pool.resize_quota(host_pool.active_size - handoff_slots)
            return handoff_slots

        result = self.reclaim_live_page_selection(
            protected_fraction=protected_fraction,
            max_pages=batch_pages,
            selection_budget_ms=selection_budget_ms,
            initial_min_pages=batch_pages if selection_budget_ms is not None else 1,
            on_allocator_pages_ready=handoff_newly_free_pages,
            reclaim_origin="donor",
        )
        # Incremental leaf actions may have made one or more complete pages
        # allocator-free.  Start their Central I/O scrub now, in the same
        # serialized HiCache event, rather than waiting for a future poll to
        # rediscover them.  The agent releases the bytes asynchronously; any
        # hot model wakes on its next clean_epoch observation.
        released_slots = min(
            host_pool.available_size(), host_pool.active_size - target
        )
        released_slots -= released_slots % self.page_size
        if released_slots:
            host_pool.resize_quota(host_pool.active_size - released_slots)
            result["async_scrub_slots"] = released_slots
            result["state"] = "scrub_started"
        if result.get("streamed_slots"):
            result["streamed_async_scrub_slots"] = result["streamed_slots"]
            result["state"] = "scrub_started"
        logger.info(
            "LatticeKV quota shrink: active=%d target=%d requested_pages=%d "
            "batch_pages=%d result=%s blockers=%s timing_ms=%s reuse=%s",
            host_pool.active_size,
            target,
            required_pages,
            batch_pages,
            result.get("reason", "started"),
            result.get("blocker_counts", {}),
            result.get("timing_ms", {}),
            result.get("reuse_summary", {}),
        )
        if (
            result.get("allocator_free_slots")
            or result.get("async_scrub_slots")
            or result.get("streamed_async_scrub_slots")
        ):
            self._clear_central_quota_wait()
        if donation_offer is not None:
            result["donation_offer"] = donation_offer
        return {"direction": "shrink", **result}

    def _maintain_local_residency(
        self, *, required_clean_pages: int = 0
    ) -> dict | None:
        """Keep one instance's current host quota usable before any handoff.

        Reclaimed pages remain in this allocator's free ranges: this is not a
        Central I/O lease change and cannot make another model grow.  The
        persistent candidate queue supplies safe leaves without a new radix
        census.  A later global scheduler may inspect the resulting health,
        but it does not select leaves here.
        """
        host_pool = self.token_to_kv_pool_host
        if not getattr(host_pool, "dynamic_page_leases", False):
            return None
        self._refresh_local_ready_pages(host_pool)
        status_reader = getattr(host_pool, "local_residency_status", None)
        if status_reader is None:
            return None
        status = status_reader()
        if required_clean_pages < 0:
            raise ValueError("required_clean_pages must be non-negative")
        if (
            status["action"] not in ("prepare", "emergency", "quota_deficit")
            and status["clean_pages"] >= required_clean_pages
        ):
            return None

        maintenance_target_pages = int(
            status.get("maintenance_target_pages", status["high_pages"])
        )
        requested_pages = max(
            0,
            # Ready candidates are not allocator-free yet.  Keep promoting
            # them until actual clean capacity reaches high; otherwise a
            # large ready set masks the next HBM backup and pushes deletion
            # back onto its synchronous admission path.
            maintenance_target_pages - status["clean_pages"],
            status["min_pages"] - status["clean_pages"],
            required_clean_pages - status["clean_pages"],
        )
        if requested_pages == 0:
            return None
        # Keep normal serving cooperative.  The persistent queue means the
        # next HiCache event resumes exactly where this wave stopped, so a
        # local low-watermark recovery need not monopolize the serialized
        # radix path in order to reach ``high`` in one invocation.
        selection_budget_ms = float(
            os.getenv("SGLANG_LATTICEKV_LOCAL_MAINTENANCE_BUDGET_MS", "2.0")
        )
        if selection_budget_ms <= 0:
            raise ValueError("SGLANG_LATTICEKV_LOCAL_MAINTENANCE_BUDGET_MS must be positive")
        clean_gap_pages = max(0, status["min_pages"] - status["clean_pages"])
        result = self.reclaim_live_page_selection(
            protected_hit_count=None,
            protected_fraction=None,
            max_pages=requested_pages,
            selection_budget_ms=selection_budget_ms,
            # The first maintenance wave must cover the concrete HBM backup
            # that triggered it, not merely yield one page and leave the
            # allocator to fail again on the remainder of the same batch.
            # It remains bounded by ``requested_pages`` and the short
            # lifecycle budget, so this is admission-oriented rather than a
            # destructive full-cache sweep.
            initial_min_pages=max(
                1,
                min(
                    requested_pages,
                    max(clean_gap_pages, required_clean_pages),
                ),
            ),
            on_allocator_pages_ready=None,
            prefer_ready_pages=True,
            reclaim_origin="local",
        )
        timing_ms = result.get("timing_ms", {})
        if result.get("allocator_free_slots", 0) and timing_ms.get("total", 0) > 0:
            record_result = getattr(host_pool, "record_local_reclaim_result", None)
            reclaimed_pages = result["allocator_free_slots"] // self.page_size
            if record_result is not None:
                record_result(reclaimed_pages, timing_ms["total"] / 1000.0)
            else:
                record_latency = getattr(host_pool, "record_local_reclaim_latency", None)
                if record_latency is not None:
                    record_latency(timing_ms["total"] / 1000.0)
        result["local_action"] = (
            "admission" if required_clean_pages > status["clean_pages"] else status["action"]
        )
        result["requested_pages"] = requested_pages
        result["required_clean_pages"] = required_clean_pages
        result["maintenance_budget_ms"] = selection_budget_ms
        result["maintenance_target_pages"] = maintenance_target_pages
        result["value_constrained"] = bool(status.get("value_constrained", False))
        self._refresh_local_ready_pages(host_pool)
        # Failing to return all the way to ``high`` is normal: the next
        # lifecycle tick can continue from the persistent queue.  Failing to
        # satisfy an imminent backup or the hard admission minimum is not;
        # record that precise deficit so the global layer can grow this quota
        # before generic host eviction turns it into a later reuse loss.
        post_status = status_reader()
        hard_clean_target = max(required_clean_pages, status["min_pages"])
        report_shortfall = getattr(host_pool, "record_unresolved_clean_shortfall", None)
        shortfall_pages = max(0, hard_clean_target - post_status["clean_pages"])
        if report_shortfall is not None and shortfall_pages:
            report_shortfall(shortfall_pages)
        result["unresolved_clean_shortfall_pages"] = shortfall_pages
        self._write_local_maintenance_audit(
            host_pool=host_pool,
            before_status=status,
            after_status=post_status,
            result=result,
            requested_pages=requested_pages,
            required_clean_pages=required_clean_pages,
            shortfall_pages=shortfall_pages,
        )
        return result

    def _write_local_maintenance_audit(
        self,
        *,
        host_pool,
        before_status: dict,
        after_status: dict,
        result: dict,
        requested_pages: int,
        required_clean_pages: int,
        shortfall_pages: int,
    ) -> None:
        """Persist local-watermark evidence only when reclaim auditing is enabled."""
        audit_dir = os.getenv("SGLANG_CENTRAL_IO_RECLAIM_AUDIT_DIR")
        if not audit_dir:
            return
        page_size = getattr(host_pool, "page_size", None)
        if page_size is None:
            page_size = self.page_size
        page_size = int(page_size)
        freed_slots = int(result.get("allocator_free_slots", 0))
        record = {
            "schema": "latticekv_local_maintenance_v1",
            "timestamp_s": time.time(),
            "model_id": getattr(host_pool, "model_id", "unknown"),
            "before": before_status,
            "after": after_status,
            "requested_pages": requested_pages,
            "required_clean_pages": required_clean_pages,
            "maintenance_target_pages": result.get("maintenance_target_pages"),
            "value_constrained": bool(result.get("value_constrained", False)),
            "reclaimed_pages": freed_slots // page_size,
            "shortfall_pages": shortfall_pages,
            "reason": result.get("reason"),
            "timing_ms": result.get("timing_ms", {}),
            "reuse_summary": result.get("reuse_summary", {}),
        }
        os.makedirs(audit_dir, exist_ok=True)
        audit_path = os.path.join(
            audit_dir,
            f"{record['model_id']}-local-maintenance.jsonl",
        )
        with open(audit_path, "a", encoding="utf-8") as audit_file:
            audit_file.write(json.dumps(record, sort_keys=True) + "\n")

    def _refresh_local_ready_pages(self, host_pool) -> None:
        """Export only page-complete safe candidates to local watermarks."""
        setter = getattr(host_pool, "set_local_ready_pages", None)
        if setter is None:
            return
        queue = self._central_reclaim_index()
        feedback_horizon = getattr(host_pool, "reclaim_feedback_horizon_s", None)
        if feedback_horizon is not None:
            queue.set_ghost_ttl_s(
                float(feedback_horizon()), now_s=time.monotonic()
            )
        setter(
            queue.ready_page_count,
            reclaim_score=queue.ready_page_reclaim_score,
        )

    def load_back(
        self, node: TreeNode, mem_quota: Optional[int] = None
    ) -> Optional[torch.Tensor]:

        start_time = time.perf_counter()
        last_hit_node = node
        nodes_to_load = []
        while node.evicted:
            assert (
                node.backuped
            ), "No backup available on evicted nodes, should not happen"
            nodes_to_load.insert(0, node)
            node = node.parent
        else:
            ancester_node = node

        # protect the ancestor nodes from eviction
        result = self.inc_lock_ref(ancester_node)
        delta = result.delta

        # load it all or not at all
        host_indices = torch.cat([n.host_value for n in nodes_to_load])
        if len(host_indices) < self.load_back_threshold or (
            len(host_indices) > mem_quota + delta if mem_quota is not None else False
        ):
            # skip loading back if the total size is too small or exceeding the memory quota
            self.dec_lock_ref(ancester_node)
            return None

        device_indices = self.cache_controller.load(
            host_indices=host_indices, node_id=last_hit_node.id
        )
        if device_indices is None:
            self.evict(EvictParams(num_tokens=len(host_indices)))
            device_indices = self.cache_controller.load(
                host_indices=host_indices, node_id=last_hit_node.id
            )
        self.dec_lock_ref(ancester_node)
        if device_indices is None:
            # no sufficient GPU memory to load back KV caches
            logger.warning(
                "load_back: FAILED to load %d tokens for node %d "
                "even after eviction (evictable_size=%d)",
                len(host_indices),
                last_hit_node.id,
                self.evictable_size_,
            )
            return None

        self.ongoing_load_back[last_hit_node.id] = last_hit_node
        offset = 0
        for node in nodes_to_load:
            node.value = device_indices[offset : offset + len(node.host_value)].clone()
            offset += len(node.host_value)
        self.evictable_size_ += len(device_indices)
        self.inc_lock_ref(last_hit_node)

        if self.metrics_collector is not None:
            self.metrics_collector.observe_load_back_duration(
                time.perf_counter() - start_time
            )
            self.metrics_collector.increment_load_back_num_tokens(len(device_indices))

        return device_indices

    def init_load_back(
        self,
        params: InitLoadBackParams,
    ):
        last_node = params.last_host_node
        mem_quota = params.mem_quota
        if last_node.evicted:
            loading_values = self.load_back(last_node, mem_quota)
            if loading_values is not None:
                logger.debug(
                    f"loading back {len(loading_values)} tokens for node {last_node.id}"
                )
                return loading_values, last_node

            while last_node.evicted:
                last_node = last_node.parent

        return (
            torch.empty((0,), dtype=torch.int64, device=self.device),
            last_node,
        )

    def ready_to_load_host_cache(self) -> int:
        """
        Notify the cache controller to start the KV cache loading.
        Return the consumer index for the schedule batch manager to track.
        """
        return self.cache_controller.start_loading()

    def flush_write_through_acks(self) -> None:
        self.writing_check()

    def check_hicache_events(self):
        self.writing_check()
        self.loading_check()
        self._maintain_local_residency()
        publish_residency = getattr(
            self.token_to_kv_pool_host, "publish_local_residency", None
        )
        if publish_residency is not None:
            publish_residency()
        self._apply_central_quota_target()
        if self.enable_storage:
            self.drain_storage_control_queues()
        if self.enable_storage_metrics:
            self.storage_metrics_collector.log_storage_metrics(
                self.cache_controller.storage_backend.get_stats()
            )

    def drain_storage_control_queues(self):
        """
        Combine prefetch revoke, backup ack, and host mem release checks
        to minimize TP synchronization and Python overhead.
        """
        self._drain_central_storage_backups()
        cc = self.cache_controller

        qsizes = torch.tensor(
            [
                cc.prefetch_revoke_queue.qsize(),
                cc.ack_backup_queue.qsize(),
                cc.host_mem_release_queue.qsize(),
            ],
            dtype=torch.int,
        )
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                qsizes, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
            )

        n_revoke, n_backup, n_release = map(int, qsizes.tolist())
        self._drain_storage_control_queues_impl(
            n_revoke=n_revoke,
            n_backup=n_backup,
            n_release=n_release,
            log_metrics=True,
        )

    # Timeout is linearly increasing with the number of pages
    def _prefetch_timeout_check_linear_func(self, operation: PrefetchOperation):
        # If hash_value has not been computed in timeout_base seconds, terminate it.
        return (
            time.monotonic() - operation.start_time
            > self.prefetch_timeout_base
            + len(operation.hash_value) * self.prefetch_timeout_per_page
        )

    def can_terminate_prefetch(self, operation: PrefetchOperation):
        can_terminate = True

        if self.prefetch_stop_policy == "best_effort":
            return can_terminate

        if len(operation.hash_value) == 0:
            completed = False
        else:
            completed = (
                operation.completed_tokens == len(operation.hash_value) * self.page_size
            )

        if self.prefetch_stop_policy == "wait_complete":
            can_terminate = completed
        elif self.prefetch_stop_policy == "timeout":
            can_terminate = completed or self.is_prefetch_timeout(operation)
        else:
            # unknown prefetch stop policy, just return True
            return True

        operation_terminated = operation.is_terminated()
        if self.tp_world_size > 1:
            states = torch.tensor(
                [1 - int(can_terminate), int(operation_terminated)],
                dtype=torch.int,
            )
            torch.distributed.all_reduce(
                states,
                op=torch.distributed.ReduceOp.MAX,
                group=self.tp_group,
            )
            can_terminate = states[0].item() == 0
            operation_terminated = states[1].item() == 1
        # the operation should be terminated if it is already terminated on any TP worker
        # or it meets the termination condition on all TP workers
        can_terminate = can_terminate or operation_terminated
        return can_terminate

    def check_prefetch_progress(self, req_id: str) -> bool:
        if req_id not in self.ongoing_prefetch:
            # there is no ongoing prefetch for this request or it has been revoked
            return True

        # todo: more policies for prefetch progress such as timeout
        # the current policy is to prefetch with best effort and terminate when queuing is over
        last_host_node, token_ids, host_indices, operation = self.ongoing_prefetch[
            req_id
        ]

        central_operation_id = getattr(operation, "central_operation_id", None)
        if central_operation_id is not None:
            return self._check_central_storage_prefetch(
                req_id,
                last_host_node,
                token_ids,
                host_indices,
                operation,
            )

        if operation.host_indices is None:
            # prefetch has not been issued due to insufficient host memory
            return True

        if not self.can_terminate_prefetch(operation):
            return False

        completed_tokens, hash_value = self.cache_controller.terminate_prefetch(
            operation
        )
        logger.debug(f"Prefetch {req_id} completed with {completed_tokens} tokens")

        min_completed_tokens = completed_tokens
        if self.tp_world_size > 1:
            # synchrnoize TP workers to make the same update to hiradix cache
            completed_tokens_tensor = torch.tensor(
                min_completed_tokens, dtype=torch.int
            )
            torch.distributed.all_reduce(
                completed_tokens_tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
            min_completed_tokens = completed_tokens_tensor.item()
        fetched_token_ids = token_ids[:min_completed_tokens]
        written_indices = host_indices[:min_completed_tokens]
        matched_length = self._insert_helper_host(
            last_host_node,
            RadixKey(
                token_ids=fetched_token_ids, extra_key=last_host_node.key.extra_key
            ),
            written_indices,
            hash_value[: min_completed_tokens // self.page_size],
        )

        self.cache_controller.mem_pool_host.free(host_indices[:matched_length])
        self.cache_controller.append_host_mem_release(
            host_indices[min_completed_tokens:completed_tokens]
        )
        last_host_node.release_host()
        del self.ongoing_prefetch[req_id]
        self.cache_controller.prefetch_tokens_occupied -= len(token_ids)

        # Track tokens actually loaded from storage for this request (L3 hits)
        loaded_from_storage = min_completed_tokens - matched_length
        self.prefetch_loaded_tokens_by_reqid[req_id] = loaded_from_storage

        if self.enable_storage_metrics:
            self.storage_metrics_collector.log_prefetched_tokens(loaded_from_storage)

        return True

    def _check_central_storage_prefetch(
        self,
        req_id: str,
        last_host_node: TreeNode,
        token_ids: List[int],
        host_indices: torch.Tensor,
        operation,
    ) -> bool:
        """Publish agent-restored KV only after the whole read acknowledgement."""
        host_pool = self.token_to_kv_pool_host
        status = host_pool.agent_storage_status(operation.central_operation_id)
        state = status.get("state")
        if state in ("writing", "reading"):
            return False

        self.ongoing_prefetch.pop(req_id, None)
        try:
            complete = (
                state == "ready"
                and status.get("completed_pages") == operation.central_page_count
                and not getattr(operation, "central_cancelled", False)
            )
            if not complete:
                # The agent never exposes a partially read prefix.  Its worker
                # has finished (or failed) before these slots become reusable.
                self.cache_controller.mem_pool_host.free(host_indices)
                self.prefetch_loaded_tokens_by_reqid[req_id] = 0
                self._write_central_storage_audit(
                    host_pool,
                    event="read_discarded",
                    operation_id=operation.central_operation_id,
                    node_id=last_host_node.id,
                    page_count=operation.central_page_count,
                    agent_state=state,
                    completed_pages=int(status.get("completed_pages", 0)),
                    cancelled=bool(getattr(operation, "central_cancelled", False)),
                    error=status.get("error"),
                )
                return True

            matched_length = self._insert_helper_host(
                last_host_node,
                RadixKey(
                    token_ids=token_ids,
                    extra_key=last_host_node.key.extra_key,
                ),
                host_indices,
                operation.hash_value,
            )
            if matched_length:
                self.cache_controller.mem_pool_host.free(host_indices[:matched_length])
            loaded_from_storage = len(token_ids) - matched_length
            self.prefetch_loaded_tokens_by_reqid[req_id] = loaded_from_storage
            if self.enable_storage_metrics:
                self.storage_metrics_collector.log_prefetched_tokens(
                    loaded_from_storage
                )
            self._write_central_storage_audit(
                host_pool,
                event="read_published",
                operation_id=operation.central_operation_id,
                node_id=last_host_node.id,
                page_count=operation.central_page_count,
                completed_pages=int(status.get("completed_pages", 0)),
                loaded_tokens=loaded_from_storage,
            )
            return True
        finally:
            last_host_node.release_host()
            self.cache_controller.prefetch_tokens_occupied = max(
                0,
                self.cache_controller.prefetch_tokens_occupied - len(token_ids),
            )

    def terminate_prefetch(self, req_id: str):
        if req_id not in self.ongoing_prefetch:
            return

        _, _, _, operation = self.ongoing_prefetch[req_id]
        if getattr(operation, "central_operation_id", None) is not None:
            # The agent owns the mmap bytes.  Do not free the reserved pages
            # until its asynchronous read reports completion or failure.
            operation.central_cancelled = True
            self._write_central_storage_audit(
                self.token_to_kv_pool_host,
                event="read_cancel_requested",
                operation_id=operation.central_operation_id,
                request_id=req_id,
                reason="terminate_prefetch",
            )
            return
        if operation.host_indices is None:
            return
        operation.mark_terminate()

    def pop_prefetch_loaded_tokens(self, req_id: str) -> int:
        """
        Pop and return the number of tokens loaded from storage for a request.
        Returns 0 if no prefetch was done or was revoked.
        This should be called after check_prefetch_progress() returns True.
        """
        return self.prefetch_loaded_tokens_by_reqid.pop(req_id, 0)

    def match_prefix(self, params: MatchPrefixParams):
        key = params.key
        empty_value = torch.empty((0,), dtype=torch.int64, device=self.device)
        key, _ = self.maybe_bigram_convert(key)
        if self.disable or len(key) == 0:
            return MatchResult(
                device_indices=empty_value,
                last_device_node=self.root_node,
                last_host_node=self.root_node,
                host_hit_length=0,
            )

        page_aligned_len = len(key)
        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
        else:
            value = empty_value

        host_hit_length = 0
        last_host_node = last_node
        while last_node.evicted:
            host_hit_length += len(last_node.host_value)
            last_node = last_node.parent
        while not last_host_node.backuped:
            last_host_node = last_host_node.parent

        if host_hit_length:
            self._record_central_host_hit(last_host_node, host_hit_length)

        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_host_node,
            host_hit_length=host_hit_length,
        )

    def prefetch_from_storage(
        self,
        req_id: str,
        last_host_node: TreeNode,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ):
        new_input_tokens = (
            convert_to_bigram_key(new_input_tokens)
            if self.is_eagle
            else new_input_tokens
        )
        # align the number of fetching tokens to the page size
        prefetch_length = len(new_input_tokens) - (
            len(new_input_tokens) % self.page_size
        )
        new_input_tokens = new_input_tokens[:prefetch_length]
        if (
            not self.enable_storage
            or prefetch_length < self.prefetch_threshold
            or self.cache_controller.prefetch_rate_limited()
        ):
            return

        last_host_node.protect_host()
        host_indices = self.cache_controller.mem_pool_host.alloc(prefetch_length)
        if host_indices is None:
            self.evict_host(prefetch_length)
            host_indices = self.cache_controller.mem_pool_host.alloc(prefetch_length)
        if host_indices is None:
            last_host_node.release_host()
            # no sufficient host memory for prefetch
            return
        central_read = getattr(
            self.token_to_kv_pool_host, "begin_agent_storage_read", None
        )
        central_exists = getattr(
            self.token_to_kv_pool_host, "agent_storage_existing_prefix", None
        )
        if central_read is not None and central_exists is not None:
            page_hashes = []
            current_hash = last_hash
            for start in range(0, len(new_input_tokens), self.page_size):
                current_hash = self.cache_controller.get_hash_str(
                    new_input_tokens[start : start + self.page_size], current_hash
                )
                page_hashes.append(current_hash)
            existing_pages = central_exists(page_hashes)
            hit_tokens = existing_pages * self.page_size
            if hit_tokens < self.prefetch_threshold:
                self.cache_controller.mem_pool_host.free(host_indices)
                last_host_node.release_host()
                return
            if hit_tokens < len(host_indices):
                self.cache_controller.mem_pool_host.free(host_indices[hit_tokens:])
                host_indices = host_indices[:hit_tokens]
                new_input_tokens = new_input_tokens[:hit_tokens]
                page_hashes = page_hashes[:existing_pages]
            ticket = central_read(page_hashes, host_indices)
            if ticket is None:
                self.cache_controller.mem_pool_host.free(host_indices)
                last_host_node.release_host()
                return
            operation = SimpleNamespace(
                central_operation_id=ticket["operation_id"],
                central_page_count=len(ticket["page_ids"]),
                central_cancelled=False,
                hash_value=page_hashes,
            )
            self.ongoing_prefetch[req_id] = (
                last_host_node,
                new_input_tokens,
                host_indices,
                operation,
            )
            self.cache_controller.prefetch_tokens_occupied += len(new_input_tokens)
            self._write_central_storage_audit(
                self.token_to_kv_pool_host,
                event="read_started",
                operation_id=ticket["operation_id"],
                node_id=last_host_node.id,
                page_ids=list(ticket["page_ids"]),
                request_id=req_id,
            )
            return
        operation = self.cache_controller.prefetch(
            req_id, host_indices, new_input_tokens, last_hash, prefix_keys
        )
        self.ongoing_prefetch[req_id] = (
            last_host_node,
            new_input_tokens,
            host_indices,
            operation,
        )
        self.cache_controller.prefetch_tokens_occupied += len(new_input_tokens)

    def _insert_helper_host(
        self, node: TreeNode, key: RadixKey, host_value, hash_value
    ):
        node.last_access_time = time.monotonic()
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)

        matched_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            # Refresh pin TTL on host insert hit
            if self._is_pinned(node):
                node.pin_expiry = time.monotonic() + node.pin_ttl
            prefix_len = self.key_match_fn(node.key, key)
            key = key[prefix_len:]
            host_value = host_value[prefix_len:]
            hash_value = hash_value[prefix_len // self.page_size :]
            matched_length += prefix_len

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode(priority=node.priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = None
            new_node.host_value = host_value.clone()
            new_node.hash_value = hash_value
            node.children[child_key] = new_node
            self._update_host_leaf_status(new_node)
            self._update_leaf_status(node)
            self._update_host_leaf_status(node)
            self._observe_central_reclaim_node(new_node)

        return matched_length

    def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
        node.last_access_time = time.monotonic()
        child_key = self.get_child_key_fn(key)
        value = []

        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = time.monotonic()
            # Refresh pin TTL on cache hit
            if self._is_pinned(child):
                child.pin_expiry = time.monotonic() + child.pin_ttl
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                if not new_node.evicted:
                    value.append(new_node.value)
                node = new_node
                break
            else:
                if not child.evicted:
                    value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = self.get_child_key_fn(key)

        return value, node

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
        # child node split into new_node -> child
        new_node = TreeNode(priority=child.priority)
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.pin_expiry = child.pin_expiry
        new_node.pin_ttl = child.pin_ttl
        # If child is pinned, new parent inherits a host_ref_counter hold
        if child.pin_expiry > 0:
            new_node.host_ref_counter += 1
        new_node.key = child.key[:split_len]
        new_node.hit_count = child.hit_count

        # split value and host value if exists
        if child.evicted:
            new_node.value = None
        else:
            new_node.value = child.value[:split_len].clone()
            child.value = child.value[split_len:].clone()
        if child.backuped:
            new_node.host_value = child.host_value[:split_len].clone()
            child.host_value = child.host_value[split_len:].clone()

        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )
        child.parent = new_node
        child.key = child.key[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node

        self._observe_central_reclaim_node(new_node)
        self._observe_central_reclaim_node(child)

        return new_node

    def insert(self, params: InsertParams) -> InsertResult:
        key = params.key
        value = params.value
        chunked = params.chunked
        priority = params.priority

        if priority is None:
            priority = 0
        key, value = self.maybe_bigram_convert(key, value)

        if len(key) == 0:
            return InsertResult(prefix_len=0)

        if self.is_eagle and value is not None:
            # Make sure the value len equal to the EAGLE bigram key len
            value = value[: len(key)]

        node = self.root_node
        child_key = self.get_child_key_fn(key)
        total_prefix_length = 0

        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            node.priority = max(node.priority, priority)
            prefix_len = self.key_match_fn(node.key, key)

            if prefix_len == len(node.key):
                if node.evicted:
                    # change the reference if the node is evicted
                    # this often happens in the case of KV cache recomputation
                    node.value = value[:prefix_len].clone()
                    self.evictable_size_ += len(node.value)
                    self._update_leaf_status(node)
                    self._update_host_leaf_status(node)
                    # update parent status as a new leaf is added into device
                    self._update_leaf_status(node.parent)
                    self._record_central_reclaim_revisit(node)
                else:
                    self._inc_hit_count(node, chunked)
                    total_prefix_length += prefix_len
            else:
                # partial match, split the node
                new_node = self._split_node(node.key, node, prefix_len)
                # shared-prefix node should also reflect max priority
                new_node.priority = max(new_node.priority, priority)
                if new_node.evicted:
                    new_node.value = value[:prefix_len].clone()
                    self.evictable_size_ += len(new_node.value)
                    self._update_leaf_status(new_node)
                    self._update_host_leaf_status(new_node)
                    # update parent status as a new leaf is added into device
                    self._update_leaf_status(new_node.parent)
                else:
                    self._inc_hit_count(new_node, chunked)
                    total_prefix_length += prefix_len
                node = new_node

            key = key[prefix_len:]
            value = value[prefix_len:]

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode(priority=priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = value.clone()
            node.children[child_key] = new_node
            self.evictable_size_ += len(value)
            self._update_leaf_status(node)
            self._update_leaf_status(new_node)

            # Compute hash_value if storage or kv events are enabled
            if self.enable_storage or self.enable_kv_cache_events:
                new_node.hash_value = compute_node_hash_values(new_node, self.page_size)

            # Emit BlockStored so the router indexes this block.
            self._record_store_event(new_node)

            if self.cache_controller.write_policy != "write_back":
                self._inc_hit_count(new_node, chunked)
            self._record_central_reclaim_revisit(new_node)
        return InsertResult(prefix_len=total_prefix_length)

    def release_aborted_request(self, rid: str):
        # Clean up storage hit tracking for aborted request
        self.prefetch_loaded_tokens_by_reqid.pop(rid, None)

        if rid not in self.ongoing_prefetch:
            return

        last_host_node, token_ids, host_indices, operation = self.ongoing_prefetch[rid]
        if getattr(operation, "central_operation_id", None) is not None:
            # Keep the lease until the agent has stopped touching its mmap
            # bytes; check_prefetch_progress will release it on the ack.
            operation.central_cancelled = True
            self._write_central_storage_audit(
                self.token_to_kv_pool_host,
                event="read_cancel_requested",
                operation_id=operation.central_operation_id,
                request_id=rid,
                reason="release_aborted_request",
            )
            return
        if operation.host_indices is None:
            return

        completed_tokens, _ = self.cache_controller.terminate_prefetch(operation)
        if self.tp_world_size > 1:
            torch.distributed.barrier(group=self.tp_group)
        last_host_node.release_host()
        del self.ongoing_prefetch[rid]
        self.cache_controller.append_host_mem_release(host_indices[:completed_tokens])
        self.cache_controller.prefetch_tokens_occupied -= len(token_ids)
