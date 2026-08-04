"""
Life cycle of a request in the prefill server

1. Bootstrap Queue
    a. Initialize a sender for each request
    b. Use the queue to store requests whose bootstrap (handshake and preallocation) has not finished
    c. Poll senders to check bootstrap state
    d. Once bootstrap is complete, move request to Waiting Queue

2. Waiting Queue
    a. Use PrefillAdder to pop requests
    b. Run forward
    c. Add the request to Inflight Queue

3. Inflight Queue
    a. Poll (non-blocking) the sender of the request
    b. Once the transfer has finished, return the request
"""

from __future__ import annotations

import logging
import time
from collections import deque
from http import HTTPStatus
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.utils import (
    FAKE_BOOTSTRAP_HOST,
    DisaggregationMode,
    KVClassType,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    get_kv_class,
    is_mla_backend,
    kv_to_page_indices,
    kv_to_page_num,
    poll_and_all_reduce_attn_cp_tp_group,
    prepare_abort,
)
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    Req,
    ScheduleBatch,
)
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, NSATokenToKVPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.observability.req_time_stats import set_schedule_time_batch

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler
    from sglang.srt.mem_cache.memory_pool import KVCache

logger = logging.getLogger(__name__)


def release_req_to_metadata_buffer(
    req: Req, allocator: ReqToMetadataIdxAllocator
) -> None:
    """
    Release the metadata buffer index allocated for a request in prefill disaggregation mode.

    This function safely releases the metadata buffer index if it was allocated.

    Args:
        req: The request object that may have a metadata_buffer_index allocated
        allocator: The ReqToMetadataIdxAllocator instance to free the index
    """
    if (
        hasattr(req, "metadata_buffer_index")
        and req.metadata_buffer_index is not None
        and req.metadata_buffer_index >= 0
    ):
        allocator.free(req.metadata_buffer_index)
        req.metadata_buffer_index = -1


class PrefillBootstrapQueue:
    """
    Store the requests in bootstrapping
    """

    def __init__(
        self,
        token_to_kv_pool: KVCache,
        draft_token_to_kv_pool: Optional[KVCache],
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        metadata_buffers: MetadataBuffers,
        tp_rank: int,
        tp_size: int,
        gpu_id: int,
        bootstrap_port: int,
        gloo_group: ProcessGroup,
        max_total_num_tokens: int,
        scheduler: Scheduler,
        pp_rank: int,
        pp_size: int,
        transfer_backend: TransferBackend,
    ):
        self.token_to_kv_pool = token_to_kv_pool
        self.draft_token_to_kv_pool = draft_token_to_kv_pool
        self.is_mla_backend = is_mla_backend(token_to_kv_pool)
        self.metadata_buffers = metadata_buffers
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.gpu_id = gpu_id
        self.bootstrap_port = bootstrap_port
        self.queue: List[Req] = []
        self.gloo_group = gloo_group
        attn_tp_group = scheduler.attn_tp_group
        self.collective_rank = attn_tp_group.rank
        self.collective_src_rank = attn_tp_group.first_rank
        self.tp_size = attn_tp_group.world_size
        self.max_total_num_tokens = max_total_num_tokens
        self.scheduler = scheduler
        self.transfer_backend = transfer_backend
        self.kv_manager = self._init_kv_manager()

        if self.scheduler.tp_worker.is_hybrid_swa:
            # FIXME: current SWA allocation allocate full kv cache size in prefill
            self.max_total_num_tokens = min(
                self.max_total_num_tokens,
                self.scheduler.tp_worker.model_runner.swa_max_total_num_tokens,
            )

    def _init_kv_manager(self) -> CommonKVManager:
        kv_args_class = get_kv_class(self.transfer_backend, KVClassType.KVARGS)
        kv_args = kv_args_class()
        kv_args.engine_rank = self.tp_rank
        kv_args.pp_rank = self.pp_rank
        kv_args.system_dp_rank = self.scheduler.dp_rank
        kv_args.prefill_start_layer = self.token_to_kv_pool.start_layer
        kv_data_ptrs, kv_data_lens, kv_item_lens = (
            self.token_to_kv_pool.get_contiguous_buf_infos()
        )

        if self.draft_token_to_kv_pool is not None:
            # We should also transfer draft model kv cache. The indices are
            # always shared with a target model.
            draft_kv_data_ptrs, draft_kv_data_lens, draft_kv_item_lens = (
                self.draft_token_to_kv_pool.get_contiguous_buf_infos()
            )
            kv_data_ptrs += draft_kv_data_ptrs
            kv_data_lens += draft_kv_data_lens
            kv_item_lens += draft_kv_item_lens

        kv_args.kv_data_ptrs = kv_data_ptrs
        kv_args.kv_data_lens = kv_data_lens
        kv_args.kv_item_lens = kv_item_lens
        if not self.is_mla_backend:
            kv_args.kv_head_num = self.token_to_kv_pool.head_num
            kv_args.total_kv_head_num = (
                self.scheduler.model_config.get_total_num_kv_heads()
            )
        kv_args.page_size = self.token_to_kv_pool.page_size

        kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.aux_item_lens = (
            self.metadata_buffers.get_buf_infos()
        )
        kv_args.ib_device = self.scheduler.server_args.disaggregation_ib_device
        kv_args.gpu_id = self.scheduler.gpu_id

        if hasattr(self.token_to_kv_pool, "get_state_buf_infos"):
            state_data_ptrs, state_data_lens, state_item_lens = (
                self.token_to_kv_pool.get_state_buf_infos()
            )
            kv_args.state_data_ptrs = state_data_ptrs
            kv_args.state_data_lens = state_data_lens
            kv_args.state_item_lens = state_item_lens

            if isinstance(self.token_to_kv_pool, SWAKVPool):
                kv_args.state_type = "swa"
            elif isinstance(self.token_to_kv_pool, HybridLinearKVPool):
                kv_args.state_type = "mamba"
                # Get state dimension info for cross-TP slice transfer
                if hasattr(self.token_to_kv_pool, "get_state_dim_per_tensor"):
                    kv_args.state_dim_per_tensor = (
                        self.token_to_kv_pool.get_state_dim_per_tensor()
                    )
            elif isinstance(self.token_to_kv_pool, NSATokenToKVPool):
                kv_args.state_type = "nsa"
            else:
                kv_args.state_type = "none"
        else:
            kv_args.state_data_ptrs = []
            kv_args.state_data_lens = []
            kv_args.state_item_lens = []
            kv_args.state_type = "none"

        kv_manager_class = get_kv_class(self.transfer_backend, KVClassType.MANAGER)
        kv_manager = kv_manager_class(
            kv_args,
            DisaggregationMode.PREFILL,
            self.scheduler.server_args,
            self.is_mla_backend,
        )
        return kv_manager

    def add(self, req: Req, num_kv_heads: int) -> None:
        if self._check_if_req_exceed_kv_capacity(req):
            return

        backend = (
            TransferBackend.FAKE
            if req.bootstrap_host == FAKE_BOOTSTRAP_HOST
            else self.transfer_backend
        )
        kv_sender_class = get_kv_class(backend, KVClassType.SENDER)

        dest_tp_ranks = [self.tp_rank]

        req.disagg_kv_sender = kv_sender_class(
            mgr=self.kv_manager,
            bootstrap_addr=f"{req.bootstrap_host}:{self.bootstrap_port}",
            bootstrap_room=req.bootstrap_room,
            dest_tp_ranks=dest_tp_ranks,
            pp_rank=self.pp_rank,
        )
        self._process_req(req)
        self.queue.append(req)

    def extend(self, reqs: List[Req], num_kv_heads: int) -> None:
        for req in reqs:
            self.add(req, num_kv_heads)

    def _check_if_req_exceed_kv_capacity(self, req: Req) -> bool:
        if len(req.origin_input_ids) > self.max_total_num_tokens:
            message = f"Request {req.rid} exceeds the maximum number of tokens: {len(req.origin_input_ids)} > {self.max_total_num_tokens}"
            logger.error(message)
            req.time_stats.trace_ctx.abort(abort_info={"reason": message})
            prepare_abort(req, message, status_code=HTTPStatus.BAD_REQUEST)
            self.scheduler.stream_output([req], req.return_logprob)
            return True
        return False

    def _process_req(self, req: Req) -> None:
        """
        Set max_new_tokens = 1, so PrefillAdder memory estimation is accurate
        """
        req.sampling_params.max_new_tokens = 1

    def _poll_authoritative_rids(
        self, rids_to_check: Optional[List[str]] = None
    ) -> tuple[List[str], List[int]]:
        """Poll rank-0's RID order across the attention TP and CP groups."""
        from sglang.srt.utils.common import broadcast_pyobj

        requested = set(rids_to_check) if rids_to_check is not None else None
        local_rids = [
            req.rid
            for req in self.queue
            if requested is None or req.rid in requested
        ]
        tp_size = int(getattr(self, "tp_size", 1))
        if tp_size == 1:
            authoritative = local_rids
        else:
            collective_rank = int(getattr(self, "collective_rank", self.tp_rank))
            collective_src_rank = int(getattr(self, "collective_src_rank", 0))
            authoritative = broadcast_pyobj(
                [local_rids] if collective_rank == collective_src_rank else None,
                collective_rank,
                self.gloo_group,
                src=collective_src_rank,
            )[0]

        local = {req.rid: req.disagg_kv_sender for req in self.queue}

        class _MissingPoller:
            @staticmethod
            def poll():
                return KVPoll.Failed

        pollers = [local.get(rid, _MissingPoller()) for rid in authoritative]
        polls = (
            [int(poller.poll()) for poller in pollers]
            if tp_size == 1
            else poll_and_all_reduce_attn_cp_tp_group(
                pollers,
                self.scheduler.attn_cp_cpu_group,
                self.gloo_group,
            )
        )
        return authoritative, polls

    def pop_bootstrapped(
        self,
        return_failed_reqs: bool = False,
        rids_to_check: Optional[List[str]] = None,
    ) -> List[Req]:
        """
        pop the reqs which has finished bootstrapping

        return_failed_reqs: For PP, on rank 0, also return the failed reqs to notify the next rank
        rids_to_check: For PP, on rank > 0, check the rids from the previous rank has consensus with the current rank.
        """

        bootstrapped_reqs = []
        failed_reqs = []
        indices_to_remove = set()

        if int(getattr(self, "tp_size", 1)) == 1 and len(self.queue) == 0:
            if return_failed_reqs is False:
                return []
            return [], []

        authoritative, polls = self._poll_authoritative_rids(rids_to_check)
        poll_by_rid = dict(zip(authoritative, polls))

        for i, req in enumerate(self.queue):
            if req.rid not in poll_by_rid:
                continue
            poll = poll_by_rid[req.rid]

            if poll == KVPoll.Bootstrapping:
                continue
            elif poll == KVPoll.Failed:
                error_message = f"Prefill bootstrap failed for request rank={self.tp_rank} {req.rid=} {req.bootstrap_room=}"
                try:
                    req.disagg_kv_sender.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.error(error_message)
                req.time_stats.trace_ctx.abort(abort_info={"reason": error_message})
                prepare_abort(
                    req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR
                )
                self.scheduler.stream_output([req], req.return_logprob)
                indices_to_remove.add(i)
                failed_reqs.append(req)
                if self.scheduler.enable_metrics:
                    self.scheduler.metrics_collector.increment_bootstrap_failed_reqs()
                if self.scheduler.enable_hicache_storage:
                    # to release prefetch events associated with the request
                    self.scheduler.tree_cache.release_aborted_request(req.rid)
                continue

            # KV.WaitingForInput - init here
            req.time_stats.set_bootstrap_done_time()
            num_kv_indices = len(req.origin_input_ids)
            if self.req_to_metadata_buffer_idx_allocator.available_size() == 0:
                break

            req.metadata_buffer_index = (
                self.req_to_metadata_buffer_idx_allocator.alloc()
            )
            assert req.metadata_buffer_index is not None

            num_pages = kv_to_page_num(num_kv_indices, self.token_to_kv_pool.page_size)
            req.disagg_kv_sender.init(num_pages, req.metadata_buffer_index)

            bootstrapped_reqs.append(req)
            indices_to_remove.add(i)
            req.time_stats.set_wait_queue_entry_time()

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        if return_failed_reqs is False:
            return bootstrapped_reqs
        else:
            return bootstrapped_reqs, failed_reqs


class SchedulerDisaggregationPrefillMixin:
    """
    Mixin for Scheduler to handle disaggregation prefill
    """

    def get_next_disagg_prefill_batch_to_run(
        self: Scheduler,
    ) -> Optional[ScheduleBatch]:
        from sglang.srt.layers.afd import afd_is_ffn

        # PF FFN receives a batch that PA has already scheduled.  Running the
        # ordinary PrefillAdder again would let rank-local request/token capacity
        # and cache history select a different subset on each active TP rank.
        if afd_is_ffn() and getattr(
            self, "_afd_ffn_authoritative_waiting", False
        ):
            return self._afd_build_authoritative_prefill_batch()

        # HACK (byronhsu): reset the batch_is_full flag because we never enter update_running_batch which resets it
        # Otherwise, it hangs under high concurrency
        self.running_batch.batch_is_full = False

        self.process_prefill_chunk()

        batch = self.get_new_batch_prefill()
        batch = self.maybe_prepare_mlp_sync_batch(batch)

        if batch:
            set_schedule_time_batch(batch)

        return batch

    def _afd_build_authoritative_prefill_batch(
        self: Scheduler,
    ) -> ScheduleBatch:
        """Prepare exactly the PA-selected PF batch or fail before forwarding."""
        from sglang.srt.managers.scheduler_afd_mixin import SchedulerAFDMixin

        metadata = self._afd_current_metadata
        dispatch_id = metadata["dispatch_id"] if metadata is not None else None
        expected_rids = list(self._afd_req_ids or ())
        local_rids = [req.rid for req in self.waiting_queue]
        if (
            metadata is None
            or expected_rids != list(metadata["req_ids"])
            or local_rids != expected_rids
            or self._afd_batchsize_attn != len(expected_rids)
        ):
            raise RuntimeError(
                f"Invalid authoritative AFD PF state dispatch={dispatch_id}: "
                f"batch_size={self._afd_batchsize_attn}, "
                f"expected_rids={expected_rids}, local_rids={local_rids}"
            )

        reqs = list(self.waiting_queue)
        req_pool_available = self.req_to_token_pool.available_size()
        req_pool_needed = sum(req.req_pool_idx is None for req in reqs)
        token_available = self.token_to_kv_pool_allocator.available_size()
        token_needed = sum(metadata["extend_lens"])
        logger.info(
            "AFD PF authoritative prepare dispatch=%s rank=%s flag=%s bs=%d "
            "req_pool_needed=%d req_pool_available=%d token_needed=%d "
            "token_available=%d rids=%s",
            dispatch_id,
            getattr(self, "tp_rank", -1),
            self._afd_ffn_authoritative_waiting,
            len(reqs),
            req_pool_needed,
            req_pool_available,
            token_needed,
            token_available,
            expected_rids,
        )

        capacity = {
            "dispatch_id": dispatch_id,
            "rank": getattr(self, "tp_rank", -1),
            "req_pool_needed": req_pool_needed,
            "req_pool_available": req_pool_available,
            "token_needed": token_needed,
            "token_available": token_available,
            "req_ids": expected_rids,
        }
        capacities = [capacity]
        if getattr(self, "tp_size", 1) > 1:
            import torch.distributed as dist

            capacities = [None] * self.tp_size
            dist.all_gather_object(
                capacities, capacity, group=self.tp_cpu_group
            )
        failures = [
            item
            for item in capacities
            if item["req_pool_needed"] > item["req_pool_available"]
            or item["token_needed"] > item["token_available"]
        ]
        if failures:
            raise RuntimeError(
                f"AFD PF authoritative batch capacity failure "
                f"dispatch={dispatch_id}: failures={failures}"
            )

        # init_next_round_input still initializes request/cache bookkeeping, but
        # PA geometry is restored immediately so local cache history cannot alter
        # the collective tensor shape.
        for req in reqs:
            req.init_next_round_input(self.tree_cache)
            SchedulerAFDMixin.afd_restore_req_geometry(self, req)

        batch = ScheduleBatch.init_new(
            reqs,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
        )
        try:
            batch.prepare_for_extend()
        except Exception as exc:
            raise RuntimeError(
                f"AFD PF authoritative batch allocation failed "
                f"dispatch={dispatch_id} rank={getattr(self, 'tp_rank', -1)} "
                f"req_ids={expected_rids}: {exc}"
            ) from exc

        from sglang.srt.observability.scheduler_metrics_mixin import PrefillStats

        batch.prefill_stats = PrefillStats.from_authoritative(
            reqs=reqs,
            extend_lens=metadata["extend_lens"],
            seq_lens=metadata["seq_lens"],
            new_token_ratio=self.new_token_ratio,
            running_reqs=self.running_batch.reqs,
            enable_priority_scheduling=self.enable_priority_scheduling,
        )
        self.waiting_queue = []
        batch = self.maybe_prepare_mlp_sync_batch(batch)
        SchedulerAFDMixin.afd_validate_prefill_batch(self, batch)
        set_schedule_time_batch(batch)
        return batch

    @torch.no_grad()
    def event_loop_normal_disagg_prefill(self: Scheduler) -> None:
        """A normal scheduler loop for prefill worker in disaggregation mode."""

        while True:
            # Receive requests
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            self.waiting_queue.extend(
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
            )

            # Get the next batch to run
            batch = self.get_next_disagg_prefill_batch_to_run()
            self.cur_batch = batch

            # Launch the current batch
            if batch:
                self._unified_dvfs_before_batch(batch)
                result = self.run_batch(batch)
                self._unified_log_prefill_observed(batch)
                self.process_batch_result(batch, result)
            else:
                self.self_check_during_idle()

            self.process_disagg_prefill_inflight_queue()

            # Update last_batch
            self.last_batch = batch

    @torch.no_grad()
    def event_loop_afd_disagg_prefill(self: Scheduler) -> None:
        """Disagg prefill event loop with AFD (Attention-FFN Disaggregation)."""
        from sglang.srt.layers.afd import afd_is_attn, afd_is_ffn, get_afd_perspective
        from sglang.srt.managers.scheduler_afd_mixin import SchedulerAFDMixin

        logger.info(
            "event_loop_afd_disagg_prefill: role=%s", get_afd_perspective()
        )
        SchedulerAFDMixin.afd_init_state(self)

        # Eager-init UCX communicator: FFN listens, Attn connects.
        # Both sides must init together for the handshake to succeed.
        # NOTE: interleaved schedule (--afd-async-schedule) uses the same
        # single shared channel — no per-mb channels needed.
        if SchedulerAFDMixin.afd_component_should_eager_init_data_plane(self):
            from sglang.srt.layers.afd import get_async_communicator
            try:
                get_async_communicator()
                logger.info(
                    "event_loop_afd_disagg_prefill: UCX communicator ready (async=%s)",
                    getattr(self.server_args, "afd_async_schedule", False),
                )
            except Exception as e:
                logger.error("event_loop_afd_disagg_prefill: AF communicator init failed: %s", e)
                raise RuntimeError(
                    f"AF communicator init failed — cannot run AFD disagg prefill without it: {e}"
                ) from e
        else:
            logger.info("event_loop_afd_disagg_prefill: deferring joining communicator init "
                        "to post-activate readiness")

        while True:
            SchedulerAFDMixin.afd_component_begin_active_iteration(self)
            if SchedulerAFDMixin.afd_component_should_leave_active_loop(self):
                return
            if SchedulerAFDMixin.afd_component_should_restart_active_iteration(self):
                continue
            # Check control again immediately before entering a potentially
            # blocking data-plane receive. All active ranks call this in order.
            SchedulerAFDMixin.afd_component_post_receive_control_checkpoint(
                self
            )
            if SchedulerAFDMixin.afd_component_should_restart_active_iteration(self):
                continue
            recv_reqs = self.recv_requests()
            extra_reqs = SchedulerAFDMixin.afd_recv_messages(self)
            if extra_reqs:
                recv_reqs = recv_reqs + extra_reqs

            if recv_reqs:
                logger.info("afd_disagg_prefill: recv %d reqs: %s",
                            len(recv_reqs), [type(r).__name__ for r in recv_reqs])
            if afd_is_ffn() and self.tp_size > 1 and not self.server_args.enable_dp_attention:
                from sglang.srt.utils.common import broadcast_pyobj

                recv_reqs = broadcast_pyobj(
                    recv_reqs,
                    self.tp_group.rank,
                    self.tp_cpu_group,
                    src=self.tp_group.ranks[0],
                )
            SchedulerAFDMixin.afd_component_post_receive_control_checkpoint(
                self
            )
            # ACTIVATE may be consumed by the post-receive checkpoint,
            # after the loop-top restart guard has already run. Old surviving
            # ranks must skip this data-plane round and meet joining ranks at
            # the next post-activation readiness preamble.
            if SchedulerAFDMixin.afd_component_should_restart_active_iteration(self):
                continue
            recv_reqs = SchedulerAFDMixin.afd_gate_work_requests(self, recv_reqs)
            SchedulerAFDMixin.afd_forward_work_requests(self, recv_reqs)
            # Use the shared AFD metadata path on both PA and PF.  It queues
            # AFDReqInput, restores the full prefill request metadata, and
            # creates missing FFN-side Req objects without forwarding duplicate
            # TokenizedGenerateReqInput objects into the PF scheduler.
            self._afd_process_input_requests(
                recv_reqs, work_already_forwarded=True
            )
            if not afd_is_ffn():
                bootstrapped = self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
                if bootstrapped:
                    logger.info("afd_disagg_prefill: %d reqs bootstrapped", len(bootstrapped))
                self.waiting_queue.extend(bootstrapped)

            if SchedulerAFDMixin.afd_ffn_should_wait(self):
                # When a component reshard has fenced admission, no more
                # AFDReqInput will ever arrive on this FFN participant. Reset the
                # batch ledger to idle so ``is_fully_idle()`` (and thus the
                # quiesce/drain AFD-quiescent check) can actually observe idle.
                # Without this the disagg prefill FFN loop keeps last_batch/
                # cur_batch non-empty forever and the drain silently spins until
                # the reshard timeout. See _afd_ffn_reset_idle_ledger.
                self._afd_ffn_reset_idle_ledger()
                # Idle freq lock for FFN (Prefill) side while waiting
                if (self._idle_lock_enabled
                        and not self._idle_freq_locked
                        and self._dvfs_hw_list):
                    self._lock_all_sm_clocks(self._idle_lock_freq)
                    self._idle_freq_locked = True
                continue

            batch = self.get_next_disagg_prefill_batch_to_run()
            if batch:
                logger.info("afd_disagg_prefill: got batch bs=%d mode=%s",
                            batch.batch_size(), batch.forward_mode)
            self.cur_batch = batch

            if batch:
                if afd_is_attn():
                    self._afd_dvfs_before_batch(batch)
                SchedulerAFDMixin.afd_send_batch_info(self, batch)
                # Record ZMQ-send wall-clock for cross-GPU latency breakdown
                from sglang.srt.layers.afd_mixin import _afd_sched_ts
                _afd_sched_ts["zmq_sent"] = time.time()
                SchedulerAFDMixin.afd_prepare_overlap(self, batch)
                is_decode = batch.forward_mode.is_decode()
                self._tier1_record_batch_start(is_prefill=not is_decode)
                if not afd_is_attn():
                    self._afd_dvfs_before_batch(batch)
                logger.info(
                    "AFD prefill before run: role=%s bs=%d extend_lens=%s req_ids=%s",
                    get_afd_perspective(), batch.batch_size(),
                    getattr(batch, "extend_lens", None), [r.rid for r in batch.reqs],
                )
                try:
                    if afd_is_ffn():
                        SchedulerAFDMixin.afd_validate_prefill_batch(self, batch)
                    result = self.run_batch(batch)
                    logger.info(
                        "AFD prefill after run: role=%s next_tokens_shape=%s",
                        get_afd_perspective(),
                        getattr(getattr(result, "next_token_ids", None), "shape", None),
                    )
                    self.process_batch_result(batch, result)
                    logger.info("AFD prefill after result: role=%s", get_afd_perspective())
                except BaseException:
                    logger.exception(
                        "AFD prefill batch failed: role=%s bs=%d extend_lens=%s",
                        get_afd_perspective(), batch.batch_size(),
                        getattr(batch, "extend_lens", None),
                    )
                    raise
                t_iter = (time.perf_counter() - self._last_decode_batch_time) * 1e6 \
                    if is_decode and hasattr(self, "_last_decode_batch_time") and self._last_decode_batch_time is not None \
                    else 0.0
                self._tier1_record_batch(batch, is_decode, t_iter)
                SchedulerAFDMixin.afd_reset_state(self)
            else:
                self.self_check_during_idle()
                if hasattr(self, "_tier1_collector") and self._tier1_collector is not None:
                    self._tier1_collector.record_idle_start()
                # Idle freq lock for Attn (PA) side when no batch
                if (self._idle_lock_enabled
                        and not self._idle_freq_locked
                        and self._dvfs_hw_list):
                    self._lock_all_sm_clocks(self._idle_lock_freq)
                    self._idle_freq_locked = True

            if not afd_is_ffn():
                self.process_disagg_prefill_inflight_queue()
            self.last_batch = batch

            # ── Tier 1 workload monitor ──
            if hasattr(self, "_tier1_monitor_check"):
                self._tier1_monitor_check()

    @torch.no_grad()
    def event_loop_overlap_disagg_prefill(self: Scheduler) -> None:
        self.result_queue = deque()

        while True:
            # Receive requests
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            self.waiting_queue.extend(
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
            )

            # Get the next batch to run
            batch = self.get_next_disagg_prefill_batch_to_run()
            self.cur_batch = batch

            # Launch the current batch
            if batch:
                self._unified_dvfs_before_batch(batch)
                batch_result = self.run_batch(batch)
                self._unified_log_prefill_observed(batch)
                self.result_queue.append((batch.copy(), batch_result))
            else:
                batch_result = None

            # Process the last batch
            if self.last_batch:
                tmp_batch, tmp_result = self.result_queue.popleft()
                self.process_batch_result(tmp_batch, tmp_result)
            elif batch is None:
                # When the server is idle, do self-check and re-init some states
                self.self_check_during_idle()

            self.process_disagg_prefill_inflight_queue()

            # Run sample of the current batch
            # It depends on the result of the last batch (e.g., grammar), so we run it after the last batch is processed.
            self.launch_batch_sample_if_needed(batch_result)

            # Update last_batch
            self.last_batch = batch

    def process_batch_result_disagg_prefill(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> None:
        from sglang.srt.layers.afd import afd_is_ffn as _is_ffn
        """
        Transfer kv for prefill completed requests and add it into disagg_prefill_inflight_queue
        Adapted from process_batch_result_prefill
        """
        (
            logits_output,
            next_token_ids,
            extend_input_len_per_req,
            extend_logprob_start_len_per_req,
            copy_done,
        ) = (
            result.logits_output,
            result.next_token_ids,
            result.extend_input_len_per_req,
            result.extend_logprob_start_len_per_req,
            result.copy_done,
        )

        if copy_done is not None:
            copy_done.synchronize()

        logprob_pt = 0
        # Transfer kv for prefill completed requests and add it into disagg_prefill_inflight_queue
        next_token_ids = result.next_token_ids.tolist()
        if batch.return_logprob:
            if logits_output.next_token_logprobs is not None:
                logits_output.next_token_logprobs = (
                    logits_output.next_token_logprobs.tolist()
                )
            if logits_output.input_token_logprobs is not None:
                logits_output.input_token_logprobs = tuple(
                    logits_output.input_token_logprobs.tolist()
                )

        for i, (req, next_token_id) in enumerate(
            zip(batch.reqs, next_token_ids, strict=True)
        ):
            if req.is_chunked <= 0:
                req.time_stats.set_prefill_finished_time()

                # There is no output_ids for prefill
                req.output_ids.append(next_token_id)
                self.tree_cache.cache_unfinished_req(req)  # update the tree and lock
                # FFN side doesn't do KV transfer — skip inflight queue
                if not _is_ffn():
                    self.disagg_prefill_inflight_queue.append(req)
                else:
                    # FFN side: release KV cache and finish immediately
                    release_kv_cache(req, self.tree_cache)
                    req.finished_reason = FINISH_LENGTH(length=0)
                if self.spec_algorithm.is_eagle() and batch.spec_info is not None:
                    req.output_topk_p = batch.spec_info.topk_p[i]
                    req.output_topk_index = batch.spec_info.topk_index[i]
                    req.hidden_states_tensor = (
                        batch.spec_info.hidden_states[i].cpu().clone()
                    )
                else:
                    req.hidden_states_tensor = None
                if req.return_logprob:
                    assert extend_logprob_start_len_per_req is not None
                    assert extend_input_len_per_req is not None
                    extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                    extend_input_len = extend_input_len_per_req[i]
                    num_input_logprobs = extend_input_len - extend_logprob_start_len
                    self.add_logprob_return_values(
                        i,
                        req,
                        logprob_pt,
                        next_token_ids,
                        num_input_logprobs,
                        logits_output,
                    )
                    logprob_pt += num_input_logprobs
                # FFN side doesn't do KV transfer — only Attn side sends KV.
                from sglang.srt.layers.afd import afd_is_ffn as _is_ffn
                if not _is_ffn():
                    self.send_kv_chunk(req, last_chunk=True)
                    req.time_stats.set_prefill_transfer_queue_entry_time()

                if req.grammar is not None:
                    # FIXME: this try-except block is for handling unexpected xgrammar issue.
                    try:
                        req.grammar.accept_token(next_token_id)
                    except ValueError as e:
                        # Grammar accept_token can raise ValueError if the token is not in the grammar.
                        # This can happen if the grammar is not set correctly or the token is invalid.
                        error_message = f"Grammar accept_token failed for req {req.rid} with token {next_token_id}: {e}"
                        release_kv_cache(req, self.tree_cache)
                        prepare_abort(
                            req,
                            error_message,
                            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                        )
                    req.grammar.finished = req.finished()
            else:
                # being chunked reqs' prefill is not finished
                req.is_chunked -= 1

                if req.return_logprob:
                    extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                    extend_input_len = extend_input_len_per_req[i]
                    if extend_logprob_start_len < extend_input_len:
                        # Update input logprobs.
                        num_input_logprobs = extend_input_len - extend_logprob_start_len
                        self.add_input_logprob_return_values(
                            i,
                            req,
                            logits_output,
                            logprob_pt,
                            num_input_logprobs,
                            last_prefill_chunk=False,
                        )
                        logprob_pt += num_input_logprobs

                if self.enable_overlap:
                    if not _is_ffn():
                        self.send_kv_chunk(req, last_chunk=False, end_idx=req.tmp_end_idx)
                req.time_stats.set_last_chunked_prefill_finish_time()

        can_run_cuda_graph = getattr(result, "can_run_cuda_graph", False)
        self.report_prefill_stats(
            prefill_stats=batch.prefill_stats,
            can_run_cuda_graph=can_run_cuda_graph,
            dp_cooperation_info=batch.dp_cooperation_info,
        )

    def process_disagg_prefill_inflight_queue(
        self: Scheduler, rids_to_check: Optional[List[str]] = None
    ) -> List[Req]:
        """
        Poll the requests in the middle of transfer. If done, return the request.
        rids_to_check: For PP, on rank > 0, check the rids from the previous rank has consensus with the current rank.
        """
        if len(self.disagg_prefill_inflight_queue) == 0:
            return []

        done_reqs = []

        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender for req in self.disagg_prefill_inflight_queue],
            self.attn_cp_cpu_group,
            self.attn_tp_cpu_group,
        )

        undone_reqs: List[Req] = []
        # Check .poll() for the reqs in disagg_prefill_inflight_queue. If Success, respond to the client and remove it from the queue
        for req, poll in zip(self.disagg_prefill_inflight_queue, polls):

            if rids_to_check is not None:
                if req.rid not in rids_to_check:
                    undone_reqs.append(req)
                    continue

                # In PP mode, the previous rank may have reached a terminal
                # state (Success/Failed) while this rank's local poll is still
                # in a transient state due to clock skew or propagation delay.
                # Treat non-terminal states as undone instead of crashing.
                if poll not in (
                    KVPoll.Success,
                    KVPoll.Failed,
                ):
                    logger.warning(
                        f"PP rank {self.pp_rank}: unexpected poll state {poll} for rid {req.rid} "
                        f"from consensus; treating as undone"
                    )
                    undone_reqs.append(req)
                    continue

            if poll in [KVPoll.WaitingForInput, KVPoll.Transferring]:
                undone_reqs.append(req)
            elif poll == KVPoll.Success:  # transfer done
                release_kv_cache(req, self.tree_cache)  # unlock the tree
                req.finished_reason = FINISH_LENGTH(length=0)
                # FIXME: clean up req's data in transfer engine
                if hasattr(req.disagg_kv_sender, "clear"):
                    req.disagg_kv_sender.clear()
                done_reqs.append(req)
                req.time_stats.set_prefill_kv_transfer_finish_time()
            elif poll == KVPoll.Failed:
                error_message = f"Prefill transfer failed for request rank={self.tp_rank} {req.rid=} {req.bootstrap_room=}"
                try:
                    req.disagg_kv_sender.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.warning(error_message)
                req.time_stats.trace_ctx.abort(abort_info={"reason": error_message})
                release_kv_cache(req, self.tree_cache)  # unlock the tree
                prepare_abort(
                    req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR
                )
                done_reqs.append(req)
                if self.enable_metrics:
                    self.metrics_collector.increment_transfer_failed_reqs()
            else:
                logger.warning(
                    f"Unexpected polling state {poll} for rid {req.rid} in inflight queue; "
                    f"treating as undone"
                )
                undone_reqs.append(req)

        for req in done_reqs:
            req.time_stats.set_completion_time()

        page_size = self.token_to_kv_pool_allocator.page_size
        kv_item_lens = (
            self.disagg_prefill_bootstrap_queue.kv_manager.kv_args.kv_item_lens
        )
        bytes_per_page_all_layers = sum(kv_item_lens)

        for req in done_reqs:
            if isinstance(req.finished_reason, FINISH_ABORT):
                continue
            metrics = req.time_stats.compute_and_observe_kv_transfer_metrics(
                num_tokens=len(req.origin_input_ids),
                page_size=page_size,
                bytes_per_page_all_layers=bytes_per_page_all_layers,
            )
            if metrics:
                # Update last-value for REST API
                if "latency_ms" in metrics:
                    self.kv_transfer_latency_ms = metrics["latency_ms"]
                if "speed_gb_s" in metrics:
                    self.kv_transfer_speed_gb_s = metrics["speed_gb_s"]

        # Stream requests which have finished transfer
        self.stream_output(
            done_reqs,
            any(req.return_logprob for req in done_reqs),
            None,
        )
        for req in done_reqs:
            req: Req

            release_req_to_metadata_buffer(
                req, self.req_to_metadata_buffer_idx_allocator
            )

        self.disagg_prefill_inflight_queue = undone_reqs

        return done_reqs

    def get_transferred_rids(self: Scheduler) -> List[str]:
        """
        Used by PP, get the transferred rids but **do not pop**
        """
        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender for req in self.disagg_prefill_inflight_queue],
            self.attn_cp_cpu_group,
            self.attn_tp_cpu_group,
        )

        transferred_rids: List[str] = []

        for req, poll in zip(self.disagg_prefill_inflight_queue, polls):
            if poll == KVPoll.Success or poll == KVPoll.Failed:
                transferred_rids.append(req.rid)

        return transferred_rids

    def process_prefill_chunk(self: Scheduler) -> None:
        chunked_req_to_exclude = set()
        if self.chunked_req:
            chunked_req_to_exclude.add(self.chunked_req)
            self.tree_cache.cache_unfinished_req(self.chunked_req, chunked=True)
            if self.enable_overlap:
                # Delay KV transfer to process_batch_result_disagg_prefill when overlap is enabled to ensure results are resolved
                self.chunked_req.tmp_end_idx = min(
                    len(self.chunked_req.fill_ids),
                    len(self.chunked_req.origin_input_ids),
                )
            else:
                from sglang.srt.layers.afd import afd_is_ffn as _is_ffn
                if not _is_ffn():
                    self.send_kv_chunk(self.chunked_req)
            self.running_batch.batch_is_full = False

        if self.last_batch and self.last_batch.forward_mode.is_extend():
            if self.last_batch.chunked_req:
                # In the context pipeline parallelism, after the last chunk, the current microbatch still track outdated chunked_req.
                # We need to discard it.
                chunked_req_to_exclude.add(self.last_batch.chunked_req)

            last_bs = self.last_batch.batch_size()
            self.last_batch.filter_batch(
                chunked_req_to_exclude=list(chunked_req_to_exclude)
            )
            if self.last_batch.batch_size() < last_bs:
                self.running_batch.batch_is_full = False

    def send_kv_chunk(
        self: Scheduler,
        req: Req,
        last_chunk: bool = False,
        end_idx: Optional[int] = None,
    ) -> None:
        """
        Send a prefilled chunk to the decode server
        """
        # AFD FFN side doesn't manage KV cache — skip KV transfer
        if req.disagg_kv_sender is None:
            return
        page_size = self.token_to_kv_pool_allocator.page_size
        start_idx = req.start_send_idx
        end_idx = (
            end_idx
            if end_idx is not None
            else min(len(req.fill_ids), len(req.origin_input_ids))
        )

        if not last_chunk:
            # if not the last chunk and the last page is partial, delay the last partial page to the next send
            end_idx = end_idx - end_idx % page_size

        kv_indices = (
            self.req_to_token_pool.req_to_token[req.req_pool_idx, start_idx:end_idx]
            .cpu()
            .numpy()
        )
        req.start_send_idx = end_idx
        state_indices = None
        if last_chunk:
            self.disagg_metadata_buffers.set_buf(req)

            # Prepare extra pool indices for hybrid models
            if isinstance(
                self.token_to_kv_pool_allocator.get_kvcache(), HybridLinearKVPool
            ):
                # Mamba hybrid model: send single mamba state index
                state_indices = [
                    self.req_to_token_pool.req_index_to_mamba_index_mapping[
                        req.req_pool_idx
                    ]
                    .cpu()
                    .numpy()
                ]
            elif isinstance(self.token_to_kv_pool_allocator.get_kvcache(), SWAKVPool):
                # SWA hybrid model: send last window KV indices
                seq_len = len(req.fill_ids)
                window_size = self.sliding_window_size
                window_start = max(0, seq_len - window_size)
                window_start = (window_start // page_size) * page_size

                window_kv_indices_full = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, window_start:seq_len
                ]

                # Translate to SWA pool indices
                window_kv_indices_swa = (
                    self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                        window_kv_indices_full
                    )
                )
                state_indices = window_kv_indices_swa.cpu().numpy()
                state_indices = kv_to_page_indices(state_indices, page_size)
            elif isinstance(
                self.token_to_kv_pool_allocator.get_kvcache(), NSATokenToKVPool
            ):
                seq_len = len(req.fill_ids)
                kv_indices_full = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, :seq_len
                ]
                state_indices = kv_indices_full.cpu().numpy()
                state_indices = kv_to_page_indices(state_indices, page_size)

        page_indices = kv_to_page_indices(kv_indices, page_size)
        if len(page_indices) == 0:
            logger.info(
                f"Skip sending kv chunk for request {req.rid=} {req.bootstrap_room=} because page_indices is empty"
            )
            return
        req.disagg_kv_sender.send(page_indices, state_indices)
