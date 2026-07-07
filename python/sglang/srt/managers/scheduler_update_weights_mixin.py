from __future__ import annotations

import logging
import time
import traceback
from typing import TYPE_CHECKING, Tuple

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.constants import (
    GPU_MEMORY_ALL_TYPES,
    GPU_MEMORY_TYPE_CUDA_GRAPH,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
)
from sglang.srt.managers.io_struct import (
    CheckWeightsReqInput,
    CheckWeightsReqOutput,
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromIPCReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
)
from sglang.srt.utils import broadcast_pyobj

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class SchedulerUpdateWeightsMixin:

    def update_weights_from_disk(
        self: Scheduler, recv_req: UpdateWeightFromDiskReqInput
    ):
        """In-place update of the weights from disk."""
        success, message = self.tp_worker.update_weights_from_disk(recv_req)
        if success:
            if recv_req.flush_cache:
                flush_cache_success = self.flush_cache()
                assert flush_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        return UpdateWeightFromDiskReqOutput(success, message, 0)

    def init_weights_update_group(
        self: Scheduler, recv_req: InitWeightsUpdateGroupReqInput
    ):
        """Initialize the online model parameter update group."""
        success, message = self.tp_worker.init_weights_update_group(recv_req)
        return InitWeightsUpdateGroupReqOutput(success, message)

    def destroy_weights_update_group(
        self: Scheduler, recv_req: DestroyWeightsUpdateGroupReqInput
    ):
        """Destroy the online model parameter update group."""
        success, message = self.tp_worker.destroy_weights_update_group(recv_req)
        return DestroyWeightsUpdateGroupReqOutput(success, message)

    def update_weights_from_distributed(
        self,
        recv_req: UpdateWeightsFromDistributedReqInput,
    ) -> Tuple[bool, str]:
        """Update the online model parameter."""
        success, message = self.tp_worker.update_weights_from_distributed(recv_req)
        if success:
            if recv_req.flush_cache:
                flush_cache_success = self.flush_cache()
                assert flush_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        return UpdateWeightsFromDistributedReqOutput(success, message)

    def update_weights_from_tensor(
        self: Scheduler, recv_req: UpdateWeightsFromTensorReqInput
    ):
        """Update the online model parameter from tensors."""
        if recv_req.disable_draft_model:
            worker = self.tp_worker
        else:
            worker = self.draft_worker or self.tp_worker
        success, message = worker.update_weights_from_tensor(recv_req)
        # TODO extract common code b/t update_weights_from_distributed and update_weights_from_tensor later
        if success:
            if recv_req.flush_cache:
                flush_cache_success = self.flush_cache()
                assert flush_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        torch.distributed.barrier(group=self.tp_cpu_group)
        return UpdateWeightsFromTensorReqOutput(success, message)

    def update_weights_from_ipc(
        self: Scheduler, recv_req: UpdateWeightsFromIPCReqInput
    ):
        """Update the online model parameter from IPC for checkpoint-engine integration."""
        success, message = self.tp_worker.update_weights_from_ipc(recv_req)
        if success:
            if recv_req.flush_cache:
                flush_cache_success = self.flush_cache()
                assert flush_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        torch.distributed.barrier(group=self.tp_cpu_group)
        return UpdateWeightsFromIPCReqOutput(success, message)

    def get_weights_by_name(self: Scheduler, recv_req: GetWeightsByNameReqInput):
        parameter = self.tp_worker.get_weights_by_name(recv_req)
        return GetWeightsByNameReqOutput(parameter)

    def release_memory_occupation(
        self: Scheduler, recv_req: ReleaseMemoryOccupationReqInput
    ):
        assert (
            self.is_fully_idle()
        ), "release_memory_occupation should be called only when server is idle."

        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        for tag in tags:
            self.offload_tags.add(tag)

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_KV_CACHE)
            self.flush_cache()

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            self.stashed_model_static_state = _export_static_state(
                self.tp_worker.model_runner.model
            )
            torch.distributed.barrier(self.tp_cpu_group)
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_WEIGHTS)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_CUDA_GRAPH)

        torch.get_device_module().synchronize()

        return ReleaseMemoryOccupationReqOutput()

    def resume_memory_occupation(
        self: Scheduler, recv_req: ResumeMemoryOccupationReqInput
    ):
        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        for tag in tags:
            self.offload_tags.remove(tag)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_CUDA_GRAPH)

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_WEIGHTS)
            torch.distributed.barrier(self.tp_cpu_group)
            _import_static_state(
                self.tp_worker.model_runner.model,
                self.stashed_model_static_state,
            )
            del self.stashed_model_static_state

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_KV_CACHE)

        return ResumeMemoryOccupationReqOutput()

    def check_weights(self: Scheduler, recv_req: CheckWeightsReqInput):
        try:
            self.tp_worker.model_runner.check_weights(action=recv_req.action)
            return CheckWeightsReqOutput(success=True, message="Success.")
        except Exception as e:
            logger.warning(f"check_weights see error: {e}")
            traceback.print_exc()
            return CheckWeightsReqOutput(success=False, message=f"{e}")

    def save_remote_model(self: Scheduler, params):
        url = params["url"]

        self.tp_worker.model_runner.save_remote_model(url)

        if self.draft_worker is not None:
            draft_url = params.get("draft_url", None)
            assert (
                draft_url is not None
            ), "draft_url must be provided when draft model is enabled"
            self.draft_worker.model_runner.save_remote_model(draft_url)

    def save_sharded_model(self: Scheduler, params):
        self.tp_worker.model_runner.save_sharded_model(
            path=params["path"],
            pattern=params["pattern"],
            max_size=params["max_size"],
        )

    def handle_reshard(self: Scheduler, recv_req):
        """Handle a reshard request (in-process TP change or weight export).

        Actions:
          - "live_reshard_tp": Full live TP reshard (drain + NCCL transfer + UCX reconnect)
          - "reshard": Perform in-process TP change
          - "export_weights": Export weights as IPC handles for other ranks
          - "ipc_reconnect": Re-create IPC communicator for new peer handshake
        """
        from sglang.srt.managers.io_struct import ReshardReqOutput

        if recv_req.action == "live_reshard_tp":
            return self._live_reshard_tp(recv_req)
        elif recv_req.action == "export_weights":
            return self._export_weights_for_reshard(recv_req)
        elif recv_req.action == "reshard":
            return self._do_reshard(recv_req)
        elif recv_req.action == "ipc_reconnect":
            return self._ipc_reconnect(recv_req)
        else:
            return ReshardReqOutput(success=False, message=f"Unknown action: {recv_req.action}")

    def _live_reshard_tp(self: Scheduler, recv_req):
        """Full live TP reshard: drain → NCCL transfer → UCX reconnect → resume.

        This is the key method that achieves <5s downtime:
          1. Pause generation (drain in-flight requests): ~10ms
          2. NCCL weight transfer to new workers: ~500ms (NVLink)
          3. UCX hot reconnect to PF: ~50ms
          4. Resume generation: ~0ms
        Total: ~560ms per reshard step

        Args in recv_req:
          new_tp_size: target TP (e.g., 2, 4, 8)
          nccl_port: port for temp NCCL group (default 29500)
        """
        from sglang.srt.managers.io_struct import (
            PauseGenerationReqInput,
            ContinueGenerationReqInput,
            ReshardReqOutput,
        )

        t0 = time.time()
        timings = {}

        new_tp = getattr(recv_req, "new_tp_size", None)
        if new_tp is None:
            return ReshardReqOutput(success=False, message="new_tp_size required", elapsed_s=0)

        nccl_port = getattr(recv_req, "nccl_port", 29500)
        old_tp = self.tp_worker.model_runner.tp_size

        logger.info("live_reshard_tp: TP%d → TP%d", old_tp, new_tp)
        self._schedule_inplace_reshard_background_prep(old_tp, new_tp)

        # In-place path: wait until continuous-batching has drained in-flight
        # decodes before rebuilding the KV pool. Pausing immediately would freeze
        # running_batch and corrupt active sequences when the pool is recreated.
        if self.server_args.inplace_reshard_max_tp is not None:
            if getattr(self, "_pending_inplace_reshard", None) is not None:
                pre_until = getattr(self, "_inplace_reshard_pre_drain_until", 0.0)
                if pre_until and time.time() < pre_until:
                    return ReshardReqOutput(
                        success=True,
                        message="pre-draining before in-place reshard",
                        elapsed_s=time.time() - t0,
                    )
                return ReshardReqOutput(
                    success=True,
                    message="already draining in-flight request(s)",
                    elapsed_s=time.time() - t0,
                )
            pre_drain_sec = float(getattr(recv_req, "pre_drain_sec", 0.0) or 0.0)
            if pre_drain_sec <= 0:
                import os

                pre_drain_sec = float(
                    os.environ.get("SGLANG_INPLACE_RESHARD_PRE_DRAIN_SEC", "0") or 0
                )
            pre_until = getattr(self, "_inplace_reshard_pre_drain_until", 0.0)
            if pre_drain_sec > 0:
                if pre_until <= 0.0:
                    self._pending_inplace_reshard = (recv_req, t0)
                    self._inplace_reshard_pre_drain_until = time.time() + pre_drain_sec
                    n_running = len(self.running_batch.reqs)
                    logger.info(
                        "  Phase 0 pre-drain: block admissions for %.1fs "
                        "(%d in-flight request(s))",
                        pre_drain_sec,
                        n_running,
                    )
                    self._publish_inplace_reshard_status(
                        phase="pre_draining",
                        active_tp=old_tp,
                        target_tp=new_tp,
                        old_tp=old_tp,
                        message=(
                            f"pre-draining {pre_drain_sec:.1f}s with {n_running} "
                            "in-flight request(s)"
                        ),
                    )
                    return ReshardReqOutput(
                        success=True,
                        message=f"pre-draining {pre_drain_sec:.1f}s",
                        elapsed_s=0,
                    )
                if time.time() < pre_until or not self._inplace_reshard_is_drained():
                    self._pending_inplace_reshard = (recv_req, t0)
                    if (
                        time.time() >= pre_until
                        and not self._inplace_reshard_is_drained()
                    ):
                        n_running = len(self.running_batch.reqs)
                        self._publish_inplace_reshard_status(
                            phase="draining",
                            active_tp=old_tp,
                            target_tp=new_tp,
                            old_tp=old_tp,
                            message=f"draining {n_running} in-flight request(s)",
                        )
                    return ReshardReqOutput(
                        success=True,
                        message="pre-draining or draining in-flight request(s)",
                        elapsed_s=time.time() - t0,
                    )
                self._inplace_reshard_pre_drain_until = 0.0
            if not self._inplace_reshard_is_drained():
                self._pending_inplace_reshard = (recv_req, t0)
                n_running = len(self.running_batch.reqs)
                logger.info(
                    "  Phase 1 drain: waiting for %d in-flight request(s)",
                    n_running,
                )
                self._publish_inplace_reshard_status(
                    phase="draining",
                    active_tp=old_tp,
                    target_tp=new_tp,
                    old_tp=old_tp,
                    message=f"draining {n_running} in-flight request(s)",
                )
                return ReshardReqOutput(
                    success=True,
                    message=f"draining {n_running} in-flight request(s)",
                    elapsed_s=0,
                )
            self._pending_inplace_reshard = None
            self._inplace_reshard_pre_drain_until = 0.0

        if self.server_args.inplace_reshard_max_tp is not None:
            from sglang.srt.reshard.inplace_reshard_background import background_prep_enabled

            if (
                background_prep_enabled()
                and not self.tp_worker.model_runner.inplace_reshard_prep_is_ready(new_tp)
            ):
                self._pending_inplace_reshard = (recv_req, t0)
                self._schedule_inplace_reshard_background_prep(old_tp, new_tp)
                return ReshardReqOutput(
                    success=True,
                    message=f"waiting for background prep TP{old_tp}→TP{new_tp}",
                    elapsed_s=time.time() - t0,
                )

        if (
            self.server_args.inplace_reshard_max_tp is not None
            and old_tp > 1
            and self.tp_rank == 0
        ):
            self._inplace_reshard_execute_pending = (recv_req, t0)
            return ReshardReqOutput(
                success=True,
                message="queued synchronized in-place reshard execute",
                elapsed_s=time.time() - t0,
            )

        return self._live_reshard_tp_execute(recv_req, t0)

    def _live_reshard_tp_execute(self: Scheduler, recv_req, t0: float):
        from sglang.srt.managers.io_struct import (
            PauseGenerationReqInput,
            ContinueGenerationReqInput,
            ReshardReqOutput,
        )

        timings = {}
        new_tp = getattr(recv_req, "new_tp_size", None)
        nccl_port = getattr(recv_req, "nccl_port", 29500)
        old_tp = self.tp_worker.model_runner.tp_size

        self._publish_inplace_reshard_status(
            phase="executing",
            active_tp=old_tp,
            target_tp=new_tp,
            old_tp=old_tp,
            message=f"committing TP{old_tp}→TP{new_tp}",
        )

        # Drop queued work before pausing so clients get a fast error instead of
        # hanging until timeout when init_running_status() clears the queue.
        self._abort_waiting_queue_for_inplace_reshard()

        # Phase 1: Pause now that no forward is in-flight.
        t1 = time.time()
        self.pause_generation(PauseGenerationReqInput(mode="in_place"))
        self.chunked_req = None
        self.running_batch = ScheduleBatch(reqs=[], batch_is_full=False)
        timings["drain_ms"] = (time.time() - t1) * 1000
        logger.info("  Phase 1 drain complete: %.1fms", timings["drain_ms"])

        # Phase 2: NCCL weight transfer
        t2 = time.time()
        self._detach_inplace_reshard_kv_refs()
        ok, msg, xfer_elapsed = self.tp_worker.model_runner.live_reshard_tp(
            new_tp=new_tp,
            nccl_port=nccl_port,
        )
        timings["transfer_ms"] = xfer_elapsed * 1000
        logger.info("  Phase 2 transfer: %.1fms - %s", timings["transfer_ms"], msg)

        if not ok:
            self.continue_generation(ContinueGenerationReqInput())
            self._publish_inplace_reshard_status(
                phase="failed",
                active_tp=old_tp,
                target_tp=new_tp,
                old_tp=old_tp,
                message=msg,
                elapsed_s=time.time() - t0,
            )
            return ReshardReqOutput(success=False, message=msg, elapsed_s=time.time()-t0)

        from sglang.srt.distributed import (
            get_pp_group,
            get_tp_group,
            get_world_group,
        )
        from sglang.srt.layers.dp_attention import (
            get_attention_cp_group,
            get_attention_tp_group,
            compute_dp_attention_world_info,
        )
        self.tp_size = new_tp
        self.server_args.tp_size = new_tp
        self.tp_worker.tp_size = new_tp
        self.attn_tp_rank, self.attn_tp_size, self.attn_dp_rank = (
            compute_dp_attention_world_info(
                self.server_args.enable_dp_attention,
                self.tp_rank,
                self.tp_size,
                self.dp_size,
                self.attn_cp_size,
            )
        )
        self.tp_group = get_tp_group()
        self.tp_cpu_group = self.tp_group.cpu_group
        self.attn_tp_group = get_attention_tp_group()
        self.attn_tp_cpu_group = self.attn_tp_group.cpu_group
        self.attn_cp_group = get_attention_cp_group()
        self.attn_cp_cpu_group = self.attn_cp_group.cpu_group
        self.pp_group = get_pp_group()
        self.world_group = get_world_group()
        self.dp_tp_group = (
            self.attn_tp_group
            if self.server_args.enable_dp_attention
            else self.tp_group
        )
        self.dp_tp_cpu_group = self.dp_tp_group.cpu_group

        # Phase 3: UCX hot reconnect (if AFD mode with UCX communicator)
        t3 = time.time()
        try:
            from sglang.srt.layers.afd import get_afd_communicator
            comm = get_afd_communicator()
            if comm is not None and hasattr(comm, 'reconnect'):
                comm.reconnect(timeout=30)
                timings["ucx_reconnect_ms"] = (time.time() - t3) * 1000
                logger.info("  Phase 3 UCX reconnect: %.1fms", timings["ucx_reconnect_ms"])
            else:
                timings["ucx_reconnect_ms"] = 0
        except Exception as e:
            timings["ucx_reconnect_ms"] = (time.time() - t3) * 1000
            logger.warning("  Phase 3 UCX reconnect failed: %s (%.1fms)", e, timings["ucx_reconnect_ms"])

        # Phase 4: Resume. In the experimental in-place path, KV metadata has
        # already been rebuilt and torch.cuda.empty_cache() can block behind
        # recent NCCL P2P work, so skip full flush here.
        t4 = time.time()
        if self.server_args.inplace_reshard_max_tp is None:
            self.flush_cache()
        else:
            # Fully reset rank0's scheduler state to EXACTLY match the freshly
            # activated rank, which calls init_running_status() and starts with an
            # empty radix tree_cache. Phase 1 already drained in-flight requests,
            # so this is safe. If rank0 kept its old waiting_queue / running_batch
            # / radix prefix cache while the joining rank started empty, the two
            # ranks would make DIFFERENT continuous-batching / prefix-hit decisions
            # for the same incoming requests, produce mismatched forward batch
            # shapes, and deadlock the TP collectives (seen only under concurrent
            # load, not single-request tests).
            # CRITICAL: model_runner rebuilt the KV pool during reshard, so the
            # scheduler's cached pool references (and the radix tree_cache built on
            # top of the OLD allocator) are now stale. Keeping them causes the
            # allocator free-list accounting to drift and triggers a
            # "token_to_kv_pool_allocator memory leak detected" assert when a
            # request finishes. Re-bind to the freshly built pool and rebuild the
            # tree_cache on the new allocator, mirroring scheduler __init__.
            self.req_to_token_pool, self.token_to_kv_pool_allocator = (
                self.tp_worker.get_memory_pool()
            )
            self._reshard_rebuild_tree_cache()
            self.tp_worker.finalize_inplace_reshard_activation()
            (
                self.max_total_num_tokens,
                self.max_prefill_tokens,
                self.max_running_requests,
                self.max_queued_requests,
                self.max_req_len,
                self.max_req_input_len,
                self.random_seed,
                self.device,
                self.forward_stream,
                _,
                _,
                _,
            ) = self.tp_worker.get_worker_info()
            self._reshard_finalize_scheduler_for_resume()
        import torch.distributed as dist
        dist.barrier(group=self.tp_cpu_group)
        self.continue_generation(ContinueGenerationReqInput())
        if self.tp_rank == 0:
            self.tp_worker.model_runner._schedule_inplace_reshard_background_memory_trim()
        if self.server_args.inplace_reshard_max_tp is not None:
            broadcast_pyobj(
                {"action": "join_active_loop", "new_tp_size": new_tp},
                self.world_group.rank,
                self.world_group.cpu_group,
                src=self.world_group.ranks[0],
            )
        timings["resume_ms"] = (time.time() - t4) * 1000
        logger.info("  Phase 4 resume: %.1fms", timings["resume_ms"])
        self._reshard_state_dump("post_reshard")

        total_ms = (time.time() - t0) * 1000
        timings["total_ms"] = total_ms
        result_msg = (
            f"live_reshard_tp TP{old_tp}→TP{new_tp} done in {total_ms:.0f}ms: "
            f"drain={timings['drain_ms']:.0f}ms "
            f"transfer={timings['transfer_ms']:.0f}ms "
            f"ucx={timings.get('ucx_reconnect_ms', 0):.0f}ms "
            f"resume={timings['resume_ms']:.0f}ms"
        )
        logger.info(result_msg)
        if self.server_args.inplace_reshard_max_tp is not None and self.tp_size > 1:
            self.enable_overlap = False
            self._inplace_reshard_restart_event_loop = True
        self._inplace_reshard_prep_key = None
        self._inplace_reshard_prep_pending = None
        self.tp_worker.model_runner.reset_inplace_reshard_prep()
        self._publish_inplace_reshard_status(
            phase="done",
            active_tp=new_tp,
            target_tp=new_tp,
            old_tp=old_tp,
            message=result_msg,
            elapsed_s=total_ms / 1000,
            timings=timings,
        )
        return ReshardReqOutput(success=True, message=result_msg, elapsed_s=total_ms/1000)

    def _detach_inplace_reshard_kv_refs(self):
        """Drop scheduler/model_runner/tree_cache refs to old KV pools before rebuild.

        Without this, model_runner allocates a fresh KV pool while the scheduler
        (and radix tree_cache) still pin the old one, roughly doubling GPU memory
        on active follower ranks after multi-hop reshard."""
        import gc as _gc

        import torch

        tc = getattr(self, "tree_cache", None)
        if tc is not None:
            try:
                tc.reset()
            except Exception:
                pass
            if hasattr(tc, "req_to_token_pool"):
                tc.req_to_token_pool = None
            if hasattr(tc, "token_to_kv_pool_allocator"):
                tc.token_to_kv_pool_allocator = None

        self.req_to_token_pool = None
        self.token_to_kv_pool_allocator = None

        tw = self.tp_worker
        tw.req_to_token_pool = None
        tw.token_to_kv_pool_allocator = None

        mr = tw.model_runner
        stale = (
            mr.req_to_token_pool,
            mr.token_to_kv_pool_allocator,
            mr.token_to_kv_pool,
        )
        mr.req_to_token_pool = None
        mr.token_to_kv_pool_allocator = None
        mr.token_to_kv_pool = None
        del stale
        _gc.collect()
        if mr.device == "cuda":
            torch.cuda.synchronize()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

    def _abort_waiting_queue_for_inplace_reshard(self):
        """Abort queued (not yet running) requests before in-place reshard."""
        from http import HTTPStatus

        from sglang.srt.disaggregation.utils import prepare_abort

        if not self.waiting_queue:
            return
        aborted = list(self.waiting_queue)
        self.waiting_queue.clear()
        msg = "Server is reconfiguring tensor parallelism; please retry."
        for req in aborted:
            prepare_abort(req, msg, status_code=HTTPStatus.SERVICE_UNAVAILABLE)
        self.stream_output(aborted, any(r.return_logprob for r in aborted))

    def _reshard_finalize_scheduler_for_resume(self):
        """Align all active TP scheduler ranks after in-place reshard.

        Clears overlap bookkeeping that would desync rank0 (already in the
        event loop) from a freshly-activated rank, and rebuilds schedule policy
        on the new tree_cache so both ranks make identical batching decisions."""
        if self.enable_overlap and hasattr(self, "result_queue"):
            while self.result_queue:
                tmp_batch, tmp_result = self.result_queue.popleft()
                self.process_batch_result(tmp_batch, tmp_result)
            self.result_queue.clear()
        self.chunked_req = None
        self.last_batch = None
        self.cur_batch = None
        if self.server_args.inplace_reshard_max_tp is not None and self.tp_size > 1:
            # Overlap lets rank0 run ahead of a freshly-joined rank and desync
            # per-forward TP collectives under concurrent load.
            self.enable_overlap = False
        self.init_schedule_policy()
        self.init_running_status()

    def _reshard_rebuild_tree_cache(self):
        """Re-point the radix tree_cache at the freshly rebuilt KV pool after an
        in-place reshard, then reset it. RadixCache caches req_to_token_pool and
        token_to_kv_pool_allocator at construction, so after the pool is rebuilt
        the cache would otherwise free indices into a stale allocator and corrupt
        its free-list accounting (memory-leak assert on request completion)."""
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            self.tp_worker.get_memory_pool()
        )
        tc = getattr(self, "tree_cache", None)
        if tc is None:
            return
        try:
            if hasattr(tc, "req_to_token_pool"):
                tc.req_to_token_pool = self.req_to_token_pool
            if hasattr(tc, "token_to_kv_pool_allocator"):
                tc.token_to_kv_pool_allocator = self.token_to_kv_pool_allocator
                if self.token_to_kv_pool_allocator is not None:
                    tc.device = self.token_to_kv_pool_allocator.device
            tc.reset()
            logger.info("reshard: rebound + reset tree_cache to new KV pool")
        except Exception as e:
            logger.warning("reshard: tree_cache rebuild failed: %s", e)

    def _export_weights_for_reshard(self: Scheduler, recv_req):
        from sglang.srt.managers.io_struct import ReshardReqOutput
        from sglang.srt.reshard.weight_exporter import WeightExporter
        from pathlib import Path

        t0 = time.time()
        try:
            model = self.tp_worker.model_runner.model
            tp_size = self.tp_worker.model_runner.tp_size
            tp_rank = self.tp_worker.model_runner.tp_rank
            ipc_dir = Path(recv_req.ipc_dir or "/tmp/sglang_reshard")
            module_type = recv_req.module_type or "prefill"
            perspective = recv_req.perspective or "full"

            exporter = WeightExporter(
                model=model, tp_size=tp_size, tp_rank=tp_rank,
                module_type=module_type, perspective=perspective,
                export_dir=ipc_dir,
            )
            path = exporter.export()
            return ReshardReqOutput(success=True, message=f"Exported to {path}", elapsed_s=time.time()-t0)
        except Exception as e:
            return ReshardReqOutput(success=False, message=str(e), elapsed_s=time.time()-t0)

    def _ipc_reconnect(self: Scheduler, recv_req):
        """Trigger IPC reconnect in the scheduler process.

        This destroys the current IPC communicator and creates a new one
        that will listen for a new handshake from a restarted peer.
        Called during graceful reshard when the peer module restarts.
        """
        from sglang.srt.managers.io_struct import ReshardReqOutput

        t0 = time.time()
        try:
            from sglang.srt.layers.afd import trigger_ipc_reconnect
            success = trigger_ipc_reconnect()
            if success:
                return ReshardReqOutput(
                    success=True,
                    message="IPC reconnect initiated in scheduler",
                    elapsed_s=time.time() - t0,
                )
            else:
                return ReshardReqOutput(
                    success=False,
                    message="Communicator does not support reconnect",
                    elapsed_s=time.time() - t0,
                )
        except Exception as e:
            import traceback
            return ReshardReqOutput(
                success=False,
                message=f"{e}\n{traceback.format_exc()}",
                elapsed_s=time.time() - t0,
            )

    def _do_reshard(self: Scheduler, recv_req):
        from sglang.srt.managers.io_struct import ReshardReqOutput
        from sglang.srt.reshard.reshard_controller import ReshardController

        t0 = time.time()
        try:
            controller = ReshardController(self.tp_worker.model_runner)
            success, elapsed = controller.begin_reshard(
                new_tp_size=recv_req.new_tp_size,
                new_tp_rank=recv_req.new_tp_rank,
            )
            return ReshardReqOutput(
                success=success,
                message="Reshard complete" if success else "Reshard failed",
                elapsed_s=elapsed,
            )
        except Exception as e:
            return ReshardReqOutput(success=False, message=str(e), elapsed_s=time.time()-t0)


def _export_static_state(model):
    return dict(
        buffers=[
            (name, buffer.detach().clone()) for name, buffer in model.named_buffers()
        ]
    )


def _import_static_state(model, static_params):
    self_named_buffers = dict(model.named_buffers())
    for name, tensor in static_params["buffers"]:
        self_named_buffers[name][...] = tensor
