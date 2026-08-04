"""AFD (Attention-FFN Disaggregation) scheduler mixin.

Provides reusable AFD scheduling helpers that can be mixed into any event loop
(normal, disagg_prefill, disagg_decode, etc.).
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import timedelta
from typing import TYPE_CHECKING

import zmq

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class SchedulerAFDMixin:
    def afd_component_init_lifecycle(self: "Scheduler") -> None:
        """Bootstrap a dedicated max-world control group on every participant rank."""
        self._afd_component_control_group = None
        self._afd_component_data_group = None
        self._afd_component_staging_control_group = None
        self._afd_component_lifecycle = None
        if not getattr(self.server_args, "enable_afd_component_reshard_participant", False):
            return
        # Standby schedulers return before afd_init_state(), but they still
        # participate in max-world QUIESCE/ACTIVATE boundaries.  Create their
        # epoch ledger at the feature lifecycle boundary; the getter remains a
        # lazy fallback for alternate initialization orderings.
        self.afd_reshard_drain_state()
        import torch.distributed as dist
        from sglang.srt.layers.afd import afd_is_attn
        from sglang.srt.reshard.afd_component_standby import AFDComponentRankLifecycle

        max_tp = int(self.server_args.afd_component_max_tp)
        active_tp = int(self.server_args.tp_size)
        ranks = list(range(max_tp))
        # These groups are intentionally not registered in parallel_state, so
        # destroy_model_parallel() cannot destroy them during serving-TP rebuild.
        # Every max-world process creates them in this exact order.
        collective_timeout = timedelta(seconds=float(self.server_args.afd_reshard_timeout))
        self._afd_component_control_group = dist.new_group(
            ranks=ranks, backend="gloo", timeout=collective_timeout
        )
        # Create every possible active-prefix control group in deterministic
        # max-world order. They are feature-local and survive serving-TP rebuilds;
        # request broadcasts can therefore never share a collective generation
        # with fence propagation or ready consensus.
        self._afd_component_active_control_groups = {}
        for prefix_tp in range(1, max_tp + 1):
            group = dist.new_group(
                ranks=list(range(prefix_tp)),
                backend="gloo",
                timeout=collective_timeout,
            )
            self._afd_component_active_control_groups[prefix_tp] = group
        self._afd_component_active_control_group = (
            self._afd_component_active_control_groups[active_tp]
        )
        self._afd_component_active_control_tp = active_tp
        self._afd_component_active_control_generation = 0
        # A newly activated serving world is not command-ready until every
        # target rank has completed scheduler/communicator initialization and
        # met at the first common post-activation loop boundary.
        self._afd_component_serving_ready = True
        self._afd_component_serving_ready_operation = None
        # Background descriptor/error collectives must never share the scheduler
        # command group above: active schedulers broadcast on it every safe point.
        # Build every required max-world group now, in one deterministic order.
        from sglang.srt.reshard.afd_component_weight_staging import (
            create_afd_component_staging_groups,
        )

        staging_backend = __import__("os").getenv(
            "AFD_COMPONENT_STAGING_BACKEND", "gloo"
        ).lower()
        staging_groups = create_afd_component_staging_groups(
            dist, ranks, staging_backend, collective_timeout
        )
        self._afd_component_gloo_data_group = staging_groups["gloo"]
        self._afd_component_nccl_data_group = staging_groups["nccl"]
        self._afd_component_staging_control_group = staging_groups["control"]
        component = "attn" if afd_is_attn() else "ffn"
        self._afd_component_lifecycle = AFDComponentRankLifecycle(
            self.tp_rank, active_tp, max_tp, component
        )
        runner = self.tp_worker.model_runner
        runner.attach_afd_component_stager(
            self._afd_component_gloo_data_group,
            self._afd_component_nccl_data_group,
            self._afd_component_staging_control_group,
            shadow_device=__import__("os").getenv("AFD_COMPONENT_SHADOW_DEVICE", "auto"),
            collective_commands_enabled=False,
            group_refresh_callback=self.afd_component_refresh_scheduler_groups,
            topology_refresh_callback=self.afd_component_refresh_pd_topology,
        )
        # Do not advertise commit capability until every rank has constructed
        # both groups and attached its stager.
        dist.barrier(group=self._afd_component_staging_control_group)
        runner._afd_component_collective_commands_enabled = True
        self._afd_component_pending_world_command = None
        self._afd_component_stop_active_loop = False
        self._afd_component_restart_active_iteration = False
        self._afd_component_readiness_schema_log_tick = 0
        # Quiesce is driven by the active serving ranks, not by rank 0's RPC
        # handler.  These fields exist on every active rank and are propagated
        # over a feature-local active-prefix CPU group at each loop iteration.
        self._afd_component_pending_fence_payload = None
        self._afd_component_fence_payload = None
        self._afd_component_fence_installed = False
        self._afd_component_fence_synced = False
        self._afd_component_fence_sealed = False
        self._afd_component_quiesce_world_done = False
        self._afd_component_quiesce_world_operation = None
        self._afd_component_quiesce_log_tick = 0
        self._afd_component_fence_monotonic = None
        self._afd_component_transfer_log_monotonic = 0.0
        # Rank 0's transport thread only sets this event.  Active scheduler
        # ranks observe the signal through the active-prefix control group.
        self._afd_component_external_control_pending = __import__(
            "threading"
        ).Event()

    def afd_component_receive_world_command(self: "Scheduler"):
        """Receive only the feature-local world command schema/group."""
        from sglang.srt.utils.common import broadcast_pyobj

        payload = broadcast_pyobj(
            None, self.tp_rank, self._afd_component_control_group, src=0
        )
        if isinstance(payload, list) and len(payload) == 1:
            payload = payload[0]
        return payload

    def afd_component_apply_safe_point(self: "Scheduler", payload):
        """Execute one already-fanned-out command between model forwards."""
        from sglang.srt.reshard.afd_component_standby import (
            AFDComponentWorldAction,
            AFDComponentWorldCommand,
        )

        cmd = AFDComponentWorldCommand.parse(payload)
        if cmd.action == AFDComponentWorldAction.QUIESCE:
            if "watermark" not in payload:
                raise RuntimeError("AFD max-world quiesce descriptor has no cut watermark")
            try:
                watermark = int(payload["watermark"])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "AFD max-world quiesce cut watermark must be an integer"
                ) from exc
            if watermark < 0 or payload["watermark"] != watermark:
                raise RuntimeError(
                    "AFD max-world quiesce cut watermark must be a non-negative integer"
                )
            # Active ranks verify their already-sealed cut; standby ranks lazily
            # create a ledger and install the same cut before ACTIVATE can reopen it.
            self.afd_reshard_drain_state().install_external_cut(watermark)
        runner = self.tp_worker.model_runner
        request = dict(payload)
        request.setdefault("operation_id", cmd.operation)
        request.setdefault("epoch", cmd.epoch)
        try:
            if cmd.action == AFDComponentWorldAction.PREPARE:
                runner._afd_component_stager.start_prepare(request)
            elif cmd.action == AFDComponentWorldAction.PREPARE_STATUS:
                return runner._afd_component_stager.prepare_status(cmd.operation)
            elif cmd.action == AFDComponentWorldAction.ACTIVATE:
                runner._afd_component_stager.activate(request)
            elif cmd.action == AFDComponentWorldAction.ABORT:
                runner._afd_component_stager.cancel_prepare("component operation aborted")
            elif cmd.action == AFDComponentWorldAction.RETIRE:
                runner._afd_component_stager.retire(request)
        except Exception as exc:
            # Staging/activation uses an all-rank error consensus.  Followers
            # must remain alive in their standby loop; rank zero converts the
            # consensus failure into the fanout failure ACK.
            runner._afd_component_stager.abort(str(exc))
            self._afd_component_last_failure = str(exc)
            abort_payload = {**payload, "action": "abort"}
            state = self._afd_component_lifecycle.apply_safe_point(abort_payload)
            self.is_afd_component_standby_rank = self._afd_component_lifecycle.is_standby
            runner.is_afd_component_standby_rank = self.is_afd_component_standby_rank
            self.tp_worker.is_afd_component_standby_rank = self.is_afd_component_standby_rank
            if self.tp_rank == 0:
                raise RuntimeError(f"AFD component fanout failed: {exc}") from exc
            return state

        if cmd.action == AFDComponentWorldAction.PREPARE_STATUS:
            return runner._afd_component_stager.prepare_status(cmd.operation)
        prev_active_tp = int(self._afd_component_lifecycle.active_tp)
        state = self._afd_component_lifecycle.apply_safe_point(payload)
        self.is_afd_component_standby_rank = self._afd_component_lifecycle.is_standby
        runner.is_afd_component_standby_rank = self.is_afd_component_standby_rank
        self.tp_worker.is_afd_component_standby_rank = self.is_afd_component_standby_rank
        if cmd.action in (
            AFDComponentWorldAction.ABORT,
            AFDComponentWorldAction.RETIRE,
        ):
            self.afd_component_clear_quiesce()
        if cmd.action == AFDComponentWorldAction.ACTIVATE:
            # Under AFD_RESHARD_FAKE_STAGING the stager activate() is a no-op:
            # the real serving process groups are NOT rebuilt and no model/KV is
            # built on joining ranks. Refreshing the scheduler's tp_size/groups
            # or promoting a joining rank into the active serving loop would
            # diverge from the live topology and crash on the missing KV runtime.
            # Undo the per-rank promotion (keep the epoch advance) so joining
            # ranks stay standby and initial active ranks keep serving. The
            # coordinator still publishes the target topology (test-only path).
            if __import__("os").getenv("AFD_RESHARD_FAKE_STAGING", "0") == "1":
                state = self._afd_component_reset_lifecycle_active_tp(prev_active_tp)
                logger.warning(
                    "[AFD-reshard] [FAKE_STAGING] skipping scheduler group "
                    "refresh and joining-rank promotion for op=%s target_tp=%s "
                    "(test-only)",
                    cmd.operation, cmd.target_tp,
                )
            # Real activation refreshes serving-group references inside the
            # stager, immediately after each target/source group rebuild and
            # before its final max-world barrier permits event-loop resumption.
            self.afd_component_complete_transition_boundary(cmd, prev_active_tp)
            if self.tp_rank < prev_active_tp:
                # ACTIVATE may run inside poll_runtime() after the loop preamble.
                # Every source-active rank must abandon that data-plane round:
                # survivors meet joining ranks at readiness, while retired ranks
                # reach the loop-top leave guard and return to standby.
                self._afd_component_restart_active_iteration = True
        return state

    def afd_component_complete_transition_boundary(
        self: "Scheduler", cmd, previous_active_tp: int
    ) -> None:
        """Publish one explicit post-ACTIVATE event-loop resume boundary.

        The stager's final barrier protects weight/runtime commit. This second,
        scheduler-owned max-world handshake protects control-plane phase: every
        rank resets its epoch-local fence/cut and active-control generation before
        old ranks resume request receive, joining ranks enter that same phase, and
        shrinking ranks leave the active loop.
        """
        import torch.distributed as dist

        descriptor = {
            "operation_id": cmd.operation,
            "epoch": int(cmd.epoch),
            "target_tp": int(cmd.target_tp),
            "resume_phase": "post_activate_ready_then_request_receive",
        }
        gathered = [None] * int(self.server_args.afd_component_max_tp)
        dist.all_gather_object(
            gathered, descriptor, group=self._afd_component_control_group
        )
        if any(item != descriptor for item in gathered):
            raise RuntimeError(
                "AFD transition boundary participants disagree: "
                f"expected={descriptor!r}, gathered={gathered!r}"
            )

        drain = self.afd_reshard_drain_state()
        if self.tp_rank == 0:
            drain.reopen_after_transition()
        else:
            drain.reopen_at_transition_boundary()
        self.afd_component_clear_quiesce()
        self._afd_component_active_control_generation = 0
        real_target_world = int(self._afd_component_lifecycle.active_tp) == int(
            cmd.target_tp
        )
        target_active_rank = real_target_world and self.tp_rank < int(cmd.target_tp)
        # Lifecycle.active_tp is replicated across the max-world, so equality
        # alone does not identify participants in the target serving prefix.
        # Retired shrink ranks completed the max-world transition above, but must
        # neither advertise pending readiness nor enter its active-only collective.
        self._afd_component_serving_ready = not target_active_rank
        self._afd_component_serving_ready_operation = (
            cmd.operation if target_active_rank else None
        )

        if target_active_rank:
            # Every target rank must stop before the first request receive in the
            # new serving world. Joining ranks consume this scheduler boundary
            # after initialization; old ranks already consumed it in ACTIVATE.
            self._afd_component_transition_boundary = {
                **descriptor,
                "resume_pending": self.tp_rank >= int(previous_active_tp),
                # ACTIVATE drops TP-bound A/F communicators. This bit belongs
                # to the transition boundary (rather than a process-global
                # status path) so rebuilding is one-shot for this topology and
                # can never be triggered by readiness observation.
                "data_plane_ready": False,
            }
        else:
            self._afd_component_transition_boundary = None
        logger.info(
            "[AFD-reshard] op=%s stage=transition_boundary rank=%d "
            "previous_tp=%d target_tp=%d resume_pending=%s",
            cmd.operation,
            self.tp_rank,
            int(previous_active_tp),
            int(cmd.target_tp),
            bool(
                self._afd_component_transition_boundary
                and self._afd_component_transition_boundary["resume_pending"]
            ),
        )

    def afd_component_resume_at_transition_boundary(self: "Scheduler") -> bool:
        """Let a joining rank consume the boundary before serving readiness."""
        boundary = getattr(self, "_afd_component_transition_boundary", None)
        if not boundary or not boundary.get("resume_pending", False):
            return False
        boundary["resume_pending"] = False
        logger.info(
            "[AFD-reshard] op=%s stage=transition_boundary resume rank=%d "
            "epoch=%d phase=%s",
            boundary["operation_id"],
            self.tp_rank,
            boundary["epoch"],
            boundary["resume_phase"],
        )
        return True

    def afd_component_complete_serving_readiness(self: "Scheduler") -> bool:
        """Complete the first all-active boundary after ACTIVATE.

        No target rank may receive a request in the new serving world first.
        Old ranks enter this barrier at their next loop top after ACTIVATE;
        joining ranks enter it in their first active iteration after scheduler
        and communicator initialization. All ranks then continue from this same
        iteration into the first request receive. This active-control collective
        cannot interfere with serving TP broadcasts or the max-world group.
        """
        # Missing/dynamically synthesized attributes mean the scheduler has no
        # readiness transition. Only an explicitly stored False plus a complete
        # boundary descriptor may consume an active-control generation.
        state = getattr(self, "__dict__", {})
        if state.get("_afd_component_serving_ready", True) is not False:
            return False
        boundary = state.get("_afd_component_transition_boundary")
        required = {
            "operation_id", "epoch", "target_tp", "resume_phase",
            "resume_pending", "data_plane_ready",
        }
        if not isinstance(boundary, dict) or not required.issubset(boundary):
            raise RuntimeError(
                "AFD serving readiness has no valid transition boundary"
            )
        if boundary["resume_pending"] is not False:
            return False
        if str(state.get("_afd_component_serving_ready_operation", "")) != str(
            boundary["operation_id"]
        ):
            raise RuntimeError("AFD serving readiness operation mismatch")

        # reset_afd_communicators() deliberately leaves reconstruction lazy.
        # Rebuild it here, while both A/F HTTP ACTIVATE paths are concurrently
        # driving their target-active ranks, before publishing serving-ready or
        # allowing a subsequent PREPARE to overlap the first user forward.
        if not boundary["data_plane_ready"]:
            if self.tp_rank >= int(boundary["target_tp"]):
                raise RuntimeError(
                    "retired AFD rank attempted data-plane serving readiness"
                )
            if self.afd_component_requires_data_plane_communicator():
                from sglang.srt.layers.afd import get_async_communicator

                try:
                    get_async_communicator()
                except Exception as exc:
                    raise RuntimeError(
                        "AFD post-activate data-plane communicator rebuild failed "
                        f"for operation {boundary['operation_id']}: {exc}"
                    ) from exc
            # Set only after construction returns. On failure the boundary is
            # retained and serving-ready remains false, making retries explicit.
            boundary["data_plane_ready"] = True

        import torch.distributed as dist

        descriptor = {
            "action": "post_activate_ready",
            "operation_id": boundary["operation_id"],
            "epoch": int(boundary["epoch"]),
            "target_tp": int(boundary["target_tp"]),
        }
        SchedulerAFDMixin.afd_component_log_pd_queues(
            self, "post_activate_ready_before"
        )
        logger.info(
            "[AFD-reshard] op=%s stage=post_activate_ready enter rank=%d "
            "epoch=%d target_tp=%d generation=%d",
            descriptor["operation_id"], self.tp_rank, descriptor["epoch"],
            descriptor["target_tp"], self._afd_component_active_control_generation,
        )
        gathered = [None] * int(boundary["target_tp"])
        dist.all_gather_object(
            gathered, descriptor, group=self._afd_component_active_control_group
        )
        if any(item != descriptor for item in gathered):
            raise RuntimeError(
                "AFD serving-ready participants disagree: "
                f"expected={descriptor!r}, gathered={gathered!r}"
            )
        self._afd_component_active_control_generation += 1
        self._afd_component_serving_ready = True
        self._afd_component_serving_ready_operation = None
        self._afd_component_transition_boundary = None
        # Trigger deferred CUDA graph capture if it was skipped during ACTIVATE.
        self._afd_maybe_deferred_graph_capture()
        SchedulerAFDMixin.afd_component_log_pd_queues(
            self, "post_activate_ready_after"
        )
        logger.info(
            "[AFD-reshard] op=%s stage=post_activate_ready rank=%d "
            "epoch=%d target_tp=%d generation=%d",
            descriptor["operation_id"], self.tp_rank, descriptor["epoch"],
            descriptor["target_tp"], self._afd_component_active_control_generation,
        )
        return True

    def _afd_maybe_deferred_graph_capture(self: "Scheduler") -> None:
        """Capture CUDA graphs if deferred during ACTIVATE for lower downtime."""
        import time as _time

        runner = getattr(self.tp_worker, "model_runner", None)
        if runner is None:
            return
        if not getattr(runner, "_afd_deferred_graph_capture", False):
            return
        runner._afd_deferred_graph_capture = False
        t0 = _time.time()
        try:
            runner.init_device_graphs()
            runner.init_piecewise_cuda_graphs()
            logger.info(
                "[AFD-reshard] deferred CUDA graph capture rank=%d %.3fs",
                self.tp_rank,
                _time.time() - t0,
            )
        except Exception as exc:
            logger.warning(
                "[AFD-reshard] deferred CUDA graph capture failed rank=%d: %s",
                self.tp_rank,
                exc,
            )

    def afd_component_should_eager_init_data_plane(self: "Scheduler") -> bool:
        """Keep joining ranks on the transition-boundary rebuild path."""
        state = getattr(self, "__dict__", {})
        boundary = state.get("_afd_component_transition_boundary")
        return not (
            state.get("_afd_component_serving_ready", True) is False
            and isinstance(boundary, dict)
            and boundary.get("resume_pending", False)
        )

    def afd_component_requires_data_plane_communicator(self: "Scheduler") -> bool:
        """Whether this post-ACTIVATE participant owns an A/F data channel."""
        args = getattr(self, "server_args", None)
        if not getattr(args, "enable_afd_component_reshard_participant", False):
            return False
        perspective = getattr(args, "afd_perspective", None)
        value = str(getattr(perspective, "value", perspective)).lower()
        if value.endswith("attn") or value.endswith("ffn"):
            return True
        raise RuntimeError(
            "AFD component participant has no Attention/FFN perspective for "
            "post-activate data-plane readiness"
        )

    def afd_component_begin_active_iteration(self: "Scheduler") -> None:
        """Run one control preamble before the iteration's request receive."""
        # ACTIVATE is a max-world operation and may demote this scheduler while
        # it is still unwinding an active-loop iteration. Route retired ranks to
        # the outer standby lifecycle before inspecting target-only readiness.
        if getattr(self, "__dict__", {}).get(
            "is_afd_component_standby_rank", False
        ):
            self._afd_component_stop_active_loop = True
            self._afd_component_serving_ready = True
            self._afd_component_serving_ready_operation = None
            self._afd_component_transition_boundary = None
            logger.info(
                "[AFD-reshard] stage=retired_to_standby rank=%d",
                self.tp_rank,
            )
            return

        # Old ranks arrive here after ACTIVATE's one-shot iteration restart.
        # Joining ranks consume their boundary here after scheduler/communicator
        # initialization, then immediately enter the same readiness generation.
        resumed = self.afd_component_resume_at_transition_boundary()
        state = getattr(self, "__dict__", {})
        if state.get("_afd_component_serving_ready", True) is False:
            boundary = state.get("_afd_component_transition_boundary")
            required = {
                "operation_id", "epoch", "target_tp", "resume_phase",
                "resume_pending", "data_plane_ready",
            }
            valid = (
                isinstance(boundary, dict)
                and required.issubset(boundary)
                and boundary.get("resume_pending") is False
                and str(state.get("_afd_component_serving_ready_operation", ""))
                == str(boundary.get("operation_id", ""))
            )
            if not valid:
                tick = int(state.get("_afd_component_readiness_schema_log_tick", 0)) + 1
                self._afd_component_readiness_schema_log_tick = tick
                if tick == 1 or tick % 100 == 0:
                    logger.error(
                        "[AFD-reshard] stage=post_activate_ready schema_rejected "
                        "rank=%d resumed=%s serving_ready=%r operation=%r "
                        "boundary=%r missing=%s",
                        self.tp_rank, resumed,
                        state.get("_afd_component_serving_ready"),
                        state.get("_afd_component_serving_ready_operation"),
                        boundary,
                        sorted(required.difference(boundary or {}))
                        if isinstance(boundary, dict) else sorted(required),
                    )
                raise RuntimeError(
                    "AFD post-activate readiness has an invalid pending boundary"
                )
            logger.info(
                "[AFD-reshard] op=%s stage=post_activate_ready pending rank=%d "
                "epoch=%d target_tp=%d resumed=%s",
                boundary["operation_id"], self.tp_rank, int(boundary["epoch"]),
                int(boundary["target_tp"]), resumed,
            )
            if not self.afd_component_complete_serving_readiness():
                raise RuntimeError("AFD post-activate readiness did not advance")
            # Readiness is this round's complete control preamble. The caller
            # continues into the first request receive in the same iteration.
            return
        if resumed:
            # Fake-staging can retain the historical resume-only boundary.
            return
        self.afd_component_sync_quiesce_request()
        self.afd_component_poll_runtime()
        self.afd_component_safe_point()

    def _afd_component_reset_lifecycle_active_tp(self, active_tp: int):
        """Restore active_tp on the per-rank lifecycle (fake-mode activate).

        Keeps the epoch advance / operation clearing performed by the ACTIVATE
        safe point, but leaves the serving topology unchanged so joining ranks
        remain standby instead of entering the active loop without a KV runtime.
        """
        from sglang.srt.reshard.afd_component_standby import AFDComponentRankState

        lifecycle = self._afd_component_lifecycle
        lifecycle.active_tp = int(active_tp)
        lifecycle.state = (
            AFDComponentRankState.ACTIVE
            if lifecycle.rank < int(active_tp)
            else AFDComponentRankState.STANDBY
        )
        self.is_afd_component_standby_rank = lifecycle.is_standby
        runner = self.tp_worker.model_runner
        runner.is_afd_component_standby_rank = self.is_afd_component_standby_rank
        self.tp_worker.is_afd_component_standby_rank = self.is_afd_component_standby_rank
        return lifecycle.state

    def afd_component_pd_queue_snapshot(self: "Scheduler") -> dict:
        """Collect compact PD queue state without assuming concrete queue APIs."""
        result = {}
        for name, attr in (
            ("decode_prealloc", "disagg_decode_prealloc_queue"),
            ("decode_transfer", "disagg_decode_transfer_queue"),
        ):
            queue = getattr(self, attr, None)
            snapshot = getattr(queue, "snapshot", None)
            if callable(snapshot):
                result[name] = snapshot()
            elif queue is not None:
                result[name] = {"len": len(getattr(queue, "queue", ()))}
        return result

    def afd_component_log_pd_queues(self: "Scheduler", stage: str) -> None:
        logger.info(
            "[AFD-reshard] stage=%s rank=%d pd_queues=%s",
            stage, int(getattr(self, "tp_rank", -1)),
            SchedulerAFDMixin.afd_component_pd_queue_snapshot(self),
        )

    def afd_component_refresh_scheduler_groups(self: "Scheduler", target_tp: int) -> None:
        """Refresh scheduler/worker references after all ranks rebuild groups."""
        SchedulerAFDMixin.afd_component_log_pd_queues(self, "group_refresh_before")
        from sglang.srt.distributed import get_pp_group, get_world_group
        from sglang.srt.distributed.parallel_state import get_tp_group
        from sglang.srt.layers.dp_attention import (
            compute_dp_attention_world_info,
            get_attention_cp_group,
            get_attention_tp_group,
        )

        self.tp_size = target_tp
        self.server_args.tp_size = target_tp
        self.tp_worker.tp_size = target_tp
        self.tp_worker.tp_group = get_tp_group()
        self.tp_worker.pp_group = get_pp_group()
        self.tp_worker.world_group = get_world_group()
        self.tp_group = self.tp_worker.tp_group
        self.tp_cpu_group = self.tp_group.cpu_group
        self.attn_tp_group = get_attention_tp_group()
        self.attn_tp_cpu_group = self.attn_tp_group.cpu_group
        self.attn_cp_group = get_attention_cp_group()
        self.attn_cp_cpu_group = self.attn_cp_group.cpu_group
        self.pp_group = self.tp_worker.pp_group
        self.world_group = self.tp_worker.world_group
        self.dp_tp_group = (
            self.attn_tp_group
            if self.server_args.enable_dp_attention
            else self.tp_group
        )
        self.dp_tp_cpu_group = self.dp_tp_group.cpu_group

        # Scheduler-owned helpers cache process-group objects at construction.
        # Rebuilding parallel state does not mutate those Python references, so
        # publish the target groups before any rank can resume its event loop.
        # This is also required on activation rollback, where PD topology is not
        # reconstructed after the source groups are rebuilt.
        decode_prealloc = getattr(self, "disagg_decode_prealloc_queue", None)
        if decode_prealloc is not None and hasattr(decode_prealloc, "gloo_group"):
            decode_prealloc.gloo_group = self.attn_tp_cpu_group
            decode_prealloc.tp_rank = self.tp_rank
            decode_prealloc.tp_size = target_tp
        decode_transfer = getattr(self, "disagg_decode_transfer_queue", None)
        if decode_transfer is not None and hasattr(decode_transfer, "gloo_group"):
            decode_transfer.gloo_group = self.attn_tp_cpu_group
            decode_transfer.tp_rank = self.tp_rank
            decode_transfer.tp_size = target_tp
        prefill_bootstrap = getattr(self, "disagg_prefill_bootstrap_queue", None)
        if prefill_bootstrap is not None and hasattr(prefill_bootstrap, "gloo_group"):
            prefill_bootstrap.gloo_group = self.attn_tp_cpu_group
            prefill_bootstrap.tp_rank = self.tp_rank
            prefill_bootstrap.tp_size = self.attn_tp_group.world_size
            prefill_bootstrap.collective_rank = self.attn_tp_group.rank
            prefill_bootstrap.collective_src_rank = self.attn_tp_group.first_rank
        grammar_manager = getattr(self, "grammar_manager", None)
        if grammar_manager is not None:
            grammar_manager.grammar_sync_group = self.dp_tp_cpu_group
            grammar_manager.grammar_sync_size = self.dp_tp_group.world_size
            grammar_manager.grammar_sync_entry = self.dp_tp_group.first_rank
            grammar_manager.is_grammar_sync_entry = self.dp_tp_group.is_first_rank

        groups = getattr(self, "_afd_component_active_control_groups", None)
        if groups is None or target_tp not in groups:
            raise RuntimeError(
                f"AFD active-control group for TP{target_tp} was not initialized"
            )
        self._afd_component_active_control_group = groups[target_tp]
        self._afd_component_active_control_tp = target_tp
        self._afd_component_active_control_generation = 0
        (
            self.attn_tp_rank,
            self.attn_tp_size,
            self.attn_dp_rank,
        ) = compute_dp_attention_world_info(
            self.server_args.enable_dp_attention,
            self.tp_rank,
            target_tp,
            self.dp_size,
            self.attn_cp_size,
        )
        SchedulerAFDMixin.afd_component_log_pd_queues(self, "group_refresh_after")

    def afd_component_refresh_pd_topology(
        self: "Scheduler", target_tp: int, generation: int
    ) -> None:
        """Rebuild PD state and publish component bootstrap topology before resume."""
        import time as _time

        from sglang.srt.disaggregation.utils import DisaggregationMode
        from sglang.srt.layers.afd import afd_is_attn

        if not afd_is_attn():
            return
        mode = getattr(self, "disaggregation_mode", None)
        if mode is None:
            mode = DisaggregationMode(self.server_args.disaggregation_mode)
        if mode not in (DisaggregationMode.PREFILL, DisaggregationMode.DECODE):
            return
        was_standby = bool(getattr(self, "is_afd_component_standby_rank", False))
        t0 = _time.time()
        logger.info(
            "[AFD-reshard] refresh_pd_topology: target_tp=%d generation=%d "
            "mode=%s was_standby=%s tp_rank=%d",
            target_tp, generation, mode.value, was_standby,
            int(getattr(self, "tp_rank", 0)),
        )

        self.server_args.afd_component_bootstrap_generation = int(generation)
        if was_standby:
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
        t_cache = _time.time()
        self.init_cache_with_memory_pool()
        t_status = _time.time()
        self.init_running_status()
        t_disagg = _time.time()
        self.init_disaggregation()
        t_done = _time.time()
        logger.info(
            "[AFD-reshard] refresh_pd_topology done: "
            "tp_rank=%d total=%.3fs cache=%.3fs status=%.3fs disagg=%.3fs",
            int(getattr(self, "tp_rank", 0)),
            t_done - t0, t_status - t_cache, t_disagg - t_status,
            t_done - t_disagg,
        )
        if mode == DisaggregationMode.DECODE:
            manager = self.disagg_decode_prealloc_queue.kv_manager
            manager.invalidate_prefill_topology_cache()
            logger.info(
                "[AFD-reshard] refresh_pd_topology: DECODE invalidated prefill cache, "
                "required_generation=%d tp_rank=%d",
                int(getattr(self.server_args, "afd_component_bootstrap_generation", 0)),
                self.tp_rank,
            )
        if was_standby:
            self._afd_component_pd_runtime_initialized = True
        logger.info(
            "[AFD-reshard] refresh_pd_topology done: tp_rank=%d total=%.3fs "
            "cache=%.3fs status=%.3fs disagg=%.3fs",
            self.tp_rank,
            t_done - t0,
            t_status - t_cache,
            t_disagg - t_status,
            t_done - t_disagg,
        )

    def afd_component_fanout_command(self: "Scheduler", payload):
        """Broadcast and execute a rank-0 command on the component max-world."""
        if self.tp_rank != 0:
            raise RuntimeError("only component rank 0 may fan out commands")
        from sglang.srt.utils.common import broadcast_pyobj

        received = broadcast_pyobj(
            [payload], self.tp_rank, self._afd_component_control_group, src=0
        )[0]
        # poll_runtime() is immediately followed by safe_point(); rank 0 has
        # already consumed this rendezvous, so it must not enter a second one.
        self._afd_component_skip_next_safe_point = True
        state = self.afd_component_apply_safe_point(received)
        if self.is_afd_component_standby_rank:
            # Rank 0 executes fanout from poll_runtime(), not from safe_point(),
            # so the follower-only post-apply branch cannot request loop exit.
            self._afd_component_stop_active_loop = True
        return state

    def afd_component_enter_active_loop(self: "Scheduler") -> None:
        """Initialize a joining scheduler and dispatch its stage-local AFD loop."""
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
            _, _, _,
        ) = self.tp_worker.get_worker_info()
        pd_runtime_initialized = bool(
            getattr(self, "_afd_component_pd_runtime_initialized", False)
        )
        if not pd_runtime_initialized:
            self.init_cache_with_memory_pool()
            self.init_running_status()
        self.init_chunked_prefill()
        self.init_diffusion_llm()
        self.init_schedule_policy()
        self.init_watch_dog_memory_saver_input_blocker()
        self.init_profiler()
        if not pd_runtime_initialized:
            self.init_disaggregation()
        self._afd_component_pd_runtime_initialized = False
        self.init_overlap()
        self.maybe_init_ngram_embedding()
        self.init_deterministic_inference_config()
        self.init_request_dispatcher()
        self.afd_component_finish_joining_scheduler_init()
        self.afd_init_state()
        self.afd_component_init_runtime()
        self._init_schedule_stream()
        self._afd_component_active_scheduler_initialized = True
        boundary = getattr(self, "_afd_component_transition_boundary", None)
        if not boundary or not boundary.get("resume_pending", False):
            raise RuntimeError(
                "joining AFD scheduler has no pending transition boundary"
            )

    def afd_component_finish_joining_scheduler_init(self: "Scheduler") -> None:
        """Initialize scheduler services skipped by the standby fast path."""
        if self.enable_lora_overlap_loading:
            from sglang.srt.lora.lora_overlap_loader import LoRAOverlapLoader

            self.lora_overlap_loader = LoRAOverlapLoader(
                self.tp_worker.model_runner.lora_manager
            )

        # Keep this after init_request_dispatcher(), matching normal and native
        # joining scheduler initialization.
        from sglang.srt.constrained.grammar_manager import GrammarManager

        self.grammar_manager = GrammarManager(self)

    def afd_component_should_leave_active_loop(self: "Scheduler") -> bool:
        """Consume the safe-point request to transition into standby."""
        if not getattr(self, "_afd_component_stop_active_loop", False):
            return False
        self._afd_component_stop_active_loop = False
        # A later grow must run the full joining scheduler initialization before
        # re-entering any data-plane loop or consuming its readiness boundary.
        if getattr(self, "__dict__", {}).get(
            "is_afd_component_standby_rank", False
        ):
            self._afd_component_active_scheduler_initialized = False
        return True

    def afd_component_should_restart_active_iteration(self: "Scheduler") -> bool:
        """Consume ACTIVATE's one-shot skip of the remaining data-plane round."""
        if not getattr(self, "_afd_component_restart_active_iteration", False):
            return False
        self._afd_component_restart_active_iteration = False
        logger.info(
            "[AFD-reshard] stage=post_activate_ready restart_iteration rank=%d",
            self.tp_rank,
        )
        return True

    def afd_component_queue_world_command(self: "Scheduler", payload) -> None:
        """Queue rank-0's next component command for a loop safe point."""
        if self.tp_rank != 0:
            raise RuntimeError("only component rank 0 may originate world commands")
        if self._afd_component_pending_world_command is not None:
            raise RuntimeError("an AFD component world command is already pending")
        self._afd_component_pending_world_command = payload

    def afd_component_request_quiesce(self: "Scheduler", payload) -> None:
        """Install rank 0's local fence request without entering a collective."""
        operation = str(payload["operation_id"])
        current = getattr(self, "_afd_component_pending_fence_payload", None)
        if current is None:
            current = getattr(self, "_afd_component_fence_payload", None)
        if current is not None and str(current.get("operation_id")) != operation:
            raise RuntimeError("another AFD component quiesce is already active")
        from sglang.srt.layers.afd import afd_is_ffn
        component = "ffn" if afd_is_ffn() else "attn"
        target_key = "target_ffn_tp" if component == "ffn" else "target_attn_tp"
        self._afd_component_pending_fence_payload = {
            **payload,
            "operation": operation,
            "component": component,
            "action": "quiesce",
            "target_tp": int(payload[target_key]),
        }
        # Requested is rank-0-local until the next pre-poll serving-TP sync.
        # Data admission remains open on every rank for the rest of this round.
        self._afd_component_fence_installed = False
        self._afd_component_fence_synced = False
        self._afd_component_fence_sealed = False
        self._afd_component_quiesce_world_done = False
        self._afd_component_quiesce_world_operation = None

    def afd_component_clear_quiesce(self: "Scheduler") -> None:
        self._afd_component_pending_fence_payload = None
        self._afd_component_fence_payload = None
        self._afd_component_fence_installed = False
        self._afd_component_fence_synced = False
        self._afd_component_fence_sealed = False
        self._afd_component_quiesce_world_done = False
        self._afd_component_quiesce_world_operation = None
        self._afd_component_fence_monotonic = None
        self._afd_component_transfer_log_monotonic = 0.0

    def afd_component_scheduler_idle_reason(self: "Scheduler"):
        """Describe the scheduler-owned queue that still prevents drain."""
        if self.is_fully_idle():
            return None

        def size(value, attr=None):
            if value is None:
                return 0
            if attr is not None:
                value = getattr(value, attr, ())
            value = getattr(value, "reqs", value)
            try:
                return len(value)
            except TypeError:
                return 0

        blockers = []
        fields = (
            ("waiting", getattr(self, "waiting_queue", None), None),
            ("running", getattr(self, "running_batch", None), None),
            ("cur", getattr(self, "cur_batch", None), None),
            ("last", getattr(self, "last_batch", None), None),
            ("result", getattr(self, "result_queue", None), None),
            ("grammar", getattr(self, "grammar_manager", None), "grammar_queue"),
            ("decode_prealloc", getattr(self, "disagg_decode_prealloc_queue", None), "queue"),
            ("decode_pending", getattr(self, "disagg_decode_prealloc_queue", None), "pending_reqs"),
            ("decode_retracted", getattr(self, "disagg_decode_prealloc_queue", None), "retracted_queue"),
            ("decode_transfer", getattr(self, "disagg_decode_transfer_queue", None), "queue"),
            ("prefill_bootstrap", getattr(self, "disagg_prefill_bootstrap_queue", None), "queue"),
            ("prefill_inflight", getattr(self, "disagg_prefill_inflight_queue", None), None),
        )
        for name, value, attr in fields:
            count = size(value, attr)
            if count:
                blockers.append(f"{name}={count}")
        if getattr(self, "chunked_req", None) is not None:
            blockers.append("chunked=1")
        return "scheduler not fully idle: " + (", ".join(blockers) or "unreported blocker")

    def afd_component_local_quiescent_reason(self: "Scheduler"):
        """Return the first local drain blocker, or None at a forward-safe idle."""
        if not getattr(self, "_afd_component_fence_installed", False):
            return "fence is not installed"
        if getattr(self, "_afd_batchsize_attn", None) is not None:
            return "AFD batch info is still being processed"
        pending = getattr(self, "_afd_pending_batch_infos", None)
        if pending:
            return f"{len(pending)} AFD batch info(s) still pending"
        try:
            from sglang.srt.layers.afd import peek_async_communicator
            comm = peek_async_communicator()
            if comm is not None:
                pending_recvs = int(getattr(comm, "_pending_recv_count", 0))
                if pending_recvs:
                    return f"{pending_recvs} AFD receive(s) still in flight"
                send_queue = getattr(comm, "_send_queue", None)
                if send_queue is not None and int(
                    getattr(send_queue, "unfinished_tasks", 0)
                ):
                    return f"{send_queue.unfinished_tasks} AFD send(s) still queued"
                if any(
                    getattr(item, "is_alive", lambda: False)()
                    for item in (getattr(comm, "_pending_sends", None) or ())
                ):
                    return "AFD send thread(s) still in flight"
                if any(
                    not getattr(item, "done", lambda: False)()
                    for item in (getattr(comm, "_send_futures", None) or ())
                ):
                    return "AFD send future(s) still in flight"
        except Exception:
            # The communicator is absent in CPU tests and before AFD startup.
            pass
        idle_reason = self.afd_component_scheduler_idle_reason()
        if idle_reason is not None:
            return idle_reason
        return None

    def afd_component_sync_quiesce_request(self: "Scheduler") -> None:
        """Propagate rank-0 fence state before polling scheduler commands.

        Every active serving rank calls this at the same between-forwards point.
        A quiesce requested by rank 0's subsequent poll therefore becomes visible
        to followers on the next iteration, without crossing a max-world command.
        """
        if self._afd_component_control_group is None:
            return
        if getattr(self, "is_afd_component_standby_rank", False):
            return
        from sglang.srt.utils.common import broadcast_pyobj

        local = None
        if self.tp_rank == 0:
            source = (
                getattr(self, "_afd_component_pending_fence_payload", None)
                or getattr(self, "_afd_component_fence_payload", None)
            )
            if source is not None:
                # Phase 1 installs only the admission fence. Existing admitted
                # batches remain dispatchable until every active rank reaches a
                # between-forwards idle boundary; the cut is sealed there.
                local = dict(source)
        received = broadcast_pyobj(
            [local] if self.tp_rank == 0 else None,
            self.tp_rank,
            self._afd_component_active_control_group,
            src=0,
        )[0]
        generation = int(
            getattr(self, "_afd_component_active_control_generation", 0)
        )
        self._afd_component_active_control_generation = generation + 1
        if received is None:
            self._afd_component_fence_synced = False
            return
        if not isinstance(received, dict):
            raise RuntimeError(
                "AFD active-control group pollution: expected quiesce dict, "
                f"got {type(received).__name__} at generation={generation}"
            )
        if received.get("action") != "quiesce" or not received.get("operation_id"):
            raise RuntimeError(
                "AFD active-control group pollution: invalid schema "
                f"at generation={generation}: {received!r}"
            )
        operation = str(received["operation_id"])
        current = getattr(self, "_afd_component_fence_payload", None)
        if current is not None and str(current.get("operation_id")) != operation:
            raise RuntimeError("active ranks received mismatched quiesce operation")
        was_installed = bool(getattr(self, "_afd_component_fence_installed", False))
        self._afd_component_fence_payload = dict(received)
        self._afd_component_pending_fence_payload = None
        self._afd_component_fence_installed = True
        self._afd_component_fence_synced = True
        if not was_installed:
            self._afd_component_fence_monotonic = __import__("time").monotonic()
            self._afd_component_transfer_log_monotonic = 0.0
            logger.info(
                "[AFD-reshard] op=%s stage=quiesce comp=%s rank=%d "
                "effective fence installed after active sync",
                operation, received.get("component", "local"), self.tp_rank,
            )

    def afd_component_progress_decode_transfers(self: "Scheduler") -> None:
        """Poll and bound pre-fence PD decode transfers while quiescing."""
        if not getattr(self, "_afd_component_fence_installed", False):
            return
        from sglang.srt.layers.afd import afd_is_attn

        if not afd_is_attn():
            return
        queue = getattr(self, "disagg_decode_transfer_queue", None)
        prealloc = getattr(self, "disagg_decode_prealloc_queue", None)

        import time

        # Quiesce must progress backend completion and AbortReq-induced failure
        # even when the normal decode polling interval has not elapsed.
        if queue is not None and getattr(queue, "queue", None):
            transferred = queue.pop_transferred()
            if transferred:
                self.waiting_queue.extend(transferred)

        now = time.monotonic()
        last_log = float(
            getattr(self, "_afd_component_transfer_log_monotonic", 0.0)
        )
        if (
            queue is not None
            and getattr(queue, "queue", None)
            and (last_log == 0.0 or now - last_log >= 1.0)
        ):
            operation = (getattr(self, "_afd_component_fence_payload", None) or {}).get(
                "operation_id", "unknown"
            )
            for detail in queue.describe(now):
                logger.info(
                    "[AFD-reshard] op=%s stage=quiesce rank=%d "
                    "decode transfer retained: %s",
                    operation, self.tp_rank, detail,
                )
            self._afd_component_transfer_log_monotonic = now

        fence_at = getattr(self, "_afd_component_fence_monotonic", None)
        if fence_at is None:
            return
        grace = float(
            getattr(self.server_args, "afd_reshard_transfer_abort_grace", 2.0)
        )
        # Never let the transfer grace exceed the operation's own timeout.
        grace = min(grace, float(self.server_args.afd_reshard_timeout))
        from sglang.srt.utils.common import broadcast_pyobj

        generation = int(
            getattr(self, "_afd_component_active_control_generation", 0)
        )
        local = None
        if self.tp_rank == 0:
            local = {
                "action": "abort_decode_transfers",
                "operation_id": str(
                    (getattr(self, "_afd_component_fence_payload", None) or {}).get(
                        "operation_id", ""
                    )
                ),
                "generation": generation,
                "rids": (
                    [decode_req.req.rid for decode_req in (queue.queue if queue else ())]
                    if now - fence_at >= grace
                    else []
                ),
                "prealloc_rids": (
                    [decode_req.req.rid for decode_req in getattr(prealloc, "queue", ())]
                    + [req.rid for req in getattr(prealloc, "pending_reqs", ())]
                    + [req.rid for req in getattr(prealloc, "retracted_queue", ())]
                    if now - fence_at >= grace
                    else []
                ),
            }
        received = broadcast_pyobj(
            [local] if self.tp_rank == 0 else None,
            self.tp_rank,
            self._afd_component_active_control_group,
            src=0,
        )[0]
        self._afd_component_active_control_generation = generation + 1
        # Accept descriptors from an older controller during rolling tests; an
        # absent field means there was no bounded preallocation cleanup request.
        if isinstance(received, dict):
            received.setdefault("prealloc_rids", [])
        expected_operation = str(
            (getattr(self, "_afd_component_fence_payload", None) or {}).get(
                "operation_id", ""
            )
        )
        if (
            not isinstance(received, dict)
            or received.get("action") != "abort_decode_transfers"
            or str(received.get("operation_id")) != expected_operation
            or received.get("generation") != generation
            or not isinstance(received.get("rids"), list)
            or not isinstance(received.get("prealloc_rids"), list)
        ):
            raise RuntimeError(
                "AFD active-control group pollution: invalid decode-transfer "
                f"abort schema at generation={generation}: {received!r}"
            )
        reason = f"AFD reshard fence grace expired after {now - fence_at:.3f}s"
        prealloc_rids = received["prealloc_rids"]
        if prealloc_rids and prealloc is not None:
            prealloc.abort_and_reap(prealloc_rids, reason)
        rids = received["rids"]
        if rids and queue is not None:
            queue.abort_and_reap(rids, reason)

    def afd_component_seal_quiesce_cut(self: "Scheduler") -> int:
        """Seal dispatch only after admitted scheduler work reached idle.

        Attention TP rank 0 owns the dispatch sequence: followers do not execute
        ``next_dispatch`` and therefore cannot derive the cut from local state.
        Once every active rank has passed ready consensus, rank 0 seals its local
        ledger and broadcasts that authoritative cut on the isolated CPU control
        group. FFN receives the same cut through the pair/max-world descriptor.
        """
        from sglang.srt.utils.common import broadcast_pyobj

        payload = self._afd_component_fence_payload
        drain = self._afd_reshard_drain
        component = payload.get("component")
        operation = str(payload.get("operation_id", ""))
        if not operation:
            raise RuntimeError("AFD seal cut requires a quiesce operation_id")

        if component == "attn":
            generation = int(
                getattr(self, "_afd_component_active_control_generation", 0)
            )
            local = None
            if self.tp_rank == 0:
                watermark = drain.cut()
                local = {
                    "action": "seal_cut",
                    "operation_id": operation,
                    "generation": generation,
                    "watermark": watermark,
                }
            received = broadcast_pyobj(
                [local] if self.tp_rank == 0 else None,
                self.tp_rank,
                self._afd_component_active_control_group,
                src=0,
            )[0]
            self._afd_component_active_control_generation = generation + 1
            if not isinstance(received, dict):
                raise RuntimeError(
                    "AFD active-control group pollution: expected seal_cut dict, "
                    f"got {type(received).__name__} at generation={generation}"
                )
            expected_keys = {
                "action", "operation_id", "generation", "watermark"
            }
            if set(received) != expected_keys or received.get("action") != "seal_cut":
                raise RuntimeError(
                    "AFD active-control group pollution: invalid seal_cut schema "
                    f"at generation={generation}: {received!r}"
                )
            if (
                str(received.get("operation_id")) != operation
                or received.get("generation") != generation
            ):
                raise RuntimeError(
                    "AFD active-control group pollution: seal_cut operation or "
                    f"generation mismatch: expected=({operation!r}, {generation}), "
                    f"received={received!r}"
                )
            try:
                watermark = int(received["watermark"])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "AFD active-control group pollution: seal_cut watermark "
                    f"must be an integer, got {received['watermark']!r}"
                ) from exc
            if watermark < 0 or received["watermark"] != watermark:
                raise RuntimeError(
                    "AFD active-control group pollution: seal_cut watermark "
                    f"must be a non-negative integer, got {received['watermark']!r}"
                )
            # This also verifies an already-installed rank-0/follower cut matches
            # the authoritative value. Followers never compare dispatch_seq.
            sealed = drain.install_external_cut(watermark)
            payload["watermark"] = watermark
        elif component == "ffn":
            if "watermark" not in payload:
                raise RuntimeError("FFN quiesce request has no Attention cut")
            watermark = int(payload["watermark"])
            sealed = drain.install_external_cut(watermark)
        else:
            raise RuntimeError(f"invalid AFD quiesce component: {component!r}")

        if sealed != watermark:
            raise RuntimeError(
                "AFD dispatch cut mismatch at safe boundary: "
                f"local={sealed}, expected={watermark}"
            )
        self._afd_component_fence_sealed = True
        return watermark

    def afd_component_advance_quiesce(self: "Scheduler") -> bool:
        """Run ready consensus only after the fence sync generation completed."""
        if self._afd_component_control_group is None:
            return False
        if getattr(self, "_afd_component_quiesce_world_done", False):
            return False
        if not getattr(self, "_afd_component_fence_installed", False):
            return False
        if not getattr(self, "_afd_component_fence_synced", False):
            # First quiesce request was installed by rank 0's poll after this
            # iteration's pre-sync. All ranks still enter ordinary max-world None.
            return False
        import torch
        import torch.distributed as dist
        from sglang.srt.utils.common import broadcast_pyobj

        received = self._afd_component_fence_payload
        operation = str(received["operation_id"])
        reason = self.afd_component_local_quiescent_reason()
        ready = torch.tensor([1 if reason is None else 0], dtype=torch.int32)
        dist.all_reduce(
            ready,
            op=dist.ReduceOp.MIN,
            group=self._afd_component_active_control_group,
        )
        self._afd_component_quiesce_log_tick += 1
        if not int(ready.item()):
            if self._afd_component_quiesce_log_tick == 1 or self._afd_component_quiesce_log_tick % 100 == 0:
                def size(value):
                    if value is None:
                        return 0
                    reqs = getattr(value, "reqs", value)
                    try:
                        return len(reqs)
                    except TypeError:
                        return 0

                drain = getattr(self, "_afd_reshard_drain", None)
                logger.info(
                    "[AFD-reshard] op=%s stage=quiesce rank=%d ready=false "
                    "reason=%s fence=%s waiting=%d running=%d cur=%d last=%d "
                    "batchsize_attn=%s pending_infos=%d pending_sends=%s "
                    "pending_recvs=%s dispatch=%s cut=%s completion=(%s,%s)",
                    operation, self.tp_rank, reason or "ready",
                    bool(self._afd_component_fence_installed),
                    size(getattr(self, "waiting_queue", None)),
                    size(getattr(self, "running_batch", None)),
                    size(getattr(self, "cur_batch", None)),
                    size(getattr(self, "last_batch", None)),
                    getattr(self, "_afd_batchsize_attn", None),
                    size(getattr(self, "_afd_pending_batch_infos", None)),
                    getattr(drain, "pending_sends", None),
                    getattr(drain, "pending_recvs", None),
                    getattr(drain, "dispatch_seq", None),
                    getattr(drain, "cut_watermark", None),
                    getattr(drain, "attn_ack", None),
                    getattr(drain, "ffn_ack", None),
                )
            return True

        if not getattr(self, "_afd_component_fence_sealed", False):
            watermark = self.afd_component_seal_quiesce_cut()
            logger.info(
                "[AFD-reshard] op=%s stage=quiesce rank=%d "
                "dispatch cut sealed watermark=%d",
                operation, self.tp_rank, watermark,
            )

        logger.info(
            "[AFD-reshard] collective enter operation=%s generation=%s "
            "action=quiesce rank=%d",
            operation, received.get("epoch"), self.tp_rank,
        )
        world = broadcast_pyobj(
            [received] if self.tp_rank == 0 else None,
            self.tp_rank,
            self._afd_component_control_group,
            src=0,
        )[0]
        self.afd_component_apply_safe_point(world)
        self._afd_component_quiesce_world_done = True
        self._afd_component_quiesce_world_operation = operation
        logger.info(
            "[AFD-reshard] collective exit operation=%s generation=%s "
            "action=quiesce rank=%d",
            operation, received.get("epoch"), self.tp_rank,
        )
        return True

    def afd_component_safe_point(self: "Scheduler") -> None:
        """Synchronize lifecycle state between completed forwards."""
        if self._afd_component_control_group is None:
            return
        # Quiesce owns this generation while fenced.  Its active-only consensus
        # prevents rank 0 from entering the max-world rendezvous by itself.
        if self.afd_component_advance_quiesce():
            return
        if getattr(self, "_afd_component_skip_next_safe_point", False):
            self._afd_component_skip_next_safe_point = False
            return
        from sglang.srt.utils.common import broadcast_pyobj

        payload = self._afd_component_pending_world_command if self.tp_rank == 0 else None
        received = broadcast_pyobj(
            [payload] if self.tp_rank == 0 else None,
            self.tp_rank,
            self._afd_component_control_group,
            src=0,
        )
        if self.tp_rank == 0:
            self._afd_component_pending_world_command = None
        payload = received[0]
        if payload is None:
            return
        self.afd_component_apply_safe_point(payload)
        if self.is_afd_component_standby_rank:
            self._afd_component_stop_active_loop = True

    """Mixin providing AFD helpers for scheduler event loops.

    Usage: call these methods from within any event loop to add AFD support.
    The host scheduler must have `self.afd_send_to_ffn` and `self.afd_recv_from_attn`
    initialized (done in `init_ipc_channels`).
    """

    def afd_init_state(self: "Scheduler"):
        """Initialize per-loop AFD state. Call once at the start of event_loop."""
        self._afd_batchsize_attn = None
        self._afd_forward_mode = None
        self._afd_req_ids = None
        self._afd_current_metadata = None
        self._afd_ffn_authoritative_waiting = False
        self._afd_pending_batch_infos: deque = deque()
        self._afd_dispatch_id = 0
        self._afd_pf_rr_cursor = 0
        server_args = getattr(self, "server_args", None)
        if getattr(server_args, "enable_afd_component_reshard_participant", False):
            # Joining schedulers initialize the full event-loop state only after
            # the ACTIVATE boundary. Preserve the ledger that boundary just
            # reopened instead of replacing it with stale/default epoch state.
            self.afd_reshard_drain_state()
        else:
            self._afd_reshard_drain = None

        self._afd_poller = None
        from sglang.srt.layers.afd import afd_is_ffn

        recv_from_attn = getattr(self, "afd_recv_from_attn", None)
        if recv_from_attn is not None and afd_is_ffn():
            self._afd_poller = zmq.Poller()
            self._afd_poller.register(recv_from_attn, zmq.POLLIN)

    def afd_component_init_runtime(self: "Scheduler") -> None:
        """Start the feature-local control participant on scheduler rank 0."""
        self._afd_component_runtime = None
        self._afd_component_control_server = None
        self._afd_component_command_queue = None
        if not getattr(self.server_args, "enable_afd_component_reshard_participant", False):
            return
        if self.pp_rank != 0 or self.tp_rank != 0:
            return
        from sglang.srt.layers.afd import afd_is_attn
        from sglang.srt.reshard.afd_component_adapter import (
            AFDComponentSchedulerRuntime, SchedulerCommandQueue,
            ZMQSchedulerControlClient, ZMQSchedulerControlServer,
        )
        stage_offset = 0 if self.server_args.afd_reshard_stage_id == "prefill" else 10
        base = int(self.server_args.afd_reshard_control_base) + stage_offset
        attn_host = __import__("os").getenv("AFD_RESHARD_ATTN_CONTROL_HOST", "127.0.0.1")
        ffn_host = __import__("os").getenv("AFD_RESHARD_FFN_CONTROL_HOST", "127.0.0.1")
        local_endpoint = f"tcp://{attn_host if afd_is_attn() else ffn_host}:{base if afd_is_attn() else base + 1}"
        peer = ZMQSchedulerControlClient(f"tcp://{ffn_host}:{base + 1}") if afd_is_attn() else None
        self._afd_component_command_queue = SchedulerCommandQueue()
        pending = getattr(
            self, "_afd_component_external_control_pending", None
        )
        if pending is None:
            pending = self._afd_component_external_control_pending = __import__(
                "threading"
            ).Event()
        self._afd_component_runtime = AFDComponentSchedulerRuntime(
            self, peer, command_timeout=float(self.server_args.afd_reshard_timeout)
        )
        self._afd_component_control_server = ZMQSchedulerControlServer(
            local_endpoint, self._afd_component_command_queue,
            default_timeout=float(self.server_args.afd_reshard_timeout),
            readiness_provider=self._afd_component_runtime.serving_readiness,
            command_pending_callback=pending.set,
        )
        logger.info("AFD component control participant listening at %s", local_endpoint)

    def afd_component_poll_runtime(self: "Scheduler") -> int:
        """Execute at most one ordinary component command at a safe point."""
        runtime = getattr(self, "_afd_component_runtime", None)
        return runtime.poll() if runtime is not None else 0

    def afd_component_poll_through_wakeup(self: "Scheduler") -> tuple[int, bool]:
        """Consume FIFO readers ahead of, and including, one phase command."""
        runtime = getattr(self, "_afd_component_runtime", None)
        command_queue = getattr(self, "_afd_component_command_queue", None)
        if runtime is None or command_queue is None:
            return 0, False
        return command_queue.poll_through_wakeup(runtime.handle)

    def afd_component_post_receive_control_checkpoint(
        self: "Scheduler",
    ) -> bool:
        """Run an ordered active-rank control checkpoint.

        Every active rank must call this at the same event-loop positions because
        each invocation performs an active-control broadcast. Callers checkpoint
        both immediately before request receive (to avoid starving queued phase
        commands behind a blocking data-plane collective) and after request/AFD
        TP broadcasts (to cover commands arriving during receive). Requests
        already received remain in the caller and are processed after the one
        scheduler command; PREPARE therefore does not become an admission fence.

        The historical method name is retained to minimize reshard call-site
        churn; it is no longer restricted to post-receive use.
        """
        if self._afd_component_control_group is None:
            return False
        if getattr(self, "is_afd_component_standby_rank", False):
            return False
        from sglang.srt.utils.common import broadcast_pyobj

        pending = False
        if self.tp_rank == 0:
            event = getattr(
                self, "_afd_component_external_control_pending", None
            )
            command_queue = getattr(self, "_afd_component_command_queue", None)
            pending = bool(
                (event is not None and event.is_set())
                or (
                    command_queue is not None
                    and command_queue.has_wakeup_pending()
                )
            )
        received = broadcast_pyobj(
            [pending] if self.tp_rank == 0 else None,
            self.tp_rank,
            self._afd_component_active_control_group,
            src=0,
        )[0]
        self._afd_component_active_control_generation += 1
        if not received:
            return False

        tick = int(
            getattr(self, "_afd_component_control_checkpoint_wakeup_log_tick", 0)
        ) + 1
        self._afd_component_control_checkpoint_wakeup_log_tick = tick
        if tick == 1 or tick % 100 == 0:
            logger.info(
                "[AFD-reshard] stage=active_control_checkpoint_pending rank=%d "
                "wakeup=%d",
                self.tp_rank, tick,
            )
        # Followers proceed directly to safe_point(), where they receive the
        # world command emitted by rank 0. Rank 0 drains FIFO status readers in
        # front of, and including, exactly one wakeup command.
        consumed, wakeup_consumed = self.afd_component_poll_through_wakeup()
        self.afd_component_safe_point()
        event = getattr(self, "_afd_component_external_control_pending", None)
        command_queue = getattr(self, "_afd_component_command_queue", None)
        if (
            self.tp_rank == 0
            and wakeup_consumed
            and event is not None
            and (
                command_queue is None
                or not command_queue.has_wakeup_pending()
            )
        ):
            event.clear()
        return True

    def afd_recv_messages(self: "Scheduler"):
        """Poll for messages from the Attn scheduler.

        Returns ALL messages (including AFDReqInput) so they can be
        broadcast to all TP ranks. AFDReqInput state is extracted later
        in _afd_process_input_requests on every rank.
        """
        recv_socket = getattr(self, "afd_recv_from_attn", None)
        if recv_socket is None:
            return []
        extra_reqs = []
        while True:
            try:
                msg = recv_socket.recv_pyobj(zmq.NOBLOCK)
                extra_reqs.append(msg)
            except zmq.ZMQError:
                break
        return extra_reqs

    def afd_set_current_metadata(self: "Scheduler", afd_req) -> None:
        """Retain the PA metadata consumed for the current PF dispatch."""
        req_ids = list(afd_req.req_ids or [])
        extend_lens = list(afd_req.extend_lens or [])
        seq_lens = list(afd_req.seq_lens or [])
        if len(req_ids) != len(extend_lens) or len(req_ids) != len(seq_lens):
            raise ValueError(
                f"Invalid AFD metadata dispatch={afd_req.dispatch_id} rid=<batch>: "
                f"req_ids={len(req_ids)}, extend_lens={len(extend_lens)}, "
                f"seq_lens={len(seq_lens)}"
            )
        geometry_by_rid = {}
        for rid, seq_len, extend_len in zip(req_ids, seq_lens, extend_lens):
            if rid in geometry_by_rid:
                raise ValueError(
                    f"Invalid AFD metadata dispatch={afd_req.dispatch_id} "
                    f"rid={rid}: duplicate request metadata"
                )
            geometry_by_rid[rid] = (int(seq_len), int(extend_len))
        self._afd_current_metadata = {
            "dispatch_id": afd_req.dispatch_id,
            "req_ids": req_ids,
            "extend_lens": extend_lens,
            "seq_lens": seq_lens,
            "geometry_by_rid": geometry_by_rid,
        }

    def afd_restore_req_geometry(self: "Scheduler", req) -> None:
        """Restore PA prefix/extend geometry after PF cache initialization."""
        import torch

        metadata = self._afd_current_metadata
        dispatch_id = metadata["dispatch_id"]
        geometry = metadata["geometry_by_rid"].get(req.rid)
        if geometry is None:
            raise RuntimeError(
                f"AFD PF metadata missing dispatch={dispatch_id} rid={req.rid}"
            )
        seq_len, extend_len = geometry
        if not 0 <= extend_len <= seq_len:
            raise RuntimeError(
                f"Invalid AFD PF geometry dispatch={dispatch_id} rid={req.rid}: "
                f"seq_len={seq_len}, extend_len={extend_len}"
            )
        req.prefix_indices = torch.arange(
            seq_len - extend_len, dtype=torch.int64
        )
        req.set_extend_input_len(extend_len)

    def afd_validate_prefill_batch(self: "Scheduler", batch: "ScheduleBatch") -> None:
        """Fail fast when PF batch geometry diverges from the PA dispatch."""
        metadata = self._afd_current_metadata
        if metadata is None:
            raise RuntimeError("AFD PF batch has no consumed PA metadata")
        dispatch_id = metadata["dispatch_id"]
        attn_rids = metadata["req_ids"]
        attn_extend = metadata["extend_lens"]
        ffn_rids = [req.rid for req in batch.reqs]
        ffn_extend = list(getattr(batch, "extend_lens", []) or [])
        if ffn_rids != attn_rids or ffn_extend != attn_extend:
            mismatch_rid = next(
                (
                    f"attn={attn_rid},ffn={ffn_rid}"
                    for attn_rid, ffn_rid in zip(attn_rids, ffn_rids)
                    if attn_rid != ffn_rid
                ),
                next(iter(set(attn_rids) ^ set(ffn_rids)), "<batch>"),
            )
            raise RuntimeError(
                f"AFD PF metadata mismatch dispatch={dispatch_id} rid={mismatch_rid}: "
                f"attn_extend={attn_extend}, ffn_extend={ffn_extend}, "
                f"attn_rids={attn_rids}, ffn_rids={ffn_rids}"
            )
        attn_total = sum(attn_extend)
        ffn_total = sum(ffn_extend)
        if ffn_total != attn_total:
            raise RuntimeError(
                f"AFD PF metadata mismatch dispatch={dispatch_id} rid=<batch>: "
                f"attn_extend={attn_total}, ffn_extend={ffn_total}"
            )

    def afd_ffn_should_wait(self: "Scheduler") -> bool:
        """Return True if FFN side should wait for Attn sync before running a batch."""
        from sglang.srt.layers.afd import afd_is_ffn

        if afd_is_ffn() and self._afd_batchsize_attn is None:
            if self._afd_poller is not None:
                self._afd_poller.poll(timeout=10)
            return True
        return False

    @staticmethod
    def _afd_req_seq_len(req) -> int:
        """Return the request's complete logical sequence length."""
        return int(req.seqlen)

    def afd_send_batch_info(self: "Scheduler", batch: "ScheduleBatch"):
        """Attn side: send AFDReqInput to FFN so it knows the current batch."""
        from sglang.srt.layers.afd import afd_is_attn
        from sglang.srt.managers.io_struct import AFDReqInput

        if not afd_is_attn():
            return
        # Admission fencing gates only newly received user work. Batches that
        # entered the local waiting/running queues before the fence must still
        # receive a dispatch id and reach FFN so pair drain can complete. The
        # strict cut, once installed at the idle safe point, is enforced by
        # AFDPairDrainProtocol.next_dispatch().

        def make_req(start, end, *, dispatch_id, group_id=0, lane_id=0):
            reqs = batch.reqs[start:end]
            if batch.forward_mode.is_decode():
                lane_extend_lens = [int(req.extend_input_len) for req in reqs]
            else:
                extend_lens = getattr(batch, "extend_lens", None)
                if extend_lens is None:
                    raise RuntimeError(
                        f"AFD dispatch={dispatch_id} has no extend_lens"
                    )
                lane_extend_lens = list(extend_lens[start:end])
            if len(lane_extend_lens) != len(reqs):
                raise RuntimeError(
                    f"Invalid AFD dispatch={dispatch_id} metadata: "
                    f"req_ids={len(reqs)}, extend_lens={len(lane_extend_lens)}"
                )
            return AFDReqInput(
                dispatch_id=dispatch_id,
                pf_group_id=group_id,
                lane_id=lane_id,
                parent_batch_size=batch.batch_size(),
                batch_size=len(reqs),
                forward_mode=batch.forward_mode,
                req_ids=[r.rid for r in reqs],
                seq_lens=[SchedulerAFDMixin._afd_req_seq_len(r) for r in reqs],
                extend_lens=lane_extend_lens,
                max_input_len=max(
                    (r.extend_input_len for r in reqs), default=0
                ),
                repr_output_len=(
                    int(
                        sum(
                            max(
                                r.seqlen - len(r.origin_input_ids),
                                1,
                            )
                            for r in reqs
                        )
                        / max(len(reqs), 1)
                    )
                    if reqs
                    else 1
                ),
                output_ids_per_req=[list(r.output_ids) for r in reqs],
                input_ids_per_req=[list(r.origin_input_ids) for r in reqs],
                max_new_tokens_per_req=[
                    r.sampling_params.max_new_tokens for r in reqs
                ],
            )

        multi_pf = getattr(
            self.server_args, "afd_multi_pf_continuation", False
        )
        if multi_pf:
            sockets = getattr(self, "afd_send_to_ffn_groups", {})
            if not sockets:
                raise RuntimeError(
                    "Shared PA has no PF scheduler control sockets"
                )

            # rid→PF affinity: once a request is assigned to a PF group,
            # all subsequent chunks of that request stay on the same group.
            if not hasattr(self, "_afd_rid_affinity"):
                self._afd_rid_affinity: dict = {}
            if not hasattr(self, "_afd_pf_rr_counter"):
                self._afd_pf_rr_counter: int = 0

            num_groups = len(sockets)
            # Partition requests by affinity, assigning new ones round-robin.
            # Use BATCH-level round-robin: all requests in the same batch go to
            # the same PF group. This avoids m_stage>1 (AsyncMbDriver) which has
            # IPC event issues when interleaving channels.
            if not hasattr(self, "_afd_rid_affinity"):
                self._afd_rid_affinity: dict = {}
            if not hasattr(self, "_afd_pf_rr_counter"):
                self._afd_pf_rr_counter: int = 0

            # Determine the target group for new (unaffinitized) requests
            new_reqs = [r for r in batch.reqs if r.rid not in self._afd_rid_affinity]
            if new_reqs:
                target_group = self._afd_pf_rr_counter % num_groups
                self._afd_pf_rr_counter += 1
            else:
                target_group = 0

            group_reqs: dict = {g: [] for g in range(num_groups)}
            for idx, req in enumerate(batch.reqs):
                rid = req.rid
                if rid in self._afd_rid_affinity:
                    g = self._afd_rid_affinity[rid]
                else:
                    g = target_group
                    self._afd_rid_affinity[rid] = g
                group_reqs[g].append(idx)

            # Evict finished requests from affinity table periodically
            active_rids = {r.rid for r in batch.reqs}
            if len(self._afd_rid_affinity) > len(active_rids) * 2:
                self._afd_rid_affinity = {
                    k: v
                    for k, v in self._afd_rid_affinity.items()
                    if k in active_rids
                }

            # Reorder batch.reqs so each group's requests are contiguous
            new_order = []
            boundaries = [0]
            group_ids = []
            for g in range(num_groups):
                indices = group_reqs[g]
                if indices:
                    new_order.extend(indices)
                    boundaries.append(boundaries[-1] + len(indices))
                    group_ids.append(g)

            if not group_ids:
                return

            # Rearrange batch in-place for the model forward
            batch.reqs = [batch.reqs[i] for i in new_order]
            if hasattr(batch, "extend_lens") and batch.extend_lens is not None:
                old_lens = list(batch.extend_lens)
                batch.extend_lens = [old_lens[i] for i in new_order]

            num_lanes = len(group_ids)
            split_indices = boundaries[1:-1] if num_lanes > 1 else []

            if self._afd_reshard_drain is not None:
                self._afd_dispatch_id = self._afd_reshard_drain.next_dispatch()
            else:
                self._afd_dispatch_id += 1
            batch.afd_split_seq_index = split_indices or None
            batch.afd_pf_group_ids = group_ids

            for lane_id, group_id in enumerate(group_ids):
                start, end = boundaries[lane_id], boundaries[lane_id + 1]
                sockets[group_id].send_pyobj(
                    make_req(
                        start,
                        end,
                        dispatch_id=self._afd_dispatch_id,
                        group_id=group_id,
                        lane_id=lane_id,
                    )
                )
            return

        send_socket = getattr(self, "afd_send_to_ffn", None)
        if send_socket is not None:
            dispatch_id = (
                self._afd_reshard_drain.next_dispatch()
                if self._afd_reshard_drain is not None
                else 0
            )
            send_socket.send_pyobj(
                make_req(0, batch.batch_size(), dispatch_id=dispatch_id)
            )

    def afd_prepare_overlap(self: "Scheduler", batch: "ScheduleBatch"):
        """Compute microbatch split points for AFD on the batch."""
        from sglang.srt.batch_overlap.afd_overlap import _split_seq_indices_m_way
        from sglang.srt.layers.afd import get_afd_micro_batch
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.disaggregation.utils import DisaggregationMode

        # If afd_send_batch_info (multi-PF dispatch) already set split indices,
        # respect them — do NOT overwrite with None.
        if batch.afd_split_seq_index is not None:
            return

        m = get_afd_micro_batch()
        if batch.batch_size() < m:
            batch.afd_split_seq_index = None
            return

        # Check if we're in PD+AF prefill mode
        is_pd_prefill = getattr(self, "disaggregation_mode", None) == (
            DisaggregationMode.PREFILL
        )

        forward_mode = batch.forward_mode
        if forward_mode == ForwardMode.EXTEND:
            if is_pd_prefill:
                batch.afd_split_seq_index = None
                return
            extend_lens = batch.extend_lens
            split_indices = _split_seq_indices_m_way(len(extend_lens), m, extend_lens)
        elif forward_mode.is_decode() or forward_mode.is_target_verify():
            # Dynamic M: use M=1 for small decode batches
            effective_m = m
            if getattr(self.server_args, "afd_dynamic_micro_batch", False):
                threshold = getattr(self.server_args, "afd_dynamic_mb_threshold", 8)
                if batch.batch_size() < threshold:
                    effective_m = 1
            if effective_m <= 1:
                batch.afd_split_seq_index = None
                return
            split_indices = _split_seq_indices_m_way(batch.batch_size(), effective_m, None)
        else:
            return

        batch.afd_split_seq_index = split_indices

    def afd_gate_work_requests(self: "Scheduler", recv_reqs):
        """Gate newly received user work during component-reshard admission fence.

        Only tokenized generate/embedding requests (including their batch
        wrappers) are deferrable user work. Scheduler controls, AbortReq, and
        AFDReqInput synchronization messages remain immediately processable.
        On reopen, deferred objects are returned unchanged before requests from
        the current receive cycle, preserving admission order. FFN is unchanged.
        """
        from collections import deque

        from sglang.srt.layers.afd import afd_is_attn
        from sglang.srt.managers.io_struct import (
            BatchTokenizedEmbeddingReqInput,
            BatchTokenizedGenerateReqInput,
            TokenizedEmbeddingReqInput,
            TokenizedGenerateReqInput,
        )

        recv_reqs = list(recv_reqs or ())
        if not afd_is_attn():
            return recv_reqs

        deferred = getattr(self, "_afd_deferred_work_requests", None)
        if deferred is None:
            deferred = self._afd_deferred_work_requests = deque()

        work_types = (
            TokenizedGenerateReqInput,
            BatchTokenizedGenerateReqInput,
            TokenizedEmbeddingReqInput,
            BatchTokenizedEmbeddingReqInput,
        )
        if getattr(self, "_afd_component_fence_installed", False):
            processable = []
            for recv_req in recv_reqs:
                if isinstance(recv_req, work_types):
                    deferred.append(recv_req)
                else:
                    processable.append(recv_req)
            return processable

        if not deferred:
            return recv_reqs
        processable = list(deferred)
        deferred.clear()
        processable.extend(recv_reqs)
        return processable

    def afd_forward_work_requests(self: "Scheduler", recv_reqs):
        """Attn side: forward admitted work requests to FFN via ZMQ."""
        from sglang.srt.layers.afd import afd_is_attn
        from sglang.srt.managers.io_struct import (
            AFDReqInput,
            AbortReq,
            BatchTokenizedEmbeddingReqInput,
            BatchTokenizedGenerateReqInput,
            TokenizedEmbeddingReqInput,
            TokenizedGenerateReqInput,
        )

        if not afd_is_attn():
            return
        send_socket = getattr(self, "afd_send_to_ffn", None)
        if send_socket is None:
            return

        for recv_req in recv_reqs:
            if isinstance(recv_req, AFDReqInput):
                continue
            if isinstance(
                recv_req,
                (BatchTokenizedGenerateReqInput, BatchTokenizedEmbeddingReqInput),
            ):
                for sub_req in recv_req:
                    send_socket.send_pyobj(sub_req)
            elif isinstance(
                recv_req,
                (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput, AbortReq),
            ):
                send_socket.send_pyobj(recv_req)

    def afd_reshard_cut_dispatch(self: "Scheduler") -> int:
        """Fence new AFD dispatches and return the strict cut watermark."""
        if self._afd_reshard_drain is None:
            raise RuntimeError("AFD component reshard runtime is disabled")
        return self._afd_reshard_drain.cut()

    def afd_reshard_drain_state(self: "Scheduler"):
        """Return, lazily creating, the feature-local drain protocol."""
        drain = getattr(self, "_afd_reshard_drain", None)
        if drain is None:
            server_args = getattr(self, "server_args", None)
            if not getattr(
                server_args, "enable_afd_component_reshard_participant", False
            ):
                raise RuntimeError("AFD component reshard runtime is disabled")
            from sglang.srt.layers.afd_reshard_comm import AFDPairDrainProtocol

            drain = AFDPairDrainProtocol()
            self._afd_reshard_drain = drain
        return drain

    def afd_reset_state(self: "Scheduler"):
        """Reset per-iteration AFD state after processing a batch."""
        self._afd_batchsize_attn = None
        self._afd_forward_mode = None
        self._afd_req_ids = None
        self._afd_current_metadata = None
        self._afd_ffn_authoritative_waiting = False
