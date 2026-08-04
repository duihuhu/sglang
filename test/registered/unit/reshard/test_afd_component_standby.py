import ast
import inspect
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.reshard.afd_component_standby import (
    AFDComponentRankLifecycle, AFDComponentRankState, AFDComponentStandbyLoop,
    AFDComponentWorldCommand,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="stage-a-cpu-only")


def command(operation, epoch, action, target_tp=None, component="attn"):
    return {"operation": operation, "epoch": epoch, "component": component,
            "action": action, "target_tp": target_tp}


class TestAFDComponentStandby(CustomTestCase):
    def test_active_prefix_grow_then_shrink_demotes_ranks(self):
        ranks = [AFDComponentRankLifecycle(rank, 2, 4, "attn") for rank in range(4)]
        for lifecycle in ranks:
            lifecycle.apply_safe_point(command("grow", 1, "prepare", 4))
            lifecycle.apply_safe_point(command("grow", 1, "activate", 4))
        self.assertTrue(all(item.state == AFDComponentRankState.ACTIVE for item in ranks))
        for lifecycle in ranks:
            lifecycle.apply_safe_point(command("shrink", 2, "demote", 1))
        self.assertEqual([item.state for item in ranks], [
            AFDComponentRankState.ACTIVE, AFDComponentRankState.STANDBY,
            AFDComponentRankState.STANDBY, AFDComponentRankState.STANDBY,
        ])

    def test_standby_loop_does_not_accept_native_schema(self):
        lifecycle = AFDComponentRankLifecycle(3, 2, 4, "ffn")
        loop = AFDComponentStandbyLoop(lifecycle, lambda: {
            "action": "activate_inplace_reshard", "old_tp_size": 2, "new_tp_size": 4
        })
        with self.assertRaisesRegex(ValueError, "misses"):
            loop.run()

    def test_command_identity_and_abort(self):
        lifecycle = AFDComponentRankLifecycle(2, 2, 4, "ffn")
        lifecycle.apply_safe_point(command("op", 1, "prepare", 4, "ffn"))
        self.assertEqual(lifecycle.state, AFDComponentRankState.PREPARED)
        with self.assertRaisesRegex(ValueError, "does not match"):
            lifecycle.apply_safe_point(command("other", 1, "activate", 4, "ffn"))
        lifecycle.apply_safe_point(command("op", 1, "abort", component="ffn"))
        self.assertEqual(lifecycle.state, AFDComponentRankState.STANDBY)

    def test_standby_does_not_own_afd_scheduler_endpoint(self):
        from sglang.srt.managers.scheduler import (
            _should_init_afd_scheduler_channels,
        )

        args = SimpleNamespace(enable_afd_component_reshard_participant=True)
        decisions = [
            _should_init_afd_scheduler_channels(
                args,
                pp_rank=0,
                global_tp_rank=rank,
                attn_tp_rank=0 if rank >= 2 else rank,
                inplace_standby_ipc=rank >= 2,
            )
            for rank in range(4)
        ]
        self.assertEqual(decisions, [True, False, False, False])
        self.assertFalse(
            _should_init_afd_scheduler_channels(
                args,
                pp_rank=0,
                global_tp_rank=2,
                attn_tp_rank=0,
                inplace_standby_ipc=False,
            )
        )

        native_args = SimpleNamespace(
            enable_afd_component_reshard_participant=False
        )
        self.assertTrue(
            _should_init_afd_scheduler_channels(
                native_args,
                pp_rank=0,
                global_tp_rank=7,
                attn_tp_rank=0,
                inplace_standby_ipc=False,
            )
        )

    def test_shutdown_and_validation(self):
        lifecycle = AFDComponentRankLifecycle(3, 2, 4, "attn")
        self.assertEqual(
            lifecycle.apply_safe_point(command("stop", 1, "shutdown")),
            AFDComponentRankState.SHUTDOWN,
        )
        with self.assertRaisesRegex(ValueError, "positive target_tp"):
            AFDComponentWorldCommand.parse(command("bad", 1, "demote", 0))


class TestAFDComponentScheduleStream(CustomTestCase):
    @staticmethod
    def _make_scheduler(*, standby=False):
        from sglang.srt.managers.scheduler import Scheduler

        scheduler = object.__new__(Scheduler)
        scheduler.server_args = SimpleNamespace(
            enable_afd_component_reshard_participant=True,
            enable_dp_attention=False,
        )
        scheduler.is_afd_component_standby_rank = standby
        scheduler.enable_lora_overlap_loading = False
        scheduler.device = "cpu"
        scheduler.device_module = MagicMock()
        scheduler.device_module.Stream.return_value = MagicMock()
        scheduler.device_module.StreamContext.return_value = MagicMock()
        return scheduler

    def test_initial_active_component_creates_schedule_stream(self):
        scheduler = self._make_scheduler()

        with patch(
            "sglang.srt.managers.scheduler.dispatch_event_loop"
        ) as dispatch:
            scheduler.run_event_loop()

        scheduler.device_module.Stream.assert_called_once_with(priority=0)
        scheduler.device_module.StreamContext.assert_called_once_with(
            scheduler.schedule_stream
        )
        dispatch.assert_called_once_with(scheduler)
        # CPU streams do not expose synchronize, so the scheduler supplies it.
        self.assertIsNone(scheduler.schedule_stream.synchronize())

    def test_standby_activation_initializes_schedule_stream_once(self):
        scheduler = self._make_scheduler(standby=True)
        scheduler.tp_rank = 2
        scheduler._afd_component_transition_boundary = {
            "operation_id": "grow",
            "epoch": 1,
            "target_tp": 4,
            "resume_phase": "request_receive",
            "resume_pending": True,
        }
        scheduler.tp_worker = MagicMock()
        scheduler.tp_worker.get_worker_info.return_value = (
            1, 2, 3, 4, 5, 6, 7, "cpu", MagicMock(), None, None, None
        )
        for method_name in (
            "init_cache_with_memory_pool",
            "init_running_status",
            "init_chunked_prefill",
            "init_diffusion_llm",
            "init_schedule_policy",
            "init_watch_dog_memory_saver_input_blocker",
            "init_profiler",
            "init_disaggregation",
            "init_overlap",
            "maybe_init_ngram_embedding",
            "init_deterministic_inference_config",
            "init_request_dispatcher",
            "afd_init_state",
            "afd_component_init_runtime",
        ):
            setattr(scheduler, method_name, MagicMock())

        grammar_manager = object()
        with patch(
            "sglang.srt.constrained.grammar_manager.GrammarManager",
            return_value=grammar_manager,
        ) as grammar_manager_cls:
            scheduler.afd_component_enter_active_loop()
        scheduler._init_schedule_stream()

        grammar_manager_cls.assert_called_once_with(scheduler)
        self.assertIs(scheduler.grammar_manager, grammar_manager)
        scheduler.device_module.Stream.assert_called_once_with(priority=0)
        self.assertTrue(scheduler._afd_component_active_scheduler_initialized)
        self.assertIsNone(scheduler.schedule_stream.synchronize())

        scheduler._afd_component_control_group = object()
        with patch(
            "sglang.srt.utils.common.broadcast_pyobj", return_value=[None]
        ) as broadcast:
            self.assertTrue(scheduler.afd_component_resume_at_transition_boundary())
            broadcast.assert_not_called()
            self.assertFalse(
                scheduler._afd_component_transition_boundary["resume_pending"]
            )

            scheduler.afd_component_safe_point()
            broadcast.assert_called_once()

    def test_joining_initializes_lora_overlap_loader_when_enabled(self):
        scheduler = self._make_scheduler(standby=True)
        scheduler.enable_lora_overlap_loading = True
        scheduler.tp_worker = MagicMock()
        lora_manager = scheduler.tp_worker.model_runner.lora_manager
        loader = object()

        with patch(
            "sglang.srt.lora.lora_overlap_loader.LoRAOverlapLoader",
            return_value=loader,
        ) as loader_cls, patch(
            "sglang.srt.constrained.grammar_manager.GrammarManager"
        ) as grammar_manager_cls:
            scheduler.afd_component_finish_joining_scheduler_init()

        loader_cls.assert_called_once_with(lora_manager)
        self.assertIs(scheduler.lora_overlap_loader, loader)
        grammar_manager_cls.assert_called_once_with(scheduler)

    def test_all_afd_event_loops_checkpoint_around_request_receive(self):
        from sglang.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin
        from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
        from sglang.srt.managers.scheduler import Scheduler

        loops = {
            "ffn": Scheduler.event_loop_afd,
            "prefill": SchedulerDisaggregationPrefillMixin.event_loop_afd_disagg_prefill,
            "decode": SchedulerDisaggregationDecodeMixin.event_loop_afd_disagg_decode,
        }

        def call_name(statement):
            value = statement.value if isinstance(statement, ast.Expr) else None
            if not isinstance(value, ast.Call):
                return None
            function = value.func
            return function.attr if isinstance(function, ast.Attribute) else None

        def guard_name(statement):
            if not isinstance(statement, ast.If) or not isinstance(
                statement.test, ast.Call
            ):
                return None
            function = statement.test.func
            return function.attr if isinstance(function, ast.Attribute) else None

        for name, loop in loops.items():
            with self.subTest(loop=name):
                function = ast.parse(textwrap.dedent(inspect.getsource(loop))).body[0]
                while_node = next(
                    node for node in function.body if isinstance(node, ast.While)
                )
                body = while_node.body
                entry_index = next(
                    index
                    for index, statement in enumerate(body)
                    if call_name(statement) == "afd_component_begin_active_iteration"
                )
                self.assertEqual(
                    guard_name(body[entry_index + 1]),
                    "afd_component_should_leave_active_loop",
                )
                self.assertEqual(
                    guard_name(body[entry_index + 2]),
                    "afd_component_should_restart_active_iteration",
                )

                recv_index = next(
                    index
                    for index, statement in enumerate(body)
                    if isinstance(statement, ast.Assign)
                    and isinstance(statement.value, ast.Call)
                    and isinstance(statement.value.func, ast.Attribute)
                    and statement.value.func.attr == "recv_requests"
                )
                checkpoint_indices = [
                    index
                    for index, statement in enumerate(body)
                    if call_name(statement)
                    == "afd_component_post_receive_control_checkpoint"
                ]
                self.assertEqual(len(checkpoint_indices), 2)
                pre_index, post_index = checkpoint_indices
                self.assertLess(pre_index, recv_index)
                self.assertGreater(post_index, recv_index)

                for checkpoint_index in checkpoint_indices:
                    restart_guard = body[checkpoint_index + 1]
                    self.assertEqual(
                        guard_name(restart_guard),
                        "afd_component_should_restart_active_iteration",
                    )
                    self.assertIsInstance(restart_guard.body[0], ast.Continue)
                self.assertEqual(pre_index + 2, recv_index)

    def test_joining_active_iteration_skips_control_and_reaches_recv_phase(self):
        from sglang.srt.managers.scheduler_afd_mixin import SchedulerAFDMixin

        scheduler = MagicMock()
        scheduler.afd_component_resume_at_transition_boundary.return_value = True

        SchedulerAFDMixin.afd_component_begin_active_iteration(scheduler)
        recv_reqs = scheduler.recv_requests()

        scheduler.afd_component_resume_at_transition_boundary.assert_called_once_with()
        scheduler.afd_component_sync_quiesce_request.assert_not_called()
        scheduler.afd_component_poll_runtime.assert_not_called()
        scheduler.afd_component_safe_point.assert_not_called()
        scheduler.recv_requests.assert_called_once_with()
        self.assertIs(recv_reqs, scheduler.recv_requests.return_value)

    def test_activate_restart_skips_current_iteration_data_plane_once(self):
        from sglang.srt.managers.scheduler_afd_mixin import SchedulerAFDMixin

        scheduler = MagicMock()
        scheduler.__dict__["_afd_component_restart_active_iteration"] = True

        self.assertTrue(
            SchedulerAFDMixin.afd_component_should_restart_active_iteration(
                scheduler
            )
        )
        scheduler.recv_requests.assert_not_called()
        self.assertFalse(
            SchedulerAFDMixin.afd_component_should_restart_active_iteration(
                scheduler
            )
        )

    def test_shrink_boundary_excludes_retired_rank_from_readiness(self):
        from sglang.srt.managers.scheduler_afd_mixin import SchedulerAFDMixin

        scheduler = MagicMock()
        scheduler.tp_rank = 1
        scheduler.server_args.afd_component_max_tp = 4
        scheduler._afd_component_control_group = object()
        scheduler._afd_component_lifecycle.active_tp = 1
        drain = scheduler.afd_reshard_drain_state.return_value
        cmd = SimpleNamespace(operation="shrink", epoch=2, target_tp=1)

        def gather(output, descriptor, group):
            output[:] = [dict(descriptor) for _ in output]

        with patch("torch.distributed.all_gather_object", side_effect=gather):
            SchedulerAFDMixin.afd_component_complete_transition_boundary(
                scheduler, cmd, previous_active_tp=4
            )

        drain.reopen_at_transition_boundary.assert_called_once_with()
        scheduler.afd_component_clear_quiesce.assert_called_once_with()
        self.assertTrue(scheduler._afd_component_serving_ready)
        self.assertIsNone(scheduler._afd_component_serving_ready_operation)
        self.assertIsNone(scheduler._afd_component_transition_boundary)

    def test_shrink_retired_rank_leaves_before_serving_readiness(self):
        from sglang.srt.managers.scheduler_afd_mixin import SchedulerAFDMixin

        scheduler = MagicMock()
        scheduler.tp_rank = 1
        scheduler.__dict__.update(
            is_afd_component_standby_rank=True,
            _afd_component_active_scheduler_initialized=True,
            _afd_component_serving_ready=False,
            _afd_component_serving_ready_operation="shrink",
            # A retired rank has completed the max-world boundary but owns no
            # target-active descriptor/readiness generation.
            _afd_component_transition_boundary=None,
        )

        SchedulerAFDMixin.afd_component_begin_active_iteration(scheduler)

        scheduler.afd_component_resume_at_transition_boundary.assert_not_called()
        scheduler.afd_component_complete_serving_readiness.assert_not_called()
        self.assertTrue(scheduler._afd_component_stop_active_loop)
        self.assertTrue(scheduler._afd_component_serving_ready)
        self.assertIsNone(scheduler._afd_component_serving_ready_operation)
        self.assertIsNone(scheduler._afd_component_transition_boundary)
        self.assertTrue(
            SchedulerAFDMixin.afd_component_should_leave_active_loop(scheduler)
        )
        self.assertFalse(scheduler._afd_component_active_scheduler_initialized)

    def test_shrink_surviving_rank_completes_serving_readiness(self):
        from sglang.srt.managers.scheduler_afd_mixin import SchedulerAFDMixin

        scheduler = MagicMock()
        scheduler.tp_rank = 0
        scheduler.__dict__.update(
            is_afd_component_standby_rank=False,
            _afd_component_serving_ready=False,
            _afd_component_serving_ready_operation="shrink",
            _afd_component_transition_boundary={
                "operation_id": "shrink",
                "epoch": 2,
                "target_tp": 1,
                "resume_phase": "post_activate_ready_then_request_receive",
                "resume_pending": False,
                "data_plane_ready": False,
            },
        )
        scheduler.afd_component_resume_at_transition_boundary.return_value = False
        scheduler.afd_component_complete_serving_readiness.return_value = True

        SchedulerAFDMixin.afd_component_begin_active_iteration(scheduler)

        scheduler.afd_component_complete_serving_readiness.assert_called_once_with()
        self.assertFalse(
            scheduler.__dict__.get("_afd_component_stop_active_loop", False)
        )

    def test_post_activate_iteration_barriers_before_first_receive(self):
        from sglang.srt.managers.scheduler_afd_mixin import SchedulerAFDMixin

        scheduler = MagicMock()
        scheduler.__dict__.update(
            _afd_component_serving_ready=False,
            _afd_component_serving_ready_operation="grow",
            _afd_component_transition_boundary={
                "operation_id": "grow",
                "epoch": 1,
                "target_tp": 4,
                "resume_phase": "post_activate_ready_then_request_receive",
                "resume_pending": False,
                "data_plane_ready": False,
            },
        )
        scheduler.afd_component_resume_at_transition_boundary.return_value = False
        scheduler.afd_component_complete_serving_readiness.return_value = True

        SchedulerAFDMixin.afd_component_begin_active_iteration(scheduler)
        scheduler.recv_requests()

        self.assertEqual(
            [call[0] for call in scheduler.method_calls],
            [
                "afd_component_resume_at_transition_boundary",
                "afd_component_complete_serving_readiness",
                "recv_requests",
            ],
        )
        scheduler.afd_component_sync_quiesce_request.assert_not_called()
        scheduler.afd_component_poll_runtime.assert_not_called()
        scheduler.afd_component_safe_point.assert_not_called()

    def test_joining_boundary_consumes_then_barriers_before_receive(self):
        from sglang.srt.managers.scheduler_afd_mixin import SchedulerAFDMixin

        scheduler = MagicMock()
        boundary = {
            "operation_id": "grow",
            "epoch": 1,
            "target_tp": 4,
            "resume_phase": "post_activate_ready_then_request_receive",
            "resume_pending": True,
            "data_plane_ready": False,
        }
        scheduler.__dict__.update(
            _afd_component_serving_ready=False,
            _afd_component_serving_ready_operation="grow",
            _afd_component_transition_boundary=boundary,
        )

        def consume():
            boundary["resume_pending"] = False
            return True

        scheduler.afd_component_resume_at_transition_boundary.side_effect = consume
        scheduler.afd_component_complete_serving_readiness.return_value = True

        SchedulerAFDMixin.afd_component_begin_active_iteration(scheduler)
        scheduler.recv_requests()

        self.assertEqual(
            [call[0] for call in scheduler.method_calls],
            [
                "afd_component_resume_at_transition_boundary",
                "afd_component_complete_serving_readiness",
                "recv_requests",
            ],
        )

    @staticmethod
    def _make_pending_readiness_scheduler(*, joining=False, rank=0, target_tp=2):
        from sglang.srt.managers.scheduler import Scheduler

        scheduler = object.__new__(Scheduler)
        scheduler.tp_rank = rank
        scheduler.server_args = SimpleNamespace(
            enable_afd_component_reshard_participant=True,
            afd_perspective="attn",
            afd_component_max_tp=target_tp,
        )
        scheduler._afd_component_active_control_group = object()
        scheduler._afd_component_active_control_generation = 0
        scheduler._afd_component_serving_ready = False
        scheduler._afd_component_serving_ready_operation = "grow"
        scheduler._afd_component_transition_boundary = {
            "operation_id": "grow",
            "epoch": 1,
            "target_tp": target_tp,
            "resume_phase": "post_activate_ready_then_request_receive",
            "resume_pending": joining,
            "data_plane_ready": False,
        }
        scheduler.is_afd_component_standby_rank = False
        scheduler.afd_component_log_pd_queues = MagicMock()
        scheduler.afd_component_sync_quiesce_request = MagicMock()
        scheduler.afd_component_poll_runtime = MagicMock()
        scheduler.afd_component_safe_point = MagicMock()
        return scheduler

    def test_surviving_and_joining_rebuild_before_collective_and_receive(self):
        from sglang.srt.managers.scheduler_afd_mixin import SchedulerAFDMixin

        for joining in (False, True):
            with self.subTest(joining=joining):
                events = []
                scheduler = self._make_pending_readiness_scheduler(joining=joining)
                original_resume = scheduler.afd_component_resume_at_transition_boundary

                def resume():
                    result = original_resume()
                    if result:
                        events.append("resume")
                    return result

                scheduler.afd_component_resume_at_transition_boundary = resume

                def gather(output, descriptor, group):
                    events.append("collective")
                    output[:] = [dict(descriptor) for _ in output]

                with patch(
                    "sglang.srt.layers.afd.get_async_communicator",
                    side_effect=lambda: events.append("rebuild") or object(),
                ) as rebuild, patch(
                    "torch.distributed.all_gather_object", side_effect=gather
                ):
                    SchedulerAFDMixin.afd_component_begin_active_iteration(scheduler)
                    events.append("recv")

                expected = ["rebuild", "collective", "recv"]
                if joining:
                    expected.insert(0, "resume")
                self.assertEqual(events, expected)
                rebuild.assert_called_once_with()
                self.assertTrue(scheduler._afd_component_serving_ready)
                self.assertIsNone(scheduler._afd_component_transition_boundary)

    def test_repeated_readiness_completion_does_not_rebuild(self):
        from sglang.srt.managers.scheduler_afd_mixin import SchedulerAFDMixin

        scheduler = self._make_pending_readiness_scheduler()

        def gather(output, descriptor, group):
            output[:] = [dict(descriptor) for _ in output]

        with patch(
            "sglang.srt.layers.afd.get_async_communicator", return_value=object()
        ) as rebuild, patch(
            "torch.distributed.all_gather_object", side_effect=gather
        ):
            self.assertTrue(
                SchedulerAFDMixin.afd_component_complete_serving_readiness(scheduler)
            )
            self.assertFalse(
                SchedulerAFDMixin.afd_component_complete_serving_readiness(scheduler)
            )

        rebuild.assert_called_once_with()

    def test_tp1_and_nonparticipant_readiness_modes(self):
        from sglang.srt.managers.scheduler_afd_mixin import SchedulerAFDMixin

        def gather(output, descriptor, group):
            output[:] = [dict(descriptor) for _ in output]

        for participant, expected_rebuilds in ((True, 1), (False, 0)):
            with self.subTest(participant=participant):
                scheduler = self._make_pending_readiness_scheduler(target_tp=1)
                scheduler.server_args.enable_afd_component_reshard_participant = (
                    participant
                )
                with patch(
                    "sglang.srt.layers.afd.get_async_communicator",
                    return_value=object(),
                ) as rebuild, patch(
                    "torch.distributed.all_gather_object", side_effect=gather
                ):
                    self.assertTrue(
                        SchedulerAFDMixin.afd_component_complete_serving_readiness(
                            scheduler
                        )
                    )
                self.assertEqual(rebuild.call_count, expected_rebuilds)
                self.assertTrue(scheduler._afd_component_serving_ready)

    def test_joining_defers_event_loop_eager_init_to_boundary(self):
        from sglang.srt.managers.scheduler_afd_mixin import SchedulerAFDMixin

        joining = self._make_pending_readiness_scheduler(joining=True)
        survivor = self._make_pending_readiness_scheduler(joining=False)

        self.assertFalse(
            SchedulerAFDMixin.afd_component_should_eager_init_data_plane(joining)
        )
        self.assertTrue(
            SchedulerAFDMixin.afd_component_should_eager_init_data_plane(survivor)
        )

    def test_communicator_rebuild_failure_preserves_pending_boundary(self):
        from sglang.srt.managers.scheduler_afd_mixin import SchedulerAFDMixin

        scheduler = self._make_pending_readiness_scheduler()
        boundary = scheduler._afd_component_transition_boundary
        with patch(
            "sglang.srt.layers.afd.get_async_communicator",
            side_effect=ConnectionError("no listener"),
        ), patch("torch.distributed.all_gather_object") as gather:
            with self.assertRaisesRegex(
                RuntimeError, "data-plane communicator rebuild failed.*no listener"
            ):
                SchedulerAFDMixin.afd_component_begin_active_iteration(scheduler)

        gather.assert_not_called()
        self.assertFalse(scheduler._afd_component_serving_ready)
        self.assertIs(scheduler._afd_component_transition_boundary, boundary)
        self.assertFalse(boundary["data_plane_ready"])
        self.assertEqual(scheduler._afd_component_serving_ready_operation, "grow")

    def test_retired_rank_never_rebuilds_data_plane(self):
        from sglang.srt.managers.scheduler_afd_mixin import SchedulerAFDMixin

        scheduler = MagicMock()
        scheduler.tp_rank = 2
        scheduler.__dict__.update(
            is_afd_component_standby_rank=True,
            _afd_component_serving_ready=False,
            _afd_component_serving_ready_operation="shrink",
            _afd_component_transition_boundary=None,
        )
        with patch(
            "sglang.srt.layers.afd.get_async_communicator"
        ) as rebuild:
            SchedulerAFDMixin.afd_component_begin_active_iteration(scheduler)
        rebuild.assert_not_called()

    def test_normal_active_iteration_runs_control_preamble_in_order(self):
        from sglang.srt.managers.scheduler_afd_mixin import SchedulerAFDMixin

        scheduler = MagicMock()
        scheduler.afd_component_resume_at_transition_boundary.return_value = False

        SchedulerAFDMixin.afd_component_begin_active_iteration(scheduler)

        self.assertEqual(
            [call[0] for call in scheduler.method_calls],
            [
                "afd_component_resume_at_transition_boundary",
                "afd_component_sync_quiesce_request",
                "afd_component_poll_runtime",
                "afd_component_safe_point",
            ],
        )

    def test_active_follower_does_not_skip_safe_point(self):
        from sglang.srt.managers.scheduler import Scheduler

        scheduler = object.__new__(Scheduler)
        scheduler.tp_rank = 1
        scheduler._afd_component_control_group = object()

        with patch(
            "sglang.srt.utils.common.broadcast_pyobj", return_value=[None]
        ) as broadcast:
            scheduler.afd_component_safe_point()

        broadcast.assert_called_once()
        self.assertFalse(
            getattr(scheduler, "_afd_component_skip_next_safe_point", False)
        )

    def test_rank_zero_fanout_skip_remains_one_shot(self):
        from sglang.srt.managers.scheduler import Scheduler

        scheduler = object.__new__(Scheduler)
        scheduler.tp_rank = 0
        scheduler._afd_component_control_group = object()
        scheduler._afd_component_pending_world_command = None
        scheduler._afd_component_skip_next_safe_point = True

        with patch(
            "sglang.srt.utils.common.broadcast_pyobj", return_value=[None]
        ) as broadcast:
            scheduler.afd_component_safe_point()
            broadcast.assert_not_called()
            self.assertFalse(scheduler._afd_component_skip_next_safe_point)

            scheduler.afd_component_safe_point()
            broadcast.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=3)
