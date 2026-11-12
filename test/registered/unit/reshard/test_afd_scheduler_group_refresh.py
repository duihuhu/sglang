"""Regression coverage for scheduler-owned process-group rebinding."""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.managers.scheduler_afd_mixin import SchedulerAFDMixin
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="stage-a-cpu-only")


class TestAFDSchedulerGroupRefresh(CustomTestCase):
    @mock.patch("sglang.srt.layers.dp_attention.compute_dp_attention_world_info")
    @mock.patch("sglang.srt.layers.dp_attention.get_attention_cp_group")
    @mock.patch("sglang.srt.layers.dp_attention.get_attention_tp_group")
    @mock.patch("sglang.srt.distributed.parallel_state.get_tp_group")
    @mock.patch("sglang.srt.distributed.get_world_group")
    @mock.patch("sglang.srt.distributed.get_pp_group")
    def test_rebinds_cached_subsystem_groups(
        self,
        get_pp_group,
        get_world_group,
        get_tp_group,
        get_attention_tp_group,
        get_attention_cp_group,
        compute_world_info,
    ):
        """PD handshake polling must not retain the pre-reshard TP group."""
        old_group = object()
        target_cpu_group = object()
        target_tp_group = SimpleNamespace(
            cpu_group=target_cpu_group,
            world_size=4,
            rank=0,
            rank_in_group=0,
            first_rank=0,
            is_first_rank=True,
        )
        get_tp_group.return_value = target_tp_group
        get_attention_tp_group.return_value = target_tp_group
        get_attention_cp_group.return_value = SimpleNamespace(cpu_group=object())
        get_pp_group.return_value = object()
        get_world_group.return_value = object()
        compute_world_info.return_value = (0, 4, 0)

        queue = lambda: SimpleNamespace(gloo_group=old_group, tp_rank=0, tp_size=2)
        scheduler = SimpleNamespace(
            tp_rank=0,
            tp_size=2,
            dp_size=1,
            attn_cp_size=1,
            server_args=SimpleNamespace(tp_size=2, enable_dp_attention=False),
            tp_worker=SimpleNamespace(),
            disagg_decode_prealloc_queue=queue(),
            disagg_decode_transfer_queue=queue(),
            disagg_prefill_bootstrap_queue=queue(),
            grammar_manager=SimpleNamespace(grammar_sync_group=old_group),
            _afd_component_active_control_groups={4: object()},
        )

        # The same callback is used for target activation and source rollback.
        SchedulerAFDMixin.afd_component_refresh_scheduler_groups(scheduler, 4)

        self.assertIs(scheduler.tp_cpu_group, target_cpu_group)
        self.assertIs(scheduler.disagg_decode_prealloc_queue.gloo_group, target_cpu_group)
        self.assertIs(scheduler.disagg_decode_transfer_queue.gloo_group, target_cpu_group)
        self.assertIs(scheduler.disagg_prefill_bootstrap_queue.gloo_group, target_cpu_group)
        self.assertEqual(scheduler.disagg_decode_prealloc_queue.tp_size, 4)
        self.assertEqual(scheduler.disagg_prefill_bootstrap_queue.tp_size, 4)
        self.assertEqual(scheduler.disagg_prefill_bootstrap_queue.collective_rank, 0)
        self.assertEqual(
            scheduler.disagg_prefill_bootstrap_queue.collective_src_rank, 0
        )
        self.assertIs(scheduler.grammar_manager.grammar_sync_group, target_cpu_group)
        self.assertEqual(scheduler.grammar_manager.grammar_sync_size, 4)


if __name__ == "__main__":
    unittest.main(verbosity=3)
