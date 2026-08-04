import threading
import time
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.afd_type import AFDPerspective
from sglang.srt.reshard.afd_component_weight_staging import (
    HybridMaxWorldTransport,
    InMemoryMaxWorldTransport,
    ModelRunnerAFDComponentStager,
    TorchDistributedNCCLMaxWorldTransport,
    _allocate_tp1_target,
    _assemble_tp1_target,
    _copy_source_shard_to_tp1_cpu,
    _target_shard,
    create_afd_component_staging_groups,
)
from sglang.srt.reshard.afd_weight_reshard import SplitRule
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="stage-a-cpu-only")


class QKVParallelLinear(torch.nn.Module):
    def __init__(self, tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(tensor.clone())
        self.head_size = 1
        self.total_num_heads = 8
        self.total_num_kv_heads = 4


class MergedColumnParallelLinear(torch.nn.Module):
    def __init__(self, tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(tensor.clone())
        self.output_sizes = (8, 8)


class RowParallelLinear(torch.nn.Module):
    def __init__(self, tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(tensor.clone())


class _Model(torch.nn.Module):
    def __init__(self, qkv, gate_up, down):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        layer = self.model.layers[0]
        layer.self_attn = torch.nn.Module()
        layer.self_attn.qkv_proj = QKVParallelLinear(qkv)
        layer.mlp = torch.nn.Module()
        layer.mlp.gate_up_proj = MergedColumnParallelLinear(gate_up)
        layer.mlp.down_proj = RowParallelLinear(down)


class _Runner:
    def __init__(self, model, tp_size=2):
        self.model = model
        self.tp_size = tp_size
        self.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen3")
        )
        self.rebuilt = []
        self.refreshed = []
        self.topology_refreshed = []
        self.demoted = []
        self.ensure_calls = []
        self._afd_component_collective_commands_enabled = True

    def afd_component_rebuild_groups(self, target_tp):
        self.rebuilt.append(target_tp)
        self.tp_size = target_tp

    def afd_component_ensure_model_shell(self, target_tp):
        self.ensure_calls.append(target_tp)

    def afd_component_refresh_runtime(self, target_tp, perspective):
        self.refreshed.append((target_tp, perspective))

    def afd_component_refresh_peer_topology(self, attn_tp, ffn_tp, *, reset_data_plane):
        self.topology_refreshed.append((attn_tp, ffn_tp, reset_data_plane))

    def afd_component_demote_rank(self, perspective):
        self.demoted.append(perspective)


class _OrderedTransport(InMemoryMaxWorldTransport):
    def __init__(self, rank, world_size, shards, events):
        super().__init__(rank, world_size, shards)
        self.events = events

    def barrier(self):
        self.events.append("barrier")


def _shard(full, rule, tp):
    if rule.kind == "column_fused":
        result = []
        for rank in range(tp):
            offset, pieces = 0, []
            for size in rule.segments:
                pieces.append(full.narrow(0, offset + rank * (size // tp), size // tp))
                offset += size
            result.append(torch.cat(pieces, 0).clone())
        return tuple(result)
    size = full.shape[rule.dim] // tp
    return tuple(full.narrow(rule.dim, rank * size, size).clone() for rank in range(tp))


class TestAFDComponentWeightStaging(CustomTestCase):
    def test_dense_qwen3_a_and_f_tp2_to_tp4_commit(self):
        qkv_full = torch.arange(16 * 4).reshape(16, 4).float()
        gate_full = torch.arange(16 * 4).reshape(16, 4).float()
        down_full = torch.arange(4 * 16).reshape(4, 16).float()
        qkv_rule = SplitRule("column_fused", 0, (8, 4, 4))
        gate_rule = SplitRule("column_fused", 0, (8, 8))
        down_rule = SplitRule("row", 1)
        qkv, gate, down = (
            _shard(qkv_full, qkv_rule, 2),
            _shard(gate_full, gate_rule, 2),
            _shard(down_full, down_rule, 2),
        )
        shards = {
            "model.layers.0.self_attn.qkv_proj.weight": qkv,
            "model.layers.0.mlp.gate_up_proj.weight": gate,
            "model.layers.0.mlp.down_proj.weight": down,
        }
        request = {
            "operation_id": "expand",
            "epoch": 1,
            "expected_attn_tp": 2,
            "expected_ffn_tp": 2,
            "target_attn_tp": 4,
            "target_ffn_tp": 4,
        }
        for perspective, expected_names in (
            (
                AFDPerspective.AFD_PERSPECTIVE_ATTN,
                {"model.layers.0.self_attn.qkv_proj.weight"},
            ),
            (
                AFDPerspective.AFD_PERSPECTIVE_FFN,
                {
                    "model.layers.0.mlp.gate_up_proj.weight",
                    "model.layers.0.mlp.down_proj.weight",
                },
            ),
        ):
            runner = _Runner(_Model(qkv[0], gate[0], down[0]))
            stager = ModelRunnerAFDComponentStager(
                runner,
                perspective,
                4,
                InMemoryMaxWorldTransport(0, 4, shards),
                shadow_device="cpu",
            )
            live_before = {
                name: {
                    "data_ptr": param.data_ptr(),
                    "storage_ptr": param.untyped_storage().data_ptr(),
                    "storage_offset": param.storage_offset(),
                    "shape": param.shape,
                    "value": param.detach().clone(),
                }
                for name, param in runner.model.named_parameters()
            }
            state = stager.prepare(request)
            self.assertEqual(state.shadow, {})
            self.assertEqual(set(state.deferred), expected_names)
            self.assertEqual(state.staged_gpu_bytes, 0)
            self.assertGreater(state.deferred_local_bytes, 0)
            live_after = dict(runner.model.named_parameters())
            for name in expected_names:
                param = live_after[name]
                before = live_before[name]
                self.assertEqual(param.data_ptr(), before["data_ptr"])
                self.assertEqual(
                    param.untyped_storage().data_ptr(), before["storage_ptr"]
                )
                self.assertEqual(param.storage_offset(), before["storage_offset"])
                self.assertEqual(param.shape, before["shape"])
                self.assertTrue(torch.equal(param, before["value"]))
            stager.activate(request)
            self.assertEqual(runner.rebuilt, [4])
            self.assertEqual(runner.refreshed[0][0], 4)
            # The final activation barrier is the retirement boundary: every
            # rank drops staging and rollback references before serving resumes.
            self.assertIsNone(stager.prepared)
            self.assertEqual(state.shadow, {})
            self.assertEqual(state.old_tensors, {})
            stager.retire(request)  # external rank-zero RETIRE is idempotent

    def test_group_refresh_happens_inside_activation_before_runtime_and_return(self):
        qkv_full = torch.arange(16 * 4).reshape(16, 4).float()
        qkv = _shard(qkv_full, SplitRule("column_fused", 0, (8, 4, 4)), 2)
        name = "model.layers.0.self_attn.qkv_proj.weight"
        request = {
            "operation_id": "ordered-refresh",
            "epoch": 1,
            "expected_attn_tp": 2,
            "expected_ffn_tp": 2,
            "target_attn_tp": 4,
            "target_ffn_tp": 4,
        }
        runner = _Runner(_Model(qkv[0], torch.ones(8, 4), torch.ones(4, 8)))
        events = []
        transport = _OrderedTransport(0, 4, {name: qkv}, events)
        original_rebuild = runner.afd_component_rebuild_groups
        runner.afd_component_rebuild_groups = lambda tp: (
            events.append(("rebuild", tp)),
            original_rebuild(tp),
        )[-1]
        original_runtime = runner.afd_component_refresh_runtime
        runner.afd_component_refresh_runtime = lambda tp, perspective: (
            events.append(("runtime", tp)),
            original_runtime(tp, perspective),
        )[-1]
        stager = ModelRunnerAFDComponentStager(
            runner,
            AFDPerspective.AFD_PERSPECTIVE_ATTN,
            4,
            transport,
            shadow_device="cpu",
            group_refresh_callback=lambda tp: events.append(("groups", tp)),
            topology_refresh_callback=lambda tp, generation: events.append(
                ("bootstrap", tp, generation)
            ),
        )

        stager.prepare(request)
        events.clear()
        stager.activate(request)

        self.assertEqual(
            events,
            [
                "barrier",
                ("rebuild", 4),
                ("groups", 4),
                ("runtime", 4),
                ("bootstrap", 4, 2),
                "barrier",
            ],
        )
        self.assertIsNone(stager.prepared)

    def test_ffn_activate_does_not_publish_pd_bootstrap(self):
        gate_full = torch.arange(16 * 4).reshape(16, 4).float()
        gate = _shard(gate_full, SplitRule("column_fused", 0, (8, 8)), 2)
        name = "model.layers.0.mlp.gate_up_proj.weight"
        down_name = "model.layers.0.mlp.down_proj.weight"
        down = _shard(torch.ones(4, 8), SplitRule("row", 1), 2)
        request = {
            "operation_id": "pf-no-bootstrap",
            "epoch": 0,
            "expected_attn_tp": 2,
            "expected_ffn_tp": 2,
            "target_attn_tp": 4,
            "target_ffn_tp": 4,
        }
        runner = _Runner(_Model(torch.ones(8, 4), gate[0], torch.ones(4, 8)))
        events = []
        stager = ModelRunnerAFDComponentStager(
            runner,
            AFDPerspective.AFD_PERSPECTIVE_FFN,
            4,
            _OrderedTransport(0, 4, {name: gate, down_name: down}, events),
            shadow_device="cpu",
            topology_refresh_callback=lambda tp, generation: events.append(
                ("bootstrap", tp, generation)
            ),
        )

        stager.prepare(request)
        events.clear()
        stager.activate(request)

        self.assertNotIn(("bootstrap", 4, 1), events)
        self.assertEqual(events[-1], "barrier")

    def test_shrink_follower_demotes_at_activation_success_point(self):
        full = torch.arange(16 * 3).reshape(16, 3).float()
        rule = SplitRule("column_fused", 0, (8, 4, 4))
        qkv = _shard(full, rule, 4)
        name = "model.layers.0.self_attn.qkv_proj.weight"
        request = {
            "operation_id": "shrink-follower",
            "epoch": 2,
            "expected_attn_tp": 4,
            "expected_ffn_tp": 4,
            "target_attn_tp": 1,
            "target_ffn_tp": 4,
        }
        rank_zero = _Runner(
            _Model(qkv[0], torch.ones(4, 3), torch.ones(3, 4)), tp_size=4
        )
        descriptor = ModelRunnerAFDComponentStager(
            rank_zero,
            AFDPerspective.AFD_PERSPECTIVE_ATTN,
            4,
            InMemoryMaxWorldTransport(0, 4, {name: qkv}),
            shadow_device="cpu",
        )._descriptor(request, 4, 1)
        runner = _Runner(_Model(qkv[1], torch.ones(4, 3), torch.ones(3, 4)), tp_size=4)
        transport = InMemoryMaxWorldTransport(1, 4, {name: qkv})
        transport._manifest = descriptor
        stager = ModelRunnerAFDComponentStager(
            runner,
            AFDPerspective.AFD_PERSPECTIVE_ATTN,
            4,
            transport,
            shadow_device="cpu",
        )

        state = stager.prepare(request)
        stager.activate(request)

        self.assertIsNone(stager.prepared)
        self.assertEqual(state.shadow, {})
        self.assertEqual(state.old_tensors, {})
        self.assertEqual(runner.demoted, [AFDPerspective.AFD_PERSPECTIVE_ATTN])

    def test_activation_rollback_refreshes_source_groups_before_return(self):
        qkv_full = torch.arange(16 * 4).reshape(16, 4).float()
        qkv = _shard(qkv_full, SplitRule("column_fused", 0, (8, 4, 4)), 2)
        name = "model.layers.0.self_attn.qkv_proj.weight"
        request = {
            "operation_id": "rollback-refresh",
            "epoch": 1,
            "expected_attn_tp": 2,
            "expected_ffn_tp": 2,
            "target_attn_tp": 4,
            "target_ffn_tp": 4,
        }
        runner = _Runner(_Model(qkv[0], torch.ones(8, 4), torch.ones(4, 8)))
        original = (
            runner.model.model.layers[0].self_attn.qkv_proj.weight.detach().clone()
        )
        events = []
        transport = _OrderedTransport(0, 4, {name: qkv}, events)
        original_rebuild = runner.afd_component_rebuild_groups
        runner.afd_component_rebuild_groups = lambda tp: (
            events.append(("rebuild", tp)),
            original_rebuild(tp),
        )[-1]

        def fail_runtime(target_tp, perspective):
            events.append(("runtime-failed", target_tp))
            raise RuntimeError("injected runtime failure")

        runner.afd_component_refresh_runtime = fail_runtime
        stager = ModelRunnerAFDComponentStager(
            runner,
            AFDPerspective.AFD_PERSPECTIVE_ATTN,
            4,
            transport,
            shadow_device="cpu",
            group_refresh_callback=lambda tp: events.append(("groups", tp)),
        )
        stager.prepare(request)
        events.clear()

        with self.assertRaisesRegex(
            RuntimeError, "ACTIVATE aborted.*injected runtime failure"
        ):
            stager.activate(request)

        self.assertEqual(
            events,
            [
                "barrier",
                ("rebuild", 4),
                ("groups", 4),
                ("runtime-failed", 4),
                ("rebuild", 2),
                ("groups", 2),
            ],
        )
        self.assertEqual(runner.tp_size, 2)
        self.assertTrue(
            torch.equal(
                runner.model.model.layers[0].self_attn.qkv_proj.weight, original
            )
        )
        self.assertIsNone(stager.prepared)

    def test_attention_tp4_to_tp1_cpu_fused_segment_major_and_bound(self):
        full = torch.arange(16 * 3).reshape(16, 3).float()
        rule = SplitRule("column_fused", 0, (8, 4, 4))
        qkv = _shard(full, rule, 4)
        gate = _shard(torch.ones(16, 4), SplitRule("column_fused", 0, (8, 8)), 4)
        down = _shard(torch.ones(4, 16), SplitRule("row", 1), 4)
        name = "model.layers.0.self_attn.qkv_proj.weight"
        shards = {
            name: qkv,
            "model.layers.0.mlp.gate_up_proj.weight": gate,
            "model.layers.0.mlp.down_proj.weight": down,
        }
        request = {
            "operation_id": "shrink",
            "epoch": 2,
            "expected_attn_tp": 4,
            "expected_ffn_tp": 4,
            "target_attn_tp": 1,
            "target_ffn_tp": 4,
        }
        runner = _Runner(_Model(qkv[0], gate[0], down[0]), tp_size=4)
        stager = ModelRunnerAFDComponentStager(
            runner,
            AFDPerspective.AFD_PERSPECTIVE_ATTN,
            4,
            InMemoryMaxWorldTransport(0, 4, shards),
            shadow_device="cpu",
            max_cpu_staging_bytes=2 * full.numel() * full.element_size(),
        )
        state = stager.prepare(request)
        self.assertTrue(torch.equal(state.shadow[name], full))
        with self.assertRaisesRegex(RuntimeError, "bounded CPU staging exceeded"):
            ModelRunnerAFDComponentStager(
                _Runner(_Model(qkv[0], gate[0], down[0]), tp_size=4),
                AFDPerspective.AFD_PERSPECTIVE_ATTN,
                4,
                InMemoryMaxWorldTransport(0, 4, shards),
                shadow_device="cpu",
                max_cpu_staging_bytes=1,
            ).prepare(request)

    def test_joining_prepare_does_not_build_model_shell(self):
        qkv_full = torch.arange(16 * 4).reshape(16, 4).float()
        qkv = _shard(qkv_full, SplitRule("column_fused", 0, (8, 4, 4)), 2)
        name = "model.layers.0.self_attn.qkv_proj.weight"
        source_runner = _Runner(_Model(qkv[0], torch.ones(8, 4), torch.ones(4, 8)))
        source_transport = InMemoryMaxWorldTransport(0, 4, {name: qkv})
        request = {
            "operation_id": "joining",
            "epoch": 1,
            "expected_attn_tp": 2,
            "expected_ffn_tp": 2,
            "target_attn_tp": 4,
            "target_ffn_tp": 4,
        }
        source_stager = ModelRunnerAFDComponentStager(
            source_runner,
            AFDPerspective.AFD_PERSPECTIVE_ATTN,
            4,
            source_transport,
            shadow_device="cpu",
        )
        descriptor = source_stager._descriptor(request, 2, 4)
        joining_runner = _Runner(None)
        joining_transport = InMemoryMaxWorldTransport(2, 4, {name: qkv})
        joining_transport._manifest = descriptor
        joining_stager = ModelRunnerAFDComponentStager(
            joining_runner,
            AFDPerspective.AFD_PERSPECTIVE_ATTN,
            4,
            joining_transport,
            shadow_device="cpu",
        )
        state = joining_stager.prepare(request)
        self.assertIsNone(joining_runner.model)
        self.assertEqual(joining_runner.ensure_calls, [])
        self.assertEqual(set(state.shadow), {name})

    def test_qwen3_32b_gqa_rule_uses_real_config_dimensions(self):
        # Qwen3-32B config: 64 Q heads, 8 KV heads, head_dim 128.
        module = QKVParallelLinear(torch.empty(5120, 64))
        module.head_size = 128
        module.total_num_heads = 64
        module.total_num_kv_heads = 8
        model = torch.nn.Module()
        model.qkv_proj = module
        from sglang.srt.layers.reshard_weights import get_tp_split_rules

        rule = get_tp_split_rules(model)["qkv_proj.weight"]
        self.assertEqual(rule, ("column_fused", 0, (8192, 1024, 1024)))
        self.assertTrue(all(segment % 4 == 0 for segment in rule[2]))

    def test_fake_staging_env_skips_real_weight_movement(self):
        import os

        qkv_full = torch.arange(16 * 4).reshape(16, 4).float()
        gate_full = torch.arange(16 * 4).reshape(16, 4).float()
        down_full = torch.arange(4 * 16).reshape(4, 16).float()
        qkv = _shard(qkv_full, SplitRule("column_fused", 0, (8, 4, 4)), 2)
        gate = _shard(gate_full, SplitRule("column_fused", 0, (8, 8)), 2)
        down = _shard(down_full, SplitRule("row", 1), 2)
        shards = {
            "model.layers.0.self_attn.qkv_proj.weight": qkv,
            "model.layers.0.mlp.gate_up_proj.weight": gate,
            "model.layers.0.mlp.down_proj.weight": down,
        }
        request = {
            "operation_id": "fake",
            "epoch": 1,
            "expected_attn_tp": 2,
            "expected_ffn_tp": 2,
            "target_attn_tp": 4,
            "target_ffn_tp": 4,
        }
        runner = _Runner(_Model(qkv[0], gate[0], down[0]))
        stager = ModelRunnerAFDComponentStager(
            runner,
            AFDPerspective.AFD_PERSPECTIVE_ATTN,
            4,
            InMemoryMaxWorldTransport(0, 4, shards),
            shadow_device="cpu",
        )
        prev = os.environ.get("AFD_RESHARD_FAKE_STAGING")
        os.environ["AFD_RESHARD_FAKE_STAGING"] = "1"
        try:
            state = stager.prepare(request)
        finally:
            if prev is None:
                os.environ.pop("AFD_RESHARD_FAKE_STAGING", None)
            else:
                os.environ["AFD_RESHARD_FAKE_STAGING"] = prev
        # READY but empty shadow: no real weights staged.
        self.assertEqual(state.state.value, "READY")
        self.assertTrue(state.no_op)
        self.assertTrue(state.fake)
        self.assertEqual(state.shadow, {})
        # activate() short-circuits: no group rebuild / H2D under fake mode.
        stager.activate(request)
        self.assertEqual(runner.rebuilt, [])
        self.assertEqual(runner.refreshed, [])
        self.assertEqual(runner.topology_refreshed, [])
        self.assertIsNone(stager.prepared)
        self.assertEqual(state.shadow, {})
        self.assertEqual(state.old_tensors, {})

    def test_fake_staging_off_by_default_does_real_staging(self):
        qkv_full = torch.arange(16 * 4).reshape(16, 4).float()
        gate_full = torch.arange(16 * 4).reshape(16, 4).float()
        down_full = torch.arange(4 * 16).reshape(4, 16).float()
        qkv = _shard(qkv_full, SplitRule("column_fused", 0, (8, 4, 4)), 2)
        gate = _shard(gate_full, SplitRule("column_fused", 0, (8, 8)), 2)
        down = _shard(down_full, SplitRule("row", 1), 2)
        shards = {
            "model.layers.0.self_attn.qkv_proj.weight": qkv,
            "model.layers.0.mlp.gate_up_proj.weight": gate,
            "model.layers.0.mlp.down_proj.weight": down,
        }
        request = {
            "operation_id": "real",
            "epoch": 1,
            "expected_attn_tp": 2,
            "expected_ffn_tp": 2,
            "target_attn_tp": 4,
            "target_ffn_tp": 4,
        }
        runner = _Runner(_Model(qkv[0], gate[0], down[0]))
        stager = ModelRunnerAFDComponentStager(
            runner,
            AFDPerspective.AFD_PERSPECTIVE_ATTN,
            4,
            InMemoryMaxWorldTransport(0, 4, shards),
            shadow_device="cpu",
        )
        # Active expansion ranks defer local transformation until ACTIVATE.
        state = stager.prepare(request)
        self.assertFalse(state.fake)
        self.assertEqual(state.shadow, {})
        self.assertEqual(
            set(state.deferred), {"model.layers.0.self_attn.qkv_proj.weight"}
        )

    def _ffn_noop_stager(self, request):
        gate = torch.ones(4, 4)
        down = torch.ones(4, 4)
        shards = {
            "model.layers.0.mlp.gate_up_proj.weight": (gate,) * 4,
            "model.layers.0.mlp.down_proj.weight": (down,) * 4,
        }
        runner = _Runner(_Model(torch.ones(4, 4), gate, down), tp_size=4)
        stager = ModelRunnerAFDComponentStager(
            runner,
            AFDPerspective.AFD_PERSPECTIVE_FFN,
            4,
            InMemoryMaxWorldTransport(0, 4, shards),
            shadow_device="cpu",
        )
        return runner, stager

    def test_ffn_noop_preserves_gathered_rank_timings(self):
        request = {
            "operation_id": "noop-rank-timings", "epoch": 2,
            "expected_attn_tp": 1, "target_attn_tp": 1,
            "expected_ffn_tp": 4, "target_ffn_tp": 4,
        }
        runner, stager = self._ffn_noop_stager(request)
        transport = stager.transport
        transport.gather_timings = lambda timings: (
            dict(timings), {**dict(timings), "activate_s": 9.0}
        )
        stager.prepare(request)
        stager.activate(request)
        completed = stager.last_completed_status
        self.assertEqual(completed["operation_id"], "noop-rank-timings")
        self.assertEqual(len(completed["rank_timings_s"]), 2)
        self.assertEqual(completed["rank_timings_s"][1]["activate_s"], 9.0)

    def test_ffn_noop_refreshes_changed_attention_peer_topology(self):
        request = {
            "operation_id": "peer-changed",
            "epoch": 2,
            "expected_attn_tp": 4,
            "target_attn_tp": 1,
            "expected_ffn_tp": 4,
            "target_ffn_tp": 4,
        }
        runner, stager = self._ffn_noop_stager(request)
        state = stager.prepare(request)
        self.assertTrue(state.no_op)

        stager.activate(request)

        self.assertEqual(runner.rebuilt, [])
        self.assertEqual(runner.topology_refreshed, [(1, 4, True)])
        self.assertIsNone(stager.prepared)

    def test_ffn_noop_does_not_reset_unchanged_topology(self):
        request = {
            "operation_id": "topology-unchanged",
            "epoch": 2,
            "expected_attn_tp": 1,
            "target_attn_tp": 1,
            "expected_ffn_tp": 4,
            "target_ffn_tp": 4,
        }
        runner, stager = self._ffn_noop_stager(request)
        stager.prepare(request)

        stager.activate(request)

        self.assertEqual(runner.topology_refreshed, [(1, 4, False)])
        self.assertIsNone(stager.prepared)

    def test_ffn_noop_refresh_failure_aborts_without_commit(self):
        request = {
            "operation_id": "peer-refresh-failure",
            "epoch": 2,
            "expected_attn_tp": 4,
            "target_attn_tp": 1,
            "expected_ffn_tp": 4,
            "target_ffn_tp": 4,
        }
        runner, stager = self._ffn_noop_stager(request)
        state = stager.prepare(request)

        def fail_refresh(*args, **kwargs):
            raise RuntimeError("injected peer refresh failure")

        runner.afd_component_refresh_peer_topology = fail_refresh
        with self.assertRaisesRegex(
            RuntimeError, "ACTIVATE aborted.*injected peer refresh failure"
        ):
            stager.activate(request)

        self.assertEqual(state.state.value, "ABORTED")
        self.assertIsNone(stager.prepared)

    def test_external_retire_does_not_discard_ready_shadow(self):
        gate = torch.ones(4, 4)
        down = torch.ones(4, 4)
        runner = _Runner(_Model(torch.ones(4, 4), gate, down), tp_size=4)
        request = {
            "operation_id": "ready",
            "epoch": 2,
            "expected_attn_tp": 1,
            "target_attn_tp": 1,
            "expected_ffn_tp": 4,
            "target_ffn_tp": 4,
        }
        stager = ModelRunnerAFDComponentStager(
            runner,
            AFDPerspective.AFD_PERSPECTIVE_FFN,
            4,
            InMemoryMaxWorldTransport(
                0,
                4,
                {
                    "model.layers.0.mlp.gate_up_proj.weight": (gate,) * 4,
                    "model.layers.0.mlp.down_proj.weight": (down,) * 4,
                },
            ),
            shadow_device="cpu",
        )
        state = stager.prepare(request)

        stager.retire(request)

        self.assertIs(stager.prepared, state)
        self.assertEqual(state.state.value, "READY")

    def test_async_prepare_returns_immediately_then_ready(self):
        full = torch.arange(16 * 4).reshape(16, 4).float()
        name = "model.layers.0.self_attn.qkv_proj.weight"
        shards = {name: _shard(full, SplitRule("column_fused", 0, (8, 4, 4)), 2)}

        class BlockingTransport(InMemoryMaxWorldTransport):
            def __init__(self):
                super().__init__(0, 4, shards)
                self.release = threading.Event()

            def stage_target_shard(self, *args, **kwargs):
                self.release.wait(1)
                return super().stage_target_shard(*args, **kwargs)

        transport = BlockingTransport()
        runner = _Runner(_Model(shards[name][0], torch.ones(8, 4), torch.ones(4, 8)))
        stager = ModelRunnerAFDComponentStager(
            runner,
            AFDPerspective.AFD_PERSPECTIVE_ATTN,
            4,
            transport,
            shadow_device="cpu",
        )
        request = {
            "operation_id": "async",
            "epoch": 1,
            "expected_attn_tp": 2,
            "expected_ffn_tp": 2,
            "target_attn_tp": 4,
            "target_ffn_tp": 4,
        }
        started = time.perf_counter()
        status = stager.start_prepare(request)
        self.assertLess(time.perf_counter() - started, 0.2)
        self.assertEqual(status["state"], "PREPARING")
        transport.release.set()
        deadline = time.monotonic() + 2
        while (
            stager.prepare_status("async")["state"] == "PREPARING"
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        final = stager.prepare_status("async")
        self.assertEqual(final["state"], "READY")
        self.assertIn("descriptor_s", final["timings_s"])
        self.assertIn("parameter_staging_s", final["timings_s"])
        self.assertIn("prepare_s", final["timings_s"])
        self.assertEqual(final["timings_s"]["staged_gpu_bytes"], 0)
        self.assertGreater(final["timings_s"]["deferred_local_bytes"], 0)

    def test_async_prepare_propagates_transport_error(self):
        name = "model.layers.0.self_attn.qkv_proj.weight"
        shard = torch.ones(8, 4)

        class FailingTransport(InMemoryMaxWorldTransport):
            def stage_target_shard(self, *args, **kwargs):
                raise RuntimeError("injected transfer failure")

        runner = _Runner(_Model(shard, torch.ones(8, 4), torch.ones(4, 8)))
        stager = ModelRunnerAFDComponentStager(
            runner,
            AFDPerspective.AFD_PERSPECTIVE_ATTN,
            4,
            FailingTransport(0, 4, {name: (shard, shard)}),
            shadow_device="cpu",
        )
        request = {
            "operation_id": "error",
            "epoch": 1,
            "expected_attn_tp": 2,
            "expected_ffn_tp": 2,
            "target_attn_tp": 4,
            "target_ffn_tp": 4,
        }
        stager.start_prepare(request)
        deadline = time.monotonic() + 2
        while (
            stager.prepare_status("error")["state"] == "PREPARING"
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        status = stager.prepare_status("error")
        self.assertEqual(status["state"], "ERROR")
        self.assertIn("injected transfer failure", status["error"])

    def test_breakdown_fields_are_explicit_and_host_load_is_zero(self):
        full = torch.arange(16 * 4).reshape(16, 4).float()
        rule = SplitRule("column_fused", 0, (8, 4, 4))
        name = "model.layers.0.self_attn.qkv_proj.weight"
        shards = _shard(full, rule, 2)
        runner = _Runner(_Model(shards[0], torch.ones(8, 4), torch.ones(4, 8)))
        stager = ModelRunnerAFDComponentStager(
            runner, AFDPerspective.AFD_PERSPECTIVE_ATTN, 4,
            InMemoryMaxWorldTransport(0, 4, {name: shards}), shadow_device="cpu",
        )
        state = stager.prepare({
            "operation_id": "metrics", "epoch": 1,
            "expected_attn_tp": 2, "expected_ffn_tp": 2,
            "target_attn_tp": 4, "target_ffn_tp": 4,
        })
        for field in (
            "local_repartition_s", "local_repartition_bytes",
            "peer_transfer_s", "peer_transfer_bytes",
            "host_stage_d2h_s", "host_stage_d2h_bytes",
            "host_load_remaining_s", "host_load_remaining_bytes",
        ):
            self.assertIn(field, state.timings_s)
        self.assertEqual(state.timings_s["host_load_remaining_s"], 0.0)
        self.assertEqual(state.timings_s["host_load_remaining_bytes"], 0)
        self.assertEqual(state.timings_s["peer_transfer_bytes"], 0)

    def test_direct_transport_receives_only_owned_target_shard(self):
        full = torch.arange(4 * 16).reshape(4, 16).float()
        shards = _shard(full, SplitRule("row", 1), 2)
        transport = InMemoryMaxWorldTransport(3, 4, {"row": shards})
        target = transport.stage_target_shard("row", None, SplitRule("row", 1), 2, 4, 3)
        self.assertTrue(torch.equal(target, full[:, 12:16]))

    def test_expand_active_defers_while_joiner_stages_and_activate_matches(self):
        full = torch.arange(16 * 4).reshape(16, 4).float()
        rule = SplitRule("column_fused", 0, (8, 4, 4))
        shards = _shard(full, rule, 2)
        name = "model.layers.0.self_attn.qkv_proj.weight"
        request = {
            "operation_id": "low-peak",
            "epoch": 1,
            "expected_attn_tp": 2,
            "expected_ffn_tp": 2,
            "target_attn_tp": 4,
            "target_ffn_tp": 4,
        }

        active_runner = _Runner(_Model(shards[0], torch.ones(8, 4), torch.ones(4, 8)))
        active = ModelRunnerAFDComponentStager(
            active_runner,
            AFDPerspective.AFD_PERSPECTIVE_ATTN,
            4,
            InMemoryMaxWorldTransport(0, 4, {name: shards}),
            shadow_device="cpu",
        )
        descriptor = active._descriptor(request, 2, 4)
        active_state = active.prepare(request)
        self.assertEqual(active_state.shadow, {})
        self.assertEqual(set(active_state.deferred), {name})
        active.activate(request)
        expected_rank0 = _shard(full, rule, 4)[0]
        self.assertTrue(
            torch.equal(
                active_runner.model.model.layers[0].self_attn.qkv_proj.weight,
                expected_rank0,
            )
        )

        joining_runner = _Runner(None)
        joining_transport = InMemoryMaxWorldTransport(2, 4, {name: shards})
        joining_transport._manifest = descriptor
        joining = ModelRunnerAFDComponentStager(
            joining_runner,
            AFDPerspective.AFD_PERSPECTIVE_ATTN,
            4,
            joining_transport,
            shadow_device="cpu",
        )
        joining_state = joining.prepare(request)
        self.assertEqual(joining_state.deferred, {})
        self.assertEqual(set(joining_state.shadow), {name})
        self.assertTrue(
            torch.equal(joining_state.shadow[name], _shard(full, rule, 4)[2])
        )

    def test_serial_tp1_assembly_matches_reference_for_all_split_rules(self):
        cases = (
            (
                torch.arange(3 * 5, dtype=torch.float32).reshape(3, 5),
                SplitRule("replicated"),
            ),
            (
                torch.arange(12 * 5, dtype=torch.float32).reshape(12, 5),
                SplitRule("column", 0),
            ),
            (
                torch.arange(3 * 16, dtype=torch.float32).reshape(3, 16),
                SplitRule("row", 1),
            ),
            (
                torch.arange(16 * 5, dtype=torch.float32).reshape(16, 5),
                SplitRule("column_fused", 0, (8, 4, 4)),
            ),
        )
        for full, rule in cases:
            with self.subTest(kind=rule.kind):
                shards = (
                    tuple(full.clone() for _ in range(4))
                    if rule.kind == "replicated"
                    else _shard(full, rule, 4)
                )
                actual = _assemble_tp1_target(shards, rule)
                expected = _target_shard(shards, rule, 1, 0)
                self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(torch.equal(actual, full))

    def test_serial_tp1_replicated_assembly_validates_sources(self):
        shards = [torch.ones(2, 3) for _ in range(4)]
        shards[3][0, 0] = 2
        with self.assertRaisesRegex(Exception, "replicated source shards differ"):
            _assemble_tp1_target(tuple(shards), SplitRule("replicated"))

    def test_hybrid_selects_transport_by_operation_direction(self):
        gloo = SimpleNamespace(rank=0, world_size=4, backend_name="gloo")
        nccl = SimpleNamespace(rank=0, world_size=4, backend_name="nccl")
        hybrid = HybridMaxWorldTransport(gloo, nccl)
        self.assertIs(hybrid.select_operation_transport(2, 4), nccl)
        self.assertIs(hybrid.select_operation_transport(4, 1), gloo)
        self.assertIs(hybrid.select_operation_transport(4, 4), gloo)

    def test_hybrid_nccl_expansion_active_rank_defers_during_prepare(self):
        full = torch.arange(16 * 4, dtype=torch.float32).reshape(16, 4)
        rule = SplitRule("column_fused", 0, (8, 4, 4))
        shards = _shard(full, rule, 2)
        name = "model.layers.0.self_attn.qkv_proj.weight"
        request = {
            "operation_id": "hybrid-expand-active",
            "epoch": 1,
            "expected_attn_tp": 2,
            "expected_ffn_tp": 2,
            "target_attn_tp": 4,
            "target_ffn_tp": 4,
        }

        class CPUOnlyNCCLOperation(TorchDistributedNCCLMaxWorldTransport):
            def __init__(self, delegate):
                self.rank = delegate.rank
                self.world_size = delegate.world_size
                self.backend_name = "nccl"
                self.delegate = delegate

            def broadcast_manifest(self, payload):
                return self.delegate.broadcast_manifest(payload)

            def stage_target_shard(self, *args, **kwargs):
                return self.delegate.stage_target_shard(*args, **kwargs)

            def materialize_deferred_target(self, *args, **kwargs):
                return self.delegate.materialize_deferred_target(*args, **kwargs)

            def consensus_error(self, error):
                return error

            def gather_timings(self, timings):
                return (dict(timings),)

        delegate = InMemoryMaxWorldTransport(0, 4, {name: shards})
        nccl = CPUOnlyNCCLOperation(delegate)
        hybrid = HybridMaxWorldTransport(delegate, nccl)
        runner = _Runner(_Model(shards[0], torch.ones(8, 4), torch.ones(4, 8)))
        stager = ModelRunnerAFDComponentStager(
            runner,
            AFDPerspective.AFD_PERSPECTIVE_ATTN,
            4,
            hybrid,
            shadow_device="cpu",
        )

        state = stager.prepare(request)

        self.assertIs(state.operation_transport, nccl)
        self.assertEqual(state.shadow, {})
        self.assertEqual(set(state.deferred), {name})

    def test_hybrid_nccl_shrink_keeps_cpu_target_with_cuda_shadow_policy(self):
        full = torch.arange(16 * 3, dtype=torch.float32).reshape(16, 3)
        rule = SplitRule("column_fused", 0, (8, 4, 4))
        shards = _shard(full, rule, 4)
        name = "model.layers.0.self_attn.qkv_proj.weight"
        request = {
            "operation_id": "hybrid-nccl-shrink",
            "epoch": 2,
            "expected_attn_tp": 4,
            "expected_ffn_tp": 4,
            "target_attn_tp": 1,
            "target_ffn_tp": 4,
        }

        class CPUOnlyNCCLShrink(TorchDistributedNCCLMaxWorldTransport):
            def __init__(self, delegate):
                self.rank = delegate.rank
                self.world_size = delegate.world_size
                self.backend_name = "nccl"
                self.delegate = delegate

            def broadcast_manifest(self, payload):
                return self.delegate.broadcast_manifest(payload)

            def stage_target_shard(self, *args, **kwargs):
                return self.delegate.stage_target_shard(*args, **kwargs)

            def consensus_error(self, error):
                return error

            def gather_timings(self, timings):
                return (dict(timings),)

        delegate = InMemoryMaxWorldTransport(0, 4, {name: shards})
        nccl = CPUOnlyNCCLShrink(delegate)
        hybrid = HybridMaxWorldTransport(delegate, nccl)
        hybrid.select_operation_transport = lambda source_tp, target_tp: nccl
        runner = _Runner(
            _Model(shards[0], torch.ones(4, 3), torch.ones(3, 4)), tp_size=4
        )
        stager = ModelRunnerAFDComponentStager(
            runner,
            AFDPerspective.AFD_PERSPECTIVE_ATTN,
            4,
            hybrid,
            shadow_device="cuda",
        )

        state = stager.prepare(request)

        self.assertIs(state.operation_transport, nccl)
        self.assertEqual(state.shadow[name].device.type, "cpu")
        self.assertTrue(torch.equal(state.shadow[name], full))

    def test_hybrid_ffn_noop_keeps_operation_transport_through_activate(self):
        class RecordingTransport(InMemoryMaxWorldTransport):
            def __init__(self, label):
                super().__init__(0, 4, {})
                self.backend_name = label
                self.consensus_calls = 0
                self.barrier_calls = 0

            def consensus_error(self, error):
                self.consensus_calls += 1
                return error

            def barrier(self):
                self.barrier_calls += 1

        gloo = RecordingTransport("gloo")
        nccl = RecordingTransport("nccl")
        hybrid = HybridMaxWorldTransport(gloo, nccl)
        runner = _Runner(
            _Model(torch.ones(8, 4), torch.ones(8, 4), torch.ones(4, 8)),
            tp_size=4,
        )
        stager = ModelRunnerAFDComponentStager(
            runner, AFDPerspective.AFD_PERSPECTIVE_FFN, 4, hybrid
        )
        request = {
            "operation_id": "hybrid-ffn-noop",
            "epoch": 1,
            "expected_attn_tp": 1,
            "target_attn_tp": 1,
            "expected_ffn_tp": 4,
            "target_ffn_tp": 4,
        }
        state = stager.prepare(request)
        self.assertTrue(state.no_op)
        self.assertIs(state.operation_transport, gloo)
        # Changing future selection cannot move this in-flight operation.
        hybrid.gloo_transport = nccl
        stager.activate(request)
        self.assertEqual(gloo.consensus_calls, 1)
        self.assertEqual(gloo.barrier_calls, 1)
        self.assertEqual(nccl.consensus_calls, 0)

    def test_hybrid_group_creation_order_and_legacy_modes(self):
        class FakeDist:
            def __init__(self):
                self.calls = []

            def new_group(self, *, ranks, backend, timeout):
                group = f"{backend}-{len(self.calls)}"
                self.calls.append((tuple(ranks), backend, timeout))
                return group

        dist = FakeDist()
        groups = create_afd_component_staging_groups(
            dist, [0, 1, 2, 3], "hybrid", "timeout"
        )
        self.assertEqual([call[1] for call in dist.calls], ["gloo", "nccl", "gloo"])
        self.assertEqual(
            groups, {"gloo": "gloo-0", "nccl": "nccl-1", "control": "gloo-2"}
        )
        for backend, expected in (
            ("gloo", ["gloo", "gloo"]),
            ("nccl", ["nccl", "gloo"]),
        ):
            with self.subTest(backend=backend):
                dist = FakeDist()
                groups = create_afd_component_staging_groups(
                    dist, [0, 1], backend, "timeout"
                )
                self.assertEqual([call[1] for call in dist.calls], expected)
                self.assertIsNone(groups["nccl" if backend == "gloo" else "gloo"])

    def test_nccl_attention_shrink_safety_no_longer_depends_on_env_gate(self):
        transport = object.__new__(TorchDistributedNCCLMaxWorldTransport)
        transport.rank = 0
        transport.world_size = 4
        request = {
            "operation_id": "shrink-safety",
            "expected_attn_tp": 4,
            "target_attn_tp": 1,
            "expected_ffn_tp": 4,
            "target_ffn_tp": 4,
        }
        runner = _Runner(None, tp_size=4)
        stager = ModelRunnerAFDComponentStager(
            runner, AFDPerspective.AFD_PERSPECTIVE_ATTN, 4, transport
        )
        import os

        previous = os.environ.pop("AFD_COMPONENT_ENABLE_NCCL_SHRINK", None)
        try:
            self.assertEqual(stager.safety_reason(request), "")
        finally:
            if previous is not None:
                os.environ["AFD_COMPONENT_ENABLE_NCCL_SHRINK"] = previous

    def test_nccl_ffn_tp4_to_tp4_remains_noop_safe(self):
        transport = object.__new__(TorchDistributedNCCLMaxWorldTransport)
        transport.rank = 0
        transport.world_size = 4
        runner = _Runner(None, tp_size=4)
        stager = ModelRunnerAFDComponentStager(
            runner, AFDPerspective.AFD_PERSPECTIVE_FFN, 4, transport
        )
        request = {
            "operation_id": "ffn-noop-safety",
            "expected_attn_tp": 1,
            "target_attn_tp": 1,
            "expected_ffn_tp": 4,
            "target_ffn_tp": 4,
        }
        self.assertEqual(stager.safety_reason(request), "")

    def test_nccl_metric_intervals_use_separate_boundaries(self):
        import inspect

        stage_source = inspect.getsource(
            TorchDistributedNCCLMaxWorldTransport.stage_target_shard
        )
        deferred_source = inspect.getsource(
            TorchDistributedNCCLMaxWorldTransport.materialize_deferred_target
        )
        for source, peer_boundary in (
            (stage_source, "transfer_started = time.perf_counter()"),
            (deferred_source, "peer_started = time.perf_counter()"),
        ):
            local_record = source.index('"local_repartition_s"')
            peer_start = source.index(peer_boundary)
            peer_record = source.index('"peer_transfer_s"')
            self.assertLess(local_record, peer_start)
            self.assertLess(peer_start, peer_record)
        self.assertNotIn("elapsed", deferred_source)

    def test_nccl_replicated_tp2_to_tp4_source_mapping_is_matched(self):
        from sglang.srt.reshard.afd_component_weight_staging import (
            TorchDistributedNCCLMaxWorldTransport,
        )

        rule = SplitRule("replicated")
        sources = [
            TorchDistributedNCCLMaxWorldTransport._expansion_source(rule, 2, 4, target)
            for target in range(4)
        ]
        self.assertEqual(sources, [0, 1, 0, 1])
        # Existing targets 0/1 retain their own source copy. New targets 2/3
        # receive from 0/1, yielding exactly one matched send/receive pair each.
        pairs = [
            (source, target)
            for target, source in enumerate(sources)
            if source != target
        ]
        self.assertEqual(pairs, [(0, 2), (1, 3)])
        replica = torch.arange(12).reshape(3, 4)
        for target in range(4):
            piece = TorchDistributedNCCLMaxWorldTransport._expansion_piece(
                replica, rule, 2, 4, target
            )
            self.assertTrue(torch.equal(piece, replica))


class TestNCCLShrinkGlooFallback(CustomTestCase):
    """Verify the NCCL transport shrink path uses Gloo CPU broadcast, not NCCL P2P."""

    def _make_nccl_transport_mock(self, rank, world_size):
        """Create a TorchDistributedNCCLMaxWorldTransport without real dist init.

        Uses mock groups that record calls for verification.
        """
        from unittest.mock import MagicMock

        transport = object.__new__(TorchDistributedNCCLMaxWorldTransport)
        transport.rank = rank
        transport.world_size = world_size
        transport.device = torch.device("cpu")
        transport.stream = MagicMock()
        transport.backend_name = "nccl"
        transport.data_group = MagicMock(name="data_group")
        transport.control_group = MagicMock(name="control_group")
        return transport

    def test_shrink_tp4_to_tp1_rank0_assembles_cpu_target(self):
        """Rank 0 receives correct full TP1 target on CPU via Gloo broadcast."""
        from unittest.mock import patch

        full = torch.arange(16 * 3, dtype=torch.float32).reshape(16, 3)
        rule = SplitRule("column_fused", 0, (8, 4, 4))
        source_tp, target_tp = 4, 1
        shards = _shard(full, rule, source_tp)

        transport = self._make_nccl_transport_mock(0, 4)
        local_tensor = shards[0]

        broadcast_calls = []

        def mock_broadcast(tensor, *, src, group):
            broadcast_calls.append(src)
            # Simulate: each source's shard is broadcast
            tensor.copy_(shards[src])

        def mock_all_gather_object(output_list, obj, *, group):
            meta = (tuple(shards[0].shape), str(shards[0].dtype))
            for i in range(len(output_list)):
                output_list[i] = meta if i < source_tp else None

        with patch("torch.distributed.broadcast", side_effect=mock_broadcast):
            with patch(
                "torch.distributed.all_gather_object",
                side_effect=mock_all_gather_object,
            ):
                result = transport.stage_target_shard(
                    "test_param", local_tensor, rule, source_tp, target_tp, 0
                )

        self.assertIsNotNone(result)
        self.assertEqual(result.device.type, "cpu")
        self.assertTrue(torch.equal(result, full))
        self.assertEqual(broadcast_calls, [0, 1, 2, 3])

    def test_shrink_tp4_to_tp1_non_rank0_returns_none(self):
        """Non-zero ranks return None from shrink path."""
        from unittest.mock import patch

        full = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8)
        rule = SplitRule("row", 1)
        source_tp, target_tp = 4, 1
        shards = _shard(full, rule, source_tp)

        for rank in (1, 2, 3):
            transport = self._make_nccl_transport_mock(rank, 4)
            local_tensor = shards[rank]

            def mock_broadcast(tensor, *, src, group):
                tensor.copy_(shards[src])

            def mock_all_gather_object(output_list, obj, *, group):
                meta = (tuple(shards[0].shape), str(shards[0].dtype))
                for i in range(len(output_list)):
                    output_list[i] = meta if i < source_tp else None

            with patch("torch.distributed.broadcast", side_effect=mock_broadcast):
                with patch(
                    "torch.distributed.all_gather_object",
                    side_effect=mock_all_gather_object,
                ):
                    result = transport.stage_target_shard(
                        "test_param", local_tensor, rule, source_tp, target_tp, rank
                    )

            self.assertIsNone(result)

    def test_shrink_does_not_call_nccl_p2p_ops(self):
        """The shrink path must never call isend, irecv, or batch_isend_irecv."""
        from unittest.mock import patch, MagicMock

        full = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8)
        rule = SplitRule("row", 1)
        source_tp, target_tp = 4, 1
        shards = _shard(full, rule, source_tp)

        transport = self._make_nccl_transport_mock(0, 4)
        local_tensor = shards[0]

        nccl_calls = []

        def track_isend(*args, **kwargs):
            nccl_calls.append("isend")
            return MagicMock()

        def track_irecv(*args, **kwargs):
            nccl_calls.append("irecv")
            return MagicMock()

        def track_batch(*args, **kwargs):
            nccl_calls.append("batch_isend_irecv")
            return []

        def mock_broadcast(tensor, *, src, group):
            tensor.copy_(shards[src])

        def mock_all_gather_object(output_list, obj, *, group):
            meta = (tuple(shards[0].shape), str(shards[0].dtype))
            for i in range(len(output_list)):
                output_list[i] = meta if i < source_tp else None

        with patch("torch.distributed.isend", side_effect=track_isend):
            with patch("torch.distributed.irecv", side_effect=track_irecv):
                with patch(
                    "torch.distributed.batch_isend_irecv", side_effect=track_batch
                ):
                    with patch(
                        "torch.distributed.broadcast", side_effect=mock_broadcast
                    ):
                        with patch(
                            "torch.distributed.all_gather_object",
                            side_effect=mock_all_gather_object,
                        ):
                            transport.stage_target_shard(
                                "test_param",
                                local_tensor,
                                rule,
                                source_tp,
                                target_tp,
                                0,
                            )

        self.assertEqual(
            nccl_calls, [], "shrink path must not call isend/irecv/batch_isend_irecv"
        )

    def test_expand_still_uses_nccl_p2p(self):
        """Expansion path continues to use NCCL batch_isend_irecv."""
        from unittest.mock import patch, MagicMock

        full = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8)
        rule = SplitRule("row", 1)
        source_tp, target_tp = 2, 4
        shards = _shard(full, rule, source_tp)

        transport = self._make_nccl_transport_mock(1, 4)
        transport.stream = MagicMock()
        transport.stream.synchronize = MagicMock()
        local_tensor = MagicMock()
        local_tensor.shape = tuple(shards[1].shape)
        local_tensor.dtype = shards[1].dtype
        local_tensor.device = torch.device("cuda")
        local_tensor.detach.return_value = local_tensor
        local_tensor.narrow.return_value.contiguous.return_value = MagicMock()

        nccl_calls = []

        def mock_all_gather_object(output_list, obj, *, group):
            meta = (tuple(shards[0].shape), str(shards[0].dtype))
            for i in range(len(output_list)):
                output_list[i] = meta if i < source_tp else None

        def mock_batch(ops):
            nccl_calls.append("batch_isend_irecv")
            return [MagicMock() for _ in ops]

        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__enter__ = MagicMock(return_value=None)
        mock_stream_ctx.__exit__ = MagicMock(return_value=False)

        with patch(
            "torch.distributed.all_gather_object", side_effect=mock_all_gather_object
        ):
            with patch("torch.distributed.batch_isend_irecv", side_effect=mock_batch):
                with patch("torch.distributed.P2POp", side_effect=lambda *a, **kw: MagicMock()):
                    with patch("torch.cuda.stream", return_value=mock_stream_ctx):
                        transport.stage_target_shard(
                            "test_param", local_tensor, rule, source_tp, target_tp, 1
                        )

        self.assertIn(
            "batch_isend_irecv", nccl_calls, "expansion must use NCCL batch_isend_irecv"
        )

    def test_shrink_replicated_validates_on_cpu(self):
        """Replicated shard mismatch detected on CPU without GPU torch.equal."""
        from unittest.mock import patch

        shard = torch.ones(3, 4, dtype=torch.float32)
        bad_shard = shard.clone()
        bad_shard[0, 0] = 99.0
        rule = SplitRule("replicated")
        source_tp, target_tp = 4, 1

        transport = self._make_nccl_transport_mock(0, 4)

        shards_to_broadcast = [shard, bad_shard, shard, shard]

        def mock_broadcast(tensor, *, src, group):
            tensor.copy_(shards_to_broadcast[src])

        def mock_all_gather_object(output_list, obj, *, group):
            meta = (tuple(shard.shape), str(shard.dtype))
            for i in range(len(output_list)):
                output_list[i] = meta if i < source_tp else None

        with patch("torch.distributed.broadcast", side_effect=mock_broadcast):
            with patch(
                "torch.distributed.all_gather_object",
                side_effect=mock_all_gather_object,
            ):
                with self.assertRaisesRegex(
                    Exception, "replicated source shards differ"
                ):
                    transport.stage_target_shard(
                        "test_param", shard, rule, source_tp, target_tp, 0
                    )

    def test_copy_source_shard_to_tp1_cpu_helper_column_fused(self):
        """Direct test of the CPU assembly helper for column_fused rule."""
        full = torch.arange(16 * 3, dtype=torch.float32).reshape(16, 3)
        rule = SplitRule("column_fused", 0, (8, 4, 4))
        source_tp = 4
        shards = _shard(full, rule, source_tp)

        target = _allocate_tp1_target(
            tuple(shards[0].shape), shards[0].dtype, "cpu", rule, source_tp
        )
        for src_rank, src_shard in enumerate(shards):
            _copy_source_shard_to_tp1_cpu(target, src_shard, rule, source_tp, src_rank)

        self.assertTrue(torch.equal(target, full))

    def test_copy_source_shard_to_tp1_cpu_helper_row(self):
        """Direct test of the CPU assembly helper for row rule."""
        full = torch.arange(3 * 16, dtype=torch.float32).reshape(3, 16)
        rule = SplitRule("row", 1)
        source_tp = 4
        shards = _shard(full, rule, source_tp)

        target = _allocate_tp1_target(
            tuple(shards[0].shape), shards[0].dtype, "cpu", rule, source_tp
        )
        for src_rank, src_shard in enumerate(shards):
            _copy_source_shard_to_tp1_cpu(target, src_shard, rule, source_tp, src_rank)

        self.assertTrue(torch.equal(target, full))


if __name__ == "__main__":
    unittest.main(verbosity=3)
