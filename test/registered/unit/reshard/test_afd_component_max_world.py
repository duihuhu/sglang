# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Directly executable four-process Gloo integration for component fan-out."""
import datetime
import os
import queue
import socket
import time
import traceback
import unittest
from multiprocessing import get_context

import torch.distributed as dist

from sglang.srt.reshard.afd_component_standby import AFDComponentRankLifecycle
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="stage-a-cpu-only")


class _FakeStager:
    def __init__(self, rank, group, trace):
        self.rank, self.group, self.trace = rank, group, trace

    def _run(self, action, command):
        gathered = [None] * dist.get_world_size(self.group)
        item = (self.rank, command["operation"], command["epoch"], action)
        dist.all_gather_object(gathered, item, group=self.group)
        assert [x[1:] for x in gathered] == [item[1:]] * 4
        self.trace.append(item[1:])

    def prepare(self, command): self._run("prepare", command)
    def activate(self, command): self._run("activate", command)
    def retire(self, command): self._run("retire", command)


def _worker(rank, port, output):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=4,
                            timeout=datetime.timedelta(seconds=20))
    # Same creation order as production; neither group belongs to serving TP.
    control = dist.new_group(ranks=list(range(4)), backend="gloo")
    data = dist.new_group(ranks=list(range(4)), backend="gloo")
    lifecycle = AFDComponentRankLifecycle(rank, 2, 4, "attn")
    trace = []
    stager = _FakeStager(rank, data, trace)
    commands = [
        {"operation": "grow", "epoch": 1, "component": "attn", "action": "prepare", "target_tp": 4},
        {"operation": "grow", "epoch": 1, "component": "attn", "action": "activate", "target_tp": 4},
        {"operation": "shrink", "epoch": 2, "component": "attn", "action": "prepare", "target_tp": 1},
        {"operation": "shrink", "epoch": 2, "component": "attn", "action": "activate", "target_tp": 1},
        {"operation": "shrink", "epoch": 2, "component": "attn", "action": "retire", "target_tp": 1},
    ]
    for source in commands:
        objects = [source if rank == 0 else None]
        dist.broadcast_object_list(objects, src=0, group=control)
        command = objects[0]
        action = command["action"]
        if action == "prepare": stager.prepare(command)
        elif action == "activate": stager.activate(command)
        else: stager.retire(command)
        lifecycle.apply_safe_point(command)
    output.put((rank, trace, lifecycle.active_tp, lifecycle.is_standby))
    dist.destroy_process_group()


def _parallel_state_worker(rank, port, output):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        SGLANG_AFD_COMPONENT_MAX_TP="4",
        SGLANG_AFD_COMPONENT_ACTIVE_TP="2",
        SGLANG_USE_MESSAGE_QUEUE_BROADCASTER="false",
    )
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=4,
        timeout=datetime.timedelta(seconds=30),
    )
    from types import SimpleNamespace

    from sglang.srt.distributed import parallel_state

    parallel_state._WORLD = SimpleNamespace(
        local_rank=rank,
        device_group=dist.group.WORLD,
    )
    try:
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=1,
            backend="gloo",
        )
        groups = {
            "tp": parallel_state.get_tp_group(),
            "attn_cp": parallel_state.get_attn_cp_group(),
            "attn_tp": parallel_state.get_attn_tp_group(),
            "moe_dp": parallel_state.get_moe_dp_group(),
            "moe_ep": parallel_state.get_moe_ep_group(),
            "moe_tp": parallel_state.get_moe_tp_group(),
            "pp": parallel_state.get_pp_group(),
        }
        output.put(
            (
                rank,
                {
                    name: (group.world_size, tuple(group.ranks))
                    for name, group in groups.items()
                },
                all(group.cpu_group is not None for group in groups.values()),
            )
        )
    finally:
        parallel_state.destroy_model_parallel()
        parallel_state._WORLD = None
        dist.destroy_process_group()


def _quiesce_consensus_worker(
    rank, port, output, status, follower_blocked, release_follower
):
    try:
        _quiesce_consensus_worker_impl(
            rank, port, output, status, follower_blocked, release_follower
        )
    except BaseException as exc:
        status.put(("error", rank, repr(exc), traceback.format_exc()))
        raise


def _quiesce_consensus_worker_impl(
    rank, port, output, status, follower_blocked, release_follower
):
    import time

    import torch

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group(
        "gloo", rank=rank, world_size=4,
        timeout=datetime.timedelta(seconds=20),
    )
    active = dist.new_group(ranks=[0, 1], backend="gloo")
    control = dist.new_group(ranks=list(range(4)), backend="gloo")
    dist.barrier()
    status.put(("dist-ready", rank))
    command = {
        "operation": "quiesce-op", "operation_id": "quiesce-op",
        "epoch": 0, "component": "attn", "action": "quiesce", "target_tp": 2,
    }

    # Production loop order begins with an active-only pre-poll fence sync.
    # With no fence, all active ranks complete this None generation first.
    if rank < 2:
        pre_sync = [None]
        dist.broadcast_object_list(pre_sync, src=0, group=active)
        assert pre_sync[0] is None

    # Followers then enter ordinary max-world directly and match rank 0's
    # PREPARE command. safe_point must not insert another active-TP collective.
    prepare = {
        "operation": "prepare-op", "epoch": 0, "component": "attn",
        "action": "prepare", "target_tp": 2,
    }
    ordinary = [prepare if rank == 0 else None]
    dist.broadcast_object_list(ordinary, src=0, group=control)
    assert ordinary[0]["action"] == "prepare"

    if rank == 0:
        from types import SimpleNamespace

        from sglang.srt.layers.afd_reshard_comm import AFDPairDrainProtocol
        from sglang.srt.layers.afd_type import AFDPerspective
        from sglang.srt.reshard.afd_component_adapter import (
            AFDComponentSchedulerRuntime,
        )

        args = SimpleNamespace(
            afd_perspective=AFDPerspective.AFD_PERSPECTIVE_ATTN,
            afd_component_max_tp=4,
            afd_micro_batch=1,
            disable_cuda_graph=True,
        )
        runner = SimpleNamespace(
            model=torch.nn.Linear(1, 1, bias=False),
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    model_type="qwen3", num_experts=None, num_local_experts=None
                )
            ),
            server_args=args,
            pp_size=1,
            tp_size=2,
        )

        class RankZeroScheduler:
            tp_rank = 0
            server_args = args
            tp_worker = SimpleNamespace(model_runner=runner)
            _afd_reshard_drain = AFDPairDrainProtocol()
            _afd_component_pending_fence_payload = None
            _afd_component_fence_payload = None
            _afd_component_fence_installed = False
            _afd_component_fence_synced = False
            _afd_component_quiesce_world_done = False
            _afd_component_quiesce_world_operation = None

            def afd_component_request_quiesce(self, payload):
                self._afd_component_pending_fence_payload = dict(payload)
                self._afd_component_fence_installed = False
                self._afd_component_fence_synced = False

            def afd_reshard_drain_state(self):
                return self._afd_reshard_drain

        runtime = AFDComponentSchedulerRuntime(RankZeroScheduler())
        request = {
            **command,
            "target_attn_tp": 2,
            "target_ffn_tp": 2,
        }
    # Complete imports and rank-0 fixture construction before beginning the
    # measured follower-forward phase. This separates spawn cost from the 5s
    # forward-delay assertion on every environment.
    dist.barrier()
    status.put(("rank-ready", rank))
    # First quiesce iteration also completes a pre-poll None sync. Rank 0's
    # handler runs only afterwards and marks its fence unsynced for this round.
    if rank < 2:
        first_quiesce_sync = [None]
        dist.broadcast_object_list(first_quiesce_sync, src=0, group=active)
        assert first_quiesce_sync[0] is None
    if rank == 1:
        follower_blocked.set()
        if not release_follower.wait(5):
            raise TimeoutError("test did not release delayed active follower")
    elif rank == 0:
        if not follower_blocked.wait(5):
            raise TimeoutError("active follower did not enter old forward")
        started = time.perf_counter()
        response = runtime.handle(request)
        status.put(
            (
                "initial",
                time.perf_counter() - started,
                request["operation"],
                response,
                runtime.scheduler._afd_component_fence_installed,
                runtime.scheduler._afd_component_pending_fence_payload is not None,
            )
        )

    # The first quiesce handler leaves fence_synced=False, so that iteration's
    # safe_point remains the ordinary max-world None generation. Rank 0 may wait
    # here for the delayed follower, but the handler has already returned.
    first_quiesce_safe_point = [None]
    dist.broadcast_object_list(first_quiesce_safe_point, src=0, group=control)
    assert first_quiesce_safe_point[0] is None

    if rank < 2:
        from sglang.srt.layers.afd_reshard_comm import AFDPairDrainProtocol

        # Next iteration pre-poll fence sync: all active ranks install the same
        # payload before any rank can enter ready consensus.
        objects = [command if rank == 0 else None]
        dist.broadcast_object_list(objects, src=0, group=active)
        installed = objects[0]["operation"] == "quiesce-op"
        # Only Attention rank 0 dispatches batch metadata. The follower's local
        # sequence intentionally remains zero and must not participate in a
        # min/max consistency check.
        drain = AFDPairDrainProtocol(dispatch_seq=31 if rank == 0 else 0)
        ready = torch.tensor([1], dtype=torch.int32)
        dist.all_reduce(ready, op=dist.ReduceOp.MIN, group=active)
        assert ready.item() == 1
        seal = [
            {
                "action": "seal_cut", "operation_id": "quiesce-op",
                "generation": 2, "watermark": drain.cut(),
            }
            if rank == 0 else None
        ]
        dist.broadcast_object_list(seal, src=0, group=active)
        assert set(seal[0]) == {
            "action", "operation_id", "generation", "watermark"
        }
        assert seal[0]["action"] == "seal_cut"
        assert seal[0]["operation_id"] == "quiesce-op"
        assert seal[0]["generation"] == 2
        local_cut = drain.install_external_cut(seal[0]["watermark"])
        assert local_cut == 31
        objects[0]["watermark"] = local_cut
    else:
        installed = False
        local_cut = None

    # Standby ranks have been waiting on exactly this second max-world
    # generation. Both active ranks enter only after follower safe-point release.
    objects = [command if rank == 0 else None]
    dist.broadcast_object_list(objects, src=0, group=control)
    observed_cut = local_cut if rank < 2 else objects[0].get("watermark")
    output.put(
        (rank, installed, ordinary[0]["action"], objects[0]["operation"],
         objects[0].get("watermark"), observed_cut)
    )
    dist.destroy_process_group()



def _active_control_isolation_worker(rank, port, output):
    import torch

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group(
        "gloo", rank=rank, world_size=4,
        timeout=datetime.timedelta(seconds=20),
    )
    # Production creation order: all max-world ranks construct every prefix.
    control_prefixes = {
        tp: dist.new_group(ranks=list(range(tp)), backend="gloo")
        for tp in range(1, 5)
    }
    serving_tp4 = dist.new_group(ranks=list(range(4)), backend="gloo")
    dist.barrier()

    # TP2 -> TP4 activation selects the pre-created TP4 control group. Its first
    # no-fence generation is independent from a serving request generation.
    control_none = [None]
    dist.broadcast_object_list(control_none, src=0, group=control_prefixes[4])
    serving_payload = [
        {"kind": "TokenizedOrAbort", "rid": "r0"} if rank == 0 else None
    ]
    dist.broadcast_object_list(serving_payload, src=0, group=serving_tp4)
    assert control_none[0] is None
    assert serving_payload[0]["kind"] == "TokenizedOrAbort"

    quiesce = {
        "operation_id": "after-grow", "operation": "after-grow",
        "epoch": 1, "component": "attn", "action": "quiesce", "target_tp": 4,
    }
    control_payload = [quiesce if rank == 0 else None]
    dist.broadcast_object_list(control_payload, src=0, group=control_prefixes[4])
    assert control_payload[0]["action"] == "quiesce"
    ready = torch.tensor([1], dtype=torch.int32)
    dist.all_reduce(ready, op=dist.ReduceOp.MIN, group=control_prefixes[4])

    # TP4 -> TP1 selects prefix-1. Retired ranks never enter this generation.
    retired_skipped = rank != 0
    if rank == 0:
        tp1_payload = [{"operation_id": "after-shrink", "action": "quiesce"}]
        dist.broadcast_object_list(tp1_payload, src=0, group=control_prefixes[1])
        assert tp1_payload[0]["operation_id"] == "after-shrink"
        retired_skipped = False
    dist.barrier()
    output.put((rank, serving_payload[0]["kind"], control_payload[0]["action"], retired_skipped))
    dist.destroy_process_group()


class _BoundaryStager:
    def __init__(self, trace):
        self.trace = trace

    def start_prepare(self, request):
        self.trace.append((request["operation"], "prepare"))

    def activate(self, request):
        self.trace.append((request["operation"], "activate"))

    def prepare_status(self, operation):
        return {"state": "READY"}

    def cancel_prepare(self, reason):
        pass

    def abort(self, reason):
        pass


class _BoundaryScheduler:
    pass


def _continuous_transition_worker(rank, port, output, status=None):
    """Model production standby/active event-loop phase ordering."""
    from types import MethodType, SimpleNamespace

    from sglang.srt.managers.scheduler_afd_mixin import SchedulerAFDMixin

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group(
        "gloo", rank=rank, world_size=4,
        timeout=datetime.timedelta(seconds=20),
    )
    control = dist.new_group(ranks=list(range(4)), backend="gloo")
    active_groups = {
        tp: dist.new_group(ranks=list(range(tp)), backend="gloo")
        for tp in range(1, 5)
    }
    trace = []

    def phase(name):
        if status is not None:
            status.put((rank, name))

    scheduler = _BoundaryScheduler()
    scheduler.tp_rank = rank
    scheduler.server_args = SimpleNamespace(
        afd_component_max_tp=4,
        enable_afd_component_reshard_participant=True,
    )
    scheduler._afd_component_control_group = control
    scheduler._afd_component_active_control_groups = active_groups
    scheduler._afd_component_active_control_group = active_groups[2]
    scheduler._afd_component_active_control_generation = 7
    scheduler._afd_component_lifecycle = AFDComponentRankLifecycle(
        rank, 2, 4, "attn"
    )
    # Production standby fast path has not called afd_init_state(). In
    # particular, joining ranks must reach QUIESCE without this attribute.
    scheduler._afd_component_pending_fence_payload = None
    scheduler._afd_component_fence_payload = None
    scheduler._afd_component_fence_installed = False
    scheduler._afd_component_fence_synced = False
    scheduler._afd_component_fence_sealed = False
    scheduler._afd_component_quiesce_world_done = False
    scheduler._afd_component_quiesce_world_operation = None
    scheduler._afd_component_fence_monotonic = None
    scheduler._afd_component_transfer_log_monotonic = 0.0
    scheduler._afd_component_stop_active_loop = False
    scheduler.is_afd_component_standby_rank = rank >= 2
    stager = _BoundaryStager(trace)
    scheduler.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            _afd_component_stager=stager,
            is_afd_component_standby_rank=rank >= 2,
        ),
        is_afd_component_standby_rank=rank >= 2,
    )
    for name in (
        "afd_component_apply_safe_point",
        "afd_component_fanout_command",
        "afd_component_receive_world_command",
        "afd_component_complete_transition_boundary",
        "afd_component_resume_at_transition_boundary",
        "afd_component_complete_serving_readiness",
        "afd_component_begin_active_iteration",
        "afd_component_clear_quiesce",
        "afd_init_state",
        "afd_reshard_drain_state",
    ):
        setattr(scheduler, name, MethodType(getattr(SchedulerAFDMixin, name), scheduler))

    def command(operation, epoch, action, target_tp):
        return {
            "operation": operation, "operation_id": operation,
            "epoch": epoch, "component": "attn", "action": action,
            "target_tp": target_tp,
        }

    def world_phase(payload):
        if rank == 0:
            scheduler.afd_component_fanout_command(payload)
        else:
            received = scheduler.afd_component_receive_world_command()
            scheduler.afd_component_apply_safe_point(received)

    # TP2 active + two standby ranks consume PREPARE/QUIESCE/ACTIVATE in one
    # world order. QUIESCE carries the authoritative Attention cut and lazily
    # creates the real-shape standby ledger before the transition boundary.
    phase("grow_prepare_enter")
    world_phase(command("grow", 0, "prepare", 4))
    phase("grow_prepare_exit")
    quiesce = command("grow", 0, "quiesce", 4)
    quiesce["watermark"] = 0
    world_phase(quiesce)
    drain = scheduler.afd_reshard_drain_state()
    if rank == 0:
        drain.ack("attn", 0)
        drain.ack("ffn", 0)
    phase("grow_activate_enter")
    world_phase(command("grow", 1, "activate", 4))
    phase("grow_activate_exit")
    boundary_ledger = scheduler.afd_reshard_drain_state()
    assert boundary_ledger.cut_watermark is None
    if rank >= 2:
        scheduler.afd_init_state()
        assert scheduler.afd_reshard_drain_state() is boundary_ledger
        assert boundary_ledger.cut_watermark is None
    scheduler._afd_component_active_control_group = active_groups[4]
    scheduler._afd_component_active_control_generation = 0

    resumed = rank >= 2

    # An immediate epoch2 PREPARE is rejected on rank 0 before touching the
    # max-world command group; followers therefore remain free to reach the
    # readiness boundary instead of waiting for a command generation.
    if rank == 0:
        from sglang.srt.reshard.afd_component_adapter import (
            AFDComponentSchedulerRuntime,
        )

        gate = object.__new__(AFDComponentSchedulerRuntime)
        gate.scheduler = scheduler
        pending = AFDComponentSchedulerRuntime.handle(
            gate,
            {
                **command("shrink", 1, "prepare", 1),
                "target_attn_tp": 1,
                "target_ffn_tp": 1,
            },
        )
        assert not pending["ok"] and pending["retryable"]
        assert trace == [("grow", "prepare"), ("grow", "activate")]

    # Old ranks wait here immediately after ACTIVATE. Joining ranks arrive only
    # after scheduler/communicator init. The helper consumes joining boundaries,
    # completes readiness, and returns all ranks into the same first request
    # receive without running a normal control preamble.
    phase("grow_ready_enter")
    scheduler.afd_component_begin_active_iteration()
    phase("grow_ready_exit")
    assert scheduler._afd_component_serving_ready
    assert scheduler._afd_component_transition_boundary is None

    # Model exactly one first data-plane generation after the barrier. If an old
    # rank had received during joining initialization this count would diverge.
    request_generation = 1
    generations = [None] * 4
    dist.all_gather_object(
        generations, request_generation, group=active_groups[4]
    )
    assert generations == [1, 1, 1, 1]
    phase("first_request_generation")

    # Immediate epoch2 PREPARE retry now reaches all four ranks safely.
    phase("shrink_prepare_enter")
    world_phase(command("shrink", 1, "prepare", 1))
    phase("shrink_prepare_exit")
    assert ("shrink", "prepare") in trace
    quiesce = command("shrink", 1, "quiesce", 1)
    quiesce["watermark"] = 0
    phase("shrink_quiesce_enter")
    world_phase(quiesce)
    phase("shrink_quiesce_exit")
    if rank == 0:
        drain = scheduler.afd_reshard_drain_state()
        drain.ack("attn", 0)
        drain.ack("ffn", 0)
    phase("shrink_activate_enter")
    world_phase(command("shrink", 2, "activate", 1))
    phase("shrink_activate_exit")
    # Production switches this reference in the stager's group-refresh callback
    # before ACTIVATE's final max-world barrier returns. Model that callback here
    # so only surviving rank 0 enters the TP1 readiness generation.
    scheduler._afd_component_active_control_group = active_groups[1]
    scheduler._afd_component_active_control_generation = 0

    # The surviving TP1 rank completes the shrink readiness generation before
    # any next max-world command. Retired ranks remain in standby world receive.
    if rank == 0:
        scheduler.afd_component_begin_active_iteration()
        assert scheduler._afd_component_serving_ready
    # Shrunk ranks have left active loops and are conceptually back in standby;
    # they still consume the next max-world PREPARE, proving operation 3 starts.
    phase("third_prepare_enter")
    world_phase(command("third", 2, "prepare", 4))
    phase("third_prepare_exit")
    output.put(
        (
            rank, tuple(trace), scheduler._afd_component_lifecycle.active_tp,
            scheduler._afd_component_lifecycle.is_standby,
            scheduler._afd_component_active_control_generation,
            scheduler._afd_reshard_drain.cut_watermark, resumed,
        )
    )
    dist.destroy_process_group()


class TestAFDComponentMaxWorld(CustomTestCase):
    def test_drain_getter_lazily_creates_and_init_preserves_ledger(self):
        from types import MethodType, SimpleNamespace

        from sglang.srt.managers.scheduler_afd_mixin import SchedulerAFDMixin

        scheduler = _BoundaryScheduler()
        scheduler.server_args = SimpleNamespace(
            enable_afd_component_reshard_participant=True
        )
        scheduler.afd_reshard_drain_state = MethodType(
            SchedulerAFDMixin.afd_reshard_drain_state, scheduler
        )
        scheduler.afd_init_state = MethodType(SchedulerAFDMixin.afd_init_state, scheduler)

        self.assertFalse(hasattr(scheduler, "_afd_reshard_drain"))
        drain = scheduler.afd_reshard_drain_state()
        self.assertIs(scheduler.afd_reshard_drain_state(), drain)
        scheduler.afd_init_state()
        self.assertIs(scheduler.afd_reshard_drain_state(), drain)

    def test_follower_boundary_requires_installed_cut(self):
        from sglang.srt.layers.afd_reshard_comm import AFDPairDrainProtocol

        drain = AFDPairDrainProtocol()
        with self.assertRaisesRegex(RuntimeError, "without an installed cut"):
            drain.reopen_at_transition_boundary()

    def test_continuous_transition_boundary_grow_shrink_then_prepare(self):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        ctx = get_context("spawn")
        output = ctx.Queue()
        status = ctx.Queue()
        processes = [
            ctx.Process(
                target=_continuous_transition_worker,
                args=(rank, port, output, status),
            )
            for rank in range(4)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(30)
            if process.is_alive():
                phases = []
                while True:
                    try:
                        phases.append(status.get_nowait())
                    except queue.Empty:
                        break
                latest = {}
                for worker_rank, worker_phase in phases:
                    latest[worker_rank] = worker_phase
                self.fail(
                    "continuous transition hung; latest worker phases="
                    f"{latest}; all phases={phases}"
                )
            self.assertEqual(process.exitcode, 0)
        results = sorted(output.get(timeout=2) for _ in range(4))
        expected_trace = (
            ("grow", "prepare"), ("grow", "activate"),
            ("shrink", "prepare"), ("shrink", "activate"),
            ("third", "prepare"),
        )
        self.assertTrue(all(item[1] == expected_trace for item in results))
        self.assertEqual([item[2] for item in results], [1] * 4)
        self.assertEqual([item[3] for item in results], [False, True, True, True])
        self.assertEqual([item[4] for item in results], [1, 0, 0, 0])
        self.assertEqual([item[5] for item in results], [None] * 4)
        self.assertEqual([item[6] for item in results], [False, False, True, True])

    def test_active_control_isolated_across_grow_and_shrink(self):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        ctx = get_context("spawn")
        output = ctx.Queue()
        processes = [
            ctx.Process(target=_active_control_isolation_worker, args=(rank, port, output))
            for rank in range(4)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(30)
            self.assertFalse(process.is_alive(), "active-control isolation hung")
            self.assertEqual(process.exitcode, 0)
        results = sorted(output.get(timeout=2) for _ in range(4))
        self.assertEqual([item[1] for item in results], ["TokenizedOrAbort"] * 4)
        self.assertEqual([item[2] for item in results], ["quiesce"] * 4)
        self.assertEqual([item[3] for item in results], [False, True, True, True])

    def test_quiesce_active_consensus_precedes_max_world(self):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        ctx = get_context("spawn")
        output = ctx.Queue()
        status = ctx.Queue()
        follower_blocked = ctx.Event()
        release_follower = ctx.Event()
        processes = [
            ctx.Process(
                target=_quiesce_consensus_worker,
                args=(
                    rank, port, output, status, follower_blocked, release_follower
                ),
            )
            for rank in range(4)
        ]
        for process in processes:
            process.start()

        # Spawn/import/process-group setup and rank-0 fixture imports have a
        # separate budget. Start the forward deadline only after the second
        # all-rank barrier reports every worker phase-ready.
        ready_ranks = set()
        initial = None
        init_deadline = time.monotonic() + 30
        while len(ready_ranks) < 4:
            remaining = init_deadline - time.monotonic()
            self.assertGreater(remaining, 0, "four-rank initialization timed out")
            try:
                message = status.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                failed = [
                    (index, process.exitcode)
                    for index, process in enumerate(processes)
                    if process.exitcode not in (None, 0)
                ]
                self.assertFalse(failed, f"worker exited during init: {failed}")
                continue
            if message[0] == "error":
                self.fail(
                    f"rank {message[1]} failed during init: {message[2]}\n"
                    f"{message[3]}"
                )
            if message[0] == "rank-ready":
                ready_ranks.add(message[1])
            elif message[0] == "initial":
                initial = message

        forward_deadline = time.monotonic() + 5
        while not follower_blocked.is_set():
            remaining = forward_deadline - time.monotonic()
            self.assertGreater(
                remaining, 0, "active follower did not enter old forward"
            )
            try:
                message = status.get(timeout=min(remaining, 0.25))
            except queue.Empty:
                failed = [
                    (index, process.exitcode)
                    for index, process in enumerate(processes)
                    if process.exitcode not in (None, 0)
                ]
                self.assertFalse(failed, f"worker exited before forward: {failed}")
                continue
            if message[0] == "error":
                self.fail(
                    f"rank {message[1]} failed before forward: {message[2]}\n"
                    f"{message[3]}"
                )
            if message[0] == "initial":
                initial = message

        if initial is None:
            initial = status.get(timeout=5)
        if initial[0] == "error":
            self.fail(
                f"rank {initial[1]} failed in handler: {initial[2]}\n{initial[3]}"
            )
        phase, elapsed, operation, response, installed, pending = initial
        self.assertEqual((phase, operation), ("initial", "quiesce-op"))
        self.assertLess(elapsed, 0.1)
        self.assertFalse(response["ok"])
        self.assertTrue(response["retryable"])
        self.assertFalse(installed)
        self.assertTrue(pending)
        release_follower.set()
        for process in processes:
            process.join(30)
            self.assertFalse(process.is_alive(), "quiesce consensus hung")
            self.assertEqual(process.exitcode, 0)
        results = sorted(output.get(timeout=2) for _ in range(4))
        self.assertEqual([item[1] for item in results], [True, True, False, False])
        self.assertEqual([item[2] for item in results], ["prepare"] * 4)
        self.assertEqual([item[3] for item in results], ["quiesce-op"] * 4)
        # The sealed cut is part of the max-world QUIESCE command, so standby
        # ranks observe the same transition descriptor before a possible grow.
        # They do not participate in active cut consensus or mutate a serving
        # dispatch ledger; receiving the watermark here is intentional.
        self.assertEqual([item[4] for item in results], [31] * 4)
        # Both active Attention ledgers are sealed at rank 0's authoritative
        # cut, while both max-world standby ranks observe that same descriptor.
        self.assertEqual([item[5] for item in results], [31] * 4)

    def test_initialize_model_parallel_active_prefix_and_singletons(self):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        ctx = get_context("spawn")
        output = ctx.Queue()
        processes = [
            ctx.Process(target=_parallel_state_worker, args=(rank, port, output))
            for rank in range(4)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(45)
            self.assertFalse(process.is_alive(), "parallel group initialization hung")
            self.assertEqual(process.exitcode, 0)
        results = sorted(output.get(timeout=2) for _ in range(4))
        expected_tp = [
            (2, (0, 1)), (2, (0, 1)), (1, (2,)), (1, (3,))
        ]
        self.assertEqual([item[1]["tp"] for item in results], expected_tp)
        self.assertEqual([item[1]["attn_tp"] for item in results], expected_tp)
        self.assertEqual([item[1]["moe_tp"] for item in results], expected_tp)
        singleton_groups = ("attn_cp", "moe_dp", "moe_ep", "pp")
        for name in singleton_groups:
            self.assertEqual(
                [item[1][name] for item in results],
                [(1, (rank,)) for rank in range(4)],
                name,
            )
        self.assertTrue(all(item[2] for item in results))

    def test_bootstrap_expected_workers_uses_cp_one(self):
        # Mirrors CommonKVBootstrapServer._is_ready without opening a listener.
        from sglang.srt.disaggregation.common.conn import CommonKVBootstrapServer

        server = object.__new__(CommonKVBootstrapServer)
        server.dp_size = 1
        server.attn_cp_size = 1
        server.attn_tp_size = 2
        server.pp_size = 1
        server._registered_count = 2
        self.assertTrue(server._is_ready())
        server._registered_count = 1
        self.assertFalse(server._is_ready())

    def test_grow_and_shrink_all_ranks_in_same_order(self):
        sock = socket.socket(); sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]; sock.close()
        ctx = get_context("spawn"); output = ctx.Queue()
        processes = [ctx.Process(target=_worker, args=(rank, port, output)) for rank in range(4)]
        for process in processes: process.start()
        for process in processes:
            process.join(30)
            self.assertFalse(process.is_alive(), "collective hung")
            self.assertEqual(process.exitcode, 0)
        results = sorted(output.get(timeout=2) for _ in range(4))
        self.assertTrue(all(item[1] == results[0][1] for item in results))
        self.assertEqual([item[2] for item in results], [1] * 4)
        self.assertEqual([item[3] for item in results], [False, True, True, True])


if __name__ == "__main__":
    unittest.main(verbosity=2)
