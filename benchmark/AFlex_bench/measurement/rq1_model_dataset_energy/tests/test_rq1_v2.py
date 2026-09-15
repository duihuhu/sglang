import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from generate_workloads_v2 import generate
from migrate_rq1_v2_qps_1_8_counts import (
    MIGRATION_ID,
    migrate,
)
from rq1lib_v2 import (
    QPS_ORDER,
    calculate_load_metrics,
    canonical_qps,
    classify_failure,
    expand,
    initial_progress,
    inventory_row,
    load,
    result_status,
    stable_id,
    workload_filename,
)


def load_runner_module():
    spec = importlib.util.spec_from_file_location("run_rq1_v2_under_test", ROOT / "scripts/run_rq1_v2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_trace(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_01_full_matrix_size_unique_ids_and_float_qps_stability():
    queue = expand("node1")
    assert len(queue) == 6 * 3 * 6 * 4 * 3 == 1296
    assert len({item["run_id"] for item in queue}) == 1296
    assert [canonical_qps(x) for x in QPS_ORDER] == ["1", "2", "4", "8"]
    for value in QPS_ORDER:
        args = ("formal_v2", "m", "native", "w")
        assert stable_id(*args, value, 1) == stable_id(*args, str(value), 1)
        if float(value).is_integer():
            assert stable_id(*args, value, 1) == stable_id(*args, int(value), 1)


def test_02_generated_trace_contract_and_sha256(tmp_path):
    result = generate(tmp_path)
    assert result["files"] == 72
    assert result["request_counts"] == {
        "1": 32,
        "2": 32,
        "4": 64,
        "8": 64,
    }
    index = load(tmp_path / "index.json")
    assert index["frozen"] and len(index["entries"]) == 72
    grouped = {}
    for entry in index["entries"]:
        path = tmp_path / entry["path"]
        rows = read_trace(path)
        expected_count = 32 if entry["qps"] in (1, 2) else 64
        assert len(rows) == expected_count
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        arrivals = [row["arrival_time_s"] for row in rows]
        assert arrivals[0] == 0
        assert arrivals[-1] == pytest.approx(
            (expected_count - 1) / entry["qps"], abs=1e-12
        )
        assert (expected_count - 1) / (
            arrivals[-1] - arrivals[0]
        ) == pytest.approx(entry["qps"])
        grouped.setdefault((entry["qps"], entry["repeat"]), []).append(arrivals)
    assert all(len(traces) == 6 and all(trace == traces[0] for trace in traces[1:]) for traces in grouped.values())
    for qps in QPS_ORDER:
        assert grouped[(qps, 1)][0] != grouped[(qps, 2)][0]
        assert grouped[(qps, 2)][0] != grouped[(qps, 3)][0]


def test_03_load_metric_formulas_and_percentiles():
    rows = [
        {"scheduled_arrival_s": 0, "sent_offset_s": 0.1, "completed_offset_s": 1.0, "arrival_lag_ms": 100, "success": True},
        {"scheduled_arrival_s": 1, "sent_offset_s": 1.2, "completed_offset_s": 2.0, "arrival_lag_ms": 200, "success": True},
        {"scheduled_arrival_s": 2, "sent_offset_s": 2.5, "completed_offset_s": 4.0, "arrival_lag_ms": 500, "success": False},
    ]
    got = calculate_load_metrics(rows, 1)
    assert got["target_qps"] == got["offered_qps"] == got["realized_offered_qps"] == 1
    assert got["send_qps"] == pytest.approx(2 / 2.4)
    assert got["completion_qps"] == got["achieved_qps"] == pytest.approx(2 / 3.9)
    assert got["drain_time_s"] == 2
    assert got["drain_after_last_send_s"] == 1.5
    assert got["arrival_lag_p90_ms"] == pytest.approx(440)
    assert got["arrival_lag_p99_ms"] == pytest.approx(494)


def test_04_canary_contract():
    queue = expand("node1", "canary_v2")
    assert len(queue) == 18
    assert {(item["workload"], item["qps"], item["repeat"]) for item in queue} == {("balanced_mpmd", 1.0, 1)}
    assert {item["point"]["request_timeout_s"] for item in queue} == {180}


def test_05_low_qps_sla_failure_never_skips_higher_qps():
    queue = expand("node1", "formal_v2", ("dense_llama3_1_8b",), ("native",), ("qa_lpld",), (), (1,))
    progress = initial_progress("node1")
    low = queue[0]
    progress["runs"][low["run_id"]] = dict(inventory_row(low), status="valid", sla_pass=False)
    assert [item["qps"] for item in queue[1:]] == [2.0, 4.0, 8.0]
    assert all(progress["runs"].get(item["run_id"], {"status": "pending"})["status"] == "pending" for item in queue[1:])
    source = (ROOT / "scripts/run_rq1_v2.py").read_text()
    assert "saturated(" not in source and "skipped_saturated" not in source


def test_06_loadgen_invalid_and_failure_classification():
    assert result_status({"status": "complete", "loadgen_healthy": False}) == "loadgen_invalid"
    assert result_status({"status": "complete", "loadgen_healthy": True}) == "valid"
    assert result_status({"status": "partial", "loadgen_healthy": True}) == "valid"
    assert classify_failure({"status": "failed", "failure_stage": "deployment", "error": "container stopped"}) == "environment_failure"
    assert classify_failure({"status": "failed", "failure_stage": "request", "error": "model kernel error"}) == "model_failure"


def test_07_static_path_and_protected_file_isolation():
    v2_files = [ROOT / "configs/matrix_v2.json", ROOT / "configs/workloads_v2.json", ROOT / "scripts/generate_workloads_v2.py", ROOT / "scripts/run_rq1_v2.py", ROOT / "scripts/report_rq1_v2.py", ROOT / "scripts/rq1lib_v2.py"]
    combined = "\n".join(path.read_text() for path in v2_files)
    assert "results/default" not in combined
    assert "data/workloads/" not in combined
    assert "data/workloads_v2" in combined and "results/v2" in combined
    assert "afd_ipc_pybind.cpp" not in combined and "cuda_ipc/conn.py" not in combined and "test_cuda_ipc_protocol.py" not in combined


def test_08_default_dry_run_inventory_is_1296_and_v2_only(tmp_path):
    result_dir = tmp_path / "v2"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/run_rq1_v2.py"), "--node", "node1", "--results", str(result_dir)], text=True, capture_output=True, check=True)
    output = json.loads(result.stdout)
    progress = load(result_dir / "progress.json")
    assert output["mode"] == "dry-run" and output["phase"] == "formal_v2" and output["selected"] == 1296
    assert len(progress["runs"]) == 1296 and {row["status"] for row in progress["runs"].values()} == {"pending"}


def test_09_formal_execute_requires_exact_canary_gate(monkeypatch, tmp_path):
    runner = load_runner_module()
    monkeypatch.setattr(sys, "argv", ["run_rq1_v2.py", "--node", "node1", "--execute", "--results", str(tmp_path)])
    with pytest.raises(RuntimeError, match="18/18"):
        runner.main()


def test_10_external_gpu_conflict_pauses_without_cleanup():
    runner = load_runner_module()
    assert runner.external_gpu_resource_conflict({"status": "failed", "error": "GPU ownership conflict with another process"})
    assert not runner.external_gpu_resource_conflict({"status": "failed", "error": "CUDA out of memory OOM"})
    source = (ROOT / "scripts/run_rq1_v2.py").read_text().lower()
    assert "docker stop" not in source and "docker rm" not in source and "kill -9" not in source


def test_11_environment_failure_pauses_queue_for_safe_retry(
    monkeypatch, tmp_path, capsys
):
    runner = load_runner_module()
    queue = expand(
        "node1",
        "formal_v2",
        ("dense_llama3_1_8b",),
        ("native",),
        ("qa_lpld",),
        (1.0,),
        (1, 2),
    )
    models, workloads, matrix, cluster = runner.configs()
    calls = []
    monkeypatch.setattr(
        runner,
        "canary_gate",
        lambda *args, **kwargs: {
            "open": True,
            "valid": 18,
            "expected": 18,
            "observed": 18,
        },
    )
    monkeypatch.setattr(runner, "expand", lambda *args, **kwargs: queue)
    monkeypatch.setattr(
        runner, "configs", lambda: (models, workloads, matrix, cluster)
    )
    model_probes = []
    monkeypatch.setattr(
        runner, "verify_model", lambda *args: (model_probes.append(args) or True)
    )
    monkeypatch.setattr(runner, "RemoteExecutor", lambda *args: object())

    def execute(*args):
        calls.append(args)
        return {
            "status": "failed",
            "failure_stage": "deployment",
            "error": "container stopped during startup",
        }

    monkeypatch.setattr(runner, "execute_item", execute)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_rq1_v2.py",
            "--node",
            "node1",
            "--phase",
            "formal_v2",
            "--execute",
            "--retry-failed",
            "--results",
            str(tmp_path),
        ],
    )
    runner.main()
    output = json.loads(capsys.readouterr().out)
    progress = load(tmp_path / "progress.json")
    first = progress["runs"][queue[0]["run_id"]]
    assert len(calls) == 1
    assert output["paused"] is True
    assert "container stopped" in output["reason"]
    assert first["status"] == "pending"
    assert first["attempts"][-1]["failure_class"] == "environment_failure"
    assert queue[1]["run_id"] not in progress["runs"]
    assert len(model_probes) == 1


def test_12_qps_matrix_migration_resets_active_and_excludes_removed(tmp_path):
    progress = initial_progress("node1")
    active = expand(
        "node1",
        "formal_v2",
        ("dense_llama3_1_8b",),
        ("native",),
        ("qa_lpld",),
        (1.0,),
        (1,),
    )[0]
    removed = dict(active)
    removed.update(
        qps=0.5,
        run_id=stable_id(
            "formal_v2",
            active["model_id"],
            active["architecture"],
            active["workload"],
            0.5,
            active["repeat"],
        ),
    )
    old_canary = dict(active)
    old_canary.update(
        phase="canary_v2",
        qps=0.25,
        run_id=stable_id(
            "canary_v2",
            active["model_id"],
            active["architecture"],
            "balanced_mpmd",
            0.25,
            1,
        ),
        workload="balanced_mpmd",
    )
    progress["runs"][active["run_id"]] = {
        **inventory_row(active),
        "status": "valid",
        "artifact": "active-artifact",
        "metrics": {"send_qps": 1.0},
        "finished_at": "old",
    }
    progress["runs"][removed["run_id"]] = {
        **inventory_row(removed),
        "status": "valid",
        "artifact": "removed-artifact",
    }
    progress["runs"][old_canary["run_id"]] = {
        **inventory_row(old_canary),
        "status": "valid",
        "artifact": "old-canary-artifact",
    }
    path = tmp_path / "progress.json"
    path.write_text(json.dumps(progress))

    result = migrate(path)
    saved = load(path)
    assert result == {
        "status": "applied",
        "reset": 1,
        "reset_by_status": {"valid": 1},
        "excluded": 1,
        "canaries_superseded": 1,
    }
    reset = saved["runs"][active["run_id"]]
    assert reset["status"] == "pending"
    assert "artifact" not in reset and "metrics" not in reset
    assert reset["superseded_results"][-1]["artifact"] == "active-artifact"
    assert saved["runs"][removed["run_id"]]["status"] == "superseded"
    assert saved["runs"][old_canary["run_id"]]["status"] == "superseded"
    assert saved["migrations"][-1]["id"] == MIGRATION_ID
    assert migrate(path) == {
        "status": "already_applied",
        "reset": 0,
        "excluded": 0,
    }
