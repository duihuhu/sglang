import hashlib, importlib.util, json, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT.parent / "benchmark" / "src"))
from rq1lib import *
from aflex_benchmark.deploy import build_plan


def load_runner_module():
    spec = importlib.util.spec_from_file_location(
        "run_rq1_under_test", ROOT / "scripts/run_rq1.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_matrix_and_stable_ids():
    q = expand("node1")
    assert len(q) == 6 * 3 * 6 * 4 * 3 == 1296
    assert len({x["run_id"] for x in q}) == 1296
    assert q[0]["run_id"] == stable_id(
        "formal",
        q[0]["model_id"],
        q[0]["architecture"],
        q[0]["workload"],
        q[0]["qps"],
        q[0]["repeat"],
    )


def test_canary_is_exactly_eighteen_and_gate():
    q = expand("node1", "canary")
    assert len(q) == 18
    p = initial_progress("node1")
    for x in q:
        p["runs"][x["run_id"]] = dict(inventory_row(x), status="valid")
    assert canary_gate(p) == {"open": True, "valid": 18, "expected": 18, "observed": 18}


def test_plans_use_eight_gpus_with_requested_split():
    models, workloads, matrix, cluster = configs()
    cluster = select_node(cluster, "node1")
    for architecture in ARCH_ORDER:
        point = make_point(
            next(iter(models)),
            architecture,
            matrix["architectures"][architecture],
            ["qa_lpld"],
            [2],
            "node1",
        )
        plan = build_plan(cluster, next(iter(models.values()))["path"], point)
        plan.validate()
        used = [(p.node, g) for p in plan.processes for g in p.gpus]
        assert len(used) == len(set(used)) == 8
        roles = {p.role: len(p.gpus) for p in plan.processes if p.gpus}
        assert roles == (
            {"native": 8}
            if architecture == "native"
            else {"P": 4, "D": 4} if architecture == "pd" else {"F": 4, "A": 4}
        )


def test_workload_index_is_frozen_and_hashed():
    index = load(ROOT / "data/workloads/index.json")
    assert index["frozen"]
    assert len(index["entries"]) == 24
    for e in index["entries"]:
        path = ROOT / "data/workloads" / e["path"]
        assert len(path.read_text().splitlines()) == 64
        assert hashlib.sha256(path.read_bytes()).hexdigest() == e["sha256"]
    long = [e for e in index["entries"] if e["name"] == "longcontext"]
    assert all((e["input_len"], e["output_len"]) == (16384, 256) for e in long)


def test_saturation_skips_only_higher_qps():
    item = next(x for x in expand("node1") if x["qps"] == 8)
    p = initial_progress("node1")
    rid = stable_id(
        "formal",
        item["model_id"],
        item["architecture"],
        item["workload"],
        4,
        item["repeat"],
    )
    p["runs"][rid] = {
        "run_id": rid,
        "status": "valid",
        "sla": {"achieved_qps_ratio": 0.5},
    }
    assert saturated(p, item, 0.9) == rid


def test_complete_canary_is_valid_even_when_sla_fails():
    runner = load_runner_module()
    summary = {"status": "complete", "successful_requests": 64, "total_requests": 64}
    sla_pass = False
    assert runner.run_status(summary, sla_pass) == "valid"


def test_dry_run_writes_inventory_without_execution():
    with tempfile.TemporaryDirectory() as d:
        cmd = [
            sys.executable,
            str(ROOT / "scripts/run_rq1.py"),
            "--node",
            "node1",
            "--model",
            "dense_llama3_1_8b",
            "--architecture",
            "native",
            "--workload",
            "qa_lpld",
            "--qps",
            "2",
            "--repeat",
            "1",
            "--results",
            d,
        ]
        result = subprocess.run(cmd, text=True, capture_output=True, check=True)
        assert "dry-run" in result.stdout
        p = load(Path(d) / "progress.json")
        assert len(p["runs"]) == 1
        assert next(iter(p["runs"].values()))["status"] == "pending"


def test_external_gpu_conflict_classifier_excludes_oom():
    runner = load_runner_module()
    conflicts = [
        "RuntimeError('planned GPU compute applications remain on node: GPU 0 busy: 123')",
        "startup failed: GPU ownership conflict with another process",
        "startup GPU ownership barrier failed: GPU 3 nvml_pid=123 owned=",
        "CUDA device or resource busy on GPU 2",
    ]
    assert all(
        runner.external_gpu_resource_conflict({"status": "failed", "error": x})
        for x in conflicts
    )
    assert not runner.external_gpu_resource_conflict(
        {"status": "complete", "error": conflicts[0]}
    )
    assert not runner.external_gpu_resource_conflict(
        {"status": "failed", "error": "CUDA out of memory (OOM) on GPU 0"}
    )


def test_formal_external_gpu_conflict_pauses_entire_queue(
    monkeypatch, tmp_path, capsys
):
    runner = load_runner_module()
    queue = expand(
        "node1",
        "formal",
        ("dense_llama3_1_8b",),
        ("native",),
        ("qa_lpld",),
        (2,),
        (1, 2),
    )
    progress = initial_progress("node1")
    for item in queue:
        progress["runs"][item["run_id"]] = inventory_row(item)
    first = progress["runs"][queue[0]["run_id"]]
    first.update(
        status="failed",
        finished_at="stale",
        sla_pass=False,
        sla={"bad": 1},
        metrics={"bad": 1},
        artifact="stale",
    )
    atomic_json(tmp_path / "progress.json", progress)
    models, workloads, matrix, cluster = configs()
    calls = []
    monkeypatch.setattr(runner, "configs", lambda: (models, workloads, matrix, cluster))
    monkeypatch.setattr(runner, "expand", lambda *args, **kwargs: queue)
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
    monkeypatch.setattr(runner, "verify_model", lambda *args: True)
    monkeypatch.setattr(runner, "RemoteExecutor", lambda *args: object())

    def execute(*args):
        calls.append(args)
        return {
            "status": "failed",
            "error": "planned GPU compute applications remain on host: GPU 0 busy: 123",
        }

    monkeypatch.setattr(runner, "execute_item", execute)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_rq1.py",
            "--node",
            "node1",
            "--phase",
            "formal",
            "--execute",
            "--retry-failed",
            "--results",
            str(tmp_path),
        ],
    )
    runner.main()
    output = json.loads(capsys.readouterr().out)
    saved = load(tmp_path / "progress.json")
    row = saved["runs"][queue[0]["run_id"]]
    assert len(calls) == 1
    assert output["paused"] is True and "planned GPU" in output["reason"]
    assert saved["pause_reason"] == output["reason"] and saved["paused_at"]
    assert (
        row["status"] == "pending"
        and row["attempts"][-1]["failure_class"] == "external_resource_conflict"
    )
    assert not ({"finished_at", "sla_pass", "sla", "metrics", "artifact"} & set(row))
    assert saved["runs"][queue[1]["run_id"]]["status"] == "pending"


def test_execute_model_probe_failure_pauses_without_blocking_queue(
    monkeypatch, tmp_path, capsys
):
    runner = load_runner_module()
    queue = expand(
        "node1",
        "formal",
        ("moe_mixtral_8x22b",),
        ("native",),
        ("qa_lpld",),
        (2,),
        (1, 2),
    )
    progress = initial_progress("node1")
    for item in queue:
        progress["runs"][item["run_id"]] = inventory_row(item)
    first = progress["runs"][queue[0]["run_id"]]
    first.update(
        status="failed",
        finished_at="stale",
        sla_pass=False,
        sla={"bad": 1},
        metrics={"bad": 1},
        artifact="stale",
    )
    atomic_json(tmp_path / "progress.json", progress)
    models, workloads, matrix, cluster = configs()
    calls = []
    monkeypatch.setattr(runner, "configs", lambda: (models, workloads, matrix, cluster))
    monkeypatch.setattr(runner, "expand", lambda *args, **kwargs: queue)
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
    monkeypatch.setattr(runner, "verify_model", lambda *args: False)
    monkeypatch.setattr(runner, "RemoteExecutor", lambda *args: object())
    monkeypatch.setattr(
        runner,
        "execute_item",
        lambda *args: (calls.append(args) or {"status": "complete"}),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_rq1.py",
            "--node",
            "node1",
            "--phase",
            "formal",
            "--execute",
            "--retry-failed",
            "--results",
            str(tmp_path),
        ],
    )
    runner.main()
    output = json.loads(capsys.readouterr().out)
    saved = load(tmp_path / "progress.json")
    row = saved["runs"][queue[0]["run_id"]]
    assert calls == []
    assert output["paused"] is True and "temporarily unavailable" in output["reason"]
    assert saved["pause_reason"] == output["reason"] and saved["paused_at"]
    assert row["status"] == "pending"
    assert not ({"finished_at", "sla_pass", "sla", "metrics", "artifact"} & set(row))
    assert saved["runs"][queue[1]["run_id"]]["status"] == "pending"
    assert all(r["status"] != "blocked_model_missing" for r in saved["runs"].values())
