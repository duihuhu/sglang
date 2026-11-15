"""CPU-only tests: never contact a host, container, GPU or HTTP endpoint."""

import contextlib
import io
import json
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark"))
from test_aflex.config import load, requests
from test_aflex.prepare_trace import convert
from test_aflex.requests import summarize
from test_aflex.run_benchmark import main, render

try:
    from sglang.test.test_utils import CustomTestCase
except ImportError:
    CustomTestCase = unittest.TestCase


class TestAFlexBenchmark(CustomTestCase):
    def config(self):
        return load(ROOT / "benchmark/test_aflex/examples/config.json")

    def test_config_and_workload(self):
        cfg = self.config()
        self.assertEqual(len(cfg["points"]), 5)
        rows = requests(ROOT / "benchmark/test_aflex/examples/synthetic.jsonl")
        self.assertEqual(rows[1]["input_len"], 16)

    def test_dvfs_library_validation(self):
        source = ROOT / "benchmark/test_aflex/examples/config.json"
        for value in (None, "relative/lib.so", "/assets/lib.so;echo", 42):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp:
                cfg = json.loads(source.read_text())
                if value is None:
                    del cfg["paths"]["dvfs_library"]
                else:
                    cfg["paths"]["dvfs_library"] = value
                path = Path(tmp) / "config.json"
                path.write_text(json.dumps(cfg))
                with self.assertRaisesRegex(ValueError, "dvfs_library"):
                    load(path)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = json.loads(source.read_text())
            cfg["paths"]["dvfs_library"] = "/assets/other.so"
            for point in cfg["points"]:
                for key in ("workload", "plan"):
                    if key in point:
                        point[key] = str(source.parent / point[key])
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError, "preflight.native_libraries"):
                load(path)

    def test_help_no_subprocess(self):
        with patch.object(
            subprocess, "run", side_effect=AssertionError("external call")
        ):
            with self.assertRaises(SystemExit) as result, contextlib.redirect_stdout(
                io.StringIO()
            ):
                main(["--help"])
            self.assertEqual(result.exception.code, 0)

    def test_render_offline_actual_schemes(self):
        with patch.object(
            subprocess, "run", side_effect=AssertionError("external call")
        ):
            commands = render(self.config())
        text = json.dumps(commands)
        for marker in (
            "--afd-attn-tp",
            "--afd-ffn-tp",
            "--afd-dvfs-decode-compositional",
            "--dvfs-policy biscale",
            "--policy round_robin",
            "--pd-disaggregation",
            "--mini-lb",
        ):
            self.assertIn(marker, text)
        self.assertNotIn("pkill", text)
        self.assertNotIn("AFlex_bench", text)
        self.assertIn("AFLEX_BENCH_OWNER", text)
        cfg = self.config()
        for command in commands:
            docker_argv = shlex.split(command["argv"][-1])
            self.assertEqual(docker_argv[:2], ["docker", "exec"])
            self.assertEqual(
                docker_argv[-1],
                "cd "
                + shlex.quote(cfg["paths"]["repo"])
                + " && export DVFS_CTRL_LIB="
                + shlex.quote(cfg["paths"]["dvfs_library"])
                + " && "
                + command["shell"],
            )

    def test_preflight_offline(self):
        with patch.object(
            subprocess, "run", side_effect=AssertionError("external call")
        ), contextlib.redirect_stdout(io.StringIO()) as out:
            main(
                [
                    "preflight",
                    "--config",
                    str(ROOT / "benchmark/test_aflex/examples/config.json"),
                ]
            )
        self.assertEqual(json.loads(out.getvalue())["status"], "OFFLINE_OK")

    def test_summary_counts_slo_union(self):
        reqs = [{}, {}, {}]
        result = summarize(
            reqs,
            [
                dict(
                    request_index=0,
                    success=True,
                    input_len=10,
                    completion_tokens=2,
                    ttft_ms=6000,
                    ttft_proc_ms=0,
                    tpot_ms=400,
                ),
                dict(request_index=1, success=False),
            ],
            2,
            12,
            1,
        )
        self.assertEqual(result["slo_violation_rate"], 100)
        self.assertEqual(result["total_tokens_all"], 12)
        self.assertEqual(result["energy_per_token_mj"], 1000)
        self.assertEqual(result["timed_out"], 1)

    def test_trace_source_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.csv"
            path.write_text(
                "TIMESTAMP,ContextTokens,GeneratedTokens\na,9000,2000\nb,0,2\nc,2,3\n"
            )
            rows = convert(path, qps=2)
            self.assertEqual(rows, convert(path, qps=2))
            self.assertEqual(len(rows), 2)
            self.assertIn(
                (4096, 1024), [(r["input_len"], r["output_len"]) for r in rows]
            )
            self.assertGreater(rows[0]["arrival_time_s"], 0)

    def test_invalid_workload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"
            path.write_text('{"input_len": 0, "output_len": 1, "arrival_time_s": 0}\n')
            with self.assertRaises(ValueError):
                requests(path)


class TestAFlexSafety(CustomTestCase):
    def test_unlock_attempts_every_host(self):
        from test_aflex import runtime as rt

        with patch.object(
            rt, "execute", side_effect=[RuntimeError("node1"), None]
        ) as execute:
            with self.assertRaisesRegex(RuntimeError, "node1"):
                rt.unlock_scope({"n1": [0], "n2": [1]})
        self.assertEqual([c.args[0] for c in execute.call_args_list], ["n1", "n2"])

    def test_cleanup_record_retained_on_timeout_and_idempotent(self):
        from test_aflex import runtime as rt

        with patch.object(rt, "OWNED", [("n1", 12)]), patch.object(rt, "CONFIG", {}):
            with patch.object(rt, "execute", side_effect=RuntimeError("timeout")):
                with self.assertRaises(RuntimeError):
                    rt.cleanup_all()
            self.assertEqual(rt.OWNED, [("n1", 12)])
            with patch.object(rt, "execute") as execute:
                rt.cleanup_all()
                rt.cleanup_all()
                self.assertEqual(execute.call_count, 1)
            self.assertEqual(rt.OWNED, [])

    def test_cleanup_failure_still_unlocks(self):
        from test_aflex import runtime as rt

        with patch.object(
            rt, "cleanup_all", side_effect=RuntimeError("cleanup")
        ), patch.object(
            rt, "unlock_scope", side_effect=RuntimeError("unlock")
        ) as unlock:
            with self.assertRaisesRegex(RuntimeError, "cleanup; unlock"):
                rt.finalize_scope({"n1": [0]})
            unlock.assert_called_once_with({"n1": [0]})

    def test_remote_preflight_checks_both_nodes_readonly(self):
        from test_aflex import runtime as rt
        from test_aflex.run_benchmark import configure

        cfg = load(ROOT / "benchmark/test_aflex/examples/config.json")
        configure(cfg)
        with patch.object(
            rt, "execute", side_effect=[RuntimeError("missing asset"), None]
        ) as execute:
            with self.assertRaisesRegex(RuntimeError, "missing asset"):
                rt.remote_preflight()
        self.assertEqual(execute.call_count, 2)
        code = execute.call_args.args[1]
        for marker in (
            "find_spec",
            "ctypes.CDLL",
            "config.json",
            "Prefill_A_GBDT.pkl",
            "is_file",
            "PYTHONDONTWRITEBYTECODE",
        ):
            self.assertIn(marker, code)
        for forbidden in (
            "lock_gpu",
            "mkdir",
            "launch_server --",
            "nvmlInit",
            "write_text",
        ):
            self.assertNotIn(forbidden, code)

    def test_run_preflight_failure_prevents_deploy(self):
        from test_aflex import run_benchmark as cli
        from test_aflex import runtime as rt

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            rt, "remote_preflight", side_effect=RuntimeError("preflight")
        ), patch.object(cli, "deploy") as deploy:
            with self.assertRaisesRegex(RuntimeError, "preflight"):
                cli.main(
                    [
                        "run",
                        "--execute",
                        "--config",
                        str(ROOT / "benchmark/test_aflex/examples/config.json"),
                        "--output",
                        str(Path(tmp) / "out.json"),
                    ]
                )
            deploy.assert_not_called()

    def test_partial_deploy_preserves_scope_for_finalization(self):
        from test_aflex import run_benchmark as cli
        from test_aflex import runtime as rt

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            rt, "remote_preflight"
        ), patch.object(
            cli, "deploy", side_effect=RuntimeError("partial")
        ), patch.object(
            rt, "finalize_scope"
        ) as finalize:
            with self.assertRaisesRegex(RuntimeError, "partial"):
                cli.main(
                    [
                        "run",
                        "--execute",
                        "--config",
                        str(ROOT / "benchmark/test_aflex/examples/config.json"),
                        "--output",
                        str(Path(tmp) / "out.json"),
                    ]
                )
            scope = finalize.call_args.args[0]
            self.assertEqual(sum(len(g) for g in scope.values()), 4)

    def test_heterogeneous_plan_parser_and_bootstrap_render(self):
        from test_aflex.aflex import PrefillSpec
        from test_aflex.config import parse_plan

        cfg = load(ROOT / "benchmark/test_aflex/examples/config.json")
        raw = json.loads((ROOT / "benchmark/test_aflex/examples/plan.json").read_text())
        raw["k_p"] = 2
        raw["prefill_specs"] = [
            dict(tp_pa=2, tp_pf=1, f_pa=930, f_pf=1170),
            dict(tp_pa=1, tp_pf=1, f_pa=930, f_pf=930),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            path.write_text(json.dumps(raw))
            plan = parse_plan(path)
            self.assertIsInstance(plan.prefill_specs[0], PrefillSpec)
            cfg["points"] = [dict(cfg["points"][0], plan=str(path))]
            with patch.object(subprocess, "run", side_effect=AssertionError("remote")):
                commands = render(cfg)
            shells = [c["shell"] for c in commands]
            for port in (24600, 24601):
                self.assertTrue(
                    any(f"--disaggregation-bootstrap-port {port}" in c for c in shells)
                )
                self.assertTrue(any(f" {port} --decode" in c for c in shells))
                self.assertTrue(
                    any(
                        f"bind(('0.0.0.0',{port}))" in " ".join(shlex.split(c))
                        for c in shells
                    )
                )
            raw["prefill_specs"][0]["tp_pa"] = 3
            path.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):
                parse_plan(path)

    def cleanup_fake(self, snapshots, force=False):
        from unittest.mock import MagicMock

        from test_aflex.process_cleanup import cleanup_session

        entries = []
        for members in snapshots:
            listing = []
            for pid, owned in members:
                entry = MagicMock()
                entry.name = str(pid)
                # stat fields after comm: state, ppid, pgrp, session, ... starttime.
                fields = ["S", "1", "42", "42"] + ["0"] * 15 + ["100"]
                (entry / "stat").read_text.return_value = f"{pid} (worker) " + " ".join(
                    fields
                )
                (entry / "environ").read_bytes.return_value = (
                    b"AFLEX_BENCH_OWNER=token" if owned else b"OTHER=1"
                )
                listing.append(entry)
            entries.append(listing)
        with patch("pathlib.Path.iterdir", side_effect=entries), patch(
            "os.killpg"
        ) as kill, patch("time.sleep"), patch("time.monotonic", side_effect=range(100)):
            cleanup_session(42, "token", term_timeout=2, force=force, kill_timeout=2)
        return kill

    def test_cleanup_missing_leader_waits_for_owned_descendants(self):
        import signal

        live = [(43, True)]
        kill = self.cleanup_fake([live, live, live, live, [], []])
        kill.assert_called_once_with(42, signal.SIGTERM)

    def test_cleanup_refuses_unowned_descendant(self):
        with self.assertRaisesRegex(RuntimeError, "ownership"):
            self.cleanup_fake([[(43, False)], [(43, False)]])

    def test_cleanup_already_exited(self):
        kill = self.cleanup_fake([[]])
        kill.assert_not_called()

    def test_cleanup_timeout_without_force(self):
        live = [(43, True)]
        with self.assertRaisesRegex(RuntimeError, "timeout"):
            self.cleanup_fake([live] * 6)

    def test_remote_missing_manifest_file_fails_readonly(self):
        import shlex

        from test_aflex import runtime as rt
        from test_aflex.run_benchmark import configure

        cfg = load(ROOT / "benchmark/test_aflex/examples/config.json")
        configure(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text("{}")
            cfg["paths"]["model"] = tmp
            cfg["paths"]["energy_model"] = tmp
            cfg["preflight"]["native_libraries"] = [str(root / "lib.so")]
            (root / "lib.so").write_text("synthetic")

            def execute(host, shell, **kwargs):
                code = shlex.split(shell)[-1]
                with patch("importlib.util.find_spec", return_value=object()), patch(
                    "ctypes.CDLL"
                ):
                    exec(code, {})

            with patch.object(rt, "execute", side_effect=execute) as transport:
                with self.assertRaisesRegex(RuntimeError, "missing prediction file"):
                    rt.remote_preflight()
            self.assertEqual(transport.call_count, 2)

    def test_cleanup_force_only_after_term_timeout(self):
        import signal

        live = [(43, True)]
        kill = self.cleanup_fake([live] * 8 + [[]], force=True)
        self.assertEqual(
            [c.args[1] for c in kill.call_args_list], [signal.SIGTERM, signal.SIGKILL]
        )


if __name__ == "__main__":
    unittest.main()
