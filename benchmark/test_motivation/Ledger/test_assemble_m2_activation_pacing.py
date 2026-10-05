import copy
import json
from pathlib import Path
import tempfile
import unittest

from assemble_m2_activation_pacing import ALL_MODES, REPLACED_MODES, assemble, cells


class AssemblyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.old, self.new, self.output = root / "old", root / "new", root / "assembled"
        self.old.mkdir()
        self.new.mkdir()
        self.hashes = {"scheduler": "frozen"}
        args = dict(stage="screen", regime="state8", modes=sorted(ALL_MODES), sessions=2,
                    reps_per_session=1, rep_start=1, waves=4, interval=2, short_per_wave=16,
                    long_per_wave=4, output_tokens=64, output=str(self.old), max_lateness=.05)
        self.manifest = dict(args=args, source_hashes=self.hashes, planned_measured_runs=48,
                             planned_measured_requests=3840,
                             commands={mode: ["serve", mode] for mode in ALL_MODES})
        new = copy.deepcopy(self.manifest)
        new["args"].update(modes=sorted(REPLACED_MODES), sessions=1, rep_start=2, output=str(self.new))
        new.update(planned_measured_runs=12, planned_measured_requests=960)
        new["commands"] = {mode: new["commands"][mode] for mode in REPLACED_MODES}
        self.write(self.old / "manifest.json", self.manifest)
        self.write(self.new / "manifest.json", new)
        self.write(self.new / "completed.json", dict(completed=True, source_hashes=self.hashes))
        retained = cells(ALL_MODES, (1, 2)) - cells(REPLACED_MODES, (2,))
        retained |= set(sorted(cells({"separate-4096"}, (2,)))[:3])
        for directory, matrix in ((self.old, retained), (self.new, cells(REPLACED_MODES, (2,)))):
            for mode, limit, order, rep in matrix:
                run_id = f"m2p-state8-{mode}-k{limit}-{order}-rep{rep}-screen"
                self.write(directory / f"{run_id}.json", {"experiment": {
                    "mode": mode, "activation_limit": limit, "order": order,
                    "rep": rep, "run_id": run_id, "stage": "screen"}})

    def write(self, path, item):
        path.write_text(json.dumps(item))

    def test_complete_view_preserves_original(self):
        result = assemble(self.old, self.new, self.output, self.hashes)
        self.assertEqual(result, dict(selected=48, excluded=3, original_completed=False))
        self.assertEqual(len(list(self.old.glob("*-screen.json"))), 39)
        self.assertFalse((self.old / "completed.json").exists())
        self.assertEqual(len(list(self.output.glob("*-screen.json"))), 48)
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(len(manifest["assembly"]["selected"]), 48)
        self.assertEqual(len(manifest["assembly"]["excluded"]), 3)
        with self.assertRaises(FileExistsError):
            assemble(self.old, self.new, self.output, self.hashes)

    def test_changed_source_rejected(self):
        with self.assertRaisesRegex(ValueError, "Source versions"):
            assemble(self.old, self.new, self.output, {"scheduler": "changed"})
        self.assertFalse(self.output.exists())

    def test_changed_workload_rejected(self):
        path = self.new / "manifest.json"
        changed = json.loads(path.read_text())
        changed["args"]["max_lateness"] = .5
        self.write(path, changed)
        with self.assertRaisesRegex(ValueError, "parameters"):
            assemble(self.old, self.new, self.output, self.hashes)

    def test_completed_original_rejected(self):
        self.write(self.old / "completed.json", dict(completed=True))
        with self.assertRaisesRegex(ValueError, "interrupted original"):
            assemble(self.old, self.new, self.output, self.hashes)

    def test_incomplete_recovery_rejected(self):
        path = self.new / "completed.json"
        self.write(path, dict(completed=False, source_hashes=self.hashes))
        with self.assertRaisesRegex(ValueError, "did not complete"):
            assemble(self.old, self.new, self.output, self.hashes)

    def test_unknown_matrix_cell_rejected(self):
        self.write(self.new / "unexpected-screen.json", {"experiment": dict(
            mode="mixed-4096", activation_limit=1, order="short-first", rep=3,
            run_id="unexpected-screen", stage="screen")})
        with self.assertRaisesRegex(ValueError, "replacement"):
            assemble(self.old, self.new, self.output, self.hashes)


if __name__ == "__main__":
    unittest.main()
