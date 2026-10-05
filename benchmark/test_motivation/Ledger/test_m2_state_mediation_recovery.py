import unittest

try:
    from recover_m2_state_mediation import recovery_selection, session_files
    from run_m2_state_mediation import session_plan
except ModuleNotFoundError as error:
    if error.name != "aiohttp":
        raise
    recovery_selection = None


@unittest.skipIf(recovery_selection is None, "Replay dependencies unavailable locally")
class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.plan = session_plan("calibrate")
        self.names = {name for session in self.plan[:4] for name in session_files(session)}
        self.names.update(f"server-{p}-{m}-rep{r}.log" for p, m, r, _ in self.plan[:5])

    def test_selects_whole_prefix_and_excludes_all_partial_session_data(self):
        extra = session_files(self.plan[4])[:3]
        adopted, excluded = recovery_selection(self.plan, self.names | set(extra), 4)
        self.assertEqual(len(adopted), 24)
        self.assertEqual(excluded, sorted(extra))

    def test_missing_one_warmup_rejects_entire_adopted_session(self):
        self.names.remove(session_files(self.plan[0])[0])
        with self.assertRaisesRegex(ValueError, "partial session"):
            recovery_selection(self.plan, self.names, 4)

    def test_requires_next_session_evidence(self):
        p, m, r, _ = self.plan[4]
        self.names.remove(f"server-{p}-{m}-rep{r}.log")
        with self.assertRaisesRegex(ValueError, "next-session"):
            recovery_selection(self.plan, self.names, 4)

    def test_rejects_unexpected_artifact(self):
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            recovery_selection(self.plan, self.names | {"m2mediate-unplanned.json"}, 4)

    def test_rejects_empty_or_all_adopted_plan(self):
        for keep in (0, len(self.plan), -1):
            with self.assertRaises(ValueError):
                recovery_selection(self.plan, self.names, keep)


if __name__ == "__main__":
    unittest.main()
