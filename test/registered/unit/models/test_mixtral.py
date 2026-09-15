"""CPU-only tests for Mixtral MoE output reduction selection."""

import unittest

from sglang.srt.layers.moe import MoeA2ABackend
from sglang.srt.models.mixtral import _should_all_reduce_moe_output
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="stage-a-cpu-only")


class TestMixtralMoEOutputReduction(CustomTestCase):
    def test_none_backend_uses_tp_all_reduce(self):
        self.assertTrue(_should_all_reduce_moe_output(2, MoeA2ABackend.NONE))

    def test_dedicated_a2a_backends_skip_tp_all_reduce(self):
        for backend in MoeA2ABackend:
            if backend.is_none():
                continue
            with self.subTest(backend=backend.value):
                self.assertFalse(_should_all_reduce_moe_output(2, backend))

    def test_tp_one_never_uses_tp_all_reduce(self):
        for backend in MoeA2ABackend:
            with self.subTest(backend=backend.value):
                self.assertFalse(_should_all_reduce_moe_output(1, backend))


if __name__ == "__main__":
    unittest.main(verbosity=3)
