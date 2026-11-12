import unittest
from types import SimpleNamespace

from sglang.srt.energy.tier1_solver import Tier1Solver, WorkloadProfile, _AFPairCandidate


def solver(**kwargs):
    return Tier1Solver(SimpleNamespace(), num_layers=1, **kwargs)


def pair(t_us=10_000):
    return _AFPairCandidate(1, 210, 1, 210, t_us / 2, t_us / 2, 1, 1, t_pair_us=t_us)


class Tier1CapacityTest(unittest.TestCase):
    def test_chatbot_one_decode_replica_is_capacity_limited(self):
        s = solver(capacity_margin=0.15)
        wl = WorkloadProfile(lambda_prefill=10, mean_ol=100, n_active_decode=10, bs_avg_d=10)
        assert not s._decode_diagnostics(wl, pair(20_000), 1, 100)["feasible"]


    def test_more_decode_replicas_restore_capacity(self):
        s = solver(capacity_margin=0.15)
        wl = WorkloadProfile(lambda_prefill=10, mean_ol=100, n_active_decode=10, bs_avg_d=10)
        assert not s._decode_diagnostics(wl, pair(20_000), 1, 100)["feasible"]
        assert s._decode_diagnostics(wl, pair(20_000), 4, 100)["feasible"]


    def test_prefill_rho_threshold_is_strict(self):
        s = solver(prefill_rho_max=0.9)
        p = pair(100_000)
        # lambda=10, batch=1 gives rho=1 and must not be considered feasible.
        rho = 10 * (s.num_layers * p.t_pair_us / 1e6) / 1
        assert rho > s.prefill_rho_max


    def test_route_skew_reduces_capacity_and_increases_busy_memory_load(self):
        wl = WorkloadProfile(lambda_prefill=2, mean_ol=100, n_active_decode=20, bs_avg_d=10)
        even = solver(route_skew_factor=1.0)._decode_diagnostics(wl, pair(), 2, 100)
        skew = solver(route_skew_factor=2.0)._decode_diagnostics(wl, pair(), 2, 100)
        assert skew["capacity_tps"] < even["capacity_tps"]
        assert skew["busy_batch"] > even["busy_batch"]


    def test_mean_ol_defaults_to_representative_output_length(self):
        s = solver()
        wl = WorkloadProfile(lambda_prefill=2, ol_rep_d=123, mean_ol=None)
        assert s._decode_diagnostics(wl, pair(), 2)["demand_tps"] == 246


if __name__ == "__main__":
    unittest.main()
