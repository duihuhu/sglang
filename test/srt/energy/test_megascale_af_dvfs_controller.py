from sglang.srt.energy.megascale_af_dvfs_controller import (
    MegaScaleAFDVFSController,
    PhaseWorkload,
)


class Prediction:
    def __init__(self, value):
        self.value = value


class Predictor:
    def predict_latency(self, phase, op, tp, freq, bs, il, ol):
        base = {("prefill", "A"): 1000, ("prefill", "F"): 1400,
                ("decode", "A"): 350, ("decode", "F"): 450}[phase, op]
        return Prediction(base * 1410 / freq)

    def predict_energy(self, phase, op, tp, freq, bs, il, ol):
        base = {("prefill", "A"): 8, ("prefill", "F"): 12,
                ("decode", "A"): 2, ("decode", "F"): 3}[phase, op]
        return Prediction(base * freq / 1410)


def controller(**kwargs):
    return MegaScaleAFDVFSController(
        Predictor(), num_layers=1, tp_a=1, tp_f=1, t_comm_us=0,
        t_drain_us=0, freqs=[690, 930, 1170, 1410], **kwargs,
    )


def test_joint_policy_uses_one_pair_for_prefill_and_decode():
    ctrl = controller(tpot_safety_factor=1.0, prefill_safety_factor=1.0)
    decision = ctrl.select_joint(
        prefill=PhaseWorkload(1, 1024), decode=PhaseWorkload(8, 1024, 64),
        ttft_slack_us=5000, tpot_slo_us=2000, micro_batch=1,
        observed_tpot_us=800, prefill_interval_us=10000,
    )
    assert decision.f_a in ctrl.freqs and decision.f_f in ctrl.freqs
    assert ctrl.last_detail.prefill_energy_mj > 0
    assert ctrl.last_detail.decode_energy_mj > 0


def test_prefill_slowdown_is_charged_to_tpot():
    ctrl = controller(tpot_safety_factor=1.0, prefill_safety_factor=1.0,
                      interference_factor=1.0)
    decision = ctrl.select_joint(
        prefill=PhaseWorkload(1, 1024), decode=PhaseWorkload(8, 1024, 64),
        ttft_slack_us=10000, tpot_slo_us=1000, micro_batch=1,
        observed_tpot_us=800, prefill_interval_us=10000,
    )
    assert decision.f_a == 1410
    assert decision.f_f == 1410
    assert ctrl.last_detail.effective_tpot_us <= 1000


def test_attention_and_ffn_can_select_different_frequencies():
    ctrl = controller(tpot_safety_factor=1.0, prefill_safety_factor=1.0)
    decision = ctrl.select_joint(
        prefill=None, decode=PhaseWorkload(4, 512, 32),
        ttft_slack_us=0, tpot_slo_us=1600, micro_batch=1,
        observed_tpot_us=0,
    )
    assert decision.f_a != decision.f_f
