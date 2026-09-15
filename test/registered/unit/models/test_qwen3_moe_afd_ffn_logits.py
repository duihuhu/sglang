"""CPU-only regression tests for Qwen3MoE pure-AF FFN logits."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.models.qwen3_moe import Qwen3MoeForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="stage-a-cpu-only")


class _StaticModel(nn.Module):
    def __init__(self, hidden_states):
        super().__init__()
        self.hidden_states = hidden_states

    def forward(self, *args, **kwargs):
        return self.hidden_states


class _RecordingLogitsProcessor(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.output = output
        self.calls = []

    def forward(self, *args):
        self.calls.append(args)
        return self.output


class TestQwen3MoeAFDFFNLogits(CustomTestCase):
    def _model(self, hidden_states, logits_output, *, is_last_rank=True):
        model = Qwen3MoeForCausalLM.__new__(Qwen3MoeForCausalLM)
        nn.Module.__init__(model)
        model.pp_group = SimpleNamespace(is_last_rank=is_last_rank)
        model.config = SimpleNamespace(vocab_size=17)
        model.model = _StaticModel(hidden_states)
        model.logits_processor = _RecordingLogitsProcessor(logits_output)
        model.lm_head = SimpleNamespace()
        model.capture_aux_hidden_states = False
        return model

    def test_ffn_returns_vocab_sized_dummy_without_logits_processor(self):
        hidden_states = torch.randn(5, 7, dtype=torch.float64)
        normal_output = LogitsProcessorOutput(next_token_logits=torch.ones(1, 1))
        model = self._model(hidden_states, normal_output)
        forward_batch = SimpleNamespace(batch_size=3)

        with mock.patch("sglang.srt.models.qwen3_moe.afd_is_ffn", return_value=True):
            output = model(
                torch.tensor([1, 2, 3]),
                torch.tensor([0, 1, 2]),
                forward_batch,
            )

        self.assertIsInstance(output, LogitsProcessorOutput)
        self.assertEqual(output.next_token_logits.shape, (3, 17))
        self.assertEqual(output.next_token_logits.dtype, hidden_states.dtype)
        self.assertEqual(output.next_token_logits.device, hidden_states.device)
        self.assertEqual(torch.count_nonzero(output.next_token_logits).item(), 0)
        self.assertIsNone(output.hidden_states)
        self.assertEqual(model.logits_processor.calls, [])

    def test_attention_keeps_normal_logits_processor(self):
        hidden_states = torch.randn(2, 7)
        normal_output = LogitsProcessorOutput(next_token_logits=torch.ones(2, 17))
        model = self._model(hidden_states, normal_output)
        input_ids = torch.tensor([4, 5])
        forward_batch = SimpleNamespace(batch_size=2)

        with mock.patch("sglang.srt.models.qwen3_moe.afd_is_ffn", return_value=False):
            output = model(input_ids, torch.tensor([0, 1]), forward_batch)

        self.assertIs(output, normal_output)
        self.assertEqual(len(model.logits_processor.calls), 1)
        args = model.logits_processor.calls[0]
        self.assertIs(args[0], input_ids)
        self.assertIs(args[1], hidden_states)
        self.assertIs(args[2], model.lm_head)
        self.assertIs(args[3], forward_batch)
        self.assertIsNone(args[4])

    def test_non_last_rank_returns_hidden_states(self):
        hidden_states = torch.randn(2, 7)
        normal_output = LogitsProcessorOutput(next_token_logits=torch.ones(2, 17))
        model = self._model(hidden_states, normal_output, is_last_rank=False)

        with mock.patch(
            "sglang.srt.models.qwen3_moe.afd_is_ffn", return_value=True
        ) as is_ffn:
            output = model(
                torch.tensor([4, 5]),
                torch.tensor([0, 1]),
                SimpleNamespace(batch_size=2),
            )

        self.assertIs(output, hidden_states)
        self.assertEqual(model.logits_processor.calls, [])
        is_ffn.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=3)
