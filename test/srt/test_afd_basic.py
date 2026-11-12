"""AFD (Attention-FFN Disaggregation) basic end-to-end tests.

Tests single/batch requests, microbatch configurations, and accuracy.
Requires 2+ GPUs on the same node for Attn + FFN.
"""

import json
import os
import unittest

import requests

from test.srt.test_afd_fixture import AFDServerBase

# Model for testing; override via env var.
# Default: Qwen3-0.6B (Dense model with LayerCommunicator, small enough for 2-GPU test)
# For MoE testing: set AFD_TEST_MODEL=Qwen/Qwen3-30B-A3B
AFD_TEST_MODEL = os.getenv("AFD_TEST_MODEL", "Qwen/Qwen3-0.6B")


class TestAFDBasicM3(AFDServerBase):
    """Test AFD with microbatch=3 (default)."""

    model = AFD_TEST_MODEL
    micro_batch = 3

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.start_attn()
        cls.start_ffn()
        cls.wait_server_ready(cls.attn_url, cls.process_attn)

    def _generate(self, prompt, max_tokens=32):
        resp = requests.post(
            f"{self.attn_url}/generate",
            json={"text": prompt, "sampling_params": {"max_new_tokens": max_tokens, "temperature": 0}},
            timeout=60,
        )
        self.assertEqual(resp.status_code, 200, f"Generate failed: {resp.text}")
        return resp.json()

    def test_single_request(self):
        """Single request should produce non-empty output."""
        result = self._generate("Hello, world!")
        self.assertIn("text", result)
        self.assertTrue(len(result["text"]) > 0, "Output text is empty")

    def test_batch_requests(self):
        """Multiple concurrent requests should all complete."""
        import concurrent.futures

        prompts = [f"Question {i}: What is {i}+{i}?" for i in range(5)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(self._generate, p, 16) for p in prompts]
            results = [f.result() for f in futures]

        for i, result in enumerate(results):
            self.assertIn("text", result, f"Request {i} missing 'text' field")
            self.assertTrue(len(result["text"]) > 0, f"Request {i} output is empty")

    def test_long_prompt(self):
        """A longer prompt exercises the EXTEND (prefill) path."""
        long_prompt = "The quick brown fox jumps over the lazy dog. " * 50
        result = self._generate(long_prompt, max_tokens=16)
        self.assertIn("text", result)
        self.assertTrue(len(result["text"]) > 0)


class TestAFDBasicM1(AFDServerBase):
    """Test AFD with microbatch=1 (no pipeline overlap)."""

    model = AFD_TEST_MODEL
    micro_batch = 1
    attn_url = "http://127.0.0.1:30200"
    ffn_url = "http://127.0.0.1:30201"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        env_override = {
            **os.environ,
            "AFD_SCHED_HOST": "127.0.0.1",
            "AFD_SCHED_PORT": "65301",
        }
        cls.start_attn(extra_args=["--port", "30200"], env=env_override)
        cls.start_ffn(extra_args=["--port", "30201"], env=env_override)
        cls.wait_server_ready(cls.attn_url, cls.process_attn)

    def test_single_request_m1(self):
        resp = requests.post(
            f"{self.attn_url}/generate",
            json={"text": "Hello!", "sampling_params": {"max_new_tokens": 16, "temperature": 0}},
            timeout=60,
        )
        self.assertEqual(resp.status_code, 200)
        result = resp.json()
        self.assertIn("text", result)
        self.assertTrue(len(result["text"]) > 0)


class TestAFDBasicM2(AFDServerBase):
    """Test AFD with microbatch=2."""

    model = AFD_TEST_MODEL
    micro_batch = 2
    attn_url = "http://127.0.0.1:30300"
    ffn_url = "http://127.0.0.1:30301"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.start_attn(extra_args=["--port", "30300"])
        cls.start_ffn(extra_args=["--port", "30301"])
        cls.wait_server_ready(cls.attn_url, cls.process_attn)

    def test_single_request_m2(self):
        resp = requests.post(
            f"{self.attn_url}/generate",
            json={"text": "Hello!", "sampling_params": {"max_new_tokens": 16, "temperature": 0}},
            timeout=60,
        )
        self.assertEqual(resp.status_code, 200)
        result = resp.json()
        self.assertIn("text", result)
        self.assertTrue(len(result["text"]) > 0)


if __name__ == "__main__":
    unittest.main()
