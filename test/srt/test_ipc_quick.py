"""Quick IPC M=3 test using existing AFD test infrastructure."""

import os
import sys
import unittest

# Force IPC backend
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"

# Allow imports from test/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srt.test_afd_fixture import AFDServerBase

AFD_TEST_MODEL = "/models/Qwen3-0.6B"


class TestIPCM3(AFDServerBase):
    model = AFD_TEST_MODEL
    micro_batch = 3

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        extra_args = [
            "--trust-remote-code",
            "--afd-comm-backend", "ipc",
            "--skip-server-warmup",
        ]
        # Start FFN first so its IPC socket is listening before Attn connects
        cls.start_ffn(extra_args=extra_args)
        cls.start_attn(extra_args=extra_args)
        cls.wait_server_ready(cls.attn_url, cls.process_attn)

    def test_single_request(self):
        import requests
        resp = requests.post(
            f"{self.attn_url}/generate",
            json={
                "text": "Hello, world!",
                "sampling_params": {"max_new_tokens": 32, "temperature": 0},
            },
            timeout=120,
        )
        self.assertEqual(resp.status_code, 200, f"Generate failed: {resp.text}")
        result = resp.json()
        self.assertIn("text", result)
        self.assertTrue(len(result["text"]) > 0, f"Output empty: {result}")
        print(f"[IPC M=3] Generated: {repr(result['text'])}")


if __name__ == "__main__":
    unittest.main()
