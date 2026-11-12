"""AFD (Attention-FFN Disaggregation) test fixture.

Provides AFDServerBase for launching paired Attn + FFN server processes.
"""

import os
import time
import unittest

import requests

from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)


def popen_launch_afd_server(
    model: str,
    base_url: str,
    perspective: str,
    micro_batch: int = 3,
    timeout: float = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    other_args: list = None,
    env: dict = None,
):
    """Launch an AFD server (attn or ffn perspective)."""
    args = [
        "--afd-perspective",
        perspective,
        "--afd-micro-batch",
        str(micro_batch),
        "--disable-overlap-schedule",
        "--disable-cuda-graph",
    ]
    if perspective == "ffn":
        args += ["--skip-server-warmup", "--watchdog-timeout", "3600"]
    if other_args:
        args += other_args

    return popen_launch_server(
        model=model,
        base_url=base_url,
        timeout=timeout,
        other_args=args,
        env=env,
    )


class AFDServerBase(unittest.TestCase):
    """Base class for AFD tests. Starts paired Attn + FFN servers."""

    model = None
    attn_url = "http://127.0.0.1:30100"
    ffn_url = "http://127.0.0.1:30101"
    process_attn = None
    process_ffn = None
    micro_batch = 3

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if cls.model is None:
            raise unittest.SkipTest("No model specified")

    @classmethod
    def start_attn(cls, extra_args=None, env=None):
        args = ["--trust-remote-code"]
        if extra_args:
            args += extra_args
        if env is None:
            env = {
                **os.environ,
                "AFD_SCHED_HOST": "127.0.0.1",
                "AFD_SCHED_PORT": "65300",
            }
        cls.process_attn = popen_launch_afd_server(
            model=cls.model,
            base_url=cls.attn_url,
            perspective="attn",
            micro_batch=cls.micro_batch,
            other_args=args,
            env=env,
        )

    @classmethod
    def start_ffn(cls, extra_args=None, env=None):
        args = ["--trust-remote-code", "--port", "30101"]
        if extra_args:
            args += extra_args
        if env is None:
            env = {
                **os.environ,
                "AFD_SCHED_HOST": "127.0.0.1",
                "AFD_SCHED_PORT": "65300",
            }
        cls.process_ffn = popen_launch_afd_server(
            model=cls.model,
            base_url=cls.ffn_url,
            perspective="ffn",
            micro_batch=cls.micro_batch,
            other_args=args,
            env=env,
        )

    @classmethod
    def wait_server_ready(cls, url, process, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH):
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = requests.get(f"{url}/health", timeout=5)
                if resp.status_code == 200:
                    return
            except requests.RequestException:
                pass
            if process.poll() is not None:
                raise RuntimeError(
                    f"Server at {url} exited with code {process.returncode}"
                )
            time.sleep(5)
        raise TimeoutError(f"Server at {url} did not start within {timeout}s")

    @classmethod
    def tearDownClass(cls):
        from sglang.test.test_utils import kill_process_tree

        for proc in [cls.process_attn, cls.process_ffn]:
            if proc is not None:
                try:
                    kill_process_tree(proc.pid)
                except Exception:
                    pass
        super().tearDownClass()
