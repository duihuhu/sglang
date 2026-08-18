"""Selection tests for the optional Central I/O HostKVCache backend."""

from __future__ import annotations

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sglang.srt.mem_cache import hiradix_cache


class HiRadixCentralHostPoolTest(unittest.TestCase):
    def test_explicit_socket_selects_central_host_pool(self):
        args = SimpleNamespace(
            hicache_ratio=2.0,
            hicache_size=0,
            hicache_mem_layout="page_first",
            hicache_storage_backend=None,
        )
        device_pool = object()
        with patch.dict(
            os.environ,
            {
                "SGLANG_CENTRAL_IO_SOCKET": "/tmp/latticekv-agent.sock",
                "SGLANG_CENTRAL_IO_MODEL_ID": "hot-a",
            },
            clear=False,
        ), patch.object(
            hiradix_cache,
            "CentralIOMHATokenToKVPoolHost",
            return_value="central-host",
        ) as central:
            result = hiradix_cache._build_mha_host_pool(device_pool, args, page_size=16)

        self.assertEqual(result, "central-host")
        central.assert_called_once_with(
            device_pool,
            2.0,
            0,
            16,
            "page_first",
            socket_path="/tmp/latticekv-agent.sock",
            model_id="hot-a",
        )


if __name__ == "__main__":
    unittest.main()
