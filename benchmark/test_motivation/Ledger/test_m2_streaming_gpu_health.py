import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from m2_streaming_gpu_health import StreamingGpuMonitor, parse_sample


LINES = ["6, 90, 1410, 42, 180, 65000", "7, 88, 1395, 41, 177, 65000"]


class SampleTests(unittest.TestCase):
    def test_same_schema_and_physical_devices(self):
        result = parse_sample(LINES)
        self.assertEqual(len(result["gpus"]), 2)
        self.assertEqual(result["gpus"][0]["sm_mhz"], 1410)
        self.assertGreater(result["observed_s"], 0)

    def test_missing_duplicate_or_wrong_devices_rejected(self):
        for values in (LINES[:1], [LINES[0], LINES[0]], [LINES[0].replace("6,", "0,"), LINES[1]]):
            with self.assertRaises(ValueError):
                parse_sample(values)

    def test_missing_columns_and_nonfinite_readings_rejected(self):
        for values in (["6, 90", LINES[1]], [LINES[0].replace("1410", "nan"), LINES[1]]):
            with self.assertRaises(ValueError):
                parse_sample(values)


class ProcessTests(unittest.IsolatedAsyncioTestCase):
    def process(self):
        return SimpleNamespace(returncode=None, stdout=SimpleNamespace(readline=AsyncMock(side_effect=[(s+"\n").encode() for s in LINES])),
                               terminate=Mock(), kill=Mock(), wait=AsyncMock(return_value=0))

    async def test_starts_once_and_cleans_up_own_process(self):
        process = self.process()
        with patch("m2_streaming_gpu_health.asyncio.create_subprocess_exec", AsyncMock(return_value=process)) as spawn:
            async with StreamingGpuMonitor() as health:
                stop = asyncio.Event()
                stop.set()
                samples = []
                await health.monitor(stop, samples)
                self.assertEqual(len(samples), 1)
            spawn.assert_awaited_once()
            self.assertIn("--loop-ms=500", spawn.call_args.args)
        process.terminate.assert_called_once()
        process.wait.assert_awaited_once()

    async def test_bad_first_sample_still_cleans_up(self):
        process = self.process()
        process.stdout.readline = AsyncMock(return_value=b"")
        with patch("m2_streaming_gpu_health.asyncio.create_subprocess_exec", AsyncMock(return_value=process)):
            with self.assertRaises(RuntimeError):
                async with StreamingGpuMonitor():
                    pass
        process.terminate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
