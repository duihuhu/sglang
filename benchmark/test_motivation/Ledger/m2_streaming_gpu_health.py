"""Read physical GPU 6/7 from one owned, persistent nvidia-smi process."""

import asyncio
import csv
import io
import math
import time


def parse_sample(lines):
    keys = ("index", "utilization", "sm_mhz", "temperature_c", "power_w", "memory_mib")
    rows = list(csv.reader(io.StringIO("\n".join(line.strip() for line in lines))))
    if len(rows) != 2 or any(len(row) != len(keys) for row in rows):
        raise ValueError("Incomplete GPU sample")
    gpus = [{key: float(value.strip()) for key, value in zip(keys, row)} for row in rows]
    if {g["index"] for g in gpus} != {6, 7} or any(not math.isfinite(v) for g in gpus for v in g.values()):
        raise ValueError("Invalid GPU sample or unexpected device")
    return {"observed_s": time.perf_counter(), "gpus": gpus}


class StreamingGpuMonitor:
    async def __aenter__(self):
        self.process = await asyncio.create_subprocess_exec(
            "nvidia-smi", "-i", "6,7",
            "--query-gpu=index,utilization.gpu,clocks.sm,temperature.gpu,power.draw,memory.used",
            "--format=csv,noheader,nounits", "--loop-ms=500",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            self.first_sample = await self.read_sample()
        except BaseException:
            await self.close()
            raise
        return self

    async def read_sample(self):
        lines = []
        for _ in range(2):
            line = await asyncio.wait_for(self.process.stdout.readline(), timeout=10)
            if not line:
                raise RuntimeError("GPU monitor exited or produced incomplete output")
            lines.append(line.decode("utf-8"))
        return parse_sample(lines)

    async def monitor(self, stop, samples):
        samples.append(self.first_sample)
        while not stop.is_set():
            samples.append(await self.read_sample())

    async def close(self):
        if self.process.returncode is None:
            try:
                self.process.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()
