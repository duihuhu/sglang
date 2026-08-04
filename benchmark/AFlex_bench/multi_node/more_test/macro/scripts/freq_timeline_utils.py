"""GPU frequency timeline capture during macro benchmark workloads."""
from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from pathlib import Path

import run_macro_benchmark as RMB

log = logging.getLogger("freq_timeline")

HERE = Path(__file__).resolve().parent
FREQ_TIMELINE_DIR = HERE / "results" / "freq_timelines"
TIMELINE_DIR = HERE / "timeline"
# Bind-mounted path visible inside operator_test on both nodes.
CONTAINER_LOG_ROOT = (
    "/workspace/sglang/benchmark/AFlex_bench/multi_node/"
    "more_test/macro/scripts/timeline/logs"
)
HOST_LOG_ROOT = TIMELINE_DIR / "logs"


class FreqMonitor:
    """Poll nvidia-smi for actual SM clock on specified GPUs."""

    def __init__(self, host: str, gpus: list[int], interval_s: float = 0.15):
        self.host = host
        self.gpus = gpus
        self.interval = interval_s
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll_loop(self):
        gpu_csv = ",".join(str(g) for g in self.gpus)
        RMB._ensure_local_node1_flag()
        use_local_docker = (
            self.host == RMB.NODE1_IP and RMB._LOCAL_NODE1
        )
        while not self._stop.is_set():
            t = time.time()
            try:
                if use_local_docker:
                    cmd = (
                        f"docker exec {RMB.CONTAINER} nvidia-smi "
                        f"--query-gpu=index,clocks.current.sm "
                        f"--format=csv,noheader,nounits -i {gpu_csv}"
                    )
                    out = subprocess.check_output(cmd, shell=True, timeout=2).decode().strip()
                else:
                    cmd = (
                        f"ssh -o StrictHostKeyChecking=no root@{self.host} "
                        f"\"docker exec {RMB.CONTAINER} nvidia-smi "
                        f"--query-gpu=index,clocks.current.sm "
                        f"--format=csv,noheader,nounits -i {gpu_csv}\""
                    )
                    out = subprocess.check_output(cmd, shell=True, timeout=3).decode().strip()
                for line in out.splitlines():
                    parts = line.strip().split(",")
                    if len(parts) == 2:
                        self.samples.append({
                            "t": round(t, 3),
                            "gpu": int(parts[0].strip()),
                            "freq_mhz": int(parts[1].strip()),
                        })
            except Exception as exc:
                log.debug("nvidia-smi poll failed on %s: %s", self.host, exc)
            elapsed = time.time() - t
            self._stop.wait(max(0.0, self.interval - elapsed))

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)


def collect_unified_dvfs_logs(scheme: str) -> dict:
    """Snapshot unified DVFS decision logs from timeline/logs/{scheme}/."""
    if scheme not in ("pd_hetero_tier_biscale", "native_tp1_tier"):
        return {}
    try:
        from timeline.dvfs_log_utils import collect_scheme_logs
    except ImportError:
        collect_scheme_logs = None
    log_dir = HOST_LOG_ROOT / scheme
    if collect_scheme_logs is not None:
        collect_scheme_logs(scheme)
    if not log_dir.exists():
        return {}
    out: dict[str, list[dict]] = {}
    for p in sorted(log_dir.glob("*.jsonl")):
        rows = []
        try:
            for line in p.read_text().splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        except Exception:
            continue
        if rows:
            out[p.name] = rows[-500:]
    return out


def collect_afd_dvfs_logs(scheme: str, rel_dir: str | None = None) -> dict:
    """Snapshot AFlex AFD DVFS decision logs from timeline/logs/{scheme}/."""
    if scheme != "aflex_tier1":
        return {}
    try:
        from timeline.dvfs_log_utils import collect_scheme_logs
        collect_scheme_logs(scheme)
    except ImportError:
        pass
    log_dir = HOST_LOG_ROOT / scheme
    if not log_dir.exists():
        return {}
    out: dict[str, list[dict]] = {}
    for p in sorted(log_dir.glob("dvfs_*.jsonl")):
        rows = []
        try:
            for line in p.read_text().splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        except Exception:
            continue
        if rows:
            out[p.name] = rows[-500:]
    return out


class FreqTimelineSession:
    def __init__(self, scheme: str, dataset: str, qps: int):
        self.scheme = scheme
        self.dataset = dataset
        self.qps = qps
        self.t_start = 0.0
        self.t_end = 0.0
        self._mon_n1: FreqMonitor | None = None
        self._mon_n2: FreqMonitor | None = None
        self._out_dir = FREQ_TIMELINE_DIR / scheme

    def start(self):
        self.t_start = time.time()
        gpus = list(range(8))
        self._mon_n1 = FreqMonitor(RMB.NODE1_IP, gpus)
        self._mon_n2 = FreqMonitor(RMB.NODE2_IP, gpus)
        self._mon_n1.start()
        self._mon_n2.start()

    def stop(self) -> dict:
        self.t_end = time.time()
        if self._mon_n1:
            self._mon_n1.stop()
        if self._mon_n2:
            self._mon_n2.stop()
        timeline = []
        if self._mon_n1:
            timeline.extend({"node": RMB.NODE1_IP, **s} for s in self._mon_n1.samples)
        if self._mon_n2:
            timeline.extend({"node": RMB.NODE2_IP, **s} for s in self._mon_n2.samples)
        timeline.sort(key=lambda x: x["t"])
        payload = {
            "scheme": self.scheme,
            "dataset": self.dataset,
            "qps": self.qps,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "duration_s": round(self.t_end - self.t_start, 2),
            "freq_samples_count": len(timeline),
            "nvidia_smi_timeline": timeline,
            "afd_dvfs_logs": collect_afd_dvfs_logs(self.scheme),
            "unified_dvfs_logs": collect_unified_dvfs_logs(self.scheme),
        }
        self._out_dir.mkdir(parents=True, exist_ok=True)
        out = self._out_dir / f"{self.dataset}_qps{self.qps}.json"
        out.write_text(json.dumps(payload, indent=2))
        log.info("  freq timeline: %d samples -> %s", len(timeline), out.name)
        return {"path": str(out.relative_to(HERE)), "samples": len(timeline)}


def patch_tier1_afd_dvfs_log(scheme: str = "aflex_tier1") -> callable:
    """Patch deploy_tier1_layout._afd_env to emit per-instance DVFS decision logs."""
    import deploy_tier1_layout as dtl

    orig = dtl._afd_env

    def _patched(role, attn_gpus, ffn_gpus, ucx_port, sched_port, host: str):
        base = orig(role, attn_gpus, ffn_gpus, ucx_port, sched_port, host)
        persp = "ffn" if role == "f" else "attn"
        cpath = f"{CONTAINER_LOG_ROOT}/{scheme}/dvfs_{persp}_{{gpu}}.jsonl"
        return base.replace(";", f" AFD_DVFS_DECISION_LOG='{cpath}';", 1)

    dtl._afd_env = _patched
    return orig


def restore_tier1_afd_env(orig):
    if orig is not None:
        import deploy_tier1_layout as dtl
        dtl._afd_env = orig


def ensure_container_log_dir(scheme: str):
    import shlex
    path = f"{CONTAINER_LOG_ROOT}/{scheme}"
    subprocess.run(
        ["docker", "exec", RMB.CONTAINER, "mkdir", "-p", path],
        check=False,
    )
    cmd = f"docker exec {RMB.CONTAINER} bash -lc {shlex.quote(f'mkdir -p {path}')}"
    subprocess.run(RMB._ssh(RMB.NODE2_IP, cmd), check=False)
