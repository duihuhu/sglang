"""Graceful Reshard Orchestrator.

Coordinates the full reshard lifecycle:
  1. Pre-launch shadow ranks on free GPUs (background, non-blocking)
  2. Drain old module via router API (new requests queued, not rejected)
  3. Wait for old module to become idle
  4. Signal old process to export weights via IPC
  5. New process inherits weights and re-slices for new TP
  6. New process initializes NCCL comm group + allocates KV cache
  7. Activate new module in router, queued requests dispatched
  8. Kill old process, free GPU memory

This class provides a high-level API that can be used both programmatically
and via CLI for multi-stage reshard demonstrations.
"""

import asyncio
import json
import logging
import os
import re
import signal
import socket
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


class ModuleType(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class Perspective(str, Enum):
    ATTN = "attn"
    FFN = "ffn"
    FULL = "full"  # For PD mode (non-PDAF)


@dataclass
class ModuleConfig:
    """Configuration for a single SGLang module."""
    module_type: ModuleType
    perspective: Perspective
    tp_size: int
    gpu_ids: List[int]
    port: int
    nccl_port: int
    bootstrap_port: int
    extra_args: List[str] = field(default_factory=list)

    @property
    def cvd(self) -> str:
        return ",".join(str(g) for g in self.gpu_ids)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@dataclass
class StageConfig:
    """Configuration for a complete service stage (all modules)."""
    name: str
    modules: List[ModuleConfig]
    bootstrap_port: int

    @property
    def gpu_count(self) -> int:
        gpus = set()
        for m in self.modules:
            gpus.update(m.gpu_ids)
        return len(gpus)

    @property
    def all_gpus(self) -> set:
        gpus = set()
        for m in self.modules:
            gpus.update(m.gpu_ids)
        return gpus

    def get_prefill_modules(self) -> List[ModuleConfig]:
        return [m for m in self.modules if m.module_type == ModuleType.PREFILL]

    def get_decode_modules(self) -> List[ModuleConfig]:
        return [m for m in self.modules if m.module_type == ModuleType.DECODE]


@dataclass
class TransitionResult:
    """Result of a stage transition."""
    success: bool
    method: str  # "graceful" | "restart"
    total_time_s: float
    downtime_s: float  # visible service interruption
    shadow_load_time_s: float
    error: Optional[str] = None


class GracefulReshardOrchestrator:
    """Orchestrates graceful TP reshard for SGLang modules.

    Example usage:
        orch = GracefulReshardOrchestrator(
            router_port=42000,
            model_path="/models/Qwen3-32B",
            python_path="/usr/bin/python3",
        )

        # Define stages
        stage1 = StageConfig(name="4GPU", modules=[...], bootstrap_port=49999)
        stage2 = StageConfig(name="6GPU", modules=[...], bootstrap_port=49998)

        # Deploy initial stage
        orch.deploy_stage(stage1)
        orch.deploy_router(stage1)

        # Graceful transition
        result = orch.transition(stage1, stage2)
        print(f"Transition: {result.method}, downtime={result.downtime_s:.2f}s")
    """

    def __init__(
        self,
        router_port: int = 42000,
        model_path: str = "/models/Qwen3-32B",
        python_path: str = "/usr/bin/python3",
        log_dir: Optional[Path] = None,
        mem_fraction: float = 0.85,
        max_running_requests: int = 128,
        ib_device: str = "mlx5_4",
        use_ipc_weights: bool = True,
    ):
        self.router_port = router_port
        self.model_path = model_path
        self.python_path = python_path
        self.log_dir = log_dir or Path("/tmp/sglang_reshard/logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.mem_fraction = mem_fraction
        self.max_running_requests = max_running_requests
        self.ib_device = ib_device
        self.use_ipc_weights = use_ipc_weights

        self._processes: Dict[str, subprocess.Popen] = {}
        self._log_files: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def deploy_stage(self, stage: StageConfig, timeout: int = 360) -> bool:
        """Deploy all modules in a stage. Blocks until all are healthy."""
        logger.info("Deploying stage: %s (%d GPUs)", stage.name, stage.gpu_count)

        for module in stage.modules:
            self._start_module(module, stage.bootstrap_port)
            time.sleep(3)

        for module in stage.modules:
            if not self._wait_health(module.port, timeout):
                logger.error("Module %s:%s (port %d) failed to start!",
                             module.module_type.value, module.perspective.value, module.port)
                return False
            logger.info("  %s/%s ready (port %d, GPU %s)",
                        module.module_type.value, module.perspective.value,
                        module.port, module.cvd)
        return True

    def deploy_router(self, stage: StageConfig, timeout: int = 60) -> bool:
        """Deploy the router pointing to the given stage's modules."""
        prefill_modules = stage.get_prefill_modules()
        decode_modules = stage.get_decode_modules()

        if not prefill_modules or not decode_modules:
            logger.error("Stage must have at least one prefill and one decode module")
            return False

        # Use the attn module port for PD routing
        prefill_port = prefill_modules[0].port
        decode_port = decode_modules[0].port

        cmd = [self.python_path, "-m", "sglang_router.launch_router",
               "--pd-disaggregation", "--mini-lb",
               "--prefill", f"http://127.0.0.1:{prefill_port}",
               "--decode", f"http://127.0.0.1:{decode_port}",
               "--host", "127.0.0.1", "--port", str(self.router_port)]

        self._popen("router", cmd, os.environ.copy())
        if not self._wait_health(self.router_port, timeout):
            logger.error("Router failed!")
            return False
        logger.info("Router ready (port %d)", self.router_port)
        return True

    def transition(self, old_stage: StageConfig, new_stage: StageConfig) -> TransitionResult:
        """Perform a graceful transition from old_stage to new_stage.

        Automatically chooses between:
          - Graceful swap (if no GPU overlap): pre-deploy shadow, then atomic swap
          - Restart swap (if GPU overlap): drain→kill→deploy→activate
        """
        t0 = time.monotonic()
        overlap = old_stage.all_gpus & new_stage.all_gpus

        if not overlap:
            return self._graceful_transition(old_stage, new_stage, t0)
        else:
            return self._restart_transition(old_stage, new_stage, t0)

    def kill_stage(self, stage: StageConfig):
        """Kill all processes belonging to a stage."""
        ports = [m.port for m in stage.modules]
        self._kill_ports(ports)

    def cleanup(self):
        """Kill all managed processes and the router."""
        all_ports = [self.router_port] + [
            p.pid for p in self._processes.values() if p.poll() is None
        ]
        # Kill by port is more reliable
        for name, proc in list(self._processes.items()):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
        self._processes.clear()
        for fh in self._log_files.values():
            try:
                fh.close()
            except Exception:
                pass
        self._log_files.clear()

    # ------------------------------------------------------------------
    # Transition strategies
    # ------------------------------------------------------------------

    def _graceful_transition(
        self, old_stage: StageConfig, new_stage: StageConfig, t0: float
    ) -> TransitionResult:
        """No GPU overlap: pre-deploy shadow in background, then atomic swap.

        If use_ipc_weights is enabled:
          1. Export old process weights via /admin/export_weights_ipc
          2. Start new process with SGLANG_RESHARD_IPC_DIR (uses dummy load + IPC override)
          3. New process loads in ~3-5s instead of ~30s
          4. Atomic router swap (downtime = drain + idle + activate ≈ 2-5s)
        """
        logger.info("[Graceful] Pre-deploying shadow: %s", new_stage.name)

        if self.use_ipc_weights:
            # Export weights from old process
            old_prefill = old_stage.get_prefill_modules()
            old_decode = old_stage.get_decode_modules()
            ipc_dir = "/tmp/sglang_reshard"

            for module in old_prefill + old_decode:
                try:
                    r = requests.post(
                        f"{module.url}/admin/export_weights_ipc",
                        json={"ipc_dir": ipc_dir,
                              "module_type": module.module_type.value,
                              "perspective": module.perspective.value},
                        timeout=30,
                    )
                    if r.status_code == 200:
                        logger.info("  Exported weights from %s (port %d)", module.module_type.value, module.port)
                    else:
                        logger.warning("  Export failed for port %d: %s", module.port, r.text[:100])
                except Exception as e:
                    logger.warning("  Export error for port %d: %s", module.port, e)

            # Deploy shadow with IPC env vars
            if not self._deploy_stage_with_ipc(new_stage, old_stage):
                # Fallback to normal deploy
                logger.info("  IPC deploy failed, falling back to disk load")
                if not self.deploy_stage(new_stage):
                    return TransitionResult(
                        success=False, method="graceful",
                        total_time_s=time.monotonic() - t0,
                        downtime_s=0, shadow_load_time_s=0,
                        error="Shadow deployment failed",
                    )
        else:
            if not self.deploy_stage(new_stage):
                return TransitionResult(
                    success=False, method="graceful",
                    total_time_s=time.monotonic() - t0,
                    downtime_s=0, shadow_load_time_s=0,
                    error="Shadow deployment failed",
                )

        shadow_time = time.monotonic() - t0

        # Atomic swap via router
        t_swap = time.monotonic()
        swap_ok = self._do_router_swap(old_stage, new_stage)
        downtime = time.monotonic() - t_swap

        if swap_ok:
            self.kill_stage(old_stage)

        return TransitionResult(
            success=swap_ok, method="graceful",
            total_time_s=time.monotonic() - t0,
            downtime_s=downtime,
            shadow_load_time_s=shadow_time,
            error=None if swap_ok else "Router swap failed",
        )

    def _deploy_stage_with_ipc(self, new_stage: StageConfig, old_stage: StageConfig) -> bool:
        """Deploy a stage using IPC weight inheritance (fast path)."""
        logger.info("  Deploying with IPC weights...")
        ipc_dir = "/tmp/sglang_reshard"

        for module in new_stage.modules:
            # Find corresponding old module for TP info
            old_modules = [m for m in old_stage.modules if m.module_type == module.module_type]
            old_tp = old_modules[0].tp_size if old_modules else 1

            extra_env = {
                "SGLANG_RESHARD_IPC_DIR": ipc_dir,
                "SGLANG_RESHARD_OLD_TP": str(old_tp),
                "SGLANG_RESHARD_MODULE_TYPE": module.module_type.value,
                "SGLANG_RESHARD_PERSPECTIVE": module.perspective.value,
            }
            self._start_module(module, new_stage.bootstrap_port, extra_env=extra_env)
            time.sleep(3)

        for module in new_stage.modules:
            if not self._wait_health(module.port, 360):
                logger.error("  IPC module %s:%s (port %d) failed!",
                             module.module_type.value, module.perspective.value, module.port)
                return False
            logger.info("  %s/%s ready (port %d, GPU %s) [IPC]",
                        module.module_type.value, module.perspective.value,
                        module.port, module.cvd)
        return True

    def _restart_transition(
        self, old_stage: StageConfig, new_stage: StageConfig, t0: float
    ) -> TransitionResult:
        """GPU overlap: must stop old before starting new. Has visible downtime."""
        logger.info("[Restart] GPU overlap detected, doing stop-start: %s → %s",
                    old_stage.name, new_stage.name)

        # Drain old modules
        self._drain_stage(old_stage)
        time.sleep(2)

        # Kill old
        self.kill_stage(old_stage)
        time.sleep(3)

        t_down = time.monotonic()

        # Deploy new
        if not self.deploy_stage(new_stage):
            return TransitionResult(
                success=False, method="restart",
                total_time_s=time.monotonic() - t0,
                downtime_s=time.monotonic() - t_down,
                shadow_load_time_s=0,
                error="New stage deployment failed",
            )

        # Activate in router
        self._activate_stage_in_router(new_stage)
        downtime = time.monotonic() - t_down

        return TransitionResult(
            success=True, method="restart",
            total_time_s=time.monotonic() - t0,
            downtime_s=downtime,
            shadow_load_time_s=0,
        )

    # ------------------------------------------------------------------
    # Router interaction
    # ------------------------------------------------------------------

    def _do_router_swap(self, old_stage: StageConfig, new_stage: StageConfig) -> bool:
        """Drain old, wait idle, activate new — atomic swap."""
        old_prefill = old_stage.get_prefill_modules()
        old_decode = old_stage.get_decode_modules()
        new_prefill = new_stage.get_prefill_modules()
        new_decode = new_stage.get_decode_modules()

        # Drain
        try:
            r = requests.post(
                f"http://127.0.0.1:{self.router_port}/admin/drain_module",
                json={
                    "prefill_urls": [m.url for m in old_prefill],
                    "decode_urls": [m.url for m in old_decode],
                },
                timeout=5,
            )
            logger.info("  Drain: %s", r.text[:100])
        except Exception as e:
            logger.warning("  Drain error: %s", e)

        # Wait idle
        if old_decode:
            self._wait_idle(old_decode[0].port)

        # Activate new
        try:
            payload = {
                "add_prefill_urls": [
                    [m.url, new_stage.bootstrap_port] for m in new_prefill
                ],
                "add_decode_urls": [m.url for m in new_decode],
                "remove_prefill_urls": [m.url for m in old_prefill],
                "remove_decode_urls": [m.url for m in old_decode],
            }
            r = requests.post(
                f"http://127.0.0.1:{self.router_port}/admin/activate_module",
                json=payload,
                timeout=5,
            )
            logger.info("  Activate: %s", r.text[:100])
            return r.status_code == 200
        except Exception as e:
            logger.error("  Activate failed: %s", e)
            return False

    def _drain_stage(self, stage: StageConfig):
        """Send drain request for all modules in a stage."""
        prefill = stage.get_prefill_modules()
        decode = stage.get_decode_modules()
        try:
            requests.post(
                f"http://127.0.0.1:{self.router_port}/admin/drain_module",
                json={
                    "prefill_urls": [m.url for m in prefill],
                    "decode_urls": [m.url for m in decode],
                },
                timeout=5,
            )
        except Exception:
            pass

    def _activate_stage_in_router(self, stage: StageConfig):
        """Activate a new stage in the router (after restart transition)."""
        prefill = stage.get_prefill_modules()
        decode = stage.get_decode_modules()
        try:
            requests.post(
                f"http://127.0.0.1:{self.router_port}/admin/activate_module",
                json={
                    "add_prefill_urls": [
                        [m.url, stage.bootstrap_port] for m in prefill
                    ],
                    "add_decode_urls": [m.url for m in decode],
                },
                timeout=5,
            )
        except Exception as e:
            logger.error("  Activate (restart) failed: %s", e)

    def _wait_idle(self, port: int, timeout: float = 60):
        """Wait for a module to become idle."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = requests.get(f"http://127.0.0.1:{port}/is_idle", timeout=3)
                if r.status_code == 200 and r.json().get("idle", False):
                    return True
            except Exception:
                return True  # If unreachable, consider it idle
            time.sleep(0.2)
        return False

    # ------------------------------------------------------------------
    # Process management
    # ------------------------------------------------------------------

    def _start_module(self, module: ModuleConfig, bootstrap_port: int, extra_env: dict = None):
        """Start a single SGLang module process."""
        cmd = [
            self.python_path, "-m", "sglang.launch_server",
            "--model-path", self.model_path,
            "--tp", str(module.tp_size),
            "--host", "127.0.0.1",
            "--port", str(module.port),
            "--disaggregation-mode", module.module_type.value,
            "--disaggregation-transfer-backend", "mooncake",
            "--disaggregation-bootstrap-port", str(bootstrap_port),
            "--disaggregation-ib-device", self.ib_device,
            "--mem-fraction-static", str(self.mem_fraction),
            "--max-running-requests", str(self.max_running_requests),
            "--nccl-port", str(module.nccl_port),
            "--skip-server-warmup",
            "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
            "--disable-radix-cache",
            "--num-reserved-decode-tokens", "512",
            "--watchdog-timeout", "600",
            "--enable-metrics",
        ]

        if module.perspective != Perspective.FULL:
            cmd += ["--afd-perspective", module.perspective.value]

        cmd += module.extra_args

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = module.cvd
        env["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
        if extra_env:
            env.update(extra_env)

        self._popen(
            f"{module.module_type.value}_{module.perspective.value}_tp{module.tp_size}",
            ["prlimit", "--memlock=unlimited:unlimited"] + cmd,
            env,
        )

    def _popen(self, name: str, cmd: List[str], env: dict):
        """Start a subprocess with logging."""
        log_path = self.log_dir / f"{name}.log"
        fh = open(log_path, "w")
        self._log_files[name] = fh
        p = subprocess.Popen(
            cmd, env=env, stdout=fh, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._processes[name] = p
        logger.info("  Started %s (PID=%d, log=%s)", name, p.pid, log_path)

    def _wait_health(self, port: int, timeout: int = 300) -> bool:
        """Wait for a service to become healthy."""
        deadline = time.time() + timeout
        # Wait for port open
        while time.time() < deadline:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(2)
                    s.connect(("127.0.0.1", port))
                    break
            except (ConnectionRefusedError, OSError):
                time.sleep(2)
        else:
            return False
        # Wait for health endpoint
        while time.time() < deadline:
            try:
                r = requests.get(f"http://127.0.0.1:{port}/health", timeout=5)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(2)
        return False

    def _kill_ports(self, ports: List[int]):
        """Kill processes listening on given ports."""
        for port in ports:
            try:
                r = subprocess.run(
                    ["ss", "-tlnp", f"sport = :{port}"],
                    capture_output=True, text=True,
                )
                for m in re.finditer(r"pid=(\d+)", r.stdout):
                    try:
                        os.kill(int(m.group(1)), signal.SIGKILL)
                    except OSError:
                        pass
            except Exception:
                pass
